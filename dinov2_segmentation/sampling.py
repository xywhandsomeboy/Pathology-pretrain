"""Deterministic WSI-balanced sampling for cervical segmentation training.

The prepared cohort intentionally keeps every tumor patch.  Sampling those
rows uniformly, however, lets a few very large lesions dominate an epoch.  The
sampler below first fixes the negative/boundary/interior quotas and then
tempers each slide's contribution inside a stratum. Repeated indices are
forbidden inside a batch because the online graph context requires unique
target nodes.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random
from typing import Iterator, Mapping, Sequence

from torch.utils.data import Sampler


NEGATIVE = "negative"
BOUNDARY = "boundary"
INTERIOR = "interior"
_CATEGORIES = (NEGATIVE, BOUNDARY, INTERIOR)


def _as_binary(value: object, *, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be 0 or 1, got {value!r}") from error
    if result not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return result


def patch_stratum(
    row: Mapping[str, object], *, interior_threshold: float = 0.999999
) -> str:
    """Classify a manifest row as negative, mixed boundary, or interior."""

    if "has_tumor" not in row or "tumor_fraction" not in row:
        raise ValueError(
            "slide-stratified sampling requires has_tumor and tumor_fraction columns"
        )
    has_tumor = _as_binary(row["has_tumor"], name="has_tumor")
    try:
        fraction = float(row["tumor_fraction"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"tumor_fraction must be numeric, got {row['tumor_fraction']!r}"
        ) from error
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"tumor_fraction must be in [0,1], got {fraction}")
    if not has_tumor:
        if fraction != 0.0:
            raise ValueError("has_tumor=0 requires tumor_fraction=0")
        return NEGATIVE
    if fraction <= 0.0:
        raise ValueError("has_tumor=1 requires tumor_fraction>0")
    return INTERIOR if fraction >= interior_threshold else BOUNDARY


class SlideStratifiedSampler(Sampler[int]):
    """Draw fixed strata with capacity-limited, tempered WSI balancing.

    Slide quotas are proportional to ``available_patches ** balance_power``.
    This tempers large-lesion dominance without repeating a tiny slide hundreds
    of times.  Each patch has a hard per-epoch repeat cap and an index can never
    occur twice in one batch. ``set_epoch`` changes the deterministic order.
    """

    def __init__(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        num_samples: int,
        batch_size: int,
        positive_fraction: float = 0.60,
        boundary_positive_fraction: float = 0.50,
        interior_threshold: float = 0.999999,
        slide_balance_power: float = 0.5,
        max_patch_repeats: int = 2,
        seed: int = 42,
    ) -> None:
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.positive_fraction = float(positive_fraction)
        self.boundary_positive_fraction = float(boundary_positive_fraction)
        self.interior_threshold = float(interior_threshold)
        self.slide_balance_power = float(slide_balance_power)
        self.max_patch_repeats = int(max_patch_repeats)
        self.seed = int(seed)
        self.epoch = 0
        if self.num_samples < 1 or self.batch_size < 1:
            raise ValueError("num_samples and batch_size must be positive")
        if not 0.0 < self.positive_fraction < 1.0:
            raise ValueError("positive_fraction must be strictly between 0 and 1")
        if not 0.0 < self.boundary_positive_fraction < 1.0:
            raise ValueError(
                "boundary_positive_fraction must be strictly between 0 and 1"
            )
        if not 0.0 < self.interior_threshold <= 1.0:
            raise ValueError("interior_threshold must be in (0,1]")
        if not 0.0 <= self.slide_balance_power <= 1.0:
            raise ValueError("slide_balance_power must be in [0,1]")
        if self.max_patch_repeats < 1:
            raise ValueError("max_patch_repeats must be positive")

        grouped: dict[str, dict[str, list[int]]] = {
            category: defaultdict(list) for category in _CATEGORIES
        }
        for index, row in enumerate(rows):
            slide_id = str(row.get("slide_id", "")).strip()
            if not slide_id:
                raise ValueError(f"row {index} has no slide_id")
            category = patch_stratum(row, interior_threshold=self.interior_threshold)
            grouped[category][slide_id].append(index)
        self._groups = {
            category: {slide: tuple(indices) for slide, indices in slides.items()}
            for category, slides in grouped.items()
        }
        for category in _CATEGORIES:
            if not self._groups[category]:
                raise ValueError(f"sampling stratum {category!r} is empty")

        positive_samples = round(self.num_samples * self.positive_fraction)
        boundary_samples = round(
            positive_samples * self.boundary_positive_fraction
        )
        self._target_counts = {
            NEGATIVE: self.num_samples - positive_samples,
            BOUNDARY: boundary_samples,
            INTERIOR: positive_samples - boundary_samples,
        }
        self._slide_quotas = {
            category: self._allocate_slide_quotas(category)
            for category in _CATEGORIES
        }

    def _allocate_slide_quotas(self, category: str) -> dict[str, int]:
        target = self._target_counts[category]
        slides = self._groups[category]
        capacity = sum(len(indices) * self.max_patch_repeats for indices in slides.values())
        if target > capacity:
            raise ValueError(
                f"Requested {target} {category} samples but repeat cap permits only "
                f"{capacity}; reduce the quota or increase max_patch_repeats"
            )
        # Weighted-fair slots give an integer allocation with an explicit hard
        # capacity.  power=1 approaches patch-uniform sampling; power=0 gives
        # equal WSI opportunity until a slide reaches its repeat cap.
        slots = []
        for slide, indices in slides.items():
            weight = len(indices) ** self.slide_balance_power
            for rank in range(1, len(indices) * self.max_patch_repeats + 1):
                slots.append((rank / weight, slide))
        slots.sort(key=lambda item: (item[0], item[1]))
        quotas = Counter(slide for _, slide in slots[:target])
        return {slide: int(quotas.get(slide, 0)) for slide in slides}

    @property
    def summary(self) -> dict[str, object]:
        return {
            "name": "slide_stratified_boundary",
            "num_samples": self.num_samples,
            "batch_size": self.batch_size,
            "positive_fraction": self.positive_fraction,
            "boundary_positive_fraction": self.boundary_positive_fraction,
            "interior_threshold": self.interior_threshold,
            "slide_balance_power": self.slide_balance_power,
            "max_patch_repeats": self.max_patch_repeats,
            "target_counts": dict(self._target_counts),
            "available_rows": {
                category: sum(map(len, self._groups[category].values()))
                for category in _CATEGORIES
            },
            "eligible_slides": {
                category: len(self._groups[category]) for category in _CATEGORIES
            },
            "slide_quota_range": {
                category: {
                    "minimum": min(self._slide_quotas[category].values()),
                    "maximum": max(self._slide_quotas[category].values()),
                }
                for category in _CATEGORIES
            },
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    @staticmethod
    def _ensure_positive_per_batch(categories: list[str], batch_size: int) -> None:
        for start in range(0, len(categories), batch_size):
            stop = min(start + batch_size, len(categories))
            if any(category != NEGATIVE for category in categories[start:stop]):
                continue
            donor = next(
                (
                    index
                    for index in range(stop, len(categories))
                    if categories[index] != NEGATIVE
                ),
                None,
            )
            if donor is None:
                donor = next(
                    (
                        index
                        for index in range(0, start)
                        if categories[index] != NEGATIVE
                        and sum(
                            item != NEGATIVE
                            for item in categories[
                                (index // batch_size) * batch_size :
                                min(
                                    ((index // batch_size) + 1) * batch_size,
                                    len(categories),
                                )
                            ]
                        )
                        > 1
                    ),
                    None,
                )
            if donor is not None:
                categories[start], categories[donor] = (
                    categories[donor],
                    categories[start],
                )

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        categories = [
            category
            for category in _CATEGORIES
            for _ in range(self._target_counts[category])
        ]
        rng.shuffle(categories)
        self._ensure_positive_per_batch(categories, self.batch_size)

        queues: dict[str, deque[int]] = {}
        for category in _CATEGORIES:
            selected = []
            for slide, quota in self._slide_quotas[category].items():
                source = list(self._groups[category][slide])
                complete_cycles, remainder = divmod(quota, len(source))
                for _ in range(complete_cycles):
                    cycle = list(source)
                    rng.shuffle(cycle)
                    selected.extend(cycle)
                if remainder:
                    cycle = list(source)
                    rng.shuffle(cycle)
                    selected.extend(cycle[:remainder])
            if len(selected) != self._target_counts[category]:
                raise RuntimeError(f"Internal {category} quota allocation mismatch")
            rng.shuffle(selected)
            queues[category] = deque(selected)

        used_in_batch: set[int] = set()
        for position, category in enumerate(categories):
            if position % self.batch_size == 0:
                used_in_batch.clear()
            queue = queues[category]
            candidate = None
            for _ in range(len(queue)):
                proposed = queue.popleft()
                if proposed not in used_in_batch:
                    candidate = proposed
                    break
                queue.append(proposed)
            if candidate is None:
                raise RuntimeError(
                    f"Cannot draw unique {category} samples for batch size "
                    f"{self.batch_size}; reduce the batch quota or add data"
                )
            used_in_batch.add(candidate)
            yield candidate
