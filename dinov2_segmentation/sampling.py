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


class WSILocalStratifiedSampler(SlideStratifiedSampler):
    """Keep the balanced population, then pack targets into local WSI regions.

    ``SlideStratifiedSampler`` controls *which* patches occur in an epoch.  This
    sampler leaves that population, its stratum counts, and per-patch repeat cap
    unchanged, but reorders it so a loader batch normally comes from one spatial
    run on a single WSI.  A positive item is swapped into an all-negative run
    when necessary, so the foreground-loss contract remains true.  The small
    number of such swaps is intentional: it retains a useful loss on normal
    slides while still making the graph receptive fields strongly overlap.

    The graph code remains responsible for constructing each target set's full
    k-hop closure.  This class changes I/O locality only; it does not introduce
    historical neighbour embeddings or a persistent fusion-feature cache.
    """

    def __init__(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        locality_tile_size: int = 4096,
        global_batch_size: int | None = None,
        **kwargs,
    ) -> None:
        self.locality_tile_size = int(locality_tile_size)
        if self.locality_tile_size < 1:
            raise ValueError("locality_tile_size must be positive")
        self._row_slide_ids: list[str] = []
        self._coordinates: list[tuple[int, int]] = []
        for index, row in enumerate(rows):
            slide_id = str(row.get("slide_id", "")).strip()
            if not slide_id:
                raise ValueError(f"row {index} has no slide_id")
            try:
                coordinate = (int(row["x"]), int(row["y"]))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "WSI-local sampling requires integer x and y manifest columns"
                ) from error
            self._row_slide_ids.append(slide_id)
            self._coordinates.append(coordinate)
        self._rows = rows
        super().__init__(rows, **kwargs)
        self.global_batch_size = int(global_batch_size or self.batch_size)
        if (self.global_batch_size < self.batch_size
                or self.global_batch_size % self.batch_size):
            raise ValueError(
                "global_batch_size must be a positive multiple of batch_size"
            )

    @property
    def summary(self) -> dict[str, object]:
        result = dict(super().summary)
        result.update(
            {
                "name": "wsi_local_stratified_boundary",
                "locality_tile_size": self.locality_tile_size,
                "global_batch_size": self.global_batch_size,
                "effective_num_samples": len(self),
                "locality_contract": (
                    "same-WSI spatial target runs; exact full k-hop closure "
                    "is still recomputed by the graph repository"
                ),
            }
        )
        return result

    def __len__(self) -> int:
        # Match the existing DDP wrapper's tail-drop rule here, before local
        # batches are rearranged.  Serial use has global_batch_size=batch_size
        # and therefore keeps the ordinary final partial batch.
        if self.global_batch_size == self.batch_size:
            return self.num_samples
        return self.num_samples - self.num_samples % self.global_batch_size

    def _spatial_key(self, index: int) -> tuple[int, int, int, int]:
        x, y = self._coordinates[index]
        tile = self.locality_tile_size
        return (y // tile, x // tile, y, x)

    def _slide_chunks(
        self, selected: Sequence[int], rng: random.Random
    ) -> list[list[int]]:
        """Build unique, spatially contiguous chunks for each selected WSI."""

        by_slide: dict[str, Counter[int]] = defaultdict(Counter)
        for index in selected:
            by_slide[self._row_slide_ids[index]][index] += 1

        chunks: list[list[int]] = []
        for slide_id in sorted(by_slide):
            remaining = by_slide[slide_id]
            ordered = sorted(remaining, key=self._spatial_key)
            # Vary the seam each epoch without breaking local coordinate order.
            if len(ordered) > 1:
                offset = rng.randrange(len(ordered))
                ordered = ordered[offset:] + ordered[:offset]
            while sum(remaining.values()):
                chunk: list[int] = []
                for index in ordered:
                    if remaining[index] == 0:
                        continue
                    # A repeated patch must never appear twice in one target
                    # batch, even when the repeat cap is active.
                    chunk.append(index)
                    remaining[index] -= 1
                    if len(chunk) == self.batch_size:
                        break
                if not chunk:
                    raise RuntimeError("Unable to make a non-empty WSI-local chunk")
                chunks.append(chunk)
        rng.shuffle(chunks)
        return chunks

    @staticmethod
    def _contains_positive(
        batch: Sequence[int], rows: Sequence[Mapping[str, object]], threshold: float
    ) -> bool:
        return any(patch_stratum(rows[index], interior_threshold=threshold) != NEGATIVE for index in batch)

    def _make_positive_batches(
        self,
        chunks: Sequence[list[int]],
        rows: Sequence[Mapping[str, object]],
    ) -> list[list[int]]:
        """Pack partial chunks and retain at least one positive per full batch."""

        full = [list(chunk) for chunk in chunks if len(chunk) == self.batch_size]
        partial = [list(chunk) for chunk in chunks if len(chunk) < self.batch_size]
        batches = full
        # A slide can have fewer unique selected patches than a batch because
        # its repeat cap is active.  Do not concatenate that slide's first and
        # second pass into one batch: rotate a repeated item to the tail until
        # another partial WSI fills the vacant position.
        pending: deque[int] = deque(
            value for chunk in partial for value in chunk
        )
        while pending:
            batch: list[int] = []
            used: set[int] = set()
            attempts_without_draw = 0
            while pending and len(batch) < self.batch_size:
                value = pending.popleft()
                if value in used:
                    pending.append(value)
                    attempts_without_draw += 1
                    if attempts_without_draw >= len(pending):
                        break
                    continue
                batch.append(value)
                used.add(value)
                attempts_without_draw = 0
            if not batch:
                raise RuntimeError("Unable to pack unique WSI-local targets")
            batches.append(batch)

        # The last serial batch may be incomplete; the distributed wrapper drops
        # it where needed.  Full batches retain the base sampler's positive
        # guarantee with one carefully checked swap.
        for target_index, target in enumerate(batches):
            if len(target) < self.batch_size or self._contains_positive(
                target, rows, self.interior_threshold
            ):
                continue
            target_set = set(target)
            donor_index = next(
                (
                    source_index
                    for source_index, source in enumerate(batches)
                    if source_index != target_index
                    and sum(
                        patch_stratum(rows[value], interior_threshold=self.interior_threshold)
                        != NEGATIVE
                        for value in source
                    ) > 1
                    and any(
                        patch_stratum(rows[value], interior_threshold=self.interior_threshold)
                        != NEGATIVE
                        and value not in target_set
                        for value in source
                    )
                    and any(value not in set(source) for value in target)
                ),
                None,
            )
            if donor_index is None:
                raise RuntimeError("Unable to retain a positive sample in every batch")
            donor = batches[donor_index]
            donor_set = set(donor)
            donor_position = next(
                position
                for position, value in enumerate(donor)
                if patch_stratum(rows[value], interior_threshold=self.interior_threshold)
                != NEGATIVE
                and value not in target_set
            )
            target_position = next(
                position
                for position, value in enumerate(target)
                if patch_stratum(rows[value], interior_threshold=self.interior_threshold)
                == NEGATIVE
                and donor[donor_position] not in target_set
                and value not in donor_set
            )
            target[target_position], donor[donor_position] = (
                donor[donor_position],
                target[target_position],
            )
        return batches

    def _group_disjoint_global_batches(
        self, batches: Sequence[list[int]]
    ) -> list[list[int]]:
        """Keep intentional repeats out of one DDP optimizer update.

        The distributed wrapper consumes consecutive local batches as one
        global batch.  Reorder full local batches so ranks in that update do
        not receive the same target patch.  This preserves the original
        sampler's no-repeat-in-a-global-batch invariant without widening the
        local receptive field.
        """

        stream = [value for batch in batches for value in batch][: len(self)]
        local_batches = [
            stream[start : start + self.batch_size]
            for start in range(0, len(stream), self.batch_size)
        ]
        ranks = self.global_batch_size // self.batch_size
        full = deque(
            batch for batch in local_batches if len(batch) == self.batch_size
        )
        tail = [batch for batch in local_batches if len(batch) < self.batch_size]
        ordered: list[list[int]] = []
        while full:
            group: list[list[int]] = []
            used: set[int] = set()
            attempts = len(full)
            while full and len(group) < ranks and attempts:
                candidate = full.popleft()
                attempts -= 1
                if set(candidate).isdisjoint(used):
                    group.append(candidate)
                    used.update(candidate)
                else:
                    full.append(candidate)
            # A serial final partial batch is retained.  In DDP the explicit
            # length truncation above makes this branch unreachable.
            if len(group) < ranks and full:
                raise RuntimeError(
                    "Unable to form a duplicate-free distributed target batch"
                )
            ordered.extend(group)
        ordered.extend(tail)
        return ordered

    def __iter__(self) -> Iterator[int]:
        # First choose exactly the same balanced population as the established
        # sampler.  Reordering afterwards makes this a locality ablation, not a
        # data-population change.
        selected = list(super().__iter__())
        rng = random.Random(self.seed + self.epoch + 1_000_003)
        chunks = self._slide_chunks(selected, rng)
        batches = self._make_positive_batches(chunks, self._rows)
        for batch in self._group_disjoint_global_batches(batches):
            if len(batch) == self.batch_size and len(batch) != len(set(batch)):
                raise RuntimeError("WSI-local sampler produced a duplicate target")
            yield from batch
