"""Metadata-driven WSI patch dataset retained for slide-level experiments.

The active Stage-1A/Stage-1B path normally uses ``ImageNet22k`` and treats
patches independently. This dataset remains available when one slide must be
loaded as a sample, but now uses the same patch identity contract as the
segmentation pipeline and no longer depends on an opaque pickle cache.
"""

from __future__ import annotations

import csv
import random
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


_DEFAULT_MAX_PATCHES = 50
_REQUIRED_COLUMNS = {"slide_name_split", "filepath", "x", "y"}


def _resolve_metadata(root: Path, metadata_csv: str | None) -> Path:
    if metadata_csv:
        path = Path(metadata_csv).expanduser()
        return path if path.is_absolute() else (root / path).resolve()
    if root.is_file():
        return root.resolve()
    preferred = root / "patch_grid_positions-TCGA_CESC-hospital.csv"
    if preferred.is_file():
        return preferred.resolve()
    candidates = sorted(root.glob("patch_grid_positions*.csv"))
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(
            f"No patch_grid_positions*.csv found under {root}. "
            "Pass ImageFolder:root=/path/to/metadata.csv explicitly."
        )
    raise ValueError(
        f"Multiple metadata CSV files found under {root}; pass metadata_csv explicitly"
    )


class ImageFolder(Dataset):
    """Return one slide and its ordered, transformed patches per sample."""

    def __init__(
        self,
        *,
        root: str,
        transform: Optional[Callable] = None,
        max_patches: int | str = _DEFAULT_MAX_PATCHES,
        sampling: str = "random",
        metadata_csv: str | None = None,
        patch_root: str | None = None,
    ) -> None:
        root_path = Path(root).expanduser()
        self.metadata_path = _resolve_metadata(root_path, metadata_csv)
        self.transform = transform
        self.max_patches = int(max_patches)
        self.sampling = str(sampling).lower()
        if self.max_patches < 0:
            raise ValueError("max_patches must be >= 0; use 0 to keep every patch")
        if self.sampling not in {"random", "ordered"}:
            raise ValueError("sampling must be 'random' or 'ordered'")
        self.patch_root = Path(patch_root).expanduser().resolve() if patch_root else None

        with self.metadata_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            missing = sorted(_REQUIRED_COLUMNS.difference(reader.fieldnames or []))
            if missing:
                raise ValueError(f"Patch metadata is missing columns: {missing}")
            grouped: OrderedDict[str, list[dict]] = OrderedDict()
            for row in reader:
                slide_id = str(row["slide_name_split"])
                grouped.setdefault(slide_id, []).append(dict(row))
        if not grouped:
            raise ValueError(f"Patch metadata contains no rows: {self.metadata_path}")
        self._slides = grouped
        self.slide_names = list(grouped)

    def __len__(self) -> int:
        return len(self.slide_names)

    def _image_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        candidates = [path]
        if not path.is_absolute():
            candidates.append(self.metadata_path.parent / path)
        if self.patch_root is not None:
            candidates.extend((self.patch_root / path.name, self.patch_root / path))
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            f"Patch image from metadata does not exist: {raw_path}. "
            "If the dataset moved, pass patch_root in the dataset string."
        )

    def __getitem__(self, index: int) -> dict:
        slide_id = self.slide_names[index]
        rows = list(self._slides[slide_id])
        if self.max_patches and len(rows) > self.max_patches:
            rows = (
                random.sample(rows, self.max_patches)
                if self.sampling == "random"
                else rows[: self.max_patches]
            )

        patches, filenames, patch_ids, coords, levels = [], [], [], [], []
        for row in rows:
            path = self._image_path(row["filepath"])
            with Image.open(path) as image:
                image = image.convert("RGB")
                patches.append(self.transform(image) if self.transform else image.copy())
            filenames.append(str(path))
            patch_ids.append(
                str(row.get("patch_id") or row.get("patch_idx") or path.stem)
            )
            coords.append((int(float(row["x"])), int(float(row["y"]))))
            levels.append(int(row.get("level") or 0))

        if len(set(patch_ids)) != len(patch_ids):
            raise ValueError(f"patch_ids are not unique inside slide {slide_id}")
        return {
            "patches": patches,
            "filenames": filenames,
            "patch_ids": patch_ids,
            "coords": coords,
            "levels": levels,
            "slide_id": slide_id,
        }
