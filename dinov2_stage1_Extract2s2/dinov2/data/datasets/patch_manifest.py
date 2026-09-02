"""Streaming dataset for metadata-rich WSI patch manifests.

Unlike the legacy slide-level ``ImageFolder`` dataset, this dataset returns one
patch per sample.  Stage-1B can therefore use a normal batch size even when a
WSI contains tens of thousands of patches.  Coordinates remain in the CSV and
are joined back to the extracted shards by ``organize_features.py``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class PatchManifest(Dataset):
    """Read independent RGB patches from a CSV containing ``filepath``."""

    def __init__(
        self,
        *,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.manifest_path = Path(root).expanduser().resolve()
        self.transform = transform
        self.target_transform = target_transform
        with self.manifest_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if "filepath" not in (reader.fieldnames or []):
                raise ValueError("Patch manifest must contain a filepath column")
            self.paths = [self._resolve(row["filepath"]) for row in reader]
        if not self.paths:
            raise ValueError(f"Patch manifest contains no rows: {self.manifest_path}")
        missing = [path for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Patch manifest references {len(missing)} missing images; first={missing[0]}"
            )

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.manifest_path.parent / path).resolve()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            sample = self.transform(image) if self.transform is not None else image.copy()
        target = ()
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, str(path)
