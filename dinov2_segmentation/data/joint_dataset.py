"""Raw patch/mask dataset used by end-to-end Stage1/Stage2 fine-tuning."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


REQUIRED_COLUMNS = {
    "slide_id",
    "patch_id",
    "image_path",
    "mask_path",
    "x",
    "y",
    "level",
}


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


class JointPatchSegmentationDataset(Dataset):
    """Load aligned raw patches and binary masks without cached model features.

    Stage1 and Stage2 features must be produced in the training process so that
    the segmentation loss retains a gradient path into both pretrained stages.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        image_size: int = 224,
        training: bool = False,
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.5,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        check_paths: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.base = self.manifest_path.parent
        self.image_size = int(image_size)
        self.training = bool(training)
        self.horizontal_flip_probability = (
            float(horizontal_flip_probability) if training else 0.0
        )
        self.vertical_flip_probability = (
            float(vertical_flip_probability) if training else 0.0
        )
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]

        with self.manifest_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_COLUMNS.difference(columns))
            if missing:
                raise ValueError(f"Joint manifest is missing columns: {missing}")
            self.rows = [dict(row) for row in reader]
        if not self.rows:
            raise ValueError(f"Manifest contains no patches: {self.manifest_path}")

        identities = set()
        for row in self.rows:
            row["image_path"] = _resolve(self.base, row["image_path"])
            row["mask_path"] = _resolve(self.base, row["mask_path"])
            row["x"] = int(row["x"])
            row["y"] = int(row["y"])
            row["level"] = int(row["level"])
            identity = (row["slide_id"], row["patch_id"], row["level"], row["x"], row["y"])
            if identity in identities:
                raise ValueError(f"Duplicate joint-training patch identity: {identity}")
            identities.add(identity)
            if check_paths:
                for field in ("image_path", "mask_path"):
                    if not row[field].is_file():
                        raise FileNotFoundError(f"{field} does not exist: {row[field]}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        with Image.open(row["mask_path"]) as source:
            mask = source.convert("L")
        if image.size != mask.size:
            raise ValueError(
                f"Image/mask size mismatch for {row['patch_id']}: "
                f"{image.size} vs {mask.size}"
            )
        if image.size != (self.image_size, self.image_size):
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            mask = mask.resize(
                (self.image_size, self.image_size), Image.Resampling.NEAREST
            )

        if random.random() < self.horizontal_flip_probability:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < self.vertical_flip_probability:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        image_tensor = (image_tensor - self.mean) / self.std
        mask_array = np.asarray(mask, dtype=np.uint8).copy()
        unique = set(np.unique(mask_array).tolist())
        if not unique.issubset({0, 1}):
            raise ValueError(
                f"Expected a binary 0/1 mask for {row['patch_id']}, got {sorted(unique)}"
            )
        return {
            "image": image_tensor,
            "mask": torch.from_numpy(mask_array.astype(np.int64, copy=False)),
            "slide_id": str(row["slide_id"]),
            "patch_id": str(row["patch_id"]),
            "coords": torch.tensor((row["x"], row["y"]), dtype=torch.int64),
            "level": row["level"],
        }
