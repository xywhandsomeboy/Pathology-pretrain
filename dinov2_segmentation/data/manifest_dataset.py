"""Manifest-based aligned raw-patch/mask/feature dataset."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .feature_store import FeatureStoreCache, make_patch_key


REQUIRED_COLUMNS = {
    "slide_id",
    "patch_id",
    "x",
    "y",
    "image_path",
    "feature_path",
}


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    array = np.asarray(mask, dtype=np.int64).copy()
    if array.ndim == 3:
        # Class-index masks should be single-channel. RGB masks require an
        # explicit color-to-class preprocessing step rather than an unsafe sum.
        if not np.array_equal(array[..., 0], array[..., 1]) or not np.array_equal(
            array[..., 0], array[..., 2]
        ):
            raise ValueError("RGB mask has different channels; convert it to class indices first")
        array = array[..., 0]
    return torch.from_numpy(array)


def _flip_dense_tokens(tokens: torch.Tensor, horizontal: bool, vertical: bool) -> torch.Tensor:
    if not horizontal and not vertical:
        return tokens
    if tokens.ndim == 2:
        side = math.isqrt(tokens.shape[0])
        if side * side != tokens.shape[0]:
            raise ValueError("Cannot geometrically flip a non-square DINO token sequence")
        grid = tokens.reshape(side, side, -1)
        if horizontal:
            grid = grid.flip(1)
        if vertical:
            grid = grid.flip(0)
        return grid.reshape_as(tokens)
    if tokens.ndim == 3:
        # Feature-store records are either [C,H,W] or [H,W,C]. The token
        # channel is normally the largest dimension for DINO ViT features.
        if tokens.shape[0] > tokens.shape[-1]:
            dimensions = ([2] if horizontal else []) + ([1] if vertical else [])
        else:
            dimensions = ([1] if horizontal else []) + ([0] if vertical else [])
        return tokens.flip(dimensions) if dimensions else tokens
    raise ValueError(f"Unsupported per-patch dense token shape {tuple(tokens.shape)}")


class PatchSegmentationDataset(Dataset):
    """Load corresponding branch-1 features and branch-2 raw patch by key."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        image_size: int | tuple[int, int] | None = 224,
        training: bool = False,
        require_mask: bool = True,
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.5,
        cache_size: int = 2,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        check_paths: bool = True,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        self.base = self.manifest_path.parent
        self.training = training
        self.require_mask = require_mask
        self.horizontal_flip_probability = horizontal_flip_probability if training else 0.0
        self.vertical_flip_probability = vertical_flip_probability if training else 0.0
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = tuple(image_size) if image_size is not None else None
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]
        self.feature_cache = FeatureStoreCache(cache_size)

        with self.manifest_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_COLUMNS.difference(columns))
            if missing:
                raise ValueError(f"Manifest is missing columns: {missing}")
            if require_mask and "mask_path" not in columns:
                raise ValueError("Training/validation manifest requires mask_path")
            self.rows = [dict(row) for row in reader]
        if not self.rows:
            raise ValueError(f"Manifest {self.manifest_path} contains no patches")

        seen = set()
        for row in self.rows:
            row["x"], row["y"] = int(row["x"]), int(row["y"])
            row["level"] = int(row.get("level") or 0)
            row["feature_index"] = (
                int(row["feature_index"]) if row.get("feature_index", "") != "" else None
            )
            row["image_path"] = _resolve(self.base, row["image_path"])
            row["feature_path"] = _resolve(self.base, row["feature_path"])
            if row.get("mask_path"):
                row["mask_path"] = _resolve(self.base, row["mask_path"])
            key = make_patch_key(
                row["slide_id"], row["patch_id"], row["x"], row["y"], row["level"]
            )
            if key in seen:
                raise ValueError(f"Duplicate patch identity in manifest: {key}")
            seen.add(key)
            if check_paths:
                for field in ("image_path", "feature_path"):
                    if not row[field].is_file():
                        raise FileNotFoundError(f"{field} does not exist: {row[field]}")
                if require_mask:
                    mask_path = row.get("mask_path")
                    if not isinstance(mask_path, Path) or not mask_path.is_file():
                        raise FileNotFoundError(f"mask_path does not exist: {mask_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        original_size = (image.height, image.width)
        mask = Image.open(row["mask_path"]) if row.get("mask_path") else None
        if mask is not None and mask.size != image.size:
            raise ValueError(
                f"Image/mask size mismatch for {row['patch_id']}: {image.size} vs {mask.size}"
            )
        if self.image_size is not None:
            height, width = self.image_size
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            if mask is not None:
                mask = mask.resize((width, height), Image.Resampling.NEAREST)

        store = self.feature_cache.get(row["feature_path"])
        feature = store.get(
            slide_id=row["slide_id"],
            patch_id=row["patch_id"],
            x=row["x"],
            y=row["y"],
            level=row["level"],
            feature_index=row["feature_index"],
        )
        dense_tokens = feature["dense_tokens"].float()
        horizontal = random.random() < self.horizontal_flip_probability
        vertical = random.random() < self.vertical_flip_probability
        if horizontal:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if mask is not None:
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if vertical:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if mask is not None:
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        dense_tokens = _flip_dense_tokens(dense_tokens, horizontal, vertical)

        image_tensor = (_image_to_tensor(image) - self.mean) / self.std
        sample = {
            "image": image_tensor,
            "dense_tokens": dense_tokens,
            "global_context": feature["global_context"].float(),
            "slide_id": row["slide_id"],
            "patch_id": row["patch_id"],
            "patch_key": feature["patch_key"],
            "coords": torch.tensor((row["x"], row["y"]), dtype=torch.int64),
            "level": row["level"],
            "original_size": torch.tensor(original_size, dtype=torch.int64),
        }
        if mask is not None:
            sample["mask"] = _mask_to_tensor(mask).long()
        return sample
