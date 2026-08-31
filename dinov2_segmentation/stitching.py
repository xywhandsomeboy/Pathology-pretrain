"""Disk-backed overlap-weighted reconstruction of WSI segmentation masks."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def blending_window(height: int, width: int) -> np.ndarray:
    """Hann blending with a non-zero floor for tissue at slide borders."""
    vertical = np.hanning(height) if height > 2 else np.ones(height)
    horizontal = np.hanning(width) if width > 2 else np.ones(width)
    return np.maximum(np.outer(vertical, horizontal), 1e-3).astype(np.float32)


class DiskBackedSlideStitcher:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        slide_id: str,
        level: int,
        num_classes: int,
        height: int,
        width: int,
    ):
        safe_slide_id = str(slide_id).replace("/", "_").replace("\\", "_")
        self.output_dir = Path(output_dir) / f"{safe_slide_id}_L{level}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shape = (int(height), int(width))
        self.num_classes = int(num_classes)
        self.scores = np.memmap(
            self.output_dir / "score_accumulator.dat",
            dtype=np.float32,
            mode="w+",
            shape=(num_classes, height, width),
        )
        self.weights = np.memmap(
            self.output_dir / "weight_accumulator.dat",
            dtype=np.float32,
            mode="w+",
            shape=(height, width),
        )
        self.scores[:] = 0
        self.weights[:] = 0

    def add(self, probability: np.ndarray, x: int, y: int):
        if probability.ndim != 3 or probability.shape[0] != self.num_classes:
            raise ValueError("probability must be [classes,height,width]")
        patch_height, patch_width = probability.shape[-2:]
        if x < 0 or y < 0 or x + patch_width > self.shape[1] or y + patch_height > self.shape[0]:
            raise ValueError(
                f"Patch ({x},{y},{patch_width},{patch_height}) exceeds slide shape {self.shape}"
            )
        weight = blending_window(patch_height, patch_width)
        self.scores[:, y : y + patch_height, x : x + patch_width] += probability * weight
        self.weights[y : y + patch_height, x : x + patch_width] += weight

    def finalize(self, stripe_height: int = 1024) -> Path:
        self.scores.flush()
        self.weights.flush()
        mask_path = self.output_dir / "segmentation.npy"
        mask = np.lib.format.open_memmap(
            mask_path, mode="w+", dtype=np.uint16, shape=self.shape
        )
        for start in range(0, self.shape[0], stripe_height):
            stop = min(start + stripe_height, self.shape[0])
            weight = np.asarray(self.weights[start:stop]).clip(min=1e-8)
            score = np.asarray(self.scores[:, start:stop]) / weight[None]
            prediction = score.argmax(axis=0).astype(np.uint16)
            prediction[np.asarray(self.weights[start:stop]) == 0] = 0
            mask[start:stop] = prediction
        mask.flush()
        return mask_path
