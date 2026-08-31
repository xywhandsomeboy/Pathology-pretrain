# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0.

"""Batch collation for independent patches and legacy slide samples."""

from __future__ import annotations

from pathlib import Path

import torch


def _stack_crops(samples, crop_name: str, dtype: torch.dtype) -> torch.Tensor:
    crop_count = len(samples[0][crop_name])
    if any(len(sample[crop_name]) != crop_count for sample in samples):
        raise ValueError(f"All samples must contain the same number of {crop_name}")
    if crop_count == 0:
        reference = samples[0]["global_crops"][0]
        return reference.new_empty((0, *reference.shape)).to(dtype)
    # View-major order is required by GCNMetaArch._encode_two_views(), which
    # reshapes this tensor to [num_views, batch, ...].
    return torch.stack(
        [sample[crop_name][view] for view in range(crop_count) for sample in samples]
    ).to(dtype)


def _collate_independent_patches(samples_list, dtype: torch.dtype) -> dict:
    transformed = [sample[0] for sample in samples_list]
    filenames = [str(sample[2]) for sample in samples_list]
    return {
        "collated_global_crops": _stack_crops(transformed, "global_crops", dtype),
        "collated_local_crops": _stack_crops(transformed, "local_crops", dtype),
        "filenames": filenames,
        "patch_ids": [Path(filename).stem for filename in filenames],
    }


def _collate_one_slide(samples_list, dtype: torch.dtype) -> dict:
    if len(samples_list) != 1:
        raise ValueError(
            "Slide-level ImageFolder samples have variable node counts and require "
            "train.batch_size_per_gpu=1"
        )
    slide = samples_list[0]
    transformed = slide["patches"]
    result = {
        "collated_global_crops": _stack_crops(transformed, "global_crops", dtype),
        "collated_local_crops": _stack_crops(transformed, "local_crops", dtype),
        "filenames": list(map(str, slide["filenames"])),
        "patch_ids": list(map(str, slide["patch_ids"])),
        "coords": torch.as_tensor(slide["coords"], dtype=torch.int64),
        "levels": torch.as_tensor(slide["levels"], dtype=torch.int64),
        "slide_ids": [str(slide["slide_id"])] * len(transformed),
    }
    return result


def collate_data_and_cast(samples_list, dtype):
    """Collate either DINO dataset triples or one metadata-rich WSI sample."""
    if not samples_list:
        raise ValueError("Cannot collate an empty batch")
    first = samples_list[0]
    if isinstance(first, dict) and "patches" in first:
        return _collate_one_slide(samples_list, dtype)
    if isinstance(first, (tuple, list)) and len(first) >= 3:
        return _collate_independent_patches(samples_list, dtype)
    raise TypeError(
        "Unsupported Stage-1 sample. Expected (transforms, target, filename) "
        "or a slide dictionary returned by ImageFolder."
    )
