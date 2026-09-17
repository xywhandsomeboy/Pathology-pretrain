"""Canonical graph views with disk-backed, pre-fusion DINO outputs.

Frozen stages use the immutable Stage1B graph inputs for every node. Fusion
training recomputes the entire requested GNN receptive field from cached raw
DINO outputs. DINO training bypasses that cache. No target-only replacement,
cross-batch fused memory, or silent fallback to old neighbors is permitted.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.checkpoint import checkpoint


class ConsistentFeatureProvider:
    def __init__(self, stage1, image_root, cache_root, *, chunk_size=8, stage="frozen"):
        self.stage1 = stage1
        self.image_root = Path(image_root).resolve()
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError("neighbor chunk size must be positive")
        self.stage = stage
        source = Path(stage1.checkpoint_path)
        stat = source.stat()
        self.identity = {
            "format": 1, "checkpoint": str(source.resolve()),
            "checkpoint_size": stat.st_size, "checkpoint_mtime_ns": stat.st_mtime_ns,
            "image_root": str(self.image_root), "image_size": stage1.image_size,
            "local_size": stage1.local_size, "local_crops": stage1.num_local_crops,
            "view": "RGB_bilinear_ImageNet_no_augmentation",
            "raw_layout": ["CLS", "global_patch_tokens", "local_crop_patch_tokens"],
        }
        token = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        self.root = Path(cache_root).resolve() / token
        self.root.mkdir(parents=True, exist_ok=True)
        metadata = self.root / "identity.json"
        if metadata.exists() and json.loads(metadata.read_text()) != self.identity:
            raise ValueError("DINO cache identity mismatch")
        if not metadata.exists():
            tmp = metadata.with_name(f".identity-{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self.identity, indent=2))
            os.replace(tmp, metadata)

    def _image(self, slide, patch):
        # This provider intentionally supports the immutable cervical export
        # layout only. Do not search for, or substitute, a different patch.
        if Path(slide).name != slide or Path(patch).name != patch:
            raise ValueError("Invalid graph patch identity")
        path = self.image_root / slide / "images" / f"{patch}.jpg"
        with Image.open(path) as source:
            image = source.convert("RGB").resize(
                (self.stage1.image_size, self.stage1.image_size), Image.Resampling.BILINEAR)
        value = torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)
        return (value - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]

    def _path(self, slide, patch):
        # Precision is part of the key; FP32/BF16 validation must never reuse
        # tensors produced under a different arithmetic protocol.
        device_type = next(self.stage1.parameters()).device.type
        enabled = torch.is_autocast_enabled() if device_type == "cuda" else torch.is_autocast_cpu_enabled()
        dtype = (torch.get_autocast_gpu_dtype() if device_type == "cuda" else torch.get_autocast_cpu_dtype()) if enabled else torch.float32
        return self.root / str(dtype).replace("torch.", "") / slide / f"{patch}.pt"

    def _cached_raw(self, slide, patches, device):
        values = [None] * len(patches)
        missing = []
        for i, patch in enumerate(patches):
            path = self._path(slide, patch)
            if path.exists():
                values[i] = torch.load(path, map_location="cpu", weights_only=True)
            else:
                missing.append(i)
        if missing:
            if torch.is_grad_enabled() and any(p.requires_grad for p in self.stage1.backbone.parameters()):
                raise RuntimeError("Raw disk cache cannot be populated by a trainable DINO")
            if any(m.training for m in self.stage1.backbone.modules()):
                raise RuntimeError("Cached DINO must be in deterministic eval mode")
            images = torch.stack([self._image(slide, patches[i]) for i in missing]).to(device)
            with torch.no_grad():
                raw = self.stage1.extract_raw(images)
            for offset, i in enumerate(missing):
                value = tuple(x[offset].detach().cpu().contiguous() for x in raw)
                path = self._path(slide, patches[i]); path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
                try:
                    torch.save(value, tmp)
                    os.replace(tmp, path)
                finally:
                    tmp.unlink(missing_ok=True)
                values[i] = value
        for value in values:
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                raise ValueError("Incomplete raw DINO cache entry")
        return tuple(torch.stack([value[k] for value in values]).to(device) for k in range(3))

    def features(self, slide, patches, device):
        outputs = []
        for start in range(0, len(patches), self.chunk_size):
            chunk = patches[start:start + self.chunk_size]
            if self.stage == "fusion":
                raw = self._cached_raw(slide, chunk, device)
                fn = self.stage1.fuse_raw
                with torch.autograd.graph.save_on_cpu():
                    value = checkpoint(fn, *raw, use_reentrant=False) if torch.is_grad_enabled() else fn(*raw)
            elif self.stage == "dino":
                images = torch.stack([self._image(slide, p) for p in chunk]).to(device)
                def fn(x):
                    return self.stage1(x)[0]
                with torch.autograd.graph.save_on_cpu():
                    value = checkpoint(fn, images, use_reentrant=False) if torch.is_grad_enabled() else fn(images)
            else:
                raise ValueError(f"Unexpected graph feature stage: {self.stage}")
            outputs.append(value)
        return torch.cat(outputs)


def configure_repository(repository, system, configuration, *, step=0):
    if configuration.get("graph_feature_policy", "legacy") != "staged_consistent":
        return
    fusion = int(configuration["stage1_partial_unfreeze_step"])
    dino = int(configuration["stage1_final_unfreeze_step"])
    stage = "frozen" if step < fusion else "fusion" if step < dino else "dino"
    provider_class = ConsistentFeatureProvider
    extra = {}
    if configuration.get("cache_frozen_dino_prefix", False):
        from .frozen_prefix import PrefixCachedFeatureProvider
        provider_class = PrefixCachedFeatureProvider
        extra["trainable_blocks"] = int(configuration["stage1_final_unfreeze_blocks"])
    repository.feature_provider = provider_class(
        system.stage1, configuration["node_image_root"], configuration["raw_feature_cache"],
        chunk_size=configuration["neighbor_chunk_size"], stage=stage, **extra)


def update_repository_stage(repository, step, args):
    provider = getattr(repository, "feature_provider", None)
    if provider is not None:
        provider.stage = ("frozen" if step < args.stage1_partial_unfreeze_step else
                          "fusion" if step < args.stage1_final_unfreeze_step else "dino")
