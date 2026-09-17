"""Reuse only the immutable DINO prefix, never a trainable node feature.

Disk entries contain canonical global/local tokens BEFORE the first trainable
block. The suffix, final norm and fusion execute afresh with gradients. This
does not change the receptive field, sampling, or cross-rank communication.
"""
from __future__ import annotations

import hashlib
import errno
import io
import json
import os
import shutil
import time

import torch
from torch.utils.checkpoint import checkpoint

from .consistent_features import ConsistentFeatureProvider
from .joint_optim import _vit_blocks


class FrozenDinoPrefix:
    def __init__(self, stage1, trainable_blocks):
        self.stage1 = stage1
        backbone = stage1.backbone
        blocks = _vit_blocks(backbone)
        if not 0 < trainable_blocks < len(blocks):
            raise ValueError("Prefix caching requires a nonempty frozen prefix and trainable suffix")
        self.cut = len(blocks) - trainable_blocks
        self.prefix, self.suffix = blocks[:self.cut], blocks[self.cut:]
        excluded = {id(p) for m in [*self.suffix, backbone.norm, backbone.head]
                    for p in m.parameters()}
        self.parameters = [(n, p) for n, p in backbone.named_parameters() if id(p) not in excluded]
        self.buffers = list(backbone.named_buffers())
        self.modules = [*backbone.patch_embed.modules(),
                        *(m for b in self.prefix for m in b.modules())]
        self.versions = self._versions()
        self.check()
        # Hash the actual frozen weights, not a filename or training step.
        # Different checkpoints may share this cache only if their prefix agrees.
        digest = hashlib.sha256()
        digest.update(repr(backbone).encode())
        digest.update(json.dumps({"cut": self.cut,
            "interpolate_offset": backbone.interpolate_offset,
            "interpolate_antialias": backbone.interpolate_antialias,
            "register_tokens": backbone.num_register_tokens}, sort_keys=True).encode())
        for name, value in [*self.parameters, *self.buffers]:
            cpu = value.detach().cpu().contiguous()
            digest.update(str((name, tuple(cpu.shape), str(cpu.dtype))).encode())
            digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
        self.digest = digest.hexdigest()

    def _versions(self):
        return tuple((id(p), p.data_ptr(), p._version, p.dtype, p.device)
                     for _, p in [*self.parameters, *self.buffers])

    def check(self):
        current = dict(self.stage1.backbone.named_parameters())
        if any(current.get(n) is not p for n, p in self.parameters):
            raise RuntimeError("Frozen DINO prefix parameters were replaced; recreate provider")
        if any(p.requires_grad for _, p in self.parameters):
            raise RuntimeError("Cannot cache a trainable DINO prefix")
        if any(m.training for m in self.modules):
            raise RuntimeError("Cached DINO prefix must remain in eval mode")
        if self._versions() != self.versions:
            raise RuntimeError("Frozen DINO prefix changed; recreate provider to select a new cache")

    def extract(self, images):
        self.check()
        if tuple(images.shape[-2:]) != (self.stage1.image_size, self.stage1.image_size):
            raise ValueError("Prefix cache requires the canonical image size")
        def run(x):
            x = self.stage1.backbone.prepare_tokens_with_masks(x, masks=None)
            for block in self.prefix:
                x = block(x)
            return x
        with torch.no_grad():
            global_tokens = run(images)
            local_tokens = run(self.stage1._local_crops(images)).view(
                self.stage1.num_local_crops, images.size(0), -1,
                self.stage1.embed_dim).transpose(0, 1)
        return global_tokens, local_tokens

    def finish(self, global_tokens, local_tokens):
        # Check also during activation-checkpoint recomputation.
        self.check()
        backbone = self.stage1.backbone
        def run(x):
            for block in self.suffix:
                x = block(x)
            return backbone.norm(x)
        global_output = run(global_tokens)
        local_output = run(local_tokens.transpose(0, 1).flatten(0, 1))
        start = 1 + backbone.num_register_tokens
        raw = (global_output[:, 0], global_output[:, start:],
               local_output[:, start:].reshape(self.stage1.num_local_crops,
                   global_output.size(0), -1, self.stage1.embed_dim).transpose(0, 1))
        return self.stage1.fuse_raw(*raw)


class PrefixCachedFeatureProvider(ConsistentFeatureProvider):
    def __init__(self, *args, trainable_blocks=4, min_free_gib=512, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainable_blocks = int(trainable_blocks)
        self.prefix = None  # Construct only after final-stage freezing is applied.
        self.prefix_hits = self.prefix_misses = 0
        self.minimum_free_bytes = int(min_free_gib * 1024**3)
        self.prefix_write_skips = 0

    def _store_prefix(self, path, value):
        # Cache is optional acceleration. Keep headroom for real checkpoints;
        # low space means recompute, never fall back to stale fused features.
        needed = sum(x.numel() * x.element_size() for x in value)
        if shutil.disk_usage(self.root).free < self.minimum_free_bytes + needed:
            self.prefix_write_skips += 1
            return
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Serialize a single node in memory, then let ordinary file writes
            # report ENOSPC reliably (PyTorch's ZIP writer can mask that errno).
            payload = io.BytesIO()
            torch.save(value, payload)
            with tmp.open("wb") as stream:
                stream.write(payload.getbuffer())
            os.replace(tmp, path)
        except OSError as error:
            if error.errno not in (errno.ENOSPC, errno.EDQUOT):
                raise
            self.prefix_write_skips += 1
        finally:
            tmp.unlink(missing_ok=True)

    def _cached_prefix(self, slide, patches, device):
        if self.prefix is None:
            self.prefix = FrozenDinoPrefix(self.stage1, self.trainable_blocks)
        self.prefix.check()
        numerics = {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        }
        protocol = hashlib.sha256(json.dumps(numerics, sort_keys=True).encode()).hexdigest()[:16]
        # Existing identity contains canonical view/crops/source; _path adds AMP.
        paths = [self.root / "frozen_prefix_v1" / self.prefix.digest /
                 device.type / protocol / self._path(slide, p).relative_to(self.root)
                 for p in patches]
        values = [None] * len(paths)
        missing = []
        for i, path in enumerate(paths):
            if path.is_file():
                values[i] = torch.load(path, map_location="cpu", weights_only=True)
                self.prefix_hits += 1
            else:
                missing.append(i)
                self.prefix_misses += 1
        if missing:
            images = torch.stack([self._image(slide, patches[i]) for i in missing]).to(device)
            raw = self.prefix.extract(images)
            for offset, i in enumerate(missing):
                value = tuple(x[offset].detach().cpu().contiguous() for x in raw)
                path = paths[i]
                self._store_prefix(path, value)
                values[i] = value
        for value in values:
            if (not isinstance(value, (list, tuple)) or len(value) != 2
                    or not all(isinstance(x, torch.Tensor) for x in value)
                    or value[0].ndim != 2 or value[1].ndim != 3
                    or value[0].size(-1) != self.stage1.embed_dim
                    or value[1].size(-1) != self.stage1.embed_dim
                    or value[1].size(0) != self.stage1.num_local_crops):
                raise ValueError("Invalid frozen DINO prefix cache entry")
        return tuple(torch.stack([v[k] for v in values]).to(device) for k in range(2))

    def features(self, slide, patches, device):
        if self.stage != "dino":
            return super().features(slide, patches, device)
        device = torch.device(device)
        outputs = []
        for start in range(0, len(patches), self.chunk_size):
            raw = self._cached_prefix(slide, patches[start:start+self.chunk_size], device)
            fn = self.prefix.finish
            with torch.autograd.graph.save_on_cpu():
                value = checkpoint(fn, *raw, use_reentrant=False) if torch.is_grad_enabled() else fn(*raw)
            outputs.append(value)
        return torch.cat(outputs)
