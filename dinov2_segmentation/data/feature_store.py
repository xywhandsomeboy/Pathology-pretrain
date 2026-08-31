"""Validated, memory-mapped per-slide cache for the segmentation branches."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


FORMAT_VERSION = 2
SUPPORTED_FORMATS = {1, 2}


def make_patch_key(slide_id: str, patch_id: str, x: int, y: int, level: int = 0) -> str:
    """Stable identity shared by raw image, DINO tokens, GNN output and mask."""
    return f"{slide_id}|L{int(level)}|X{int(x)}|Y{int(y)}|{patch_id}"


def _as_tensor(value, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    try:
        return torch.as_tensor(value).contiguous()
    except Exception as error:
        raise TypeError(f"{name} cannot be converted to a tensor") from error


def _validate_dense_shape(value, count: int, name="dense_tokens"):
    if value.ndim not in (3, 4) or value.shape[0] != count:
        raise ValueError(
            f"{name} must be [N,T,C], [N,C,H,W], or [N,H,W,C] with N={count}"
        )


def _validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required = {
        "format_version",
        "slide_id",
        "patch_ids",
        "coords",
        "levels",
        "global_context",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Feature store is missing fields: {missing}")
    version = int(payload["format_version"])
    if version not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported feature format {version}; supported versions are {SUPPORTED_FORMATS}"
        )
    if version == 1 and "dense_tokens" not in payload:
        raise ValueError("Version-1 feature store is missing dense_tokens")
    if version == 2 and "dense_tokens_file" not in payload:
        raise ValueError("Version-2 feature store is missing dense_tokens_file")

    patch_ids = [str(value) for value in payload["patch_ids"]]
    coords = _as_tensor(payload["coords"], "coords").to(torch.int64)
    levels = _as_tensor(payload["levels"], "levels").to(torch.int64)
    global_context = _as_tensor(payload["global_context"], "global_context")
    count = len(patch_ids)
    if coords.shape != (count, 2):
        raise ValueError(f"coords must be [{count},2], got {tuple(coords.shape)}")
    if levels.shape != (count,):
        raise ValueError(f"levels must be [{count}], got {tuple(levels.shape)}")
    if global_context.ndim != 2 or global_context.shape[0] != count:
        raise ValueError("global_context must be [N,C] with the same N as patch_ids")
    if len(set(patch_ids)) != count:
        raise ValueError("patch_ids must be unique inside one slide feature store")
    coordinate_ids = [
        (int(level), int(x), int(y))
        for level, (x, y) in zip(levels.tolist(), coords.tolist())
    ]
    if len(set(coordinate_ids)) != count:
        raise ValueError("(level,x,y) coordinates must be unique inside one slide")

    payload = dict(payload)
    payload.update(
        format_version=version,
        slide_id=str(payload["slide_id"]),
        patch_ids=patch_ids,
        coords=coords,
        levels=levels,
        global_context=global_context,
    )
    if version == 1:
        dense_tokens = _as_tensor(payload["dense_tokens"], "dense_tokens")
        _validate_dense_shape(dense_tokens, count)
        payload["dense_tokens"] = dense_tokens
    else:
        payload["dense_tokens_file"] = str(payload["dense_tokens_file"])
    return payload


def save_slide_feature_store(
    output_path: str | Path,
    *,
    slide_id: str,
    patch_ids: Sequence[str],
    coords,
    dense_tokens,
    global_context,
    levels=None,
    node_features=None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save metadata/context and externalize dense tokens as a NumPy sidecar.

    The sidecar is opened with ``mmap_mode='r'`` by workers, so a 30k-patch WSI
    does not make every DataLoader process load all DINO tokens into RAM.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = len(patch_ids)
    if levels is None:
        levels = torch.zeros(count, dtype=torch.int64)
    if isinstance(dense_tokens, np.ndarray):
        dense_array = dense_tokens
    else:
        dense_array = _as_tensor(dense_tokens, "dense_tokens").numpy()
    _validate_dense_shape(dense_array, count)
    dense_path = output_path.with_name(f"{output_path.stem}.dense_tokens.npy")
    payload = {
        "format_version": FORMAT_VERSION,
        "slide_id": str(slide_id),
        "patch_ids": list(patch_ids),
        "coords": coords,
        "levels": levels,
        "dense_tokens_file": dense_path.name,
        "global_context": global_context,
        "metadata": dict(metadata or {}),
    }
    if node_features is not None:
        node_features = _as_tensor(node_features, "node_features")
        if node_features.ndim != 2 or node_features.shape[0] != count:
            raise ValueError("node_features must be [N,C] with the same N as patch_ids")
        payload["node_features"] = node_features
    payload = _validate_payload(payload)

    dense_temporary = dense_path.with_suffix(".npy.tmp")
    with dense_temporary.open("wb") as stream:
        np.save(stream, dense_array, allow_pickle=False)
    dense_temporary.replace(dense_path)
    payload_temporary = output_path.with_suffix(".pt.tmp")
    torch.save(payload, payload_temporary)
    payload_temporary.replace(output_path)
    return output_path


def _safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class SlideFeatureStore:
    """Read-only, validated access to one WSI feature store."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        payload = _safe_torch_load(self.path)
        if not isinstance(payload, dict):
            raise TypeError(f"{self.path} is not a feature-store dictionary")
        self.payload = _validate_payload(payload)
        self.slide_id = self.payload["slide_id"]
        count = len(self.payload["patch_ids"])
        if self.payload["format_version"] == 1:
            self._dense_tokens = self.payload["dense_tokens"]
        else:
            dense_path = self.path.parent / self.payload["dense_tokens_file"]
            if not dense_path.is_file():
                raise FileNotFoundError(f"Dense-token sidecar does not exist: {dense_path}")
            self._dense_tokens = np.load(dense_path, mmap_mode="r", allow_pickle=False)
            _validate_dense_shape(self._dense_tokens, count, "dense-token sidecar")
        self._by_patch_id = {
            patch_id: index for index, patch_id in enumerate(self.payload["patch_ids"])
        }

    def __len__(self) -> int:
        return len(self.payload["patch_ids"])

    def index_of(self, patch_id: str) -> int:
        try:
            return self._by_patch_id[str(patch_id)]
        except KeyError as error:
            raise KeyError(f"patch_id={patch_id!r} is absent from {self.path}") from error

    def get(
        self,
        *,
        slide_id: str,
        patch_id: str,
        x: int,
        y: int,
        level: int = 0,
        feature_index: int | None = None,
    ) -> dict[str, Any]:
        if str(slide_id) != self.slide_id:
            raise ValueError(
                f"manifest slide_id={slide_id!r} does not match store slide_id={self.slide_id!r}"
            )
        index = self.index_of(patch_id) if feature_index is None else int(feature_index)
        if index < 0 or index >= len(self):
            raise IndexError(f"feature_index={index} is outside [0,{len(self)})")
        stored_patch_id = self.payload["patch_ids"][index]
        stored_x, stored_y = self.payload["coords"][index].tolist()
        stored_level = int(self.payload["levels"][index])
        expected = make_patch_key(slide_id, patch_id, x, y, level)
        actual = make_patch_key(
            self.slide_id, stored_patch_id, stored_x, stored_y, stored_level
        )
        if actual != expected:
            raise ValueError(
                "Raw patch and cached features are misaligned: "
                f"manifest={expected!r}, feature_store={actual!r}"
            )
        dense_tokens = self._dense_tokens[index]
        if not isinstance(dense_tokens, torch.Tensor):
            dense_tokens = torch.from_numpy(np.array(dense_tokens, copy=True))
        return {
            "dense_tokens": dense_tokens,
            "global_context": self.payload["global_context"][index],
            "feature_index": index,
            "patch_key": actual,
        }


class FeatureStoreCache:
    """Small per-worker LRU cache so adjacent patches reuse WSI metadata/maps."""

    def __init__(self, capacity: int = 2):
        if capacity < 1:
            raise ValueError("cache capacity must be positive")
        self.capacity = capacity
        self._stores: OrderedDict[str, SlideFeatureStore] = OrderedDict()

    def get(self, path: str | Path) -> SlideFeatureStore:
        key = str(Path(path).resolve())
        if key in self._stores:
            self._stores.move_to_end(key)
            return self._stores[key]
        store = SlideFeatureStore(key)
        self._stores[key] = store
        if len(self._stores) > self.capacity:
            self._stores.popitem(last=False)
        return store
