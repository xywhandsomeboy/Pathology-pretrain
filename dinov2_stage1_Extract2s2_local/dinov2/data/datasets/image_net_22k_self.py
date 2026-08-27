# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from gzip import GzipFile
from io import BytesIO
from mmap import ACCESS_READ, mmap
import os
from typing import Any, Callable, List, Optional, Set, Tuple
import warnings

import numpy as np

from .extended import ExtendedVisionDataset
from .image_net_22k import ImageNet22k  # 继承原类

_Labels = int

_DEFAULT_MMAP_CACHE_SIZE = 16  # Warning: This can exhaust file descriptors


@dataclass
class _ClassEntry:
    block_offset: int
    maybe_filename: Optional[str] = None


@dataclass
class _Entry:
    class_index: int  # noqa: E701
    start_offset: int
    end_offset: int
    filename: str


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"

    @property
    def length(self) -> int:
        return {
            _Split.TRAIN: 2_299_631,
            _Split.VAL: 561_050,
        }[self]

    def entries_path(self):
        return f"imagenet21kp_{self.value}.txt"


def _get_tarball_path(class_id: str) -> str:
    return f"{class_id}.tar"


def _make_mmap_tarball(tarballs_root: str, mmap_cache_size: int):
    @lru_cache(maxsize=mmap_cache_size)
    def _mmap_tarball(class_id: str) -> mmap:
        tarball_path = _get_tarball_path(class_id)
        tarball_full_path = os.path.join(tarballs_root, tarball_path)
        with open(tarball_full_path) as f:
            return mmap(fileno=f.fileno(), length=0, access=ACCESS_READ)

    return _mmap_tarball


class ImageNet22kSelf(ImageNet22k):
    def __init__(self, root, split, extra, transform=None):
        """
        root: ImageFolder 数据集的根目录
        split: "train" / "val" 等（可以不严格用）
        extra: 生成的 entries.npy 和 class-ids.npy 的目录
        transform: 图像变换
        """
        self.root = root
        super().__init__(split=split, extra=extra, transform=transform)

    def get_image_data(self, index):
        """重写读取方式：直接从磁盘加载而不是 tar"""
        entry = self._entries[index]
        img_path = os.path.join(self.root, entry["class_id"], entry["filename"])
        with open(img_path, "rb") as f:
            return f.read()

    def __getitem__(self, index):
        """按 ImageNet22k 接口返回 (image, target)"""
        img_data = self.get_image_data(index)
        img = Image.open(os.BytesIO(img_data)).convert("RGB")
        target = self._entries[index]["class_index"]
        if self.transform is not None:
            img = self.transform(img)
        return img, target
