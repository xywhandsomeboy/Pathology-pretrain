# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0.

"""Training API with lazy entry imports so ``python -m`` stays unambiguous."""

from .ssl_meta_arch import SSLMetaArch
from .gcn_meta_arch import GCNMetaArch


def get_args_parser(*args, **kwargs):
    from .train import get_args_parser as implementation

    return implementation(*args, **kwargs)


def main(*args, **kwargs):
    from .train import main as implementation

    return implementation(*args, **kwargs)


__all__ = ["GCNMetaArch", "SSLMetaArch", "get_args_parser", "main"]
