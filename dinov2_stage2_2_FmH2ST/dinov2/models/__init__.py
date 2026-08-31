# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from . import gcn as gnn

logger = logging.getLogger("gnn")

def build_model(args):
    args.arch = args.arch.removesuffix("_memeff")
    if "gnn" in args.arch:
        gnn_kwargs = dict(
            num_layer=args.num_layer,
            emb_dim=args.emb_dim,
            JK=args.JK,
            drop_ratio=args.dropout_ratio,
            gnn_type=args.gnn_type,
            edge_dim=args.edge_dim,
            block_chunks=args.block_chunks,
            use_residual=getattr(args, "use_residual", True),
            use_layernorm=getattr(args, "use_layernorm", True),
            edge_injection=getattr(args, "edge_injection", "message_and_attention"),
        )
        
        student = gnn.__dict__[args.arch](**gnn_kwargs,)
        embed_dim = student.emb_dim
    return student, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.gcn)
