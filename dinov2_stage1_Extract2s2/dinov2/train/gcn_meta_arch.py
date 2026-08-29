# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import math

import torch
from torch import nn

from dinov2.models import build_model_from_cfg
from dinov2.utils.utils import has_batchnorms
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model

from dinov2.models.vision_transformer import BlockChunk


logger = logging.getLogger("dinov2")


class SpatialPatchAggregator(nn.Module):
    """Preserve the 2-D organization of raw DINO patch tokens."""

    def __init__(self, in_dim, hidden_dim=256, out_dim=1024):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1, groups=hidden_dim, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1, groups=hidden_dim, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.GELU(),
        )
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, patch_tokens):
        batch_size, num_tokens, channels = patch_tokens.shape
        side = math.isqrt(num_tokens)
        if side * side != num_tokens:
            raise ValueError(f"DINO token count {num_tokens} is not a square grid")
        x = patch_tokens.transpose(1, 2).reshape(batch_size, channels, side, side)
        x = self.spatial(self.compress(x)).mean(dim=(-2, -1))
        return self.proj(x)


class NodeFeatureFusion(nn.Module):
    def __init__(self, global_dim, out_dim, spatial_scale_init=0.1):
        super().__init__()
        self.global_proj = nn.Identity() if global_dim == out_dim else nn.Linear(global_dim, out_dim)
        self.spatial_scale = nn.Parameter(torch.tensor(float(spatial_scale_init)))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, global_feature, spatial_feature):
        return self.norm(self.global_proj(global_feature) + self.spatial_scale * spatial_feature)


class GCNMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()

        student_backbone, embed_dim = build_model_from_cfg(cfg)
        student_model_dict["backbone"] = student_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")
            student_backbone.load_state_dict(chkpt["model"], strict=False)

        self.embed_dim = embed_dim
        self.need_to_synchronize_fsdp_streams = True

        student_model_dict["spatial_agg"] = SpatialPatchAggregator(
            embed_dim, cfg.feature.spatial_agg_hidden, cfg.feature.out_dim
        )
        student_model_dict["node_fusion"] = NodeFeatureFusion(
            embed_dim, cfg.feature.out_dim, cfg.feature.spatial_fusion_alpha
        )

        self.student = nn.ModuleDict(student_model_dict)

        logger.info("DINO extractor built with spatially fused %d-D output", cfg.feature.out_dim)

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def dino_encoder(self, images):
        n_global_crops = 2
        batch_size = images["collated_global_crops"].shape[0] // n_global_crops
        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        output = self.student.backbone(global_crops, masks=None, is_training=True)
        global_feature = output["x_norm_clstoken"].view(n_global_crops, batch_size, -1).mean(0)
        patch_tokens = output["x_norm_patchtokens"]
        spatial_feature = self.student.spatial_agg(patch_tokens.float()).view(
            n_global_crops, batch_size, -1
        ).mean(0).to(global_feature.dtype)
        return self.student.node_fusion(global_feature, spatial_feature)
        

    def forward_backward(self, images):
        criterion = nn.CrossEntropyLoss()
        loss_dict = {}
        loss_accumulator = 0  # for backprop

        ### DINO encoder
        node_features = self.dino_encoder(images)

        # self.fsdp_synchronize_streams()

        return node_features
        

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            self.student.spatial_agg._streams = self.student.backbone._streams
            self.student.node_fusion._streams = self.student.backbone._streams
            self.need_to_synchronize_fsdp_streams = False

    def train(self, mode: bool = True): 
        super().train(mode)

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        # below will synchronize all student subnetworks across gpus:
        for k, v in self.student.items():
            print(k)
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
