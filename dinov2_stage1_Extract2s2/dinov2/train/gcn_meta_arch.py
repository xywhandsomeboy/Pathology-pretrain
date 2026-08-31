# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import math

import torch
from torch import nn
import torch.nn.functional as F

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


class LocalCropFusion(nn.Module):
    """Fuse independently encoded local crops without inventing spatial adjacency."""

    def __init__(self, num_local_crops, feature_dim, hidden_dim):
        super().__init__()
        if num_local_crops < 1:
            raise ValueError("LocalCropFusion requires at least one local crop")
        self.num_local_crops = num_local_crops
        self.mlp = nn.Sequential(
            nn.Linear(num_local_crops * feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, local_features):
        # local_features: [num_local_crops, batch, feature_dim]
        if local_features.size(0) != self.num_local_crops:
            raise ValueError(
                f"Expected {self.num_local_crops} local crops, got {local_features.size(0)}"
            )
        return self.mlp(local_features.permute(1, 0, 2).flatten(1))


class NodeFeatureFusion(nn.Module):
    """Fuse CLS, global spatial, and local spatial features with a CLS residual."""

    def __init__(self, global_dim, out_dim, hidden_dim, fusion_scale_init=0.1):
        super().__init__()
        self.global_proj = nn.Identity() if global_dim == out_dim else nn.Linear(global_dim, out_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(out_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.fusion_scale = nn.Parameter(torch.tensor(float(fusion_scale_init)))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, cls_feature, global_spatial_feature, local_spatial_feature):
        cls_feature = self.global_proj(cls_feature)
        correction = self.fusion_mlp(
            torch.cat((cls_feature, global_spatial_feature, local_spatial_feature), dim=-1)
        )
        return self.norm(cls_feature + self.fusion_scale * correction)

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
        if cfg.feature.out_dim != embed_dim:
            raise ValueError(
                "Stage-1A CLS guidance requires feature.out_dim to match the "
                f"DINO embedding dimension ({embed_dim}), got {cfg.feature.out_dim}."
            )
        if cfg.crops.local_crops_number < 1:
            raise ValueError("The local spatial branch requires crops.local_crops_number >= 1")
        self.need_to_synchronize_fsdp_streams = True

        # Global and local crops use independent convolutional encoders because
        # they represent different fields of view and have different token grids.
        student_model_dict["spatial_agg"] = SpatialPatchAggregator(
            embed_dim, cfg.feature.spatial_agg_hidden, cfg.feature.out_dim
        )
        student_model_dict["local_spatial_agg"] = SpatialPatchAggregator(
            embed_dim, cfg.feature.local_spatial_agg_hidden, cfg.feature.out_dim
        )
        student_model_dict["local_crop_fusion"] = LocalCropFusion(
            cfg.crops.local_crops_number,
            cfg.feature.out_dim,
            cfg.feature.local_fusion_hidden,
        )
        student_model_dict["node_fusion"] = NodeFeatureFusion(
            embed_dim,
            cfg.feature.out_dim,
            cfg.feature.node_fusion_hidden,
            cfg.feature.spatial_fusion_alpha,
        )

        self.student = nn.ModuleDict(student_model_dict)

        if cfg.feature.freeze_backbone:
            for parameter in self.student.backbone.parameters():
                parameter.requires_grad_(False)
            self.student.backbone.eval()

        logger.info("DINO extractor built with spatially fused %d-D output", cfg.feature.out_dim)

    def forward(self, inputs):
        return self.extract_features(inputs)

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def _encode_two_views(self, images, return_patch_tokens=False):
        n_global_crops = 2
        batch_size = images["collated_global_crops"].shape[0] // n_global_crops
        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        local_crops = images["collated_local_crops"].cuda(non_blocking=True)
        n_local_crops = self.cfg.crops.local_crops_number
        if local_crops.shape[0] != n_local_crops * batch_size:
            raise ValueError(
                f"Expected {n_local_crops * batch_size} local crops, got {local_crops.shape[0]}"
            )
        if self.cfg.feature.freeze_backbone:
            with torch.no_grad():
                output = self.student.backbone(global_crops, masks=None, is_training=True)
                local_output = self.student.backbone(local_crops, masks=None, is_training=True)
        else:
            output = self.student.backbone(global_crops, masks=None, is_training=True)
            local_output = self.student.backbone(local_crops, masks=None, is_training=True)

        global_features = output["x_norm_clstoken"].view(n_global_crops, batch_size, -1)
        patch_tokens = output["x_norm_patchtokens"].view(
            n_global_crops, batch_size, -1, self.embed_dim
        )
        spatial_features = self.student.spatial_agg(
            patch_tokens.flatten(0, 1).float()
        ).view(
            n_global_crops, batch_size, -1
        ).to(global_features.dtype)
        local_patch_tokens = local_output["x_norm_patchtokens"]
        local_spatial_features = self.student.local_spatial_agg(
            local_patch_tokens.float()
        ).view(n_local_crops, batch_size, -1).to(global_features.dtype)
        local_feature = self.student.local_crop_fusion(local_spatial_features.float()).to(
            global_features.dtype
        )
        fused_features = torch.stack([
            self.student.node_fusion(
                global_features[view], spatial_features[view], local_feature
            )
            for view in range(n_global_crops)
        ])
        outputs = (fused_features, spatial_features, local_feature, global_features)
        if return_patch_tokens:
            return outputs + (patch_tokens,)
        return outputs

    @torch.no_grad()
    def extract_features(self, images):
        """Stage 1B: average the two trained fused views for .npz output."""
        fused_features, _, _, _ = self._encode_two_views(images)
        return fused_features.mean(dim=0)

    @torch.no_grad()
    def extract_decoder_features(self, images):
        """Export both graph nodes and unpooled DINO tokens for segmentation.

        Stage-1B uses two identical deterministic global views, so averaging
        them is spatially valid. The returned dense tokens retain the DINO
        patch grid and must be kept in exactly the same patch order as the
        graph-node features.
        """
        fused, _, _, _, patch_tokens = self._encode_two_views(
            images, return_patch_tokens=True
        )
        return {
            "node_features": fused.mean(dim=0),
            "dense_tokens": patch_tokens.mean(dim=0),
        }

    def forward_pretrain(self, images):
        """Stage 1A: train fusion using view consistency and frozen CLS guidance."""
        fused, spatial, local_feature, global_cls = self._encode_two_views(images)
        view_loss = (
            1.0
            - F.cosine_similarity(fused[0].float(), fused[1].float(), dim=-1)
        ).mean()

        teacher_losses = []
        for view in range(2):
            teacher = global_cls[view].detach()
            teacher_losses.append(
                (1.0 - F.cosine_similarity(fused[view].float(), teacher.float(), dim=-1)).mean()
            )
        teacher_loss = torch.stack(teacher_losses).mean()
        spatial_view_loss = (
            1.0
            - F.cosine_similarity(spatial[0].float(), spatial[1].float(), dim=-1)
        ).mean()
        spatial_teacher_loss = torch.stack([
            (
                1.0
                - F.cosine_similarity(
                    spatial[view].float(), global_cls[view].detach().float(), dim=-1
                )
            ).mean()
            for view in range(2)
        ]).mean()
        global_teacher = global_cls.detach().mean(dim=0)
        local_teacher_loss = (
            1.0
            - F.cosine_similarity(
                local_feature.float(), global_teacher.float(), dim=-1
            )
        ).mean()
        total_loss = (
            self.cfg.feature.view_loss_weight * view_loss
            + self.cfg.feature.teacher_loss_weight * teacher_loss
            + self.cfg.feature.spatial_view_loss_weight * spatial_view_loss
            + self.cfg.feature.spatial_teacher_loss_weight * spatial_teacher_loss
            + self.cfg.feature.local_teacher_loss_weight * local_teacher_loss
        )
        self.backprop_loss(total_loss)
        self.fsdp_synchronize_streams()
        return {
            "view_consistency": view_loss.detach(),
            "teacher_alignment": teacher_loss.detach(),
            "spatial_view_consistency": spatial_view_loss.detach(),
            "spatial_teacher_alignment": spatial_teacher_loss.detach(),
            "local_teacher_alignment": local_teacher_loss.detach(),
            "total_loss": total_loss.detach(),
        }

    def forward_backward(self, images):
        """Backward-compatible alias for Stage 1B feature extraction."""
        return self.extract_features(images)
        

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            self.student.spatial_agg._streams = self.student.backbone._streams
            self.student.local_spatial_agg._streams = self.student.backbone._streams
            self.student.local_crop_fusion._streams = self.student.backbone._streams
            self.student.node_fusion._streams = self.student.backbone._streams
            self.need_to_synchronize_fsdp_streams = False

    def train(self, mode: bool = True): 
        super().train(mode)
        if self.cfg.feature.freeze_backbone:
            self.student.backbone.eval()
        return self

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
