"""Trainable Stage1 -> Stage2 -> segmentation systems.

This module intentionally uses the Stage2 source tree's ``dinov2`` package.
Its ViT implementation is byte-identical to the Stage1 source tree while its
GNN implementation contains the trained GATv2 changes. The launcher therefore
runs from ``dinov2_stage2_2_FmH2ST`` with the repository root on PYTHONPATH.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from omegaconf import OmegaConf
import torch
from torch import nn
import torch.nn.functional as F

from dinov2.models import vision_transformer as vits
from dinov2.models.gcn import GNN

from dinov2_segmentation.models import GlobalLocalSegmentationModel
from dinov2_segmentation.models.model_v2 import GlobalLocalSegmentationModelV2


def _load(path: str | Path):
    path = Path(path).expanduser().resolve()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class SpatialPatchAggregator(nn.Module):
    """The exact trained Stage1 spatial-token convolutional aggregator."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 1024):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                3,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                3,
                stride=2,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                3,
                stride=2,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.GELU(),
        )
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = patch_tokens.shape
        side = math.isqrt(num_tokens)
        if side * side != num_tokens:
            raise ValueError(f"DINO token count {num_tokens} is not a square grid")
        feature = patch_tokens.transpose(1, 2).reshape(
            batch_size, channels, side, side
        )
        feature = self.spatial(self.compress(feature)).mean(dim=(-2, -1))
        return self.proj(feature)


class LocalCropFusion(nn.Module):
    def __init__(self, num_local_crops: int, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.num_local_crops = int(num_local_crops)
        self.mlp = nn.Sequential(
            nn.Linear(self.num_local_crops * feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, local_features: torch.Tensor) -> torch.Tensor:
        if local_features.size(0) != self.num_local_crops:
            raise ValueError(
                f"Expected {self.num_local_crops} local crops, got {local_features.size(0)}"
            )
        return self.mlp(local_features.permute(1, 0, 2).flatten(1))


class NodeFeatureFusion(nn.Module):
    def __init__(
        self,
        global_dim: int,
        out_dim: int,
        hidden_dim: int,
        fusion_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.global_proj = (
            nn.Identity() if global_dim == out_dim else nn.Linear(global_dim, out_dim)
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(out_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.fusion_scale = nn.Parameter(torch.tensor(float(fusion_scale_init)))
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        cls_feature: torch.Tensor,
        global_spatial_feature: torch.Tensor,
        local_spatial_feature: torch.Tensor,
    ) -> torch.Tensor:
        cls_feature = self.global_proj(cls_feature)
        correction = self.fusion_mlp(
            torch.cat(
                (cls_feature, global_spatial_feature, local_spatial_feature), dim=-1
            )
        )
        return self.norm(cls_feature + self.fusion_scale * correction)


class TrainableStage1(nn.Module):
    """Recreate Stage1 and expose online node plus dense token features."""

    def __init__(self, config_path: str | Path, checkpoint_path: str | Path):
        super().__init__()
        self.config_path = str(Path(config_path).expanduser().resolve())
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        cfg = OmegaConf.load(self.config_path)
        student = cfg.student
        architecture = str(student.arch).removesuffix("_memeff")
        if architecture not in vits.__dict__:
            raise ValueError(f"Unsupported Stage1 ViT architecture: {architecture}")
        self.backbone = vits.__dict__[architecture](
            img_size=int(cfg.crops.global_crops_size),
            patch_size=int(student.patch_size),
            init_values=float(student.layerscale),
            ffn_layer=str(student.ffn_layer),
            block_chunks=int(student.block_chunks),
            qkv_bias=bool(student.qkv_bias),
            proj_bias=bool(student.proj_bias),
            ffn_bias=bool(student.ffn_bias),
            num_register_tokens=int(student.num_register_tokens),
            interpolate_offset=float(student.interpolate_offset),
            interpolate_antialias=bool(student.interpolate_antialias),
            drop_path_rate=float(student.drop_path_rate),
            drop_path_uniform=bool(student.drop_path_uniform),
        )
        self.embed_dim = int(self.backbone.embed_dim)
        self.image_size = int(cfg.crops.global_crops_size)
        self.local_size = int(cfg.crops.local_crops_size)
        self.num_local_crops = int(cfg.crops.local_crops_number)
        if self.num_local_crops < 1 or self.num_local_crops > 5:
            raise ValueError("Joint Stage1 supports one to five deterministic local crops")
        out_dim = int(cfg.feature.out_dim)
        if out_dim != self.embed_dim:
            raise ValueError("Stage1 node and dense token dimensions must match")
        self.spatial_agg = SpatialPatchAggregator(
            self.embed_dim,
            int(cfg.feature.spatial_agg_hidden),
            out_dim,
        )
        self.local_spatial_agg = SpatialPatchAggregator(
            self.embed_dim,
            int(cfg.feature.local_spatial_agg_hidden),
            out_dim,
        )
        self.local_crop_fusion = LocalCropFusion(
            self.num_local_crops,
            out_dim,
            int(cfg.feature.local_fusion_hidden),
        )
        self.node_fusion = NodeFeatureFusion(
            self.embed_dim,
            out_dim,
            int(cfg.feature.node_fusion_hidden),
            float(cfg.feature.spatial_fusion_alpha),
        )

        payload = _load(self.checkpoint_path)
        state = payload.get("model", payload)
        if not isinstance(state, dict):
            raise TypeError("Stage1 checkpoint does not contain a state dictionary")
        prefix = "student."
        state = {
            str(key)[len(prefix) :]: value
            for key, value in state.items()
            if str(key).startswith(prefix)
        }
        incompatible = self.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Stage1 checkpoint is incompatible with joint fine-tuning: "
                f"missing={incompatible.missing_keys[:10]}, "
                f"unexpected={incompatible.unexpected_keys[:10]}"
            )
        # The Stage1 pretraining config froze DINO, but supervised joint
        # fine-tuning explicitly updates every Stage1 parameter.
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def _local_crops(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        size = min(self.local_size, height, width)
        positions = (
            (0, 0),
            (width - size, 0),
            (0, height - size),
            (width - size, height - size),
            ((width - size) // 2, (height - size) // 2),
        )
        crops = [
            images[:, :, y : y + size, x : x + size]
            for x, y in positions[: self.num_local_crops]
        ]
        return torch.cat(crops, dim=0)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Stage1 images must be [B,3,H,W], got {images.shape}")
        if tuple(images.shape[-2:]) != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
            )
        output = self.backbone(images, masks=None, is_training=True)
        cls_feature = output["x_norm_clstoken"]
        dense_tokens = output["x_norm_patchtokens"]
        global_spatial = self.spatial_agg(dense_tokens.float()).to(cls_feature.dtype)

        batch_size = images.size(0)
        local_output = self.backbone(
            self._local_crops(images), masks=None, is_training=True
        )
        local_spatial = self.local_spatial_agg(
            local_output["x_norm_patchtokens"].float()
        ).view(self.num_local_crops, batch_size, -1)
        local_feature = self.local_crop_fusion(local_spatial.float()).to(
            cls_feature.dtype
        )
        node_features = self.node_fusion(
            cls_feature, global_spatial, local_feature
        )
        return node_features, dense_tokens


@dataclass(frozen=True)
class Stage2Runtime:
    num_layers: int
    context_edge_mode: str
    use_edge_attr: bool
    edge_dim: int


def build_trainable_stage2(
    config_path: str | Path, checkpoint_path: str | Path
) -> tuple[GNN, Stage2Runtime]:
    cfg = OmegaConf.load(Path(config_path).expanduser().resolve())
    gcn = cfg.gcn
    model = GNN(
        num_layer=int(gcn.num_layer),
        emb_dim=int(gcn.emb_dim),
        JK=str(gcn.JK),
        drop_ratio=float(gcn.dropout_ratio),
        gnn_type=str(gcn.gnn_type),
        edge_dim=int(gcn.edge_dim),
        block_chunks=int(gcn.block_chunks),
        use_residual=bool(gcn.use_residual),
        use_layernorm=bool(gcn.use_layernorm),
        edge_injection=str(gcn.edge_injection),
    )
    payload = _load(checkpoint_path)
    state = payload.get("model", payload.get("student", payload))
    if not isinstance(state, dict):
        raise TypeError("Stage2 checkpoint does not contain a state dictionary")
    prefix = "student.gcn."
    gnn_state = {
        str(key)[len(prefix) :]: value
        for key, value in state.items()
        if str(key).startswith(prefix)
    }
    incompatible = model.load_state_dict(gnn_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Stage2 checkpoint is incompatible with joint fine-tuning: "
            f"missing={incompatible.missing_keys[:10]}, "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    runtime = Stage2Runtime(
        num_layers=int(gcn.num_layer),
        context_edge_mode=str(gcn.context_edge_mode),
        use_edge_attr=bool(gcn.context_use_edge_attr),
        edge_dim=int(gcn.edge_dim),
    )
    return model, runtime


class JointSegmentationSystem(nn.Module):
    """Container that makes all three trainable stages checkpointable."""

    model_version = "joint_stage1_stage2_decoder_v1"

    def __init__(
        self,
        *,
        decoder_version: str,
        stage1_config: str | Path,
        stage1_checkpoint: str | Path,
        stage2_config: str | Path,
        stage2_checkpoint: str | Path,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.stage1 = TrainableStage1(stage1_config, stage1_checkpoint)
        self.stage2, self.stage2_runtime = build_trainable_stage2(
            stage2_config, stage2_checkpoint
        )
        if decoder_version == "v1":
            self.decoder = GlobalLocalSegmentationModel(
                num_classes=num_classes,
                token_dim=self.stage1.embed_dim,
                context_dim=self.stage1.embed_dim,
            )
        elif decoder_version == "v2":
            self.decoder = GlobalLocalSegmentationModelV2(
                num_classes=num_classes,
                token_dim=self.stage1.embed_dim,
                context_dim=self.stage1.embed_dim,
            )
        else:
            raise ValueError("decoder_version must be 'v1' or 'v2'")
        self.decoder_version = decoder_version
        self.model_version = f"joint_stage1_stage2_decoder_{decoder_version}"

    def decode(
        self,
        images: torch.Tensor,
        dense_tokens: torch.Tensor,
        contexts: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(images, dense_tokens, contexts)
