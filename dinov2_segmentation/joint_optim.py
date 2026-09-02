"""Literature-backed parameter groups and warmup-cosine scheduling."""

from __future__ import annotations

import math

import torch
from torch import nn


def _vit_blocks(backbone: nn.Module) -> list[nn.Module]:
    blocks: list[nn.Module] = []
    for item in backbone.blocks:
        if isinstance(item, nn.ModuleList):
            # DINO BlockChunk prefixes later chunks with parameter-free
            # Identity modules to retain global layer indices. They must not
            # count as real layers in layer-wise LR decay.
            blocks.extend(
                block for block in item if any(True for _ in block.parameters())
            )
        else:
            if any(True for _ in item.parameters()):
                blocks.append(item)
    return blocks


def _no_weight_decay(name: str, parameter: nn.Parameter) -> bool:
    return (
        parameter.ndim <= 1
        or name.endswith(".bias")
        or name.endswith("cls_token")
        or name.endswith("mask_token")
        or name.endswith("pos_embed")
    )


def build_joint_adamw(
    system,
    *,
    decoder_lr: float = 2e-4,
    stage2_lr: float = 5e-5,
    stage1_fusion_lr: float = 5e-5,
    stage1_backbone_lr: float = 2e-5,
    layer_decay: float = 0.9,
    weight_decay: float = 0.05,
    betas: tuple[float, float] = (0.9, 0.999),
) -> tuple[torch.optim.AdamW, list[dict]]:
    """Create disjoint, named AdamW groups for the full joint model.

    The pretrained ViT uses layer-wise LR decay from input to output. Stage1
    fusion, Stage2 GATv2 and the randomly initialized decoder receive separate
    rates so that supervised adaptation is strongest near the task head.
    """

    for value, name in (
        (decoder_lr, "decoder_lr"),
        (stage2_lr, "stage2_lr"),
        (stage1_fusion_lr, "stage1_fusion_lr"),
        (stage1_backbone_lr, "stage1_backbone_lr"),
        (weight_decay, "weight_decay"),
    ):
        if value < 0 or (name != "weight_decay" and value == 0):
            raise ValueError(f"{name} must be positive")
    if not 0 < layer_decay <= 1:
        raise ValueError("layer_decay must be in (0,1]")

    backbone = system.stage1.backbone
    blocks = _vit_blocks(backbone)
    block_layers: dict[int, int] = {}
    for layer_id, block in enumerate(blocks, start=1):
        for parameter in block.parameters():
            block_layers[id(parameter)] = layer_id
    final_layer_id = len(blocks) + 1

    grouped: dict[tuple[str, float, float], list[nn.Parameter]] = {}
    descriptions: dict[tuple[str, float, float], dict] = {}
    assigned: set[int] = set()

    def add(name: str, parameter: nn.Parameter, lr: float, decay: float) -> None:
        if not parameter.requires_grad:
            return
        identity = id(parameter)
        if identity in assigned:
            raise RuntimeError(f"Parameter assigned twice: {name}")
        assigned.add(identity)
        key = (name, float(lr), float(decay))
        grouped.setdefault(key, []).append(parameter)
        descriptions.setdefault(
            key,
            {"name": name, "lr": float(lr), "weight_decay": float(decay)},
        )

    for name, parameter in backbone.named_parameters():
        if id(parameter) in block_layers:
            layer_id = block_layers[id(parameter)]
        elif name.startswith("norm") or name.startswith("head"):
            layer_id = final_layer_id
        else:
            layer_id = 0
        scale = layer_decay ** (final_layer_id - layer_id)
        decay = 0.0 if _no_weight_decay(name, parameter) else weight_decay
        add(
            f"stage1_backbone_layer_{layer_id:02d}_{'nodecay' if decay == 0 else 'decay'}",
            parameter,
            stage1_backbone_lr * scale,
            decay,
        )

    fusion_modules = (
        system.stage1.spatial_agg,
        system.stage1.local_spatial_agg,
        system.stage1.local_crop_fusion,
        system.stage1.node_fusion,
    )
    for module_name, module in zip(
        ("spatial_agg", "local_spatial_agg", "local_crop_fusion", "node_fusion"),
        fusion_modules,
    ):
        for name, parameter in module.named_parameters():
            decay = 0.0 if _no_weight_decay(name, parameter) else weight_decay
            add(
                f"stage1_{module_name}_{'nodecay' if decay == 0 else 'decay'}",
                parameter,
                stage1_fusion_lr,
                decay,
            )

    for module_name, module, learning_rate in (
        ("stage2_gatv2", system.stage2, stage2_lr),
        (f"decoder_{system.decoder_version}", system.decoder, decoder_lr),
    ):
        for name, parameter in module.named_parameters():
            decay = 0.0 if _no_weight_decay(name, parameter) else weight_decay
            add(
                f"{module_name}_{'nodecay' if decay == 0 else 'decay'}",
                parameter,
                learning_rate,
                decay,
            )

    expected = {id(p) for p in system.parameters() if p.requires_grad}
    if assigned != expected:
        raise RuntimeError(
            f"Optimizer coverage mismatch: missing={len(expected - assigned)}, "
            f"unexpected={len(assigned - expected)}"
        )
    parameter_groups = []
    metadata = []
    for key, parameters in grouped.items():
        description = dict(descriptions[key])
        description["parameter_count"] = sum(p.numel() for p in parameters)
        parameter_groups.append(
            {
                "params": parameters,
                "lr": description["lr"],
                "weight_decay": description["weight_decay"],
                "group_name": description["name"],
            }
        )
        metadata.append(description)
    optimizer = torch.optim.AdamW(parameter_groups, betas=betas)
    return optimizer, metadata


class WarmupCosineScheduler:
    """Per-update linear warm-up followed by one cosine decay to ``min_ratio``."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        warmup_steps: int,
        min_ratio: float = 0.01,
    ) -> None:
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        if warmup_steps < 0 or warmup_steps >= total_steps:
            raise ValueError("warmup_steps must be in [0,total_steps)")
        if not 0 <= min_ratio <= 1:
            raise ValueError("min_ratio must be in [0,1]")
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_ratio = float(min_ratio)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.current_step = 0
        self._apply()

    def _factor(self) -> float:
        if self.warmup_steps and self.current_step < self.warmup_steps:
            return float(self.current_step + 1) / self.warmup_steps
        decay_steps = max(1, self.total_steps - self.warmup_steps - 1)
        progress = min(
            1.0,
            max(0.0, (self.current_step - self.warmup_steps) / decay_steps),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_ratio + (1.0 - self.min_ratio) * cosine

    def _apply(self) -> None:
        factor = self._factor()
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def step(self) -> None:
        self.current_step = min(self.current_step + 1, self.total_steps - 1)
        self._apply()

    def state_dict(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_ratio": self.min_ratio,
            "base_lrs": list(self.base_lrs),
            "current_step": self.current_step,
        }

    def load_state_dict(self, state: dict) -> None:
        for name in ("total_steps", "warmup_steps"):
            if int(state[name]) != getattr(self, name):
                raise ValueError(f"Scheduler {name} differs from the checkpoint")
        if not math.isclose(float(state["min_ratio"]), self.min_ratio):
            raise ValueError("Scheduler min_ratio differs from the checkpoint")
        loaded_lrs = list(map(float, state["base_lrs"]))
        if len(loaded_lrs) != len(self.base_lrs):
            raise ValueError("Scheduler parameter-group count differs from checkpoint")
        self.base_lrs = loaded_lrs
        self.current_step = int(state["current_step"])
        self._apply()
