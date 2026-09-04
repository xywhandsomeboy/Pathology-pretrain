"""Jointly fine-tune Stage1, Stage2 GATv2, and Decoder V1 or V2."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dinov2_segmentation.data import JointPatchSegmentationDataset
from dinov2_segmentation.joint_graph import JointGraphRepository
from dinov2_segmentation.joint_model import JointSegmentationSystem
from dinov2_segmentation.joint_optim import (
    WarmupCosineScheduler,
    _vit_blocks,
    build_joint_adamw,
)
from dinov2_segmentation.losses import segmentation_loss
from dinov2_segmentation.probability_metrics import (
    BinaryProbabilityMetrics,
    binary_confusion_metrics,
)
from dinov2_segmentation.profiles import validate_experiment_profile
from dinov2_segmentation.sampling import SlideStratifiedSampler


PAPER_REFERENCES = {
    "adamw": "https://arxiv.org/abs/1711.05101",
    "cosine_schedule": "https://arxiv.org/abs/1608.03983",
    "layerwise_lr_decay": "https://arxiv.org/abs/2106.08254",
    "gradual_unfreezing": "https://arxiv.org/abs/1801.06146",
    "dinov2": "https://arxiv.org/abs/2304.07193",
    "gatv2": "https://arxiv.org/abs/2105.14491",
    "dense_prediction_adapter": "https://arxiv.org/abs/2205.08534",
    "tversky": "https://arxiv.org/abs/1706.05721",
    "focal_tversky": "https://arxiv.org/abs/1810.07842",
    "pathology_color_augmentation": "https://pubmed.ncbi.nlm.nih.gov/31466046/",
}


def _load(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-version", choices=("v1", "v2"), required=True)
    parser.add_argument(
        "--decoder-drop-path-rate",
        type=float,
        default=0.1,
        help="Maximum stochastic-depth probability inside the decoder",
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--stage1-config", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-config", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment-profile",
        choices=("current", "S", "ST", "STA"),
        default="current",
        help="Recorded ablation identity; profile flags are supplied by the launcher",
    )
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--decoder-lr", type=float, default=2e-4)
    parser.add_argument("--stage2-lr", type=float, default=5e-5)
    parser.add_argument("--stage1-fusion-lr", type=float, default=5e-5)
    parser.add_argument("--stage1-backbone-lr", type=float, default=2e-5)
    parser.add_argument("--layer-decay", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--cross-entropy-weight", type=float, default=1.0)
    parser.add_argument(
        "--dice-weight",
        "--overlap-weight",
        dest="dice_weight",
        type=float,
        default=1.0,
        help="Weight for the selected Dice or Tversky overlap term",
    )
    parser.add_argument(
        "--overlap-loss",
        choices=("dice", "foreground_tversky"),
        default="dice",
    )
    parser.add_argument("--tversky-alpha", type=float, default=0.3, help="FP weight")
    parser.add_argument("--tversky-beta", type=float, default=0.7, help="FN weight")
    parser.add_argument(
        "--tumor-class-weight",
        type=float,
        default=1.0,
        help="Relative class-1 weight in cross entropy; overlap loss remains unchanged",
    )
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument(
        "--sampling-mode",
        choices=("uniform", "slide_stratified"),
        default="uniform",
    )
    parser.add_argument("--sampling-positive-fraction", type=float, default=0.60)
    parser.add_argument(
        "--sampling-boundary-positive-fraction", type=float, default=0.50
    )
    parser.add_argument(
        "--sampling-interior-threshold", type=float, default=0.999999
    )
    parser.add_argument("--sampling-slide-balance-power", type=float, default=0.5)
    parser.add_argument("--sampling-max-patch-repeats", type=int, default=2)
    parser.add_argument(
        "--sampling-epoch-samples",
        type=int,
        default=0,
        help="Samples per balanced epoch; zero keeps the manifest length",
    )
    parser.add_argument(
        "--color-augmentation", choices=("none", "mild"), default="none"
    )
    parser.add_argument(
        "--probability-metric-bins",
        type=int,
        default=0,
        help="Streaming validation histogram bins; zero disables probability metrics",
    )
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--gradient-audit-updates", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument(
        "--decoder-only-epochs",
        type=int,
        default=0,
        help="Train only the decoder for this many initial epochs",
    )
    parser.add_argument(
        "--stage1-top-unfreeze-epoch",
        type=int,
        default=0,
        help="Epoch at which selected top DINO blocks begin joint training",
    )
    parser.add_argument(
        "--stage1-unfreeze-blocks",
        type=int,
        default=0,
        help="Number of top DINO blocks to unfreeze; zero means all blocks",
    )
    parser.add_argument(
        "--final-phase-pretrained-lr-scale",
        type=float,
        default=1.0,
        help="LR multiplier for Stage1 fusion and Stage2 after DINO unfreezing",
    )
    parser.add_argument(
        "--final-phase-decoder-lr-scale",
        type=float,
        default=1.0,
        help="Decoder LR multiplier after DINO unfreezing",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many non-improving validation epochs; zero disables",
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--early-stopping-start-epoch",
        type=int,
        default=0,
        help="Do not count early-stopping patience before this zero-based epoch",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _loader(path: Path, args: argparse.Namespace, training: bool) -> DataLoader:
    dataset = JointPatchSegmentationDataset(
        path,
        image_size=args.image_size,
        training=training,
        color_augmentation=args.color_augmentation if training else "none",
    )
    generator = torch.Generator().manual_seed(args.seed + int(training))
    sampler = None
    if training and args.sampling_mode == "slide_stratified":
        sampler = SlideStratifiedSampler(
            dataset.rows,
            num_samples=args.sampling_epoch_samples or len(dataset),
            batch_size=args.batch_size,
            positive_fraction=args.sampling_positive_fraction,
            boundary_positive_fraction=args.sampling_boundary_positive_fraction,
            interior_threshold=args.sampling_interior_threshold,
            slide_balance_power=args.sampling_slide_balance_power,
            max_patch_repeats=args.sampling_max_patch_repeats,
            seed=args.seed,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=args.workers > 0,
        generator=generator,
    )


def _gradient_norm(module: torch.nn.Module) -> float:
    squares = []
    for parameter in module.parameters():
        if parameter.grad is not None:
            squares.append(parameter.grad.detach().float().square().sum())
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt().cpu())


_BACKBONE_GROUP = re.compile(r"^stage1_backbone_layer_(\d+)_")


def _training_phase(
    system: JointSegmentationSystem,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    epoch: int,
    args: argparse.Namespace,
) -> dict:
    """Configure gradual unfreezing and phase-specific LR multipliers."""

    if epoch < args.decoder_only_epochs:
        phase_name = "decoder_only"
    elif epoch < args.stage1_top_unfreeze_epoch:
        phase_name = "adapters_and_decoder"
    else:
        phase_name = "top_backbone_joint"

    backbone_layers = [
        int(match.group(1))
        for group in optimizer.param_groups
        if (match := _BACKBONE_GROUP.match(str(group.get("group_name", ""))))
    ]
    final_layer = max(backbone_layers, default=0)
    if args.stage1_unfreeze_blocks > 0:
        first_enabled_backbone_layer = max(
            1, final_layer - args.stage1_unfreeze_blocks
        )
    else:
        first_enabled_backbone_layer = 0

    scales = []
    active_groups = []
    for group in optimizer.param_groups:
        name = str(group.get("group_name", ""))
        match = _BACKBONE_GROUP.match(name)
        if name.startswith("decoder_"):
            active = True
            scale = (
                args.final_phase_decoder_lr_scale
                if phase_name == "top_backbone_joint"
                else 1.0
            )
        elif name.startswith("stage2_") or (
            name.startswith("stage1_") and match is None
        ):
            active = phase_name != "decoder_only"
            scale = (
                args.final_phase_pretrained_lr_scale
                if phase_name == "top_backbone_joint"
                else 1.0
            )
        elif match is not None:
            layer_id = int(match.group(1))
            active = phase_name == "top_backbone_joint" and (
                first_enabled_backbone_layer == 0
                or layer_id >= first_enabled_backbone_layer
            )
            scale = 1.0
        else:
            raise RuntimeError(f"Unrecognized optimizer group: {name}")
        effective_scale = scale if active else 0.0
        scales.append(effective_scale)
        for parameter in group["params"]:
            parameter.requires_grad_(active)
        if active:
            active_groups.append(name)
    scheduler.set_phase_scales(scales)

    phase = {
        "name": phase_name,
        "epoch": epoch,
        "active_optimizer_groups": len(active_groups),
        "total_optimizer_groups": len(optimizer.param_groups),
        "first_enabled_backbone_layer": (
            first_enabled_backbone_layer
            if phase_name == "top_backbone_joint"
            else None
        ),
        "stage1_unfreeze_blocks": (
            args.stage1_unfreeze_blocks
            if phase_name == "top_backbone_joint"
            else 0
        ),
    }
    return phase


def _set_runtime_modes(
    system: JointSegmentationSystem,
    *,
    training: bool,
    phase: dict | None,
) -> None:
    system.train(training)
    if not training or phase is None:
        return
    if phase["name"] == "decoder_only":
        system.stage1.eval()
        system.stage2.eval()
        system.decoder.train(True)
        return
    if phase["name"] == "adapters_and_decoder":
        system.stage1.backbone.eval()
        system.stage2.train(True)
        return

    # Keep frozen lower ViT blocks deterministic while the selected top blocks
    # use their normal train-time behavior.
    system.stage1.backbone.eval()
    unfreeze_blocks = int(phase["stage1_unfreeze_blocks"])
    blocks = _vit_blocks(system.stage1.backbone)
    selected = blocks if unfreeze_blocks == 0 else blocks[-unfreeze_blocks:]
    for block in selected:
        block.train(True)


def _confusion_update(
    confusion: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int,
) -> None:
    prediction = logits.detach().argmax(dim=1)
    valid = target != ignore_index
    encoded = target[valid] * num_classes + prediction[valid]
    confusion += torch.bincount(
        encoded.cpu(), minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)


def _metrics(
    totals: dict,
    samples: int,
    confusion: torch.Tensor,
    probability_metrics: BinaryProbabilityMetrics | None = None,
) -> dict:
    result = {name: value / max(samples, 1) for name, value in totals.items()}
    result.update(binary_confusion_metrics(confusion))
    result["confusion"] = confusion.tolist()
    if probability_metrics is not None:
        result.update(probability_metrics.compute())
    return result


def _run_epoch(
    system: JointSegmentationSystem,
    graph_repository: JointGraphRepository,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    *,
    optimizer=None,
    scheduler=None,
    scaler=None,
    gradient_audit: dict | None = None,
    training_phase: dict | None = None,
) -> dict:
    training = optimizer is not None
    _set_runtime_modes(system, training=training, phase=training_phase)
    totals = {
        "loss": 0.0,
        "cross_entropy": 0.0,
        "dice_loss": 0.0,
        "tversky_loss": 0.0,
    }
    confusion = torch.zeros((args.num_classes, args.num_classes), dtype=torch.int64)
    probability_metrics = (
        BinaryProbabilityMetrics(args.probability_metric_bins)
        if args.probability_metric_bins > 0
        else None
    )
    samples = 0
    accumulation = args.gradient_accumulation if training else 1
    micro_count = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    maximum = args.max_train_batches if training else args.max_val_batches
    total_batches = min(len(loader), maximum) if maximum > 0 else len(loader)
    start = time.monotonic()

    for batch_index, batch in enumerate(loader):
        if maximum > 0 and batch_index >= maximum:
            break
        images = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            node_features, dense_tokens = system.stage1(images)
            contexts = graph_repository.contextualize(
                system.stage2,
                node_features,
                list(batch["slide_id"]),
                list(batch["patch_id"]),
                num_hops=system.stage2_runtime.num_layers,
                use_edge_attr=system.stage2_runtime.use_edge_attr,
                update_memory=training,
            )
            logits = system.decode(images, dense_tokens, contexts)
            loss, parts = segmentation_loss(
                logits,
                target,
                ignore_index=args.ignore_index,
                cross_entropy_weight=args.cross_entropy_weight,
                dice_weight=args.dice_weight,
                tumor_class_weight=args.tumor_class_weight,
                overlap_loss=args.overlap_loss,
                tversky_alpha=args.tversky_alpha,
                tversky_beta=args.tversky_beta,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite joint loss at batch {batch_index}: {loss}")
        if training:
            scaler.scale(loss).backward()
            micro_count += 1
            should_step = micro_count == accumulation or batch_index + 1 == total_batches
            if should_step:
                scaler.unscale_(optimizer)
                # Average accumulated micro-batch gradients without weakening
                # the final, possibly shorter accumulation window.
                for parameter in system.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(micro_count)
                audit_enabled = (
                    gradient_audit is not None
                    and training_phase is not None
                    and training_phase["name"] != "decoder_only"
                )
                if audit_enabled and not gradient_audit.get("complete", False):
                    observed = {
                        "stage1_grad_norm": _gradient_norm(system.stage1),
                        "stage2_grad_norm": _gradient_norm(system.stage2),
                        "decoder_grad_norm": _gradient_norm(system.decoder),
                    }
                    full_joint_phase = (
                        training_phase["name"] == "top_backbone_joint"
                    )
                    if full_joint_phase:
                        observed["stage1_backbone_grad_norm"] = _gradient_norm(
                            system.stage1.backbone
                        )
                    gradient_audit["updates_observed"] = int(
                        gradient_audit.get("updates_observed", 0)
                    ) + 1
                    if full_joint_phase:
                        gradient_audit["full_joint_updates_observed"] = int(
                            gradient_audit.get("full_joint_updates_observed", 0)
                        ) + 1
                    for name, value in observed.items():
                        gradient_audit[name] = max(
                            float(gradient_audit.get(name, 0.0)), float(value)
                        )
                    stage_keys = (
                        "stage1_grad_norm",
                        "stage2_grad_norm",
                        "decoder_grad_norm",
                        "stage1_backbone_grad_norm",
                    )
                    gradient_audit["complete"] = all(
                        math.isfinite(float(gradient_audit.get(name, 0.0)))
                        and float(gradient_audit.get(name, 0.0)) > 0
                        for name in stage_keys
                    )
                    if (
                        not gradient_audit["complete"]
                        and full_joint_phase
                        and gradient_audit["full_joint_updates_observed"]
                        >= args.gradient_audit_updates
                    ):
                        raise RuntimeError(
                            "Supervised loss did not reach every stage within the "
                            f"gradient audit window: {gradient_audit}"
                        )
                torch.nn.utils.clip_grad_norm_(system.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                micro_count = 0

        count = images.size(0)
        samples += count
        totals["loss"] += float(loss.detach()) * count
        for name, value in parts.items():
            totals[name] += float(value) * count
        _confusion_update(
            confusion,
            logits,
            target,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
        )
        if probability_metrics is not None:
            probability_metrics.update(
                logits,
                target,
                ignore_index=args.ignore_index,
            )
        if training and (batch_index + 1) % args.log_interval == 0:
            elapsed = time.monotonic() - start
            print(
                json.dumps(
                    {
                        "batch": batch_index + 1,
                        "batches": total_batches,
                        "loss": float(loss.detach()),
                        "seconds": elapsed,
                        "lr_min": min(group["lr"] for group in optimizer.param_groups),
                        "lr_max": max(group["lr"] for group in optimizer.param_groups),
                    }
                ),
                flush=True,
            )
    return _metrics(totals, samples, confusion, probability_metrics)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _configuration(args: argparse.Namespace) -> dict:
    keys = (
        "experiment_profile",
        "decoder_version",
        "decoder_drop_path_rate",
        "num_classes",
        "image_size",
        "epochs",
        "batch_size",
        "gradient_accumulation",
        "decoder_lr",
        "stage2_lr",
        "stage1_fusion_lr",
        "stage1_backbone_lr",
        "layer_decay",
        "weight_decay",
        "warmup_ratio",
        "min_lr_ratio",
        "clip_grad",
        "cross_entropy_weight",
        "dice_weight",
        "tumor_class_weight",
        "overlap_loss",
        "tversky_alpha",
        "tversky_beta",
        "ignore_index",
        "sampling_mode",
        "sampling_positive_fraction",
        "sampling_boundary_positive_fraction",
        "sampling_interior_threshold",
        "sampling_slide_balance_power",
        "sampling_max_patch_repeats",
        "sampling_epoch_samples",
        "color_augmentation",
        "probability_metric_bins",
        "amp_dtype",
        "seed",
        "gradient_audit_updates",
        "decoder_only_epochs",
        "stage1_top_unfreeze_epoch",
        "stage1_unfreeze_blocks",
        "final_phase_pretrained_lr_scale",
        "final_phase_decoder_lr_scale",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "early_stopping_start_epoch",
    )
    return {key: getattr(args, key) for key in keys}


def main() -> None:
    args = parse_args()
    if args.num_classes != 2:
        raise ValueError("The cervical workflow requires binary background/tumor output")
    if not 0.0 <= args.decoder_drop_path_rate < 1.0:
        raise ValueError("decoder-drop-path-rate must be in [0, 1)")
    for name in (
        "train_manifest",
        "val_manifest",
        "graph_dir",
        "stage1_config",
        "stage1_checkpoint",
        "stage2_config",
        "stage2_checkpoint",
    ):
        value = getattr(args, name).expanduser().resolve()
        setattr(args, name, value)
        if not value.exists():
            raise FileNotFoundError(f"Missing --{name.replace('_', '-')}: {value}")
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.gradient_accumulation < 1
        or args.gradient_audit_updates < 2
    ):
        raise ValueError(
            "epochs, batch-size and gradient-accumulation must be positive; "
            "gradient-audit-updates must be at least 2"
        )
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup-ratio must be in [0,1)")
    if not 0 <= args.decoder_only_epochs <= args.stage1_top_unfreeze_epoch < args.epochs:
        raise ValueError(
            "Require 0 <= decoder-only-epochs <= stage1-top-unfreeze-epoch < epochs"
        )
    if args.stage1_unfreeze_blocks < 0:
        raise ValueError("stage1-unfreeze-blocks must be non-negative")
    if args.tumor_class_weight <= 0:
        raise ValueError("tumor-class-weight must be positive")
    if args.cross_entropy_weight < 0 or args.dice_weight < 0:
        raise ValueError("loss component weights must be non-negative")
    if args.cross_entropy_weight == 0 and args.dice_weight == 0:
        raise ValueError("at least one loss component must be enabled")
    if args.tversky_alpha <= 0 or args.tversky_beta <= 0:
        raise ValueError("Tversky alpha and beta must be positive")
    if not 0 < args.sampling_positive_fraction < 1:
        raise ValueError("sampling-positive-fraction must be in (0,1)")
    if not 0 < args.sampling_boundary_positive_fraction < 1:
        raise ValueError("sampling-boundary-positive-fraction must be in (0,1)")
    if not 0 < args.sampling_interior_threshold <= 1:
        raise ValueError("sampling-interior-threshold must be in (0,1]")
    if not 0 <= args.sampling_slide_balance_power <= 1:
        raise ValueError("sampling-slide-balance-power must be in [0,1]")
    if args.sampling_max_patch_repeats < 1:
        raise ValueError("sampling-max-patch-repeats must be positive")
    if args.sampling_epoch_samples < 0:
        raise ValueError("sampling-epoch-samples must be non-negative")
    if args.probability_metric_bins != 0 and args.probability_metric_bins < 16:
        raise ValueError("probability-metric-bins must be zero or at least 16")
    validate_experiment_profile(args)
    for name in (
        "final_phase_pretrained_lr_scale",
        "final_phase_decoder_lr_scale",
    ):
        if not 0 < getattr(args, name) <= 1:
            raise ValueError(f"{name.replace('_', '-')} must be in (0,1]")
    if (
        args.early_stopping_patience < 0
        or args.early_stopping_min_delta < 0
        or args.early_stopping_start_epoch < 0
    ):
        raise ValueError("Early-stopping settings must be non-negative")

    _set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    complete_path = args.output_dir / "complete"
    if complete_path.exists():
        print(f"Joint run is already complete: {args.output_dir}", flush=True)
        return
    last_checkpoint = args.output_dir / "checkpoint_last.pt"
    if args.resume is None and last_checkpoint.exists():
        raise FileExistsError(f"Use --resume for existing run: {last_checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = _loader(args.train_manifest, args, training=True)
    val_loader = _loader(args.val_manifest, args, training=False)
    system = JointSegmentationSystem(
        decoder_version=args.decoder_version,
        stage1_config=args.stage1_config,
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_config=args.stage2_config,
        stage2_checkpoint=args.stage2_checkpoint,
        num_classes=args.num_classes,
        decoder_drop_path_rate=args.decoder_drop_path_rate,
    ).to(device)
    optimizer, group_metadata = build_joint_adamw(
        system,
        decoder_lr=args.decoder_lr,
        stage2_lr=args.stage2_lr,
        stage1_fusion_lr=args.stage1_fusion_lr,
        stage1_backbone_lr=args.stage1_backbone_lr,
        layer_decay=args.layer_decay,
        weight_decay=args.weight_decay,
    )
    train_batches = (
        min(len(train_loader), args.max_train_batches)
        if args.max_train_batches > 0
        else len(train_loader)
    )
    updates_per_epoch = math.ceil(train_batches / args.gradient_accumulation)
    total_steps = max(1, updates_per_epoch * args.epochs)
    warmup_steps = min(total_steps - 1, round(total_steps * args.warmup_ratio))
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_ratio=args.min_lr_ratio,
    )
    scaler = _make_scaler(device.type == "cuda" and args.amp_dtype == "fp16")

    runtime = system.stage2_runtime
    train_graphs = JointGraphRepository(
        args.graph_dir,
        expected_edge_mode=runtime.context_edge_mode,
    )
    val_graphs = JointGraphRepository(
        args.graph_dir,
        expected_edge_mode=runtime.context_edge_mode,
    )
    run_manifest = {
        "format_version": 1,
        "model_version": system.model_version,
        "configuration": _configuration(args),
        "stage1_config": str(args.stage1_config),
        "stage1_checkpoint": str(args.stage1_checkpoint),
        "stage2_config": str(args.stage2_config),
        "stage2_checkpoint": str(args.stage2_checkpoint),
        "graph_dir": str(args.graph_dir),
        "stage2_runtime": runtime.__dict__,
        "sampling": getattr(
            train_loader.sampler,
            "summary",
            {
                "name": "uniform_patch",
                "num_samples": len(train_loader.dataset),
            },
        ),
        "optimizer_groups": group_metadata,
        "scheduler": {
            "name": "linear_warmup_single_cosine_decay",
            "updates_per_epoch": updates_per_epoch,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "minimum_lr_ratio": args.min_lr_ratio,
        },
        "gradual_unfreezing": {
            "decoder_only_epochs": args.decoder_only_epochs,
            "stage1_top_unfreeze_epoch": args.stage1_top_unfreeze_epoch,
            "stage1_unfreeze_blocks": args.stage1_unfreeze_blocks,
            "final_phase_pretrained_lr_scale": args.final_phase_pretrained_lr_scale,
            "final_phase_decoder_lr_scale": args.final_phase_decoder_lr_scale,
        },
        "early_stopping": {
            "patience": args.early_stopping_patience,
            "minimum_delta": args.early_stopping_min_delta,
            "start_epoch": args.early_stopping_start_epoch,
        },
        "paper_references": PAPER_REFERENCES,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    start_epoch = 0
    best_dice = -1.0
    history_path = args.output_dir / "history.json"
    history: list[dict] = []
    gradient_audit: dict = {}
    early_stopping_best = -1.0
    epochs_without_improvement = 0
    if args.resume is not None:
        checkpoint = _load(args.resume)
        if checkpoint.get("model_version") != system.model_version:
            raise ValueError("Resume checkpoint model version differs")
        checkpoint_configuration = dict(checkpoint.get("configuration", {}))
        # Checkpoints created before the opt-in class weighting flag have the
        # same semantics as the new default and remain safely resumable.
        backward_compatible_defaults = {
            "experiment_profile": "current",
            "tumor_class_weight": 1.0,
            "cross_entropy_weight": 1.0,
            "overlap_loss": "dice",
            "tversky_alpha": 0.3,
            "tversky_beta": 0.7,
            "sampling_mode": "uniform",
            "sampling_positive_fraction": 0.60,
            "sampling_boundary_positive_fraction": 0.50,
            "sampling_interior_threshold": 0.999999,
            "sampling_slide_balance_power": 0.5,
            "sampling_max_patch_repeats": 2,
            "sampling_epoch_samples": 0,
            "color_augmentation": "none",
            "probability_metric_bins": 0,
        }
        for name, value in backward_compatible_defaults.items():
            checkpoint_configuration.setdefault(name, value)
        if checkpoint_configuration != _configuration(args):
            raise ValueError("Resume checkpoint configuration differs")
        system.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint.get("best_dice", best_dice))
        gradient_audit.update(checkpoint.get("gradient_audit", {}))
        early_stopping_best = float(
            checkpoint.get("early_stopping_best", best_dice)
        )
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        if history_path.is_file():
            history = [
                record
                for record in json.loads(history_path.read_text(encoding="utf-8"))
                if int(record["epoch"]) < start_epoch
            ]

    for epoch in range(start_epoch, args.epochs):
        set_sampler_epoch = getattr(train_loader.sampler, "set_epoch", None)
        if callable(set_sampler_epoch):
            set_sampler_epoch(epoch)
        phase = _training_phase(system, optimizer, scheduler, epoch, args)
        print(
            json.dumps(
                {
                    "training_phase": phase,
                    "active_lr_min": min(
                        group["lr"]
                        for group in optimizer.param_groups
                        if group["lr"] > 0
                    ),
                    "active_lr_max": max(group["lr"] for group in optimizer.param_groups),
                }
            ),
            flush=True,
        )
        train_metrics = _run_epoch(
            system,
            train_graphs,
            train_loader,
            device,
            args,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            gradient_audit=gradient_audit,
            training_phase=phase,
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                system,
                val_graphs,
                val_loader,
                device,
                args,
            )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr_min": min(group["lr"] for group in optimizer.param_groups),
            "lr_max": max(group["lr"] for group in optimizer.param_groups),
            "training_phase": phase["name"],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        improved = val_metrics["tumor_dice"] > best_dice
        best_dice = max(best_dice, val_metrics["tumor_dice"])
        should_check_early_stopping = (
            args.early_stopping_patience > 0
            and epoch >= args.early_stopping_start_epoch
        )
        early_stopping_improved = (
            val_metrics["tumor_dice"]
            > early_stopping_best + args.early_stopping_min_delta
        )
        if should_check_early_stopping:
            if early_stopping_improved:
                early_stopping_best = val_metrics["tumor_dice"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        elif val_metrics["tumor_dice"] > early_stopping_best:
            early_stopping_best = val_metrics["tumor_dice"]
        state = {
            "format_version": 1,
            "model_version": system.model_version,
            "epoch": epoch,
            "model": system.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_dice": best_dice,
            "configuration": _configuration(args),
            "gradient_audit": gradient_audit,
            "early_stopping_best": early_stopping_best,
            "epochs_without_improvement": epochs_without_improvement,
            "run_manifest": run_manifest,
        }
        _atomic_torch_save(state, last_checkpoint)
        if improved:
            _atomic_torch_save(state, args.output_dir / "checkpoint_best.pt")
        history_path.write_text(
            json.dumps(history, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "gradient_audit.json").write_text(
            json.dumps(gradient_audit, indent=2) + "\n", encoding="utf-8"
        )
        if (
            should_check_early_stopping
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            early_stopping = {
                "stopped_early": True,
                "stop_epoch": epoch,
                "best_dice": best_dice,
                "reference_dice": early_stopping_best,
                "epochs_without_improvement": epochs_without_improvement,
                "patience": args.early_stopping_patience,
                "minimum_delta": args.early_stopping_min_delta,
                "start_epoch": args.early_stopping_start_epoch,
            }
            (args.output_dir / "early_stopping.json").write_text(
                json.dumps(early_stopping, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps({"early_stopping": early_stopping}), flush=True)
            break
    if not gradient_audit.get("complete", False):
        raise RuntimeError(
            "Joint training ended before the Stage1 DINO backbone, Stage1 fusion, "
            "Stage2 and decoder all received "
            f"a non-zero supervised gradient: {gradient_audit}"
        )
    early_stopping_path = args.output_dir / "early_stopping.json"
    if not early_stopping_path.exists():
        early_stopping_path.write_text(
            json.dumps(
                {
                    "stopped_early": False,
                    "stop_epoch": history[-1]["epoch"],
                    "best_dice": best_dice,
                    "reference_dice": early_stopping_best,
                    "epochs_without_improvement": epochs_without_improvement,
                    "patience": args.early_stopping_patience,
                    "minimum_delta": args.early_stopping_min_delta,
                    "start_epoch": args.early_stopping_start_epoch,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    complete_path.touch()


if __name__ == "__main__":
    main()
