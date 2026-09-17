"""Jointly fine-tune Stage1, Stage2 GATv2, and versioned segmentation decoders."""

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
    remap_warmup_cosine_state,
)
from dinov2_segmentation.losses import segmentation_loss
from dinov2_segmentation.probability_metrics import (
    BinaryProbabilityMetrics,
    binary_confusion_metrics,
)
from dinov2_segmentation.profiles import validate_experiment_profile
from dinov2_segmentation.sampling import (
    SlideStratifiedSampler,
    WSILocalStratifiedSampler,
)


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


# Probability calibration scans every valid pixel and is not part of the
# optimization objective. Keep it off the training hot path and run it only on
# periodic/full validation epochs. Confusion-derived metrics remain exact on
# every epoch.
_VALIDATION_PROBABILITY_METRIC_INTERVAL = 5
_HISTORY_CONFUSION_METRICS = (
    "tumor_dice",
    "tumor_precision",
    "tumor_recall",
    "tumor_f2",
    "predicted_tumor_fraction",
)
_HISTORY_PROBABILITY_METRICS = (
    "approx_pr_auc",
    "best_f2_threshold",
    "best_threshold_f2",
)


def _load(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-version", choices=("v1", "v2", "v3", "v4", "v5"), required=True)
    parser.add_argument("--diffusion-loss-weight", type=float, default=0.1)
    parser.add_argument("--diffusion-reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--diffusion-edge-weight", type=float, default=0.5)
    parser.add_argument("--diffusion-boundary-boost", type=float, default=4.0)
    parser.add_argument("--diffusion-boundary-radius", type=int, default=2)
    parser.add_argument(
        "--decoder-drop-path-rate",
        type=float,
        default=0.1,
        help="Maximum stochastic-depth probability inside the decoder",
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--graph-feature-policy", choices=("legacy", "staged_consistent"), default="legacy")
    parser.add_argument("--raw-feature-cache", type=Path)
    parser.add_argument("--cache-frozen-dino-prefix", action="store_true",
                        help="Cache canonical tokens before the trainable DINO suffix on disk")
    parser.add_argument("--node-image-root", type=Path)
    parser.add_argument("--neighbor-chunk-size", type=int, default=8)
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
    parser.add_argument("--retain-progress-checkpoints", action="store_true")
    parser.add_argument("--async-full-validation", action="store_true",
                        help="Queue retained epoch weights; never run validation in the trainer")
    parser.add_argument("--monitor-interval-steps", type=int, default=2000)
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
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20_000,
        help=(
            "Fixed optimizer-update warmup. For short smoke tests it is capped at "
            "total_steps - 1"
        ),
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help=argparse.SUPPRESS,
    )
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
        choices=("uniform", "slide_stratified", "wsi_local_stratified"),
        default="uniform",
    )
    parser.add_argument(
        "--sampling-locality-tile-size",
        type=int,
        default=4096,
        help=(
            "Coordinate tile size for WSI-local target ordering; this only "
            "changes training I/O locality, not graph neighbourhood semantics"
        ),
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
        "--unfreeze-schedule", choices=("legacy", "separate_gnn_fusion"),
        default="legacy",
        help="Separate decoder, GNN, Stage1 fusion, and DINO phases; legacy preserves existing runs",
    )
    parser.add_argument(
        "--decoder-only-steps",
        type=int,
        default=20_000,
        help="Optimizer updates used to align the decoder before adapters unfreeze",
    )
    parser.add_argument(
        "--stage1-partial-unfreeze-step", "--fusion-unfreeze-step",
        type=int,
        default=60_000,
        help="Fusion unfreeze boundary in separate_gnn_fusion; first DINO boundary in legacy",
    )
    parser.add_argument(
        "--stage1-partial-unfreeze-blocks",
        type=int,
        default=2,
        help="Number of top DINO Transformer blocks trained in the partial phase",
    )
    parser.add_argument(
        "--stage1-final-unfreeze-step", "--dino-unfreeze-step",
        type=int,
        default=100_000,
        help="Optimizer update at which the final top-DINO phase starts",
    )
    parser.add_argument(
        "--stage1-final-unfreeze-blocks", "--dino-unfreeze-blocks",
        type=int,
        default=4,
        help="Number of top DINO Transformer blocks trained in the final phase",
    )
    parser.add_argument(
        "--checkpoint-interval-steps",
        type=int,
        default=20_000,
        help="Write a resumable in-epoch checkpoint every N optimizer updates; zero disables",
    )
    # Kept parseable so an old command fails with a focused migration message
    # instead of argparse's generic unknown-option error. They never control a
    # new run.
    parser.add_argument("--decoder-only-epochs", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--stage1-top-unfreeze-epoch", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--stage1-unfreeze-blocks", type=int, help=argparse.SUPPRESS)
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
    return parser.parse_args(argv)


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


def _loader(path: Path, args: argparse.Namespace, training: bool, execution=None) -> DataLoader:
    dataset = JointPatchSegmentationDataset(
        path,
        image_size=args.image_size,
        training=training,
        color_augmentation=args.color_augmentation if training else "none",
    )
    generator = torch.Generator().manual_seed(args.seed + int(training))
    sampler = None
    if training and args.sampling_mode in {
        "slide_stratified", "wsi_local_stratified"
    }:
        sampler_class = (
            WSILocalStratifiedSampler
            if args.sampling_mode == "wsi_local_stratified"
            else SlideStratifiedSampler
        )
        sampler_kwargs = dict(
            rows=dataset.rows,
            num_samples=args.sampling_epoch_samples or len(dataset),
            # Existing global stratification keeps its historic DDP behaviour.
            # The local sampler deliberately uses a rank-local batch because
            # WholeBatchShardSampler slices the stream at that boundary.
            batch_size=(
                args.batch_size
                if args.sampling_mode == "wsi_local_stratified"
                else args.batch_size
                * (execution.world_size if execution is not None else 1)
            ),
            positive_fraction=args.sampling_positive_fraction,
            boundary_positive_fraction=args.sampling_boundary_positive_fraction,
            interior_threshold=args.sampling_interior_threshold,
            slide_balance_power=args.sampling_slide_balance_power,
            max_patch_repeats=args.sampling_max_patch_repeats,
            seed=args.seed,
        )
        if args.sampling_mode == "wsi_local_stratified":
            sampler_kwargs["locality_tile_size"] = args.sampling_locality_tile_size
            sampler_kwargs["global_batch_size"] = (
                args.batch_size * (execution.world_size if execution is not None else 1)
            )
        sampler = sampler_class(**sampler_kwargs)
    if training:
        from dinov2_segmentation.distributed_execution import (
            EpochRandomSampler,
            WholeBatchSampler,
        )

        if sampler is None:
            sampler = EpochRandomSampler(dataset, seed=args.seed)
        if execution is not None and execution.distributed:
            batch_sampler = execution.shard_batches(
                sampler, args.batch_size, training=True
            )
        else:
            batch_sampler = WholeBatchSampler(sampler, args.batch_size)
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.workers,
            pin_memory=(
                execution.device.type == "cuda"
                if execution is not None
                else torch.cuda.is_available()
            ),
            persistent_workers=args.workers > 0,
            generator=generator,
        )
    if execution is not None and execution.distributed:
        from torch.utils.data import SequentialSampler

        sampler = SequentialSampler(dataset)
        batch_sampler = execution.shard_batches(sampler, args.batch_size, training=False)
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.workers,
            pin_memory=execution.device.type == "cuda",
            persistent_workers=args.workers > 0,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
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


def _phase_spec(curriculum_step: int, args: argparse.Namespace) -> tuple[str, int]:
    """Return the phase name and number of trainable top ViT blocks."""

    curriculum_step = int(curriculum_step)
    if curriculum_step < args.decoder_only_steps:
        return "decoder_only", 0
    if getattr(args, "unfreeze_schedule", "legacy") == "separate_gnn_fusion":
        if curriculum_step < args.stage1_partial_unfreeze_step:
            return "gnn_and_decoder", 0
        if curriculum_step < args.stage1_final_unfreeze_step:
            return "fusion_gnn_decoder", 0
        return "dino_joint", int(args.stage1_final_unfreeze_blocks)
    if curriculum_step < args.stage1_partial_unfreeze_step:
        return "adapters_and_decoder", 0
    if curriculum_step < args.stage1_final_unfreeze_step:
        return "top2_backbone_joint", int(args.stage1_partial_unfreeze_blocks)
    return "top4_backbone_joint", int(args.stage1_final_unfreeze_blocks)


def _training_phase(
    system: JointSegmentationSystem,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    curriculum_step: int,
    args: argparse.Namespace,
    *,
    epoch: int | None = None,
) -> dict:
    """Configure step-based unfreezing and phase-specific LR multipliers."""

    curriculum_step = int(curriculum_step)
    phase_name, unfreeze_blocks = _phase_spec(curriculum_step, args)

    backbone_layers = [
        int(match.group(1))
        for group in optimizer.param_groups
        if (match := _BACKBONE_GROUP.match(str(group.get("group_name", ""))))
    ]
    final_layer = max(backbone_layers, default=0)
    if unfreeze_blocks > 0:
        first_enabled_backbone_layer = max(
            1, final_layer - unfreeze_blocks
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
                if phase_name in ("top4_backbone_joint", "dino_joint")
                else 1.0
            )
        elif name.startswith("stage2_") or (name.startswith("stage1_") and match is None):
            active = phase_name != "decoder_only"
            if name.startswith("stage1_") and phase_name == "gnn_and_decoder":
                active = False
            scale = (
                args.final_phase_pretrained_lr_scale
                if phase_name in ("top4_backbone_joint", "dino_joint")
                else 1.0
            )
        elif match is not None:
            layer_id = int(match.group(1))
            active = unfreeze_blocks > 0 and (
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
    # Legacy resumes retain their previous cache behavior. In the new schedule,
    # persistent graph features only change once their Stage1 producer trains.
    system.refresh_graph_memory = (
        getattr(args, "unfreeze_schedule", "legacy") == "legacy"
        or phase_name not in ("decoder_only", "gnn_and_decoder")
    )
    if getattr(args, "graph_feature_policy", "legacy") == "staged_consistent":
        system.refresh_graph_memory = False

    phase = {
        "name": phase_name,
        "curriculum_step": curriculum_step,
        "epoch": epoch,
        "active_optimizer_groups": len(active_groups),
        "total_optimizer_groups": len(optimizer.param_groups),
        "first_enabled_backbone_layer": (
            first_enabled_backbone_layer
            if unfreeze_blocks > 0
            else None
        ),
        "stage1_unfreeze_blocks": int(unfreeze_blocks),
        "refresh_graph_memory": system.refresh_graph_memory,
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
    if phase["name"] == "gnn_and_decoder":
        system.stage1.eval()
        return
    if phase["name"] in ("adapters_and_decoder", "fusion_gnn_decoder"):
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
        encoded, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)


def _metrics(
    totals: dict,
    samples: int,
    confusion: torch.Tensor,
    probability_metrics: BinaryProbabilityMetrics | None = None,
) -> dict:
    denominator = max(samples, 1)
    result = {
        name: (
            float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
        )
        / denominator
        for name, value in totals.items()
    }
    if 'boundary_predicted_pixels' in totals:
        from dinov2_segmentation.diffusion_losses import boundary_scores, BOUNDARY_COUNTS
        result.update(boundary_scores(totals))
        for name in BOUNDARY_COUNTS:
            result.pop(name, None)
    confusion = confusion.detach().cpu()
    confusion_metrics = binary_confusion_metrics(confusion)
    result.update(
        {name: confusion_metrics[name] for name in _HISTORY_CONFUSION_METRICS}
    )
    result["confusion"] = confusion.tolist()
    if probability_metrics is not None:
        probability_results = probability_metrics.compute()
        result.update(
            {name: probability_results[name] for name in _HISTORY_PROBABILITY_METRICS}
        )
    return result


def _should_collect_probability_metrics(epoch: int, total_epochs: int) -> bool:
    """Collect calibration diagnostics periodically and on the final epoch."""
    completed_epoch = int(epoch) + 1
    return (
        completed_epoch % _VALIDATION_PROBABILITY_METRIC_INTERVAL == 0
        or completed_epoch == int(total_epochs)
    )


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
    curriculum_state: dict | None = None,
    epoch: int | None = None,
    batch_offset: int = 0,
    initial_train_progress: dict | None = None,
    progress_callback=None,
    execution=None,
    collect_probability_metrics: bool = False,
) -> dict:
    training = optimizer is not None
    if training and (scheduler is None or scaler is None):
        raise ValueError("Training requires scheduler and scaler")
    if training and curriculum_state is None:
        # Backward-compatible support for focused unit tests that call the
        # epoch runner directly. Production always supplies the persistent
        # state so phase progress is checkpointed independently of LR time.
        curriculum_state = {
            "step": int((training_phase or {}).get("curriculum_step", 0)),
            "phase_name": (training_phase or {}).get("name"),
            "phase_transitions": [],
        }
    active_phase = training_phase
    if training:
        configured = _training_phase(
            system,
            optimizer,
            scheduler,
            int(curriculum_state["step"]),
            args,
            epoch=epoch,
        )
        previous_name = curriculum_state.get("phase_name")
        if previous_name != configured["name"]:
            curriculum_state.setdefault("phase_transitions", []).append(
                {
                    "optimizer_step": int(curriculum_state["step"]),
                    "from": previous_name,
                    "to": configured["name"],
                }
            )
        curriculum_state["phase_name"] = configured["name"]
        active_phase = configured
        if execution is not None:
            execution.prepare_model(system, phase_signature=configured["name"])
    _set_runtime_modes(system, training=training, phase=active_phase)
    overlap_metric = (
        "dice_loss" if args.overlap_loss == "dice" else "tversky_loss"
    )
    metric_names = ("loss", "cross_entropy", overlap_metric)
    auxiliary_version = getattr(system, 'decoder_version', None) in ('v4', 'v5')
    if auxiliary_version:
        from dinov2_segmentation.diffusion_losses import DIFFUSION_METRICS, BOUNDARY_COUNTS
        metric_names += ('segmentation_loss',) + BOUNDARY_COUNTS
        if training:
            metric_names += DIFFUSION_METRICS[1:]
    seed_progress = bool(
        initial_train_progress is not None
        and (execution is None or execution.is_primary)
    )
    saved_totals = (
        initial_train_progress.get("totals", {}) if seed_progress else {}
    )
    totals = {
        name: torch.tensor(
            float(saved_totals.get(name, 0.0)), device=device, dtype=torch.float64
        )
        for name in metric_names
    }
    if seed_progress:
        saved_confusion = torch.as_tensor(
            initial_train_progress.get("confusion", []), dtype=torch.int64
        )
        if tuple(saved_confusion.shape) != (args.num_classes, args.num_classes):
            raise ValueError("Progress checkpoint confusion matrix has the wrong shape")
        confusion = saved_confusion.to(device=device)
        samples = int(initial_train_progress.get("samples", 0))
    else:
        confusion = torch.zeros(
            (args.num_classes, args.num_classes), device=device, dtype=torch.int64
        )
        samples = 0
    probability_metrics = (
        BinaryProbabilityMetrics(args.probability_metric_bins, device=device)
        if (
            not training
            and collect_probability_metrics
            and args.probability_metric_bins > 0
        )
        else None
    )
    accumulation = args.gradient_accumulation if training else 1
    micro_count = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    maximum = args.max_train_batches if training else args.max_val_batches
    batch_offset = int(batch_offset)
    if batch_offset < 0:
        raise ValueError("batch_offset must be non-negative")
    remaining_batches = len(loader)
    if maximum > 0:
        remaining_batches = min(
            remaining_batches, max(0, int(maximum) - batch_offset)
        )
    total_batches = batch_offset + remaining_batches
    start = time.monotonic()

    for relative_batch_index, batch in enumerate(loader):
        if relative_batch_index >= remaining_batches:
            break
        batch_index = batch_offset + relative_batch_index
        if training:
            desired_name, _ = _phase_spec(int(curriculum_state["step"]), args)
            if desired_name != active_phase["name"]:
                if micro_count != 0:
                    raise RuntimeError(
                        "A training phase cannot change inside a gradient-accumulation window"
                    )
                previous_name = active_phase["name"]
                active_phase = _training_phase(
                    system,
                    optimizer,
                    scheduler,
                    int(curriculum_state["step"]),
                    args,
                    epoch=epoch,
                )
                curriculum_state["phase_name"] = active_phase["name"]
                transition = {
                    "optimizer_step": int(curriculum_state["step"]),
                    "from": previous_name,
                    "to": active_phase["name"],
                }
                curriculum_state.setdefault("phase_transitions", []).append(transition)
                _set_runtime_modes(system, training=True, phase=active_phase)
                if execution is not None:
                    execution.prepare_model(
                        system, phase_signature=active_phase["name"]
                    )
                if execution is None or execution.is_primary:
                    print(json.dumps({"training_phase_transition": transition}), flush=True)
        if training and getattr(graph_repository, "feature_provider", None) is not None:
            from dinov2_segmentation.consistent_features import update_repository_stage
            update_repository_stage(graph_repository, int(curriculum_state["step"]), args)
        images = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            if execution is None:
                node_features, dense_tokens = system.stage1(images)
                contexts = graph_repository.contextualize(
                    system.stage2,
                    node_features,
                    list(batch["slide_id"]),
                    list(batch["patch_id"]),
                    num_hops=system.stage2_runtime.num_layers,
                    use_edge_attr=system.stage2_runtime.use_edge_attr,
                    update_memory=training and getattr(system, "refresh_graph_memory", True),
                )
                logits = system.decode(images, dense_tokens, contexts)
            else:
                logits = execution.forward(
                    system, graph_repository, images,
                    list(batch["slide_id"]), list(batch["patch_id"]),
                    training=training,
                )
            auxiliary_output = None
            if isinstance(logits, dict):
                auxiliary_output = logits.get('diffusion')
                logits = logits['logits']
            loss_function = segmentation_loss
            if training and execution is not None and execution.distributed:
                from dinov2_segmentation.distributed_losses import segmentation_loss_distributed

                loss_function = segmentation_loss_distributed
            loss, parts = loss_function(
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
            if auxiliary_version:
                parts['segmentation_loss'] = loss
                if training:
                    if auxiliary_output is None:
                        raise RuntimeError('V4/V5 training requires diffusion output inside the DDP forward')
                    from dinov2_segmentation.diffusion_losses import diffusion_auxiliary_loss
                    auxiliary_loss, auxiliary_parts = diffusion_auxiliary_loss(
                        auxiliary_output, target, ignore_index=args.ignore_index,
                        boundary_boost=args.diffusion_boundary_boost,
                        boundary_radius=args.diffusion_boundary_radius,
                        reconstruction_weight=args.diffusion_reconstruction_weight,
                        edge_weight=args.diffusion_edge_weight,
                        distributed=bool(execution is not None and execution.distributed),
                    )
                    parts.update(auxiliary_parts)
                    loss = loss + args.diffusion_loss_weight * auxiliary_loss
        finite = (
            execution.all_finite(loss)
            if training and execution is not None else bool(torch.isfinite(loss))
        )
        if not finite:
            raise FloatingPointError(
                f"Non-finite joint loss at batch {batch_index}: {loss}"
            )

        count = images.size(0)
        samples += count
        totals["loss"].add_(loss.detach().to(torch.float64), alpha=count)
        for name, value in parts.items():
            totals[name].add_(value.detach().to(torch.float64), alpha=count)
        if auxiliary_version:
            from dinov2_segmentation.diffusion_losses import boundary_counts
            for name, value in boundary_counts(logits, target, args.ignore_index).items():
                totals[name].add_(value.to(torch.float64))
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

        if training:
            scaler.scale(loss).backward()
            micro_count += 1
            should_step = (
                micro_count == accumulation
                or relative_batch_index + 1 == remaining_batches
            )
            if should_step:
                scaler.unscale_(optimizer)
                # Average accumulated micro-batch gradients without weakening
                # the final, possibly shorter accumulation window.
                for parameter in system.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(micro_count)
                audit_enabled = (
                    gradient_audit is not None
                    and active_phase is not None
                    and active_phase["name"] != "decoder_only"
                )
                if audit_enabled and not gradient_audit.get("complete", False):
                    observed = {
                        "stage1_grad_norm": _gradient_norm(system.stage1),
                        "stage2_grad_norm": _gradient_norm(system.stage2),
                        "decoder_grad_norm": _gradient_norm(system.decoder),
                    }
                    full_joint_phase = active_phase["name"] in ("top4_backbone_joint", "dino_joint")
                    if full_joint_phase:
                        observed["stage1_backbone_grad_norm"] = _gradient_norm(
                            system.stage1.backbone
                        )
                    if execution is not None:
                        observed = execution.reduce_max_values(observed)
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
                previous_scale = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                update_was_skipped = bool(
                    scaler.is_enabled() and float(scaler.get_scale()) < previous_scale
                )
                optimizer.zero_grad(set_to_none=True)
                if not update_was_skipped:
                    scheduler.step()
                    curriculum_state["step"] = int(curriculum_state["step"]) + 1
                micro_count = 0
                should_checkpoint = bool(
                    not update_was_skipped
                    and progress_callback is not None
                    and args.checkpoint_interval_steps > 0
                    and int(curriculum_state["step"]) > 0
                    and int(curriculum_state["step"])
                    % int(args.checkpoint_interval_steps)
                    == 0
                )
                if should_checkpoint:
                    if execution is not None:
                        snapshot = execution.reduce_metrics(
                            totals, samples, confusion, None
                        )
                        snapshot_totals, snapshot_samples, snapshot_confusion, _ = snapshot
                    else:
                        snapshot_totals = {
                            name: float(value.detach().cpu())
                            for name, value in totals.items()
                        }
                        snapshot_samples = samples
                        snapshot_confusion = confusion.detach().cpu()
                    progress_callback(
                        {
                            "next_batch_index": batch_index + 1,
                            "total_batches": total_batches,
                            "totals": snapshot_totals,
                            "samples": int(snapshot_samples),
                            "confusion": snapshot_confusion.tolist(),
                        }
                    )
        if (
            training and (batch_index + 1) % args.log_interval == 0
            and (execution is None or execution.is_primary)
        ):
            elapsed = time.monotonic() - start
            print(
                json.dumps(
                    {
                        "batch": batch_index + 1,
                        "batches": total_batches,
                        "optimizer_step": int(curriculum_state["step"]),
                        "training_phase": active_phase["name"],
                        "loss": float(loss.detach()),
                        **({name: float(parts[name].detach()) for name in DIFFUSION_METRICS}
                           if auxiliary_version else {}),
                        "seconds": elapsed,
                        "lr_min": min(group["lr"] for group in optimizer.param_groups),
                        "lr_max": max(group["lr"] for group in optimizer.param_groups),
                    }
                ),
                flush=True,
            )
    if execution is not None:
        totals, samples, confusion, probability_metrics = execution.reduce_metrics(
            totals, samples, confusion, probability_metrics
        )
    return _metrics(totals, samples, confusion, probability_metrics)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_save(payload, path: Path) -> None:
    """Write metadata without exposing a partially written JSON file."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
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
        "warmup_steps",
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
        "sampling_locality_tile_size",
        "color_augmentation",
        "probability_metric_bins",
        "amp_dtype",
        "seed",
        "gradient_audit_updates",
        "decoder_only_steps",
        "stage1_partial_unfreeze_step",
        "stage1_partial_unfreeze_blocks",
        "stage1_final_unfreeze_step",
        "stage1_final_unfreeze_blocks",
        "checkpoint_interval_steps",
        "final_phase_pretrained_lr_scale",
        "final_phase_decoder_lr_scale",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "early_stopping_start_epoch",
    )
    if args.decoder_version in ('v4', 'v5'):
        keys += ('diffusion_loss_weight', 'diffusion_reconstruction_weight',
                 'diffusion_edge_weight', 'diffusion_boundary_boost', 'diffusion_boundary_radius')
    configuration = {key: getattr(args, key) for key in keys}
    if getattr(args, "unfreeze_schedule", "legacy") != "legacy":
        configuration["unfreeze_schedule"] = args.unfreeze_schedule
    if getattr(args, "graph_feature_policy", "legacy") != "legacy":
        if getattr(args, "unfreeze_schedule", "legacy") != "separate_gnn_fusion":
            raise ValueError("staged_consistent requires separate_gnn_fusion scheduling")
        if args.raw_feature_cache is None or args.node_image_root is None or args.neighbor_chunk_size < 1:
            raise ValueError("staged_consistent requires a raw cache, image root and positive chunk size")
        configuration.update(graph_feature_policy=args.graph_feature_policy,
                             raw_feature_cache=str(args.raw_feature_cache.resolve()),
                             node_image_root=str(args.node_image_root.resolve()),
                             neighbor_chunk_size=args.neighbor_chunk_size)
    if getattr(args, "cache_frozen_dino_prefix", False):
        if configuration.get("graph_feature_policy") != "staged_consistent":
            raise ValueError("Frozen prefix caching requires staged_consistent features")
        configuration["cache_frozen_dino_prefix"] = True
    return configuration


def _migration_history(
    checkpoint_path: Path, completed_epochs: int, checkpoint: dict | None = None
) -> list[dict]:
    history_path = checkpoint_path.parent / "history.json"
    # The checkpoint is the atomic state boundary. Its embedded history must
    # win over the human-readable sidecar, which can lag after an interruption.
    if checkpoint is not None and isinstance(checkpoint.get("history"), list):
        payload = checkpoint["history"]
    elif history_path.is_file():
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        raise ValueError(
            "Checkpoint recovery requires embedded history or history.json beside "
            "the checkpoint: "
            f"{history_path}"
        )
    if not isinstance(payload, list):
        raise ValueError("Serial migration history must be a JSON list")
    history = [
        record for record in payload
        if isinstance(record, dict) and int(record.get("epoch", -1)) < completed_epochs
    ]
    epochs = [int(record["epoch"]) for record in history]
    if epochs != list(range(completed_epochs)):
        raise ValueError(
            "Serial migration history is not a contiguous prefix ending at the "
            f"checkpoint epoch: {epochs}"
        )
    return history


_LEGACY_SCHEDULE_CONFIGURATION = {
    "warmup_ratio",
    "decoder_only_epochs",
    "stage1_top_unfreeze_epoch",
    "stage1_unfreeze_blocks",
}
_STEP_SCHEDULE_CONFIGURATION = {
    "warmup_steps",
    "decoder_only_steps",
    "stage1_partial_unfreeze_step",
    "stage1_partial_unfreeze_blocks",
    "stage1_final_unfreeze_step",
    "stage1_final_unfreeze_blocks",
    "checkpoint_interval_steps",
}


def _migration_configuration_changes(
    source: dict, target: dict
) -> tuple[dict, bool]:
    """Validate an explicit serial-to-DDP configuration conversion.

    Scientific settings remain strict. The only permitted changes are the
    per-rank batch geometry, checkpoint frequency, and the one-time authorized
    conversion from the historical epoch curriculum to the step curriculum.
    """

    source_is_legacy = bool(_LEGACY_SCHEDULE_CONFIGURATION & set(source))
    ignored = set(_STEP_SCHEDULE_CONFIGURATION)
    if source_is_legacy:
        ignored |= _LEGACY_SCHEDULE_CONFIGURATION
    source_common = {key: value for key, value in source.items() if key not in ignored}
    target_common = {key: value for key, value in target.items() if key not in ignored}
    if set(source_common) != set(target_common):
        raise ValueError(
            "Serial migration configuration keys differ outside the authorized "
            "schedule conversion: "
            f"missing={sorted(set(target_common) - set(source_common))}, "
            f"unexpected={sorted(set(source_common) - set(target_common))}"
        )
    incompatible = {
        name: (source_common[name], target_common[name])
        for name in target_common
        if name != "batch_size" and source_common[name] != target_common[name]
    }
    if incompatible:
        raise ValueError(
            "Serial migration only permits batch geometry and the explicit "
            f"epoch-to-step schedule conversion; differing values: {incompatible}"
        )
    if not source_is_legacy:
        schedule_differences = {
            name: (source.get(name), target.get(name))
            for name in _STEP_SCHEDULE_CONFIGURATION - {"checkpoint_interval_steps"}
            if source.get(name) != target.get(name)
        }
        if schedule_differences:
            raise ValueError(
                "Step-based curriculum settings differ during migration: "
                f"{schedule_differences}"
            )
    changes = {
        "batch_size": {
            "source": int(source["batch_size"]),
            "target_per_rank": int(target["batch_size"]),
        }
    }
    if source_is_legacy:
        changes["training_schedule"] = {
            "source": {
                key: source[key]
                for key in sorted(_LEGACY_SCHEDULE_CONFIGURATION)
                if key in source
            },
            "target": {
                key: target[key]
                for key in sorted(_STEP_SCHEDULE_CONFIGURATION)
                if key in target
            },
        }
    elif source.get("checkpoint_interval_steps") != target.get(
        "checkpoint_interval_steps"
    ):
        changes["checkpoint_interval_steps"] = {
            "source": source.get("checkpoint_interval_steps"),
            "target": target.get("checkpoint_interval_steps"),
        }
    return changes, source_is_legacy


def _validate_migration_optimizer(
    checkpoint: dict,
    optimizer: torch.optim.Optimizer,
    source_manifest: dict,
    target_group_metadata: list[dict],
) -> None:
    source_optimizer = checkpoint["optimizer"]
    if not isinstance(source_optimizer, dict):
        raise ValueError("Serial migration optimizer state must be a dictionary")
    source_groups = source_optimizer.get("param_groups")
    target_groups = optimizer.state_dict().get("param_groups")
    if not isinstance(source_groups, list) or len(source_groups) != len(target_groups):
        raise ValueError("Serial and DDP optimizer group counts differ")
    stable_group_fields = (
        "group_name",
        "weight_decay",
        "betas",
        "eps",
        "amsgrad",
        "maximize",
    )
    for index, (source, target) in enumerate(zip(source_groups, target_groups)):
        if len(source.get("params", ())) != len(target.get("params", ())):
            raise ValueError(f"Optimizer parameter count differs in group {index}")
        for name in stable_group_fields:
            if source.get(name) != target.get(name):
                raise ValueError(
                    f"Optimizer group {index} field {name!r} differs during migration"
                )
    if source_manifest.get("optimizer_groups") != target_group_metadata:
        raise ValueError("Serial and DDP optimizer group metadata differ")


def _prepare_serial_to_ddp_migration(
    checkpoint_path: Path,
    checkpoint: dict,
    *,
    args: argparse.Namespace,
    system: JointSegmentationSystem,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    target_group_metadata: list[dict],
    target_execution: dict,
    target_sampling: dict,
    target_updates_per_epoch: int,
) -> tuple[list[dict], dict, dict, int, dict | None]:
    required = {
        "format_version",
        "model_version",
        "epoch",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "best_dice",
        "configuration",
        "gradient_audit",
        "early_stopping_best",
        "epochs_without_improvement",
        "run_manifest",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Serial migration checkpoint lacks keys: {missing}")
    if int(checkpoint["format_version"]) not in (1, 2):
        raise ValueError("Unsupported serial migration checkpoint format")
    if checkpoint["model_version"] != system.model_version:
        raise ValueError("Serial migration checkpoint model version differs")
    if not isinstance(checkpoint["gradient_audit"], dict):
        raise ValueError("Serial migration gradient audit must be a dictionary")
    if not isinstance(checkpoint["scaler"], dict):
        raise ValueError("Serial migration scaler state must be a dictionary")

    source_configuration = checkpoint["configuration"]
    target_configuration = _configuration(args)
    if not isinstance(source_configuration, dict):
        raise ValueError("Serial migration configuration must be a dictionary")
    configuration_changes, source_is_legacy = _migration_configuration_changes(
        source_configuration, target_configuration
    )

    source_manifest = checkpoint["run_manifest"]
    if not isinstance(source_manifest, dict):
        raise ValueError("Serial migration run manifest must be a dictionary")
    if source_manifest.get("configuration") != source_configuration:
        raise ValueError("Checkpoint and run-manifest configurations differ")
    source_execution = checkpoint.get("execution")
    manifest_execution = source_manifest.get("execution")
    if source_execution is None:
        if manifest_execution is not None:
            raise ValueError("Serial checkpoint execution metadata is inconsistent")
        source_execution_record = {
            "mode": "serial",
            "world_size": 1,
            "legacy_without_execution_manifest": True,
        }
    else:
        if source_execution != manifest_execution:
            raise ValueError("Serial checkpoint execution metadata is inconsistent")
        if (
            source_execution.get("mode") != "serial"
            or int(source_execution.get("world_size", 0)) != 1
        ):
            raise ValueError("--migrate-resume accepts only a serial checkpoint")
        source_execution_record = source_execution

    for name in (
        "graph_dir",
        "stage1_config",
        "stage1_checkpoint",
        "stage2_config",
        "stage2_checkpoint",
    ):
        source = source_manifest.get(name)
        if source is None or Path(source).expanduser().resolve() != getattr(args, name):
            raise ValueError(f"Serial migration source {name} differs")
    if source_manifest.get("sampling") != target_sampling:
        raise ValueError("Serial migration sampling population differs")

    source_batch_size = int(source_configuration["batch_size"])
    source_accumulation = int(source_configuration["gradient_accumulation"])
    source_effective_batch = source_batch_size * source_accumulation
    target_effective_batch = int(target_execution["effective_batch_size"])
    if source_effective_batch != target_effective_batch:
        raise ValueError(
            "Serial and DDP effective batch sizes differ: "
            f"serial={source_effective_batch}, ddp={target_effective_batch} "
            f"(world_size={target_execution['world_size']}, "
            f"per_rank={target_execution['per_rank_batch_size']}, "
            f"gradient_accumulation={args.gradient_accumulation})"
        )
    target_global_micro_batch = (
        int(target_execution["per_rank_batch_size"])
        * int(target_execution["world_size"])
    )
    if source_batch_size != target_global_micro_batch:
        raise ValueError(
            "Serial and DDP global micro-batch sizes differ; an in-epoch sample "
            f"cursor cannot be preserved: serial={source_batch_size}, "
            f"ddp={target_global_micro_batch}"
        )
    if source_accumulation != int(args.gradient_accumulation):
        raise ValueError(
            "Serial and DDP gradient-accumulation windows differ; in-epoch "
            "migration requires identical optimizer-update boundaries"
        )
    if source_execution is not None:
        recorded_effective_batch = int(source_execution.get("effective_batch_size", -1))
        if recorded_effective_batch != source_effective_batch:
            raise ValueError("Serial checkpoint effective batch metadata is inconsistent")
        for name, target_source in target_execution["sources"].items():
            if source_execution.get("sources", {}).get(name) != target_source:
                raise ValueError(f"Serial migration execution source {name} differs")
        for name in ("max_train_batches", "max_val_batches"):
            if source_execution.get(name) != target_execution.get(name):
                raise ValueError(f"Serial migration {name} differs")
    elif args.max_train_batches or args.max_val_batches:
        raise ValueError(
            "Legacy serial checkpoints without execution metadata can migrate only "
            "with max-train-batches=0 and max-val-batches=0"
        )

    _validate_migration_optimizer(
        checkpoint, optimizer, source_manifest, target_group_metadata
    )
    source_scheduler_manifest = source_manifest.get("scheduler")
    if not isinstance(source_scheduler_manifest, dict):
        raise ValueError("Serial migration run manifest lacks scheduler metadata")
    if source_scheduler_manifest.get("name") != "linear_warmup_single_cosine_decay":
        raise ValueError("Serial migration scheduler type differs")
    source_scheduler = checkpoint["scheduler"]
    if not isinstance(source_scheduler, dict):
        raise ValueError("Serial migration scheduler state must be a dictionary")
    missing_scheduler_manifest = sorted(
        {"updates_per_epoch", "total_steps", "warmup_steps"}
        - set(source_scheduler_manifest)
    )
    if missing_scheduler_manifest:
        raise ValueError(
            "Serial migration scheduler manifest lacks keys: "
            f"{missing_scheduler_manifest}"
        )
    missing_scheduler_state = sorted(
        {"total_steps", "warmup_steps", "base_lrs"} - set(source_scheduler)
    )
    if missing_scheduler_state:
        raise ValueError(
            f"Serial migration scheduler state lacks keys: {missing_scheduler_state}"
        )
    for name in ("total_steps", "warmup_steps"):
        if int(source_scheduler_manifest[name]) != int(source_scheduler[name]):
            raise ValueError(f"Serial scheduler {name} metadata is inconsistent")
    source_updates_per_epoch = int(source_scheduler_manifest["updates_per_epoch"])
    if source_updates_per_epoch < 1:
        raise ValueError("Serial scheduler updates_per_epoch must be positive")
    source_total_expected = source_updates_per_epoch * int(source_configuration["epochs"])
    if int(source_scheduler["total_steps"]) != max(1, source_total_expected):
        raise ValueError("Serial scheduler total_steps does not match its configuration")
    if list(source_scheduler["base_lrs"]) != list(scheduler.state_dict()["base_lrs"]):
        raise ValueError("Serial and DDP scheduler base learning rates differ")

    epoch = int(checkpoint["epoch"])
    epoch_complete = bool(checkpoint.get("epoch_complete", True))
    completed_epochs = epoch + 1 if epoch_complete else epoch
    if epoch < 0 or completed_epochs > args.epochs:
        raise ValueError(f"Serial migration checkpoint epoch is invalid: {epoch}")
    history = _migration_history(checkpoint_path, completed_epochs, checkpoint)
    target_state = scheduler.state_dict()
    if epoch_complete:
        remapped_scheduler, scheduler_provenance = remap_warmup_cosine_state(
            source_scheduler,
            source_updates_per_epoch=source_updates_per_epoch,
            target_updates_per_epoch=target_updates_per_epoch,
            target_total_steps=int(target_state["total_steps"]),
            target_warmup_steps=int(target_state["warmup_steps"]),
            completed_epochs=completed_epochs,
        )
        resume_progress = None
    else:
        resume_progress = checkpoint.get("train_progress")
        if not isinstance(resume_progress, dict):
            raise ValueError("In-epoch migration checkpoint lacks train_progress")
        next_batch_index = int(checkpoint.get("next_batch_index", -1))
        total_batches = int(resume_progress.get("total_batches", -1))
        if not 0 < next_batch_index <= total_batches:
            raise ValueError("In-epoch migration checkpoint has an invalid batch cursor")
        expected_source_step = min(
            epoch * source_updates_per_epoch
            + math.ceil(next_batch_index / source_accumulation),
            int(source_scheduler["total_steps"]) - 1,
        )
        if int(source_scheduler["current_step"]) != expected_source_step:
            raise ValueError(
                "In-epoch scheduler position disagrees with its batch cursor: "
                f"current={source_scheduler['current_step']}, "
                f"expected={expected_source_step}"
            )
        target_next_batch_index = min(
            next_batch_index, int(target_execution["train_batches_per_rank"])
        )
        target_current_step = min(
            epoch * target_updates_per_epoch
            + math.ceil(target_next_batch_index / int(args.gradient_accumulation)),
            int(target_state["total_steps"]) - 1,
        )
        resume_progress = dict(resume_progress)
        resume_progress["next_batch_index"] = target_next_batch_index
        resume_progress["total_batches"] = int(
            target_execution["train_batches_per_rank"]
        )
        remapped_scheduler = dict(source_scheduler)
        remapped_scheduler.update(
            {
                "total_steps": int(target_state["total_steps"]),
                "warmup_steps": int(target_state["warmup_steps"]),
                "current_step": target_current_step,
            }
        )
        scheduler_provenance = {
            "policy": "absolute_optimizer_update_position",
            "completed_epochs": completed_epochs,
            "next_batch_index": next_batch_index,
            "target_next_batch_index": target_next_batch_index,
            "source": {
                "updates_per_epoch": source_updates_per_epoch,
                "total_steps": int(source_scheduler["total_steps"]),
                "warmup_steps": int(source_scheduler["warmup_steps"]),
                "current_step": int(source_scheduler["current_step"]),
            },
            "target": {
                "updates_per_epoch": target_updates_per_epoch,
                "total_steps": int(target_state["total_steps"]),
                "warmup_steps": int(target_state["warmup_steps"]),
                "current_step": target_current_step,
            },
        }

    if "curriculum_step" in checkpoint:
        curriculum_step = int(checkpoint["curriculum_step"])
        curriculum_mapping = "preserved_from_step_checkpoint"
    elif source_is_legacy:
        decoder_epochs = int(source_configuration.get("decoder_only_epochs", 0))
        top_epoch = int(source_configuration.get("stage1_top_unfreeze_epoch", 0))
        if completed_epochs <= decoder_epochs:
            curriculum_step = int(args.decoder_only_steps)
            curriculum_mapping = "legacy_decoder_phase_completed"
        elif completed_epochs <= top_epoch:
            curriculum_step = int(args.stage1_partial_unfreeze_step)
            curriculum_mapping = "legacy_adapter_phase_completed"
        else:
            curriculum_step = int(args.stage1_final_unfreeze_step)
            curriculum_mapping = "legacy_backbone_phase_completed"
    else:
        raise ValueError("Step-based migration checkpoint lacks curriculum_step")
    if curriculum_step < 0:
        raise ValueError("Migration curriculum_step must be non-negative")
    checkpoint_stat = checkpoint_path.stat()
    provenance = {
        "kind": (
            "serial_to_ddp_epoch_checkpoint"
            if epoch_complete
            else "serial_to_ddp_step_checkpoint"
        ),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_size": checkpoint_stat.st_size,
        "source_checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "source_epoch": epoch,
        "source_execution": source_execution_record,
        "target_execution": target_execution,
        "source_epoch_complete": epoch_complete,
        "source_next_batch_index": (
            None if epoch_complete else int(checkpoint["next_batch_index"])
        ),
        "configuration_changes": configuration_changes,
        "effective_batch_size": {
            "source": source_effective_batch,
            "target": target_effective_batch,
        },
        "scheduler_remap": scheduler_provenance,
        "curriculum_remap": {
            "policy": curriculum_mapping,
            "source": checkpoint.get("curriculum_step"),
            "target": curriculum_step,
        },
        "learning_rate_discontinuity_expected": bool(
            int(source_scheduler["warmup_steps"])
            != int(target_state["warmup_steps"])
        ),
        "history_source": (
            "checkpoint_embedded"
            if isinstance(checkpoint.get("history"), list)
            else str(checkpoint_path.parent / "history.json")
        ),
    }
    return history, remapped_scheduler, provenance, curriculum_step, resume_progress


def main(args=None, execution=None) -> None:
    args = parse_args() if args is None else args
    is_primary = execution is None or execution.is_primary
    init_checkpoint = getattr(args, "init_checkpoint", None)
    migrate_resume = getattr(args, "migrate_resume", None)
    selected_checkpoints = sum(
        value is not None for value in (args.resume, init_checkpoint, migrate_resume)
    )
    if selected_checkpoints > 1:
        raise ValueError(
            "--resume, --init-checkpoint and --migrate-resume are mutually exclusive"
        )
    if migrate_resume is not None:
        migrate_resume = Path(migrate_resume).expanduser().resolve()
        args.migrate_resume = migrate_resume
        if not migrate_resume.is_file():
            raise FileNotFoundError(migrate_resume)
        if (
            execution is None
            or not execution.distributed
            or getattr(args, "execution_mode", None) != "ddp"
        ):
            raise ValueError("--migrate-resume is supported only by multi-rank DDP")
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
    legacy_phase_flags = {
        "decoder-only-epochs": args.decoder_only_epochs,
        "stage1-top-unfreeze-epoch": args.stage1_top_unfreeze_epoch,
        "stage1-unfreeze-blocks": args.stage1_unfreeze_blocks,
    }
    supplied_legacy_flags = [
        name for name, value in legacy_phase_flags.items() if value is not None
    ]
    if supplied_legacy_flags:
        raise ValueError(
            "Epoch-based unfreezing is retired; replace "
            + ", ".join(f"--{name}" for name in supplied_legacy_flags)
            + " with the step-based curriculum options"
        )
    if args.warmup_steps < 0:
        raise ValueError("warmup-steps must be non-negative")
    if not (
        0 <= args.decoder_only_steps
        < args.stage1_partial_unfreeze_step
        < args.stage1_final_unfreeze_step
    ):
        raise ValueError(
            "Require 0 <= decoder-only-steps < stage1-partial-unfreeze-step "
            "< stage1-final-unfreeze-step"
        )
    if (
        args.stage1_final_unfreeze_blocks < 1
        or (getattr(args, "unfreeze_schedule", "legacy") == "legacy" and (
            args.stage1_partial_unfreeze_blocks < 1
            or args.stage1_final_unfreeze_blocks < args.stage1_partial_unfreeze_blocks
        ))
    ):
        raise ValueError(
            "Require 1 <= stage1-partial-unfreeze-blocks <= "
            "stage1-final-unfreeze-blocks"
        )
    if args.checkpoint_interval_steps < 0:
        raise ValueError("checkpoint-interval-steps must be non-negative")
    if args.decoder_version in ('v4', 'v5'):
        for name in ('diffusion_loss_weight', 'diffusion_reconstruction_weight',
                     'diffusion_edge_weight', 'diffusion_boundary_boost', 'diffusion_boundary_radius'):
            if not math.isfinite(float(getattr(args, name))) or getattr(args, name) < 0:
                raise ValueError(f'{name} must be finite and non-negative')
    if getattr(args, 'retain_progress_checkpoints', False):
        if args.checkpoint_interval_steps <= 0 or args.monitor_interval_steps <= 0:
            raise ValueError('Retention requires positive save and monitor intervals')
        if args.monitor_interval_steps % args.checkpoint_interval_steps:
            raise ValueError('Monitor interval must be divisible by save interval')
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
    if args.sampling_locality_tile_size < 1:
        raise ValueError("sampling-locality-tile-size must be positive")
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
    progress_checkpoint = args.output_dir / "checkpoint_progress.pt"
    if migrate_resume is not None and (
        last_checkpoint.exists() or progress_checkpoint.exists()
    ):
        raise FileExistsError("--migrate-resume requires a fresh output directory")
    if args.resume is None and (
        last_checkpoint.exists() or progress_checkpoint.exists()
    ):
        existing = (
            progress_checkpoint if progress_checkpoint.exists() else last_checkpoint
        )
        raise FileExistsError(f"Use --resume for existing run: {existing}")

    device = (
        execution.device if execution is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_loader = _loader(args.train_manifest, args, training=True, execution=execution)
    val_loader = (None if args.async_full_validation else
                  _loader(args.val_manifest, args, training=False, execution=execution))
    system = JointSegmentationSystem(
        decoder_version=args.decoder_version,
        stage1_config=args.stage1_config,
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_config=args.stage2_config,
        stage2_checkpoint=args.stage2_checkpoint,
        num_classes=args.num_classes,
        decoder_drop_path_rate=args.decoder_drop_path_rate,
    ).to(device)
    if init_checkpoint is not None:
        initial = _load(init_checkpoint)
        if initial.get("model_version") != system.model_version:
            raise ValueError("Initialization checkpoint model version differs")
        system.load_state_dict(initial["model"], strict=True)
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
    if train_batches == 0:
        raise ValueError("No training batches; reduce per-GPU batch size or world size")
    updates_per_epoch = math.ceil(train_batches / args.gradient_accumulation)
    total_steps = max(1, updates_per_epoch * args.epochs)
    if total_steps <= args.stage1_final_unfreeze_step:
        raise ValueError(
            "The run ends before the final top-DINO phase can receive an update: "
            f"total_steps={total_steps}, final_unfreeze_step="
            f"{args.stage1_final_unfreeze_step}"
        )
    warmup_steps = min(total_steps - 1, int(args.warmup_steps))
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
    val_graphs = None if args.async_full_validation else JointGraphRepository(
        args.graph_dir,
        expected_edge_mode=runtime.context_edge_mode,
    )
    from dinov2_segmentation.consistent_features import configure_repository
    configure_repository(train_graphs, system, _configuration(args))
    if val_graphs is not None:
        configure_repository(val_graphs, system, _configuration(args))
    run_manifest = {
        "format_version": 2,
        "model_version": system.model_version,
        "configuration": _configuration(args),
        "stage1_config": str(args.stage1_config),
        "stage1_checkpoint": str(args.stage1_checkpoint),
        "stage2_config": str(args.stage2_config),
        "stage2_checkpoint": str(args.stage2_checkpoint),
        "graph_dir": str(args.graph_dir),
        "stage2_runtime": runtime.__dict__,
        "sampling": getattr(
            getattr(train_loader.batch_sampler, "sampler", train_loader.sampler),
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
            "requested_warmup_steps": int(args.warmup_steps),
            "effective_warmup_steps": warmup_steps,
            "minimum_lr_ratio": args.min_lr_ratio,
        },
        "gradual_unfreezing": {
            "clock": "optimizer_updates",
            "decoder_only_steps": args.decoder_only_steps,
            "stage1_partial_unfreeze_step": args.stage1_partial_unfreeze_step,
            "stage1_partial_unfreeze_blocks": args.stage1_partial_unfreeze_blocks,
            "stage1_final_unfreeze_step": args.stage1_final_unfreeze_step,
            "stage1_final_unfreeze_blocks": args.stage1_final_unfreeze_blocks,
            "final_phase_pretrained_lr_scale": args.final_phase_pretrained_lr_scale,
            "final_phase_decoder_lr_scale": args.final_phase_decoder_lr_scale,
        },
        "checkpointing": {
            "epoch_checkpoint": "checkpoint_last.pt",
            "step_checkpoint": "checkpoint_progress.pt",
            "interval_optimizer_steps": args.checkpoint_interval_steps,
            "resume_granularity": "batch_cursor_at_optimizer_boundary",
        },
        "early_stopping": {
            "patience": args.early_stopping_patience,
            "minimum_delta": args.early_stopping_min_delta,
            "start_epoch": args.early_stopping_start_epoch,
        },
        "metrics": {
            "checkpoint_and_early_stopping": "val.tumor_dice",
            "confusion_accumulation": "device_epoch",
            "confusion_metrics": [
                "tumor_dice",
                "tumor_precision",
                "tumor_recall",
                "tumor_f2",
                "predicted_tumor_fraction",
            ],
            "training_probability_metrics": False,
            "validation_probability_metric_interval_epochs": (
                _VALIDATION_PROBABILITY_METRIC_INTERVAL
            ),
            "periodic_probability_metrics": [
                "approx_pr_auc",
                "best_f2_threshold",
                "best_threshold_f2",
            ],
        },
        "paper_references": PAPER_REFERENCES,
    }
    if getattr(args, "unfreeze_schedule", "legacy") == "separate_gnn_fusion":
        run_manifest["gradual_unfreezing"] = {
            "schedule": "separate_gnn_fusion", "clock": "optimizer_updates",
            "gnn_unfreeze_step": args.decoder_only_steps,
            "fusion_unfreeze_step": args.stage1_partial_unfreeze_step,
            "dino_unfreeze_step": args.stage1_final_unfreeze_step,
            "dino_unfreeze_blocks": args.stage1_final_unfreeze_blocks,
            "dino_final_norm_trainable": True,
            "phases": ["decoder_only", "gnn_and_decoder", "fusion_gnn_decoder", "dino_joint"],
            "final_phase_pretrained_lr_scale": args.final_phase_pretrained_lr_scale,
            "final_phase_decoder_lr_scale": args.final_phase_decoder_lr_scale,
            "persistent_graph_cache_updates": "current global batch only, starting at fusion unfreeze",
        }
    if getattr(args, "graph_feature_policy", "legacy") == "staged_consistent":
        run_manifest["gradual_unfreezing"]["persistent_graph_cache_updates"] = "disabled; no target-only writes"
        run_manifest["graph_features"] = {
            "policy": "staged_consistent", "frozen": "immutable Stage1B inputs for all graph nodes",
            "fusion": "disk-cached raw DINO outputs; current fusion for entire requested receptive field",
            "dino": "current DINO and fusion for entire requested receptive field; raw cache bypassed",
            "graph_views": "canonical RGB, bilinear resize, ImageNet normalization; no random augmentation",
            "segmentation_views": "existing paired image/mask augmentations retained",
            "raw_cache_population": "on demand on disk; no fused-feature persistence",
            "static_edge_attributes": "original baseline spatial and semantic weights retained",
        }
        if run_manifest["configuration"].get("cache_frozen_dino_prefix", False):
            run_manifest["graph_features"]["dino"] = (
                "weight-hashed frozen-prefix tokens on disk; current trainable suffix, "
                "final norm and fusion recomputed for entire requested receptive field")
    if args.async_full_validation:
        run_manifest['validation_execution'] = {
            'mode': 'independent_process', 'trainer_waits_for_validation': False,
            'epoch_queue': 'full_validation/candidate_epoch_*.json',
            'best_checkpoint': 'full_validation/checkpoint_best_full.pt',
            'early_stopping': 'completed_contiguous_epoch_results_checked_at_epoch_end',
        }
    if args.decoder_version in ('v4', 'v5'):
        run_manifest['auxiliary_task'] = {
            'name': 'boundary_weighted_conditional_diffusion_reconstruction',
            'recipe_version': 1,
            'base_decoder': 'v1' if args.decoder_version == 'v4' else 'v2',
            'diffusion_steps': system.decoder.diffusion.steps,
            'schedule': 'cosine_alpha_bar_s0.008_max_beta0.999',
            'prediction': 'epsilon',
            'training': 'one_random_timestep_per_patch',
            'target': 'same_augmented_RGB_patch_in_minus1_plus1',
            'condition': 'pooled_base_semantics_and_predicted_class_probabilities',
            'label_use': 'loss_weighting_only',
            'reconstruction_stability': 'multiply_x0_errors_by_sqrt_alpha_bar',
            'validation': 'segmentation_only_no_diffusion_sampling',
            'boundary_f1_tolerance_resized_pixels': 2,
            'loss_weights': {name: getattr(args, name) for name in (
                'diffusion_loss_weight', 'diffusion_reconstruction_weight',
                'diffusion_edge_weight', 'diffusion_boundary_boost', 'diffusion_boundary_radius')},
        }
    execution_metadata = None
    if execution is not None:
        from dinov2_segmentation.train_joint_parallel import execution_manifest

        execution_metadata = execution_manifest(args, execution, train_loader)
        run_manifest["execution"] = execution_metadata
        run_manifest["initialization_checkpoint"] = (
            str(init_checkpoint) if init_checkpoint else None
        )
        source_sampler = getattr(train_loader.batch_sampler, "sampler", train_loader.sampler)
        run_manifest["sampling"] = getattr(
            source_sampler, "summary", run_manifest["sampling"]
        )

    start_epoch = 0
    resume_batch_offset = 0
    resume_train_progress = None
    best_dice = -1.0
    history_path = args.output_dir / "history.json"
    history: list[dict] = []
    gradient_audit: dict = {}
    curriculum_state = {
        "step": 0,
        "phase_name": None,
        "phase_transitions": [],
    }
    early_stopping_best = -1.0
    epochs_without_improvement = 0
    checkpoint = None
    if migrate_resume is not None:
        source_checkpoint = _load(migrate_resume)
        (
            history,
            remapped_scheduler,
            migration_provenance,
            migrated_curriculum_step,
            resume_train_progress,
        ) = (
            _prepare_serial_to_ddp_migration(
                migrate_resume,
                source_checkpoint,
                args=args,
                system=system,
                optimizer=optimizer,
                scheduler=scheduler,
                target_group_metadata=group_metadata,
                target_execution=execution_metadata,
                target_sampling=run_manifest["sampling"],
                target_updates_per_epoch=updates_per_epoch,
            )
        )
        system.load_state_dict(source_checkpoint["model"], strict=True)
        optimizer.load_state_dict(source_checkpoint["optimizer"])
        scheduler.load_state_dict(remapped_scheduler)
        scaler.load_state_dict(source_checkpoint["scaler"])
        source_epoch_complete = bool(
            source_checkpoint.get("epoch_complete", True)
        )
        start_epoch = int(source_checkpoint["epoch"]) + int(source_epoch_complete)
        if not source_epoch_complete:
            resume_batch_offset = int(source_checkpoint["next_batch_index"])
            resume_batch_offset = min(
                resume_batch_offset,
                int(execution_metadata["train_batches_per_rank"]),
            )
        curriculum_state.update(
            {
                "step": migrated_curriculum_step,
                "phase_name": source_checkpoint.get("training_phase"),
                "phase_transitions": list(
                    source_checkpoint.get("phase_transitions", [])
                    if not source_epoch_complete
                    else []
                ),
            }
        )
        best_dice = float(source_checkpoint["best_dice"])
        gradient_audit.update(source_checkpoint["gradient_audit"])
        early_stopping_best = float(source_checkpoint["early_stopping_best"])
        epochs_without_improvement = int(
            source_checkpoint["epochs_without_improvement"]
        )
        run_manifest["migration"] = migration_provenance
        checkpoint = dict(source_checkpoint)
        checkpoint.update(
            {
                "model": system.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "configuration": _configuration(args),
                "run_manifest": run_manifest,
                "execution": execution_metadata,
                "history": list(history),
                "format_version": 2,
                "epoch_complete": source_epoch_complete,
                "curriculum_step": int(curriculum_state["step"]),
                "next_batch_index": resume_batch_offset,
                "train_progress": resume_train_progress,
            }
        )
    elif args.resume is not None:
        checkpoint = _load(args.resume)
        if int(checkpoint.get("format_version", 1)) != 2:
            raise ValueError(
                "Ordinary --resume requires a step-curriculum checkpoint. Use "
                "--migrate-resume to convert an old epoch-scheduled serial checkpoint."
            )
        if execution is not None and checkpoint.get("execution") != execution_metadata:
            raise ValueError(
                "Resume execution or dataset differs. Use --init-checkpoint in a fresh "
                "output directory to change GPU count or data; this restarts optimizer/scheduler."
            )
        if execution is not None:
            checkpoint_manifest = checkpoint.get("run_manifest", {})
            run_manifest["initialization_checkpoint"] = checkpoint_manifest.get(
                "initialization_checkpoint"
            )
            if "migration" in checkpoint_manifest:
                run_manifest["migration"] = checkpoint_manifest["migration"]
        if checkpoint.get("model_version") != system.model_version:
            raise ValueError("Resume checkpoint model version differs")
        checkpoint_configuration = dict(checkpoint.get("configuration", {}))
        # Checkpoints created before the opt-in class weighting flag have the
        # same semantics as the new default and remain safely resumable.
        backward_compatible_defaults = {
            "experiment_profile": "current",
            "tumor_class_weight": 1.0,
            "cross_entropy_weight": 1.0,
            "decoder_drop_path_rate": 0.1,
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
            "sampling_locality_tile_size": 4096,
            "color_augmentation": "none",
            "probability_metric_bins": 0,
        }
        for name, value in backward_compatible_defaults.items():
            checkpoint_configuration.setdefault(name, value)
        from dinov2_segmentation.checkpoint_retention import validate_resume_configuration
        operational_changes = validate_resume_configuration(checkpoint_configuration, _configuration(args))
        if operational_changes and is_primary:
            print(json.dumps({'resume_operational_changes': operational_changes}), flush=True)
            run_manifest["resume_operational_changes"] = operational_changes
        system.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        epoch_complete = bool(checkpoint.get("epoch_complete", True))
        start_epoch = int(checkpoint["epoch"]) + int(epoch_complete)
        if "curriculum_step" not in checkpoint:
            raise ValueError("Resume checkpoint lacks curriculum_step")
        curriculum_state.update(
            {
                "step": int(checkpoint["curriculum_step"]),
                "phase_name": checkpoint.get("training_phase"),
                "phase_transitions": list(
                    checkpoint.get("phase_transitions", [])
                    if not epoch_complete
                    else []
                ),
            }
        )
        if not epoch_complete:
            resume_batch_offset = int(checkpoint.get("next_batch_index", -1))
            resume_train_progress = checkpoint.get("train_progress")
            if resume_batch_offset < 1 or not isinstance(
                resume_train_progress, dict
            ):
                raise ValueError("In-epoch resume checkpoint has no valid progress cursor")
        best_dice = float(checkpoint.get("best_dice", best_dice))
        gradient_audit.update(checkpoint.get("gradient_audit", {}))
        early_stopping_best = float(
            checkpoint.get("early_stopping_best", best_dice)
        )
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        history = _migration_history(args.resume, start_epoch, checkpoint)

    def checkpoint_state(
        *,
        epoch: int,
        epoch_complete: bool,
        train_progress: dict | None = None,
    ) -> dict:
        state = {
            "format_version": 2,
            "model_version": system.model_version,
            "epoch": int(epoch),
            "epoch_complete": bool(epoch_complete),
            "next_batch_index": (
                0
                if epoch_complete
                else int(train_progress["next_batch_index"])
            ),
            "train_progress": None if epoch_complete else train_progress,
            "curriculum_step": int(curriculum_state["step"]),
            "training_phase": curriculum_state.get("phase_name"),
            "phase_transitions": list(
                curriculum_state.get("phase_transitions", [])
            ),
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
            "history": list(history),
        }
        if execution_metadata is not None:
            state["execution"] = execution_metadata
        return state

    if is_primary:
        _atomic_json_save(run_manifest, args.output_dir / "run_manifest.json")
        _atomic_json_save(history, history_path)
        if getattr(args, 'retain_progress_checkpoints', False):
            _atomic_json_save({
                'mode': 'synchronous_writer',
                'save_interval_steps': args.checkpoint_interval_steps,
                'monitor_interval_steps': args.monitor_interval_steps,
                'retain_all': True,
            }, args.output_dir / 'checkpoint_queue_writer.json')
        if migrate_resume is not None:
            # Persist the converted state before the first DDP forward. A
            # launch failure can then use ordinary strict --resume.
            converted = checkpoint_state(
                epoch=int(source_checkpoint["epoch"]),
                epoch_complete=bool(source_checkpoint.get("epoch_complete", True)),
                train_progress=resume_train_progress,
            )
            _atomic_torch_save(
                converted,
                last_checkpoint
                if converted["epoch_complete"]
                else progress_checkpoint,
            )
            (args.output_dir / "gradient_audit.json").write_text(
                json.dumps(gradient_audit, indent=2) + "\n", encoding="utf-8"
            )
    if execution is not None:
        execution.barrier()

    for epoch in range(start_epoch, args.epochs):
        epoch_sampler = train_loader.batch_sampler
        set_sampler_epoch = getattr(epoch_sampler, "set_epoch", None)
        if callable(set_sampler_epoch):
            set_sampler_epoch(epoch)
        epoch_batch_offset = resume_batch_offset if epoch == start_epoch else 0
        set_start_batch = getattr(epoch_sampler, "set_start_batch", None)
        if epoch_batch_offset:
            if not callable(set_start_batch):
                raise RuntimeError("Training sampler does not support in-epoch resume")
            set_start_batch(epoch_batch_offset)
        if not (epoch == start_epoch and resume_train_progress is not None):
            curriculum_state["phase_transitions"] = []
        epoch_curriculum_start = int(curriculum_state["step"])
        phase_name, unfreeze_blocks = _phase_spec(
            epoch_curriculum_start, args
        )
        if is_primary:
            print(
                json.dumps(
                    {
                        "training_phase": {
                            "name": phase_name,
                            "curriculum_step": epoch_curriculum_start,
                            "epoch": epoch,
                            "stage1_unfreeze_blocks": unfreeze_blocks,
                        },
                        "resumed_batch_index": epoch_batch_offset,
                    }
                ),
                flush=True,
            )

        def save_progress(train_progress: dict) -> None:
            if execution is not None:
                execution.barrier()
            if is_primary:
                state = checkpoint_state(
                    epoch=epoch,
                    epoch_complete=False,
                    train_progress=train_progress,
                )
                _atomic_torch_save(state, progress_checkpoint)
                if getattr(args, 'retain_progress_checkpoints', False):
                    from dinov2_segmentation.checkpoint_retention import retain_progress
                    retain_progress(progress_checkpoint, state, args.monitor_interval_steps)
                print(
                    json.dumps(
                        {
                            "step_checkpoint": str(progress_checkpoint),
                            "epoch": epoch,
                            "next_batch_index": train_progress["next_batch_index"],
                            "curriculum_step": state["curriculum_step"],
                        }
                    ),
                    flush=True,
                )
            if execution is not None:
                execution.barrier()

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
            curriculum_state=curriculum_state,
            epoch=epoch,
            batch_offset=epoch_batch_offset,
            initial_train_progress=(
                resume_train_progress if epoch == start_epoch else None
            ),
            progress_callback=save_progress,
            execution=execution,
        )
        resume_batch_offset = 0
        resume_train_progress = None
        if execution is not None:
            execution.sync_buffers(system)
        if args.async_full_validation:
            record = {
                'epoch': epoch, 'train': train_metrics, 'val': None,
                'validation_status': 'queued_independent_process',
                'lr_min': min(g['lr'] for g in optimizer.param_groups),
                'lr_max': max(g['lr'] for g in optimizer.param_groups),
                'training_phase': curriculum_state['phase_name'],
                'curriculum_step_start': epoch_curriculum_start,
                'curriculum_step_end': int(curriculum_state['step']),
                'phase_transitions': list(curriculum_state['phase_transitions']),
            }
            history.append(record)
            from dinov2_segmentation.checkpoint_retention import (
                queue_epoch_validation, completed_epoch_validation_state,
            )
            # Rank 0 reads external files once. Broadcast its decision so a
            # result published between rank reads cannot diverge DDP control.
            decision = None
            if is_primary:
                decision = completed_epoch_validation_state(
                    args.output_dir, history, start_epoch=args.early_stopping_start_epoch,
                    min_delta=args.early_stopping_min_delta, patience=args.early_stopping_patience,
                )
            if execution is not None and execution.distributed:
                import torch.distributed as dist
                envelope = [decision]
                dist.broadcast_object_list(envelope, src=0)
                decision = envelope[0]
            best_dice = max(best_dice, decision['best_dice'])
            early_stopping_best = max(early_stopping_best, decision['reference_dice'])
            epochs_without_improvement = decision['epochs_without_improvement']
            state = checkpoint_state(epoch=epoch, epoch_complete=True)
            if is_primary:
                _atomic_torch_save(state, last_checkpoint)
                request = queue_epoch_validation(last_checkpoint, state)
                _atomic_json_save(history, history_path)
                _atomic_json_save(gradient_audit, args.output_dir / 'gradient_audit.json')
                _atomic_json_save(decision, args.output_dir / 'async_early_stopping_status.json')
                print(json.dumps({'epoch_training_complete': record,
                                  'queued_full_validation': str(request)}), flush=True)
                if progress_checkpoint.exists():
                    progress_checkpoint.unlink()
                if decision['stopped_early']:
                    _atomic_json_save(dict(decision, stop_epoch=epoch,
                                           reason='completed_async_epoch_results'),
                                      args.output_dir / 'early_stopping.json')
            if execution is not None:
                execution.barrier()
            if decision['stopped_early']:
                break
            continue
        with torch.no_grad():
            val_metrics = _run_epoch(
                system,
                val_graphs,
                val_loader,
                device,
                args,
                execution=execution,
                collect_probability_metrics=_should_collect_probability_metrics(
                    epoch, args.epochs
                ),
            )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr_min": min(group["lr"] for group in optimizer.param_groups),
            "lr_max": max(group["lr"] for group in optimizer.param_groups),
            "training_phase": curriculum_state["phase_name"],
            "curriculum_step_start": epoch_curriculum_start,
            "curriculum_step_end": int(curriculum_state["step"]),
            "phase_transitions": list(curriculum_state["phase_transitions"]),
        }
        history.append(record)
        if is_primary:
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
        state = checkpoint_state(epoch=epoch, epoch_complete=True)
        if is_primary:
            _atomic_torch_save(state, last_checkpoint)
            if improved:
                _atomic_torch_save(state, args.output_dir / "checkpoint_best.pt")
            _atomic_json_save(history, history_path)
            (args.output_dir / "gradient_audit.json").write_text(
                json.dumps(gradient_audit, indent=2) + "\n", encoding="utf-8"
            )
            if progress_checkpoint.exists():
                progress_checkpoint.unlink()
        if execution is not None:
            execution.barrier()
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
            if is_primary:
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
    if execution is not None and is_primary and start_epoch >= args.epochs:
        # A completed checkpoint can be restored into a fresh directory. Keep
        # its history and last state alongside the completion marker there.
        checkpoint["run_manifest"] = run_manifest
        _atomic_torch_save(checkpoint, last_checkpoint)
        _atomic_json_save(history, history_path)
        (args.output_dir / "gradient_audit.json").write_text(
            json.dumps(gradient_audit, indent=2) + "\n", encoding="utf-8"
        )
    early_stopping_path = args.output_dir / "early_stopping.json"
    if is_primary and not early_stopping_path.exists():
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
    if is_primary:
        complete_path.touch()
    if execution is not None:
        execution.barrier()


if __name__ == "__main__":
    main()
