"""Jointly fine-tune Stage1, Stage2 GATv2, and Decoder V1 or V2."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dinov2_segmentation.data import JointPatchSegmentationDataset
from dinov2_segmentation.joint_graph import JointGraphRepository
from dinov2_segmentation.joint_model import JointSegmentationSystem
from dinov2_segmentation.joint_optim import (
    WarmupCosineScheduler,
    build_joint_adamw,
)
from dinov2_segmentation.losses import segmentation_loss


PAPER_REFERENCES = {
    "adamw": "https://arxiv.org/abs/1711.05101",
    "cosine_schedule": "https://arxiv.org/abs/1608.03983",
    "layerwise_lr_decay": "https://arxiv.org/abs/2106.08254",
    "dinov2": "https://arxiv.org/abs/2304.07193",
    "gatv2": "https://arxiv.org/abs/2105.14491",
    "dense_prediction_adapter": "https://arxiv.org/abs/2205.08534",
}


def _load(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-version", choices=("v1", "v2"), required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--stage1-config", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-config", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--gradient-audit-updates", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
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
    )
    generator = torch.Generator().manual_seed(args.seed + int(training))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training,
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


def _metrics(totals: dict, samples: int, confusion: torch.Tensor) -> dict:
    result = {name: value / max(samples, 1) for name, value in totals.items()}
    true_positive = float(confusion[1, 1])
    false_positive = float(confusion[0, 1])
    false_negative = float(confusion[1, 0])
    denominator_dice = 2 * true_positive + false_positive + false_negative
    denominator_iou = true_positive + false_positive + false_negative
    result["tumor_dice"] = (
        2 * true_positive / denominator_dice if denominator_dice else 1.0
    )
    result["tumor_iou"] = true_positive / denominator_iou if denominator_iou else 1.0
    result["pixel_accuracy"] = float(confusion.diag().sum()) / max(
        float(confusion.sum()), 1.0
    )
    result["confusion"] = confusion.tolist()
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
) -> dict:
    training = optimizer is not None
    system.train(training)
    totals = {"loss": 0.0, "cross_entropy": 0.0, "dice_loss": 0.0}
    confusion = torch.zeros((args.num_classes, args.num_classes), dtype=torch.int64)
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
                dice_weight=args.dice_weight,
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
                if gradient_audit is not None and not gradient_audit.get("complete", False):
                    observed = {
                        "stage1_grad_norm": _gradient_norm(system.stage1),
                        "stage2_grad_norm": _gradient_norm(system.stage2),
                        "decoder_grad_norm": _gradient_norm(system.decoder),
                    }
                    gradient_audit["updates_observed"] = int(
                        gradient_audit.get("updates_observed", 0)
                    ) + 1
                    for name, value in observed.items():
                        gradient_audit[name] = max(
                            float(gradient_audit.get(name, 0.0)), float(value)
                        )
                    stage_keys = tuple(observed)
                    gradient_audit["complete"] = all(
                        math.isfinite(float(gradient_audit[name]))
                        and float(gradient_audit[name]) > 0
                        for name in stage_keys
                    )
                    if (
                        not gradient_audit["complete"]
                        and gradient_audit["updates_observed"]
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
    return _metrics(totals, samples, confusion)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _configuration(args: argparse.Namespace) -> dict:
    keys = (
        "decoder_version",
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
        "dice_weight",
        "ignore_index",
        "amp_dtype",
        "seed",
        "gradient_audit_updates",
    )
    return {key: getattr(args, key) for key in keys}


def main() -> None:
    args = parse_args()
    if args.num_classes != 2:
        raise ValueError("The cervical workflow requires binary background/tumor output")
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
        "optimizer_groups": group_metadata,
        "scheduler": {
            "name": "linear_warmup_single_cosine_decay",
            "updates_per_epoch": updates_per_epoch,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "minimum_lr_ratio": args.min_lr_ratio,
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
    if args.resume is not None:
        checkpoint = _load(args.resume)
        if checkpoint.get("model_version") != system.model_version:
            raise ValueError("Resume checkpoint model version differs")
        if checkpoint.get("configuration") != _configuration(args):
            raise ValueError("Resume checkpoint configuration differs")
        system.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint.get("best_dice", best_dice))
        gradient_audit.update(checkpoint.get("gradient_audit", {}))
        if history_path.is_file():
            history = [
                record
                for record in json.loads(history_path.read_text(encoding="utf-8"))
                if int(record["epoch"]) < start_epoch
            ]

    for epoch in range(start_epoch, args.epochs):
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
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        improved = val_metrics["tumor_dice"] > best_dice
        best_dice = max(best_dice, val_metrics["tumor_dice"])
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
    if not gradient_audit.get("complete", False):
        raise RuntimeError(
            "Joint training ended before Stage1, Stage2 and decoder all received "
            f"a non-zero supervised gradient: {gradient_audit}"
        )
    complete_path.touch()


if __name__ == "__main__":
    main()
