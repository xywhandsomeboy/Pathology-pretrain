"""Train the global-local pathology segmentation decoder."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dinov2_segmentation.data import PatchSegmentationDataset
from dinov2_segmentation.losses import segmentation_loss
from dinov2_segmentation.models import GlobalLocalSegmentationModel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--token-dim", type=int, default=1024)
    parser.add_argument("--context-dim", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--detail-depth", type=int, default=4)
    parser.add_argument("--fusion-depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def model_config(args) -> dict:
    return {
        "num_classes": args.num_classes,
        "token_dim": args.token_dim,
        "context_dim": args.context_dim,
        "channels": args.channels,
        "detail_depth": args.detail_depth,
        "fusion_depth": args.fusion_depth,
        "num_heads": args.num_heads,
        "window_size": args.window_size,
        "drop_path_rate": args.drop_path_rate,
    }


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_grad_scaler(enabled: bool):
    # torch.amp.GradScaler(device, ...) is new; retain compatibility with the
    # PyTorch versions commonly used by the original DINOv2 environment.
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _make_loader(path, args, training):
    dataset = PatchSegmentationDataset(
        path,
        image_size=args.image_size,
        training=training,
        require_mask=True,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=args.workers > 0,
    )


def _run_epoch(model, loader, device, args, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "cross_entropy": 0.0, "dice_loss": 0.0}
    samples = 0
    amp_enabled = device.type == "cuda" and not args.no_amp
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        dense = batch["dense_tokens"].to(device, non_blocking=True)
        context = batch["global_context"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=amp_enabled,
        ):
            logits = model(image, dense, context)
            loss, parts = segmentation_loss(
                logits,
                target,
                ignore_index=args.ignore_index,
                dice_weight=args.dice_weight,
            )
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        count = image.shape[0]
        samples += count
        totals["loss"] += float(loss.detach()) * count
        for name in parts:
            totals[name] += float(parts[name]) * count
    return {name: value / max(samples, 1) for name, value in totals.items()}


def main():
    args = parse_args()
    if args.num_classes < 2:
        raise ValueError("num_classes must include background and be at least 2")
    _set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = _make_loader(args.train_manifest, args, training=True)
    val_loader = _make_loader(args.val_manifest, args, training=False) if args.val_manifest else None
    config = model_config(args)
    model = GlobalLocalSegmentationModel(**config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    scaler = _make_grad_scaler(device.type == "cuda" and not args.no_amp)
    start_epoch, best_loss = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if checkpoint["model_config"] != config:
            raise ValueError("Resume checkpoint model_config differs from command-line config")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_loss", best_loss))

    history = []
    for epoch in range(start_epoch, args.epochs):
        train_metrics = _run_epoch(model, train_loader, device, args, optimizer, scaler)
        with torch.no_grad():
            val_metrics = (
                _run_epoch(model, val_loader, device, args) if val_loader else train_metrics
            )
        scheduler.step()
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_loss": min(best_loss, val_metrics["loss"]),
            "model_config": config,
            "image_size": args.image_size,
        }
        torch.save(state, args.output_dir / "checkpoint_last.pt")
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            state["best_loss"] = best_loss
            torch.save(state, args.output_dir / "checkpoint_best.pt")
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
