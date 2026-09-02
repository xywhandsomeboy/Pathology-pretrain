"""Train Decoder V2 without changing the V1 training entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dinov2_segmentation.models.model_v2 import GlobalLocalSegmentationModelV2
from dinov2_segmentation.train import (
    _make_grad_scaler,
    _make_loader,
    _run_epoch,
    _set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--token-dim", type=int, default=1024)
    parser.add_argument("--context-dim", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--high-resolution-depth", type=int, default=3)
    parser.add_argument("--correction-depth", type=int, default=4)
    parser.add_argument("--channel-expansion", type=int, default=2)
    parser.add_argument("--attention-reduction", type=int, default=16)
    parser.add_argument("--residual-scale-init", type=float, default=1e-2)
    parser.add_argument("--max-upsample-stages", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
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
        "high_resolution_depth": args.high_resolution_depth,
        "correction_depth": args.correction_depth,
        "channel_expansion": args.channel_expansion,
        "attention_reduction": args.attention_reduction,
        "residual_scale_init": args.residual_scale_init,
        "max_upsample_stages": args.max_upsample_stages,
    }


def _load_history(path: Path, start_epoch: int) -> list[dict]:
    if not path.is_file():
        return []
    history = json.loads(path.read_text(encoding="utf-8"))
    return [record for record in history if int(record["epoch"]) < start_epoch]


def main():
    args = parse_args()
    if args.num_classes < 2:
        raise ValueError("num_classes must include background and be at least 2")
    if args.correction_depth < 1:
        raise ValueError("correction_depth must be at least 1")
    _set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = args.output_dir / "checkpoint_last.pt"
    if args.resume is None and last_checkpoint.exists():
        raise FileExistsError(
            f"V2 output already contains {last_checkpoint}; use --resume or a new directory"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = _make_loader(args.train_manifest, args, training=True)
    val_loader = (
        _make_loader(args.val_manifest, args, training=False)
        if args.val_manifest
        else None
    )
    config = model_config(args)
    model = GlobalLocalSegmentationModelV2(**config).to(device)
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
        checkpoint_version = checkpoint.get("model_version")
        if checkpoint_version != model.model_version:
            raise ValueError(
                f"Cannot resume V2 from model_version={checkpoint_version!r}"
            )
        if checkpoint["model_config"] != config:
            raise ValueError(
                "Resume checkpoint model_config differs from V2 command-line config"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_loss", best_loss))

    history_path = args.output_dir / "history.json"
    history = _load_history(history_path, start_epoch)
    for epoch in range(start_epoch, args.epochs):
        train_metrics = _run_epoch(
            model, train_loader, device, args, optimizer, scaler
        )
        with torch.no_grad():
            val_metrics = (
                _run_epoch(model, val_loader, device, args)
                if val_loader
                else train_metrics
            )
        scheduler.step()
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        current_best = min(best_loss, val_metrics["loss"])
        state = {
            "format_version": 1,
            "model_version": model.model_version,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_loss": current_best,
            "model_config": config,
            "image_size": args.image_size,
        }
        torch.save(state, last_checkpoint)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save(state, args.output_dir / "checkpoint_best.pt")
        history_path.write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
