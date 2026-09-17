"""Optional serial or torchrun/DDP execution for joint segmentation training.

The original ``train_joint`` CLI stays serial. This entry point selects the
execution layer while sharing its model, training phases and checkpoint writer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dinov2_segmentation import train_joint
from dinov2_segmentation.distributed_execution import DistributedExecution


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--execution-mode", choices=("serial", "ddp"), default="serial")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="auto uses CUDA when available; CPU uses Gloo for distributed checks",
    )
    parser.add_argument(
        "--init-checkpoint", type=Path,
        help="Load joint model weights only into a fresh run; optimizer and schedule restart",
    )
    parser.add_argument(
        "--migrate-resume",
        type=Path,
        help=(
            "Explicitly migrate a completed serial epoch checkpoint to DDP while "
            "retaining model and optimizer state and rebasing its schedule"
        ),
    )
    if "--help" in argv or "-h" in argv:
        print(parser.format_help())
    execution_args, remaining = parser.parse_known_args(argv)
    args = train_joint.parse_args(remaining)
    args.execution_mode = execution_args.execution_mode
    args.device = execution_args.device
    args.init_checkpoint = execution_args.init_checkpoint
    args.migrate_resume = execution_args.migrate_resume
    selected = [
        name
        for name, value in (
            ("--resume", args.resume),
            ("--init-checkpoint", args.init_checkpoint),
            ("--migrate-resume", args.migrate_resume),
        )
        if value is not None
    ]
    if len(selected) > 1:
        raise ValueError(f"{', '.join(selected)} cannot be combined")
    for name in ("init_checkpoint", "migrate_resume"):
        value = getattr(args, name)
        if value is not None:
            value = value.expanduser().resolve()
            if not value.is_file():
                raise FileNotFoundError(value)
            setattr(args, name, value)
    if args.migrate_resume is not None and args.execution_mode != "ddp":
        raise ValueError("--migrate-resume requires --execution-mode ddp")
    return args


def execution_manifest(args, execution, train_loader):
    """Record execution and data identity; reject silent schedule/data migration."""
    paths = {}
    for name in (
        "train_manifest", "val_manifest", "graph_dir", "stage1_config",
        "stage1_checkpoint", "stage2_config", "stage2_checkpoint",
    ):
        path = Path(getattr(args, name)).resolve()
        stat = path.stat()
        paths[name] = {
            "path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        }
    batches = len(train_loader)
    if args.max_train_batches > 0:
        batches = min(batches, args.max_train_batches)
    return {
        "version": 1,
        "mode": args.execution_mode,
        "world_size": execution.world_size,
        "per_rank_batch_size": args.batch_size,
        "effective_batch_size": (
            args.batch_size * execution.world_size * args.gradient_accumulation
        ),
        "train_batches_per_rank": batches,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "dropped_training_samples": getattr(train_loader.batch_sampler, "dropped_samples", 0),
        "validation_padding": False,
        "distributed_loss": "global_sufficient_statistics" if execution.distributed else "local_batch",
        "graph_context": "autograd_global_targets" if execution.distributed else "local_targets",
        "sources": paths,
    }


def main(argv=None):
    args = parse_args(argv)
    execution = DistributedExecution.from_environment(
        mode=args.execution_mode, device=None if args.device == "auto" else args.device
    )
    try:
        train_joint.main(args=args, execution=execution)
    finally:
        execution.close()


if __name__ == "__main__":
    main()
