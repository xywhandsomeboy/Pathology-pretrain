#!/usr/bin/env python3
"""Monitor joint cervical-segmentation runs and stop clear overfitting early.

The training process writes ``history.json`` only after validation and after the
last/best checkpoints have been saved.  This monitor therefore acts on complete
epoch evidence and leaves a resumable checkpoint behind.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Detection:
    reason: str
    epoch: int
    best_epoch: int
    best_val_dice: float
    val_dice: float
    train_dice: float
    val_loss: float
    train_loss: float


def _metric(record: dict[str, Any], split: str, name: str) -> float:
    return float(record[split][name])


def detect_overfitting(history: list[dict[str, Any]]) -> Detection | None:
    """Return evidence only for a collapse or a sustained generalization gap."""
    if len(history) < 3:
        return None

    best_index = max(
        range(len(history)), key=lambda index: _metric(history[index], "val", "tumor_dice")
    )
    best = history[best_index]
    current = history[-1]
    best_dice = _metric(best, "val", "tumor_dice")
    val_dice = _metric(current, "val", "tumor_dice")
    train_dice = _metric(current, "train", "tumor_dice")
    val_loss = _metric(current, "val", "loss")
    train_loss = _metric(current, "train", "loss")
    gap = train_dice - val_dice

    collapse = (
        best_dice >= 0.65
        and gap >= 0.15
        and (val_dice < 0.50 or best_dice - val_dice >= 0.15)
    )
    if collapse:
        return Detection(
            reason="validation_tumor_dice_collapse",
            epoch=int(current["epoch"]),
            best_epoch=int(best["epoch"]),
            best_val_dice=best_dice,
            val_dice=val_dice,
            train_dice=train_dice,
            val_loss=val_loss,
            train_loss=train_loss,
        )

    if len(history) < 4 or best_index >= len(history) - 2:
        return None
    recent = history[-3:]
    train_losses = [_metric(record, "train", "loss") for record in recent]
    val_losses = [_metric(record, "val", "loss") for record in recent]
    val_dices = [_metric(record, "val", "tumor_dice") for record in recent]
    sustained = (
        gap >= 0.12
        and best_dice - val_dice >= 0.03
        and train_losses[0] > train_losses[1] > train_losses[2]
        and val_losses[0] < val_losses[1] < val_losses[2]
        and val_dices[0] > val_dices[1] > val_dices[2]
    )
    if sustained:
        return Detection(
            reason="sustained_train_val_divergence",
            epoch=int(current["epoch"]),
            best_epoch=int(best["epoch"]),
            best_val_dice=best_dice,
            val_dice=val_dice,
            train_dice=train_dice,
            val_loss=val_loss,
            train_loss=train_loss,
        )
    return None


def _training_pids(output_dir: Path) -> list[int]:
    target = str(output_dir.resolve())
    matches: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
            decoded = [arg.decode(errors="replace") for arg in args if arg]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "dinov2_segmentation.train_joint" not in decoded:
            continue
        for index, arg in enumerate(decoded[:-1]):
            if arg == "--output-dir" and str(Path(decoded[index + 1]).resolve()) == target:
                matches.append(int(entry.name))
                break
    if not matches:
        return []
    match_set = set(matches)
    roots = []
    for pid in matches:
        try:
            fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
            parent_pid = int(fields[3])
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
        if parent_pid not in match_set:
            roots.append(pid)
    return sorted(roots or matches)


def _write_detection(output_dir: Path, detection: Detection, pids: list[int]) -> None:
    payload = {
        "detected_at": datetime.now().astimezone().isoformat(),
        "reason": detection.reason,
        "epoch": detection.epoch,
        "best_epoch": detection.best_epoch,
        "best_val_dice": detection.best_val_dice,
        "val_dice": detection.val_dice,
        "train_dice": detection.train_dice,
        "val_loss": detection.val_loss,
        "train_loss": detection.train_loss,
        "terminated_root_pids": pids,
    }
    destination = output_dir / "overfitting_detected.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _run_once(work_root: Path, stop_on_detection: bool, seen: dict[Path, int]) -> bool:
    changed = False
    runs_root = work_root / "decoder_runs"
    for history_path in sorted(runs_root.glob("*/*/history.json")):
        output_dir = history_path.parent
        if (output_dir / "overfitting_detected.json").exists():
            continue
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            continue
        if not isinstance(history, list) or not history:
            continue
        if seen.get(history_path) != len(history):
            latest = history[-1]
            print(
                json.dumps(
                    {
                        "observed_at": datetime.now().astimezone().isoformat(),
                        "run": str(output_dir.relative_to(runs_root)),
                        "completed_epochs": len(history),
                        "epoch": int(latest["epoch"]),
                        "train_loss": _metric(latest, "train", "loss"),
                        "train_dice": _metric(latest, "train", "tumor_dice"),
                        "val_loss": _metric(latest, "val", "loss"),
                        "val_dice": _metric(latest, "val", "tumor_dice"),
                    }
                ),
                flush=True,
            )
            seen[history_path] = len(history)
            changed = True
        detection = detect_overfitting(history)
        if detection is None:
            continue
        pids = _training_pids(output_dir)
        print(
            json.dumps(
                {
                    "overfitting_detected": str(output_dir.relative_to(runs_root)),
                    **detection.__dict__,
                    "training_root_pids": pids,
                    "stop_on_detection": stop_on_detection,
                }
            ),
            flush=True,
        )
        if stop_on_detection:
            _write_detection(output_dir, detection, pids)
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        changed = True
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--stop-on-detection", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")

    seen: dict[Path, int] = {}
    while True:
        _run_once(args.work_root.resolve(), args.stop_on_detection, seen)
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
