#!/usr/bin/env python3
"""Compare completed Stage-1A runs at their preserved checkpoint endpoints.

The comparison deliberately uses only completed runs.  For each ``model_final``,
archived same-budget endpoint, and cosine-cycle endpoint, it summarizes the same
trailing iteration window from ``training_metrics.json`` and ranks candidates
by mean total loss.  This avoids selecting a checkpoint from one unusually easy
training batch while retaining a strict equal-step comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


METRICS = (
    "total_loss",
    "view_consistency",
    "teacher_alignment",
    "spatial_view_consistency",
    "spatial_teacher_alignment",
    "local_teacher_alignment",
)
ITERATION_PATTERN = re.compile(r"model_(\d+)\.rank_0\.pth$")


def _load_metrics(path: Path) -> tuple[list[dict[str, float]], dict[str, int]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            missing = [name for name in ("iteration", *METRICS) if name not in record]
            if missing:
                raise ValueError(f"{path}:{line_number} lacks metrics: {missing}")
            if not all(math.isfinite(float(record[name])) for name in METRICS):
                raise ValueError(f"Non-finite metric at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"No metric records in {path}")
    # A no-resume relaunch may append a fresh iteration-0 segment to an
    # existing metrics file. Only the final monotonic segment can correspond
    # to model_final; mixing the abandoned prefix would bias the comparison.
    restart_indices = [
        index
        for index in range(1, len(records))
        if int(records[index]["iteration"]) <= int(records[index - 1]["iteration"])
    ]
    segment_start = restart_indices[-1] if restart_indices else 0
    selected = records[segment_start:]
    iterations = [int(record["iteration"]) for record in selected]
    if iterations != sorted(iterations) or len(iterations) != len(set(iterations)):
        raise ValueError(f"Final metric segment is not strictly increasing in {path}")
    provenance = {
        "raw_records": len(records),
        "selected_records": len(selected),
        "discarded_prefix_records": segment_start,
        "restart_boundaries": len(restart_indices),
    }
    return selected, provenance


def _checkpoint_iteration(path: Path, final_iteration: int) -> int:
    if path.name == "model_final.rank_0.pth":
        return final_iteration
    match = ITERATION_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Cannot infer checkpoint iteration from {path}")
    return int(match.group(1))


def _candidate_checkpoints(run_dir: Path, final_iteration: int) -> list[Path]:
    final_checkpoint = run_dir / "model_final.rank_0.pth"
    if not final_checkpoint.is_file():
        raise FileNotFoundError(
            f"Stage-1 run is not complete (missing {final_checkpoint})"
        )
    candidates = [final_checkpoint]
    for archive_name in ("budget_checkpoints", "cycle_checkpoints"):
        archive_dir = run_dir / archive_name
        if archive_dir.is_dir():
            candidates.extend(sorted(archive_dir.glob("model_*.rank_0.pth")))

    # model_final and the last cycle endpoint may be hard links to one inode.
    unique = []
    seen_inodes = set()
    seen_iterations = set()
    for path in candidates:
        inode = (path.stat().st_dev, path.stat().st_ino)
        iteration = _checkpoint_iteration(path, final_iteration)
        if inode in seen_inodes or iteration in seen_iterations:
            continue
        seen_inodes.add(inode)
        seen_iterations.add(iteration)
        if iteration > final_iteration:
            raise ValueError(
                f"Checkpoint {path} is newer than the final metric iteration "
                f"{final_iteration}"
            )
        unique.append(path)
    return unique


def _summarize_candidate(
    run_dir: Path,
    checkpoint: Path,
    records: list[dict[str, float]],
    window_iterations: int,
) -> dict:
    final_iteration = int(records[-1]["iteration"])
    checkpoint_iteration = _checkpoint_iteration(checkpoint, final_iteration)
    start_iteration = checkpoint_iteration - window_iterations + 1
    window = [
        record
        for record in records
        if start_iteration <= int(record["iteration"]) <= checkpoint_iteration
    ]
    if len(window) < 2:
        raise ValueError(
            f"Not enough metrics before checkpoint {checkpoint}; got {len(window)}"
        )
    summary = {
        "run": run_dir.name,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_iteration": checkpoint_iteration,
        "window_start_iteration": int(window[0]["iteration"]),
        "window_end_iteration": int(window[-1]["iteration"]),
        "window_records": len(window),
        "metrics": {},
    }
    for name in METRICS:
        values = [float(record[name]) for record in window]
        summary["metrics"][name] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    return summary


def compare(run_dirs: list[Path], window_iterations: int) -> dict:
    candidates = []
    for run_dir in run_dirs:
        run_dir = run_dir.expanduser().resolve()
        metrics_path = run_dir / "training_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing Stage-1 metrics: {metrics_path}")
        records, metrics_log = _load_metrics(metrics_path)
        final_iteration = int(records[-1]["iteration"])
        for checkpoint in _candidate_checkpoints(run_dir, final_iteration):
            candidate = _summarize_candidate(
                run_dir,
                checkpoint,
                records,
                window_iterations,
            )
            candidate["metrics_log"] = metrics_log
            candidates.append(candidate)
    if len(candidates) < 2:
        raise ValueError("At least two completed Stage-1 candidates are required")
    candidates.sort(
        key=lambda item: (
            item["metrics"]["total_loss"]["mean"],
            item["metrics"]["total_loss"]["std"],
            item["checkpoint_iteration"],
        )
    )
    return {
        "selection_rule": "lowest trailing-window mean total_loss, then std",
        "window_iterations": window_iterations,
        "winner": candidates[0],
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--window-iterations", type=int, default=1000)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.window_iterations < 1:
        parser.error("--window-iterations must be positive")

    result = compare(args.run_dirs, args.window_iterations)
    print(
        "Stage-1 winner: "
        f"{result['winner']['run']} @ {result['winner']['checkpoint_iteration']} "
        f"(mean total_loss="
        f"{result['winner']['metrics']['total_loss']['mean']:.8f})"
    )
    for index, candidate in enumerate(result["candidates"], 1):
        total = candidate["metrics"]["total_loss"]
        print(
            f"{index}. {candidate['run']} @ {candidate['checkpoint_iteration']}: "
            f"total_loss={total['mean']:.8f}±{total['std']:.8f}, "
            f"records={candidate['window_records']}"
        )
        discarded = candidate["metrics_log"]["discarded_prefix_records"]
        if discarded:
            print(
                f"   selected final monotonic log segment; "
                f"discarded {discarded} prefix records"
            )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output_json)
        print(f"Saved comparison: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
