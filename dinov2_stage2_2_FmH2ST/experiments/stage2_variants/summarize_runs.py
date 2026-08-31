#!/usr/bin/env python3
"""Print a compact Markdown comparison of Stage-2 training runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "total_loss",
    "node_reconstruction",
    "graph_contrastive",
    "edge_existence",
    "edge_weight",
)


def load_metrics(run_dir: Path) -> tuple[dict, float]:
    path = run_dir / "training_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    if not rows:
        raise ValueError(f"No metric rows found in {path}")
    best = min(float(row["total_loss"]) for row in rows if "total_loss" in row)
    return rows[-1], best


def formatted(value) -> str:
    return "-" if value is None else f"{float(value):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()

    columns = ("run", "iteration", *METRICS, "best_total_loss")
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for run_dir in args.run_dirs:
        latest, best = load_metrics(run_dir)
        values = [
            str(run_dir),
            str(latest.get("iteration", "-")),
            *(formatted(latest.get(metric)) for metric in METRICS),
            formatted(best),
        ]
        print("| " + " | ".join(values) + " |")


if __name__ == "__main__":
    main()
