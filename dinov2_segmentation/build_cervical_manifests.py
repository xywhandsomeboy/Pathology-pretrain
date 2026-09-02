"""Attach one Stage-2 feature-store set to the frozen decoder selections."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = (
    "slide_id",
    "patch_id",
    "x",
    "y",
    "level",
    "image_path",
    "mask_path",
    "feature_path",
    "feature_index",
    "has_tumor",
    "tumor_fraction",
    "tissue_fraction",
)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--feature-store-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summaries = []
    for split in ("train", "valid", "test"):
        selection_path = args.selection_dir / f"{split}.csv"
        with selection_path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"Decoder selection contains no {split} patches")
        for row in rows:
            store_path = (args.feature_store_dir / f"{row['slide_id']}.pt").resolve()
            if not store_path.is_file():
                raise FileNotFoundError(f"Missing decoder feature store: {store_path}")
            row["feature_path"] = str(store_path)
            row["feature_index"] = ""
        _write(args.output_dir / f"{split}.csv", rows)
        summaries.append(
            f"{split}={len(rows)} (tumor={sum(int(row['has_tumor']) for row in rows)})"
        )
    print("Wrote decoder manifests: " + ", ".join(summaries))


if __name__ == "__main__":
    main()
