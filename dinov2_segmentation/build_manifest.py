"""Convert patch-coordinate CSV metadata into a segmentation manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _basename_index(root: Path, wanted: set[str]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates = set()
    for path in root.rglob("*"):
        if path.is_file() and path.name in wanted:
            if path.name in index:
                duplicates.add(path.name)
            else:
                index[path.name] = path.resolve()
    if duplicates:
        names = ", ".join(sorted(duplicates)[:5])
        raise ValueError(f"Patch basenames are ambiguous under {root}: {names}")
    return index


def build_manifest(args) -> int:
    with args.patch_csv.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        required = {args.slide_column, args.path_column, "x", "y"}
        missing = sorted(required.difference(columns))
        if missing:
            raise ValueError(f"Patch CSV is missing columns: {missing}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Patch CSV contains no rows: {args.patch_csv}")

    unresolved = {
        Path(row[args.path_column]).name
        for row in rows
        if not Path(row[args.path_column]).expanduser().is_file()
    }
    basename_paths = (
        _basename_index(args.patch_root.resolve(), unresolved)
        if args.patch_root is not None and unresolved
        else {}
    )
    output_rows = []
    for row in rows:
        slide_id = str(row[args.slide_column])
        raw_path = Path(row[args.path_column]).expanduser()
        image_path = raw_path.resolve() if raw_path.is_file() else basename_paths.get(raw_path.name)
        if image_path is None:
            raise FileNotFoundError(
                f"Cannot resolve patch {raw_path}; pass --patch-root if the dataset moved"
            )
        patch_id = str(
            row.get(args.patch_id_column, "")
            or row.get("patch_id", "")
            or image_path.stem
        )
        feature_path = (args.feature_dir / f"{slide_id}.pt").resolve()
        if not feature_path.is_file() and not args.allow_missing_features:
            raise FileNotFoundError(f"Missing slide feature store: {feature_path}")
        output = {
            "slide_id": slide_id,
            "patch_id": patch_id,
            "x": int(float(row["x"])),
            "y": int(float(row["y"])),
            "level": int(row.get(args.level_column) or 0),
            "image_path": str(image_path),
            "feature_path": str(feature_path),
            "feature_index": "",
        }
        if args.mask_root is not None:
            mask_path = (args.mask_root / f"{patch_id}{args.mask_suffix}").resolve()
            if not mask_path.is_file() and not args.allow_missing_masks:
                raise FileNotFoundError(f"Missing patch mask: {mask_path}")
            output["mask_path"] = str(mask_path)
        output_rows.append(output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    return len(output_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-csv", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-root", type=Path)
    parser.add_argument("--mask-root", type=Path)
    parser.add_argument("--mask-suffix", default=".png")
    parser.add_argument("--slide-column", default="slide_name_split")
    parser.add_argument("--path-column", default="filepath")
    parser.add_argument("--patch-id-column", default="patch_id")
    parser.add_argument("--level-column", default="level")
    parser.add_argument("--allow-missing-features", action="store_true")
    parser.add_argument("--allow-missing-masks", action="store_true")
    args = parser.parse_args()
    count = build_manifest(args)
    print(f"Wrote {count} aligned patch rows to {args.output}")


if __name__ == "__main__":
    main()
