import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
from openpyxl import Workbook

from dinov2_segmentation.prepare_cervical_data import (
    _binary_mask,
    _scaled_polygons,
    _write_selections,
    discover_ready,
)


def _write_workbook(path: Path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(
        [
            "Filename",
            "Category",
            "train/valid/test",
            "ExcludedFromAnnotation",
        ]
    )
    sheet.append(["normal.isyntax", "normal_inflammation", "train", 0])
    sheet.append(["lesion.isyntax", "high_grade", "valid", 0])
    workbook.save(path)


def test_readiness_collapses_grades_but_preserves_normal(tmp_path):
    data_root = tmp_path / "raw"
    data_root.mkdir()
    normal_geojson = {"type": "FeatureCollection", "features": []}
    lesion_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"classification": {"name": "High Grade"}},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [16, 0], [16, 16], [0, 16], [0, 0]]],
                },
            }
        ],
    }
    manifest = []
    for slide_id, geojson in (("normal", normal_geojson), ("lesion", lesion_geojson)):
        slide_path = data_root / f"{slide_id}.isyntax"
        slide_path.write_bytes(b"slide")
        annotation_path = data_root / f"{slide_id}.geojson"
        annotation_path.write_text(json.dumps(geojson), encoding="utf-8")
        for path in (slide_path, annotation_path):
            manifest.append({"path": f"raw/{path.name}", "size": path.stat().st_size})
    manifest_path = tmp_path / "download.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    workbook_path = tmp_path / "metadata.xlsx"
    _write_workbook(workbook_path)
    args = Namespace(
        data_root=data_root,
        download_manifest=manifest_path,
        metadata_xlsx=workbook_path,
        minimum_ready=2,
        minimum_train=1,
        minimum_valid=1,
    )

    ready, status = discover_ready(args)

    assert status["ready"] is True
    assert {row["slide_id"]: row["binary_slide_class"] for row in ready} == {
        "lesion": 1,
        "normal": 0,
    }
    assert status["label_map"]["low_grade"] == 1
    assert status["label_map"]["malignant"] == 1


def test_all_tumor_grades_render_to_one_binary_value():
    features = []
    for offset, name in enumerate(("Low Grade", "High grade", "Malignant")):
        left = 4 + offset * 8
        features.append(
            {
                "properties": {"classification": {"name": name}},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[left, 4], [left + 4, 4], [left + 4, 8], [left, 8], [left, 4]]
                    ],
                },
            }
        )
    mask = np.asarray(_binary_mask(_scaled_polygons(features, 1), 0, 0, 32))

    assert set(np.unique(mask)) == {0, 1}
    assert mask.sum() > 0


def test_training_selection_is_deterministic_and_keeps_all_positive(tmp_path):
    rows = []
    for index in range(20):
        rows.append(
            {
                "slide_name_split": "slide",
                "slide_id": "slide",
                "patch_id": f"p{index}",
                "filepath": f"p{index}.jpg",
                "image_path": f"p{index}.jpg",
                "mask_path": f"p{index}.png",
                "x": index,
                "y": 0,
                "level": 1,
                "split": "train",
                "has_tumor": int(index < 2),
                "tumor_fraction": "0.1" if index < 2 else "0",
                "tissue_fraction": "1",
            }
        )
    _write_selections(rows, tmp_path, seed=42, negative_ratio=3)
    first = (tmp_path / "decoder_selection/train.csv").read_text()
    _write_selections(rows, tmp_path, seed=42, negative_ratio=3)
    second = (tmp_path / "decoder_selection/train.csv").read_text()
    selected = list(csv.DictReader(first.splitlines()))

    assert first == second
    assert {row["patch_id"] for row in selected if int(row["has_tumor"])} == {"p0", "p1"}
