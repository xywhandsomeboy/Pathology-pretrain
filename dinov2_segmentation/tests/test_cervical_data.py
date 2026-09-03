import csv
import json
import os
import shutil
import subprocess
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from openpyxl import Workbook

from dinov2_segmentation.prepare_cervical_data import (
    _binary_mask,
    _is_exact_file,
    _scaled_polygons,
    _write_selections,
    discover_ready,
)


DOWNLOAD_TOOLS_AVAILABLE = all(
    shutil.which(command) for command in ("curl", "flock", "jq", "xargs")
)


def test_zero_byte_manifest_entry_is_not_training_ready(tmp_path):
    empty_path = tmp_path / "empty.isyntax"
    empty_path.touch()

    assert not _is_exact_file(empty_path, {empty_path.name: 0})


def _run_downloader(tmp_path: Path, remote_manifest: Path, dataset_dir: Path):
    repo_root = Path(__file__).resolve().parents[2]
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CERVICAL_SEGMENT_DATA_DIR": str(state_dir),
            "CERVICAL_DOWNLOAD_BASE_URL": remote_manifest.parent.as_uri(),
            "CERVICAL_DOWNLOAD_MANIFEST_URL": remote_manifest.as_uri(),
            "CERVICAL_DOWNLOAD_MANIFEST": str(state_dir / "manifest.json"),
            "CERVICAL_DOWNLOAD_PARALLEL": "2",
            "CERVICAL_DOWNLOAD_LOCK_FILE": str(tmp_path / "download.lock"),
        }
    )
    return subprocess.run(
        ["bash", str(repo_root / "scripts/download_cervical_dataset.sh"), str(dataset_dir)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.skipif(not DOWNLOAD_TOOLS_AVAILABLE, reason="download shell tools unavailable")
def test_downloader_normalizes_remote_space_and_materializes_zero_byte_files(tmp_path):
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    remote_files = {
        "lesion .geojson": b'{"type":"FeatureCollection","features":[]}',
        "empty.csv": b"",
        "oversize.bin": b"x",
    }
    manifest = []
    for name, content in remote_files.items():
        path = remote_dir / name
        path.write_bytes(content)
        manifest.append({"path": f"raw/{name}", "size": len(content)})
    remote_manifest = remote_dir / "manifest.json"
    remote_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "oversize.bin").write_bytes(b"too large")

    result = _run_downloader(tmp_path, remote_manifest, dataset_dir)

    assert result.returncode == 0, result.stderr
    assert (dataset_dir / "lesion.geojson").read_bytes() == remote_files["lesion .geojson"]
    assert not (dataset_dir / "lesion .geojson").exists()
    assert (dataset_dir / "empty.csv").exists()
    assert (dataset_dir / "empty.csv").stat().st_size == 0
    assert (dataset_dir / "oversize.bin").read_bytes() == b"x"
    assert len(list((dataset_dir / ".download-quarantine").glob("oversize.bin.*.oversize"))) == 1


@pytest.mark.skipif(not DOWNLOAD_TOOLS_AVAILABLE, reason="download shell tools unavailable")
def test_downloader_does_not_replace_manifest_with_invalid_json(tmp_path):
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    remote_manifest = remote_dir / "manifest.json"
    remote_manifest.write_text('{"unexpected":"object"}', encoding="utf-8")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    existing_manifest = state_dir / "manifest.json"
    original = '[{"path":"existing.bin","size":1}]'
    existing_manifest.write_text(original, encoding="utf-8")

    result = _run_downloader(tmp_path, remote_manifest, tmp_path / "dataset")

    assert result.returncode != 0
    assert existing_manifest.read_text(encoding="utf-8") == original


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
            manifest_name = path.name
            if slide_id == "lesion" and path.suffix == ".geojson":
                manifest_name = f"{path.stem} .geojson"
            manifest.append(
                {"path": f"raw/{manifest_name}", "size": path.stat().st_size}
            )
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
