"""Checkpoint-embedded history is the authoritative migration transaction."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

# Match the joint trainer's explicitly selected DINO source tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dinov2_stage2_2_FmH2ST"))

from dinov2_segmentation.train_joint import _migration_history


@pytest.mark.parametrize(
    "sidecar_text",
    (
        json.dumps([{"epoch": 0, "origin": "stale-sidecar"}]),
        '[{"epoch": 0, "origin": "truncated-sidecar"}',
    ),
)
def test_migration_prefers_embedded_history_over_stale_or_truncated_sidecar(
    tmp_path, sidecar_text
):
    checkpoint_path = tmp_path / "checkpoint_progress.pt"
    checkpoint_path.touch()
    (tmp_path / "history.json").write_text(sidecar_text, encoding="utf-8")
    embedded = [
        {"epoch": 0, "origin": "checkpoint"},
        {"epoch": 1, "origin": "checkpoint"},
    ]

    restored = _migration_history(
        checkpoint_path,
        completed_epochs=2,
        checkpoint={"history": embedded},
    )

    assert restored == embedded


def test_migration_rejects_noncontiguous_embedded_history_even_with_valid_sidecar(
    tmp_path,
):
    checkpoint_path = tmp_path / "checkpoint_progress.pt"
    checkpoint_path.touch()
    valid_sidecar = [{"epoch": epoch} for epoch in range(3)]
    (tmp_path / "history.json").write_text(
        json.dumps(valid_sidecar), encoding="utf-8"
    )
    embedded_with_gap = [{"epoch": 0}, {"epoch": 2}]

    with pytest.raises(ValueError, match="not a contiguous prefix"):
        _migration_history(
            checkpoint_path,
            completed_epochs=3,
            checkpoint={"history": embedded_with_gap},
        )
