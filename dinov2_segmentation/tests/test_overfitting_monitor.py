from __future__ import annotations

import io
import json
import signal
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from scripts import monitor_cervical_overfitting as monitor
from scripts.monitor_cervical_overfitting import detect_overfitting


def _record(
    epoch: int,
    train_loss: float,
    train_dice: float,
    val_loss: float,
    val_dice: float,
):
    return {
        "epoch": epoch,
        "train": {"loss": train_loss, "tumor_dice": train_dice},
        "val": {"loss": val_loss, "tumor_dice": val_dice},
    }


class OverfittingMonitorTest(unittest.TestCase):
    def test_detects_three_epoch_loss_divergence_after_dice_plateau(self):
        history = [
            _record(0, 0.4066, 0.8880, 0.6365, 0.7357),
            _record(1, 0.2450, 0.9369, 0.7586, 0.7399),
            _record(2, 0.2148, 0.9448, 0.8610, 0.7387),
        ]
        detection = detect_overfitting(history)
        self.assertIsNotNone(detection)
        self.assertEqual(detection.reason, "sustained_validation_loss_divergence")

    def test_does_not_stop_while_validation_dice_keeps_improving(self):
        history = [
            _record(0, 0.40, 0.88, 0.50, 0.70),
            _record(1, 0.30, 0.92, 0.61, 0.75),
            _record(2, 0.20, 0.94, 0.72, 0.78),
        ]
        self.assertIsNone(detect_overfitting(history))

    def test_discovers_legacy_and_improved_run_layouts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            work_root = Path(temporary_directory)
            legacy = work_root / "decoder_runs" / "baseline" / "v1" / "history.json"
            improved = (
                work_root
                / "decoder_runs_improved"
                / "STA"
                / "weighted_pretrain_distance_context"
                / "v2"
                / "history.json"
            )
            unrelated = (
                work_root
                / "decoder_runs_improved"
                / "too"
                / "shallow"
                / "history.json"
            )
            for path in (legacy, improved, unrelated):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("[]\n", encoding="utf-8")

            discovered = {
                (runs_root.name, history_path.relative_to(work_root))
                for runs_root, history_path in monitor._discover_run_histories(work_root)
            }

            self.assertEqual(
                discovered,
                {
                    ("decoder_runs", legacy.relative_to(work_root)),
                    ("decoder_runs_improved", improved.relative_to(work_root)),
                },
            )

    def test_improved_run_uses_its_exact_output_directory_as_stop_target(self):
        history = [
            _record(0, 0.4066, 0.8880, 0.6365, 0.7357),
            _record(1, 0.2450, 0.9369, 0.7586, 0.7399),
            _record(2, 0.2148, 0.9448, 0.8610, 0.7387),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            work_root = Path(temporary_directory)
            output_dir = (
                work_root
                / "decoder_runs_improved"
                / "ST"
                / "weighted_pretrain_distance_context"
                / "v1"
            )
            output_dir.mkdir(parents=True)
            (output_dir / "history.json").write_text(
                json.dumps(history), encoding="utf-8"
            )

            with (
                mock.patch.object(monitor, "_training_pids", return_value=[4321]) as pids,
                mock.patch.object(monitor, "_write_detection") as write_detection,
                mock.patch.object(monitor.os, "kill") as kill,
                redirect_stdout(io.StringIO()) as stdout,
            ):
                changed = monitor._run_once(work_root, True, {})

            self.assertTrue(changed)
            pids.assert_called_once_with(output_dir)
            write_detection.assert_called_once()
            self.assertEqual(write_detection.call_args.args[0], output_dir)
            kill.assert_called_once_with(4321, signal.SIGTERM)
            events = [json.loads(line) for line in stdout.getvalue().splitlines()]
            detection_event = events[-1]
            self.assertEqual(detection_event["runs_root"], "decoder_runs_improved")
            self.assertEqual(
                detection_event["overfitting_detected"],
                "ST/weighted_pretrain_distance_context/v1",
            )


if __name__ == "__main__":
    unittest.main()
