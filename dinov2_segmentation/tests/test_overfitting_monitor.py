from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
