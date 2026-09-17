import csv
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from dinov2_segmentation.validate_progress_checkpoints import (
    _allocate_proportional_quotas,
    _atomic_json,
    _fingerprint,
    _fingerprint_token,
    _processed_marker,
    _discard_processed_snapshot,
    _publish_result,
    build_stratified_monitor_manifest,
    capture_progress_checkpoint,
    parse_args,
    watch,
)


class AsyncProgressValidationTests(unittest.TestCase):
    def test_restart_skips_published_queue_without_evaluating_and_accepts_new_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            source = run / "checkpoint_progress.pt"
            source.write_bytes(b"completed-generation")
            snapshot = capture_progress_checkpoint(source)
            result = {
                "status": "complete", "protocol_id": "fixed-subset",
                "checkpoint": {
                    "fingerprint": _fingerprint(snapshot), "sha256": "test-sha",
                    "curriculum_step": 40000, "epoch": 1, "next_batch_index": 20000,
                },
                "metrics": {"tumor_dice": 0.89},
            }
            # Simulate a crash after publication, before spool cleanup.
            result_path = _publish_result(run, snapshot, result)
            args = parse_args(["--watch-root", str(root), "--device", "cpu", "--once"])
            with mock.patch(
                "dinov2_segmentation.validate_progress_checkpoints.evaluate_snapshot"
            ) as evaluate:
                watch(args)
                evaluate.assert_not_called()
            self.assertFalse(snapshot.exists())
            self.assertTrue(result_path.is_file())
            self.assertEqual(source.read_bytes(), b"completed-generation")
            self.assertEqual(
                (run / "async_validation/checkpoint_best_monitor.pt").read_bytes(),
                b"completed-generation",
            )
            self.assertIsNone(capture_progress_checkpoint(source))
            replacement = run / "new.pt"
            replacement.write_bytes(b"new-generation")
            os.replace(replacement, source)
            next_snapshot = capture_progress_checkpoint(source)
            self.assertIsNotNone(next_snapshot)
            self.assertFalse(_discard_processed_snapshot(next_snapshot))
            self.assertEqual(next_snapshot.read_bytes(), b"new-generation")

    def test_failed_validation_stays_queued_for_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            source = run / "checkpoint_progress.pt"
            source.write_bytes(b"retry-me")
            snapshot = capture_progress_checkpoint(source)
            failure = snapshot.with_suffix(".failure.json")
            _atomic_json({"status": "failed"}, failure)
            self.assertFalse(_discard_processed_snapshot(snapshot))
            self.assertTrue(snapshot.exists())
            self.assertTrue(failure.exists())

    def test_proportional_quotas_are_exact_and_bounded(self):
        counts = {("a", "negative"): 11, ("a", "boundary"): 5, ("b", "interior"): 4}
        quotas = _allocate_proportional_quotas(counts, 9)
        self.assertEqual(sum(quotas.values()), 9)
        self.assertTrue(all(0 <= quotas[key] <= counts[key] for key in counts))

    def test_monitor_manifest_is_deterministic_and_stratified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "valid.csv"
            fields = ("slide_id", "patch_id", "has_tumor", "tumor_fraction", "image_path", "mask_path")
            rows = []
            for index in range(60):
                category = index % 3
                rows.append(
                    {
                        "slide_id": f"slide-{index % 2}",
                        "patch_id": f"patch-{index:03d}",
                        "has_tumor": 0 if category == 0 else 1,
                        "tumor_fraction": 0.0 if category == 0 else (0.5 if category == 1 else 1.0),
                        "image_path": f"image-{index}.jpg",
                        "mask_path": f"mask-{index}.png",
                    }
                )
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            first = root / "subset-a.csv"
            second = root / "subset-b.csv"
            first_metadata = build_stratified_monitor_manifest(source, first, subset_size=30, seed=7)
            second_metadata = build_stratified_monitor_manifest(source, second, subset_size=30, seed=7)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_metadata["manifest_sha256"], second_metadata["manifest_sha256"])
            selected = list(csv.DictReader(first.open(newline="", encoding="utf-8")))
            self.assertEqual(len(selected), 30)
            self.assertEqual(len({row["patch_id"] for row in selected}), 30)
            by_stratum = {
                (item["slide_id"], item["category"]): item["selected_count"]
                for item in first_metadata["strata"]
            }
            self.assertEqual(set(by_stratum.values()), {5})

    def test_capture_keeps_each_atomic_generation_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            source = run_dir / "checkpoint_progress.pt"
            source.write_bytes(b"generation-one")
            first = capture_progress_checkpoint(source)
            replacement = run_dir / "checkpoint_progress.pt.tmp"
            replacement.write_bytes(b"generation-two")
            os.replace(replacement, source)
            second = capture_progress_checkpoint(source)
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), b"generation-one")
            self.assertEqual(second.read_bytes(), b"generation-two")
            self.assertTrue(first.with_suffix(".capture.json").is_file())
            self.assertTrue(second.with_suffix(".capture.json").is_file())

            token = _fingerprint_token(_fingerprint(second))
            _atomic_json({"processed": True}, _processed_marker(run_dir, token))
            second.unlink()
            self.assertIsNone(capture_progress_checkpoint(source))


if __name__ == "__main__":
    unittest.main()
