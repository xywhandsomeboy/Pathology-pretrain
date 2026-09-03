from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from dinov2_segmentation.stitching import DiskBackedSlideStitcher


class StitchingThresholdTest(unittest.TestCase):
    def test_validation_selected_binary_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            probability = np.array(
                [
                    [[0.6, 0.4], [0.8, 0.2]],
                    [[0.4, 0.6], [0.2, 0.8]],
                ],
                dtype=np.float32,
            )
            stitcher = DiskBackedSlideStitcher(
                Path(directory),
                slide_id="slide",
                level=1,
                num_classes=2,
                height=2,
                width=2,
            )
            stitcher.add(probability, 0, 0)
            path = stitcher.finalize(tumor_threshold=0.3)
            prediction = np.load(path)
            np.testing.assert_array_equal(
                prediction, np.array([[1, 1], [0, 1]], dtype=np.uint16)
            )

    def test_threshold_rejects_non_binary_output(self):
        with tempfile.TemporaryDirectory() as directory:
            stitcher = DiskBackedSlideStitcher(
                Path(directory),
                slide_id="slide",
                level=1,
                num_classes=3,
                height=2,
                width=2,
            )
            with self.assertRaisesRegex(ValueError, "binary"):
                stitcher.finalize(tumor_threshold=0.4)


if __name__ == "__main__":
    unittest.main()
