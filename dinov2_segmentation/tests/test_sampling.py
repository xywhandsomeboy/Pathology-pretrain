from __future__ import annotations

from collections import Counter
import unittest

from dinov2_segmentation.sampling import (
    BOUNDARY,
    INTERIOR,
    NEGATIVE,
    SlideStratifiedSampler,
    WSILocalStratifiedSampler,
    patch_stratum,
)


def _rows() -> list[dict[str, object]]:
    rows = []
    for category, has_tumor, fraction in (
        (NEGATIVE, 0, 0.0),
        (BOUNDARY, 1, 0.4),
        (INTERIOR, 1, 1.0),
    ):
        for slide_index in range(4):
            for patch_index in range(6):
                rows.append(
                    {
                        "slide_id": f"{category}-slide-{slide_index}",
                        "patch_id": f"{category}-{slide_index}-{patch_index}",
                        "has_tumor": has_tumor,
                        "tumor_fraction": fraction,
                        "x": patch_index * 512,
                        "y": slide_index * 512,
                    }
                )
    return rows


class SlideStratifiedSamplerTest(unittest.TestCase):
    def test_exact_strata_batch_uniqueness_and_slide_balance(self):
        rows = _rows()
        sampler = SlideStratifiedSampler(
            rows,
            num_samples=100,
            batch_size=10,
            positive_fraction=0.6,
            boundary_positive_fraction=0.5,
            seed=7,
        )
        indices = list(sampler)
        self.assertEqual(len(indices), 100)
        self.assertLessEqual(max(Counter(indices).values()), 2)
        for start in range(0, len(indices), 10):
            batch = indices[start : start + 10]
            self.assertEqual(len(batch), len(set(batch)))
            self.assertTrue(any(int(rows[index]["has_tumor"]) for index in batch))

        strata = Counter(patch_stratum(rows[index]) for index in indices)
        self.assertEqual(
            strata,
            Counter({NEGATIVE: 40, BOUNDARY: 30, INTERIOR: 30}),
        )
        for category in (NEGATIVE, BOUNDARY, INTERIOR):
            by_slide = Counter(
                str(rows[index]["slide_id"])
                for index in indices
                if patch_stratum(rows[index]) == category
            )
            self.assertLessEqual(max(by_slide.values()) - min(by_slide.values()), 1)

    def test_epoch_order_is_deterministic_but_changes(self):
        rows = _rows()
        sampler = SlideStratifiedSampler(
            rows, num_samples=48, batch_size=8, seed=11
        )
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))

    def test_manifest_consistency_is_validated(self):
        with self.assertRaisesRegex(ValueError, "requires .*tumor_fraction"):
            patch_stratum({"slide_id": "s", "has_tumor": 1})
        with self.assertRaisesRegex(ValueError, "requires tumor_fraction=0"):
            patch_stratum(
                {"slide_id": "s", "has_tumor": 0, "tumor_fraction": 0.2}
            )

    def test_wsi_local_sampler_preserves_population_and_batches(self):
        rows = _rows()
        common = dict(
            num_samples=100,
            batch_size=10,
            positive_fraction=0.6,
            boundary_positive_fraction=0.5,
            seed=7,
        )
        baseline = SlideStratifiedSampler(rows, **common)
        local = WSILocalStratifiedSampler(
            rows, **common, locality_tile_size=1024, global_batch_size=20
        )
        baseline_indices = list(baseline)
        local_indices = list(local)
        self.assertEqual(Counter(local_indices), Counter(baseline_indices))
        self.assertEqual(local.summary["name"], "wsi_local_stratified_boundary")
        self.assertEqual(local.summary["locality_tile_size"], 1024)
        self.assertEqual(local.summary["global_batch_size"], 20)
        for start in range(0, len(local_indices), 10):
            batch = local_indices[start : start + 10]
            self.assertEqual(len(batch), len(set(batch)))
            self.assertTrue(any(int(rows[index]["has_tumor"]) for index in batch))
            # Full chunks are one WSI; this tiny fixture deliberately forces
            # repeat-cap partial chunks, which can pack a few WSIs together.
            self.assertLessEqual(
                len({str(rows[index]["slide_id"]) for index in batch}), 4
            )
        for start in range(0, len(local_indices), 20):
            self.assertEqual(
                len(local_indices[start : start + 20]),
                len(set(local_indices[start : start + 20])),
            )

    def test_wsi_local_sampler_requires_coordinates(self):
        rows = _rows()
        del rows[0]["x"]
        with self.assertRaisesRegex(ValueError, "requires integer x and y"):
            WSILocalStratifiedSampler(rows, num_samples=12, batch_size=4)


if __name__ == "__main__":
    unittest.main()
