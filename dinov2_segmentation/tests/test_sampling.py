from __future__ import annotations

from collections import Counter
import unittest

from dinov2_segmentation.sampling import (
    BOUNDARY,
    INTERIOR,
    NEGATIVE,
    SlideStratifiedSampler,
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


if __name__ == "__main__":
    unittest.main()
