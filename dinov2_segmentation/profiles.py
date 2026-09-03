"""Stable identities and validation rules for training-strategy ablations."""

from __future__ import annotations

import argparse
import math


def validate_experiment_profile(args: argparse.Namespace) -> None:
    if args.experiment_profile == "current":
        return
    expected = {
        "S": ("dice", "none"),
        "ST": ("foreground_tversky", "none"),
        "STA": ("foreground_tversky", "mild"),
    }
    overlap_loss, color_augmentation = expected[args.experiment_profile]
    mismatches = []
    for name, actual, required in (
        ("sampling_mode", args.sampling_mode, "slide_stratified"),
        ("overlap_loss", args.overlap_loss, overlap_loss),
        ("color_augmentation", args.color_augmentation, color_augmentation),
    ):
        if actual != required:
            mismatches.append(f"{name}={actual!r}, expected {required!r}")
    for name, required in (
        ("sampling_positive_fraction", 0.60),
        ("sampling_boundary_positive_fraction", 0.50),
        ("sampling_slide_balance_power", 0.50),
        ("cross_entropy_weight", 1.0),
        ("dice_weight", 1.0),
        ("tumor_class_weight", 1.0),
    ):
        actual = float(getattr(args, name))
        if not math.isclose(actual, required, rel_tol=0.0, abs_tol=1e-12):
            mismatches.append(f"{name}={actual!r}, expected {required!r}")
    if args.sampling_max_patch_repeats != 2:
        mismatches.append(
            "sampling_max_patch_repeats="
            f"{args.sampling_max_patch_repeats!r}, expected 2"
        )
    if args.experiment_profile in {"ST", "STA"}:
        for name, required in (("tversky_alpha", 0.3), ("tversky_beta", 0.7)):
            actual = float(getattr(args, name))
            if not math.isclose(actual, required, rel_tol=0.0, abs_tol=1e-12):
                mismatches.append(f"{name}={actual!r}, expected {required!r}")
    if args.probability_metric_bins < 16:
        mismatches.append("probability_metric_bins must be at least 16")
    if mismatches:
        raise ValueError(
            f"Experiment profile {args.experiment_profile} is inconsistent: "
            + "; ".join(mismatches)
        )
