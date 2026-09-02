"""Infer and stitch WSI masks with the independent Decoder V2."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dinov2_segmentation.data import PatchSegmentationDataset
from dinov2_segmentation.models.model_v2 import GlobalLocalSegmentationModelV2
from dinov2_segmentation.stitching import DiskBackedSlideStitcher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _slide_extents(dataset):
    extents = defaultdict(lambda: [0, 0])
    for row in dataset.rows:
        with Image.open(row["image_path"]) as image:
            width, height = image.size
        key = (row["slide_id"], row["level"])
        extents[key][0] = max(extents[key][0], row["x"] + width)
        extents[key][1] = max(extents[key][1], row["y"] + height)
    return extents


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    expected_version = GlobalLocalSegmentationModelV2.model_version
    if checkpoint.get("model_version") != expected_version:
        raise ValueError(
            "infer_v2 only accepts V2 checkpoints; got "
            f"model_version={checkpoint.get('model_version')!r}"
        )
    model = GlobalLocalSegmentationModelV2(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    image_size = int(checkpoint.get("image_size", 224))
    dataset = PatchSegmentationDataset(
        args.manifest,
        image_size=image_size,
        training=False,
        require_mask=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    num_classes = checkpoint["model_config"]["num_classes"]
    stitchers = {
        key: DiskBackedSlideStitcher(
            args.output_dir,
            slide_id=key[0],
            level=key[1],
            num_classes=num_classes,
            width=extent[0],
            height=extent[1],
        )
        for key, extent in _slide_extents(dataset).items()
    }

    amp_enabled = device.type == "cuda" and not args.no_amp
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=amp_enabled,
            ):
                logits = model(
                    batch["image"].to(device, non_blocking=True),
                    batch["dense_tokens"].to(device, non_blocking=True),
                    batch["global_context"].to(device, non_blocking=True),
                )
                probability = logits.softmax(dim=1)
            for index in range(probability.shape[0]):
                height, width = map(
                    int, batch["original_size"][index].tolist()
                )
                patch_probability = F.interpolate(
                    probability[index : index + 1],
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[0].float().cpu().numpy()
                x, y = map(int, batch["coords"][index].tolist())
                key = (
                    batch["slide_id"][index],
                    int(batch["level"][index]),
                )
                stitchers[key].add(patch_probability, x, y)
    for key, stitcher in stitchers.items():
        path = stitcher.finalize()
        print(f"Saved {key[0]} level {key[1]}: {path}")


if __name__ == "__main__":
    main()
