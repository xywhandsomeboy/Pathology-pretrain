"""Export one full-WSI Stage-2 context vector per graph node."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf
import torch

from dinov2.configs import dinov2_default_config
from dinov2.train.gcn_meta_arch import GCNMetaArch


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _model_state(payload):
    if not isinstance(payload, dict):
        raise TypeError("Stage-2 checkpoint must contain a state dictionary")
    state = payload.get("student", payload.get("model", payload))
    if not isinstance(state, dict):
        raise TypeError("Checkpoint student/model entry must be a state dictionary")
    for prefix in ("student.", "module."):
        if state and all(str(key).startswith(prefix) for key in state):
            state = {str(key)[len(prefix) :]: value for key, value in state.items()}
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.merge(
        OmegaConf.create(dinov2_default_config),
        OmegaConf.load(args.config_file),
    )
    device = torch.device(args.device)
    model = GCNMetaArch(cfg).to(device)
    incompatible = model.student.load_state_dict(_model_state(_load(args.checkpoint)), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Stage-2 checkpoint is incompatible with the current architecture: "
            f"missing={incompatible.missing_keys[:10]}, "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exported = skipped = 0
    for graph_path in sorted(args.graph_dir.glob("*.pt")):
        graph = _load(graph_path)
        missing = [name for name in ("patch_ids", "slide_id") if not hasattr(graph, name)]
        if missing:
            raise ValueError(f"{graph_path} lacks context-export metadata: {missing}")
        output_path = args.output_dir / f"{graph.slide_id}.pt"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        with torch.no_grad():
            context = model.extract_context(graph).detach().cpu()
        if len(context) != len(graph.patch_ids):
            raise ValueError(f"Context/node count mismatch for {graph_path}")
        temporary = output_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "format_version": 1,
                "slide_id": str(graph.slide_id),
                "patch_ids": list(map(str, graph.patch_ids)),
                "global_context": context,
                "graph_path": str(graph_path.resolve()),
            },
            temporary,
        )
        temporary.replace(output_path)
        exported += 1
    print(f"Exported {exported} context files; skipped {skipped} existing files")


if __name__ == "__main__":
    main()
