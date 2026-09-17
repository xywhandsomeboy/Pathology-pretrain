"""Asynchronously validate immutable snapshots of joint-training progress weights.

The training process owns ``checkpoint_progress.pt`` and replaces it atomically.
This module never writes into the trainer's history or official best-checkpoint
files.  A lightweight capture thread hard-links every observed generation into
an immutable spool, while the main thread validates those snapshots one by one.

Intermediate validation deliberately uses a fixed deterministic subset.  It is
for comparing progress checkpoints and detecting regressions; the trainer's
full epoch-end validation remains authoritative for ``checkpoint_best.pt``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Iterable


SCHEMA_VERSION = 1
DEFAULT_SUBSET_SIZE = 50_000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _fingerprint_token(fingerprint: dict) -> str:
    encoded = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def _processed_marker(run_dir: Path, token: str) -> Path:
    return run_dir / "async_validation" / "processed" / f"{token}.json"


def _category(row: dict, interior_threshold: float) -> str:
    has_tumor = int(row["has_tumor"])
    fraction = float(row["tumor_fraction"])
    if has_tumor == 0:
        if fraction != 0.0:
            raise ValueError("has_tumor=0 requires tumor_fraction=0")
        return "negative"
    if fraction <= 0.0:
        raise ValueError("has_tumor=1 requires tumor_fraction>0")
    return "interior" if fraction >= interior_threshold else "boundary"


def _allocate_proportional_quotas(
    counts: dict[tuple[str, str], int], target_size: int
) -> dict[tuple[str, str], int]:
    total = sum(counts.values())
    if not 0 < target_size <= total:
        raise ValueError(f"subset size must be in [1,{total}], got {target_size}")
    exact = {
        key: target_size * count / total
        for key, count in counts.items()
    }
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = target_size - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda key: (-(exact[key] - quotas[key]), key[0], key[1]),
    )
    for key in order[:remaining]:
        quotas[key] += 1
    if sum(quotas.values()) != target_size:
        raise RuntimeError("stratified quota allocation produced the wrong size")
    if any(quotas[key] > counts[key] for key in counts):
        raise RuntimeError("stratified quota exceeds its source population")
    return quotas


def _stable_score(seed: int, slide_id: str, patch_id: str) -> int:
    payload = f"{seed}\0{slide_id}\0{patch_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def build_stratified_monitor_manifest(
    source: Path,
    destination: Path,
    *,
    subset_size: int = DEFAULT_SUBSET_SIZE,
    seed: int = 42,
    interior_threshold: float = 0.999999,
) -> dict:
    """Create an exact-size, deterministic slide/category-proportional subset."""

    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    source_fingerprint = _fingerprint(source)
    metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "source_fingerprint": source_fingerprint,
        "subset_size": int(subset_size),
        "seed": int(seed),
        "interior_threshold": float(interior_threshold),
    }
    if destination.is_file() and metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in expected.items()):
            if existing.get("manifest_sha256") == _sha256(destination):
                return existing

    counts: Counter[tuple[str, str]] = Counter()
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        required = {"slide_id", "patch_id", "has_tumor", "tumor_fraction"}
        if fieldnames is None or not required.issubset(fieldnames):
            raise ValueError(
                f"validation manifest lacks fields: {sorted(required - set(fieldnames or []))}"
            )
        for row in reader:
            counts[(row["slide_id"], _category(row, interior_threshold))] += 1
    quotas = _allocate_proportional_quotas(dict(counts), int(subset_size))

    # Each heap retains the rows with the smallest stable hashes in one stratum.
    heaps: dict[tuple[str, str], list[tuple[int, int, int, dict]]] = defaultdict(list)
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        for row_index, row in enumerate(reader):
            key = (row["slide_id"], _category(row, interior_threshold))
            quota = quotas[key]
            if quota == 0:
                continue
            score = _stable_score(seed, row["slide_id"], row["patch_id"])
            item = (-score, -row_index, row_index, dict(row))
            heap = heaps[key]
            if len(heap) < quota:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)

    selected = sorted(
        (item[2], item[3])
        for heap in heaps.values()
        for item in heap
    )
    if len(selected) != subset_size:
        raise RuntimeError(
            f"selected {len(selected)} validation rows, expected {subset_size}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row for _, row in selected)
    os.replace(temporary, destination)

    selected_counts = Counter(
        (row["slide_id"], _category(row, interior_threshold))
        for _, row in selected
    )
    metadata = {
        **expected,
        "created_at": _utc_now(),
        "source_rows": sum(counts.values()),
        "strata": [
            {
                "slide_id": key[0],
                "category": key[1],
                "source_count": counts[key],
                "selected_count": selected_counts[key],
            }
            for key in sorted(counts)
        ],
        "manifest": str(destination),
        "manifest_sha256": _sha256(destination),
    }
    _atomic_json(metadata, metadata_path)
    return metadata


def capture_progress_checkpoint(source: Path) -> Path | None:
    """Hard-link one atomic checkpoint generation into its validation spool."""

    source = source.expanduser().resolve()
    if not source.is_file():
        return None
    run_dir = source.parent
    spool = run_dir / "async_validation" / "spool"
    spool.mkdir(parents=True, exist_ok=True)
    temporary = spool / f".capture-{os.getpid()}-{time.time_ns()}.pt"
    try:
        os.link(source, temporary)
    except FileNotFoundError:
        return None
    fingerprint = _fingerprint(temporary)
    token = _fingerprint_token(fingerprint)
    destination = spool / f"checkpoint_{token}.pt"
    if destination.exists() or _processed_marker(run_dir, token).is_file():
        temporary.unlink()
        return None
    os.replace(temporary, destination)
    _atomic_json(
        {
            "schema_version": SCHEMA_VERSION,
            "captured_at": _utc_now(),
            "source": str(source),
            "snapshot": str(destination),
            "fingerprint": fingerprint,
            "token": token,
        },
        destination.with_suffix(".capture.json"),
    )
    return destination


def _source_path(run_manifest: dict, name: str) -> Path:
    direct = run_manifest.get(name)
    if isinstance(direct, str) and direct:
        return Path(direct).expanduser().resolve()
    execution = run_manifest.get("execution", {})
    source = execution.get("sources", {}).get(name, {}) if isinstance(execution, dict) else {}
    if isinstance(source, dict) and source.get("path"):
        return Path(source["path"]).expanduser().resolve()
    raise ValueError(f"checkpoint run_manifest does not identify {name}")


def _validation_args(configuration: dict, manifest: Path, workers: int) -> SimpleNamespace:
    return SimpleNamespace(
        val_manifest=manifest,
        image_size=int(configuration.get("image_size", 224)),
        color_augmentation="none",
        workers=int(workers),
        batch_size=int(configuration["batch_size"]),
        seed=int(configuration.get("seed", 42)),
        max_train_batches=0,
        max_val_batches=0,
        num_classes=int(configuration.get("num_classes", 2)),
        overlap_loss=str(configuration.get("overlap_loss", "dice")),
        probability_metric_bins=int(configuration.get("probability_metric_bins", 0)),
        amp_dtype=str(configuration.get("amp_dtype", "bf16")),
        ignore_index=int(configuration.get("ignore_index", 255)),
        cross_entropy_weight=float(configuration.get("cross_entropy_weight", 1.0)),
        dice_weight=float(configuration.get("dice_weight", 1.0)),
        tumor_class_weight=float(configuration.get("tumor_class_weight", 1.0)),
        tversky_alpha=float(configuration.get("tversky_alpha", 0.3)),
        tversky_beta=float(configuration.get("tversky_beta", 0.7)),
        log_interval=0,
    )


def _wait_for_free_cuda_memory(minimum_gib: float, poll_seconds: int) -> None:
    if minimum_gib <= 0:
        return
    import torch

    threshold = int(minimum_gib * 1024**3)
    while True:
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes >= threshold:
            return
        print(
            json.dumps(
                {
                    "async_validation_waiting_for_memory": True,
                    "free_gib": free_bytes / 1024**3,
                    "required_gib": minimum_gib,
                }
            ),
            flush=True,
        )
        time.sleep(poll_seconds)


def evaluate_snapshot(
    snapshot: Path,
    *,
    protocol_root: Path,
    subset_size: int,
    subset_seed: int,
    workers: int,
    device_name: str,
    probability_metrics: bool,
) -> tuple[dict, Path]:
    """Evaluate one immutable checkpoint and return its result and subset path."""

    import torch

    from dinov2_segmentation import train_joint
    from dinov2_segmentation.joint_graph import JointGraphRepository
    from dinov2_segmentation.joint_model import JointSegmentationSystem

    started_at = _utc_now()
    started = time.monotonic()
    checkpoint_sha256 = _sha256(snapshot)
    checkpoint = train_joint._load(snapshot)
    if int(checkpoint.get("format_version", -1)) != 2:
        raise ValueError("asynchronous validation requires format_version=2")
    if bool(checkpoint.get("epoch_complete", True)):
        raise ValueError("asynchronous progress validation expects epoch_complete=False")
    configuration = checkpoint.get("configuration")
    run_manifest = checkpoint.get("run_manifest")
    if not isinstance(configuration, dict) or not isinstance(run_manifest, dict):
        raise ValueError("checkpoint lacks configuration or run_manifest")
    val_manifest = _source_path(run_manifest, "val_manifest")
    graph_dir = _source_path(run_manifest, "graph_dir")
    stage1_config = _source_path(run_manifest, "stage1_config")
    stage1_checkpoint = _source_path(run_manifest, "stage1_checkpoint")
    stage2_config = _source_path(run_manifest, "stage2_config")
    stage2_checkpoint = _source_path(run_manifest, "stage2_checkpoint")
    for path in (
        val_manifest,
        graph_dir,
        stage1_config,
        stage1_checkpoint,
        stage2_config,
        stage2_checkpoint,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    protocol_root.mkdir(parents=True, exist_ok=True)
    subset_manifest = protocol_root / (
        f"valid_monitor_n{subset_size}_seed{subset_seed}.csv"
    )
    subset_metadata = build_stratified_monitor_manifest(
        val_manifest,
        subset_manifest,
        subset_size=subset_size,
        seed=subset_seed,
        interior_threshold=float(
            configuration.get("sampling_interior_threshold", 0.999999)
        ),
    )
    args = _validation_args(configuration, subset_manifest, workers)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation requested but CUDA is unavailable")
    torch.set_float32_matmul_precision("high")
    train_joint._set_seed(args.seed)
    loader = train_joint._loader(subset_manifest, args, training=False, execution=None)
    system = JointSegmentationSystem(
        decoder_version=str(configuration["decoder_version"]),
        stage1_config=stage1_config,
        stage1_checkpoint=stage1_checkpoint,
        stage2_config=stage2_config,
        stage2_checkpoint=stage2_checkpoint,
        num_classes=args.num_classes,
        decoder_drop_path_rate=float(configuration.get("decoder_drop_path_rate", 0.1)),
    ).to(device)
    if checkpoint.get("model_version") != system.model_version:
        raise ValueError("checkpoint model_version differs from reconstructed model")
    system.load_state_dict(checkpoint["model"], strict=True)
    graph_repository = JointGraphRepository(
        graph_dir,
        expected_edge_mode=system.stage2_runtime.context_edge_mode,
    )
    from dinov2_segmentation.consistent_features import configure_repository
    configure_repository(graph_repository, system, configuration,
                         step=int(checkpoint.get("curriculum_step", 0)))
    with torch.no_grad():
        metrics = train_joint._run_epoch(
            system,
            graph_repository,
            loader,
            device,
            args,
            collect_probability_metrics=probability_metrics,
        )
    protocol = {
        "scope": "fixed_stratified_monitor_subset",
        "official_model_selection": False,
        "source_validation_manifest": str(val_manifest),
        "source_validation_fingerprint": subset_metadata["source_fingerprint"],
        "subset_manifest": str(subset_manifest),
        "subset_size": subset_size,
        "subset_seed": subset_seed,
        "subset_sha256": subset_metadata["manifest_sha256"],
        "batch_size": args.batch_size,
        "amp_dtype": args.amp_dtype,
        "probability_metrics": bool(probability_metrics),
        "graph_dir": str(graph_dir),
        "loss": {
            "overlap_loss": args.overlap_loss,
            "cross_entropy_weight": args.cross_entropy_weight,
            "overlap_weight": args.dice_weight,
            "tumor_class_weight": args.tumor_class_weight,
            "tversky_alpha": args.tversky_alpha,
            "tversky_beta": args.tversky_beta,
        },
    }
    if configuration.get("graph_feature_policy", "legacy") != "legacy":
        protocol["graph_feature_policy"] = configuration["graph_feature_policy"]
        protocol["graph_view"] = "canonical_all_receptive_field_nodes_v1"
    protocol_id = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "duration_seconds": time.monotonic() - started,
        "checkpoint": {
            "snapshot": str(snapshot),
            "fingerprint": _fingerprint(snapshot),
            "sha256": checkpoint_sha256,
            "format_version": int(checkpoint["format_version"]),
            "model_version": checkpoint["model_version"],
            "epoch": int(checkpoint["epoch"]),
            "epoch_complete": False,
            "next_batch_index": int(checkpoint["next_batch_index"]),
            "curriculum_step": int(checkpoint["curriculum_step"]),
            "training_phase": checkpoint.get("training_phase"),
        },
        "profile": configuration.get("experiment_profile"),
        "decoder_version": configuration.get("decoder_version"),
        "protocol_id": protocol_id,
        "protocol": protocol,
        "metrics": metrics,
    }
    # Release the extra model between checkpoints so the idle watcher consumes
    # no training GPU memory.
    del graph_repository, system, loader, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, subset_manifest


def _result_name(result: dict, token: str) -> str:
    checkpoint = result["checkpoint"]
    return (
        f"step_{checkpoint['curriculum_step']:09d}_"
        f"epoch_{checkpoint['epoch']:03d}_batch_{checkpoint['next_batch_index']:06d}_"
        f"{token}.json"
    )


def _publish_result(run_dir: Path, snapshot: Path, result: dict) -> Path:
    validation_root = run_dir / "async_validation"
    token = _fingerprint_token(result["checkpoint"]["fingerprint"])
    result_path = validation_root / "results" / _result_name(result, token)
    _atomic_json(result, result_path)
    _atomic_json(result, validation_root / "latest.json")

    best_path = validation_root / "best.json"
    current_best = None
    if best_path.is_file():
        current_best = json.loads(best_path.read_text(encoding="utf-8"))
    comparable = (
        current_best is None
        or current_best.get("protocol_id") == result.get("protocol_id")
    )
    improved = bool(
        comparable
        and (
            current_best is None
            or float(result["metrics"]["tumor_dice"])
            > float(current_best["metrics"]["tumor_dice"])
        )
    )
    if improved:
        best_checkpoint = validation_root / "checkpoint_best_monitor.pt"
        temporary = validation_root / (
            f".checkpoint_best_monitor.pt.tmp-{os.getpid()}-{time.time_ns()}"
        )
        os.link(snapshot, temporary)
        os.replace(temporary, best_checkpoint)
        best_result = dict(result)
        best_result["retained_checkpoint"] = str(best_checkpoint)
        _atomic_json(best_result, best_path)
    _atomic_json(
        {
            "schema_version": SCHEMA_VERSION,
            "processed_at": _utc_now(),
            "token": token,
            "result": str(result_path),
            "checkpoint_sha256": result["checkpoint"]["sha256"],
        },
        _processed_marker(run_dir, token),
    )
    return result_path


def _discover_progress(root: Path) -> Iterable[Path]:
    for path in root.rglob("checkpoint_progress.pt"):
        if "async_validation" not in path.parts and path.is_file():
            # This trainer enqueues selected generations synchronously. Polling
            # its rolling file would wrongly validate the intermediate 1k saves.
            if (path.parent / 'checkpoint_queue_writer.json').is_file():
                continue
            yield path


def _discover_spool(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("async_validation/spool/checkpoint_*.pt")
            if path.is_file()
        ),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )


def _discard_processed_snapshot(snapshot: Path) -> bool:
    """Skip a completed generation left queued across a validator restart.

    Publication persists the processed marker before removing the spool link.
    A crash in between must not trigger another GPU evaluation on restart.
    """
    token = _fingerprint_token(_fingerprint(snapshot))
    if not _processed_marker(snapshot.parents[2], token).is_file():
        return False
    snapshot.unlink()
    snapshot.with_suffix(".capture.json").unlink(missing_ok=True)
    snapshot.with_suffix(".failure.json").unlink(missing_ok=True)
    return True


def _capture_loop(root: Path, poll_seconds: int, stop: threading.Event) -> None:
    while not stop.is_set():
        for source in _discover_progress(root):
            try:
                captured = capture_progress_checkpoint(source)
                if captured is not None:
                    print(
                        json.dumps({"captured_progress_checkpoint": str(captured)}),
                        flush=True,
                    )
            except Exception as error:  # keep capture alive; evaluator reports details
                print(
                    json.dumps(
                        {
                            "capture_error": str(error),
                            "source": str(source),
                        }
                    ),
                    flush=True,
                )
        stop.wait(poll_seconds)


def watch(args: argparse.Namespace) -> None:
    root = args.watch_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".async_validation.lock"
    lock_stream = lock_path.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"another async validator owns {lock_path}") from error

    protocol_root = root / ".async_validation_protocol"
    stop = threading.Event()
    capture_thread = threading.Thread(
        target=_capture_loop,
        args=(root, args.poll_seconds, stop),
        name="checkpoint-capture",
        daemon=True,
    )
    capture_thread.start()
    print(
        json.dumps(
            {
                "async_validation_started": _utc_now(),
                "watch_root": str(root),
                "device": args.device,
                "subset_size": args.subset_size,
                "official_best_unchanged": True,
            }
        ),
        flush=True,
    )
    try:
        while True:
            snapshots = _discover_spool(root)
            if not snapshots:
                if args.once:
                    return
                time.sleep(args.poll_seconds)
                continue
            snapshot = snapshots[0]
            run_dir = snapshot.parents[2]
            try:
                if _discard_processed_snapshot(snapshot):
                    print(
                        json.dumps({"skipped_processed_checkpoint": str(snapshot)}),
                        flush=True,
                    )
                    continue
                if args.device.startswith("cuda"):
                    _wait_for_free_cuda_memory(
                        args.min_free_memory_gib, args.poll_seconds
                    )
                result, _ = evaluate_snapshot(
                    snapshot,
                    protocol_root=protocol_root,
                    subset_size=args.subset_size,
                    subset_seed=args.subset_seed,
                    workers=args.workers,
                    device_name=args.device,
                    probability_metrics=args.probability_metrics,
                )
                result_path = _publish_result(run_dir, snapshot, result)
                print(
                    json.dumps(
                        {
                            "async_validation_complete": str(result_path),
                            "curriculum_step": result["checkpoint"]["curriculum_step"],
                            "tumor_dice": result["metrics"]["tumor_dice"],
                            "tumor_precision": result["metrics"]["tumor_precision"],
                            "tumor_recall": result["metrics"]["tumor_recall"],
                        }
                    ),
                    flush=True,
                )
                snapshot.unlink()
                capture_metadata = snapshot.with_suffix(".capture.json")
                capture_metadata.unlink(missing_ok=True)
            except Exception as error:
                failure = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "failed_at": _utc_now(),
                    "snapshot": str(snapshot),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                _atomic_json(failure, snapshot.with_suffix(".failure.json"))
                print(json.dumps({"async_validation_error": failure}), flush=True)
                if args.once:
                    raise
                time.sleep(args.retry_seconds)
    finally:
        stop.set()
        capture_thread.join(timeout=max(1, args.poll_seconds + 1))
        fcntl.flock(lock_stream, fcntl.LOCK_UN)
        lock_stream.close()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--subset-size", type=int, default=DEFAULT_SUBSET_SIZE)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--retry-seconds", type=int, default=300)
    parser.add_argument("--min-free-memory-gib", type=float, default=24.0)
    parser.add_argument("--probability-metrics", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    if args.subset_size < 1:
        raise ValueError("subset-size must be positive")
    if args.poll_seconds < 1 or args.retry_seconds < 1:
        raise ValueError("poll and retry intervals must be positive")
    if args.min_free_memory_gib < 0:
        raise ValueError("min-free-memory-gib must be non-negative")
    return args


def main(argv=None) -> None:
    watch(parse_args(argv))


if __name__ == "__main__":
    main()
