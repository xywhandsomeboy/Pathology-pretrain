# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
from functools import partial

from fvcore.common.checkpoint import PeriodicCheckpointer
import torch

from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import (
    collate_data_and_cast,
    DataAugmentationDINO,
    DataAugmentationStage1Extraction,
)
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler

from dinov2.train.gcn_meta_arch import GCNMetaArch
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    cosine_cycle_epochs = int(getattr(cfg.optim, "cosine_cycle_epochs", 0))
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
        cycle_iters=cosine_cycle_epochs * OFFICIAL_EPOCH_LENGTH,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


@torch.no_grad()
def _save_feature_chunk(output_dir, chunk_index, chunk, export_dense_tokens):
    rank = distributed.get_global_rank()
    rank_suffix = "" if distributed.get_global_size() == 1 else f"_rank{rank:03d}"
    stem = f"part{chunk_index:05d}{rank_suffix}"
    filenames = list(chunk["filenames"])
    node_features = torch.cat(chunk["node_features"], dim=0).contiguous()
    np.savez_compressed(
        os.path.join(output_dir, f"filenames_pretrained_s1_{stem}.npz"),
        np.asarray(filenames),
    )
    np.savez_compressed(
        os.path.join(output_dir, f"features_pretrained_s1_{stem}.npz"),
        node_features.numpy(),
    )
    if export_dense_tokens:
        payload = {
            "format_version": 1,
            "filenames": filenames,
            "patch_ids": list(chunk["patch_ids"]),
            "node_features": node_features,
            "dense_tokens": torch.cat(chunk["dense_tokens"], dim=0).contiguous(),
        }
        for key in ("coords", "levels"):
            if chunk[key]:
                payload[key] = torch.cat(chunk[key], dim=0).contiguous()
        if chunk["slide_ids"]:
            payload["slide_ids"] = list(chunk["slide_ids"])
        torch.save(
            payload,
            os.path.join(output_dir, f"decoder_features_s1_{stem}.pt"),
        )
    logger.info("Saved Stage-1 feature chunk %s with %d patches", stem, len(filenames))


def _empty_feature_chunk():
    return {
        "filenames": [],
        "patch_ids": [],
        "slide_ids": [],
        "coords": [],
        "levels": [],
        "node_features": [],
        "dense_tokens": [],
    }


@torch.no_grad()
def do_test(cfg, model):
    """Stage-1B deterministic extraction with rank-safe, aligned shards."""
    model.eval()
    data_transform = DataAugmentationStage1Extraction(
        image_size=cfg.crops.global_crops_size,
        local_size=cfg.crops.local_crops_size,
        num_local_crops=cfg.crops.local_crops_number,
    )

    collate_fn = partial(
        collate_data_and_cast,
        dtype=torch.float,
    )

    # setup data loader
    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        transform=data_transform,
        target_transform=lambda _: (),
    )
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        seed=42,  # TODO: Fix this -- cfg.train.seed
        sampler_type=SamplerType.EPOCH,
        sampler_advance=0,
        drop_last=False,
        collate_fn=collate_fn,
    )
    output_dir = os.path.join(cfg.train.output_dir, "embeddings")
    os.makedirs(output_dir, exist_ok=True)
    export_dense_tokens = bool(getattr(cfg.feature, "export_dense_tokens", False))
    chunk = _empty_feature_chunk()
    save_interval = 10
    chunk_idx = 0
    logger.info("Extracting %d patches in %d batches", len(data_loader.dataset), len(data_loader))
    for idx, data in enumerate(data_loader):
        if export_dense_tokens:
            decoder_features = model.extract_decoder_features(data)
            outputs = decoder_features["node_features"]
            chunk["dense_tokens"].append(
                decoder_features["dense_tokens"].detach().cpu().to(torch.float16)
            )
        else:
            outputs = model.forward_backward(data)
        outputs = outputs.detach().cpu().float()
        filenames = list(map(str, data["filenames"]))
        if len(outputs) != len(filenames):
            raise ValueError("Stage-1 outputs and filenames have different lengths")
        chunk["filenames"].extend(filenames)
        chunk["patch_ids"].extend(
            list(map(str, data.get("patch_ids", [os.path.splitext(os.path.basename(v))[0] for v in filenames])))
        )
        chunk["slide_ids"].extend(list(map(str, data.get("slide_ids", []))))
        for key in ("coords", "levels"):
            if key in data:
                chunk[key].append(torch.as_tensor(data[key]).cpu())
        chunk["node_features"].append(outputs)

        if (idx + 1) % save_interval == 0:
            _save_feature_chunk(output_dir, chunk_idx, chunk, export_dense_tokens)
            chunk_idx += 1
            chunk = _empty_feature_chunk()
    if chunk["node_features"]:
        _save_feature_chunk(output_dir, chunk_idx, chunk, export_dense_tokens)
        

def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    # setup optimizer

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)
    
    # checkpointer
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)

    # MODEL.WEIGHTS initializes the frozen DINO before FSDP wrapping. Resume
    # only from checkpoints produced in this Stage-1 output directory.
    start_iter = checkpointer.resume_or_load("", resume=resume).get("iteration", -1) + 1

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    data_transform = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )

    collate_fn = partial(
        collate_data_and_cast,
        dtype=inputs_dtype,
    )

    # setup data loader

    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        transform=data_transform,
        target_transform=lambda _: (),
    )
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=cfg.train.seed + start_iter,
        sampler_type=sampler_type,
        sampler_advance=0,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        if iteration >= max_iter:
            return

        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_pretrain(data)

        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for module in model.student.values():
                    module.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for module in model.student.values():
                    module.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        if distributed.get_global_size() > 1:
            for value in loss_dict.values():
                torch.distributed.all_reduce(value)
        reduced = {
            key: value.item() / distributed.get_global_size()
            for key, value in loss_dict.items()
        }
        if not math.isfinite(reduced["total_loss"]):
            raise FloatingPointError(f"Non-finite Stage-1 loss: {reduced}")
        metric_logger.update(lr=lr, wd=wd, **reduced)
        periodic_checkpointer.step(iteration)
        iteration += 1

    metric_logger.synchronize_between_processes()
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}



def main(args):
    cfg = setup(args)

    if not cfg.MODEL.WEIGHTS:
        raise ValueError(
            "MODEL.WEIGHTS is required. Pass it on the command line or set "
            "DINO_WEIGHTS/STAGE1_WEIGHTS in the maintained launch scripts."
        )
    model = GCNMetaArch(cfg).to(torch.device("cuda"))
    checkpoint = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
    checkpoint = checkpoint.get(
        "student", checkpoint.get("teacher", checkpoint.get("model", checkpoint))
    )
    if checkpoint and all(key.startswith("student.") for key in checkpoint):
        checkpoint = {key.removeprefix("student."): value for key, value in checkpoint.items()}
    # print(checkpoint.keys())
    incompatible = model.student.load_state_dict(checkpoint, strict=False)
    backbone_missing = [
        key for key in incompatible.missing_keys if key.startswith("backbone.")
    ]
    if backbone_missing:
        raise RuntimeError(
            f"DINO checkpoint is incompatible: {len(backbone_missing)} "
            "backbone tensors are missing."
        )
    spatial_missing = [
        key for key in incompatible.missing_keys
        if key.startswith((
            "spatial_agg.",
            "local_spatial_agg.",
            "local_crop_fusion.",
            "node_fusion.",
        ))
    ]
    if spatial_missing:
        message = (
            f"Checkpoint has no trained spatial feature branch "
            f"({len(spatial_missing)} missing tensors). Train it in "
            "Stage-1A or load a compatible checkpoint before producing "
            "final Stage-1 features."
        )
        if args.eval_only and cfg.feature.require_trained_spatial:
            raise RuntimeError(message)
        if args.eval_only:
            logger.warning("%s Continuing only because feature.require_trained_spatial=false.", message)
        else:
            logger.info("%s This is expected for a new Stage-1A run.", message)

    # model.prepare_for_distributed_training()
    
    # basedir=os.path.dirname(cfg.MODEL.WEIGHTS)
    # checkpointer = FSDPCheckpointer(model, save_dir=basedir)
    # # 指定 epoch 10 的 checkpoint（只需要提供文件夹或 rank0 文件即可，FSDP 会自动找到其他 rank 的分片）
    # checkpointer.load(cfg.MODEL.WEIGHTS)
    # # 保存完整模型
    # torch.save(model.state_dict(), cfg.MODEL.WEIGHTS.replace(".rank_0",""))
    # model.load_state_dict(cfg.MODEL.WEIGHTS.replace(".rank_0",""))

    logger.info("Model:\n{}".format(model))
    if args.eval_only:

        # iteration = (
        #     FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
        #     .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
        #     .get("iteration", -1)
        #     + 1
        # )
        
        print("Evaluation...")
        return do_test(cfg, model)

    model.prepare_for_distributed_training()
    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
