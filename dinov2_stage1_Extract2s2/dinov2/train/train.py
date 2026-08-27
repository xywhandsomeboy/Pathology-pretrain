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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, ShardedStateDictConfig, FullStateDictConfig
import torch

from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler

from dinov2.train.gcn_meta_arch import GCNMetaArch
import os
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
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.student["momentum"],
        final_value=cfg.student["final_momentum"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
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
def do_test(cfg, model):
    model.eval()
    inputs_dtype = torch.float
    # setup data preprocessing

    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    
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
    # sampler_type = SamplerType.INFINITE
    sampler_type = SamplerType.EPOCH
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        seed=42,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=False,
        collate_fn=collate_fn,
    )
    output_dir = os.path.join(cfg.train.output_dir,"embeddings")
    os.makedirs(output_dir,exist_ok=True)
    all_filenames = []
    all_outputs = []
    save_interval = 10  # 每多少个样本保存一次
    chunk_idx = 0
    print("dataset size", len(data_loader.dataset))
    num_batches = len(data_loader.dataset) // data_loader.batch_size//10
    print("num batches per epoch:", num_batches)
    for idx, data in enumerate(data_loader):
        filename = data["filenames"]
        outputs = model.forward_backward(data)
        print(outputs.shape)
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().numpy().tolist()
        all_filenames.extend(filename)
        all_outputs+=outputs
        # 每 save_interval 保存一次
        if (idx + 1) % save_interval == 0:

            np.savez_compressed(os.path.join(output_dir, f"filenames_pretrained_s1_part{chunk_idx}.npz"),
                    np.array(all_filenames))
            np.savez_compressed(os.path.join(output_dir, f"features_pretrained_s1_part{chunk_idx}.npz"),
                    np.array(all_outputs))

            print(f"Saved chunk {chunk_idx}, size={len(all_filenames)}")

            # 清空缓存，进入下一批
            chunk_idx += 1
            all_filenames, all_outputs = [], []
    if all_outputs:
        np.savez_compressed(os.path.join(output_dir, f"filenames_pretrained_s1_part{chunk_idx}.npz"),
                np.array(all_filenames))
        np.savez_compressed(os.path.join(output_dir, f"features_pretrained_s1_part{chunk_idx}.npz"),
                np.array(all_outputs))
        print(f"Saved final chunk {chunk_idx}, size={len(all_filenames)}")
        

def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    # setup optimizer

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)
    
    # checkpointer
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)

    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # setup data preprocessing

    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    
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
    # sampler_type = SamplerType.INFINITE
    sampler_type = SamplerType.RANDOM
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
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
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return

        features = model.forward_backward(data)



def main(args):
    cfg = setup(args)
    
    model = GCNMetaArch(cfg).to(torch.device("cuda"))
    checkpoint = torch.load(cfg.MODEL.WEIGHTS)['teacher']
    # print(checkpoint.keys())
    model.student.load_state_dict(checkpoint)

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

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
