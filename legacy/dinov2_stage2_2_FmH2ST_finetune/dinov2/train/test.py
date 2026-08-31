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
import torch.distributed as dist

from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler

from dinov2.train.gcn_meta_arch import GCNMetaArch

import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,   # ← 新增
)

torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("gnn")


def test_get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("GNN training", add_help=add_help)
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


# -------------------------
# Distributed helpers
# -------------------------
def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def gather_tensor(tensor):
    """
    DDP 下 all_gather；单卡原样返回
    注意：all_gather要求所有进程的tensor形状必须相同
    """
    if get_world_size() == 1:
        return tensor
    # 确保tensor是连续的
    tensor = tensor.contiguous()
    tensor_list = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=0)


def gather_variable_tensor(tensor):
    if get_world_size() == 1:
        return tensor

    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)

    tensor = tensor.contiguous()
    local_size = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
    size_list = [torch.zeros_like(local_size) for _ in range(get_world_size())]
    dist.all_gather(size_list, local_size)
    sizes = [int(item.item()) for item in size_list]
    max_size = max(sizes)

    if tensor.shape[0] < max_size:
        pad_shape = (max_size - tensor.shape[0],) + tuple(tensor.shape[1:])
        pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, pad_tensor], dim=0)

    gather_list = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(gather_list, tensor)
    return torch.cat([chunk[:size] for chunk, size in zip(gather_list, sizes)], dim=0)


# -------------------------
# Test function
# -------------------------
@torch.no_grad()
def do_test(cfg, model):
    model.eval()

    inputs_dtype = torch.float32
    collate_fn = partial(
        collate_data_and_cast,
        dtype=inputs_dtype,
    )

    # ===== Dataset & DataLoader =====
    # 测试时禁用随机子图采样，使用固定子图确保可重复性
    import os
    os.environ["USE_RANDOM_SUBGRAPH"] = "False"
    
    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        table=cfg.eval.table,
        jsonfile=cfg.train.jsonfile,
        transform=None,
        target_transform=lambda _: (),
    )
    
    # 确保测试时使用固定子图
    if hasattr(dataset, 'use_random_subgraph'):
        dataset.use_random_subgraph = False
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'use_random_subgraph'):
        dataset.dataset.use_random_subgraph = False

    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        seed=0,
        sampler_type=SamplerType.EPOCH,
        sampler_advance=0,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # ===== Containers =====
    all_preds = []
    all_targets = []

    # ===== tqdm (rank0 only) =====
    is_main = (get_rank() == 0)
    iterator = tqdm(
        data_loader,
        desc="Testing",
        ncols=100,
        leave=False,
        disable=not is_main,
    )

    # ===== Test Loop =====
    # 重要：所有进程必须同步执行循环，否则all_gather会死锁
    for batch_idx, data in enumerate(iterator):
        targets, outputs = model.test_forward_backward(data)
        # outputs: (B, C)
        # targets: (B,)

        preds = outputs.argmax(dim=1)
        
        # 确保tensor是连续的（all_gather的要求）
        preds = preds.contiguous()
        targets = targets.contiguous()

        # 重要：all_gather必须在所有进程上同时调用
        # gather_tensor会收集所有进程的数据并拼接
        preds_gathered = gather_variable_tensor(preds)
        targets_gathered = gather_variable_tensor(targets)

        # 只在rank0收集结果（因为gather_tensor已经收集了所有进程的数据）
        if is_main:
            all_preds.append(preds_gathered.cpu())
            all_targets.append(targets_gathered.cpu())

    # ===== Merge (rank0 only) =====
    if is_main:
        print("Start Merge...")
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
    else:
        all_preds = None
        all_targets = None
    
    # 同步所有进程
    if get_world_size() > 1:
        dist.barrier()

    # ===== Metrics (rank0 only) =====
    if is_main:
        acc = accuracy_score(all_targets, all_preds)
        precision = precision_score(
            all_targets, all_preds, average="macro", zero_division=0
        )
        recall = recall_score(
            all_targets, all_preds, average="macro", zero_division=0
        )
        f1 = f1_score(
            all_targets, all_preds, average="macro", zero_division=0
        )

        print("\n========== Overall Metrics (Macro) ==========")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")

        cm = confusion_matrix(all_targets, all_preds)
        print("\n========== Confusion Matrix ==========")
        print(cm)

        print("\n========== Per-class Metrics ==========")
        print(
            classification_report(
                all_targets,
                all_preds,
                digits=4,
                zero_division=0,
            )
        )

        print("GT class counts   :", np.bincount(all_targets))
        print("Pred class counts :", np.bincount(all_preds))
        
        # 诊断信息：数据集大小和类别分布
        print(f"\n========== Dataset Info ==========")
        print(f"Total test samples: {len(all_targets)}")
        print(f"Dataset path: {cfg.train.dataset_path}")
        print(f"Table file: {cfg.eval.table}")
        print(f"JSON file: {cfg.train.jsonfile}")
        
        # 计算类别分布
        unique, counts = np.unique(all_targets, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"Class distribution: {class_dist}")
        
        # 计算每个类别的准确率
        print(f"\n========== Per-Class Accuracy ==========")
        for class_id in unique:
            mask = (all_targets == class_id)
            class_acc = (all_preds[mask] == all_targets[mask]).sum() / mask.sum()
            print(f"Class {class_id}: {class_acc:.4f} ({counts[class_id]} samples)")

        return {
            "acc": acc,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "confusion_matrix": cm,
            "class_distribution": class_dist,
        }

    return None


def test_main(args):
    cfg = setup(args)
    with open(cfg.train.jsonfile, "r") as file:
        label_to_id = json.load(file)
        cfg.train.classes = len(label_to_id)
    
    model = GCNMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()
    
    logger.info("Model:\n{}".format(model))
    
    if args.eval_only:
    
        checkpointer = FSDPCheckpointer(
            model,
            save_dir=cfg.train.output_dir,
        )
    
        checkpoint = checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS,
            resume=not args.no_resume,
        )
    
        iteration = checkpoint.get("iteration", -1) + 1
        # print(checkpoint.keys())
    
        print("Evaluation...")
        return do_test(cfg, model)

if __name__ == "__main__":
    args = test_get_args_parser(add_help=True).parse_args()
    test_main(args)
