# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
from functools import partial
from pathlib import Path

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
from collections import Counter
from torch import nn
import torch.distributed as dist

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


def get_args_parser(add_help: bool = True):
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
    """
    Gather tensors whose first dimension may differ across ranks.
    This keeps evaluation complete without forcing drop_last=True.
    """
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


def _training_iteration_from_path(path: Path):
    try:
        return int(path.parent.name.replace("training_", ""))
    except ValueError:
        return -1


def resolve_pretrained_checkpoint(pretrained_path: str) -> str:
    path = Path(pretrained_path)
    if path.exists():
        return str(path)

    parts = path.parts
    if "eval" in parts:
        eval_index = parts.index("eval")
        run_dir = Path(*parts[:eval_index])
        eval_dir = run_dir / "eval"
        candidates = sorted(
            eval_dir.glob("training_*/student_checkpoint.pth"),
            key=_training_iteration_from_path,
        )
        if candidates:
            fallback = candidates[-1]
            logger.warning(
                "Pretrained checkpoint %s not found, falling back to latest available eval checkpoint %s",
                pretrained_path,
                fallback,
            )
            return str(fallback)

    raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")


def load_student_pretrained_weights(model, pretrained_path: str):
    resolved_path = resolve_pretrained_checkpoint(pretrained_path)
    checkpoint = torch.load(resolved_path, map_location="cpu")
    state_dict = checkpoint.get("student", checkpoint)
    if all(key.startswith("student.") for key in state_dict.keys()):
        state_dict = {key.removeprefix("student."): value for key, value in state_dict.items()}

    missing, unexpected = model.student.load_state_dict(state_dict, strict=False)
    logger.info("Loaded pretrained weights from %s", resolved_path)
    logger.info("Missing keys: %s", missing)
    logger.info("Unexpected keys: %s", unexpected)
    return resolved_path, missing, unexpected


def iter_gnn_layers(gnn_backbone):
    for module in gnn_backbone.gnns:
        if isinstance(module, torch.nn.ModuleList):
            for layer in module:
                yield layer
        else:
            yield module


def freeze_gnn_layers(model, num_layers_to_freeze: int):
    if num_layers_to_freeze <= 0:
        return

    layers = list(iter_gnn_layers(model.student["gcn"]))
    num_layers_to_freeze = min(num_layers_to_freeze, len(layers))
    for layer in layers[:num_layers_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False

    logger.info("Froze %d/%d GNN layers", num_layers_to_freeze, len(layers))


# -------------------------
# Test function
# -------------------------
@torch.no_grad()
def do_eval(cfg, model):
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
        augment=False,
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

def do_test(cfg, model, iteration):
    new_state_dict = model.student.state_dict()

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        student_ckp_path = os.path.join(eval_dir, "student_checkpoint.pth")
        torch.save({"student": new_state_dict}, student_ckp_path)

def do_train(cfg, model, dataset, resume=False):
    model.train()
    inputs_dtype = torch.float32
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
        max_to_keep=5,
    )

    # setup data preprocessing

    collate_fn = partial(
        collate_data_and_cast,
        dtype=inputs_dtype,
    )

    # setup data loader
    # 训练时启用随机子图采样（数据增强）
    os.environ["USE_RANDOM_SUBGRAPH"] = "True"
    
    # 确保训练时使用随机子图（数据增强）
    if hasattr(dataset, 'use_random_subgraph'):
        dataset.use_random_subgraph = True
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'use_random_subgraph'):
        dataset.dataset.use_random_subgraph = True
    
    # 选择采样器类型
    # 如果配置中启用了加权采样，使用加权采样器
    use_weighted_sampling = getattr(cfg.train, 'use_weighted_sampling', False)
    weighted_mode = getattr(cfg.train, 'weighted_sampling_mode', 'balanced')
    
    if use_weighted_sampling:
        sampler_type = SamplerType.WEIGHTED_SHARDED_INFINITE
        logger.info(f"Using weighted sampling with mode: {weighted_mode}")
        # 获取类别权重（如果数据集支持）
        sampler_weights = None
        if hasattr(dataset, 'get_class_weights'):
            sampler_weights = dataset.get_class_weights(mode=weighted_mode)
        else:
            logger.warning("Dataset doesn't support get_class_weights(), falling back to regular sampling")
            sampler_type = SamplerType.SHARDED_INFINITE
    else:
        sampler_type = SamplerType.SHARDED_INFINITE
        sampler_weights = None
    
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
        sampler_weights=sampler_weights,
    )

    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"
    total_correct = 0
    total_samples = 0
    
    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
#         current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return

        # apply schedules

        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses

        optimizer.zero_grad(set_to_none=True)
        loss_dict,batch_correct,batch_total = model.forward_backward(data)

        batch_accuracy = loss_dict.get("accuracy", 0)

        batch_stats = torch.tensor(
            [batch_correct, batch_total],
            device=loss_dict["total_loss"].device,
            dtype=torch.float32,
        )
        if distributed.get_global_size() > 1:
            torch.distributed.all_reduce(batch_stats)

        total_correct += int(batch_stats[0].item())
        total_samples += int(batch_stats[1].item())

        global_accuracy = total_correct / total_samples if total_samples > 0 else 0
        # clip gradients

        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    # v.clip_grad_norm_(cfg.optim.clip_grad)
                    params = [p for p in v.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(params, cfg.optim.clip_grad)

            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    # v.clip_grad_norm_(cfg.optim.clip_grad)
                    params = [p for p in v.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(params, cfg.optim.clip_grad)
            optimizer.step()

        # logging
        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        loss_dict_reduced["global_accuracy"] = global_accuracy

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError
        # losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(**loss_dict_reduced)
#         metric_logger.update(current_batch_size=current_batch_size)
        # metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        # checkpointing and testing

        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            do_eval(cfg, model)
            model.train()
            torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)
    with open(cfg.train.jsonfile, "r") as file:
        label_to_id = json.load(file)
        cfg.train.classes = len(label_to_id)
    
    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        table=cfg.train.table,
        jsonfile=cfg.train.jsonfile,
        transform=None,
        target_transform=lambda _: (),
        augment=True,
    )

    targets = dataset.labels  # List[int]，每个图像对应的类别编号
    class_counts = Counter(targets)
    counts = np.array([class_counts.get(i, 0) for i in range(cfg.train.classes)], dtype=np.int64)
    nonzero_mask = counts > 0
    if not np.any(nonzero_mask):
        raise RuntimeError("No valid training samples were found after dataset filtering.")

    empirical_prior = np.zeros(cfg.train.classes, dtype=np.float32)
    empirical_prior[nonzero_mask] = counts[nonzero_mask] / counts[nonzero_mask].sum()
    if getattr(cfg.train, "use_weighted_sampling", False):
        target_prior = np.full(cfg.train.classes, 1.0 / cfg.train.classes, dtype=np.float32)
    else:
        target_prior = empirical_prior.copy()

    if not np.all(nonzero_mask):
        missing_classes = np.where(~nonzero_mask)[0].tolist()
        logger.warning("Classes missing from training split, setting their loss weight to 0: %s", missing_classes)

    use_class_weighted_loss = bool(getattr(cfg.train, "use_class_weighted_loss", False))
    if getattr(cfg.train, "use_weighted_sampling", False) and use_class_weighted_loss:
        logger.warning("Weighted sampler and class-weighted loss were both enabled; disabling class-weighted loss to avoid double reweighting.")
        use_class_weighted_loss = False

    if use_class_weighted_loss:
        weights = np.zeros(cfg.train.classes, dtype=np.float32)
        weights[nonzero_mask] = counts[nonzero_mask].sum() / (nonzero_mask.sum() * counts[nonzero_mask])
        weights_tensor = torch.tensor(weights, dtype=torch.float32).cuda()
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    class_prior_tensor = torch.tensor(target_prior, dtype=torch.float32)
    model = GCNMetaArch(cfg, criterion, class_prior=class_prior_tensor).to(torch.device("cuda"))
    if cfg.gcn.pretrained_weights:
        load_student_pretrained_weights(model, cfg.gcn.pretrained_weights)

    freeze_layers = int(getattr(cfg.train, "freeze_gnn_layers", 0))
    freeze_gnn_layers(model, freeze_layers)

    model.prepare_for_distributed_training()

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    do_train(cfg, model, dataset, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
