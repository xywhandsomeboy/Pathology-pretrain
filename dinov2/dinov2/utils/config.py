# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import math
import logging
import os

from omegaconf import OmegaConf

import dinov2.distributed as distributed
from dinov2.logging import setup_logging
from dinov2.utils import utils
from dinov2.configs import dinov2_default_config


logger = logging.getLogger("dinov2")


def apply_scaling_rules_to_cfg(cfg):  # to fix
    if cfg.optim.scaling_rule == "sqrt_wrt_1024":
        base_lr = cfg.optim.base_lr
        cfg.optim.lr = base_lr
        cfg.optim.lr *= math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_global_size() / 1024.0)
        logger.info(f"sqrt scaling learning rate; base: {base_lr}, new: {cfg.optim.lr}")
    else:
        raise NotImplementedError
    return cfg


def write_config(cfg, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_cfg_from_args(args):
    args.output_dir = os.path.abspath(args.output_dir) # 获取绝对路径 ， 在训练下工作目录和你提交任务时候的目录是不一样的 ，例：sbatch /home/user/projects/dinov2/submit.sh sbatch submit.sh 这两个提交的是同一个作业但是他们的工作目录不一样
    args.opts += [f"train.output_dir={args.output_dir}"]#将绝对路径追加到opts ，中用以覆盖老的参数，这样可以最小化对开发者的开发习惯影响
    default_cfg = OmegaConf.create(dinov2_default_config)#防止污染原始参数这里创建一个新的，这里应该是一些原始的默认的参数，也是方便我们偷懒
    cfg = OmegaConf.load(args.config_file)#加载自己配置的参数文件
    cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))#将默认的参数，绝对的路径还有自己的参数合并获得最终的参数
    return cfg


def default_setup(args):
    distributed.enable(overwrite=True)
    seed = getattr(args, "seed", 0)
    rank = distributed.get_global_rank()

    global logger
    setup_logging(output=args.output_dir, level=logging.INFO)
    logger = logging.getLogger("dinov2")

    utils.fix_random_seeds(seed + rank)
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg_from_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    default_setup(args)
    apply_scaling_rules_to_cfg(cfg)
    write_config(cfg, args.output_dir)
    return cfg
