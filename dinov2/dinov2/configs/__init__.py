# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import pathlib

from omegaconf import OmegaConf


def load_config(config_name: str): #这个函数的主要意义是让omegaconf可以在工作目录未知的情况下也能够找到文件，加载文件并返回Dictconfig文件（和传统的yamlload比做了一些升级，对引用方式上也做了改变，以前的key变成了属性）
    config_filename = config_name + ".yaml"#偷懒用的，让开发者不用管后缀，也是消除了格式带来的影响
    return OmegaConf.load(pathlib.Path(__file__).parent.resolve() / config_filename) # __file__当前运行文件的自身路径，这里指/home/user/90T/xiayw/CerviPath/dinov2/dinov2/train/checkpoint_merge_fsdp.py/
                                                                                    # Path(__file__).parent 取所在目录  ；  转化为绝对路径（原因同上） ； /加上了你输入的名字 ；


dinov2_default_config = load_config("ssl_default_config")


def load_and_merge_config(config_name: str):
    default_config = OmegaConf.create(dinov2_default_config)
    loaded_config = load_config(config_name)
    return OmegaConf.merge(default_config, loaded_config)
