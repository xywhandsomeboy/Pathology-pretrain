# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random

from torch_geometric.data import Data, Batch

# def collate_data_and_cast(samples_list, dtype):
#     """
#     Args:
#         samples_list: list of samples from Dataset.__getitem__
#                       每个 sample 是一个 dict，包含:
#                           - graph: 遮掩后的图 (torch_geometric.data.Data)
#                           - original_graph: 原始图 (torch_geometric.data.Data)
#                           - slide_name: 可选
#     """
#     assert len(samples_list) == 1, "只支持 batch_size = 1 的情况"
#     sample = samples_list[0]

#     masked_graph = sample["graph"]
#     original_graph = sample["original_graph"]

#     # 确保是 torch_geometric.data.Data 类型
#     if not isinstance(masked_graph, Data):
#         raise TypeError(f"Expected torch_geometric.data.Data, got {type(masked_graph)}")
#     if not isinstance(original_graph, Data):
#         raise TypeError(f"Expected torch_geometric.data.Data, got {type(original_graph)}")

#     # 可选：将其中的特征转换 dtype
#     masked_graph.x = masked_graph.x.to(dtype=torch.float32)
#     if hasattr(masked_graph, "edge_attr") and masked_graph.edge_attr is not None:
#         masked_graph.edge_attr = masked_graph.edge_attr.to(dtype=torch.float32)

#     original_graph.x = original_graph.x.to(dtype=torch.float32)
#     if hasattr(original_graph, "edge_attr") and original_graph.edge_attr is not None:
#         original_graph.edge_attr = original_graph.edge_attr.to(dtype=torch.float32)

#     return {
#         "graph": masked_graph,
#         "original_graph": original_graph,
#         "slide_name": sample.get("slide_name", None),
#     }


def collate_data_and_cast(samples_list, dtype=torch.float32):
    """
    批量 collate 多个样本，支持 batch_size > 1。

    Args:
        samples_list: list of samples from Dataset.__getitem__
                      每个 sample 是一个 dict，包含:
                          - graph: 遮掩后的图 (torch_geometric.data.Data)
                          - original_graph: 原始图 (torch_geometric.data.Data)
                          - slide_name: 可选
        dtype: torch.dtype, 默认 torch.float32

    Returns:
        一个 dict，包含 batched 的图对象。
    """
    # 分别收集 masked_graph 与 original_graph
    masked_graphs = []
    original_graphs = []
    slide_names = []

    for sample in samples_list:
        masked_graph = sample["graph"]
        original_graph = sample["original_graph"]

        if not isinstance(masked_graph, Data):
            raise TypeError(f"Expected torch_geometric.data.Data, got {type(masked_graph)}")
        if not isinstance(original_graph, Data):
            raise TypeError(f"Expected torch_geometric.data.Data, got {type(original_graph)}")

        # 转 dtype
        masked_graph.x = masked_graph.x.to(dtype=dtype)
        if hasattr(masked_graph, "edge_attr") and masked_graph.edge_attr is not None:
            masked_graph.edge_attr = masked_graph.edge_attr.to(dtype=dtype)

        original_graph.x = original_graph.x.to(dtype=dtype)
        if hasattr(original_graph, "edge_attr") and original_graph.edge_attr is not None:
            original_graph.edge_attr = original_graph.edge_attr.to(dtype=dtype)

        masked_graphs.append(masked_graph)
        original_graphs.append(original_graph)
        slide_names.append(sample.get("slide_name", None))

    # 使用 PyG 的 Batch.from_data_list 合并多个图
    batched_masked_graph = Batch.from_data_list(masked_graphs)
    batched_original_graph = Batch.from_data_list(original_graphs)

    return {
        "graph": batched_masked_graph,
        "original_graph": batched_original_graph,
        "slide_name": slide_names,
    }

