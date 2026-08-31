# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0.

"""Stage-2 PyG batch collation."""

import torch
from torch_geometric.data import Batch, Data


def collate_data_and_cast(samples_list, dtype=torch.float32):
    if not samples_list:
        raise ValueError("Cannot collate an empty graph batch")
    graphs, slide_names = [], []
    for sample in samples_list:
        graph = sample["original_graph"]
        if not isinstance(graph, Data):
            raise TypeError(f"Expected torch_geometric.data.Data, got {type(graph)}")
        graph = graph.clone()
        graph.x = graph.x.to(dtype=dtype)
        if graph.edge_attr is not None:
            graph.edge_attr = graph.edge_attr.to(dtype=dtype)
        graphs.append(graph)
        slide_names.append(str(sample["slide_name"]))
    return {
        "original_graph": Batch.from_data_list(graphs),
        "slide_name": slide_names,
    }
