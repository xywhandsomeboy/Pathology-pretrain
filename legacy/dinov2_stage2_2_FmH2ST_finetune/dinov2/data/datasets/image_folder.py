import os
import json
from typing import Optional, Callable

import pandas as pd
import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph, dropout_edge
import numpy as np
# 全局参数：子图节点数上限
NUM_NODES = 50000


class ImageFolder(Dataset):
    def __init__(
        self,
        *,
        root: str,
        table: str,
        jsonfile: str,
        transform: Optional[Callable] = None,
        use_random_subgraph: bool = True,  # train=True / test=False
        augment: bool = True,              # 是否启用增强（仅训练）
        node_noise_std: float = 0.05,      # 节点特征噪声
        node_drop_prob: float = 0.1,      # 节点 dropout
        edge_drop_prob: float = 0.1,      # 边 dropout
    ) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.use_random_subgraph = use_random_subgraph

        self.augment = augment and use_random_subgraph
        self.node_noise_std = node_noise_std
        self.node_drop_prob = node_drop_prob
        self.edge_drop_prob = edge_drop_prob

        # ===== Load metadata =====
        self.file_info = pd.read_csv(table, encoding="gbk")
        with open(jsonfile, "r") as f:
            self.label_to_id = json.load(f)
        self.num_labels = len(self.label_to_id)

        # 过滤合法 label
        self.file_info = self.file_info[
            self.file_info["cases.disease_type"].isin(self.label_to_id.keys())
        ]
        print("check:", len(self.file_info))

        # ===== Scan graph files =====
        all_samples = os.listdir(self.root)
        self.samples = []
        self.labels = []

        slide_names = set(self.file_info["file_name_HE"].values)

        for f in all_samples:
            if not f.endswith(".pt"):
                continue

            slide_id = f.replace(".pt", "")
            if slide_id not in slide_names:
                continue

            graph_path = os.path.join(self.root, f)
            row = self.file_info[self.file_info["file_name_HE"] == slide_id]
            label = self.label_to_id[row["cases.disease_type"].values[0]]

            try:
                graph = torch.load(graph_path, map_location="cpu")
            except Exception as e:
                print(f"[Warning] Failed to load {f}: {e}")
                continue

            if not isinstance(graph, Data):
                continue
            if graph.x is None or graph.edge_index is None or graph.edge_index.numel() == 0:
                continue

            # 确保 graph.x 和 edge_index 是合法的
            graph.x = graph.x.float()
            if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
                graph.edge_attr = graph.edge_attr.float()

            self.samples.append(f)
            self.labels.append(label)

        # ===== 统计节点数信息 =====
        node_counts = []
        for f in self.samples:
            graph_path = os.path.join(self.root, f)
            graph = torch.load(graph_path, map_location="cpu")
            if hasattr(graph, "num_nodes"):
                node_counts.append(graph.num_nodes)
        
        if len(node_counts) > 0:
            node_counts = np.array(node_counts)
            print(f"Node statistics for loaded graphs:")
            print(f"  Max nodes   : {node_counts.max()}")
            print(f"  Min nodes   : {node_counts.min()}")
            print(f"  Median nodes: {int(np.median(node_counts))}")
            print(f"  Mean nodes  : {node_counts.mean():.2f}")

        print(
            f"Loaded {len(self.samples)} graphs | "
            f"use_random_subgraph={self.use_random_subgraph} | "
            f"augment={self.augment}"
        )

    def get_class_weights(self, mode='balanced'):
        """
        计算类别权重
        
        Args:
            mode: 'balanced' - 使用sklearn的balanced权重
                  'inverse' - 使用类别频率的倒数
                  'sqrt_inverse' - 使用类别频率平方根的倒数
        
        Returns:
            weights: 每个样本的权重列表
        """
        import numpy as np
        from collections import Counter
        
        labels = np.array(self.labels)
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_to_count = dict(zip(unique_labels, counts))
        
        print(f"\nClass distribution:")
        for label, count in sorted(label_to_count.items()):
            print(f"  Class {label}: {count} samples")
        
        if mode == 'balanced':
            # sklearn style: n_samples / (n_classes * np.bincount(y))
            total_samples = len(labels)
            n_classes = len(unique_labels)
            class_weights = {}
            for label in unique_labels:
                class_weights[label] = total_samples / (n_classes * label_to_count[label])
        elif mode == 'inverse':
            # Inverse frequency
            max_count = max(counts)
            class_weights = {label: max_count / count for label, count in label_to_count.items()}
        elif mode == 'sqrt_inverse':
            # Square root of inverse frequency
            max_count = max(counts)
            class_weights = {label: np.sqrt(max_count / count) for label, count in label_to_count.items()}
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Create sample weights
        sample_weights = [class_weights[label] for label in labels]
        
        print(f"\nClass weights ({mode}):")
        for label in sorted(unique_labels):
            print(f"  Class {label}: {class_weights[label]:.4f}")
        
        return sample_weights

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        slide_name = self.samples[idx]
        label = self.labels[idx]

        graph = torch.load(os.path.join(self.root, slide_name), map_location="cpu")

        # ===== 子图采样 =====
        if graph.num_nodes > NUM_NODES:
            graph = self.get_subgraph(graph)

        # ===== 数据增强（仅训练） =====
        if self.augment:
            graph = self.augment_graph(graph)

        return {
            "graph": graph,
            "slide_name": slide_name,
            "label": label,
        }

    # ------------------------------------------------------------------
    # Subgraph sampling
    # ------------------------------------------------------------------
    def get_subgraph(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        num_keep = NUM_NODES

        if self.use_random_subgraph:
            node_idx = torch.randperm(num_nodes)[:num_keep]
        else:
            node_idx = torch.arange(min(num_keep, num_nodes))

        # 🚨 过滤非法索引
        node_idx = node_idx[(node_idx >= 0) & (node_idx < num_nodes)]
        if node_idx.numel() == 0:
            raise RuntimeError(f"Empty node_idx after filtering, num_nodes={num_nodes}")

        edge_index, edge_attr = subgraph(
            subset=node_idx,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            relabel_nodes=True,
            num_nodes=num_nodes,
        )

        return Data(
            x=data.x[node_idx],
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=data.pos[node_idx] if hasattr(data, "pos") else None,
        )

    # ------------------------------------------------------------------
    # Graph augmentation (train only)
    # ------------------------------------------------------------------
    def augment_graph(self, data: Data) -> Data:
        data = data.clone()

        # 1️⃣ Node feature noise
        if data.x is not None and self.node_noise_std > 0:
            noise = torch.randn_like(data.x) * self.node_noise_std
            data.x = data.x + noise

        # 2️⃣ Node dropout
        if self.node_drop_prob > 0:
            keep_mask = torch.rand(data.num_nodes) > self.node_drop_prob
            if keep_mask.sum() > 0:
                node_idx = keep_mask.nonzero(as_tuple=False).view(-1)
                data = self._subgraph_from_nodes(data, node_idx)

        # 3️⃣ Edge dropout
        if self.edge_drop_prob > 0 and data.edge_index.numel() > 0:
            edge_index, edge_mask = dropout_edge(
                data.edge_index,
                p=self.edge_drop_prob,
                force_undirected=False,
            )
            data.edge_index = edge_index
            if data.edge_attr is not None:
                data.edge_attr = data.edge_attr[edge_mask]

        return data

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------
    def _subgraph_from_nodes(self, data, node_idx):
        num_nodes = data.num_nodes
        node_idx = node_idx.long()
        node_idx = node_idx[(node_idx >= 0) & (node_idx < num_nodes)]
        if node_idx.numel() == 0:
            raise RuntimeError(f"Empty node_idx after filtering, num_nodes={num_nodes}")

        edge_index, edge_attr = subgraph(
            subset=node_idx,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            relabel_nodes=True,
            num_nodes=num_nodes,
        )

        return Data(
            x=data.x[node_idx],
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=data.pos[node_idx] if hasattr(data, "pos") else None,
        )