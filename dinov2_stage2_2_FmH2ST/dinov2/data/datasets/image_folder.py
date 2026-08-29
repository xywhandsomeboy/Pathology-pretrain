import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import random
import pandas as pd
import pickle
import gzip
from pathlib import Path
import torch
from typing import Callable, List, Optional, Tuple, Union
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

def auto_set_graph_filters(root, node_quantile=0.05, edge_quantile=0.05):
    """
    自动根据数据统计设置 min_nodes 和 min_edges

    Args:
        root (str): 存放图文件的目录
        node_quantile (float): 节点数分位数，低于这个分位数的图会被丢弃
        edge_quantile (float): 边数分位数，低于这个分位数的图会被丢弃

    Returns:
        min_nodes (int), min_edges (int)
    """
    node_counts = []
    edge_counts = []

    for fname in os.listdir(root):
        path = os.path.join(root, fname)
        g = torch.load(path)
        
        if not isinstance(g, Data):
            continue  # 跳过非 Data 对象
        if g.x is None or len(g.edge_index)==0:
            continue  # 跳过异常图
            
        if hasattr(g, 'num_nodes'):
            node_counts.append(g.num_nodes)
        else:
            node_counts.append(g.x.size(0))
        
        if hasattr(g, 'edge_index'):
            edge_counts.append(g.edge_index.size(1))
        else:
            edge_counts.append(0)

    node_counts = np.array(node_counts)
    edge_counts = np.array(edge_counts)

    # 根据分位数计算阈值
    min_nodes = max(1, int(np.quantile(node_counts, node_quantile)))
    min_edges = max(1, int(np.quantile(edge_counts, edge_quantile)))

    print(f"[Auto Filter] Nodes: min={min_nodes}, median={int(np.median(node_counts))}, max={int(np.max(node_counts))}")
    print(f"[Auto Filter] Edges: min={min_edges}, median={int(np.median(edge_counts))}, max={int(np.max(edge_counts))}")

    return min_nodes, min_edges
    
class ImageFolder(Dataset):
    def __init__(
        self, 
        *, 
        root: str, 
        transform: Optional[Callable] = None,
        mask_ratio: float = 0.75,  # 遮掩比例
        mask_strategy: str = "edge",  # 遮掩策略
        rand_walk: bool = False,
        min_nodes: Optional[int] = None,
        min_edges: Optional[int] = None,
    ) -> None:
        self.root = root
        self.transform = transform
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        self.rand_walk = rand_walk
        
        if min_nodes is None or min_edges is None:
            min_node, min_edge = auto_set_graph_filters(root, node_quantile=0.01, edge_quantile=0.01)
            if min_nodes is None:
                min_nodes = min(min_node, 5)
            if min_edges is None:
                min_edges = min(min_edge, 10)
                
        print("check",min_nodes,min_edges)
        # 初步列出所有文件
        all_samples = os.listdir(self.root)
        self.samples = []

        for f in all_samples:
            graph_path = os.path.join(self.root, f)
            try:
                graph = torch.load(graph_path)
            except Exception as e:
                print(f"Failed to load {f}: {e}")
                continue

            if not isinstance(graph, Data):
                continue  # 跳过非 Data 对象
            if graph.x is None or len(graph.edge_index)==0:
                continue  # 跳过异常图
                
            # 检查节点数和边数
            num_nodes = graph.num_nodes if hasattr(graph, 'num_nodes') else graph.x.size(0)
            num_edges = graph.edge_index.size(1) if hasattr(graph, 'edge_index') and graph.edge_index is not None else 0

            if num_nodes >= min_nodes and num_edges >= min_edges:
                self.samples.append(f)
            else:
                print(f"Skipping {f}: nodes={num_nodes}, edges={num_edges}")

        print(f"Loaded {len(self.samples)} graphs after filtering.")
    
    def __len__(self):
        return len(self.samples)

    
    def __getitem__(self, idx):
        slide = {}
        slide_name = self.samples[idx]
        graph = torch.load(os.path.join(self.root, slide_name))
        graph = self.get_random_subgraph(graph)
        # print("check -slide",slide_name,graph.x.size(0),graph.edge_index.size(1))
        # Mask sampling is now performed inside GCNMetaArch so that random,
        # spatial-region and random-walk masks share one learnable mask token
        # and reach exactly the configured ratio.
        slide["graph"] = graph
        slide["original_graph"] = graph
        slide["slide_name"] = slide_name
        
        return slide
    
    def get_random_subgraph(self,data, ratio=0.2):
        """
        从 PyG Data 对象中随机抽取 ratio 比例的节点，返回诱导子图。
        """
        num_nodes = data.num_nodes
        if num_nodes <=5000:
            return data
        num_keep = 5000
    
        # 随机选择节点索引
        node_idx = torch.randperm(num_nodes)[:num_keep]
    
        # 构建诱导子图（保留与这些节点相关的边和属性）
        edge_index, edge_attr = subgraph(
            node_idx, 
            data.edge_index, 
            data.edge_attr, 
            relabel_nodes=True
        )
    
        # 生成新的 Data
        sub_data = data.__class__(
            x=data.x[node_idx],
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=data.pos[node_idx] if hasattr(data, 'pos') else None
        )
    
        return sub_data
        
    def _mask_graph(self, graph):
        """根据策略对graph进行随机遮掩"""
        if self.mask_strategy == "node":
            return self._mask_node_features(graph)
        elif self.mask_strategy == "edge":
            return self._mask_edges(graph)
        elif self.mask_strategy == "patch":
            return self._mask_patches(graph)
        elif self.mask_strategy == "feature":
            return self._mask_feature_channels(graph)
        else:
            return self._mask_node_features(graph)  # 默认节点遮掩
    
    def _mask_node_features(self, graph):
        """遮掩节点特征"""
        masked_graph = graph.clone()
        
        if hasattr(graph, 'x') and graph.x is not None:
            num_nodes = graph.x.size(0)
            num_features = graph.x.size(1)
            
            # 随机选择要遮掩的节点
            num_mask = max(1, int(num_nodes * self.mask_ratio))
            mask_indices = torch.randperm(num_nodes)[:num_mask]
            
            # 创建遮掩版本和原始版本
            original_features = graph.x.clone()
            
            # 用均值遮掩（保持特征分布）
            mean_features = graph.x.mean(dim=0, keepdim=True)
            masked_graph.x[mask_indices] = mean_features.repeat(num_mask, 1)
            
            # 保存遮掩信息用于损失计算
            masked_graph.mask_indices = mask_indices
            masked_graph.original_features = original_features
            masked_graph.mask_type = "node"
            
        return masked_graph
    
    def _mask_edges(self, graph):
        """遮掩边"""
        masked_graph = graph.clone()
        
        if hasattr(graph, 'edge_index') and graph.edge_index is not None:
            num_edges = graph.edge_index.size(1)
            num_mask = max(1, int(num_edges * self.mask_ratio))
            
            # 随机选择要遮掩的边
            mask_indices = torch.randperm(num_edges)[:num_mask]
            
            # 保存原始边信息
            original_edge_index = graph.edge_index.clone()
            original_edge_attr = None
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                original_edge_attr = graph.edge_attr.clone()
            
            # 创建遮掩的边索引（移除被遮掩的边）
            if num_mask > 0:
                keep_indices = torch.ones(num_edges, dtype=torch.bool)
                keep_indices[mask_indices] = False
                masked_graph.edge_index = graph.edge_index[:, keep_indices]
                
                # 如果存在边特征，也相应处理
                if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                    masked_graph.edge_attr = graph.edge_attr[keep_indices]
                
                # 保存遮掩信息
                masked_graph.masked_edge_indices = mask_indices
                masked_graph.original_edge_index = original_edge_index
                masked_graph.original_edge_attr = original_edge_attr
                masked_graph.mask_type = "edge"
        
        return masked_graph
    
    def _mask_patches(self, graph):
        """遮掩图块（基于空间位置）"""
        masked_graph = graph.clone()
        
        # 假设graph包含位置信息pos
        if hasattr(graph, 'pos') and graph.pos is not None:
            num_nodes = graph.pos.size(0)
            num_mask = max(1, int(num_nodes * self.mask_ratio))
            
            # 基于空间位置的聚类遮掩
            from sklearn.cluster import KMeans
            pos_np = graph.pos.numpy()
            n_clusters = max(2, int(num_nodes * 0.1))  # 聚类数量
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(pos_np)
            
            # 随机选择要遮掩的聚类
            mask_cluster = torch.randint(0, n_clusters, (1,)).item()
            mask_indices = torch.tensor([i for i, c in enumerate(clusters) if c == mask_cluster])
            
            if len(mask_indices) > 0:
                # 遮掩节点特征
                if hasattr(graph, 'x') and graph.x is not None:
                    original_features = graph.x.clone()
                    mean_features = graph.x.mean(dim=0, keepdim=True)
                    masked_graph.x[mask_indices] = mean_features.repeat(len(mask_indices), 1)
                    
                    masked_graph.mask_indices = mask_indices
                    masked_graph.original_features = original_features
                    masked_graph.mask_type = "patch"
        
        return masked_graph
    
    def _mask_feature_channels(self, graph):
        """遮掩特征通道"""
        masked_graph = graph.clone()
        
        if hasattr(graph, 'x') and graph.x is not None:
            num_features = graph.x.size(1)
            num_mask = max(1, int(num_features * self.mask_ratio))
            
            # 随机选择要遮掩的特征通道
            mask_channels = torch.randperm(num_features)[:num_mask]
            
            original_features = graph.x.clone()
            
            # 遮掩选中的特征通道
            masked_graph.x[:, mask_channels] = 0
            
            # 保存遮掩信息
            masked_graph.mask_channels = mask_channels
            masked_graph.original_features = original_features
            masked_graph.mask_type = "feature"
        
        return masked_graph
    
    def _random_walk_mask(self, graph, walk_length=None):
        """基于随机游走的遮掩"""
        if walk_length is None:
            walk_length = max(3, int(graph.x.size(0) * 0.1))  # 自适应游走长度
        
        if not hasattr(graph, 'edge_index') or graph.edge_index is None:
            return self._mask_node_features(graph)
        
        num_nodes = graph.x.size(0)
        num_mask = max(1, int(num_nodes * self.mask_ratio))
        
        # 从随机节点开始随机游走
        start_node = torch.randint(0, num_nodes, (1,))
        current_node = start_node
        
        mask_indices = [current_node.item()]
        visited = set([current_node.item()])
        
        for step in range(walk_length - 1):
            # 找到当前节点的邻居
            neighbors = self._get_neighbors(current_node, graph.edge_index)
            
            # 过滤已访问的邻居
            unvisited_neighbors = [n for n in neighbors if n.item() not in visited]
            
            if len(unvisited_neighbors) == 0:
                # 如果没有未访问的邻居，随机跳转
                next_node = torch.randint(0, num_nodes, (1,))
            else:
                # 随机选择下一个未访问的邻居
                next_node = unvisited_neighbors[torch.randint(0, len(unvisited_neighbors), (1,))]
            
            mask_indices.append(next_node.item())
            visited.add(next_node.item())
            current_node = next_node
            
            if len(mask_indices) >= num_mask:
                break
        
        mask_indices = torch.tensor(mask_indices)
        return self._apply_random_walk_mask(graph, mask_indices)
    
    def _get_neighbors(self, node, edge_index):
        """获取节点的邻居"""
        mask = edge_index[0] == node
        neighbors = edge_index[1][mask]
        return neighbors.unique()
    
    def _apply_random_walk_mask(self, graph, mask_indices):
        """应用随机游走遮掩"""
        masked_graph = graph.clone()
        
        if hasattr(graph, 'x') and graph.x is not None:
            original_features = graph.x.clone()
            num_mask = len(mask_indices)
            
            if num_mask > 0:
                # 用均值遮掩
                mean_features = graph.x.mean(dim=0, keepdim=True)
                masked_graph.x[mask_indices] = mean_features.repeat(num_mask, 1)
                
                # 保存遮掩信息
                masked_graph.mask_indices = mask_indices
                masked_graph.original_features = original_features
                masked_graph.mask_type = "random_walk"
        
        return masked_graph
