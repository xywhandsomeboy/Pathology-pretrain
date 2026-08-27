# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging

import torch
from torch import nn
from dinov2.models import build_model_from_cfg
from dinov2.utils.utils import has_batchnorms
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model

from dinov2.models.gcn import GNNChunk
from dinov2.models.gcn import GNN, GNN_graphpred
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax as pyg_softmax
try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("gnn")


class GCNMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()

        logger.info("OPTIONS -- GNN")
        self.need_to_synchronize_fsdp_streams = True
        GNN_backbone, embed_dim = build_model_from_cfg(cfg)
        
        student_model_dict["gcn"] = GNN_backbone
        pool_hidden_dim = max(embed_dim // 4, 64)
        student_model_dict["pool_gate"] = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, pool_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(pool_hidden_dim, 1),
        )

        if cfg.gcn.mask_strategy == "edge":
            student_model_dict["linear_pred_edges"] = torch.nn.Linear(embed_dim, 1)
        elif cfg.gcn.mask_strategy == "node":
            student_model_dict["linear_pred_nodes"] = torch.nn.Linear(embed_dim, embed_dim)

        self.student = nn.ModuleDict(student_model_dict)
        
        logger.info(f"Student and GCN are built: they are both {cfg.gcn.arch} network.")

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def global_pooling(self, node_rep, batch):
        gate_logits = self.student.pool_gate(node_rep).squeeze(-1)
        gate_weights = pyg_softmax(gate_logits, batch)
        weighted_rep = node_rep * gate_weights.unsqueeze(-1)
        return global_add_pool(weighted_rep, batch)

    def forward_backward(self, images):
        mse_criterion = nn.MSELoss()
        loss_dict = {}
        loss_accumulator = 0

        # 获取遮掩图和原始图
        masked_graph = images["graph"]  # 遮掩后的图（模型输入）
        original_graph = images["original_graph"]  # 原始图（监督信号）

        device = next(self.student.parameters()).device
        original_graph = original_graph.to(device)
        masked_graph = masked_graph.to(device)

        ### GCN encoder - 使用遮掩图作为输入
        x = masked_graph.x.to(device)
        edge_index = masked_graph.edge_index.to(device)
        edge_attr = masked_graph.edge_attr.to(device) if hasattr(masked_graph, 'edge_attr') else None

        node_rep = self.student.gcn(x, edge_index, edge_attr)
        batch = masked_graph.batch if hasattr(masked_graph, "batch") else torch.zeros(
            node_rep.size(0), dtype=torch.long, device=device
        )

        ### 任务1: 边重建 - 与原始图的边比较
        if hasattr(masked_graph, 'masked_edge_indices') and masked_graph.masked_edge_indices is not None:
            mask_indices = masked_graph.masked_edge_indices.to(device)
            original_edge_index = original_graph.edge_index.to(device)
        
            # 被遮掩的边在原始图中的真实连接
            true_masked_edges = original_edge_index[:, mask_indices].t()
        
            # 生成这些边的预测表示
            edge_rep = node_rep[true_masked_edges[:, 0]] + node_rep[true_masked_edges[:, 1]]
            pred_edge_weights = self.student.linear_pred_edges(edge_rep)
        
            # 取出原始图的真实边权重
            if hasattr(original_graph, 'edge_attr') and original_graph.edge_attr is not None:
                true_edge_weights = original_graph.edge_attr[mask_indices].to(device)
            else:
                # 如果没有edge_attr，就默认全1（表示存在）
                true_edge_weights = torch.ones(len(mask_indices), 1, dtype=torch.float32).to(device)
        
            # 边权重重建损失（MSE）
            edge_loss = F.mse_loss(pred_edge_weights, true_edge_weights)
            loss_accumulator += edge_loss
            loss_dict["edge_reconstruction"] = edge_loss

        ### 任务2: 节点特征重建 - 与原始节点特征比较
        if hasattr(masked_graph, 'mask_indices') and masked_graph.mask_indices is not None:
            mask_indices = masked_graph.mask_indices.to(device)
            if len(mask_indices) > 0:
                # 预测被遮掩节点的特征
                pred_features = self.student.linear_pred_nodes(node_rep[mask_indices])

                # 获取原始图中这些节点的真实特征
                true_features = original_graph.x[mask_indices].to(device)

                feature_loss = mse_criterion(pred_features, true_features)
                loss_accumulator += feature_loss
                loss_dict["node_reconstruction"] = feature_loss

        ### 任务3: 图级表示学习 - 与原始图结构一致性
        # 使用原始图的邻接矩阵作为监督信号
        original_adj = self._build_adjacency_matrix(original_graph).to(device)
        current_adj = self._build_adjacency_matrix_from_rep(node_rep, original_graph.edge_index.to(device))

        structure_loss = mse_criterion(current_adj, original_adj)
        structure_loss_weight = float(getattr(self.cfg.gcn, "structure_loss_weight", 0.02))
        loss_accumulator += structure_loss * structure_loss_weight
        loss_dict["structure_consistency"] = structure_loss

        ### 任务4: masked/original 图级表示一致性
        original_x = original_graph.x.to(device)
        original_edge_index = original_graph.edge_index.to(device)
        original_edge_attr = original_graph.edge_attr.to(device) if hasattr(original_graph, "edge_attr") else None
        original_node_rep = self.student.gcn(original_x, original_edge_index, original_edge_attr)
        original_batch = original_graph.batch if hasattr(original_graph, "batch") else torch.zeros(
            original_node_rep.size(0), dtype=torch.long, device=device
        )

        masked_graph_rep = self.global_pooling(node_rep, batch)
        original_graph_rep = self.global_pooling(original_node_rep, original_batch)
        graph_consistency_loss = 1 - F.cosine_similarity(masked_graph_rep, original_graph_rep, dim=-1).mean()
        graph_consistency_weight = float(getattr(self.cfg.gcn, "graph_consistency_weight", 0.2))
        loss_accumulator += graph_consistency_weight * graph_consistency_loss
        loss_dict["graph_consistency"] = graph_consistency_loss

        loss_dict["total_loss"] = loss_accumulator

        self.backprop_loss(loss_accumulator)
        self.fsdp_synchronize_streams()

        return loss_dict

    def _build_adjacency_matrix(self, graph):
        """从图构建邻接矩阵"""
        num_nodes = graph.x.size(0)
        adj = torch.zeros((num_nodes, num_nodes))
        adj[graph.edge_index[0], graph.edge_index[1]] = 1
        return adj

    def _build_adjacency_matrix_from_rep(self, node_rep, edge_index):
        """从节点表示重建邻接矩阵"""
        # 使用节点表示计算相似度作为邻接概率
        similarity = torch.mm(node_rep, node_rep.t())
        return similarity

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
#             self.student.dino_head._streams = self.student.backbone._streams
            self.need_to_synchronize_fsdp_streams = False

    def train(self):
        super().train()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        # below will synchronize all student subnetworks across gpus:
        for k, v in self.student.items():
            print(k)
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={GNNChunk})(self.student[k])
