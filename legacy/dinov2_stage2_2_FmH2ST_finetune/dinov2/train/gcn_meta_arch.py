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
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax
try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("gnn")


class GCNMetaArch(nn.Module):
    def __init__(self, cfg, criterion=None, class_prior=None):
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

        classifier_dropout = float(getattr(cfg.train, "classifier_dropout", 0.2))
        student_model_dict["classifier"] = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Dropout(classifier_dropout),
            torch.nn.Linear(embed_dim, cfg.train.classes),
        )

        self.student = nn.ModuleDict(student_model_dict)
        self.criterion = criterion
        self.collapse_penalty_weight = float(getattr(cfg.train, "collapse_penalty_weight", 0.0))
        if class_prior is None:
            class_prior = torch.full((cfg.train.classes,), 1.0 / cfg.train.classes, dtype=torch.float32)
        self.register_buffer("class_prior", class_prior.float())

        logger.info(f"Student and GCN are built: they are both {cfg.gcn.arch} network.")

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def global_pooling(self, node_rep, batch):
        """Gated attention pooling for graph classification."""
        gate_logits = self.student.pool_gate(node_rep).squeeze(-1)
        gate_weights = pyg_softmax(gate_logits, batch)
        weighted_rep = node_rep * gate_weights.unsqueeze(-1)
        return global_add_pool(weighted_rep, batch)
    
    def forward_backward(self, images):
    
        loss_dict = {}
        loss_accumulator = 0

        # 获取遮掩图和原始图
        original_graph = images["graph"]  # 遮掩后的图（模型输入)
        label = images["label"] 

        device = next(self.student.parameters()).device
        original_graph = original_graph.to(device) # [4,
        label = label.to(device) # [4]
        
        ### GCN encoder - 使用遮掩图作为输入
        x = original_graph.x.to(device)# [324,
        edge_index = original_graph.edge_index.to(device) # [2 
        edge_attr = original_graph.edge_attr.to(device) if hasattr(original_graph, 'edge_attr') else None # [2510
        # print(len(original_graph),len(x),len(edge_index),len(edge_attr))
        # with torch.no_grad():
        node_rep = self.student.gcn(x, edge_index, edge_attr) # [324,1024]

        # 关键：使用全局池化将节点表示聚合成图表示
        if hasattr(original_graph, 'batch'):
            # 使用批处理信息进行全局池化
            graph_rep = self.global_pooling(node_rep, original_graph.batch)  # [batch_size, 1024]
        else:
            # 如果没有批处理，假设只有一个图
            graph_rep = torch.mean(node_rep, dim=0, keepdim=True)  # [1, 1024]
        
        # 分类器
        output = self.student.classifier(graph_rep)  # [batch_size, num_classes]

        # print(label.shape,output.shape,node_rep.shape,graph_rep.shape,len(original_graph))
        cls_loss = self.criterion(output, label)
        probs = F.softmax(output, dim=-1)
        mean_probs = probs.mean(dim=0).clamp_min(1e-8)

        collapse_loss = torch.tensor(0.0, device=device)
        if self.collapse_penalty_weight > 0:
            target_prior = self.class_prior.to(device=device, dtype=mean_probs.dtype).clamp_min(1e-8)
            collapse_loss = torch.sum(mean_probs * (mean_probs.log() - target_prior.log()))

        # 计算准确率
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == label).sum().item()
        total = label.size(0)
        accuracy = correct / total

        loss_accumulator += cls_loss + self.collapse_penalty_weight * collapse_loss
        loss_dict["classifier"] = cls_loss
        loss_dict["collapse_penalty"] = collapse_loss
        loss_dict["accuracy"] = torch.tensor(accuracy, device=device)  # batch准确率
        loss_dict["pred_max_prob"] = mean_probs.max().detach()
        loss_dict["total_loss"] = loss_accumulator

        self.backprop_loss(loss_accumulator)
        self.fsdp_synchronize_streams()

        return loss_dict,correct,total

    def test_forward_backward(self, images):
        # 获取遮掩图和原始图
        original_graph = images["graph"]  # 遮掩后的图（模型输入)
        label = images["label"] 

        device = next(self.student.parameters()).device
        original_graph = original_graph.to(device) # [4,
        label = label.to(device) # [4]
        
        ### GCN encoder - 使用遮掩图作为输入
        x = original_graph.x.to(device)# [324,
        edge_index = original_graph.edge_index.to(device) # [2 
        edge_attr = original_graph.edge_attr.to(device) if hasattr(original_graph, 'edge_attr') else None # [2510
        # print(len(original_graph),len(x),len(edge_index),len(edge_attr))
        node_rep = self.student.gcn(x, edge_index, edge_attr) # [324,1024]

        # 关键：使用全局池化将节点表示聚合成图表示
        if hasattr(original_graph, 'batch'):
            # 使用批处理信息进行全局池化
            graph_rep = self.global_pooling(node_rep, original_graph.batch)  # [batch_size, 1024]
        else:
            # 如果没有批处理，假设只有一个图
            graph_rep = torch.mean(node_rep, dim=0, keepdim=True)  # [1, 1024]
        
        # 分类器
        output = self.student.classifier(graph_rep)  # [batch_size, num_classes]
        
        return label,output

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

    def train(self, mode: bool = True): 
        super().train(mode)

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

        for k, v in self.student.items():
            student_model_cfg = self.cfg.compute_precision.student[k]

            if k == "gcn":
                # 逐个 wrap GNNChunk（最关键）
                for i, chunk in enumerate(v.gnns):
                    v.gnns[i] = get_fsdp_wrapper(
                        student_model_cfg,
                        modules_to_wrap={GNNChunk}
                    )(chunk)
            else:
                self.student[k] = get_fsdp_wrapper(
                    student_model_cfg,
                    modules_to_wrap={GNNChunk}
                )(v)

    # def prepare_for_distributed_training(self):
    #     logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
    #     if has_batchnorms(self.student):
    #         raise NotImplementedError
    #     # below will synchronize all student subnetworks across gpus:
    #     for k, v in self.student.items():
    #         print(k)
    #         student_model_cfg = self.cfg.compute_precision.student[k]
    #         self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={GNNChunk})(self.student[k])
