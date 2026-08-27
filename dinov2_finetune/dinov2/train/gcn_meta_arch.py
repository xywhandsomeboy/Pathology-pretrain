# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging

import torch
from torch import nn

from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov2.models import build_model_from_cfg
from dinov2.layers import DINOHead
from dinov2.utils.utils import has_batchnorms
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model

from dinov2.models.vision_transformer import BlockChunk
from dinov2.models.gcn import GNN, GNN_graphpred
import torch.nn.functional as F
from torch_geometric.data import Data
try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("dinov2")


class GCNMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()

        student_backbone, embed_dim = build_model_from_cfg(cfg)
        student_model_dict["backbone"] = student_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")
            student_backbone.load_state_dict(chkpt["model"], strict=False)

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head

        logger.info("OPTIONS -- DINO")
        if self.do_dino:
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
            logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
            logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                self.koleo_loss = KoLeoLoss()

        else:
            logger.info("OPTIONS -- DINO -- not using DINO")

        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")
        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert max(cfg.ibot.mask_ratio_min_max) > 0, "please provide a positive mask ratio tuple for ibot"
            assert cfg.ibot.mask_sample_probability > 0, "please provide a positive mask probability for ibot"
            self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else cfg.dino.head_n_prototypes
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
                logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
                logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
                logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
                ibot_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
            else:
                logger.info("OPTIONS -- IBOT -- head shared with DINO")

        self.need_to_synchronize_fsdp_streams = True

        student_model_dict["mlp"] = nn.Sequential(
            nn.Linear(cfg.dino.head_n_prototypes, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, cfg.gcn.emb_dim)
        )
        student_model_dict["gcn"] = GNN(
            cfg.gcn.num_layer,
            cfg.gcn.emb_dim,
            JK=cfg.gcn.JK,
            drop_ratio=cfg.gcn.dropout_ratio,
            gnn_type=cfg.gcn.gnn_type,
            edge_dim=3
        )

        student_model_dict["linear_pred_edges"] = torch.nn.Linear(cfg.gcn.emb_dim, 4)

        self.student = nn.ModuleDict(student_model_dict)

        student_dtype = next(self.student.backbone.parameters()).dtype

        
        logger.info(f"Student and GCN are built: they are both {cfg.student.arch} network.")

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def dino_encoder(self, images):
        n_global_crops = 2
        assert n_global_crops == 2
        batch_size = int(images["collated_global_crops"].shape[0]/n_global_crops)
        n_local_crops = self.cfg.crops.local_crops_number
        
        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        local_crops = images["collated_local_crops"].cuda(non_blocking=True)

        student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.backbone(
            [global_crops, local_crops], masks=[None, None], is_training=True
        )

        inputs_for_student_head_list = []

        # 1a: local crops cls tokens
        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
        
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        # 1c: global crops patch tokens
        if self.do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            B = student_global_backbone_output_dict["x_norm_patchtokens"].shape[0]
            num_tokens = student_global_backbone_output_dict["x_norm_patchtokens"].shape[1]
            ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"].flatten(0, 1)

            if not self.ibot_separate_head:
                
                inputs_for_student_head_list.append(ibot_student_patch_tokens.unsqueeze(0))
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(ibot_student_patch_tokens)

        # 2: run
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

        # 3a: local crops cls tokens
        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3b: global crops cls tokens
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3c: global crops patch tokens
        if self.do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)

        student_global_cls_tokens_after_head = student_global_cls_tokens_after_head.view(n_global_crops,batch_size,-1)
        student_local_cls_tokens_after_head = student_local_cls_tokens_after_head.view(n_local_crops,batch_size,-1)
        student_global_masked_patch_tokens_after_head = student_global_masked_patch_tokens_after_head.view((n_global_crops,batch_size,num_tokens,-1))
        
        student_global_mean_feat = student_global_cls_tokens_after_head.mean(dim=0)
        student_local_mean_feat = student_local_cls_tokens_after_head.mean(dim=0)
        student_global_patch_mean_feat = student_global_masked_patch_tokens_after_head.mean(dim=(0, 2))

        cat_feat = (student_global_mean_feat+student_global_patch_mean_feat+student_local_mean_feat)/3
        
        node_features = self.student.mlp(cat_feat)
        return node_features
        
    def construct_graph(
        self,
        node_features,
        coords,
        radius=50.0,
        sim_threshold=0.7,
        mask_ratio=0.75,
        mask_mode="remove",  # "remove" 或 "flag"
    ):
        """
        Args:
            node_features: Tensor [N, D] 每个 patch 的特征
            coords: Tensor [N, 2] 每个 patch 的坐标
            radius: float, 邻居半径
            sim_threshold: float, 余弦相似度阈值
            mask_ratio: float, mask 的边比例（用于自监督）
            mask_mode: str, "remove" 删除 Mask 边, "flag" 保留边但添加标记
        """
        N = node_features.shape[0]

        # ---------- 距离矩阵 ----------
        dist_matrix = torch.cdist(coords, coords, p=2)  # [N, N]
        sim_matrix = F.cosine_similarity(
            node_features.unsqueeze(1),  # [N, 1, D]
            node_features.unsqueeze(0),  # [1, N, D]
            dim=-1
        )  # [N, N]

        # ---------- 邻居约束 ----------
        edge_index_list = []
        edge_attr_list = []
        edge_label_list = []
        undirected_pairs = []  # 记录唯一无向边的节点对 (i,j)

        for i in range(N):
            for j in range(i + 1, N):
                d = dist_matrix[i, j].item()
                s = sim_matrix[i, j].item()

                if d < radius or s > sim_threshold:
                    # 添加无向边 (成对)
                    edge_index_list.append([i, j])
                    edge_index_list.append([j, i])

                    # edge_attr = [distance, similarity, mask_flag]
                    edge_attr_list.append([d, s, 0.0])
                    edge_attr_list.append([d, s, 0.0])

                    # label: 0=none, 1=distance, 2=similarity, 3=both
                    edge_label = int((d < radius)) + int((s > sim_threshold)) * 2
                    edge_label_list.append(edge_label)
                    edge_label_list.append(edge_label)

                    undirected_pairs.append([i, j])  # 只存一次 (i,j)

        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()  # [2, E]
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)                  # [E, 3]
        edge_labels = torch.tensor(edge_label_list, dtype=torch.long)                  # [E]

        # ---------- Mask 无向边 ----------
        num_undirected_edges = len(undirected_pairs)
        num_mask = int(num_undirected_edges * mask_ratio)
        perm = torch.randperm(num_undirected_edges)
        masked_pairs = [undirected_pairs[i] for i in perm[:num_mask]]

        masked_edge_pairs = torch.tensor(masked_pairs, dtype=torch.long)  # [num_mask, 2]
        mask_edge_label = []
        for u, v in masked_pairs:
            idx = ((edge_index[0] == u) & (edge_index[1] == v)).nonzero(as_tuple=True)[0]
            label = edge_labels[idx]
            mask_edge_label.append(label.item())
        mask_edge_label = F.one_hot(torch.tensor(mask_edge_label), num_classes=4).float()

        # ---------- 根据 mask_mode 修改 ----------
        if mask_mode == "remove":
            # 删除这些边
            remove_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
            for u, v in masked_pairs:
                idx_uv = ((edge_index[0] == u) & (edge_index[1] == v)).nonzero(as_tuple=True)[0]
                idx_vu = ((edge_index[0] == v) & (edge_index[1] == u)).nonzero(as_tuple=True)[0]
                remove_mask[idx_uv] = True
                remove_mask[idx_vu] = True

            keep_mask = ~remove_mask
            edge_index = edge_index[:, keep_mask]
            edge_attr = edge_attr[keep_mask]
            edge_labels = edge_labels[keep_mask]

        elif mask_mode == "flag":
            # 在 edge_attr 中标记这些边
            for u, v in masked_pairs:
                idx_uv = ((edge_index[0] == u) & (edge_index[1] == v)).nonzero(as_tuple=True)[0]
                idx_vu = ((edge_index[0] == v) & (edge_index[1] == u)).nonzero(as_tuple=True)[0]
                edge_attr[idx_uv, 2] = 1.0
                edge_attr[idx_vu, 2] = 1.0

        else:
            raise ValueError(f"Unsupported mask_mode: {mask_mode}")

        # ---------- 封装 PyG Data ----------
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        data.masked_edge_pairs = masked_edge_pairs      # [num_mask, 2]
        data.mask_edge_label = mask_edge_label          # [num_mask, 4]

        return data

    def forward_backward(self, images):
        criterion = nn.CrossEntropyLoss()
        loss_dict = {}
        loss_accumulator = 0  # for backprop

        ### DINO encoder
        node_features = self.dino_encoder(images)

        ### construct graph
        edge_batch = self.construct_graph(node_features, images["coords"])

        ### GCN encoder
        node_rep = self.student.gcn(edge_batch.x, edge_batch.edge_index, edge_batch.edge_attr)

        ### predict the edge types.
        masked_edge_pairs = edge_batch.masked_edge_pairs.to(node_rep.device)
        edge_rep = node_rep[masked_edge_pairs[:, 0]] + node_rep[masked_edge_pairs[:, 1]]
        pred_edge = self.student.linear_pred_edges(edge_rep)

        edge_label = torch.argmax(edge_batch.mask_edge_label, dim=1).to(pred_edge.device)

        loss_accumulator += criterion(pred_edge, edge_label)
        loss_dict["gcn_global_mask"] = loss_accumulator

        self.backprop_loss(loss_accumulator)
        self.fsdp_synchronize_streams()

        return loss_dict
        
    # def forward_backward(self, images):
    #     criterion = nn.CrossEntropyLoss()
    #     loss_dict = {}
    #     loss_accumulator = 0  # for backprop
    #     ### DINO encoder
    #     node_features = self.dino_encoder(images)
    #     ### construct graph
    #     edge_batch = self.construct_graph(node_features,images["coords"])
    #     # print("index ",edge_batch.edge_index.shape)
    #     # print("attr ",edge_batch.edge_attr.shape)
    #     # print("x ",edge_batch.x.shape)
    #     ### GCN encoder
    #     node_rep = self.student.gcn(edge_batch.x, edge_batch.edge_index, edge_batch.edge_attr)
    #     ### predict the edge types.
    #     masked_edge_index = edge_batch.edge_index[:, edge_batch.masked_edge_idx]
    #     edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
    #     pred_edge = self.student.linear_pred_edges(edge_rep)
    #     edge_label = torch.argmax(edge_batch.mask_edge_label, dim = 1)
    #     device = pred_edge.device   # 保证跟预测结果一致
    #     edge_label = edge_label.to(device)

    #     # print("check ",pred_edge.shape,edge_label.shape) # torch.Size([12, 2]) torch.Size([12])
    #     loss_accumulator +=criterion(pred_edge, edge_label)
        
    #     loss_dict["gcn_global_mask"]=loss_accumulator
        
    #     self.backprop_loss(loss_accumulator)

    #     self.fsdp_synchronize_streams()

    #     return loss_dict

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            self.student.dino_head._streams = self.student.backbone._streams
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
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])

