# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging
import math
import os

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
try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("dinov2")


class SpatialPatchAggregator(nn.Module):
    """修改 1：子图内部 patch 特征的空间感知聚合。

    原实现直接对子图内所有 DINO patch feature 做 mean，丢掉了
    "patch 在子图中的空间位置" 信息。这里改为：

        DINO patch tokens -> 重排成 2D 网格 -> 1x1 通道压缩 ->
        多尺度 Depthwise Conv (3x3 / 5x5) -> 自适应池化(下采样) -> 固定尺寸向量

    即保留局部空间结构与多尺度语义后，再压缩成一张子图的节点特征，
    仍然保持：1 张病理子图 = 1 个 GNN 节点。
    """

    def __init__(self, in_dim, hidden_dim=256, pool_size=4):
        super().__init__()
        self.pool_size = pool_size
        # 通道压缩：head_n_prototypes 维度太大，先用 1x1 conv 压缩再做空间卷积
        self.compress = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
        # 多尺度 Depthwise Conv：3x3 捕捉局部结构，5x5 提供更大感受野
        self.dw_conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.dw_conv5 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        # 下采样到固定尺寸，保证输出与 patch 网格大小无关
        self.pool = nn.AdaptiveAvgPool2d(pool_size)

    def forward(self, patch_tokens):
        # patch_tokens: [B, T, C]，T = H * W（如 224/16 -> 14x14=196）
        B, T, C = patch_tokens.shape
        H = W = int(math.sqrt(T))
        assert H * W == T, f"patch token 数量 {T} 不是完全平方数，无法重排成空间网格"
        x = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        x = self.compress(x)
        x = torch.cat([self.dw_conv3(x), self.dw_conv5(x)], dim=1)
        x = self.fuse(x)
        x = self.pool(x)  # [B, hidden, P, P]
        return x.flatten(1)  # [B, hidden * P * P]，固定尺寸


class DINOFeatureContainer:
    """DINO 原始结果保存容器。

    每次 dino_encoder 前向时收集 DINOv2 处理后的原始特征（未经 GCN 等
    下游模块污染），连同坐标 / slide 名等元信息一起落盘为 .pt 文件，
    便于后续离线分析、二次建图或特征提取复用。
    同时在内存中缓存最近一次的原始结果（self.latest）供即时访问。
    """

    def __init__(self, save_dir, save_every=1, max_files=0):
        """
        Args:
            save_dir: 保存目录
            save_every: 每多少次调用保存一次（1 = 每次都存）
            max_files: 磁盘上最多保留的文件数，0 = 不限制（超出时删最早的）
        """
        self.save_dir = save_dir
        self.save_every = max(1, int(save_every))
        self.max_files = int(max_files)
        self.latest = {}
        self._call_count = 0
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"DINOFeatureContainer -- saving raw DINO results to {save_dir}")

    def save(self, results, meta=None):
        """保存一次前向的原始结果。

        Args:
            results: dict[str, Tensor]，各阶段的原始特征（自动 detach + 移回 CPU）
            meta: dict，元信息（如 coords / slide_name），非 Tensor 值原样保存
        """
        self._call_count += 1
        payload = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in results.items()
        }
        if meta:
            payload["meta"] = {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in meta.items()
            }
        self.latest = payload

        if self._call_count % self.save_every != 0:
            return
        save_path = os.path.join(self.save_dir, f"dino_raw_{self._call_count:08d}.pt")
        torch.save(payload, save_path)
        logger.info(f"DINOFeatureContainer -- saved {save_path}")

        # 可选：限制磁盘上保留的文件数量，删除最早的旧文件
        if self.max_files > 0:
            existing = sorted(
                f for f in os.listdir(self.save_dir) if f.startswith("dino_raw_") and f.endswith(".pt")
            )
            for old in existing[: max(0, len(existing) - self.max_files)]:
                os.remove(os.path.join(self.save_dir, old))


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

        # ---------- 修改 1：空间感知的子图特征聚合（替代原来的 patch mean）----------
        agg_hidden = cfg.gcn.spatial_agg_hidden
        agg_pool = cfg.gcn.spatial_agg_pool_size
        self.spatial_agg_out_dim = agg_hidden * agg_pool * agg_pool
        student_model_dict["spatial_agg"] = SpatialPatchAggregator(
            in_dim=cfg.dino.head_n_prototypes,
            hidden_dim=agg_hidden,
            pool_size=agg_pool,
        )
        # node feature = [global CLS mean, local CLS mean, 空间聚合特征] -> MLP
        student_model_dict["mlp"] = nn.Sequential(
            nn.Linear(cfg.dino.head_n_prototypes * 2 + self.spatial_agg_out_dim, 2048),
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

        # ---------- 修改 3：Graph Context Contrastive 投影头 ----------
        student_model_dict["projection"] = nn.Sequential(
            nn.Linear(cfg.gcn.emb_dim, cfg.gcn.emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.gcn.emb_dim, cfg.gcn.contrast_proj_dim),
        )

        # ---------- 修改 2：两阶段 Edge Learning 预测头 ----------
        # 第一阶段：Edge Existence Prediction，预测 P(E_ij = 1)
        student_model_dict["edge_existence_head"] = nn.Sequential(
            nn.Linear(cfg.gcn.emb_dim * 2, cfg.gcn.emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.gcn.emb_dim, 1),
        )
        # 第二阶段：Edge Type Prediction，仅在 E_ij = 1 时预测边类型
        # 0 = Spatial, 1 = Semantic, 2 = Spatial + Semantic
        student_model_dict["edge_type_head"] = nn.Sequential(
            nn.Linear(cfg.gcn.emb_dim * 2, cfg.gcn.emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.gcn.emb_dim, 3),
        )

        self.student = nn.ModuleDict(student_model_dict)

        # ---------- DINO 原始结果保存容器 ----------
        self.dino_feature_container = None
        if cfg.gcn.save_dino_features:
            self.dino_feature_container = DINOFeatureContainer(
                save_dir=os.path.join(cfg.train.output_dir, cfg.gcn.dino_feature_dir),
                save_every=cfg.gcn.dino_feature_save_every,
                max_files=cfg.gcn.dino_feature_max_files,
            )
        self._dino_raw_results = {}

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

        # 修改 1：不再对 patch feature 简单 mean，
        # 而是重排成空间网格后做多尺度 Depthwise Conv + 池化，保留局部空间结构
        # [n_global_crops, B, T, C] -> [n_global_crops*B, T, C]
        global_patch_grid_feat = self.student.spatial_agg(
            student_global_masked_patch_tokens_after_head.flatten(0, 1).float()
        )
        student_global_patch_spatial_feat = global_patch_grid_feat.view(
            n_global_crops, batch_size, -1
        ).mean(dim=0).to(student_global_mean_feat.dtype)

        cat_feat = torch.cat(
            [student_global_mean_feat, student_local_mean_feat, student_global_patch_spatial_feat],
            dim=-1,
        )

        node_features = self.student.mlp(cat_feat)

        # 收集 DINO 处理后的原始结果，供容器落盘（均为进入 GCN 前的特征）
        if self.dino_feature_container is not None:
            self._dino_raw_results = {
                "global_cls_after_head": student_global_cls_tokens_after_head,   # [2, B, head_dim]
                "local_cls_after_head": student_local_cls_tokens_after_head,     # [n_local, B, head_dim]
                "global_patch_after_head": student_global_masked_patch_tokens_after_head,  # [2, B, T, head_dim]
                "global_patch_spatial_feat": student_global_patch_spatial_feat,  # [B, agg_dim] 修改1的空间聚合输出
                "cat_feat": cat_feat,                                            # [B, 2*head_dim+agg_dim]
                "node_features": node_features,                                  # [B, emb_dim]
            }
        return node_features
        
    def build_spatial_graph(self, node_features, coords):
        """修改 3.2：仅根据子图在 WSI 中的真实坐标构建初始空间图。

        边是否存在只由空间邻接（d < radius）决定，
        不让 DINO similarity 参与初始建图。
        约定：edge_index 前 M 条为无向对 (i, j)，后 M 条为反向 (j, i)，
        M 为无向边数，后续对“无向对”的 mask 可直接同步到两个方向。
        """
        N = node_features.shape[0]
        device = node_features.device
        radius = self.cfg.gcn.spatial_radius

        coords = coords.to(device).float()
        dist_matrix = torch.cdist(coords, coords, p=2)
        with torch.no_grad():
            sim_matrix = F.cosine_similarity(
                node_features.unsqueeze(1), node_features.unsqueeze(0), dim=-1
            )

        adj = (dist_matrix < radius) & (~torch.eye(N, dtype=torch.bool, device=device))
        src, dst = adj.nonzero(as_tuple=True)
        keep = src < dst  # 只保留上三角，得到唯一无向边 (i < j)
        src, dst = src[keep], dst[keep]

        undirected_pairs = torch.stack([src, dst], dim=1)  # [M, 2]
        d = dist_matrix[src, dst] / radius  # 距离按 radius 归一化，数值更稳定
        s = sim_matrix[src, dst]
        edge_attr_half = torch.stack([d, s, torch.zeros_like(d)], dim=1)  # [M, 3]

        edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
        edge_attr = torch.cat([edge_attr_half, edge_attr_half], dim=0)
        return edge_index, edge_attr, undirected_pairs, coords

    def sample_view_masks(self, num_undirected_edges, device):
        """修改 3.4：对同一空间图随机丢边，生成两个不同的 Graph View。"""
        drop = self.cfg.gcn.view_edge_drop
        masks = []
        for _ in range(2):
            if num_undirected_edges == 0:
                masks.append(torch.zeros(0, dtype=torch.bool, device=device))
                continue
            keep = torch.rand(num_undirected_edges, device=device) >= drop
            if not keep.any():  # 保证视图里至少保留一条边
                keep[torch.randint(num_undirected_edges, (1,), device=device)] = True
            masks.append(keep)
        return masks

    def apply_view_mask(self, edge_index, edge_attr, keep_mask):
        """把无向对级别的 keep_mask 同步应用到两个方向的有向边上。"""
        keep_full = torch.cat([keep_mask, keep_mask])
        return edge_index[:, keep_full], edge_attr[keep_full]

    def graph_context_contrastive_loss(self, h1, h2):
        """修改 3.4/3.6：节点级 InfoNCE。

        Positive：同一节点在两个 Graph View 下的 context-aware 表示；
        Negative：图内其它（语义/空间上不同的）节点，
        避免所有节点退化成相同 embedding。
        """
        N = h1.shape[0]
        if N < 2:
            return h1.sum() * 0.0
        z1 = F.normalize(self.student.projection(h1), dim=-1)
        z2 = F.normalize(self.student.projection(h2), dim=-1)
        tau = self.cfg.gcn.contrast_temperature
        logits = z1 @ z2.t() / tau
        labels = torch.arange(N, device=z1.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        return loss

    def add_semantic_edges(self, context_h, edge_index, edge_attr, undirected_pairs, coords):
        """修改 3 的出口：用 context-aware 特征的相似度发现潜在语义关系。

        与原来直接用 DINOv2 feature 的 cos(x_i, x_j) 不同，这里比较的是
        sim(h_i, h_j)，其中 h 已包含各自组织邻域的上下文信息。
        为每条无向边给出类型标签：
            0 = Spatial（仅空间相邻）
            1 = Semantic（仅语义相似，空间上可以很远）
            2 = Spatial + Semantic（既相邻又语义相似）
        """
        N = context_h.shape[0]
        device = context_h.device
        radius = self.cfg.gcn.spatial_radius
        threshold = self.cfg.gcn.semantic_sim_threshold
        max_new = self.cfg.gcn.semantic_max_edges
        M = undirected_pairs.shape[0]

        type_labels = torch.zeros(M, dtype=torch.long, device=device)
        if M == 0 and N < 2:
            return edge_index, edge_attr, undirected_pairs, type_labels

        with torch.no_grad():
            sim = F.cosine_similarity(context_h.unsqueeze(1), context_h.unsqueeze(0), dim=-1)

        # 空间相邻且语义相似 -> 已有空间边升级为 Spatial+Semantic
        if M > 0:
            s_spatial = sim[undirected_pairs[:, 0], undirected_pairs[:, 1]].detach()
            type_labels[s_spatial > threshold] = 2

        # 非空间相邻但语义相似 -> 新增 Semantic 边
        spatial_adj = torch.zeros(N, N, dtype=torch.bool, device=device)
        if M > 0:
            spatial_adj[undirected_pairs[:, 0], undirected_pairs[:, 1]] = True
            spatial_adj[undirected_pairs[:, 1], undirected_pairs[:, 0]] = True
        cand = (sim > threshold) & (~spatial_adj) & (~torch.eye(N, dtype=torch.bool, device=device))
        cand = torch.triu(cand, diagonal=1)
        idx = cand.nonzero(as_tuple=False)  # [K, 2]

        if idx.shape[0] > max_new:
            vals = sim[idx[:, 0], idx[:, 1]].detach()
            idx = idx[vals.topk(max_new).indices]

        if idx.shape[0] > 0:
            si, sj = idx[:, 0], idx[:, 1]
            d = torch.sqrt(((coords[si] - coords[sj]) ** 2).sum(-1)) / radius
            s = sim[si, sj].detach()
            new_attr = torch.stack([d, s, torch.zeros_like(d)], dim=1)
            # 插入后保持约定：前一半 = (i, j)，后一半 = (j, i)
            edge_index = torch.cat(
                [edge_index[:, :M], torch.stack([si, sj]), torch.stack([sj, si]), edge_index[:, M:]],
                dim=1,
            )
            edge_attr = torch.cat([edge_attr[:M], new_attr, new_attr, edge_attr[M:]], dim=0)
            undirected_pairs = torch.cat([undirected_pairs, idx], dim=0)
            type_labels = torch.cat(
                [type_labels, torch.ones(idx.shape[0], dtype=torch.long, device=device)]
            )
        return edge_index, edge_attr, undirected_pairs, type_labels

    def mask_edges_and_sample_negatives(self, N, edge_index, edge_attr, undirected_pairs, type_labels):
        """修改 2 的数据准备：mask 一部分边作为待预测样本，并采样真正的负样本节点对。

        正样本：被 mask 掉的边（E_ij = 1）
        负样本：图中本来就没有任何边的节点对（E_ij = 0），
                让 existence prediction 拥有真正的正负样本。
        """
        device = edge_index.device
        M = undirected_pairs.shape[0]

        num_mask = min(M, max(1, int(M * self.cfg.gcn.edge_mask_ratio))) if M > 0 else 0
        perm = torch.randperm(M, device=device)
        mask_sel = perm[:num_mask]
        keep_mask = torch.ones(M, dtype=torch.bool, device=device)
        keep_mask[mask_sel] = False

        masked_edge_index, masked_edge_attr = self.apply_view_mask(edge_index, edge_attr, keep_mask)
        masked_pairs = undirected_pairs[mask_sel]          # [num_mask, 2]
        masked_type_labels = type_labels[mask_sel]         # [num_mask]

        # 负样本：不存在任何边的节点对（基于 mask 前的完整边集，避免和正样本重叠）
        num_neg = self.cfg.gcn.num_neg_pairs
        if N >= 2 and num_neg > 0:
            adj = torch.zeros(N, N, dtype=torch.bool, device=device)
            if M > 0:
                adj[undirected_pairs[:, 0], undirected_pairs[:, 1]] = True
                adj[undirected_pairs[:, 1], undirected_pairs[:, 0]] = True
            neg_chunks = []
            total = 0
            tries = 0
            while total < num_neg and tries < 100:
                tries += 1
                need = num_neg - total
                i = torch.randint(0, N, (need * 4,), device=device)
                j = torch.randint(0, N, (need * 4,), device=device)
                valid = (i != j) & (~adj[i, j])
                pairs = torch.stack([i[valid], j[valid]], dim=1)[:need]
                if pairs.shape[0] > 0:
                    neg_chunks.append(pairs)
                    total += pairs.shape[0]
            neg_pairs = (
                torch.cat(neg_chunks, dim=0)
                if neg_chunks
                else torch.zeros((0, 2), dtype=torch.long, device=device)
            )
        else:
            neg_pairs = torch.zeros((0, 2), dtype=torch.long, device=device)

        return masked_edge_index, masked_edge_attr, masked_pairs, masked_type_labels, neg_pairs

    def forward_backward(self, images):
        loss_dict = {}
        loss_accumulator = 0  # for backprop

        ### 修改 1：DINO encoder + 空间感知聚合 -> 子图级 node feature
        node_features = self.dino_encoder(images)
        coords = images["coords"]
        if coords.shape[0] == 2 * node_features.shape[0]:
            # 兼容每个子图给出两个 global crop 坐标的情况，取均值
            coords = coords.view(2, node_features.shape[0], 2).mean(dim=0)

        ### DINO 原始结果保存：将本次前向的原始特征连同坐标 / slide 名落盘
        if self.dino_feature_container is not None and self._dino_raw_results:
            self.dino_feature_container.save(
                self._dino_raw_results,
                meta={"coords": coords, "slide_name": images.get("slide_name")},
            )

        ### 修改 3.2：仅根据 WSI 坐标建立初始空间图
        edge_index, edge_attr, undirected_pairs, coords = self.build_spatial_graph(node_features, coords)

        ### 修改 3.3/3.4：两个 Graph View -> GNN -> Graph Context Contrastive Learning
        view_mask1, view_mask2 = self.sample_view_masks(undirected_pairs.shape[0], node_features.device)
        h1 = self.student.gcn(node_features, *self.apply_view_mask(edge_index, edge_attr, view_mask1))
        h2 = self.student.gcn(node_features, *self.apply_view_mask(edge_index, edge_attr, view_mask2))
        contrast_loss = self.graph_context_contrastive_loss(h1, h2)

        ### 修改 3 出口：context-aware 相似度 -> 潜在语义边（可能空间上很远）
        context_h = (h1 + h2) / 2
        edge_index, edge_attr, undirected_pairs, edge_type_labels = self.add_semantic_edges(
            context_h, edge_index, edge_attr, undirected_pairs, coords
        )

        ### 修改 2：Mask Graph -> GNN -> Edge Existence -> Edge Type
        (masked_edge_index, masked_edge_attr, masked_pairs,
         masked_type_labels, neg_pairs) = self.mask_edges_and_sample_negatives(
            node_features.shape[0], edge_index, edge_attr, undirected_pairs, edge_type_labels
        )
        node_rep = self.student.gcn(node_features, masked_edge_index, masked_edge_attr)

        # 第一阶段：Edge Existence Prediction（正样本 = 被 mask 的边，负样本 = 无边节点对）
        all_pairs = torch.cat([masked_pairs, neg_pairs], dim=0)
        if all_pairs.shape[0] > 0:
            exist_label = torch.cat([
                torch.ones(masked_pairs.shape[0], device=node_rep.device),
                torch.zeros(neg_pairs.shape[0], device=node_rep.device),
            ])
            edge_rep = torch.cat([node_rep[all_pairs[:, 0]], node_rep[all_pairs[:, 1]]], dim=-1)
            exist_logits = self.student.edge_existence_head(edge_rep).squeeze(-1)
            existence_loss = F.binary_cross_entropy_with_logits(exist_logits, exist_label)
        else:
            existence_loss = node_features.sum() * 0.0

        # 第二阶段：Edge Type Prediction，仅对 E_ij = 1 的边预测（训练时用真值正边）
        if masked_pairs.shape[0] > 0:
            pos_rep = torch.cat([node_rep[masked_pairs[:, 0]], node_rep[masked_pairs[:, 1]]], dim=-1)
            type_logits = self.student.edge_type_head(pos_rep)
            type_loss = F.cross_entropy(type_logits, masked_type_labels)
        else:
            type_loss = node_features.sum() * 0.0

        loss_accumulator += contrast_loss * self.cfg.gcn.contrast_weight
        loss_accumulator += existence_loss
        loss_accumulator += type_loss

        loss_dict["graph_contrastive"] = contrast_loss.detach()
        loss_dict["edge_existence"] = existence_loss.detach()
        loss_dict["edge_type"] = type_loss.detach()

        self.backprop_loss(loss_accumulator)
        self.fsdp_synchronize_streams()

        return loss_dict

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

