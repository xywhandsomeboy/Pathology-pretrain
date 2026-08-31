# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import math
import os

import torch
from torch import nn

from dinov2.models import build_model_from_cfg
from dinov2.utils.utils import has_batchnorms
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model

from dinov2.models.vision_transformer import BlockChunk
from dinov2.models.gcn import GNN
import torch.nn.functional as F


logger = logging.getLogger("dinov2")


class SpatialPatchAggregator(nn.Module):
    """Aggregate raw DINO patch tokens without destroying their 2-D layout."""

    def __init__(self, in_dim, hidden_dim=256, out_dim=1024):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1, groups=hidden_dim, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1, groups=hidden_dim, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, patch_tokens):
        # patch_tokens: [B, T, C], where T is the ViT token grid area.
        batch_size, num_tokens, channels = patch_tokens.shape
        grid_size = math.isqrt(num_tokens)
        if grid_size * grid_size != num_tokens:
            raise ValueError(
                f"patch token count {num_tokens} is not a square grid; "
                "spatial aggregation requires H * W tokens"
            )
        x = patch_tokens.transpose(1, 2).reshape(
            batch_size, channels, grid_size, grid_size
        )
        x = self.pool(self.spatial(self.compress(x))).flatten(1)
        return self.proj(x)


class NodeFeatureFusion(nn.Module):
    """Fuse the global DINO feature with a learnable spatial residual."""

    def __init__(self, global_dim, out_dim, spatial_scale_init=0.1):
        super().__init__()
        self.global_proj = (
            nn.Identity() if global_dim == out_dim else nn.Linear(global_dim, out_dim)
        )
        self.spatial_scale = nn.Parameter(torch.tensor(float(spatial_scale_init)))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, global_feature, spatial_feature):
        return self.norm(
            self.global_proj(global_feature) + self.spatial_scale * spatial_feature
        )


class LearnableMaskToken(nn.Module):
    """Replace selected nodes with a learned token instead of graph-wide mean."""

    def __init__(self, dim):
        super().__init__()
        self.token = nn.Parameter(torch.zeros(1, dim))
        nn.init.normal_(self.token, std=0.02)

    def forward(self, node_features, mask):
        return torch.where(mask.unsqueeze(-1), self.token.to(node_features.dtype), node_features)


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
        self.need_to_synchronize_fsdp_streams = True

        # One WSI image patch remains one GNN node. Raw 2-D DINO tokens are
        # aggregated inside that patch and fused with its global CLS feature.
        student_model_dict["spatial_agg"] = SpatialPatchAggregator(
            in_dim=embed_dim,
            hidden_dim=cfg.gcn.spatial_agg_hidden,
            out_dim=cfg.gcn.emb_dim,
        )
        student_model_dict["node_fusion"] = NodeFeatureFusion(
            global_dim=embed_dim,
            out_dim=cfg.gcn.emb_dim,
            spatial_scale_init=cfg.gcn.spatial_fusion_alpha,
        )
        student_model_dict["gcn"] = GNN(
            cfg.gcn.num_layer,
            cfg.gcn.emb_dim,
            JK=cfg.gcn.JK,
            drop_ratio=cfg.gcn.dropout_ratio,
            gnn_type=cfg.gcn.gnn_type,
            edge_dim=3
        )
        student_model_dict["mask_token"] = LearnableMaskToken(cfg.gcn.emb_dim)
        student_model_dict["node_decoder"] = nn.Sequential(
            nn.Linear(cfg.gcn.emb_dim, cfg.gcn.emb_dim),
            nn.GELU(),
            nn.Linear(cfg.gcn.emb_dim, cfg.gcn.emb_dim),
        )

        # ---------- 修改 3：Graph Context Contrastive 投影头 ----------
        student_model_dict["projection"] = nn.Sequential(
            nn.Linear(cfg.gcn.emb_dim, cfg.gcn.emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.gcn.emb_dim, cfg.gcn.contrast_proj_dim),
        )

        # Symmetric pair representation: [h_i+h_j, |h_i-h_j|, h_i*h_j].
        pair_dim = cfg.gcn.emb_dim * 3
        student_model_dict["edge_existence_head"] = nn.Sequential(
            nn.Linear(pair_dim, cfg.gcn.emb_dim),
            nn.GELU(),
            nn.Linear(cfg.gcn.emb_dim, 1),
        )
        student_model_dict["edge_type_head"] = nn.Sequential(
            nn.Linear(pair_dim, cfg.gcn.emb_dim),
            nn.GELU(),
            nn.Linear(cfg.gcn.emb_dim, 3),
        )
        student_model_dict["edge_weight_head"] = nn.Sequential(
            nn.Linear(pair_dim, cfg.gcn.emb_dim),
            nn.GELU(),
            nn.Linear(cfg.gcn.emb_dim, 2),
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

        logger.info(
            "DINO backbone and graph pretraining heads are built with %d-D node features",
            cfg.gcn.emb_dim,
        )

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def dino_encoder(self, images):
        n_global_crops = 2
        batch_size = images["collated_global_crops"].shape[0] // n_global_crops
        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        global_output = self.student.backbone(
            global_crops, masks=None, is_training=True
        )
        global_cls = global_output["x_norm_clstoken"].view(
            n_global_crops, batch_size, self.embed_dim
        )
        raw_patch_tokens = global_output["x_norm_patchtokens"]
        num_tokens = raw_patch_tokens.shape[1]
        raw_patch_tokens = raw_patch_tokens.view(
            n_global_crops, batch_size, num_tokens, self.embed_dim
        )

        global_feature = global_cls.mean(dim=0)
        spatial_feature = self.student.spatial_agg(
            raw_patch_tokens.flatten(0, 1).float()
        ).view(
            n_global_crops, batch_size, -1
        ).mean(dim=0).to(global_feature.dtype)
        node_features = self.student.node_fusion(global_feature, spatial_feature)

        if self.dino_feature_container is not None:
            self._dino_raw_results = {
                "global_cls": global_cls,
                "global_patch_tokens": raw_patch_tokens,
                "global_feature": global_feature,
                "spatial_feature": spatial_feature,
                "node_features": node_features,
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
        spatial_weight = 1.0 - dist_matrix[src, dst] / radius
        semantic_weight = sim_matrix[src, dst]
        edge_attr_half = torch.stack(
            [spatial_weight, semantic_weight, torch.zeros_like(spatial_weight)], dim=1
        )

        edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
        edge_attr = torch.cat([edge_attr_half, edge_attr_half], dim=0)
        return edge_index, edge_attr, undirected_pairs, coords

    def sample_view_masks(self, num_undirected_edges, device):
        """Optionally drop a very small number of edges in a weak view."""
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

    def make_noisy_view(self, node_features, edge_index, edge_attr, num_pairs):
        """Create a weak view while preserving the underlying tissue graph."""
        feature_noise = torch.randn_like(node_features) * self.cfg.gcn.feature_noise_std
        view_features = node_features + feature_noise

        feature_mask_ratio = self.cfg.gcn.contrast_feature_mask_ratio
        if feature_mask_ratio > 0:
            keep = torch.rand_like(view_features) >= feature_mask_ratio
            view_features = view_features * keep.to(view_features.dtype)

        view_attr = edge_attr.clone()
        if view_attr.numel() > 0 and self.cfg.gcn.edge_weight_noise_std > 0:
            # The last channel is a self-loop marker and must not be perturbed.
            noise = torch.randn_like(view_attr[:, :2]) * self.cfg.gcn.edge_weight_noise_std
            view_attr[:, :2] = view_attr[:, :2] * (1.0 + noise)
            view_attr[:, 0].clamp_(min=0.0)
            view_attr[:, 1].clamp_(min=-1.0, max=1.0)

        keep_mask = self.sample_view_masks(num_pairs, node_features.device)[0]
        view_index, view_attr = self.apply_view_mask(edge_index, view_attr, keep_mask)
        return view_features, view_index, view_attr

    def reliable_negative_mask(self, coords, visual_h, context_h, undirected_pairs=None):
        """Only mark pairs that are far and dissimilar in both semantic spaces."""
        num_nodes = visual_h.shape[0]
        if num_nodes < 2:
            return torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=visual_h.device)
        with torch.no_grad():
            distance = torch.cdist(coords.float(), coords.float())
            visual_sim = F.cosine_similarity(
                visual_h.unsqueeze(1), visual_h.unsqueeze(0), dim=-1
            )
            context_sim = F.cosine_similarity(
                context_h.unsqueeze(1), context_h.unsqueeze(0), dim=-1
            )
            mask = (
                (
                    distance
                    >= self.cfg.gcn.spatial_radius
                    * self.cfg.gcn.reliable_neg_distance_ratio
                )
                & (visual_sim <= self.cfg.gcn.reliable_neg_visual_sim_max)
                & (context_sim <= self.cfg.gcn.reliable_neg_context_sim_max)
            )
            mask.fill_diagonal_(False)
            if undirected_pairs is not None and undirected_pairs.numel() > 0:
                mask[undirected_pairs[:, 0], undirected_pairs[:, 1]] = False
                mask[undirected_pairs[:, 1], undirected_pairs[:, 0]] = False
        return mask

    def graph_context_contrastive_loss(self, h1, h2, reliable_negatives):
        """Masked InfoNCE: other nodes are negatives only when they are reliable."""
        num_nodes = h1.shape[0]
        if num_nodes == 0:
            return h1.sum() * 0.0
        z1 = F.normalize(self.student.projection(h1).float(), dim=-1)
        z2 = F.normalize(self.student.projection(h2).float(), dim=-1)
        tau = self.cfg.gcn.contrast_temperature
        logits = z1 @ z2.t() / tau
        allowed = reliable_negatives | torch.eye(num_nodes, dtype=torch.bool, device=z1.device)
        labels = torch.arange(num_nodes, device=z1.device)
        forward_loss = F.cross_entropy(logits.masked_fill(~allowed, float("-inf")), labels)
        backward_loss = F.cross_entropy(
            logits.t().masked_fill(~allowed.t(), float("-inf")), labels
        )
        alignment = (1.0 - (z1 * z2).sum(dim=-1)).mean()
        return (
            (forward_loss + backward_loss) * 0.5
            + self.cfg.gcn.contrast_alignment_weight * alignment
        )

    def _mask_target_count(self, num_nodes):
        if num_nodes < 2:
            return 0
        requested = max(1, round(num_nodes * self.cfg.gcn.node_mask_ratio))
        return min(num_nodes - 1, requested)

    def sample_node_reconstruction_masks(self, coords, undirected_pairs):
        """Return exact-ratio random, spatial-region, and random-walk masks."""
        num_nodes = coords.shape[0]
        device = coords.device
        target = self._mask_target_count(num_nodes)
        empty = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if target == 0:
            return {"random": empty, "region": empty.clone(), "random_walk": empty.clone()}

        random_mask = empty.clone()
        random_mask[torch.randperm(num_nodes, device=device)[:target]] = True

        # A nearest-neighbour region around one random seed is spatially contiguous
        # and always reaches the requested mask ratio.
        seed = torch.randint(num_nodes, (1,), device=device)
        seed_distance = torch.cdist(coords[seed].float(), coords.float()).squeeze(0)
        region_mask = empty.clone()
        region_mask[seed_distance.argsort()[:target]] = True

        adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
        if undirected_pairs.numel() > 0:
            adjacency[undirected_pairs[:, 0], undirected_pairs[:, 1]] = True
            adjacency[undirected_pairs[:, 1], undirected_pairs[:, 0]] = True
        walk_mask = empty.clone()
        current = torch.randint(num_nodes, (1,), device=device).squeeze(0)
        stagnant_steps = 0
        while int(walk_mask.sum()) < target:
            previous_count = int(walk_mask.sum())
            walk_mask[current] = True
            neighbours = adjacency[current].nonzero(as_tuple=False).flatten()
            if neighbours.numel() == 0 or stagnant_steps >= max(4, num_nodes):
                remaining = (~walk_mask).nonzero(as_tuple=False).flatten()
                if remaining.numel() == 0:
                    break
                current = remaining[
                    torch.randint(remaining.numel(), (1,), device=device)
                ].squeeze(0)
                stagnant_steps = 0
                continue
            current = neighbours[
                torch.randint(neighbours.numel(), (1,), device=device)
            ].squeeze(0)
            stagnant_steps = stagnant_steps + 1 if previous_count == int(walk_mask.sum()) else 0

        # A disconnected graph may force restarts; trim only as a safety guard.
        if int(walk_mask.sum()) > target:
            selected = walk_mask.nonzero(as_tuple=False).flatten()[:target]
            walk_mask.zero_()
            walk_mask[selected] = True
        return {"random": random_mask, "region": region_mask, "random_walk": walk_mask}

    def node_reconstruction_losses(self, node_features, edge_index, edge_attr, masks):
        losses = {}
        zero = node_features.sum() * 0.0
        for name, mask in masks.items():
            if not mask.any():
                losses[name] = zero
                continue
            masked_features = self.student.mask_token(node_features, mask)
            context = self.student.gcn(masked_features, edge_index, edge_attr)
            prediction = self.student.node_decoder(context[mask])
            target = node_features.detach()[mask]
            mse = F.mse_loss(prediction.float(), target.float())
            cosine = (1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)).mean()
            losses[name] = (
                self.cfg.gcn.node_recon_mse_weight * mse
                + self.cfg.gcn.node_recon_cos_weight * cosine
            )
        combined = torch.stack(list(losses.values())).mean() if losses else zero
        return combined, losses

    @staticmethod
    def pair_features(node_representation, pairs):
        left = node_representation[pairs[:, 0]]
        right = node_representation[pairs[:, 1]]
        return torch.cat([left + right, (left - right).abs(), left * right], dim=-1)

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
            distance = torch.sqrt(((coords[si] - coords[sj]) ** 2).sum(-1))
            spatial_weight = (1.0 - distance / radius).clamp(min=0.0)
            semantic_weight = sim[si, sj].detach()
            new_attr = torch.stack(
                [spatial_weight, semantic_weight, torch.zeros_like(spatial_weight)], dim=1
            )
            edge_attr_half = torch.cat([edge_attr[:M], new_attr], dim=0)
            undirected_pairs = torch.cat([undirected_pairs, idx], dim=0)
            src, dst = undirected_pairs[:, 0], undirected_pairs[:, 1]
            # Preserve the invariant used by every graph masking operation:
            # first half forward pairs, second half the same pairs reversed.
            edge_index = torch.cat(
                [torch.stack([src, dst]), torch.stack([dst, src])], dim=1
            )
            edge_attr = torch.cat([edge_attr_half, edge_attr_half], dim=0)
            type_labels = torch.cat(
                [type_labels, torch.ones(idx.shape[0], dtype=torch.long, device=device)]
            )
        return edge_index, edge_attr, undirected_pairs, type_labels

    def mask_edges_and_sample_negatives(
        self, edge_index, edge_attr, undirected_pairs, type_labels, reliable_negatives
    ):
        """Mask positive edges and sample only explicitly reliable non-edges."""
        device = edge_index.device
        num_edges = undirected_pairs.shape[0]

        num_mask = (
            min(num_edges, max(1, round(num_edges * self.cfg.gcn.edge_mask_ratio)))
            if num_edges > 0
            else 0
        )
        perm = torch.randperm(num_edges, device=device)
        mask_sel = perm[:num_mask]
        keep_mask = torch.ones(num_edges, dtype=torch.bool, device=device)
        keep_mask[mask_sel] = False

        masked_edge_index, masked_edge_attr = self.apply_view_mask(edge_index, edge_attr, keep_mask)
        masked_pairs = undirected_pairs[mask_sel]
        masked_type_labels = type_labels[mask_sel]
        masked_edge_targets = edge_attr[:num_edges][mask_sel, :2]

        candidates = torch.triu(reliable_negatives, diagonal=1).nonzero(as_tuple=False)
        if candidates.shape[0] > 0:
            candidates = candidates[torch.randperm(candidates.shape[0], device=device)]
        max_negatives = self.cfg.gcn.num_neg_pairs
        if num_mask > 0:
            max_negatives = min(max_negatives, num_mask * self.cfg.gcn.negatives_per_positive)
        neg_pairs = candidates[:max_negatives]

        return (
            masked_edge_index,
            masked_edge_attr,
            masked_pairs,
            masked_type_labels,
            masked_edge_targets,
            neg_pairs,
        )

    def forward_backward(self, images):
        loss_dict = {}

        # 1) One image patch -> one spatially aggregated DINO node feature.
        node_features = self.dino_encoder(images)
        zero = node_features.sum() * 0.0
        coords = images["coords"]
        if coords.shape[0] == 2 * node_features.shape[0]:
            coords = coords.view(2, node_features.shape[0], 2).mean(dim=0)

        ### DINO 原始结果保存：将本次前向的原始特征连同坐标 / slide 名落盘
        if self.dino_feature_container is not None and self._dino_raw_results:
            self.dino_feature_container.save(
                self._dino_raw_results,
                meta={"coords": coords, "slide_name": images.get("slide_name")},
            )

        # Initial graph is spatial only; DINO similarity is an edge attribute,
        # not a criterion for whether the initial edge exists.
        edge_index, edge_attr, undirected_pairs, coords = self.build_spatial_graph(node_features, coords)

        # 2) Graph-MAE: all three masks use the same learned mask token and
        # node reconstruction head, with both MSE and cosine objectives.
        node_masks = self.sample_node_reconstruction_masks(coords, undirected_pairs)
        node_recon_loss, _ = self.node_reconstruction_losses(
            node_features, edge_index, edge_attr, node_masks
        )

        # 3) Two weak noisy views preserve topology by default. Feature/edge
        # noise and a small channel mask provide the contrastive perturbation.
        view1 = self.make_noisy_view(
            node_features, edge_index, edge_attr, undirected_pairs.shape[0]
        )
        view2 = self.make_noisy_view(
            node_features, edge_index, edge_attr, undirected_pairs.shape[0]
        )
        h1 = self.student.gcn(*view1)
        h2 = self.student.gcn(*view2)
        context_h = (h1 + h2) / 2
        contrast_negatives = self.reliable_negative_mask(
            coords, node_features.detach(), context_h.detach(), undirected_pairs
        )
        contrast_loss = self.graph_context_contrastive_loss(
            h1, h2, contrast_negatives
        )

        # Context-aware similarity provides provisional semantic relation labels.
        edge_index, edge_attr, undirected_pairs, edge_type_labels = self.add_semantic_edges(
            context_h, edge_index, edge_attr, undirected_pairs, coords
        )

        # 4) Edge learning first predicts existence with positive and reliable
        # negative pairs, then relation type and continuous attributes for positives.
        edge_negatives = self.reliable_negative_mask(
            coords, node_features.detach(), context_h.detach(), undirected_pairs
        )
        (
            masked_edge_index,
            masked_edge_attr,
            masked_pairs,
            masked_type_labels,
            masked_edge_targets,
            neg_pairs,
        ) = self.mask_edges_and_sample_negatives(
            edge_index,
            edge_attr,
            undirected_pairs,
            edge_type_labels,
            edge_negatives,
        )
        node_rep = self.student.gcn(node_features, masked_edge_index, masked_edge_attr)

        all_pairs = torch.cat([masked_pairs, neg_pairs], dim=0)
        # Existence is a binary task only when both classes are available.
        # Uncertain non-edges are deliberately not used as fallback negatives.
        if masked_pairs.shape[0] > 0 and neg_pairs.shape[0] > 0:
            exist_label = torch.cat([
                torch.ones(masked_pairs.shape[0], device=node_rep.device),
                torch.zeros(neg_pairs.shape[0], device=node_rep.device),
            ])
            edge_rep = self.pair_features(node_rep, all_pairs)
            exist_logits = self.student.edge_existence_head(edge_rep).squeeze(-1)
            existence_loss = F.binary_cross_entropy_with_logits(exist_logits, exist_label)
        else:
            existence_loss = zero

        if masked_pairs.shape[0] > 0:
            pos_rep = self.pair_features(node_rep, masked_pairs)
            type_logits = self.student.edge_type_head(pos_rep)
            type_loss = F.cross_entropy(type_logits, masked_type_labels)
            predicted_edge_weights = self.student.edge_weight_head(pos_rep)
            edge_weight_loss = F.mse_loss(
                predicted_edge_weights.float(), masked_edge_targets.float()
            )
        else:
            type_loss = zero
            edge_weight_loss = zero

        weighted_losses = {
            "node_reconstruction": node_recon_loss * self.cfg.gcn.node_recon_weight,
            "graph_contrastive": contrast_loss * self.cfg.gcn.contrast_weight,
            "edge_existence": existence_loss * self.cfg.gcn.edge_existence_weight,
            "edge_relation": type_loss * self.cfg.gcn.edge_relation_weight,
            "edge_weight": edge_weight_loss * self.cfg.gcn.edge_weight_weight,
        }
        loss_accumulator = sum(weighted_losses.values(), zero)
        loss_dict.update({name: loss.detach() for name, loss in weighted_losses.items()})

        self.backprop_loss(loss_accumulator)
        self.fsdp_synchronize_streams()

        return loss_dict

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            backbone_streams = self.student.backbone._streams
            for name, module in self.student.items():
                if name != "backbone" and hasattr(module, "_streams"):
                    module._streams = backbone_streams
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
