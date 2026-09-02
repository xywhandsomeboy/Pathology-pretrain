import logging
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from dinov2.models import build_model_from_cfg
from dinov2.models.gcn import GNNChunk
from dinov2.utils.utils import has_batchnorms
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler

logger = logging.getLogger("gnn")


class LearnableMaskToken(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.token = nn.Parameter(torch.empty(1, dim))
        nn.init.normal_(self.token, std=0.02)

    def forward(self, x, mask):
        return torch.where(mask.unsqueeze(-1), self.token.to(x.dtype), x)


class GCNMetaArch(nn.Module):
    """Offline graph pretraining aligned with the finetune graph objectives."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None
        gcn, dim = build_model_from_cfg(cfg)
        pair_dim = dim * 3
        self.edge_weight_objective = bool(
            getattr(cfg.gcn, "edge_weight_objective", True)
        )
        self.edge_existence_objective = bool(
            getattr(cfg.gcn, "edge_existence_objective", True)
        )
        if self.edge_weight_objective and int(cfg.gcn.edge_dim) <= 0:
            raise ValueError("The edge-weight objective requires edge_dim > 0")
        if (
            not self.edge_existence_objective
            and float(cfg.gcn.edge_existence_weight) != 0.0
        ):
            raise ValueError(
                "edge_existence_weight must be zero when the edge-existence "
                "objective is disabled"
            )
        student_modules = {
            "gcn": gcn,
            "mask_token": LearnableMaskToken(dim),
            "node_decoder": nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)),
            "projection": nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, cfg.gcn.contrast_proj_dim)),
        }
        if self.edge_existence_objective:
            student_modules["edge_existence_head"] = nn.Sequential(
                nn.Linear(pair_dim, dim), nn.GELU(), nn.Linear(dim, 1)
            )
        if self.edge_weight_objective:
            student_modules["edge_weight_head"] = nn.Sequential(
                nn.Linear(pair_dim, dim),
                nn.GELU(),
                nn.Linear(dim, cfg.gcn.edge_dim),
            )
        self.student = nn.ModuleDict(student_modules)
        self.need_to_synchronize_fsdp_streams = True
        logger.info("Offline graph pretraining model built with %d-D nodes", dim)

    def forward(self, inputs):
        raise NotImplementedError

    @torch.no_grad()
    def extract_context(self, graph):
        """Return one contextualized feature per node without masking/noise.

        This is the deterministic branch-1 output consumed by the segmentation
        decoder. Call it on the complete WSI graph; random training subgraphs
        would break the one-to-one mapping to raw patches.
        """
        device = next(self.student.parameters()).device
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        expected_edge_mode = str(
            getattr(self.cfg.gcn, "context_edge_mode", "dual")
        )
        declared_edge_mode = getattr(graph, "edge_mode", None)
        if (
            declared_edge_mode is not None
            and str(declared_edge_mode) != expected_edge_mode
        ):
            raise ValueError(
                f"Expected context graph edge_mode={expected_edge_mode!r}, got "
                f"{str(declared_edge_mode)!r}"
            )
        use_edge_attr = bool(getattr(self.cfg.gcn, "context_use_edge_attr", True))
        edge_attr = getattr(graph, "edge_attr", None)
        if use_edge_attr:
            if edge_attr is None:
                edge_attr = torch.ones(
                    (edge_index.size(1), self.cfg.gcn.edge_dim),
                    device=device,
                    dtype=x.dtype,
                )
            else:
                edge_attr = edge_attr.to(device)
            if edge_attr.ndim == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            if edge_attr.size(-1) != self.cfg.gcn.edge_dim:
                raise ValueError(
                    f"Expected {self.cfg.gcn.edge_dim}-D edge_attr, got {edge_attr.size(-1)}"
                )
        else:
            edge_attr = None
        # Move only graph-computation tensors. Patch identities and any legacy
        # decoder metadata remain on CPU.
        return self.student.gcn(
            x,
            edge_index,
            edge_attr,
            use_edge_attr=use_edge_attr,
        )

    def backprop_loss(self, loss):
        (self.fp16_scaler.scale(loss) if self.fp16_scaler is not None else loss).backward()

    @staticmethod
    def _unique_pairs(edge_index):
        if edge_index.numel() == 0:
            return edge_index.new_empty((0, 2))
        pairs = torch.sort(edge_index.t(), dim=1).values
        return torch.unique(pairs[pairs[:, 0] != pairs[:, 1]], dim=0)

    def _node_masks(self, graph, pairs):
        n, device = graph.x.size(0), graph.x.device
        target = 0 if n < 2 else min(n - 1, max(1, round(n * self.cfg.gcn.node_mask_ratio)))
        empty = torch.zeros(n, dtype=torch.bool, device=device)
        if target == 0:
            return {name: empty.clone() for name in ("random", "region", "random_walk")}
        random_mask = empty.clone()
        random_mask[torch.randperm(n, device=device)[:target]] = True
        region_mask = empty.clone()
        if getattr(graph, "pos", None) is not None:
            seed = torch.randint(n, (1,), device=device)
            distance = torch.cdist(graph.pos[seed].float(), graph.pos.float()).squeeze(0)
            region_mask[distance.argsort()[:target]] = True
        else:
            region_mask[torch.randperm(n, device=device)[:target]] = True
        adjacency = [[] for _ in range(n)]
        for left, right in pairs.detach().cpu().tolist():
            adjacency[left].append(right)
            adjacency[right].append(left)
        selected, current = set(), int(torch.randint(n, (1,)).item())
        while len(selected) < target:
            selected.add(current)
            candidates = [v for v in adjacency[current] if v not in selected]
            if candidates:
                current = candidates[int(torch.randint(len(candidates), (1,)).item())]
            else:
                remaining = list(set(range(n)) - selected)
                if not remaining:
                    break
                current = remaining[int(torch.randint(len(remaining), (1,)).item())]
        walk_mask = empty.clone()
        walk_mask[torch.tensor(list(selected), device=device)] = True
        return {"random": random_mask, "region": region_mask, "random_walk": walk_mask}

    def _node_reconstruction(self, graph, pairs):
        losses, details = [], {}
        for name, mask in self._node_masks(graph, pairs).items():
            if mask.any():
                h = self.student.gcn(self.student.mask_token(graph.x, mask), graph.edge_index, graph.edge_attr)
                pred, target = self.student.node_decoder(h[mask]), graph.x.detach()[mask]
                mse = F.mse_loss(pred.float(), target.float())
                cosine = (1 - F.cosine_similarity(pred.float(), target.float(), dim=-1)).mean()
                loss = self.cfg.gcn.node_recon_mse_weight * mse + self.cfg.gcn.node_recon_cos_weight * cosine
            else:
                loss = graph.x.sum() * 0.0
            losses.append(loss)
            details[f"node_reconstruction_{name}"] = loss
        return torch.stack(losses).mean(), details

    def _contrast_edge_keep_mask(self, edge_index, num_nodes):
        """Drop undirected edge pairs without stranding low-degree nodes."""
        keep = torch.ones(
            edge_index.size(1), dtype=torch.bool, device=edge_index.device
        )
        drop_ratio = float(self.cfg.gcn.view_edge_drop)
        if edge_index.numel() == 0 or drop_ratio <= 0:
            return keep

        source, target = edge_index
        non_loop = source != target
        if not non_loop.any():
            return keep

        edge_slots = non_loop.nonzero(as_tuple=False).flatten()
        left = torch.minimum(source[non_loop], target[non_loop])
        right = torch.maximum(source[non_loop], target[non_loop])
        pair_keys = left * num_nodes + right
        unique_keys, inverse = torch.unique(pair_keys, return_inverse=True)
        pair_left = torch.div(unique_keys, num_nodes, rounding_mode="floor")
        pair_right = torch.remainder(unique_keys, num_nodes)

        # Sample once per undirected pair so i->j and j->i are changed together.
        drop_pair = torch.rand(
            unique_keys.numel(), device=edge_index.device
        ) < drop_ratio

        min_neighbors = max(
            0, int(getattr(self.cfg.gcn, "contrast_min_neighbors", 0))
        )
        if min_neighbors and drop_pair.any():
            degree = torch.zeros(
                num_nodes, dtype=torch.long, device=edge_index.device
            )
            ones = torch.ones_like(pair_left)
            degree.scatter_add_(0, pair_left, ones)
            degree.scatter_add_(0, pair_right, ones)

            dropped_degree = torch.zeros_like(degree)
            dropped_left = pair_left[drop_pair]
            dropped_right = pair_right[drop_pair]
            dropped_degree.scatter_add_(0, dropped_left, torch.ones_like(dropped_left))
            dropped_degree.scatter_add_(0, dropped_right, torch.ones_like(dropped_right))
            required = degree.clamp(max=min_neighbors)
            underconnected = degree - dropped_degree < required
            if underconnected.any():
                # Restore every sampled pair incident to an under-connected
                # node. This is conservative and keeps the operation vectorized
                # for large WSI graphs.
                drop_pair &= ~(
                    underconnected[pair_left] | underconnected[pair_right]
                )

        keep[edge_slots] = ~drop_pair[inverse]
        return keep

    def _noisy_view(self, graph):
        x = graph.x + torch.randn_like(graph.x) * self.cfg.gcn.feature_noise_std
        if self.cfg.gcn.contrast_feature_mask_ratio > 0:
            x *= (torch.rand_like(x) >= self.cfg.gcn.contrast_feature_mask_ratio).to(x.dtype)
        edge_attr = (
            graph.edge_attr.clone() if graph.edge_attr is not None else None
        )
        if (
            edge_attr is not None
            and edge_attr.numel()
            and self.cfg.gcn.edge_weight_noise_std > 0
        ):
            edge_attr *= 1 + torch.randn_like(edge_attr) * self.cfg.gcn.edge_weight_noise_std
        keep = self._contrast_edge_keep_mask(graph.edge_index, graph.x.size(0))
        kept_edge_attr = edge_attr[keep] if edge_attr is not None else None
        return self.student.gcn(x, graph.edge_index[:, keep], kept_edge_attr)

    def _reliable_negatives(self, graph, visual_h, context_h, pairs):
        visual = F.normalize(visual_h.float(), dim=-1)
        context = F.normalize(context_h.float(), dim=-1)
        mask = ((visual @ visual.t()) <= self.cfg.gcn.reliable_neg_visual_sim_max) & ((context @ context.t()) <= self.cfg.gcn.reliable_neg_context_sim_max)
        if getattr(graph, "pos", None) is not None and pairs.numel():
            edge_distance = (graph.pos[pairs[:, 0]].float() - graph.pos[pairs[:, 1]].float()).norm(dim=-1)
            reference = edge_distance.median().clamp_min(1e-6)
            mask &= torch.cdist(graph.pos.float(), graph.pos.float()) >= reference * self.cfg.gcn.reliable_neg_distance_ratio
        mask.fill_diagonal_(False)
        if pairs.numel():
            mask[pairs[:, 0], pairs[:, 1]] = False
            mask[pairs[:, 1], pairs[:, 0]] = False
        return mask

    def _contrastive_loss(self, h1, h2, negatives):
        z1 = F.normalize(self.student.projection(h1).float(), dim=-1)
        z2 = F.normalize(self.student.projection(h2).float(), dim=-1)
        logits = z1 @ z2.t() / self.cfg.gcn.contrast_temperature
        allowed = negatives | torch.eye(z1.size(0), dtype=torch.bool, device=z1.device)
        labels = torch.arange(z1.size(0), device=z1.device)
        nce = (F.cross_entropy(logits.masked_fill(~allowed, -torch.inf), labels) + F.cross_entropy(logits.t().masked_fill(~allowed.t(), -torch.inf), labels)) / 2
        return nce + self.cfg.gcn.contrast_alignment_weight * (1 - (z1 * z2).sum(-1)).mean()

    @staticmethod
    def _pair_features(h, pairs):
        left, right = h[pairs[:, 0]], h[pairs[:, 1]]
        return torch.cat((left + right, (left - right).abs(), left * right), dim=-1)

    def _edge_losses(self, graph, pairs, negatives):
        zero = graph.x.sum() * 0.0
        if (
            not pairs.numel()
            or not (
                self.edge_existence_objective or self.edge_weight_objective
            )
        ):
            return zero, zero
        count = min(pairs.size(0), max(1, round(pairs.size(0) * self.cfg.gcn.edge_mask_ratio)))
        positive = pairs[torch.randperm(pairs.size(0), device=graph.x.device)[:count]]
        selected_key = positive[:, 0] * graph.x.size(0) + positive[:, 1]
        directed = torch.sort(graph.edge_index.t(), dim=1).values
        directed_key = directed[:, 0] * graph.x.size(0) + directed[:, 1]
        keep = ~torch.isin(directed_key, selected_key)
        kept_edge_attr = (
            graph.edge_attr[keep] if graph.edge_attr is not None else None
        )
        h = self.student.gcn(graph.x, graph.edge_index[:, keep], kept_edge_attr)
        existence = zero
        if self.edge_existence_objective:
            candidates = torch.triu(negatives, diagonal=1).nonzero()
            if candidates.numel():
                candidates = candidates[
                    torch.randperm(candidates.size(0), device=graph.x.device)
                ]
            negative = candidates[
                : min(
                    candidates.size(0),
                    count * self.cfg.gcn.negatives_per_positive,
                )
            ]
            if negative.numel():
                all_pairs = torch.cat((positive, negative))
                labels = torch.cat(
                    (
                        torch.ones(count, device=h.device),
                        torch.zeros(negative.size(0), device=h.device),
                    )
                )
                logits = self.student.edge_existence_head(
                    self._pair_features(h, all_pairs)
                ).squeeze(-1)
                existence = F.binary_cross_entropy_with_logits(logits, labels)
        if not self.edge_weight_objective:
            return existence, zero
        targets = []
        for pair in positive:
            match = (directed == pair).all(dim=1).nonzero(as_tuple=False)[0, 0]
            targets.append(graph.edge_attr[match])
        predicted = self.student.edge_weight_head(self._pair_features(h, positive))
        return existence, F.mse_loss(predicted.float(), torch.stack(targets).float())

    def forward_backward(self, images):
        source_graph = images["original_graph"]
        device = next(self.student.parameters()).device
        # Do not call PyG Data.to(device): decoder metadata (especially
        # dense_tokens) is not part of graph pretraining and can exhaust VRAM.
        graph = SimpleNamespace(
            x=source_graph.x.to(device),
            edge_index=source_graph.edge_index.to(device),
            edge_attr=(
                source_graph.edge_attr.to(device)
                if getattr(source_graph, "edge_attr", None) is not None
                else None
            ),
            pos=(
                source_graph.pos.to(device)
                if getattr(source_graph, "pos", None) is not None
                else None
            ),
        )
        expected_edge_dim = int(self.cfg.gcn.edge_dim)
        if graph.edge_attr is None:
            if expected_edge_dim != 0:
                graph.edge_attr = torch.ones(
                    (graph.edge_index.size(1), expected_edge_dim),
                    device=graph.x.device,
                )
        else:
            if graph.edge_attr.ndim == 1:
                graph.edge_attr = graph.edge_attr.unsqueeze(-1)
            if graph.edge_attr.size(-1) != expected_edge_dim:
                raise ValueError(
                    f"Expected {expected_edge_dim}-D edge_attr, got "
                    f"{graph.edge_attr.size(-1)}. Rebuild old .pt graphs with "
                    "build_graphs.py using the edge mode required by this variant."
                )
        pairs = self._unique_pairs(graph.edge_index)
        node_loss, details = self._node_reconstruction(graph, pairs)
        h1, h2 = self._noisy_view(graph), self._noisy_view(graph)
        context = (h1 + h2) / 2
        negatives = self._reliable_negatives(graph, graph.x.detach(), context.detach(), pairs)
        contrast_loss = self._contrastive_loss(h1, h2, negatives)
        existence_loss, weight_loss = self._edge_losses(graph, pairs, negatives)
        total = self.cfg.gcn.node_recon_weight * node_loss + self.cfg.gcn.contrast_weight * contrast_loss + self.cfg.gcn.edge_existence_weight * existence_loss + self.cfg.gcn.edge_weight_weight * weight_loss
        losses = {**details, "node_reconstruction": node_loss, "graph_contrastive": contrast_loss, "edge_existence": existence_loss, "edge_weight": weight_loss, "total_loss": total}
        self.backprop_loss(total)
        self.fsdp_synchronize_streams()
        return losses

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            self.need_to_synchronize_fsdp_streams = False

    def train(self, mode=True):
        return super().train(mode)

    def get_maybe_fused_params_for_submodel(self, model):
        # ``fuse_params_groups`` returns a ``dict_values`` view.  Materialize
        # it here so parameter groups from the independent student modules can
        # be concatenated below.  This changes only the container type; group
        # membership and every optimizer option remain identical.
        groups = list(fuse_params_groups(get_params_groups_with_decay(model=model, lr_decay_rate=self.cfg.optim.layerwise_decay, patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult)))
        for group in groups:
            group["foreach"] = True
        return groups

    def get_params_groups(self):
        return sum((self.get_maybe_fused_params_for_submodel(model) for model in self.student.values()), [])

    def prepare_for_distributed_training(self):
        if has_batchnorms(self.student):
            raise NotImplementedError
        for name, model in self.student.items():
            self.student[name] = get_fsdp_wrapper(self.cfg.compute_precision.student[name], modules_to_wrap={GNNChunk})(model)
