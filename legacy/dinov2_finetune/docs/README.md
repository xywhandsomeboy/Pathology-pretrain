# dinov2_finetune — GCN × DINOv2 病理图自监督学习

> CerviPath 宫颈癌病理分析流水线的图学习项目：以预训练好的 **DINOv2 作为视觉特征提取器**
> （不讨论 DINOv2 本身的预训练），将 **一张从大病理 WSI 切出的子图 = 一个 GNN 节点**，
> 通过空间图建模与自监督代理任务，让模型同时学到
> **局部视觉 + 空间结构 + 组织级语义关系**。

---

## 1. 核心思想

原模型更接近「先人工定义关系，再让 GNN 预测关系」；本项目改造为：

> **先利用真实空间结构让 GNN 理解组织上下文，再学习语义关系，
> 同时让模型自己学习「是否应该建立边」。**

整体流水线：

```text
                大病理 WSI
                     ↓
               病理子图（每个子图一个节点）
                     ↓
                  DINOv2（已训练好的权重，冻结/初始化使用）
                     ↓
        ┌─────────────────────┐
        │ Spatial Feature     │   修改 1：空间感知特征聚合
        │ Aggregation / Conv  │   （替代原来的 patch mean）
        └──────────┬──────────┘
                   ↓
             Node Features
                   ↓
          根据 WSI 真实坐标建立空间图
                   ↓
             Spatial Graph
                   ↓
                  GNN
             ↙           ↘
    Context Contrastive    Edge Learning（修改 2）
    （修改 3）                 ↓
             ↓               Mask Edge
    Context-aware feature      ↓
             ↓                GNN
      Semantic Relation        ↓
             ↓            Edge Existence（P(E_ij=1)）
        Semantic Edge          ↓
                          Edge Type（S / M / S+M）
```

---

## 2. 三个修改点

### 修改 1：子图特征的空间感知聚合（Node Representation）

**原实现**：DINO 的 global CLS mean、local CLS mean、global patch mean 三者相加取平均，
再经 MLP 得到节点特征 —— 简单 mean 把子图内部的空间结构（哪个 patch 在左上、哪个在右下）压掉了。

**新实现**（`SpatialPatchAggregator`，位于 `dinov2/train/gcn_meta_arch.py`）：

```text
DINO patch tokens → 重排成 14×14 空间网格 → 1×1 通道压缩（65536→256）
                  → 多尺度 Depthwise Conv（3×3 局部结构 + 5×5 大感受野）
                  → 自适应池化下采样 → 固定尺寸向量
```

最终节点特征 = `[global CLS mean, local CLS mean, 空间聚合特征]` → MLP，
仍然保持 **1 张子图 = 1 个 GNN 节点**，但保留了局部空间结构与多尺度语义。

### 修改 2：两阶段 Edge Learning

**原实现**：规则建边（`d < 50` 或 `sim > 0.7`）→ mask 边 → GNN → 单个 4 分类头预测边类型。
问题：把「有没有关系」和「是什么关系」混在一起，模型从未真正学习「应不应该建边」。

**新实现**：

```text
             h_i, h_j
                 ↓
        ┌────────┴────────┐
        ↓                 ↓
   Edge Existence      Edge Type（仅当 E_ij = 1）
      0 / 1           S / M / S+M
```

- **第一阶段 Edge Existence**：`edge_existence_head` 预测 \(P(E_{ij}=1)\)（BCE 损失）。
  正样本 = 被 mask 的真实边；**负样本 = 图中本来就没有任何边的节点对**（`num_neg_pairs` 采样），
  让存在性预测拥有真正的正负样本。
- **第二阶段 Edge Type**：`edge_type_head` 只对 \(E_{ij}=1\) 的边预测 3 类类型：
  `0 = Spatial`（仅空间相邻）、`1 = Semantic`（仅语义相似，空间上可以很远）、`2 = Spatial+Semantic`。

### 修改 3：Graph Context Contrastive Learning

解决「单张子图之间的相似度不能代表它在整个组织中的语义」：希望得到
\(h_i = f(x_i, \text{邻域}_i)\) 而不是 \(h_i = x_i\)。

1. **仅根据空间关系建立初始图**：用子图在 WSI 中的真实坐标 \(p_i=(x_i,y_i)\) 构建
   \(E_{spatial}\)（邻接半径 `spatial_radius`），**不让 DINO 相似度参与初始建图**。
2. **GNN 提取上下文特征**：\(H = \text{GNN}(X, E_{spatial})\)，此时
   \(h_A = f(x_A, x_B, x_D, \ldots)\) 已包含 A 周围的组织信息（context-aware）。
3. **双视图一致性对比**：对同一空间图随机丢边生成两个 Graph View，分别过 GNN 得
   \(h_i^{(1)}, h_i^{(2)}\)，用节点级 **InfoNCE** 拉近同一节点的两个视图：
   - **Positive**：同一个节点在不同 view 下的表示（同一组织区域，邻接略有扰动也应相似）；
   - **Negative**：图内其它节点，防止所有节点坍缩成相同 embedding。
4. **出口 — 发现语义边**：用 context-aware 特征的 \(\text{sim}(h_i, h_j) > \texttt{semantic_sim_threshold}\)
   补充 Semantic 边（两个子图可能空间上很远，但各自组织上下文语义一致）；
   空间相邻且语义相似的空间边升级为 `Spatial+Semantic`。
   这比直接 \(\cos(x_i, x_j)\) 更符合「结合组织上下文后的语义相似性」。

---

## 3. 训练损失

`forward_backward` 每个迭代返回三项损失（分别记录在训练日志中）：

| 损失键 | 含义 | 形式 |
|---|---|---|
| `graph_contrastive` | 图上下文对比学习（×`contrast_weight`） | 对称 InfoNCE |
| `edge_existence` | 边存在性预测 | BCEWithLogits |
| `edge_type` | 边类型预测（仅正边） | CrossEntropy（3 类） |

---

## 4. DINO 原始结果保存容器（DINOFeatureContainer）

`dino_encoder` 每次前向都会把 **进入 GCN 前的原始结果**收集进容器并落盘为 `.pt`：

| 键 | 内容 | 形状 |
|---|---|---|
| `global_cls_after_head` | global CLS tokens（DINO head 后） | `[2, B, head_dim]` |
| `local_cls_after_head` | local CLS tokens | `[n_local, B, head_dim]` |
| `global_patch_after_head` | global patch tokens（保留空间维度） | `[2, B, T, head_dim]` |
| `global_patch_spatial_feat` | 修改 1 的空间聚合输出 | `[B, agg_dim]` |
| `cat_feat` | MLP 前的拼接特征 | `[B, 2×head_dim+agg_dim]` |
| `node_features` | 最终子图级节点特征 | `[B, emb_dim]` |
| `meta.coords` / `meta.slide_name` | 子图坐标 / WSI 名 | — |

文件按调用序号递增保存至 `{output_dir}/dino_raw_features/dino_raw_xxxxxxxx.pt`，
自动 detach + 移回 CPU；容器同时在内存中缓存最近一次结果（`model.dino_feature_container.latest`）。

---

## 5. 相关文件结构（本项目改动涉及的文件，完整不省略）

```text
dinov2_finetune/
├── pretrain.sh                          # SLURM 提交脚本（submitit 启动器方式）
├── pretrain_imgnet22k.sh
├── checkpoint_merge_fsdp.py             # FSDP 分片权重合并
├── checkpoint_merge_fsdp.sh
├── conda.yaml / conda-extras.yaml       # 环境定义
├── requirements.txt / requirements-dev.txt / requirements-extras.txt
├── dinov2/
│   ├── configs/
│   │   ├── ssl_default_config.yaml      # ★ 默认配置（gcn 段含全部新超参 + 容器开关）
│   │   └── train/
│   │       ├── vitg14.yaml
│   │       ├── vitl14.yaml
│   │       ├── vitl16_short.yaml
│   │       └── vitl16_short_imgnet22k.yaml
│   ├── data/
│   │   ├── collate.py                   # ★ batch_size=1，单 slide：crops + coords[x,y] + slide_name
│   │   ├── adapters.py / augmentations.py / loaders.py / samplers.py
│   │   ├── transforms.py / masking.py / __init__.py
│   ├── models/
│   │   ├── gcn.py                       # GIN/GCN/GAT/GraphSAGE 卷积与 GNN 主干（GAT 用 edge_dim=3）
│   │   ├── vision_transformer.py
│   │   └── __init__.py
│   ├── train/
│   │   ├── gcn_meta_arch.py             # ★ 核心：GCNMetaArch、SpatialPatchAggregator、
│   │   │                                #         DINOFeatureContainer、建图/对比/两阶段边学习
│   │   ├── train.py                     # ★ do_train / do_test（保存 student 权重）
│   │   ├── ssl_meta_arch.py             # 官方 SSL 架构（保留未用）
│   │   └── __init__.py
│   ├── run/train/train.py               # submitit 启动器入口（集群提交用）
│   ├── loss/                            # DINOLoss / iBOTPatchLoss / KoLeoLoss
│   ├── fsdp/                            # FSDP 封装与 FSDPCheckpointer
│   ├── distributed/ / logging/ / utils/ / eval/ / hub/ / layers/ / thirdparty/
└── docs/
    └── README.md                        # 本文件
```

> 注：带 ★ 为本项目三个修改点与容器直接涉及的文件。

---

## 6. 配置说明（`dinov2/configs/ssl_default_config.yaml` 的 `gcn` 段）

```yaml
gcn:
  num_layer: 5                 # GNN 层数
  emb_dim: 128                 # 节点嵌入维度
  JK: "last"                   # Jumping Knowledge 模式
  dropout_ratio: 0
  gnn_type: "gat"              # gin / gcn / gat / graphsage

  # 修改1：空间特征聚合
  spatial_agg_hidden: 256      # 1x1 压缩后的通道数
  spatial_agg_pool_size: 4     # 池化输出网格尺寸（输出维度 = hidden × pool²）

  # 修改3.2：初始空间图（仅坐标）
  spatial_radius: 50.0         # 空间邻接半径（WSI 坐标单位）

  # 修改3出口：语义边发现
  semantic_sim_threshold: 0.7  # context-aware 相似度阈值
  semantic_max_edges: 4096     # 每次最多新增的语义边数（按相似度取 top-k）

  # 修改2：两阶段边学习
  edge_mask_ratio: 0.75        # 被 mask 的边比例（作为正样本）
  num_neg_pairs: 128           # 负样本节点对数量（图中本不存在的边）

  # 修改3：Graph Context Contrastive Learning
  contrast_weight: 1.0         # 对比损失权重
  contrast_temperature: 0.5    # InfoNCE 温度
  contrast_proj_dim: 128       # 投影头输出维度
  view_edge_drop: 0.2          # 构造 Graph View 时的随机丢边比例

  # DINO 原始结果保存容器
  save_dino_features: false    # 开关，默认关闭
  dino_feature_dir: "dino_raw_features"
  dino_feature_save_every: 1   # 每 N 次前向保存一次
  dino_feature_max_files: 0    # 0 = 不限制；>0 超出则删最早的文件
```

`compute_precision.student` 中为每个 student 子模块配置了 FSDP 精度，
新增模块：`spatial_agg`、`projection`、`edge_existence_head`、`edge_type_head`（均 fp32）。

---

## 7. 数据约定

- 每个训练样本 = **一张 WSI 的所有子图**（`batch_size_per_gpu=1`）。
- 样本需提供：每个子图的 global / local crops，以及子图在 WSI 中的坐标 `x, y` 和 `slide_name`。
- `collate.py` 将坐标整理为 `coords: [num_patches, 2]`，节点数与子图数一一对应。

---

## 8. 运行方式

**集群（SLURM + submitit）**：

```bash
bash pretrain.sh
# 等价于：
python -m torch.distributed.launch dinov2/run/train/train.py \
  --master_port=4582 --nnodes=1 --ngpus=1 --partition=gpu \
  --config-file dinov2/configs/train/vitl16_short.yaml \
  --output-dir dinov2/results/gcn_he0819_pretrain \
  train.dataset_path=ImageFolder:root=/path/to/HE/data
```

**非集群 / 本地调试**：绕过 submitit，直接用 `dinov2/train/train.py` 入口，
并用 `CUDA_VISIBLE_DEVICES` 指定 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 python -m dinov2.train.train \
  --config-file dinov2/configs/ssl_default_config.yaml \
  --output-dir ./outputs/gcn_local_test \
  gcn.save_dino_features=true
```

命令行可覆盖任意配置项（`KEY.SUB VALUE` 形式），例如打开原始结果保存：
`gcn.save_dino_features=true gcn.dino_feature_max_files=500`。

---

## 9. 输出产物

| 路径 | 内容 |
|---|---|
| `{output_dir}/model_*.pth` | FSDP 训练 checkpoint（可用 `checkpoint_merge_fsdp.py` 合并） |
| `{output_dir}/eval/*/student_checkpoint.pth` | 评估点保存的 student 权重（backbone + GNN + 各预测头） |
| `{output_dir}/training_metrics.json` | 训练指标（含 `graph_contrastive` / `edge_existence` / `edge_type`） |
| `{output_dir}/dino_raw_features/*.pt` | DINO 原始结果（开启 `save_dino_features` 后） |
