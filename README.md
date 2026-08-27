# CerviPath DINOv2 系列项目说明

本目录包含基于 **Meta DINOv2** 官方仓库二次开发的多个项目，均用于**宫颈癌病理图像（TCGA-CESC / HE 染色）**的自监督表征学习，核心思路是 **DINOv2 视觉特征 + 图神经网络（GNN）**。

五个核心项目：

| 目录 | 阶段 | 定位 |
|---|---|---|
| `dinov2` | 预训练基座 | 官方 DINOv2 自监督预训练 + resume 续训（HE 病理图） |
| `dinov2_finetune` | 早期实验 | GCN + DINO 联合自监督训练（边预测任务） |
| `dinov2_stage1_Extract2s2` | Stage 1 | 用预训练模型提取 patch 嵌入（`--eval-only`） |
| `dinov2_stage1_Extract2s2_local` | Stage 1 本地版 | Stage 1 的本地调试增强版（提取 patch token 级特征） |
| `dinov2_stage2_2_FmH2ST` | Stage 2 | 在图数据上的图自监 督学习（多任务重建） |

> 另有 `dinov2_stage2_2_FmH2ST_finetune`（Stage 2 微调项目）不在本文重点范围内。

---

## 1. 总体流程（数据流）

```
HE 病理图 (ImageNet_like / TCGA-CESC WSI patch)
        │
        ▼
【预训练】dinov2
   官方 DINOv2 自监督预训练（DINO/iBOT）
   ViT-L/16 权重（resume 续训）
        │  输出: 预训练权重 → 初始化 Stage 1 backbone
        ▼
【Stage 1】dinov2_stage1_Extract2s2
   DINOv2 / GCN-DINO 模型（--eval-only）
   提取 global + local crops 特征
        │  输出: embeddings/features_*.npz（每 patch 一个嵌入）
        ▼
【建图】graph_build.ipynb
   嵌入 + patch 坐标 (x, y)
   KNN(k=8) + 空间/语义相似度 → 每张 WSI 一个图 (.pt)
        │  输出: Graph-1024-251111-thre0.8/（每图约 1024 节点）
        ▼
【Stage 2】dinov2_stage2_2_FmH2ST
   GAT 图自监督学习
   mask 边/节点 → 重建（边权重、节点特征、邻接结构、图级一致性）
        │
        ▼
   图级病理表征（下游任务）
```

---

## 2. 各项目详解

### 2.1 `dinov2` —— 预训练基座（官方 DINOv2 + 续训）

**定位**：流水线最上游的**预训练基座**。基于官方 DINOv2 的 ViT-L/16 自监督预训练（DINO/iBOT），在 HE 病理图上训练并支持 resume 续训，产出权重用于初始化下游 Stage 1 的 backbone。

- **架构**：保留官方 `SSLMetaArch` 完整自监督结构（DINO + iBOT + KoLeo 正则、teacher EMA 0.992→1.0、teacher temperature schedule）
- **权重兼容** [dinov2/train/ssl_meta_arch.py](dinov2/dinov2/train/ssl_meta_arch.py) 新增 `replace_name()`：把 checkpoint 中 `blocks.N` 扁平索引重映射为 `blocks.chunk.chunk_idx` 嵌套索引，**消除 BlockChunk 分块数差异**，使不同 `block_chunks` 配置可共用同一套权重
- **续训** [dinov2/train/ssl_meta_arch_stage2.py](dinov2/dinov2/train/ssl_meta_arch_stage2.py)：第二阶段预训练，`load_state_dict` 加载已有权重继续训练
- **模型合并** [dinov2/run/train/model_merge.py](dinov2/dinov2/run/train/model_merge.py)：submitit 提交的 FSDP 分片权重合并（`checkpoint_merge_fsdp.sh` 调用，4 GPU）
- **训练脚本** `pretrain_imgnet22k.sh`：`vitl16_short_imgnet22k.yaml`（batch_size=64），数据 `/home/user/90T/liyu/CerviPath/Data/Pretrain(+_extra)`，输出 `dinov2_he1023_pretrain-resume-1024`
- **实际结果** `dinov2/results/dinov2_he1023_pretrain-resume-1024/`（config.yaml、slurm 提交记录、日志）
- 环境：自带 `.venv`（Python 3.13）

### 2.2 `dinov2_finetune` —— GCN+DINO 联合预训练（早期实验）

**定位**：DINOv2 backbone 与 GNN 联合自监督训练的最初版本，核心任务为**图边类型预测**。

- **架构** [dinov2/train/gcn_meta_arch.py](dinov2_finetune/dinov2/train/gcn_meta_arch.py)
  - `dino_encoder`：2 个 global crop + 8 个 local crop → backbone → DINO head → 三类 token 均值特征拼接后求平均
  - `mlp`：将 DINO 特征投影到图嵌入维度
  - `GNN`（[dinov2/models/gcn.py](dinov2_finetune/dinov2/models/gcn.py)，基于 torch_geometric 的 GINConv）
  - 建图 `construct_graph`：坐标距离（`radius=50`）+ 特征余弦相似度（`sim_threshold=0.7`）确定连边，mask 75% 的边，用 `linear_pred_edges` 预测边的 4 类标签（none / distance / similarity / both），`CrossEntropyLoss`
- **数据集** [dinov2/data/datasets/image_folder.py](dinov2_finetune/dinov2/data/datasets/image_folder.py)：WSI slide patch 级数据（`patch_grid_positions-TCGA_CESC-hospital.csv` + 预构建的 `slide_patch_map.pkl.gz`），每 slide 最多 50 个 patch，样本带 `coords` 坐标（建图必需）
- **训练脚本** `pretrain.sh`：单卡（`--ngpus=1`），输出 `gcn_he0819_pretrain`，`batch_size_per_gpu=1`
- **train.py 改动**：使用 `GCNMetaArch`；删除 teacher EMA 更新与 teacher temperature schedule；新增 FSDP `ShardedStateDictConfig`/`FullStateDictConfig` 分片保存；checkpoint 每 epoch 保存

### 2.3 `dinov2_stage1_Extract2s2` —— Stage 1 嵌入提取（主项目）

**定位**：**特征提取**阶段。`pretrain.sh` 的训练部分与 finetune 相同的 GCN-DINO 架构；`pretrain_imgnet22k.sh` 则以 **`--eval-only`** 模式跑完整数据集，输出每个 patch 的嵌入。

- **提取逻辑** [dinov2/train/train.py](dinov2_stage1_Extract2s2/dinov2/train/train.py) 的 `do_test` 被重写：
  - fp32 前向、`shuffle=False`、`SamplerType.EPOCH`、`drop_last=False`
  - 遍历全部数据 → 保存 `embeddings/features_pretrained_s1_part{N}.npz`（压缩格式）与文件名索引
  - 支持分 chunk 断点式提取，每个 chunk 完成后立即落盘
- **架构** [dinov2/train/gcn_meta_arch.py](dinov2_stage1_Extract2s2/dinov2/train/gcn_meta_arch.py)：保留完整 DINO 组件（dino_head / ibot_head），`forward_backward` 仅调用 `dino_encoder` 返回特征（**不计算损失**）
- **实际结果** `dinov2/results/pretrained_s1_embeddings-1024`、`pretrained_s1_embeddings-65536`（两个不同 batch size 的提取任务），数据源为 ImageNet22k（HE 病理图），batch_size=64
- 提取出的嵌入即后续建图的输入（见 `CerviPath/Graph/embeddings-1024`）

### 2.4 `dinov2_stage1_Extract2s2_local` —— Stage 1 本地调试增强版

**定位**：与 2.3 同阶段，代码基本一致，但做了**细粒度特征提取**增强（`2s2` = 两个尺度：global crop + local crop 的 patch token 级特征）：

- [gcn_meta_arch.py](dinov2_stage1_Extract2s2_local/dinov2/train/gcn_meta_arch.py) 新增 `extract_local_patch_tokens`：提取 **local crops 的全部 patch tokens**（非仅 cls token），`is_training=False` 推理模式
- [train.py](dinov2_stage1_Extract2s2_local/dinov2/train/train.py) 的 `do_test` 新增 **dense tile 提取**：96×96 tile、stride 96、batch 256，遍历整图（配合 `return_original=True`），用于获取高密度局部特征
- **目录精简**：删除官方文档/安装杂项，脚本收敛到 `script/` 目录；configs 提供 `vitl14.yaml` / `vitg14.yaml`（完整大模型配置）与精简的 `vitl16_short.yaml`
- 主要用途：本地快速调试 + 提取 patch token 级稠密特征（供建图时使用更细粒度的语义）

### 2.5 `dinov2_stage2_2_FmH2ST` —— Stage 2 图自监督学习

**定位**：直接在**预构建的图数据**上做图自监督学习，架构与 Stage 1 完全不同（**不再使用 DINO**，纯 GNN）。

- **架构** [dinov2/train/gcn_meta_arch.py](dinov2_stage2_2_FmH2ST/dinov2/train/gcn_meta_arch.py)：
  - `GNN_backbone`（cfg.gcn 构建）+ `pool_gate`（gated attention pooling：LayerNorm → Linear → GELU → Linear → softmax → `global_add_pool`）
  - 输入：`masked_graph`（遮掩后的图）与 `original_graph`（监督信号）
  - **4 个自监督任务**：
    1. `edge_reconstruction`：被 mask 边的权重重建（MSE）
    2. `node_reconstruction`：被 mask 节点的特征重建（MSE）
    3. `structure_consistency`：邻接矩阵一致性（权重 0.02）
    4. `graph_consistency`：masked/original 图级表示余弦一致性（权重 0.2）
- **模型** [dinov2/models/gcn.py](dinov2_stage2_2_FmH2ST/dinov2/models/gcn.py)：GNN 支持 `block_chunks` 分块（`GNNChunk`，兼容 FSDP 自动分片）、残差连接、LayerNorm；`gnn_type` 可选 gin / gcn / gat / graphsage
- **数据集** [dinov2/data/datasets/image_folder.py](dinov2_stage2_2_FmH2ST/dinov2/data/datasets/image_folder.py)：加载 `.pt` 图文件（PyG `Data`），含：
  - `auto_set_graph_filters`：按节点/边数分位数自动过滤小图
  - 随机子图采样（>5000 节点时取 5000）
  - mask 策略：`edge` / `node` / `patch` / `feature`，可选用随机游走遮掩（`rand_walk`）
- **配置** [dinov2/configs/ssl_default_config.yaml](dinov2_stage2_2_FmH2ST/dinov2/configs/ssl_default_config.yaml) 新增 `gcn` 节：`arch=gnn`、`num_layer=5`、`emb_dim=1024`、`JK=sum`、`gnn_type=gat`、`edge_dim=1`、`mask_strategy=edge`；epochs=400，batch_size=4（`vitl16_short.yaml`）
- **建图方式** [dinov2/data/datasets/graph_build.ipynb](dinov2_stage2_2_FmH2ST/dinov2/data/datasets/graph_build.ipynb)：读取 Stage 1 嵌入 + patch 坐标，`build_graph(coords, features, k=8, alpha=0.5)`（KNN + 空间/语义相似度加权）→ 每张 WSI 一个图（约 1064 节点、8512 边）→ `Graph-1024-251111-thre0.8/`（1024 节点/图、相似度阈值 0.8）
- **实际结果** `dinov2/results/gcn_1112_pretrain_embeddings-1024`（含 `-b4` batch4、`-b4-mask75` mask 75% 等变体）

---

## 3. 异同点对比

### 3.1 相同点

1. **同一代码基线**：全部 fork 自官方 DINOv2 仓库，git 历史一致，核心训练框架（`dinov2/run/train/train.py`、FSDP、`data/loaders.py`、submitit/slurm 提交）未动
2. **同一业务场景**：宫颈癌 HE 病理图像（TCGA-CESC），数据根目录均为 `ImageNet_like` 系列
3. **统一的技术选型**：FSDP（分布式）、slurm 作业脚本（`pretrain*.sh` + `checkpoint_merge_fsdp.sh`）、提交后输出 `Job.%j.out/err`（图相关项目另用 torch_geometric）
4. **相同的集成方式**：都通过自定义/改写 meta arch（`GCNMetaArch` / `SSLMetaArch`）+ 改写 `dinov2/train/train.py` 接入训练循环
5. **checkpoint 处理**：均使用 FSDP 分片保存 + 合并脚本汇总权重（dinov2 用 `run/train/model_merge.py`，其余用根目录 `checkpoint_merge_fsdp.py`）

### 3.2 不同点

| 维度 | dinov2 | dinov2_finetune | stage1_Extract2s2 | stage1_..._local | stage2_2_FmH2ST |
|---|---|---|---|---|---|
| 阶段 | 预训练基座 | 早期实验 | Stage 1 提取 | Stage 1 本地版 | Stage 2 图学习 |
| 架构 | 官方 SSLMetaArch（DINO/iBOT） | DINO + GNN（边预测） | DINO 特征提取（无损失） | DINO + dense patch token 提取 | 纯 GNN（GAT，多任务重建） |
| 建图 | — | 训练时动态建图（距离+相似度） | — | — | 预构建图（KNN k=8） |
| 数据集 | ImageNet22k（HE 病理图） | WSI slide patch（csv+pkl 缓存） | ImageNet22k 图像 | 同左 + dense tile | `.pt` 图文件（PyG Data） |
| 任务/损失 | DINO/iBOT 自监督损失 | 边 4 分类（CE） | 无（仅前向） | 无（仅前向） | 边/节点重建 + 结构/图一致性（MSE+余弦） |
| batch_size | 64 | 1 | 64 | 64 | 4 |
| teacher EMA / temp | 保留（官方） | 移除 | 保留组件（不训练） | 保留组件（不训练） | 不存在 |
| 训练脚本 | `pretrain_imgnet22k.sh` → `dinov2_he1023_pretrain-resume-1024` | `pretrain.sh` → `gcn_he0819_pretrain` | `pretrain.sh`（训练）+ `pretrain_imgnet22k.sh`（`--eval-only` 提取） | 同左（`script/` 下） | `pretrain.sh` → `gcn_1112_pretrain_embeddings-1024` |
| 产物 | 预训练权重（resume 续训） | — | `embeddings/features_*.npz` | dense patch token | 图级表征（含 `-b4-mask75` 变体） |
| 目录结构 | 完整官方仓库（含 `.venv`） | 完整官方仓库 | 完整官方仓库 | 精简（`script/`、无官方文档） | 完整官方仓库 |

### 3.3 关键代码差异速查

- **`ssl_meta_arch.py`**：仅 dinov2 修改 —— 新增 `replace_name()` 做 BlockChunk 权重索引重映射；其余项目保留官方原版
- **`gcn_meta_arch.py`**：其余四个版本互不相同 —— finetune 版含 `construct_graph` + 边分类；stage1 版保留 DINO 组件但 `forward_backward` 只返回特征；local 版增加 `extract_local_patch_tokens`；stage2 版完全重写（`pool_gate` + 4 任务重建，logger 为 "gnn"）
- **`train.py` 的 `do_test`**：dinov2/finetune/stage2 保留官方 checkpoint 保存逻辑；stage1 重写为嵌入提取（npz 落盘）；local 版再加 dense tile 逻辑
- **`ssl_default_config.yaml`**：仅 stage2 新增 `gcn` 配置节（num_layer/emb_dim/gnn_type/mask_strategy 等）
- **`image_folder.py`**：finetune/stage1 为 WSI patch 数据集；stage2 为图数据集（过滤、子图采样、多策略 mask）；dinov2 使用官方 ImageNet22k 数据集
- **独有脚本**：dinov2 含 `ssl_meta_arch_stage2.py`（续训）与 `run/train/model_merge.py`（权重合并）；其余项目用根目录 `checkpoint_merge_fsdp.py`

---

## 4. 运行入口与关键接口

### 4.1 Python 入口文件与调用链

所有项目共用官方 DINOv2 的"启动器 + 训练逻辑"两层架构，**Python 入口均为 `dinov2/run/train/train.py`**（submitit 作业启动器，各项目完全相同），真正的训练主逻辑在各项目的 `dinov2/train/train.py`，核心模型架构在 `dinov2/train/*_meta_arch.py`。

| 项目 | Python 入口 | 实际训练主逻辑 | 核心架构文件 |
|---|---|---|---|
| `dinov2` | `dinov2/run/train/train.py` | `dinov2/train/train.py`（官方版） | `dinov2/train/ssl_meta_arch.py`（+ `replace_name()` 加载 UNI） |
| `dinov2_finetune` | 同上 | `dinov2/train/train.py`（改：使用 GCNMetaArch） | `dinov2/train/gcn_meta_arch.py` + `dinov2/models/gcn.py` |
| `dinov2_stage1_Extract2s2` | 同上 | `dinov2/train/train.py`（改：`do_test` 重写为嵌入提取） | `dinov2/train/gcn_meta_arch.py` |
| `dinov2_stage1_Extract2s2_local` | 同上 | 同 stage1 + dense tile 提取 | 同 stage1 + `extract_local_patch_tokens` |
| `dinov2_stage2_2_FmH2ST` | 同上 | `dinov2/train/train.py`（改：纯 GNN 训练） | `dinov2/train/gcn_meta_arch.py`（重写）+ `dinov2/models/gcn.py` |

调用链（以 finetune 为例，其余项目同构）：

```
dinov2/run/train/train.py         ← 入口（submitit 启动器，仅 59 行）
  main() → submit_jobs(Trainer)  ← 提交 slurm 作业
  └─ Trainer.__call__() → from dinov2.train import main   （延迟导入）
       └─ dinov2/train/train.py 的 main()
            ├─ model = GCNMetaArch(cfg) / SSLMetaArch(cfg)  ← 架构在此实例化
            ├─ do_train() / do_test()
            └─ dinov2/train/gcn_meta_arch.py               ← GCN 核心（边预测 / 重建任务）
                 └─ dinov2/models/gcn.py                   ← GNN / GINConv 模型定义
```

### 4.2 slurm 启动脚本（各项目）

| 项目 | 训练/提取脚本 | 输出目录 | 说明 |
|---|---|---|---|
| `dinov2` | `pretrain_imgnet22k.sh` | `dinov2/results/dinov2_he1023_pretrain-resume-1024` | 从 UNI 初始化 → HE 病理图预训练/续训 |
| `dinov2_finetune` | `pretrain.sh` | `dinov2/results/gcn_he0819_pretrain` | GCN+DINO 边预测训练（`--ngpus=1`） |
| `dinov2_stage1_Extract2s2` | `pretrain.sh`（训练）+ `pretrain_imgnet22k.sh`（`--eval-only` 提取） | `gcn_he0819_pretrain` / `pretrained_s1_embeddings-{1024,65536}` | 提取的嵌入用于后续建图 |
| `dinov2_stage1_Extract2s2_local` | `script/pretrain.sh` + `script/pretrain_imgnet22k.sh` | 同上 | 同 stage1（本地调试版） |
| `dinov2_stage2_2_FmH2ST` | `pretrain.sh` | `dinov2/results/gcn_1112_pretrain_embeddings-1024`（含 `-b4`、`-b4-mask75` 变体） | 图自监督训练（GAT 多任务重建） |

FSDP 分片权重合并：`dinov2` 用 `checkpoint_merge_fsdp.sh`（内部调 `run/train/model_merge.py`，submitit 提交）；其余项目用同脚本名（内部 `torchrun --nproc_per_node=4 checkpoint_merge_fsdp.py`）。

```bash
# 预训练（DINOv2 基座，可选 resume 续训）
sbatch dinov2/pretrain_imgnet22k.sh

# Stage 1：GCN-DINO 训练
sbatch dinov2_stage1_Extract2s2/pretrain.sh

# Stage 1：提取嵌入（--eval-only）
sbatch dinov2_stage1_Extract2s2/pretrain_imgnet22k.sh

# 建图（手动执行 notebook）
#   dinov2_stage2_2_FmH2ST/dinov2/data/datasets/graph_build.ipynb

# Stage 2：图自监督训练
sbatch dinov2_stage2_2_FmH2ST/pretrain.sh

# 合并 FSDP 分片权重
sbatch dinov2_stage2_2_FmH2ST/checkpoint_merge_fsdp.sh
```

> 注意：所有脚本中的 `--chdir` 与数据路径均指向服务器原路径（`/home/li_yu/Proj04_he/...`），在本地运行前需按实际环境修改；stage1/stage2 的 `pretrain.sh` 均带 `--no-resume`，且实际执行时仅用 1 张卡（`--ngpus=1`，Slurm 申请 4 卡仅作资源预留）。

### 4.3 命令行参数与配置覆盖接口

基于 fvcore/yacs 配置体系，启动命令可直接在末尾用 `键=值` 覆盖 yaml 配置：

```bash
python dinov2/run/train/train.py \
  --master_port=4582 \
  --nnodes=1 --ngpus=1 --partition=gpu \
  --config-file dinov2/configs/train/vitl16_short.yaml \
  --output-dir dinov2/results/my_exp \
  train.dataset_path=ImageFolder:root=/path/to/data \
  train.batch_size_per_gpu=2 \
  student.pretrained_weights=/path/to/weight.pth \
  gcn.mask_strategy=edge
```

常用参数：

| 参数 | 作用 |
|---|---|
| `--config-file` | 训练配置（`dinov2/configs/train/*.yaml`） |
| `--output-dir` | 输出目录（`dinov2/results/<实验名>`） |
| `--eval-only` | 仅推理/提取（stage1 提取嵌入时使用） |
| `--ngpus` / `--nnodes` | GPU 数 / 节点数 |
| `--master_port` | 分布式通信端口 |
| `train.dataset_path` | 数据集（`ImageFolder:root=...` / `ImageNet22k:root=...:extra=...`） |
| `train.batch_size_per_gpu` | 每卡 batch size |
| `student.pretrained_weights` | **预训练权重路径（backbone 初始化，核心接口）** |
| `gcn.*` | stage2 专属图配置（`mask_strategy` / `structure_loss_weight` 等） |

### 4.4 预训练权重加载接口（backbone 初始化）

机制（`gcn_meta_arch.py` / `ssl_meta_arch.py` 的 `cfg.student.pretrained_weights`）：

- 配置非空时，`torch.load` 后 `load_state_dict(..., strict=False)` **部分加载**：backbone 用预训练权重初始化，GCN 等新增模块保持随机初始化
- 权重格式差异：`dinov2` 项目产出 fvcore 格式（含 `"model"` 键）；**UNI 权重是裸 state_dict（无 `"model"` 键）**，加载时需注意匹配

完整权重链路：

```
UNI 病理大模型（ViT-L/16，约 1.2GB）
  CerviPath/UNI/weights/models--MahmoodLab--UNI/pytorch_model.bin
  │  replace_name() 重映射键名（扁平 blocks.N → chunk 嵌套，兼容 block_chunks 差异）
  ▼
【dinov2】默认从 UNI 初始化 → HE 病理图预训练/续训
  │  输出：dinov2_he1023_pretrain-resume-1024（fvcore 格式）
  ▼
【dinov2_finetune / stage1】student.pretrained_weights=该权重
  只初始化 backbone，GCN 从头训练（或提取嵌入）
```

### 4.5 核心配置项速查（`dinov2/configs/ssl_default_config.yaml`）

| 配置节 | 关键项 | 说明 |
|---|---|---|
| `student` | `arch` / `patch_size` / `block_chunks` / `pretrained_weights` | 模型结构与权重初始化 |
| `dino` | `loss_weight` / `head_n_prototypes` | 设为 0 即关闭 DINO 损失（仅保留特征提取） |
| `ibot` | `loss_weight` / `separate_head` | iBOT 损失开关 |
| `teacher` | `momentum_teacher` / `final_momentum_teacher` | 仅 dinov2（官方 SSLMetaArch）使用 |
| `gcn`（仅 stage2） | `arch=gnn` / `num_layer=5` / `emb_dim=1024` / `gnn_type=gat` / `mask_strategy=edge` / `structure_loss_weight=0.02` / `graph_consistency_weight=0.2` | 图网络与自监督任务配置 |
| `train` | `dataset_path` / `batch_size_per_gpu` / `OFFICIAL_EPOCH_LENGTH` / `cache_dataset` | 数据与训练参数 |


---

## 5. 周边目录说明

- `Data/`：原始数据（`Pretrain`/`Pretrain_extra` HE 病理图、`TCGA` WSI 数据、`hospital` 元信息）
- `Graph/`：Stage 1 产物与建图结果（`embeddings-1024`、`Graph-1024-251111-thre*`）
- `dinov2_stage2_2_FmH2ST_finetune/`：Stage 2 微调项目（含 `ACCURACY_DIFFERENCE_ROOT_CAUSE.md` 精度差异根因分析）
- `UNI/`：其他自监督模型相关代码

---

## 6. 项目完整目录树（所有层级展开，不省略）

> 说明：
> - 本树反映服务器磁盘上的**真实完整结构**，目录层级全部展开、无省略（文件数量巨大的数据目录以「文件名模式 × 数量」标注，如 `dummy_0/` 内含 10001 个 patch png，全部 230 个 `dummy_*` 目录逐一列出）。
> - 标注 **【未纳入版本控制】** 的目录（`Data/`、`Graph/`、`UNI/`、各项目 `results/`、`.venv/`）因体积过大（合计数十 GB / 数百万文件）无法随本仓库上传 GitHub，其内部结构在此完整描述，如需复现请按第 4 节运行入口重新生成。
> - `__pycache__/`、`.ipynb_checkpoints/`、`*.egg-info/`、`*.whl`、`Job.*.out/err` 等缓存/构建产物已忽略，不展示。

```
CerviPath/
├── Data/
│   ├── hospital/
│   │   ├── rawdata/
│   │   │   ├── 114731_0722151317681_h.jpg
│   │   │   ├── 114731_0722151407169_h.jpg
│   │   │   ├── 114855_0722203758966_h.jpg
│   │   │   └── … 其余 1838 个文件（.jpg × 1841）
│   │   └── cervival-summary-简化1029update.csv
│   ├── meta/
│   │   ├── patch/
│   │   │   ├── patch_grid_positions-hospitalAlignTCGA-disease2.json
│   │   │   └── patch_grid_positions-TCGA-disease2-test-lesstest.csv
│   │   ├── hospital-meta.csv
│   │   └── TCGA-meta.csv
│   ├── Pretrain/
│   │   ├── blocks/
│   │   │   ├── dummy_0.log
│   │   │   ├── dummy_1.log
│   │   │   ├── dummy_10.log
│   │   │   └── … 其余 227 个文件（.log × 230）
│   │   ├── extracted/
│   │   │   ├── dummy_0/  （10001 个 patch 文件）
│   │   │   ├── dummy_1/  （10001 个 patch 文件）
│   │   │   ├── dummy_10/  （10001 个 patch 文件）
│   │   │   ├── dummy_100/  （10001 个 patch 文件）
│   │   │   ├── dummy_101/  （10001 个 patch 文件）
│   │   │   ├── dummy_102/  （10001 个 patch 文件）
│   │   │   ├── dummy_103/  （10001 个 patch 文件）
│   │   │   ├── dummy_104/  （10001 个 patch 文件）
│   │   │   ├── dummy_105/  （10001 个 patch 文件）
│   │   │   ├── dummy_106/  （10001 个 patch 文件）
│   │   │   ├── dummy_107/  （10001 个 patch 文件）
│   │   │   ├── dummy_108/  （10001 个 patch 文件）
│   │   │   ├── dummy_109/  （10001 个 patch 文件）
│   │   │   ├── dummy_11/  （10001 个 patch 文件）
│   │   │   ├── dummy_110/  （10001 个 patch 文件）
│   │   │   ├── dummy_111/  （10001 个 patch 文件）
│   │   │   ├── dummy_112/  （10001 个 patch 文件）
│   │   │   ├── dummy_113/  （10001 个 patch 文件）
│   │   │   ├── dummy_114/  （10001 个 patch 文件）
│   │   │   ├── dummy_115/  （10001 个 patch 文件）
│   │   │   ├── dummy_116/  （10001 个 patch 文件）
│   │   │   ├── dummy_117/  （10001 个 patch 文件）
│   │   │   ├── dummy_118/  （10001 个 patch 文件）
│   │   │   ├── dummy_119/  （10001 个 patch 文件）
│   │   │   ├── dummy_12/  （10001 个 patch 文件）
│   │   │   ├── dummy_120/  （10001 个 patch 文件）
│   │   │   ├── dummy_121/  （10001 个 patch 文件）
│   │   │   ├── dummy_122/  （10001 个 patch 文件）
│   │   │   ├── dummy_123/  （10001 个 patch 文件）
│   │   │   ├── dummy_124/  （10001 个 patch 文件）
│   │   │   ├── dummy_125/  （10001 个 patch 文件）
│   │   │   ├── dummy_126/  （10001 个 patch 文件）
│   │   │   ├── dummy_127/  （10001 个 patch 文件）
│   │   │   ├── dummy_128/  （10001 个 patch 文件）
│   │   │   ├── dummy_129/  （10001 个 patch 文件）
│   │   │   ├── dummy_13/  （10001 个 patch 文件）
│   │   │   ├── dummy_130/  （10001 个 patch 文件）
│   │   │   ├── dummy_131/  （10001 个 patch 文件）
│   │   │   ├── dummy_132/  （10001 个 patch 文件）
│   │   │   ├── dummy_133/  （10001 个 patch 文件）
│   │   │   ├── dummy_134/  （10001 个 patch 文件）
│   │   │   ├── dummy_135/  （10001 个 patch 文件）
│   │   │   ├── dummy_136/  （10001 个 patch 文件）
│   │   │   ├── dummy_137/  （10001 个 patch 文件）
│   │   │   ├── dummy_138/  （10001 个 patch 文件）
│   │   │   ├── dummy_139/  （10001 个 patch 文件）
│   │   │   ├── dummy_14/  （10001 个 patch 文件）
│   │   │   ├── dummy_140/  （10001 个 patch 文件）
│   │   │   ├── dummy_141/  （10001 个 patch 文件）
│   │   │   ├── dummy_142/  （10001 个 patch 文件）
│   │   │   ├── dummy_143/  （10001 个 patch 文件）
│   │   │   ├── dummy_144/  （10001 个 patch 文件）
│   │   │   ├── dummy_145/  （10001 个 patch 文件）
│   │   │   ├── dummy_146/  （10001 个 patch 文件）
│   │   │   ├── dummy_147/  （10001 个 patch 文件）
│   │   │   ├── dummy_148/  （10001 个 patch 文件）
│   │   │   ├── dummy_149/  （10001 个 patch 文件）
│   │   │   ├── dummy_15/  （10001 个 patch 文件）
│   │   │   ├── dummy_150/  （10001 个 patch 文件）
│   │   │   ├── dummy_151/  （10001 个 patch 文件）
│   │   │   ├── dummy_152/  （10001 个 patch 文件）
│   │   │   ├── dummy_153/  （10001 个 patch 文件）
│   │   │   ├── dummy_154/  （10001 个 patch 文件）
│   │   │   ├── dummy_155/  （10001 个 patch 文件）
│   │   │   ├── dummy_156/  （10001 个 patch 文件）
│   │   │   ├── dummy_157/  （10001 个 patch 文件）
│   │   │   ├── dummy_158/  （10001 个 patch 文件）
│   │   │   ├── dummy_159/  （10001 个 patch 文件）
│   │   │   ├── dummy_16/  （10001 个 patch 文件）
│   │   │   ├── dummy_160/  （10001 个 patch 文件）
│   │   │   ├── dummy_161/  （10001 个 patch 文件）
│   │   │   ├── dummy_162/  （10001 个 patch 文件）
│   │   │   ├── dummy_163/  （10001 个 patch 文件）
│   │   │   ├── dummy_164/  （10001 个 patch 文件）
│   │   │   ├── dummy_165/  （10001 个 patch 文件）
│   │   │   ├── dummy_166/  （10001 个 patch 文件）
│   │   │   ├── dummy_167/  （10001 个 patch 文件）
│   │   │   ├── dummy_168/  （10001 个 patch 文件）
│   │   │   ├── dummy_169/  （10001 个 patch 文件）
│   │   │   ├── dummy_17/  （10001 个 patch 文件）
│   │   │   ├── dummy_170/  （10001 个 patch 文件）
│   │   │   ├── dummy_171/  （10001 个 patch 文件）
│   │   │   ├── dummy_172/  （10001 个 patch 文件）
│   │   │   ├── dummy_173/  （10001 个 patch 文件）
│   │   │   ├── dummy_174/  （10001 个 patch 文件）
│   │   │   ├── dummy_175/  （10001 个 patch 文件）
│   │   │   ├── dummy_176/  （10001 个 patch 文件）
│   │   │   ├── dummy_177/  （10001 个 patch 文件）
│   │   │   ├── dummy_178/  （10001 个 patch 文件）
│   │   │   ├── dummy_179/  （10001 个 patch 文件）
│   │   │   ├── dummy_18/  （10001 个 patch 文件）
│   │   │   ├── dummy_180/  （10001 个 patch 文件）
│   │   │   ├── dummy_181/  （10001 个 patch 文件）
│   │   │   ├── dummy_182/  （10001 个 patch 文件）
│   │   │   ├── dummy_183/  （10001 个 patch 文件）
│   │   │   ├── dummy_184/  （10001 个 patch 文件）
│   │   │   ├── dummy_185/  （10001 个 patch 文件）
│   │   │   ├── dummy_186/  （10001 个 patch 文件）
│   │   │   ├── dummy_187/  （10001 个 patch 文件）
│   │   │   ├── dummy_188/  （10001 个 patch 文件）
│   │   │   ├── dummy_189/  （10001 个 patch 文件）
│   │   │   ├── dummy_19/  （10001 个 patch 文件）
│   │   │   ├── dummy_190/  （10001 个 patch 文件）
│   │   │   ├── dummy_191/  （10001 个 patch 文件）
│   │   │   ├── dummy_192/  （10001 个 patch 文件）
│   │   │   ├── dummy_193/  （10001 个 patch 文件）
│   │   │   ├── dummy_194/  （10001 个 patch 文件）
│   │   │   ├── dummy_195/  （10001 个 patch 文件）
│   │   │   ├── dummy_196/  （10001 个 patch 文件）
│   │   │   ├── dummy_197/  （10001 个 patch 文件）
│   │   │   ├── dummy_198/  （10001 个 patch 文件）
│   │   │   ├── dummy_199/  （10001 个 patch 文件）
│   │   │   ├── dummy_2/  （10001 个 patch 文件）
│   │   │   ├── dummy_20/  （10001 个 patch 文件）
│   │   │   ├── dummy_200/  （10001 个 patch 文件）
│   │   │   ├── dummy_201/  （10001 个 patch 文件）
│   │   │   ├── dummy_202/  （10001 个 patch 文件）
│   │   │   ├── dummy_203/  （10001 个 patch 文件）
│   │   │   ├── dummy_204/  （10001 个 patch 文件）
│   │   │   ├── dummy_205/  （10001 个 patch 文件）
│   │   │   ├── dummy_206/  （10001 个 patch 文件）
│   │   │   ├── dummy_207/  （10001 个 patch 文件）
│   │   │   ├── dummy_208/  （10001 个 patch 文件）
│   │   │   ├── dummy_209/  （10001 个 patch 文件）
│   │   │   ├── dummy_21/  （10001 个 patch 文件）
│   │   │   ├── dummy_210/  （10001 个 patch 文件）
│   │   │   ├── dummy_211/  （10001 个 patch 文件）
│   │   │   ├── dummy_212/  （10001 个 patch 文件）
│   │   │   ├── dummy_213/  （10001 个 patch 文件）
│   │   │   ├── dummy_214/  （10001 个 patch 文件）
│   │   │   ├── dummy_215/  （10001 个 patch 文件）
│   │   │   ├── dummy_216/  （10001 个 patch 文件）
│   │   │   ├── dummy_217/  （10001 个 patch 文件）
│   │   │   ├── dummy_218/  （10001 个 patch 文件）
│   │   │   ├── dummy_219/  （10001 个 patch 文件）
│   │   │   ├── dummy_22/  （10001 个 patch 文件）
│   │   │   ├── dummy_220/  （10001 个 patch 文件）
│   │   │   ├── dummy_221/  （10001 个 patch 文件）
│   │   │   ├── dummy_222/  （10001 个 patch 文件）
│   │   │   ├── dummy_223/  （10001 个 patch 文件）
│   │   │   ├── dummy_224/  （10001 个 patch 文件）
│   │   │   ├── dummy_225/  （10001 个 patch 文件）
│   │   │   ├── dummy_226/  （10001 个 patch 文件）
│   │   │   ├── dummy_227/  （10001 个 patch 文件）
│   │   │   ├── dummy_228/  （10001 个 patch 文件）
│   │   │   ├── dummy_229/  （9632 个 patch 文件）
│   │   │   ├── dummy_23/  （10001 个 patch 文件）
│   │   │   ├── dummy_24/  （10001 个 patch 文件）
│   │   │   ├── dummy_25/  （10001 个 patch 文件）
│   │   │   ├── dummy_26/  （10001 个 patch 文件）
│   │   │   ├── dummy_27/  （10001 个 patch 文件）
│   │   │   ├── dummy_28/  （10001 个 patch 文件）
│   │   │   ├── dummy_29/  （10001 个 patch 文件）
│   │   │   ├── dummy_3/  （10001 个 patch 文件）
│   │   │   ├── dummy_30/  （10001 个 patch 文件）
│   │   │   ├── dummy_31/  （10001 个 patch 文件）
│   │   │   ├── dummy_32/  （10001 个 patch 文件）
│   │   │   ├── dummy_33/  （10001 个 patch 文件）
│   │   │   ├── dummy_34/  （10001 个 patch 文件）
│   │   │   ├── dummy_35/  （10001 个 patch 文件）
│   │   │   ├── dummy_36/  （10001 个 patch 文件）
│   │   │   ├── dummy_37/  （10001 个 patch 文件）
│   │   │   ├── dummy_38/  （10001 个 patch 文件）
│   │   │   ├── dummy_39/  （10001 个 patch 文件）
│   │   │   ├── dummy_4/  （10001 个 patch 文件）
│   │   │   ├── dummy_40/  （10001 个 patch 文件）
│   │   │   ├── dummy_41/  （10001 个 patch 文件）
│   │   │   ├── dummy_42/  （10001 个 patch 文件）
│   │   │   ├── dummy_43/  （10001 个 patch 文件）
│   │   │   ├── dummy_44/  （10001 个 patch 文件）
│   │   │   ├── dummy_45/  （10001 个 patch 文件）
│   │   │   ├── dummy_46/  （10001 个 patch 文件）
│   │   │   ├── dummy_47/  （10001 个 patch 文件）
│   │   │   ├── dummy_48/  （10001 个 patch 文件）
│   │   │   ├── dummy_49/  （10001 个 patch 文件）
│   │   │   ├── dummy_5/  （10001 个 patch 文件）
│   │   │   ├── dummy_50/  （10001 个 patch 文件）
│   │   │   ├── dummy_51/  （10001 个 patch 文件）
│   │   │   ├── dummy_52/  （10001 个 patch 文件）
│   │   │   ├── dummy_53/  （10001 个 patch 文件）
│   │   │   ├── dummy_54/  （10001 个 patch 文件）
│   │   │   ├── dummy_55/  （10001 个 patch 文件）
│   │   │   ├── dummy_56/  （10001 个 patch 文件）
│   │   │   ├── dummy_57/  （10001 个 patch 文件）
│   │   │   ├── dummy_58/  （10001 个 patch 文件）
│   │   │   ├── dummy_59/  （10001 个 patch 文件）
│   │   │   ├── dummy_6/  （10001 个 patch 文件）
│   │   │   ├── dummy_60/  （10001 个 patch 文件）
│   │   │   ├── dummy_61/  （10001 个 patch 文件）
│   │   │   ├── dummy_62/  （10001 个 patch 文件）
│   │   │   ├── dummy_63/  （10001 个 patch 文件）
│   │   │   ├── dummy_64/  （10001 个 patch 文件）
│   │   │   ├── dummy_65/  （10001 个 patch 文件）
│   │   │   ├── dummy_66/  （10001 个 patch 文件）
│   │   │   ├── dummy_67/  （10001 个 patch 文件）
│   │   │   ├── dummy_68/  （10001 个 patch 文件）
│   │   │   ├── dummy_69/  （10001 个 patch 文件）
│   │   │   ├── dummy_7/  （10001 个 patch 文件）
│   │   │   ├── dummy_70/  （10001 个 patch 文件）
│   │   │   ├── dummy_71/  （10001 个 patch 文件）
│   │   │   ├── dummy_72/  （10001 个 patch 文件）
│   │   │   ├── dummy_73/  （10001 个 patch 文件）
│   │   │   ├── dummy_74/  （10001 个 patch 文件）
│   │   │   ├── dummy_75/  （10001 个 patch 文件）
│   │   │   ├── dummy_76/  （10001 个 patch 文件）
│   │   │   ├── dummy_77/  （10001 个 patch 文件）
│   │   │   ├── dummy_78/  （10001 个 patch 文件）
│   │   │   ├── dummy_79/  （10001 个 patch 文件）
│   │   │   ├── dummy_8/  （10001 个 patch 文件）
│   │   │   ├── dummy_80/  （10001 个 patch 文件）
│   │   │   ├── dummy_81/  （10001 个 patch 文件）
│   │   │   ├── dummy_82/  （10001 个 patch 文件）
│   │   │   ├── dummy_83/  （10001 个 patch 文件）
│   │   │   ├── dummy_84/  （10001 个 patch 文件）
│   │   │   ├── dummy_85/  （10001 个 patch 文件）
│   │   │   ├── dummy_86/  （10001 个 patch 文件）
│   │   │   ├── dummy_87/  （10001 个 patch 文件）
│   │   │   ├── dummy_88/  （10001 个 patch 文件）
│   │   │   ├── dummy_89/  （10001 个 patch 文件）
│   │   │   ├── dummy_9/  （10001 个 patch 文件）
│   │   │   ├── dummy_90/  （10001 个 patch 文件）
│   │   │   ├── dummy_91/  （10001 个 patch 文件）
│   │   │   ├── dummy_92/  （10001 个 patch 文件）
│   │   │   ├── dummy_93/  （10001 个 patch 文件）
│   │   │   ├── dummy_94/  （10001 个 patch 文件）
│   │   │   ├── dummy_95/  （10001 个 patch 文件）
│   │   │   ├── dummy_96/  （10001 个 patch 文件）
│   │   │   ├── dummy_97/  （10001 个 patch 文件）
│   │   │   ├── dummy_98/  （10001 个 patch 文件）
│   │   │   └── dummy_99/  （10001 个 patch 文件）
│   │   ├── dummy_0.tar
│   │   ├── dummy_1.tar
│   │   ├── dummy_10.tar
│   │   └── … 其余 229 个文件（.tar × 230、.log × 1、.py × 1）
│   ├── Pretrain_extra/
│   │   ├── class-ids.npy
│   │   └── entries.npy
│   └── TCGA/
│       └── rawdata/
│           └── TCGA-CESC/
│               ├── TCGA-2W-A8YY-01A-01-TSA.E5237533-AB19-41FB-84D5-80C3FF6D30EF.svs
│               ├── TCGA-2W-A8YY-01Z-00-DX1.2BEC2531-DA98-429B-83BB-3428D3B6FB1E.svs
│               ├── TCGA-4J-AA1J-01A-02-TS2.2191FB69-62CF-4B52-A989-0CE3423A610E.svs
│               └── … 其余 601 个文件（.svs × 604）
├── Graph/
│   ├── embeddings-1024/
│   │   ├── 114855_0722203758966_h_0_coords.npy
│   │   ├── 114855_0722203758966_h_0_features.npy
│   │   ├── 114855_0722203826691_h_0_coords.npy
│   │   └── … 其余 4373 个文件（.npy × 4376）
│   ├── Graph-1024-251111-thre/
│   │   ├── 114855_0722203758966_h_0.pt
│   │   ├── 114855_0722203826691_h_0.pt
│   │   ├── 125908bd0b6d6fa0_0.pt
│   │   └── … 其余 2179 个文件（.pt × 2182）
│   └── Graph-1024-251111-thre0.8/
│       ├── 114855_0722203758966_h_0.pt
│       ├── 114855_0722203826691_h_0.pt
│       ├── 125908bd0b6d6fa0_0.pt
│       └── … 其余 2179 个文件（.pt × 2182）
├── UNI/
│   └── weights/
│       ├── models--MahmoodLab--UNI/
│       │   ├── config.json
│       │   ├── gitattributes
│       │   ├── pytorch_model.bin
│       │   ├── README.md
│       │   ├── requesting_access.png
│       │   └── uni.jpg
│       └── models--MahmoodLab--UNI2-h/
│           ├── config.json
│           ├── config_r.json
│           ├── gitattributes
│           ├── pytorch_model.bin
│           ├── README.md
│           └── requesting_access.png
├── dinov2/
│   ├── .venv/  [Python 虚拟环境（未纳入版本控制）]
│   ├── dinov2/
│   │   ├── configs/
│   │   │   ├── eval/
│   │   │   │   ├── vitb14_pretrain.yaml
│   │   │   │   ├── vitb14_reg4_pretrain.yaml
│   │   │   │   ├── vitg14_pretrain.yaml
│   │   │   │   ├── vitg14_reg4_pretrain.yaml
│   │   │   │   ├── vitl14_pretrain.yaml
│   │   │   │   ├── vitl14_reg4_pretrain.yaml
│   │   │   │   ├── vits14_pretrain.yaml
│   │   │   │   └── vits14_reg4_pretrain.yaml
│   │   │   ├── train/
│   │   │   │   ├── vitg14.yaml
│   │   │   │   ├── vitl14.yaml
│   │   │   │   ├── vitl16_short.yaml
│   │   │   │   └── vitl16_short_imgnet22k.yaml
│   │   │   ├── __init__.py
│   │   │   └── ssl_default_config.yaml
│   │   ├── data/
│   │   │   ├── datasets/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── decoders.py
│   │   │   │   ├── extended.py
│   │   │   │   ├── image_net.py
│   │   │   │   ├── image_net_22k.py
│   │   │   │   └── image_net_22k_self.py
│   │   │   ├── __init__.py
│   │   │   ├── adapters.py
│   │   │   ├── augmentations.py
│   │   │   ├── collate.py
│   │   │   ├── loaders.py
│   │   │   ├── masking.py
│   │   │   ├── samplers.py
│   │   │   └── transforms.py
│   │   ├── distributed/
│   │   │   └── __init__.py
│   │   ├── eval/
│   │   │   ├── depth/
│   │   │   │   ├── models/
│   │   │   │   │   ├── backbones/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── vision_transformer.py
│   │   │   │   │   ├── decode_heads/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── decode_head.py
│   │   │   │   │   │   ├── dpt_head.py
│   │   │   │   │   │   └── linear_head.py
│   │   │   │   │   ├── depther/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── base.py
│   │   │   │   │   │   └── encoder_decoder.py
│   │   │   │   │   ├── losses/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── gradientloss.py
│   │   │   │   │   │   └── sigloss.py
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── builder.py
│   │   │   │   ├── ops/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── wrappers.py
│   │   │   │   └── __init__.py
│   │   │   ├── segmentation/
│   │   │   │   ├── hooks/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── optimizer.py
│   │   │   │   ├── models/
│   │   │   │   │   ├── backbones/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── vision_transformer.py
│   │   │   │   │   ├── decode_heads/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── linear_head.py
│   │   │   │   │   └── __init__.py
│   │   │   │   ├── utils/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── colormaps.py
│   │   │   │   └── __init__.py
│   │   │   ├── segmentation_m2f/
│   │   │   │   ├── core/
│   │   │   │   │   ├── anchor/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── builder.py
│   │   │   │   │   │   └── point_generator.py
│   │   │   │   │   ├── box/
│   │   │   │   │   │   ├── samplers/
│   │   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   │   ├── base_sampler.py
│   │   │   │   │   │   │   ├── mask_pseudo_sampler.py
│   │   │   │   │   │   │   ├── mask_sampling_result.py
│   │   │   │   │   │   │   └── sampling_result.py
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── builder.py
│   │   │   │   │   ├── utils/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── dist_utils.py
│   │   │   │   │   │   └── misc.py
│   │   │   │   │   └── __init__.py
│   │   │   │   ├── models/
│   │   │   │   │   ├── backbones/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── adapter_modules.py
│   │   │   │   │   │   ├── drop_path.py
│   │   │   │   │   │   ├── vit.py
│   │   │   │   │   │   └── vit_adapter.py
│   │   │   │   │   ├── decode_heads/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── mask2former_head.py
│   │   │   │   │   ├── losses/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── cross_entropy_loss.py
│   │   │   │   │   │   ├── dice_loss.py
│   │   │   │   │   │   └── match_costs.py
│   │   │   │   │   ├── plugins/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── msdeformattn_pixel_decoder.py
│   │   │   │   │   ├── segmentors/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── encoder_decoder_mask2former.py
│   │   │   │   │   ├── utils/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── assigner.py
│   │   │   │   │   │   ├── point_sample.py
│   │   │   │   │   │   ├── positional_encoding.py
│   │   │   │   │   │   └── transformer.py
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── builder.py
│   │   │   │   ├── ops/
│   │   │   │   │   └── modules/
│   │   │   │   │       ├── __init__.py
│   │   │   │   │       └── ms_deform_attn.py
│   │   │   │   └── __init__.py
│   │   │   ├── __init__.py
│   │   │   ├── knn.py
│   │   │   ├── linear.py
│   │   │   ├── log_regression.py
│   │   │   ├── metrics.py
│   │   │   ├── setup.py
│   │   │   └── utils.py
│   │   ├── fsdp/
│   │   │   └── __init__.py
│   │   ├── hub/
│   │   │   ├── depth/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── decode_heads.py
│   │   │   │   ├── encoder_decoder.py
│   │   │   │   └── ops.py
│   │   │   ├── text/
│   │   │   │   ├── dinotxt_model.py
│   │   │   │   ├── dinov2_wrapper.py
│   │   │   │   ├── text_tower.py
│   │   │   │   ├── text_transformer.py
│   │   │   │   ├── tokenizer.py
│   │   │   │   └── vision_tower.py
│   │   │   ├── __init__.py
│   │   │   ├── backbones.py
│   │   │   ├── classifiers.py
│   │   │   ├── depthers.py
│   │   │   ├── dinotxt.py
│   │   │   └── utils.py
│   │   ├── layers/
│   │   │   ├── __init__.py
│   │   │   ├── attention.py
│   │   │   ├── block.py
│   │   │   ├── dino_head.py
│   │   │   ├── drop_path.py
│   │   │   ├── layer_scale.py
│   │   │   ├── mlp.py
│   │   │   ├── patch_embed.py
│   │   │   └── swiglu_ffn.py
│   │   ├── logging/
│   │   │   ├── __init__.py
│   │   │   └── helpers.py
│   │   ├── loss/
│   │   │   ├── __init__.py
│   │   │   ├── dino_clstoken_loss.py
│   │   │   ├── ibot_patch_loss.py
│   │   │   └── koleo_loss.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── vision_transformer.py
│   │   ├── results/  [训练产物（权重/日志/checkpoint，未纳入版本控制）]
│   │   │   └── dinov2_he1023_pretrain-resume-1024/  （含 28 个文件）
│   │   ├── run/
│   │   │   ├── eval/
│   │   │   │   ├── knn.py
│   │   │   │   ├── linear.py
│   │   │   │   └── log_regression.py
│   │   │   ├── train/
│   │   │   │   ├── model_merge.py
│   │   │   │   └── train.py
│   │   │   ├── __init__.py
│   │   │   └── submit.py
│   │   ├── thirdparty/
│   │   │   └── CLIP/
│   │   │       ├── clip/
│   │   │       │   └── simple_tokenizer.py
│   │   │       └── LICENSE
│   │   ├── train/
│   │   │   ├── __init__.py
│   │   │   ├── checkpoint_merge_fsdp.py
│   │   │   ├── ssl_meta_arch.py
│   │   │   ├── ssl_meta_arch_stage2.py
│   │   │   └── train.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── cluster.py
│   │   │   ├── config.py
│   │   │   ├── dtype.py
│   │   │   ├── param_groups.py
│   │   │   └── utils.py
│   │   └── __init__.py
│   ├── docs/
│   │   ├── ChannelAdaptiveDINO.png
│   │   └── README_CHANNEL_ADAPTIVE_DINO.md
│   ├── notebooks/
│   │   ├── depth_estimation.ipynb
│   │   ├── dinotxt.ipynb
│   │   └── semantic_segmentation.ipynb
│   ├── scripts/
│   │   └── lint.sh
│   ├── .gitignore
│   ├── checkpoint_merge_fsdp.sh
│   ├── CODE_OF_CONDUCT.md
│   └── … 其余 15 个文件（.md × 4、.txt × 3、.sh × 2、.yaml × 2、.py × 2、.gitignore × 1、(无扩展名) × 1、.toml × 1、.cfg × 1、.ipynb × 1）
├── dinov2_finetune/
│   ├── .github/
│   │   └── workflows/
│   │       └── lint.yaml
│   ├── dinov2/
│   │   ├── configs/
│   │   │   ├── eval/
│   │   │   │   ├── vitb14_pretrain.yaml
│   │   │   │   ├── vitb14_reg4_pretrain.yaml
│   │   │   │   ├── vitg14_pretrain.yaml
│   │   │   │   ├── vitg14_reg4_pretrain.yaml
│   │   │   │   ├── vitl14_pretrain.yaml
│   │   │   │   ├── vitl14_reg4_pretrain.yaml
│   │   │   │   ├── vits14_pretrain.yaml
│   │   │   │   └── vits14_reg4_pretrain.yaml
│   │   │   ├── train/
│   │   │   │   ├── vitg14.yaml
│   │   │   │   ├── vitl14.yaml
│   │   │   │   ├── vitl16_short.yaml
│   │   │   │   └── vitl16_short_imgnet22k.yaml
│   │   │   ├── __init__.py
│   │   │   └── ssl_default_config.yaml
│   │   ├── data/
│   │   │   ├── datasets/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── decoders.py
│   │   │   │   ├── extended.py
│   │   │   │   ├── image_folder.py
│   │   │   │   ├── image_net.py
│   │   │   │   ├── image_net_22k.py
│   │   │   │   └── image_net_22k_self.py
│   │   │   ├── __init__.py
│   │   │   ├── adapters.py
│   │   │   ├── augmentations.py
│   │   │   ├── collate.py
│   │   │   ├── loaders.py
│   │   │   ├── masking.py
│   │   │   ├── samplers.py
│   │   │   └── transforms.py
│   │   ├── distributed/
│   │   │   └── __init__.py
│   │   ├── eval/
│   │   │   ├── depth/
│   │   │   │   ├── models/
│   │   │   │   │   ├── backbones/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── vision_transformer.py
│   │   │   │   │   ├── decode_heads/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── decode_head.py
│   │   │   │   │   │   ├── dpt_head.py
│   │   │   │   │   │   └── linear_head.py
│   │   │   │   │   ├── depther/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── base.py
│   │   │   │   │   │   └── encoder_decoder.py
│   │   │   │   │   ├── losses/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── gradientloss.py
│   │   │   │   │   │   └── sigloss.py
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── builder.py
│   │   │   │   ├── ops/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── wrappers.py
│   │   │   │   └── __init__.py
│   │   │   ├── segmentation/
│   │   │   │   ├── hooks/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── optimizer.py
│   │   │   │   ├── models/
│   │   │   │   │   ├── backbones/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── vision_transformer.py
│   │   │   │   │   ├── decode_heads/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── linear_head.py
│   │   │   │   │   └── __init__.py
│   │   │   │   ├── utils/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── colormaps.py
│   │   │   │   └── __init__.py
│   │   │   ├── segmentation_m2f/
│   │   │   │   ├── core/
│   │   │   │   │   ├── anchor/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── builder.py
│   │   │   │   │   │   └── point_generator.py
│   │   │   │   │   ├── box/
│   │   │   │   │   │   ├── samplers/
│   │   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   │   ├── base_sampler.py
│   │   │   │   │   │   │   ├── mask_pseudo_sampler.py
│   │   │   │   │   │   │   ├── mask_sampling_result.py
│   │   │   │   │   │   │   └── sampling_result.py
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── builder.py
│   │   │   │   │   ├── utils/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── dist_utils.py
│   │   │   │   │   │   └── misc.py
│   │   │   │   │   └── __init__.py
│   │   │   │   ├── models/
│   │   │   │   │   ├── backbones/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── adapter_modules.py
│   │   │   │   │   │   ├── drop_path.py
│   │   │   │   │   │   ├── vit.py
│   │   │   │   │   │   └── vit_adapter.py
│   │   │   │   │   ├── decode_heads/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── mask2former_head.py
│   │   │   │   │   ├── losses/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── cross_entropy_loss.py
│   │   │   │   │   │   ├── dice_loss.py
│   │   │   │   │   │   └── match_costs.py
│   │   │   │   │   ├── plugins/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── msdeformattn_pixel_decoder.py
│   │   │   │   │   ├── segmentors/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── encoder_decoder_mask2former.py
│   │   │   │   │   ├── utils/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── assigner.py
│   │   │   │   │   │   ├── point_sample.py
│   │   │   │   │   │   ├── positional_encoding.py
│   │   │   │   │   │   └── transformer.py
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── builder.py
│   │   │   │   ├── ops/
│   │   │   │   │   └── modules/
│   │   │   │   │       ├── __init__.py
│   │   │   │   │       └── ms_deform_attn.py
│   │   │   │   └── __init__.py
│   │   │   ├── __init__.py
│   │   │   ├── knn.py
│   │   │   ├── linear.py
│   │   │   ├── log_regression.py
│   │   │   ├── metrics.py
│   │   │   ├── setup.py
│   │   │   └── utils.py
│   │   ├── fsdp/
│   │   │   └── __init__.py
│   │   ├── hub/
│   │   │   ├── depth/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── decode_heads.py
│   │   │   │   ├── encoder_decoder.py
│   │   │   │   └── ops.py
│   │   │   ├── text/
│   │   │   │   ├── dinotxt_model.py
│   │   │   │   ├── dinov2_wrapper.py
│   │   │   │   ├── text_tower.py
│   │   │   │   ├── text_transformer.py
│   │   │   │   ├── tokenizer.py
│   │   │   │   └── vision_tower.py
│   │   │   ├── __init__.py
│   │   │   ├── backbones.py
│   │   │   ├── classifiers.py
│   │   │   ├── depthers.py
│   │   │   ├── dinotxt.py
│   │   │   └── utils.py
│   │   ├── layers/
│   │   │   ├── __init__.py
│   │   │   ├── attention.py
│   │   │   ├── block.py
│   │   │   ├── dino_head.py
│   │   │   ├── drop_path.py
│   │   │   ├── layer_scale.py
│   │   │   ├── mlp.py
│   │   │   ├── patch_embed.py
│   │   │   └── swiglu_ffn.py
│   │   ├── logging/
│   │   │   ├── __init__.py
│   │   │   └── helpers.py
│   │   ├── loss/
│   │   │   ├── __init__.py
│   │   │   ├── dino_clstoken_loss.py
│   │   │   ├── ibot_patch_loss.py
│   │   │   └── koleo_loss.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── gcn.py
│   │   │   └── vision_transformer.py
│   │   ├── run/
│   │   │   ├── eval/
│   │   │   │   ├── knn.py
│   │   │   │   ├── linear.py
│   │   │   │   └── log_regression.py
│   │   │   ├── train/
│   │   │   │   └── train.py
│   │   │   ├── __init__.py
│   │   │   └── submit.py
│   │   ├── thirdparty/
│   │   │   └── CLIP/
│   │   │       ├── clip/
│   │   │       │   └── simple_tokenizer.py
│   │   │       └── LICENSE
│   │   ├── train/
│   │   │   ├── __init__.py
│   │   │   ├── gcn_meta_arch.py
│   │   │   ├── ssl_meta_arch.py
│   │   │   └── train.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── cluster.py
│   │   │   ├── config.py
│   │   │   ├── dtype.py
│   │   │   ├── param_groups.py
│   │   │   └── utils.py
│   │   └── __init__.py
│   ├── docs/
│   │   ├── ChannelAdaptiveDINO.png
│   │   └── README_CHANNEL_ADAPTIVE_DINO.md
│   ├── notebooks/
│   │   ├── depth_estimation.ipynb
│   │   ├── dinotxt.ipynb
│   │   └── semantic_segmentation.ipynb
│   ├── scripts/
│   │   └── lint.sh
│   ├── .gitignore
│   ├── checkpoint_merge_fsdp.py
│   ├── checkpoint_merge_fsdp.sh
│   └── … 其余 17 个文件（.md × 4、.py × 3、.sh × 3、.txt × 3、.yaml × 2、.gitignore × 1、(无扩展名) × 1、.toml × 1、.cfg × 1、.ipynb × 1）
├── dinov2_stage1_Extract2s2/
│   ├── .github/
│   │   └── workflows/
│   │       └── lint.yaml
│   ├── dinov2/
│   │   ├── configs/
│   │   │   ├── eval/
│   │   │   │   ├── vitb14_pretrain.yaml
│   │   │   │   ├── vitb14_reg4_pretrain.yaml
│   │   │   │   ├── vitg14_pretrain.yaml
│   │   │   │   ├── vitg14_reg4_pretrain.yaml
│   │   │   │   ├── vitl14_pretrain.yaml
│   │   │   │   ├── vitl14_reg4_pretrain.yaml
│   │   │   │   ├── vits14_pretrain.yaml
│   │   │   │   └── vits14_reg4_pretrain.yaml
│   │   │   ├── train/
│   │   │   │   ├── vitg14.yaml
│   │   │   │   ├── vitl14.yaml
│   │   │   │   ├── vitl16_short.yaml
│   │   │   │   └── vitl16_short_imgnet22k.yaml
│   │   │   ├── __init__.py
│   │   │   └── ssl_default_config.yaml
│   │   ├── data/
│   │   │   ├── datasets/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── decoders.py
│   │   │   │   ├── extended.py
│   │   │   │   ├── image_folder.py
│   │   │   │   ├── image_net.py
│   │   │   │   ├── image_net_22k.py
│   │   │   │   └── image_net_22k_self.py
│   │   │   ├── __init__.py
│   │   │   ├── adapters.py
│   │   │   ├── augmentations.py
│   │   │   ├── collate.py
│   │   │   ├── loaders.py
│   │   │   ├── masking.py
│   │   │   ├── samplers.py
│   │   │   └── transforms.py
│   │   ├── distributed/
│   │   │   └── __init__.py
│   │   ├── eval/
│   │   │   ├── depth/
│   │   │   │   ├── models/
│   │   │   │   │   ├── backbones/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── vision_transformer.py
│   │   │   │   │   ├── decode_heads/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── decode_head.py
│   │   │   │   │   │   ├── dpt_head.py
│   │   │   │   │   │   └── linear_head.py
│   │   │   │   │   ├── depther/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── base.py
│   │   │   │   │   │   └── encoder_decoder.py
│   │   │   │   │   ├── losses/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── gradientloss.py
│   │   │   │   │   │   └── sigloss.py
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── builder.py
│   │   │   │   ├── ops/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── wrappers.py
│   │   │   │   └── __init__.py
│   │   │   ├── segmentation/
│   │   │   │   ├── hooks/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── optimizer.py
│   │   │   │   ├── models/
│   │   │   │   │   ├── backbones/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── vision_transformer.py
│   │   │   │   │   ├── decode_heads/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── linear_head.py
│   │   │   │   │   └── __init__.py
│   │   │   │   ├── utils/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── colormaps.py
│   │   │   │   └── __init__.py
│   │   │   ├── segmentation_m2f/
│   │   │   │   ├── core/
│   │   │   │   │   ├── anchor/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── builder.py
│   │   │   │   │   │   └── point_generator.py
│   │   │   │   │   ├── box/
│   │   │   │   │   │   ├── samplers/
│   │   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   │   ├── base_sampler.py
│   │   │   │   │   │   │   ├── mask_pseudo_sampler.py
│   │   │   │   │   │   │   ├── mask_sampling_result.py
│   │   │   │   │   │   │   └── sampling_result.py
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── builder.py
│   │   │   │   │   ├── utils/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── dist_utils.py
│   │   │   │   │   │   └── misc.py
│   │   │   │   │   └── __init__.py
│   │   │   │   ├── models/
│   │   │   │   │   ├── backbones/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── adapter_modules.py
│   │   │   │   │   │   ├── drop_path.py
│   │   │   │   │   │   ├── vit.py
│   │   │   │   │   │   └── vit_adapter.py
│   │   │   │   │   ├── decode_heads/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── mask2former_head.py
│   │   │   │   │   ├── losses/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── cross_entropy_loss.py
│   │   │   │   │   │   ├── dice_loss.py
│   │   │   │   │   │   └── match_costs.py
│   │   │   │   │   ├── plugins/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── msdeformattn_pixel_decoder.py
│   │   │   │   │   ├── segmentors/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── encoder_decoder_mask2former.py
│   │   │   │   │   ├── utils/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── assigner.py
│   │   │   │   │   │   ├── point_sample.py
│   │   │   │   │   │   ├── positional_encoding.py
│   │   │   │   │   │   └── transformer.py
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── builder.py
│   │   │   │   ├── ops/
│   │   │   │   │   └── modules/
│   │   │   │   │       ├── __init__.py
│   │   │   │   │       └── ms_deform_attn.py
│   │   │   │   └── __init__.py
│   │   │   ├── __init__.py
│   │   │   ├── knn.py
│   │   │   ├── linear.py
│   │   │   ├── log_regression.py
│   │   │   ├── metrics.py
│   │   │   ├── setup.py
│   │   │   └── utils.py
│   │   ├── fsdp/
│   │   │   └── __init__.py
│   │   ├── hub/
│   │   │   ├── depth/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── decode_heads.py
│   │   │   │   ├── encoder_decoder.py
│   │   │   │   └── ops.py
│   │   │   ├── text/
│   │   │   │   ├── dinotxt_model.py
│   │   │   │   ├── dinov2_wrapper.py
│   │   │   │   ├── text_tower.py
│   │   │   │   ├── text_transformer.py
│   │   │   │   ├── tokenizer.py
│   │   │   │   └── vision_tower.py
│   │   │   ├── __init__.py
│   │   │   ├── backbones.py
│   │   │   ├── classifiers.py
│   │   │   ├── depthers.py
│   │   │   ├── dinotxt.py
│   │   │   └── utils.py
│   │   ├── layers/
│   │   │   ├── __init__.py
│   │   │   ├── attention.py
│   │   │   ├── block.py
│   │   │   ├── dino_head.py
│   │   │   ├── drop_path.py
│   │   │   ├── layer_scale.py
│   │   │   ├── mlp.py
│   │   │   ├── patch_embed.py
│   │   │   └── swiglu_ffn.py
│   │   ├── logging/
│   │   │   ├── __init__.py
│   │   │   └── helpers.py
│   │   ├── loss/
│   │   │   ├── __init__.py
│   │   │   ├── dino_clstoken_loss.py
│   │   │   ├── ibot_patch_loss.py
│   │   │   └── koleo_loss.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── gcn.py
│   │   │   └── vision_transformer.py
│   │   ├── results/  [训练产物（权重/日志/checkpoint，未纳入版本控制）]
│   │   │   ├── pretrained_s1_embeddings-1024/  （含 7 个文件）
│   │   │   └── pretrained_s1_embeddings-65536/  （含 14 个文件）
│   │   ├── run/
│   │   │   ├── eval/
│   │   │   │   ├── knn.py
│   │   │   │   ├── linear.py
│   │   │   │   └── log_regression.py
│   │   │   ├── train/
│   │   │   │   └── train.py
│   │   │   ├── __init__.py
│   │   │   └── submit.py
│   │   ├── thirdparty/
│   │   │   └── CLIP/
│   │   │       ├── clip/
│   │   │       │   └── simple_tokenizer.py
│   │   │       └── LICENSE
│   │   ├── train/
│   │   │   ├── __init__.py
│   │   │   ├── gcn_meta_arch.py
│   │   │   ├── ssl_meta_arch.py
│   │   │   └── train.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── cluster.py
│   │   │   ├── config.py
│   │   │   ├── dtype.py
│   │   │   ├── param_groups.py
│   │   │   └── utils.py
│   │   └── __init__.py
│   ├── docs/
│   │   ├── ChannelAdaptiveDINO.png
│   │   └── README_CHANNEL_ADAPTIVE_DINO.md
│   ├── notebooks/
│   │   ├── depth_estimation.ipynb
│   │   ├── dinotxt.ipynb
│   │   └── semantic_segmentation.ipynb
│   ├── scripts/
│   │   └── lint.sh
│   ├── .gitignore
│   ├── checkpoint_merge_fsdp.py
│   ├── checkpoint_merge_fsdp.sh
│   └── … 其余 17 个文件（.md × 4、.py × 3、.sh × 3、.txt × 3、.yaml × 2、.gitignore × 1、(无扩展名) × 1、.toml × 1、.cfg × 1、.ipynb × 1）
├── dinov2_stage1_Extract2s2_local/
│   ├── dinov2/
│   │   ├── configs/
│   │   │   ├── eval/
│   │   │   │   ├── vitb14_pretrain.yaml
│   │   │   │   ├── vitb14_reg4_pretrain.yaml
│   │   │   │   ├── vitg14_pretrain.yaml
│   │   │   │   ├── vitg14_reg4_pretrain.yaml
│   │   │   │   ├── vitl14_pretrain.yaml
│   │   │   │   ├── vitl14_reg4_pretrain.yaml
│   │   │   │   ├── vits14_pretrain.yaml
│   │   │   │   └── vits14_reg4_pretrain.yaml
│   │   │   ├── train/
│   │   │   │   ├── vitg14.yaml
│   │   │   │   ├── vitl14.yaml
│   │   │   │   ├── vitl16_short.yaml
│   │   │   │   └── vitl16_short_imgnet22k.yaml
│   │   │   ├── __init__.py
│   │   │   └── ssl_default_config.yaml
│   │   ├── data/
│   │   │   ├── datasets/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── decoders.py
│   │   │   │   ├── extended.py
│   │   │   │   ├── image_folder.py
│   │   │   │   ├── image_net.py
│   │   │   │   ├── image_net_22k.py
│   │   │   │   └── image_net_22k_self.py
│   │   │   ├── __init__.py
│   │   │   ├── adapters.py
│   │   │   ├── augmentations.py
│   │   │   ├── collate.py
│   │   │   ├── loaders.py
│   │   │   ├── masking.py
│   │   │   ├── samplers.py
│   │   │   └── transforms.py
│   │   ├── distributed/
│   │   │   └── __init__.py
│   │   ├── eval/
│   │   │   ├── depth/
│   │   │   │   ├── models/
│   │   │   │   │   ├── backbones/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── vision_transformer.py
│   │   │   │   │   ├── decode_heads/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── decode_head.py
│   │   │   │   │   │   ├── dpt_head.py
│   │   │   │   │   │   └── linear_head.py
│   │   │   │   │   ├── depther/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── base.py
│   │   │   │   │   │   └── encoder_decoder.py
│   │   │   │   │   ├── losses/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── gradientloss.py
│   │   │   │   │   │   └── sigloss.py
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── builder.py
│   │   │   │   ├── ops/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── wrappers.py
│   │   │   │   └── __init__.py
│   │   │   ├── segmentation/
│   │   │   │   ├── hooks/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── optimizer.py
│   │   │   │   ├── models/
│   │   │   │   │   ├── backbones/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── vision_transformer.py
│   │   │   │   │   ├── decode_heads/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── linear_head.py
│   │   │   │   │   └── __init__.py
│   │   │   │   ├── utils/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── colormaps.py
│   │   │   │   └── __init__.py
│   │   │   ├── segmentation_m2f/
│   │   │   │   ├── core/
│   │   │   │   │   ├── anchor/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── builder.py
│   │   │   │   │   │   └── point_generator.py
│   │   │   │   │   ├── box/
│   │   │   │   │   │   ├── samplers/
│   │   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   │   ├── base_sampler.py
│   │   │   │   │   │   │   ├── mask_pseudo_sampler.py
│   │   │   │   │   │   │   ├── mask_sampling_result.py
│   │   │   │   │   │   │   └── sampling_result.py
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── builder.py
│   │   │   │   │   ├── utils/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── dist_utils.py
│   │   │   │   │   │   └── misc.py
│   │   │   │   │   └── __init__.py
│   │   │   │   ├── models/
│   │   │   │   │   ├── backbones/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── adapter_modules.py
│   │   │   │   │   │   ├── drop_path.py
│   │   │   │   │   │   ├── vit.py
│   │   │   │   │   │   └── vit_adapter.py
│   │   │   │   │   ├── decode_heads/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── mask2former_head.py
│   │   │   │   │   ├── losses/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── cross_entropy_loss.py
│   │   │   │   │   │   ├── dice_loss.py
│   │   │   │   │   │   └── match_costs.py
│   │   │   │   │   ├── plugins/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── msdeformattn_pixel_decoder.py
│   │   │   │   │   ├── segmentors/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── encoder_decoder_mask2former.py
│   │   │   │   │   ├── utils/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── assigner.py
│   │   │   │   │   │   ├── point_sample.py
│   │   │   │   │   │   ├── positional_encoding.py
│   │   │   │   │   │   └── transformer.py
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── builder.py
│   │   │   │   ├── ops/
│   │   │   │   │   └── modules/
│   │   │   │   │       ├── __init__.py
│   │   │   │   │       └── ms_deform_attn.py
│   │   │   │   └── __init__.py
│   │   │   ├── __init__.py
│   │   │   ├── knn.py
│   │   │   ├── linear.py
│   │   │   ├── log_regression.py
│   │   │   ├── metrics.py
│   │   │   ├── setup.py
│   │   │   └── utils.py
│   │   ├── fsdp/
│   │   │   └── __init__.py
│   │   ├── hub/
│   │   │   ├── depth/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── decode_heads.py
│   │   │   │   ├── encoder_decoder.py
│   │   │   │   └── ops.py
│   │   │   ├── text/
│   │   │   │   ├── dinotxt_model.py
│   │   │   │   ├── dinov2_wrapper.py
│   │   │   │   ├── text_tower.py
│   │   │   │   ├── text_transformer.py
│   │   │   │   ├── tokenizer.py
│   │   │   │   └── vision_tower.py
│   │   │   ├── __init__.py
│   │   │   ├── backbones.py
│   │   │   ├── classifiers.py
│   │   │   ├── depthers.py
│   │   │   ├── dinotxt.py
│   │   │   └── utils.py
│   │   ├── layers/
│   │   │   ├── __init__.py
│   │   │   ├── attention.py
│   │   │   ├── block.py
│   │   │   ├── dino_head.py
│   │   │   ├── drop_path.py
│   │   │   ├── layer_scale.py
│   │   │   ├── mlp.py
│   │   │   ├── patch_embed.py
│   │   │   └── swiglu_ffn.py
│   │   ├── logging/
│   │   │   ├── __init__.py
│   │   │   └── helpers.py
│   │   ├── loss/
│   │   │   ├── __init__.py
│   │   │   ├── dino_clstoken_loss.py
│   │   │   ├── ibot_patch_loss.py
│   │   │   └── koleo_loss.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── gcn.py
│   │   │   └── vision_transformer.py
│   │   ├── run/
│   │   │   ├── eval/
│   │   │   │   ├── knn.py
│   │   │   │   ├── linear.py
│   │   │   │   └── log_regression.py
│   │   │   ├── train/
│   │   │   │   └── train.py
│   │   │   ├── __init__.py
│   │   │   └── submit.py
│   │   ├── thirdparty/
│   │   │   └── CLIP/
│   │   │       ├── clip/
│   │   │       │   └── simple_tokenizer.py
│   │   │       └── LICENSE
│   │   ├── train/
│   │   │   ├── __init__.py
│   │   │   ├── gcn_meta_arch.py
│   │   │   ├── ssl_meta_arch.py
│   │   │   └── train.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── cluster.py
│   │   │   ├── config.py
│   │   │   ├── dtype.py
│   │   │   ├── param_groups.py
│   │   │   └── utils.py
│   │   └── __init__.py
│   ├── docs/
│   │   ├── ChannelAdaptiveDINO.png
│   │   └── README_CHANNEL_ADAPTIVE_DINO.md
│   ├── notebooks/
│   │   ├── depth_estimation.ipynb
│   │   ├── dinotxt.ipynb
│   │   └── semantic_segmentation.ipynb
│   └── script/
│       ├── pretrain.sh
│       └── pretrain_imgnet22k.sh
├── dinov2_stage2_2_FmH2ST/
│   ├── .github/
│   │   └── workflows/
│   │       └── lint.yaml
│   ├── dinov2/
│   │   ├── configs/
│   │   │   ├── eval/
│   │   │   │   ├── vitb14_pretrain.yaml
│   │   │   │   ├── vitb14_reg4_pretrain.yaml
│   │   │   │   ├── vitg14_pretrain.yaml
│   │   │   │   ├── vitg14_reg4_pretrain.yaml
│   │   │   │   ├── vitl14_pretrain.yaml
│   │   │   │   ├── vitl14_reg4_pretrain.yaml
│   │   │   │   ├── vits14_pretrain.yaml
│   │   │   │   └── vits14_reg4_pretrain.yaml
│   │   │   ├── train/
│   │   │   │   ├── vitg14.yaml
│   │   │   │   ├── vitl14.yaml
│   │   │   │   ├── vitl16_short.yaml
│   │   │   │   └── vitl16_short_imgnet22k.yaml
│   │   │   ├── __init__.py
│   │   │   └── ssl_default_config.yaml
│   │   ├── data/
│   │   │   ├── datasets/
│   │   │   │   ├── 126173_0615115446229_h_0.jpg
│   │   │   │   ├── 134402_0111162216505_h_0.jpg
│   │   │   │   ├── 136883_0314180117113_h_0.jpg
│   │   │   │   ├── __init__.py
│   │   │   │   ├── decoders.py
│   │   │   │   ├── extended.py
│   │   │   │   ├── graph_build.ipynb
│   │   │   │   ├── image_folder.py
│   │   │   │   ├── image_net.py
│   │   │   │   ├── image_net_22k.py
│   │   │   │   └── image_net_22k_self.py
│   │   │   ├── __init__.py
│   │   │   ├── adapters.py
│   │   │   ├── augmentations.py
│   │   │   ├── collate.py
│   │   │   ├── loaders.py
│   │   │   ├── masking.py
│   │   │   ├── samplers.py
│   │   │   └── transforms.py
│   │   ├── distributed/
│   │   │   └── __init__.py
│   │   ├── eval/
│   │   │   ├── depth/
│   │   │   │   ├── models/
│   │   │   │   │   ├── backbones/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── vision_transformer.py
│   │   │   │   │   ├── decode_heads/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── decode_head.py
│   │   │   │   │   │   ├── dpt_head.py
│   │   │   │   │   │   └── linear_head.py
│   │   │   │   │   ├── depther/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── base.py
│   │   │   │   │   │   └── encoder_decoder.py
│   │   │   │   │   ├── losses/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── gradientloss.py
│   │   │   │   │   │   └── sigloss.py
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── builder.py
│   │   │   │   ├── ops/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── wrappers.py
│   │   │   │   └── __init__.py
│   │   │   ├── segmentation/
│   │   │   │   ├── hooks/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── optimizer.py
│   │   │   │   ├── models/
│   │   │   │   │   ├── backbones/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── vision_transformer.py
│   │   │   │   │   ├── decode_heads/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── linear_head.py
│   │   │   │   │   └── __init__.py
│   │   │   │   ├── utils/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── colormaps.py
│   │   │   │   └── __init__.py
│   │   │   ├── segmentation_m2f/
│   │   │   │   ├── core/
│   │   │   │   │   ├── anchor/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── builder.py
│   │   │   │   │   │   └── point_generator.py
│   │   │   │   │   ├── box/
│   │   │   │   │   │   ├── samplers/
│   │   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   │   ├── base_sampler.py
│   │   │   │   │   │   │   ├── mask_pseudo_sampler.py
│   │   │   │   │   │   │   ├── mask_sampling_result.py
│   │   │   │   │   │   │   └── sampling_result.py
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── builder.py
│   │   │   │   │   ├── utils/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── dist_utils.py
│   │   │   │   │   │   └── misc.py
│   │   │   │   │   └── __init__.py
│   │   │   │   ├── models/
│   │   │   │   │   ├── backbones/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── adapter_modules.py
│   │   │   │   │   │   ├── drop_path.py
│   │   │   │   │   │   ├── vit.py
│   │   │   │   │   │   └── vit_adapter.py
│   │   │   │   │   ├── decode_heads/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── mask2former_head.py
│   │   │   │   │   ├── losses/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── cross_entropy_loss.py
│   │   │   │   │   │   ├── dice_loss.py
│   │   │   │   │   │   └── match_costs.py
│   │   │   │   │   ├── plugins/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── msdeformattn_pixel_decoder.py
│   │   │   │   │   ├── segmentors/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── encoder_decoder_mask2former.py
│   │   │   │   │   ├── utils/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── assigner.py
│   │   │   │   │   │   ├── point_sample.py
│   │   │   │   │   │   ├── positional_encoding.py
│   │   │   │   │   │   └── transformer.py
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── builder.py
│   │   │   │   ├── ops/
│   │   │   │   │   └── modules/
│   │   │   │   │       ├── __init__.py
│   │   │   │   │       └── ms_deform_attn.py
│   │   │   │   └── __init__.py
│   │   │   ├── __init__.py
│   │   │   ├── knn.py
│   │   │   ├── linear.py
│   │   │   ├── log_regression.py
│   │   │   ├── metrics.py
│   │   │   ├── setup.py
│   │   │   └── utils.py
│   │   ├── fsdp/
│   │   │   └── __init__.py
│   │   ├── hub/
│   │   │   ├── depth/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── decode_heads.py
│   │   │   │   ├── encoder_decoder.py
│   │   │   │   └── ops.py
│   │   │   ├── text/
│   │   │   │   ├── dinotxt_model.py
│   │   │   │   ├── dinov2_wrapper.py
│   │   │   │   ├── text_tower.py
│   │   │   │   ├── text_transformer.py
│   │   │   │   ├── tokenizer.py
│   │   │   │   └── vision_tower.py
│   │   │   ├── __init__.py
│   │   │   ├── backbones.py
│   │   │   ├── classifiers.py
│   │   │   ├── depthers.py
│   │   │   ├── dinotxt.py
│   │   │   └── utils.py
│   │   ├── layers/
│   │   │   ├── __init__.py
│   │   │   ├── attention.py
│   │   │   ├── block.py
│   │   │   ├── dino_head.py
│   │   │   ├── drop_path.py
│   │   │   ├── layer_scale.py
│   │   │   ├── mlp.py
│   │   │   ├── patch_embed.py
│   │   │   └── swiglu_ffn.py
│   │   ├── logging/
│   │   │   ├── __init__.py
│   │   │   └── helpers.py
│   │   ├── loss/
│   │   │   ├── __init__.py
│   │   │   ├── dino_clstoken_loss.py
│   │   │   ├── ibot_patch_loss.py
│   │   │   └── koleo_loss.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── gcn.py
│   │   │   └── vision_transformer.py
│   │   ├── results/  [训练产物（权重/日志/checkpoint，未纳入版本控制）]
│   │   │   ├── gcn_1112_pretrain_embeddings-1024/  （含 135 个文件）
│   │   │   ├── gcn_1112_pretrain_embeddings-1024-b4/  （含 111 个文件）
│   │   │   └── gcn_1112_pretrain_embeddings-1024-b4-mask75/  （含 129 个文件）
│   │   ├── run/
│   │   │   ├── eval/
│   │   │   │   ├── knn.py
│   │   │   │   ├── linear.py
│   │   │   │   └── log_regression.py
│   │   │   ├── train/
│   │   │   │   └── train.py
│   │   │   ├── __init__.py
│   │   │   └── submit.py
│   │   ├── thirdparty/
│   │   │   └── CLIP/
│   │   │       ├── clip/
│   │   │       │   └── simple_tokenizer.py
│   │   │       └── LICENSE
│   │   ├── train/
│   │   │   ├── __init__.py
│   │   │   ├── gcn_meta_arch.py
│   │   │   ├── ssl_meta_arch.py
│   │   │   └── train.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── cluster.py
│   │   │   ├── config.py
│   │   │   ├── dtype.py
│   │   │   ├── param_groups.py
│   │   │   └── utils.py
│   │   └── __init__.py
│   ├── docs/
│   │   ├── ChannelAdaptiveDINO.png
│   │   └── README_CHANNEL_ADAPTIVE_DINO.md
│   ├── notebooks/
│   │   ├── depth_estimation.ipynb
│   │   ├── dinotxt.ipynb
│   │   └── semantic_segmentation.ipynb
│   ├── scripts/
│   │   └── lint.sh
│   ├── .gitignore
│   ├── checkpoint_merge_fsdp.py
│   ├── checkpoint_merge_fsdp.sh
│   └── … 其余 16 个文件（.md × 4、.py × 3、.txt × 3、.sh × 2、.yaml × 2、.gitignore × 1、(无扩展名) × 1、.toml × 1、.cfg × 1、.ipynb × 1）
├── dinov2_stage2_2_FmH2ST_finetune/
│   ├── dinov2/
│   │   ├── configs/
│   │   │   ├── eval/
│   │   │   │   ├── vitb14_pretrain.yaml
│   │   │   │   ├── vitb14_reg4_pretrain.yaml
│   │   │   │   ├── vitg14_pretrain.yaml
│   │   │   │   ├── vitg14_reg4_pretrain.yaml
│   │   │   │   ├── vitl14_pretrain.yaml
│   │   │   │   ├── vitl14_reg4_pretrain.yaml
│   │   │   │   ├── vits14_pretrain.yaml
│   │   │   │   └── vits14_reg4_pretrain.yaml
│   │   │   ├── train/
│   │   │   │   ├── vitg14.yaml
│   │   │   │   ├── vitl14.yaml
│   │   │   │   ├── vitl16_short.yaml
│   │   │   │   └── vitl16_short_imgnet22k.yaml
│   │   │   ├── __init__.py
│   │   │   └── ssl_default_config.yaml
│   │   ├── data/
│   │   │   ├── datasets/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── decoders.py
│   │   │   │   ├── extended.py
│   │   │   │   ├── graph_build.ipynb
│   │   │   │   ├── image_folder.py
│   │   │   │   ├── image_net.py
│   │   │   │   ├── image_net_22k.py
│   │   │   │   └── image_net_22k_self.py
│   │   │   ├── __init__.py
│   │   │   ├── adapters.py
│   │   │   ├── augmentations.py
│   │   │   ├── collate.py
│   │   │   ├── loaders.py
│   │   │   ├── masking.py
│   │   │   ├── samplers.py
│   │   │   └── transforms.py
│   │   ├── distributed/
│   │   │   └── __init__.py
│   │   ├── eval/
│   │   │   ├── depth/
│   │   │   │   ├── models/
│   │   │   │   │   ├── backbones/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── vision_transformer.py
│   │   │   │   │   ├── decode_heads/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── decode_head.py
│   │   │   │   │   │   ├── dpt_head.py
│   │   │   │   │   │   └── linear_head.py
│   │   │   │   │   ├── depther/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── base.py
│   │   │   │   │   │   └── encoder_decoder.py
│   │   │   │   │   ├── losses/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── gradientloss.py
│   │   │   │   │   │   └── sigloss.py
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── builder.py
│   │   │   │   ├── ops/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── wrappers.py
│   │   │   │   └── __init__.py
│   │   │   ├── segmentation/
│   │   │   │   ├── hooks/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── optimizer.py
│   │   │   │   ├── models/
│   │   │   │   │   ├── backbones/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── vision_transformer.py
│   │   │   │   │   ├── decode_heads/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── linear_head.py
│   │   │   │   │   └── __init__.py
│   │   │   │   ├── utils/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── colormaps.py
│   │   │   │   └── __init__.py
│   │   │   ├── segmentation_m2f/
│   │   │   │   ├── core/
│   │   │   │   │   ├── anchor/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── builder.py
│   │   │   │   │   │   └── point_generator.py
│   │   │   │   │   ├── box/
│   │   │   │   │   │   ├── samplers/
│   │   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   │   ├── base_sampler.py
│   │   │   │   │   │   │   ├── mask_pseudo_sampler.py
│   │   │   │   │   │   │   ├── mask_sampling_result.py
│   │   │   │   │   │   │   └── sampling_result.py
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── builder.py
│   │   │   │   │   ├── utils/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── dist_utils.py
│   │   │   │   │   │   └── misc.py
│   │   │   │   │   └── __init__.py
│   │   │   │   ├── models/
│   │   │   │   │   ├── backbones/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── adapter_modules.py
│   │   │   │   │   │   ├── drop_path.py
│   │   │   │   │   │   ├── vit.py
│   │   │   │   │   │   └── vit_adapter.py
│   │   │   │   │   ├── decode_heads/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── mask2former_head.py
│   │   │   │   │   ├── losses/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── cross_entropy_loss.py
│   │   │   │   │   │   ├── dice_loss.py
│   │   │   │   │   │   └── match_costs.py
│   │   │   │   │   ├── plugins/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── msdeformattn_pixel_decoder.py
│   │   │   │   │   ├── segmentors/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── encoder_decoder_mask2former.py
│   │   │   │   │   ├── utils/
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── assigner.py
│   │   │   │   │   │   ├── point_sample.py
│   │   │   │   │   │   ├── positional_encoding.py
│   │   │   │   │   │   └── transformer.py
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── builder.py
│   │   │   │   ├── ops/
│   │   │   │   │   └── modules/
│   │   │   │   │       ├── __init__.py
│   │   │   │   │       └── ms_deform_attn.py
│   │   │   │   └── __init__.py
│   │   │   ├── __init__.py
│   │   │   ├── knn.py
│   │   │   ├── linear.py
│   │   │   ├── log_regression.py
│   │   │   ├── metrics.py
│   │   │   ├── setup.py
│   │   │   └── utils.py
│   │   ├── fsdp/
│   │   │   └── __init__.py
│   │   ├── hub/
│   │   │   ├── depth/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── decode_heads.py
│   │   │   │   ├── encoder_decoder.py
│   │   │   │   └── ops.py
│   │   │   ├── text/
│   │   │   │   ├── dinotxt_model.py
│   │   │   │   ├── dinov2_wrapper.py
│   │   │   │   ├── text_tower.py
│   │   │   │   ├── text_transformer.py
│   │   │   │   ├── tokenizer.py
│   │   │   │   └── vision_tower.py
│   │   │   ├── __init__.py
│   │   │   ├── backbones.py
│   │   │   ├── classifiers.py
│   │   │   ├── depthers.py
│   │   │   ├── dinotxt.py
│   │   │   └── utils.py
│   │   ├── layers/
│   │   │   ├── __init__.py
│   │   │   ├── attention.py
│   │   │   ├── block.py
│   │   │   ├── dino_head.py
│   │   │   ├── drop_path.py
│   │   │   ├── layer_scale.py
│   │   │   ├── mlp.py
│   │   │   ├── patch_embed.py
│   │   │   └── swiglu_ffn.py
│   │   ├── logging/
│   │   │   ├── __init__.py
│   │   │   └── helpers.py
│   │   ├── loss/
│   │   │   ├── __init__.py
│   │   │   ├── dino_clstoken_loss.py
│   │   │   ├── ibot_patch_loss.py
│   │   │   └── koleo_loss.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── gcn.py
│   │   │   └── vision_transformer.py
│   │   ├── run/
│   │   │   ├── eval/
│   │   │   │   ├── knn.py
│   │   │   │   ├── linear.py
│   │   │   │   ├── log_regression.py
│   │   │   │   └── test.py
│   │   │   ├── train/
│   │   │   │   └── train.py
│   │   │   ├── __init__.py
│   │   │   └── submit.py
│   │   ├── thirdparty/
│   │   │   └── CLIP/
│   │   │       ├── clip/
│   │   │       │   └── simple_tokenizer.py
│   │   │       └── LICENSE
│   │   ├── train/
│   │   │   ├── __init__.py
│   │   │   ├── checkpoint.ipynb
│   │   │   ├── gcn_meta_arch.py
│   │   │   ├── ssl_meta_arch.py
│   │   │   ├── test.py
│   │   │   └── train.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── cluster.py
│   │   │   ├── config.py
│   │   │   ├── dtype.py
│   │   │   ├── param_groups.py
│   │   │   └── utils.py
│   │   └── __init__.py
│   ├── docs/
│   │   ├── ChannelAdaptiveDINO.png
│   │   └── README_CHANNEL_ADAPTIVE_DINO.md
│   ├── notebooks/
│   │   ├── depth_estimation.ipynb
│   │   ├── dinotxt.ipynb
│   │   └── semantic_segmentation.ipynb
│   ├── scripts/
│   │   └── lint.sh
│   ├── ACCURACY_DIFFERENCE_ROOT_CAUSE.md
│   ├── checkpoint_merge_fsdp.py
│   ├── checkpoint_merge_fsdp.sh
│   └── … 其余 19 个文件（.md × 6、.py × 4、.sh × 3、.txt × 3、.yaml × 2、(无扩展名) × 1、.toml × 1、.cfg × 1、.ipynb × 1）
└── README.md
```
