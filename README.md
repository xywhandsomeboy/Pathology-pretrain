# CerviPath

CerviPath 是宫颈病理图像的 DINOv2–GNN 预训练与分割项目。仓库保留了早期实验，
但当前开发只围绕一条主线进行，避免不同 fork 中同名 `dinov2` 包互相污染。

## 当前主线

| 顺序 | 目录 | 职责 | 维护入口 |
|---|---|---|---|
| 0 | `dinov2/` | 病理图 DINOv2 基座与权重 | 仅提供 backbone |
| 1 | `dinov2_stage1_Extract2s2/` | 冻结 DINO；训练 global/local 空间融合；导出节点和 dense token | `pretrain.sh`、`pretrain_imgnet22k.sh` |
| 2 | `dinov2_stage2_2_FmH2ST/` | 建立 WSI patch 图并进行图自监督预训练 | `build_graphs.py`、`pretrain.sh`、`export_context.py` |
| 3 | `dinov2_segmentation/` | 全局语义分支与高分辨率细节分支融合分割 | `train.py`、`infer.py` |

主线数据流：

```text
原图 patch + (slide_id, patch_id, x, y, level)
  -> Stage-1A 训练空间融合模块
  -> Stage-1B 导出 node_features + dense_tokens
  -> Stage 2 按同一 patch 顺序建图并导出 global_context
  -> 对齐为单 slide feature store
  -> 双分支分割与 WSI 坐标拼接
```

详细流程见 [docs/PIPELINE.md](docs/PIPELINE.md)，字段约束见
[docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md)。

## 历史项目

以下目录保留用于代码对照和复现实验，不属于当前分割主线：

- `legacy/dinov2_finetune/`：早期 DINO + 动态 GNN 联合实验。
- `legacy/dinov2_stage1_Extract2s2_local/`：旧 local tile 提取实验；其坐标是 patch 内相对坐标。
- `legacy/dinov2_stage2_2_FmH2ST_finetune/`：旧分类微调项目。

不要从主线代码导入这些目录。历史边界和兼容性说明见
[docs/LEGACY.md](docs/LEGACY.md)。

## 运行原则

每个 fork 都包含名为 `dinov2` 的 Python 包，因此必须从对应项目目录启动：

```bash
# Stage-1A：训练空间融合模块
cd dinov2_stage1_Extract2s2
DINO_WEIGHTS=/absolute/path/to/dino_checkpoint.pth bash pretrain.sh

# Stage-1B：导出图节点和分割 dense tokens
STAGE1_WEIGHTS=/absolute/path/to/merged_stage1_checkpoint.pth \
  bash pretrain_imgnet22k.sh

# 按 patch CSV 把 Stage-1 shard 整理为每张 WSI 一组文件
../dinov2/.venv/bin/python organize_features.py \
  --embeddings-dir dinov2/results/stage1b_features/embeddings \
  --patch-csv ../Data/meta/patch/patch_grid_positions-TCGA-disease2-test-lesstest.csv \
  --output-dir ../Graph/embeddings-current \
  --require-dense-tokens \
  --skip-unmatched

# Stage 2：从已有成对 npy 建图
cd ../dinov2_stage2_2_FmH2ST
PYTHON_BIN=/path/to/python-with-pyg
"$PYTHON_BIN" build_graphs.py \
  --embeddings-dir ../Graph/embeddings-current \
  --metadata-dir ../Graph/embeddings-current \
  --output-dir ../Graph/graphs-current \
  --require-decoder-metadata

# Stage 2 图预训练
GRAPH_ROOT=../Graph/graphs-current \
  PYTHON_BIN="$PYTHON_BIN" bash pretrain.sh

# Stage 2 多版本对比（baseline / spatial_only / spatial_bias）
# 详细说明见 experiments/stage2_variants/README.md
EMBEDDINGS_DIR=../Graph/embeddings-current \
  METADATA_DIR=../Graph/embeddings-current \
  PYTHON_BIN="$PYTHON_BIN" \
  bash experiments/stage2_variants/prepare_graphs.sh baseline
EMBEDDINGS_DIR=../Graph/embeddings-current \
  METADATA_DIR=../Graph/embeddings-current \
  PYTHON_BIN="$PYTHON_BIN" \
  bash experiments/stage2_variants/prepare_graphs.sh spatial_only

# 两张空闲 GPU 上同时启动两个独立 Stage 2 版本
PYTHON_BIN="$PYTHON_BIN" \
  bash experiments/stage2_variants/run_parallel.sh \
  comparison_01 baseline spatial_only

# 用合并后的 Stage-2 权重对完整 WSI 图导出节点 context
"$PYTHON_BIN" export_context.py \
  --config-file dinov2/configs/train/vitl16_short.yaml \
  --checkpoint /absolute/path/to/merged_stage2_checkpoint.pth \
  --graph-dir ../Graph/graphs-current \
  --output-dir ../Graph/context-current

# 合并某张 WSI 的轻量图、Stage1 dense tokens 与 Stage2 context
cd ..
dinov2/.venv/bin/python -m dinov2_segmentation.prepare_slide_store \
  --graph Graph/graphs-current/SLIDE.pt \
  --context Graph/context-current/SLIDE.pt \
  --stage1-metadata Graph/embeddings-current/SLIDE_metadata.pt \
  --dense-tokens Graph/embeddings-current/SLIDE_dense_tokens.npy \
  --output Graph/segmentation-features/SLIDE.pt

# 分割训练必须回到仓库根目录
dinov2/.venv/bin/python -m dinov2_segmentation.train --help
```

所有机器相关路径均通过环境变量或命令行传入，不再写入主线配置。Stage 2 的 Python
环境必须安装 `torch_geometric`；大图建图还需要 `scikit-learn` 的 BallTree。

## 数据与生成物

- `Data/`：原始/切分图像及 patch 元数据，不纳入 Git。
- `Graph/`：Stage 1 嵌入和 Stage 2 图，不纳入 Git。
- `UNI/`：外部权重，不纳入 Git。
- `**/results/`：训练日志和 checkpoint，不纳入 Git。

这些目录没有在整理过程中移动或删除。运行
`python3 scripts/check_project.py` 可进行不导入深度学习依赖的结构检查。
