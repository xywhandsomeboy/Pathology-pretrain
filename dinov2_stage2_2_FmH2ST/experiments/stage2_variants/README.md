# Stage 2 三版本对比

这里仅保留本轮确定的三个 Stage 2 版本。它们共享同一份 Stage 1 节点特征、
GATv2 主体和空间建图参数，但分别使用独立的运行目录与完整解析配置。
其中 `baseline` 保持 Git 提交 `d17b6a9` 的数值路径和全部代理任务不变。

## 三个版本的准确含义

| 编号 | 名称 | 预训练图与 GATv2 | 代理任务 | 预训练后导出给 Decoder 的图 |
| --- | --- | --- | --- | --- |
| 1 | `baseline` | 原版二维边权：空间高斯权重 + 特征余弦语义权重；同时进入注意力与消息 | 节点重建、图对比、边存在、边权回归全部保留 | 原版二维带权图 |
| 2 | `distance_only` | 边只由坐标距离 KNN 和最大距离阈值决定；图中不保存 `edge_attr`，GATv2 不读取边权 | 保留节点重建、图对比和边存在；删除边权回归，关闭边权噪声 | 同一纯距离、无边权图 |
| 3 | `weighted_pretrain_distance_context` | 与 `baseline` 完全相同 | 与 `baseline` 完全相同 | 改用纯距离、无边权图；导出时显式绕过 GATv2 的边权编码器 |

这里的“仅通过距离连边”不是全连接图：先按 `(x,y)` 做空间 KNN，再用最大空间
距离过滤。节点的 DINO 特征不参与边的建立，也不会生成空间或语义边权。

第 3 版有意制造训练/导出的输入差异，用于单独判断“预训练时学习边权、下游时
去掉边权”是否有效。它不会删除 checkpoint 中学到的边编码器参数，只是在
`extract_context()` 时不调用这些参数。

这里删除的是预先写入图文件的静态空间/语义边权。版本 2 和版本 3 的下游阶段仍
保留 GATv2 根据两端节点特征动态算出的注意力系数，否则模型就不再是 GATv2。

## 目录

```text
experiments/stage2_variants/variants/
├── baseline/variant.env
├── distance_only/variant.env
└── weighted_pretrain_distance_context/variant.env

Graph/stage2_variants/          # 生成物，不进入 Git
├── dual/                       # 版本 1/3 的预训练图、版本 1 的下游图
└── distance/                   # 版本 2 的全部图、版本 3 的下游图

dinov2/results/stage2_variants/ # checkpoint 与日志，不进入 Git
├── baseline/<run_id>/
├── distance_only/<run_id>/
└── weighted_pretrain_distance_context/<run_id>/
```

`dual` 与 `distance` 使用完全相同的空间拓扑参数；区别是 `dual` 保存二维边权，
`distance` 完全不保存 `edge_attr`。每个图目录都有
`stage2_graph_manifest.json`，防止不同建图设置被误用。

## 1. 准备图

以下命令均从 `dinov2_stage2_2_FmH2ST` 目录执行：

```bash
PYTHON_BIN=/path/to/python-with-pyg
EMBEDDINGS_DIR=../Graph/embeddings-current
METADATA_DIR=../Graph/embeddings-current

PYTHON_BIN="$PYTHON_BIN" EMBEDDINGS_DIR="$EMBEDDINGS_DIR" \
METADATA_DIR="$METADATA_DIR" \
  bash experiments/stage2_variants/prepare_graphs.sh baseline

PYTHON_BIN="$PYTHON_BIN" EMBEDDINGS_DIR="$EMBEDDINGS_DIR" \
METADATA_DIR="$METADATA_DIR" \
  bash experiments/stage2_variants/prepare_graphs.sh distance_only
```

版本 3 复用上述 `dual` 预训练图和 `distance` 下游图，不需要第三次生成。
也可以在命令末尾加 `pretrain` 或 `context`，只准备对应用途的图；默认 `all`。

## 2. 单独训练

```bash
bash experiments/stage2_variants/run_variant.sh baseline 0 compare_01
bash experiments/stage2_variants/run_variant.sh distance_only 1 compare_01
bash experiments/stage2_variants/run_variant.sh \
  weighted_pretrain_distance_context 0 compare_01
```

参数依次是版本名、物理 GPU 编号和 `run_id`。脚本默认拒绝覆盖已有结果，也会拒绝
在显存占用超过 1GB 的 GPU 上启动。每次运行都会保存解析后的 `config.yaml` 和
`variant_manifest.txt`。

## 3. 两张 GPU 并行训练

```bash
bash experiments/stage2_variants/run_parallel.sh \
  compare_01 baseline distance_only
```

第三个版本可在任意一个 GPU 空闲后启动。为保证对比公平，三个版本应使用相同的
Stage 1 特征、seed、KNN 参数、距离阈值、训练轮数和 Decoder 设置。

## 4. 导出完整 WSI 上下文

必须使用该次运行目录保存的 `config.yaml`，并选择对应的下游图：

```bash
# 版本 1：带权图
python export_context.py \
  --config-file dinov2/results/stage2_variants/baseline/compare_01/config.yaml \
  --checkpoint /path/to/baseline_checkpoint.pth \
  --graph-dir ../Graph/stage2_variants/dual \
  --output-dir ../Graph/context-stage2-variants/baseline/compare_01

# 版本 2/3：纯距离无边权图
python export_context.py \
  --config-file dinov2/results/stage2_variants/weighted_pretrain_distance_context/compare_01/config.yaml \
  --checkpoint /path/to/version3_checkpoint.pth \
  --graph-dir ../Graph/stage2_variants/distance \
  --output-dir ../Graph/context-stage2-variants/weighted_pretrain_distance_context/compare_01
```

程序会核对图中的 `edge_mode` 与配置，避免把 `dual` 和 `distance` 图混用。

## 5. 汇总预训练指标

```bash
python experiments/stage2_variants/summarize_runs.py \
  dinov2/results/stage2_variants/baseline/compare_01 \
  dinov2/results/stage2_variants/distance_only/compare_01 \
  dinov2/results/stage2_variants/weighted_pretrain_distance_context/compare_01
```

预训练损失只能说明代理任务优化情况；最终仍应比较相同 Decoder 下的 Dice、mIoU、
Boundary F1 和 HD95。版本 2 没有边权回归头，因此其 `edge_weight` 指标固定为零，
不能与版本 1/3 的该项损失横向比较。
