# Stage2 多版本对比

这里集中管理 Stage2 消融实验。所有版本共享
`dinov2_stage2_2_FmH2ST/dinov2` 源码和同一份 Stage1 节点特征，差异只由
版本定义和最终保存到各自输出目录的 `config.yaml` 决定，不复制整套 DINOv2。

## 版本

| 名称 | 图目录 | edge_attr | GATv2 边注入方式 |
| --- | --- | --- | --- |
| `baseline` | `Graph/stage2_variants/dual` | 空间 + 语义，2 维 | 同时进入注意力和消息值 |
| `spatial_only` | `Graph/stage2_variants/spatial` | 仅空间，1 维 | 同时进入注意力和消息值 |
| `spatial_bias` | `Graph/stage2_variants/spatial` | 仅空间，1 维 | 只作为注意力偏置 |

`spatial_only` 和 `spatial_bias` 共享同一批纯空间图，因此只需要生成一次。
三种图的拓扑规则相同，都是空间 KNN 加最大距离过滤；区别只在是否保存语义边属性。
每个图目录保存 `stage2_graph_manifest.json`，每次训练保存
`variant_manifest.txt` 和解析后的 `config.yaml`，用于核对数据来源和源码哈希。

以下命令均从 `dinov2_stage2_2_FmH2ST` 根目录执行。运行环境必须同时提供
`torch_geometric` 和 `torch_scatter`；如果共享环境没有这些包，通过
`PYTHON_BIN=/path/to/python-with-pyg` 显式选择已有环境。

## 目录

```text
experiments/stage2_variants/
├── variants/
│   ├── baseline/variant.env
│   ├── spatial_only/variant.env
│   └── spatial_bias/variant.env
├── prepare_graphs.sh
├── run_variant.sh
├── run_parallel.sh
└── summarize_runs.py

Graph/stage2_variants/                 # 大文件，不进入 Git
├── dual/
└── spatial/

dinov2/results/stage2_variants/        # checkpoint 和日志，不进入 Git
├── baseline/<run_id>/
├── spatial_only/<run_id>/
└── spatial_bias/<run_id>/
```

## 1. 生成图

Stage1B 特征整理完成后执行：

```bash
bash experiments/stage2_variants/prepare_graphs.sh baseline
bash experiments/stage2_variants/prepare_graphs.sh spatial_only
```

默认读取 `../Graph/embeddings-1024`。也可以显式指定：

```bash
EMBEDDINGS_DIR=/path/to/organized_stage1 \
METADATA_DIR=/path/to/organized_stage1 \
bash experiments/stage2_variants/prepare_graphs.sh baseline
```

图默认要求包含 `patch_ids`、`levels` 和 `slide_id`，以保证以后能够与分割 Decoder
严格对齐。只有进行不面向 Decoder 的临时测试时，才设置
`REQUIRE_DECODER_METADATA=0`。

## 2. 单独运行

```bash
bash experiments/stage2_variants/run_variant.sh baseline 1 comparison_01
```

参数依次是版本、GPU 编号和 `run_id`。脚本默认拒绝在显存占用超过 1GB 的 GPU
上启动，也拒绝覆盖已有配置或 checkpoint。必要时可以显式设置
`ALLOW_BUSY_GPU=1`，但正式对比实验不建议这样做。

## 3. 两张 GPU 并行运行

```bash
bash experiments/stage2_variants/run_parallel.sh comparison_01 baseline spatial_only
```

默认映射为第一个版本使用 GPU 0，第二个版本使用 GPU 1。第三个版本可在其中一个
完成后运行，或者下一轮比较 `baseline` 与 `spatial_bias`。

后台运行示例：

```bash
nohup bash experiments/stage2_variants/run_parallel.sh \
  comparison_01 baseline spatial_only \
  > dinov2/results/stage2_variants/comparison_01.launcher.log 2>&1 &
```

## 4. 汇总 Stage2 代理指标

```bash
python3 experiments/stage2_variants/summarize_runs.py \
  dinov2/results/stage2_variants/baseline/comparison_01 \
  dinov2/results/stage2_variants/spatial_only/comparison_01
```

该脚本只汇总 Stage2 预训练损失。最终判断仍应使用相同 Decoder 设置下的 Dice、
mIoU、Boundary F1 和 HD95。

## 5. 导出各版本完整 WSI 上下文

导出时必须使用该次运行目录中保存的解析后配置，并使用与该版本一致的图目录。例如：

```bash
PYTHON_BIN=/path/to/python-with-pyg
RUN_DIR=dinov2/results/stage2_variants/spatial_bias/comparison_01
"$PYTHON_BIN" export_context.py \
  --config-file "$RUN_DIR/config.yaml" \
  --checkpoint "$RUN_DIR/model_final.rank_0.pth" \
  --graph-dir ../Graph/stage2_variants/spatial \
  --output-dir ../Graph/context-stage2-variants/spatial_bias/comparison_01
```

实际 checkpoint 名称以运行目录中的 `last_checkpoint.rank_0` 为准。不同版本的
context 也必须放在不同目录，后续才能使用同一 Decoder 设置公平比较。

## 公平比较约束

- 所有版本使用同一个 Stage1 checkpoint 和同一批节点特征。
- 节点顺序、坐标、KNN 参数、距离阈值、训练 seed 和训练轮数保持一致。
- 不共用输出目录，不从其他结构不兼容的 Stage2 checkpoint 续训。
- 每个输出目录内的解析后 `config.yaml` 是该次运行的最终配置依据。
- 启动前必须归档 baseline 的源码状态、图 schema 和日志。
