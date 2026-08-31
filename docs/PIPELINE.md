# 当前训练与分割流程

## 0. Patch 与坐标

切图阶段必须生成一行一个 patch 的元数据。当前代码消费绝对左上角坐标 `(x, y)`，
默认 `level=0`。现有 CSV 中的 `row/col` 是网格索引，而 `x/y` 是像素坐标；建图和
最终 WSI 拼接都使用后者。

WSI 切图程序本身不在当前仓库中。仓库中保留的是已切 patch、CSV 坐标和图文件，
因此不要把 `stage1_Extract2s2_local` 产生的 patch 内 tile 坐标当成 WSI 坐标。

## 1. Stage-1A：训练融合特征

目录：`dinov2_stage1_Extract2s2/`

- DINO backbone 冻结。
- 两个 global 训练视图经过 global spatial CNN；确定性增强只用于 Stage-1B 导出。
- local crops 经过独立 local spatial CNN 和 MLP 融合。
- CLS、global spatial、local spatial 三者进入 node fusion。
- 输出维度保持 1024，作为 Stage 2 图节点。

入口：`pretrain.sh`，实际转到 `pretrain_stage1a.sh`。必须通过 `DINO_WEIGHTS` 指定
DINO 初始化权重。

## 2. Stage-1B：确定性导出

入口：`pretrain_imgnet22k.sh`。必须通过 `STAGE1_WEIGHTS` 指定训练完成且已合并的
Stage-1A 权重。

输出目录中的两套格式用途不同：

- `features_pretrained_s1_part*.npz` 和 `filenames_*.npz`：兼容已有 Stage 2 建图数据。
- `decoder_features_s1_part*.pt`：新格式，包含 `format_version`、`filenames`、
  `patch_ids`、`node_features` 和未池化 `dense_tokens`；有 slide 元数据时还会包含
  `slide_ids/coords/levels`。

多 GPU 导出会自动在文件名加入 rank，避免不同进程覆盖同一个 chunk。

随后运行 `organize_features.py`，用 patch CSV 将 shard 还原成每张 WSI 的
`*_features.npy`、`*_coords.npy`、`*_metadata.pt` 和独立的
`*_dense_tokens.npy`（memory-map 友好）。用于分割时加
`--require-dense-tokens`，从这一刻起节点特征、dense token、坐标和 patch ID 共享
同一个有序列表。

## 3. Stage 2：建图和图预训练

新建图入口是 `dinov2_stage2_2_FmH2ST/build_graphs.py`。它读取同名的
`*_features.npy`、`*_coords.npy` 和可选 `*_metadata.pt`，使用空间 KNN 候选和
物理距离上限建立边：

- 边通道 0：空间高斯权重。
- 边通道 1：节点特征余弦相似度映射到 `[0, 1]`。

`edge_index` 的存在与否只由坐标 KNN 和物理距离上限决定；余弦相似度仅是已存在边的
属性，不会产生语义远程边。主线图编码器使用带双通道边属性的 GATv2：先联合目标节点、
来源节点和边嵌入，再计算动态注意力，并按目标节点的入边归一化。外层仍保留残差、
LayerNorm 和 JK-Sum，节点输入输出维度均为 1024。

`graph_build.ipynb` 是历史实验记录，不再作为生产入口。Stage 2 遮掩和噪声视图只在
`GCNMetaArch` 中生成，数据集不再执行第二套 mask。

仓库现有 `Graph-1024-251111-thre0.8` 是单通道旧图。当前配置 `gcn.edge_dim=2`，
必须先从 Stage1 整理后的 per-slide arrays 运行 `build_graphs.py` 生成
`Graph/graphs-current`；
加载器会拒绝把旧图静默当成新图。

图训练入口是 `pretrain.sh`，通过 `GRAPH_ROOT` 指定图目录，并通过 `PYTHON_BIN`
选择安装了 PyG 的环境。

## 4. 分割缓存

分割不在每个 patch batch 内重新运行整张图。正确步骤是：

1. 对完整 WSI 图调用 Stage 2 的 `model.extract_context(graph)`。
2. 保存 `global_context` 和完全相同顺序的 `patch_ids`。
3. 用 `python -m dinov2_segmentation.prepare_slide_store` 合并轻量 graph、Stage1
   metadata/dense tokens 与 context。
4. 用 manifest 将原图、mask 和 feature store 对齐。

禁止使用 Stage 2 训练数据集的随机 5000 节点子图导出 context；这会破坏完整 WSI
语义及节点一一对应关系。

维护入口 `export_context.py` 会直接读取磁盘上的完整图，不经过训练数据集，并把
`patch_ids` 与 context 一起保存。它要求与当前架构兼容的合并后 Stage 2 checkpoint；
旧单通道图任务的 checkpoint 不兼容。

## 5. 分割网络

- 语义分支：DINO dense token + 完整 WSI 的 GNN context，输出 H/4 特征。
- 细节分支：对应原图 patch，ConvNeXtV2 与 HRFormer-style block 交替，保持 H/4。
- 融合 decoder：拼接两分支并交替执行 attention/conv，最终恢复原 patch 分辨率。
- WSI 推理：依据 `(x, y, level)` 放回，重叠区加权融合。

训练和推理说明见 `dinov2_segmentation/README.md`。
