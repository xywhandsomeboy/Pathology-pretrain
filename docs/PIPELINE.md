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

长周期实验会另外保留 `budget_checkpoints/model_0009999.rank_0.pth`，用于和旧版在
完全相同的 10,000-step 预算下做诊断；200-epoch 周期端点保存在
`cycle_checkpoints/`。最终选择仍由 `scripts/compare_stage1_checkpoints.py` 在所有完整
候选的相同 trailing window 上统一完成。工具若发现日志由 no-resume 重启追加而成，
只选择与最终 checkpoint 对应的最后一个严格单调片段，并在 JSON 中记录被丢弃的前缀
数量，禁止把两次运行的指标混合统计。

## 2. Stage-1B：确定性导出

入口：`pretrain_imgnet22k.sh`。必须通过 `STAGE1_WEIGHTS` 指定训练完成且已合并的
Stage-1A 权重。

输出目录中的两套格式用途不同：

- `features_pretrained_s1_part*.npz` 和 `filenames_*.npz`：兼容已有 Stage 2 建图数据。
- `decoder_features_s1_part*.pt`：新格式，包含 `format_version`、`filenames`、
  `patch_ids`、`node_features` 和未池化 `dense_tokens`；有 slide 元数据时还会包含
  `slide_ids/coords/levels`。

多 GPU 导出会自动在文件名加入 rank，避免不同进程覆盖同一个 chunk。

完整训练集原始 CSV 不在当前迁移后的工作区，但身份和坐标没有丢失：
`Data/Pretrain_extra/entries.npy` 保留全部 2,299,631 个 patch 文件名，
`Graph/embeddings-1024` 保留 2188 个 slide split 的坐标和节点顺序。先运行
`scripts/reconstruct_patch_metadata.py`，按已验证的 `patch_idx` 与 `(y,x)` 同序契约
重建完整 manifest。参考目录的旧特征只用于核对节点数量，绝不会写入重建结果；
Stage 2 的节点特征仍全部来自获选 Stage-1 checkpoint。

随后运行 `organize_features.py`，用这个完整 patch manifest 将 shard 还原成每张 WSI 的
`*_features.npy`、`*_coords.npy`、`*_metadata.pt` 和独立的
`*_dense_tokens.npy`（memory-map 友好）。用于分割时加
`--require-dense-tokens`，从这一刻起节点特征、dense token、坐标和 patch ID 共享
同一个有序列表。

## 3. Stage 2：建图和图预训练

新建图入口是 `dinov2_stage2_2_FmH2ST/build_graphs.py`。它读取同名的
`*_features.npy`、`*_coords.npy` 和可选 `*_metadata.pt`，使用空间 KNN 候选和
物理距离上限建立边。`edge_index` 的存在与否始终只由坐标决定，支持两种 schema：

- `dual`：边通道 0 是空间高斯权重，边通道 1 是节点特征余弦相似度映射到
  `[0, 1]`；
- `distance`：使用相同空间拓扑，但完全不保存 `edge_attr`。

余弦相似度只可能成为 `dual` 已存在边的属性，不会产生语义远程边。原版图编码器
使用带双通道边属性的 GATv2；纯距离版本使用同一个 GATv2 主体但不输入静态边权。
两者都根据节点特征计算动态注意力并按目标节点的入边归一化。外层仍保留残差、
LayerNorm 和 JK-Sum，节点输入输出维度均为 1024。三个正式版本的准确边界见
`dinov2_stage2_2_FmH2ST/experiments/stage2_variants/README.md`。

`graph_build.ipynb` 是历史实验记录，不再作为生产入口。Stage 2 遮掩和噪声视图只在
`GCNMetaArch` 中生成，数据集不再执行第二套 mask。

仓库现有 `Graph-1024-251111-thre0.8` 是单通道旧图，既不满足原版 `dual` 的二维
schema，也不满足 `distance` 的无属性 schema。必须先从 Stage1 整理后的 per-slide
arrays 运行 `build_graphs.py` 重新生成；加载器会拒绝把旧图静默当成新图。

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
`patch_ids` 与 context 一起保存。导出必须使用该次训练保存的解析后配置；程序会核对
配置要求的 `context_edge_mode` 与完整图，避免 `dual`/`distance` 混用。旧单通道图
任务的 checkpoint 不兼容。

## 5. 分割网络

- 语义分支：DINO dense token + 完整 WSI 的 GNN context，输出 H/4 特征。
- 细节分支：对应原图 patch，ConvNeXtV2 与 HRFormer-style block 交替，保持 H/4。
- 融合 decoder：拼接两分支并交替执行 attention/conv，最终恢复原 patch 分辨率。
- WSI 推理：依据 `(x, y, level)` 放回，重叠区加权融合。

训练和推理说明见 `dinov2_segmentation/README.md`。
