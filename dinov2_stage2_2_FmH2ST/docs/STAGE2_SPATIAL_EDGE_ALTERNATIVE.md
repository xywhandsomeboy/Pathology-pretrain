# Stage2 纯空间边备选方案（暂缓实施）

状态：**设计备选，当前不实施**
记录日期：2026-08-31

## 决策

当前模型结构和正在进行的预训练保持不变。先保留现有模型作为 baseline，
等当前训练完成并保存 checkpoint、配置与指标后，再通过独立实验分支评估本方案。

本文件只记录后续候选改动，不代表这些改动已经进入当前模型。

## 当前 baseline

1. 图拓扑在 Stage2 开始前建立。
2. 每个 patch 是一个节点，节点特征来自 Stage1。
3. 候选边由二维空间坐标的 KNN 建立，默认 `k=8`。
4. 候选边还会经过最大距离阈值过滤。未显式指定阈值时，使用：

   ```text
   max_distance = 3 * median(nearest-neighbour distance)
   ```

5. 当前 `edge_attr` 有两个通道：

   ```text
   [spatial_weight, semantic_weight]
   ```

   - `spatial_weight`：根据 patch 间空间距离计算的高斯权重。
   - `semantic_weight`：根据两个 Stage1 节点特征的余弦相似度计算。

6. 当前 GATv2 会先编码 `edge_attr`，然后让边嵌入同时影响注意力分数和消息内容。
7. Stage2 正式导出上下文时使用完整原图，不使用随机遮挡、噪声或临时删边。

## 已经生效、无需等待的对比视图设置

以下内容属于训练增强，不改变磁盘上的原始图，因此已经保留在当前 baseline 中：

- 对比视图临时删边率为 `5%`。
- 双向边作为同一个无向边对同步删除。
- 自环保留。
- 原图中邻居数不少于 2 的节点，在对比视图中至少保留 2 个邻居。
- 以上规则只影响 Stage2 训练时的 noisy view，不影响完整图推理。

## 备选方案 A：边完全由空间关系描述

目标是将图结构与视觉语义解耦：空间关系由图提供，语义关系由 GATv2 从节点特征中学习。

### 拓扑

边是否存在只允许使用坐标和物理距离：

```text
(i, j) is an edge
if j is a spatial KNN neighbour of i
and distance(i, j) <= max_distance
```

禁止使用以下信息决定是否连边：

- Stage1 特征相似度；
- GATv2 上下文相似度；
- 类别或分割标签；
- Decoder 输出。

Stage2 训练和训练结束后的完整图推理必须使用同一套建边规则。

### 边属性

删除静态 `semantic_weight`，只保留一个空间通道：

```text
edge_attr = [spatial_weight]
edge_dim = 1
```

空间权重候选定义：

```text
spatial_weight = exp(-(distance ** 2) / (2 * sigma ** 2))
```

这样可以避免 Stage1 特征既用于生成 `semantic_weight`，又被 GATv2 再次用于动态注意力所造成的信息重复。

## 备选方案 B：空间权重只作为注意力偏置（优先评估）

当前实现让边嵌入同时进入注意力和消息值。候选修改是让空间权重只影响“邻居有多重要”，
不再直接改变“邻居传递什么内容”。

候选公式：

```text
score_ij = GATv2Score(h_i, h_j) + beta * log(spatial_weight_ij + eps)
alpha_ij = softmax_j(score_ij)
message_ij = alpha_ij * W * h_j
```

其中 `beta` 使用可学习标量。需要通过实验比较两种初始化：

- `beta = 0`：先从纯节点注意力开始，逐渐学习空间先验强度；
- `beta = 1`：从完整空间先验开始。

优先尝试 `beta = 0`，降低静态距离先验在训练初期压制动态注意力的风险。

## 坐标与图结构的后续检查项

1. 如果 WSI patch 来自不同倍率或金字塔层级，先把坐标换算到同一物理单位（推荐微米）。
2. 不允许直接用不同 level 的像素坐标计算距离。
3. 如果数据全部来自相同倍率、patch 大小和步长，可以继续使用原图像素坐标。
4. 保留 `KNN + 最大距离阈值`，避免 KNN 跨越组织空洞连接远处区域。
5. 评估是否在写入图文件前将 KNN 边显式对称化。
6. 自环仍由 GATv2 前向传播时添加，不写入永久图也可以。

## 实施影响

实施纯空间边方案时至少需要同步修改：

- `dinov2/data/datasets/graph_builder.py`
  - 停止计算并保存 `semantic_weight`；
  - 将图格式版本升级；
  - 评估是否显式对称化空间 KNN 边。
- `dinov2/configs/ssl_default_config.yaml`
  - `edge_dim: 2` 改为 `edge_dim: 1`；
  - 加入空间偏置方式和 `beta` 初始化配置。
- `dinov2/models/gcn.py`
  - 不再将空间边嵌入加到消息值；
  - 把空间权重作为 attention bias；
  - 加入可学习 `beta`。
- `dinov2/train/gcn_meta_arch.py`
  - 将边属性校验从二维更新为一维；
  - 决定是否保留边权重重建代理任务。
- `build_graphs.py` 和数据集加载检查
  - 重新生成并验证新 schema 的图文件。

## checkpoint 与数据兼容性

- Stage1 特征可以复用，不需要因为这项修改重新训练 Stage1。
- 当前两通道 `.pt` 图不能直接当作新的一通道图继续使用，应重新建图。
- 当前 Stage2 checkpoint 的 `edge_encoder` 输入维度为 2，与新模型不兼容。
- 新方案需要从头训练 Stage2，不能把旧 Stage2 checkpoint 当作严格续训点。
- 不覆盖当前图和结果，建议使用独立目录：

  ```text
  Graph/graphs-spatial-v2/
  dinov2/results/stage2_graph_pretrain_spatial_v2/
  ```

## 实验比较

至少保留以下三个实验：

1. `Baseline`：当前双通道边属性和当前 GATv2。
2. `Spatial-Only`：一维空间边属性，但仍使用现有 edge encoder 注入方式。
3. `Spatial-Bias`：一维空间边属性，只作为注意力偏置，不进入消息值。

除上述差异外，三组实验应固定：

- Stage1 特征；
- 图的节点顺序与 patch 集合；
- 训练轮数、随机种子和优化器；
- 节点遮挡、边任务和对比增强参数；
- 下游分割 Decoder 与训练集划分。

评价指标不只观察 Stage2 代理损失，还应至少比较：

- Dice；
- mIoU；
- Boundary F1；
- HD95；
- 不同 WSI 上的稳定性和方差。

## 启动修改前的条件

只有满足以下条件后才开始实现本备选方案：

1. 当前预训练正常结束；
2. 当前 checkpoint、完整配置和训练日志已经归档；
3. baseline 图和结果目录保持只读、不被覆盖；
4. 为新图 schema 使用独立输出目录；
5. 明确记录从二维 `edge_attr` 到一维 `edge_attr` 的不兼容变更。
