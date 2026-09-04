# 分割训练过拟合迭代记录

## Iteration 1：肿瘤类加权 Cross Entropy

### 触发证据

`distance_only/v1` 在 decoder-only 阶段完成 epoch 0–2 后出现连续分叉：

| epoch | train loss | train tumor Dice | val loss | val tumor Dice |
|---:|---:|---:|---:|---:|
| 0 | 0.4066 | 0.8880 | 0.6365 | 0.7357 |
| 1 | 0.2450 | 0.9369 | 0.7586 | 0.7399 |
| 2 | 0.2148 | 0.9448 | 0.8610 | 0.7387 |

训练 loss 连续下降、验证 loss 连续上升 35.3%，验证 Dice 在约 0.74 停滞，最终训练/验证
Dice 差距为 0.2061。epoch 2 的验证混淆矩阵对应肿瘤 Precision 约 0.824、Recall 约
0.669，预测肿瘤像素比例约 23.6%，低于真实的 29.1%。训练进程已终止，
`checkpoint_last.pt` 与最佳 epoch 1 的 `checkpoint_best.pt` 均保留。

### 数据诊断

- train/valid 按 WSI 隔离，分别包含 90/44 张切片，无 slide overlap。
- train/valid 的肿瘤 patch 比例分别为 58.9%/38.6%。
- train/valid 的平均肿瘤像素比例分别为 48.0%/29.1%。
- 训练集中每张 WSI 含 107–14,163 个 patch；普通 patch 级 shuffle 会让大切片贡献更多
  梯度。
- 当前在线增强仅包含水平和垂直翻转，尚未覆盖病理染色变化。

### 本轮单变量修改

新增 `--tumor-class-weight`，首次正式迭代固定为 `1.5`：

```text
Cross Entropy class weights = [background: 1.0, tumor: 1.5]
soft Dice                  = unchanged
```

网络、数据清单、图、学习率、AdamW、warm-up/cosine 调度和渐进解冻设置保持原对照不变。
默认权重仍为 `1.0`，因此其他正在运行或排队的原始实验不会改变语义。旧 checkpoint 缺少
该新字段时按 `1.0` 兼容恢复，禁止把 `1.5` 从原始 checkpoint 中途接入。

### 验证与监控

- 16 项分割单元测试全部通过。
- 三阶段 smoke test 覆盖 decoder-only、Stage1/Stage2 adapter 联合和 Stage1 顶层解冻；
  梯度审计确认 Stage1 backbone、Stage1 fusion、Stage2 GATv2、Decoder 均收到非零梯度。
- 监控器新增三轮 loss 分叉检测：训练 loss 连降、验证 loss 连升至少 20%、训练/验证
  Dice 差距至少 0.18，且验证 Dice 未继续创新高时停止。

### 论文依据与后续候选

- Focal Tversky 分析了医学分割中高 Precision、低 Recall 时提高假阴性代价的必要性；
  本轮先采用更温和、可单独归因的肿瘤类加权 CE：<https://arxiv.org/abs/1810.07842>。
- 若本轮仍出现跨 WSI 泛化分叉，下一项单变量候选为 RandStainNA 染色增强，其在组织分类
  与细胞核分割中用于提升染色变化下的泛化：<https://arxiv.org/abs/2206.12694>。
- 再下一项候选为按 WSI 平衡采样，减小单张大切片对梯度的支配；不得与染色增强在同一
  首次试验中同时启用，以保留消融可解释性。

## Iteration 2：提高 Decoder stochastic depth

### 触发证据

采用 WSI 平衡采样的 S、ST、STA 三个版本均在 epoch 1 达到最佳验证 Dice（分别为
0.7861、0.7865、0.7873），之后训练 Dice 持续上升至约 0.959，而 epoch 5 验证 Dice
下降至 0.758–0.761、验证 loss 上升至 0.897–0.999。退化在 epoch 2 的
`decoder_only` 阶段已经开始，因此不能归因于 Stage1/Stage2 解冻。

### 本轮单变量修改

将 Decoder 内残差块的最大 stochastic-depth 概率从 `0.1` 提高到 `0.2`。采样策略、
S/ST/STA 损失定义、颜色增强、AdamW、各参数组学习率、warm-up/cosine 调度和解冻时刻
全部保持不变。新实验使用 `v1_dp020` 输出目录，不覆盖已终止实验及其最佳权重。

Stochastic Depth 在训练时随机绕过残差分支、推理时使用完整网络，用于正则化深层残差
网络：<https://arxiv.org/abs/1603.09382>。现有 AdamW 与 cosine 调度分别继续依据
<https://arxiv.org/abs/1711.05101> 和 <https://arxiv.org/abs/1608.03983>。
