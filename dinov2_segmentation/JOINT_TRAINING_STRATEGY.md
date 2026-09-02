# Stage1–Stage2–Decoder 联合微调策略

最终六个分割模型均进行端到端监督微调，Stage1 与 Stage2 不冻结。离线 Stage1B 特征只用于
建立固定图拓扑和初始化邻居特征记忆；当前目标 patch 的 dense tokens、节点特征和 GATv2
context 都在训练进程内重新计算，分割损失可回传至三个阶段。

## 文献依据

- [DINOv2](https://arxiv.org/abs/2304.07193)：采用自监督 ViT 的 patch-level dense features
  作为像素级下游任务的基础表征。
- [BEiT](https://arxiv.org/abs/2106.08254)：预训练 ViT 下游微调采用 layer-wise learning-rate
  decay，使靠近输入的通用表征更新更小。
- [ViT-Adapter](https://arxiv.org/abs/2205.08534)：预训练 plain ViT 与卷积/任务适配模块联合
  用于语义分割等 dense prediction，支持让新任务模块比预训练主干更快适配。
- [AdamW](https://arxiv.org/abs/1711.05101)：使用与梯度更新解耦的 weight decay。
- [SGDR](https://arxiv.org/abs/1608.03983)：采用 cosine learning-rate decay。本项目最终微调
  使用一次 warm-up 后的单周期 cosine，不做周期重启。
- [GATv2](https://arxiv.org/abs/2105.14491)：Stage2 使用 query-dependent dynamic attention，
  并在分割损失下继续更新。

## 默认参数组

| 参数组 | 峰值学习率 | 说明 |
|---|---:|---|
| Decoder V1/V2 | `2e-4` | 新初始化的任务层，适配最快 |
| Stage2 GATv2 | `5e-5` | 保留图预训练知识并允许监督修正 |
| Stage1 spatial/local/fusion | `5e-5` | 预训练聚合层，中等更新幅度 |
| Stage1 ViT 顶层 | `2e-5` | 预训练视觉主干，使用最小基础学习率 |

ViT-L 的 24 个 Transformer block 从输出到输入按 `0.9` 逐层衰减，最底层实际峰值约为
`1.44e-6`。`0.9` 是针对 24 层 ViT-L 和较小医学标注集的保守工程取值：保留 BEiT 的逐层
衰减原则，同时避免使用 `0.65` 时最底层几乎完全不更新，从而满足 Stage1 联合微调要求。

所有组由 AdamW 优化，weight decay 为 `0.05`；bias、归一化参数、CLS/mask token 和位置
嵌入不做 decay。默认使用 BF16、梯度裁剪 `1.0`、micro-batch `1`、梯度累计 `8`。

## 调度与验证

- 总更新步的前 5% 线性 warm-up。
- 之后单周期 cosine 衰减至各组峰值学习率的 1%。
- scheduler 按 optimizer update 而不是按 epoch 更新，并随 checkpoint 保存/恢复。
- 损失为 Cross Entropy + soft Dice；最佳模型按验证集肿瘤 Dice 保存，同时记录肿瘤 IoU、
  pixel accuracy 和混淆矩阵。
- FiLM 的零初始化会让 Stage2 在第一个更新尚无梯度，因此审计前 10 个 optimizer update；
  Stage1、Stage2、Decoder 必须在窗口内都出现有限非零梯度，否则训练立即失败。
