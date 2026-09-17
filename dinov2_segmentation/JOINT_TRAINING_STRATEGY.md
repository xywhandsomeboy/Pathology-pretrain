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

## 按 optimizer step 渐进解冻（legacy 调度）

下表为保留的 `legacy` 调度。将 GNN、融合模块和 DINO 分开解冻的
`separate_gnn_fusion` 调度见 [训练流程说明](TRAINING_WORKFLOW.md)。

数据规模扩大后，一个 epoch 约包含 20 万次 optimizer update。继续用“第 3/8 个 epoch”
作为解冻边界，会让 GATv2 和 DINO 分别等待数百万乃至数千万个 patch 后才开始更新。因此
新训练使用独立的 `curriculum_step`（即 phase step）控制参数解冻；它只在 optimizer
update 后递增，不等同于 scheduler 的 `current_step`，并随 checkpoint 保存和恢复。

| `curriculum_step` | 可训练模块 | 目的 |
|---:|---|---|
| `0–19,999` | Decoder | 先让随机初始化的分割头对齐预训练特征 |
| `20,000–59,999` | Decoder + Stage2 GATv2 + Stage1 spatial/local/fusion | 适配图上下文和节点融合，同时保持 ViT 主干稳定 |
| `60,000–99,999` | 上述模块 + DINO 顶部 2 个 Transformer block | 温和引入高层视觉语义更新 |
| `>=100,000` | 上述模块 + DINO 顶部 4 个 Transformer block | 进入最终端到端联合微调 |

生产入口对应参数为 `--decoder-only-steps 20000`、
`--stage1-partial-unfreeze-step 60000 --stage1-partial-unfreeze-blocks 2`、
`--stage1-final-unfreeze-step 100000 --stage1-final-unfreeze-blocks 4`。这些阈值按全局
optimizer update 计数，不因串行/DDP 的每进程 batch 表述而改变。

阶段边界发生在 optimizer update 之间，而不是任意 micro-batch 中间。切换阶段时会同时更新
`requires_grad`、模块 train/eval 模式、参数组学习率缩放，并在 DDP 下按新的可训练参数集合
安全重建梯度同步包装。没有解冻的 DINO block 继续保持冻结和 eval 状态。

`curriculum_step` 必须独立持久化，不能从 scheduler 步数反推。尤其是旧 S/ST 在完成 epoch 0 后
已经积累约 20 万个 scheduler step；迁移时若直接把这个数当成解冻进度，会跳过 GATv2
适配和 DINO 顶部 2 层阶段。旧 epoch-0 checkpoint 因此显式映射为 `curriculum_step=20,000`：
Decoder 对齐视为完成，随后仍依次执行 40,000 步适配、40,000 步顶部 2 层微调。scheduler
保留其真实 optimizer 进度而不回退，迁移来源、映射值和学习率切换写入 provenance。

这是一条为保住旧 S/ST 已完成计算而设计的迁移路径；它们经历过旧的 epoch 调度和比例
warmup，而全新 STA 从 step 0 使用本策略。因此迁移后的 S/ST 与 fresh STA 不再是严格的
单变量公平对照，报告结果时必须注明。旧运行本身的 epoch 调度记录属于实验历史，不回写
或改写。

## 学习率、检查点与验证

- 固定前 `20,000` 个 optimizer step 线性 warm-up（`--warmup-steps 20000`），不再使用
  总训练步数的百分比。这样数据量或 epoch 数变化时，达到峰值学习率所需的实际更新数
  保持不变。
- warm-up 后使用单周期 cosine 衰减至各组峰值学习率的 1%。scheduler 仍按 optimizer
  update 更新，并与独立的 `curriculum_step` 一同保存/恢复。
- 每 `20,000` 个 optimizer step 保存可恢复的 `checkpoint_progress.pt`
  （`--checkpoint-interval-steps 20000`）。它包含模型、AdamW、
  AMP scaler、scheduler、`curriculum_step`、当前 epoch/batch 游标和累计训练统计；串行与
  DDP 的保存都只发生在完整 optimizer update 边界，可用于同模式续训或受校验的串行到
  DDP 迁移。
  sampler 会从保存的 batch 游标确定性重放索引且不读取已经完成的样本，但 worker 内随机增强
  和在线图缓存不承诺逐位重现，因此恢复后的浮点轨迹可能有轻微差异。
- 损失为 Cross Entropy + soft Dice；最佳模型按验证集肿瘤 Dice 保存。完整混淆矩阵在设备端
  累积到 epoch 结束，再统一计算肿瘤 Dice、precision、recall、F2 和预测肿瘤比例。
- 梯度审计从 GATv2/融合适配阶段开始累计 Stage1、Stage2 和 Decoder 梯度，并在顶部 4 层
  阶段把 DINO backbone 梯度纳入完成条件；所有值都必须有限且非零，否则在完整联合阶段的
  审计窗口结束时立即失败。FiLM 零初始化允许梯度在短审计窗口内延迟一个更新出现。
