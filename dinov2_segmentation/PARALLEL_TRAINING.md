# 可切换串行 / DDP 的联合分割训练

新入口 `dinov2_segmentation.train_joint_parallel` 支持同一模型在单 GPU
单进程（`serial`）或多 GPU 数据并行（`ddp`）下训练。Stage1、Stage2、Decoder
结构和分阶段解冻规则沿用已有联合训练；DDP 在每张 GPU 上保存完整模型副本，
分摊训练样本并同步梯度。它不是把不同的 S/ST/STA 模型分配成一个模型的并行分支。

新增入口和输出目录独立于正在运行的旧入口。创建代码不会自动启动任务或迁移现有进程。

## 启动方式

在仓库根目录执行，GPU 编号由调用者显式指定。下面的单卡 `1` 和双卡 `0,1`
均为命令示例，文档中的命令未自动启动；实际运行前请根据服务器上的任务占用选择 GPU。

```bash
# 只显示命令并校验已有输入，不创建输出、不占用 GPU。
DRY_RUN=1 RUN_SUFFIX=_trial01 bash scripts/run_cervical_parallel_variant.sh \
  serial S weighted_pretrain_distance_context v1 1

DRY_RUN=1 RUN_SUFFIX=_trial01 bash scripts/run_cervical_parallel_variant.sh \
  ddp S weighted_pretrain_distance_context v1 0,1

# 真正启动单卡任务。
BATCH_SIZE=16 GRADIENT_ACCUMULATION=1 RUN_SUFFIX=_trial01 \
bash scripts/run_cervical_parallel_variant.sh \
  serial S weighted_pretrain_distance_context v1 1

# 真正启动双卡 DDP；下面将每卡 batch 调为 8，使有效 batch 与上面相同。
BATCH_SIZE=8 GRADIENT_ACCUMULATION=1 RUN_SUFFIX=_trial01 \
bash scripts/run_cervical_parallel_variant.sh \
  ddp S weighted_pretrain_distance_context v1 0,1
```

位置参数依次是：`serial|ddp`、`S|ST|STA`、Stage2 版本、`v1|v2`、GPU 列表，
以及可选的 **Stage2 预训练运行目录名**（不是本轮训练名称）。默认该目录名为
`s1_iter39999_compare_01_retry2`。Stage2 支持 `baseline`、`distance_only` 和
`weighted_pretrain_distance_context`；前者读取 `graphs/dual`，后两者读取
`graphs/distance`。`serial` 要求一个 GPU，脚本的 `ddp` 模式要求至少两个不同的 GPU。

输出位于：

```text
<work-root>/decoder_runs_parallel/<S|ST|STA>/<stage2-version>/<v1|v2>_<serial|ddp><RUN_SUFFIX>/
```

未指定 `RUN_SUFFIX` 时，脚本自动添加时间和进程号，避免意外复用之前的实验目录。
指定例如 `_trial01` 可以稳定定位同一轮实验。输出锁防止两个进程同时写入同一个目录。
已经完成的目录会直接返回；存在 `checkpoint_last.pt` 时自动续训；没有检查点的非空
目录会被拒绝，需要改用新的后缀并保留原目录供排查。

## 数据和训练设置

默认数据根目录是 `Data/cervical_segmentation_latest_area_20260904`，可用
`CERVICAL_WORK_ROOT` 指向另一套已经完成预处理的数据。脚本要求
`preprocessing.complete`、训练/验证清单、对应图目录及预训练检查点均已存在。
它直接复用准备好的训练/验证划分，不自行下载、切图、重新分配 WSI 或创建测试集。
因此新增下载的数据需要先通过数据准备流程生成完整的新快照，再启动新一轮训练；
仅增加下载文件不会改变这个入口读取的样本。运行期间应保持这套数据清单及图文件不变。
运行记录会保存训练/验证清单、图目录、Stage1/Stage2 配置和检查点的绝对路径、
大小及 `mtime_ns`。完整续训严格比较这些输入标识及配置，不能用原运行目录承接
不同的数据快照。这是文件/目录元数据检查，不是对所有图文件进行内容哈希；
仍需保持快照不可变。

默认 `BATCH_SIZE=16` 指 **每张 GPU 的 batch**。名义有效 batch 为：

```text
每卡 batch × GPU 数量 × GRADIENT_ACCUMULATION
```

例如每卡 16、双卡、累积 1 次，对应有效 batch 32；单卡则是 16。
串行保留最后一个不完整 batch。DDP 为保证所有 rank 同步且局部 batch 等大，
每个 epoch 略过不足一个全局 batch 的尾部（最多 `每卡 batch × 卡数 - 1` 条），
数量记录为 `execution.dropped_training_samples`；下一 epoch 会重新打乱顺序。
这不会从数据清单删除 WSI，验证集也不丢弃或补齐任何样本。
两种模式的最后一个梯度累积组可能更小。学习率不会随卡数自动放大，
对照实验可以用单卡 batch 16 / 双卡每卡 batch 8 控制有效 batch。

沿用当前无面积损失的设置：S 使用交叉熵 + Dice，ST 和 STA 使用交叉熵 +
前景 Tversky（FP 权重 0.3，FN 权重 0.7），STA 额外启用温和颜色增强。
不包含肿瘤面积平方差损失。WSI 分层采样、正常样本保留及 Stage2 图规则沿用既有配置。

默认训练 50 epoch，Decoder DropPath 为 0.2，学习率分别为 Decoder `1e-4`、
Stage2 `1e-5`、Stage1 融合层 `1e-5`、Stage1 主干 `2e-6`，主干逐层衰减系数 0.8。
使用 10% warmup + cosine 衰减，最低学习率比例 0.01。前 3 epoch 只训练 Decoder，
从第 4 epoch 加入 Stage2 和 Stage1 融合层，从第 9 epoch 解冻 Stage1 顶部 4 层；
最终阶段的学习率缩放为 0.5。

可调整的环境变量包括 `BATCH_SIZE`、`GRADIENT_ACCUMULATION`、`DECODER_WORKERS`
（每进程，默认 8）、`DECODER_EPOCHS`、`DECODER_DROP_PATH_RATE`、`PYTHON_BIN`、
`PARALLEL_OUTPUT_ROOT`、`STAGE1_ROOT`、`STAGE2_ROOT`。不建议仅为了启用并行而更换
数据或预训练权重，否则比较结果同时包含这些变化的影响。

## 续训与切换执行模式

使用相同模式、GPU 数量、每卡 batch、参数、数据及 `RUN_SUFFIX` 重新执行命令，
可以从该目录的 `checkpoint_last.pt` 恢复模型、优化器和调度器。完整续训严格比较
模式、world size、有效 batch、每个 epoch 的 batch 数及前述输入元数据。
改变卡数可能改变每个 epoch 的更新步数，入口会拒绝此类优化器/调度器续接。

模型权重可以跨单卡/DDP 加载。需要换模式或 GPU 数量时，用独立的新后缀及
`INIT_CHECKPOINT` 进行 **仅模型权重初始化**，这是新训练，优化器、调度器和 epoch
计数重新开始：

```bash
INIT_CHECKPOINT=/absolute/path/to/previous/run/checkpoint_last.pt \
BATCH_SIZE=8 RUN_SUFFIX=_warmstart01 \
bash scripts/run_cervical_parallel_variant.sh \
  ddp S weighted_pretrain_distance_context v1 0,1
```

`INIT_CHECKPOINT` 不可与已经有续训检查点的输出目录混用。模型结构仍需匹配，例如
V1 权重不能直接用于 V2。串行与 DDP 共用不带 DDP `module.` 前缀的模型保存方式。

## 验证、性能和短测试

训练时，交叉熵的分子/分母以及 Dice/Tversky 所需的交集、预测量等统计量会通过
可反向传播的 `all_reduce(SUM)` 在各 rank 间汇总，再计算同一同步 step 的全局
batch 损失。图分支也通过可反向传播的 `all_gather` 汇集各 rank 当前目标节点的
新鲜特征，使同一 WSI 的全局 batch 目标节点在图上下文中共同参与计算，梯度可以
回到对应 rank 的特征提取器。

在同一组样本、匹配初始权重、相同确定性前向计算及有效 batch 的条件下，这些训练
损失与梯度可对应串行全局 batch 的计算。不同随机增强、DropPath 和浮点归约次序
仍可能使实际曲线不同；梯度累积也仍是多个 microbatch 损失的累积。

验证样本按 rank 分片且不补齐重复样本，各 rank 的混淆矩阵等统计量汇总后计算
Dice / Precision / Recall，指标对应实际验证样本的全局统计。验证阶段各 rank
可能有不同数量的 batch，因此验证损失使用本地 batch 损失按样本数加权汇总，
其中 Dice/Tversky 损失仍会受 batch 划分影响；这一限制不影响通过全局混淆矩阵
计算的 Dice / Precision / Recall。

DDP 加速效果取决于数据读取、整张 WSI 图的处理开销、卡间通信和显存压力，
不会保证按 GPU 数量线性提速；多卡不会把一个模型副本的显存平均拆到多张卡。
Stage1 图像编码和 Decoder 分摊到各卡；为保留跨卡节点之间的梯度，当前同步 step
的全局目标节点图上下文会在每个 rank 计算一次，这部分 GNN 计算不是分片加速。
每卡默认 8 个数据加载 worker，双卡即共 16 个，应结合 CPU、磁盘和内存余量调整。
DDP 接口和通信机制可参考 [PyTorch 2.0 官方文档](https://pytorch.org/docs/2.0/generated/torch.nn.parallel.DistributedDataParallel.html)。

代码级测试可以只使用 CPU 执行，其中包含双进程检查，需要允许进程通过本地
socket 通信，不需要占用训练 GPU：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' dinov2/.venv/bin/python -m pytest -q dinov2_segmentation/tests
```

可使用独立的输出后缀限制 batch 数做短测试，下面会覆盖 3 个解冻阶段：

```bash
MAX_TRAIN_BATCHES=2 MAX_VAL_BATCHES=2 DECODER_EPOCHS=9 \
DECODER_WORKERS=0 BATCH_SIZE=1 RUN_SUFFIX=_smoke01 \
bash scripts/run_cervical_parallel_variant.sh \
  ddp S weighted_pretrain_distance_context v1 0,1
```

这些 batch 上限是每个 rank 的上限，仅用于流程检查；不要把受限验证集的结果当成
完整验证成绩。正式运行应取消这两个变量或设为 0。

## 断开电脑后继续运行

在服务器的 `tmux` 会话内启动命令，再按 `Ctrl-b`、`d` 脱离会话：

```bash
tmux new-session -s cervical_parallel
# 在新会话中进入仓库，执行上面的训练命令。
```

关闭本地电脑、SSH 或 IDE 不会停止这个服务器会话。服务器自身重启仍会中断任务，
之后可按上面的检查点续训方式恢复。脚本本身不创建定时器、不监控或停止其他训练。
