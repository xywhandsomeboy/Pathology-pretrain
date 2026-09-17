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
使用固定 `20,000` optimizer step warmup + cosine 衰减，最低学习率比例 0.01；warmup
不再随总 epoch 数或数据规模成比例增长。

解冻由独立的 `curriculum_step` 驱动：`0–20,000` 步只训练 Decoder；`20,000–60,000` 步
加入 Stage2 GATv2 和 Stage1 融合模块；`60,000–100,000` 步再解冻 DINO 顶部 2 层；
从 `100,000` 步起解冻 DINO 顶部 4 层。`curriculum_step` 与 scheduler 的 `current_step`
分别保存、分别校验；改变 scheduler 进度不会隐式跳过解冻阶段。阶段可在 epoch 内切换，
DDP 会在 optimizer update 边界按新的可训练参数集合重建同步包装。

每 `20,000` 个 optimizer step 写入可恢复的 `checkpoint_progress.pt`，因此后续运行不必等待
一个超大 epoch 及完整验证结束才获得恢复点。checkpoint 同时保存模型、优化器、AMP scaler、
scheduler、`curriculum_step`、epoch/batch 游标和累计统计；串行/DDP 都在完整 optimizer update
边界协调保存。epoch 末的 `checkpoint_last.pt` 与按验证 Dice 选择的
`checkpoint_best.pt` 仍保留原有职责。前者标记 `epoch_complete=True`；epoch 内的
`checkpoint_progress.pt` 标记 `epoch_complete=False`，并带有 `next_batch_index` 与
`train_progress`，恢复时不会重新读取已经完成的 batch。

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

### 保留完整训练状态迁移到 DDP

对于由旧入口 `dinov2_segmentation.train_joint` 写出的串行 checkpoint，可使用
显式的 `MIGRATE_RESUME` 通道迁移到 DDP。它与普通 `--resume`、仅加载模型权重的
`INIT_CHECKPOINT` 分开，避免意外放宽常规断点恢复的严格校验：

```bash
MIGRATE_RESUME=/absolute/serial/run/checkpoint_last.pt \
BATCH_SIZE=8 RUN_SUFFIX=_from_serial_epoch_boundary_01 \
bash scripts/run_cervical_parallel_variant.sh \
  ddp S weighted_pretrain_distance_context v1 0,1
```

迁移要求使用全新的并行输出目录。新格式可以使用协调写出的 step checkpoint 在 epoch
内部恢复；旧格式仍必须使用完整串行 epoch 结束后生成的 checkpoint。除“每进程 batch”、
step-checkpoint 频率以及下面明确列出的旧 epoch 调度转换外，模型版本、输入路径、损失、
采样和优化配置必须相同。源串行与目标 DDP 不仅有效累计 batch 必须相同，单次全局
micro-batch 和梯度累计次数也必须分别相同；这样 epoch 内游标才有唯一对应关系。例如源串行
batch 16、累计 1 次，对应双卡 DDP 每卡 batch 8、累计 1 次。迁移会保留模型、AdamW 动量、AMP scaler、
scheduler、独立 `curriculum_step`、epoch 内游标、已完成 epoch、最佳指标、early-stopping
状态、gradient audit 和历史记录。step checkpoint 只在 optimizer update 边界写入，
目标执行方式会先严格校验其完整性及有效全局 batch，再继续未完成的 epoch。

DDP 会略过不足一个全局 batch 的尾部，因此每个 epoch 的更新步数可能比串行少一
步。迁移器保留 scheduler 已发生的 optimizer 进度，同时独立恢复 `curriculum_step`，并将
源/目标步数、解冻阶段和迁移来源写入 `run_manifest.json`。它不会宣称两种执行方式
在随机增强、浮点归约次序或在线图缓存状态上逐位一致。

旧训练代码在训练段和完整验证段都结束后才写 checkpoint，不保存 epoch 内的
sampler、batch 或随机数状态。因此没有 `checkpoint_last.pt` 时不能无损迁移；直接
结束这种进程会丢失当前 epoch 的内存中进度。新代码的 step checkpoint 不能追溯恢复
旧进程尚未落盘的内存状态。

旧 S/ST 的 epoch-0 checkpoint 使用历史上的 epoch 解冻与比例 warmup。迁移器不会把其
约 20 万个 scheduler step 直接当作新解冻进度，而是显式设为 `curriculum_step=20,000`：
Decoder 对齐阶段视为完成，随后仍执行 GATv2/融合适配、DINO 顶部 2 层和顶部 4 层三个
阶段。scheduler 保留旧优化器真实进度，固定 warmup 策略带来的学习率切换及映射依据写入
provenance。旧 S/ST 与从 step 0 使用新调度的 fresh STA 因而不再是严格公平对照；比较时
必须标注这一差异。旧运行目录中的 epoch 调度记录保持原样，作为历史事实保留。

当前 S/ST/STA 交接可由下面的专用脚本托管：

```bash
tmux new-session -d -s cervical_ddp_handoff \
  'cd /home/user/90T/xiayw/CerviPath && bash scripts/handoff_cervical_serial_to_ddp.sh'
```

脚本会等待 S、ST 都产生 checkpoint，在目标 GPU 没有其他计算任务时才停止精确
匹配的旧进程，然后以双卡 DDP 依次运行 S、ST、STA。S/ST 使用完整状态迁移；已经
在首个 checkpoint 前退出的 STA 只能从 Stage1/Stage2 预训练权重重新开始。默认
每卡 batch 8，以保持原串行全局 batch 16；三个模型采用队列而非同时占用同一对
GPU，以避免主机内存和 I/O 过度竞争。交接日志写在
`<work-root>/logs/ddp_handoff/`。

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

可使用独立的输出后缀限制 batch 数做短执行测试。正式阈值下，这个命令只检查数据、前后向
和 DDP 流程，不会覆盖四个解冻阶段；阶段切换由使用缩小阈值的代码级测试单独验证：

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

## 独立的 step-checkpoint 验证进程

`checkpoint_progress.pt` 首先是恢复点，不能在没有验证结果时当作最优模型。为了在超大
epoch 中间观察权重效果，可启动一个与训练状态完全隔离的单进程验证器：

```bash
tmux new-session -d -s cervical_async_validation \
  'cd /home/user/90T/xiayw/CerviPath && \
   exec bash scripts/watch_cervical_progress_validation.sh \
   >> Data/cervical_segmentation_latest_area_20260904/logs/async_validation.log 2>&1'
```

验证器每 30 秒观察 `decoder_runs_parallel`，先把新一代
`checkpoint_progress.pt` 通过硬链接固定到不可变 spool，再按顺序验证。捕获线程在验证
期间仍继续工作，因此训练覆盖同名 progress 文件不会让结果与权重错配。默认协议为：

- 从原验证清单确定性生成固定 50,000-patch 子集；
- 子集按 WSI 及 negative/boundary/interior 原始占比进行分层；
- 使用 checkpoint 记录的每 rank batch（当前为 8）、损失配置、图和 BF16 设置；
- 输出 Loss、Dice、Precision、Recall、F2、混淆矩阵等；
- 先等待目标 GPU 至少有 24 GiB 空闲显存，再加载模型；
- 验证结束释放模型显存，不改变训练进程、scheduler、history 或 early stopping。

结果写到每个运行目录的 `async_validation/results/`，`latest.json` 指向最近结果，
`best.json` 与 `checkpoint_best_monitor.pt` 保存同一验证协议下子集 Dice 最好的中间权重。
名称中的 `monitor` 表示它只用于快速趋势比较，不能替代完整验证选出的正式
`checkpoint_best.pt`。epoch 末仍由训练进程对全部 968,133 个验证 patch 运行验证，并用
完整验证 Dice 决定正式 best。

服务器只有两张 GPU；DDP 运行时独立验证器必然和其中一张卡共享算力。默认使用 GPU 0、
验证 batch 较小且进程 CPU nice 值为 10，但仍会降低 DDP 吞吐。可通过
`ASYNC_VALIDATION_GPU_ID`、`ASYNC_VALIDATION_SUBSET_SIZE`、
`ASYNC_VALIDATION_WORKERS` 和 `ASYNC_VALIDATION_MIN_FREE_GIB` 调整。验证失败只会保留
spool 和错误记录供重试，不会终止训练。
