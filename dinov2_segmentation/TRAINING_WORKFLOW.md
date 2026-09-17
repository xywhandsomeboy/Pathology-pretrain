# 联合训练流程：分层解冻、特征一致性与独立验证

本说明对应当前代码提供的可选流程。运行配置、权重和数据不包含在源码提交中；
更新仓库不会自动启动训练，也不会改写运行进程已经加载的参数。

## 入口与兼容

- `train_joint.py`：联合训练；默认保留 `legacy` 调度和特征策略。
- `train_joint_parallel.py`：串行 / DDP 包装，支持显式断点恢复和迁移。
- `train_joint_staged.py`：在相同训练器上启用 `separate_gnn_fusion` 调度。
- `validate_progress_checkpoints.py`：另起进程消费磁盘上的权重及验证任务。

V1 解码器的模型、细节分支、语义分支、融合解码器和基础模块没有结构修改。
共同训练入口也包含 V3/V4/V5 的可选分支，因此仓库一并包含其实现依赖；
只有明确选择对应 `--decoder-version` 才启用这些分支。

## 分层解冻与特征来源

使用 `--unfreeze-schedule separate_gnn_fusion` 与
`--graph-feature-policy staged_consistent` 可分别控制 GNN、融合层和 DINO。
以下为 20k / 60k / 100k 边界的配置示例，具体运行以保存的配置为准：

| 参数更新步 | 可训练模块 | 图输入特征 |
|---|---|---|
| 0–19,999 | 分割解码器 | 固定预训练节点特征 |
| 20,000–59,999 | 解码器、GNN | 输入特征仍固定；不因 GNN 更新而刷新 |
| 60,000–99,999 | 上述模块、Stage1 融合模块 | 复用原始 DINO 输出，用当前融合参数计算所需邻域 |
| 100,000 起 | 上述模块、配置指定的 DINO 尾部 block | 用当前上游参数计算所需特征 |

相关参数：`--decoder-only-steps`、`--fusion-unfreeze-step`、
`--dino-unfreeze-step`、`--dino-unfreeze-blocks`。
邻域跳数从所用 Stage2 配置的实际层数读取，不在训练流程中固定为五层。
`--raw-feature-cache`、`--node-image-root` 指定磁盘原始特征缓存和节点图像来源。
融合 / DINO 解冻时，精确策略计算目标的完整所需邻域；缓存不能跨不兼容的来源复用。
可选冻结前缀缓存由 `--cache-frozen-dino-prefix` 启用，并校验来源及精度协议。
本流程不启用实验性的 EMA 历史特征队列。

## 保存和独立验证

配置 `--checkpoint-interval-steps 1000 --retain-progress-checkpoints`
可每 1,000 次参数更新保留独立 progress 权重。
`--monitor-interval-steps 2000` 控制送入子集验证队列的步数间隔；
`--async-full-validation` 使 epoch 全量验证由独立进程负责。
这些频率是显式配置示例，不是所有旧启动脚本的默认值。

训练器持久化模型、优化器、scheduler、课程步数及恢复所需状态；
验证任务及候选权重保留在磁盘。验证器处理不可变快照、记录完成状态并避免
重复评估已完成的相同任务。子集和全量指标必须分开比较。
独立进程避免训练器等待验证，但共享 GPU/磁盘时仍可能争用资源。

`scripts/watch_cervical_progress_validation.sh` 提供验证器启动入口。
其默认子集大小为 50,000，可由 `ASYNC_VALIDATION_SUBSET_SIZE` 覆盖。

## 断点与检查

`--resume` 恢复兼容运行；`--init-checkpoint` 仅加载模型作为新运行初始化；
`--migrate-resume` 用于明确支持的串行到 DDP 迁移。三者互斥。
恢复会检查训练配置；`curriculum_step` 与 scheduler 进度分别保存。

测试覆盖阶段边界、恢复、缓存前向与梯度一致性、DDP 同步、权重留存、
验证任务去重和异步 epoch 验证。运行测试需 Stage2 源码与仓库根目录均在
`PYTHONPATH` 中，且环境提供 PyTorch、PyG 和 pytest。
