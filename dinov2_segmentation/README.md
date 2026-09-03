# DINO–GNN 全局语义 + 高分辨率局部分割

这个目录实现已经确认的双分支方案。Stage1/Stage2 的预训练彼此独立，但最终分割训练使用
`train_joint.py` 联合微调 Stage1、Stage2 GATv2 和 Decoder；离线 Stage1B 结果只用于建立
图拓扑及初始化邻居特征记忆，不能代替最终训练中的在线前向。

## 最终网络

| 路径 | 输入 | 处理 | 输出 |
|---|---|---|---|
| 全局语义分支 | 当前 patch 的 DINO dense tokens + 该节点的整张 WSI GNN context | 1×1 投影；GNN context 通过 FiLM 调制 token map；ConvNeXtV2 block 卷积上采样 | H/4 的高层语义图 |
| 高分辨率分支 | 与上述节点坐标完全相同的原图 patch | 两层浅 stem 到 H/4；之后 ConvNeXtV2 与 HRFormer 局部窗口注意力交替，始终不再降采样 | H/4 的纹理、边缘和细胞细节图 |
| 融合 Decoder | 两个 H/4 特征图 | 拼接 + 1×1 Conv；通道注意力；HRFormer 与 ConvNeXtV2 交替融合；两次 CNN 上采样 | 与原 patch 同分辨率的类别 logits |

这里的“高分辨率”明确指 H/4 主干，而不是把 224×224 的全分辨率直接送进自注意力。这样既保留边界，又避免注意力的显存平方增长。全局语义来自分支一，分支二没有深层低分辨率语义金字塔。

## 数据流

1. Stage1B 对全部 WSI patch 生成初始 `node_features`，随后仅依据固定坐标建立 dual 或
   distance 图；图保留 `patch_ids`、坐标、level 和初始节点特征。
2. 最终监督 batch 从原图重新执行可训练 Stage1，在线产生 DINO dense tokens 以及融合
   CLS/global/local crop 的节点特征。
3. 当前目标节点的在线特征替换图记忆中的对应节点。对目标节点提取 5-hop 子图，恰好覆盖
   5 层 GATv2 的完整感受野；Stage2 输出在线 context，梯度继续回传到 Stage1。
4. 每个已见节点的在线特征以 detached 形式刷新邻居记忆，避免每次把整张 WSI 的全部原图
   同时放入显存；当前目标节点的计算图不会 detach。
5. Decoder 同时接收原图、在线 dense tokens 和在线 GATv2 context。分割损失一次反向传播
   更新 Stage1、Stage2 与 V1/V2。

联合训练 manifest 直接使用数据准备阶段的 CSV，必须包含
`slide_id, patch_id, x, y, level, image_path, mask_path`。mask 严格限制为 `{0,1}`：正常/背景
为 0，Low Grade、High Grade 与 Malignant 均为肿瘤 1。

已有 `patch_grid_positions*.csv` 时可以转换 manifest：

```bash
/path/to/python -m dinov2_segmentation.build_manifest \
  --patch-csv Data/meta/patch/patch_grid_positions.csv \
  --patch-root /path/to/moved/patches \
  --feature-dir features \
  --mask-root masks \
  --output manifests/train.csv
```

## 训练

从 Stage2 目录运行，并把 Stage2 源码与仓库根目录加入 `PYTHONPATH`：

```bash
PYTHONPATH=/path/to/CerviPath/dinov2_stage2_2_FmH2ST:/path/to/CerviPath \
/path/to/python -m dinov2_segmentation.train_joint \
  --decoder-version v1 \
  --train-manifest decoder_selection/train.csv \
  --val-manifest decoder_selection/valid.csv \
  --graph-dir graphs/dual \
  --stage1-config stage1/config.yaml \
  --stage1-checkpoint stage1/model.pth \
  --stage2-config stage2/config.yaml \
  --stage2-checkpoint stage2/model_final.rank_0.pth \
  --output-dir outputs/baseline/v1 \
  --num-classes 2 --epochs 50 --batch-size 1 \
  --gradient-accumulation 8
```

默认优化依据相关论文采用 AdamW、5% 线性 warm-up 和单周期 cosine decay。默认峰值学习率为：
Decoder `2e-4`、Stage2 GATv2 `5e-5`、Stage1 聚合/融合层 `5e-5`、Stage1 ViT 顶层
`2e-5`；24 个 ViT block 从输出到输入按 `0.9` 逐层衰减。bias、归一化参数和 token/位置
参数不做 weight decay。首个监督更新必须产生非零 Stage1、Stage2、Decoder 梯度，否则立即
停止，并把审计结果写入 `gradient_audit.json`。

默认损失为 Cross Entropy + soft Dice，`255` 为 ignore index。模型选择依据验证集肿瘤 Dice，
同时记录 tumor IoU、pixel accuracy 和完整混淆矩阵。

需要提高肿瘤漏检代价时，可为新的独立实验添加 `--tumor-class-weight 1.5`。该选项只对
Cross Entropy 的 class 1 加权，soft Dice 保持不变；默认值 `1.0` 保持原始实验行为。

针对当前数据中“大病灶与病灶内部 patch 占主导、验证 recall 低于 precision”的问题，另有
互不覆盖原输出的 S（WSI/边界分层采样）、ST（S + 前景 Tversky）和 STA（ST + 温和颜色
增强）三个训练策略版本。它们同时记录 precision、recall、F2、PR-AUC 近似值和验证集最优
F2 阈值。完整参数、实测数据分布与启动命令见
[`IMBALANCE_AWARE_VARIANTS.md`](IMBALANCE_AWARE_VARIANTS.md)。

完整的论文依据、参数选择和工程取舍见
[`JOINT_TRAINING_STRATEGY.md`](JOINT_TRAINING_STRATEGY.md)。
实际过拟合停止证据与后续单变量实验见
[`OVERFITTING_ITERATIONS.md`](OVERFITTING_ITERATIONS.md)。

## WSI 推理与拼接

推理 manifest 可以不含 `mask_path`：

```bash
/path/to/python -m dinov2_segmentation.infer \
  --manifest manifests/test.csv \
  --checkpoint outputs/segmentation/checkpoint_best.pt \
  --output-dir outputs/wsi_masks
```

推理会先把每个 patch 的概率恢复到原 patch 尺寸，再依据 `(x,y)` 放回对应 WSI。重叠区用带非零边界的 Hann 权重融合。累加器和最终 `segmentation.npy` 都使用磁盘映射，避免整张 WSI 一次性进入内存；不同 `level` 会分别输出，未覆盖区域为背景 0。

## 关键约束

- `dense_tokens` 必须由当前可训练 Stage1 在线产生，来自未做空间平均的 DINO patch tokens，
  不包含 CLS token。
- CLS token 已在 Stage1 的 node fusion 中与 global/local spatial feature 融合，再经 Stage2 进入 `global_context`，因此没有从最终分支一中丢失。
- `global_context[i]` 必须由包含节点 `i` 的精确 5-hop 感受野产生，并与 graph node order 一致。
- 原图 patch、mask、dense token 和 GNN context 必须共享相同裁切坐标和 level。
- 训练时几何翻转先同步作用于原图和 mask；DINO token 由翻转后的原图在线生成。
- 当前代码只实现单一 H/4 高分辨率流，没有采用 FmH2ST 的双图层级结构，因此不会改变既定图预训练代理任务。
