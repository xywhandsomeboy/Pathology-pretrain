# 宫颈 WSI 不平衡改良版本 S / ST / STA

这些版本是训练策略消融，不改变 Decoder V1/V2、Stage1 或 Stage2 的网络定义，也不覆盖
原来的 `decoder_runs/`。改良前基线永久保存在 Git 标签
`segmentation-pre-imbalance-variants-20260903`。

## 为什么没有照搬通用的“60% 阳性 patch”建议

冻结数据集的真实统计为：

- 191 张 WSI：train/valid/test 为 90/44/57；
- 训练 patch 97,350 个，其中阳性 57,294 个，占 58.85%；
- 训练 patch 的肿瘤像素比例为 47.96%，验证集为 29.08%；
- 68.59% 的阳性 patch 有至少 95% 肿瘤像素，64.58% 完全是肿瘤；
- 阳性 patch 数量最多的 15 张 WSI 提供全部阳性 patch 的 87.90%。

因此首要问题不是继续增加阳性总比例，而是减少大病灶/病灶内部 patch 的支配，并增加
含正常—肿瘤过渡的边界 patch。

## 三个逐步叠加的版本

| Profile | 相对原版的唯一新增策略 | 目的 |
|---|---|---|
| S | 60% 阳性；阳性中边界/内部各 50%；按 WSI patch 数平方根温和均衡 | 控制大病灶支配并增加边界监督 |
| ST | S + 肿瘤前景 Tversky，`0.3*FP + 0.7*FN` | 在保持采样相同的条件下单独检验漏检惩罚 |
| STA | ST + torchvision mild `ColorJitter` | 单独检验染色/扫描颜色变化的泛化影响 |

这里 CE 与 overlap loss 的外部系数都保持 1.0，CE 肿瘤权重保持 1.0。这样 S→ST 只改变
overlap loss，ST→STA 只增加颜色增强。不要把 Tversky 内部的 FP/FN 系数与 CE/Tversky
外部组合系数混为一谈。

边界 patch 定义为 `0 < tumor_fraction < 0.999999`，内部 patch 为其余阳性 patch。每个分层
中一张 WSI 的目标配额与其可用 patch 数的平方根成比例：它比原来的面积线性贡献更均衡，
又不会强迫只有一个 patch 的 WSI 与大型病灶完全等量。每个 patch 每 epoch 最多出现两次，
同一个 batch 中禁止重复 patch，满足 GNN 在线目标节点必须唯一的约束。每个 epoch 调用
`set_epoch`，随机顺序可复现但不会逐 epoch 固定。

## 新增诊断

所有改良 profile 使用 256-bin 流式概率直方图，不在内存保存全部 WSI 像素。每轮记录：

- tumor precision、recall、specificity、F2；
- 预测/真实肿瘤像素比例；
- 肿瘤像素和背景像素上的平均肿瘤概率；
- 近似 PR-AUC；
- 验证集 F2 最优阈值及该阈值下的 precision/recall。

默认原版仍使用 uniform patch sampling、CE+Dice、无颜色增强、无概率直方图，因此已有命令
和旧 checkpoint 的训练语义不变。旧 checkpoint 恢复时会自动补齐这些默认配置字段。

## 启动一个独立版本

```bash
# 只打印命令，不启动
DRY_RUN=1 scripts/run_cervical_improved_variant.sh \
  S distance_only v1 0

# 实际启动 STA；输出写到 decoder_runs_improved，不覆盖原版
scripts/run_cervical_improved_variant.sh \
  STA weighted_pretrain_distance_context v2 1
```

参数依次是：

1. `S`、`ST` 或 `STA`；
2. `baseline`、`distance_only` 或 `weighted_pretrain_distance_context`；
3. `v1` 或 `v2`；
4. GPU 编号；
5. 可选的 Stage2 run id。

建议先在当前表现最好的 Stage2/Decoder 组合上完成 S、ST、STA 三项消融，再决定是否扩展到
全部六种架构组合，避免一次混入多个变量。

## 推理阈值

`infer.py` 和 `infer_v2.py` 新增可选参数：

```bash
--tumor-threshold 0.37
```

阈值必须来自验证集记录的 `best_f2_threshold`，测试集不能用于选择阈值。不传该参数时仍按
原来的 `argmax` 输出，保证旧行为不变。当前没有小连通域删除、opening 或腐蚀后处理。

## 论文依据

- Tversky loss: <https://arxiv.org/abs/1706.05721>
- Focal Tversky（只作为后续备选，本轮未启用 focal）: <https://arxiv.org/abs/1810.07842>
- 病理颜色增强/染色归一化比较: <https://pubmed.ncbi.nlm.nih.gov/31466046/>
