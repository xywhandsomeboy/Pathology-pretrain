# Decoder 设计备忘录

## V1 双分支融合入口

需要保留并重点检查的融合步骤：

```text
Semantic feature + Detail feature
              ↓
    Concat（沿通道维拼接）
              ↓
          1×1 Conv
              ↓
         LayerNorm
              ↓
            GELU
              ↓
      Channel Attention
```

简写：

```text
Concat → 1×1 Conv → LayerNorm → GELU → Channel Attention
```

当前默认通道关系：

```text
[B, 128, H/4, W/4] + [B, 128, H/4, W/4]
                        ↓ Concat
                 [B, 256, H/4, W/4]
                        ↓ 1×1 Conv
                 [B, 128, H/4, W/4]
```

作用：先对齐两个分支的空间尺寸，再沿通道维拼接；使用 `1×1 Conv` 完成通道压缩
和跨分支线性融合，随后通过 `LayerNorm` 稳定特征分布，用 `GELU` 引入非线性，
最后由 Channel Attention 显式学习不同融合通道的重要性。

当前实现位置：`dinov2_segmentation/models/fusion_decoder.py` 中的
`AttentionConvFusionDecoder.input_projection`。

## 候选训练改动：肿瘤区域加权

状态：**仅记录，当前四个模型不启用。**

若模型进入 Stage1 融合层与 Stage2 联合训练阶段后，仍持续表现为 Precision 高、
Recall 低，并且预测肿瘤像素比例明显低于真实比例，可在一组新的对照实验中调整
二分类交叉熵：

```text
background class weight = 1.0
tumor class weight      = 1.5（首选）
```

Dice Loss 继续保持现状。当前 Dice 对出现的背景类和肿瘤类等权平均，已经具有类别
平衡作用；先只提高交叉熵中的肿瘤权重，可以较温和地提升漏检肿瘤像素的代价。

建议备选权重为 `1.25 / 1.5 / 2.0`。必须使用新的输出目录从第 1 个 epoch 重新训练，
不得在现有检查点中途切换损失，以免混淆实验条件。是否启用以第 4–5 个 epoch 的
验证集 Recall、Precision、Dice 和预测肿瘤像素比例为依据。
