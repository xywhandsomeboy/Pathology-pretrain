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
