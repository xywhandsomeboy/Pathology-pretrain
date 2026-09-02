# V2：全分辨率逐级残差修正 Decoder

状态：已独立实现，尚未启动训练。

## 文件边界

- `dinov2_segmentation/models/v2_decoder.py`：全分辨率分支、CBAM 注意力、全局
  上采样和逐级残差修正模块。
- `dinov2_segmentation/models/model_v2.py`：V2 总模型。
- `dinov2_segmentation/train_v2.py`：V2 专用训练入口。
- `dinov2_segmentation/infer_v2.py`：V2 专用推理与 WSI 拼接入口。
- `dinov2_segmentation/tests/test_model_v2.py`：CPU 结构不变量测试。

V1 的模型文件、训练入口和 checkpoint 契约均未修改。Stage2 的配置、图数据和
运行脚本也不属于 V2 的修改范围。

## 数据流

```text
DINO dense tokens + GNN context
               ↓
全局语义逐级上采样到 H×W ─────────────── G0
                                             │
原始 patch → CBAM + 3×3 Conv（不下采样）→ H │
                                             ↓
                   G(k+1) = G(k) + α(k)R(k)
                                             ↓
                                      segmentation head
```

高分辨率分支中的全部卷积均为 `stride=1`，输入图像为多大，其输出特征就保持多大。
CBAM 依次执行通道注意力和空间注意力；`3×3 Conv` 在不改变 H×W 的前提下完成
通道扩展与局部纹理提取。

每一个修正块执行：

```text
Concat(Gk, H)
→ 1×1 Conv + LayerNorm + GELU
→ CBAM
→ 3×3 Conv（通道扩展）
→ LayerNorm + GELU
→ 3×3 Conv（恢复到全局通道数）
→ Rk
→ G(k+1) = G(k) + α(k)R(k)
```

每一级只有最后这一次残差相加。不同修正块拥有互不共享的可学习标量 `α(k)`，默认
初始化为 `1e-2`。没有额外的最外层长残差，也没有 DenseNet 式跨层连接。

## 训练

从 `CerviPath` 根目录运行，并为 V2 使用独立输出目录：

```bash
/home/user/90T/xiayw/CerviPath/dinov2/.venv/bin/python \
  -m dinov2_segmentation.train_v2 \
  --train-manifest manifests/train.csv \
  --val-manifest manifests/val.csv \
  --output-dir outputs/segmentation_v2 \
  --num-classes 4 \
  --image-size 224 \
  --batch-size 4 \
  --high-resolution-depth 3 \
  --correction-depth 4
```

`--correction-depth` 控制残差修正次数。由于所有修正都在 H×W 上执行，V2 的显存
开销明显高于 V1，改变输入尺寸后应重新评估 batch size。

V2 checkpoint 强制保存：

```text
model_version = v2_full_resolution_residual_correction
```

因此 `train_v2.py` 和 `infer_v2.py` 会拒绝加载 V1 checkpoint，防止两个版本混用。

## 推理

```bash
/home/user/90T/xiayw/CerviPath/dinov2/.venv/bin/python \
  -m dinov2_segmentation.infer_v2 \
  --manifest manifests/test.csv \
  --checkpoint outputs/segmentation_v2/checkpoint_best.pt \
  --output-dir outputs/wsi_masks_v2
```
