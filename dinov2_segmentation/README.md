# DINO–GNN 全局语义 + 高分辨率局部分割

这个目录实现了已经确认的双分支方案，并与 Stage1/Stage2 预训练解耦。分割训练不会再次逐 patch 运行整张 WSI 的图网络；它读取事先按坐标严格对齐的缓存。

## 最终网络

| 路径 | 输入 | 处理 | 输出 |
|---|---|---|---|
| 全局语义分支 | 当前 patch 的 DINO dense tokens + 该节点的整张 WSI GNN context | 1×1 投影；GNN context 通过 FiLM 调制 token map；ConvNeXtV2 block 卷积上采样 | H/4 的高层语义图 |
| 高分辨率分支 | 与上述节点坐标完全相同的原图 patch | 两层浅 stem 到 H/4；之后 ConvNeXtV2 与 HRFormer 局部窗口注意力交替，始终不再降采样 | H/4 的纹理、边缘和细胞细节图 |
| 融合 Decoder | 两个 H/4 特征图 | 拼接 + 1×1 Conv；HRFormer 与 ConvNeXtV2 交替融合；两次 CNN 上采样 | 与原 patch 同分辨率的类别 logits |

这里的“高分辨率”明确指 H/4 主干，而不是把 224×224 的全分辨率直接送进自注意力。这样既保留边界，又避免注意力的显存平方增长。全局语义来自分支一，分支二没有深层低分辨率语义金字塔。

## 数据流

1. Stage1 对每个 WSI patch 输出两份内容：用于建图的 `node_features` 和未池化的 `dense_tokens`。配置 `feature.export_dense_tokens: true` 后，Stage1B 会额外写出 `decoder_features_s1_part*.pt`。
2. 使用同一个有序 patch 列表建整张 WSI 图。图中只写入轻量的 `patch_ids`、
   `levels` 和 `slide_id`；`dense_tokens` 单独保存在 Stage1 per-slide 文件中，避免
   Stage2 每次加载图时同时读取数 GB 的 decoder 特征。
3. 加载训练好的 Stage2，先执行 `model.eval()`，再对完整图调用 `model.extract_context(graph)`；不能使用数据集里的随机 5000 节点子图。保存示例：

   ```python
   model.eval()
   context = model.extract_context(graph).cpu()
   torch.save({
       "patch_ids": graph.patch_ids,
       "global_context": context,
   }, "slide_context.pt")
   ```

4. 合并成分割用的单 WSI 缓存：

   ```bash
   /path/to/python -m dinov2_segmentation.prepare_slide_store \
     --graph graph/TCGA-example.pt \
     --context context/TCGA-example.pt \
     --stage1-metadata embeddings/TCGA-example_metadata.pt \
     --dense-tokens embeddings/TCGA-example_dense_tokens.npy \
     --output features/TCGA-example.pt
   ```

   `context` 文件默认必须携带与 graph 完全同序的 `patch_ids`。旧的裸 tensor 只有在
   人工核验顺序后才能使用 `--allow-unkeyed-context`，避免历史文件静默错位。

5. 准备 CSV manifest。格式见 `example_manifest.csv`。每一行必须包含：

   - `slide_id, patch_id, x, y, level`
   - `image_path, feature_path`
   - 训练时还需 `mask_path`
   - `feature_index` 可省略；省略后按 `patch_id` 查找

读取样本时会同时比较 `slide_id + level + x + y + patch_id`。图节点排序、坐标或图像错一项都会报错，防止模型在错误 patch 之间进行融合。

feature store 当前格式为 v2：较小的 metadata/context 保存在 `.pt`，体积最大的 DINO
dense tokens 自动写到同目录 `.dense_tokens.npy`，训练 worker 通过 memory map 只读取
当前 patch。旧 v1 单文件 store 仍可读取。

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

从 `CerviPath` 目录运行：

```bash
/path/to/python -m dinov2_segmentation.train \
  --train-manifest manifests/train.csv \
  --val-manifest manifests/val.csv \
  --output-dir outputs/segmentation \
  --num-classes 4 \
  --image-size 224 \
  --batch-size 8
```

第一阶段只训练本目录的高分辨率分支和 Decoder。DINO 与 GNN 已经离线生成缓存，因此天然冻结。默认损失为 Cross Entropy + soft Dice，mask 必须是单通道类别索引图；`255` 为 ignore index。RGB 颜色 mask 需要先显式转换为类别编号，数据集不会猜测颜色含义。

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

- `dense_tokens` 必须来自未做空间平均的 DINO patch tokens，不包含 CLS token。
- CLS token 已在 Stage1 的 node fusion 中与 global/local spatial feature 融合，再经 Stage2 进入 `global_context`，因此没有从最终分支一中丢失。
- `global_context[i]` 必须由包含节点 `i` 的完整 WSI 图产生，并与 graph node order 一致。
- 原图 patch、mask、dense token 和 GNN context 必须共享相同裁切坐标和 level。
- 训练时几何翻转会同步作用于原图、mask 和 DINO token 网格；GNN context 是向量，不需要翻转。
- 当前代码只实现单一 H/4 高分辨率流，没有采用 FmH2ST 的双图层级结构，因此不会改变既定图预训练代理任务。
