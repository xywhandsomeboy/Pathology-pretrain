# Patch 身份与中间文件契约

## 唯一身份

从原图到最终 mask，任何 patch 都使用以下五元组：

```text
(slide_id, level, x, y, patch_id)
```

- `slide_id`：一张 WSI 的稳定名称。
- `level`：金字塔层级，未使用多层级时为 0。
- `x/y`：该 level 坐标系中的 patch 左上角像素位置，顺序固定为 `(x, y)`。
- `patch_id`：slide 内唯一字符串，推荐使用原 patch 文件 stem。

数组、图节点和 context 的第 `i` 项必须始终代表同一个五元组。只依赖数组下标而不保存
身份信息的旧文件只能用于旧 Stage 2 预训练，不能直接进入新分割 decoder。

## Stage 1 decoder shard

`decoder_features_s1_part*.pt` 是字典：

| 字段 | 形状/类型 | 含义 |
|---|---|---|
| `format_version` | `int` | 当前为 1 |
| `filenames` | `list[str]` | 原 patch 文件名 |
| `patch_ids` | `list[str]` | 与特征同序 |
| `node_features` | `[N,C]` | 用于建图 |
| `dense_tokens` | `[N,T,C]` | 未做空间平均的 DINO token |
| `slide_ids` | 可选 `list[str]` | slide 数据集可直接提供 |
| `coords` | 可选 `[N,2]` | `(x,y)` |
| `levels` | 可选 `[N]` | 金字塔 level |

## Stage 2 graph

新图必须包含 `x [N,C]`、`edge_index [2,E]`、`edge_attr [E,2]` 和 `pos [N,2]`。
用于分割时还必须包含 `patch_ids`、`levels` 与 `slide_id`。`dense_tokens` 不写进图，
而是由 Stage1 独立保存，防止图训练与 context 导出读取巨大的 decoder 张量。

训练数据集仅保留图训练张量；patch 身份只用于完整图 context 导出，不进入 PyG batch。

## Slide feature store

`dinov2_segmentation` 的单 slide 缓存包含：

- `format_version, slide_id, patch_ids, coords, levels`
- `dense_tokens_file`：外部 `.npy` sidecar，形状 `[N,T,C]`，DataLoader 以只读
  memory map 按 patch 访问，不整张载入内存
- `global_context [N,Cg]`
- 可选 `node_features [N,C]`

加载 manifest 时会重新比较完整五元组，任意坐标、level、slide 或 patch ID 不一致都会
报错，不允许静默融合错误 patch。

## 分割 manifest

必须列：`slide_id,patch_id,x,y,image_path,feature_path`。推荐显式提供 `level` 与
`feature_index`。训练/验证还需要 `mask_path`。

路径可以是绝对路径，也可以相对于 manifest 所在目录。mask 必须是单通道类别索引，
RGB 颜色 mask 需提前转换，`255` 为默认 ignore index。
