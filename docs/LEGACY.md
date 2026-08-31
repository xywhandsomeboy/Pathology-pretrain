# 历史代码边界

历史目录没有删除，因为其中包含实验依据和可能仍需读取的 checkpoint。它们不参与当前
主线 import，也不作为新命令入口。

| 目录/文件 | 状态 | 原因 |
|---|---|---|
| `legacy/dinov2_finetune/` | 只读参考 | 早期在线动态建图与边分类任务，不是当前 Stage 1 |
| `legacy/dinov2_stage1_Extract2s2_local/` | 只读参考 | 96×96 tile 坐标为 patch 内 `(y,x)`，不是 WSI `(x,y)` |
| `legacy/dinov2_stage2_2_FmH2ST_finetune/` | 只读参考 | 面向旧分类任务，不是当前 segmentation decoder |
| `graph_build.ipynb` | 实验记录 | 包含多套重复函数和旧绝对路径，已由 `build_graphs.py` 取代 |
| 旧 `.npz`/`.npy` | Stage 2 兼容 | 没有完整身份字段，不能单独供分割对齐 |
| 旧 PyG `.pt` | 旧 Stage 2 兼容 | `Graph-1024-251111-thre0.8` 的边是单通道，不能用于当前双通道 Stage 2，也不能直接建分割缓存 |
| 旧 Stage 2 checkpoint | 只读参考 | 由单通道边和旧代理任务训练，不能加载到当前双通道多任务模型 |

由于每个 fork 都提供顶层包 `dinov2`，不要把多个 fork 同时加入 `PYTHONPATH`。运行
Stage 1/Stage 2 时进入对应目录；运行 `dinov2_segmentation` 时回到仓库根目录。

旧脚本中的 `/home/li_yu/...` 路径仅存在于历史目录或历史工具中。当前主线脚本不依赖
这些路径。
