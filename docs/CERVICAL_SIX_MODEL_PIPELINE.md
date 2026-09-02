# 宫颈 WSI 六模型自动流程

入口是 `scripts/run_cervical_six_models.sh`。它由一个持久化 `tmux` 会话执行，
客户端或 SSH 断开不会终止任务。

## 触发条件

触发器不看整个下载目录的字节百分比，而按可训练切片计数。一张 WSI 只有在下载
manifest 中的 `.isyntax` 与 `.geojson` 都通过精确文件大小校验、出现在官方 xlsx
划分中且未被排除时，才算 ready。默认至少要求：

- 总计 32 张；
- 官方 train 至少 16 张、valid 至少 8 张；
- cohort 同时含正常与肿瘤病例。

达到阈值后立即冻结当时完整的 cohort；后续下载的新切片不会在同一轮实验中改变
训练集，避免六个模型看到不同数据。

## 二分类标签契约

GeoJSON 是精细的 level-0 多边形标注。所有病变等级严格合并为一个肿瘤类：

| 原始标注 | 训练类别 |
| --- | --- |
| 背景、正常/炎症 | `0` |
| `Low Grade` / `Low grade` | `1` |
| `High Grade` / `High grade` | `1` |
| `Malignant` | `1` |

转换会统一大小写和空格，但遇到任何其他非空标注会直接停止。`_mask.png` 只用于
快速组织区域筛选；训练 mask 始终由 GeoJSON 在目标 level 重新栅格化，因此不会
把绿色预览像素泄漏进模型输入。

默认从 iSyntax level 1 读取 224×224 patch，步长 224，保留组织占比至少 20% 的
patch，并无条件保留与病变多边形相交的 patch。坐标 `(x,y)` 始终处于 level 1
坐标系，原图、二分类 mask、Stage1 dense token、Stage2 节点和 Decoder manifest
共享同一个 `patch_id`。

## 执行顺序

1. CPU 切图并生成二分类 mask、固定官方 train/valid/test 划分；
2. 等待当前正在运行的 `baseline` 与 `distance_only` Stage2 正常完成；
3. 在空闲 GPU 上训练 `weighted_pretrain_distance_context`；
4. 用已选定的 Stage1 checkpoint 为新 cohort 流式提取节点特征和 dense token；
5. 分别建立 dual-edge 与 distance-only 完整 WSI 图；
6. 导出三个 Stage2 版本的完整 WSI context；
7. 严格按 patch 身份生成三个版本的 feature store 与相同数据选择的 manifest；
8. 分三轮训练，每轮在两张 GPU 上并行训练 Decoder V1 和 V2。

最终六个模型为：

| Stage2 context | Decoder |
| --- | --- |
| `baseline` | V1、V2 |
| `distance_only` | V1、V2 |
| `weighted_pretrain_distance_context` | V1、V2 |

训练产物默认写入 `Data/cervical_segmentation_partial/`，该目录由 `.gitignore`
排除。状态日志为 `Data/cervical_segmentation_partial/logs/pipeline.log`，全部完成后
生成 `Data/cervical_segmentation_partial/six_models.complete`。
