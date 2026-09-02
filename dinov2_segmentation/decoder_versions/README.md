# Decoder 版本目录

这个目录只用于隔离分割 Decoder 的不同设计，不参与 Stage1 或 Stage2 预训练。

## 版本位置

- `v1_current/`：记录当前已实现版本。实际代码仍保留在
  `dinov2_segmentation/models/`，不移动、不复制，避免破坏现有导入路径和 checkpoint。
- `v2_new/`：第二版 Decoder 的设计与使用说明。V2 使用独立的模型、训练和推理
  文件，不接入也不改变 V1 的 `train.py` 或 `infer.py`。

## 隔离原则

1. Decoder 版本开发不得修改 `dinov2_stage2_2_FmH2ST/` 下的训练配置、图数据或运行脚本。
2. V2 完成并测试前，不修改当前 `GlobalLocalSegmentationModel` 的默认行为。
3. 两个 Decoder 共用相同的 Stage1/Stage2 初始化权重、图拓扑、原图 patch 和 mask；最终
   dense tokens 与 GNN context 均在线生成并联合微调，便于公平比较。
4. 两个版本使用独立输出目录和 checkpoint，禁止相互续训或覆盖。
