# V1：当前 Decoder

状态：已实现，保持不变。

代码入口：

- `dinov2_segmentation/models/model.py`
- `dinov2_segmentation/models/fusion_decoder.py`
- `dinov2_segmentation/models/detail_encoder.py`
- `dinov2_segmentation/models/semantic_encoder.py`

当前结构是双分支分割网络：DINO dense tokens 与整张 WSI 的 GNN context 构成全局语义
分支，原图 patch 构成高分辨率细节分支。拼接投影后先使用显式 Channel Attention
学习跨通道权重，再由 HRFormer/ConvNeXtV2 交替模块完成融合与上采样。

本目录只记录版本身份，不保存代码副本，避免同一实现出现两份后逐渐不一致。
