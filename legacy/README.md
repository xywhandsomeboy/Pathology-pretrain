# Legacy experiments

这里集中保存不再参与当前 CerviPath 主线的历史实验 fork：

- `dinov2_finetune/`：早期 DINO + 在线动态图任务。
- `dinov2_stage1_Extract2s2_local/`：旧 local tile 特征实验。
- `dinov2_stage2_2_FmH2ST_finetune/`：旧分类微调实验。

这些目录用于复现和代码对照，不应加入当前主线的 `PYTHONPATH`，也不应从其中启动
Stage1、Stage2 或 segmentation。历史文件中的绝对路径和旧配置按原样保留。

当前入口和数据契约请返回仓库根目录阅读 `README.md` 与 `docs/PIPELINE.md`。
