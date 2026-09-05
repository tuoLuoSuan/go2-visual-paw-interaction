# Data and code availability status

本文件描述当前 **公开 GitHub 研究包** 的实际状态。该仓库可供读者浏览和核验，但尚不等同于具有 DOI 与明确数据许可证的正式归档数据集。

| 材料 | 当前位置与访问状态 |
|---|---|
| FORMAL-02 十份 schema v4 JSON | 已包含于 `data/formal02/json/` |
| FORMAL-02 十份运行日志 | 已包含于 `data/formal02/logs/` |
| FORMAL-02 独立观察标注与纠错记录 | 已包含于 `data/formal02/` |
| 图 4 源数据和重建脚本 | 已包含于 `data/formal02/source_data_outcomes.csv` 与 `figures/build_figures.py` |
| FORMAL-03 结构化记录 | 已脱敏后包含于 `data/formal03_standing/`，并与主实验分开 |
| 趴姿与站姿部署模型 | 已包含于 `models/`，模型卡记录 SHA-256 |
| 论文趴姿 MLP/GRU 同任务对比 | 已包含于 `data/policy_comparison/`；两个原始 `.pt` 权重与训练配置位于 `models/prone_comparison/` |
| 对比指标重建脚本 | `simulation/src/reconstruct_metrics.py`，复算距离与存活指标；不复算未随逐步数据保存的关节目标范围 |
| 原始视频 | 保留于私人研究归档，本次不上传 GitHub |
| 相机内参工件 | Windows 最终归档中不存在实体文件；仅记录 ID `intrinsics_calib_20260824_accepted`、SHA-256 `b5b23de64a0dd16f69097f27508a6d595901cb58a8a6f0893ed4c549d5443d48`、RMS 1.5719145083 px，并标为作者接受的偏差 |
| MediaPipe task 模型 | 不包含；准确来源与许可配对尚未闭环 |
| 稳定公开标识符 | 尚未分配；当前没有创建 Zenodo、OSF 或机构仓库 DOI |

论文已获 CCICS 2026 录用；代码链接、数据仓库标识符、许可和视频访问方式仍须与最终出版声明保持一致。当前 GitHub 仓库已经公开，但没有 DOI 或开放许可证，且不包含原始视频、第三方 MediaPipe task 模型和相机内参实体文件，因此不能描述为完整的公共数据集或从零复现包。当前元数据见 [PAPER.md](PAPER.md)，本次材料更新见 [v0.2.0](RELEASE_NOTES_v0.2.0.md)。

## English availability summary

Code, deployment models, structured prone-trial records, observer annotations, correction records and a separate standing extension are publicly accessible in this repository. The archived prone MLP/GRU comparison is provided in `data/policy_comparison/`, with original checkpoint files in `models/prone_comparison/` and a distance/survival reconstruction script in `simulation/src/reconstruct_metrics.py`. Raw videos and participant/administrative records are retained privately and are not included in this release; no automatic access entitlement or request-service commitment is made. Camera calibration, third-party hand-model assets and complete training lineage are not provided. No archival DOI or open licence has been assigned.
