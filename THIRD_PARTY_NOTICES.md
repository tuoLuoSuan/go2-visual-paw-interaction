# Third-party components

官方获取入口：[Unitree SDK 2 Python](https://github.com/unitreerobotics/unitree_sdk2_python)、[Unitree MuJoCo](https://github.com/unitreerobotics/unitree_mujoco)、[CycloneDDS](https://github.com/eclipse-cyclonedds/cyclonedds)、[MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)。这些链接是获取/文档入口，不代表已核实当前下载版本与历史实验完全相同；未锁定的历史 commit 或模型许可仍按下表处理。

本仓库只保存作者项目代码、部署权重和小型证据副本，不捆绑下列第三方源码树或运行时资产。使用者应从官方来源获取相应组件，并遵守其当前许可证和硬件使用条件。

| 组件 | 本研究记录的版本/状态 | 是否捆绑 | 发布边界 |
|---|---|---:|---|
| Unitree SDK 2 Python | VM 记录为 1.0.1 | 否 | 从 Unitree 官方仓库获取；真机控制需单独完成网络与安全配置 |
| CycloneDDS | VM 已安装，精确版本未在最终查询中取得 | 否 | 由 Unitree SDK 运行环境提供；本仓库不复制系统库 |
| Unitree MuJoCo robot assets | 研究环境中使用 | 否 | 场景和机器人资产从其官方来源获取；本仓库不复制完整第三方树 |
| MediaPipe | 0.10.35 | 否 | Python 依赖由使用者安装 |
| `hand_landmarker.task` | VM 文件哈希已记录，但准确下载来源与模型许可证配对未在 Windows 归档中闭环 | 否 | 在来源与许可核验完成前禁止随本仓库分发 |
| OpenCV contrib | 5.0.0.93 | 否 | Python 依赖由使用者安装 |
| MuJoCo Python | 3.11.0 | 否 | Python 依赖由使用者安装 |
| PyTorch | 2.13.0+cpu | 否 | 仅训练/参考评测需要；NumPy 部署模型不要求加载 PyTorch checkpoint |
| NumPy | Windows 2.4.6；VM 2.2.6 | 否 | 两套环境分别锁定，不应强行合并为一个运行环境 |
| Matplotlib | 3.11.1 | 否 | 用于图表重建 |

当前公开研究包没有软件、结构化数据或图像的授权许可证文件。仓库公开可见仅用于阅读与学术核验，不会把第三方许可证自动套用于作者代码、JSON、图像或视频；如需允许复制、修改或再分发，仍应由作者分别确定并补充相应授权方式。
