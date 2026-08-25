# Third-party components

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

当前 Private 审阅包没有软件许可证文件，也没有把软件许可证自动套用于 JSON、图像或视频。转为 Public 前必须由作者和导师分别确认代码、数据和图像的授权方式。
