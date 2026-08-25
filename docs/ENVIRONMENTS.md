# 运行环境记录

本项目使用两套用途不同的环境。版本来自最终归档中的实测环境锁；私人用户名、IP、网卡名、VM 路径和第三方本地目录已从本说明中移除。

## Windows 训练与离线评测

| 组件 | 版本 |
|---|---|
| Python | 3.13.13 |
| NumPy | 2.4.6 |
| PyTorch | 2.13.0+cpu |
| MuJoCo | 3.11.0 |
| Matplotlib | 3.11.1 |

## Ubuntu 真机控制与视觉

| 组件 | 版本/状态 |
|---|---|
| Python | 3.10.12 |
| MediaPipe | 0.10.35 |
| NumPy | 2.2.6 |
| OpenCV contrib | 5.0.0.93 |
| Unitree SDK 2 Python | 1.0.1 |
| CycloneDDS | 已安装；最终记录没有取得精确版本号 |

两套环境承担不同职责。Windows 版本用于训练、评测、图表和发布包 QA；Ubuntu 版本用于 GO2 图像获取与真机执行。版本差异不应被误写成同一环境的依赖冲突。
