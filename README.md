# GO2 Visual Paw Interaction

这是一个面向 Unitree GO2 EDU 的视觉引导前足人机交互研究原型。系统使用机器人自带前置单目 RGB 相机检测用户手部，在受约束的固定交互区域内判断手位于画面左侧或右侧，随后由趴姿控制器选择对应前足、执行接近、接触保持和安全退出。

本仓库是论文审阅用的 **Private GitHub 研究包**，不是成熟产品，也没有授予公开复用许可证。

## 研究范围

- 硬件：Unitree GO2 EDU，自带前置单目 RGB 相机。
- 感知：MediaPipe 手部检测、画面侧别判断、目标滤波与丢失处理。
- 执行：趴姿前足交互为主线；站姿交互仅作为独立的定性扩展。
- 场景：单名操作者、室内木地板、预先限定的固定交互区域。
- 边界：二维像素位置不是通用的机器人三维坐标。本项目不主张在任意三维位置完成视觉伺服握手。

## 系统结构

```text
GO2 单目图像
  -> 手部检测与画面区域门控
  -> 侧别信号与时间滤波
  -> 行为状态机
  -> 前足轨迹 / 策略增量 / VMC 稳定约束
  -> 接触事件、保持阶段与安全退出
```

## 真机证据口径

FORMAL-02 包含 10 次连续、被命令执行的趴姿真机尝试：

- 独立视频观察接触：10/10。
- 独立视频观察保持不少于 0.6 s：10/10。
- 无执行中止地完成：9/10。
- 执行中止：1/10，FORMAL-02-002 在观察到接触后出现 `TRACKING_ERROR`。
- 安全退出组成完成：10/10。
- 选足正确性：未测量；侧视机位不能可靠映射画面左右与机器人左右。

原始 trial 002 JSON 因当时的记录器缺陷保留了错误的 `execution_status=ok`。仓库不改写该原始记录，而是通过同目录下的 `CORRECTION_RECORD.md` 和逐次运行日志确定派生执行状态。

FORMAL-03 单独记录站姿扩展：三个录制段均得到执行层 `STAND_HS_OK`，但没有与 FORMAL-02 等价的独立端点标注，因此不与趴姿结果合并，也不表述为经过独立确认的 3/3 握手成功率。

## 目录

```text
real_go2/                 真机入口、策略推理、接触检测和 trial 记录
vision/                   单目视觉与几何辅助模块
simulation/               MuJoCo 训练、评测和控制辅助模块
models/                   趴姿主模型与站姿扩展模型
data/formal02/            十次趴姿 JSON、日志、标注、纠错和图源数据
data/formal03_standing/   站姿扩展记录，和主实验严格分开
schemas/                  trial schema v4 与数据字典
figures/                  论文图、图源数据和照片溯源
tools/                    schema 与发布包审计工具
docs/                     实验范围、环境、来源和可用性说明
```

## 离线检查

以下命令不连接机器人，也不发布控制消息：

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s simulation/tests -v
python -m unittest discover -s vision/tests -v
python tools/release_audit.py
```

真实机器人入口只建议先做源码审查、`--help` 和明确的 dry-run。自动验收不会执行任何联网控制或运动命令。

## 安全警告

真机低层控制可能导致突然运动、跌倒、夹伤、过热或硬件损坏。任何真机复现都必须由具备相应经验的人员负责，并至少满足：遥控器和急停可用、周围净空、机器人温度和姿态门限有效、操作者能够随时终止，以及先完成无运动检查。仓库作者不把研究代码描述为经过产品级安全认证的控制软件。

## 复现边界

本仓库没有捆绑 Unitree SDK、CycloneDDS、Unitree MuJoCo 资产或 `hand_landmarker.task`。相机内参工件在 Ubuntu VM 中曾按 ID 和哈希核验，但未进入 Windows 最终归档，因此这里不会声称标定文件已经随仓库提供。详见 `docs/ENVIRONMENTS.md`、`docs/DATA_AVAILABILITY.md` 和 `THIRD_PARTY_NOTICES.md`。

## English summary

This private research package contains the paper-aligned code, deployment models, compact real-robot evidence and figure sources for a Unitree GO2 EDU monocular-RGB paw-interaction prototype. The main experiment comprises ten commanded prone trials: observer-rated contact and a hold of at least 0.6 s were reported in 10/10 trial segments, while clean execution completed in 9/10 attempts and one attempt aborted with a tracking error after observed contact. Correct paw selection was not measurable from the lateral recording view. The three standing recordings are a separate qualitative extension. The system is restricted to a predefined interaction zone and does not claim general monocular 3D localization.
