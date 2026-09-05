# 快速开始：先离线看懂和核验

以下流程不连接机器狗、不启动训练，也不运行真机入口。命令在仓库根目录执行，Windows 示例使用 PowerShell。

## 1. 下载与最小环境

安装 Python 和 Git 后：

```console
git clone https://github.com/tuoLuoSuan/go2-visual-paw-interaction.git
cd go2-visual-paw-interaction
python -m venv .venv
```

PowerShell 激活：`.\.venv\Scripts\Activate.ps1`。如果学校电脑不允许激活，不必修改系统策略，直接把下面命令中的 `python` 换成 `.\.venv\Scripts\python.exe`。Linux/macOS 使用 `source .venv/bin/activate`。

仅核对已记录的论文距离指标和做 NumPy 推理：

```console
python -m pip install numpy==2.4.6
python simulation/src/reconstruct_metrics.py data/policy_comparison/mlp --tol 1e-9
python simulation/src/reconstruct_metrics.py data/policy_comparison/gru --tol 1e-9
```

预期两次均打印 `[RECON] OK`。精确版本和历史运行环境见 [ENVIRONMENTS.md](ENVIRONMENTS.md)；不支持该版本的 Python/平台需要兼容环境，不能把换版本运行叫作原环境复现。

## 2. 验证策略能离线推理

```console
python -m unittest discover -s simulation/tests -p test_policy_runner_release.py -v
```

该组测试加载公开的趴姿 MLP `.npz`，检查 27 维输入、6 维输出及有限数值，不读取相机或发送机器人消息。合成输入只用于检查程序，不是新实验结果。

## 3. 完整离线检查

```console
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m unittest discover -s simulation/tests -v
python -m unittest discover -s vision/tests -v
python tools/release_audit.py
```

最后一条会检查公开目录并**重新生成** `MANIFEST.sha256`，不是只读验签工具。下载后应先核对已有 manifest，再运行会改写它的命令。Linux/macOS 可用 `sha256sum -c MANIFEST.sha256`；PowerShell 可用：

```powershell
Get-Content MANIFEST.sha256 | ForEach-Object {
    $expected, $relative = $_ -split '  ', 2
    if ((Get-FileHash -LiteralPath $relative -Algorithm SHA256).Hash.ToLower() -ne $expected) {
        throw "SHA-256 mismatch: $relative"
    }
}
```

这些离线检查不等于真机安全认证、重新进行正式实验或重新训练模型。

## 4. 从数据追到论文表格和图

- 模型对比：先运行步骤 1 的两条重算命令，再对照 [manuscript_distance_table.csv](../data/policy_comparison/manuscript_distance_table.csv)。先按原始精度计算，再按表格位数显示，不对已舍入的数再次做分析。
- 真机结果：先读 [CORRECTION_RECORD.md](../data/formal02/CORRECTION_RECORD.md)，再看 [标注](../data/formal02/FORMAL-02_ANNOTATION.md) 和 [结果图源数据](../figures/source_data_outcomes.csv)。trial 2 的原 JSON 有已知错误，不能直接把十个 `ok` 当作十次完整执行。
- 原有示意图与结果图的生成器是 `figures/build_figures.py`。它会覆盖 `figures/` 下同名派生图，并额外输出 PDF/TIFF；仅在自己的副本中运行 `python figures/build_figures.py`。字体不同会改变版式，不保证与最新版六页论文逐像素一致。

## 5. 哪些东西没有打包

完整训练、仿真重跑和真机部署另需外部组件：

- [Unitree SDK 2 Python](https://github.com/unitreerobotics/unitree_sdk2_python)：机器人通信。
- [Unitree MuJoCo](https://github.com/unitreerobotics/unitree_mujoco)：机器人仿真模型与场景。`unitree_robots/go2/scene.xml` 是项目历史预期位置；本包未锁定该外部资产的历史 commit，不保证当前官方主分支等价。
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)：手部检测依赖及模型说明。本项目实际使用的 task 文件版本/许可配对仍未闭环，不能随意下载一个文件就声称是实验同一版本。
- 相机标定文件、原始视频和完整训练起点不在公开包内。

第三方组件遵守各自许可；详见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。`requirements-training.txt` 和 `requirements-vision.txt` 分别面向训练参考与视觉环境，不应不加区分地装成一套环境。

当前 `simulation/src/` 中 v4 训练/评测入口不是论文历史趴姿比较的精确冻结入口，不能直接替代后声称“重现了原实验”。先完成上述无需外部资产的核验，再决定是否研究仿真或硬件复现。
