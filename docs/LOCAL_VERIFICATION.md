# Local release verification

- 验证日期：2026-08-25
- 私人源仓库实现起点：`6ab587061e6bb778e64556a0c581c8157997651a`
- 已批准发布设计提交：`ba9010459892b118f8d42fac08f24b25b43c5ab4`
- 发布目录：独立 Git 仓库 `go2-visual-paw-interaction`

## 自动检查

| 检查组 | 发现 | 通过 | 失败 |
|---|---:|---:|---:|
| 发布审计与正式证据 | 11 | 11 | 0 |
| 控制、策略与部署推理 | 30 | 30 | 0 |
| 视觉、内参与几何辅助 | 42 | 42 | 0 |
| 合计 | 83 | 83 | 0 |

执行命令：

```powershell
D:\Application\Python\python.exe -m unittest discover -s tests -v
D:\Application\Python\python.exe -m unittest discover -s simulation/tests -v
D:\Application\Python\python.exe -m unittest discover -s vision/tests -v
D:\Application\Python\python.exe tools/release_audit.py
```

发布审计输出：`RELEASE_AUDIT_OK`。

## 证据与哈希

- FORMAL-02 JSON：10 份；与 `FINAL_MANIFEST_20260825.json` 逐文件 SHA-256 一致。
- FORMAL-02 运行日志：10 份。
- 趴姿模型 SHA-256：`9de29f01893534b20cd395de82d3d6096a41a1c17d0db1b43d586b59a00f7958`。
- 站姿扩展模型 SHA-256：`5a6cdb1a2bc86a99cb223196d9eac4ebf748e07aef1e32f08f3ca4cebdb822d5`。
- FORMAL-03 四份发布副本均只脱敏 `params.network_interface`；原始和发布哈希已机器复核并登记。
- 原始视频、私钥/连接目录、同意书、MediaPipe task 模型和第三方大目录：0 个进入发布包。

## 已确认的代码边界

FORMAL-02 使用 `side-only` 模式，不读取相机外参。发布 QA 中发现父项目的一个相机外参测试把 `Rz(yaw) @ Ry(pitch) @ Rx(roll)` 的 `pitch=90°` 结果写反；发布副本仅修正该测试的数学说明和期望值，`hand_to_fr_target.py` 运行逻辑未改变。一般三维相机到机身映射仍被列为未随本研究验证的能力。

## 转为 Public 前仍需解决

1. 作者公开显示名称与最终引用信息。
2. 代码、结构化数据和真实实验图片的分别授权方案。
3. 接受偏差的相机内参实体文件取回与归档。
4. `hand_landmarker.task` 的准确下载来源、版本和许可闭环。
5. 面向投稿读者的稳定数据仓库标识符或经过期刊认可的访问路线。
