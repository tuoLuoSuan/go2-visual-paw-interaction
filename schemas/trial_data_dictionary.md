# P0-2 数据字典 + v3→v4 迁移说明（阶段 A readiness）

## 字段字典（v4，完整规则见 P0-2_trial_schema_v4.md 与 catalog JSON）
- 时间字段：`*_wall_ms` = epoch 毫秒（`wall_clock_source` 注明来源，默认 system-utc）；
  `*_mono_ms` = `time.monotonic()` 毫秒，仅用于时长/单调性，不与墙钟换算。
- 坐标系/单位：关节角 rad；tau N·m；温度 °C；px/py 归一化 0~1（画面坐标系，
  视觉 schema v2 单独定义）。
- 缺失语义：未测量字段写 `not_measured`（string）；不可用数字写 `null`；
  禁止用 0/""/placeholder 冒充缺失。
- 独立单位：trial 为阶段 C 主键；训练 run/seed 为仿真比较独立单位；episode/
  时间步只是 run 内观测（P1-1 冻结后生效）。
- raw/derived：trial JSON 为 raw 记录；summary/table 必须可由
  `tools/reconstruct_summary.py`（P1-4）从 raw 重建并 SHA 校验。

## v3 → v4 迁移说明
| v3 字段 | v4 对应 | 迁移规则 |
|---|---|---|
| outcome | execution_status | ok/aborted 原样映射；v3 无 not_run |
| endpoints.reach_success / handshake_success | 同名 | v3 全部 not_measured → 原样保留，不重写 |
| contact.detector_trigger_count | 同名 | 保留；contact_ground_truth 仍 not_measured |
| contact（无 contact_hold_s） | contact.contact_hold_s | 旧文件缺失 → 版本化读取器只读报告缺失，不回填 |
| stages（结束统一回填） | stages 记录点 | 旧文件不改写；读取器标记 "legacy-backfill" 注释 |
| session（commit【待补】/placeholder/空 trial_index） | identity.* | 旧文件不改写；读取器报告缺失字段 |
| run_id | trial_id | 保留原值；正式 trial 将由 --session-id/--trial-index 生成 |
| 无 event_trace/clock_sync | v4 新字段 | 旧文件不新增字段；读取器在报告层记 not_measured |

旧 13 份 JSON 永不重写；`00_legacy_recompute_v4.json` 的 before/after SHA
证明读取前后一致。
