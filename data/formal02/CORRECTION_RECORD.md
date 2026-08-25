# 记录更正说明（CORRECTION RECORD，2026-08-25）

## 缺陷
FORMAL-02 录制期间，real_vmc_reach_m8_v11.py 的记录构造把
execution_status 硬编码为 "ok"（v11 补丁器缺陷），导致 trial 2 的 JSON
execution_status=ok 与其日志不符。

## 事实（以逐次运行日志为准，日志已入库）
- trial 2（FORMAL-02-002）实际结果：**M8_ABORTED reason=TRACKING_ERROR**
  （日志：remediation/formal_trials/FORMAL-02/formal02_2.log，
  日志含 `[RESULT] M8_ABORTED reason=TRACKING_ERROR`）
- 其余 9 次日志均为 M8_OK，与 JSON 一致

## 更正后分母（正式统计口径）
- 趴姿 10 次尝试：**9 ok + 1 aborted（trial 2, TRACKING_ERROR）**，全部计入分母
- 盲法标注独立判定不变：10/10 接触、10/10 保持 ≥0.6s
  （标注不依赖 execution_status；trial 2 的接触观察亦成立）

## 处置
- 已生成 JSON **不改写**（保持原始证据）；本文件作为更正记录与 JSON 并列入档
- v11 补丁器已修复（execution_status 取自 ctl.abort_reason），
  未来运行记录正确；修复后脚本重新生成/部署
- 本记录 SHA 与 JSON SHA 一并进入 FINAL_MANIFEST
