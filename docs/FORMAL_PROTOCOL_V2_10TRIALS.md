# 正式实验协议 v2（10 次趴姿握手，单视频连续录制）

> 依据 GPT 2026-08-24 定稿。论文范围：**单名操作者、室内木地板、固定交互
> 区域下，GO2 EDU 视觉引导趴姿握手系统的工程可行性验证**（pilot/demo 级，
> 不声称统计显著优于）。

## 0. 前置（全部 fail-closed）
- 狗趴姿起始、温度 <55°C、遥控急停在手、净空 1m、木地板防滑
- consent：consent-20260824-01（已登记，覆盖本 session）
- 固定交互区域：手在画面中心 ±15°、30~50cm、趴姿高度 10~25cm
- 三脚架固定拍摄：**1080p 30fps**，画面必须同时包含整只狗、手、支撑脚；
  一个连续视频文件，中途不停机

## 1. 十次序列（预注册，顺序不打乱重录）
| 次 | 手侧 | 次 | 手侧 |
|---|---|---|---|
| 1 | L | 6 | L |
| 2 | R | 7 | L |
| 3 | R | 8 | R |
| 4 | L | 9 | R |
| 5 | R | 10 | L |
（L/R = 手伸在狗左/右侧；5L+5R）

## 2. 每次流程
1. 操作者对着镜头报"第 N 次"
2. 确认狗为相同趴姿起始（每次之间用遥控恢复官方趴姿）
3. 手放入指定侧固定区域，保持至自动结束（或按终止规则）
4. 程序自动：站起→趴下→低层→伸直→追手→手离开 3s 收回
5. 成功/失败/中止**全部保留**，JSON 逐次落盘（trial-index=N）
6. 两次之间停 5~10 秒

## 3. 命令（每次一条，N=1..10）
```bash
nohup .venv-go2-vision/bin/python3 real_go2/real_vmc_reach_m8_v11.py \
  --network-interface YOUR_NETWORK_INTERFACE --site-check --confirm GO2-M8-20260818 \
  --hand-port 4300 --kp 120 --kd 10 --hold-kp 40 --hold-kd 4 --tau-clip 30 \
  --reach-seconds 4 --settle-seconds 2 --track-slew-fast-rad 0.012 \
  --slew-switch-rad 0.15 --predict-latency 0.10 --predict-max-lost 0.25 \
  --contact-hold-s 0.6 --policy real_go2/best_mlp_prone_px_v4.npz \
  --max-start-temp 70 --hold-seconds 90 --hand-lost-frames 1500 \
  --session-id FORMAL-02 --trial-index N --floor wood-indoor --light indoor \
  --calibration-id intrinsics_calib_20260824_accepted --commit <HEAD> \
  --trial-log-dir evidence/formal_trials > /tmp/formal_N.log 2>&1 &
```

## 4. 标注（1 名必需，第 2 名加分）
- 找与项目无关的人，盲法标注 10 次（ANNOTATION_KIT.md 的 4 问）
- 视频：一个连续文件（含报数声）；SHA 登记 lineage
- 现有 实验视频.mp4（FORMAL-01）降级为**预实验展示**，不进正式分母

## 5. 分母规则
10 次全部计入；无手检出/中止/失败不删除；endpoint 按独立标注判定；
不声称显著优于任何基线（本实验无对照组，仅可行性验证）。

## 6. 站立姿态握手（附加实验，按 GPT 2026-08-24 定位）
- 定位：跨姿态扩展示范/系统通用性验证；**数据与趴姿严格分开**，不得合并成
  13 次成功率
- 完成度对应写法：
  - 1 次成功 → 补充视频/案例展示，不做定量结论
  - 连续 3 次成功（视频+JSON 齐）→ 初步扩展实验
  - 10 次 → 单独统计的第二组结果（本论文目标不要求）
- 最快路径：明天连续录 3 次站立成功（视频+JSON），不追求 10 次
- 未按时完成 → 不拖延趴姿主线，直接写进 Discussion 未来工作
- 站立安全红线不变：预置卸重→预抬爪→追手；姿态守卫 0.30；温度门；逐次签核
