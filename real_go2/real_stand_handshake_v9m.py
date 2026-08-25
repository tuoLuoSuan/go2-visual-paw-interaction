#!/usr/bin/env python3
"""真机站立握手：低层站立 + standing_px RL 策略追手（12 关节驱动）。

前置安全门（已过）：real_stand_probe.py kp=100 站 15s 零升温。
本脚本在此基础上接视觉+策略：官方站起 -> 接管 -> 纯PD站稳 3s ->
策略 50Hz 追手 -> 手离开 3s -> 斜坡降趴 -> 恢复 mcf -> StandDown。

温度纪律与 real_stand_probe.py 相同：80°C 硬限 + 5s/3°C 速率门 + 漂移 0.15。

用法（VM）：
  .venv-go2-vision/bin/python3 real_go2/real_stand_handshake.py \
    --network-interface YOUR_NETWORK_INTERFACE --confirm GO2-STAND-HS-20260822 \
    --policy real_go2/best_mlp_standing_px_v2.npz \
    --hand-port 4300 --kp 100 --active-kp 120 --hold-seconds 90
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "real_go2"))

import numpy as np

from contact_detector import ContactSpikeDetector
from hand_predictor import HandPredictor
from policy_runner import PolicyRunner
from trial_record import build_trial_record, write_trial_record

TEMP_HARD_LIMIT = 80.0
TEMP_RISE_LIMIT = 3.0
TEMP_RISE_WINDOW = 5.0
CONFIRM_PASSPHRASE = "GO2-STAND-HS-20260822"
LIE_Q = [-0.06, 1.24, -2.75, 0.08, 1.25, -2.78,
         -0.40, 1.26, -2.77, 0.41, 1.25, -2.78]
STAND_LIMITS = {0: (-0.5, 0.5), 1: (-0.6, 1.2), 2: (-2.2, -0.8)}
FR_IDX = (0, 1, 2)
FL_IDX = (3, 4, 5)


def quat_to_pitch_roll(qq):
    w, x, y, z = (float(v) for v in qq)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return pitch, roll


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-interface", required=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--site-check", action="store_true")
    parser.add_argument("--policy", required=False, default="",
                        help="standing_px 策略 npz 路径（必填用于追手）")
    parser.add_argument("--hand-port", type=int, default=4300)
    parser.add_argument("--kp", type=float, default=100.0)
    parser.add_argument("--kd", type=float, default=4.0)
    parser.add_argument("--active-kp", type=float, default=120.0)
    parser.add_argument("--active-kd", type=float, default=8.0)
    parser.add_argument("--hold-seconds", type=float, default=90.0)
    parser.add_argument("--hand-lost-frames", type=int, default=1500)
    parser.add_argument("--predict-latency", type=float, default=0.10)
    parser.add_argument("--py-bias", type=float, default=0.0,
                        help="喂给策略的 py 垂直补偿（相机几何差；负值=手读得更高）")
    parser.add_argument("--debug-joints", action="store_true")
    parser.add_argument("--max-start-temp", type=float, default=70.0)
    parser.add_argument("--trial-log-dir", default="evidence/real_trials")
    args = parser.parse_args(argv)

    if args.dry_run:
        print(f"[DRY-RUN] 站立握手：StandUp -> 接管 -> 站稳3s -> 策略追手"
              f"(kp={args.kp}/{args.active_kp}) -> 手离开3s -> 降趴 -> 恢复")
        print(f"[DRY-RUN] 温度门 起始{args.max_start_temp:.0f}C / "
              f"硬限{TEMP_HARD_LIMIT:.0f}C / 5s+{TEMP_RISE_LIMIT}C 中止")
        return 0
    if args.site_check:
        checks = [("狗当前站立且四周1m无障碍", "请清场"),
                  ("遥控器在手", "请拿好遥控器")]
        for label, tip in checks:
            ok = input(f"  [检查] {label}? [y/N] ").strip().lower() == "y"
            print(f"  [检查] {label}: {'通过' if ok else '未通过'}")
            if not ok:
                print(f"[ABORT] {tip}", file=sys.stderr)
                return 4
    if args.confirm != CONFIRM_PASSPHRASE:
        print("[ABORT] 确认口令错误", file=sys.stderr)
        return 5
    if not args.policy:
        print("[ABORT] --policy 必填", file=sys.stderr)
        return 5

    from datetime import datetime as _dt
    session_start = _dt.now().isoformat()   # 会话真实开始时间（trial 证据）

    cyclonedds_home = WORKSPACE / "cyclonedds" / "install"
    if cyclonedds_home.exists():
        os.environ.setdefault("CYCLONEDDS_HOME", str(cyclonedds_home))
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber)
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
        MotionSwitcherClient)
    from unitree_sdk2py.go2.sport.sport_client import SportClient

    ChannelFactoryInitialize(0, args.network_interface)
    crc = CRC()
    shared = {"msg": None}
    lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
    lowstate_sub.Init(lambda m: shared.__setitem__("msg", m), 10)
    deadline = time.monotonic() + 5.0
    while shared["msg"] is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if shared["msg"] is None:
        print("[ABORT] 5s 内未收到 lowstate", file=sys.stderr)
        return 6
    temps = [float(shared["msg"].motor_state[i].temperature)
             for i in range(12)]
    print(f"[INFO] 起始电机温度: max={max(temps, default=0):.0f} C")
    if max(temps, default=0) > args.max_start_temp:
        print(f"[ABORT] 起始温度超上限 {args.max_start_temp:.0f}C",
              file=sys.stderr)
        return 8

    sport = SportClient()
    sport.SetTimeout(10)
    sport.Init()
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()

    print("[INFO] StandUp() 官方站立 ...")
    print("  code:", sport.StandUp())
    time.sleep(3.0)

    released = False
    for attempt in range(10):
        try:
            _, result = msc.CheckMode()
        except Exception:
            result = None
        if not (result or {}).get("name", ""):
            released = True
            break
        msc.ReleaseMode()
        time.sleep(0.2)
    if not released:
        print("[ABORT] 无法释放模式", file=sys.stderr)
        return 7

    policy = PolicyRunner(args.policy)
    if policy.obs_dim != 29 or policy.act_dim != 12:
        print(f"[ABORT] 策略维度不符: obs={policy.obs_dim} "
              f"act={policy.act_dim}（期望 29/12 standing_px）",
              file=sys.stderr)
        return 9
    # policy 文件 SHA-256（trial 证据；正式实验前预计算，不临时生成）
    import hashlib as _hashlib
    policy_sha = _hashlib.sha256()
    with open(args.policy, "rb") as _pf:
        for _chunk in iter(lambda: _pf.read(65536), b""):
            policy_sha.update(_chunk)
    policy_sha256 = policy_sha.hexdigest()
    print(f"[INFO] 策略已加载: {args.policy}（{policy.obs_dim}->"
          f"{policy.act_dim}，50Hz，sha256={policy_sha256[:16]}...）")
    if hasattr(policy, "reset"):
        policy.reset()   # GRU 隐藏态清零

    # HandStream（从 M8 复用类）
    from real_vmc_reach_m8 import HandStream
    hand_stream = HandStream(args.hand_port)
    predictor = HandPredictor(latency_s=args.predict_latency,
                              max_lost_s=0.25)
    contact_det = ContactSpikeDetector(hold_s=0.6, warmup=1.0)

    publisher = None
    targets = None
    hand_frames = 0
    contact_count = 0
    aborted = False
    abort_reason = ""
    try:
        print("[INFO] 低层接管就绪，冻结站立位（纯 PD）")
        deadline = time.monotonic() + 0.5
        while shared["msg"] is None and time.monotonic() < deadline:
            time.sleep(0.002)
        targets = [float(shared["msg"].motor_state[i].q) for i in range(12)]
        print("[INFO] 冻结站立关节角:", [round(v, 3) for v in targets])

        publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        publisher.Init()
        low_cmd = unitree_go_msg_dds__LowCmd_()
        low_cmd.head[0] = 0xFE
        low_cmd.head[1] = 0xEF
        low_cmd.level_flag = 0xFF
        for i in range(20):
            low_cmd.motor_cmd[i].mode = 0x01
            low_cmd.motor_cmd[i].q = float("nan")
            low_cmd.motor_cmd[i].kp = 0
            low_cmd.motor_cmd[i].dq = float("nan")
            low_cmd.motor_cmd[i].kd = 0
            low_cmd.motor_cmd[i].tau = 0

        z0 = None
        temp_last = temps
        t_last = time.monotonic()
        max_temp = max(temps, default=0)
        max_drift = 0.0
        side = 0
        cycle = 0
        policy_interval = 10
        hand_lost_frames = 0
        last_contact_t = -9.0
        hold_until = time.monotonic() + args.hold_seconds

        def build():
            for i in range(12):
                low_cmd.motor_cmd[i].mode = 0x01
                low_cmd.motor_cmd[i].q = float(targets[i])
                low_cmd.motor_cmd[i].dq = 0.0
                kp, kd = args.kp, args.kd
                if i in (FR_IDX if side == 0 else FL_IDX):
                    kp, kd = args.active_kp, args.active_kd
                low_cmd.motor_cmd[i].kp = kp
                low_cmd.motor_cmd[i].kd = kd
                low_cmd.motor_cmd[i].tau = 0.0
            low_cmd.crc = crc.Crc(low_cmd)
            return low_cmd

        def stop():
            for i in range(20):
                low_cmd.motor_cmd[i].mode = 0x01
                low_cmd.motor_cmd[i].q = float("nan")
                low_cmd.motor_cmd[i].kp = 0
                low_cmd.motor_cmd[i].dq = float("nan")
                low_cmd.motor_cmd[i].kd = 0
                low_cmd.motor_cmd[i].tau = 0
            low_cmd.crc = crc.Crc(low_cmd)
            return low_cmd

        def abort(reason):
            nonlocal aborted, abort_reason
            aborted = True
            abort_reason = reason
            publisher.Write(stop())

        print("[INFO] 站稳窗口 5s（首 1s 豁免瞬态，之后门限 0.15）...")
        settle_duration = 5.0
        settle_until = time.monotonic() + settle_duration
        gate_start = time.monotonic() + 1.0
        next_dbg = time.monotonic()
        while time.monotonic() < settle_until:
            msg = shared["msg"]
            if msg is None:
                continue
            measured = [float(msg.motor_state[i].q) for i in range(12)]
            drift = max(abs(measured[i] - targets[i]) for i in range(12))
            max_drift = max(max_drift, drift)
            now = time.monotonic()
            if now >= next_dbg:
                i_max = max(range(12), key=lambda i: abs(measured[i] - targets[i]))
                print(f"[DBG] settle t={settle_duration - (settle_until - now):.1f}s " f"drift={drift:.4f} joint={i_max}")
                next_dbg = now + 0.5
            if now >= gate_start and drift > 0.15:
                abort(f"站稳窗口漂移超限 {drift:.3f} rad")
                break
            publisher.Write(build())
            time.sleep(0.002)

        print("[INFO] 预置卸重 1.5s（后腿微蹲，前爪卸重）...")
        unload_q = [float(targets[i]) for i in range(12)]
        unload_steps = int(round(1.5 / 0.002))
        for step in range(1, unload_steps + 1):
            p = step / unload_steps
            targets = [unload_q[i] for i in range(12)]
            for i in range(12):
                if i % 3 == 1 and i >= 6:
                    targets[i] = unload_q[i] - 0.04 * p
                elif i % 3 == 2 and i >= 6:
                    targets[i] = unload_q[i] - 0.03 * p
            publisher.Write(build())
            time.sleep(0.002)
        print("[INFO] 预抬爪 1.5s（空载前爪离地）...")
        pre_q = [float(targets[i]) for i in range(12)]
        pre_steps = int(round(1.5 / 0.002))
        for step in range(1, pre_steps + 1):
            p = step / pre_steps
            targets = [pre_q[i] for i in range(12)]
            targets[FR_IDX[1]] = pre_q[FR_IDX[1]] - 0.05 * p
            targets[FR_IDX[2]] = pre_q[FR_IDX[2]] - 0.08 * p
            publisher.Write(build())
            time.sleep(0.002)
        print("[INFO] 保持官方站姿追手（不坡到训练低姿：真机力矩上限仅为仿真 40%）")

        print(f"[INFO] 策略追手阶段（最长 {args.hold_seconds:.0f}s）...")
        while not aborted and time.monotonic() < hold_until:
            msg = shared["msg"]
            if msg is None:
                continue
            measured = [float(msg.motor_state[i].q) for i in range(12)]
            dq = [float(msg.motor_state[i].dq) for i in range(12)]
            imu_q = tuple(float(v) for v in msg.imu_state.quaternion)
            pitch, roll = quat_to_pitch_roll(imu_q)
            if z0 is None:
                z0 = 0.335
            support_idx = [i for i in range(12)
                           if i not in (FR_IDX if side == 0 else FL_IDX)]
            drift = max(abs(measured[i] - targets[i]) for i in support_idx)
            max_drift = max(max_drift, drift)
            if drift > 0.30:
                abort(f"追手阶段支撑漂移超限 {drift:.3f} rad")
                break
            if abs(pitch) > 0.30 or abs(roll) > 0.30:
                abort(f"姿态超限 pitch={pitch:.2f} roll={roll:.2f}")
                break
            now = time.monotonic()
            if now - t_last >= 5.0:
                temps_now = [float(msg.motor_state[i].temperature)
                             for i in range(12)]
                max_temp = max(max_temp, max(temps_now, default=0))
                rise = max((a - b for a, b in zip(temps_now, temp_last)
                            if b > 0.0), default=0.0)
                print(f"[INFO] t={settle_duration - (settle_until - now):.0f}s "
                      f"max_temp={max_temp:.0f}C rise={rise:.1f}C "
                      f"drift={drift:.3f}")
                if max_temp >= TEMP_HARD_LIMIT:
                    abort(f"温度硬上限 {TEMP_HARD_LIMIT:.0f}C")
                    break
                if rise >= TEMP_RISE_LIMIT:
                    abort(f"5s 升温 {rise:.1f}C 超速率门")
                    break
                temp_last = temps_now
                t_last = now
            # 视觉 + 预测
            hand_stream.poll()
            raw = hand_stream.target
            if raw is not None:
                predictor.update(now, raw[0], raw[1])
            hand_pos = predictor.predict(now)
            if hand_pos is not None:
                px, py = hand_pos
                hand_frames += 1
                side = 0 if px >= 0.5 else 1
                hand_lost_frames = 0
                if cycle % policy_interval == 0:
                    obs = np.concatenate([
                        np.asarray(measured) / math.pi,
                        np.asarray(dq) / 10.0,
                        np.asarray([pitch, roll]),
                        np.asarray([px, max(0.0, min(1.0, py + args.py_bias))]),
                        np.asarray([float(side)])])
                    deltas = policy.act(obs)
                    act_idx = FR_IDX if side == 0 else FL_IDX
                    if getattr(args, "debug_joints", False):
                        print(f"[DBGJ] py={py:.3f} side={side} raw="
                              f"{[round(float(deltas[i]), 4) for i in act_idx]}")
                    for i in range(12):
                        if i not in act_idx:
                            continue
                        if i % 3 == 1:
                            lo, hi = -0.6, 1.35
                        elif i % 3 == 2:
                            lo, hi = -2.2, -0.7
                        else:
                            lo, hi = -0.5, 0.5
                        delta = deltas[i]
                        if (hand_pos is not None and py < 0.60
                                and now - last_contact_t > 1.0):
                            if i % 3 == 1:
                                delta -= 0.020
                            elif i % 3 == 2:
                                delta -= 0.012

                        new = float(np.clip(targets[i] + delta, lo, hi))
                        targets[i] = float(np.clip(
                            new, targets[i] - 0.05, targets[i] + 0.05))
                idxs = FR_IDX if side == 0 else FL_IDX
                tau_abs = max(abs(float(msg.motor_state[i].tau_est))
                              for i in idxs)
                if contact_det.update(now, tau_abs):
                    contact_count += 1
                    last_contact_t = now
                    print(f"[INFO] 握到！tau 尖峰 {tau_abs:.1f}Nm",
                          file=sys.stderr)
            else:
                hand_lost_frames += 1
                if hand_lost_frames >= args.hand_lost_frames:
                    print("[INFO] 手已离开，进入收尾")
                    break
            publisher.Write(build())
            cycle += 1
            time.sleep(0.002)

        print("[INFO] 斜坡降回趴姿（2.5s）...")
        start_q = [float(targets[i]) for i in range(12)]
        descend_steps = int(round(2.5 / 0.002))
        for step in range(1, descend_steps + 1):
            p = step / descend_steps
            targets = [start_q[i] + (LIE_Q[i] - start_q[i]) * p
                       for i in range(12)]
            publisher.Write(build())
            time.sleep(0.002)
        publisher.Write(stop())
    except Exception as error:
        print(f"[ERROR] 主流程异常: {error}", file=sys.stderr)
        aborted = True
        abort_reason = f"EXCEPTION {type(error).__name__}: {error}"
    finally:
        time.sleep(0.3)
        restored = False
        select_mode_code = None
        for attempt in range(12):
            try:
                code, _ = msc.SelectMode("mcf")
                select_mode_code = int(code)
                if code == 0:
                    restored = True
                    break
            except Exception:
                pass
            time.sleep(0.5)
        check_ok = False
        check_mode_name = ""
        for _ in range(20):
            time.sleep(0.1)
            try:
                _, result = msc.CheckMode()
                if result and result.get("name"):
                    check_ok = True
                    check_mode_name = str(result.get("name"))
                    break
            except Exception:
                continue
        action_ok = False
        restore_code = None
        try:
            code = sport.StandDown()
            restore_code = int(code)
            action_ok = (code == 0)
        except Exception:
            pass
        print(f"[INFO] 恢复: SelectMode={restored} CheckMode={check_ok} "
              f"StandDown={action_ok}")

    class _Ctl:
        pass
    ctl = _Ctl()
    ctl.aborted = aborted
    ctl.abort_reason = abort_reason
    ctl.max_track_err = 0.0
    ctl.max_track_err_joint = ""
    ctl.max_pitch_deg = 0.0
    try:
        trial = build_trial_record(
            args, ctl, temps,
            hold_metrics={"hand_frames": hand_frames,
                          "last_hand_px": None,
                          "contact_trigger_count": contact_count,
                          "max_detector_hold_s": 0.0,
                          "contact_ground_truth": "not_measured",
                          "contact_confirmation_source": ""},
            retreat_evidence={
                "retreat_completed": restored and check_ok and action_ok,
                "select_mode_code": select_mode_code,
                "check_mode_name": check_mode_name,
                "restore_code": restore_code,
                "failure": "" if not aborted else abort_reason,
                "steps": [],
            },
            session={
                "start_t": session_start,
                "firmware": "mcf(version【待补】)",
                "camera": "GO2_front_RGB(VideoClient)",
                "calibration_id": "placeholder",
                "floor": "", "light": "", "trial_index": "",
                "policy_sha256": policy_sha256,
            })
        out_path = write_trial_record(trial, args.trial_log_dir)
        print(f"[EVIDENCE] {out_path}")
    except Exception as error:
        print(f"[WARN] 试验记录写入失败: {error}", file=sys.stderr)

    if aborted:
        print(f"[RESULT] STAND_HS_ABORTED reason={abort_reason}")
        return 3
    print(f"[RESULT] STAND_HS_OK hand_frames={hand_frames} "
          f"contact={contact_count} max_temp={max_temp:.0f}C "
          f"max_drift={max_drift:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
