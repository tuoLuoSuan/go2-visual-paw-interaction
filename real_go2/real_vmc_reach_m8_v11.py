#!/usr/bin/env python3
"""GO2 真机低层趴姿双前腿握手，里程碑 8。

狗趴着（官方 StandDown）-> 低层接管 -> 两条前腿（FR+FL）同时向前伸直
-> 视觉检测手位置（UDP 流，M3 视觉进程发 'x y z'）：
    手在右边（y<0）-> 右前爪(FR) 再往前够
    手在左边（y>0）-> 左前爪(FL) 再往前够
-> 保持握手 -> 收回 -> 恢复控制 -> 结束站立收工（--restore-action 默认 standup）。

基于 M6（趴姿低层验证 M6_OK：零升温、pitch 5.9°、完全伸直保持 5.4s）。

安全：dry-run 默认、现场检查、口令、看门狗（支撑腿 0.25 / FR 0.60 /
姿态 45° / 温度 80C+速率门）、遥控器最高优先。
"""
import argparse
import math
import os
import socket
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
for _src in ("simulation/src", "robot/src"):
    _path = WORKSPACE / _src
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import standing_paw_lift_common as common
import vmc_balance as vmc
from hand_predictor import HandPredictor
from contact_detector import ContactSpikeDetector
from trial_record_v4 import (build_trial_record_v4, write_trial_record_v4,
                             TraceEmitter, StageTimer, sha256_file)

POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0
CONFIRM_PASSPHRASE = "GO2-M8-20260818"

FR_JOINT_INDICES = (0, 1, 2)   # FR_hip, FR_thigh, FR_calf
FL_JOINT_INDICES = (3, 4, 5)   # FL_hip, FL_thigh, FL_calf
SUPPORT_ERR_LIMIT = 0.25
FR_ERR_LIMIT = 0.60
REAR_DRIFT_LIMIT_DEG = 25.0
FRONT_DRIFT_LIMIT_DEG = -30.0
TEMP_HARD_LIMIT = 80.0
TEMP_RISE_LIMIT = 3.0
TEMP_RISE_WINDOW = 5.0

# 趴姿双前腿向前打直（用户要求"向前"不是"向下"）：
# 仿真扫描：FR_t=-0.90（大腿水平前伸）+ FR_c=-1.05（小腿接近伸直，留极限余量
# 避免 -0.84 贴死关节极限产生电机振动噪音）
# -> paw_x=0.559（前伸 56cm）、paw_z=0.022（贴地）
# 关节极限：FR_t [-1.57, 3.49]、FR_c [-2.72, -0.84]
REACH_DT = -2.13   # thigh 前摆偏移：1.23 -> -0.90
REACH_DC = 1.69    # calf 伸直偏移：-2.74 -> -1.05
FR_T_JOINT_MIN = -1.00   # thigh 前摆下限（目标 -0.90）
FR_C_LIMIT_HI = -0.84    # calf 伸直上限（关节极限）
FR_C_LIMIT_LO = -2.72    # calf 折叠下限


class ReachControl:
    def __init__(self, kp=150.0, kd=10.0, hold_s=8.0, vmc_kp_rot=300.0,
                 vmc_kd_rot=30.0, tau_clip=50.0, crc=None):
        self.kp = float(kp)
        self.kd = float(kd)
        self.active_kp = float(kp)
        self.active_kd = float(kd)
        self.active_kp_fr = float(kp)   # 前腿分腿增益：只让追手腿发力
        self.active_kp_fl = float(kp)
        self.hold_s = float(hold_s)
        self.vmc_kp_rot = float(vmc_kp_rot)
        self.vmc_kd_rot = float(vmc_kd_rot)
        self.tau_clip = (None if tau_clip is None else float(tau_clip))
        self.crc = crc
        self.low_state = None
        self.low_cmd = None
        self.freeze_q = [0.0] * 12
        self.freeze_imu = (1.0, 0.0, 0.0, 0.0)
        self.fr_target = None    # FR 追踪关节目标
        self.fl_target = None    # FL 追踪关节目标
        self.max_track_err = 0.0
        self.max_track_err_joint = ""
        self.max_pitch_deg = 0.0
        self.contact_count = 0   # 真机"握到"检测触发次数（试验证据）
        self.hand_frames = 0     # 保持阶段检测到手的总帧数（试验证据）
        self.last_hand_px = None  # 最后一次手位置（选足证据）
        self.contact_hold_frames = 0      # 当前连续接触保持帧数
        self.max_contact_hold_frames = 0  # 最长连续接触保持帧数
        self.aborted = False
        self.abort_reason = ""
        self.model = None
        self.data = None
        self.ctx = None
        self._temp_time = None
        self._temp_last = 0.0

    def init_low_cmd(self, factory):
        cmd = factory()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0
        for i in range(20):
            cmd.motor_cmd[i].mode = 0x01
            cmd.motor_cmd[i].q = POS_STOP_F
            cmd.motor_cmd[i].kp = 0
            cmd.motor_cmd[i].dq = VEL_STOP_F
            cmd.motor_cmd[i].kd = 0
            cmd.motor_cmd[i].tau = 0
        return cmd

    def build_cmd(self, targets, tau):
        cmd = self.low_cmd
        # 保持阶段用低增益防振动（hold_kp）；伸直/追手用高增益
        kp_use = getattr(self, "active_kp", self.kp)
        kd_use = getattr(self, "active_kd", self.kd)
        for i in range(12):
            cmd.motor_cmd[i].mode = 0x01
            cmd.motor_cmd[i].q = float(targets[i])
            cmd.motor_cmd[i].dq = 0.0
            # 前腿(FR+FL)分腿增益：追手腿高增益，另一条保持低增益（降发热）；
            # 支撑腿(RR/RL)用低增益（防止伸直时支撑腿固件 PD 输出大电流）
            if i in (0, 1, 2):
                cmd.motor_cmd[i].kp = getattr(self, "active_kp_fr", kp_use)
                cmd.motor_cmd[i].kd = kd_use
            elif i in (3, 4, 5):
                cmd.motor_cmd[i].kp = getattr(self, "active_kp_fl", kp_use)
                cmd.motor_cmd[i].kd = kd_use
            else:
                cmd.motor_cmd[i].kp = min(kp_use, 30.0)
                cmd.motor_cmd[i].kd = min(kd_use, 4.0)
            cmd.motor_cmd[i].tau = float(tau[i])
        cmd.crc = self.crc.Crc(cmd)
        return cmd

    def build_stop_cmd(self):
        cmd = self.low_cmd
        for i in range(20):
            cmd.motor_cmd[i].mode = 0x01
            cmd.motor_cmd[i].q = POS_STOP_F
            cmd.motor_cmd[i].kp = 0
            cmd.motor_cmd[i].dq = VEL_STOP_F
            cmd.motor_cmd[i].kd = 0
            cmd.motor_cmd[i].tau = 0
        cmd.crc = self.crc.Crc(cmd)
        return cmd

    def setup_model(self):
        self.ctx = common.load_context(
            WORKSPACE / "unitree_mujoco" / "unitree_robots" / "go2" / "scene.xml")
        self.model = self.ctx.model
        self.data = common.mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002

    def targets(self):
        out = list(self.freeze_q)
        if self.fr_target is not None:
            for i, value in zip(FR_JOINT_INDICES, self.fr_target):
                out[i] = float(value)
        if self.fl_target is not None:
            for i, value in zip(FL_JOINT_INDICES, self.fl_target):
                out[i] = float(value)
        return out

    def realtime_balance(self, measured_q, imu_quat, targets):
        ctx = self.ctx
        data = self.data
        base = ctx.base_qpos_address
        qpos = np.zeros(self.model.nq, dtype=float)
        for address, value in zip(ctx.qpos_addresses, measured_q):
            qpos[address] = float(value)
        w, x, y, z = (float(v) for v in imu_quat)
        qpos[base + 3] = w
        qpos[base + 4] = x
        qpos[base + 5] = y
        qpos[base + 6] = z
        qpos[base + 2] = 0.32
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        common.mujoco.mj_forward(self.model, data)

        ref_qpos = np.zeros(self.model.nq, dtype=float)
        for address, value in zip(ctx.qpos_addresses, self.freeze_q):
            ref_qpos[address] = float(value)
        ref_qpos[base + 2] = 0.32
        fw, fx, fy, fz = (float(v) for v in self.freeze_imu)
        ref_qpos[base + 3] = fw
        ref_qpos[base + 4] = fx
        ref_qpos[base + 5] = fy
        ref_qpos[base + 6] = fz
        cfg = common.StandingConfig(
            vmc_enabled=True, vmc_kp_pos=300.0, vmc_kd_pos=30.0,
            vmc_kp_rot=self.vmc_kp_rot, vmc_kd_rot=self.vmc_kd_rot,
        ).validate()
        force, moment = vmc.base_wrench(ctx, data, tuple(ref_qpos), cfg)
        regions = []
        for index in range(data.ncon):
            c = data.contact[index]
            first, second = int(c.geom1), int(c.geom2)
            other = second if first == ctx.floor_geom_id else first
            for name, gid in ctx.foot_geom_ids.items():
                if other == int(gid) and name not in regions:
                    regions.append(name)
        vmc_tau = vmc.support_foot_torques(
            ctx, data, force, moment, tuple(regions), ctx.dof_addresses)
        out = [float(v) for v in vmc_tau]
        # 支撑腿(RR/RL)的 VMC 力矩降权：双前腿伸直时身体姿态变化，
        # VMC 会持续对抗支撑腿 -> 大电流过热。支撑腿主要靠结构 + 低增益 PD。
        support_weight = 0.3
        for i in range(6, 12):
            out[i] *= support_weight
        if self.tau_clip is not None:
            out = [float(max(-self.tau_clip, min(self.tau_clip, v)))
                   for v in out]
        return out, regions

    def check(self, measured_q):
        targets = self.targets()
        errs = [abs(targets[i] - measured_q[i]) for i in range(12)]
        err = max(errs)
        if err > self.max_track_err:
            self.max_track_err = err
            names = ("FR_h", "FR_t", "FR_c", "FL_h", "FL_t", "FL_c",
                     "RR_h", "RR_t", "RR_c", "RL_h", "RL_t", "RL_c")
            self.max_track_err_joint = names[int(errs.index(err))]
        moving = set(FR_JOINT_INDICES) | set(FL_JOINT_INDICES)
        support_err = max(errs[i] for i in range(12) if i not in moving)
        front_err = max(errs[i] for i in moving)
        names = ("FR_h", "FR_t", "FR_c", "FL_h", "FL_t", "FL_c",
                 "RR_h", "RR_t", "RR_c", "RL_h", "RL_t", "RL_c")
        if support_err > SUPPORT_ERR_LIMIT:
            worst = int(max(range(12), key=lambda i: errs[i]
                            if i not in moving else -1.0))
            self.abort("SUPPORT_LEG_DRIFT",
                       f"joint={names[worst]} err={support_err:.4f} rad "
                       f"target={targets[worst]:.3f} meas={measured_q[worst]:.3f}")
            return False
        if front_err > FR_ERR_LIMIT:
            worst = int(max(moving, key=lambda i: errs[i]))
            self.abort("TRACKING_ERROR",
                       f"joint={names[worst]} err={front_err:.4f} rad "
                       f"target={targets[worst]:.3f} meas={measured_q[worst]:.3f}")
            return False
        imu = self.low_state.imu_state
        qw, qx, qy, qz = (float(v) for v in imu.quaternion)
        pitch = math.degrees(math.asin(
            max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))))
        roll = math.degrees(math.atan2(
            2.0 * (qw * qx + qy * qz),
            1.0 - 2.0 * (qx * qx + qy * qy)))
        self.max_pitch_deg = max(self.max_pitch_deg, abs(pitch))
        if abs(pitch) > 45.0 or abs(roll) > 45.0:
            self.abort("ORIENTATION", f"pitch {pitch:.1f} roll {roll:.1f}")
            return False
        fw, fx, fy, fz = (float(v) for v in self.freeze_imu)
        frozen_pitch = math.degrees(math.asin(
            max(-1.0, min(1.0, 2.0 * (fw * fy - fz * fx)))))
        drift = pitch - frozen_pitch
        if drift > REAR_DRIFT_LIMIT_DEG:
            self.abort("REAR_LEAN_DRIFT",
                       f"pitch {pitch:.1f} vs frozen {frozen_pitch:.1f}")
            return False
        if drift < FRONT_DRIFT_LIMIT_DEG:
            self.abort("FRONT_LEAN_DRIFT",
                       f"pitch {pitch:.1f} vs frozen {frozen_pitch:.1f}")
            return False
        temps = [float(self.low_state.motor_state[i].temperature)
                 for i in range(12)]
        max_temp = max(temps, default=0.0)
        if max_temp > TEMP_HARD_LIMIT:
            self.abort("TEMP_HIGH", f"max motor temp {max_temp:.0f} C")
            return False
        now = time.monotonic()
        if self._temp_time is not None and now - self._temp_time >= TEMP_RISE_WINDOW:
            if max_temp - self._temp_last >= TEMP_RISE_LIMIT:
                self.abort("TEMP_RISING_FAST",
                           f"{self._temp_last:.0f} -> {max_temp:.0f} C "
                           f"in {TEMP_RISE_WINDOW:.0f}s")
                return False
            self._temp_time = now
            self._temp_last = max_temp
        elif self._temp_time is None:
            self._temp_time = now
            self._temp_last = max_temp
        return True

    def abort(self, reason, detail=""):
        self.aborted = True
        self.abort_reason = reason
        if detail:
            print(f"[ABORT] {reason}: {detail}", file=sys.stderr)
        else:
            print(f"[ABORT] {reason}", file=sys.stderr)


LIE_Q = [-0.06, 1.24, -2.75, 0.08, 1.25, -2.78,
         -0.40, 1.26, -2.77, 0.41, 1.25, -2.78]


class HandStream:
    """UDP 手位置流：'x y z'（3D 模式）或 'L'/'R'/'N'（左右信号模式）。"""

    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", int(port)))
        self.sock.setblocking(False)
        self.target = None      # (px, py) 平滑后手位置 / None
        self.side = None        # 兼容旧三态（保留）
        self.count = 0
        self.last_rx = time.monotonic()
        self._smooth = None     # EMA 平滑状态（丝滑关键：滤视觉抖动）

    def poll(self):
        got = False
        while True:
            try:
                payload, _ = self.sock.recvfrom(128)
            except BlockingIOError:
                break
            self.count += 1
            got = True
            self.last_rx = time.monotonic()
            text = payload.decode("utf-8", "replace").strip().upper()
            if text in ("NONE", "LOST", "", "N"):
                self.target = None
                self.side = None
                self._smooth = None
                continue
            if text in ("L", "R", "C"):
                self.side = {"L": "LEFT", "R": "RIGHT",
                             "C": "CENTER"}[text]
                self.target = None
                self._smooth = None
                continue
            try:
                values = tuple(float(v) for v in text.split())
            except ValueError:
                continue
            if len(values) == 2 and all(math.isfinite(v) for v in values):
                # EMA 平滑：滤掉 MediaPipe 检测抖动，爪子丝滑跟踪
                raw = (values[0], values[1])
                if self._smooth is None:
                    self._smooth = raw
                else:
                    alpha = 0.25   # 更平滑：滤抖更狠，快速移动仍跟得上
                    self._smooth = (
                        self._smooth[0] + alpha * (raw[0] - self._smooth[0]),
                        self._smooth[1] + alpha * (raw[1] - self._smooth[1]))
                self.target = self._smooth
                self.side = ("LEFT" if self._smooth[0] < 0.5 else "RIGHT")
        # 关键修复：超过 1 秒没收到新包 = 视觉停/手丢失 -> 清空目标
        # （否则旧坐标变成"幽灵手"，手拿开程序不结束）
        if not got and time.monotonic() - self.last_rx > 1.0:
            self.target = None
            self.side = None
            self._smooth = None

    def close(self):
        self.sock.close()


def run(args):
    print(f"[M8] 趴姿双前腿握手 {args.hold_seconds}s (Kp={args.kp})")
    trace = TraceEmitter()
    stages = StageTimer(trace)
    stages.start("reach")
    _t0_wall_ms = int(time.time() * 1000)
    _t0_mono_ms = int(time.monotonic() * 1000)
    if not args.dry_run:
        for name, ok in (
            ("狗已趴卧（官方平衡）", args.site_check),
            ("遥控器在手且可随时接管", args.site_check),
            ("四周 1m 无障碍、地面平整", args.site_check),
            ("观察员在场、无他人靠近", args.site_check),
        ):
            print(f"  [检查] {name}: {'通过' if ok else '未确认'}")
            if not ok:
                print("[ABORT] 现场条件未全部确认", file=sys.stderr)
                return 4
        if args.confirm != CONFIRM_PASSPHRASE:
            print("[ABORT] 确认口令错误", file=sys.stderr)
            return 5

    if args.dry_run:
        print("[DRY-RUN] 时序预览：")
        print("  1. StandUp() 官方站立 -> StandDown() 官方趴下")
        print("  2. ReleaseMode -> 低层接管（mcf）")
        print("  3. 冻结趴姿，纯 PD 稳定窗口")
        print("  4. 双前腿（FR+FL）thigh 前摆 + calf 伸直（完全伸直）")
        print("  5. 保持 + UDP 视觉手位置：手左->FL 再够，手右->FR 再够")
        print("  6. 收回 -> 回退趴卧 -> PosStopF -> SelectMode('mcf')")
        print(f"  7. 结束动作：{args.restore_action}（默认站立收工）")
        print(f"  温度纪律：起始门限 {args.max_start_temp:.0f}C，"
              f"看门狗 {TEMP_HARD_LIMIT:.0f}C + 速率门")
        print("[DRY-RUN] M8_DRY_RUN_OK")
        return 0

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
    ctl = ReachControl(kp=args.kp, kd=args.kd, hold_s=args.hold_seconds,
                       vmc_kp_rot=args.vmc_kp_rot, vmc_kd_rot=args.vmc_kd_rot,
                       tau_clip=args.tau_clip, crc=CRC())
    ctl.setup_model()

    shared = {"msg": None}
    lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
    lowstate_sub.Init(lambda m: shared.__setitem__("msg", m), 10)
    deadline = time.monotonic() + 5.0
    while shared["msg"] is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if shared["msg"] is None:
        print("[ABORT] 5s 内未收到 lowstate", file=sys.stderr)
        return 6
    ctl.low_state = shared["msg"]
    temps = [float(ctl.low_state.motor_state[i].temperature)
             for i in range(12)]
    print(f"[INFO] 起始电机温度: max={max(temps, default=0):.0f} C")
    if max(temps, default=0) > args.max_start_temp:
        print(f"[ABORT] 起始温度超上限 {args.max_start_temp:.0f}C，等冷却",
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
    time.sleep(2.0)
    print("[INFO] StandDown() 官方趴下 ...")
    print("  code:", sport.StandDown())
    time.sleep(2.0)

    released = False
    for attempt in range(10):
        try:
            _, result = msc.CheckMode()
        except Exception as error:
            print(f"[WARN] CheckMode 异常: {error}", file=sys.stderr)
            result = None
        name = (result or {}).get("name", "")
        if not name:
            released = True
            break
        msc.ReleaseMode()
        time.sleep(0.2)

    publisher = None
    try:
        print("[INFO] 低层接管就绪，立即冻结趴姿（防塌）")
        freeze_deadline = time.monotonic() + 0.5
        while shared["msg"] is None and time.monotonic() < freeze_deadline:
            time.sleep(0.002)
        for i in range(12):
            ctl.freeze_q[i] = float(shared["msg"].motor_state[i].q)
        ctl.freeze_imu = tuple(
            float(v) for v in shared["msg"].imu_state.quaternion)
        print("[INFO] 冻结关节角:", [round(v, 3) for v in ctl.freeze_q])

        publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        publisher.Init()
        ctl.low_cmd = ctl.init_low_cmd(unitree_go_msg_dds__LowCmd_)

        print(f"[INFO] 稳定窗口（纯PD，最长 {args.settle_seconds}s）...")
        settle_until = time.monotonic() + args.settle_seconds
        prev_q = None
        stable_since = None
        measured = None
        imu_q = None
        while time.monotonic() < settle_until:
            msg = shared["msg"]
            if msg is None:
                continue
            ctl.low_state = msg
            measured = [float(msg.motor_state[i].q) for i in range(12)]
            imu_q = tuple(float(v) for v in msg.imu_state.quaternion)
            publisher.Write(ctl.build_cmd(ctl.targets(), [0.0] * 12))
            if prev_q is not None:
                max_delta = max(abs(measured[i] - prev_q[i])
                                for i in range(12))
                if max_delta < 0.005:
                    if stable_since is None:
                        stable_since = time.monotonic()
                    elif time.monotonic() - stable_since >= 1.0:
                        print(f"[INFO] 关节已收敛（max_delta={max_delta:.4f}）",
                              file=sys.stderr)
                        break
                else:
                    stable_since = None
            prev_q = measured
            time.sleep(0.002)
        if measured is None:
            print("[ABORT] 稳定窗口未收到任何 lowstate", file=sys.stderr)
            return 6
        for i in range(12):
            ctl.freeze_q[i] = measured[i]
        ctl.freeze_imu = imu_q
        print("[INFO] 稳定后冻结:", [round(v, 3) for v in ctl.freeze_q])

        # ---- 双前腿完全伸直 ----
        # FR 目标
        ctl.fr_target = [float(ctl.freeze_q[j]) for j in FR_JOINT_INDICES]
        fr_pose = [
            float(ctl.freeze_q[FR_JOINT_INDICES[0]]),
            max(FR_T_JOINT_MIN,
                float(ctl.freeze_q[FR_JOINT_INDICES[1]]) + REACH_DT),
            min(FR_C_LIMIT_HI, max(FR_C_LIMIT_LO,
                                   float(ctl.freeze_q[FR_JOINT_INDICES[2]])
                                   + REACH_DC)),
        ]
        # FL 目标（镜像：hip 用冻结值，thigh/calf 同偏移）
        ctl.fl_target = [float(ctl.freeze_q[j]) for j in FL_JOINT_INDICES]
        fl_pose = [
            float(ctl.freeze_q[FL_JOINT_INDICES[0]]),
            max(FR_T_JOINT_MIN,
                float(ctl.freeze_q[FL_JOINT_INDICES[1]]) + REACH_DT),
            min(FR_C_LIMIT_HI, max(FR_C_LIMIT_LO,
                                   float(ctl.freeze_q[FL_JOINT_INDICES[2]])
                                   + REACH_DC)),
        ]
        print(f"[INFO] 双前腿伸直: FR目标={[round(v, 3) for v in fr_pose]} "
              f"FL目标={[round(v, 3) for v in fl_pose]}")

        # HandStream 提前创建（伸直阶段就收包预热，伸直完立即响应手）
        hand_stream = None
        predictor = None
        if args.hand_port:
            hand_stream = HandStream(args.hand_port)
            print(f"[INFO] 手位置流: UDP 127.0.0.1:{args.hand_port}"
                  f"（伸直阶段预热）")
            if args.predict_latency > 0.0:
                predictor = HandPredictor(
                    latency_s=args.predict_latency,
                    max_lost_s=args.predict_max_lost)
                print(f"[INFO] Kalman 预测: 前推 {args.predict_latency:.3f}s，"
                      f"手丢失外推 {args.predict_max_lost:.3f}s")

        reach_until = time.monotonic() + args.reach_seconds
        cycle = 0
        while time.monotonic() < reach_until and not ctl.aborted:
            msg = shared["msg"]
            if msg is None:
                continue
            ctl.low_state = msg
            measured = [float(msg.motor_state[i].q) for i in range(12)]
            imu_q = tuple(float(v) for v in msg.imu_state.quaternion)
            # 伸直阶段持续 poll 手（预热 EMA 滤波 + Kalman 收敛，
            # 保持阶段立即平滑响应）
            if hand_stream is not None:
                hand_stream.poll()
                if predictor is not None and hand_stream.target is not None:
                    predictor.update(time.monotonic(),
                                     hand_stream.target[0],
                                     hand_stream.target[1])
            g = args.track_gain
            slew = args.slew_limit_rad
            for i in range(3):
                desired = (ctl.fr_target[i]
                           + (fr_pose[i] - ctl.fr_target[i]) * g)
                ctl.fr_target[i] = float(np.clip(
                    desired,
                    ctl.fr_target[i] - slew,
                    ctl.fr_target[i] + slew))
                desired = (ctl.fl_target[i]
                           + (fl_pose[i] - ctl.fl_target[i]) * g)
                ctl.fl_target[i] = float(np.clip(
                    desired,
                    ctl.fl_target[i] - slew,
                    ctl.fl_target[i] + slew))
            if not ctl.check(measured):
                break
            tau, _ = ctl.realtime_balance(measured, imu_q, ctl.targets())
            publisher.Write(ctl.build_cmd(ctl.targets(), tau))
            cycle += 1
            if cycle % 250 == 0:
                print(f"[INFO] 伸直 t={cycle * 0.002:.1f}s FR={[round(v, 3) for v in ctl.fr_target]} "
                      f"FL={[round(v, 3) for v in ctl.fl_target]}",
                      file=sys.stderr)
            time.sleep(0.002)
        stages.end("reach")
        stages.start("track")
        print(f"[INFO] 伸直阶段 {cycle} 次（约 {cycle * 0.002:.1f}s）")

        # ---- 保持 + 视觉选爪 ----
        # 核心逻辑：手在 -> 一直保持打直+够手；手离开持续
        # --hand-lost-frames 帧 -> 退出收回。hold 上限防视觉故障。
        # hand_stream 已在伸直阶段创建并预热（EMA 已收敛，立即响应）。
        hold_until = time.monotonic() + ctl.hold_s
        cycle = 0
        last_side = None
        hand_lost_frames = 0
        hand_lost_required = max(1, int(args.hand_lost_frames))
        # 保持阶段：打直已到位，前腿低增益（结构支撑，几乎不发力防振动）。
        # 检测到手 -> 只切追手那条腿高增益发力；手离开 -> 全回低增益。
        ctl.active_kp = args.hold_kp
        ctl.active_kd = args.hold_kd
        ctl.active_kp_fr = args.hold_kp
        ctl.active_kp_fl = args.hold_kp
        # A5：接触检测（爪子碰到手 -> tau_est 尖峰 -> 暂停追手保持）
        contact_det = None
        if args.contact_hold_s > 0.0:
            contact_det = ContactSpikeDetector(hold_s=args.contact_hold_s,
                                               warmup=0.5)
        # RL 策略模式（部署版）：加载 checkpoint（纯 numpy 推理，无需 torch）
        policy_runner = None
        policy_interval = 10
        policy_fr_lo = policy_fr_hi = policy_fl_lo = policy_fl_hi = None
        if args.policy:
            from policy_runner import PolicyRunner
            policy_runner = PolicyRunner(args.policy)
            policy_interval = max(1, int(round(500.0 / args.policy_hz)))
            print(f"[INFO] RL 策略已加载: {args.policy} "
                  f"（{args.policy_hz:.0f}Hz 推理，动作增量+限位+0.05 安全钳）")
            # 关节限位与训练环境一致（相对打直位）
            policy_fr_lo = [ctl.freeze_q[0] - 0.35, -1.55, -2.0]
            policy_fr_hi = [ctl.freeze_q[0] + 0.35, fr_pose[1], -1.05]
            policy_fl_lo = [ctl.freeze_q[3] - 0.35, -1.55, -2.0]
            policy_fl_hi = [ctl.freeze_q[3] + 0.35, fl_pose[1], -1.05]
        while time.monotonic() < hold_until and not ctl.aborted:
            msg = shared["msg"]
            if msg is None:
                continue
            ctl.low_state = msg
            measured = [float(msg.motor_state[i].q) for i in range(12)]
            imu_q = tuple(float(v) for v in msg.imu_state.quaternion)
            # 连续追踪：手位置 (px, py) -> 前腿关节（单侧响应！）
            #   px (0-1 左右): 手在左 -> 只有 FL(左爪) 动；手在右 -> 只有 FR(右爪) 动
            #   py (0-1 上下):  抬爪幅度 -> thigh 前摆（放开范围）
            # 范围：hip 关节极限 ±1.047，摆到 0.8；thigh 极限 -1.57，
            #       打直位 -0.90 可再前摆 0.6（到 -1.50）。不额外限制。
            hand_pos = None
            if hand_stream is not None:
                hand_stream.poll()
                raw = hand_stream.target  # (px, py) 或 None
                if predictor is not None:
                    # A2/A3：Kalman 延迟前推 + 手丢失短时轨迹记忆
                    # （论文 GRU 记忆的规则版：手快速移动短暂丢帧时
                    #  按最后速度外推 --predict-max-lost 秒继续追）
                    now = time.monotonic()
                    if raw is not None:
                        predictor.update(now, raw[0], raw[1])
                    hand_pos = predictor.predict(now)
                else:
                    hand_pos = raw
                if hand_pos is not None:
                    px, py = hand_pos
                    ctl.last_hand_px = float(px)
                    side_x = -1.0 if px < 0.5 else 1.0   # -1=左爪(FL), +1=右爪(FR)
                    if policy_runner is not None and cycle % policy_interval == 0:
                        # RL 策略模式：观测与训练一致（27维），输出前腿增量
                        dq_now = [float(msg.motor_state[i].dq)
                                  for i in range(12)]
                        obs = np.concatenate([
                            np.asarray(measured, dtype=float) / math.pi,
                            np.asarray(dq_now, dtype=float) / 10.0,
                            np.asarray([px, py], dtype=float),
                            np.asarray([1.0 if px < 0.5 else 0.0])])
                        deltas = policy_runner.act(obs)
                        for i, j in enumerate(FR_JOINT_INDICES):
                            new = float(np.clip(ctl.fr_target[i] + deltas[i],
                                                policy_fr_lo[i],
                                                policy_fr_hi[i]))
                            ctl.fr_target[i] = float(np.clip(
                                new,
                                ctl.fr_target[i] - 0.03,
                                ctl.fr_target[i] + 0.03))
                        for i, j in enumerate(FL_JOINT_INDICES):
                            new = float(np.clip(ctl.fl_target[i] + deltas[3 + i],
                                                policy_fl_lo[i],
                                                policy_fl_hi[i]))
                            ctl.fl_target[i] = float(np.clip(
                                new,
                                ctl.fl_target[i] - 0.03,
                                ctl.fl_target[i] + 0.03))
                    # 左右映射（面对面镜像）：你的右手 -> 小狗左爪(FL)，
                    # 你的左手 -> 小狗右爪(FR)。实测右手 px=0.11(画面左)，
                    # 左手 px=0.78(画面右)。
                    # 所以 px<0.5(画面左=你的右手) -> 小狗 FL(左爪)；
                    #     px>0.5(画面右=你的左手) -> 小狗 FR(右爪)。
                    amp_x = abs(px - 0.5) * 2.0
                    # 抬升映射按实测标定：用户手放正常握手位 py≈0.35
                    # -> 中等抬升 amp_y=0.5；手更高(py 0.2) -> 全抬；手低(py 0.6+) -> 不抬
                    amp_y = float(np.clip((0.60 - py) / 0.40, 0.0, 1.0))
                    # 关节映射（仿真几何校准 + 用户实测微调）：
                    #   hip: 左右侧摆 0~0.35（管左右）
                    #   thigh: 前摆 0.9->-1.5（管抬升，系数 0.45 校准：
                    #          py=0.35 时 z≈0.17 与手对齐，原来 0.60 偏高）
                    #   calf: 收回 -1.05->-1.3（配合抬升）
                    hip_target = 0.35 * amp_x
                    thigh_target = -0.90 - 0.45 * amp_y   # -0.9 ~ -1.35
                    calf_target = -1.05 - 0.20 * amp_y    # -1.05 ~ -1.25
                    fl_pose_extra = list(fl_pose)
                    fr_pose_extra = list(fr_pose)
                    if side_x < 0:
                        # 手在左：FL 动
                        fl_pose_extra[0] = min(0.35,
                                               fl_pose[0] + hip_target)
                        fl_pose_extra[1] = max(-1.55, thigh_target)
                        fl_pose_extra[2] = max(-2.0, calf_target)
                    else:
                        # 手在右：FR 动
                        fr_pose_extra[0] = max(-0.35,
                                               fr_pose[0] - hip_target)
                        fr_pose_extra[1] = max(-1.55, thigh_target)
                        fr_pose_extra[2] = max(-2.0, calf_target)
                    # A1：敏捷-精准双模 slew（论文双奖励切换思想）——
                    # 爪子离手目标远 -> 大步快追；近 -> 小步精细对齐
                    track_slew = args.track_slew_rad
                    if side_x < 0:
                        dist = max(abs(fl_pose_extra[i] - ctl.fl_target[i])
                                   for i in range(3))
                    else:
                        dist = max(abs(fr_pose_extra[i] - ctl.fr_target[i])
                                   for i in range(3))
                    if dist > args.slew_switch_rad:
                        track_slew = args.track_slew_fast_rad
                    # A5：接触检测（tau_est 尖峰 -> 握到，暂停追手保持）
                    contact_hold = False
                    if contact_det is not None:
                        idxs = (3, 4, 5) if side_x < 0 else (0, 1, 2)
                        tau_abs = max(
                            abs(float(msg.motor_state[i].tau_est))
                            for i in idxs)
                        if contact_det.update(time.monotonic(), tau_abs):
                            ctl.contact_count += 1
                            print(f"[INFO] 握到！tau 尖峰 {tau_abs:.1f}Nm，"
                                  f"保持 {args.contact_hold_s:.1f}s",
                                  file=sys.stderr)
                        contact_hold = contact_det.in_hold(time.monotonic())
                        if contact_hold:
                            ctl.contact_hold_frames += 1
                            ctl.max_contact_hold_frames = max(
                                ctl.max_contact_hold_frames,
                                ctl.contact_hold_frames)
                        else:
                            ctl.contact_hold_frames = 0
                    if not contact_hold and policy_runner is None:
                        for i in range(3):
                            desired = (ctl.fl_target[i]
                                       + (fl_pose_extra[i] - ctl.fl_target[i])
                                       * 0.10)
                            ctl.fl_target[i] = float(np.clip(
                                desired,
                                ctl.fl_target[i] - track_slew,
                                ctl.fl_target[i] + track_slew))
                            desired = (ctl.fr_target[i]
                                       + (fr_pose_extra[i] - ctl.fr_target[i])
                                       * 0.10)
                            ctl.fr_target[i] = float(np.clip(
                                desired,
                                ctl.fr_target[i] - track_slew,
                                ctl.fr_target[i] + track_slew))
                    if side_x != last_side:
                        which = "左爪(FL)" if side_x < 0 else "右爪(FR)"
                        print(f"[INFO] 手 px={px:.2f} py={py:.2f} -> "
                              f"{which} hip={hip_target:.2f} "
                              f"thigh={thigh_target:.2f} calf={calf_target:.2f}",
                              file=sys.stderr)
                        last_side = side_x
            if hand_pos is None:
                # 手离开：全部回低增益（不发力），连续计数超过阈值才恢复
                ctl.active_kp = args.hold_kp
                ctl.active_kd = args.hold_kd
                ctl.active_kp_fr = args.hold_kp
                ctl.active_kp_fl = args.hold_kp
                hand_lost_frames += 1
                if hand_lost_frames == hand_lost_required:
                    print(f"[INFO] 手已离开 {hand_lost_frames} 帧，"
                          f"恢复收回", file=sys.stderr)
                if hand_lost_frames >= hand_lost_required:
                    break
            else:
                # 手在：只给追手那条腿高增益，另一条前腿保持低增益（降发热）
                ctl.active_kp = args.kp
                ctl.active_kd = args.kd
                ctl.hand_frames += 1
                if side_x < 0:
                    ctl.active_kp_fl = args.kp
                    ctl.active_kp_fr = args.hold_kp
                else:
                    ctl.active_kp_fr = args.kp
                    ctl.active_kp_fl = args.hold_kp
                hand_lost_frames = 0
            if not ctl.check(measured):
                break
            if args.hold_vmc_scale > 0.0:
                tau, _ = ctl.realtime_balance(measured, imu_q, ctl.targets())
                tau = [t * args.hold_vmc_scale for t in tau]
            else:
                # 纯 PD：趴姿下 VMC 无接触脚恒为零输出（vmc_balance.py
                # support_foot_torques 空接触直接返回 0），白算 500Hz
                # MuJoCo 只费 CPU -> 跳过，控制更稳更省电
                tau = [0.0] * 12
            publisher.Write(ctl.build_cmd(ctl.targets(), tau))
            cycle += 1
            if cycle % 500 == 0:
                print(f"[INFO] 保持 t={cycle * 0.002:.1f}s FR={[round(v, 3) for v in ctl.fr_target]} "
                      f"FL={[round(v, 3) for v in ctl.fl_target]}",
                      file=sys.stderr)
            time.sleep(0.002)
        stages.end("track")
        stages.start("retreat")
        print(f"[INFO] 保持阶段 {cycle} 次（约 {cycle * 0.002:.1f}s）")
        if hand_stream is not None:
            print(f"[INFO] UDP 流: 收 {hand_stream.count} 包")
            hand_stream.close()

        # 收回双爪（到位即停，最多 8s 兜底）
        print("[INFO] 收回 FR+FL（回冻结姿态）...")
        ctl.active_kp = args.hold_kp
        ctl.active_kd = args.hold_kd
        ctl.active_kp_fr = args.hold_kp
        ctl.active_kp_fl = args.hold_kp
        retract_until = time.monotonic() + 8.0
        converged_ticks = 0
        retract_start = time.monotonic()
        while time.monotonic() < retract_until and not ctl.aborted:
            msg = shared["msg"]
            if msg is None:
                continue
            ctl.low_state = msg
            measured = [float(msg.motor_state[i].q) for i in range(12)]
            imu_q = tuple(float(v) for v in msg.imu_state.quaternion)
            for i in range(3):
                desired = (ctl.fr_target[i]
                           + (ctl.freeze_q[FR_JOINT_INDICES[i]]
                              - ctl.fr_target[i]) * 0.02)
                ctl.fr_target[i] = float(np.clip(
                    desired,
                    ctl.fr_target[i] - args.slew_limit_rad,
                    ctl.fr_target[i] + args.slew_limit_rad))
                desired = (ctl.fl_target[i]
                           + (ctl.freeze_q[FL_JOINT_INDICES[i]]
                              - ctl.fl_target[i]) * 0.02)
                ctl.fl_target[i] = float(np.clip(
                    desired,
                    ctl.fl_target[i] - args.slew_limit_rad,
                    ctl.fl_target[i] + args.slew_limit_rad))
            if not ctl.check(measured):
                break
            if args.hold_vmc_scale > 0.0:
                tau, _ = ctl.realtime_balance(measured, imu_q, ctl.targets())
                tau = [t * args.hold_vmc_scale for t in tau]
            else:
                tau = [0.0] * 12
            publisher.Write(ctl.build_cmd(ctl.targets(), tau))
            err = max(
                abs(ctl.fr_target[i] - ctl.freeze_q[FR_JOINT_INDICES[i]])
                for i in range(3))
            err = max(err, max(
                abs(ctl.fl_target[i] - ctl.freeze_q[FL_JOINT_INDICES[i]])
                for i in range(3)))
            converged_ticks = converged_ticks + 1 if err < 0.01 else 0
            if converged_ticks >= 150:  # 0.3s 稳定即停
                break
            time.sleep(0.002)
        print(f"[INFO] FR+FL 已收回（用时 {time.monotonic() - retract_start:.1f}s）")
    except Exception as error:
        print(f"[ERROR] 主流程异常: {error}", file=sys.stderr)
        ctl.aborted = True
        ctl.abort_reason = f"EXCEPTION {type(error).__name__}: {error}"

    # safe release（计时：找出收尾链路瓶颈）
    recovery_start = time.monotonic()
    print("[INFO] 收尾：若已在冻结趴姿直接恢复控制，否则斜坡回 LIE_Q ...")
    start_q = list(ctl.freeze_q)
    if shared.get("msg") is not None:
        try:
            start_q = [float(shared["msg"].motor_state[i].q)
                       for i in range(12)]
        except Exception:
            pass
    dist_to_freeze = max(
        abs(start_q[i] - ctl.freeze_q[i]) for i in range(12))
    if publisher is None:
        print("[WARN] publisher 未创建，请用遥控器让狗趴下", file=sys.stderr)
    elif dist_to_freeze < 0.05:
        # 正常手离开路径：收回后已在冻结趴姿（官方趴卧位），跳过回退斜坡
        publisher.Write(ctl.build_stop_cmd())
        print(f"[INFO] 已在冻结趴姿(偏差 {dist_to_freeze:.3f}rad)，"
              f"跳过回退，已发 PosStopF")
    else:
        # 异常中止路径：斜坡回标准趴姿，限速 ~1 rad/s
        max_dist = max(abs(LIE_Q[i] - start_q[i]) for i in range(12))
        descend_steps = max(150, int(round(max_dist * 500)))
        for step in range(1, descend_steps + 1):
            p = step / descend_steps
            targets = [start_q[i] + (LIE_Q[i] - start_q[i]) * p
                       for i in range(12)]
            publisher.Write(ctl.build_cmd(targets, [0.0] * 12))
            time.sleep(0.002)
        for _ in range(125):
            publisher.Write(ctl.build_cmd(LIE_Q, [0.0] * 12))
            time.sleep(0.002)
        publisher.Write(ctl.build_stop_cmd())
        print(f"[INFO] 已回退 LIE_Q（{descend_steps * 0.002:.1f}s）"
              f"并发送 PosStopF")
    time.sleep(0.2)
    t_mode = time.monotonic()
    restored = False
    select_mode_code = None
    for attempt in range(12):
        try:
            code, _ = msc.SelectMode("mcf")
            select_mode_code = int(code)
            print(f"[INFO] SelectMode('mcf') code={code} (attempt {attempt + 1})")
            if code == 0:
                restored = True
                break
        except Exception as error:
            print(f"[WARN] SelectMode 异常: {error}", file=sys.stderr)
        time.sleep(0.5)
    if not restored:
        print("[WARN] SelectMode('mcf') 未成功；请用遥控器接管",
              file=sys.stderr)
    print(f"[INFO] SelectMode 阶段用时 {time.monotonic() - t_mode:.1f}s")
    t_check = time.monotonic()
    check_ok = False
    check_mode_name = ""
    for _ in range(20):
        time.sleep(0.1)
        try:
            _, result = msc.CheckMode()
        except Exception:
            continue
        if result and result.get("name"):
            check_ok = True
            check_mode_name = str(result.get("name"))
            break
    stages.end("retreat")
    print(f"[INFO] 运动服务已恢复（CheckMode 用时 "
          f"{time.monotonic() - t_check:.1f}s），执行 {args.restore_action}() ...")
    action_fn = (sport.StandDown if args.restore_action == "standdown"
                 else sport.StandUp)
    t_action = time.monotonic()
    action_ok = False
    restore_code = None
    for attempt in range(5):
        try:
            code = action_fn()
            restore_code = int(code)
            print(f"  code: {code} (attempt {attempt + 1})")
            if code == 0:
                action_ok = True
                break
        except Exception as error:
            print(f"[WARN] {args.restore_action} 异常: {error}",
                  file=sys.stderr)
        time.sleep(1.0)
    recovery_ok = restored and check_ok and action_ok
    retreat_evidence = {
        "retreat_completed": recovery_ok,
        "select_mode_code": select_mode_code,
        "check_mode_name": check_mode_name,
        "restore_code": restore_code,
        "failure": "" if recovery_ok else ctl.abort_reason,
        "steps": [],
    }
    print(f"[INFO] {args.restore_action} 阶段用时 "
          f"{time.monotonic() - t_action:.1f}s；"
          f"收尾总用时 {time.monotonic() - recovery_start:.1f}s"
          f"（不含手离开确认 3s）")
    temps = []
    try:
        temps = [float(ctl.low_state.motor_state[i].temperature)
                 for i in range(12)]
    except Exception:
        pass
    if temps:
        print(f"[INFO] 测试后温度 max={max(temps):.0f}C。"
              f"{'狗已趴下散热' if args.restore_action == 'standdown' else '狗已站立（用户要求）；如温度偏高建议尽快趴下/关机'}",
              file=sys.stderr)

    # ---- 真机试验记录（schema v2，纯函数构造，见 trial_record.py）----
    try:
        from datetime import datetime as _dt
        pass  # v4 imported at module level
        _t_end_wall = int(time.time() * 1000)
        _t_end_mono = int(time.monotonic() * 1000)
        _code_sha = sha256_file(__file__)
        _schema_sha = sha256_file(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "trial_record_v4.py"))
        _model_sha = sha256_file(args.policy) if getattr(args, "policy", "") else "none"
        trial = build_trial_record_v4(
            trial_id=f"{args.session_id or 'm8'}-{args.trial_index:03d}",
            session_id=args.session_id or f"m8-{_t0_wall_ms}",
            trial_index=args.trial_index,
            execution_status="aborted" if getattr(ctl, "abort_reason", "") else "ok",
            abort_reason=getattr(ctl, "abort_reason", ""),
            wall_clock_source="system-utc",
            started_at_wall_ms=_t0_wall_ms, ended_at_wall_ms=_t_end_wall,
            started_at_monotonic_ms=_t0_mono_ms, ended_at_monotonic_ms=_t_end_mono,
            clock_sync={"robot_clock_offset_ms": None, "robot_clock_uncertainty_ms": None,
                        "video_clock_offset_ms": None, "video_clock_uncertainty_ms": None,
                        "sync_method": "not_measured", "estimated_at_wall_ms": None},
            stages=stages.dump(), event_trace=trace.dump(),
            identity={"commit": args.commit, "code_sha256": _code_sha,
                      "model_sha256": _model_sha, "schema_sha256": _schema_sha,
                      "firmware": args.firmware, "calibration_id": args.calibration_id,
                      "floor": args.floor, "light": args.light,
                      "deploy_params": {"kp": getattr(args, "kp", 0),
                                        "contact_hold_s": getattr(args, "contact_hold_s", 0.6)},
                      "human_intervention": []},
            contact={"contact_hold_s": getattr(args, "contact_hold_s", 0.6),
                     "detector_trigger_count": getattr(ctl, "contact_count", 0),
                     "contact_ground_truth": "not_measured",
                     "contact_confirmation_source": "none"},
            endpoints={"reach_success": "not_measured",
                       "handshake_success": "not_measured"},
            paw_selection={"expected_paw": "not_measured", "selected_paw": "not_measured",
                           "paw_selected_correctly": "not_measured"},
            safety_retreat={"retreat_completed": bool(recovery_ok),
                            "select_mode_code": select_mode_code,
                            "check_mode_name": check_mode_name,
                            "restore_code": restore_code,
                            "safety_alarms": []},
        )
        out_path = write_trial_record_v4(
            trial, getattr(args, "trial_log_dir", "evidence/real_trials"))
        print(f"[EVIDENCE] {out_path}")
    except Exception as error:
        print(f"[WARN] 试验记录写入失败: {error}", file=sys.stderr)

    if ctl.aborted:
        print(f"[RESULT] M8_ABORTED reason={ctl.abort_reason}")
        return 3
    print(f"[RESULT] M8_OK max_track_err={ctl.max_track_err:.4f} rad "
          f"joint={ctl.max_track_err_joint} "
          f"pitch={ctl.max_pitch_deg:.2f}deg")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-interface", required=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--site-check", action="store_true")
    parser.add_argument("--hold-seconds", type=float, default=120.0,
                        help="保持阶段安全上限（秒）；手在期间一直保持，"
                             "手离开 --hand-lost-frames 帧才恢复")
    parser.add_argument("--hand-lost-frames", type=int, default=1500,
                        help="手离开多少帧（约 2ms/帧）后恢复收回；"
                             "默认 1500 帧 ≈ 3 秒，防视觉抖动")
    parser.add_argument("--reach-seconds", type=float, default=4.0,
                        help="伸直阶段时长（slew 0.008 下 2.1rad 约 1s，"
                             "4s 余量）")
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--kp", type=float, default=150.0,
                        help="追手时前腿高增益（发力）")
    parser.add_argument("--kd", type=float, default=10.0)
    parser.add_argument("--hold-kp", type=float, default=40.0,
                        help="保持阶段前腿增益（打直后不发力防振动；"
                             "检测到手才切高增益）")
    parser.add_argument("--hold-kd", type=float, default=4.0)
    parser.add_argument("--track-gain", type=float, default=0.10,
                        help="追手步进增益（大=更快更灵活）")
    parser.add_argument("--slew-limit-rad", type=float, default=0.008,
                        help="伸直阶段每 tick 关节目标最大变化（快）")
    parser.add_argument("--track-slew-rad", type=float, default=0.004,
                        help="追手阶段每 tick 关节目标最大变化（慢=丝滑）")
    parser.add_argument("--track-slew-fast-rad", type=float, default=0.012,
                        help="A1 双模追手：爪子离手目标远时的大步 slew"
                             "（敏捷-精准切换的'敏捷'档）")
    parser.add_argument("--slew-switch-rad", type=float, default=0.15,
                        help="A1 双模追手：关节空间距离超过此值切快 slew")
    parser.add_argument("--predict-latency", type=float, default=0.0,
                        help="A2：Kalman 前推延迟补偿秒数（0=关闭预测，"
                             "纯 EMA 旧行为；建议 0.08~0.15）")
    parser.add_argument("--predict-max-lost", type=float, default=0.25,
                        help="A3：手丢失后按最后速度外推的最长秒数，"
                             "超时才走'手离开 3s 收回'逻辑")
    parser.add_argument("--contact-hold-s", type=float, default=0.0,
                        help="A5：爪子碰到手（tau_est 尖峰）后暂停追手保持"
                             "的秒数；0=关闭接触检测")
    parser.add_argument("--policy", default="",
                        help="RL 策略 checkpoint 路径（prone_px MLP）；"
                             "空=用手工几何映射")
    parser.add_argument("--policy-hz", type=float, default=50.0,
                        help="策略推理频率（与训练一致，50Hz）")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--trial-index", type=int, default=1)
    parser.add_argument("--calibration-id", default=None)
    parser.add_argument("--floor", default="")
    parser.add_argument("--light", default="")
    parser.add_argument("--commit", default="")
    parser.add_argument("--firmware", default="mcf")
    parser.add_argument("--trial-log-dir", default="evidence/real_trials",
                        help="真机试验记录 JSON 输出目录")
    parser.add_argument("--vmc-kp-rot", type=float, default=300.0)
    parser.add_argument("--vmc-kd-rot", type=float, default=30.0)
    parser.add_argument("--hold-vmc-scale", type=float, default=0.0,
                        help="保持/追手/收回阶段 VMC 前馈缩放。默认 0=纯 PD"
                             "（趴姿下 VMC 无接触脚恒零输出，跳过省 CPU、"
                             "降发热；不稳时设 0.5~1.0 恢复）")
    parser.add_argument("--tau-clip", type=float, default=50.0)
    parser.add_argument("--hand-port", type=int, default=0,
                        help="UDP 手位置流端口（M3 视觉进程）；0=不接视觉")
    parser.add_argument("--max-start-temp", type=float, default=70.0,
                        help="起始温度门（用户要求 70C；看门狗 80C 硬上限"
                             "不变，注意余量）")
    parser.add_argument("--restore-action", default="standup",
                        choices=("standup", "standdown"),
                        help="结束后恢复动作：默认 standup 站立收工；"
                             "担心发热可选 standdown 趴下散热")
    args = parser.parse_args(argv)
    if args.dry_run and not args.network_interface:
        args.network_interface = "dry-run-iface"
    if not args.network_interface:
        parser.error("--network-interface 必填（dry-run 可任意值）")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
