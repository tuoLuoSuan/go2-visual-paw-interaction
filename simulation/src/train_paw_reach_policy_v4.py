#!/usr/bin/env python3
"""趴姿前爪追手策略训练（Playful DoggyBot 方法迁移到本项目 M8 任务）。

任务：GO2 趴姿、双前腿向前打直；策略网络（MLP/GRU）输入本体感受 + 手目标
3D 位置，输出前腿 6 关节目标角增量，PD 执行。手目标在爪可达带内随机游走
（模拟真机"手在动、狗去够"）。

论文对照点：
- 感知-控制解耦：手位置当观测，策略只管关节目标（论文 Fig.2 同款）
- r_pos = exp(-d/alpha)+1 指数距离奖励（论文 Eq.1 同款）
- 延迟注入：观测手位置随机延迟（论文关键 trick，真机成功率显著提升）
- MLP vs GRU 记忆网络对比（论文 Table I 同款实验）
- 能耗正则（论文 energy conservation，也对应本项目温度纪律）

训练在 CPU 上跑（MuJoCo 无 Intel GPU 后端；任务比论文简单得多，
32-64 并行环境足够）。产物 checkpoint 可加载进真机 M8 保持循环替换
手工几何映射（--policy 开关，后续接入）。

用法：
  python simulation/src/train_paw_reach_policy.py --envs 32 --iterations 300
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import mujoco
import torch
import torch.nn as nn

import standing_paw_lift_common as common

WS = Path(__file__).resolve().parent.parent.parent   # repo 根（unitree_mujoco 同级）
SCENE = WS / "unitree_mujoco/unitree_robots/go2/scene.xml"
# ---- 姿态常量（与 real_vmc_reach_m8.py 一致）----
LIE_Q = [-0.06, 1.24, -2.75, 0.08, 1.25, -2.78,
         -0.40, 1.26, -2.77, 0.41, 1.25, -2.78]
REACH_DT = -2.13   # thigh 前摆偏移 -> 水平前伸
REACH_DC = 1.69    # calf 伸直偏移
# 前腿关节范围（真机映射的极限）
HIP_RANGE = 0.35
THIGH_LO, THIGH_HI = -1.55, None   # None=用 LIE 冻结值
CALF_LO, CALF_HI = -2.0, -0.84
# 手目标可达带（关节空间，论文式课程：高度从低到高）
BAND_THIGH = (-1.35, -0.90)   # (min, max)
BAND_CALF = (-1.25, -1.05)
ACTION_SCALE = 0.02   # 每步目标角增量上限（slew 思想）

TIMESTEP = 0.004     # 仿真步长（趴姿任务）
CTRL_HZ = 50         # 策略控制频率
SUBSTEPS = int(round(1.0 / CTRL_HZ / TIMESTEP))   # 20ms / 4ms = 5
TIMESTEP_STAND = 0.002   # 站立任务仿真步长（平衡更敏感，用 2ms）
SUBSTEPS_STAND = int(round(1.0 / CTRL_HZ / TIMESTEP_STAND))  # 10
EPISODE_STEPS = 200  # 4s @ 50Hz

FR_IDX = (0, 1, 2)
FL_IDX = (3, 4, 5)


def pitch_deg(qq):
    w0, x0, y0, z0 = (float(v) for v in qq)
    return math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w0 * y0 - z0 * x0)))))


class PawReachEnv:
    """单个 MuJoCo 环境（向量化：外部循环 N 个实例）。"""

    def __init__(self, ctx, base_qpos, straight_pose, latency_range,
                 reward_scale=0.06, calf_hi=-0.84, torque_noise=0.0,
                 smooth_weight=0.05, joint_weight=0.0):
        self.ctx = ctx
        self.model = ctx.model
        self.data = mujoco.MjData(self.model)
        self.base_qpos = base_qpos          # 趴姿 base qpos（含高度标定）
        self.straight_pose = list(straight_pose)  # 双前腿打直目标角（12 维）
        self.latency_range = latency_range
        # v2 根治参数：奖励尺度越小越逼策略用大臂精准追手；calf_hi 与
        # 真机限位一致（-1.05 避极限振动区）；torque_noise 域随机化抗抖；
        # joint_weight 整臂协同奖励（对齐 M8 几何映射，堵死只动小腿）
        self.reward_scale = float(reward_scale)
        self.calf_hi = float(calf_hi)
        self.torque_noise = float(torque_noise)
        self.smooth_weight = float(smooth_weight)
        self.joint_weight = float(joint_weight)
        self.scratch = mujoco.MjData(self.model)
        self.actuator_ids = tuple(int(ctx.actuator_by_joint[n])
                                  for n in ctx.joint_names)
        self.base = ctx.base_qpos_address
        self._qlo = [0.0] * 12
        self._qhi = [0.0] * 12
        for i in range(12):
            if i in FR_IDX or i in FL_IDX:
                hip_i = i % 3 == 0
                thigh_i = i % 3 == 1
                if hip_i:
                    lo = self.straight_pose[i] - HIP_RANGE
                    hi = self.straight_pose[i] + HIP_RANGE
                elif thigh_i:
                    lo = THIGH_LO
                    hi = self.straight_pose[i]
                else:
                    lo = CALF_LO
                    hi = self.calf_hi
                self._qlo[i], self._qhi[i] = lo, hi
            else:
                self._qlo[i] = self._qhi[i] = self.straight_pose[i]
        self.reset()

    # ---- 手目标（关节空间游走 + FK 到 3D 位置）----
    def _hand_point(self):
        """在可达带内随机采样（hip 偏移, thigh, calf），返回 12 维关节角。"""
        q = list(self.straight_pose)
        side = np.random.rand() < 0.5
        leg = FL_IDX if side else FR_IDX
        hip = self.straight_pose[leg[0]] + (HIP_RANGE * np.random.rand()
                                            if side else -HIP_RANGE * np.random.rand())
        thigh = float(np.random.uniform(*BAND_THIGH))
        calf = float(np.random.uniform(*BAND_CALF))
        q[leg[0]], q[leg[1]], q[leg[2]] = hip, thigh, calf
        return q

    def _fk_foot(self, q):
        """给定关节角（12 维），返回 FR/FL 足端世界坐标（base 趴姿下）。"""
        qpos = np.zeros(self.model.nq, dtype=float)
        for address, value in zip(self.ctx.qpos_addresses, q):
            qpos[address] = float(value)
        qpos[self.base:self.base + 3] = self.base_qpos[self.base:self.base + 3]
        qpos[self.base + 3:self.base + 7] = (1.0, 0.0, 0.0, 0.0)
        self.scratch.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.scratch)
        return {name: np.array(self.scratch.geom_xpos[gid], dtype=float)
                for name, gid in self.ctx.foot_geom_ids.items()}

    def reset(self):
        data = self.data
        data.qpos[:] = self.base_qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, data)
        self.targets = [float(self.straight_pose[i]) for i in range(12)]
        self.joint_point = self._hand_point()      # 关节空间手点
        self.hand_pos = self._fk_foot(self.joint_point)  # 3D 手位置
        self.side = 0 if np.random.rand() < 0.5 else 1   # 0=FR 右爪, 1=FL 左爪
        self.latency = float(np.random.uniform(*self.latency_range))
        self.hist = [self.hand_pos["FL" if self.side == 1
                                    else "FR"].copy()]  # 延迟注入历史
        self.prev_action = np.zeros(6, dtype=float)
        self.t = 0

    def obs(self):
        data = self.data
        q = np.array([float(data.qpos[a]) for a in self.ctx.qpos_addresses])
        dq = np.array([float(data.qvel[a]) for a in self.ctx.dof_addresses])
        # 延迟注入：观测 T_latency 秒前的手位置（论文 trick）
        k = max(1, int(round(self.latency * CTRL_HZ)))
        hand_delayed = self.hist[-k] if len(self.hist) >= k else self.hist[0]
        paw_fr = np.array(data.geom_xpos[self.ctx.foot_geom_ids["FR"]])
        paw_fl = np.array(data.geom_xpos[self.ctx.foot_geom_ids["FL"]])
        rel_fr = hand_delayed - paw_fr
        rel_fl = hand_delayed - paw_fl
        side = np.array([float(self.side)], dtype=float)
        return np.concatenate([q / math.pi, dq / 10.0, rel_fr, rel_fl,
                               hand_delayed - np.array(self.base_qpos[
                                   self.base:self.base + 3]), side])

    def step(self, action_delta):
        """执行一次控制（20ms = 5 个 4ms 子步）。返回 obs, reward, done。"""
        data = self.data
        targets = list(self.targets)
        for i in range(6):
            # 动作只作用于前腿 6 关节（i<3 -> FR，i>=3 -> FL）
            j = FR_IDX[i] if i < 3 else FL_IDX[i - 3]
            targets[j] = float(np.clip(targets[j] + action_delta[i],
                                       self._qlo[j], self._qhi[j]))
        self.targets = targets

        tau_sum = 0.0
        for _ in range(SUBSTEPS):
            q = [float(data.qpos[a]) for a in self.ctx.qpos_addresses]
            dq = [float(data.qvel[a]) for a in self.ctx.dof_addresses]
            tau = [60.0 * (self.straight_pose[i] - q[i]) + 5.0 * (-dq[i])
                   for i in range(12)]
            for i, j in enumerate(FR_IDX):
                tau[j] = 250.0 * (targets[j] - q[j]) + 15.0 * (-dq[j])
            for i, j in enumerate(FL_IDX):
                tau[j] = 250.0 * (targets[j] - q[j]) + 15.0 * (-dq[j])
            for i in range(6, 12):
                tau[i] = 30.0 * (self.straight_pose[i] - q[i]) + 4.0 * (-dq[i])
            tau = [max(-60.0, min(60.0, v)) for v in tau]
            if self.torque_noise > 0.0:
                tau = [v + np.random.uniform(-self.torque_noise,
                                             self.torque_noise) for v in tau]
                tau = [max(-60.0, min(60.0, v)) for v in tau]
            for aid, v in zip(self.actuator_ids, tau):
                data.ctrl[aid] = v
            mujoco.mj_step(self.model, data)
            tau_sum += sum(v * v for v in tau) / SUBSTEPS

        # 手目标：关节空间 OU 游走，保持可达带内
        jp = list(self.joint_point)
        leg = FL_IDX if self.side == 1 else FR_IDX
        jp[leg[1]] = float(np.clip(jp[leg[1]]
                                   + np.random.normal(0.0, 0.02),
                                   *BAND_THIGH))
        jp[leg[2]] = float(np.clip(jp[leg[2]]
                                   + np.random.normal(0.0, 0.02),
                                   *BAND_CALF))
        jp[leg[0]] = float(np.clip(jp[leg[0]]
                                   + np.random.normal(0.0, 0.01),
                                   self._qlo[leg[0]], self._qhi[leg[0]]))
        self.joint_point = jp
        self.hand_pos = self._fk_foot(jp)

        paw = np.array(data.geom_xpos[self.ctx.foot_geom_ids[
            "FL" if self.side == 1 else "FR"]])
        dist = float(np.linalg.norm(paw - self.hand_pos["FL" if self.side == 1
                                                         else "FR"]))
        # 论文式奖励（v2：reward_scale 收窄逼精准追手 -> 逼大臂参与）
        r_pos = math.exp(-dist / self.reward_scale) + 1.0
        r_smooth = -self.smooth_weight * float(np.sum(action_delta ** 2))
        r_energy = -1e-4 * tau_sum
        # 非活动腿保持打直
        other_leg = FR_IDX if self.side == 1 else FL_IDX
        dev_other = max(abs(self.targets[j] - self.straight_pose[j])
                        for j in other_leg)
        r_keep = -0.5 * dev_other
        reward = r_pos + r_smooth + r_energy + r_keep

        # 终止：趴姿坍塌（pitch 过大 / base 掉太低）
        pitch = pitch_deg(tuple(float(v)
                                for v in data.qpos[self.base + 3:self.base + 7]))
        base_z = float(data.qpos[self.base + 2])
        done = abs(pitch) > 40.0 or base_z < 0.05
        if done:
            reward -= 10.0
        self.t += 1
        done = done or self.t >= EPISODE_STEPS
        self.hist.append(self.hand_pos[
            "FL" if self.side == 1 else "FR"].copy())
        self.prev_action = action_delta.copy()
        return self.obs(), reward, done, {"dist": dist, "pitch": pitch}

    def done_info(self):
        return {"dist": float(np.linalg.norm(
            np.array(self.data.geom_xpos[self.ctx.foot_geom_ids[
                "FL" if self.side == 1 else "FR"]])
            - self.hand_pos["FL" if self.side == 1 else "FR"]))}


# ---- 站立追手任务（第二关：需要学平衡）----
STAND_Q = (0.0, 0.8, -1.5) * 4        # 官方站立关节角
# 手目标带限定在站立爪高及以上（握手高度）：thigh 0.2~-0.6 为前伸抬爪
# 姿态。若包含贴地目标，策略会学"下蹲够手"作弊（蹲下去离手更近）
STAND_BAND_THIGH = (-0.6, 0.2)
STAND_BAND_CALF = (-2.0, -1.5)
# 站立关节限位（【绝对角度】不是偏移量！站立位 thigh=0.8/calf=-1.5，
# 允许抬爪前摆到 thigh -0.6、伸小腿到 calf -2.0）
STAND_LIMITS = {0: (-0.5, 0.5), 1: (-0.6, 1.2), 2: (-2.2, -0.8)}   # 按关节模 3


class StandingPawReachEnv:
    """GO2 站立 + 单前爪追踪手目标（论文同款'平衡+精准'任务）。"""

    def __init__(self, ctx, stand_qpos, latency_range, substeps,
                 walk_scale=1.0, balance_weight=1.0, init_noise=0.02,
                 reward_scale=0.06, joint_weight=0.0, tau_clip=60.0,
                 mask_active_only=False):
        self.ctx = ctx
        self.model = ctx.model
        self.data = mujoco.MjData(self.model)
        self.stand_qpos = stand_qpos      # 含标定后 base 高度的站立 qpos
        self.latency_range = latency_range
        self.substeps = substeps
        self.walk_scale = float(walk_scale)      # 手移动幅度（0=固定手）
        self.balance_weight = float(balance_weight)  # 平衡奖励权重
        self.init_noise = float(init_noise)      # 初始关节扰动幅度
        self.reward_scale = float(reward_scale)  # 追手奖励尺度
        self.joint_weight = float(joint_weight)  # 整臂协同奖励权重
        self.tau_clip = float(tau_clip)  # 力矩钳位（v3 对齐真机）
        self.mask_active_only = bool(mask_active_only)  # 部署模式：只动活动腿
        self.scratch = mujoco.MjData(self.model)
        self.actuator_ids = tuple(int(ctx.actuator_by_joint[n])
                                  for n in ctx.joint_names)
        self.base = ctx.base_qpos_address
        self.stand_joints = [float(stand_qpos[a])
                             for a in ctx.qpos_addresses]
        self.z0 = float(stand_qpos[self.base + 2])
        self._qlo = [0.0] * 12
        self._qhi = [0.0] * 12
        for i in range(12):
            # 绝对角度限位（旧版错当成偏移量：站立 calf -1.5 被裁到
            # [-3.7,-2.3] 全是更弯的蹲姿 -> 策略被迫下蹲）
            lo, hi = STAND_LIMITS[i % 3]
            self._qlo[i] = lo
            self._qhi[i] = hi
        self.reset()

    def _hand_point(self):
        q = list(self.stand_joints)
        side = np.random.rand() < 0.5
        leg = FL_IDX if side else FR_IDX
        q[leg[0]] = self.stand_joints[leg[0]] + (
            0.3 * np.random.rand() if side else -0.3 * np.random.rand())
        q[leg[1]] = float(np.random.uniform(*STAND_BAND_THIGH))
        q[leg[2]] = float(np.random.uniform(*STAND_BAND_CALF))
        return q

    def _fk_foot(self, q):
        qpos = np.zeros(self.model.nq, dtype=float)
        for address, value in zip(self.ctx.qpos_addresses, q):
            qpos[address] = float(value)
        qpos[self.base:self.base + 3] = self.stand_qpos[self.base:self.base + 3]
        qpos[self.base + 3:self.base + 7] = (1.0, 0.0, 0.0, 0.0)
        self.scratch.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.scratch)
        return {name: np.array(self.scratch.geom_xpos[gid], dtype=float)
                for name, gid in self.ctx.foot_geom_ids.items()}

    def reset(self):
        data = self.data
        data.qpos[:] = self.stand_qpos
        data.qvel[:] = 0.0
        # 初始微扰：小角度 + 小噪声（鲁棒性）
        for a in self.ctx.qpos_addresses:
            data.qpos[a] += np.random.uniform(-self.init_noise,
                                              self.init_noise)
        data.qpos[self.base + 3:self.base + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.model, data)
        self.targets = [float(data.qpos[a]) for a in self.ctx.qpos_addresses]
        self.joint_point = self._hand_point()
        self.hand_pos = self._fk_foot(self.joint_point)
        self.side = 0 if np.random.rand() < 0.5 else 1
        self.latency = float(np.random.uniform(*self.latency_range))
        self.hist = [self.hand_pos["FL" if self.side == 1
                                    else "FR"].copy()]
        self.prev_action = np.zeros(12, dtype=float)
        self.t = 0

    def obs(self):
        data = self.data
        q = np.array([float(data.qpos[a]) for a in self.ctx.qpos_addresses])
        dq = np.array([float(data.qvel[a]) for a in self.ctx.dof_addresses])
        k = max(1, int(round(self.latency * CTRL_HZ)))
        hand_delayed = self.hist[-k] if len(self.hist) >= k else self.hist[0]
        paw_fr = np.array(data.geom_xpos[self.ctx.foot_geom_ids["FR"]])
        paw_fl = np.array(data.geom_xpos[self.ctx.foot_geom_ids["FL"]])
        pitch, roll = self._base_rpy()
        side = np.array([float(self.side)], dtype=float)
        return np.concatenate([q / math.pi, dq / 10.0,
                               np.array([pitch, roll]),
                               hand_delayed - paw_fr, hand_delayed - paw_fl,
                               hand_delayed - np.array(self.stand_qpos[
                                   self.base:self.base + 3]), side])

    def _base_rpy(self):
        qq = tuple(float(v) for v in self.data.qpos[self.base + 3:self.base + 7])
        w0, x0, y0, z0 = qq
        roll = math.atan2(2.0 * (w0 * x0 + y0 * z0),
                          1.0 - 2.0 * (x0 * x0 + y0 * y0))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w0 * y0 - z0 * x0))))
        return pitch, roll

    def step(self, action_delta):
        data = self.data
        targets = list(self.targets)
        for i in range(12):
            targets[i] = float(np.clip(targets[i] + action_delta[i],
                                       self._qlo[i], self._qhi[i]))
        self.targets = targets

        tau_sum = 0.0
        for _ in range(self.substeps):
            q = [float(data.qpos[a]) for a in self.ctx.qpos_addresses]
            dq = [float(data.qvel[a]) for a in self.ctx.dof_addresses]
            tau = [0.0] * 12
            for i in range(12):
                # 支撑腿高刚度（模拟真机固件刚性位置控制）：
                # 纯 PD kp=80 时策略会学"下蹲够手"作弊，kp=300 站姿近乎
                # 锁定，策略只能学站立伸手（零动作测试：kp300 站 0.4s
                # 只下沉 2mm）
                kp, kd = 300.0, 10.0
                if i in (FR_IDX if self.side == 0 else FL_IDX):
                    kp, kd = 150.0, 10.0    # 活动腿（抬爪）
                tau[i] = kp * (targets[i] - q[i]) + kd * (-dq[i])
            tau = [max(-60.0, min(60.0, v)) for v in tau]
            for aid, v in zip(self.actuator_ids, tau):
                data.ctrl[aid] = v
            mujoco.mj_step(self.model, data)
            tau_sum += sum(v * v for v in tau) / self.substeps

        # 手目标关节空间 OU 游走
        jp = list(self.joint_point)
        leg = FL_IDX if self.side == 1 else FR_IDX
        jp[leg[1]] = float(np.clip(jp[leg[1]] + np.random.normal(0.0, 0.02)
                                   * self.walk_scale, *STAND_BAND_THIGH))
        jp[leg[2]] = float(np.clip(jp[leg[2]] + np.random.normal(0.0, 0.02)
                                   * self.walk_scale, *STAND_BAND_CALF))
        jp[leg[0]] = float(np.clip(jp[leg[0]] + np.random.normal(0.0, 0.01)
                                   * self.walk_scale,
                                   self._qlo[leg[0]], self._qhi[leg[0]]))
        self.joint_point = jp
        self.hand_pos = self._fk_foot(jp)

        paw = np.array(data.geom_xpos[self.ctx.foot_geom_ids[
            "FL" if self.side == 1 else "FR"]])
        dist = float(np.linalg.norm(paw - self.hand_pos["FL" if self.side == 1
                                                         else "FR"]))
        r_pos = math.exp(-dist / 0.06) + 1.0
        pitch, roll = self._base_rpy()
        r_balance = -0.3 * (abs(pitch) + abs(roll))
        base_z = float(data.qpos[self.base + 2])
        r_height = -1.5 * abs(base_z - self.z0)
        r_smooth = -0.05 * float(np.sum(action_delta ** 2))
        r_energy = -1e-4 * tau_sum
        # 非活动前腿保持站立位
        other_leg = FR_IDX if self.side == 1 else FL_IDX
        dev_other = max(abs(self.targets[j] - self.stand_joints[j])
                        for j in other_leg)
        r_keep = -0.3 * dev_other
        r_alive = 0.5    # 存活奖励：让策略先认识到"活着就有分"
        reward = r_pos + r_smooth + r_energy + r_keep + r_alive \
            + (r_balance + r_height) * self.balance_weight

        # 下沉 3.5cm 即判死：不给"下蹲够手"留任何收益空间
        done = base_z < self.z0 - 0.035 or abs(pitch) > 0.6 or abs(roll) > 0.6
        if done:
            reward -= 10.0
        self.t += 1
        done = done or self.t >= EPISODE_STEPS
        self.hist.append(self.hand_pos[
            "FL" if self.side == 1 else "FR"].copy())
        self.prev_action = action_delta.copy()
        return self.obs(), reward, done, {"dist": dist,
                                          "pitch": math.degrees(pitch)}

    def done_info(self):
        return {"dist": float(np.linalg.norm(
            np.array(self.data.geom_xpos[self.ctx.foot_geom_ids[
                "FL" if self.side == 1 else "FR"]])
            - self.hand_pos["FL" if self.side == 1 else "FR"]))}


def obs_dim_for(task):
    if task == "standing":
        return 36     # 站立多 pitch/roll 两项
    if task == "standing_px":
        return 29     # 12q + 12dq + pitch/roll + px + py + side
    if task == "prone_px":
        return 27     # 12q + 12dq + px + py + side（与真机视觉接口一致）
    return 34         # prone：3D 手位置版本


def act_dim_for(task):
    return 6 if task in ("prone", "prone_px") else 12


# ---- 像素坐标版趴姿任务（部署专用：观测直接吃真机视觉的 px,py）----
def m8_px_to_joints(px, py, straight_pose):
    """M8 真机标定几何：像素 (px,py) -> 活动腿关节目标。与 real_vmc_reach_m8.py
    保持完全一致（side_x 镜像、amp_y 抬升标定 0.60/0.40、thigh 0.45/calf 0.20）。"""
    side_x = -1.0 if px < 0.5 else 1.0
    amp_x = abs(px - 0.5) * 2.0
    amp_y = float(np.clip((0.60 - py) / 0.40, 0.0, 1.0))
    hip = 0.35 * amp_x
    thigh = -0.90 - 0.45 * amp_y
    calf = -1.05 - 0.20 * amp_y
    q = list(straight_pose)
    leg = FL_IDX if side_x < 0 else FR_IDX
    q[leg[0]] = straight_pose[leg[0]] + (hip if side_x < 0 else -hip)
    q[leg[1]] = thigh
    q[leg[2]] = calf
    return q


class PronePxEnv(PawReachEnv):
    """趴姿 + 像素坐标观测：与真机接口零缝隙（部署版）。"""

    def __init__(self, ctx, base_qpos, straight_pose, latency_range,
                 walk_scale=1.0, reward_scale=0.06, calf_hi=-0.84,
                 torque_noise=0.0, smooth_weight=0.05, joint_weight=0.0):
        super().__init__(ctx, base_qpos, straight_pose, latency_range,
                         reward_scale=reward_scale, calf_hi=calf_hi,
                         torque_noise=torque_noise,
                         smooth_weight=smooth_weight,
                         joint_weight=joint_weight)
        self.walk_scale = float(walk_scale)

    def _hand_point(self):
        # 像素空间手点（左右映射已含在 m8_px_to_joints 里）
        self.px = float(np.random.uniform(0.15, 0.85))
        self.py = float(np.random.uniform(0.15, 0.55))
        return m8_px_to_joints(self.px, self.py, self.straight_pose)

    def reset(self):
        self.px = float(np.random.uniform(0.15, 0.85))
        self.py = float(np.random.uniform(0.15, 0.55))
        self.side = 0 if self.px >= 0.5 else 1
        self.joint_point = m8_px_to_joints(self.px, self.py,
                                           self.straight_pose)
        self.hand_pos = self._fk_foot(self.joint_point)
        self.latency = float(np.random.uniform(*self.latency_range))
        self.hist = [self.hand_pos["FL" if self.side == 1
                                    else "FR"].copy()]
        self.hist_px = [self.px]
        self.hist_py = [self.py]
        # 复用父类 reset 的物理部分
        data = self.data
        data.qpos[:] = self.base_qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, data)
        self.targets = [float(self.straight_pose[i]) for i in range(12)]
        self.prev_action = np.zeros(6, dtype=float)
        self.t = 0

    def _step_hand(self):
        # 像素空间 OU 游走
        self.px = float(np.clip(self.px + np.random.normal(0.0, 0.05)
                                * self.walk_scale, 0.05, 0.95))
        self.py = float(np.clip(self.py + np.random.normal(0.0, 0.03)
                                * self.walk_scale, 0.05, 0.75))
        self.side = 0 if self.px >= 0.5 else 1
        self.joint_point = m8_px_to_joints(self.px, self.py,
                                           self.straight_pose)
        self.hand_pos = self._fk_foot(self.joint_point)

    def obs(self):
        data = self.data
        q = np.array([float(data.qpos[a]) for a in self.ctx.qpos_addresses])
        dq = np.array([float(data.qvel[a]) for a in self.ctx.dof_addresses])
        k = max(1, int(round(self.latency * CTRL_HZ)))
        px_d = self.hist_px[-k] if len(self.hist_px) >= k else self.hist_px[0]
        py_d = self.hist_py[-k] if len(self.hist_py) >= k else self.hist_py[0]
        side = np.array([float(self.side)], dtype=float)
        return np.concatenate([q / math.pi, dq / 10.0,
                               np.array([px_d, py_d]), side])

    def step(self, action_delta):
        # 与 PawReachEnv.step 相同，但手走像素空间、追加像素历史
        data = self.data
        targets = list(self.targets)
        for i in range(6):
            j = FR_IDX[i] if i < 3 else FL_IDX[i - 3]
            targets[j] = float(np.clip(targets[j] + action_delta[i],
                                       self._qlo[j], self._qhi[j]))
        self.targets = targets
        tau_sum = 0.0
        for _ in range(SUBSTEPS):
            q = [float(data.qpos[a]) for a in self.ctx.qpos_addresses]
            dq = [float(data.qvel[a]) for a in self.ctx.dof_addresses]
            tau = [60.0 * (self.straight_pose[i] - q[i]) + 5.0 * (-dq[i])
                   for i in range(12)]
            for i, j in enumerate(FR_IDX):
                tau[j] = 250.0 * (targets[j] - q[j]) + 15.0 * (-dq[j])
            for i, j in enumerate(FL_IDX):
                tau[j] = 250.0 * (targets[j] - q[j]) + 15.0 * (-dq[j])
            for i in range(6, 12):
                tau[i] = 30.0 * (self.straight_pose[i] - q[i]) + 4.0 * (-dq[i])
            tau = [max(-60.0, min(60.0, v)) for v in tau]
            if self.torque_noise > 0.0:
                tau = [v + np.random.uniform(-self.torque_noise,
                                             self.torque_noise) for v in tau]
                tau = [max(-60.0, min(60.0, v)) for v in tau]
            for aid, v in zip(self.actuator_ids, tau):
                data.ctrl[aid] = v
            mujoco.mj_step(self.model, data)
            tau_sum += sum(v * v for v in tau) / SUBSTEPS
        self._step_hand()
        paw = np.array(data.geom_xpos[self.ctx.foot_geom_ids[
            "FL" if self.side == 1 else "FR"]])
        dist = float(np.linalg.norm(paw - self.hand_pos["FL" if self.side == 1
                                                         else "FR"]))
        r_pos = math.exp(-dist / self.reward_scale) + 1.0
        r_smooth = -self.smooth_weight * float(np.sum(action_delta ** 2))
        r_energy = -1e-4 * tau_sum
        other_leg = FR_IDX if self.side == 1 else FL_IDX
        dev_other = max(abs(self.targets[j] - self.straight_pose[j])
                        for j in other_leg)
        r_keep = -0.5 * dev_other
        # 整臂协同奖励：活动腿目标对齐 M8 几何映射（hip/thigh/calf 全动，
        # 堵死"只动小腿"的偷懒解）；joint_weight=0 关闭
        r_joint = 0.0
        if self.joint_weight > 0.0:
            leg = FL_IDX if self.side == 1 else FR_IDX
            r_joint = -self.joint_weight * float(np.mean([
                (self.targets[j] - self.joint_point[j]) ** 2 for j in leg]))
        reward = r_pos + r_smooth + r_energy + r_keep + r_joint
        pitch = pitch_deg(tuple(float(v)
                                for v in data.qpos[self.base + 3:self.base + 7]))
        base_z = float(data.qpos[self.base + 2])
        done = abs(pitch) > 40.0 or base_z < 0.05
        if done:
            reward -= 10.0
        self.t += 1
        done = done or self.t >= EPISODE_STEPS
        self.hist.append(self.hand_pos[
            "FL" if self.side == 1 else "FR"].copy())
        self.hist_px.append(self.px)
        self.hist_py.append(self.py)
        self.prev_action = action_delta.copy()
        return self.obs(), reward, done, {"dist": dist, "pitch": pitch}


class StandingPxEnv(StandingPawReachEnv):
    """站立 + 像素坐标观测（部署版）：px,py 直接喂策略，与真机接口一致。"""

    def _px_to_joints(self):
        side_x = -1.0 if self.px < 0.5 else 1.0
        amp_x = abs(self.px - 0.5) * 2.0
        amp_y = float(np.clip((0.60 - self.py) / 0.40, 0.0, 1.0))
        q = list(self.stand_joints)
        leg = FL_IDX if side_x < 0 else FR_IDX
        hip = 0.3 * amp_x
        q[leg[0]] = self.stand_joints[leg[0]] + (hip if side_x < 0 else -hip)
        q[leg[1]] = 0.2 - 0.8 * amp_y     # 站立抬爪带 0.2 -> -0.6
        q[leg[2]] = -1.5 - 0.5 * amp_y    # -1.5 -> -2.0
        return q

    def reset(self):
        self.px = float(np.random.uniform(0.15, 0.85))
        self.py = float(np.random.uniform(0.15, 0.55))
        self.side = 0 if self.px >= 0.5 else 1
        self.joint_point = self._px_to_joints()
        self.hand_pos = self._fk_foot(self.joint_point)
        self.latency = float(np.random.uniform(*self.latency_range))
        self.hist = [self.hand_pos["FL" if self.side == 1
                                    else "FR"].copy()]
        self.hist_px = [self.px]
        self.hist_py = [self.py]
        data = self.data
        data.qpos[:] = self.stand_qpos
        data.qvel[:] = 0.0
        for a in self.ctx.qpos_addresses:
            data.qpos[a] += np.random.uniform(-self.init_noise,
                                              self.init_noise)
        data.qpos[self.base + 3:self.base + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.model, data)
        self.targets = [float(data.qpos[a]) for a in self.ctx.qpos_addresses]
        self.prev_action = np.zeros(12, dtype=float)
        self.t = 0

    def _step_hand(self):
        self.px = float(np.clip(self.px + np.random.normal(0.0, 0.05)
                                * self.walk_scale, 0.05, 0.95))
        self.py = float(np.clip(self.py + np.random.normal(0.0, 0.03)
                                * self.walk_scale, 0.05, 0.75))
        self.side = 0 if self.px >= 0.5 else 1
        self.joint_point = self._px_to_joints()
        self.hand_pos = self._fk_foot(self.joint_point)

    def obs(self):
        data = self.data
        q = np.array([float(data.qpos[a]) for a in self.ctx.qpos_addresses])
        dq = np.array([float(data.qvel[a]) for a in self.ctx.dof_addresses])
        k = max(1, int(round(self.latency * CTRL_HZ)))
        px_d = self.hist_px[-k] if len(self.hist_px) >= k else self.hist_px[0]
        py_d = self.hist_py[-k] if len(self.hist_py) >= k else self.hist_py[0]
        pitch, roll = self._base_rpy()
        side = np.array([float(self.side)], dtype=float)
        return np.concatenate([q / math.pi, dq / 10.0,
                               np.array([pitch, roll]),
                               np.array([px_d, py_d]), side])

    def step(self, action_delta):
        # 与 StandingPawReachEnv.step 相同，但手走像素空间、追加像素历史
        data = self.data
        targets = list(self.targets)
        act_idx = FR_IDX if self.side == 0 else FL_IDX
        for i in range(12):
            if self.mask_active_only and i not in act_idx:
                continue
            targets[i] = float(np.clip(targets[i] + action_delta[i],
                                       self._qlo[i], self._qhi[i]))
        self.targets = targets
        tau_sum = 0.0
        for _ in range(self.substeps):
            q = [float(data.qpos[a]) for a in self.ctx.qpos_addresses]
            dq = [float(data.qvel[a]) for a in self.ctx.dof_addresses]
            tau = [0.0] * 12
            for i in range(12):
                kp, kd = 300.0, 10.0
                if i in (FR_IDX if self.side == 0 else FL_IDX):
                    kp, kd = 150.0, 10.0
                tau[i] = kp * (targets[i] - q[i]) + kd * (-dq[i])
            tau = [max(-self.tau_clip, min(self.tau_clip, v)) for v in tau]
            for aid, v in zip(self.actuator_ids, tau):
                data.ctrl[aid] = v
            mujoco.mj_step(self.model, data)
            tau_sum += sum(v * v for v in tau) / self.substeps
        self._step_hand()
        paw = np.array(data.geom_xpos[self.ctx.foot_geom_ids[
            "FL" if self.side == 1 else "FR"]])
        dist = float(np.linalg.norm(paw - self.hand_pos["FL" if self.side == 1
                                                         else "FR"]))
        r_pos = math.exp(-dist / self.reward_scale) + 1.0
        pitch, roll = self._base_rpy()
        r_balance = -0.3 * (abs(pitch) + abs(roll))
        base_z = float(data.qpos[self.base + 2])
        r_height = -1.5 * abs(base_z - self.z0)
        r_smooth = -0.05 * float(np.sum(action_delta ** 2))
        r_energy = -1e-4 * tau_sum
        other_leg = FR_IDX if self.side == 1 else FL_IDX
        dev_other = max(abs(self.targets[j] - self.stand_joints[j])
                        for j in other_leg)
        r_keep = -0.3 * dev_other
        r_alive = 0.5
        # 整臂协同：活动腿目标对齐站立像素映射
        r_joint = 0.0
        if self.joint_weight > 0.0:
            leg = FL_IDX if self.side == 1 else FR_IDX
            r_joint = -self.joint_weight * float(np.mean([
                (self.targets[j] - self.joint_point[j]) ** 2 for j in leg]))
        reward = r_pos + r_smooth + r_energy + r_keep + r_alive + r_joint \
            + (r_balance + r_height) * self.balance_weight
        done = base_z < self.z0 - 0.035 or abs(pitch) > 0.6 or abs(roll) > 0.6
        if done:
            reward -= 10.0
        self.t += 1
        done = done or self.t >= EPISODE_STEPS
        self.hist.append(self.hand_pos[
            "FL" if self.side == 1 else "FR"].copy())
        self.hist_px.append(self.px)
        self.hist_py.append(self.py)
        self.prev_action = action_delta.copy()
        return self.obs(), reward, done, {"dist": dist,
                                          "pitch": math.degrees(pitch)}


OBS_DIM = 12 + 12 + 3 + 3 + 3 + 1   # 34


class GRUPolicy(nn.Module):
    """带记忆策略：GRU + MLP（论文 Table I 的 GRU 主干）。"""

    def __init__(self, obs_dim, act_dim=6, hidden=128, log_std_init=-1.5):
        super().__init__()
        self.gru = nn.GRU(obs_dim, hidden, batch_first=False)
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())
        self.mean = nn.Linear(hidden, act_dim)
        self.value = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))

    def forward(self, x, h):
        out, h = self.gru(x.unsqueeze(0), h)
        f = self.net(out.squeeze(0))
        return self.mean(f), self.log_std, self.value(f), h


class MLPPolicy(nn.Module):
    """无记忆策略：MLP（论文 Table I 的 MLP 基线，对照组）。"""

    def __init__(self, obs_dim, act_dim=6, hidden=128, log_std_init=-1.5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())
        self.mean = nn.Linear(hidden, act_dim)
        self.value = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))

    def forward(self, x, h=None):
        f = self.net(x)
        return self.mean(f), self.log_std, self.value(f), None


def imitation_update(policy, opt, obs, cur, mapped, act_dim=6, mb=256,
                     epochs=4, backbone="mlp"):
    """模仿预热：监督学习 M8 映射（含大臂运动）。prone：前腿 6 关节
    cur[:, :6] + delta 应等于 mapped[:, :6]；standing：全部 12 关节
    （mapped 中非活动腿=站立位，顺带监督平衡）。
    GRU 主干：minibatch 从零隐藏态起（截断监督，与 ppo_update 同约定）。"""
    n = len(obs)
    idx = np.arange(n)
    cur = np.asarray(cur, dtype=np.float32)
    mapped = np.asarray(mapped, dtype=np.float32)
    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, n, mb):
            b = idx[start:start + mb]
            o = torch.tensor(obs[b], dtype=torch.float32)
            if backbone == "gru":
                h = torch.zeros(1, len(b), 128)
                mean, _, _, _ = policy(o, h)
            else:
                mean, _, _, _ = policy(o)       # (mb, act_dim) 增量
            cur_b = torch.tensor(cur[b])
            mapped_b = torch.tensor(mapped[b])
            new_target = cur_b[:, :act_dim] + mean
            loss = ((new_target - mapped_b[:, :act_dim]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()


def ppo_update(policy, opt, obs, acts, logp, adv, ret, h_in=None,
               epochs=4, mb=256, clip=0.2, ent_coef=1e-3, backbone="mlp"):
    n = len(obs)
    idx = np.arange(n)
    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, n, mb):
            b = idx[start:start + mb]
            o = torch.tensor(obs[b], dtype=torch.float32)
            a = torch.tensor(acts[b], dtype=torch.float32)
            olp = torch.tensor(logp[b], dtype=torch.float32)
            advb = torch.tensor(adv[b], dtype=torch.float32)
            retb = torch.tensor(ret[b], dtype=torch.float32)
            advb = (advb - advb.mean()) / (advb.std() + 1e-8)
            if backbone == "gru":
                # 截断 BPTT：用 rollout 里该步的真实初始隐藏态
                h = torch.tensor(h_in[b], dtype=torch.float32).permute(1, 0, 2)
                mean, log_std, val, _ = policy(o, h)
            else:
                mean, log_std, val, _ = policy(o)
            dist = torch.distributions.Normal(mean, log_std.exp())
            nlp = dist.log_prob(a).sum(-1)
            ratio = torch.exp(nlp - olp)
            surr1 = ratio * advb
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advb
            loss_pi = -torch.min(surr1, surr2).mean() \
                - ent_coef * dist.entropy().sum(-1).mean()
            loss_v = ((val.squeeze(-1) - retb) ** 2).mean()
            opt.zero_grad()
            (loss_pi + 0.5 * loss_v).backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()


def make_envs(args):
    # 注意：MuJoCo 在 Windows 上打不开含非 ASCII 字符的路径，
    # 本地中文目录下请用 --scene 指向纯英文路径（如 junction 名）
    ctx = common.load_context(Path(args.scene))
    model = ctx.model
    if args.task in ("standing", "standing_px"):
        model.opt.timestep = TIMESTEP_STAND
        stand_qpos = np.asarray(ctx.standing_qpos, dtype=float).copy()
        cls = StandingPxEnv if args.task == "standing_px" \
            else StandingPawReachEnv
        envs = [cls(ctx, stand_qpos, args.latency_range, SUBSTEPS_STAND,
                    walk_scale=getattr(args, "walk_scale", 1.0),
                    balance_weight=getattr(args, "balance_weight", 1.0),
                    init_noise=getattr(args, "init_noise", 0.02),
                    reward_scale=getattr(args, "reward_scale", 0.06),
                    joint_weight=getattr(args, "joint_weight", 0.0),
                    tau_clip=getattr(args, "tau_clip", 60.0),
                    mask_active_only=getattr(args, "mask_active_only", False))
                for _ in range(args.envs)]
        joints = [float(stand_qpos[a]) for a in ctx.qpos_addresses]
        print(f"[INFO] 模型加载: {args.scene}")
        print(f"[INFO] 站立{'像素版' if args.task == 'standing_px' else ''} "
              f"base_z={float(stand_qpos[ctx.base_qpos_address + 2]):.3f}"
              f" 站立关节={[round(v, 2) for v in joints]}")
        return ctx, envs
    model.opt.timestep = TIMESTEP
    base = ctx.base_qpos_address
    straight_pose = [float(LIE_Q[i]) for i in range(12)]
    # hip 保持 LIE 冻结值；thigh 前摆 REACH_DT；calf 伸直 REACH_DC
    for j in (FR_IDX[1], FL_IDX[1]):
        straight_pose[j] = LIE_Q[j] + REACH_DT
    for j in (FR_IDX[2], FL_IDX[2]):
        straight_pose[j] = LIE_Q[j] + REACH_DC
    # 标定趴姿 base 高度（与 verify_m8_dynamics.py 同款二分）
    probe = mujoco.MjData(model)
    qpos = np.zeros(model.nq, dtype=float)
    for address, value in zip(ctx.qpos_addresses, straight_pose):
        qpos[address] = float(value)
    qpos[base + 3] = 1.0
    lo, hi = -0.2, 0.7
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        qpos[base + 2] = mid
        probe.qpos[:] = qpos
        mujoco.mj_forward(model, probe)
        if min(float(mujoco.mj_geomDistance(
                model, probe, ctx.floor_geom_id, int(g), 0.3,
                np.zeros(6))) for g in ctx.foot_geom_ids.values()) > 0.0005:
            hi = mid
        else:
            lo = mid
    qpos[base + 2] = lo
    base_qpos = qpos.copy()
    if args.task == "prone_px":
        envs = [PronePxEnv(ctx, base_qpos, straight_pose, args.latency_range,
                           walk_scale=getattr(args, "walk_scale", 1.0),
                           reward_scale=getattr(args, "reward_scale", 0.06),
                           calf_hi=getattr(args, "calf_hi", -0.84),
                           torque_noise=getattr(args, "torque_noise", 0.0),
                           smooth_weight=getattr(args, "smooth_weight",
                                                 0.05),
                           joint_weight=getattr(args, "joint_weight", 0.0))
                for _ in range(args.envs)]
        print(f"[INFO] 模型加载: {args.scene}")
        print(f"[INFO] 趴姿像素版 base_z={lo:.3f}（观测直接吃 px,py，部署版）")
        return ctx, envs
    envs = [PawReachEnv(ctx, base_qpos, straight_pose, args.latency_range,
                        reward_scale=getattr(args, "reward_scale", 0.06),
                        calf_hi=getattr(args, "calf_hi", -0.84),
                        torque_noise=getattr(args, "torque_noise", 0.0),
                        smooth_weight=getattr(args, "smooth_weight", 0.05))
            for _ in range(args.envs)]
    print(f"[INFO] 模型加载: {args.scene}")
    print(f"[INFO] 趴姿 base_z={lo:.3f} 打直前腿={straight_pose[:6]}")
    return ctx, envs


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # ---- 证据留存：训练 manifest（论文证据包，GPT 任务书 5.2）----
    import json
    import platform
    import subprocess
    from datetime import datetime
    from pathlib import Path as _Path
    out_dir = _Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"],
                                    capture_output=True, text=True,
                                    timeout=10).stdout.strip()
        git_dirty = subprocess.run(["git", "status", "--porcelain"],
                                   capture_output=True, text=True,
                                   timeout=10).stdout.strip()
    except Exception:
        git_commit, git_dirty = "unknown", ""
    manifest = {
        "run_id": f"train_{args.task}_{args.backbone}_"
                  f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_seed{args.seed}",
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
        "git_dirty": bool(git_dirty),
        "task": args.task,
        "backbone": args.backbone,
        "seed": args.seed,
        "resume": str(args.resume or ""),
        "config": {k: (str(v) if not isinstance(v, (int, float, bool))
                       else v) for k, v in vars(args).items()},
        "python": platform.python_version(),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[EVIDENCE] manifest: {out_dir / 'manifest.json'}")
    ctx, envs = make_envs(args)
    obs_dim = obs_dim_for(args.task)
    act_dim = act_dim_for(args.task)
    # 站立任务探索噪声必须小：12 关节 × std 0.22 的随机扰动会瞬间推翻狗，
    # 策略根本见不到"存活"轨迹。站立用 std 0.05（log_std -3.0）起步。
    log_std_init = -3.0 if args.task in ("standing", "standing_px") else -1.5
    policy = (GRUPolicy(obs_dim, act_dim, log_std_init=log_std_init)
              if args.backbone == "gru"
              else MLPPolicy(obs_dim, act_dim, log_std_init=log_std_init))
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        policy.load_state_dict(ckpt["policy"])
        print(f"[INFO] 从 {args.resume} 续训（课程学习第二阶段）")
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_mean = 1e9
    for it in range(args.iterations):
        rollout = {"obs": [], "acts": [], "logp": [], "rew": [],
                   "done": [], "val": [], "dist": [], "h": [],
                   "cur": [], "mapped": []}
        h_state = None
        if args.backbone == "gru":
            h_state = torch.zeros(1, args.envs, 128)
        for step in range(EPISODE_STEPS):
            batch_obs = np.zeros((args.envs, obs_dim), dtype=np.float32)
            for e, env in enumerate(envs):
                batch_obs[e] = env.obs()
            x = torch.tensor(batch_obs, dtype=torch.float32)
            with torch.no_grad():
                if args.backbone == "gru":
                    # 存下该步的初始隐藏态（PPO 更新时做正确的截断 BPTT）
                    rollout["h"].append(h_state.numpy().copy())
                    mean, log_std, value, h_state = policy(x, h_state)
                else:
                    mean, log_std, value, _ = policy(x)
                std = log_std.exp().expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                act = dist.sample()
                logp = dist.log_prob(act).sum(-1)
                val = value.squeeze(-1).numpy()
                actn = act.numpy()
                logpn = logp.numpy()
            next_obs = np.zeros((args.envs, obs_dim), dtype=np.float32)
            rews = np.zeros(args.envs)
            dones = np.zeros(args.envs, dtype=bool)
            dists = np.zeros(args.envs)
            for e, env in enumerate(envs):
                no, r, d, info = env.step(actn[e])
                next_obs[e], rews[e], dones[e], dists[e] = no, r, d, info["dist"]
                if d:
                    env.reset()
                    # 回合结束：该环境隐藏态清零（否则记忆串到新回合）
                    if h_state is not None:
                        h_state[:, e] = 0.0
            rollout["obs"].append(batch_obs)
            rollout["acts"].append(actn)
            rollout["logp"].append(logpn)
            rollout["rew"].append(rews)
            rollout["done"].append(dones)
            rollout["val"].append(val)
            rollout["dist"].append(dists)
            # 模仿预热用：当前前腿目标 + M8 映射目标（12 维）
            cur = np.zeros((args.envs, 12), dtype=np.float32)
            mapped = np.zeros((args.envs, 12), dtype=np.float32)
            for e, env in enumerate(envs):
                cur[e] = np.asarray(env.targets, dtype=np.float32)
                mapped[e] = np.asarray(env.joint_point, dtype=np.float32)
            rollout["cur"].append(cur)
            rollout["mapped"].append(mapped)
            batch_obs = next_obs
        # GAE + 回报（标准 PPO）
        with torch.no_grad():
            last = torch.tensor(batch_obs, dtype=torch.float32)
            if args.backbone == "gru":
                _, _, last_val, _ = policy(last, h_state)
            else:
                _, _, last_val, _ = policy(last)
            last_val = last_val.squeeze(-1).numpy()
        adv = np.zeros((EPISODE_STEPS, args.envs))
        ret = np.zeros((EPISODE_STEPS, args.envs))
        gae = np.zeros(args.envs)
        next_val = last_val
        for t in reversed(range(EPISODE_STEPS)):
            delta = (rollout["rew"][t] + args.gamma * next_val
                     * (1 - rollout["done"][t]) - rollout["val"][t])
            gae = delta + args.gamma * args.lam * gae * \
                (1 - rollout["done"][t])
            adv[t] = gae
            ret[t] = gae + rollout["val"][t]
            next_val = rollout["val"][t]
        obs_f = np.asarray(rollout["obs"], dtype=np.float32).reshape(-1, obs_dim)
        acts_f = np.asarray(rollout["acts"], dtype=np.float32).reshape(-1,
                                                                      act_dim)
        logp_f = np.asarray(rollout["logp"], dtype=np.float32).reshape(-1)
        adv_f = np.asarray(adv, dtype=np.float32).reshape(-1)
        ret_f = np.asarray(ret, dtype=np.float32).reshape(-1)
        ret_f = (ret_f - ret_f.mean()) / (ret_f.std() + 1e-8)
        h_f = None
        if args.backbone == "gru":
            # (steps, 1, envs, 128) -> (steps*envs, 1, 128)
            h_f = np.asarray(rollout["h"], dtype=np.float32).reshape(-1, 1, 128)
        if it < args.imitation_iters:
            # 阶段 0：模仿学习预热（监督 M8 映射 -> 整臂协同 + 大臂参与）
            cur_f = np.asarray(rollout["cur"],
                               dtype=np.float32).reshape(-1, 12)
            mapped_f = np.asarray(rollout["mapped"],
                                  dtype=np.float32).reshape(-1, 12)
            imitation_update(policy, opt, obs_f, cur_f, mapped_f,
                             act_dim=act_dim, backbone=args.backbone)
        else:
            ppo_update(policy, opt, obs_f, acts_f, logp_f, adv_f, ret_f,
                       h_in=h_f, backbone=args.backbone)
        mean_dist = float(np.mean(rollout["dist"]))
        mean_rew = float(np.mean(rollout["rew"]))
        print(f"[ITER {it + 1}/{args.iterations}] dist={mean_dist:.4f}m "
              f"reward={mean_rew:.2f}")
        if mean_dist < best_mean:
            best_mean = mean_dist
            torch.save({"policy": policy.state_dict(), "backbone":
                        args.backbone, "obs_dim": obs_dim, "task":
                        args.task, "act_dim": act_dim},
                       out_dir / f"best_{args.backbone}_{args.task}.pt")
    print(f"[RESULT] TRAIN_OK best_mean_dist={best_mean:.4f}m "
          f"checkpoint={out_dir / ('best_' + args.backbone + '_' + args.task + '.pt')}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--backbone", choices=("mlp", "gru"), default="mlp")
    p.add_argument("--task", choices=("prone", "prone_px", "standing",
                                      "standing_px"),
                   default="prone",
                   help="prone=趴姿3D手位置(研究版)；prone_px=趴姿像素坐标"
                        "(部署版)；standing=站立3D；standing_px=站立像素"
                        "(部署版)")
    p.add_argument("--resume", default="",
                   help="从 checkpoint 续训（课程学习第二阶段用）")
    p.add_argument("--hand-walk-scale", type=float, default=1.0,
                   help="手目标移动幅度；0=固定手（课程第一阶段先学平衡）")
    p.add_argument("--balance-weight", type=float, default=1.0,
                   help="平衡+高度奖励权重（站立任务学平衡阶段调大，如 6）")
    p.add_argument("--init-noise", type=float, default=0.02,
                   help="站立任务初始关节扰动幅度（大=练抗干扰）")
    p.add_argument("--reward-scale", type=float, default=0.06,
                   help="追手奖励 exp(-d/scale)+1 的尺度；v2 根治版用 0.03"
                        "逼策略精准追手（否则只动小腿偷懒）")
    p.add_argument("--calf-hi", type=float, default=-0.84,
                   help="小腿伸直上限；v2 与真机一致用 -1.05 避极限振动")
    p.add_argument("--mask-active-only", action="store_true",
                   help="只把动作增量施加到活动腿（部署模式：冻身体）")
    p.add_argument("--tau-clip", type=float, default=60.0,
                   help="力矩钳位 Nm（v3 对齐真机 23.7Nm 用 24）")
    p.add_argument("--torque-noise", type=float, default=0.0,
                   help="力矩域随机化噪声幅值 Nm（v2 用 2.0 抗真机抖动）")
    p.add_argument("--smooth-weight", type=float, default=0.05,
                   help="动作平滑惩罚权重（v2 用 0.15~0.2 降抖）")
    p.add_argument("--joint-weight", type=float, default=0.0,
                   help="整臂协同奖励权重（活动腿对齐 M8 映射，"
                        "v2 用 0.5 逼 hip/thigh/calf 全动）")
    p.add_argument("--imitation-iters", type=int, default=0,
                   help="模仿学习预热轮数（监督 M8 映射含大臂运动，"
                        "v3 用 300；之后切换 PPO）")
    p.add_argument("--latency-range", type=float, nargs=2,
                   default=(0.03, 0.15),
                   help="观测手位置延迟注入范围（秒）；论文关键 trick")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="simulation/output/policies")
    p.add_argument("--scene", default=str(SCENE),
                   help="GO2 scene.xml 路径（Windows 中文路径需改为纯英文）")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
