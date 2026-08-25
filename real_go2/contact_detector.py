"""爪子接触检测：电机力矩/电流尖峰 -> "握到手了"（A5 触觉闭环）。

论文（Playful DoggyBot）承认其系统没有力/触觉反馈，咬没咬到靠视觉判断。
本模块用 GO2 lowstate 的 tau_est（估算力矩）做接触检测：爪子空摆时
tau_est 只有重力补偿量级；碰到手的瞬间固件 PD 为维持位置会产生力矩
尖峰 -> 检测到即判定接触。纯 numpy，可离线用录制数据/合成数据测试。

用法：
    det = ContactSpikeDetector(ratio=2.0, min_abs=1.0, hold_s=0.6)
    每 tick:  hit = det.update(t, abs_tau_of_active_leg)
    hit=True -> 握到（触发 hold_s 秒握手保持，期间暂停追手）
"""
import numpy as np


class ContactSpikeDetector:
    def __init__(self, alpha=0.05, ratio=2.0, min_abs=1.0,
                 warmup=0.5, hold_s=0.6, debounce_s=1.0):
        self.alpha = float(alpha)
        self.ratio = float(ratio)
        self.min_abs = float(min_abs)
        self.warmup = float(warmup)
        self.hold_s = float(hold_s)
        self.debounce_s = float(debounce_s)
        self.ema = None            # |tau| 指数平均（基线）
        self.last_trigger = -1e9   # 上次触发时间
        self.hold_until = 0.0      # 握手保持截止时间
        self.started = 0.0         # 第一次 update 时间

    def update(self, t, tau_abs):
        """输入当前活动腿 |tau|（Nm），返回是否触发新接触。"""
        if self.ema is None:
            self.ema = float(tau_abs)
            self.started = float(t)
            return False
        self.ema = self.alpha * float(tau_abs) + (1.0 - self.alpha) * self.ema
        # 预热期只建基线，不触发（防刚切换高增益时的假尖峰）
        if float(t) - self.started < self.warmup:
            return False
        if (float(t) - self.last_trigger < self.debounce_s
                or self.in_hold(t)):
            return False
        spike = float(tau_abs) > self.min_abs and \
            float(tau_abs) > self.ratio * max(self.ema, 1e-6)
        if spike:
            self.last_trigger = float(t)
            self.hold_until = float(t) + self.hold_s
            return True
        return False

    def in_hold(self, t):
        """当前是否处于握手保持窗口内。"""
        return float(t) < self.hold_until
