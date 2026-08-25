"""手位置 Kalman 预测器：延迟前推补偿 + 手丢失短时轨迹记忆。

设计动机（Playful DoggyBot 论文思想落地）：
1. 感知-控制链路有固定延迟（摄像头 -> MediaPipe -> UDP -> EMA -> 关节映射），
   爪子"总慢半拍"。用常速度模型把滤波后的位置**前推到 t + latency_s**，
   让爪子提前到位。
2. 论文用 GRU 补目标短暂丢失；这里是规则版：手丢失后继续按最后速度
   外推 max_lost_s 秒（速度线性衰减），超过才返回 None，交给上层
   "手离开 3 秒收回"逻辑。

用法（M8 保持循环，500Hz 主循环 + ~15-30Hz 视觉测量）：
    pred = HandPredictor(latency_s=0.12, max_lost_s=0.25)
    收到测量:  pred.update(t, px, py)
    每 tick:    pos = pred.predict(t)   # 前推到 t+latency；丢失期间继续外推
                if pos is None: -> 真正手离开
纯 numpy，可在无狗环境下单元测试。
"""
import numpy as np


class HandPredictor:
    """2D 常速度 Kalman 滤波器 + 延迟前推 + 丢失外推。

    state x = [px, vx, py, vy]；测量 z = [px, py]。
    """

    def __init__(self, latency_s=0.0, max_lost_s=0.25,
                 q_pos=1e-4, q_vel=0.5, r_pos=2e-3,
                 v_clip=(1.2, 1.2), decay=True):
        self.latency_s = max(0.0, float(latency_s))
        self.max_lost_s = max(0.0, float(max_lost_s))
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        self.r_pos = float(r_pos)
        self.v_clip = tuple(float(v) for v in v_clip)
        self.decay = bool(decay)
        self.x = None       # [px, vx, py, vy]
        self.P = None       # 4x4
        self.last_t = None  # 最近一次测量的时间戳
        self.n_updates = 0

    @property
    def ready(self):
        return self.x is not None

    def _build_fq(self, dt):
        dt = max(1e-3, float(dt))
        f = np.array([[1.0, dt, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, dt],
                      [0.0, 0.0, 0.0, 1.0]])
        q = np.zeros((4, 4))
        q[0, 0] = self.q_pos * dt
        q[1, 1] = self.q_vel * dt
        q[2, 2] = self.q_pos * dt
        q[3, 3] = self.q_vel * dt
        return f, q

    def update(self, t, px, py):
        """喂入一次测量（t 单调递增）。"""
        px = float(px)
        py = float(py)
        if self.x is None:
            self.x = np.array([px, 0.0, py, 0.0])
            self.P = np.eye(4)
            self.last_t = float(t)
            self.n_updates = 1
            return
        dt = max(1e-3, float(t) - self.last_t)
        self.last_t = float(t)
        f, q = self._build_fq(dt)
        self.x = f @ self.x
        self.P = f @ self.P @ f.T + q
        h = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0]])
        r = np.eye(2) * self.r_pos
        z = np.array([px, py])
        y = z - h @ self.x
        s = h @ self.P @ h.T + r
        k = self.P @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.P = (np.eye(4) - k @ h) @ self.P
        # 限速防抖动尖峰
        self.x[1] = float(np.clip(self.x[1], -self.v_clip[0], self.v_clip[0]))
        self.x[3] = float(np.clip(self.x[3], -self.v_clip[1], self.v_clip[1]))
        self.n_updates += 1

    def predict(self, t):
        """返回前推到 t + latency_s 的位置；丢失超过 max_lost_s 返回 None。"""
        if self.x is None:
            return None
        age = float(t) - self.last_t
        if age > self.max_lost_s:
            return None
        if self.decay and age > 0.0 and self.max_lost_s > 0.0:
            # 手丢失期间速度线性衰减到 0：短时按轨迹外推，长了自然停
            decay = max(0.0, 1.0 - age / self.max_lost_s)
        else:
            decay = 1.0
        lead = self.latency_s + age * decay
        px = self.x[0] + self.x[1] * lead
        py = self.x[2] + self.x[3] * lead
        return float(px), float(py)

    def reset(self):
        self.x = None
        self.P = None
        self.last_t = None
        self.n_updates = 0
