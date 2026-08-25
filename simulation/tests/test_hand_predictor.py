"""Tests for HandPredictor（Kalman 延迟前推 + 手丢失轨迹记忆）。

纯 numpy，无需狗/仿真环境。
"""
import unittest

from real_go2.hand_predictor import HandPredictor


class TestHandPredictor(unittest.TestCase):
    def test_first_update_initializes(self):
        p = HandPredictor(latency_s=0.1)
        self.assertIsNone(p.predict(0.0))
        p.update(0.0, 0.4, 0.5)
        self.assertTrue(p.ready)
        px, py = p.predict(0.0)
        self.assertAlmostEqual(px, 0.4, places=6)
        self.assertAlmostEqual(py, 0.5, places=6)

    def test_tracks_constant_velocity_and_compensates_latency(self):
        # 手以 0.3 px/s 匀速右移，视觉 30Hz；链路延迟 0.12s
        latency = 0.12
        v = 0.3
        p = HandPredictor(latency_s=latency, max_lost_s=0.25)
        t = 0.0
        for _ in range(30):  # 先收敛 1 秒
            p.update(t, 0.4 + v * t, 0.5)
            t += 1.0 / 30.0
        # 最后时刻：预测应接近 t+latency 的真值，而不是 t 时刻的测量值
        px_pred, _ = p.predict(t)
        true_ahead = 0.4 + v * (t + latency)
        true_now = 0.4 + v * t
        err_pred = abs(px_pred - true_ahead)
        err_naive = abs(true_now - true_ahead)
        self.assertLess(err_pred, err_naive * 0.5)  # 补偿掉一半以上延迟
        self.assertLess(err_pred, 0.03)

    def test_hand_lost_keeps_predicting_then_none(self):
        p = HandPredictor(latency_s=0.05, max_lost_s=0.25)
        t = 0.0
        for i in range(20):
            p.update(t, 0.3 + 0.2 * t, 0.5)
            t += 1.0 / 30.0
        # 丢失 0.2s 内继续给预测
        lost_t = t + 0.2
        self.assertIsNotNone(p.predict(lost_t))
        # 超过 max_lost_s 返回 None
        self.assertIsNone(p.predict(t + 0.25 + 1e-3))

    def test_lost_prediction_decays_to_stop(self):
        p = HandPredictor(latency_s=0.0, max_lost_s=0.25)
        t = 0.0
        for i in range(20):
            p.update(t, 0.3 + 0.4 * t, 0.5)
            t += 1.0 / 30.0
        pos_early, _ = p.predict(t + 0.05)
        pos_late, _ = p.predict(t + 0.20)
        # 速度衰减：后期位置应接近停滞（不再继续外推很远）
        delta = abs(pos_late - pos_early)
        # 匀速外推 0.15s 会走 0.06；衰减后应显著小于
        self.assertLess(delta, 0.03)

    def test_noise_smoothed_and_velocity_clipped(self):
        p = HandPredictor(latency_s=0.0, max_lost_s=0.25, v_clip=(0.6, 0.6))
        t = 0.0
        last_raw = 0.5
        for i in range(60):
            true = 0.5 + 0.1 * t
            noisy = true + (0.02 if i % 2 == 0 else -0.02)
            p.update(t, noisy, 0.5)
            last_raw = noisy
            t += 1.0 / 30.0
        px, _ = p.predict(t)
        # 滤波后位置应比原始测量更接近真值
        true_now = 0.5 + 0.1 * t
        self.assertLess(abs(px - true_now), abs(last_raw - true_now))

    def test_reset(self):
        p = HandPredictor()
        p.update(0.0, 0.5, 0.5)
        self.assertTrue(p.ready)
        p.reset()
        self.assertIsNone(p.predict(1.0))


if __name__ == "__main__":
    unittest.main()
