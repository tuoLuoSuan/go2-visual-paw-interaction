"""Tests for ContactSpikeDetector（A5 爪子接触"握到"检测）。"""
import unittest

from real_go2.contact_detector import ContactSpikeDetector


def run_series(det, taus, dt=0.002, t0=0.0):
    hits = []
    t = t0
    for tau in taus:
        hits.append(det.update(t, tau))
        t += dt
    return hits


class TestContactSpikeDetector(unittest.TestCase):
    def test_no_spike_no_trigger(self):
        det = ContactSpikeDetector(warmup=0.2, ratio=2.0, min_abs=1.0)
        taus = [2.0] * 500           # 恒定 2 Nm 基线（重力补偿量级）
        hits = run_series(det, taus)
        self.assertFalse(any(hits))

    def test_spike_triggers_once(self):
        det = ContactSpikeDetector(warmup=0.2, ratio=2.0, min_abs=1.0,
                                   debounce_s=0.1)
        taus = [2.0] * 300 + [9.0] * 30 + [2.0] * 300   # 尖峰 9 Nm
        hits = run_series(det, taus)
        self.assertEqual(sum(hits), 1)
        # 触发点应在尖峰段内
        self.assertGreater(sum(hits[300:330]), 0)

    def test_hold_window(self):
        det = ContactSpikeDetector(warmup=0.2, ratio=2.0, min_abs=1.0,
                                   debounce_s=0.1, hold_s=0.4)
        taus = [2.0] * 300 + [9.0] * 30 + [2.0] * 500
        hits = run_series(det, taus)
        self.assertEqual(sum(hits), 1)
        idx = hits.index(True)
        t_hit = idx * 0.002
        self.assertTrue(det.in_hold(t_hit + 0.3))
        self.assertFalse(det.in_hold(t_hit + 0.5))

    def test_debounce_blocks_second_spike(self):
        det = ContactSpikeDetector(warmup=0.2, ratio=2.0, min_abs=1.0,
                                   debounce_s=0.5, hold_s=0.0)
        # 两个尖峰间隔 0.2s < debounce 0.5s -> 只触发一次
        taus = ([2.0] * 200 + [9.0] * 20 + [2.0] * 100
                + [9.0] * 20 + [2.0] * 200)
        hits = run_series(det, taus)
        self.assertEqual(sum(hits), 1)

    def test_warmup_ignores_initial_spike(self):
        det = ContactSpikeDetector(warmup=0.5, ratio=2.0, min_abs=1.0)
        taus = [9.0] * 100 + [2.0] * 400   # 一开始就尖峰 -> 预热期忽略
        hits = run_series(det, taus)
        self.assertFalse(any(hits))


if __name__ == "__main__":
    unittest.main()
