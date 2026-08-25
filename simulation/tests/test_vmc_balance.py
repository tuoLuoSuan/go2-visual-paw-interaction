import math
from pathlib import Path
import sys
import unittest


WORKSPACE = Path(__file__).resolve().parents[2]
SOURCE = WORKSPACE / "simulation" / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import vmc_balance as vmc
import standing_paw_lift_common as common


class VmcBalanceTest(unittest.TestCase):
    def test_quaternion_error_identity_is_zero(self):
        error = vmc.quaternion_error((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        for value in error:
            self.assertAlmostEqual(value, 0.0, places=9)

    def test_quaternion_error_ninety_degrees_about_z(self):
        # reference: identity; current: 90 deg about z (w=cos45, z=sin45)
        current = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
        error = vmc.quaternion_error((1.0, 0.0, 0.0, 0.0), current)
        self.assertAlmostEqual(error[0], 0.0, places=6)
        self.assertAlmostEqual(error[1], 0.0, places=6)
        # 旋转向量按 2*sin(theta/2) 小角度约定：90° -> 2*sin(45°) = sqrt(2)
        self.assertAlmostEqual(abs(error[2]), math.sqrt(2.0), places=6)

    def test_clip_torques_respects_ranges(self):
        clipped = vmc.clip_torques((-6.0, 0.0, 6.0), ((-5.0, 5.0),) * 3)
        self.assertEqual(clipped, (-5.0, 0.0, 5.0))

    def test_config_rejects_negative_vmc_gains(self):
        for field in ("vmc_kp_pos", "vmc_kd_pos", "vmc_kp_rot", "vmc_kd_rot"):
            with self.assertRaises(ValueError):
                common.StandingConfig(**{field: -1.0}).validate()

    def test_config_rejects_non_bool_enable(self):
        with self.assertRaises(ValueError):
            common.StandingConfig(vmc_enabled=1).validate()

    def test_distribute_wrench_keeps_feet_positive(self):
        # 对称三足：后脚 (x=-0.2,y=0)，两前脚 (x=0.1,y=±0.14)
        feet = ((-0.20, 0.00, 0.02), (0.10, 0.14, 0.02), (0.10, -0.14, 0.02))
        base = (0.0, 0.0, 0.30)
        force = (0.0, 0.0, 120.0)
        moment = (0.0, 8.0, 0.0)  # nose-up pitch moment（可行：Fz 全为正）
        distributed = vmc.distribute_wrench(feet, base, force, moment)
        fz = [item[0][2] for item in distributed]
        self.assertGreaterEqual(min(fz), -1e-9)
        total = sum(-(foot[0] - base[0]) * value
                    for foot, item in zip(feet, distributed)
                    for value in [item[0][2]])
        self.assertAlmostEqual(total, moment[1], places=6)
        self.assertAlmostEqual(sum(fz), force[2], places=6)
        # 期望解：后脚承担更多（0.2F0 - 0.1(F1+F2) = 8）
        self.assertGreater(fz[0], fz[1])

    def test_distribute_wrench_pure_vertical_is_equal(self):
        feet = ((-0.20, 0.00, 0.02), (0.10, 0.14, 0.02), (0.10, -0.14, 0.02))
        distributed = vmc.distribute_wrench(
            feet, (0.0, 0.0, 0.30), (0.0, 0.0, 90.0), (0.0, 0.0, 0.0)
        )
        self.assertEqual([round(item[0][2], 6) for item in distributed],
                         [30.0, 30.0, 30.0])


if __name__ == "__main__":
    unittest.main()
