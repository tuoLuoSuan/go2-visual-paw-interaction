import math
from pathlib import Path
import sys
import unittest


WORKSPACE = Path(__file__).resolve().parents[2]
SOURCE = WORKSPACE / "simulation" / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import standing_paw_lift_common as common


EXPECTED_JOINTS = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)


class StandingPawLiftCommonTest(unittest.TestCase):
    def test_fix_stand_joint_targets_use_canonical_order(self):
        target = common.fix_stand_joint_targets()

        self.assertEqual(
            target,
            (0.0, 0.8, -1.5) * 4,
        )
        self.assertEqual(
            common.CANONICAL_JOINT_NAMES,
            EXPECTED_JOINTS,
        )

    def test_config_rejects_nonpositive_and_invalid_height(self):
        with self.assertRaises(ValueError):
            common.StandingConfig(sample_rate_hz=0.0).validate()
        with self.assertRaises(ValueError):
            common.StandingConfig(target_lifts_m=(0.1, 0.0)).validate()
        with self.assertRaises(ValueError):
            common.StandingConfig(target_lifts_m=(0.2, 0.1)).validate()

    def test_config_rejects_out_of_range_load_compensation_scale(self):
        self.assertEqual(
            common.StandingConfig(load_compensation_scale=1.0).validate()
            .load_compensation_scale,
            1.0,
        )
        common.StandingConfig(load_compensation_scale=0.0).validate()
        for invalid in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                common.StandingConfig(load_compensation_scale=invalid).validate()

    def test_config_rejects_nonpositive_target_reach_tolerance(self):
        with self.assertRaises(ValueError):
            common.StandingConfig(target_reach_tolerance_m=0.0).validate()
        with self.assertRaises(ValueError):
            common.StandingConfig(target_reach_tolerance_m=-0.01).validate()

    def test_config_rejects_nonpositive_base_drift_limit(self):
        with self.assertRaises(ValueError):
            common.StandingConfig(base_drift_limit_m=0.0).validate()

    def test_config_rejects_nonpositive_shift_margin_target(self):
        with self.assertRaises(ValueError):
            common.StandingConfig(shift_com_margin_target_m=0.0).validate()

    def test_support_margin_is_orientation_independent(self):
        triangle = (
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
        )

        inside = common.support_polygon_margin(
            (0.2, 0.2),
            triangle,
        )
        reversed_inside = common.support_polygon_margin(
            (0.2, 0.2),
            tuple(reversed(triangle)),
        )
        outside = common.support_polygon_margin(
            (0.8, 0.8),
            triangle,
        )

        self.assertGreater(inside, 0.0)
        self.assertAlmostEqual(inside, reversed_inside)
        self.assertLess(outside, 0.0)
        self.assertTrue(math.isfinite(inside))

    def test_support_margin_accepts_four_foot_polygon(self):
        square = (
            (-0.2, -0.1),
            (0.2, -0.1),
            (0.2, 0.1),
            (-0.2, 0.1),
        )

        self.assertGreater(
            common.support_polygon_margin((0.0, 0.0), square),
            0.0,
        )
        self.assertLess(
            common.support_polygon_margin((0.3, 0.0), square),
            0.0,
        )

    def test_support_margin_rejects_degenerate_polygon(self):
        with self.assertRaisesRegex(ValueError, "退化"):
            common.support_polygon_margin(
                (0.0, 0.0),
                ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
            )


if __name__ == "__main__":
    unittest.main()
