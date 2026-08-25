import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from hand_to_fr_target import (
    CameraExtrinsics,
    FrTargetConfig,
    clamp_to_box,
    palm_to_fr_target,
)


class CameraExtrinsicsTest(unittest.TestCase):
    def test_identity_transform(self):
        extrinsics = CameraExtrinsics(0.0, 0.0, 0.0, (0.0, 0.0, 0.0))
        point = (0.10, -0.05, 0.40)
        for expected, actual in zip(point, extrinsics.transform(point)):
            self.assertAlmostEqual(expected, actual, places=9)

    def test_yaw_ninety_degrees_maps_x_to_y(self):
        extrinsics = CameraExtrinsics(0.0, 0.0, 90.0, (0.0, 0.0, 0.0))
        x, y, z = extrinsics.transform((1.0, 0.0, 0.0))
        self.assertAlmostEqual(x, 0.0, places=9)
        self.assertAlmostEqual(y, 1.0, places=9)
        self.assertAlmostEqual(z, 0.0, places=9)

    def test_translation_is_added(self):
        extrinsics = CameraExtrinsics(0.0, 0.0, 0.0, (0.05, -0.02, 0.30))
        x, y, z = extrinsics.transform((0.0, 0.0, 0.0))
        self.assertAlmostEqual(x, 0.05, places=9)
        self.assertAlmostEqual(y, -0.02, places=9)
        self.assertAlmostEqual(z, 0.30, places=9)

    def test_pitch_ninety_maps_camera_z_to_base_x(self):
        # Under Rz(yaw) @ Ry(pitch) @ Rx(roll), +90 deg pitch maps the
        # camera +z axis to the base +x axis.  This tests the mathematical
        # Euler convention only; a physical OpenCV-camera-to-base mapping
        # requires all calibrated axes and is not inferred from one angle.
        extrinsics = CameraExtrinsics(0.0, 90.0, 0.0, (0.20, 0.0, 0.30))
        x, y, z = extrinsics.transform((0.0, 0.0, 0.50))  # 0.5 m straight ahead
        self.assertAlmostEqual(x, 0.70, places=6)
        self.assertAlmostEqual(y, 0.0, places=6)
        self.assertAlmostEqual(z, 0.30, places=6)
        # Rotation about y leaves the camera y coordinate unchanged.
        x2, y2, z2 = extrinsics.transform((0.0, -0.30, 0.50))
        self.assertAlmostEqual(x2, 0.70, places=6)
        self.assertAlmostEqual(y2, -0.30, places=6)
        self.assertAlmostEqual(z2, 0.30, places=6)

    def test_json_roundtrip(self):
        extrinsics = CameraExtrinsics(
            2.0, -5.0, 90.0, (0.05, 0.0, 0.35),
            calibrated=True, created="2026-08-17T00:00:00",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extrinsics.json"
            extrinsics.save(path)
            loaded = CameraExtrinsics.load(path)
        self.assertEqual(loaded.to_dict(), extrinsics.to_dict())
        self.assertTrue(loaded.calibrated)


class FrTargetTest(unittest.TestCase):
    def test_uncalibrated_extrinsics_gate_the_target(self):
        extrinsics = CameraExtrinsics(0.0, 0.0, 0.0, (0.0, 0.0, 0.0),
                                      calibrated=False)
        target = palm_to_fr_target((0.10, -0.02, 0.40), extrinsics)
        self.assertFalse(target.reachable)
        self.assertIn("EXTRINSICS_UNCALIBRATED", target.notes)
        # position is still computed for offline logging
        self.assertIsNotNone(target.position)

    def test_allow_uncalibrated_enables_offline_demo(self):
        extrinsics = CameraExtrinsics(0.0, 0.0, 0.0, (0.0, 0.0, 0.0),
                                      calibrated=False)
        config = FrTargetConfig(allow_uncalibrated=True)
        target = palm_to_fr_target((0.10, -0.02, 0.40), extrinsics, config)
        self.assertTrue(target.reachable)

    def test_approach_offset_is_applied_below_palm(self):
        extrinsics = CameraExtrinsics(0.0, 0.0, 0.0, (0.0, 0.0, 0.0),
                                      calibrated=True)
        config = FrTargetConfig(
            reach_box_m=(0.05, 0.35, -0.22, 0.22, 0.02, 0.50)
        )
        target = palm_to_fr_target((0.10, -0.02, 0.40), extrinsics, config)
        self.assertIsNotNone(target.position)
        self.assertAlmostEqual(target.position[2], 0.40 - 0.06, places=9)
        self.assertAlmostEqual(target.position[0], 0.10, places=9)
        self.assertAlmostEqual(target.position[1], -0.02, places=9)

    def test_out_of_reach_target_is_clamped_and_flagged(self):
        extrinsics = CameraExtrinsics(0.0, 0.0, 0.0, (0.0, 0.0, 0.0),
                                      calibrated=True)
        target = palm_to_fr_target((5.0, 5.0, 5.0), extrinsics)
        self.assertTrue(target.clamped)
        self.assertIn("CLAMPED_TO_REACH_BOX", target.notes)
        # 默认可达盒为证据化凸包 (0.10..0.28, -0.20..-0.08, 0.04..0.12)
        self.assertEqual(
            target.position,
            (0.28, -0.08, 0.12),
        )

    def test_invalid_palm_position_is_rejected(self):
        extrinsics = CameraExtrinsics(0.0, 0.0, 0.0, (0.0, 0.0, 0.0),
                                      calibrated=True)
        target = palm_to_fr_target(None, extrinsics)
        self.assertIsNone(target.position)
        self.assertFalse(target.reachable)
        self.assertIn("INVALID_PALM_POSITION", target.notes)

    def test_reach_box_validation(self):
        with self.assertRaises(ValueError):
            FrTargetConfig(reach_box_m=(0.35, 0.05, -0.22, 0.22, 0.02, 0.30)).validate()

    def test_clamp_inside_box_is_identity(self):
        point = (0.10, 0.00, 0.10)
        box = (0.05, 0.35, -0.22, 0.22, 0.02, 0.30)
        clamped, changed = clamp_to_box(point, box)
        self.assertEqual(clamped, point)
        self.assertFalse(changed)


if __name__ == "__main__":
    unittest.main()
