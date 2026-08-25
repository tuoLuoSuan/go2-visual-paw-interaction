import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from camera_intrinsics import CameraIntrinsics


class CameraIntrinsicsTest(unittest.TestCase):
    def test_validates_positive_focal_and_center(self):
        with self.assertRaises(ValueError):
            CameraIntrinsics(960, 540, 0.0, 500.0, 480.0, 270.0).validate()
        with self.assertRaises(ValueError):
            CameraIntrinsics(960, 540, 500.0, -1.0, 480.0, 270.0).validate()
        with self.assertRaises(ValueError):
            CameraIntrinsics(960, 540, 500.0, 500.0, float("nan"), 270.0).validate()

    def test_crop_allows_principal_point_outside_roi(self):
        base = CameraIntrinsics(960, 540, 500.0, 500.0, 480.0, 270.0)
        # ROI entirely left of the principal point -> negative cx
        cropped = base.crop(600, 0, 300, 200)
        self.assertLess(cropped.cx, 0.0)

    def test_scaled_intrinsics_track_resolution_ratio(self):
        base = CameraIntrinsics(1920, 1080, 1000.0, 1000.0, 960.0, 540.0)
        half = base.scaled(960, 540)
        self.assertEqual((half.width, half.height), (960, 540))
        self.assertAlmostEqual(half.fx, 500.0)
        self.assertAlmostEqual(half.fy, 500.0)
        self.assertAlmostEqual(half.cx, 480.0)
        self.assertAlmostEqual(half.cy, 270.0)

    def test_unproject_project_roundtrip(self):
        intrinsics = CameraIntrinsics(960, 540, 500.0, 500.0, 480.0, 270.0)
        point = (0.10, -0.05, 0.40)
        u, v = intrinsics.project(point)
        recovered = intrinsics.unproject(u, v, 0.40)
        for expected, actual in zip(point, recovered):
            self.assertAlmostEqual(expected, actual, places=9)

    def test_unproject_rejects_nonpositive_depth(self):
        intrinsics = CameraIntrinsics(960, 540, 500.0, 500.0, 480.0, 270.0)
        with self.assertRaises(ValueError):
            intrinsics.unproject(100.0, 100.0, 0.0)

    def test_crop_shifts_principal_point_keeps_focal(self):
        base = CameraIntrinsics(960, 540, 500.0, 500.0, 480.0, 270.0)
        cropped = base.crop(100, 50, 400, 300)
        self.assertEqual((cropped.width, cropped.height), (400, 300))
        self.assertAlmostEqual(cropped.fx, 500.0)
        self.assertAlmostEqual(cropped.fy, 500.0)
        self.assertAlmostEqual(cropped.cx, 380.0)
        self.assertAlmostEqual(cropped.cy, 220.0)

    def test_crop_rejects_out_of_bounds_region(self):
        base = CameraIntrinsics(960, 540, 500.0, 500.0, 480.0, 270.0)
        with self.assertRaises(ValueError):
            base.crop(900, 0, 100, 100)

    def test_crop_then_unproject_matches_full_frame(self):
        base = CameraIntrinsics(960, 540, 500.0, 500.0, 480.0, 270.0)
        # a point at depth 0.5 that lands inside the crop region
        point = (0.02, -0.01, 0.50)
        u, v = base.project(point)
        cropped = base.crop(200, 100, 400, 300)
        if 200 <= u < 600 and 100 <= v < 400:
            local_u, local_v = u - 200, v - 100
            recovered = cropped.unproject(local_u, local_v, 0.50)
            for expected, actual in zip(point, recovered):
                self.assertAlmostEqual(expected, actual, places=9)

    def test_json_roundtrip_preserves_distortion(self):
        intrinsics = CameraIntrinsics(
            1920, 1080, 1013.7, 1011.2, 955.3, 540.1,
            dist=(-0.31, 0.11, 0.0001, -0.0002, -0.02),
            calibrated=True, rms=0.42, created="2026-08-17T00:00:00",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intrinsics.json"
            intrinsics.save(path)
            loaded = CameraIntrinsics.load(path)
        self.assertEqual(loaded.to_dict(), intrinsics.to_dict())
        self.assertTrue(loaded.calibrated)
        self.assertEqual(loaded.dist, intrinsics.dist)

    def test_from_dict_accepts_minimal_payload(self):
        payload = {
            "width": 640, "height": 480,
            "fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0,
        }
        intrinsics = CameraIntrinsics.from_dict(payload)
        self.assertFalse(intrinsics.calibrated)
        self.assertEqual(intrinsics.dist, (0.0, 0.0, 0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
