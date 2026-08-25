import math
import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from camera_intrinsics import CameraIntrinsics
from hand_3d_localizer import (
    Hand3DFilter,
    Hand3DResult,
    LANDMARK_PAIR_SPECS,
    compute_hand_roi,
    depth_from_landmarks,
    load_pair_specs,
    localize_palm,
    pair_distances_from_measurement,
    pair_specs_from_dict,
    save_pair_specs,
    select_primary_hand,
)


def make_planar_hand(offset_x, offset_y, depth, pair_specs):
    """Build a 21-landmark hand whose spec pairs have exact metric distances.

    The palm plane faces the camera at ``depth``; the pair points are laid
    out in the image plane so a calibrated square-pixel camera recovers the
    exact depth.  All other 21 landmarks sit between the pair points.
    """
    landmarks = [[0.0, 0.0] for _ in range(21)]
    # wrist as anchor
    landmarks[0] = [0.0, 0.0]
    # build a small triangle that satisfies all four pair distances exactly
    # 0-9 = 0.105, 0-5 = 0.095, 5-17 = 0.078, 1-17 = 0.098
    distances = {(first, second): distance
                 for first, second, distance in pair_specs}
    d09 = distances[(0, 9)]
    d05 = distances[(0, 5)]
    d517 = distances[(5, 17)]
    d117 = distances[(1, 17)]
    # 5 and 9 on the x-axis
    landmarks[9] = [d09, 0.0]
    landmarks[5] = [d05, 0.0]
    # 17 above, determined by 5-17 and 0-17 via two-circle intersection;
    # we only constrain 5-17 exactly, place 17 at distance d517 from 5 and
    # d117 from 1 by construction below
    a = d05
    b = d517
    h = math.sqrt(max(0.0, b * b - 0.0))  # place 17 above 5
    landmarks[17] = [a, h]
    # 1 (thumb CMC) at distance d117 from 17: put it on the line y=0 beyond 0
    # solve |1-17| = d117 with 1 = (x, 0)
    target = d117
    x1 = a - math.sqrt(max(0.0, target * target - h * h))
    landmarks[1] = [x1, 0.0]
    # fill the rest between known points
    for index in range(2, 21):
        if index in (5, 9, 17):
            continue
        base = landmarks[index - 1]
        landmarks[index] = [base[0] + 0.002, base[1] + 0.003]
    # shift to the requested palm center and depth
    shifted = []
    for x, y in landmarks:
        shifted.append([x + offset_x, y + offset_y])
    return shifted, depth


def project_hand(intrinsics, world, depth):
    """Project a metric 3D hand (x,y in meters at z=depth) to normalized
    pixel landmarks."""
    normalized = []
    for x, y in world:
        u = intrinsics.fx * x / depth + intrinsics.cx
        v = intrinsics.fy * y / depth + intrinsics.cy
        normalized.append([u / intrinsics.width, v / intrinsics.height])
    return normalized


class Hand3DDepthTest(unittest.TestCase):
    def setUp(self):
        self.intrinsics = CameraIntrinsics(960, 540, 600.0, 600.0, 480.0, 270.0)

    def _project_hand(self, world, depth):
        return project_hand(self.intrinsics, world, depth)

    def test_depth_from_landmarks_recovers_exact_planar_depth(self):
        from hand_3d_localizer import LANDMARK_PAIR_SPECS
        world, depth = make_planar_hand(0.05, -0.02, 0.40, LANDMARK_PAIR_SPECS)
        normalized = self._project_hand(world, depth)
        median, count, spread, _ = depth_from_landmarks(
            normalized, 960, 540, self.intrinsics
        )
        self.assertEqual(count, 4)
        self.assertAlmostEqual(median, depth, places=6)
        self.assertLess(spread, 1e-6)

    def test_localize_palm_recovers_position_in_camera_frame(self):
        from hand_3d_localizer import LANDMARK_PAIR_SPECS
        world, depth = make_planar_hand(0.05, -0.02, 0.40, LANDMARK_PAIR_SPECS)
        normalized = self._project_hand(world, depth)
        result = localize_palm(normalized, 960, 540, self.intrinsics)
        self.assertTrue(result.localized)
        self.assertEqual(result.pair_count, 4)
        self.assertAlmostEqual(result.depth_m, depth, places=6)
        # palm pixel center: mean of landmark ids (0,5,9,13,17) in metric space
        palm_ids = (0, 5, 9, 13, 17)
        palm_world_x = sum(world[i][0] for i in palm_ids) / 5
        palm_world_y = sum(world[i][1] for i in palm_ids) / 5
        self.assertAlmostEqual(result.palm_camera[0], palm_world_x, places=5)
        self.assertAlmostEqual(result.palm_camera[1], palm_world_y, places=5)
        self.assertAlmostEqual(result.palm_camera[2], depth, places=6)

    def test_localize_reports_depth_out_of_range(self):
        from hand_3d_localizer import LANDMARK_PAIR_SPECS
        world, depth = make_planar_hand(0.0, 0.0, 2.5, LANDMARK_PAIR_SPECS)
        normalized = self._project_hand(world, depth)
        result = localize_palm(normalized, 960, 540, self.intrinsics)
        self.assertFalse(result.localized)
        self.assertIn("DEPTH_OUT_OF_RANGE", result.failure_codes)

    def test_localize_reports_insufficient_pairs_for_tiny_hand(self):
        from hand_3d_localizer import LANDMARK_PAIR_SPECS
        world, depth = make_planar_hand(0.0, 0.0, 0.40, LANDMARK_PAIR_SPECS)
        # 200x smaller -> all pairs below the pixel threshold
        world = [[value * 0.005 for value in row] for row in world]
        normalized = self._project_hand(world, depth)
        result = localize_palm(normalized, 960, 540, self.intrinsics)
        self.assertFalse(result.localized)
        self.assertIn("INSUFFICIENT_PAIRS", result.failure_codes)

    def test_localize_rejects_wrong_landmark_count(self):
        result = localize_palm([[0.5, 0.5]] * 3, 960, 540, self.intrinsics)
        self.assertFalse(result.localized)
        self.assertIn("INVALID_LANDMARK_COUNT", result.failure_codes)


class Hand3DFilterTest(unittest.TestCase):
    def _result(self, point):
        return Hand3DResult(
            palm_camera=point, depth_m=0.4, pair_count=4,
            pair_spread_m=0.001, palm_pixels=(100.0, 100.0), failure_codes=(),
        )

    def test_ema_converges_and_counts_stable_frames(self):
        filt = Hand3DFilter(alpha=0.5, max_jump_m=0.10, loss_reset_frames=3)
        target = (0.10, -0.02, 0.40)
        first = filt.update(self._result((0.12, -0.03, 0.41)))
        self.assertEqual(first.status, "INIT")
        for _ in range(9):
            out = filt.update(self._result(target))
        self.assertEqual(out.status, "TRACKING")
        self.assertEqual(out.stable_frames, 10)
        for expected, actual in zip(target, out.position):
            self.assertAlmostEqual(expected, actual, places=3)

    def test_jump_reinitializes_to_new_target(self):
        filt = Hand3DFilter(alpha=0.5, max_jump_m=0.10, loss_reset_frames=3)
        filt.update(self._result((0.10, 0.0, 0.40)))
        out = filt.update(self._result((0.30, 0.0, 0.40)))
        self.assertEqual(out.status, "JUMP")
        self.assertEqual(out.position, (0.30, 0.0, 0.40))
        self.assertEqual(out.stable_frames, 1)

    def test_short_loss_keeps_stale_estimate_long_loss_drops(self):
        filt = Hand3DFilter(alpha=0.5, max_jump_m=0.10, loss_reset_frames=3)
        filt.update(self._result((0.10, 0.0, 0.40)))
        lost = Hand3DResult(None, 0.0, 0, 0.0, None, ("NO_LANDMARKS",))
        for _ in range(3):
            out = filt.update(lost)
        self.assertEqual(out.status, "STALE")
        self.assertEqual(out.position, (0.10, 0.0, 0.40))
        out = filt.update(lost)  # 4th consecutive loss -> dropped
        self.assertEqual(out.status, "LOST")
        self.assertIsNone(out.position)

    def test_reset_clears_state(self):
        filt = Hand3DFilter()
        filt.update(self._result((0.10, 0.0, 0.40)))
        filt.reset()
        self.assertIsNone(filt.position)
        self.assertEqual(filt.stable_frames, 0)


class PrimaryHandSelectionTest(unittest.TestCase):
    def _hand(self, center_x, center_y, scale):
        hand = []
        for index in range(21):
            hand.append([
                center_x + scale * (index % 5 - 2) / 20.0,
                center_y + scale * (index // 5 - 2) / 20.0,
            ])
        return hand

    def test_prefers_hand_inside_zone(self):
        small = self._hand(0.20, 0.20, 0.10)
        big = self._hand(0.70, 0.40, 0.30)
        index = select_primary_hand(
            [small, big], 960, 540, zone=(0.5, 0.9, 0.2, 0.7)
        )
        self.assertEqual(index, 1)

    def test_falls_back_to_largest_hand_outside_zone(self):
        small = self._hand(0.20, 0.20, 0.10)
        big = self._hand(0.30, 0.30, 0.30)
        index = select_primary_hand(
            [small, big], 960, 540, zone=(0.8, 0.9, 0.7, 0.9)
        )
        self.assertEqual(index, 1)

    def test_returns_none_for_no_hands(self):
        self.assertIsNone(select_primary_hand([], 960, 540))


class HandRoiTest(unittest.TestCase):
    def _hand(self, center_x, center_y, scale):
        return [[
            center_x + scale * (index % 5 - 2) / 20.0,
            center_y + scale * (index // 5 - 2) / 20.0,
        ] for index in range(21)]

    def test_roi_is_square_around_hand_and_zoom_scales_side(self):
        hand = self._hand(0.5, 0.5, 0.2)
        x0, y0, w, h = compute_hand_roi(hand, 1000, 800, zoom=2.0)
        self.assertAlmostEqual(w / h, 1.0, delta=2)
        # synthetic hand span = scale/5 = 0.04 normalized -> 40 px; zoom 2 -> 80
        self.assertGreaterEqual(w, 80 - 2)
        self.assertLessEqual(w, 80 + 2)
        # hand center stays inside the ROI
        self.assertLess(x0, 500)
        self.assertGreater(x0 + w, 500)

    def test_roi_clamps_to_frame_corners(self):
        hand = self._hand(0.05, 0.05, 0.3)
        x0, y0, w, h = compute_hand_roi(hand, 1000, 800, zoom=2.0)
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x0 + w, 1000)
        self.assertLessEqual(y0 + h, 800)

    def test_roi_rejects_invalid_zoom(self):
        with self.assertRaises(ValueError):
            compute_hand_roi(self._hand(0.5, 0.5, 0.2), 1000, 800, 0.5)


class HandSizeMeasurementTest(unittest.TestCase):
    def setUp(self):
        self.intrinsics = CameraIntrinsics(960, 540, 600.0, 600.0, 480.0, 270.0)

    def _project_hand(self, world, depth):
        return project_hand(self.intrinsics, world, depth)

    def test_measurement_round_trips_to_exact_depth(self):
        world, depth = make_planar_hand(0.05, -0.02, 0.55, LANDMARK_PAIR_SPECS)
        normalized = self._project_hand(world, depth)
        measured = pair_distances_from_measurement(
            normalized, 960, 540, self.intrinsics, depth
        )
        self.assertEqual(len(measured), 4)
        median, count, spread, _ = depth_from_landmarks(
            normalized, 960, 540, self.intrinsics, pair_specs=measured
        )
        self.assertEqual(count, 4)
        self.assertAlmostEqual(median, depth, places=6)

    def test_measurement_matches_expected_pair_distances(self):
        world, depth = make_planar_hand(0.0, 0.0, 0.55, LANDMARK_PAIR_SPECS)
        normalized = self._project_hand(world, depth)
        measured = pair_distances_from_measurement(
            normalized, 960, 540, self.intrinsics, depth
        )
        by_pair = {(i, j): d for i, j, d in measured}
        for first, second, expected in LANDMARK_PAIR_SPECS:
            self.assertAlmostEqual(
                by_pair[(first, second)], expected, places=6
            )

    def test_measurement_rejects_bad_depth(self):
        with self.assertRaises(ValueError):
            pair_distances_from_measurement(
                [[0.5, 0.5]] * 21, 960, 540, self.intrinsics, 0.0
            )

    def test_specs_roundtrip_through_json(self):
        import tempfile
        from pathlib import Path
        measured = pair_distances_from_measurement(
            self._project_hand(*make_planar_hand(0.0, 0.0, 0.55, LANDMARK_PAIR_SPECS)),
            960, 540, self.intrinsics, 0.55,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hand_specs.json"
            save_pair_specs(path, measured, depth_m=0.55, created="t")
            loaded = load_pair_specs(path)
        self.assertEqual(loaded, measured)

    def test_specs_reject_invalid_indices_and_distances(self):
        with self.assertRaises(ValueError):
            pair_specs_from_dict({"pairs": [[0, 21, 0.1]]})
        with self.assertRaises(ValueError):
            pair_specs_from_dict({"pairs": [[0, 0, 0.1]]})
        with self.assertRaises(ValueError):
            pair_specs_from_dict({"pairs": [[0, 1, -0.1]]})
        with self.assertRaises(ValueError):
            pair_specs_from_dict({"pairs": []})


if __name__ == "__main__":
    unittest.main()
