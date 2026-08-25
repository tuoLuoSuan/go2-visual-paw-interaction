"""Monocular 3D hand localization from MediaPipe-style normalized landmarks.

Pure-Python so the geometry can be regression-tested anywhere.  Depth comes
from known palm-scale distances between landmark pairs: for a calibrated
pinhole camera, depth = expected_world_distance * sqrt(fx*fy) / pixel_distance.
The median over several pairs is robust against single-landmark noise and mild
occlusion, which is enough for a soft handshake approach region (we never
claim contact-grade accuracy).

Landmarks are normalized (0..1) image coordinates, matching
``HandLandmarkerResult.hand_landmarks`` entries that expose ``.x``/``.y``.
"""

from dataclasses import dataclass
import math

from camera_intrinsics import CameraIntrinsics


PALM_LANDMARK_IDS = (0, 5, 9, 13, 17)

# Landmark index pairs with average adult metric distances (meters).
# Wrist=0, thumb CMC=1, index MCP=5, middle MCP=9, pinky MCP=17.
LANDMARK_PAIR_SPECS = (
    (0, 9, 0.105),
    (0, 5, 0.095),
    (5, 17, 0.078),
    (1, 17, 0.098),
)

DEFAULT_MIN_PAIR_PIXELS = 15.0
DEFAULT_MIN_DEPTH_M = 0.10
DEFAULT_MAX_DEPTH_M = 1.20


def compute_hand_roi(hand, width, height, zoom):
    """Return a square ROI around the hand for digital-zoom re-detection.

    The ROI side is the hand's pixel span times ``zoom``, centered on the
    hand bounding box, clamped to the frame.  Returns ``(x0, y0, w, h)``.
    """
    if not math.isfinite(float(zoom)) or float(zoom) < 1.0:
        raise ValueError("zoom 必须是不小于 1 的有限数")
    landmarks_xy = _normalized_xy(hand)
    if not landmarks_xy:
        raise ValueError("手部地标为空")
    xs = [point[0] * width for point in landmarks_xy]
    ys = [point[1] * height for point in landmarks_xy]
    center_x = 0.5 * (min(xs) + max(xs))
    center_y = 0.5 * (min(ys) + max(ys))
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    side = max(1.0, float(span) * float(zoom))
    half = side / 2.0
    x0 = max(0, int(center_x - half))
    y0 = max(0, int(center_y - half))
    x1 = min(int(width), int(math.ceil(center_x + half)))
    y1 = min(int(height), int(math.ceil(center_y + half)))
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def select_primary_hand(hands, width, height, zone=None):
    """Pick one deterministic primary hand from possibly several detections.

    ``hands`` is a sequence of 21-landmark sequences (normalized ``(x, y)``
    or MediaPipe landmark objects).  When ``zone`` is given as
    ``(x_min, x_max, y_min, y_max)`` normalized, hands whose palm center is
    inside the zone are preferred; ties are broken by the largest landmark
    pixel span.  Without a zone, or when no hand is inside it, the largest
    hand wins.  Returns the index into ``hands``, or ``None`` when empty.
    """
    if not hands:
        return None
    normalized = [_normalized_xy(hand) for hand in hands]

    def palm_center(hand):
        return (
            sum(hand[index][0] for index in PALM_LANDMARK_IDS) / len(PALM_LANDMARK_IDS),
            sum(hand[index][1] for index in PALM_LANDMARK_IDS) / len(PALM_LANDMARK_IDS),
        )

    def pixel_span(hand):
        best = 0.0
        for first in hand:
            for second in hand:
                distance = math.hypot(
                    (second[0] - first[0]) * width,
                    (second[1] - first[1]) * height,
                )
                best = max(best, distance)
        return best

    def in_zone(hand):
        if zone is None:
            return True
        x, y = palm_center(hand)
        x_min, x_max, y_min, y_max = (float(value) for value in zone)
        return x_min <= x <= x_max and y_min <= y <= y_max

    candidates = [
        index for index, hand in enumerate(normalized)
        if in_zone(hand)
    ] or list(range(len(normalized)))
    return max(candidates, key=lambda index: pixel_span(normalized[index]))


def pair_distances_from_measurement(landmarks, width, height, intrinsics,
                                    depth_m, pair_specs=LANDMARK_PAIR_SPECS):
    """Measure the user's real per-pair metric distances from one photo.

    With the hand held at a known ``depth_m`` in front of a calibrated
    camera, each pair's world distance is
    ``depth * pixel_distance / sqrt(fx*fy)``.  The result can be fed back
    into ``localize_palm`` as custom ``pair_specs``, replacing the generic
    adult-average table with the user's own hand (review item: hand-size
    prior calibration).
    """
    intrinsics.validate()
    depth_m = float(depth_m)
    if not math.isfinite(depth_m) or depth_m <= 0.0:
        raise ValueError("测量距离必须是有限正数")
    landmarks_xy = _normalized_xy(landmarks)
    if len(landmarks_xy) != 21:
        raise ValueError("MediaPipe 手部地标必须为 21 个")
    focal = math.sqrt(float(intrinsics.fx) * float(intrinsics.fy))
    measured = []
    for first, second, _ in pair_specs:
        x1, y1 = landmarks_xy[first]
        x2, y2 = landmarks_xy[second]
        pixel_distance = math.hypot(
            (x2 - x1) * width,
            (y2 - y1) * height,
        )
        if pixel_distance < 1e-9:
            raise ValueError("地标对像素距离过小，无法测量")
        measured.append((
            int(first), int(second),
            float(depth_m * pixel_distance / focal),
        ))
    return tuple(measured)


def pair_specs_from_dict(payload):
    """Parse custom landmark-pair specs from JSON-like data."""
    pairs = tuple(
        (int(first), int(second), float(distance))
        for first, second, distance in payload["pairs"]
    )
    if not pairs:
        raise ValueError("手部地标对规格不能为空")
    for first, second, distance in pairs:
        if not (0 <= first < 21 and 0 <= second < 21 and first != second):
            raise ValueError("地标对索引必须在 0..20 且不重复")
        if not math.isfinite(distance) or distance <= 0.0:
            raise ValueError("地标对距离必须是有限正数")
    return pairs


def pair_specs_to_dict(pair_specs, depth_m=0.0, created=""):
    return {
        "pairs": [
            [int(first), int(second), float(distance)]
            for first, second, distance in pair_specs
        ],
        "depth_m": float(depth_m),
        "created": str(created),
    }


def save_pair_specs(path, pair_specs, depth_m=0.0, created=""):
    import json
    from pathlib import Path
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(
            pair_specs_to_dict(pair_specs, depth_m, created),
            handle, ensure_ascii=False, indent=2,
        )
        handle.write("\n")
    return output.resolve()


def load_pair_specs(path):
    import json
    from pathlib import Path
    with Path(path).open(encoding="utf-8") as handle:
        return pair_specs_from_dict(json.load(handle))


@dataclass(frozen=True)
class Hand3DResult:
    palm_camera: object  # (x, y, z) in camera frame, or None on failure
    depth_m: float
    pair_count: int
    pair_spread_m: float
    palm_pixels: tuple  # (x, y) image position of the palm center, or None
    failure_codes: tuple

    @property
    def localized(self):
        return self.palm_camera is not None


def _normalized_xy(landmarks):
    """Accept either ``(x, y)`` tuples or MediaPipe landmark objects."""
    result = []
    for landmark in landmarks:
        if hasattr(landmark, "x") and hasattr(landmark, "y"):
            result.append((float(landmark.x), float(landmark.y)))
        else:
            result.append((float(landmark[0]), float(landmark[1])))
    return result


def _pair_depths(landmarks_xy, width, height, intrinsics, pair_specs,
                 min_pair_pixels):
    depths = []
    for first, second, expected in pair_specs:
        x1, y1 = landmarks_xy[first]
        x2, y2 = landmarks_xy[second]
        pixel_distance = math.hypot(
            (x2 - x1) * width,
            (y2 - y1) * height,
        )
        if pixel_distance < min_pair_pixels:
            continue
        focal = math.sqrt(float(intrinsics.fx) * float(intrinsics.fy))
        depths.append(float(expected) * focal / pixel_distance)
    return depths


def depth_from_landmarks(landmarks, width, height, intrinsics,
                         pair_specs=LANDMARK_PAIR_SPECS,
                         min_pair_pixels=DEFAULT_MIN_PAIR_PIXELS):
    """Return ``(median_depth_m, pair_count, spread_m, pair_depths)``.

    ``spread_m`` is ``max - min`` over the used pairs; a large spread means
    the hand pose is foreshortened and the median is less trustworthy.
    """
    if min_pair_pixels <= 0.0:
        raise ValueError("最小像素距离必须是正数")
    landmarks_xy = _normalized_xy(landmarks)
    if len(landmarks_xy) != 21:
        raise ValueError("MediaPipe 手部地标必须为 21 个")
    depths = _pair_depths(
        landmarks_xy, width, height, intrinsics, pair_specs, min_pair_pixels
    )
    if not depths:
        return None, 0, 0.0, ()
    ordered = sorted(depths)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else 0.5 * (ordered[middle - 1] + ordered[middle])
    )
    spread = max(ordered) - min(ordered)
    return float(median), len(ordered), float(spread), tuple(ordered)


def localize_palm(landmarks, width, height, intrinsics,
                  pair_specs=LANDMARK_PAIR_SPECS,
                  min_pair_pixels=DEFAULT_MIN_PAIR_PIXELS,
                  min_depth_m=DEFAULT_MIN_DEPTH_M,
                  max_depth_m=DEFAULT_MAX_DEPTH_M):
    """Localize the palm center in the camera frame, or a structured failure."""
    if not isinstance(intrinsics, CameraIntrinsics):
        raise TypeError("intrinsics 必须是 CameraIntrinsics")
    intrinsics.validate()
    failures = []
    palm_pixels = None
    try:
        landmarks_xy = _normalized_xy(landmarks)
    except (TypeError, ValueError, IndexError):
        return Hand3DResult(None, 0.0, 0, 0.0, None, ("INVALID_LANDMARKS",))
    if len(landmarks_xy) != 21:
        failures.append("INVALID_LANDMARK_COUNT")
        return Hand3DResult(None, 0.0, 0, 0.0, None, tuple(failures))

    palm_x = sum(landmarks_xy[index][0] for index in PALM_LANDMARK_IDS) / len(PALM_LANDMARK_IDS)
    palm_y = sum(landmarks_xy[index][1] for index in PALM_LANDMARK_IDS) / len(PALM_LANDMARK_IDS)
    palm_pixels = (float(palm_x * width), float(palm_y * height))

    depth, pair_count, spread, _ = depth_from_landmarks(
        landmarks_xy, width, height, intrinsics, pair_specs, min_pair_pixels
    )
    if depth is None or pair_count == 0:
        failures.append("INSUFFICIENT_PAIRS")
    if depth is not None and not (min_depth_m <= depth <= max_depth_m):
        failures.append("DEPTH_OUT_OF_RANGE")
    if failures:
        return Hand3DResult(None, 0.0, 0, 0.0, palm_pixels, tuple(failures))

    point = intrinsics.unproject(palm_pixels[0], palm_pixels[1], depth)
    return Hand3DResult(
        palm_camera=point,
        depth_m=float(depth),
        pair_count=int(pair_count),
        pair_spread_m=float(spread),
        palm_pixels=palm_pixels,
        failure_codes=(),
    )


@dataclass(frozen=True)
class FilteredHand3D:
    position: object  # smoothed (x, y, z) camera-frame point, or None
    raw_position: object
    status: str      # INIT | TRACKING | STALE | JUMP | LOST
    stable_frames: int

    @property
    def localized(self):
        return self.position is not None


class Hand3DFilter:
    """EMA jitter filter with jump re-init and loss reset.

    - A position jump larger than ``max_jump_m`` is treated as a new target
      (the filter re-initializes instead of averaging two different hands).
    - A short localization loss keeps the last estimate flagged ``STALE``;
      after ``loss_reset_frames`` consecutive losses the target is dropped
      (``LOST``) so a stale position is never reported as a live target.
    """

    def __init__(self, alpha=0.30, max_jump_m=0.10, loss_reset_frames=5):
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("alpha 必须在 (0, 1] 内")
        if float(max_jump_m) <= 0.0:
            raise ValueError("max_jump_m 必须是正数")
        if int(loss_reset_frames) < 1:
            raise ValueError("loss_reset_frames 必须为正整数")
        self.alpha = float(alpha)
        self.max_jump_m = float(max_jump_m)
        self.loss_reset_frames = int(loss_reset_frames)
        self.position = None
        self.lost_frames = 0
        self.stable_frames = 0

    def _distance(self, first, second):
        return math.sqrt(sum(
            (a - b) ** 2 for a, b in zip(first, second)
        ))

    def update(self, result):
        if not isinstance(result, Hand3DResult):
            raise TypeError("update 需要 Hand3DResult")
        raw = result.palm_camera
        if raw is None:
            self.lost_frames += 1
            self.stable_frames = 0
            if self.position is None:
                return FilteredHand3D(None, None, "LOST", 0)
            if self.lost_frames > self.loss_reset_frames:
                self.position = None
                return FilteredHand3D(None, None, "LOST", 0)
            return FilteredHand3D(
                self.position, None, "STALE", self.stable_frames
            )
        self.lost_frames = 0
        if self.position is None:
            self.position = raw
            self.stable_frames = 1
            return FilteredHand3D(raw, raw, "INIT", self.stable_frames)
        if self._distance(raw, self.position) > self.max_jump_m:
            self.position = raw
            self.stable_frames = 1
            return FilteredHand3D(raw, raw, "JUMP", self.stable_frames)
        smoothed = tuple(
            self.alpha * float(new) + (1.0 - self.alpha) * float(old)
            for new, old in zip(raw, self.position)
        )
        self.position = smoothed
        self.stable_frames += 1
        return FilteredHand3D(
            self.position, raw, "TRACKING", self.stable_frames
        )

    def reset(self):
        self.position = None
        self.lost_frames = 0
        self.stable_frames = 0
