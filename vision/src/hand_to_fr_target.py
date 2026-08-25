"""Map a localized palm position to an FR foot target in the robot base frame.

Pure-Python transform chain: camera frame (OpenCV convention: z forward)
-> robot base frame (x forward, y left, z up) via configurable extrinsics
-> FR foot approach target below the palm.

The extrinsic values are NOT yet calibrated on the real robot.  Every
``FrTarget`` therefore carries ``calibrated`` from the extrinsic config, and
``palm_to_fr_target`` refuses to emit a ``reachable=True`` target from
uncalibrated extrinsics unless ``allow_uncalibrated`` is explicitly enabled
(offline demo only).  This mirrors the project rule that simulation/vision
outputs never authorize real motion.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class CameraExtrinsics:
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    translation: tuple  # (x, y, z) camera origin in base frame, meters
    calibrated: bool = False
    created: str = ""

    def validate(self):
        for name, value in (
            ("roll_deg", self.roll_deg),
            ("pitch_deg", self.pitch_deg),
            ("yaw_deg", self.yaw_deg),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"外参 {name} 必须是有限数")
        if len(self.translation) != 3 or any(
            not math.isfinite(float(value)) for value in self.translation
        ):
            raise ValueError("外参平移必须是 3 个有限数")
        return self

    def rotation_matrix(self):
        """R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
        roll = math.radians(self.roll_deg)
        pitch = math.radians(self.pitch_deg)
        yaw = math.radians(self.yaw_deg)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )

    def transform(self, point):
        """Map a camera-frame point to the base frame."""
        self.validate()
        if len(point) != 3 or any(
            not math.isfinite(float(value)) for value in point
        ):
            raise ValueError("被变换点必须是 3 个有限数")
        rotation = self.rotation_matrix()
        x, y, z = (float(value) for value in point)
        tx, ty, tz = (float(value) for value in self.translation)
        return (
            rotation[0][0] * x + rotation[0][1] * y + rotation[0][2] * z + tx,
            rotation[1][0] * x + rotation[1][1] * y + rotation[1][2] * z + ty,
            rotation[2][0] * x + rotation[2][1] * y + rotation[2][2] * z + tz,
        )

    def to_dict(self):
        return {
            "roll_deg": float(self.roll_deg),
            "pitch_deg": float(self.pitch_deg),
            "yaw_deg": float(self.yaw_deg),
            "translation": [float(value) for value in self.translation],
            "calibrated": bool(self.calibrated),
            "created": str(self.created),
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            roll_deg=float(payload["roll_deg"]),
            pitch_deg=float(payload["pitch_deg"]),
            yaw_deg=float(payload["yaw_deg"]),
            translation=tuple(float(value) for value in payload["translation"]),
            calibrated=bool(payload.get("calibrated", False)),
            created=str(payload.get("created", "")),
        ).validate()

    def save(self, path):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return output.resolve()

    @classmethod
    def load(cls, path):
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


@dataclass(frozen=True)
class FrTargetConfig:
    approach_offset_m: tuple = (0.0, 0.0, -0.06)
    # SIMULATION-ONLY default (2026-08-17 solver v2 grid scan convex hull).
    # The REAL robot low-level VMC tracker (real_go2/real_vmc_track_m2.py
    # FR_WORKSPACE: x 0.10-0.30, y -0.20--0.02, z 0.15-0.25) must be passed
    # explicitly - anyone using this default for real-machine targets gets
    # the wrong box (z cap 0.12 vs real 0.25) and silently clamps wrongly.
    reach_box_m: tuple = (0.10, 0.28, -0.20, -0.08, 0.04, 0.12)
    allow_uncalibrated: bool = False

    def validate(self):
        if len(self.approach_offset_m) != 3 or any(
            not math.isfinite(float(value)) for value in self.approach_offset_m
        ):
            raise ValueError("接近偏移必须是 3 个有限数")
        if len(self.reach_box_m) != 6:
            raise ValueError("可达盒必须是 (x_min, x_max, y_min, y_max, z_min, z_max)")
        x_min, x_max, y_min, y_max, z_min, z_max = (
            float(value) for value in self.reach_box_m
        )
        if not (x_min < x_max and y_min < y_max and z_min < z_max):
            raise ValueError("可达盒边界必须严格递增")
        return self


@dataclass(frozen=True)
class FrTarget:
    position: object     # approach target in base frame (clamped), or None
    raw_position: object  # pre-clamp position in base frame, or None
    clamped: bool
    reachable: bool
    calibrated: bool
    notes: tuple

    @property
    def valid(self):
        return self.position is not None and self.reachable


def clamp_to_box(point, box):
    x_min, x_max, y_min, y_max, z_min, z_max = (float(value) for value in box)
    x, y, z = (float(value) for value in point)
    clamped = (
        max(x_min, min(x_max, x)),
        max(y_min, min(y_max, y)),
        max(z_min, min(z_max, z)),
    )
    return clamped, clamped != (x, y, z)


def palm_to_fr_target(palm_camera, extrinsics, config=None):
    """Palm 3D point (camera frame) -> FR foot approach target (base frame)."""
    if not isinstance(extrinsics, CameraExtrinsics):
        raise TypeError("extrinsics 必须是 CameraExtrinsics")
    configuration = (FrTargetConfig() if config is None else config).validate()
    extrinsics.validate()
    if palm_camera is None or len(palm_camera) != 3 or any(
        not math.isfinite(float(value)) for value in palm_camera
    ):
        return FrTarget(None, None, False, False, extrinsics.calibrated,
                        ("INVALID_PALM_POSITION",))
    palm_base = extrinsics.transform(palm_camera)
    raw = tuple(
        float(value) + float(offset)
        for value, offset in zip(palm_base, configuration.approach_offset_m)
    )
    position, clamped = clamp_to_box(raw, configuration.reach_box_m)
    notes = []
    if clamped:
        notes.append("CLAMPED_TO_REACH_BOX")
    if not extrinsics.calibrated and not configuration.allow_uncalibrated:
        notes.append("EXTRINSICS_UNCALIBRATED")
    reachable = not notes or notes == ["CLAMPED_TO_REACH_BOX"]
    return FrTarget(
        position=position,
        raw_position=raw,
        clamped=clamped,
        reachable=reachable,
        calibrated=extrinsics.calibrated,
        notes=tuple(notes),
    )
