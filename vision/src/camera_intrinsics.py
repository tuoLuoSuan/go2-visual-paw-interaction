"""Pinhole camera intrinsics storage and projection helpers.

Pure-Python (no cv2/mediapipe) so the math can be regression-tested on any
platform.  Convention follows OpenCV: image origin top-left, x right, y down,
camera frame z forward.  Distortion coefficients are stored as
``(k1, k2, p1, p2, k3)``; for localization we currently ignore distortion
(MediaPipe landmarks come from the raw image, and the handshake target is a
soft approach region, not a contact point) but keep the coefficients in the
JSON so a calibrated camera's distortion is never silently lost.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)
    calibrated: bool = False
    rms: float = 0.0
    created: str = ""

    def validate(self):
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("内参图像尺寸必须为正整数")
        for name, value in (
            ("fx", self.fx),
            ("fy", self.fy),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"内参 {name} 必须是有限正数")
        for name, value in (
            ("cx", self.cx),
            ("cy", self.cy),
        ):
            # principal point may sit outside a cropped ROI (negative ok)
            if not math.isfinite(float(value)):
                raise ValueError(f"内参 {name} 必须是有限数")
        if len(self.dist) != 5 or any(
            not math.isfinite(float(value)) for value in self.dist
        ):
            raise ValueError("畸变系数必须是 5 个有限数")
        return self

    def scaled(self, width, height):
        """Return intrinsics valid for a different detection resolution."""
        self.validate()
        scale_x = float(width) / float(self.width)
        scale_y = float(height) / float(self.height)
        return CameraIntrinsics(
            width=int(width),
            height=int(height),
            fx=float(self.fx) * scale_x,
            fy=float(self.fy) * scale_y,
            cx=float(self.cx) * scale_x,
            cy=float(self.cy) * scale_y,
            dist=self.dist,
            calibrated=self.calibrated,
            rms=self.rms,
            created=self.created,
        ).validate()

    def crop(self, x0, y0, crop_width, crop_height):
        """Return intrinsics valid for a pixel crop (ROI digital zoom).

        A pure crop keeps the focal lengths; only the principal point
        shifts into the ROI coordinate system.  Chain with ``scaled`` when
        the ROI is also resized.
        """
        self.validate()
        x0, y0 = int(x0), int(y0)
        crop_width, crop_height = int(crop_width), int(crop_height)
        if x0 < 0 or y0 < 0 or crop_width <= 0 or crop_height <= 0:
            raise ValueError("裁剪区域无效")
        if x0 + crop_width > self.width or y0 + crop_height > self.height:
            raise ValueError("裁剪区域超出图像范围")
        return CameraIntrinsics(
            width=crop_width,
            height=crop_height,
            fx=self.fx,
            fy=self.fy,
            cx=float(self.cx) - x0,
            cy=float(self.cy) - y0,
            dist=self.dist,
            calibrated=self.calibrated,
            rms=self.rms,
            created=self.created,
        ).validate()

    def unproject(self, pixel_x, pixel_y, depth):
        """Map pixel + depth to a 3D point in the camera frame."""
        self.validate()
        depth = float(depth)
        if not math.isfinite(depth) or depth <= 0.0:
            raise ValueError("深度必须是有限正数")
        return (
            float((float(pixel_x) - self.cx) * depth / self.fx),
            float((float(pixel_y) - self.cy) * depth / self.fy),
            depth,
        )

    def project(self, point):
        """Map a 3D camera-frame point to pixel coordinates (no distortion)."""
        self.validate()
        x, y, z = (float(value) for value in point)
        if not math.isfinite(z) or z <= 0.0:
            raise ValueError("投影深度必须是有限正数")
        return (
            float(self.fx * x / z + self.cx),
            float(self.fy * y / z + self.cy),
        )

    def to_dict(self):
        return {
            "width": int(self.width),
            "height": int(self.height),
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "dist": [float(value) for value in self.dist],
            "calibrated": bool(self.calibrated),
            "rms": float(self.rms),
            "created": str(self.created),
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            width=int(payload["width"]),
            height=int(payload["height"]),
            fx=float(payload["fx"]),
            fy=float(payload["fy"]),
            cx=float(payload["cx"]),
            cy=float(payload["cy"]),
            dist=tuple(float(value) for value in payload.get("dist", (0.0,) * 5)),
            calibrated=bool(payload.get("calibrated", False)),
            rms=float(payload.get("rms", 0.0)),
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
