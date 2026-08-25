#!/usr/bin/env python3
"""GO2 real handshake, milestone 3 vision feed: hand -> UDP target stream.

Detects the user's hand with MediaPipe, localizes the palm in the camera
frame (palm-scale depth), maps it to the FR approach target in the robot
base frame via the configured extrinsics, and streams the target over UDP
to the VMC2 process (``real_vmc_track_m2.py --fr-target-udp PORT``).

Stream format: one packet per frame, UTF-8 text:
  "x y z"   -> hand localized & mapped, base-frame meters
  "none"    -> hand lost / not in zone / localization failed

Safety: this process never sends any command to the robot; it only emits
targets.  The VMC2 side validates the workspace and verifies IK reachability
before moving the paw, and its watchdogs abort on any tracking error.

Usage (Ubuntu VM, dog connected, remote in hand):
  python3 real_go2/real_vision_feed_m3.py \
      --model vision/models/hand_landmarker.task \
      --network-interface YOUR_NETWORK_INTERFACE \
      --intrinsics vision/output/calibration/intrinsics_placeholder.json \
      --extrinsics vision/output/calibration/extrinsics_placeholder.json \
      --target-port 3999 --no-display
"""

import argparse
import os
import socket
import sys
import time
from pathlib import Path

import cv2

WORKSPACE = Path(__file__).resolve().parents[1]
for _src in ("vision/src", "simulation/src", "robot/src"):
    _path = WORKSPACE / _src
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

from camera_intrinsics import CameraIntrinsics
from detect_hand_camera import (
    ReadyStabilizer,
    ZONE_X_MAX, ZONE_X_MIN, ZONE_Y_MAX, ZONE_Y_MIN,
)
from detect_hand_go2 import create_video_client
from go2_video_adapter import Go2FrameReader
from hand_3d_localizer import (
    Hand3DFilter,
    Hand3DResult,
    PALM_LANDMARK_IDS,
    localize_palm,
    select_primary_hand,
)
from hand_to_fr_target import CameraExtrinsics, FrTargetConfig, palm_to_fr_target

ZONE = (ZONE_X_MIN, ZONE_X_MAX, ZONE_Y_MIN, ZONE_Y_MAX)


def palm_in_zone(hand):
    x = sum(hand[index].x for index in PALM_LANDMARK_IDS) / len(PALM_LANDMARK_IDS)
    y = sum(hand[index].y for index in PALM_LANDMARK_IDS) / len(PALM_LANDMARK_IDS)
    return ZONE_X_MIN <= x <= ZONE_X_MAX and ZONE_Y_MIN <= y <= ZONE_Y_MAX


class TargetSender:
    """UDP target stream toward the VMC2 process (127.0.0.1:port)."""

    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = ("127.0.0.1", int(port))
        self.sent = 0
        self.last_target = None

    def send(self, target):
        if target is None:
            payload = b"none"
        elif isinstance(target, str):
            payload = target.encode("utf-8")  # 'N' 无手
        elif len(target) == 2:
            # (px, py) 归一化坐标
            payload = ("%.4f %.4f" % (float(target[0]), float(target[1]))
                       ).encode("utf-8")
        else:
            payload = ("%.4f %.4f %.4f" % tuple(float(v) for v in target)
                       ).encode("utf-8")
        try:
            self.sock.sendto(payload, self.addr)
        except OSError:
            pass  # no listener yet (VMC2 starts first on the real run)
        self.sent += 1
        self.last_target = target

    def close(self):
        self.sock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--network-interface", required=True)
    parser.add_argument("--mock-image")
    parser.add_argument("--intrinsics")
    parser.add_argument("--extrinsics")
    parser.add_argument("--target-port", type=int, default=3999,
                        help="UDP 目标端口（VMC2 --fr-target-udp）")
    parser.add_argument("--side-only", action="store_true",
                        help="只发左右信号（'L'/'R'/'N'，基于手在画面中心"
                             "左/右侧），供趴姿握手选爪；不依赖 3D 外参标定")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--every-n-frames", type=int, default=1,
                        help="每 N 帧检测一次（视觉与 VMC 抢 CPU，真机上设 "
                             "3-5 保 VMC 控制率）")
    parser.add_argument("--tflite-threads", type=int, default=1,
                        help="MediaPipe/TFLite 推理线程数（默认 1，避免抢 "
                             "VMC 实时循环的 CPU）")
    parser.add_argument("--log-every", type=int, default=60,
                        help="每 N 帧打印一次目标")
    args = parser.parse_args(argv)

    is_mock = args.network_interface == "mock"
    if is_mock and not args.mock_image:
        parser.error("mock 模式必须提供 --mock-image")

    calib = WORKSPACE / "vision" / "output" / "calibration"
    intrinsics = CameraIntrinsics.load(
        args.intrinsics or calib / "intrinsics_placeholder.json"
    )
    extrinsics = None
    if args.extrinsics:
        extrinsics = CameraExtrinsics.load(args.extrinsics)
    elif (calib / "extrinsics_placeholder.json").exists():
        extrinsics = CameraExtrinsics.load(calib / "extrinsics_placeholder.json")
    if extrinsics is None:
        print("[WARN] 未配置外参：只能发 2D zone 门控的 'none'（不追手）",
              file=sys.stderr)
    # The VMC2 side tracks targets in FR_WORKSPACE
    # (x 0.10-0.30, y -0.20--0.02, z 0.15-0.25).  Clamp to the SAME box here
    # so a streamed target is never dropped as out-of-workspace downstream
    # and never asks the dog to track a rear-tipping height (>0.25 m
    # measured to tip the dog backward).
    fr_config = FrTargetConfig(
        approach_offset_m=(0.0, 0.0, 0.0),
        reach_box_m=(0.10, 0.30, -0.20, -0.02, 0.15, 0.25),
        allow_uncalibrated=True,
    )

    video_client = create_video_client(
        network_interface=args.network_interface,
        mock_image=args.mock_image,
    )
    frame_reader = Go2FrameReader(video_client, max_consecutive_failures=10)
    stabilizer = ReadyStabilizer(required_frames=10)
    hand_filter = Hand3DFilter(alpha=0.30, max_jump_m=0.10, loss_reset_frames=5)
    base_kwargs = {
        "model_asset_path": args.model,
        "delegate": python.BaseOptions.Delegate.CPU,
    }
    try:
        base_kwargs["num_threads"] = max(1, int(args.tflite_threads))
        python.BaseOptions(**base_kwargs)
    except TypeError:
        # 老版本 mediapipe 不支持 num_threads
        base_kwargs.pop("num_threads", None)
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(**base_kwargs),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    sender = TargetSender(args.target_port)
    print(f"[INFO] UDP 目标流 -> 127.0.0.1:{args.target_port}",
          file=sys.stderr)

    last_timestamp_ms = 0
    frames = 0
    every_n = max(1, int(args.every_n_frames))
    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                frame = frame_reader.read()
                if frame is None:
                    continue
                if not is_mock:
                    frame = cv2.resize(frame, (960, 540),
                                       interpolation=cv2.INTER_AREA)
                frames += 1
                # downsampled detection: skip inference, keep sending the
                # last known target so the VMC loop keeps tracking
                if frames % every_n != 0:
                    continue
                height, width = frame.shape[:2]
                timestamp_ms = max(last_timestamp_ms + 1,
                                   int(time.monotonic() * 1000))
                last_timestamp_ms = timestamp_ms
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                # side_only 模式：选最大手（不限制 zone——狗主动找手够手）
                if args.side_only:
                    primary = select_primary_hand(result.hand_landmarks,
                                                  width, height)
                else:
                    primary = select_primary_hand(result.hand_landmarks,
                                                  width, height, zone=ZONE)
                hand = (result.hand_landmarks[primary]
                        if primary is not None else None)
                in_zone = bool(hand is not None and palm_in_zone(hand))
                is_ready = stabilizer.update(in_zone)

                # hand 3D (camera frame) -> base frame target
                target = None
                notes = []
                if args.side_only:
                    # 连续手位置：发 'px py'（0-1 归一化，替代三态 L/R/C）
                    # 爪子实时跟手：px 左右、py 上下
                    if hand is not None:
                        px = sum(hand[i].x for i in PALM_LANDMARK_IDS) \
                            / len(PALM_LANDMARK_IDS)
                        py = sum(hand[i].y for i in PALM_LANDMARK_IDS) \
                            / len(PALM_LANDMARK_IDS)
                        target = (px, py)
                    else:
                        target = "N"
                elif hand is not None and is_ready:
                    if extrinsics is not None:
                        filtered = hand_filter.update(localize_palm(
                            hand, width, height,
                            intrinsics.scaled(width, height)))
                        if filtered.position is not None:
                            fr_target = palm_to_fr_target(
                                filtered.position, extrinsics, fr_config)
                            if fr_target.position is not None:
                                target = fr_target.position
                            notes = list(fr_target.notes)
                    else:
                        notes.append("NO_EXTRINSICS")
                elif hand is not None:
                    notes.append("STABILIZING")
                else:
                    hand_filter.update(Hand3DResult(
                        palm_camera=None, depth_m=0.0, pair_count=0,
                        pair_spread_m=0.0, palm_pixels=None,
                        failure_codes=("NO_HAND",)))

                sender.send(target)

                if frames % args.log_every == 0:
                    status = "追踪中" if target is not None else "none"
                    extra = (" | ".join(notes) if notes else "")
                    print(f"[INFO] frame={frames} send={sender.sent} "
                          f"status={status} {extra} "
                          f"ready={is_ready} zone={in_zone}",
                          file=sys.stderr)
                if not args.no_display:
                    if hand is not None:
                        for landmark in hand:
                            x = max(0, min(width - 1, int(landmark.x * width)))
                            y = max(0, min(height - 1, int(landmark.y * height)))
                            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                    if args.side_only:
                        # side_only：显示手位置 px/py
                        color = (0, 255, 0) if target is not None else (0, 165, 255)
                        text = "NO HAND"
                        if isinstance(target, tuple):
                            text = ("HAND px=%.2f py=%.2f"
                                    % (target[0], target[1]))
                        # 左右中分界线
                        cv2.line(frame, (int(0.40 * width), 0),
                                 (int(0.40 * width), height),
                                 (255, 0, 0), 2)
                        cv2.line(frame, (int(0.60 * width), 0),
                                 (int(0.60 * width), height),
                                 (255, 0, 0), 2)
                        cv2.putText(frame, "L", (int(0.05 * width), 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    (255, 0, 0), 2)
                        cv2.putText(frame, "C", (int(0.45 * width), 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    (255, 0, 0), 2)
                        cv2.putText(frame, "R", (int(0.85 * width), 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    (255, 0, 0), 2)
                    else:
                        zone_left = int(ZONE_X_MIN * width)
                        zone_right = int(ZONE_X_MAX * width)
                        zone_top = int(ZONE_Y_MIN * height)
                        zone_bottom = int(ZONE_Y_MAX * height)
                        color = (0, 255, 0) if target is not None else (0, 165, 255)
                        cv2.rectangle(frame, (zone_left, zone_top),
                                      (zone_right, zone_bottom), color, 3)
                        text = (f"{'TRACK' if target is not None else 'NONE'} | "
                                f"send={sender.sent}")
                        if target is not None:
                            text += f" | tgt=({target[0]:.2f},{target[1]:.2f},{target[2]:.2f})"
                    cv2.putText(frame, text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    try:
                        cv2.imshow("GO2 Vision Feed M3 - Q to quit", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            break
                    except cv2.error:
                        pass  # 无 DISPLAY：跳过显示
                if args.max_frames and frames >= args.max_frames:
                    break
    finally:
        sender.close()
    print(f"[RESULT] VISION_FEED_DONE frames={frames} sent={sender.sent}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
