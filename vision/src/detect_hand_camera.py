import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat


PALM_LANDMARK_IDS = (0, 5, 9, 13, 17)

ZONE_X_MIN = 0.55
ZONE_X_MAX = 0.85
ZONE_Y_MIN = 0.25
ZONE_Y_MAX = 0.60


class DetectionCsvLogger:
    FIELDNAMES = (
        "timestamp_ms",
        "palm_x",
        "palm_y",
        "consecutive_frames",
        "status",
    )

    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.csv_path.open(
            "w", encoding="utf-8", newline=""
        )
        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=self.FIELDNAMES,
        )
        self.writer.writeheader()

    def write(
        self,
        timestamp_ms,
        palm_x,
        palm_y,
        consecutive_frames,
        status,
    ):
        self.writer.writerow(
            {
                "timestamp_ms": timestamp_ms,
                "palm_x": palm_x,
                "palm_y": palm_y,
                "consecutive_frames": consecutive_frames,
                "status": status,
            }
        )
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


class ReadyStabilizer:
    def __init__(self, required_frames=10):
        self.required_frames = required_frames
        self.consecutive_frames = 0

    def update(self, in_zone):
        if in_zone:
            self.consecutive_frames += 1
        else:
            self.consecutive_frames = 0

        return self.consecutive_frames >= self.required_frames

def annotate_frame(frame, result, stabilizer):
    height, width = frame.shape[:2]

    zone_left = int(ZONE_X_MIN * width)
    zone_right = int(ZONE_X_MAX * width)
    zone_top = int(ZONE_Y_MIN * height)
    zone_bottom = int(ZONE_Y_MAX * height)

    in_handshake_zone = False
    palm_x = None
    palm_y = None

    for hand in result.hand_landmarks:
        for landmark in hand:
            x = max(0, min(width - 1, int(landmark.x * width)))
            y = max(0, min(height - 1, int(landmark.y * height)))
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        palm_x_normalized = sum(
            hand[index].x for index in PALM_LANDMARK_IDS
        ) / len(PALM_LANDMARK_IDS)

        palm_y_normalized = sum(
            hand[index].y for index in PALM_LANDMARK_IDS
        ) / len(PALM_LANDMARK_IDS)

        palm_x = int(palm_x_normalized * width)
        palm_y = int(palm_y_normalized * height)

        cv2.circle(frame, (palm_x, palm_y), 8, (255, 0, 255), -1)

        if (
            ZONE_X_MIN <= palm_x_normalized <= ZONE_X_MAX
            and ZONE_Y_MIN <= palm_y_normalized <= ZONE_Y_MAX
        ):
            in_handshake_zone = True

    is_ready = stabilizer.update(in_handshake_zone)
    zone_color = (0, 255, 0) if is_ready else (0, 165, 255)

    cv2.rectangle(
        frame,
        (zone_left, zone_top),
        (zone_right, zone_bottom),
        zone_color,
        3,
    )

    if is_ready:
        status = "READY"
    elif in_handshake_zone:
        status = (
            f"HOLD {stabilizer.consecutive_frames}/"
            f"{stabilizer.required_frames}"
        )
    elif result.hand_landmarks:
        status = "MOVE HAND INTO ZONE"
    else:
        status = "NO HAND"

    cv2.putText(
        frame,
        status,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        zone_color,
        2,
    )

    return frame, palm_x, palm_y, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)

    if not camera.isOpened():
        print(
            f"无法打开摄像头：/dev/video{args.camera}",
            file=sys.stderr,
        )
        return 2

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=args.model),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_count = 0
    last_timestamp_ms = 0
    stabilizer = ReadyStabilizer(required_frames=10)

    log_path = Path("vision/output/logs") / (
        "hand_detection_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + ".csv"
    )
    logger = DetectionCsvLogger(log_path)
    print(f"CSV_LOG_STARTED path={log_path}")

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = camera.read()

                if not ok or frame is None:
                    print("摄像头读取失败", file=sys.stderr)
                    return 3

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = Image(
                    image_format=ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp_ms = max(
                    last_timestamp_ms + 1,
                    int(time.monotonic() * 1000),
                )
                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                annotated, palm_x, palm_y, status = annotate_frame(
                    frame,
                    result,
                    stabilizer,
                )

                logger.write(
                    timestamp_ms=timestamp_ms,
                    palm_x=palm_x,
                    palm_y=palm_y,
                    consecutive_frames=stabilizer.consecutive_frames,
                    status=status,
                )
                frame_count += 1

                if not args.headless:
                    cv2.imshow(
                        "GO2 Hand Detection - Q to quit",
                        annotated,
                    )

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

                if (
                    args.max_frames > 0
                    and frame_count >= args.max_frames
                ):
                    break
    finally:
        logger.close()
        camera.release()
        cv2.destroyAllWindows()

    print(f"CSV_LOG_SAVED path={log_path}")
    print(f"CAMERA_STREAM_OK frames={frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
