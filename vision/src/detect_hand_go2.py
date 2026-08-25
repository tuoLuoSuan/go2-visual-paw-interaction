import argparse
import sys
import time
from pathlib import Path

import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

from detect_hand_camera import (
    DetectionCsvLogger,
    ReadyStabilizer,
    annotate_frame,
)
from go2_video_adapter import Go2FrameReader


class MockVideoClient:
    def __init__(self, image_path):
        self.image_data = Path(image_path).read_bytes()

    def GetImageSample(self):
        return 0, self.image_data


def create_video_client(
    network_interface,
    mock_image=None,
    channel_initializer=None,
    video_client_class=None,
):
    if network_interface == "mock":
        if not mock_image:
            raise ValueError("mock 模式必须提供 --mock-image")
        return MockVideoClient(mock_image)

    if channel_initializer is None:
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
        )

        channel_initializer = ChannelFactoryInitialize

    if video_client_class is None:
        from unitree_sdk2py.go2.video.video_client import (
            VideoClient,
        )

        video_client_class = VideoClient

    channel_initializer(0, network_interface)
    video_client = video_client_class()
    video_client.SetTimeout(3.0)
    video_client.Init()
    return video_client


def main():
    parser = argparse.ArgumentParser(
        description="GO2 front camera hand detection"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--network-interface", required=True)
    parser.add_argument("--mock-image")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if (
        args.network_interface == "mock"
        and not args.mock_image
    ):
        parser.error("mock 模式必须提供 --mock-image")

    video_client = create_video_client(
        network_interface=args.network_interface,
        mock_image=args.mock_image,
    )
    frame_reader = Go2FrameReader(
        video_client,
        max_consecutive_failures=10,
    )
    is_mock = args.network_interface == "mock"

    logger = DetectionCsvLogger(args.output_csv)
    stabilizer = ReadyStabilizer(required_frames=10)

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=args.model
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_count = 0
    last_timestamp_ms = 0
    exit_code = 0

    try:
        with vision.HandLandmarker.create_from_options(
            options
        ) as landmarker:
            while True:
                frame = frame_reader.read()

                if frame is None:
                    print(
                        "GO2_FRAME_SKIPPED "
                        f"total={frame_reader.total_failures} "
                        f"error={frame_reader.last_error}",
                        file=sys.stderr,
                    )
                    continue

                if not is_mock:
                    frame = cv2.resize(
                        frame,
                        (960, 540),
                        interpolation=cv2.INTER_AREA,
                    )

                timestamp_ms = max(
                    last_timestamp_ms + 1,
                    int(time.monotonic() * 1000),
                )
                last_timestamp_ms = timestamp_ms

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = Image(
                    image_format=ImageFormat.SRGB,
                    data=rgb,
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                annotated, palm_x, palm_y, status = (
                    annotate_frame(
                        frame,
                        result,
                        stabilizer,
                    )
                )

                logger.write(
                    timestamp_ms=timestamp_ms,
                    palm_x=palm_x,
                    palm_y=palm_y,
                    consecutive_frames=(
                        stabilizer.consecutive_frames
                    ),
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
    except RuntimeError as error:
        print(
            f"GO2_CAMERA_ABORTED error={error}",
            file=sys.stderr,
        )
        exit_code = 4
    finally:
        logger.close()
        cv2.destroyAllWindows()

    if exit_code != 0:
        return exit_code

    mode = "MOCK" if is_mock else "REAL"
    print(
        f"GO2_{mode}_DETECTION_OK "
        f"frames={frame_count} "
        f"failures={frame_reader.total_failures}"
    )
    print(f"OUTPUT_CSV={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
