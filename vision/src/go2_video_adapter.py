import cv2
import numpy as np


def _require_complete_jpeg(image_bytes):
    if image_bytes is None:
        raise ValueError("GO2 相机没有返回图像数据")

    payload = bytes(image_bytes)

    if not payload:
        raise ValueError("GO2 相机没有返回图像数据")

    if (
        payload.startswith(b"\xff\xd8")
        and not payload.endswith(b"\xff\xd9")
    ):
        raise ValueError("GO2 相机 JPEG 数据不完整")

    return payload


def decode_compressed_image(image_bytes):
    payload = _require_complete_jpeg(image_bytes)
    encoded = np.frombuffer(payload, dtype=np.uint8)

    try:
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise ValueError("无法解码 GO2 相机图像") from error

    if image is None:
        raise ValueError("无法解码 GO2 相机图像")

    return image


def get_frame_from_video_client(video_client):
    response_code, image_data = video_client.GetImageSample()

    if response_code != 0:
        raise RuntimeError(
            f"GO2 相机请求失败，错误码：{response_code}"
        )

    return decode_compressed_image(image_data)


class Go2FrameReader:
    def __init__(
        self,
        video_client,
        max_consecutive_failures=10,
    ):
        if max_consecutive_failures < 1:
            raise ValueError("最大连续失败帧数必须大于 0")

        self.video_client = video_client
        self.max_consecutive_failures = max_consecutive_failures
        self.total_failures = 0
        self.consecutive_failures = 0
        self.last_error = None

    def read(self):
        try:
            frame = get_frame_from_video_client(self.video_client)
        except (RuntimeError, ValueError, TypeError) as error:
            self.total_failures += 1
            self.consecutive_failures += 1
            self.last_error = str(error)

            if (
                self.consecutive_failures
                >= self.max_consecutive_failures
            ):
                raise RuntimeError(
                    "GO2 相机连续 "
                    f"{self.consecutive_failures} 帧读取失败，"
                    f"最后错误：{self.last_error}"
                ) from error

            return None

        self.consecutive_failures = 0
        self.last_error = None
        return frame
