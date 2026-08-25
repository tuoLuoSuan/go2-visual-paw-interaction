import argparse
import sys
import time
from pathlib import Path

import mujoco


CANONICAL_LEG_JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)


def load_go2_model(scene_path):
    path = Path(scene_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"找不到 MuJoCo 场景：{path}"
        )

    # 注意：用 absolute() 而不是 resolve()——resolve() 会把 Windows
    # junction 还原成真实路径（可能含非 ASCII 字符，MuJoCo 打不开），
    # absolute() 保留调用方给的词法路径（英文 junction 名可直接用）。
    return mujoco.MjModel.from_xml_path(
        str(path.absolute())
    )


def leg_joint_names(model):
    missing = [
        name
        for name in CANONICAL_LEG_JOINT_NAMES
        if mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        ) < 0
    ]

    if missing:
        raise ValueError(
            "GO2 模型缺少腿部关节："
            + ", ".join(missing)
        )

    return CANONICAL_LEG_JOINT_NAMES


def create_go2_data(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return data
def positive_seconds(value):
    seconds = float(value)

    if seconds <= 0:
        raise argparse.ArgumentTypeError(
            "viewer-seconds 必须大于 0"
        )

    return seconds


def run_viewer(model, data, viewer_seconds):
    if viewer_seconds <= 0:
        raise ValueError(
            "viewer_seconds 必须大于 0"
        )

    from mujoco import viewer

    deadline = time.monotonic() + viewer_seconds

    with viewer.launch_passive(
        model,
        data,
    ) as active_viewer:
        while (
            active_viewer.is_running()
            and time.monotonic() < deadline
        ):
            active_viewer.sync()
            time.sleep(1.0 / 60.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Load and inspect the official "
            "GO2 MuJoCo model"
        )
    )
    parser.add_argument(
        "--scene",
        required=True,
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Load and validate without "
            "opening a window"
        ),
    )
    parser.add_argument(
        "--viewer-seconds",
        type=positive_seconds,
        default=10.0,
        help="Maximum Viewer runtime in seconds",
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    try:
        scene_path = Path(args.scene).resolve()
        model = load_go2_model(scene_path)
        joints = leg_joint_names(model)
        data = create_go2_data(model)

        if not args.headless:
            run_viewer(
                model,
                data,
                args.viewer_seconds,
            )
    except (
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            f"GO2_MUJOCO_LOAD_ERROR: {error}",
            file=sys.stderr,
        )
        return 2

    print("GO2_MUJOCO_LOAD_OK")
    print(f"MUJOCO_VERSION={mujoco.__version__}")
    print(f"SCENE_PATH={scene_path}")
    print(f"MODEL_NQ={model.nq}")
    print(f"MODEL_NV={model.nv}")
    print(f"LEG_JOINT_COUNT={len(joints)}")
    print(
        f"HEADLESS={str(args.headless).lower()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
