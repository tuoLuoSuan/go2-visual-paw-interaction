from dataclasses import dataclass
import math
from pathlib import Path

try:
    import mujoco
except ModuleNotFoundError:  # pragma: no cover - installed in Ubuntu env
    mujoco = None

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - installed in Ubuntu env
    np = None


CANONICAL_JOINT_NAMES = (
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
FOOT_GEOM_NAMES = ("FR", "FL", "RR", "RL")
FIX_STAND_JOINT_TARGETS = (0.0, 0.8, -1.5) * 4


@dataclass(frozen=True)
class StandingConfig:
    target_lifts_m: tuple = (0.10, 0.20, 0.30)
    sample_rate_hz: float = 500.0
    shift_seconds: float = 2.0
    hold_seconds: float = 1.0
    minimum_lift_seconds: float = 2.0
    support_slip_limit_m: float = 0.005
    target_height_error_m: float = 0.005
    pitch_limit_deg: float = 0.5
    roll_limit_deg: float = 5.0
    base_drop_limit_m: float = 0.020
    tracking_error_limit_rad: float = 0.08
    penetration_limit_m: float = 0.002
    saturation_duration_s: float = 0.026
    restore_joint_error_rad: float = 0.03
    restore_base_error_m: float = 0.01
    restore_orientation_error_deg: float = 1.0
    load_compensation_scale: float = 1.0
    target_reach_tolerance_m: float = 0.010
    dynamics_kp: float = 80.0
    dynamics_kd: float = 5.0
    dynamics_tau_ff: float = 0.0
    base_drift_limit_m: float = 0.15
    shift_com_margin_target_m: float = 0.005
    fr_joint_delta_limit_rad: float = 0.35
    rear_support_maneuver: bool = False
    vmc_enabled: bool = False
    vmc_kp_pos: float = 300.0
    vmc_kd_pos: float = 30.0
    vmc_kp_rot: float = 150.0
    vmc_kd_rot: float = 15.0

    def validate(self):
        heights = tuple(float(value) for value in self.target_lifts_m)
        if not heights or any(
            not math.isfinite(value) or value <= 0.0
            for value in heights
        ):
            raise ValueError("目标高度必须是有限正数")
        if any(next_value <= value for value, next_value in zip(
            heights, heights[1:]
        )):
            raise ValueError("目标高度必须严格递增")

        numeric_fields = (
            self.sample_rate_hz,
            self.shift_seconds,
            self.hold_seconds,
            self.minimum_lift_seconds,
            self.support_slip_limit_m,
            self.target_height_error_m,
            self.pitch_limit_deg,
            self.roll_limit_deg,
            self.base_drop_limit_m,
            self.tracking_error_limit_rad,
            self.penetration_limit_m,
            self.saturation_duration_s,
            self.restore_joint_error_rad,
            self.restore_base_error_m,
            self.restore_orientation_error_deg,
            self.target_reach_tolerance_m,
            self.base_drift_limit_m,
            self.shift_com_margin_target_m,
            self.fr_joint_delta_limit_rad,
        )
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in numeric_fields
        ):
            raise ValueError("站立抬爪配置必须全部为有限正数")
        compensation = float(self.load_compensation_scale)
        if (
            not math.isfinite(compensation)
            or compensation < 0.0
            or compensation > 1.0
        ):
            raise ValueError("负载补偿比例必须在 [0, 1] 内")
        for name, value in (
            ("dynamics_kp", self.dynamics_kp),
            ("dynamics_kd", self.dynamics_kd),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"动力学增益 {name} 必须是有限正数")
        if (
            not math.isfinite(float(self.dynamics_tau_ff))
            or float(self.dynamics_tau_ff) < 0.0
        ):
            raise ValueError("动力学前馈增益必须是非负有限数")
        for name, value in (
            ("vmc_kp_pos", self.vmc_kp_pos),
            ("vmc_kd_pos", self.vmc_kd_pos),
            ("vmc_kp_rot", self.vmc_kp_rot),
            ("vmc_kd_rot", self.vmc_kd_rot),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"VMC 增益 {name} 必须是非负有限数")
        if not isinstance(self.vmc_enabled, bool):
            raise ValueError("vmc_enabled 必须是布尔值")
        return self


@dataclass(frozen=True)
class StandingTarget:
    qpos: tuple
    joint_targets: tuple


@dataclass(frozen=True)
class StandingContext:
    scene_path: Path
    model: object
    standing_qpos: tuple
    joint_names: tuple
    qpos_addresses: tuple
    dof_addresses: tuple
    actuator_by_joint: dict
    foot_geom_ids: dict
    floor_geom_id: int
    base_body_id: int
    base_qpos_address: int
    base_dof_address: int


@dataclass(frozen=True)
class StandingMeasurement:
    base_position: tuple
    base_quaternion: tuple
    base_roll: float
    base_pitch: float
    base_yaw: float
    center_of_mass: tuple
    foot_positions: dict
    foot_surface_heights: dict
    support_contacts: dict
    support_margin_m: float
    maximum_penetration_m: float
    joint_positions: tuple
    joint_velocities: tuple


def fix_stand_joint_targets():
    return FIX_STAND_JOINT_TARGETS


def _convex_hull(points):
    unique = sorted({
        (float(point[0]), float(point[1]))
        for point in points
    })
    if len(unique) < 3:
        raise ValueError("支撑多边形退化")

    def cross(origin, first, second):
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(
            lower[-2], lower[-1], point
        ) <= 1e-12:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(
            upper[-2], upper[-1], point
        ) <= 1e-12:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        raise ValueError("支撑多边形退化")
    return tuple(hull)


def support_polygon_margin(point_xy, support_xy):
    point = (float(point_xy[0]), float(point_xy[1]))
    if not all(math.isfinite(value) for value in point):
        raise ValueError("支撑多边形输入无效")
    points = tuple(
        (float(value[0]), float(value[1]))
        for value in support_xy
    )
    if not points or any(
        not all(math.isfinite(value) for value in item)
        for item in points
    ):
        raise ValueError("支撑多边形输入无效")

    hull = _convex_hull(points)
    distances = []
    for index, first in enumerate(hull):
        second = hull[(index + 1) % len(hull)]
        edge_x = second[0] - first[0]
        edge_y = second[1] - first[1]
        length = math.hypot(edge_x, edge_y)
        if length <= 1e-12:
            raise ValueError("支撑多边形退化")
        distances.append(
            (edge_x * (point[1] - first[1])
             - edge_y * (point[0] - first[0])) / length
        )
    return float(min(distances))


def _require_mujoco():
    if mujoco is None or np is None:
        raise RuntimeError(
            "站立抬爪模型操作需要 Ubuntu 仿真环境的 mujoco 和 numpy"
        )
    return mujoco


def _object_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo 模型缺少对象：{name}")
    return int(object_id)


def _actuator_map(model, joint_names):
    joint_ids = {
        _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name): name
        for name in joint_names
    }
    found = {}
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        name = joint_ids.get(joint_id)
        if name is None:
            continue
        if name in found:
            raise ValueError(f"关节执行器不唯一：{name}")
        found[name] = int(actuator_id)
    missing = [name for name in joint_names if name not in found]
    if missing:
        raise ValueError("模型缺少腿部执行器：" + ", ".join(missing))
    return {name: found[name] for name in joint_names}


def _quaternion_to_rpy(quaternion):
    w, x, y, z = (float(value) for value in quaternion)
    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch_value = 2.0 * (w * y - z * x)
    pitch_value = max(-1.0, min(1.0, pitch_value))
    pitch = math.asin(pitch_value)
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return roll, pitch, yaw


def load_context(scene_path):
    _require_mujoco()
    from load_go2_model import create_go2_data, leg_joint_names, load_go2_model

    # 用 absolute() 不用 resolve()：resolve 会把 Windows junction 还原成
    # 真实路径（可能含非 ASCII 字符，MuJoCo 打不开），absolute 保留词法路径
    scene = Path(scene_path).absolute()
    model = load_go2_model(scene)
    joint_names = tuple(leg_joint_names(model))
    if joint_names != CANONICAL_JOINT_NAMES:
        raise ValueError("GO2 腿部关节顺序不符合合同")

    data = create_go2_data(model)
    qpos = np.asarray(data.qpos, dtype=float).copy()
    qpos_addresses = []
    dof_addresses = []
    for name in joint_names:
        joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"腿部关节不是 hinge：{name}")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))

    for address, target in zip(qpos_addresses, FIX_STAND_JOINT_TARGETS):
        qpos[address] = target

    base_body_id = _object_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "base_link",
    )
    base_joint_id = int(model.body_jntadr[base_body_id])
    if base_joint_id < 0:
        raise ValueError("base_link 缺少自由基座关节")
    base_qpos_address = int(model.jnt_qposadr[base_joint_id])
    base_dof_address = int(model.jnt_dofadr[base_joint_id])

    foot_geom_ids = {
        name: _object_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )
        for name in FOOT_GEOM_NAMES
    }
    floor_geom_id = _object_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "floor",
    )

    # The official scene's free-base default is above the ground while the
    # FixStand joint target is already geometrically standing.  Normalize the
    # context once using the lowest named foot surface; this is an initial
    # model-frame alignment, not a relaxed seed correction.
    alignment_data = mujoco.MjData(model)
    alignment_data.qpos[:] = qpos
    alignment_data.qvel[:] = 0.0
    mujoco.mj_forward(model, alignment_data)
    floor_z = float(alignment_data.geom_xpos[floor_geom_id][2])
    surface_heights = tuple(
        float(alignment_data.geom_xpos[geom_id][2])
        - float(model.geom_size[geom_id, 0])
        - floor_z
        for geom_id in foot_geom_ids.values()
    )
    if not surface_heights or not all(math.isfinite(value) for value in surface_heights):
        raise ValueError("站立初始脚面高度无效")
    qpos[base_qpos_address + 2] -= min(surface_heights)

    return StandingContext(
        scene_path=scene,
        model=model,
        standing_qpos=tuple(float(value) for value in qpos),
        joint_names=joint_names,
        qpos_addresses=tuple(qpos_addresses),
        dof_addresses=tuple(dof_addresses),
        actuator_by_joint=_actuator_map(model, joint_names),
        foot_geom_ids=foot_geom_ids,
        floor_geom_id=floor_geom_id,
        base_body_id=base_body_id,
        base_qpos_address=base_qpos_address,
        base_dof_address=base_dof_address,
    )


def fix_stand_qpos(context):
    qpos = list(context.standing_qpos)
    for address, target in zip(
        context.qpos_addresses,
        FIX_STAND_JOINT_TARGETS,
    ):
        qpos[address] = target
    return StandingTarget(
        qpos=tuple(float(value) for value in qpos),
        joint_targets=FIX_STAND_JOINT_TARGETS,
    )


def measure_state(context, qpos, qvel=None):
    _require_mujoco()
    data = mujoco.MjData(context.model)
    qpos_array = np.asarray(qpos, dtype=float)
    if qpos_array.shape != (context.model.nq,):
        raise ValueError("qpos 维度与 GO2 模型不匹配")
    data.qpos[:] = qpos_array
    if qvel is None:
        data.qvel[:] = 0.0
    else:
        qvel_array = np.asarray(qvel, dtype=float)
        if qvel_array.shape != (context.model.nv,):
            raise ValueError("qvel 维度与 GO2 模型不匹配")
        data.qvel[:] = qvel_array
    mujoco.mj_forward(context.model, data)

    base = context.base_qpos_address
    base_position = tuple(float(value) for value in data.qpos[base:base + 3])
    quaternion = tuple(float(value) for value in data.qpos[base + 3:base + 7])
    roll, pitch, yaw = _quaternion_to_rpy(quaternion)
    center_of_mass = tuple(
        float(value) for value in data.subtree_com[context.base_body_id]
    )

    foot_positions = {}
    foot_surface_heights = {}
    for name, geom_id in context.foot_geom_ids.items():
        position = tuple(float(value) for value in data.geom_xpos[geom_id])
        foot_positions[name] = position
        foot_surface_heights[name] = float(
            position[2] - context.model.geom_size[geom_id, 0]
        )

    support_contacts = {name: False for name in FOOT_GEOM_NAMES}
    maximum_penetration = 0.0
    for index in range(data.ncon):
        contact = data.contact[index]
        first = int(contact.geom1)
        second = int(contact.geom2)
        maximum_penetration = max(
            maximum_penetration,
            max(0.0, -float(contact.dist)),
        )
        foot_name = None
        if first == context.floor_geom_id:
            foot_name = next(
                (name for name, geom_id in context.foot_geom_ids.items()
                 if geom_id == second),
                None,
            )
        elif second == context.floor_geom_id:
            foot_name = next(
                (name for name, geom_id in context.foot_geom_ids.items()
                 if geom_id == first),
                None,
            )
        if foot_name is not None:
            support_contacts[foot_name] = True

    support_margin = support_polygon_margin(
        center_of_mass[:2],
        tuple(foot_positions[name][:2] for name in FOOT_GEOM_NAMES),
    )
    joint_positions = tuple(
        float(data.qpos[address]) for address in context.qpos_addresses
    )
    joint_velocities = tuple(
        float(data.qvel[address]) for address in context.dof_addresses
    )
    finite_values = (
        *base_position,
        *quaternion,
        roll,
        pitch,
        yaw,
        *center_of_mass,
        support_margin,
        maximum_penetration,
        *joint_positions,
        *joint_velocities,
    )
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("MuJoCo 状态包含非有限数")

    return StandingMeasurement(
        base_position=base_position,
        base_quaternion=quaternion,
        base_roll=float(roll),
        base_pitch=float(pitch),
        base_yaw=float(yaw),
        center_of_mass=center_of_mass,
        foot_positions=foot_positions,
        foot_surface_heights=foot_surface_heights,
        support_contacts=support_contacts,
        support_margin_m=float(support_margin),
        maximum_penetration_m=float(maximum_penetration),
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
    )
