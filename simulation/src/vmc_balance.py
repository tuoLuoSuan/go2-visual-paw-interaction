"""Virtual Model Control (VMC) balance layer for the GO2 simulator.

The passive PD + load-feedforward controller tracks joint targets but lets
the free base settle wherever force balance takes it.  VMC adds a virtual
spring-damper between the measured base pose and the designed base pose of
the current trajectory sample, converts that virtual wrench into joint
torques through the support feet's Jacobians, and adds it to the actuator
command.  This is the simplified equivalent of the balance controller the
official MCF uses for three-legged stances.

Simulation-only: no SDK, no DDS, no actuator writes.
"""

import math

import standing_paw_lift_common as common


def quaternion_error(reference, current):
    """Rotation vector (rad) rotating ``current`` toward ``reference``.

    Both are (w, x, y, z) quaternions.  Returns (x, y, z) small-angle
    rotation vector; accurate for errors below ~1 rad which covers our
    reach maneuvers (bounded pitch).
    """
    rw, rx, ry, rz = (float(value) for value in reference)
    cw, cx, cy, cz = (float(value) for value in current)
    # q_err = q_ref * conj(q_cur)
    ew = rw * cw + rx * cx + ry * cy + rz * cz
    ex = -rw * cx + rx * cw - ry * cz + rz * cy
    ey = -rw * cy + rx * cz + ry * cw - rz * cx
    ez = -rw * cz - rx * cy + ry * cx + rz * cw
    if ew < 0.0:
        ex, ey, ez, ew = -ex, -ey, -ez, -ew
    return (2.0 * ex, 2.0 * ey, 2.0 * ez)


def base_wrench(context, data, reference_qpos, config):
    """Virtual wrench (F, M) at the base origin toward the reference pose."""
    base = context.base_qpos_address
    measured_pos = common.np.asarray(data.qpos[base:base + 3], dtype=float)
    measured_quat = common.np.asarray(data.qpos[base + 3:base + 7], dtype=float)
    reference_pos = common.np.asarray(
        reference_qpos[base:base + 3], dtype=float
    )
    reference_quat = common.np.asarray(
        reference_qpos[base + 3:base + 7], dtype=float
    )
    position_error = reference_pos - measured_pos
    rotation_error = common.np.asarray(
        quaternion_error(reference_quat, measured_quat), dtype=float
    )
    base_dof = context.base_dof_address
    linear_velocity = common.np.asarray(
        data.qvel[base_dof:base_dof + 3], dtype=float
    )
    angular_velocity = common.np.asarray(
        data.qvel[base_dof + 3:base_dof + 6], dtype=float
    )
    force = (
        float(config.vmc_kp_pos) * position_error
        - float(config.vmc_kd_pos) * linear_velocity
    )
    moment = (
        float(config.vmc_kp_rot) * rotation_error
        - float(config.vmc_kd_rot) * angular_velocity
    )
    return force, moment


def distribute_wrench(foot_positions, base_position, force, moment):
    """Distribute a base wrench to support feet with Fz >= 0.

    Naive equal distribution of the pitch/roll moment lifts a support foot
    (one side presses, the other is pulled).  The vertical forces are
    solved as a least-norm distribution subject to the moment balance and
    Fz_i >= 0 (active-set, a few iterations for 3-4 feet).  The horizontal
    force parts share equally; the vertical moment (Mz) goes through the
    foot moments.
    """
    count = len(foot_positions)
    if count == 0:
        return ()
    if count == 1:
        return ((force, moment),)
    positions = common.np.asarray(foot_positions, dtype=float)  # (n, 3)
    origin = common.np.asarray(base_position, dtype=float)
    relative = positions - origin
    fx = float(force[0]) / count
    fy = float(force[1]) / count
    mz = float(moment[2]) / count
    # vertical balance about the base origin:
    #   sum Fz_i = Fz
    #   sum (-(x_i - x0)) Fz_i = My
    #   sum  (y_i - y0)  Fz_i = Mx
    a = -relative[:, 0]
    b = relative[:, 1]
    fz_target = float(force[2])
    my_target = float(moment[1])
    mx_target = float(moment[0])
    free = list(range(count))
    fz = common.np.zeros(count, dtype=float)
    for _ in range(count + 2):
        if not free:
            break
        rows = []
        rhs = []
        if free:
            rows.append([1.0] * len(free))
            rhs.append(fz_target - float(fz.sum()))
        if free:
            rows.append([float(a[i]) for i in free])
            rhs.append(my_target - float((a * fz).sum()))
            rows.append([float(b[i]) for i in free])
            rhs.append(mx_target - float((b * fz).sum()))
        matrix = common.np.asarray(rows, dtype=float)
        target = common.np.asarray(rhs, dtype=float)
        try:
            solution = common.np.linalg.lstsq(matrix, target, rcond=None)[0]
        except common.np.linalg.LinAlgError:
            solution = common.np.zeros(len(free), dtype=float)
        trial = fz.copy()
        for index, value in zip(free, solution):
            trial[index] = float(value)
        if float(trial.min()) >= -1e-9:
            fz = trial
            break
        # drop the most-negative foot from the free set
        worst = int(trial.argmin())
        free.remove(worst)
    fz = common.np.clip(fz, 0.0, None)
    result = []
    for index in range(count):
        per_foot_force = (fx, fy, float(fz[index]))
        per_foot_moment = (0.0, 0.0, mz)
        result.append((per_foot_force, per_foot_moment))
    return tuple(result)


def support_foot_torques(context, data, force, moment, support_regions,
                         dof_addresses):
    """Map the base wrench to joint torques through support-foot Jacobians.

    For each planted foot the (positive-constrained) wrench is projected
    with ``mj_jac``'s translational and rotational Jacobians.
    """
    mujoco = common._require_mujoco()
    base = context.base_qpos_address
    base_position = common.np.asarray(data.qpos[base:base + 3], dtype=float)
    nv = data.qvel.shape[0]
    jacp = common.np.zeros((3, nv), dtype=float)
    jacr = common.np.zeros((3, nv), dtype=float)
    foot_positions = tuple(
        tuple(float(value) for value in data.geom_xpos[context.foot_geom_ids[region]])
        for region in support_regions
    )
    if not foot_positions:
        return (0.0,) * len(dof_addresses)
    distributed = distribute_wrench(
        foot_positions, base_position, force, moment
    )
    torques = common.np.zeros(len(dof_addresses), dtype=float)
    for region, (per_foot_force, per_foot_moment) in zip(
        support_regions, distributed
    ):
        geom_id = context.foot_geom_ids[region]
        point = data.geom_xpos[geom_id]
        body_id = int(context.model.geom_bodyid[geom_id])
        mujoco.mj_jac(context.model, data, jacp, jacr, point, body_id)
        for index, dof in enumerate(dof_addresses):
            torques[index] += (
                float(jacp[0, dof] * per_foot_force[0]
                      + jacp[1, dof] * per_foot_force[1]
                      + jacp[2, dof] * per_foot_force[2])
                + float(jacr[0, dof] * per_foot_moment[0]
                        + jacr[1, dof] * per_foot_moment[1]
                        + jacr[2, dof] * per_foot_moment[2])
            )
    return tuple(float(value) for value in torques)


def clip_torques(torques, ctrl_ranges):
    """Re-clip a torque tuple to the actuator ranges."""
    clipped = []
    for value, (lower, upper) in zip(torques, ctrl_ranges):
        clipped.append(float(max(lower, min(upper, value))))
    return tuple(clipped)
