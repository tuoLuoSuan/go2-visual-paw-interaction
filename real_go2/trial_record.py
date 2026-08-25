"""真机试验记录构造 v2（schema_version=2，GPT 预实验证据要求 P0-2/3/4/5）。

语义约定（不得违反）：
- 检测器触发 ≠ 物理接触：contact_ground_truth/contact_confirmed 无真值时为
  "not_measured"；
- 选足正确性只在 expected_paw 与 selected_paw 都有值时才计算；
- safe_retreat_completed 由组成证据按固定规则计算，不用"未 abort"替代；
- 每个阶段记录开始/完成时间、判据、状态与失败原因。
"""
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = 3   # 草稿状态（正式实验前可能再修订，未冻结）


class Stage:
    """单阶段记录。"""

    def __init__(self, name):
        self.name = name
        self.start_t = None
        self.end_t = None
        self.status = "not_started"   # not_started/passed/failed/not_measured
        self.criterion = ""
        self.failure = ""

    def start(self, t, criterion=""):
        self.start_t = t.isoformat()
        self.criterion = criterion
        self.status = "running"

    def finish(self, t, passed):
        self.end_t = t.isoformat()
        self.status = "passed" if passed else "failed"

    def mark_not_measured(self, reason):
        self.status = "not_measured"
        self.failure = reason

    def to_dict(self):
        return {"name": self.name, "start_t": self.start_t,
                "end_t": self.end_t, "status": self.status,
                "criterion": self.criterion, "failure": self.failure}


def _hand_side(px):
    return "left" if px < 0.5 else "right"   # 画面侧（非狗侧）


def _selected_paw(px):
    return "FL" if px < 0.5 else "FR"        # 镜像映射：画面左->狗左爪FL


def build_trial_record(args, ctl, temps, hold_metrics, retreat_evidence,
                      session, now=None):
    """构造 trial dict（schema v2）。

    hold_metrics: {hand_frames, last_hand_px, contact_trigger_count,
                   max_detector_hold_s, contact_ground_truth,
                   contact_confirmation_source}
    retreat_evidence: {retreat_completed, select_mode_code, check_mode_name,
                       restore_code, failure, steps}
    session: {start_t, end_t, firmware, camera, calibration_id, floor,
              light, trial_index, policy_sha256}
    """
    now = now or datetime.now()
    hand_frames = int(hold_metrics.get("hand_frames", 0))
    last_px = hold_metrics.get("last_hand_px")
    triggers = int(hold_metrics.get("contact_trigger_count", 0))
    max_hold_s = float(hold_metrics.get("max_detector_hold_s", 0.0))
    contact_hold_s = float(getattr(args, "contact_hold_s", 0.0))

    stages = {}
    for name in ("hand_detect", "paw_select", "target_reach",
                 "contact_detector_trigger", "hold", "safe_retreat"):
        stages[name] = Stage(name)

    # 阶段1 检测到手
    stages["hand_detect"].start(now, "视觉流输出非空")
    if hand_frames > 0:
        stages["hand_detect"].finish(now, True)
    else:
        stages["hand_detect"].finish(now, False)
        stages["hand_detect"].failure = "全程未检测到手"

    # 阶段2 选足：执行了选择动作；正确性无独立真值 -> not_measured
    #（判据 expected_paw == selected_paw 在 expected 为 not_measured 时不可评，
    #  不得记为 passed）
    if last_px is not None:
        stages["paw_select"].start(now, "由 px<0.5 镜像规则选择前足")
        stages["paw_select"].finish(now, True)
        stages["paw_select"].mark_not_measured(
            "选择动作已执行，但 expected_paw 无独立真值，正确性未验证")
    else:
        stages["paw_select"].mark_not_measured("无手位置，未发生选足")

    # 阶段3 到达：无足端3D定位 -> not_measured
    stages["target_reach"].mark_not_measured(
        "足端-目标距离需相机标定【待标定】")

    # 阶段4 接触检测器触发（软件输出；非物理接触确认）
    # 真实的 contact_confirm 语义在 contact.contact_confirmed（无真值=
    # not_measured），本阶段只描述检测器是否触发
    if triggers > 0:
        stages["contact_detector_trigger"].start(
            now, "tau_est 尖峰检测器触发（非物理接触确认）")
        stages["contact_detector_trigger"].finish(now, True)
    else:
        stages["contact_detector_trigger"].mark_not_measured(
            "检测器未触发")

    # 阶段5 保持：检测器保持判据完成（非物理保持证明）
    if triggers > 0:
        stages["hold"].start(now, f"连续检测器保持 >= {contact_hold_s:.2f}s")
        stages["hold"].finish(now, max_hold_s >= contact_hold_s - 1e-6)
        if max_hold_s < contact_hold_s - 1e-6:
            stages["hold"].failure = (f"最长保持 {max_hold_s:.2f}s "
                                      f"< {contact_hold_s:.2f}s")
    else:
        stages["hold"].mark_not_measured("无接触触发")

    # 阶段6 安全退出：由组成证据计算
    stages["safe_retreat"].start(
        now, "收回完成 且 SelectMode=0 且 CheckMode 有名 且 恢复动作=0")
    stages["safe_retreat"].finish(now, bool(retreat_evidence.get(
        "retreat_completed", False)))
    if not retreat_evidence.get("retreat_completed", False):
        stages["safe_retreat"].failure = retreat_evidence.get(
            "failure", "组成证据不完整")

    contact_truth = hold_metrics.get("contact_ground_truth", "not_measured")
    contact_confirmed = ("not_measured" if contact_truth == "not_measured"
                         else bool(contact_truth and triggers > 0))

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": f"m8_{now.strftime('%Y%m%d-%H%M%S')}",
        "start_t": session.get("start_t"),
        "end_t": now.isoformat(),
        "outcome": "aborted" if ctl.aborted else "ok",
        "abort_reason": ctl.abort_reason or "",
        "controller": ("policy:" + str(getattr(args, "policy", ""))
                       if getattr(args, "policy", "") else "handcrafted"),
        "policy_sha256": session.get("policy_sha256", ""),
        "firmware": session.get("firmware", "mcf(version【待补】)"),
        "camera": session.get("camera", "GO2_front_RGB(VideoClient)"),
        "calibration_id": session.get("calibration_id", "placeholder"),
        "floor": session.get("floor", ""),
        "light": session.get("light", ""),
        "trial_index": session.get("trial_index", ""),
        "paw_selection": {
            "detected_hand_side": (_hand_side(last_px)
                                   if last_px is not None else None),
            "expected_paw": "not_measured",   # 无独立真值来源
            "selected_paw": (_selected_paw(last_px)
                             if last_px is not None else None),
            "paw_selected_correctly": "not_measured",
        },
        "contact": {
            "detector_triggered": bool(triggers > 0),
            "detector_trigger_count": triggers,
            "max_detector_hold_s": max_hold_s,
            "contact_ground_truth": contact_truth,
            "contact_confirmation_source": hold_metrics.get(
                "contact_confirmation_source", ""),
            "contact_confirmed": contact_confirmed,
        },
        "stages": {k: v.to_dict() for k, v in stages.items()},
        "retreat_evidence": {
            "retreat_completed": bool(retreat_evidence.get(
                "retreat_completed", False)),
            "select_mode_code": retreat_evidence.get("select_mode_code"),
            "check_mode_name": retreat_evidence.get("check_mode_name", ""),
            "restore_code": retreat_evidence.get("restore_code"),
            "failure": retreat_evidence.get("failure", ""),
            "steps": retreat_evidence.get("steps", []),
        },
        "endpoints": {
            "reach_success": "not_measured",
            "handshake_success": "not_measured",
        },
        "metrics": {
            "max_joint_tracking_error_rad": float(
                getattr(ctl, "max_track_err", 0.0)),
            "max_joint_tracking_error_joint": getattr(
                ctl, "max_track_err_joint", ""),
            "max_pitch_deg": float(getattr(ctl, "max_pitch_deg", 0.0)),
            "end_temp_max_c": float(max(temps, default=0.0)),
        },
        "params": {k: str(v) for k, v in vars(args).items()},
        "python": platform.python_version(),
    }


def write_trial_record(trial, log_dir, git_commit=None):
    out_dir = Path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if git_commit is None:
        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True,
                text=True, timeout=5).stdout.strip()
        except Exception:
            git_commit = "unknown"
    trial["git_commit"] = git_commit
    out_path = out_dir / f"{trial['run_id']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trial, f, ensure_ascii=False, indent=2)
    return out_path
