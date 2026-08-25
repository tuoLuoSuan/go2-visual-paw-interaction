"""P0-2 trial schema v4 validator + deterministic fixture generator.

Usage:
  python validate_trial_schema.py <trial.json> [--expect-valid|--expect-invalid]
  python validate_trial_schema.py --write-fixtures <dir>

Exit codes: 0 = expectation met, 1 = validation errors, 2 = usage error.
"""
import json
import sys
from pathlib import Path

PLACEHOLDER_MARKERS = ("placeholder", "【待补】", "TBD", "todo", "none_placeholder")
SHA_LEN = 64
HEX = set("0123456789abcdefABCDEF")


def _is_sha(v) -> bool:
    return isinstance(v, str) and len(v) == SHA_LEN and all(c in HEX for c in v)


def validate(trial: dict) -> list:
    errs = []
    if trial.get("schema_version") != "4":
        errs.append("schema_version != 4")
    if not trial.get("trial_id"):
        errs.append("trial_id empty")
    if not trial.get("session_id"):
        errs.append("session_id empty")
    if not isinstance(trial.get("trial_index"), int) or trial["trial_index"] < 1:
        errs.append("trial_index missing or <1")
    if trial.get("execution_status") not in ("ok", "aborted", "not_run"):
        errs.append("execution_status invalid")
    if trial.get("execution_status") == "aborted" and not trial.get("abort_reason"):
        errs.append("aborted without abort_reason")
    # timestamps
    w0, w1 = trial.get("started_at_wall_ms"), trial.get("ended_at_wall_ms")
    m0, m1 = trial.get("started_at_monotonic_ms"), trial.get("ended_at_monotonic_ms")
    if not all(isinstance(x, int) for x in (w0, w1, m0, m1)):
        errs.append("timestamp fields missing/non-int")
    elif w1 < w0 or m1 < m0:
        errs.append("trial start>end")
    # clock_sync
    cs = trial.get("clock_sync") or {}
    for k in ("robot_clock_offset_ms", "robot_clock_uncertainty_ms",
              "video_clock_offset_ms", "video_clock_uncertainty_ms"):
        if k not in cs:
            errs.append(f"clock_sync missing {k}")
    if not cs.get("sync_method"):
        errs.append("clock_sync.sync_method empty")
    # stages
    stages = trial.get("stages")
    if not isinstance(stages, list) or not stages:
        errs.append("stages empty")
    else:
        for s in stages:
            if not s.get("name"):
                errs.append("stage name empty")
                continue
            if s.get("status") not in ("not_started", "passed", "failed", "not_measured"):
                errs.append(f"stage {s['name']} status invalid")
            ks = ("record_point_start_ms_wall", "record_point_end_ms_wall",
                  "record_point_start_ms_mono", "record_point_end_ms_mono")
            if not all(isinstance(s.get(k), int) for k in ks):
                errs.append(f"stage {s['name']} missing record-point timestamps")
                continue
            if s["record_point_end_ms_wall"] < s["record_point_start_ms_wall"]:
                errs.append(f"stage {s['name']} wall start>end")
            if s["record_point_end_ms_mono"] < s["record_point_start_ms_mono"]:
                errs.append(f"stage {s['name']} mono start>end")
        # uniform backfill detection
        n_stages = len(stages)
        uniform = all(
            s["record_point_start_ms_wall"] == s["record_point_end_ms_wall"] == w1
            and s["record_point_start_ms_mono"] == s["record_point_end_ms_mono"] == m1
            for s in stages)
        if uniform:
            errs.append("uniform end-time backfill detected")
        # trace recompute consistency (v3.1): stage record points must be
        # recomputable from the event trace
        trace_map = {}
        for ev in trial.get("event_trace") or []:
            trace_map[(ev.get("stage_name"), ev.get("record_point"))] = ev.get("mono_ms")
        for s in stages:
            if not isinstance(s.get("record_point_start_ms_mono"), int):
                continue
            st = trace_map.get((s["name"], "stage_start"))
            en = trace_map.get((s["name"], "stage_end"))
            if st != s["record_point_start_ms_mono"] or en != s["record_point_end_ms_mono"]:
                errs.append(f"stage {s['name']} times not recomputable from trace")
    # event_trace
    trace = trial.get("event_trace")
    if not isinstance(trace, list) or not trace:
        errs.append("event_trace empty")
    else:
        prev_id, prev_mono, prev_wall = -1, -1, -1
        for ev in trace:
            for k in ("event_id", "wall_ms", "mono_ms"):
                if not isinstance(ev.get(k), int):
                    errs.append(f"trace missing {k}")
            if not ev.get("stage_name") or not ev.get("record_point") or not ev.get("source"):
                errs.append("trace missing stage_name/record_point/source")
            if ev.get("event_id", -1) <= prev_id:
                errs.append("trace event_id not strictly increasing")
            if ev.get("mono_ms", -1) < prev_mono:
                errs.append("trace mono_ms decreasing")
            if ev.get("wall_ms", -1) < prev_wall:
                errs.append("trace wall_ms decreasing")
            prev_id, prev_mono, prev_wall = ev["event_id"], ev["mono_ms"], ev["wall_ms"]
    # identity
    ident = trial.get("identity") or {}
    commit = ident.get("commit", "")
    if not commit or any(m in commit for m in PLACEHOLDER_MARKERS):
        errs.append("commit empty or placeholder")
    for k in ("code_sha256", "model_sha256", "schema_sha256"):
        v = ident.get(k, "")
        if not _is_sha(v) and v != "none":
            errs.append(f"identity.{k} not 64-hex (and not 'none')")
    if (ident.get("model_sha256") == "none"
            and trial.get("endpoints", {}).get("reach_success") not in ("not_measured",)
            or ident.get("model_sha256") == "none"
            and trial.get("endpoints", {}).get("handshake_success") not in ("not_measured",)):
        errs.append("model_sha256=none with measured endpoint")
    cal = ident.get("calibration_id", "MISSING")
    if cal == "MISSING":
        errs.append("identity.calibration_id missing")
    elif isinstance(cal, str) and (any(m in cal.lower() for m in PLACEHOLDER_MARKERS)):
        errs.append("calibration_id placeholder")
    if not ident.get("firmware"):
        errs.append("identity.firmware empty")
    if not isinstance(ident.get("deploy_params"), dict):
        errs.append("identity.deploy_params not object")
    if not isinstance(ident.get("human_intervention"), list):
        errs.append("identity.human_intervention not list")
    # contact (fail-closed)
    contact = trial.get("contact") or {}
    if "contact_hold_s" not in contact:
        errs.append("contact_hold_s missing (fail-closed)")
    elif not isinstance(contact["contact_hold_s"], (int, float)) or contact["contact_hold_s"] <= 0:
        errs.append("contact_hold_s <= 0 (fail-closed)")
    if contact.get("contact_ground_truth") not in ("not_measured", "true", "false"):
        errs.append("contact_ground_truth invalid")
    if contact.get("contact_ground_truth") == "true" and \
            contact.get("contact_confirmation_source") != "independent_annotation":
        errs.append("contact_ground_truth=true without independent_annotation source")
    if not contact.get("contact_confirmation_source"):
        errs.append("contact_confirmation_source empty")
    # endpoints
    eps = trial.get("endpoints") or {}
    for k in ("reach_success", "handshake_success"):
        if eps.get(k) not in ("not_measured", "true", "false"):
            errs.append(f"endpoints.{k} invalid")
    # only-detector-trigger rule
    if eps.get("handshake_success") == "true" and \
            contact.get("contact_ground_truth") != "true":
        errs.append("handshake_success=true without measured contact_ground_truth=true")
    # paw_selection
    ps = trial.get("paw_selection") or {}
    if ps.get("expected_paw") not in ("not_measured", "FL", "FR", "RL", "RR", "NONE"):
        errs.append("paw_selection.expected_paw invalid")
    if ps.get("paw_selected_correctly") not in ("not_measured", "true", "false"):
        errs.append("paw_selected_correctly invalid")
    # safety_retreat independent recompute
    sr = trial.get("safety_retreat") or {}
    if not isinstance(sr.get("retreat_completed"), bool):
        errs.append("safety_retreat.retreat_completed not bool")
    if sr.get("retreat_completed") is True:
        if not isinstance(sr.get("select_mode_code"), int):
            errs.append("retreat completed but select_mode_code missing")
        if not sr.get("check_mode_name"):
            errs.append("retreat completed but check_mode_name missing")
        if not isinstance(sr.get("restore_code"), int):
            errs.append("retreat completed but restore_code missing")
    if not isinstance(sr.get("safety_alarms"), list):
        errs.append("safety_retreat.safety_alarms not list")
    return errs


def positive_fixture() -> dict:
    return {
        "schema_version": "4",
        "trial_id": "fixture-ok-0001",
        "session_id": "fixture-session-0001",
        "trial_index": 1,
        "execution_status": "ok",
        "abort_reason": "",
        "wall_clock_source": "system-utc",
        "started_at_wall_ms": 1000, "ended_at_wall_ms": 5000,
        "started_at_monotonic_ms": 100, "ended_at_monotonic_ms": 500,
        "clock_sync": {
            "robot_clock_offset_ms": None, "robot_clock_uncertainty_ms": None,
            "video_clock_offset_ms": None, "video_clock_uncertainty_ms": None,
            "sync_method": "not_measured", "estimated_at_wall_ms": None,
        },
        "stages": [
            {"name": "stand", "record_point_start_ms_wall": 1000, "record_point_end_ms_wall": 2000,
             "record_point_start_ms_mono": 100, "record_point_end_ms_mono": 200, "status": "passed"},
            {"name": "track", "record_point_start_ms_wall": 2000, "record_point_end_ms_wall": 4000,
             "record_point_start_ms_mono": 200, "record_point_end_ms_mono": 400, "status": "passed"},
        ],
        "event_trace": [
            {"event_id": 0, "stage_name": "stand", "record_point": "stage_start",
             "wall_ms": 1000, "mono_ms": 100, "source": "main"},
            {"event_id": 1, "stage_name": "stand", "record_point": "stage_end",
             "wall_ms": 2000, "mono_ms": 200, "source": "main"},
            {"event_id": 2, "stage_name": "track", "record_point": "stage_start",
             "wall_ms": 2000, "mono_ms": 200, "source": "main"},
            {"event_id": 3, "stage_name": "track", "record_point": "stage_end",
             "wall_ms": 4000, "mono_ms": 400, "source": "main"},
        ],
        "identity": {
            "commit": "b8c69e4853eaf988572195732c89b3720f6038ab",
            "code_sha256": "a" * 64, "model_sha256": "b" * 64, "schema_sha256": "c" * 64,
            "firmware": "mcf-1.0", "calibration_id": "calib-fixture-01",
            "floor": "flat", "light": "indoor",
            "deploy_params": {"kp": 300},
            "human_intervention": [],
        },
        "contact": {
            "contact_hold_s": 0.6, "detector_trigger_count": 3,
            "contact_ground_truth": "true", "contact_confirmation_source": "independent_annotation",
        },
        "endpoints": {"reach_success": "not_measured", "handshake_success": "true"},
        "paw_selection": {"expected_paw": "FR", "selected_paw": "FR",
                          "paw_selected_correctly": "not_measured"},
        "safety_retreat": {
            "retreat_completed": True, "select_mode_code": 0,
            "check_mode_name": "mcf", "restore_code": 0, "safety_alarms": [],
        },
    }


def negative_fixtures() -> dict:
    out = {}
    base = positive_fixture()
    d = json.loads(json.dumps(base)); d["identity"]["commit"] = ""; out["n1_empty_commit"] = d
    d = json.loads(json.dumps(base)); d["identity"]["calibration_id"] = "placeholder"; out["n2_placeholder_calib"] = d
    d = json.loads(json.dumps(base)); d["trial_index"] = 0; out["n3_empty_trial_index"] = d
    d = json.loads(json.dumps(base))
    for s in d["stages"]:
        s["record_point_start_ms_wall"] = s["record_point_end_ms_wall"] = d["ended_at_wall_ms"]
        s["record_point_start_ms_mono"] = s["record_point_end_ms_mono"] = d["ended_at_monotonic_ms"]
    out["n4_uniform_backfill"] = d
    d = json.loads(json.dumps(base)); d["contact"].pop("contact_hold_s"); out["n5_missing_contact_hold_s"] = d
    d = json.loads(json.dumps(base)); d["contact"]["contact_hold_s"] = 0.0; out["n5b_zero_contact_hold_s"] = d
    d = json.loads(json.dumps(base)); d["contact"]["contact_ground_truth"] = "not_measured"; out["n6_detector_only"] = d
    d = json.loads(json.dumps(base)); d["safety_retreat"].pop("restore_code"); out["n7_retreat_component_missing"] = d
    d = json.loads(json.dumps(base)); d["event_trace"][1]["event_id"] = 0; out["n8_trace_nonmonotonic"] = d
    d = json.loads(json.dumps(base)); d["event_trace"][1]["mono_ms"] = 90; out["n8b_trace_mono_decreasing"] = d
    d = json.loads(json.dumps(base)); d["ended_at_wall_ms"] = 500; out["n9_start_gt_end"] = d
    return out


def main(argv) -> int:
    if len(argv) >= 2 and argv[1] == "--write-fixtures":
        dstdir = Path(argv[2])
        dstdir.mkdir(parents=True, exist_ok=True)
        (dstdir / "positive_fixture.json").write_text(
            json.dumps(positive_fixture(), ensure_ascii=False, indent=2), encoding="utf-8")
        for name, doc in negative_fixtures().items():
            (dstdir / f"{name}.json").write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[FIXTURES] wrote {1 + len(negative_fixtures())} files to {dstdir}")
        return 0
    if len(argv) < 2:
        print("usage: validate_trial_schema.py <file> [--expect-valid|--expect-invalid] | --write-fixtures <dir>")
        return 2
    with open(argv[1], "r", encoding="utf-8") as f:
        trial = json.load(f)
    errs = validate(trial)
    expect_invalid = "--expect-invalid" in argv
    if errs:
        print(f"[INVALID] {len(errs)} errors:")
        for e in errs:
            print("  -", e)
        return 0 if expect_invalid else 1
    print("[VALID]")
    return 1 if expect_invalid else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
