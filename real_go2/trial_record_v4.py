"""P0-2 trial schema v4: record builder, trace emitter, writer, validator.

Stdlib-only so it can run on both Windows and the Ubuntu VM.
Imported by the v11 caller candidates (real_stand_handshake_v11.py,
real_vmc_reach_m8_v11.py).
"""
import hashlib
import json
import time
from pathlib import Path


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class TraceEmitter:
    """Append-only event trace with strictly increasing event_id."""

    def __init__(self):
        self.events = []
        self._next_id = 0

    def mark(self, stage_name: str, record_point: str, source: str = "main"):
        ev = {
            "event_id": self._next_id,
            "stage_name": stage_name,
            "record_point": record_point,
            "wall_ms": int(time.time() * 1000),
            "mono_ms": int(time.monotonic() * 1000),
            "source": source,
        }
        self._next_id += 1
        self.events.append(ev)
        return ev

    def dump(self):
        return list(self.events)


class StageTimer:
    """Collects real record-point timestamps per stage."""

    def __init__(self, trace: TraceEmitter):
        self.trace = trace
        self.stages = {}  # name -> {start wall/mono, end wall/mono}

    def start(self, name: str):
        ev = self.trace.mark(name, "stage_start")
        self.stages[name] = {
            "record_point_start_ms_wall": ev["wall_ms"],
            "record_point_start_ms_mono": ev["mono_ms"],
        }

    def end(self, name: str, status: str = "passed"):
        ev = self.trace.mark(name, "stage_end")
        s = self.stages.setdefault(name, {})
        s.update({
            "record_point_end_ms_wall": ev["wall_ms"],
            "record_point_end_ms_mono": ev["mono_ms"],
            "status": status,
        })

    def dump(self):
        out = []
        for name in sorted(self.stages, key=lambda n: self.stages[n].get(
                "record_point_start_ms_mono", 0)):
            s = self.stages[name]
            if "record_point_start_ms_wall" in s and "record_point_end_ms_wall" in s:
                out.append({
                    "name": name,
                    **s,
                    "status": s.get("status", "not_measured"),
                })
        return out


def build_trial_record_v4(*, trial_id, session_id, trial_index,
                          execution_status, abort_reason,
                          wall_clock_source,
                          started_at_wall_ms, ended_at_wall_ms,
                          started_at_monotonic_ms, ended_at_monotonic_ms,
                          clock_sync, stages, event_trace, identity,
                          contact, endpoints, paw_selection,
                          safety_retreat) -> dict:
    return {
        "schema_version": "4",
        "trial_id": trial_id,
        "session_id": session_id,
        "trial_index": trial_index,
        "execution_status": execution_status,
        "abort_reason": abort_reason,
        "wall_clock_source": wall_clock_source,
        "started_at_wall_ms": started_at_wall_ms,
        "ended_at_wall_ms": ended_at_wall_ms,
        "started_at_monotonic_ms": started_at_monotonic_ms,
        "ended_at_monotonic_ms": ended_at_monotonic_ms,
        "clock_sync": clock_sync,
        "stages": stages,
        "event_trace": event_trace,
        "identity": identity,
        "contact": contact,
        "endpoints": endpoints,
        "paw_selection": paw_selection,
        "safety_retreat": safety_retreat,
    }


def write_trial_record_v4(trial: dict, out_dir) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"trial_v4_{trial['trial_id']}.json"
    p.write_text(json.dumps(trial, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


def validate_trial_v4(trial: dict) -> list:
    """Re-export of the contract validator (kept in tools/validate_trial_schema.py)."""
    import sys
    from pathlib import Path as _P
    tools = _P(__file__).resolve().parents[2] / "tools"
    sys.path.insert(0, str(tools))
    from validate_trial_schema import validate
    return validate(trial)
