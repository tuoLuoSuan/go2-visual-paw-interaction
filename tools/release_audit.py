from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL_SHA = "9de29f01893534b20cd395de82d3d6096a41a1c17d0db1b43d586b59a00f7958"
EXPECTED_STANDING_MODEL_SHA = "5a6cdb1a2bc86a99cb223196d9eac4ebf748e07aef1e32f08f3ca4cebdb822d5"
FORBIDDEN_NAMES = {".ds_ssh", ".ds_remote", ".learnings", "hand_landmarker.task"}
FORBIDDEN_SUFFIXES = {".mp4", ".mov", ".avi", ".pem", ".key"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".yml", ".yaml", ".cff"}
EXCLUDED_MANIFEST_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv"}
PATTERNS = {
    "private-key": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    "github-token": re.compile(r"(?:ghp_|github_pat_)[A-Za-z0-9_]+"),
    "windows-user-path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.I),
    "linux-home-path": re.compile(r"/home/(?!\[)[A-Za-z0-9._-]+(?:/[^\s\"']*)?"),
    "private-ip": re.compile(
        r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
        r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
    ),
    "private-interface": re.compile(r"\benx[0-9a-f]{8,}\b", re.I),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(part in EXCLUDED_MANIFEST_PARTS for part in path.parts)
        and path.suffix.lower() not in {".pyc", ".pyo"}
        and path.name != "MANIFEST.sha256"
    )


def audit() -> list[str]:
    errors: list[str] = []
    files = release_files()
    for path in files:
        rel = path.relative_to(ROOT)
        if any(part in FORBIDDEN_NAMES for part in rel.parts):
            errors.append(f"forbidden path: {rel.as_posix()}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden media/key suffix: {rel.as_posix()}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            for label, pattern in PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"{label}: {rel.as_posix()}")

    trials = sorted((ROOT / "data/formal02/json").glob("trial_v4_FORMAL-02-*.json"))
    expected_ids = [f"FORMAL-02-{index:03d}" for index in range(1, 11)]
    actual_ids = [json.loads(path.read_text(encoding="utf-8"))["trial_id"] for path in trials]
    if actual_ids != expected_ids:
        errors.append(f"FORMAL-02 trial IDs mismatch: {actual_ids}")

    model = ROOT / "models/prone/best_mlp_prone_px_v4.npz"
    if not model.exists() or sha256(model) != EXPECTED_MODEL_SHA:
        errors.append("prone model missing or SHA-256 mismatch")
    standing_model = ROOT / "models/standing_extension/best_gru_standing_px_v4_mask.npz"
    if not standing_model.exists() or sha256(standing_model) != EXPECTED_STANDING_MODEL_SHA:
        errors.append("standing model missing or SHA-256 mismatch")

    correction = ROOT / "data/formal02/CORRECTION_RECORD.md"
    correction_text = correction.read_text(encoding="utf-8") if correction.exists() else ""
    for required in ("FORMAL-02-002", "TRACKING_ERROR", "9 ok + 1 aborted"):
        if required not in correction_text:
            errors.append(f"correction sidecar missing statement: {required}")

    standing_trials = sorted((ROOT / "data/formal03_standing/json").glob("*.json"))
    if len(standing_trials) != 4:
        errors.append(f"FORMAL-03 JSON count is {len(standing_trials)}, expected 4")
    for path in standing_trials:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("params", {}).get("network_interface") != "redacted_private_interface":
            errors.append(f"FORMAL-03 interface not redacted: {path.name}")
    return errors


def write_manifest() -> None:
    rows = [
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in release_files()
    ]
    (ROOT / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    found = audit()
    if found:
        print("RELEASE_AUDIT_FAILED")
        print("\n".join(f"- {item}" for item in found))
        sys.exit(1)
    write_manifest()
    print("RELEASE_AUDIT_OK")
