import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import release_audit


class ReleaseAuditTest(unittest.TestCase):
    def test_private_ip_pattern_distinguishes_versions_from_ipv4(self):
        pattern = release_audit.PATTERNS["private-ip"]
        for harmless in ("Python 3.10.12", "MediaPipe 0.10.35", "limit 10.0"):
            self.assertIsNone(pattern.search(harmless), harmless)
        addresses = (
            "10" + ".1.2.3",
            "192" + ".168.60.128",
            "172" + ".16.0.1",
        )
        for address in addresses:
            self.assertIsNotNone(pattern.search(address), address)

    def test_linux_home_pattern_does_not_match_its_own_source(self):
        pattern = release_audit.PATTERNS["linux-home-path"]
        source = (ROOT / "tools/release_audit.py").read_text(encoding="utf-8")
        self.assertIsNone(pattern.search(source))
        sample = "runtime path: /" + "home/researcher/workspace"
        self.assertIsNotNone(pattern.search(sample))

    def test_current_release_passes(self):
        self.assertEqual(release_audit.audit(), [])

    def test_forbidden_model_fixture_is_rejected(self):
        fixture = ROOT / "hand_landmarker.task"
        self.assertFalse(fixture.exists())
        fixture.write_bytes(b"test fixture")
        try:
            errors = release_audit.audit()
            self.assertTrue(any("forbidden path" in item for item in errors))
        finally:
            fixture.unlink(missing_ok=True)

    def test_manifest_excludes_git_and_itself(self):
        release_audit.write_manifest()
        text = (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8")
        self.assertNotIn("/.git/", text)
        self.assertNotIn("MANIFEST.sha256", text)

    def test_manifest_uses_lf_line_endings(self):
        release_audit.write_manifest()
        payload = (ROOT / "MANIFEST.sha256").read_bytes()
        self.assertNotIn(b"\r\n", payload)

    def test_crlf_text_fixture_is_rejected(self):
        fixture = ROOT / "crlf_fixture.md"
        self.assertFalse(fixture.exists())
        fixture.write_bytes(b"first\r\nsecond\r\n")
        try:
            errors = release_audit.audit()
            self.assertTrue(any("crlf-text" in item for item in errors))
        finally:
            fixture.unlink(missing_ok=True)

    def test_release_file_list_excludes_python_caches(self):
        paths = [path.relative_to(ROOT).as_posix() for path in release_audit.release_files()]
        self.assertFalse(any("__pycache__" in path for path in paths))
        self.assertFalse(any(path.endswith(".pyc") for path in paths))


if __name__ == "__main__":
    unittest.main()
