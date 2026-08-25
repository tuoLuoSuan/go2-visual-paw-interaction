import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from validate_trial_schema import validate

EXPECTED_MODEL_SHA = "9de29f01893534b20cd395de82d3d6096a41a1c17d0db1b43d586b59a00f7958"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Formal02EvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = sorted((ROOT / "data/formal02/json").glob("trial_v4_FORMAL-02-*.json"))
        cls.docs = [json.loads(path.read_text(encoding="utf-8")) for path in cls.paths]

    def test_ten_ordered_unique_trials(self):
        expected = [f"FORMAL-02-{index:03d}" for index in range(1, 11)]
        actual = [doc["trial_id"] for doc in self.docs]
        self.assertEqual(actual, expected)
        self.assertEqual(len(set(actual)), 10)

    def test_each_raw_record_satisfies_schema_validator(self):
        for path, doc in zip(self.paths, self.docs):
            self.assertEqual(validate(doc), [], path.name)

    def test_model_identity_is_consistent(self):
        for doc in self.docs:
            self.assertEqual(doc["identity"]["model_sha256"], EXPECTED_MODEL_SHA)
        model = ROOT / "models/prone/best_mlp_prone_px_v4.npz"
        self.assertEqual(sha256(model), EXPECTED_MODEL_SHA)

    def test_raw_trial_002_is_preserved_and_corrected_by_sidecar(self):
        trial_002 = self.docs[1]
        self.assertEqual(trial_002["trial_id"], "FORMAL-02-002")
        self.assertEqual(trial_002["execution_status"], "ok")
        correction = (ROOT / "data/formal02/CORRECTION_RECORD.md").read_text(encoding="utf-8")
        readout = (ROOT / "data/formal02/formal02_trial_readout.md").read_text(encoding="utf-8")
        self.assertIn("M8_ABORTED reason=TRACKING_ERROR", correction)
        self.assertIn("9 ok + 1 aborted", correction)
        self.assertIn("| FORMAL-02-002 | left | yes | yes | aborted | TRACKING_ERROR |", readout)

    def test_endpoint_boundaries_remain_distinct(self):
        for doc in self.docs:
            self.assertEqual(doc["contact"]["contact_ground_truth"], "not_measured")
            self.assertEqual(doc["endpoints"]["reach_success"], "not_measured")
            self.assertEqual(doc["endpoints"]["handshake_success"], "not_measured")
            self.assertEqual(doc["paw_selection"]["paw_selected_correctly"], "not_measured")


if __name__ == "__main__":
    unittest.main()
