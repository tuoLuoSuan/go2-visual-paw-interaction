"""Check archived comparison integrity; no training or robot connection."""

import csv
import gzip
import hashlib
import importlib.util
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/policy_comparison"
spec = importlib.util.spec_from_file_location(
    "comparison_reconstruction", ROOT / "simulation/src/reconstruct_metrics.py"
)
reconstruction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reconstruction)


class ComparisonEvidenceTest(unittest.TestCase):
    def test_checkpoint_hashes_match_archived_summaries(self):
        for backbone in ("mlp", "gru"):
            with self.subTest(backbone=backbone):
                summary = json.loads((DATA / backbone / "summary.json").read_text(encoding="utf-8"))
                checkpoint = ROOT / f"models/prone_comparison/best_{backbone}_prone_px.pt"
                self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), summary["checkpoint_sha256"])
                self.assertEqual(summary["task"], "prone_px")
                self.assertEqual(summary["backbone"], backbone)
                self.assertEqual(summary["eval_seed"], 42)
                self.assertEqual(summary["episodes"], 20)
                self.assertEqual(summary["latency_range"], [0.03, 0.15])

    def test_complete_finite_step_and_episode_denominators(self):
        for backbone in ("mlp", "gru"):
            with self.subTest(backbone=backbone):
                with gzip.open(DATA / backbone / "step_metrics.csv.gz", "rt", encoding="utf-8") as handle:
                    steps = list(csv.DictReader(handle))
                with (DATA / backbone / "episode_metrics.csv").open(encoding="utf-8") as handle:
                    episodes = list(csv.DictReader(handle))
                self.assertEqual(len(steps), 4000)
                self.assertEqual(len(episodes), 20)
                self.assertEqual({int(row["env_index"]) for row in episodes}, set(range(20)))
                self.assertEqual({int(row["episode"]) for row in episodes}, set(range(1, 21)))
                self.assertTrue(all(math.isfinite(float(r["distance_m"])) and float(r["distance_m"]) >= 0 for r in steps))
                for env in range(20):
                    self.assertEqual([int(r["step"]) for r in steps if int(r["env_index"]) == env], list(range(200)))
                for row in episodes:
                    self.assertEqual(int(row["steps"]), 200)
                    self.assertEqual(row["termination"], "time_limit")

    def test_reconstruction_and_manuscript_rounding(self):
        with (DATA / "manuscript_distance_table.csv").open(encoding="utf-8") as handle:
            table = {row["backbone"].lower(): row for row in csv.DictReader(handle)}
        for backbone in ("mlp", "gru"):
            with self.subTest(backbone=backbone):
                summary, rebuilt = reconstruction.reconstruct(DATA / backbone)
                for key, value in rebuilt.items():
                    self.assertTrue(math.isfinite(value))
                    self.assertTrue(math.isfinite(summary["metrics"][key]))
                    self.assertAlmostEqual(value, summary["metrics"][key], delta=1e-9)
                for column, key in (("mean_m", "mean_dist_m"), ("median_m", "median_dist_m"), ("p90_m", "p90_dist_m")):
                    self.assertEqual(f"{rebuilt[key]:.4f}", table[backbone][column])
                self.assertEqual(table[backbone]["training_seeds"], "1")

    def test_training_origins_are_distinct_and_scene_is_external(self):
        manifests = {}
        for backbone in ("mlp", "gru"):
            path = ROOT / f"models/prone_comparison/{backbone}_training_manifest.json"
            manifests[backbone] = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(manifests[backbone]["seed"], 0)
            self.assertEqual(manifests[backbone]["config"]["scene"], "EXTERNAL_GO2_SCENE_XML_NOT_BUNDLED")
        self.assertTrue(manifests["mlp"]["resume"])
        self.assertFalse(manifests["gru"]["resume"])

    def test_all_new_source_copies_match_provenance_hashes(self):
        with (ROOT / "docs/POLICY_COMPARISON_PROVENANCE.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row["release_path"] for row in rows}), 12)
        for row in rows:
            with self.subTest(path=row["release_path"]):
                self.assertEqual(hashlib.sha256((ROOT / row["release_path"]).read_bytes()).hexdigest(), row["release_sha256"])
                self.assertRegex(row["source_sha256"], r"^[0-9a-f]{64}$")
                if row["change_note"] == "exact copy":
                    self.assertEqual(row["source_sha256"], row["release_sha256"])


if __name__ == "__main__":
    unittest.main()
