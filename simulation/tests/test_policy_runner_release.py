import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_go2.policy_runner import PolicyRunner


class PolicyRunnerReleaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = ROOT / "models/prone/best_mlp_prone_px_v4.npz"
        cls.runner = PolicyRunner(model)

    def test_metadata(self):
        self.assertEqual(self.runner.obs_dim, 27)
        self.assertEqual(self.runner.act_dim, 6)
        self.assertEqual(self.runner.task, "prone_px")
        self.assertEqual(self.runner.backbone, "mlp")

    def test_single_observation(self):
        output = self.runner.act(np.zeros(27, dtype=np.float64))
        self.assertEqual(output.shape, (6,))
        self.assertTrue(np.isfinite(output).all())

    def test_batch_observation(self):
        output = self.runner.act(np.zeros((5, 27), dtype=np.float64))
        self.assertEqual(output.shape, (5, 6))
        self.assertTrue(np.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
