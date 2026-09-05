# Prone MLP / GRU checkpoint comparison

This dataset supports the manuscript's descriptive simulation comparison, not the real-robot contact endpoints. Both policies have 27 inputs and 6 outputs; the standing extension's 29/12 GRU is unrelated to this comparison.

## Protocol and results

| Field | MLP | GRU |
|---|---|---|
| Archived run ID | eval_prone_px_mlp_20260823-053232_seed42 | eval_prone_px_gru_20260823-091219_seed42 |
| Training seed | 0 | 0 |
| Training origin | Continued from earlier weights | Random initialization |
| Evaluation seed | 42 | 42 |
| Trajectories / steps per trajectory | 20 / 200 | 20 / 200 |
| Observation delay range (s) | 0.03–0.15 | 0.03–0.15 |
| Mean distance (m) | 0.10527811406583186 | 0.11868563477533765 |
| Median distance (m) | 0.10387402196193171 | 0.11041595217716818 |
| P90 distance (m) | 0.16244780272645032 | 0.19272295321987695 |
| Time steps below 0.05 m | 10.15% | 9.375% |
| Full 200-step trajectories | 20/20 | 20/20 |

One checkpoint per architecture is available. Evaluation trajectories are not independent training seeds. Different training origins and source histories prevent causal attribution to architecture; no significance test or general ranking is supported.

## Files and variables

Each of `mlp/` and `gru/` contains:

- `summary.json`: archived configuration, checkpoint hash and aggregate metrics; paths and Git commits inside it refer to the private research history, not this public release.
- `step_metrics.csv.gz`: recorded step distances (`distance_m`, metres), zero-based `step` and `env_index`. The legacy `episode` field is a shared completion counter, **not a unique trajectory ID**. In these runs, identify each trajectory using `env_index` (0–19); each has steps 0–199 with no reset before the horizon.
- `episode_metrics.csv`: one row per completed trajectory. `episode` is the 1-based completion index; `env_index` maps to the step file. `steps` is an integer count; distance columns use metres, `close_rate_steps` is a proportion and `termination=time_limit` means the 200-step horizon was reached.

`manuscript_distance_table.csv` is the rounded display source, not raw observations. Training configuration and original weights are in [models/prone_comparison](../../models/prone_comparison/MODEL_CARD.md). Source-to-public file hashes and transformations are in [provenance](../../docs/POLICY_COMPARISON_PROVENANCE.csv).

## Reconstruct the archived summary

From the repository root, with NumPy installed:

```console
python simulation/src/reconstruct_metrics.py data/policy_comparison/mlp --tol 1e-9
python simulation/src/reconstruct_metrics.py data/policy_comparison/gru --tol 1e-9
```

Expected: `[RECON] OK` and exit 0 for each. The script pools all 4000 recorded distances to compute mean, median, P90 and the fraction strictly below 0.05 m. Final distance and survival derive from the 20 episode rows. Its survival threshold is fixed at 200 steps and is appropriate for these two archives only.

The step data do not contain joint-target vectors, so `joint_target_range_rad` and `thigh_range_rad` in the original summary are **not reconstructed by this script**. They are not used for the manuscript distance table. Do not call this a reconstruction of every field in the original summary.

## Reproducibility and access limits

Metric reconstruction and checkpoint inspection are available without robot hardware or simulator assets. Re-running the original simulation and re-training from scratch are different claims: historical evaluator/source lineage and the MLP's overwritten starting checkpoint are not completely archived. Existing v4 training/evaluation scripts are reference code, not a substitute for the historical prone comparison evaluator. The archived metadata's `git_dirty=false` does not prove that untracked scripts were captured by the named commit.

Text copies use LF line endings; numeric values are preserved. Original compressed step files and `.pt` checkpoints are copied byte-for-byte. Only the machine-specific scene path in each training manifest is replaced with `EXTERNAL_GO2_SCENE_XML_NOT_BUNDLED`. No trial, failed trajectory or checkpoint was newly collected or excluded in this publication update.

The files are publicly accessible under the repository's current rights status; no open data or software licence has been assigned. Cite the associated [manuscript](../../docs/PAPER.md) and repository version when discussing this evidence.
