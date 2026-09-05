# Archived prone comparison checkpoints

These are the exact PyTorch files identified in the manuscript's two archived evaluation summaries. They are **not** the standing-extension GRU.

| File | Architecture | Input / output | SHA-256 |
|---|---|---|---|
| `best_mlp_prone_px.pt` | 27 → 128 → 128 → 128 → 6, Tanh hidden layers | 27 / 6 | `899319bd00b3afb02406b56ad5eb3e491f35ed864258bb9d3fa9cb77e00bf494` |
| `best_gru_prone_px.pt` | 27 → GRU(128) → 128 → 128 → 6 | 27 / 6 | `833fd1a2ec85c90374fbcbde06fa8f0b6ebf072e70f8356ec4e840596621b54a` |

Both use task ID `prone_px`. Inputs comprise 12 normalized joint positions, 12 normalized joint velocities, image-space `px`, `py`, and an image-side indicator. Six outputs correspond to front-leg joint increments, not Cartesian paw positions. The GRU carries hidden state within a trajectory and requires reset between trajectories.

The main robot experiment used the MLP's NumPy deployment representation in [../prone/](../prone/MODEL_CARD.md). The comparison GRU was not deployed in FORMAL-02. The existing standing GRU in [../standing_extension/](../standing_extension/MODEL_CARD.md) has 29 inputs / 12 outputs.

## Training provenance

The adjacent `mlp_training_manifest.json` and `gru_training_manifest.json` preserve the recorded configurations: seed 0, 32 environments, 2500 iterations in the recorded stage. The MLP resumed earlier weights; the GRU did not. The prior MLP checkpoint was overwritten and cannot be reconstructed from this package. Recorded private-repository commits are historical references, not public commits or a complete training lineage. Scene paths are sanitized; no third-party scene is bundled.

See [evaluation data and limitations](../../data/policy_comparison/README.md) for metric definitions. These two checkpoints are not sufficient for an architecture-level conclusion.

## Loading and safety

Inspect with a current PyTorch version and `torch.load(path, map_location="cpu", weights_only=True)`. Do not disable safe loading for an untrusted checkpoint. Verify the SHA-256 before using existing legacy loaders/exporters, some of which use `weights_only=False`.

Offline tensor inspection does not authorize applying model outputs to hardware. No new robot-control test was performed for this release. Weights have no separately assigned open licence; repository visibility alone does not grant redistribution rights.
