# Public package verification — 2026-09-05

Scope: local, hardware-free verification of the v0.2.0 manuscript-evidence additions. This is not a new robot experiment, training run, simulator rerun or safety certification.

## Environment

Windows; Python 3.13.13, NumPy 2.4.6, PyTorch 2.13.0+cpu, MuJoCo 3.11.0, Matplotlib 3.11.1. The three unittest groups below use the release checkout only; checkpoint tensor inspection additionally uses PyTorch. No robot entry point or network-control code was executed.

## Commands and results

| Command | Found / passed | Failed / skipped | Exit |
|---|---:|---:|---:|
| `python -X utf8 -m unittest discover -s tests -v` | 18 / 18 | 0 / 0 | 0 |
| `python -m unittest discover -s simulation/tests -v` | 30 / 30 | 0 / 0 | 0 |
| `python -m unittest discover -s vision/tests -v` | 42 / 42 | 0 / 0 | 0 |

Total: **90 tests passed**. Five added evidence tests check the two checkpoint hashes, 4000 finite step distances and 20 complete trajectories per run, all seven reconstructed metrics, displayed table rounding, training-origin differences and the 12-file provenance mapping.

Both commands below returned exit 0 and `[RECON] OK` with a tolerance of `1e-9`:

```console
python -X utf8 simulation/src/reconstruct_metrics.py data/policy_comparison/mlp --tol 1e-9
python -X utf8 simulation/src/reconstruct_metrics.py data/policy_comparison/gru --tol 1e-9
```

Rebuilt mean distances: MLP `0.10527811406583186 m`; GRU `0.11868563477533765 m`. Both recorded mean survival lengths are 200 steps. Joint-target-range fields cannot be reconstructed from the published step columns and are outside this check.

## Additional checks

- Safe PyTorch loading (`weights_only=True`) confirms both comparison checkpoints are 27-input / 6-output policies. All stored MLP policy tensors equal the public deployment `.npz` tensors after the documented float64 conversion.
- `CITATION.cff` parses as YAML and validates against the Citation File Format project's official schema fetched on 2026-09-05. The paper is typed `unpublished` pending verified publication metadata, while its note records acceptance.
- New source/public SHA mapping covers 12 curated files. Compressed data and checkpoints are unchanged; text uses LF. Training manifests only replace the historical machine-specific scene path.
- Existing FORMAL-02 / FORMAL-03 evidence, deployment models and photos are preserved. `docs/LOCAL_VERIFICATION.md` remains the historical 2026-08-25 record; its old test count is not this release's count.

## Remaining boundaries

No claim is made that this package reproduces the full original training chain. The starting MLP checkpoint, exact frozen evaluator lineage, private video and external assets remain incomplete or excluded. Licence choice, public manuscript PDF and archival DOI remain unresolved. Publishing this package does not establish EI indexing or a final proceedings venue.
