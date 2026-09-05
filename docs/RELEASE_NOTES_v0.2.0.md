# v0.2.0 — Manuscript evidence update

Date: 2026-09-05. This release updates the public evidence package; it does not change the robot controller or report new experiments.

## Added

- CCICS 2026 accepted-manuscript metadata, author order, shared-first-authorship statement and a result-to-material index.
- Original 27-input / 6-output prone MLP and GRU comparison checkpoints, archived step/episode measurements, recorded training configurations, rounded table source and metric reconstruction script.
- Source/public SHA-256 provenance mapping for the new evidence, with text normalization and scene-path sanitization recorded.
- README display of two existing public experiment photos and a hardware-free quick start.
- Paper `preferred-citation` separate from software attribution.

## Unchanged and unresolved

- FORMAL-02 remains ten commanded trials: observer contact and hold 10/10, clean execution 9/10, one abort, selected-paw correctness not measured. Original evidence and its correction sidecar are unchanged.
- Standing remains a separate qualitative extension, not the GRU comparator and not pooled with prone results.
- The two simulated checkpoints have one training seed each and unequal training origins. Reconstructing recorded metrics is not reproducing training from scratch.
- No manuscript PDF, original video, participant forms, private settings, payment/registration documents or third-party model assets are added.
- No publication DOI, confirmed proceedings venue, EI indexing, archival DOI or open licence is asserted. PDF sharing and licence choice await the authors/rights holders.

The release tag identifies this evidence package, not a complete frozen copy of every submitted manuscript artifact. Verification commands and scope are in [VERIFICATION_20260905.md](VERIFICATION_20260905.md).
