# FORMAL-02 per-trial readout

## Readout rules

- Independent unit: one commanded real-robot attempt (`trial`).
- Observer endpoints: taken from `remediation/formal_trials/FORMAL-02_ANNOTATION.md`.
- Execution endpoint: taken from each `formal02_N.log`, with `CORRECTION_RECORD.md` governing conflicts between a raw JSON field and the run log.
- Online detector count: taken from each schema v4 JSON; it is a within-trial event count, not an independent sample and not contact ground truth.
- Safety retreat: the JSON composition fields (`retreat_completed`, `select_mode_code`, `restore_code`) were checked; trial 2 was also checked against its run log.
- Raw JSON files remain unchanged.

## Per-trial table

| Trial | Hand side in video | Observer contact | Observer hold ≥0.6 s | Derived execution | Abort reason | Detector events | Start/end temp (°C) | Max tracking error (rad) | JSON duration (s) | Safety retreat |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---|
| FORMAL-02-001 | right | yes | yes | complete | — | 14 | 33/34 | 0.5669 | 36.890 | complete |
| FORMAL-02-002 | left | yes | yes | aborted | TRACKING_ERROR | 11 | 38/40 | 0.6004 | 38.690 | complete |
| FORMAL-02-003 | left | yes | yes | complete | — | 15 | 44/46 | 0.5410 | 39.280 | complete |
| FORMAL-02-004 | right | yes | yes | complete | — | 17 | 49/49 | 0.5406 | 39.840 | complete |
| FORMAL-02-005 | left | yes | yes | complete | — | 21 | 50/50 | 0.5629 | 46.020 | complete |
| FORMAL-02-006 | right | yes | yes | complete | — | 18 | 51/53 | 0.5425 | 40.370 | complete |
| FORMAL-02-007 | right | yes | yes | complete | — | 21 | 53/54 | 0.5595 | 43.860 | complete |
| FORMAL-02-008 | left | yes | yes | complete | — | 21 | 55/56 | 0.5471 | 44.800 | complete |
| FORMAL-02-009 | right | yes | yes | complete | — | 28 | 56/58 | 0.5172 | 52.930 | complete |
| FORMAL-02-010 | left | yes | yes | complete | — | 32 | 59/61 | 0.5509 | 56.230 | complete |

The hand-side labels are image-view labels supplied by the observer. They do not establish robot-left versus robot-right paw correctness because the recording viewpoint was lateral.

## Denominator-preserving summary

| Endpoint | Numerator / denominator | Interpretation |
|---|---:|---|
| Observer-rated contact | 10/10 | Contact was observed in every recorded trial segment. |
| Observer-rated sustained contact ≥0.6 s | 10/10 | Sustained contact was rated in every recorded trial segment. |
| Clean execution completion | 9/10 | Nine attempts completed without an execution abort. |
| Execution abort | 1/10 | FORMAL-02-002 aborted after observed contact because of tracking error. |
| Safety retreat completed | 10/10 | The recorded retreat composition completed for every attempt; trial 2's log shows the recovery sequence. |
| Correct paw selection | not measured | The lateral view cannot reliably map image left/right to robot left/right. |

## Aggregate ranges used by the draft

- Online detector events: 11–32 per trial.
- Start temperature: 33–59 °C.
- End temperature: 34–61 °C.
- Max tracking error for the nine completed attempts: 0.5172–0.5669 rad.
- Tracking error recorded for the aborted attempt: 0.6004 rad.
- Sum of the ten JSON trial durations: 438.900 s.
- Span from the first JSON start timestamp to the last JSON end timestamp: 1005.276 s.

The 7 min 56 s video duration does not cover the entire 1005.276 s wall-clock span between the first and last trial records. The manuscript therefore describes the media as one file containing ten announced trial segments, not as a continuously synchronized recording of all inter-trial intervals.

## Source traceability

- Trial JSON: `remediation/formal_trials/FORMAL-02/trial_v4_FORMAL-02-001.json` through `trial_v4_FORMAL-02-010.json`.
- Run logs: `remediation/formal_trials/FORMAL-02/formal02_1.log` through `formal02_10.log`.
- Observer record: `remediation/formal_trials/FORMAL-02_ANNOTATION.md`.
- Sidecar correction: `remediation/formal_trials/FORMAL-02/CORRECTION_RECORD.md`.
- Video: `趴姿10次录制.mp4`, SHA-256 `ba2997b3f1e209e6c9a16c583b183715c01a8b76f5be77875445535236a60b0e`.
- Model: `simulation/output/policies/prone_px_v4/best_mlp_prone_px.npz`, SHA-256 `9de29f01893534b20cd395de82d3d6096a41a1c17d0db1b43d586b59a00f7958`.
- Trial code commit: `98f7bb80191a1914db9ff5b8c735d7ab901f3593`.
