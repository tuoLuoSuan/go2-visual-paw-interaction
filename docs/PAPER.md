# Associated manuscript

## Bibliographic information

**Constrained Monocular Vision-Guided Handshake Interaction for Quadruped Robots**

Liangbin Wu¹†, Shouchen Chen²†, Yihuang Zheng², Hongxin Chen² and Xiaowei Chen³*.

1. School of Information Engineering, Fujian Polytechnic of Water Conservancy and Electric Power, Fujian, China.
2. College of Computer and Cyber Security, Fujian Normal University, Fujian, China.
3. Fujian Provincial Key Laboratory of Network Security and Cryptology, College of Computer and Cyber Security, Fujian Normal University, Fujian, China.

† Liangbin Wu and Shouchen Chen contributed equally and share first authorship. * Corresponding author: Xiaowei Chen, chenxw@fjnu.edu.cn.

Status checked on 2026-09-20 against the authors' acceptance notice dated 2026-08-27: **accepted by the 3rd International Conference on Computational Intelligence and Communication System (CCICS 2026), manuscript CC178**. The notice lists alternative proceedings series; the final publication venue, volume, pages, DOI and EI indexing have not been verified. Use “accepted”, not “published” or “EI-indexed”.

The current author-prepared version uses a nine-page JPCS layout after the organizer invited an 8–10 page expansion. A template is not proof of final publication venue. The author version and its LaTeX source are included in [`docs/manuscript/`](manuscript/); the acceptance letter, reviewer correspondence, registration materials and raw videos remain private.

## Current manuscript files

- [CC178 expanded nine-page PDF](manuscript/CC178_Expanded_JPCS_9pages.pdf)
- [LaTeX source package](manuscript/CC178_Expanded_LaTeX_Source.zip)

The expanded version adds implementation details, archived training settings, endpoint decision rules and a descriptive MLP/GRU distance figure. It does not add robot trials or change the ten-trial denominators. The paper reports the author-prepared submission version, not a publisher-formatted proceedings version.

## What the paper reports

A GO2 EDU uses its built-in monocular RGB camera to detect a hand in a constrained interaction region. Image-space information drives a prone front-paw interaction controller, with execution gates and withdrawal handling. Ten consecutive prone attempts yielded observer-rated contact and a hold of at least 0.6 s in 10/10 segments, while logs support 9/10 clean executions and one tracking-error abort after observed contact. A separate standing demonstration and descriptive MLP/GRU checkpoint evaluation provide supplementary engineering context.

These observations do not establish unrestricted 3D reaching, stability improvement, contact-force accuracy, cross-user generalization or a general MLP-versus-GRU ranking.

## Result-to-material index

| Manuscript content | Public material | Interpretation |
|---|---|---|
| Prone observer contact / hold, 10/10 | [Observer annotation](../data/formal02/FORMAL-02_ANNOTATION.md) | One non-operator observer; not sensor ground truth |
| Prone clean execution, 9/10; trial 2 abort | [Raw records and logs](../data/formal02/README.md), [correction](../data/formal02/CORRECTION_RECORD.md) | Raw JSON error retained; correction sidecar governs derived status |
| Paw selection | [Observer annotation](../data/formal02/FORMAL-02_ANNOTATION.md) | Not measured from lateral view |
| MLP/GRU distance table | [Evaluation dataset](../data/policy_comparison/README.md), [checkpoint cards](../models/prone_comparison/MODEL_CARD.md) | Single training seed per architecture; unequal training origins |
| Standing extension | [FORMAL-03](../data/formal03_standing/README.md) | Separate qualitative execution evidence |
| Representative experiment photos | [Photo provenance](../figures/figure_photo_provenance.md) | Existing public frames, not a substitute for complete trial evidence |

The public figures and structured records support the claims in the manuscript. The PDF and source package above are the exact author-prepared nine-page build; repository figures remain separately indexed for code/data inspection.

## Citation

Wu, L., Chen, S., Zheng, Y., Chen, H., and Chen, X. (2026). *Constrained Monocular Vision-Guided Handshake Interaction for Quadruped Robots*. Accepted by CCICS 2026, manuscript CC178; publication details pending.

The machine-readable [CITATION.cff](../CITATION.cff) separates software attribution from the paper's author list. Its preferred citation is temporarily typed `unpublished` because acceptance is confirmed but publication metadata are not. Add the verified proceedings citation and DOI when available.
