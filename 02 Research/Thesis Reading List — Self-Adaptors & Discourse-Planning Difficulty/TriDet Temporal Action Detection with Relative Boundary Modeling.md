---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://openaccess.thecvf.com/content/CVPR2023/html/Shi_TriDet_Temporal_Action_Detection_With_Relative_Boundary_Modeling_CVPR_2023_paper.html
Year: 2023
Should Refer:
  - Architecture
  - Benchmark
Reading Status: Reading
Venue: CVPR
Code Link: https://github.com/sssste/TriDet
Relatedness: Strongly
Topic: Video-Backbone
snippet: "Study the relative boundary distribution and multi-granularity temporal features for short, ambiguous self-contact events. Use it as a second strong detector or borrow only its boundary head."
Authors: Dingfeng Shi, Yujie Zhong, Qiong Cao, Lin Ma, Jia Li, Dacheng Tao
tags:
  - Temporal-Action-Localization
  - Boundary-Detection
  - Transformer
Tier: Essential
Assigned Date: 2026-07-14
Citation Key: shi2023tridet
Local PDF: "[[99 Assets/Papers/CVPR 2027/TriDet Temporal Action Detection with Relative Boundary Modeling.pdf]]"
Zotero URI: "zotero://select/library/items/WXLMGWFL"
Zotero PDF URI: "zotero://open-pdf/library/items/RNSAP9RN"
---
# Reading objective

Study the relative boundary distribution and multi-granularity temporal features for short, ambiguous self-contact events. Use it as a second strong detector or borrow only its boundary head.

## Extraction checklist

- [ ] Exact task, inputs, outputs, and claimed novelty
- [ ] Dataset size, split unit, annotations, and license
- [ ] Strongest relevant baseline and metric protocol
- [ ] Component or loss worth reproducing
- [ ] Failure mode or limitation that affects [[CVPR 2027 Submission Roadmap — Temporal Self-Contact]]

## Notes

## Reading Summary

**Abstract**

TriDet is a one-stage framework for temporal action detection (TAD) — localizing the start/end boundaries and category of each action instance in an untrimmed video. The paper's central observation is that action boundaries in real video are inherently ambiguous (annotators and models alike struggle to place an exact frame), so instead of regressing a single boundary offset, TriDet models the boundary as an estimated relative probability distribution around it via a "Trident-head." It pairs this with a Scalable-Granularity Perception (SGP) layer in the feature pyramid, replacing the self-attention typically used in transformer-based TAD feature pyramids, which the authors show suffers from a rank-collapse problem on video features.

**Research Question**

How can a temporal action detector produce more precise boundary localization under inherent boundary ambiguity, and can this be done more efficiently than transformer-based feature pyramids, which the authors argue lose discriminative power (rank deficiency) when self-attention is applied directly to video features?

**Methodology**

TriDet is built on the ActionFormer codebase and Detectron2. The Trident-head predicts, for each candidate boundary, a relative probability distribution over nearby positions rather than a point estimate, giving the model a principled way to express boundary uncertainty. The SGP layer replaces self-attention in the feature pyramid with an efficient mechanism for aggregating information across multiple temporal granularities, addressing a rank-loss problem the authors identify empirically in self-attention applied to video features. The method is evaluated on three benchmarks in the main paper — THUMOS14, HACS, and EPIC-Kitchens-100 — and the released codebase additionally reports ActivityNet results; all use pre-extracted video features (e.g., I3D, SlowFast) rather than end-to-end video input.

**Findings**

Modeling the boundary as a distribution rather than a point, combined with a lighter-weight multi-granularity aggregation mechanism than self-attention, improves precision specifically on short and ambiguous action segments while reducing compute relative to transformer-based feature pyramids — the paper frames this explicitly as a rank/discriminability argument against naively porting self-attention into dense per-frame video feature pyramids.

**Results**

TriDet reports an average mAP of 69.3% on THUMOS14, 2.5 points above the prior best, while running at only 74.6% of that method's latency. The public repository reports THUMOS14 average mAP of 69.27% (mAP at tIoU 0.3–0.7: 83.62/80.07/72.94/62.35/47.35), HACS average 38.69 (tIoU 0.5/0.75/0.95: 56.90/39.33/11.24), ActivityNet average 36.77 (54.71/38.01/8.35), and EPIC-Kitchens average mAP of 23.76 (noun) and 25.51 (verb) across tIoU 0.1–0.5.

**Conclusion**

The authors conclude that explicit boundary-distribution modeling and granularity-aware, attention-free feature aggregation together give a more accurate and cheaper one-stage TAD framework than prior transformer-heavy designs. For the two-layer thesis this is a Layer 2/RQ2 methodological candidate: the Trident-head's core motivation — precisely localizing short, inherently ambiguous temporal boundaries — maps directly onto the problem of detecting the onset and offset of brief self-adaptor/fidgeting episodes in continuous video, making TriDet (or its SGP layer specifically) a plausible detector head or comparison baseline for the contact-aware pipeline's temporal segmentation stage. It carries no cognitive-load or domain-generalization content itself, so its relevance is architectural rather than to RQ1 or RQ3, and it does not bear on the DENSO/driver-monitoring positioning directly.

*Sources: [arXiv:2303.07347](https://arxiv.org/abs/2303.07347), [CVPR 2023 Open Access page](https://openaccess.thecvf.com/content/CVPR2023/html/Shi_TriDet_Temporal_Action_Detection_With_Relative_Boundary_Modeling_CVPR_2023_paper.html), [GitHub repository](https://github.com/dingfengshi/TriDet)*

