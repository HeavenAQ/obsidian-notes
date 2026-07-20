---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640485.pdf
Year: 2022
Should Refer:
  - Architecture
  - Benchmark
Reading Status: Reading
Venue: ECCV
Code Link: https://github.com/happyharrycn/actionformer_release
Relatedness: Strongly
Topic: Video-Backbone
snippet: "Implement this as the primary generic temporal localization baseline. Extract the feature format, temporal pyramid, local attention, boundary regression, loss, and mAP protocol."
Authors: Chen-Lin Zhang, Jianxin Wu, Yin Li
tags:
  - Temporal-Action-Localization
  - Transformer
  - Benchmark
Tier: Essential
Assigned Date: 2026-07-12
Citation Key: "zhang2022actionformer"
Local PDF: "[[99 Assets/Papers/CVPR 2027/ActionFormer Localizing Moments of Actions with Transformers.pdf]]"
Zotero URI: "zotero://select/library/items/I69UFHR3"
Zotero PDF URI: "zotero://open-pdf/library/items/WK3MUG6N"
---
# Reading objective

Implement this as the primary generic temporal localization baseline. Extract the feature format, temporal pyramid, local attention, boundary regression, loss, and mAP protocol.

## Extraction checklist

- [ ] Exact task, inputs, outputs, and claimed novelty
- [ ] Dataset size, split unit, annotations, and license
- [ ] Strongest relevant baseline and metric protocol
- [ ] Component or loss worth reproducing
- [ ] Failure mode or limitation that affects [[CVPR 2027 Submission Roadmap — Temporal Self-Contact]]

## Notes

## Reading Summary

**Abstract**
ActionFormer applies Transformer self-attention to temporal action localization in untrimmed video, proposing a single-shot detector that identifies action instances and their boundaries directly, without the region-proposal or predefined anchor-window machinery that dominated prior temporal action detection pipelines. The paper reports large accuracy gains over prior methods on three standard benchmarks and releases code that has since become a widely used baseline in the temporal localization literature (it is, for instance, the localization head used in SocialGesture, tonight's other read).

**Research Question**
The paper asks whether a purpose-built Transformer architecture — combining multiscale feature representations with local self-attention — can localize and classify action instances in long, untrimmed videos more accurately than proposal-based or anchor-based detectors, while remaining simple (single-shot, no proposal generation stage).

**Methodology**
ActionFormer combines a multiscale feature pyramid (built by downsampling an input feature sequence across several temporal resolutions) with local self-attention at each scale, so the model can reason about both fine-grained and coarse-grained temporal context efficiently. A lightweight convolutional decoder then operates on every time step of every pyramid level to jointly (a) classify whether an action is present and which category it belongs to, and (b) regress the action's start and end boundaries relative to that time step — collapsing detection into a dense, per-moment classification-plus-regression problem rather than a two-stage propose-then-classify pipeline. The model is evaluated on three standard temporal action localization benchmarks: THUMOS14, ActivityNet-1.3, and EPIC-Kitchens 100, using the standard mean Average Precision (mAP) protocol across a range of temporal IoU thresholds.

**Findings**
The core qualitative finding is that a well-designed, purely Transformer-based dense detector can outperform the anchor- and proposal-based paradigms that had previously dominated temporal action localization, and that local (rather than global) self-attention combined with a multiscale pyramid is important for handling the wide range of action durations found in real untrimmed video. The design has proven durable as a backbone/baseline choice in later work (including domain-specific gesture and behavior localization tasks), suggesting the architecture generalizes well beyond the benchmarks it was validated on.

**Results**
ActionFormer achieves 71.0% mAP at tIoU = 0.5 on THUMOS14, a 14.1 absolute percentage point improvement over the best prior model at the time. On ActivityNet-1.3 it reaches 36.6% average mAP, and on EPIC-Kitchens 100 it improves average mAP by +13.5 percentage points over prior work. These are single-shot, non-ensembled results using standard input features (e.g., I3D/TSN-derived), reported without additional post-processing beyond standard soft-NMS-style suppression.

**Conclusion**
The authors conclude that Transformer-based dense detection is a strong and simple alternative to proposal-based temporal action localization across diverse benchmark types (sports actions, everyday activities, egocentric kitchen actions). For the thesis, this is a Layer 2 (RQ2, methods) backbone paper: it is the direct methodological candidate for the "fidgeting dynamics" stage of the proposed pipeline — turning frame-level hand-face contact/pose signals into temporally localized self-adaptor bouts — and its cross-domain robustness (sports to kitchens) is a mild positive signal for the kind of cross-dataset generalization the thesis's RQ3 targets, though the paper itself does not evaluate domain generalization or cognitive load and so contributes architecture rather than thematic DENSO positioning.

*Sources: [arXiv:2202.07925](https://arxiv.org/abs/2202.07925), [GitHub code release](https://github.com/happyharrycn/actionformer_release), [ECCV 2022 proceedings](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640485.pdf)*

