---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2007.15815
Year: 2021
Should Refer: []
Reading Status: Reading
Assigned Date: 2026-07-21
Venue: journal
Topic: Self-Contact-Hand-Face-CV
snippet: IEEE TAFFC journal extension of Lin et al. (2020, IEEE FG) already read — fuller fidgeting feature set (multi-modal deep denoising auto-encoders + Fisher vector encoding); read for methods detail beyond the conference version.
Authors: W. Lin, I. Orton, Q. Li, G. Pavarini, M. Mahmoud
tags:
  - Self-Adaptor
  - Distress
  - Hand-Face-CV
Tier: Recommended
Local PDF: "[[99 Assets/Papers/CVPR 2027/Looking At The Body Automatic Analysis of Body Gestures and Self-Adaptors in Psychological Distress.pdf]]"
Zotero URI: "zotero://select/library/items/H7634NRU"
Citation Key: "linLookingBodyAutomatic2021"
Zotero PDF URI: "zotero://open-pdf/library/items/B24ZWRXG"
---

## Reading Summary

**Abstract**
This paper (a TAFFC journal extension of Lin et al.'s 2020 FG conference paper) presents an automatic pipeline for detecting self-adaptors and fidgeting from full-body video and uses these detected behavioral cues to predict psychological distress (depression and anxiety). The authors introduce a new audio-visual interview dataset with self-reported distress labels, a hierarchical computer-vision detector for self-adaptors and fidgeting, and a multimodal fusion classifier combining fidgeting features with facial Action Units, gaze, and speech (MFCCs).

**Research Question**
Can self-adaptors and fidgeting behavior be automatically and reliably detected from ordinary RGB video of naturalistic (non-acted) interviews, and, if so, do these automatically detected behavioral cues carry predictive signal for psychological distress (depression/anxiety) beyond or in combination with other modalities?

**Methodology**
The authors collected a new dataset of 35 participants (18 high-distress, 17 low-distress by PHQ-8/GAD-7 screening) recorded during semi-structured, DAIC-inspired peer-support interviews (~7h50m total), with distress labeled via PHQ-8 (depression), GAD-7 (anxiety), SSS-8 (somatic symptoms), PSS (perceived stress), and the Big Five Inventory. Body pose (OpenPose), facial Action Units and gaze (OpenFace 2.2), and diarized audio (MFCCs) were extracted. A hierarchical self-adaptor detector first identifies location events (hand-to-hand, hand-to-face, hand-to-arm, hand-to-leg, leg-crossing, etc.) via bounding-box overlap on OpenPose keypoints, then a DYNAMIC/STATIC classifier operating on optical-flow FFT/standard-deviation/mean features distinguishes fidgeting (rhythmic, non-functional movement) from static self-touch. Detected fidgeting is validated against manual annotation (Krippendorff's alpha up to 1.00) and against an independent, previously-published acted fidgeting dataset (Mahmoud et al., 2013). For distress classification, generic body-gesture statistics and fidgeting features are fused with AUs, gaze, and MFCCs via a Multi-modal Deep Denoising Auto-Encoder, aggregated per-video with Improved Fisher Vector encoding over a Gaussian Mixture Model, and classified with logistic regression or an MLP (with label smoothing) after Random Forest feature selection, evaluated via participant-independent 3-fold cross-validation.

**Findings**
Fidgeting is not just noise but a behaviorally meaningful, learnable signal: a purely statistical linear-regression analysis on hand-crafted body-gesture features already reached moderate predictive power for depression, and adding fidgeting features improved this substantially. Feature-level analysis suggested that head and leg movement duration/frequency, plus right-hand fidgeting directed at the leg/arm/face, were positively associated with higher depression scores, while generic hand-gesture regularity (rather than raw amount of hand motion) was the more informative hand-related cue. In the deep multimodal pipeline, fidgeting features combined with a co-occurring "is-speaking" signal improved distress classification over facial/audio features alone, and cross-dataset validation showed the detector generalizes to an independent, previously unseen acted dataset.

**Results**
The self-adaptor detector achieved high precision/recall on manually labeled frames (e.g., F1 = 1.00 for hand-to-hand and leg self-adaptors; F1 = 0.77–0.91 for hand-to-arm/leg/face events), and the DYNAMIC/STATIC fidgeting classifier reached roughly 83–90% accuracy across categories. In the statistical analysis of depression classification, a linear model using only global body motion reached F1 = 34.43%, full body-gesture statistics reached F1 = 66.81%, an exhaustive feature search over body gestures alone reached F1 = 82.70%, and combining searched body-gesture and fidgeting features reached the best F1 = 83.38%. On the independent cross-dataset validation (Mahmoud et al.'s acted fidgeting videos), the detector achieved 80% recall for fidget-vs-no-fidget detection and 76.8–86.5% recall across specific fidget types (leg, hand-to-face, hand-to-arm, hand-cross), outperforming the prior state of the art (59% recognition accuracy in the original Mahmoud et al. system).

**Conclusion**
The authors conclude that self-adaptors and fidgeting, automatically detected from ordinary video via a hierarchical pose-based pipeline, constitute a valid and generalizable behavioral signal that meaningfully improves multimodal distress detection, particularly when combined with speech co-occurrence information. For the thesis, this paper is highly relevant to both layers: it is close prior art for Layer 1 (self-adaptors/fidgeting as markers of an internal difficulty state — here psychological distress rather than discourse-planning difficulty specifically) and directly informs Layer 2 methodology, since its hierarchical self-adaptor/fidgeting detection pipeline (bounding-box contact detection + optical-flow dynamics classifier) is essentially a lightweight, non-neural predecessor to the contact-aware detection pipeline the thesis proposes, and its cross-dataset validation result is a useful generalization benchmark to compare against for RQ3.

*Sources: [arXiv:2007.15815](https://arxiv.org/abs/2007.15815), [IEEE Xplore](https://ieeexplore.ieee.org/document/9506822/)*
