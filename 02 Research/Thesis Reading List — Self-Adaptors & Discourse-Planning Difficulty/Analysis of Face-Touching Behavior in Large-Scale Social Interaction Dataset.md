---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://doi.org/10.1145/3382507.3418876
Year: 2020
Should Refer:
  - None
Reading Status: Read
Venue: ICMI
Code Link: https://github.com/IIT-PAVIS/Face-Touching-Behavior
Relatedness: weakly
Topic: Self-Contact / Hand-Face CV
snippet: |2-
    • provide a dataset of hand-face interaction 
    • too old and from a relatively small conference
    • used several approaches including rule-based, hand-crafted features, NN, CNN
    • CNN based approach (ResNet152) for feature learning-based detection worked the best
Authors: C. Beyan, M. Bustreo, M. Shahid, G. L. Bailo, N. Carissimi, A. Del Bue
tags:
  - Dataset
  - Hand-Face-CV
  - Benchmark
Tier: Essential
Assigned Date: 2026-07-02
---
**Background.**  face-touch detection lacked public benchmarks. 

**Research question.**  can face-touching be reliably annotated and detected in group-interaction video? 

**Methodology.**  public frame-level annotations over 64 small-group videos with ROIs and OpenPose keypoints, plus baseline classifiers. 

**Conclusion.**  high inter-annotator agreement and workable baselines -- the practical starting point for candidate generation, weak-supervision seeds, and visual pretraining.

## Reading Summary

**Abstract**

This ICMI 2020 paper releases the first public frame-level annotations of face-touching behavior in a large social-interaction video corpus and benchmarks several detection methods on them. The annotations cover 64 videos (12–30 minutes each) from four-person meetings in IIT-PAVIS's Leadership Corpus, with roughly 74K face-touch frames and 2M no-face-touch frames. A CNN-based feature-learning approach performs best among rule-based, hand-crafted-feature, and learned-feature methods.

**Research Question**

Can face-touching behavior be reliably annotated at frame level in naturalistic group-interaction video, and how well do rule-based versus supervised (hand-crafted vs. learned feature) approaches detect it?

**Methodology**

Sixteen annotators labeled every frame of 64 single-person views from four-people meetings as face-touch or no-face-touch, achieving almost-perfect inter-annotator agreement (Cohen's Kappa = 0.89). The release includes OpenPose body/face/hand keypoints for all frames, face ROIs, hand-crafted features (face and hand bounding boxes, face parts, hand keypoints), and fixed cross-validation splits. Three detector families were evaluated: a rule-based method over pose geometry, a neural classifier on hand-crafted features, and end-to-end CNN feature learning on image data.

**Findings**

Frame-level face-touch annotation is feasible and highly reliable in real meeting video, and learned visual features clearly beat geometric rules and hand-crafted pose features, which suffer when keypoint detection fails (occlusion, blur, unusual postures). The heavy class imbalance (~74K positive vs. ~2M negative frames) makes the problem a realistic rare-event detection benchmark.

**Results**

The CNN feature-learning approach reached 83.76% F1 and 0.84 Matthews Correlation Coefficient, the best of the evaluated methods; annotation agreement was Cohen's Kappa 0.89. Annotations, features, splits, and baseline code are public ([github.com/IIT-PAVIS/Face-Touching-Behavior](http://github.com/IIT-PAVIS/Face-Touching-Behavior); data via the PAVIS Leadership Corpus form).

**Conclusion**

The authors conclude that large-scale, reliable face-touch annotation plus learned visual features yields workable detection in unconstrained group interaction, and they provide the benchmark to build on. For the thesis, this is the practical starting point for self-adaptor candidate generation: its annotations and pretrained detection setup can seed weak supervision and visual pretraining before finer self-adaptor/discourse alignment. Note: the full ACM text was paywalled, so this summary draws on the abstract, the official GitHub release, and secondary sources; depth on ablations is limited by access.

*Sources: *[*https://dl.acm.org/doi/10.1145/3382507.3418876*](https://dl.acm.org/doi/10.1145/3382507.3418876)*, *[*https://github.com/IIT-PAVIS/Face-Touching-Behavior*](https://github.com/IIT-PAVIS/Face-Touching-Behavior)*, *[*https://www.semanticscholar.org/paper/19c433ca972ce22f72b584e3bb6307e8ee721149*](https://www.semanticscholar.org/paper/19c433ca972ce22f72b584e3bb6307e8ee721149)