---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2410.21086
Year: 2024
Should Refer:
  - None
Reading Status: Skim/Skip
Venue: arXiv
Relatedness: not relevant
Topic: Action-Recognition Backbone
snippet: ""
Authors: Jiyao Wang, Xiao Yang, Zhenyu Wang, Ximeng Wei, Ange Wang, Dengbo He, Kaishun Wu
Tags:
  - Cognition
  - Multimodal
Tier: Recommended
Assigned Date: 2026-07-05
---
Found via **CogMamba** (2026-07-04 night). Adjacent literature in CogMamba's immediate task family (CogMamba's reference list wasn't accessible to confirm direct citation): multi-task video-based driver state + rPPG physiological estimation with a mixture-of-experts architecture. Directly supports the DENSO driver cognitive-load monitoring framing (RQ3 / PhD positioning).

## Reading Summary

**Abstract**

This paper proposes VDMoE, a video-only Driver Monitoring System that jointly estimates a driver's cognitive load, drowsiness, heart rate, and respiration rate from RGB facial video, targeting SAE Level-2/3 conditional-autonomy contexts where drivers alternate between over-load (non-driving-related tasks) and under-load (drowsiness). The authors validate it on a newly collected dataset, MCDD (42 participants), plus two public datasets, and claim it is the first model to jointly and non-invasively estimate all four states from video alone.

**Research Question**

Can a single, lightweight, RGB-video-only model exploit the coupling between drowsiness, cognitive load, and physiological signals (rather than modeling each with a separate task-specific model) to monitor driver state non-invasively and efficiently enough for in-vehicle deployment?

**Methodology**

VDMoE extracts key facial landmark regions (eyes, mouth) from video frames rather than processing full frames, and separately derives a Spatial-Temporal Map (STMap) via RGB→YUV color-space transformation and band-pass filtering to expose remote-photoplethysmography (rPPG) signal relevant to heart/respiration rate. These inputs feed a Mixture-of-Experts architecture with a heterogeneous gating mechanism and separate spatial and temporal experts, using lightweight two-layer MLPs (rather than CNN or Transformer experts) as the nonlinear units, followed by four task-specific heads (drowsiness, cognitive load, HR, RR). A "prior-inclusive regularization" term nudges the joint output distribution toward statistical priors from human-factors research (e.g., the expected inverse relationship between cognitive load and drowsiness) to speed convergence and reduce overfitting to individual subjects. The model is benchmarked against single-task and multi-task CNN/Transformer baselines (e.g., ResNet3D, ViViT, PhysMLE, BigSmall) on the authors' new MCDD dataset (42 subjects, ~105,840s of driving-simulator video and physiological recordings under induced non-driving-related-task cognitive load) and two public datasets.

**Findings**

Modeling drowsiness, cognitive load, and physiological state jointly — rather than as independent single-task problems — improves accuracy on every task simultaneously, supporting the paper's premise that these states are functionally coupled. The lightweight MLP-expert MoE design also achieves this while being substantially cheaper than CNN/Transformer-based multi-task baselines, and the prior-inclusive regularization measurably improves convergence stability in ablations.

**Results**

On MCDD, VDMoE achieves the best result on every reported metric: 84.31% accuracy / 69.84 F1 for drowsiness and 79.96% accuracy / 68.81 F1 (77.13 sensitivity, 80.77 specificity) for cognitive load, both significantly ahead of the next-best baselines (e.g., ResNet3D+LSTM-style baselines around 78-81% drowsiness accuracy, ~72% cognitive-load accuracy). For the physiological tasks, VDMoE reaches a heart-rate MAE of 10.32 bpm (vs. 12.03 for the next-best baseline, PhysMLE) and a respiration-rate MAE of 4.98 breaths/min (vs. 5.04-5.12 for the closest baselines), all improvements statistically significant by paired t-test.

**Conclusion**

The authors position VDMoE as a deployable, real-time core for in-vehicle Level-2/3 driver monitoring systems, and release the MCDD dataset to support further research. For the thesis, this is a Layer 2 methods paper whose framing — multi-task, efficient, video-only cognitive-load and physiological-state estimation for driving — is close to a direct precedent for the RQ3 domain-generalization/application layer, and it is the clearest available example of the "driver cognitive-load monitoring" application that anchors the DENSO PhD positioning bridge; it is not itself a domain-generalization paper, so it complements rather than substitutes for REVELIO/DomainBed-style cross-domain evaluation.

*Sources: *[*arXiv:2410.21086 full text*](https://arxiv.org/pdf/2410.21086)*, *[*Semantic Scholar*](https://www.semanticscholar.org/paper/Efficient-Mixture-of-Expert-for-Video-based-Driver-Wang-Yang/83f2656487f05a5677a8b7711b0c1963cb49a02d)*, *[*themoonlight.io*](http://themoonlight.io/)[* literature review*](https://www.themoonlight.io/en/review/efficient-mixture-of-expert-for-video-based-driver-state-and-physiological-multi-task-estimation-in-conditional-autonomous-driving)