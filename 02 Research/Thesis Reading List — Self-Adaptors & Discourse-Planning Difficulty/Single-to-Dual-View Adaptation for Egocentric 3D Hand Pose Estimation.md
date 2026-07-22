---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2403.04381
Year: 2024
Should Refer: []
Reading Status: Reading
Assigned Date: 2026-07-21
Venue: CVPR
Topic: Domain-Generalization
snippet: ""
Authors: Ruicong Liu, Takehiko Ohkawa, Mingfang Zhang, Yoichi Sato
tags:
  - Domain-Generalization
  - Egocentric
Tier: Recommended
Local PDF: "[[99 Assets/Papers/CVPR 2027/Single-to-Dual-View Adaptation for Egocentric 3D Hand Pose Estimation.pdf]]"
Zotero URI: "zotero://select/library/items/VL7C686Z"
Citation Key: "liuSingletoDualViewAdaptationEgocentric2024"
Zotero PDF URI: "zotero://open-pdf/library/items/IL86ZUW3"
---
Found via **DF-Mamba** (2026-07-05 night) — direct reference [38]. Unsupervised adaptation of a pre-trained single-view egocentric hand-pose estimator to dual-view setups without multi-view annotation; relevant to RQ3 (domain generalization / cross-setup robustness) for a contact-aware hand-pose pipeline. Same author group as DF-Mamba (Ohkawa, Sato lab).

## Reading Summary

**Abstract**
This paper proposes S2DHand, an unsupervised method for adapting a pre-trained single-view egocentric 3D hand pose estimator to work accurately in dual-view (stereo) settings, without requiring any multi-view labels or known camera parameters. Using only unlabeled pairs of images from two arbitrarily placed cameras, the method exploits two geometric "stereo constraints" — cross-view consensus and rotation invariance between the two camera coordinate systems — to generate reliable pseudo-labels for adaptation, achieving large accuracy gains over both the single-view baseline and existing unsupervised adaptation techniques.

**Research Question**
Given a hand pose estimator trained only on single-view (monocular) data, can it be adapted, in a fully unsupervised and source-free manner, to exploit an arbitrary pair of egocentric cameras with unknown relative pose so as to substantially improve 3D hand pose accuracy under dual-view inference, and does this adaptation generalize across both in-dataset and cross-dataset (synthetic-to-real) settings?

**Methodology**
The backbone hand pose estimator is DetNet, pre-trained on either the same-domain AssemblyHands dataset (in-dataset setting) or purely synthetic data (Rendered Handpose + GANerated Hands; cross-dataset setting), then evaluated on AssemblyHands' real multi-camera egocentric headset recordings (412K training / 62K testing frames, four synchronized cameras per headset). The adaptation framework maintains a dynamically updated estimator and a momentum-averaged teacher network; the teacher's dual-view predictions are combined into pseudo-labels via two modules — an attention-based merging module that weights each view's joint predictions by heatmap confidence, and a rotation-guided refinement module that adjusts predictions so the implied inter-camera rotation stays consistent with an initialization estimate (solved via BFGS) — and these are combined via a weighted average to supervise the student network. Evaluation uses a novel dual-view MPJPE metric (Dual-M) alongside standard monocular MPJPE (Mono-M), compared against unsupervised domain-adaptation baselines (ADDA, DAGEN, RegDA, SFDAHPE) with ablations on both pseudo-labeling modules and hyperparameters.

**Findings**
Adapting a single-view estimator with the two combined stereo constraints consistently improves accuracy across all camera-pair layouts and both in-dataset and cross-dataset regimes, and the two pseudo-labeling modules are complementary: attention-based merging works well when initial predictions are already fairly accurate, while rotation-guided refinement becomes important specifically when predictions are poor (e.g., under extreme viewing angles or partial hand truncation), correcting cases the merging module alone cannot fix. The method requires no multi-view annotations or camera calibration and is source-free, making it far cheaper to deploy across arbitrary new camera rigs than conventional multi-view training methods.

**Results**
Averaged across all AssemblyHands camera pairs, S2DHand improves both Mono-M and Dual-M metrics by over 10% relative to the pre-trained single-view baseline, with the largest single-pair improvement exceeding 20%. Under the harder cross-dataset setting (synthetic-to-real), S2DHand reduces error from a Source-Only baseline of 56.18/50.59 mm (Mono-M/Dual-M, Headset1) to 48.44/45.92 mm, outperforming the strongest competing unsupervised adaptation baseline (RegDA: 51.41/47.85 mm) while remaining source-free, unlike the comparably strong SFDAHPE baseline. Ablations confirm both pseudo-labeling modules contribute (combined use gives the best result, e.g., 20.16/19.98 mm vs. 24.96/22.67 mm with neither module active, Headset1 in-dataset), and performance stabilizes once at least roughly 1000 unlabeled adaptation pairs are available.

**Conclusion**
The authors conclude that geometric stereo constraints alone — without any labels or camera calibration — are sufficient to adapt a monocular hand pose estimator into an accurate, flexible dual-view estimator compatible with arbitrary camera layouts, a practically important property given the proliferation of multi-camera AR/VR headsets. For the thesis, this is a direct Layer 2/RQ3 methods contribution: it is a concrete, benchmarked example of unsupervised cross-domain/cross-setup adaptation for exactly the kind of hand-pose backbone a contact-aware self-adaptor detector would depend on, and its source-free, label-free adaptation recipe is a template for making a hand-face contact pipeline robust to unseen camera rigs or recording setups — directly relevant to domain-generalization claims and, by extension, to the DENSO positioning around robust, deployable perception under varying real-world sensing conditions.

*Sources: [arXiv:2403.04381](https://arxiv.org/abs/2403.04381), [CVPR 2024 Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Single-to-Dual-View_Adaptation_for_Egocentric_3D_Hand_Pose_Estimation_CVPR_2024_paper.html), [Sato Lab project page](https://www.ut-vision.org/publication/2024-liu-single/), [GitHub (S2DHand)](https://github.com/ut-vision/S2DHand)*