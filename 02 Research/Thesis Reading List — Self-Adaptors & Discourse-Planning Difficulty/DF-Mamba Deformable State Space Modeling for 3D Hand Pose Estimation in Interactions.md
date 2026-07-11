---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2512.02727
Year: 2026
Should Refer: []
Reading Status: Reading
Venue: WACV/ICCV
Topic: Self-Contact / Hand-Face CV
snippet: "DENSO IT Lab paper (WACV 2026) — interaction-aware 3D hand pose with deformable state-space modeling; read before outreach: self-adaptor detection can be pitched as downstream of this line of work."
Authors: T. Ohkawa, M. Ohi, K. Goto, Y. Sekikawa, N. Inoue
tags:
  - Contact-Aware
  - Hand-Face-CV
Tier: Recommended
Assigned Date: 2026-07-05
---
## Reading Summary

**Abstract**

DF-Mamba proposes a new visual backbone, "Deformable Mamba," for 3D hand pose estimation (HPE) from images. Co-authored by a DENSO IT Laboratory / Institute of Science Tokyo team (Ohkawa, Sekikawa, Inoue et al.), it replaces the ubiquitous ResNet-50 backbone with a state-space-model (Mamba) design that adaptively scans image features rather than sweeping a fixed grid, aiming to better capture global context (inter-joint, inter-hand, hand-object cues) under the severe occlusions typical of real hand interactions.

**Research Question**

Can a deformable, data-driven state-space scanning mechanism extract visual features for 3D hand pose estimation more effectively than CNN backbones (ResNet) or existing fixed-grid Mamba backbones (VMamba, Spatial-Mamba), across single-hand, two-hand, hand-object, RGB and depth scenarios, without sacrificing inference speed?

**Methodology**

The paper introduces Deformable State-Space Modeling (DSSM): a generalization of the Mamba SSM equations in which the scan path is no longer a fixed 2D sweep but a set of K=9 learnable anchors with input-dependent offsets, letting the model selectively sample the most informative regions of the feature map (e.g., ignoring background, attending more densely near joints). DF-Mamba stacks this into a "tribrid" backbone of convolution blocks (early layers), DSSM+FFN blocks (mid/high layers), and gated-convolution blocks (efficiency layers), keeping parameter count (~42M) and FLOPs comparable to ResNet-50. It is dropped into two existing 3D HPE pipelines (Jiang et al. 2023's transformer decoder and Zhou et al. 2020's regression decoder) and evaluated on five datasets: InterHand2.6M (two-hand RGB), RHP (synthetic single-hand), NYU (depth), DexYCB (hand-object), and AssemblyHands (egocentric hand-object), against ResNet-50, ConvNeXt-T, ViT-S, Swin-T, VMamba-T, and Spatial-Mamba-T baselines.

**Findings**

DF-Mamba's deformable scanning is specifically what helps in occlusion-heavy and data-scarce regimes: it is the only backbone that improves over ResNet-50 on the depth-only NYU setting and it gives the largest gains on the egocentric, hand-object AssemblyHands and two-hand InterHand2.6M splits, where fixed-grid Transformer/Mamba backbones (Swin, VMamba, Spatial-Mamba) actually underperform CNNs. Ablations confirm both the deformable-scan mechanism and the tribrid block combination (rather than any single block type) are necessary for the gains, and that 9 anchors (3×3) is the sweet spot for offset flexibility.

**Results**

On InterHand2.6M, DF-Mamba reaches 7.94mm / 10.53mm / 9.32mm MPJPE for single-hand / two-hand / all, versus ResNet-50's 8.10 / 10.96 / 9.63mm. On NYU (depth), it achieves the best mean error of 7.96mm. On DexYCB it gets 17.80mm MPJPE / 87.31% AUC, and on AssemblyHands (egocentric) 18.78mm MPJPE / 86.12% AUC — best among all compared backbones on every dataset. It does so at 112.2 FPS (vs. ResNet-50's 109.2 FPS) and comparable model size (42M params, 4.9G FLOPs), and removing the deformable scan raises InterHand2.6M MPJPE from 9.32mm back up to 9.47mm.

**Conclusion**

The authors conclude DF-Mamba is a practical, near-drop-in replacement for ResNet-50 across diverse 3D hand-pose scenarios, and flag future work on scaling to full-body pose, large-scale pretraining, and hand-mesh decoders. For the thesis, this is a Layer 2 (RQ2, methods) contribution: a state-of-the-art, efficient hand-pose backbone directly relevant to the pose→hand-face-contact stage of a contact-aware self-adaptor detection pipeline. It is also directly strategically valuable for DENSO positioning, since it is co-authored by, and acknowledges funding from, the DENSO IT Lab Recognition, Control & Learning Algorithm Collaborative Research Chair at Institute of Science Tokyo — the exact PhD target group.

*Sources: *[*arXiv:2512.02727 abstract*](https://arxiv.org/abs/2512.02727)*, *[*arXiv full text*](https://arxiv.org/html/2512.02727)*, *[*ResearchGate preprint*](https://www.researchgate.net/publication/398269379_DF-Mamba_Deformable_State_Space_Modeling_for_3D_Hand_Pose_Estimation_in_Interactions)