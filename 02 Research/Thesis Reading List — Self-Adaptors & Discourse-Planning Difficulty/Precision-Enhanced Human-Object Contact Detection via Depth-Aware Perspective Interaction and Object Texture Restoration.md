---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2412.09920
Year: 2025
Should Refer: []
Reading Status: Reading
Venue: AAAI
Code Link: https://github.com/YuxiaoWang-AI/PIHOT
Topic: Self-Contact-Hand-Face-CV
snippet: ""
Authors: Y. Wang, W. Neng, Z. Wei, Y. Lei, W. Xue, N. Zhuang, Y. Xu, X. Jiang, Q. Liu
tags:
  - Contact-Aware
Tier: Recommended
Assigned Date: 2026-07-15
---
Found via **Detecting Human-Object Contact in Images (HOT)** (2026-07-11 night) — adjacent-literature substitute, not a backward reference (published after HOT). Proposes PIHOT, which explicitly builds on the HOT detector/dataset, adding depth-aware perspective interaction and object-texture restoration to sharpen contact-region boundaries. A direct methodological successor for RQ2's contact-detection stage, useful for improving boundary precision in a future hand-face contact detector.

## Reading Summary

**Abstract**
PIHOT tackles human-object contact (HOT) detection — segmenting the precise regions where a person's body touches an object — and specifically targets the failure mode where an object is partly occluded by the human body, causing 2D-overlap-based methods to misjudge contact (e.g., a person merely standing near a wall in 2D projection looks like they are touching it). The authors combine an object-inpainting module that restores the texture and shape of occluded object regions with a depth-aware module that estimates and compares depth maps of the original and repaired images, using the depth difference as direct evidence of whether human and object are actually at the same physical distance from the camera (true contact) or merely visually overlapping (false contact).

**Research Question**
Can restoring occluded object texture and explicitly modeling depth/perspective relationships between humans and objects reduce the false-contact errors that plague 2D-segmentation-based HOT detectors, and produce sharper, more accurate contact-region boundaries?

**Methodology**
Building on the HOT task and dataset introduced by Chen et al. (CVPR 2023), PIHOT adds three components on top of a shared ResNet-50 backbone: an Object Inpainting (OI) module that uses a dilated human mask and a LaMa inpainting model to remove the person and reconstruct the occluded object's texture; a Space Perception Operation (SPO) module that runs a ZoeDepth model over both the original and the object-repaired image to obtain two depth maps, whose normalized difference yields a relative spatial-position feature; and two attention mechanisms — Instances Perspective Interaction (IPI), which fuses object features into the contact-branch features via cross-attention, and Instances Depth Space Interaction (IDSI), which further fuses the depth-difference signal into that representation — before a Contact Perception Operation (CPO) module produces the final 18-class (17 body-part-contact classes + background) segmentation map, trained with per-pixel cross-entropy loss. Evaluation uses three benchmarks: HOT-Annotated (V-COCO/HAKE/Watch-n-Patch, ~15,000 images), HOT-Generated (PROX/SMPL-X, ~20,000 images), and their union (Full Set), scored with semantic contact accuracy (SC-Acc.), contact accuracy (C-Acc.), mean IoU, and weighted IoU, benchmarked against the prior state of the art, DHOT.

**Findings**
The core qualitative insight is that occlusion, not raw visual ambiguity, is the main source of contact-boundary error in prior HOT models: once the occluded object's shape and texture are hallucinated back in and its depth is compared against the human's depth, the network can distinguish "close in the image but far in depth" (false contact, e.g., a person and a wall behind them) from genuine physical contact far more reliably than a segmentation model working from RGB alone. The ablations show that each added component (inpainting, perspective attention, depth-space attention) contributes incrementally, with the object-inpainting step generally providing the largest single jump, confirming that recovering occluded structure — rather than the depth-difference computation per se — is the primary driver of the gain.

**Results**
On HOT-Annotated, PIHOT reaches 45.3 SC-Acc., 80.7 C-Acc., 0.236 mIoU, and 0.286 wIoU, improving over the strongest DHOT variant by roughly 11%, 14%, 10%, and 10% respectively; on HOT-Generated it reaches 34.9, 76.3, 0.169, and 0.212, improving by roughly 15%, 41%, 22%, and 27%; on the combined Full Set it improves over the second-best method by 16%, 17%, 6%, and 9%, reaching 77.5 C-Acc. and 0.273/0.221 for wIoU/mIoU respectively. Ablations on HOT-Annotated show the baseline model at 40.5/73.8/0.214/0.263, rising to 45.3/80.7/0.236/0.286 with all four modules (OI, IPI, SPO, IDSI) active; the paper reports the overall average improvement over DHOT across metrics and datasets as roughly 13%, 27.5%, 16%, and 18.5% for SC-Acc., C-Acc., mIoU, and wIoU respectively.

**Conclusion**
The authors conclude that jointly leveraging occlusion-aware texture restoration and depth-based spatial reasoning substantially improves HOT segmentation accuracy under real-world occlusion, and note as a limitation that the approach may still struggle in extreme-occlusion or dynamic-scene settings, suggesting future unification with 3D/video-based contact detection. For the thesis this is a Layer 2/RQ2 methods paper: it is a direct, publicly-coded (LaMa + ZoeDepth pipeline) template for the contact-aware detection stage of the proposed pose→hand-face-contact pipeline, since self-adaptor detection faces an analogous occlusion problem (a hand touching the face is, by definition, partially occluding both), and its depth-difference trick for disambiguating true from apparent contact could transfer directly to hand-face self-contact detection.

*Sources: [arXiv abstract page](https://arxiv.org/abs/2412.09920), [arXiv HTML full text](https://arxiv.org/html/2412.09920v2), [AAAI 2025 proceedings page](https://ojs.aaai.org/index.php/AAAI/article/view/32883), [GitHub code repository](https://github.com/YuxiaoWang-AI/PIHOT)*
