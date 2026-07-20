---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://openaccess.thecvf.com/content/CVPR2026/html/Yang_SAM_3D_Body_Robust_Full-Body_Human_Mesh_Recovery_CVPR_2026_paper.html
Year: 2026
Should Refer:
  - Architecture
  - Benchmark
Reading Status: Reading
Venue: CVPR
Code Link: https://github.com/facebookresearch/sam-3d-body
Relatedness: Strongly
Topic: Self-Contact-Hand-Face-CV
snippet: "This is a current strong full-body HMR baseline. Test whether its hand recovery and unusual-pose generalization improve contact-candidate quality over supplied SMPL-H on a manageable subset."
Authors: Xitong Yang, Devansh Kukreja, Don Pinkus, Taosha Fan, Jinhyung Park, Soyong Shin, Jinkun Cao, Jia-Wei Liu, Nicolas Ugrinovic, Anushka Sagar, Jitendra Malik, Matt Feiszli, Piotr Dollar, Kris Kitani
tags:
  - Human-Mesh-Recovery
  - Hand-Pose
  - Foundation-Model
Tier: Essential
Assigned Date: 2026-07-14
Citation Key: "yang2026sam3dbody"
Local PDF: "[[99 Assets/Papers/CVPR 2027/SAM 3D Body Robust Full-Body Human Mesh Recovery.pdf]]"
Zotero URI: "zotero://select/library/items/4N5FTVAD"
Zotero PDF URI: "zotero://open-pdf/library/items/TQ3CJIQV"
---
# Reading objective

This is a current strong full-body HMR baseline. Test whether its hand recovery and unusual-pose generalization improve contact-candidate quality over supplied SMPL-H on a manageable subset.

## Extraction checklist

- [ ] Exact task, inputs, outputs, and claimed novelty
- [ ] Dataset size, split unit, annotations, and license
- [ ] Strongest relevant baseline and metric protocol
- [ ] Component or loss worth reproducing
- [ ] Failure mode or limitation that affects [[CVPR 2027 Submission Roadmap — Temporal Self-Contact]]

## Notes

## Reading Summary

**Abstract**

SAM 3D Body (3DB), from Meta Superintelligence Labs, is a promptable model for single-image full-body 3D human mesh recovery (HMR) that estimates body, feet, and hand pose together. It introduces the Momentum Human Rig (MHR), a new parametric mesh representation that decouples skeletal structure from surface shape, and pairs it with a promptable encoder–decoder architecture (accepting optional 2D keypoint, mask, or camera prompts, echoing the original Segment Anything interaction pattern) and a large VLM-driven data engine built to mine and annotate rare poses and hard viewpoints at scale.

**Research Question**

Can a promptable encoder–decoder architecture, combined with a data engine that actively mines for rare/hard in-the-wild cases rather than relying on random sampling, produce a full-body (body+feet+hands) mesh recovery model that generalizes substantially better than prior HMR methods to unusual poses, ambiguous viewpoints, and out-of-distribution imaging conditions — while remaining competitive with specialist hand-only estimators on hand accuracy?

**Methodology**

3DB uses a shared image encoder with separate body and hand decoders; the hand decoder attends to hand crops via cross-attention to refine hand pose specifically, while MHR tokens and a camera token parameterize the output mesh and camera. Training supervision comes from a multi-stage annotation pipeline combining manual 2D keypoint annotation, differentiable optimization, multi-view geometry, and dense keypoint detection, curated by a VLM-based data engine that searches tens of millions of images to prioritize high-value (rare-pose, rare-viewpoint, occluded) samples for annotation rather than annotating randomly. Evaluation spans five standard HMR benchmarks plus five newly introduced challenging benchmarks (EE4D-Phy, EE4D-Proc, Harmony4D, Goliath Synthetic, and SA1B-Hard, the last organized by explicit failure categories: depth-ambiguous, orientation-ambiguous, scale-ambiguous poses and narrow/medium field-of-view). Metrics are MPJPE, PA-MPJPE, PVE (all mm) and PCK@0.05; on hands specifically it is compared against LookMa, METRO, HaMeR, MaskHand, and WiLoR using MPJPE and F-scores. A separate large-scale human preference study (7,800 participants, six pairwise comparisons against HMR2.0b, CameraHMR, PromptHMR, SMPLerX-H, and others) evaluates perceptual quality directly, since geometric metrics do not always track human judgments of plausibility.

**Findings**

3DB's advantage is concentrated in exactly the conditions its data engine was designed to target: on the SA-1B-Hard categorical breakdown, its lead over CameraHMR and PromptHMR is largest on depth-ambiguous, orientation-ambiguous, and scale-ambiguous poses and on narrow field-of-view images — i.e., the cases where naive random-sampling training data is thin. On hands, 3DB is close to but does not fully surpass the dedicated hand-only specialist WiLoR, reflecting a familiar full-body-vs-specialist tradeoff (similar in spirit to the wrist/finger tradeoff addressed by Hand4Whole++, read the same night), but it does so while also producing full-body context that hand-only methods cannot.

**Results**

On standard in-the-wild benchmarks 3DB variants (3DB-H, 3DB-DINOv3) outperform HMR2.0b, CameraHMR, PromptHMR, and SMPLerX-H on MPJPE/PVE/PA-MPJPE. On the new challenging benchmarks (EE4D-Phy/Proc, Harmony4D, Goliath Synthetic, SA1B-Hard), 3DB again reports lower MPJPE/PVE than CameraHMR, PromptHMR, and NLF, with categorical analysis showing the largest margins under depth/orientation/scale-ambiguous conditions (e.g., on the aux:depth_ambiguous slice, PVE/MPJPE/PA-MPJPE for CameraHMR are 126.25/102.25/81.33 and for PromptHMR 109.58/91.77/69.24, versus markedly lower values for 3DB). On hand-only evaluation, 3DB-H reaches 6.3 MPJPE / 5.5 PVE, close to WiLoR's 5.1/5.5 and ahead of METRO (6.3/6.5) and LookMa (8.1/8.6). The human preference study reports win rates (out of 80 pairwise comparisons per baseline) favoring 3DB across all six baselines tested. Exact overall win-rate percentages and the full first-benchmark table were not fully extractable in this session from the parsed PDF text; the qualitative direction (3DB wins on generalization and is preferred by raters) is well supported by the text obtained.

**Conclusion**

The authors conclude that a VLM-driven, actively-mined data engine paired with a promptable, jointly-trained body+hand decoder architecture is what closes the generalization gap on rare poses and hard viewpoints, more so than architectural novelty alone. For the two-layer thesis this paper mainly strengthens Layer 2/RQ3: its explicit categorical benchmark design (depth/orientation/scale ambiguity, FOV variation) is essentially a template for the cross-dataset/domain-generalization stress-testing the contact-aware pipeline will need, and its open-source status (3DB and MHR both released) makes it a plausible pose-estimation backbone or comparison baseline. It carries no direct cognitive-load or driver-monitoring framing, so its DENSO-positioning value is indirect, via generalization methodology rather than the application bridge itself.

*Sources: [AI at Meta publication page](https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/), [full paper PDF](https://holographica.space/wp-content/uploads/2025/11/584770213_869757652066297_8126547710241554369_n.pdf), [Hugging Face model card](https://huggingface.co/facebook/sam-3d-body-dinov3), [arXiv:2602.15989](https://arxiv.org/abs/2602.15989)*

