---
base: "[[Document Hub.base]]"
Status: In-Progress
Category:
  - Research
  - Computer Vision
  - Thesis
tags:
  - CVPR
  - Contact-Aware
  - Hand-Face-CV
  - Temporal-Action-Localization
  - Dataset
Created time: 2026-07-12
Last updated time: 2026-07-12
Target Venue: CVPR 2027
Target Submission Window: November 2026
---

# CVPR 2027 Submission Roadmap — Temporal Self-Contact

> [!info] Local literature library
> All roadmap papers and their PDFs are managed in [[Local Paper Library.base]]. See [[Local Paper Workflow]] for the Zotero, Better BibTeX, Zotero Integration, and PDF++ workflow.

> [!important] One-sentence objective
> Submit a **computer-vision paper carved out of the thesis**, centered on a new benchmark and method for **temporally localizing physical self-contact in monocular conversational video**. The psycholinguistic self-adaptor/disfluency analysis is a downstream demonstration, not the CVPR paper's main claim.

## Executive decision

### Recommended paper

**Working title:** *Contact in Conversation: Temporal Self-Contact Localization in Monocular Video*

**Core claim:** existing work covers static self-contact pose recovery, hand–face reconstruction, and generic social gestures, but does not provide a benchmark and contact-specific temporal model for localizing naturally occurring self-contact events in long conversational video.

**Three contributions, in priority order:**

1. **Benchmark:** a subject-independent temporal self-contact benchmark built as an annotation layer over Seamless Interaction, with onset/offset, hand, target body region, occlusion/uncertainty, hard negatives, and fixed splits.
2. **Method:** a lightweight **Temporal Contact Graph Network** that combines visual evidence with noisy geometric contact cues and predicts event boundaries and contacted body regions.
3. **Scale:** confidence-weighted pseudo-label learning from the supplied SMPL-H sequences, followed by fine-tuning on a manually annotated gold set.

### Why the wording matters

- Do **not** claim the first self-contact dataset. TUCH introduced 3DCP/MTP/DSC; Goliath-SC contains 383K shape-registered self-contact poses; BioTUCH contains synchronized contact/video/mocap data.
- The defensible novelty is the intersection: **natural conversation + monocular video + temporal event boundaries + body-region labels + subject-independent benchmark**.
- Do **not** label an event “self-adaptor” merely because physical contact occurred. The CV benchmark should label observable contact. Communicative function belongs to the thesis downstream study.

## Current landscape and the exact research gap

| Literature branch | What already exists | Remaining gap for this paper |
| --- | --- | --- |
| Static full-body self-contact | [[On Self-Contact and Human Pose (TUCH)]] introduces contact-aware pose data and fitting; [[Generative Modeling of Shape-Dependent Self-Contact Human Poses]] introduces Goliath-SC and shape-conditioned priors | Not temporal event detection in natural conversation |
| Contact-timed pose refinement | [[Contact-Aware Refinement of Human Pose Pseudo-Ground Truth via Bioimpedance Sensing]] uses measured skin contact to refine pose | Requires wearable sensing; only a small proof-of-concept capture, not vision-only conversational localization |
| Hand–face reconstruction | [[Decaf Monocular Deformation Capture for Face and Hand Interactions]], [[DICE End-to-end Deformation Capture of Hand-Face Interactions from a Single Image]], and [[Capturing Head Avatar with Hand Contacts from a Monocular Video]] reconstruct contact/deformation | Reconstruction is not long-video event localization; DICE is single-image and Decaf is optimized for hand–face capture |
| Whole-body/hand HMR | [[SMPLer-X Scaling Up Expressive Human Pose and Shape Estimation]], [[SAM 3D Body Robust Full-Body Human Mesh Recovery]], [[Enhancing Hands in 3D Whole-Body Pose Estimation with Conditional Hands Modulator]] | Strong initialization, but no explicit temporal self-contact event head |
| Occlusion-aware HMR | [[DPMesh Exploiting Diffusion Prior for Occluded Human Mesh Recovery]], [[OnlineHMR Video-based Online World-Grounded Human Mesh Recovery]] | Handles pose stability/occlusion, not self-contact semantics and boundaries |
| Social gesture benchmarks | [[SocialGesture Delving into Multi-person Gesture Understanding]] demonstrates that a natural social-gesture dataset plus temporal localization baselines is CVPR-relevant | Its deictic gestures differ from subtle self-contact and self-occlusion |
| Generic temporal localization | [[ActionFormer Localizing Moments of Actions with Transformers]], [[TriDet Temporal Action Detection with Relative Boundary Modeling]], [[End-to-End Spatio-Temporal Action Localisation with Video Transformers]] | Generic RGB features do not explicitly represent hand-to-body contact geometry |

## Research question and hypotheses

**RQ-CV:** Can explicit contact geometry and temporal consistency improve the localization of subtle self-contact events over generic RGB, 2D-proximity, and framewise 3D-contact baselines?

| ID | Testable hypothesis | Evidence required |
| --- | --- | --- |
| H1 | Explicit hand-to-body contact graph features improve event localization over RGB-only video features | Higher event mAP and segment F1 on the same subject-independent split |
| H2 | Temporal modeling reduces false positives from momentary hand-near-body poses and reduces fragmented predictions | Better boundary error, fewer fragments/event, and stronger F1@50 |
| H3 | Confidence-weighted geometric pseudo-labels improve low-label performance | Label-fraction curve at 10%, 25%, 50%, 100% gold data |
| H4 | Occlusion-aware reliability gating improves performance when hands are partially hidden | Separate results on visible vs. occluded/uncertain subsets |
| H5 | The method transfers across conversational conditions | Train improvised → test naturalistic, and vendor/site holdout |

## Benchmark specification

### Primary data source

Use **Seamless Interaction** because it supplies 4,000+ hours from 4,000+ participants, HD video, 30 Hz SMPL-H, body/face keypoints, aligned speech, naturalistic/improvised conditions, and participant/vendor metadata. It is CC-BY-NC 4.0. Release **annotations, IDs, timestamps, splits, and evaluation code**, not a duplicated 27 TB video corpus.

### Label schema — freeze this before scaling annotation

| Field | Values |
| --- | --- |
| `start`, `end` | Frame/time boundaries of physical contact |
| `hand` | left / right / both / uncertain |
| `target_region` | face-head / neck / opposite-hand / arm / torso / leg / other |
| `motion` | static-rest / rub / scratch / tap / repeated-fidget / uncertain |
| `visibility` | visible / partly-occluded / fully-occluded-but-inferable |
| `confidence` | certain / probable / ambiguous |
| `contact` | contact / hard-negative-near-contact |

**Hard negatives must be deliberately sampled:** hand near face without touch, ordinary co-speech gestures crossing the face, phone/object interaction, folded arms without hand–skin contact, pose-estimation failure, and out-of-frame hands.

### Annotation target

- **Pilot:** 100 candidate clips from at least 25 participants.
- **Benchmark v0.1:** at least 500 verified positive events plus matched hard negatives.
- **Submission target:** 1,500–2,500 verified events across at least 200 participants and multiple vendors, with 15–20% double-annotated.
- Measure Cohen's kappa for categorical labels and boundary agreement (mean/median absolute onset/offset difference or temporal IoU) for segments.
- Use active candidate mining rather than watching random footage: over-sample low hand-to-body mesh distance, 2D hand/body overlap, low hand velocity near the body, and existing occlusion flags.

### Fixed evaluation splits

1. **Primary:** participant-disjoint train/dev/test.
2. **Site robustness:** hold out one or more vendors.
3. **Condition shift:** improvised → naturalistic and naturalistic → improvised.
4. **Difficulty slices:** visible vs. occluded; hand–face vs. other regions; short vs. long events; static vs. dynamic contact.

No frame, clip, session, or conversation partner from a test participant may leak into training.

## Proposed method: Temporal Contact Graph Network

### Representation

At each timestep, create nodes for the left hand, right hand, and coarse body regions. Candidate edges encode:

- minimum hand-to-region mesh distance;
- 2D keypoint/region distance and overlap;
- relative velocity and approach/separation direction;
- hand/body visibility and SMPL-H validity;
- optional local visual tokens from hand and target-region crops.

### Model

1. **Visual encoder:** start frozen with VideoMAE V2 or a comparable public video backbone. Do not train a large video model from scratch.
2. **Geometry encoder:** small MLP/graph-attention block over contact candidates.
3. **Reliability gate:** down-weight invalid SMPL-H, occluded keypoints, and inconsistent framewise geometry rather than treating pseudo-3D as ground truth.
4. **Temporal neck:** multi-scale local attention or 1D temporal pyramid inspired by ActionFormer/TriDet.
5. **Heads:** framewise contact state, target-region class, and onset/offset boundary regression.
6. **Training:** supervised gold loss + confidence-weighted pseudo-label loss + temporal smoothness/boundary loss.

### Keep the novelty disciplined

The method contribution is **contact-specific structured temporal reasoning under noisy geometry**. Avoid adding an LLM, diffusion model, multimodal speech fusion, causal model, or full mesh-reconstruction objective unless the core benchmark and method are already complete.

## Required baselines

| Priority | Baseline | Purpose |
| --- | --- | --- |
| P0 | Majority/random and duration prior | Sanity check |
| P0 | 2D wrist-to-face/body keypoint threshold + temporal hysteresis | Proves whether the task needs more than proximity |
| P0 | SMPL-H Euclidean-distance threshold | Measures the value and noise of supplied geometry |
| P0 | RGB VideoMAE V2 + linear/MLP frame head | Strong generic visual baseline |
| P0 | ActionFormer or TriDet on frozen video features | Strong temporal localization baseline |
| P1 | DICE framewise hand–face contact, where applicable | Contact-specialist baseline for hand–face subset |
| P1 | TUCH/contact fitting or BioTUCH-style contact optimization without wearable timing | 3D contact-aware comparison where reproducible |
| P1 | RGB + geometry early concatenation | Shows the proposed graph/gating is not ordinary feature fusion |
| P2 | SAM 3D Body or Hand4Whole++ replacing supplied SMPL-H on a subset | Tests dependency on Meta-provided geometry |

## Metrics and reporting

- **Primary:** event mAP at temporal IoU thresholds 0.1:0.1:0.7 and average mAP.
- **Segmentation:** F1@10, F1@25, F1@50; edit score if reporting dense segmentation.
- **Framewise:** macro-F1 and AUPRC; accuracy alone is misleading under class imbalance.
- **Boundary quality:** median onset and offset absolute error in seconds.
- **Attributes:** macro-F1 for contacted body region and hand.
- **Robustness:** report all metrics by visibility, region, duration, condition, and vendor holdout.
- **Efficiency:** parameters, FLOPs or throughput, and preprocessing cost.
- Report mean ± standard deviation over at least three seeds for the main table.

## Mandatory ablations

1. RGB only vs. geometry only vs. naïve concatenation vs. full model.
2. Remove temporal neck / use framewise predictions.
3. Remove reliability gate.
4. Remove pseudo-label pretraining.
5. Euclidean-only vs. Euclidean + non-adjacent/geodesic contact prior.
6. No local hand/target-region crops.
7. No hard-negative mining.
8. Gold label fractions: 10%, 25%, 50%, 100%.
9. Different temporal windows and sampling rates.

## Reading curriculum

### Sprint A — read first (July 12–26)

| Order | Paper | What must be extracted |
| --- | --- | --- |
| 1 | [[On Self-Contact and Human Pose (TUCH)]] | Contact definitions, non-adjacent surface constraints, DSC/MTP labels, evaluation weaknesses |
| 2 | [[Generative Modeling of Shape-Dependent Self-Contact Human Poses]] | What Goliath-SC already claims; avoid invalid “first dataset” language; shape-conditioned contact prior |
| 3 | [[Contact-Aware Refinement of Human Pose Pseudo-Ground Truth via Bioimpedance Sensing]] | Contact timing, adversarial near-contact gestures, BioTUCH loss, dataset limitations |
| 4 | [[DICE End-to-end Deformation Capture of Hand-Face Interactions from a Single Image]] | Per-vertex contact heads, plausibility metrics, weak supervision, full-body generalization limit |
| 5 | [[SocialGesture Delving into Multi-person Gesture Understanding]] | Annotation protocol, dataset paper structure, temporal localization metrics/baselines, release strategy |
| 6 | [[Seamless Interaction Dyadic Audiovisual Motion Modeling and Large-Scale Dataset]] | File structure, split units, licenses, SMPL-H validity, vendor/participant metadata, download cost |

### Sprint B — implementation foundations (July 27–August 16)

| Order | Paper | What must be implemented or borrowed |
| --- | --- | --- |
| 7 | [[ActionFormer Localizing Moments of Actions with Transformers]] | Reproducible temporal localization baseline and feature format |
| 8 | [[TriDet Temporal Action Detection with Relative Boundary Modeling]] | Boundary modeling for short ambiguous events |
| 9 | [[VideoMAE Masked Autoencoders Are Data-Efficient Learners for Self-Supervised Video Pre-Training]] and [VideoMAE V2](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_VideoMAE_V2_Scaling_Video_Masked_Autoencoders_With_Dual_Masking_CVPR_2023_paper.html) | Frozen feature extraction and compute planning |
| 10 | [[End-to-End Spatio-Temporal Action Localisation with Video Transformers]] | Sparse-vs-dense supervision and coherent tube/event prediction |
| 11 | [[SAM 3D Body Robust Full-Body Human Mesh Recovery]] | 2026 full-body baseline, prompts, and generalization claims |
| 12 | [[Enhancing Hands in 3D Whole-Body Pose Estimation with Conditional Hands Modulator]] | Why whole-body systems lose hand detail and how to modularly improve it |

### Sprint C — read as needed for ablations and paper writing (August–September)

- [[DPMesh Exploiting Diffusion Prior for Occluded Human Mesh Recovery]] — occlusion-aware HMR baseline.
- [[OnlineHMR Video-based Online World-Grounded Human Mesh Recovery]] — temporal consistency and online inference.
- [[Capturing Head Avatar with Hand Contacts from a Monocular Video]] — video contact/depth ordering.
- [[Decaf Monocular Deformation Capture for Face and Hand Interactions]] — sequence-level contact/deformation capture.
- [[Bringing Inputs to Shared Domains for 3D Interacting Hands Recovery in the Wild (InterWild)]] — hand occlusion and interaction recovery.
- [[Revisiting Skeleton-Based Action Recognition (PoseC3D)]] and [[Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition (ST-GCN)]] — skeleton-only baselines.

### How to read each paper

For every P0 paper, record only five things: task/claim, exact input-output, supervision/data, strongest relevant baseline, and one design choice or failure mode that changes this project. Reproduce tables/metrics only when they affect experimental design.

## Week-by-week execution plan

| Dates | Deliverable | Exit criterion |
| --- | --- | --- |
| Jul 12–19 | Scope freeze, dataset access test, 20-video qualitative audit, read TUCH/Goliath-SC/BioTUCH | Written gap statement survives advisor challenge; raw video + SMPL-H can be loaded |
| Jul 20–26 | Annotation guide v1, label tool, 100 candidate clips, read DICE/SocialGesture/Seamless | Two annotators can apply the schema without repeated clarification |
| Jul 27–Aug 2 | Candidate miner v0: 2D proximity + SMPL-H distance; participant-disjoint split generator | Candidate recall is high on the manually audited sample |
| Aug 3–9 | P0 heuristic baselines; 250–500 verified events; annotation QA dashboard | A reproducible evaluation script emits every primary metric |
| Aug 10–16 | Frozen VideoMAE baseline and ActionFormer/TriDet baseline | Generic temporal baseline runs end-to-end on fixed split |
| Aug 17–23 | Double annotation and IAA; benchmark v0.1; geometry-quality audit | IAA acceptable and error taxonomy documented |
| Aug 24–30 | Contact graph features and geometry-only model | Beats raw distance thresholds or exposes why geometry fails |
| Aug 31–Sep 6 | Full fusion model v0; reliability gate | Full model runs reproducibly with logged configs/seeds |
| Sep 7–13 | Pseudo-label mining at larger scale; label-fraction experiment | Weak supervision improves at least one low-label setting |
| Sep 14–20 | Boundary head, hard-negative mining, model v1 | Clear gain over generic temporal baseline on dev set |
| Sep 21–27 | Submission-scale gold set and frozen test set | No further test-label inspection; dataset card drafted |
| Sep 28–Oct 4 | Main experiments, 3 seeds, primary ablations | Main table and ablation table complete |
| Oct 5–11 | Vendor/condition/occlusion generalization | Robustness table complete; failures visualized |
| Oct 12–18 | DICE/TUCH/HMR specialist comparisons where feasible | Reviewer-obvious baselines covered or limitations explicitly justified |
| Oct 19–25 | Paper draft v1, figures, qualitative timeline visualizations | Full 8-page draft reviewed by advisor/collaborator |
| Oct 26–Nov 1 | Experiment freeze; code cleanup; release package and supplement draft | Every paper number generated from versioned result files |
| Nov 2–6 | Internal abstract/profile deadline; paper draft v2 | OpenReview profiles ready; title/abstract/authors frozen internally |
| Nov 7–12 | Final writing, anonymization, supplement, video | Submission-ready package one day before expected window |

> [!warning] Deadline status
> As of 2026-07-12, the official CVPR 2027 paper deadline was not posted on the official site located during this review. CVPR 2026 used **Nov 7, 2025 AoE** for abstract, **Nov 13** for paper, and **Nov 20** for supplement. Treat the dates above as conservative internal planning dates and replace them immediately when CVPR 2027 publishes its call.

## Go/no-go gates

### August 23 — data feasibility

Continue the CVPR plan only if candidate mining finds enough genuine events, the videos are publishable under the annotation-layer plan, the supplied geometry is usable at least for proposals, and annotation agreement is defensible.

### September 20 — method feasibility

Continue only if the contact-specific model clearly improves over 2D proximity and shows a consistent dev-set advantage over a strong RGB temporal baseline. If geometry is too noisy, narrow the claim to uncertainty-aware fusion or hand–face contact rather than pretending full-body geometry works.

### October 18 — CVPR readiness

Submit only if all are true:

- benchmark size and diversity are credible;
- test split is participant-disjoint and untouched;
- method beats heuristic, RGB-only, geometry-only, and generic temporal baselines;
- gains survive at least three seeds and at least two robustness slices;
- mandatory ablations and qualitative failure analysis are complete;
- code, split files, and annotation schema can be released.

Otherwise, preserve the work and target ICCV 2027 with a larger benchmark and stronger cross-dataset validation.

## Risk register and fallback decisions

| Risk | Early diagnostic | Fallback |
| --- | --- | --- |
| Too few natural self-contact events | Random 20-hour estimate and active-miner precision | Narrow to hand–face + hand–hand; mine more hours; include matched hard negatives |
| SMPL-H contact distances are unreliable | Compare distances with 100 manually labeled clips | Use geometry only for proposals; train RGB + 2D contact graph; test a stronger HMR on a subset |
| DICE fails on full conversational frames | Evaluate 100 hand–face cases | Keep as hand–face subset baseline, not a system component |
| Benchmark seems too small | Event/participant/vendor coverage by Sep 1 | Recruit annotators or delay to ICCV; do not inflate with correlated frames |
| Method looks like feature concatenation | Naïve fusion performs equally | Make reliability-gated structured edges the core; otherwise revise method before scaling experiments |
| Behavioral story dominates | Abstract mentions disfluency before benchmark/method | Move behavior to one downstream table and the final motivation paragraph |
| Compute bottleneck | Feature extraction throughput measured in July | Freeze encoders, precompute features, use short candidate windows, avoid end-to-end giant models |

## Immediate actions — next seven days

- [ ] Confirm advisor/collaborator agreement on the physical-contact-only CVPR task.
- [ ] Download 10 naturalistic and 10 improvised Seamless interaction pairs with video + SMPL-H + keypoints.
- [ ] Render body meshes/keypoints over video and manually inspect hand–face, hand–hand, hand–torso, and false near-contact cases.
- [ ] Write the annotation guide with at least 20 positive and 20 hard-negative examples.
- [ ] Read and extract gaps from Goliath-SC, BioTUCH, and SocialGesture.
- [ ] Implement a first wrist-to-face 2D threshold baseline and event-merging hysteresis.
- [ ] Create a repository with immutable split files, config-driven experiments, seed control, and result JSON export.
- [ ] Create/update all co-author OpenReview profiles well before November.

## Primary sources

- [CVPR 2026 call and prior-cycle deadlines](https://cvpr.thecvf.com/Conferences/2026/CallForPapers)
- [CVPR 2027 conference announcement](https://cvpr.thecvf.com/Conferences/2026/News/Closing)
- [Seamless Interaction dataset and license](https://github.com/facebookresearch/seamless_interaction)
- [TUCH — CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Muller_On_Self-Contact_and_Human_Pose_CVPR_2021_paper.html)
- [Goliath-SC — ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ohkawa_Generative_Modeling_of_Shape-Dependent_Self-Contact_Human_Poses_ICCV_2025_paper.html)
- [BioTUCH — ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Forte_Contact-Aware_Refinement_of_Human_Pose_Pseudo-Ground_Truth_via_Bioimpedance_Sensing_ICCV_2025_paper.html)
- [DICE — ICLR 2025](https://frank-zy-dou.github.io/projects/DICE/index.html)
- [SocialGesture — CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Cao_SocialGesture_Delving_into_Multi-person_Gesture_Understanding_CVPR_2025_paper.html)
- [SAM 3D Body — CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_SAM_3D_Body_Robust_Full-Body_Human_Mesh_Recovery_CVPR_2026_paper.html)
- [Hand4Whole++ — CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Moon_Enhancing_Hands_in_3D_Whole-Body_Pose_Estimation_with_Conditional_Hands_CVPR_2026_paper.html)
- [ActionFormer — ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640485.pdf)
- [TriDet — CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Shi_TriDet_Temporal_Action_Detection_With_Relative_Boundary_Modeling_CVPR_2023_paper.html)
- [End-to-End Spatio-Temporal Action Localisation — CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Gritsenko_End-to-End_Spatio-Temporal_Action_Localisation_with_Video_Transformers_CVPR_2024_paper.html)
