---
base: "[[Research Hub.base]]"
---
# Master's Thesis Plan (v3 — 2026-07-12)

## Working Title

**Self-Adaptors as Markers of Discourse-Planning Difficulty: A Contact-Aware Multimodal Analysis**

- Dataset:  https://github.com/facebookresearch/seamless_interaction 

*(Cross-domain multimodal modeling is the method, not the headline — behavioral question first for EDICS, ML rigor inside.)*

## Strategy: Two-Layer Design

Layer 1 satisfies the EDICS / Human Sciences degree requirement; Layer 2 builds the ML-researcher evidence for PhD applications (DENSO IT Lab, Institute of Science Tokyo).

| Layer | Contribution | Audience |
| --- | --- | --- |
| 1. Human-sciences core | Hypothesis-driven behavioral finding: do non-instrumental self-adaptors signal discourse-planning difficulty in conversation? Statistical analysis grounded in Ekman & Friesen (1969) and Sekine & Hotta (2025). | EDICS thesis committee / advisor (Prof. Sekine — multimodal communication) |
| 2. ML methodology | Contact-aware self-adaptor detection pipeline (pose → hand-face contact → fidgeting dynamics), multimodal fusion (video + audio + transcript), cross-dataset generalization benchmark. Framed in-thesis as "generalizability across recording conditions." | PhD admissions (DENSO IT Lab); ML/multimodal venue reviewers |

## CVPR-Level Carve-Out: Submit the CV Component as Its Own Paper

> [!info] Detailed execution document
> The deep literature review, benchmark specification, proposed method, reading order, baseline/ablation matrix, risk register, and week-by-week November submission plan are maintained in [[CVPR 2027 Submission Roadmap — Temporal Self-Contact]].

**Goal:** carve out one thesis component as a computer-vision paper, without forcing the entire human-sciences thesis to become a CVPR paper. The paper should be framed around **temporal self-contact / self-adaptor detection in monocular conversational video**, not around psycholinguistic theory testing.

**Provisional paper title:** *Temporal Self-Contact Detection in Conversational Video*.

### Core CVPR framing

The CV paper should make the behavioral thesis a downstream motivation, while the main contribution is CV-native:

1. **New task + benchmark:** temporal self-contact / communicative self-adaptor event detection in natural conversation.
2. **Method contribution:** an occlusion-aware temporal contact model that predicts self-contact/contact-region events over time, using mesh/contact priors rather than simple 2D proximity.
3. **Scale hook:** weak or pseudo-label supervision from SMPL-H/SMPL-X-style fits plus a smaller manually annotated gold set.

This lets the thesis split cleanly:

| Output | Focus | Best venue logic |
| --- | --- | --- |
| CV carve-out paper | Benchmark + method for temporal self-contact detection | CVPR / ICCV; WACV as pragmatic safety |
| Thesis / behavior paper | Self-adaptors as markers of discourse-planning difficulty | ICMI / IEEE FG / ACII / psycholinguistics venue |

### What to focus on for a CVPR-level submission

| Component | Must-have for top CV venue | Concrete thesis action |
| --- | --- | --- |
| Task definition | Precise event labels: onset/offset, contact body-region pair, occlusion flag, confidence | Freeze annotation schema before broad experiments |
| Dataset/benchmark | Subject-independent splits, inter-annotator agreement, released labels/splits if license allows | Annotate a gold evaluation set; release annotation layer rather than raw video if dataset license restricts redistribution |
| Method | More than a wrapper around TUCH/DICE/SMPLer-X: temporal contact field + occlusion handling + contact prior | Build one clean model contribution and ablate every term |
| Metrics | Temporal event mAP@tIoU, segment F1, contact precision/recall, contact-IoU; pose metrics only if pose is evaluated | Report both event-level and contact-level metrics |
| Baselines | 2D proximity/keypoint heuristics, fidgeting/optical-flow baseline, DICE/TUCH lifted per-frame, skeleton/video backbones | Implement simple baselines early to prove the gap |
| Ablations | No temporal module, no geodesic/contact prior, no occlusion flag, 2D vs 3D representation, different body-region pairs | Make ablations the evidence that the method is not merely application engineering |
| Generalization | Cross-subject first; optionally cross-corpus if time permits | Avoid participant leakage at all costs |

Related reading anchors: [[On Self-Contact and Human Pose (TUCH)]], [[DICE End-to-end Deformation Capture of Hand-Face Interactions from a Single Image]], [[Decaf Monocular Deformation Capture for Face and Hand Interactions]], [[Bringing Inputs to Shared Domains for 3D Interacting Hands Recovery in the Wild (InterWild)]].

### Submission calendar and decision gates

> Current planning date: **2026-07-12**. Dates marked **estimated** must be rechecked on the official site before committing.

| Venue | Status for this project | Main dates/deadlines | Use in strategy |
| --- | --- | --- | --- |
| **WACV 2027** | Realistic early/safety target | Official Round 2 registration: **2026-08-21 AoE**; paper: **2026-08-28 AoE**; supplement: **2026-08-30 AoE**; conference: **2027-01-04–08** | Only submit if a minimal benchmark + baselines are already solid; otherwise use as internal sprint deadline |
| **CVPR 2027** | Main stretch target | Official conference dates: **2027-06-20–24**, Seattle. Paper deadline is not yet confirmed here; plan around the usual **mid-November 2026** cycle and verify once the CVPR 2027 dates page is live. CVPR 2026 used abstract **2025-11-07 AoE**, paper **2025-11-13 AoE**, supplement **2025-11-20 AoE**. | Submit only if benchmark, baselines, main method, and ablations are complete by late October 2026 |
| **ICCV 2027** | Same-tier fallback with more runway | Official conference dates: **2027-10-02–08**, Hong Kong. Paper deadline not yet confirmed; based on ICCV 2025, expect early March 2027 and verify once official dates post. | Best target if the CVPR version would be premature; use extra months for annotation, ablations, and stronger baselines |
| **ECCV 2026 / 2028** | Not the near-term target | ECCV 2026 paper deadline already passed: registration **2026-02-26**, paper **2026-03-05**, supplement **2026-03-12**. Next ECCV is expected in 2028, dates not yet useful for this thesis timeline. | Treat as post-thesis/top-tier future target, not the current 9-month plan |

Sources to recheck before submission planning: [WACV 2027 dates](https://wacv.thecvf.com/Conferences/2027/Dates), [CVF upcoming conferences](https://www.thecvf.com/), [CVPR 2026 dates as deadline-cycle reference](https://cvpr.thecvf.com/Conferences/2026/Dates), [ICCV 2025 dates as deadline-cycle reference](https://iccv.thecvf.com/Conferences/2025/Dates), [ECCV 2026 dates](https://eccv.ecva.net/Conferences/2026/Dates).

### Minimum viable CVPR paper checklist

- [ ] By **2026-07-31**: freeze task definition, labels, metrics, and annotation guide.
- [ ] By **2026-08-21**: complete a small gold eval set + inter-annotator agreement estimate; decide whether WACV Round 2 is realistic.
- [ ] By **2026-09-15**: finish baseline suite: 2D proximity, optical-flow/fidgeting, DICE/TUCH per-frame lift, skeleton/video backbone.
- [ ] By **2026-10-15**: finish main temporal contact model and core ablations.
- [ ] By **2026-10-31**: CVPR go/no-go. If contact-IoU/event-mAP gains are not strong and ablations are incomplete, skip CVPR and target ICCV 2027.
- [ ] By **2026-11-10**: paper draft, figures, benchmark table, qualitative videos, and supplement ready if going for CVPR.

**Go/no-go rule:** do not submit to CVPR just because the story is interesting. Submit only if the CV contribution can be stated as: *we introduce the first temporal self-contact benchmark for conversational video and a temporal occlusion-aware contact model that substantially outperforms 2D, single-frame, and generic video baselines under subject-independent evaluation.*

### Venue timing — why ICCV/ECCV are the realistic top-tier targets (not a downgrade)

> [!important] The lever is **timing, not tier.** CVPR, ICCV and ECCV are the **same tier** (~25–30% acceptance, equal prestige). ICCV/ECCV are **not easier** — the only difference for *this* project is *when* the deadline lands.

- CVPR's paper deadline is **mid-November**; ICCV and ECCV deadlines fall in **early spring (~Feb–Mar)** — several months later. Because they alternate years, in almost any window one of them has a deadline well after the nearest CVPR.
- A **benchmark + method** paper is the most time-sensitive kind: annotation, inter-annotator agreement, baselines, and ablations all consume calendar. A November deadline forces a rushed, incomplete submission; a March deadline lets me submit a *mature* version of the same idea. **Same bar, better-prepared paper → higher odds.**
- The staggered cadence (CVPR Nov → ICCV/ECCV Mar → CVPR Nov) is a **built-in resubmission ladder at equal tier**, not a downgrade. WACV is the one genuinely notch-easier venue, used only as a safety net / internal sprint.
- **Recommended target order for this project:** primary top-tier = **ICCV 2027** (~Mar 2027 deadline, ~4 extra months); **WACV 2027** (~Aug–Sep 2026) as an optional early sprint *only if* a minimal benchmark + baselines are already solid; **CVPR 2027** (~Nov 2026) only if unexpectedly ahead of schedule.

#### Verified deadlines (checked 2026-07-12 — re-confirm on official site before committing)

| Venue | Conference | Paper deadline | Confidence | Role for this project |
| --- | --- | --- | --- | --- |
| **WACV 2027** | Jan 4–8, 2027 · Disney Springs | Round 2 enrollment **Aug 21, 2026**; paper **~Aug 28–Sep 1, 2026** | Confirmed (site live) | Safety / sprint. Submit only if a minimal benchmark + baselines already work |
| **CVPR 2027** | Jun 19–26, 2027 · Seattle | Not yet official. Expect abstract **~mid-Nov 2026**, paper ~1 week later (CVPR 2026: abstract 11/07, paper 11/13) | Estimated | Stretch. Only if benchmark + method + ablations complete by **late Oct 2026** |
| **ICCV 2027** | Oct 2027 · Hong Kong | Not yet official. Expect **~early Mar 2027** (ICCV 2025: Mar 7; 2023: Mar 8) | Estimated | **Most realistic top-tier.** Use the extra months for annotation, ablations, stronger baselines |
| **ECCV 2028** | 2028 · TBA | ~Mar 2028 (biennial; ECCV 2026 was Mar 5, 2026 — already passed) | Estimated | Post-thesis / future target, not the 9-month plan |

#### What to focus on for each venue

- **All three top-tier:** the *content* is identical — the benchmark + occlusion-aware temporal contact model + ablations (see the focus table above). Only the **polish level** differs.
- **WACV (Aug/Sep 2026):** lead with the **benchmark + strong baselines**; a solid single method contribution is enough (more application-friendly reviewing). Good forcing function to finish the gold eval set.
- **CVPR (Nov 2026):** the method must **clearly beat baselines with complete ablations** by the deadline — no partial ablations. Highest polish, least runway.
- **ICCV (Mar 2027):** spend the extra ~4 months to (1) **expand the gold annotation set + report inter-annotator agreement**, (2) add **cross-corpus generalization** (RQ3), (3) add temporal-HMR baselines (DanceHMR, PMCE) and **qualitative video** results. This is where the paper becomes genuinely competitive.

Sources (re-check before committing): [WACV 2027 dates](https://wacv.thecvf.com/Conferences/2027/Dates), [CVPR 2027 (dates page)](https://cvpr.thecvf.com/), [CVPR 2026 deadline reference](https://cvpr.thecvf.com/Conferences/2026/Dates), [ICCV 2027 (CVF)](https://www.thecvf.com/), [ICCV 2025 deadline reference](https://iccv.thecvf.com/Conferences/2025/Dates). Full execution plan: [[CVPR 2027 Submission Roadmap — Temporal Self-Contact]].

## Research Questions

1. **RQ1 (behavioral):** Are self-adaptor occurrence, duration, and kinematics elevated around loci of discourse-planning difficulty (disfluencies, repairs, delayed turn starts)?
2. **RQ2 (methodological):** Can a contact-aware multimodal model detect self-adaptors and planning difficulty reliably from video + audio + transcript?
3. **RQ3 (robustness):** Do detection and the behavioral association generalize across corpora and recording conditions?

## Theoretical Grounding

- Self-adaptor definition & taxonomy: Ekman & Friesen (1969)
- Self-adaptors and lexical retrieval / cognitive load: Sekine & Hotta (2025); Mueller & Grunwald self-touch studies
- Turn-taking & repair as difficulty loci: Sacks, Schegloff & Jefferson (1974)

Detailed theory notes: [[The Repertoire of Nonverbal Behavior Categories, Origins, Usage, and Coding]]

## Method Pipeline

4. **Contact detection:** pose/segmentation-based hand-face contact (upgrade Lin et al. 2020's OpenPose bbox with SAM-style segmentation; contact-aware priors from Decaf/DICE/TUCH)
5. **Fidgeting dynamics:** optical-flow trajectories → spectral/cyclic features
6. **Difficulty labels:** disfluency & repair annotation from transcripts + audio (Romana et al. line of work)
7. **Fusion:** video + audio + transcript; ablate modality contributions
8. **Statistical analysis (Layer 1):** mixed-effects models of self-adaptor measures vs. difficulty loci
9. **Generalization (Layer 2):** DomainBed-style cross-corpus evaluation

## Datasets (prioritized)

Primary 2–3 corpora keep the behavioral analysis tractable in a master's timeline; the rest are optional transfer targets.

| Priority | Dataset | Role | Access |
| --- | --- | --- | --- |
| 1 | NoXi | Main discourse-alignment corpus (video, audio, turn-taking annotations) | Academic request |
| 2 | AMI | Meeting-style discourse & repair analysis (transcripts, dialog acts) | Public (CC BY 4.0) |
| 3 | One transfer target: UDIVA or CEJC | Cross-domain generalization test | Academic request / public with conditions |
| Optional | Face-Touching-Behavior | Visual pretraining for contact detection | Public |
| Optional | FluencyBank Timestamped | Disfluency supervision/evaluation | Consortium |
| Optional | DAIC-WOZ | Auxiliary distress-domain transfer | Academic request |

## Milestones & Deliverables

- [ ] Discuss two-layer plan and ML-heavy methods with Prof. Sekine (aim: co-authored methods paper → stronger recommendation letter)
- [ ] Confirm EDICS thesis format requirements in admissions guideline/brochure
- [ ] Secure dataset access (NoXi, transfer target); confirm ethics/usage approvals
- [ ] Build contact-aware detection pipeline; internal validation
- [ ] Run behavioral analysis (RQ1) on primary corpus
- [ ] Cross-corpus benchmark (RQ3)
- [ ] **First-author submission to ICMI / ACII / Interspeech-ICASSP workshop before PhD applications** — single biggest odds-booster
- [ ] Release open-source code + reproducible benchmark
- [ ] Thesis writing: behavioral framing first, ML methods inside, driver cognitive-load application in Future Work only

## PhD Positioning — ML Researcher Track (DENSO IT Lab, Institute of Science Tokyo)

Target: DENSO IT Lab Recognition, Control & Learning Algorithm collaborative research chair ([lab site](https://d-itlab.comp.isct.ac.jp/)).

**Outreach pitch:** contact-aware multimodal estimation of cognitive load from nonverbal behavior — discourse-planning difficulty as the thesis instantiation, driver cognitive-load monitoring as the DENSO-relevant application (their MIRU 2024 driver-workload dataset).

| Thesis pillar | Lab's related work | Connection |
| --- | --- | --- |
| Contact-aware self-adaptor detection (Decaf, DICE, TUCH) | DF-Mamba: 3D hand pose estimation in interactions (WACV 2026) | Self-adaptor detection as downstream of interaction-aware hand pose |
| Action-recognition backbones (ST-GCN, PoseC3D, VideoMAE) | Human motion & action recognition (Cervantes et al.); egocentric activity recognition (EvIs-Kitchen) | Shared backbone layer for skeleton/video behavior modeling |
| Cross-domain benchmarking (DomainBed, SimMMDG, RNA-Net) | OOD detection (AAAI 2025); optimizer selection for OOD; robust MoE routing | Domain generalization / OOD is a core lab theme |
| Multimodal fusion (audio + video + transcript) | Audio-visual event recognition; speech + inertial sensing | Same fusion methodology, different signals |
| Self-adaptors as cognitive-load markers | Driver workload prediction dataset (MIRU 2024) | Strongest pitch — direct value to sponsoring company |

**Likely contacts:** Rei Kawakami or Ikuro Sato (human-centric CV/ML); Nakamasa Inoue (multimodal learning).

**Action items:**

- [ ] Draft outreach email to faculty drawing the bridges above, before formal application
- [ ] Check admission route (Institute of Science Tokyo, Dept. of Computer Science) and deadlines
