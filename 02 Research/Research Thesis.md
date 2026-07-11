---
base: "[[Research Hub.base]]"
---
# Master's Thesis Plan (v2 — 2026-07-03)

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