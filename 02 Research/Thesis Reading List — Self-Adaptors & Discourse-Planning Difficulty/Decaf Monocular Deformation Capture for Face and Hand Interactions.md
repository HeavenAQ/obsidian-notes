---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2309.16670
Year: 2023
Should Refer:
  - Architecture
Reading Status: Skim-Skip
Venue: SIGGRAPH-Asia
Relatedness: Weakly
Topic: Self-Contact-Hand-Face-CV
snippet: |2-
    • the predecessor of DICE 
    • capture hand-face interactions and deformation with a single RGB video
    • can be replaced by DICE
Authors: Soshi Shimada, Vladislav Golyanik, Patrick Perez, Christian Theobalt
tags:
  - Contact-Aware
  - Hand-Face-CV
Tier: Essential
Assigned Date: 2026-07-04
---
**Background.**  independent hand/face tracking fails under contact and heavy occlusion. 

**Research question.**  can hand-face interaction be captured jointly with contact-induced deformation from monocular video? 

**Methodology.**  deformation capture modeling skin deformation and physically plausible contact. 

**Conclusion.**  plausible contact reconstruction under occlusion -- the language and method template for a genuinely contact-aware (not proximity-based) visual branch.

## Reading Summary

**Abstract**

Decaf is the first monocular RGB-video motion-capture method that jointly regresses 3D hand and face motion together with the non-rigid face deformations caused by their interaction. Instead of tracking hand and face independently — which yields depth ambiguity, interpenetration, and missing deformation — it treats the hand as an articulated object that actively deforms the face, and it reconstructs contact regions and skin deformation from a single camera. It was published in ACM TOG (SIGGRAPH Asia 2023) by Shimada, Golyanik, Pérez, and Theobalt (MPI Informatics / [Valeo.ai](http://valeo.ai/)).

**Research Question**

Can hand-face interactions — including physically plausible contact and contact-induced non-rigid skin deformation — be captured in 3D from single monocular RGB video, a setting where naive independent hand/face tracking produces collisions and implausible depth?

**Methodology**

The authors first build a new hand-face motion and interaction capture dataset acquired with a markerless multi-view camera system. Raw multi-view reconstructions are refined with position-based dynamics (PBD) plus a non-uniform stiffness estimation of head tissues, yielding plausible per-frame annotations of surface deformations, hand-face contact regions, and head-hand positions (released publicly, ~8 GB). The tracking method itself is neural: a variational auto-encoder supplies a hand-face depth prior, and dedicated modules estimate per-vertex contact probabilities and deformations, which then guide a fitting/tracking stage that produces the final interacting hand-and-face reconstruction. Evaluation compares against several monocular hand/face reconstruction baselines applicable in this setting, using both quantitative plausibility/accuracy measures and qualitative comparisons, including in-the-wild videos (chin rest, cheek poke, face-touch while resting).

**Findings**

Jointly modeling contact and deformation resolves the classic failure modes of independent tracking: reconstructions show correct relative hand-face depth, no interpenetration, and realistic skin displacement at the contact site. The contact-estimation module is the key ingredient — it converts "hand near face" into an explicit surface-level contact representation, which is exactly the conceptual move needed for genuinely contact-aware (rather than proximity-heuristic) self-adaptor detection.

**Results**

The paper reports that Decaf's reconstructions are quantitatively and qualitatively more plausible than all applicable baselines on its new dataset (plausibility metrics such as collision/penetration measures and deformation accuracy alongside standard reconstruction errors). Exact table values were not accessible in this pass (the full PDF tables weren't retrievable), so treat the specific numbers as to-be-checked against the paper PDF; the headline claim is consistent superiority over independent-tracking baselines on both accuracy and physical-plausibility measures.

**Conclusion**

The authors conclude that monocular capture of interacting hands and faces with contact and deformation is feasible and markedly more realistic than treating the two independently, and they release the first dataset with deformation/contact annotations for this problem. For the thesis this is a core RQ2 (methods-layer) paper: it supplies the vocabulary and architecture template for the contact-aware visual branch of the self-adaptor detection pipeline, and — as a hand-face-contact capture method with automotive-adjacent authorship ([Valeo.ai](http://valeo.ai/)) — it also reinforces the DENSO-facing framing of contact-aware human-state sensing.

*Sources: *[*https://arxiv.org/abs/2309.16670*](https://arxiv.org/abs/2309.16670)*, *[*https://vcai.mpi-inf.mpg.de/projects/Decaf/*](https://vcai.mpi-inf.mpg.de/projects/Decaf/)