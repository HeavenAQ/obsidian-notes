---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2406.17988
Year: 2025
Should Refer:
  - Challenges
  - Architecture
Reading Status: Read
Venue: ICLR
Code Link: https://github.com/Qingxuan-Wu/DICE
Relatedness: strongly
Topic: Self-Contact / Hand-Face CV
snippet: |2-
    • hand-face interaction 
    • SOTA with better generalization
    • mostly profile images, unknown performance on full body
Authors: Qingxuan Wu, Zhiyang Dou, Sirui Xu, et al.
Tags:
  - Contact-Aware
  - Hand-Face CV
  - Weak Supervision
Tier: Essential
Assigned Date: 2026-07-03
---
## Notes

- challenges in hand-face deformation and interactions
    - self-occlusions involved in hand-face interaction
    - diversity of hand and face poses, contacts, and deformations
    - ambiguity in the single-view setting
- operates at an interactive rate (20 fps) on an Nvidia 4090 GPU, whereas Decaf requires more than 15 seconds for a single image

## Reading Summary

![[99 Assets/Media/image 23.png]]

**Abstract**

DICE (ICLR 2025) is the first end-to-end learning method that recovers hand-face interactions — hand pose, face pose, contact, and face deformation — from a single monocular RGB image. It replaces the slow optimization pipeline of Decaf (the only prior method) with a feed-forward Transformer, adds a weakly-supervised training scheme on in-the-wild images, and achieves state-of-the-art accuracy and physical plausibility at interactive speed (~20 fps vs. >15 s/image for Decaf).

**Research Question**

Can hand/face poses, hand-face contacts, and skin deformations be predicted jointly, end-to-end, from one image — and can such a model generalize beyond the studio-bound Decaf training distribution to in-the-wild imagery?

**Methodology**

![[99 Assets/Media/image 24.png]]

- *Problem Formulation*
    - Input image: $I \in \R^{224 \times 224 \times 3}$
    - vertices of the hand mesh: $V_H \in \R^{778 \times 3}$
    - vertices of the face mesh: $V_F \in \R^{5023 \times 3}$
    - vertices of the face deformation: $D \in \R^{5023 \times 3}$
    - per-vertex hand contact probabilities: $C_H \in \R^{778}$
    - per-vertex face contact probabilities: $C_F \in \R^{5023}$

DICE uses a two-branch Transformer that disentangles global mesh vertex regression from local deformation-field estimation; contacts (per-vertex labels for hand and face) and deformations are regressed in the same branch to model their causal coupling (contacts cause deformations). Supervision on the Decaf dataset combines mesh losses with interaction losses: Chamfer-distance touch/collision terms, binary cross-entropy on contact labels, and adaptively weighted deformation losses. Because Decaf covers only 8 subjects captured against green screens, the authors add weakly-supervised training on in-the-wild images without 3D ground truth (keypoint and depth-based supervision plus annotation-free touch/collision losses). Evaluation follows standard mesh-recovery metrics — PVE, MPJPE, PA-MPJPE — plus physical-plausibility metrics: Collision Distance, Non-Collision Ratio, Touchness Ratio, and their harmonic-mean F-Score; baselines include Decaf and a METRO variant extended with contact/deformation heads.

**Findings**

End-to-end joint estimation works and generalizes: DICE surpasses prior methods in accuracy on the Decaf benchmark and on challenging in-the-wild images. Ablations show (i) the two-branch design (separating deformation/interaction from vertex regression) improves both accuracy and plausibility, and (ii) adding in-the-wild weak supervision improves all reconstruction error metrics while maintaining high plausibility — i.e., the weak-supervision recipe is what breaks the studio-data overfitting.

**Results**

On the Decaf benchmark DICE achieves the best reported PVE/MPJPE/PA-MPJPE and the best contact-estimation F-Score (with precision/recall/accuracy reported) among all baselines including Decaf itself and contact-augmented METRO; exact table values are in the paper (Tabs. 1–3). Inference runs at roughly 20 fps on an Nvidia RTX 4090, versus more than 15 seconds per image for Decaf's optimization — a ~300× speedup that makes video-rate contact analysis feasible.

**Conclusion**

The authors conclude that disentangled end-to-end regression plus weak in-the-wild supervision yields accurate, plausible, fast, and generalizable hand-face interaction capture. For the thesis this is a core RQ2 methods paper: it is the strongest current model for the contact-detection stage of the contact-aware self-adaptor pipeline (pose → hand-face contact → fidgeting dynamics), its weak-supervision strategy is a template for adapting contact models to conversational video without 3D labels, and its in-the-wild generalization emphasis feeds RQ3 and the DENSO positioning.

*Sources: *[*https://arxiv.org/abs/2406.17988*](https://arxiv.org/abs/2406.17988)* (full-text HTML including references), *[*https://openreview.net/forum?id=rfrtFwnF62*](https://openreview.net/forum?id=rfrtFwnF62)*, *[*https://github.com/Qingxuan-Wu/DICE*](https://github.com/Qingxuan-Wu/DICE)*, *[*https://frank-zy-dou.github.io/projects/DICE/index.html*](https://frank-zy-dou.github.io/projects/DICE/index.html)