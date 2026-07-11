---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2509.23393
Year: 2025
Should Refer: []
Reading Status: Reading
Venue: ICCV
Topic: Self-Contact-Hand-Face-CV
snippet: ""
Authors: Takehiko Ohkawa, Jihyun Lee, Shunsuke Saito, Jason Saragih, Fabian Prado, Yichen Xu, Shoou-I Yu, Ryosuke Furuta, Yoichi Sato, Takaaki Shiratori
tags:
  - Contact-Aware
Tier: Recommended
Assigned Date: 2026-07-07
---
Found via **DF-Mamba** (2026-07-05 night) — direct reference [50]. Introduces Goliath-SC, a large self-contact human-pose dataset (383K poses, 130 subjects) with body-shape-conditioned generative modeling of self-contact priors. Full-body self-contact rather than hand-face specifically, but directly relevant methodology for contact-aware pose modeling (RQ2). Same first author as DF-Mamba.

## Reading Summary

**Abstract**

This ICCV 2025 paper (Ohkawa, Lee, Saito, Saragih, Prada, Xu, Yu, Furuta, Sato, Shiratori) addresses full-body self-contact human pose modeling — poses where a person's body touches itself (e.g., an arm resting on the belly). It observes that the geometry of such contact depends on the individual's body shape (a heavier person's self-touch pose differs from a thinner person's for the "same" gesture), a factor prior self-contact datasets and models largely ignored. The authors introduce Goliath-SC, a large, shape-registered self-contact pose dataset, and a body-shape-conditioned generative diffusion model over self-contact poses, which they use to refine single-image 3D pose estimates into physically consistent, contact-respecting poses.

**Research Question**

Can self-contact human poses be modeled generatively in a way that correctly accounts for body-shape dependence, and does conditioning on body shape improve both the realism of generated self-contact poses and the accuracy of self-contact-aware pose estimation from a single image?

**Methodology**

The authors built Goliath-SC, described as the first large-scale self-contact dataset with precise body-shape registration, comprising roughly 383,000 self-contact poses across 130 subjects captured performing natural daily-life self-touch behaviors (face, body, hands). On top of this they train a body-part-wise latent diffusion model with self-attention, conditioned on SMPL-X-style body-shape parameters, to model the distribution of plausible self-contact poses per individual shape. This learned prior is then used as a refinement/regularization step applied to an initial single-view 3D pose estimate, nudging it toward a shape-consistent, self-contact-respecting pose. The refined approach is compared against BUDDI (a prior diffusion-based human-contact model, already in this reading list) and SMPLer-X (a direct-regression pose/shape foundation model, also in this reading list) on held-out subjects from the Goliath-SC evaluation split.

**Findings**

The central qualitative finding is that shape conditioning is not a minor detail but a necessary ingredient: without conditioning on individual body shape, a generative model of self-contact poses cannot correctly capture the pose distribution, because the same semantic self-contact gesture (e.g., touching one's stomach) maps to substantially different joint configurations depending on body shape. Incorporating this shape-conditioned prior as a refinement step over an initial pose estimate produces more physically plausible, contact-consistent poses than either a shape-agnostic contact prior (BUDDI) or a strong direct-regression foundation model (SMPLer-X) that doesn't explicitly reason about contact at all.

**Results**

The paper reports that the shape-conditional diffusion refinement surpasses both BUDDI and SMPLer-X on the Goliath-SC evaluation set with unseen subjects. I was not able to retrieve the full quantitative results tables (the arXiv full text exceeded what could be fetched in this session), so exact metric values (e.g., contact-precision/recall, per-joint position error) are not reproduced here; this summary is based on the abstract, the ICCV listing, and secondary write-ups (including the authors' own lab page), and should be treated as qualitative pending a direct read of the paper's tables.

**Conclusion**

The authors conclude that shape-aware generative modeling meaningfully improves self-contact pose estimation over shape-agnostic priors and direct-regression baselines, establishing Goliath-SC and the shape-conditioned diffusion prior as a new reference point for contact-aware 3D human pose work. For the thesis, this feeds directly into Layer 2/RQ2: it is closely relevant methodological precedent for the "pose → hand-face contact" stage of the planned contact-aware self-adaptor detection pipeline, even though its scope is whole-body self-contact rather than hand-face specifically. It is also a notable positioning find: co-author Yoichi Sato is one of the three named PhD-target chair members (Kawakami/Sato/Inoue) at the DENSO IT Lab, Institute of Science Tokyo, so this is direct, current evidence of that lab's active work on exactly the shape/contact-aware pose modeling problem central to the thesis's Layer 2 methodology. (Note: depth of this summary is limited by incomplete full-text access; a follow-up pass reading the full PDF directly is recommended before citing specific numbers from this paper.)

*Sources: *[*arXiv:2509.23393 abstract*](https://arxiv.org/abs/2509.23393)*, *[*ICCV 2025 Open Access page*](https://openaccess.thecvf.com/content/ICCV2025/html/Ohkawa_Generative_Modeling_of_Shape-Dependent_Self-Contact_Human_Poses_ICCV_2025_paper.html)*, *[*author project page*](https://tkhkaeio.github.io/projects/25-scgen/)