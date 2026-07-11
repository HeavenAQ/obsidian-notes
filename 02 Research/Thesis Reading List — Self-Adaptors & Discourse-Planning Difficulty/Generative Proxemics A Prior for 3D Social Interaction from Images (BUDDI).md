---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2306.09337
Year: 2024
Should Refer: []
Reading Status: To Read
Venue: CVPR
Code Link: https://github.com/muelea/buddi
Relatedness: weakly
Topic: Self-Contact / Hand-Face CV
snippet: |-
  • human–human close-interaction prior, not self-contact
    • diffusion model (BUDDI) over two people's 3D poses
    • used as a prior to reconstruct 2 people from 1 image w/o contact labels
    • contact-from-images methodology transfers to the thesis
Authors: Lea Müller, Vickie Ye, Georgios Pavlakos, Michael J. Black, Angjoo Kanazawa
tags:
  - Contact-Aware
  - Dataset
  - Multimodal
Tier: Recommended
Assigned Date: 2026-07-07
---
## Notes

- Same author (Müller) as TUCH, but this is **human–human** close interaction (proxemics), not self-contact.
- The relevance is methodological: **contact-conditioned 3D reconstruction from a single image using a learned generative (diffusion) prior over contact**.
- No contact annotation needed at inference — the prior supplies the constraint.

## Reading Summary

**Abstract**

CVPR 2024 (Müller, Ye, Pavlakos, Black, Kanazawa). Proxemics — how people position themselves relative to others — conveys social cues but is hard to reconstruct from images because of mutual occlusion and scarce 3D training data. The paper learns a prior over the 3D proxemics of two people in close social interaction and uses it for single-view 3D reconstruction, via a novel denoising diffusion model, **BUDDI**, that models the joint distribution over the two bodies' poses.

**Research Question**

Can a learned generative prior over 3D proxemics (the relative pose/contact of two interacting people) enable accurate, annotation-free single-view 3D reconstruction of close social interactions?

**Methodology**

(1) Build 3D training data of interacting people from image datasets with contact annotations. (2) Train **BUDDI**, a denoising diffusion model that learns the *joint* distribution over the SMPL-X poses (and relative placement) of two people in close interaction — so sampling produces realistic two-person interactions. (3) At inference, use BUDDI as a **prior in an optimization** that refines noisy initial two-person estimates from a single image, *without any contact annotation*, letting the diffusion prior pull the reconstruction toward plausible proxemics.

**Findings**

Sampling from the generative proxemics model yields realistic 3D human interactions (validated by a perceptual study). Used as a prior, it recovers accurate and plausible 3D social interactions from noisy initial estimates.

**Results**

Outperforms state-of-the-art methods on single-view close-interaction reconstruction; code, data, and model are public at [muelea.github.io/buddi](http://muelea.github.io/buddi).

**Conclusion**

A diffusion prior over two-person proxemics is an effective, annotation-free route to 3D social interaction from images. For the thesis this is a **methods-layer template**: it shows how to encode contact as a learned prior over 3D bodies and inject it into reconstruction — directly transferable to a self-contact / self-adaptor prior, even though the paper itself targets human–human rather than self-contact.

*Sources: *[*https://arxiv.org/abs/2306.09337*](https://arxiv.org/abs/2306.09337)*, *[*https://github.com/muelea/buddi*](https://github.com/muelea/buddi)*, *[*https://muelea.github.io/buddi*](https://muelea.github.io/buddi)