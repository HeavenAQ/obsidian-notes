---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2303.18246
Year: 2023
Should Refer: []
Reading Status: To Read
Venue: CVPR
Code Link: https://ipman.is.tue.mpg.de
Relatedness: weakly
Topic: Self-Contact / Hand-Face CV
snippet: |-
  • physics-informed HPS: stability via pressure / CoP / CoM + floor contact
    • differentiable intuitive-physics terms (no physics engine)
    • human–ground contact analog of self-contact terms
    • releases MoYo (multi-view, GT contact/CoM/pressure)
Authors: Shashank Tripathi, Lea Müller, Chun-Hao P. Huang, Omid Taheri, Michael J. Black, Dimitrios Tzionas
Tags:
  - Contact-Aware
  - Dataset
  - Theory
Tier: Recommended
Assigned Date: 2026-07-07
---
## Notes

- Contact-aware, but the contact is **human–ground** (stability) rather than self-contact.
- Relevant as an example of **differentiable physical/contact priors** improving pose robustness — the same style of term used for self-contact.
- Müller is 2nd author; same MPI-IS SMPL/SMPL-X line as TUCH.

## Reading Summary

**Abstract**

CVPR 2023 (Tripathi, Müller, Huang, Taheri, Black, Tzionas). Estimating 3D humans from images often yields implausible bodies that lean, float, or penetrate the floor, because methods ignore that bodies are usually *supported by the scene*. Physics engines can enforce plausibility but are non-differentiable, use unrealistic proxy bodies, and are hard to integrate. Instead, the paper uses **intuitive-physics (IP)** terms inferred directly from a 3D SMPL body interacting with the scene.

**Research Question**

Can intuitive-physics terms — stability and floor contact inferred from a SMPL body — improve monocular 3D pose plausibility *without* a full physics engine, in a differentiable way that plugs into existing optimizers and regressors?

**Methodology**

Inspired by biomechanics, infer from the SMPL body: a **pressure heatmap**, the **Center of Pressure (CoP)** from that heatmap, and the body's **Center of Mass (CoM)**. Build **IPMAN**, which encourages a "stable" configuration by rewarding plausible floor contact and **overlap of CoP and CoM** (a static-stability criterion). The IP terms are intuitive, fast, differentiable, and integrated into both optimization and regression pipelines. Evaluate on standard datasets and **MoYo**, a new multi-view dataset with GT 3D bodies (complex yoga poses), body–floor contact, CoM, and pressure.

**Findings**

Adding IP/stability terms removes the characteristic floating/leaning/penetrating failure modes and yields more physically plausible bodies.

**Results**

IPMAN improves accuracy for static poses while not hurting dynamic ones, producing more plausible results than the state of the art; code and MoYo are public at [ipman.is.tue.mpg.de](http://ipman.is.tue.mpg.de/).

**Conclusion**

Intuitive-physics contact/stability terms are cheap, differentiable, and broadly integrable. For the thesis this is a **contact-aware methods reference** — human–ground contact as the analog of self-contact terms — and a template for differentiable physical priors that make pose estimation more reliable, an argument that transfers to robust in-cabin behavior sensing.

*Sources: *[*https://arxiv.org/abs/2303.18246*](https://arxiv.org/abs/2303.18246)*, *[*https://ipman.is.tue.mpg.de*](https://ipman.is.tue.mpg.de/)