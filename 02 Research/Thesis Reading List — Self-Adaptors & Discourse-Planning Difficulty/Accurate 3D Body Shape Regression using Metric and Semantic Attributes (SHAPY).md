---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2206.07036
Year: 2022
Should Refer: []
Reading Status: To-Read
Venue: CVPR
Code Link: https://github.com/muelea/shapy
Relatedness: Weakly
Topic: Self-Contact-Hand-Face-CV
snippet: |-
  • body SHAPE from images via metric + linguistic/semantic attributes
    • appearance side of SMPL-X (shape term β), not contact
    • background/foundation reference; weak relevance to self-adaptors
    • releases HBW (Human Bodies in the Wild) benchmark
Authors: Vasileios Choutas, Lea Müller, Chun-Hao P. Huang, Siyu Tang, Dimitrios Tzionas, Michael J. Black
tags:
  - Dataset
  - Weak-Supervision
Tier: Background
Assigned Date: 2026-07-07
Local PDF: "[[99 Assets/Papers/CVPR 2027/Accurate 3D Body Shape Regression using Metric and Semantic Attributes (SHAPY).pdf]]"
Zotero URI: "zotero://select/library/items/CJV274FQ"
Citation Key: "choutasAccurate3DBody2022"
Zotero PDF URI: "zotero://open-pdf/library/items/K9Q636P6"
---
## Notes

- About **body shape** (the SMPL-X shape term β), not contact or self-touch — included as **background** on the appearance side of the human model used across the contact papers.
- Same author group / SMPL-X line; weak direct relevance to self-adaptor detection.
- Releases the **HBW (Human Bodies in the Wild)** benchmark with GT 3D scans.

## Reading Summary

**Abstract**

CVPR 2022 (Choutas, Müller, Huang, Tang, Tzionas, Black). Regresses 3D body shape from a single RGB image by exploiting two cheap supervision sources: (1) internet "fashion" model images with a small set of **anthropometric measurements**, and (2) **linguistic shape attributes** (e.g., "tall", "muscular") for a range of 3D meshes and images. The network, **SHAPY**, regresses 3D pose and shape without needing large paired image–scan datasets.

**Research Question**

Can accurate 3D body-shape estimation be achieved *without* large 3D-scan-paired image data, by leveraging metric measurements and linguistic (semantic) shape attributes as proxy supervision?

**Methodology**

Define mappings between **attributes and body measurements** (A2B / B2A) and between linguistic ratings and the SMPL-X shape space; supervise SHAPY with metric attributes on internet images plus crowd-sourced linguistic attribute ratings. Introduce **HBW (Human Bodies in the Wild)**, photos paired with ground-truth 3D body scans, as an evaluation benchmark.

**Findings**

Attribute-based supervision effectively substitutes for scarce paired 3D data, letting the model learn metrically meaningful body shape from in-the-wild images.

**Results**

On the new HBW benchmark, SHAPY significantly outperforms state-of-the-art methods on 3D body-shape estimation; code and data are public.

**Conclusion**

Linguistic + metric attributes are strong, cheap proxies for 3D shape supervision. For the thesis this is a **background reference** on the body-shape component (β) of the SMPL-X model that the self-contact methods build on — tangential to self-adaptor detection itself, but useful for understanding the appearance half of the human model.

*Sources: *[*https://arxiv.org/abs/2206.07036*](https://arxiv.org/abs/2206.07036)*, *[*https://github.com/muelea/shapy*](https://github.com/muelea/shapy)*, *[*https://shapy.is.tue.mpg.de*](https://shapy.is.tue.mpg.de/)