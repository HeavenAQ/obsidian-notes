---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Authors: István Sárándi, Gerard Pons-Moll
Year: 2024
Venue: NeurIPS
Paper Link: https://arxiv.org/abs/2407.07532
Code Link: https://github.com/isarandi/nlf
Topic: Domain-Generalization
Tier: Recommended
tags:
  - Human-Mesh-Recovery
  - Domain-Generalization
  - Continuous-Field
Reading Status: To-Read
Assigned Date: 2026-07-14
---

Found via **SAM 3D Body: Robust Full-Body Human Mesh Recovery** (2026-07-14 night) — direct reference and comparison baseline on SAM 3D Body's new challenging benchmarks. NLF's core contribution is unifying heterogeneous pose/shape datasets and annotation formats (meshes, 2D/3D skeletons, dense pose) via a continuous neural field of point localizers queryable at any body location — directly relevant to RQ3 cross-dataset/domain-generalization benchmarking, since it is explicitly designed to generalize across annotation schemas rather than a single dataset's conventions.
