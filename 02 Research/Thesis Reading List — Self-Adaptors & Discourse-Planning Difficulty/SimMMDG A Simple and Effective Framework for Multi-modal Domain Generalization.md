---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://proceedings.neurips.cc/paper_files/paper/2023/hash/f88bec15cc4cb56b432ee040bb63f94f-Abstract-Conference.html
Year: 2023
Should Refer: []
Venue: NeurIPS
Code Link: https://github.com/donghao51/SimMMDG
Topic: Domain Generalization
snippet: ""
Authors: H. Dong, I. Nejjar, H. Sun, E. Chatzi, O. Fink
tags:
  - Domain-Generalization
  - Multimodal
Tier: Background
---
**Background.**  forcing all modalities into one joint space hurts DG. 

**Research question / methodology.**  split features into modality-shared (supervised contrastive) and modality-specific (distance loss) parts, plus a cross-modal translation module for missing-modality robustness. 

**Conclusion.**  strong multimodal DG -- well suited to self-adaptors, whose visual signal is partly shared with, partly independent of, speech timing.