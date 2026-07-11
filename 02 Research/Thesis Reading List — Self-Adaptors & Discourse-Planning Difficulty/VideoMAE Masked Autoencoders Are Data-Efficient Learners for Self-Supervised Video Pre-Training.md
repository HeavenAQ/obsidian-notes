---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2203.12602
Year: 2022
Should Refer: []
Venue: NeurIPS
Code Link: https://github.com/MCG-NJU/VideoMAE
Topic: Action-Recognition Backbone
snippet: ""
Authors: Z. Tong, Y. Song, J. Wang, L. Wang
tags:
  - Video-Backbone
Tier: Background
---
**Background.**  video pretraining is data-hungry. 

**Research question / methodology.**  masked autoencoding with an extremely high tube-masking ratio on a plain ViT. 

**Conclusion.**  a data-efficient SOTA backbone -- ideal RGB branch when self-touch labels are sparse/expensive (you already use VideoMAEv2).