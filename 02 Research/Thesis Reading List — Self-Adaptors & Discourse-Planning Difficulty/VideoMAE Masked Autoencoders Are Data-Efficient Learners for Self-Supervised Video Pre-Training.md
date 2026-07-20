---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2203.12602
Year: 2022
Should Refer: []
Venue: NeurIPS
Code Link: https://github.com/MCG-NJU/VideoMAE
Topic: Action-Recognition-Backbone
snippet: ""
Authors: Zhan Tong, Yibing Song, Jue Wang, Limin Wang
tags:
  - Video-Backbone
Tier: Background
Citation Key: "tong2022videomae"
Local PDF: "[[99 Assets/Papers/CVPR 2027/VideoMAE Masked Autoencoders Are Data-Efficient Learners for Self-Supervised Video Pre-Training.pdf]]"
Zotero URI: "zotero://select/library/items/GXUUH6L5"
Zotero PDF URI: "zotero://open-pdf/library/items/65FI7Z9H"
---
**Background.**  video pretraining is data-hungry. 

**Research question / methodology.**  masked autoencoding with an extremely high tube-masking ratio on a plain ViT. 

**Conclusion.**  a data-efficient SOTA backbone -- ideal RGB branch when self-touch labels are sparse/expensive (you already use VideoMAEv2).