---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2104.13586
Year: 2022
Should Refer: []
Venue: CVPR
Code Link: https://github.com/kennymckormick/pyskl
Topic: Action-Recognition Backbone
snippet: ""
Authors: H. Duan, Y. Zhao, K. Chen, et al.
Tags:
  - Skeleton
  - Video Backbone
Tier: Background
---
**Background.**  GCN skeleton methods are brittle to noisy pose. 

**Research question / methodology.**  represent skeletons as 3D heatmap volumes processed by a 3D-CNN. 

**Conclusion.**  more robust to pose-estimation noise and easy to fuse with RGB -- often a better practical baseline than graph models for messy conversational video.