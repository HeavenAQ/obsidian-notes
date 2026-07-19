---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2605.20992
Year: 2026
Should Refer: []
Reading Status: To-Read
Venue: arXiv
Topic: Self-Contact-Hand-Face-CV
snippet: ""
Authors: Hao Xu, Yilin Liu, Yinqiao Wang, Chi-Wing Fu, Niloy J. Mitra
tags:
  - Contact-Aware
  - Video
Tier: Recommended
---
Found via **Learning a Contact Potential Field for Modeling the Hand-Object Interaction (CPF)** (2026-07-17 night) — adjacent-literature substitute, not a direct reference (CPF's own reference list, from the 2020/2021 ICCV version, predates 2023). Extends contact-aware hand-object modeling from CPF's single-image, static-grasp setting to open-world monocular video: CHOIR reconstructs 4D (temporally evolving) hand motion, object pose, and contact from in-the-wild footage using contact as an explicit coupling signal, with a contact-aware joint optimization stage that enforces temporal consistency — directly relevant to the temporal/fidgeting-dynamics stage of RQ2's contact-aware self-adaptor pipeline (pose → hand-face contact → fidgeting dynamics).
