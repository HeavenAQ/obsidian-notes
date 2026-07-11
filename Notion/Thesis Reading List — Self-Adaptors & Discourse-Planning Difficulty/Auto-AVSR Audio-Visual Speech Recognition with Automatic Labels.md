---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2303.14307
Year: 2023
Should Refer: []
Reading Status: To Read
Venue: ICASSP
Code Link: https://github.com/mpc001/auto_avsr
Topic: Action-Recognition Backbone
snippet: ""
Authors: P. Ma, A. Haliassos, A. Fernandez-Lopez, H. Chen, S. Petridis, M. Pantic
Tags:
  - Multimodal
  - Video Backbone
Tier: Recommended
---
Recommended via tonight's paper "Missingness-resilient Video-enhanced Multimodal Disfluency Detection" (Mohapatra et al., Interspeech 2024), where it is a direct reference and the main audio-visual baseline (lip-based AV speech model). Relevant as a strong AV backbone and as evidence that lip-region pipelines built for ASR transfer poorly to paralinguistic tasks like disfluency — whole-face context wins, a useful design constraint for self-adaptor-aware models.