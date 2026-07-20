---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2205.06887
Year: 2023
Should Refer: []
Reading Status: Reading
Venue: AAAI
Code Link: https://github.com/pritamqu/AVCAffe
Topic: Dataset-Corpus
snippet: ""
Authors: P. Sarkar, A. Posen, A. Etemad
tags:
  - Dataset
  - Cognition
  - Multimodal
Tier: Recommended
Assigned Date: 2026-07-11
Local PDF: "[[99 Assets/Papers/CVPR 2027/AVCAffe A Large Scale Audio-Visual Dataset of Cognitive Load and Affect for Remote Work.pdf]]"
Zotero URI: "zotero://select/library/items/38SMC5GG"
Citation Key: "sarkarAVCAffeLargeScale2023"
Zotero PDF URI: "zotero://open-pdf/library/items/N8EXZ5DD"
---
Found via Mueller et al. 2019 (tonight's theory paper) — adjacent-literature substitute, not a direct reference (the 2019 paper's citations predate 2023). The ML descendant of behavioral cognitive-load research: 106 participants, 108 hours of audio-visual video-conferencing interaction with self-reported cognitive-load labels (mental demand, effort, etc.) — a candidate corpus for linking observable nonverbal behavior to load in conversation (RQ1→RQ2 bridge).

## Reading Summary

**Abstract**
Sarkar, Posen, and Etemad (AAAI 2023) introduce AVCAffe, the first audio-visual dataset to jointly annotate cognitive load and affect, recorded by simulating remote-work video-conferencing. 106 participants from 18 countries of origin (ages 18–57, balanced gender) completed a series of cognitively engaging collaborative tasks over a video-conferencing platform, yielding 108 hours of video (58,000+ short clips) paired with self-reported ground truth for cognitive-load attributes (mental demand, temporal demand, effort, and others, from NASA-TLX) and affect (arousal, valence, from SAM).

**Research Question**
The paper asks whether audio-visual signals captured during naturalistic remote video-conferencing work can be used to estimate a person's cognitive load and affective state, and how unimodal versus multimodal models, and short- versus long-timescale evaluation, compare on this task.

**Methodology**
Participants collaborated on multiple cognitively demanding tasks over video conferencing while being recorded; after each task they completed NASA-TLX (cognitive load: mental demand, physical demand, temporal demand, effort, performance, frustration) and SAM (arousal/valence) questionnaires. Because raw NASA-TLX scores (0–21) were too sparse and fine-grained to model reliably, cognitive load was reformulated as binary classification (High if score > 10, Low otherwise), while affect was kept as 5-class classification; three attributes (frustration, physical demand, performance) were dropped from the baselines for insufficient variance. The authors benchmark deep audio backbones (VGG-16, ResNet-18 on mel-spectrograms) and visual backbones (MC3-18, ResNet3D-18, R(2+1)D-18 on face-crop video clips) individually and via late fusion, evaluated at two timescales — short 6-second segments and long 2–10 minute videos — using weighted F1 against a random-classifier reference.

**Findings**
Multimodal (audio+visual) fusion generally outperforms unimodal models across attributes and timescales, with one notable exception: for "Effort" on short segments, visual-only models are competitive with or better than the multimodal fusion, and visual-only features are broadly stronger than audio-only ones once face-crops (rather than full frames) are used as visual input. Cognitive-load attributes (temporal demand, effort) show the opposite short-vs-long trend compared with affect attributes (arousal, valence) — cognitive-load estimation benefits more from long-video context, while affect estimation is relatively stronger on short segments — which the authors flag as an open question about how differently cognitive load and affect manifest temporally, both perceptually and computationally.

**Results**
The best multimodal models (weighted F1, chance-level baselines in parentheses) reach: temporal demand 66.7 long / 61.2 short (33.9/35.1 random); arousal 44.0 short / 40.5 long (26.2–32.9 random); valence 43.9 short / 41.7 long (30.3–34.0 random). For Effort, the best score overall is a visual-only R(2+1)D model at 69.4 (long) / 65.5 (short), exceeding the best multimodal Effort scores. All reported models clear the random-classifier reference by a wide margin, confirming the task is learnable but far from solved (best F1s cluster in the 60s, not the 80s–90s).

**Conclusion**
The authors position AVCAffe as the largest originally-collected English-language affective-computing dataset and the first with audio-visual cognitive-load annotations, offering it as a new, deliberately challenging benchmark for the community. For the thesis, this feeds Layer 2/RQ3 most directly: as a cognitive-load-labeled audio-visual corpus from a domain (solo remote video-conferencing tasks) distinct from both face-to-face conversation and driving, AVCAffe is a strong candidate for cross-dataset/domain-generalization benchmarking of the thesis's contact-aware multimodal cognitive-load pipeline, and its binary-load-from-NASA-TLX labeling scheme is a useful template for handling label sparsity in the thesis's own annotation design.

*Sources: [arXiv abstract/full text](https://arxiv.org/abs/2205.06887), [GitHub](https://github.com/pritamqu/AVCAffe), [author site](https://pritamsarkar.com/)*