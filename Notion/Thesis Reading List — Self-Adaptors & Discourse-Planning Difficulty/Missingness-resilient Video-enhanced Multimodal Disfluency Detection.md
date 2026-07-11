---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2406.06964
Year: 2024
Should Refer: []
Reading Status: Reading
Venue: Interspeech
Topic: Multimodal Disfluency & Discourse
snippet: ""
Authors: P. Mohapatra, S. Likhite, S. Biswas, B. Islam, Q. Zhu
Tags:
  - Multimodal
  - Disfluency
Tier: Essential
Assigned Date: 2026-07-02
---
**Background.**  disfluency systems assume clean speech and full modalities. 

**Research question.**  can video improve disfluency detection while staying robust to missing video? 

**Methodology.**  multimodal fusion explicitly designed for missing-modality resilience. 

**Conclusion.**  video helps and degrades gracefully -- the baseline family to beat by adding self-adaptor events, and a realistic evaluation setup for open corpora.

## Reading Summary

**Abstract**

This Interspeech 2024 paper presents a practical audio-visual disfluency detection framework designed to remain usable when the video modality is missing at inference. The authors curate a 3.3-hour paired, annotated audio-visual dataset from FluencyBank and propose a unified fusion network with weight-sharing modality-agnostic encoders, plus early- and late-fusion variants for the complete-modality case. Across five disfluency-detection tasks the multimodal approach beats audio-only baselines by about 10 points absolute, and by 7 points even with half the video missing.

**Research Question**

Can visual cues improve speech disfluency detection over the dominant audio-only paradigm, and can a fusion architecture be made resilient to arbitrary missingness of the video modality during training and inference?

**Methodology**

Dataset: FluencyBank videos segmented using SEP-28k-style annotations (Lea et al.), distilled by majority vote to 4,004 3-second clips over five disfluency classes (blocks 469, word repetition 337, sound repetition 428, interjection 701, prolongation 407, plus 1,660 fluent), released with train/test splits. Audio features come from frozen wav2vec 2.0 (base) with a CNN temporal decimator; video features come from a 3D-CNN/ResNet-18 pipeline over the full facial region. The DAV-unified model projects both modalities into a common latent space and passes them through a shared 16-head transformer encoder with learnable per-modality scaling; modality dropout (p = 0.5 on video) trains missingness resilience. Baselines: audio-only, video-only, audio-text AT-Dsflnt, and lip-based Auto-AVSR; metrics are balanced accuracy and F1 over 3 seeds with 20% held-out data.

**Findings**

Visual information carries substantial disfluency signal — video-only models sometimes match full multimodal models, which the authors attribute to disfluency's paralinguistic expression in facial gestures. Unlike lip-reading tasks, cropping to the lip region hurts (average −14% video-only, −11.5% unified), so whole-face context matters. The unified weight-sharing design degrades gracefully as video availability drops and transfers zero-shot to audio-only SEP-28k without losing to a pure audio model.

**Results**

Average accuracy across the five tasks: DAV-unified 0.74 vs. audio-only 0.64 (≈10 points absolute; up to 17 points on some tasks); with 50% of video missing at inference the unified model still gives a 7-point average boost over audio-only. Per-task balanced accuracy for DAV-unified ranges from 0.69 (prolongation) to 0.79 (interjection); Auto-AVSR (lip-focused) manages only 0.56 average, and zero-shot transfer to SEP-28k yields up to +6 balanced-accuracy points over audio-only.

**Conclusion**

Video meaningfully improves disfluency detection and can be incorporated robustly even when partially missing, via a unified weight-sharing encoder. For the thesis this is the natural baseline family to extend: whole-face visual context already helps detect disfluency, so adding explicit self-adaptor/self-touch event features is a plausible next increment, and the missingness-resilient design fits realistic conversational corpora.

*Sources: *[*https://arxiv.org/abs/2406.06964*](https://arxiv.org/abs/2406.06964)* (full text PDF), *[*https://www.isca-archive.org/interspeech_2024/mohapatra24_interspeech.html*](https://www.isca-archive.org/interspeech_2024/mohapatra24_interspeech.html)*, *[*https://github.com/payalmohapatra/Multimodal-Speech-Disfluency*](https://github.com/payalmohapatra/Multimodal-Speech-Disfluency)