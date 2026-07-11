---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://ieeexplore.ieee.org/document/10095629/
Year: 2023
Should Refer: []
Reading Status: Reading
Venue: ICASSP
Topic: Multimodal Disfluency & Discourse
snippet: ""
Authors: A. Romana, K. Koishida
tags:
  - Multimodal
  - Disfluency
Tier: Recommended
Assigned Date: 2026-07-07
---
Recommended reading, found via the reference list of Romana, Koishida & Mower Provost (2024) TASLP paper — this ICASSP 2023 paper is the direct precursor work in which the authors first introduced the multimodal (fine-tuned BERT + fine-tuned W2V2) fusion approach for disfluency detection that the TASLP paper later expands and re-evaluates with newer ASR/acoustic models.

## Reading Summary

**Abstract**

This ICASSP 2023 paper by Romana and Koishida (Microsoft Applied Sciences) is the earlier, precursor work to the group's later TASLP journal paper already in this reading list. It evaluates disfluency detection and categorization using automatically-derived text (via ASR transcription) versus acoustic features, and introduces a multimodal fusion approach that combines the two, motivated by the observation that ASR transcripts of disfluent speech are themselves unreliable.

**Research Question**

Can combining acoustic features with automatically-derived (ASR-based) linguistic features improve disfluency detection and categorization over either modality alone, and how much does the unreliability of ASR transcription on disfluent speech limit a text-only approach?

**Methodology**

The language branch fine-tunes a BERT-style model on transcripts produced by an ASR system (Whisper, fine-tuned); the acoustic branch fine-tunes a speech representation model on the raw audio. Several fusion architectures are compared against each unimodal baseline: a simple multimodal perceptron/MLP fusion, a BLSTM-based fusion, and a transformer-based fusion. Performance is measured with unweighted (macro) and weighted F1 for disfluency detection/categorization.

**Findings**

Acoustic-only features clearly outperform ASR-transcript-only features, supporting the paper's motivating claim that automatic transcription degrades badly on disfluent speech and therefore undermines text-based disfluency detection. Multimodal fusion further improves over the acoustic-only model, and among fusion strategies, the recurrent (BLSTM) fusion outperforms both the simpler perceptron fusion and the more complex transformer fusion — a somewhat counterintuitive result suggesting that, at least for this task/data scale, sequence-order-aware but lighter-weight fusion generalizes better than a full transformer.

**Results**

Reported (approximate, macro/weighted) F1 scores: ASR-language-only ≈0.35 macro / 0.47 weighted; acoustic-only ≈0.45 macro / 0.64 weighted; multimodal perceptron ≈0.49 macro / 0.64 weighted; multimodal BLSTM ≈0.52 macro / 0.69 weighted (best overall); multimodal transformer ≈0.50 macro / 0.67 weighted. I could not access the original IEEE Xplore PDF directly in this session (institutional paywall), so these figures are drawn from secondary aggregation rather than a direct read of the paper's tables and should be treated as approximate pending verification against the primary source.

**Conclusion**

The authors conclude that multimodal fusion of acoustic and automatically-transcribed linguistic features is more robust for disfluency detection/categorization than either modality alone, and specifically that ASR-transcript unreliability is a key reason text-only approaches underperform on disfluent speech. For the thesis, this is a direct anchor for both layers: it feeds Layer 1/RQ1 (disfluency as observable evidence potentially linked to discourse-planning difficulty) and Layer 2/RQ2 (multimodal fusion architecture choices for video+audio+transcript pipelines), and it offers a concrete fusion-architecture comparison (BLSTM > transformer > perceptron, at least in this setting) worth benchmarking the thesis's own fusion design against. As the direct precursor to the same authors' later TASLP paper already in this list, reading both together clarifies how the group's approach evolved with newer ASR/acoustic backbones.

*Sources: *[*IEEE Xplore listing*](https://ieeexplore.ieee.org/document/10095629/)*, secondary aggregation of reported results (primary PDF paywalled; not independently verified from the original tables)*