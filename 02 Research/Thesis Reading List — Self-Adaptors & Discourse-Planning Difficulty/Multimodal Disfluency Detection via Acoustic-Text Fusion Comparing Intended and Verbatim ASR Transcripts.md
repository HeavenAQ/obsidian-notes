---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://www.isca-archive.org/diss_2025/ferreira25_diss.pdf
Year: 2025
Should Refer: []
Reading Status: Reading
Venue: DiSS
Topic: Multimodal-Disfluency-and-Discourse
snippet: ""
Authors: M. H. L. Ferreira, A. I. Ferreira, L. R. Gris, R. Tanaka, I. J. S. Filho, F. Oliveira, A. Galvão Filho
tags:
  - Disfluency
  - Multimodal
Tier: Recommended
Assigned Date: 2026-07-15
---
Found via **Toward a Multimodal Approach for Disfluency Detection and Categorization** (2026-07-07 night) — adjacent literature substitute (the original paper's own reference list was inaccessible behind an IEEE paywall in this session). A 2025 continuation of the same acoustic+ASR-transcript multimodal disfluency detection research line, comparing intended-style vs. verbatim ASR transcripts fused with several acoustic encoders. Directly relevant to Layer 1/RQ1 and Layer 2/RQ2 fusion-architecture design.

## Reading Summary

**Abstract**
This DiSS 2025 paper studies a specific, previously under-examined design choice in multimodal disfluency detection: given that automatic speech recognition (ASR) systems can be tuned to produce either "intended-style" transcripts (clean, disfluency-omitting, e.g. off-the-shelf Whisper) or "verbatim-style" transcripts (literal, disfluency-preserving, e.g. fine-tuned CrisperWhisper), which style is more useful when fused with acoustic features for detecting disfluency? The authors build a dual-stream fusion model (acoustic encoder + text encoder, concatenated and classified by an MLP) and systematically compare Whisper-OTS versus CrisperWhisper transcripts fused with several acoustic encoders (Wav2vec 2.0, Whisper, and the non-verbal-vocalization model Voc2vec) on the SEP-28k-Extended stuttering/disfluency corpus.

**Research Question**
When acoustic features are fused with an ASR transcript for multimodal disfluency detection, does a cleaner "intended-style" transcript or a more literal "verbatim-style" transcript yield better detection performance, and why?

**Methodology**
The architecture is a dual-stream fusion network: the raw audio waveform is passed through a speech encoder (Wav2vec 2.0, Whisper encoder, Voc2vec, or their feature-extractor-only variants) and mean-pooled into an acoustic embedding, while in parallel an ASR model (Whisper-OTS for intended transcripts, CrisperWhisper for verbatim transcripts) generates a transcription that is tokenized with RoBERTa-Large and mean-pooled into a text embedding; the two embeddings are concatenated and fed to a single-hidden-layer MLP with ReLU and dropout 0.5, trained with cross-entropy loss via AdamW (learning rate 5×10⁻⁴ with cosine decay, batch size 64, 20 epochs, mixed precision on a single A100). The task is framed as binary fluent/disfluent classification on SEP-28k-Extended, an enhanced version of the SEP-28k stuttering corpus (>28,000 clips) with speaker-independent splits and richer metadata, with ground truth derived by thresholding annotator disfluency-type scores.

**Findings**
The results consistently support the authors' hypothesis: pairing acoustic features with the cleaner, intended-style Whisper-OTS transcript outperforms pairing the same acoustic features with the verbatim CrisperWhisper transcript, across every tested speech encoder. The proposed mechanism is that acoustic encoders (especially Wav2vec 2.0) already capture the raw acoustic manifestation of a disfluency (hesitation, repeated sound, prolongation) directly from the waveform, so what the text stream needs to contribute is not a redundant verbatim transcription of that same disfluency but a contrasting "clean" reference of the speaker's intended utterance; the mismatch between the messy acoustic signal and the clean intended text becomes the discriminative signal, rather than the text stream trying to encode the disfluency itself. A secondary finding is that raw/non-contextual acoustic representations (Voc2vec, and feature-extractor-only variants of Wav2vec 2.0/Voc2vec) perform surprisingly well unimodally, suggesting that lower-level acoustic properties carry substantial disfluency-relevant signal even without full contextual language-model-style encoding.

**Results**
Table 1 reports F1-Macro across encoder/modality combinations: unimodal Whisper-OTS and CrisperWhisper audio-only both score 0.325 (weak), while Wav2vec 2.0 audio-only reaches 0.691, Voc2vec 0.627–0.628, and the feature-extractor-only variants 0.624–0.628. In the multimodal setting, Wav2vec 2.0 acoustic features fused with Whisper-OTS (intended) transcripts achieve the best overall score at 0.702 F1-Macro, versus 0.668 when the same acoustic stream is fused with CrisperWhisper (verbatim) transcripts; this intended-over-verbatim advantage holds for every encoder tested (e.g., Whisper-OTS text lifts Whisper-OTS audio from 0.325 to 0.605, versus 0.583 with verbatim text; CrisperWhisper audio goes from 0.325 to 0.607 with intended text versus 0.596 with verbatim text).

**Conclusion**
The authors conclude that multimodal disfluency detection benefits more from fusing acoustic features with intended-style ASR transcripts than verbatim ones, because the acoustic–text discrepancy itself functions as the detection signal, and they suggest exploring intermediate-layer Wav2vec 2.0 representations as a next step. For the thesis this is a Layer 1/RQ1-adjacent and Layer 2/RQ2 paper: it offers direct, empirically-grounded guidance on ASR-transcript-style choice for the multimodal fusion architecture (video + audio + transcript) the thesis proposes, and its "acoustic-text discrepancy as signal" framing is a transferable idea for the self-adaptor pipeline, where a mismatch between fluent transcript content and disfluent/hesitant delivery could analogously flag discourse-planning difficulty.

*Sources: [ISCA Archive paper page](https://www.isca-archive.org/diss_2025/ferreira25_diss.html), [ISCA Archive PDF](https://www.isca-archive.org/diss_2025/ferreira25_diss.pdf)*