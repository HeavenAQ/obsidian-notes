---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2403.06487
Year: 2024
Should Refer:
  - None
Reading Status: Skim/Skip
Venue: LREC
Relatedness: not relevant
Topic: Gesture & Turn-Taking
snippet: |2-
    • LLM can generate natural spoken language but still struggle with when to take the turn
    • trains a Voice Activity Projection (VAP) cross-attention Transformer to continuously predict upcoming speaker activity across different languages.
    • achieves better cross-language turn-taking prediction
Authors: K. Inoue, B. Jiang, E. Ekstedt, T. Kawahara, G. Skantze
Tags:
  - Turn-Taking
  - Multimodal
Tier: Recommended
Assigned Date: 2026-07-07
---
Recommended reading, in the research lineage of Sacks, Schegloff & Jefferson (1974) — a computational descendant of the classic turn-taking systematics, this LREC-COLING 2024 paper trains a Voice Activity Projection (VAP) cross-attention Transformer to continuously predict upcoming speaker activity across English, Mandarin, and Japanese dyadic dialogue, showing multilingual training improves cross-language turn-taking prediction.

## Reading Summary

**Abstract**

This LREC-COLING 2024 paper by Inoue, Jiang, Ekstedt, Kawahara, and Skantze extends Voice Activity Projection (VAP) — a continuous, transformer-based model that predicts the near-future voice activity of both speakers in a dyadic dialogue — to a multilingual setting spanning English, Mandarin, and Japanese. It asks whether a single model trained across languages can match language-specific models, and probes what the model implicitly learns about language identity and prosody along the way.

**Research Question**

The paper poses five explicit questions: (RQ1) does a VAP model trained on one language transfer to another; (RQ2) can one multilingual model match monolingual performance across all three languages; (RQ3) does the multilingual model implicitly learn to identify the input language; (RQ4) how much does the model rely on pitch, a cue long thought central to turn-taking; and (RQ5) does the choice of audio encoder (English-pretrained CPC vs. massively multilingual wav2vec2/MMS) matter.

**Methodology**

VAP encodes each speaker's raw audio channel with a frozen, Librispeech-pretrained Contrastive Predictive Coding (CPC) encoder, passes each through a per-speaker self-attention transformer, then fuses the two streams with a cross-attention transformer; two linear heads predict (a) the joint future voice-activity state over a 2-second window discretized into 4 time bins per speaker (256-way classification) and (b) current voice activity as an auxiliary task. Models are trained/evaluated on three matched ~92.5-hour dyadic corpora: Switchboard (English telephone calls), HKUST (Mandarin telephone calls), and a Japanese travel-agency task-dialogue corpus. Evaluation uses test-set cross-entropy loss and balanced accuracy on a standard shift-vs-hold prediction task at silences longer than 0.25s. A separate experiment adds a language-identification head to probe RQ3, and a pitch-flattening manipulation (via Praat) tests RQ4.

**Findings**

Monolingual models transfer poorly to other languages, confirming that turn-taking cues are not fully language-universal, but a single multilingual model trained on all three languages matches or slightly exceeds monolingual performance in every language — a "free lunch" generalization result. The multilingual model also implicitly learns to distinguish the three languages almost perfectly, despite never being explicitly supervised to do so in the base setup. Pitch flattening degrades Mandarin and Japanese prediction accuracy noticeably more than English, suggesting tonal/pitch-accent languages lean more heavily on pitch as a turn-taking cue. The English-pretrained CPC encoder outperforms the multilingual MMS encoder, suggesting encoder architecture/training regime matters more than language-matched pretraining for this task.

**Results**

Test loss: English-trained/English-tested 2.387 vs. multilingual model on English 2.396 (near parity); cross-lingual transfer is much worse (e.g., English-trained model tested on Japanese: loss 2.956 vs. 2.329 for the Japanese-trained model). Balanced accuracy on shift/hold prediction: English monolingual 79.59% vs. multilingual 77.16%; Mandarin monolingual 84.49% vs. multilingual 84.60%; Japanese monolingual 74.20% vs. multilingual 76.54% (multilingual matches or beats monolingual in 2 of 3 languages). Language-identification F1 reached 99.99%. Pitch-flattening accuracy changes: Mandarin −2.02pp (mono) / −2.30pp (multi); Japanese −1.37pp / −1.81pp; English ≈ 0 (+0.09/+0.12pp). CPC beat MMS by roughly 1–3 percentage points of balanced accuracy across all three languages.

**Conclusion**

The authors conclude that a single multilingual VAP model is a practical, near-lossless replacement for separate monolingual turn-taking models, and that reliance on prosodic (pitch) cues for turn-taking varies systematically by language. For the thesis, this is a strong Layer 2/RQ2–RQ3 reference: it is a concrete, well-quantified example of cross-domain (here cross-lingual) generalization for a continuous, transformer-based conversational-timing model, directly parallel to the kind of cross-dataset/cross-domain generalization benchmarking planned for the self-adaptor detection pipeline. It also sits squarely in the Gesture & Turn-Taking topic adjacent to Layer 1's discourse-planning-difficulty framing. Notably, first author Koji Inoue is one of the three named PhD-target chair members (Kawakami/Sato/Inoue) at the DENSO IT Lab at Science Tokyo, so this paper is directly evidence of that lab's active, current research line in multimodal/cross-domain conversational-timing modeling — a positioning-relevant find beyond its content value.

*Sources: *[*arXiv:2403.06487*](https://arxiv.org/abs/2403.06487)*, *[*full-text PDF*](https://arxiv.org/pdf/2403.06487)