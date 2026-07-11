---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://dl.acm.org/doi/10.1145/3678957.3685741
Year: 2024
Should Refer: []
Reading Status: Reading
Venue: ICMI
Topic: Multimodal-Disfluency-and-Discourse
snippet: ""
Authors: Olcay Türk, Stefan Lazarov, Yu Wang, Hendrik Buschmeier, Angela Grimminger, Petra Wagner
tags:
  - Multimodal
  - Cognition
Tier: Recommended
Assigned Date: 2026-07-11
---
Found via **Predicting States of Understanding in Explanatory Interactions Using Cognitive Load-Related Linguistic Cues** (2026-07-06 night) — likely direct predecessor/reference (overlapping authorship and shared MUNDEX/explanatory-dialogue corpus with tonight's paper, though the exact reference list of the 2026 paper could not be verified due to a fetch rate limit in this session). Builds a random-forest classifier over multimodal cues (vocalizations, facial expressions, torso/head/hand movement) from a dyadic explanation corpus (21 pairs) to differentiate understanding from non-understanding windows, an important multimodal precursor to tonight's linguistic-cue-focused follow-up.

## Reading Summary

**Abstract**
Türk, Lazarov, Wang, Buschmeier, Grimminger, and Wagner (ICMI 2024) study how explainees in dyadic explanatory dialogues nonverbally and vocally signal whether they understand an ongoing explanation. Using the MUNDEX corpus of 21 explainer–explainee pairs explaining how to play a board game, they train a random-forest classifier over multimodal cues — vocalizations, facial expressions, and torso, head, and hand movements — to distinguish windows of understanding from windows of non-understanding.

**Research Question**
The paper asks whether moment-to-moment understanding versus non-understanding in an explanatory interaction can be predicted from multimodal nonverbal and vocal cues, and which modalities or cue types carry the most discriminative information about the listener's comprehension state.

**Methodology**
The MUNDEX corpus was built by recording dyads in which one participant explains a board game to the other; ground-truth understanding/non-understanding windows were obtained through a retrospective video-recall procedure in which participants later reviewed the recording and marked segments according to their own contemporaneous state of understanding. Multimodal features were extracted across the labeled windows spanning acoustic/vocalization cues, facial expression, and torso/head/hand movement, and a random-forest classifier was trained to discriminate the two classes from this feature set.

**Findings**
Windows of understanding and non-understanding are behaviorally separable using a broad multimodal feature set, and no single modality dominates — cues from vocalization, face, and body movement all contribute, consistent with discourse-state signaling being an inherently multimodal phenomenon rather than one carried by a single channel. At the same time, the authors note that generalizability is currently limited by data sparsity and a high degree of individual variation in how understanding is expressed nonverbally, meaning the classifier's cues and weights likely do not transfer cleanly across speakers without further work.

**Results**
Secondary-source descriptions of the paper confirm that the random-forest classifier differentiates understanding from non-understanding sequences above a meaningful margin using the combined multimodal predictor set, but the exact accuracy/F1 figures reported in the paper's results tables could not be retrieved in this session — the ACM Digital Library page did not return full text (paywalled), and a follow-up attempt to cross-check numbers via the related 2026 arXiv paper was blocked by a fetch rate limit. This summary is therefore built from the abstract and consistent third-party descriptions rather than the primary results table, and precise metric values should be treated as unverified until the full PDF is consulted.

**Conclusion**
The authors conclude that listener understanding state is legible from multimodal nonverbal and vocal behavior even without access to speech content, though robust, generalizable prediction remains an open problem given inter-individual variability. For the thesis, this feeds Layer 1 (RQ1) most directly: it is a close methodological cousin of the core hypothesis — that a cognitive/discourse state (there, "understanding"; here, "discourse-planning difficulty") leaves a detectable multimodal nonverbal signature — and it also offers a Layer 2 (RQ2) precedent for combining vocal, facial, and body-movement features via a classical ML classifier (random forest) as a baseline fusion strategy before moving to deep multimodal fusion.

*Sources: [ACM Digital Library record](https://dl.acm.org/doi/10.1145/3678957.3685741), [related follow-up paper (arXiv)](https://arxiv.org/pdf/2603.20079)*