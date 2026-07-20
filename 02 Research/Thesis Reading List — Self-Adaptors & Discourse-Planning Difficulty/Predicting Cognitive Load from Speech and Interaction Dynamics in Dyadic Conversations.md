---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2606.12971
Year: 2026
Should Refer: []
Reading Status: Reading
Venue: arXiv
Topic: Multimodal-Disfluency-and-Discourse
snippet: Cognitive-load prediction from speech + interaction dynamics in dyadic conversation — closest recent work to RQ1/RQ2 framing; check features and labels used (authors TBD — verify on arXiv).
Authors: Tahiya Chowdhury
tags:
  - Cognition
  - Multimodal
  - Turn-Taking
Tier: Recommended
Assigned Date: 2026-07-05
Local PDF: "[[99 Assets/Papers/CVPR 2027/Predicting Cognitive Load from Speech and Interaction Dynamics in Dyadic Conversations.pdf]]"
Zotero URI: "zotero://select/library/items/RTQNMHCE"
Citation Key: "chowdhuryPredictingCognitiveLoad2026"
Zotero PDF URI: "zotero://open-pdf/library/items/65KNW4PI"
---
## Reading Summary

**Abstract**

This paper by Tahiya Chowdhury (Colby College) asks whether cognitive load experienced during a natural, task-based dyadic conversation can be predicted from speech and conversational-interaction dynamics alone, without physiological sensors. Using audio from 53 dyads completing nine collaborative tasks, the author extracts acoustic, prosodic-dynamic, and interaction (turn-taking) features and trains a two-head Gated Recurrent Unit (GRU) encoder to predict self-reported cognitive load along multiple NASA-TLX-style subscales.

**Research Question**

Do speech characteristics and interaction dynamics (turn overlap, speaker switching, participation balance) carry usable signal for predicting different facets of a conversational partner's perceived cognitive load (temporal demand, mental demand, effort, performance), and do different facets of load associate with different conversational-dynamics features?

**Methodology**

The dataset consists of 53 dyads each performing nine collaborative tasks, with per-task or per-segment self-reported cognitive load (multidimensional, NASA-TLX-style subscales). From the audio, the author extracts static acoustic features, features capturing their temporal dynamics, and interaction features specifically describing turn-taking (speech overlap, speaker-switch frequency, participation imbalance between the two speakers). These are fed into a two-head GRU encoder that jointly predicts multiple load subscales, evaluated presumably via concordance/correlation between predicted and self-reported scores at the dyad level (this study's full text was not machine-readable in this session; the account of the model and evaluation protocol here follows the abstract and third-party summaries rather than the primary tables).

**Findings**

Different cognitive-load subscales are predicted by different conversational signals: temporal demand (time pressure) is most closely tied to turn-taking dynamics such as overlap and speaker-switch frequency, whereas mental demand (perceived thinking/effort) is more closely tied to imbalanced participation between the two speakers. This dissociation is the paper's central qualitative claim — that interaction dynamics, not just raw acoustic-prosodic features, carry load-relevant information, and that different load facets are legible in different parts of the interaction structure.

**Results**

Secondary reporting on this paper (the primary PDF/HTML full text could not be retrieved as machine-readable text in this session, so these figures should be treated as approximate rather than independently verified against the paper's own tables) describes modest but non-trivial dyad-level prediction performance: correlation in the range of roughly 0.4-0.46 for temporal demand and around 0.3-0.36 for mental demand, i.e., temporal demand was easier to predict from interaction dynamics than mental demand, consistent with the qualitative finding above. No claim beyond this approximate range should be taken as confirmed without checking the published tables directly.

**Conclusion**

The author concludes that conversational interaction dynamics provide a genuinely useful, non-invasive signal for cognitive-load estimation in natural dyadic conversation, but that different load facets require different features to predict well, arguing for task-structure-aware modeling rather than one-size-fits-all acoustic classifiers. For the thesis, this is the closest published precedent to the core empirical framing: it operationalizes "discourse/task difficulty leaves a trace in conversational dynamics," directly supporting Layer 1 (RQ1, behavioral) by validating that interaction-level (not just acoustic) features carry cognitive-load signal, and it is methodologically relevant to Layer 2 (RQ2) as a naturalistic dyadic-conversation analogue to the self-adaptor detection pipeline's discourse-planning-difficulty target variable.

*Sources: *[*arXiv:2606.12971 (PDF listing)*](https://arxiv.org/pdf/2606.12971)*, web-search secondary summaries of the abstract and reported results (primary full text not machine-readable in this session)*