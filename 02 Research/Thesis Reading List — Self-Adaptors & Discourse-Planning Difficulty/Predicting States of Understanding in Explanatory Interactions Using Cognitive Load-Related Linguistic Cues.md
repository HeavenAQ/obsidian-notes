---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2603.20079
Year: 2026
Should Refer:
  - None
Reading Status: Skim-Skip
Venue: arXiv
Relatedness: Not-Relevant
Topic: Multimodal-Disfluency-and-Discourse
snippet: ""
Authors: Yu Wang, Hendrik Buschmeier, et al.
tags:
  - Cognition
  - Multimodal
  - Disfluency
Tier: Recommended
Assigned Date: 2026-07-06
Local PDF: "[[99 Assets/Papers/CVPR 2027/Predicting States of Understanding in Explanatory Interactions Using Cognitive Load-Related Linguistic Cues.pdf]]"
Zotero URI: "zotero://select/library/items/QXCI5P2C"
Citation Key: "wangPredictingStatesUnderstanding2026"
Zotero PDF URI: "zotero://open-pdf/library/items/AEQG9GUK"
---
Found via **Predicting Cognitive Load from Speech and Interaction Dynamics in Dyadic Conversations** (2026-07-05 night) — adjacent literature, not a direct reference (source paper's reference list was not accessible in this session). Uses cognitive-load-related linguistic cues (speaker utterance surprisal/syntactic complexity, listener gaze variation) to predict moment-by-moment listener understanding state in explanatory dialogue. Very close conceptually to RQ1 (discourse-planning/processing difficulty leaving a trace in conversational behavior), extending a 2024 ICMI precursor ("Predictability of Understanding in Explanatory Interactions Based on Multimodal Cues") into a full linguistic-cue model.

## Reading Summary

**Abstract**

Wang, Türk, Grimminger and Buschmeier (Bielefeld/Paderborn TRR318; arXiv, submitted March 2026, [cs.CL](http://cs.cl/)) study whether linguistic cues tied to cognitive load — how surprising or syntactically complex an explainer's utterances are, and how much an explainee's gaze behaviour varies — can predict a listener's moment-by-moment state of understanding during face-to-face explanatory dialogue.

**Research Question**

Do cognitive-load-related linguistic and gaze cues (speaker utterance surprisal, speaker syntactic complexity, listener gaze-behaviour variation) track, and help predict, a listener's fine-grained understanding state during an ongoing explanation, and does adding these cues improve automatic prediction over text-only features?

**Methodology**

The study uses the MUNDEX corpus of face-to-face dyadic interactions recorded at Bielefeld, in which one participant explains a board game's rules to another and they then play it together, with multimodal annotation (gaze, gesture, adaptors, prosody) from multiple camera angles. Listener understanding was self-annotated post hoc via retrospective video recall into four states: Understanding, Partial Understanding, Non-Understanding, and Misunderstanding. Three cues were operationalized: information value of the speaker's utterance via language-model surprisal, syntactic complexity of the speaker's utterance, and variation in the listener's interactive gaze. Statistical analyses relate each cue to the listener's understanding-state label, and a classification experiment compares two off-the-shelf classifiers against a fine-tuned German BERT-based multimodal classifier combining textual features with the three cognitive-load cues.

**Findings**

The individual cues shift systematically with the listener's level of understanding — more complex or less predictable explainer utterances and more variable listener gaze tend to co-occur with poorer understanding states — consistent with the hypothesis that cognitive-load markers leak into observable dialogue behavior. Automatic classification of the four understanding states is shown to be generally achievable, and combining the three cognitive-load cues with textual features improves classification over textual features alone, supporting the idea that nonverbal and paralinguistic load markers carry information complementary to lexical content.

**Results**

The paper reports statistical relationships between each of the three cues and the four annotated understanding states, and reports classification performance for the two off-the-shelf classifiers and the fine-tuned multimodal German BERT model, with the cue-augmented multimodal model outperforming text-only baselines. A web-search-and-abstract pass in this session could not retrieve the exact accuracy/F1 figures from the full text (a fetch rate limit blocked PDF/HTML retrieval), so precise numbers should be verified directly against the paper before being cited elsewhere.

**Conclusion**

The authors conclude that cognitive-load-related linguistic and gaze cues are a useful, learnable signal of a listener's real-time understanding state in explanatory dialogue, and that multimodal models combining these cues with text improve prediction of comprehension breakdowns. This is a strong RQ1 (behavioral/discourse-planning) fit: it operationalizes "processing/comprehension difficulty leaves a trace in observable behavior" from the listener side, complementing the thesis's speaker-side self-adaptor hypothesis, and its retrospective-annotation-plus-multimodal-classifier design is a useful methodological template for Layer 2's discourse/disfluency fusion component. Depth here is limited by access: the full-text PDF could not be retrieved in this session, so this summary relies on the abstract and the paper's ICMI 2024 companion study; a follow-up pass should pull the full PDF for exact classification metrics.

*Sources: *[*https://arxiv.org/abs/2603.20079*](https://arxiv.org/abs/2603.20079)*, *[*https://dl.acm.org/doi/10.1145/3678957.3685741*](https://dl.acm.org/doi/10.1145/3678957.3685741)* (companion ICMI 2024 study), *[*https://trr318.uni-paderborn.de/en/news-item/hand-gestures-nods-and-co*](https://trr318.uni-paderborn.de/en/news-item/hand-gestures-nods-and-co)* (MUNDEX corpus description)*