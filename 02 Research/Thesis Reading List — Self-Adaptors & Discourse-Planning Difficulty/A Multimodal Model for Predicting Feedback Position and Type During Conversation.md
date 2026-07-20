---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://www.sciencedirect.com/science/article/pii/S0167639324000384
Year: 2024
Should Refer: []
Reading Status: Reading
Assigned Date: 2026-07-19
Venue: Computer-Speech-Language
Topic: Multimodal-Disfluency-and-Discourse
snippet: ""
Authors: Auriane Boudin, Roxane Bertrand, Stéphane Rauzy, Magalie Ochs, Philippe Blache
tags:
  - Multimodal
  - Discourse
Tier: Recommended
---
Found via **Predictability of Understanding in Explanatory Interactions Based on Multimodal Cues** (2026-07-11 night) — adjacent-literature substitute, not a verified direct reference (ACM Digital Library page was paywalled and did not return the source paper's reference list in this session). Models morpho-syntactic, prosodic, and gestural predictors of listener feedback position/type in conversation — a close methodological parallel to predicting a discourse state (there, feedback-worthy moments; for the thesis, discourse-planning difficulty) from multimodal speaker/listener cues.

## Reading Summary

**Abstract**
This paper investigates conversational feedback — the reactions (e.g., backchannels) a listener produces in response to a speaker — and builds a computational model that jointly predicts both where feedback will occur and what type it will be, using a fine-grained multimodal feature set drawn from the speaker's own production and applied to French conversational corpora.

**Research Question**
Which multimodal features of a speaker's production (morpho-syntactic, prosodic, gestural) predict the position and the fine-grained type of listener feedback, and can generic versus specific feedback — with specific feedback further split into positive/negative and expected/unexpected subtypes — be automatically and jointly predicted from these speaker-side cues?

**Methodology**
The study draws on French multimodal conversational corpora (PACO and Cheese!) annotated for feedback occurrences. Three modalities of the speaker's own production are used as predictors: morpho-syntactic features (e.g., syntactic completion, discourse markers), prosodic features (pitch, pauses, intensity), and mimogestural features (head and hand gestures, gaze). Building on a fine-grained labeling scheme that separates generic from specific feedback and further subdivides specific feedback by valence and expectedness, a computational classifier is trained to jointly predict feedback position and type from these multimodal cues, with feature-importance analysis used to identify which modalities matter most for which feedback subtype.

**Findings**
Listener feedback is not randomly timed or shaped but is systematically triggered by identifiable multimodal configurations in the speaker's own production: different combinations of prosodic, syntactic, and gestural cues differentially predict generic versus the finer-grained specific-feedback subtypes. This supports treating feedback as jointly speaker-driven and listener-produced, since much of its timing and form appears to be anticipated in the structure of the speaker's discourse production itself.

**Results**
The paper reports that its model constitutes the state of the art for this joint position-and-type feedback prediction task on French corpora. Specific accuracy/F1 tables were not retrievable in this session — the ScienceDirect version is paywalled and the open-access HAL PDF (hal-04551398) returned an access-denied page — so this results summary is necessarily based on the abstract and secondary sources (ILCB research summary, SSRN listing) rather than the full quantitative results, and depth here is limited by access rather than by the paper's actual content.

**Conclusion**
The authors conclude that multimodal, speaker-side production cues carry systematic, exploitable structure for predicting listener feedback, reframing conversational feedback as a phenomenon that is substantially encoded in advance within the speaker's own multimodal discourse production. For the thesis, this is a close Layer 1 (RQ1) methodological parallel: predicting a discourse state (here, feedback-worthy moments; for the thesis, discourse-planning difficulty) from multimodal speaker cues, and it also informs Layer 2's multimodal feature-fusion design.

*Sources: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0167639324000384), [ILCB feature](https://www.ilcb.fr/35074-2/), [HAL record](https://hal.science/hal-04551398) (PDF access blocked in this session)*
