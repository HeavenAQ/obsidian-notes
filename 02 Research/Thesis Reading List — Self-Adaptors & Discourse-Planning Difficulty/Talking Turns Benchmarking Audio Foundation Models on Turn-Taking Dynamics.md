---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2503.01174
Year: 2025
Should Refer: []
Reading Status: Reading
Venue: arXiv
Topic: Gesture-and-Turn-Taking
snippet: ""
Authors: Siddhant Arora, Zhiyun Lu, Chung-Cheng Chiu, Ruoming Pang, Shinji Watanabe
tags:
  - Turn-Taking
  - Benchmark
Tier: Recommended
Assigned Date: 2026-07-15
---
Found via **Predicting Cognitive Load from Speech and Interaction Dynamics in Dyadic Conversations** (2026-07-05 night) — adjacent literature, not a direct reference (that paper's own reference list was not accessible in this session since its full text could not be retrieved as machine-readable text). Benchmarks audio foundation models specifically on turn-taking dynamics (overlap, backchannel, switch timing), continuing the same "conversational-dynamics-as-signal" research line the source paper uses for cognitive-load prediction. Relevant to RQ1/RQ2 (turn-taking as a discourse-planning-adjacent signal).

## Reading Summary

**Abstract**
This paper asks whether the current generation of audio foundation models (audio FMs) — increasingly used to build spoken dialogue systems — actually understand and can perform turn-taking, the fluent alternation of speaking and listening that makes human conversation feel natural rather than either interrupt-heavy or full of dead air. The authors introduce an evaluation protocol built around a supervised "judge" model trained to predict turn-taking events from human-human conversation, then use it to run the first comprehensive study of existing spoken dialogue systems and several audio FMs, finding systematic weaknesses such as failing to recognize when to speak, interrupting too aggressively, and rarely backchanneling.

**Research Question**
Can recently proposed audio foundation models understand, predict, and correctly perform turn-taking events (yielding the floor, interrupting, backchanneling, staying silent) at a level approaching natural human-human conversation, and how can this capability be measured systematically?

**Methodology**
The authors define a fine-grained taxonomy of conversational units and events: Inter-Pausal Units (IPUs, continuous speech bounded by silences longer than 200ms), pauses, gaps, overlaps, backchannels, and interruptions. They train a causal "judge" model that consumes audio (using features from a pretrained Whisper model) and, in 40ms increments, predicts which of five turn-taking labels applies next — NA (not speaking), BC (backchannel), I (interruption), T (turn change), or C (continuation) — via a softmax classifier trained on human-human conversational data. This judge is then used as an automatic evaluator: it is applied to recorded interactions from real spoken dialogue systems (the full-duplex system Moshi and a VAD-based cascaded pipeline) collected in a user study, and separately used to benchmark several open-source and proprietary audio FMs accessible via API on curated test sets built from the Switchboard corpus, measuring how well each system's turn-taking behavior (and each FM's turn-taking predictions) matches human conversational norms.

**Findings**
The user study of deployed dialogue systems shows concrete, qualitatively distinct failure modes: the full-duplex Moshi system tends to interrupt users too aggressively and often fails to recognize when the user intends to continue speaking, while the VAD-based cascaded system suffers from higher latency and excessive inactivity (dead air) because it waits for clear silence before responding. Across the broader set of benchmarked audio FMs, the consistent qualitative pattern is a substantial gap from human-level turn-taking competence: many models do not reliably know when to speak up, and backchanneling (the short acknowledgment tokens like "mm-hm" that signal active listening without claiming the floor) is particularly underdeveloped, which the authors treat as a key missing ingredient for natural-feeling conversational AI.

**Results**
The paper reports statistical characterizations of turn-taking behavior (event rates per minute, duration proportions of overlaps/gaps/backchannels) for the compared systems, and evaluates the judge model's own reliability using discrimination metrics before applying it downstream; per the abstract and available secondary-source summaries, the study finds "significant room for improvement" across both open-source and proprietary audio FMs on Switchboard-derived turn-taking prediction tests, though this session's access to the full paper (via web search summaries and the arXiv abstract rather than the complete PDF) means exact per-model numeric scores (e.g., precise F1/accuracy figures per FM) could not be independently verified from primary text and are not restated here to avoid fabrication.

**Conclusion**
The authors conclude that despite progress in conversational capability, current audio foundation models still fall meaningfully short of human-level turn-taking, and they release their evaluation platform to spur further work, with particular attention flagged for multilingual and lower-resource settings. For the thesis, this is a Layer 1/RQ1-adjacent and Layer 2/RQ2 paper: its formalized taxonomy of turn-taking events (overlaps, gaps, interruptions, backchannels) and its judge-model evaluation methodology are directly reusable for treating turn-taking irregularity as a discourse-planning-difficulty signal correlated with self-adaptor production, and its benchmark-style evaluation across multiple audio FMs is a template for the kind of cross-system/cross-domain generalization testing relevant to RQ3.

*Sources: [arXiv abstract page](https://arxiv.org/abs/2503.01174), [Moonlight literature review summary](https://www.themoonlight.io/en/review/talking-turns-benchmarking-audio-foundation-models-on-turn-taking-dynamics)*