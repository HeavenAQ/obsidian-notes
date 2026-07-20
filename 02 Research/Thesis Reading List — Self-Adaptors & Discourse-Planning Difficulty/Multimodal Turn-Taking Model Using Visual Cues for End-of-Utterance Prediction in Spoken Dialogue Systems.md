---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://www.isca-archive.org/interspeech_2023/kurata23_interspeech.html
Year: 2023
Should Refer: []
Reading Status: Reading
Assigned Date: 2026-07-19
Venue: Interspeech
Topic: Gesture-and-Turn-Taking
snippet: ""
Authors: F. Kurata, M. Saeki, S. Fujie, Y. Matsuyama
tags:
  - Turn-Taking
  - Multimodal
  - Gesture
Tier: Recommended
Local PDF: "[[99 Assets/Papers/CVPR 2027/Multimodal Turn-Taking Model Using Visual Cues for End-of-Utterance Prediction in Spoken Dialogue Systems.pdf]]"
Zotero URI: "zotero://select/library/items/FCNLJRUI"
Citation Key: "kurataMultimodalTurnTakingModel2023"
Zotero PDF URI: "zotero://open-pdf/library/items/PKMSSXID"
---
Found via **Multilingual Turn-taking Prediction Using Voice Activity Projection** (2026-07-07 night) — cited in the reference list as a recent multimodal end-of-utterance model. Adds visual cues (eye/mouth/head movement via 3D-CNN) to acoustic/linguistic turn-taking prediction, with eye movements shown to contribute most — directly relevant to Layer 2's multimodal fusion design and Gesture & Turn-Taking topic.

## Reading Summary

**Abstract**
This ISCA Best Student Paper Award-winning study proposes a multimodal model for predicting end-of-utterance probability in spoken dialogue systems, adding visual cues (gaze/eye, mouth, and head movement) to conventional acoustic and linguistic information. An end-to-end 3D-CNN visual feature extractor is used to capture these cues jointly, and an ablation study quantifies each visual cue's relative contribution.

**Research Question**
Do visual cues meaningfully improve end-of-utterance (turn-taking) prediction beyond acoustic and linguistic information alone, and — among gaze/eye, mouth, and head movement — which visual cue contributes most to that improvement?

**Methodology**
End-of-utterance prediction is framed as a supervised sequence classification task that fuses acoustic, linguistic (verbal), and visual streams. Rather than hand-crafting separate features per visual cue, an end-to-end 3D-CNN operates directly on video to extract a joint representation spanning eye, mouth, and head movement. The full multimodal model (visual + acoustic + verbal) is compared against an acoustic+verbal-only baseline using AUC, and a targeted ablation study isolates the individual contribution of each visual cue category to prediction performance.

**Findings**
Visual cues carry information that is genuinely complementary to acoustic and linguistic cues for anticipating when a speaker is about to yield the conversational floor. Among the visual channels, eye movements are the single most informative cue for this prediction — more so than mouth or head movement — suggesting gaze dynamics are a privileged signal for discourse/turn-transition planning that should be prioritized when compute or data constraints force a choice among modalities in a fusion architecture.

**Results**
Adding multimodal visual cues improved the AUC for end-of-utterance prediction from 0.896 (acoustic + verbal baseline) to 0.920. The ablation study confirmed that eye movements contributed more to this gain than either mouth or head movement.

**Conclusion**
The authors conclude that end-to-end visual feature extraction, fused with acoustic and linguistic streams, produces a measurable and meaningful improvement in turn-taking prediction, with gaze as the standout visual channel. For the thesis, this feeds Layer 2 (RQ2) directly as a video+audio+transcript multimodal-fusion precedent for discourse-timing prediction, and it is conceptually adjacent to Layer 1's behavioral question, since turn-transition planning and discourse-planning difficulty are closely related conversational-timing phenomena; the eye-movement-dominance finding is a useful prior for weighting modalities in the thesis's own fusion design.

*Sources: [ISCA Archive](https://www.isca-archive.org/interspeech_2023/kurata23_interspeech.html), [Fujie Lab publications](https://www.fujielab.org/works/), [Waseda Interspeech 2023 award note](https://www.teai-waseda.jp/en/interspeech-2023-best-student-award/)*