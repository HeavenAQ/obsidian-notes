---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Authors: Wenqi Jia, Miao Liu, Hao Jiang, Ishwarya Ananthabhotla, James M. Rehg, Vamsi Krishna Ithapu, Ruohan Gao
Year: 2024
Venue: CVPR
Paper Link: https://arxiv.org/abs/2312.12870
Topic: Multimodal-Disfluency-and-Discourse
Tier: Recommended
tags:
  - multimodal
  - conversation
  - egocentric
  - speaking-listening
Reading Status: Reading
Assigned Date: 2026-07-21
Local PDF: "[[99 Assets/Papers/CVPR 2027/The Audio-Visual Conversational Graph From an Egocentric-Exocentric Perspective.pdf]]"
Zotero URI: "zotero://select/library/items/JFJRPVJK"
Citation Key: "jiaAudioVisualConversationalGraph2024"
Zotero PDF URI: "zotero://open-pdf/library/items/AZ3BDHCY"
---

Found via tonight's SocialGesture read (reference [22] in its related-work section, shared co-authors Jia/Rehg): this paper proposes a unified audio-visual framework for jointly predicting speaking/listening conversational behavior for the camera wearer and social partners, directly relevant to the thesis's multimodal fusion methodology (RQ2) for modeling discourse-planning-related behavior in conversation.

## Reading Summary

**Abstract**
This paper introduces the Ego-Exocentric Conversational Graph Prediction problem: given only a single egocentric (camera-wearer) video of a multi-person conversation, jointly infer not just the camera wearer's own speaking/listening behavior but also the speaking/listening relationships among all the *other* social partners in the scene — exocentric interactions the camera cannot directly "see" from a third-person view. The authors propose AV-CONV, a unified audio-visual self-attention model that fuses per-person visual (cropped head) features, multi-channel audio, and spatial position masks across time, subjects, and modalities to predict this complete directed conversational graph.

**Research Question**
Can a model infer the complete multi-party conversational structure — who is speaking to or listening to whom, including relationships between people other than the camera wearer — using only egocentric audio-visual input, and which combination of audio, visual, and spatial-positional cues, and which attention mechanisms (temporal, cross-subject, cross-modal), are necessary to do so accurately?

**Methodology**
The authors define a Conversational Graph with an egocentric bipartite subgraph (camera wearer ↔ each partner) and an exocentric non-bipartite subgraph (partner ↔ partner), with four binary edge attributes per pair (speaking-to and listening-to, in each direction). They repurpose the Egocentric Concurrent Conversations Dataset (50 participants, 10–30 minute sessions in groups of five, each wearing a headset with an Intel SLAM camera and a six-microphone array, ~20 hours total) to generate ground-truth labels for these edges. The AV-CONV model extracts per-person head-crop visual features and location-aware multi-channel audio features via separate ResNet-18 backbones, then applies a three-part Conversational Attention module (cross-time, cross-subject, and cross-modality/global-local self-attention, in the spirit of space-time attention for video understanding) to fuse them, followed by separate classifiers for each of the four egocentric and four exocentric edge-attribute types (with pairwise-concatenated features used for the exocentric case). It is compared against two adapted baselines built from prior single-task methods (SAAL, an active-auditory-attention localizer, and an ASL+3D-layout heuristic) and evaluated with accuracy and mean Average Precision.

**Findings**
Jointly modeling all conversational sub-tasks together, rather than solving them one at a time, meaningfully helps performance, and each of the three attention types (temporal, cross-subject, cross-modal) contributes complementary information — cross-time attention gives the largest single gain, reflecting the value of aggregating audio evidence over neighboring frames for detecting who is speaking. Modality ablations show audio is essential for speaking-related predictions while positional/visual cues matter more for listening-related predictions, and combining audio with spatial position masks captures most of the achievable performance, with head-image appearance adding a smaller further gain.

**Results**
AV-CONV reaches 86.15% egocentric accuracy and 81.04% exocentric accuracy on average across all eight edge-attribute types, outperforming the SAAL baseline by roughly 4.45% on average on egocentric tasks and outperforming the heuristic ASL+Layout baseline by a large margin on both egocentric and exocentric tasks (overall Ego-Exo average accuracy of 83.51% for AV-CONV vs. 60.40% for ASL+Layout). The full three-way attention model beats a no-attention "direct concatenation" baseline by about 4.15% (egocentric) and 4.61% (exocentric) average accuracy, and beats single-attention-type ablations as well; the modality ablation shows audio+mask together reaches 85.03%/79.78% (ego/exo average accuracy), close to the full three-modality model's 86.15%/81.04%.

**Conclusion**
The authors conclude that exocentric conversational structure is substantially recoverable from purely egocentric audio-visual input when audio, visual appearance, and spatial position are fused via time-subject-modality self-attention, opening a path toward richer social-scene understanding from a single wearable camera. For the thesis, this is a Layer 2/RQ2 methodology paper: its audio-visual-positional multimodal fusion architecture (and its explicit joint egocentric/exocentric multi-party framing) is a directly transferable template for fusing video, audio, and interlocutor-position information to model discourse-relevant conversational states (e.g., co-occurrence of self-adaptors with speaking/listening roles and turn structure), and its dataset/task design is a useful reference point for building a multimodal, multi-party evaluation setting for self-adaptor and discourse-planning-difficulty modeling.

*Sources: [arXiv:2312.12870](https://arxiv.org/abs/2312.12870), [CVPR 2024 Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Jia_The_Audio-Visual_Conversational_Graph_From_an_Egocentric-Exocentric_Perspective_CVPR_2024_paper.html), [Project page](https://vjwq.github.io/AV-CONV/)*
