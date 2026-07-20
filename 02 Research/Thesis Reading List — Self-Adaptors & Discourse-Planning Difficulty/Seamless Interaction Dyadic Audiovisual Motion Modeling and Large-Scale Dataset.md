---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://ai.meta.com/research/publications/seamless-interaction-dyadic-audiovisual-motion-modeling-and-large-scale-dataset/
Year: 2025
Should Refer:
  - Architecture
  - Benchmark
Reading Status: Reading
Venue: Technical Report
Code Link: https://github.com/facebookresearch/seamless_interaction
Relatedness: Strongly
Topic: Dataset-Corpus
snippet: "This is the substrate for the proposed benchmark. Audit participant/session/vendor identifiers, naturalistic and improvised splits, SMPL-H validity, keypoints, occlusion fields, download size, and CC-BY-NC release obligations."
Authors: "Vasu Agrawal, Akinniyi Akinyemi, Kathryn Alvero, Morteza Behrooz, Julia Buffalini, Fabio Maria Carlucci, Joy Chen, Junming Chen, Zhang Chen, Shiyang Cheng, Praveen Chowdary, Joe Chuang, Antony D'Avirro, Jon Daly, Ning Dong, Mark Duppenthaler, Cynthia Gao, Jeff Girard, Martin Gleize, Sahir Gomez, Hongyu Gong, Srivathsan Govindarajan, Brandon Han, Sen He, Denise Hernandez, Yordan Hristov, Rongjie Huang, Hirofumi Inaguma, Somya Jain, Raj Janardhan, Qingyao Jia, Christopher Klaiber, Dejan Kovachev, Moneish Kumar, Hang Li, Yilei Li, Pavel Litvin, Wei Liu, Guangyao Ma, Jing Ma, Martin Ma, Xutai Ma, Lucas Mantovani, Sagar Miglani, Sreyas Mohan, Louis-Philippe Morency, Evonne Ng, Kam-Woh Ng, Tu Anh Nguyen, Amia Oberai, Benjamin Peloquin, Juan Pino, Jovan Popovic, Omid Poursaeed, Fabian Prada, Alice Rakotoarison, Alexander Richard, Christophe Ropers, Safiyyah Saleem, Vasu Sharma, Alex Shcherbyna, Jia Shen, Jie Shen, Anastasis Stathopoulos, Anna Sun, Paden Tomasello, Tuan Tran, Arina Turkatenko, Bo Wan, Chao Wang, Jeff Wang, Mary Williamson, Carleigh Wood, Tao Xiang, Yilin Yang, Zhiyuan Yao, Chen Zhang, Jiemin Zhang, Xinyue Zhang, Jason Zheng, Pavlo Zhyzheria, Jan Zikes, Michael Zollhoefer"
tags:
  - Dataset
  - Multimodal
  - Conversation
Tier: Essential
Assigned Date: 2026-07-12
Citation Key: "agrawal2025seamless"
Local PDF: "[[99 Assets/Papers/CVPR 2027/Seamless Interaction Dyadic Audiovisual Motion Modeling and Large-Scale Dataset.pdf]]"
Zotero URI: "zotero://select/library/items/JQHFKSQF"
Zotero PDF URI: "zotero://open-pdf/library/items/YCRA9QA6"
---
# Reading objective

This is the substrate for the proposed benchmark. Audit participant/session/vendor identifiers, naturalistic and improvised splits, SMPL-H validity, keypoints, occlusion fields, download size, and CC-BY-NC release obligations.

## Extraction checklist

- [ ] Exact task, inputs, outputs, and claimed novelty
- [ ] Dataset size, split unit, annotations, and license
- [ ] Strongest relevant baseline and metric protocol
- [ ] Component or loss worth reproducing
- [ ] Failure mode or limitation that affects [[CVPR 2027 Submission Roadmap — Temporal Self-Contact]]

## Notes

## Reading Summary

**Abstract**
Seamless Interaction is a Meta AI paper that introduces both a massive dyadic conversation dataset and a suite of generative models built on it. The dataset contains over 4,000 hours of face-to-face interaction footage from more than 4,000 participants across diverse relationship types and conversational contexts, and it is used to train models that generate dyadic motion — body gestures and facial expressions — aligned with speech and with an interlocutor's visual behavior.

**Research Question**
The paper asks how to build AI systems that both comprehend and generate the dyadic embodied dynamics of face-to-face communication — that is, models that produce socially appropriate nonverbal motion (gesture, facial expression) conditioned jointly on a person's own speech and on their conversational partner's behavior — and what data infrastructure is required to make that possible at scale.

**Methodology**
The Seamless Interaction Dataset was collected as naturalistic and improvised dyadic sessions structured around prompts grounded in interpersonal psychological theory, covering a wide range of conversational topics, interpersonal stances, and participant relationships, with train/dev/test/private-test splits and rich metadata (personality annotations, participant-relationship labels, transcripts). Interactions are represented multimodally via parametric human body/hand models, a facial "imitator" representation, speech, and text transcripts. On top of this substrate the authors build a family of dyadic motion models — an audio-only model, an audiovisual model that conditions on both participants' behavior, and joint/cascaded face+body variants — with additional controllable variants for emotion, expressivity level, and semantically relevant gesture generation, including an integration with an LLM for driving generated speech and gesture. Evaluation combines structured human studies (dedicated Dyadic Body and Dyadic Face rating protocols covering dimensions such as lifelikeness, clarity of intent, and turn-taking behavior) with automatic metrics, plus an analysis of how well the automatic metrics track the human judgments.

**Findings**
The paper's central qualitative claim is that dyadic nonverbal behavior can be modeled as a joint function of a speaker's own speech and their partner's visual and acoustic behavior, and that doing so — rather than modeling each participant's motion in isolation — is necessary for generating motion that reads as socially coordinated (e.g., appropriately timed turn-taking, listener backchannels). The controllability experiments further show that emotional tone, expressivity, and gesture semantics can be manipulated as separate conditioning axes without retraining the base motion model.

**Results**
The dataset scale itself is the headline quantitative result: 4,000+ hours of footage, 4,000+ participants, and roughly 1,300 distinct conversational and activity-based prompts, released under a CC BY-SA 4.0 license. The paper reports automatic-metric comparisons across its audio-only, audiovisual, and joint/cascaded model variants and correlates these metrics against the human-study ratings; the full quantitative tables are extensive and were not exhaustively extracted here due to how the source renders results tables, but the qualitative pattern reported is that audiovisual conditioning and joint face+body modeling outperform audio-only and independently modeled baselines on the human-rated dimensions.

**Conclusion**
The authors position Seamless Interaction as infrastructure for socially intelligent AI — virtual agents, telepresence, and multimodal content analysis — enabled primarily by dataset scale and multimodal richness rather than by a single novel architecture. For the thesis, this is chiefly a Layer 2 (RQ2/RQ3) resource: its scale and multimodal representation (video + audio + transcript + parametric pose) make it a plausible additional domain for cross-dataset generalization benchmarking, and its dyadic conditioning architecture is a relevant precedent for multimodal fusion design, though it does not address self-adaptors, cognitive load, or discourse-planning difficulty directly, so its DENSO-positioning contribution is methodological (scale, fusion) rather than thematic.

*Sources: [arXiv:2506.22554](https://arxiv.org/abs/2506.22554), [arXiv HTML](https://arxiv.org/html/2506.22554v2), [AI at Meta publication page](https://ai.meta.com/research/publications/seamless-interaction-dyadic-audiovisual-motion-modeling-and-large-scale-dataset/)*

