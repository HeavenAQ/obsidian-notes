---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://openaccess.thecvf.com/content/CVPR2025/html/Cao_SocialGesture_Delving_into_Multi-person_Gesture_Understanding_CVPR_2025_paper.html
Year: 2025
Should Refer:
  - Architecture
  - Benchmark
Reading Status: Reading
Venue: CVPR
Code Link: https://www.irohxucao.com/SocialGesture/
Relatedness: Strongly
Topic: Gesture-Turn-Taking
snippet: "Use this as the closest CVPR precedent for a natural social-gesture benchmark. Extract its annotation pipeline, split logic, temporal localization baselines, metrics, and release strategy."
Authors: Xu Cao, Pranav Virupaksha, Wenqi Jia, Bolin Lai, Fiona Ryan, Sangmin Lee, James M. Rehg
tags:
  - Dataset
  - Gesture
  - Temporal-Action-Localization
Tier: Essential
Assigned Date: 2026-07-12
Citation Key: "cao2025socialgesture"
Local PDF: "[[99 Assets/Papers/CVPR 2027/SocialGesture Delving into Multi-person Gesture Understanding.pdf]]"
Zotero URI: "zotero://select/library/items/CVBK8KKR"
Zotero PDF URI: "zotero://open-pdf/library/items/JM2EGRWA"
---
# Reading objective

Use this as the closest CVPR precedent for a natural social-gesture benchmark. Extract its annotation pipeline, split logic, temporal localization baselines, metrics, and release strategy.

## Extraction checklist

- [ ] Exact task, inputs, outputs, and claimed novelty
- [ ] Dataset size, split unit, annotations, and license
- [ ] Strongest relevant baseline and metric protocol
- [ ] Component or loss worth reproducing
- [ ] Failure mode or limitation that affects [[CVPR 2027 Submission Roadmap — Temporal Self-Contact]]

## Notes

## Reading Summary

**Abstract**
SocialGesture introduces the first large-scale dataset built specifically for multi-person gesture analysis, addressing the fact that prior gesture-recognition datasets almost exclusively feature isolated, single-person gestures in controlled settings. The dataset supports video-based gesture recognition and temporal localization, and the authors also propose a visual question answering (VQA) benchmark to test vision-language models on social gesture understanding, finding that current models — both action-recognition backbones and VLMs — struggle substantially with the task.

**Research Question**
The paper asks whether existing gesture-recognition and vision-language architectures can detect, classify, and spatially/temporally localize deictic gestures (pointing, showing, giving, reaching) when they occur naturally between multiple interacting people, rather than in isolated single-person clips, and where current models' capabilities break down.

**Methodology**
The authors curated 9,889 gesture video clips (plus 4,304 non-gesture clips) from YouTube and Ego4D footage of natural multi-person scenarios — group social games (e.g., One Night Ultimate Werewolf), variety entertainment, educational play, product reviews, group meals, and cooking — split into 293 training and 79 test raw videos, annotated by 12 trained annotators using a three-step annotate/verify/consensus protocol. Each gesture instance carries temporal boundaries, a keyframe, initiator and target bounding boxes, a category label among the four deictic types (grounded in McNeill's gesture taxonomy), and free-text VQA-style descriptions. Three benchmark tasks are defined: (1) temporal action localization, using I3D, R(2+1)D, and VideoMAEv2 features feics into an ActionFormer localization head; (2) gesture recognition, both binary (gesture vs. non-gesture) and four-way classification, using CNN-based (TSN, TANet, SlowFast) and transformer-based (TimeSformer, MViTv2, VideoSwin, UniFormerV2) backbones, tested on both full frames and per-person cropped regions; and (3) a three-part VQA benchmark (global perception, gesture understanding, gesture localization) evaluated zero-shot and after LoRA/full fine-tuning on VLMs including Qwen2-VL, Qwen2.5-VL, InternVL-2.5, LLaVA-NeXT-Video, GPT-4o-mini, and Claude 3.7 Sonnet.

**Findings**
Across all three task families, models perform far worse than on comparable single-person or non-social benchmarks. Temporal localization is the hardest task, which the authors attribute to gestures being small, subtle sub-components of larger multi-person actions and to pretraining features (even strong ones like VideoMAEv2) not being aligned with multi-person interaction structure. Classification improves modestly when using person-cropped regions rather than full frames, suggesting the difficulty is partly about disentangling which person is performing which gesture toward whom. VLMs handle coarse perception (e.g., scene description) and gesture-type understanding reasonably but fail badly at spatial gesture localization, and even the strongest closed models never exceed 70% accuracy on the simplest sub-task (counting people in the scene), indicating a genuine gap in multi-person visual reasoning rather than a labeling artifact.

**Results**
On temporal localization with ActionFormer, average mAP (IoU 0.3–0.7) tops out at 14.73 with VideoMAEv2 features (vs. 10.73 for I3D, 7.29 for R(2+1)D), rising to 19.19 for I3D when the sliding-window stride is reduced from 16 to 4. On binary gesture-vs-non-gesture classification the best model (UniFormerV2-B/16) reaches only 84.43% accuracy despite fine-tuning on over 10,000 clips; on four-way gesture-type classification the best full-frame model (VideoSwin-L) reaches 56.18% top-1 accuracy, improving to 64.72% (UniFormerV2-B/16) when using ground-truth person crops. On the VQA benchmark, zero-shot VLM accuracy on gesture localization sub-tasks (target localization, target classification) stays below roughly 30–65% across all tested models, and LoRA/full fine-tuning of Qwen2-VL-7B improves some sub-tasks (e.g., target localization to ~49–51%) while degrading others (e.g., gesture detection accuracy dropping to ~33%), reflecting overfitting on imbalanced binary labels.

**Conclusion**
The authors conclude that multi-person social gesture understanding requires a form of incremental, relational visual reasoning — who is gesturing, toward whom, doing what — that current action-recognition architectures and VLMs largely lack, and they release SocialGesture to motivate future work on this gap. For the thesis, this is primarily a Layer 2 (RQ2) methodology precedent: it is a close analogue for building an annotated, temporally localized gesture/behavior benchmark and directly demonstrates the ActionFormer-based localization pipeline the thesis's own fidgeting-dynamics stage would use (nicely complementing tonight's ActionFormer read). It is a useful contrast case for Layer 1 (RQ1) as well, since deictic gestures are communicative/intentional whereas self-adaptors are defined by being non-instrumental — the paper's annotation and evaluation framework is transferable, but its subject matter sits conceptually on the opposite side of the instrumental/non-instrumental gesture distinction central to the thesis's behavioral hypothesis.

*Sources: [arXiv:2504.02244](https://arxiv.org/abs/2504.02244), [arXiv HTML full text](https://arxiv.org/html/2504.02244v1), [CVPR 2025 Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Cao_SocialGesture_Delving_into_Multi-person_Gesture_Understanding_CVPR_2025_paper.html), [project page](https://www.irohxucao.com/SocialGesture/)*

