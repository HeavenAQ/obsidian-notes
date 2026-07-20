---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2604.07651
Year: 2026
Should Refer: []
Reading Status: Reading
Assigned Date: 2026-07-19
Venue: arXiv
Topic: Domain-Generalization
snippet: ""
Authors: Keito Inoshita, Nobuhiro Hayashida, Akira Imanishi
tags:
  - Cognition
  - Multimodal
  - Driving
Tier: Recommended
Local PDF: "[[99 Assets/Papers/CVPR 2027/Cognitive-Causal Multi-Task Learning with Psychological State Conditioning for Assistive Driving Perception.pdf]]"
Zotero URI: "zotero://select/library/items/D83C57UW"
Citation Key: "inoshitaCognitiveCausalMultiTaskLearning2026"
Zotero PDF URI: "zotero://open-pdf/library/items/6LXJEE8C"
---
Found via **Multimodal Brain-Computer Interface for In-Vehicle Driver Cognitive Load Measurement: Dataset and Baselines (CL-Drive)** (2026-07-17 night) — adjacent-literature substitute, not a direct reference (CL-Drive's own accessible reference list predates 2023). Proposes CauPsi, a causal multi-task framework for assistive driving perception that estimates a driver psychological/internal-state signal from facial expression and body posture (Cross-Task Psychological Conditioning) and propagates it through a causal task chain from environmental perception to behavior recognition, evaluated on the AIDE dataset (82.71% mean accuracy). Same driver cognitive/psychological-state-monitoring research thread as tonight's CL-Drive paper, and a recent (2026) continuation relevant to RQ3's DENSO driver cognitive-load positioning.

## Reading Summary

**Abstract**
CauPsi is a cognitive-science-grounded causal multi-task learning framework for advanced driver-assistance systems that explicitly models the hierarchical dependencies among four recognition tasks — Traffic Context Recognition (TCR), Vehicle Context Recognition (VCR), Driver Emotion Recognition (DER), and Driver Behavior Recognition (DBR) — rather than treating them as flat, independent objectives. It introduces a Causal Task Chain that propagates upstream predictions to downstream tasks via learnable prototype embeddings, and Cross-Task Psychological Conditioning (CTPC), which estimates a self-supervised psychological-state signal from the driver's facial expression and body posture and injects it into all four tasks. Evaluated on the AIDE dataset, CauPsi reaches 82.71% mean accuracy with only 5.05M parameters.

**Research Question**
Can explicitly modeling the causal, hierarchical structure of driver cognition — environmental perception cascading into an internal psychological state, which in turn modulates behavior — improve multi-task assistive driving perception relative to flat multi-task baselines, and can a psychological-state signal learned without explicit annotation (purely from facial and postural cues) meaningfully condition both downstream and upstream task predictions?

**Methodology**
The framework is evaluated on the AIDE dataset (multi-view driver monitoring: front, left, right, and in-cabin camera views, annotated for TCR, VCR, DER, and DBR). Two mechanisms are combined: the Causal Task Chain, which passes upstream task predictions to downstream tasks through learnable prototype embeddings to realize a differentiable "cognitive cascade" from perception to behavioral regulation; and CTPC, which derives an unsupervised psychological-state embedding from facial expression and body posture and conditions every task head on it, including the upstream perception tasks. CauPsi is compared against prior multimodal multi-task baselines on AIDE (the same benchmark used by frameworks such as MMTL-UniAD), and ablations remove each mechanism in turn to isolate its contribution.

**Findings**
Structuring the tasks causally — rather than as independent flat objectives — better mirrors how driving cognition actually cascades from environment to internal state to behavior, and this structuring measurably improves recognition, particularly for the downstream, harder-to-predict tasks (behavior and emotion). Notably, injecting the self-supervised psychological-state signal into upstream perception tasks (not just downstream ones) also helps, supporting the claim that internal driver state has a genuine modulatory effect on even early-stage perception. Despite receiving no explicit psychological annotation, the learned signal shows systematic, task-label-dependent structure, suggesting it captures a real latent construct rather than noise.

**Results**
CauPsi achieves 82.71% mean accuracy on AIDE with 5.05M parameters, a +1.0% overall improvement over prior work, with disproportionate gains on Driver Emotion Recognition (+3.65%) and Driver Behavior Recognition (+7.53%). Ablations show the Causal Task Chain is particularly load-bearing for behavior recognition (removing it drops DBR by 8.70 points), while removing CTPC most hurts Vehicle Context Recognition (−1.65%) and DBR (−2.95%).

**Conclusion**
The authors conclude that combining a cognitive-science-motivated causal task structure with a self-supervised psychological-conditioning signal yields a more accurate and more parameter-efficient driver-perception system than flat multi-task alternatives. For the thesis, this is a strong Layer 2 (RQ2/RQ3) touchpoint: it is a very recent (2026), camera-based, facial-expression-and-posture-driven estimate of driver internal/psychological state — structurally close to the thesis's own contact-aware self-adaptor pipeline — and it directly reinforces the DENSO driver-cognitive-load positioning.

*Sources: [arXiv:2604.07651](https://arxiv.org/abs/2604.07651), [arXiv HTML](https://arxiv.org/html/2604.07651)*
