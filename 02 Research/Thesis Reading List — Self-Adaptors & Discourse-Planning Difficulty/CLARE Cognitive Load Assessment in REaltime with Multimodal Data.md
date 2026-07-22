---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2404.17098
Year: 2024
Should Refer: []
Reading Status: Reading
Assigned Date: 2026-07-20
Venue: arXiv
Topic: Dataset-Corpus
snippet: ""
Authors: Anubhav Bhatti, Prithila Angkan, Behnam Behinaein, Zunayed Mahmud, Dirk Rodenburg, Heather Braund, P. James Mclellan, Aaron J. Ruberto, Geoffery Harrison, Daryl Wilson, Adam Szulewski, Dan Howes, Ali Etemad, Paul Hungler
tags:
  - Dataset
  - Cognition
  - Multimodal
Tier: Recommended
Local PDF: "[[99 Assets/Papers/CVPR 2027/CLARE Cognitive Load Assessment in REaltime with Multimodal Data.pdf]]"
Zotero URI: "zotero://select/library/items/9ZZM7XKI"
Citation Key: "bhattiCLARECognitiveLoad2024"
Zotero PDF URI: "zotero://open-pdf/library/items/FWGN4467"
---
Found via **AVCAffe: A Large Scale Audio-Visual Dataset of Cognitive Load and Affect for Remote Work** (2026-07-11 night) — adjacent-literature substitute, not a verified direct reference (AVCAffe's reference list was not accessible in this session). Multimodal (physiological + gaze) real-time cognitive-load assessment dataset/benchmark from 24 participants with self-reported ground truth — a methodologically close cousin of AVCAffe's cognitive-load labeling approach, useful as a comparison point for RQ3 cross-dataset benchmarking design.

## Reading Summary

**Abstract**
CLARE introduces a multimodal dataset and benchmark for real-time cognitive load assessment, collecting synchronized ECG, EDA, EEG, and gaze data from 24 participants performing the MATB-II multitasking operator-workload simulation (a standard aviation/human-factors task battery combining system monitoring, tracking, resource management, and scheduling sub-tasks), with self-reported cognitive load ratings collected every 10 seconds as ground truth.

**Research Question**
Can real-time, self-reported cognitive load be predicted from physiological and gaze signals at fine temporal granularity (10-second windows), which modality combinations are most informative, and how well do models generalize to unseen participants (leave-one-subject-out) versus within-subject folds?

**Methodology**
24 participants completed four 9-minute MATB-II sessions of varying task complexity, with a self-reported cognitive load score collected every 10 seconds; signals (ECG, EDA, EEG, gaze) were segmented into matching 10-second windows and both handcrafted statistical/modality-specific features and raw-signal deep learning pipelines (CNN and Transformer architectures) were used to predict a binarized (high vs. low, threshold at self-report ≥5) cognitive load label. The authors benchmark eight classical ML algorithms (Gradient Boosting, LightGBM, LDA, Logistic Regression, MLP, Random Forest, SVM, XGBoost) alongside CNN and Transformer deep models, across every single-modality and multimodal combination of the four signal types, under two evaluation protocols: 10-fold cross-validation and leave-one-subject-out (LOSO), reporting accuracy and F1.

**Findings**
Across both evaluation schemes the Transformer model using all four modalities was the most accurate and most consistent performer, and multimodal combinations consistently beat single-modality inputs, though which single modality was strongest differed by protocol: ECG was most informative for within-subject (10-fold) prediction, while EEG was most informative for cross-subject (LOSO) generalization. The much larger performance drop from 10-fold to LOSO (e.g., best accuracy falling from roughly 85% to roughly 73%) is itself a substantive finding, illustrating how much harder cross-subject cognitive-load generalization is than within-subject prediction — directly relevant to any claim about generalizing a cognitive-load or difficulty marker across new people.

**Results**
In 10-fold cross-validation, the Transformer using all modalities reached 85.58% accuracy (F1 = 81.18), the best tri-modal combination (EDA+EEG+Gaze) reached 84.93% (F1 = 78.18), and the best single modality (ECG via CNN) reached 78.45% (F1 = 76.41). In LOSO cross-validation, performance dropped substantially: the Transformer with all modalities reached 72.70% accuracy (F1 = 69.46), the best single modality (EEG) reached 67.77% (F1 = 57.03), and most classical ML models performed only modestly above chance. All models met real-time constraints, with inference times under the 10-second ground-truth collection interval.

**Conclusion**
The authors conclude that multimodal physiological and gaze fusion improves real-time cognitive load classification over any single modality, that deep learning (particularly Transformer-based fusion) outperforms classical ML especially in the harder cross-subject setting, and that generalization to unseen individuals remains a substantially harder problem than within-subject prediction. For the thesis this is a Layer 2/Layer 3 (RQ2 multimodal fusion design, RQ3 cross-dataset/cross-subject generalization) dataset and benchmark paper: while its modalities (ECG/EDA/EEG/gaze) differ from the thesis's planned video+audio+transcript self-adaptor pipeline, its 10-fold-vs-LOSO evaluation protocol and its explicit demonstration of the generalization gap are directly relevant templates for the cross-dataset/domain-generalization benchmarking the thesis and DENSO positioning require.

*Sources: [arXiv:2404.17098](https://arxiv.org/abs/2404.17098), [arXiv HTML v2](https://arxiv.org/html/2404.17098v2)*
