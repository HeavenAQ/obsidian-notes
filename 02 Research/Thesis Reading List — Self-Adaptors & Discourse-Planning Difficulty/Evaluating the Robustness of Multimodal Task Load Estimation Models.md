---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2024.1371181/full
Year: 2024
Should Refer:
  - None
Reading Status: Skim/Skip
Venue: (journal)
Relatedness: not relevant
Topic: Domain Generalization
snippet: ""
Authors: Andreas Foltyn, Jessica Deuschel, Nadine R. Lang-Richter, Nina Holzer, Maximilian P. Oppelt
Tags:
  - Domain Generalization
  - Cognition
  - Multimodal
  - Benchmark
Tier: Recommended
Assigned Date: 2026-07-06
---
Found via **ADABase** (2026-07-05 night) — adjacent literature / direct continuation by the same author group (Foltyn and Oppelt are ADABase co-authors); ADABase's own reference/citation list beyond the fetched excerpt was not accessible in this session, so this is treated as an adjacent-literature substitute rather than a confirmed citation. Tests how multimodal task-load estimation models trained on one scenario (e.g., n-back) generalize to a different scenario (e.g., driving) — i.e., a direct robustness/domain-generalization follow-up to ADABase's own multi-paradigm design. Strongly supports RQ3.

## Reading Summary

**Abstract**

Foltyn, Deuschel, Lang-Richter, Holzer and Oppelt (Fraunhofer IIS; Frontiers in Computer Science, 2024) ask whether multimodal cognitive/task-load estimation models trained on one scenario remain accurate when deployed on a different scenario. Using the ADABase dataset's two task paradigms (an n-Back working-memory test and a k-Drive simulated-driving task), they train classical ML and deep learning classifiers, with several fusion strategies, on n-Back data only and evaluate both in-distribution (n-Back) and under distribution shift (k-Drive).

**Research Question**

RQ1: how do model architecture, fusion strategy, and preprocessing choices affect predictive performance and uncertainty calibration when train and test data come from the same scenario? RQ2: how does an out-of-scenario distribution shift affect classification accuracy and the quality of uncertainty estimates?

**Methodology**

Data come from ADABase (46 usable subjects), with binary low/high task-load labels derived from task difficulty level rather than self-report, across modalities ECG, EDA/GSR, EMG, eye tracking, PPG, respiration, and skin temperature. Classical ML models (logistic regression, SVM, XGBoost) and three deep architectures (FCN, ResNet-1D, ResNet1D-GRU) were trained with subject-wise 4x4 nested cross-validation and Optuna-based hyperparameter search. Three fusion schemes were compared: late fusion (averaging unimodal predictions), concatenation/early fusion of features, and gated fusion (learned per-modality weighting, deep-learning only). Metrics included F1-score, AUROC, Expected Calibration Error (ECE), and a Rejection Ratio quantifying whether uncertainty correlates with misclassification.

**Findings**

In-distribution (n-Back), eye tracking is consistently the strongest unimodal signal and classic ML (especially logistic regression) tends to edge out deep learning; early/feature fusion gives the best raw accuracy and calibration. Under the k-Drive distribution shift, the ranking inverts: eye tracking degrades sharply (likely due to uncontrolled lighting), other modalities such as ECG and EMG improve, deep models like ResNet1D-GRU catch up to or beat logistic regression, and early fusion loses its robustness advantage while late fusion holds up best across both scenarios. Subject-wise normalization consistently helps most in both scenarios.

**Results**

In-distribution, the best unimodal F1 is eye tracking at about 0.85 across most models, and the best fusion result is logistic regression with concatenated features (F1 0.86, calibration error 7.92). Under the k-Drive shift, unimodal F1 rises for ECG/EMG (e.g., EMG reaches 0.86 for both logistic regression and ResNet1D-GRU) while eye tracking falls to 0.71-0.79; late-fusion logistic regression reaches F1 0.82 (calibration error 29.25), and multimodal fusion overall still does not beat the best single modality but narrows the robustness gap. A normalization ablation shows subject-wise z-scoring clearly outperforming no normalization and global normalization in both n-Back (F1 0.86 vs. 0.58/0.75) and k-Drive (F1 0.76 vs. 0.65/0.08).

**Conclusion**

The authors conclude that no single model or fusion method is simultaneously best in-distribution and most robust to scenario shift; late fusion offers a good practical compromise between accuracy and uncertainty calibration, and modality importance itself is scenario-dependent, arguing against relying on a single "best" unimodal signal for deployment. This feeds directly into Layer 2 methodology and RQ3 (cross-domain/cross-dataset generalization): it is essentially a domain-shift stress test companion to ADABase/REVELIO (same author group already in this reading list), and its late-fusion-for-robustness finding is directly transferable to the thesis's planned cross-dataset benchmarking of the contact-aware self-adaptor pipeline. It also strengthens the DENSO positioning by demonstrating exactly the kind of scenario-transfer (lab task to driving) evaluation a driver cognitive-load application would require.

*Sources: *[*https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2024.1371181/full*](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2024.1371181/full)