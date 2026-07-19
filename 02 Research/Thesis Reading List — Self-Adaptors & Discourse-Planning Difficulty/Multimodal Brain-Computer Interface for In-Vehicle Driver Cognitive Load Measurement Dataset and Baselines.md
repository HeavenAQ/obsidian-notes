---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2304.04273
Year: 2024
Should Refer: []
Reading Status: Reading
Venue: IEEE-Trans-Intell-Transp-Syst
Topic: Dataset-Corpus
snippet: ""
Authors: P. Angkan, B. Behinaein, Z. Mahmud, A. Bhatti, D. Rodenburg, P. Hungler, A. Etemad
tags:
  - Dataset
  - Cognition
  - Multimodal
Tier: Recommended
Assigned Date: 2026-07-17
---
Found via **Real-Time Driver Cognitive Workload Recognition: Attention-Enabled Learning With Multimodal Information Fusion** (2026-07-11 night) — adjacent-literature substitute, not a direct reference (the source paper is IEEE Xplore–paywalled and its reference list could not be retrieved in this session). Introduces CL-Drive, a multimodal driver cognitive-load dataset (EEG, ECG, EDA, and eye-tracking from 21 participants across 9 escalating-complexity simulated-driving scenarios, with subjective cognitive-load ground truth every 10 seconds) plus binary/ternary classification baselines. Same driver-cognitive-load research thread as tonight's paper; a strong candidate dataset for RQ3 cross-domain generalization benchmarking and for strengthening the DENSO driver-cognitive-load positioning.

## Reading Summary

**Abstract**
This paper introduces CL-Drive, a multimodal dataset for measuring driver cognitive load, built around a brain-computer-interface (BCI) framing: EEG is the primary signal, supplemented by ECG, electrodermal activity (EDA), and eye-tracking (gaze), all collected from 21 participants driving in an immersive vehicle simulator. Unlike prior cognitive-load datasets that induce load with an unrelated secondary task, CL-Drive treats driving itself, at nine escalating complexity levels, as the primary source of load, and gathers dense subjective cognitive-load ratings (PAAS 1-9 scale) every 10 seconds throughout each 3-minute scenario. The paper also reports extensive binary and ternary classification baselines across classical ML and deep models, under both 10-fold and leave-one-subject-out (LOSO) evaluation.

**Research Question**
The paper asks whether driver cognitive load, induced by the driving task itself rather than by a secondary task, can be reliably measured from multimodal physiological and ocular signals, and how classification performance varies across modality combinations, model families, and generalization regimes (within-subject 10-fold versus cross-subject LOSO).

**Methodology**
Twenty-three participants (data retained for 21) drove a full-motion, three-screen immersive simulator through nine pre-built scenarios of increasing complexity (highway driving, night driving, snow, tennis-ball and slalom precision challenges, 90-degree and 3-point turns, narrow-alley navigation), each lasting 3 minutes and separated by 2-minute rest and re-baselining periods. EEG was recorded with a 4-channel Muse S headband (256 Hz), ECG and EDA with Shimmer3 wearables, and gaze with Tobii Pro Glasses 2. Every 10 seconds, participants verbally reported their cognitive load on the 9-point PAAS scale, yielding dense ground truth (an explicit design response to sparser labeling in prior cognitive-load datasets). After band-pass/notch filtering and per-modality artifact handling, the authors extracted domain-standard features per 10-second segment (e.g., EEG spectral-band power, spectral entropy, Hjorth parameters, Lempel-Ziv complexity, Higuchi fractal dimension; HRV-derived ECG features; phasic/tonic EDA statistics; saccade/fixation/pupil gaze statistics) and additionally trained VGG-style and ResNet-style 1D-CNNs directly on raw, per-modality signals with late feature-level fusion for the multimodal condition. Labels were collapsed into binary (low/high) and ternary (low/medium/high) classes. Nine classical classifiers (AdaBoost, Decision Tree, Naive Bayes, k-NN, LDA, Random Forest, SVM, XGBoost, MLP) and the two deep architectures (on features and on raw signals) were evaluated under both 10-fold cross-validation and the more demanding LOSO protocol, across all single- and multi-modality combinations.

**Findings**
Across every evaluation setting, adding modalities monotonically improved average accuracy, with the full four-modality combination (EEG+ECG+EDA+Gaze) consistently outperforming tri-, bi-, and uni-modal subsets, confirming that auxiliary peripheral and ocular signals carry complementary cognitive-load information beyond EEG alone. Tree-ensemble methods (XGBoost, Random Forest) trained on hand-crafted features were the strongest classical models, while among deep networks the VGG-style architecture trained on extracted features generally outperformed its ResNet counterpart and both raw-signal variants under 10-fold and LOSO-binary settings (though ResNet-on-raw-data became competitive in the harder LOSO-ternary setting). As expected, ternary classification was substantially harder than binary, and LOSO evaluation - the more realistic test of subject-generalization - was consistently harder than 10-fold cross-validation, underscoring that cross-subject generalization remains the harder, more clinically/practically relevant regime.

**Results**
The best 10-fold binary accuracy was 83.67% (F1 82.05%), achieved by XGBoost using all four modalities; the best LOSO binary accuracy dropped to 76.17% (F1 71.72%), achieved by the feature-based VGG network on EEG+ECG+EDA. For 10-fold ternary classification the best accuracy was 74.08% (XGBoost, EEG+ECG+EDA+Gaze), while LOSO ternary - the hardest setting - topped out at 64.53% (ResNet on raw EEG+ECG+Gaze). The consistent accuracy gap between 10-fold and LOSO (roughly 7-10 points across settings) quantifies the generalization cost of held-out-subject evaluation on this task.

**Conclusion**
The authors conclude that driving-induced cognitive load can be measured with reasonable accuracy from EEG plus auxiliary wearable and ocular signals, and they position CL-Drive (publicly released) as filling a gap in the literature: prior driver cognitive-load work either lacked physiological multimodality or used sparse, end-of-task labels, whereas CL-Drive offers dense, frequent self-report labels tied to a primary driving task rather than a secondary distractor task. For the two-layer thesis, this paper is a Layer 2 / RQ3 resource rather than a methods contribution in itself: it is a concrete, well-documented driver-cognitive-load dataset and multimodal baseline suite that could serve as one leg of a cross-domain generalization benchmark (e.g., paired with conversational self-adaptor/cognitive-load data), and its explicit driving-simulator framing and BCI emphasis directly strengthens the DENSO driver-cognitive-load positioning bridge in the PhD framing.

*Sources: [arXiv:2304.04273v2 (HTML full text)](https://arxiv.org/html/2304.04273v2), [arXiv abstract page](https://arxiv.org/abs/2304.04273v2), [GitHub - Prithila05/CL-Drive](https://github.com/Prithila05/CL-Drive)*
