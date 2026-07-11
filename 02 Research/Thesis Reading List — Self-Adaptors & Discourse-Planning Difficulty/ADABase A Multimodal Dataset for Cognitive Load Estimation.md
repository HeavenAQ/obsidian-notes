---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://www.mdpi.com/1424-8220/23/1/340
Year: 2023
Should Refer:
  - None
Reading Status: Skim/Skip
Venue: (journal)
Relatedness: not relevant
Topic: Dataset / Corpus
snippet: ""
Authors: M. P. Oppelt, et al.
tags:
  - Dataset
  - Cognition
  - Multimodal
Tier: Recommended
Assigned Date: 2026-07-05
---
Found via REVELIO (tonight's paper) — direct reference and the dataset REVELIO extends. Published in Sensors (MDPI), 2023. Driver cognitive-load dataset: 51 subjects, n-back + simulator-based k-drive test, physiological (ECG, EDA, EMG, PPG, respiration, skin temp, eye tracking) + behavioral (facial action units) signals with NASA-TLX. Directly on-target for the DENSO driver cognitive-load monitoring application (RQ3 / PhD positioning).

## Reading Summary

**Abstract**

ADABase (Autonomous Driving Cognitive Load Assessment Database) is a multimodal dataset paper for driver cognitive-load detection. Oppelt, Foltyn, Deuschel, Lang(-Richter), Holzer, Eskofier and Yang induce single- and dual-task cognitive load of increasing intensity in 51 subjects while recording ECG, EDA, EMG, PPG, respiration, skin temperature, eye-tracking, and facial-video action units, alongside subjective (NASA-TLX) and performance (reaction time, recall/precision) measures, then train and evaluate machine-learning models to distinguish cognitive-load levels from single- and multi-modal inputs.

**Research Question**

Can cognitive load be reliably induced, measured across physiological, behavioral, subjective, and performance modalities, and then classified/regressed from that multimodal signal — and which combinations of modalities are most informative, in both a standard laboratory paradigm (n-back) and a novel simulator-based driving paradigm (k-drive) meant to better match real semi-autonomous-vehicle interaction?

**Methodology**

51 subjects (24 female, 26 male, 1 undisclosed) each underwent two cognitive-load-inducing paradigms: the well-established n-back working-memory test at three difficulty levels (single- and dual-task), and a novel simulator-based "k-drive" test with multiple levels of load plus a secondary in-cabin entertainment-system task, motivated by real semi-autonomous driving. Throughout, the authors recorded physiological signals (ECG, EDA, EMG, PPG, respiration, skin temperature), eye-tracking, facial video (from which action units are extracted), subjective NASA-TLX ratings after each phase, and task performance (recall, precision, reaction time for primary and secondary tasks). They extracted statistical/expert features per modality, ran significance tests (Holm-corrected t-tests) across load levels, and then trained classifiers (SVM, kNN, XGBoost) for binary low-vs-high load classification, a three-class under/medium/over-load classification, and an XGBoost regressor predicting a novel continuous cognitive-load target combining subjective and performance measures. A public 30-subject release (of the 51 recorded) is provided.

**Findings**

Statistically significant, modality-specific changes with increasing cognitive load were found across nearly all recorded signals (heart rate up, heart-rate variability down, skin temperature down, several facial action units changing in both n-back and k-drive), confirming that both paradigms successfully induced graded cognitive load. Combining modalities (multimodal fusion) improved classification over any single modality, and moving from simple binary (low/high) classification to finer-grained multi-level classification substantially degrades performance, indicating cognitive-load level is much easier to detect coarsely than to grade precisely. Eye-tracker features alone performed comparably well across both task paradigms, while some modalities (e.g., EDA) helped more for one paradigm (k-drive) than the other (n-back).

**Results**

For the harder three-class (under/medium/over-load) task, low-load was correctly identified about 92% of the time on k-drive using all features, but high-load was frequently confused with medium-load (only ~69% correctly classified), yielding an F1 of 0.72 ± 0.09 using all features on k-drive. For the continuous cognitive-load regression target, an XGBoost regressor achieved R² of 0.51 ± 0.07 using only performance-based targets, R² of 0.48 ± 0.08 using only NASA-TLX subjective ratings, and a higher R² using a linear combination of both (exact combined value not confirmed beyond this point, as the fetched text was truncated before that figure and the discussion/conclusion sections).

**Conclusion**

ADABase's main contribution is the dataset and evaluation protocol itself — a rare same-subject, multi-paradigm (n-back + driving-simulator), multi-modality cognitive-load corpus released publicly to enable future multimodal algorithm development — rather than a single best-performing model. For the thesis, this is a Layer 2/RQ3 resource: it is one of the direct data sources that REVELIO (already in this reading list) builds on for cross-domain generalization benchmarking, and its explicit driver/autonomous-vehicle framing is directly relevant to the DENSO driver-cognitive-load-monitoring positioning bridge; the coarse-vs-fine-grained classification difficulty finding is also a useful methodological caution for any self-adaptor/discourse-difficulty labeling scheme with more than two levels.

*Sources: *[*MDPI Sensors 23(1):340*](https://www.mdpi.com/1424-8220/23/1/340)*, *[*PMC9823940*](https://pmc.ncbi.nlm.nih.gov/articles/PMC9823940/)* (note: full text retrieval was truncated before the Discussion/Conclusion and reference sections in this session, so this summary is based on the Abstract, Introduction, Methods, and the Results sections that were retrieved)*