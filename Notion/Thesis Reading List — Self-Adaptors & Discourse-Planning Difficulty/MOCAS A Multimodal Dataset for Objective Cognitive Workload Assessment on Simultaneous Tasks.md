---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2210.03065
Year: 2024
Should Refer:
  - None
Reading Status: Skim/Skip
Venue: (journal)
Relatedness: not relevant
Topic: Dataset / Corpus
snippet: ""
Authors: Wonse Jo, Ruiqi Wang, Go-Eum Cha, Sangwoo Sun, Ramviyas Parasuraman Senthilkumaran, Daniel Foti, Byung-Cheol Min
Tags:
  - Dataset
  - Cognition
  - Multimodal
Tier: Recommended
Assigned Date: 2026-07-06
---
Found via **VDMoE** (2026-07-05 night) — direct reference [34]. Multimodal cognitive-workload dataset (physiological wearables + webcam behavioral features) collected from realistic CCTV-monitoring tasks with simultaneous secondary tasks, rated via NASA-TLX/ISA. Directly supports RQ3 (cognitive/task-load estimation, dataset diversity beyond driving). Published in IEEE Transactions on Affective Computing, 2024 (Venue set to generic "(journal)" — not in the fixed Venue option list).

## Reading Summary

**Abstract**

Jo, Wang, Cha, Sun, Senthilkumaran, Foti and Min (Purdue SMART Lab; IEEE Transactions on Affective Computing, 2024, originally posted to arXiv in 2022) introduce MOCAS, an open multimodal dataset for objective cognitive workload assessment collected during a realistic simulated CCTV-monitoring task rather than an abstract lab game, paired with physiological sensors, a facial webcam, and rich subjective and personality annotations.

**Research Question**

Can objective cognitive workload be assessed from multimodal physiological and behavioral signals collected during an ecologically realistic human-machine monitoring task, and does incorporating personality traits and emotional state alongside physiological/behavioral signals improve on existing game-based cognitive-workload datasets and models?

**Methodology**

21 of 30 recruited participants performed a simultaneous CCTV-monitoring task, watching 1, 2, or 4 real-time video feeds from patrol robots at 3 speed levels (9 workload conditions) and clicking to flag abnormal objects, with a scoring scheme (+1 correct, -3 miss) to keep engagement realistic. Physiological data (EEG via Emotiv Insight; PPG/GSR/HR/IBI/skin temperature via Empatica E4) and behavioral data (facial video, eye-aspect-ratio, facial action units via MediaPipe, mouse movement) were recorded alongside subjective workload ratings (NASA-TLX, ISA), emotion ratings (SAM), and Big-Five personality questionnaires. For validation, the authors ran three-class (low/medium/high) cognitive-workload classification using unimodal LSTM models per modality and a Late-Fusion LSTM (per-modality LSTM followed by a 1-D CNN over concatenated features) for multimodal fusion, evaluated under both a trial-independent (shuffled, 5-fold cross-validation) and a subject-independent (leave-one-subject-out) protocol.

**Findings**

The CCTV task reliably elicited distinct workload levels, validated by both NASA-TLX/ISA self-reports (clear, statistically significant main effects of camera number and robot speed) and by classification performance. Personality traits show only modest correlations with physiological/behavioral means and with task performance, suggesting individual differences matter but are not dominant. Multimodal fusion clearly outperforms any single modality, EEG band power and eye-aspect-ratio are relatively strong unimodal cognitive-workload indicators, and subject-independent (leave-one-subject-out) generalization is markedly harder than within-subject evaluation — a concrete illustration of the domain-generalization gap central to this thesis's RQ3.

**Results**

Two-way repeated-measures ANOVA on NASA-TLX showed strong main effects of camera number (F(2,38)=49.13, p<.001, partial eta-squared=.72) and robot speed (F(2,38)=21.03, p<.001, partial eta-squared=.53), with no interaction. Friedman tests on ISA (chi-square=88.34, p<.001) and SAM arousal/valence (chi-square=49.16 / 63.28, both p<.001) confirmed distinct emotional/workload elicitation. For three-class workload classification, trial-independent multimodal fusion reached 72.33% accuracy (best unimodal: EEG band-power 69.81%, heart rate 68.84%); under the harder subject-independent (LOSO) protocol, fusion accuracy dropped to 46.13% (best unimodal about 37.55% for EEG band-power), underscoring poor cross-subject generalization for the paper's baseline model.

**Conclusion**

The authors position MOCAS as the first open dataset combining physiological and behavioral cognitive-workload signals with both cognitive and emotional annotations plus personality/background metadata from a realistic, non-game task, and they identify subject-independent generalization as the key open problem for future modeling on the dataset. For the thesis, MOCAS is primarily an RQ3 (domain-generalization/benchmark diversity) resource: a concrete, accessible, non-driving cognitive-workload dataset with a documented LOSO generalization gap, useful either as an additional cross-dataset generalization target for the contact-aware/self-adaptor pipeline or as a comparison point for the DENSO-relevant driver cognitive-load framing, since a monitoring-task dataset shows the same subject-generalization weakness that domain-generalization methods need to address.

*Sources: *[*https://arxiv.org/abs/2210.03065*](https://arxiv.org/abs/2210.03065)*, *[*https://arxiv.org/html/2210.03065v2*](https://arxiv.org/html/2210.03065v2)*, *[*https://dl.acm.org/doi/abs/10.1109/TAFFC.2024.3414330*](https://dl.acm.org/doi/abs/10.1109/TAFFC.2024.3414330)