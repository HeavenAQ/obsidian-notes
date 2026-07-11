---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://ieeexplore.ieee.org/document/10447676/
Year: 2024
Should Refer:
  - None
Reading Status: Skim/Skip
Venue: ICASSP
Relatedness: not relevant
Topic: Domain Generalization
snippet: ""
Authors: Jiyao Wang, Ange Wang, Haolong Hu, Kaishun Wu, Dengbo He
Tags:
  - Domain Generalization
  - Cognition
Tier: Recommended
Assigned Date: 2026-07-07
---
Found via **MOCAS: A Multimodal Dataset for Objective Cognitive Workload Assessment on Simultaneous Tasks** (2026-07-06 night) — adjacent literature (MOCAS's own reference list predates 2023); this ICASSP 2024 paper by the same broader driver-cognitive-load research thread as VDMoE and the Hierarchical Style-Aware DG paper already in this list (Jiyao Wang et al.) introduces a multi-source domain-generalization benchmark (CogDG-ECG) across three cognitive-load datasets, directly addressing the cross-dataset generalization gap MOCAS's own LOSO results expose. Strong RQ3 relevance and DENSO-adjacent (driver cognitive load via ECG).

## Reading Summary

**Abstract**

This ICASSP 2024 paper by Wang, Wang, Hu, Wu, and He tackles cognitive load estimation from electrocardiography (ECG), a non-invasive alternative to EEG that is increasingly feasible via wearables and steering-wheel sensors. It identifies that ECG-based cognitive-load models trained in one dataset/environment fail to generalize to another due to domain shift (different subjects, devices, and load-inducing tasks), and proposes CogDG-ECG, a plug-in framework for multi-source domain generalization (MSDG), evaluated on a newly introduced cross-dataset benchmark.

**Research Question**

Can a cognitive-load estimation model trained on ECG from multiple source domains generalize to a completely unseen target domain (different dataset/subjects/task) without any access to target-domain data during training — i.e., true domain generalization rather than domain adaptation?

**Methodology**

CogDG-ECG adds two regularizers on top of a substitutable backbone encoder: (1) Adversarial Domain Alignment, which uses a gradient-reversal-layer domain discriminator to force the encoder toward domain-invariant features; and (2) Uncertainty Variation Estimation, which perturbs the learned feature distribution with Gaussian noise scaled by its own variance and uses a contrastive (InfoNCE-style) loss to keep the synthesized "out-of-distribution" features plausible. The three losses (classification, adversarial, contrastive) are jointly optimized. The benchmark combines two public datasets (DMTD: automated-driving cognitive load; CLAS: mental-arithmetic-induced load) and one proprietary dataset (MNBD: n-back task during manual driving), using a leave-one-dataset-out protocol (two combined as source domains, one held out as target). Comparisons span classic ML (SVM, KNN, LDA, LightGBM), deep learning (ANN, LSTM, TCN), and domain-generalization baselines (AD, DSU, IFL), scored on Accuracy, F1, and Sensitivity averaged over 5 seeded runs with paired t-tests.

**Findings**

Domain-generalization baselines generally beat plain ML/DL models, but CogDG-ECG's combination of adversarial alignment and uncertainty-based feature augmentation outperforms all of them on most targets. An ablation shows neither regularizer alone is sufficient — removing either measurably degrades performance, indicating the two mechanisms address complementary aspects of the domain-shift problem (alignment removes existing domain variance; uncertainty estimation prepares the model for unseen variance). The authors also note that a larger source dataset does not automatically help when it is redundant relative to the target domain's characteristics (observed with DMTD), and that the ideal number of synthesized auxiliary features (K) is dataset-dependent.

**Results**

CogDG-ECG achieves the best or statistically indistinguishable-from-best accuracy on all three leave-one-out targets: CLAS accuracy 74.06% (best baseline DSU: 72.28%, p<0.05), DMTD accuracy 61.21% (best baseline: 60.03%), and MNBD accuracy 69.33% (best baseline: 68.63%), with parallel gains in F1 and Sensitivity on CLAS and MNBD reaching significance. The ablation table shows accuracy dropping from 74.06%/61.21%/69.33% (full method) to 70.22%/58.84%/67.10% (without the uncertainty term) and 73.37%/60.34%/67.96% (without the adversarial term) across CLAS/DMTD/MNBD respectively.

**Conclusion**

The authors conclude that an end-to-end, plug-in domain-generalization framework meaningfully improves cross-dataset robustness for ECG-based cognitive load estimation, and that the new MSDG protocol/benchmark itself is a useful contribution for future work in this space. For the thesis, this paper is a direct RQ3 anchor: it is a concrete, quantified precedent for cross-domain generalization of a physiological cognitive-load signal (rather than video), and its leave-one-dataset-out benchmarking protocol is a template the thesis's own domain-generalization evaluation (Layer 2) could adapt. It is also DENSO-adjacent in spirit — driver cognitive-load estimation is the explicit DENSO-relevant application named in the thesis plan, even though this paper uses ECG rather than the planned video/audio/transcript modalities.

*Sources: *[*ICASSP 2024 IEEE Xplore listing*](https://ieeexplore.ieee.org/document/10447676/)*, *[*full-text PDF (author's site)*](https://personal.hkust-gz.edu.cn/hedengbo/assets/publicationPDFs/Wang_ICASSP_2024a.pdf)