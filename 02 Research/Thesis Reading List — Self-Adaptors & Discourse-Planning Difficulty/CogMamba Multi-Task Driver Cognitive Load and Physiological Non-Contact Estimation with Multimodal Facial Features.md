---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12473509/
Year: 2025
Should Refer: []
Reading Status: Skim/Skip
Venue: (journal)
Relatedness: not relevant
snippet: Non-contact driver cognitive-load + heart/respiratory rate estimation from facial RGB video (Mamba temporal model) — key citation for the DENSO driver cognitive-load framing in Future Work and outreach.
Authors: ""
tags:
  - Cognition
  - Multimodal
Tier: Recommended
Assigned Date: 2026-07-04
---
## Reading Summary

**Abstract**

CogMamba (Xie & Guo, Sichuan University; Sensors 25(18):5620, Sept 2025) is a multi-task, non-contact model that estimates driver cognitive load, heart rate (HR), and respiratory rate (RR) jointly from RGB facial video. It targets the autonomous-driving takeover scenario: when drivers engaged in non-driving-related tasks (NDRTs) must rapidly retake control, their cognitive load determines safety, but existing detection relies on invasive sensors or eye-tracking alone. CogMamba uses multimodal facial features plus the Mamba state-space architecture for efficient temporal modeling.

**Research Question**

Can driver cognitive load and physiological state be estimated accurately, efficiently, and without any contact sensors, from in-cabin RGB video alone — and does such a model generalize across datasets?

**Methodology**

Rather than full-frame video, CogMamba extracts compact multimodal facial inputs: an STMap (spatial-temporal map of facial color signals, the standard rPPG representation) plus key facial features — landmarks, eye regions, and mouth area. A Mamba-based temporal encoder captures local and global temporal dependencies with linear complexity; a bidirectional fusion mechanism shares representations across tasks, and a lightweight two-layer MLP head branches into three outputs: HR regression, RR regression, and binary cognitive-load classification. Training uses task-specific losses — truncated cross-entropy for cognitive load (robust to noisy subjective questionnaire labels) and smooth-L1-style regression losses for HR/RR — combined as L_total = L_cog + λL_hr + λL_rr. Evaluation is on two public driver datasets (eDream and MCDD), including feature-ablation and a cross-dataset generalization test.

**Findings**

Joint multi-task estimation works: physiological signals (HR/RR) and cognitive load share facial evidence, and parameter sharing with branch-only task heads suppresses task interference. Using STMap + selected facial regions instead of raw frames cuts parameters and compute substantially, making in-vehicle deployment plausible. Most relevant to the thesis, the model reportedly retains performance when trained on one dataset and tested on another — evidence that compact physiologically grounded facial representations generalize across domains better than end-to-end video features.

**Results**

The paper claims superior performance over existing methods on both eDream and MCDD and "excellent robustness" in the cross-dataset generalization test, with ablations showing each facial feature stream contributes. Exact table values (accuracy/MAE figures) were not retrievable in this pass — the accessible MDPI capture ended at the methods section and PMC was blocked — so specific numbers should be confirmed against the PDF; the summary above reflects the abstract, methods, and figure captions.

**Conclusion**

The authors conclude that non-contact, video-only, multi-task driver-state estimation is practical and generalizes across datasets, providing a path for real-world driver monitoring. For the thesis this is the keystone application citation for the DENSO framing: it validates cognitive-load estimation from nonverbal/facial video in the driving domain (RQ3 generalization + PhD positioning), and its Mamba-based temporal design connects directly to the state-space modeling thread (cf. DF-Mamba from the DENSO IT Lab group). Note it ignores hand/body behavior — the contact-aware self-adaptor channel is exactly the complementary signal the thesis adds.

*Sources: *[*https://www.mdpi.com/1424-8220/25/18/5620*](https://www.mdpi.com/1424-8220/25/18/5620)* (partial full text), *[*https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12473509/*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12473509/)