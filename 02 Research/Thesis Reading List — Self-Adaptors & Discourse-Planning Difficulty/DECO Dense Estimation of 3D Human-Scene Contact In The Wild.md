---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2309.15273
Year: 2023
Should Refer: []
Reading Status: Reading
Assigned Date: 2026-07-20
Venue: ICCV
Code Link: https://github.com/sha2nkt/deco
Topic: Self-Contact-Hand-Face-CV
snippet: ""
Authors: S. Tripathi, A. Chatterjee, J-C. Passy, H. Yi, D. Tzionas, M. J. Black
tags:
  - Contact-Aware
Tier: Recommended
Local PDF: "[[99 Assets/Papers/CVPR 2027/DECO Dense Estimation of 3D Human-Scene Contact In The Wild.pdf]]"
Zotero URI: "zotero://select/library/items/WZGP5U65"
Citation Key: "tripathiDECODenseEstimation2023"
Zotero PDF URI: "zotero://open-pdf/library/items/T89UN4JV"
---
Found via **Detecting Human-Object Contact in Images (HOT)** (2026-07-11 night) — adjacent-literature substitute, not a backward reference (HOT is CVPR 2023 and predates DECO; DECO is a forward continuation of the same research line, sharing authors Tzionas and Black with HOT). Extends 2D contact heatmaps to dense, per-vertex 3D human-scene contact estimation from in-the-wild images. Directly relevant methodological successor for the contact-detection stage of the thesis's contact-aware pipeline (RQ2).

## Reading Summary

**Abstract**
DECO tackles the problem of inferring dense, vertex-level 3D contact between the full human body surface and the surrounding scene or objects from a single in-the-wild RGB image. The authors first build DAMON, a new dataset of dense vertex-level 3D contact annotations on the SMPL body mesh paired with complex, naturalistic human-object and human-scene interaction images, then train DECO, a contact detector that reasons jointly about which body parts are involved and about the surrounding scene context.

**Research Question**
Can 3D contact be estimated at fine (vertex-level) resolution, on the full body surface, directly from a single arbitrary in-the-wild image — rather than being restricted to 2D contact maps, sparse body joints, or coarse pre-defined body regions — and does explicitly modeling both body-part identity and scene context improve this estimation over prior approaches?

**Methodology**
The authors crowd-source DAMON via Amazon Mechanical Turk, collecting dense, vertex-level 3D contact annotations (distinguishing human-supported vs. scene-supported contact, with semantic object labels) on top of images drawn from existing human-object interaction datasets. DECO's architecture processes an image through two parallel branches — a body-part-driven branch and a scene-context-driven branch — that attend respectively to the contacting body regions and to the surrounding scene, fused via cross-attention to predict per-vertex contact on the SMPL mesh. The model is trained with both 3D vertex-level losses (from DAMON) and, notably, is compatible with 2D contact-area annotations (from the earlier HOT dataset) as an auxiliary signal, which the baseline BSTRO cannot use. Evaluation uses precision, recall, F1, and geodesic error (cm) on three datasets: DAMON, RICH, and the out-of-domain BEHAVE dataset, benchmarked against BSTRO, POSA-PIXIE, and POSA-GT.

**Findings**
The central qualitative finding is that combining part-driven and scene-driven attention is what drives DECO's improvement over prior work, with the two components trading off precision and recall: the scene-context branch chiefly improves recall (helping detect subtler or less body-adjacent contact), while the part-driven branch chiefly improves precision. DECO also generalizes substantially better than BSTRO to unseen, out-of-domain images (BEHAVE), which the authors attribute to DECO's ability to train jointly on 2D and 3D contact labels, letting it scale with more, more-diverse training images than a 3D-label-only method like BSTRO.

**Results**
On RICH, DECO reaches F1 = 0.70 (17.92 cm geodesic error) versus BSTRO's F1 = 0.63 (18.39 cm); on DAMON, DECO reaches F1 = 0.55 (21.32 cm) versus BSTRO's F1 = 0.46 (38.06 cm) — a roughly 16 cm geodesic-error improvement; and on the out-of-domain BEHAVE set, DECO reaches F1 = 0.18 (46.33 cm) versus BSTRO's F1 = 0.04 (50.45 cm), a large relative margin that the authors read as evidence of superior generalization. DECO also strongly outperforms HOI-focused baselines CHORE (F1 = 0.08) and PHOSA on contact-derived body-object interaction metrics.

**Conclusion**
The authors conclude that dense, vertex-level 3D contact can be learned directly from in-the-wild images by combining body-part and scene-context reasoning, and that this generalizes better across datasets than prior joint- or patch-level contact methods, in part because it can exploit heterogeneous (2D and 3D) contact supervision. For the thesis, DECO is a Layer 2 (RQ2) methods paper: it is a direct methodological analog for the contact-detection stage of the self-adaptor pipeline (detecting hand-face contact rather than hand-scene/object contact), and its cross-dataset evaluation protocol (train on one set, test out-of-domain on BEHAVE) is a template for the cross-dataset generalization benchmarking the thesis needs for RQ3.

*Sources: [arXiv:2309.15273](https://arxiv.org/abs/2309.15273), [ICCV 2023 paper PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Tripathi_DECO_Dense_Estimation_of_3D_Human-Scene_Contact_In_The_Wild_ICCV_2023_paper.pdf), [project page](https://deco.is.tue.mpg.de/)*
