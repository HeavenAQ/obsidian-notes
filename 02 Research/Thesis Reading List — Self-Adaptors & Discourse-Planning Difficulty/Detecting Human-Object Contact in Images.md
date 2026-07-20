---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://hot.is.tue.mpg.de/
Year: 2023
Should Refer: []
Reading Status: Reading
Venue: CVPR
Code Link: https://github.com/yixchen/HOT
Topic: Self-Contact-Hand-Face-CV
snippet: ""
Authors: Y. Chen, S. K. Dwivedi, M. J. Black, D. Tzionas
tags:
  - Contact-Aware
Tier: Recommended
Assigned Date: 2026-07-11
Local PDF: "[[99 Assets/Papers/CVPR 2027/Detecting Human-Object Contact in Images.pdf]]"
Zotero URI: "zotero://select/library/items/YWFRU2T9"
Citation Key: "chenDetectingHumanObjectContact2023"
Zotero PDF URI: "zotero://open-pdf/library/items/E2NHEW7V"
---
Found via **Generative Modeling of Shape-Dependent Self-Contact Human Poses** (2026-07-07 night) — direct reference [5]. Introduces HOT, a dataset and detector for human-object (rather than self-) contact in images. Directly relevant methodological precedent for the contact-detection stage of the thesis's contact-aware pipeline (RQ2), even though it targets human-scene rather than hand-face contact.

## Reading Summary

**Abstract**
Chen, Dwivedi, Black, and Tzionas (CVPR 2023) introduce HOT ("Human-Object conTact"), the first dataset and detector for full-body human-object contact in single color images. HOT combines automatically generated contact annotations from the PROX dataset (via 3D mesh proximity) with manually annotated in-the-wild images from V-COCO, HAKE, and Watch-n-Patch, yielding 35,287 images and 162,267 contact-area annotations, each labeled with the involved SMPL-X body part. They train a part-attention-guided contact detector that outputs 2D contact heatmaps and per-pixel body-part labels from a single RGB image.

**Research Question**
The paper asks whether a general-purpose detector can localize where, and on which body part, a person is in physical contact with an object or scene from a single 2D image — extending prior work that only handled narrow cases such as foot-ground or hand-object contact — and whether such a detector can rival specialized part-specific contact detectors.

**Methodology**
HOT has two complementary parts: "HOT-Generated" (20,205 images / 95,179 contact areas), where contact is computed automatically from PROX's 3D SMPL-X meshes and 3D scene meshes via vertex-proximity and normal-compatibility thresholds, and "HOT-Annotated" (15,082 images / 67,088 contact areas from V-COCO, HAKE, and Watch-n-Patch), where 12 trained professional annotators drew polygons around contact regions with two rounds of quality control. The detector uses a dilated ResNet-50 backbone feeding a two-branch decoder: an attention branch that predicts a per-body-part attention mask (supervised early in training by rendered SMPL-X part segmentations) and a contact branch that predicts per-pixel contact; the two are combined via a part-attention operation. Evaluation uses semantic contact accuracy (SC-Acc), contact accuracy (C-Acc), mean IoU, and weighted IoU, benchmarked against ResNet+UperNet and ResNet+PPM segmentation baselines, with ablations removing or de-supervising the attention branch, cross-subset transfer experiments, and comparisons against the part-specific detectors ContactDynamics (foot) and ContactHands (hand). Notably, the annotation protocol explicitly instructs annotators to *ignore* human-human and self-contact, meaning HOT is scoped to external object/scene contact only.

**Findings**
The part-attention mechanism is the key contributor to performance, letting the model reason about contact using the context of nearby body parts and surrounding scene rather than raw pixel classification alone. Manual "HOT-Annotated" data transfers well to the automatically generated "HOT-Generated" set, but not vice versa, and combining both consistently boosts performance, indicating the two annotation strategies are complementary rather than redundant. The general-purpose full-body detector performs on par with specialized foot- and hand-contact detectors, suggesting one model could serve as a drop-in replacement for multiple part-specific pipelines. Downstream, predicted contact improves contact-driven 3D human pose fitting and helps a state-of-the-art 3D body-scene contact estimator generalize to in-the-wild images.

**Results**
On "HOT-Annotated," the full model reaches SC-Acc 40.7 / C-Acc 70.7 / mIoU 0.215 / wIoU 0.260, versus the best segmentation baseline (ResNet+UperNet) at 35.1 / 62.6 / 0.195 / 0.227; on the combined "Full Set" the model scores 36.4 / 66.3 / 0.209 / 0.251. Removing the attention branch drops SC-Acc to 24.1 (Full Set: 19.4), confirming its necessity. Against part-specific detectors, HOT matches ContactDynamics on foot-ground contact (59.2% vs. 58.6%) and ContactHands on hand-object contact recognition (63.5% vs. 62.2% at IoU 0.4). Using HOT's predicted contact to replace PROX's handcrafted contact heuristics in pose optimization reduces vertex-to-vertex error to 172.3mm (vs. 183.3mm with no contact constraint, 174.0mm with PROX's heuristic, and 163.0mm with ground-truth contact). Extending the RICH/BSTRO 3D contact model's training data with HOT's lifted pseudo-3D labels raises its in-the-wild F1 from 0.231 to 0.636.

**Conclusion**
The authors conclude that a general, part-attention-guided full-body contact detector can match specialized part-specific detectors while additionally generalizing across contact types, and that combining automatically generated and manually annotated data is an effective, scalable recipe for training such a detector; they flag transformer-based architectures and self-/human-human contact as future work. For the thesis, this is a direct Layer 2 (RQ2) methodological precedent for the contact-detection stage of the contact-aware self-adaptor pipeline — its architecture (part-attention over a segmentation backbone) and evaluation metrics (SC-Acc/C-Acc/mIoU) are transferable to hand-face contact detection. Critically, HOT's annotators were explicitly told to ignore self-contact, so the thesis's contact-aware detector is precisely extending this contact-detection paradigm into the territory HOT deliberately left out.

*Sources: [arXiv abstract](https://arxiv.org/abs/2303.03373), [arXiv PDF (full text)](https://arxiv.org/pdf/2303.03373), [project page](https://hot.is.tue.mpg.de/), [GitHub](https://github.com/yixchen/HOT)*