---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Authors: Vimal Mollyn, Chris Harrison
Year: 2024
Venue: UIST
Paper Link: https://arxiv.org/abs/2509.01786
Topic: Self-Contact-Hand-Face-CV
Tier: Recommended
tags:
  - touch-detection
  - on-body-sensing
  - ar-vr
Reading Status: Reading
Assigned Date: 2026-07-19
Local PDF: "[[99 Assets/Papers/CVPR 2027/EgoTouch On-Body Touch Input Using AR-VR Headset Cameras.pdf]]"
Zotero URI: "zotero://select/library/items/M2V3PVGL"
Citation Key: "mollynEgoTouchOnBodyTouch2024"
Zotero PDF URI: "zotero://open-pdf/library/items/VDZ4RW8I"
---

Found via tonight's BioTUCH read (direct reference): EgoTouch is cited in BioTUCH's related work as a camera-based approach to on-body touch detection, offering a vision-only contrast to BioTUCH's bioimpedance sensing — useful for weighing sensing-modality tradeoffs for the contact-detection stage of the thesis's pipeline.

## Reading Summary

**Abstract**
EgoTouch demonstrates high-accuracy, sensor-free detection of bare-hand on-body ("on-skin") touch input for AR/VR using only the RGB camera already built into modern XR headsets. The system infers not just touch onset/offset but rich metadata — force (light vs. hard), finger identity, angle of attack, and rotation — and is shown to be robust across lighting conditions, skin tones, and body motion such as walking.

**Research Question**
Can a single egocentric RGB camera, without any special instrumentation of the user's skin or hands, reliably and richly detect bare-skin touch events — accurately enough to support practical on-body touch interfaces — using purely visual cues such as shadows and skin deformation?

**Methodology**
Ground-truth touch labels (presence and force) were collected using a custom touch sensor worn along the index finger and palm that stays invisible to the camera, time-synced with egocentric RGB video from 15 participants spanning diverse skin tones and hair densities across varied lighting conditions and activities, including walking. A vision model is trained to map camera frames to touch state, force level, finger identity, contact angle, and rotation without manual annotation, since labels come directly from the worn sensor. The system is evaluated under three protocols — person-specific models, cross-user models with no calibration, and cross-user models with limited personal calibration data — situating it against prior camera- and sensor-based on-body/skin touch approaches such as ActiTouch, SkinTrack, and PressureVision.

**Findings**
Touch on skin can be inferred with high fidelity from visual shadow and skin-deformation cues alone, provided the training data has precise, high-resolution ground truth from a co-registered physical sensor. A modest amount of per-user calibration closes most of the gap between a fully person-specific model and a generic cross-user model, indicating that camera-only on-skin touch sensing is close to being a practically deployable, generalizable input modality rather than a person-specific novelty.

**Results**
The system achieves greater than 96% touch detection accuracy with roughly a 5% false-positive rate, and 98% accuracy for light-vs-hard force classification. F1 scores are .992 for person-specific models, .878 for cross-user models with no calibration, and .957 with limited personal calibration data.

**Conclusion**
The authors conclude that camera-only, sensor-free on-skin touch sensing has reached a level of accuracy and robustness sufficient to serve as a general-purpose AR/VR input modality. For the thesis, this is a direct Layer 2 (RQ2) methodological reference point for the contact-detection stage of the pose → hand-face-contact pipeline: it demonstrates a pure-vision alternative to bioimpedance-based contact sensing (cf. BioTUCH), and its cross-user F1 (.878 without calibration vs. .992 person-specific) is a concrete, relevant data point for RQ3's domain-generalization framing of contact detection across individuals.

*Sources: [ACM DL](https://dl.acm.org/doi/10.1145/3654777.3676455), [arXiv HTML](https://arxiv.org/html/2509.01786v1), [Project page](https://vimalmollyn.com/research/egotouch/)*
