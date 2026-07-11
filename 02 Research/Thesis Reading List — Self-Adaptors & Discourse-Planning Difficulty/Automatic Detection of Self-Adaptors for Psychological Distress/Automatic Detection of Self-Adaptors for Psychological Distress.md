---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://doi.org/10.1109/FG47880.2020.00032
Year: 2020
Should Refer:
  - Method
Reading Status: Read
Venue: IEEE FG
Relatedness: strongly
Topic: Self-Contact / Hand-Face CV
Dataset: self-collected
snippet: |2-
    • Kinda old but the fidgeting detection part is worth taking a look
    • using openpose’s bbox for self-adaptor detection → use segment anything instead
    • fidgeting detection: optical flow (trajectory) → FFT / Mean / Std → cycle detection
    • 35 participants → 7h 50m footages
Authors: W. Lin, I. Orton, M. Liu, M. Mahmoud
Tags:
  - Self-Adaptor
  - Distress
Tier: Essential
Assigned Date: 2026-07-01
---
## Notes

![[99 Assets/Media/image 19.png]]

### Fidgeting

![[99 Assets/Media/image 20.png]]

- Used optical flow for tracking
- FFT to convert tracks from time domain to frequency domain $[0.5Hz, \;2.5Hz]$
- 

[[Detailed explanation of the Fidget Detection]]

### Self-adaptors

![[99 Assets/Media/image 21.png]]

> All self-adaptors, except for HF, must be longer than 100 frames (around 4 seconds with the frame rate of 26). This reduces noise from detected self-adaptor events.

- used bboxes to detect each body part

### Distress Classification

![[99 Assets/Media/image 22.png]]

- applied a Random Forest to select important features from the per-video representation
- classifier
    - logistic regression based classifier (LR) using a binary threshold of 0.5; and 2) 
    - MLP with two `softmax` outputs for binary classification.
        - label smoothing is applied
        - $L_{-} n e w=L \times(1-s)+\frac{s}{n}$
            - L is the original one-hot label $[0, 1]$
            - s is the smoothing strength
            - n is the number of classes

---

## Reading Summary

**Abstract**

Psychological distress detection research has mostly used facial, head, and vocal modalities; body movement is comparatively unexplored due to sparse data and the difficulty of automatically extracting useful body features. The authors propose a hierarchical computer vision model to automatically detect self-adaptors and fidgeting (a self-adaptor subtype long associated with anxiety and depression), fuse these features with audiovisual signals via a multimodal deep denoising autoencoder and Improved Fisher Vector encoding, and show that the resulting representation can predict self-reported distress. They also introduce a new dataset of full-body interview videos with distress labels to enable this work.

**Research Question**

Can self-adaptor and fidgeting behavior be reliably detected automatically from video, and does incorporating these automatically detected fidget features improve automatic prediction of self-reported psychological distress (depression, anxiety) beyond facial/audio features alone?

**Methodology**

The authors collected a new dataset (35 participants, ~7h50m total) using a DAIC-style semi-structured interview: a human interviewer asked open-ended questions while participants were kept unaware of the study's true behavioral-analysis purpose, to preserve naturalistic behavior. Distress was measured with self-report questionnaires (PHQ-8 depression, GAD-7 anxiety, SSS-8 somatic symptoms, PSS perceived stress) and personality via the Big Five Inventory. Detection uses a two-stage hierarchical pipeline: (1) a self-adaptor detector using OpenPose/OpenFace keypoint bounding boxes to flag overlapping body regions (hand-to-face, hand-to-hand, hand-to-leg, etc.) as self-adaptor events; (2) a DYNAMIC/STATIC action classifier operating on optical-flow FFT/mean/std features in a sliding window to distinguish active fidgeting from static contact. Detector outputs were validated against manually labeled video (inter-rater reliability via Krippendorff's alpha). For distress prediction, fidget, gaze, facial Action Unit, and MFCC audio features are compressed per-frame via a multimodal Deep Denoising Auto-Encoder, aggregated per-video with a GMM + Improved Fisher Vector encoding, passed through Random Forest feature selection, then classified with logistic regression or an MLP under participant-independent 3-fold cross-validation.

**Findings**

The self-adaptor location detector achieved high precision/recall (F1 = 1.00 for hand-to-hand contact; 0.77–1.00 for other locations) with strong inter-annotator agreement (Krippendorff's alpha 0.82–1.00). The DYNAMIC/STATIC fidget-action classifier reached 83–90% accuracy. Adding automatically detected fidget features improved distress classification performance over facial-AU/gaze/audio-only baselines in most configurations, and including the “participant speaking" co-occurrence signal further improved results — suggesting fidgeting-while-speaking is a particularly informative cue. Facial Action Units remained highly predictive; MFCCs contributed comparatively little. Fidgeting was consistently useful for anxiety prediction and helpful for depression under the right configuration. Cross-dataset validation on an independent, actor-performed fidgeting dataset showed good generalization.

**Results**

Detector cross-validation: H2H self-adaptor F1 1.00, other locations 0.77–1.00; DYNAMIC/STATIC accuracy 83.3–89.5% across categories. Cross-dataset validation (Mahmoud et al. dataset): fidget-presence recall ~80% (precision 0.79), with location-specific recall of 78.4% (leg), 86.5% (hand-to-face), 78.7% (hand-to-arm), and 76.8% (hand-cross) — improving on the prior state of the art for each fidget type.

**Conclusion**

This is the first system to automatically detect self-adaptor/fidgeting behavior from full-body pose video and to show that these automatically-extracted features carry incremental predictive value for self-reported psychological distress, establishing proof-of-concept that CV-based self-adaptor detection is both technically feasible and behaviorally meaningful — directly relevant methodologically to detecting self-adaptors as markers of discourse-planning difficulty rather than only distress.

*Sources: *[*Cambridge PDF*](https://www.cl.cam.ac.uk/~mmam3/pub/FG2020_Self_Adaptors_for_Psychological_Distress.pdf)*, *[*GitHub (code)*](https://github.com/LinWeizheDragon/AutoFidgetDetection)