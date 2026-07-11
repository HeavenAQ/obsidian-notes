---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://doi.org/10.1145/3581641.3584063
Year: 2023
Should Refer:
  - None
Reading Status: Read
Venue: IUI
Relatedness: Weakly
Topic: Self-Contact-Hand-Face-CV
snippet: |2-
    • to reduce the risk of contagious disease by avoiding face-touching behavior 
    • using neck-mounted wearable with an infrared camera to predict non-touch and touched behavior
Authors: H. Lim, R. Zhang, S. Pendyal, J. Jo, C. Zhang
tags:
  - Hand-Face-CV
  - Contact-Aware
Tier: Recommended
Assigned Date: 2026-07-06
---
Recommended via tonight's paper "Analysis of Face-Touching Behavior in Large Scale Social Interaction Dataset" (Beyan et al., ICMI 2020) as an adjacent-literature substitute, not a direct reference — that paper's references predate 2023. D-Touch (IUI 2023) continues the hand-face contact detection line with a neck-mounted IR-camera wearable that recognizes and even predicts fine-grained hand-face touch types, offering a sensing-based complement to the video-only detection paradigm and fine-grained touch taxonomies useful for self-contact annotation schemes.

![[99 Assets/Media/image 17.png]]

## Reading Summary

**Abstract**

Lim, Zhang, Pendyal, Jo and Zhang (Cornell SciFi Lab; IUI 2023) present D-Touch, a neck-mounted wearable with an infrared camera that looks upward at the underside of the head and hands to both recognize and, for the first time in this line of work, predict fine-grained hand-face touching activity before contact happens.

**Research Question**

Can a single neck-worn camera not only detect that a hand is near the face but distinguish which facial region is, or is about to be, touched, and how far in advance can hand-face contact be predicted to enable just-in-time behavioral intervention?

**Methodology**

A custom neck-mounted IR camera (worn like a tie-clip) streams images of the underside of the chin and head to a deep-learning classifier trained to recognize 17 "Facial-related Activities": 11 discrete touch positions (mouth, eyes, forehead, cheek, chin, etc.) plus 6 non-touch daily activities (eating, drinking, phone calls, wearing glasses). Data collection combined lab sessions with an in-the-wild deployment in 10 participants' homes with no behavioral constraints. Two questions were tested: recognition accuracy on the full 17-class taxonomy, and early prediction — given only the first ~150ms after a hand enters frame, whether the system can already forecast if an approaching hand will touch the risky facial T-zone versus perform another activity, plus a follow-up study on how much advance warning users need to actually interrupt the movement.

**Findings**

The system reliably distinguishes not just touch-vs-no-touch but which facial region or non-touch activity is occurring, using a single miniature camera rather than the two-sensor (wrist+neck) setups common in prior wearable work. Predicting the touch category from partial, early visual evidence is feasible well before the hand actually contacts the face, and a brief 150ms advance warning is enough for most users to interrupt the movement in a subsequent intervention experiment, though performance drops from lab to in-the-wild conditions due to variation in face/hand shape, movement style, and neck length across users.

**Results**

Recognition of the 17-class taxonomy achieved 92.1% average accuracy in controlled settings. Early prediction of T-zone touch versus other Facial-related Activities reached 82.12% accuracy at 150ms after the hand first appeared in frame in the lab study, dropping to 72.3% in the unconstrained 10-participant home deployment. A separate intervention experiment found a 72.1% success rate for participants stopping a face-touch when alerted 150ms in advance.

**Conclusion**

The authors conclude D-Touch is the first wearable system to both recognize fine-grained touch location and predict touch behavior before contact, and they discuss form-factor, energy, and ecological-validity limitations (bulky prototype, controlled-alarm-timing artifacts) as next steps toward a deployable single-device intervention system. For the thesis, this is a Layer 2 methodological input on contact-aware sensing: a strong, real-time hand-face-touch pipeline offering both a fine-grained touch taxonomy useful for annotation schemes and a working example of early/predictive detection, complementary to the video-only self-contact CV methods (TUCH, DICE, Decaf) already in the reading list. It is less central to RQ1/RQ2 directly but is useful background for real-world deployment framing and for the DENSO-relevant angle of low-latency, wearable behavioral sensing.

*Sources: *[*https://dl.acm.org/doi/10.1145/3581641.3584063*](https://dl.acm.org/doi/10.1145/3581641.3584063)*, *[*https://czhang.org/assets/pdf/DTouch.pdf*](https://czhang.org/assets/pdf/DTouch.pdf)