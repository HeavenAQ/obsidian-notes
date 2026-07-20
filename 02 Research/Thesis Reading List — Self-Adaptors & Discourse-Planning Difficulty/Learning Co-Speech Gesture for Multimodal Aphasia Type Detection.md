---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2310.11710
Year: 2023
Should Refer: []
Reading Status: Reading
Venue: EMNLP
Code Link: https://github.com/DSAIL-SKKU/Multimodal-Aphasia-Type-Detection_EMNLP_2023
Topic: Multimodal-Disfluency-and-Discourse
snippet: ""
Authors: D. Lee, S. Son, H. Jeon, S. Kim, J. Han
tags:
  - Multimodal
  - Gesture
  - Disfluency
Tier: Recommended
Assigned Date: 2026-07-17
Local PDF: "[[99 Assets/Papers/CVPR 2027/Learning Co-Speech Gesture for Multimodal Aphasia Type Detection.pdf]]"
Zotero URI: "zotero://select/library/items/2FBZXUC4"
Citation Key: "leeLearningCoSpeechGesture2023"
Zotero PDF URI: "zotero://open-pdf/library/items/J88LAVN9"
---
Recommended via tonight's paper "The Role of Self-Adaptors in Lexical Retrieval" (Sekine & Hotta 2025) as an adjacent-literature substitute, not a direct reference — that paper's own references are all psychology works predating 2023. This EMNLP 2023 paper is a computational descendant of the same research line: it models how co-speech gesture patterns reflect word-retrieval impairment, using a speech-gesture graph neural network to classify aphasia types (SOTA F1 84.2%), showing gesture features can outperform acoustic features for retrieval-difficulty-related classification.

## Reading Summary

**Abstract**
This EMNLP 2023 paper proposes a multimodal graph neural network for classifying the *type* of aphasia (a language disorder from brain damage) using both speech and co-speech gesture, rather than merely detecting the presence of aphasia as prior work had done. The model learns the correlation between disfluency-related speech content and aligned gesture patterns for each aphasia type through a heterogeneous graph encoder, then uses that correlation to sensitize textual word embeddings to gesture information before a final multimodal fusion and classification step. The approach achieves state-of-the-art results (F1 84.2%) on the AphasiaBank corpus and, notably, shows that gesture features are more informative than acoustic features for this task.

**Research Question**
The paper asks whether, and how, co-speech gesture patterns - not just speech/linguistic content - can be used to automatically distinguish between clinically distinct aphasia types (e.g., fluent Wernicke-type versus non-fluent Broca-type aphasia), motivated by the clinical observation that people with aphasia rely more heavily on gesture as a compensatory communication strategy when word retrieval and fluency are impaired, and that this reliance differs systematically by aphasia type.

**Methodology**
The dataset is drawn from AphasiaBank's Cinderella Story Recall task, yielding 507 subjects and 3,651 aligned text/gesture/audio segments (50 tokens each, ~23.8s average duration) labeled into four categories: Control, Fluent, Non-Comprehension, and Non-Fluent aphasia. Transcriptions were generated with Whisper ASR (chosen because it captures fillers and disfluencies at the token level, unlike prior clean-speech-oriented ASR pipelines), and text, gesture, and audio streams were time-aligned. The model architecture has four components: (1) a Speech-Gesture Graph Encoder that builds a heterogeneous graph connecting disfluency-related keyword nodes (encoded via pretrained RoBERTa) to co-occurring gesture nodes (23 upper-body MediaPipe pose keypoints through a BiLSTM) and audio nodes (eGeMAPS acoustic descriptors via OpenSmile, also through a BiLSTM), updated via a GraphSAGE-style heterogeneous aggregation; (2) a Gesture-aware Word Embedding Layer that re-weights the pretrained RoBERTa word embeddings using the graph-refined disfluency-node representations, making textual features sensitive to gesture context; (3) a Multimodal Fusion Encoder based on the Multimodal Transformer (MulT) architecture, using cross-modal attention (vision→text, audio→text) followed by self-attention; and (4) a final fully-connected classification head trained with cross-entropy loss. The model is evaluated with group-stratified 5-fold cross-validation against classical ML baselines (SVM, Random Forest, Decision Tree, Logistic Regression, AdaBoost) and established multimodal fusion baselines (MulT, MISA, MAG, SP-Transformer), with ablations isolating the contribution of each modality and of the graph/gesture-aware components.

**Findings**
Deep multimodal fusion models outperformed classical feature-concatenation baselines, and the proposed graph-based approach outperformed all fusion baselines, with its advantage concentrated on the harder minority classes (e.g., Non-Comprehension, which most baselines classified with F1 near zero). Ablations showed that text alone is the strongest single modality, but visual (gesture) features meaningfully outperform acoustic features as a unimodal signal, and that the largest single performance jump comes from the gesture-aware word embedding step - i.e., letting gesture information reshape how textual/disfluency content is represented, not just concatenating gesture features alongside text. The model also showed a systematic gender gap in performance (models trained and tested on female subjects performed better than on male subjects), which the authors relate to documented clinical differences in aphasia presentation by gender.

**Results**
The full model reached F1 = 84.2% (Precision 0.837, Recall 0.852) using 50-token segments, versus the best fusion baseline (MulT and MISA) at F1 = 76.1% and the best classical baseline (Random Forest) at F1 = 74.2%; using 30-token segments instead dropped the full model to F1 = 73.2%. In the unimodal ablation, text alone reached F1 = 70.0%, gesture alone F1 = 62.9%, and acoustic alone F1 = 50.1%, while simple T+V+A concatenation without the graph encoder reached F1 = 76.5%; adding the graph encoder alone raised this to F1 = 79.2%, and adding the gesture-aware embedding layer on top produced the full model's F1 = 84.2% - i.e., roughly two-thirds of the total gain over naive fusion came from the gesture-aware embedding step specifically.

**Conclusion**
The authors conclude that gesture is not merely a redundant or secondary channel but carries clinically discriminative information about the type of underlying language-processing impairment, and that explicitly modeling cross-modal (speech-gesture) correlations via a graph structure - rather than late concatenation - is what unlocks that information. For the two-layer thesis, this paper is most directly a Layer 2 / RQ2 methods precedent: its graph-based, gesture-aware multimodal fusion architecture (disfluency-conditioned re-weighting of textual representations using aligned gesture and audio streams) is a concrete template for fusing video, audio, and transcript signals around moments of discourse-planning difficulty, and its clinical population (aphasia, a canonical discourse/word-retrieval-difficulty condition) also gives indirect empirical support to Layer 1's premise that word-retrieval/discourse difficulty and gesture behavior are linked, even though the source study is clinical rather than about non-instrumental self-adaptors specifically.

*Sources: [ACL Anthology page](https://aclanthology.org/2023.emnlp-main.577/), [arXiv:2310.11710 HTML](https://arxiv.org/html/2310.11710), [GitHub - DSAIL-SKKU/Multimodal-Aphasia-Type-Detection_EMNLP_2023](https://github.com/DSAIL-SKKU/Multimodal-Aphasia-Type-Detection_EMNLP_2023)*