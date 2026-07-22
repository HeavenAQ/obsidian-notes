---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2602.20163
Year: 2026
Should Refer: []
Reading Status: Reading
Assigned Date: 2026-07-20
Venue: IJCAI
snippet: ""
Authors: Navya Martin Kollapally, Christa Akers, Renjith Nelson Joseph
Topic: Multimodal-Disfluency-and-Discourse
tags:
  - Multimodal
  - Gesture
  - Disfluency
Tier: Recommended
Local PDF: "[[99 Assets/Papers/CVPR 2027/Graph Modelling Analysis of Speech-Gesture Interaction for Aphasia Severity Estimation.pdf]]"
Zotero URI: "zotero://select/library/items/PNTHY8ER"
Citation Key: "kollapallyGraphModellingAnalysis2026"
Zotero PDF URI: "zotero://open-pdf/library/items/NZJ26CHF"
---
Found via **Learning Co-Speech Gesture for Multimodal Aphasia Type Detection** (2026-07-17 night) — adjacent-literature substitute, not a direct reference (that EMNLP 2023 paper's own reference list contains no qualifying 2023+ CS entries; its only 2023 citations are a psychology/linguistics systematic review and a general medical-AI bibliometrics review). A direct continuation of the same aphasia speech-gesture-graph research line: represents each participant's discourse as a directed multimodal graph over lexical items and gestures (word-word, gesture-word, word-gesture edges) and uses GraphSAGE to learn participant-level embeddings for aphasia *severity* estimation, finding severity emerges from structured speech-gesture interaction rather than isolated lexical distribution — directly relevant to Layer 1's premise that gesture-discourse coupling indexes underlying language-processing difficulty, and a further methods precedent for RQ2's multimodal fusion design.

## Reading Summary

**Abstract**
This paper proposes a graph neural network framework for estimating the severity of aphasia — an acquired language disorder, typically caused by stroke — from spontaneous discourse rather than from isolated clinical test items. Each participant's transcript is represented as a directed multimodal graph over lexical items, gestures, and paraphasia (word-finding error) events, and the resulting graph- and node-level features are used both in classical regression models and to train a GraphSAGE-based graph neural network that predicts the clinical WAB Aphasia Quotient severity score.

**Research Question**
Can the severity of aphasia, as measured by the clinical Western Aphasia Battery-Revised (WAB-R), be predicted more reliably from the structured interaction between speech and gesture in spontaneous discourse than from isolated lexical, part-of-speech, or paraphasia-error features alone — and does explicitly encoding gesture as a graph node alongside words materially improve that prediction?

**Methodology**
The dataset is drawn from the Non-Protocol section of AphasiaBank (part of TalkBank): 192 participants (spanning no-aphasia, mild, moderate, severe, and very-severe categories), selected from a pool of 512 for having complete WAB scoring, totaling 287 hours of transcribed clinician-patient interaction across personal narratives, picture description, story retelling, and procedural discourse tasks. From each CHAT-format transcript the authors extract word tokens, morphosyntactic (POS) tags, and gesture markers, heuristically pairing each gesture with its nearest word token (within five tokens) to approximate temporal/functional alignment, and separately extract semantic, phonemic, and neologistic paraphasia error tags. A directed multimodal graph is built per participant (nodes = words/gestures, edges = word-word, gesture-word, word-gesture transitions weighted by co-occurrence frequency), from which the authors derive both (a) graph-level summary statistics (density, reciprocity, transitivity, gesture-to-node ratio, degree entropy, edge-weight statistics, POS rates, paraphasia rates) and (b) a 7-dimensional node-level feature vector for GNN input. These features feed two parallel modeling paths: a two-layer GraphSAGE regressor trained end-to-end on the graphs (5-fold cross-validation, MSE loss, Adam optimizer), and a ridge regression baseline trained on the hand-engineered graph-derived features, both evaluated by RMSE, MAE, Pearson r, and Spearman correlation against WAB-AQ and three WAB subtest scores.

**Findings**
Semantic and overall paraphasia rates showed a clear positive association with aphasia severity (odds ratios of 1.25 and 2.80 per 0.01 increase, respectively), while phonological paraphasia showed little association — consistent with semantic word-finding errors being a more severity-linked marker than phonological substitutions. Critically, gesture-to-node ratio was negatively correlated with WAB score, meaning heavier reliance on gesture tracked with more severe aphasia, and an ablation removing gesture nodes from the GNN substantially degraded performance (RMSE rose from 9.29 to 18.9, Pearson r dropped from 0.703 to 0.67) — direct evidence that gesture usage carries independent severity-relevant signal beyond lexical or paraphasia features alone. The authors note the model struggles specifically at the very-severe end of the aphasia spectrum, where discourse (and therefore the graph) becomes sparse and low-information.

**Results**
The ablation study on the ridge-regression / ridge-derived feature path showed R² rising from 0.089 (graph-only features) to 0.099 (+POS) to 0.120 (graph+POS+paraphasia combined; adjusted R² = 0.087), while POS-only or paraphasia-only models each achieved only R² ≈ 0.002, underscoring that no single feature type is sufficient alone. On the main severity target (WAB-AQ), the GraphSAGE GNN achieved RMSE = 9.29 ± 1.47, Pearson r = 0.703 ± 0.082, Spearman = 0.675 ± 0.094, closely comparable in correlation to the ridge regression baseline (RMSE = 16.42 ± 4.76, Pearson r = 0.703 ± 0.093), with the GNN clearly ahead on RMSE despite using only the raw graph topology and 7D node features rather than hand-engineered ones.

**Conclusion**
The authors conclude that aphasia severity is not encoded in isolated lexical or paraphasia distributions but emerges from the structured interaction between speech and gesture, and that a graph neural network can learn this structure directly from discourse graphs with performance comparable to an explicitly feature-engineered regression baseline, though both degrade in very-severe, sparse-discourse cases. For the thesis, this is a strong Layer 1 (RQ1, behavioral) analog: it is essentially the same paradigm the thesis proposes for self-adaptors and discourse-planning difficulty — nonverbal (here, gestural) behavior indexing an underlying difficulty in language/discourse production — applied to a clinical population, and it also offers a concrete methods precedent (graph-based multimodal fusion of gesture and lexical/discourse structure) relevant to Layer 2's RQ2 fusion design.

*Sources: [arXiv:2602.20163](https://arxiv.org/abs/2602.20163), [PDF](https://arxiv.org/pdf/2602.20163)*
