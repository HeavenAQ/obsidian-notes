---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://doi.org/10.1145/3136755.3136780
Year: 2017
Should Refer: []
Reading Status: Reading
Venue: ICMI
Topic: Dataset / Corpus
snippet: ""
Authors: A. Cafaro, J. Wagner, T. Baur, S. Dermouche, M. T. Torres, C. Pelachaud, E. Andre, M. Valstar
tags:
  - Dataset
  - Turn-Taking
  - Multimodal
Tier: Essential
Assigned Date: 2026-07-02
---
**Background.**  few open corpora capture spontaneous expert-novice dyads with rich social signals. 

**Research question / methodology.**  build a multimodal mediated-interaction database with audio-visual recordings annotated for gestures, turn-taking, dialogue acts, engagement, and interruptions; multilingual and mediated. 

**Conclusion.**  a discourse-rich, domain-shiftable corpus -- the proposed main discourse-alignment dataset.

## Reading Summary

**Abstract**

This ICMI 2017 resource paper introduces NoXi (NOvice eXpert Interaction database), a multilingual corpus of screen-mediated, dyadic novice–expert conversations built around information exchange and retrieval. It emphasizes spontaneous, adaptive behavior and deliberately includes unexpected situations such as conversational interruptions. Synchronized audio, video, and Kinect 2.0 depth data plus continuous and discrete annotations are publicly distributed via a web interface.

**Research Question**

As a corpus paper, its driving question is infrastructural: how to capture natural, spontaneous dyadic knowledge-exchange interactions — with enough multimodal signal and annotation coverage to study social behaviors like engagement, turn-taking, and interruption handling — across languages and recording sites.

**Methodology**

Data collection ran at three sites (France, Germany, UK) with an expert and a novice conversing through a screen-mediated setup on self-selected expertise topics. In total 87 people were recorded across 84 dyadic sessions totaling 25 hours 18 minutes; the three main languages are English (40 interactions), French (25), and German (19), with sessions also in Spanish, Indonesian, Arabic, and Italian (7 languages, 58 topics overall). Each participant was captured with synchronized audio, HD video, and Kinect 2.0 depth/skeleton streams; 6,373 acoustic features per chunk were extracted with openSMILE, and manual word- and sentence-level transcriptions are provided. Annotations span low-level social signals (gestures, smiles, head movements), functional descriptors (turn-taking, dialogue acts), and interaction-level states (engagement, interest, fluidity), created largely with the NOVA annotation tool.

**Findings**

The mediated novice–expert design reliably elicits spontaneous, engagement-varying conversation, including planned interruption episodes, making the corpus suited to studying adaptive behavior rather than scripted interaction. Its multi-site, multi-language structure yields natural domain shift (site, language, speaker role), and the layered annotation scheme connects low-level body signals to discourse-level functions in one resource.

**Results**

This is a dataset paper, so its substantive output is the resource itself: 25h18m of synchronized audio/video/depth recordings from 87 participants in 84 dyads across 7 languages, with public continuous and discrete annotations and transcriptions. It has since become the basis of the MultiMediate challenge series (e.g., backchannel detection, engagement estimation), which serves as external validation of its utility.

**Conclusion**

The authors conclude NoXi fills the gap for spontaneous, mediated, multilingual dyadic corpora with rich social-signal annotation. For the thesis, it is the proposed main discourse-alignment dataset: full-body views for self-adaptor detection, turn-taking and dialogue-act annotations for discourse-planning-difficulty proxies, and cross-language/site splits for domain-generalization experiments.

*Sources: *[*https://d-nb.info/1250171288/34*](https://d-nb.info/1250171288/34)* (full text PDF), *[*https://dl.acm.org/doi/10.1145/3136755.3136780*](https://dl.acm.org/doi/10.1145/3136755.3136780)*, *[*https://multimediate-challenge.org/datasets/Dataset_NoXi/*](https://multimediate-challenge.org/datasets/Dataset_NoXi/)