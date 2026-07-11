---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-07T14:58:00
Status: Done
Last updated time: 2026-07-07T14:58:00
Last edited by: Heaven Chen
Category:
  - LLM
  - Theory
  - ML
---
> **Authors:** Ariel Goldstein, Leonard Niekerken, Ariel Schain, Zaid Zada, Samuel A. Nastase, Harshvardhan Gazula, …, Werner Doyle, Orrin Devinsky, …, **Uri Hasson** (Hasson Lab — Princeton Neuroscience Institute / NYU / HUJI). **Venue:** Nature Human Behaviour, 2025. **DOI:** 10.1038/s41562-025-02105-9.

> *(Author list abbreviated from the article metadata; senior author Uri Hasson, first author Ariel Goldstein.)*

## Notes

- Not a CV paper — this is **cognitive neuroscience × speech/LLM**, relevant to the thesis's **"Multimodal Disfluency & Discourse"** strand and its neurocognitive framing.
- Core idea: a *single* multimodal DNN (**Whisper**, acoustic→speech→language) provides one **unified embedding space** that predicts brain activity during **real, unscripted conversation**, for both **speech production and comprehension**.
- Strongest thesis link: it validates using **multimodal embeddings** (not separate acoustic vs. linguistic pipelines) to model natural discourse, and cleanly separates **production vs. comprehension** — the two sides of a conversation where discourse-planning difficulty and disfluency live.
- Method template worth borrowing: **encoding models** (embedding → neural signal) evaluated by predicting held-out data, with per-lag temporal analysis.

## Reading Summary

**Abstract**

This Nature Human Behaviour (2025) study from the Hasson lab links acoustic, speech and linguistic representations to intracranial brain activity recorded during **real-life, free-flowing conversations**, using a multimodal acoustic-to-speech-to-language model (**Whisper**). From the same model they extract three embedding types per word — **acoustic**, **speech**, and **language** — and show that a unified embedding space predicts neural responses across the cortical language network with high accuracy during both speaking and listening.

**Research Question**

Can a single multimodal deep model that transforms sound → contextual speech → language provide a **unified representational space** that captures how the human brain processes language during everyday, natural conversation — across the full acoustic-to-semantic hierarchy and across production vs. comprehension?

**Methodology**

Intracranial **ECoG** (electrocorticography) was recorded from epilepsy patients engaged in **spontaneous everyday conversations** (no experimenter intervention; ~100 h total — ~50 h comprehension / 289,971 words and ~50 h production / 230,238 words). Audio + transcripts were fed to **Whisper**, and three embeddings were extracted per word: **acoustic** (low-level, input side of the speech encoder), **speech** (contextual, speech-encoder), and **language** (decoder stack, ~layer 3–4). For each electrode they fit **linear encoding models** mapping embeddings → neural activity, separately per **lag** (−2,000 to +2,000 ms around word onset) and per condition (production/comprehension), using **10-fold leave-one-out cross-validation** and evaluating by the **correlation between predicted and actual neural signal on held-out conversations**, with non-parametric FWER multiple-comparison control. **Variance partitioning** separated unique speech- vs language-embedding contributions; controls compared a unimodal LM (**GPT-2**) and continuous (non-word-locked) signals.

**Findings**

The embeddings predicted neural activity with **remarkable accuracy across hundreds of thousands of words**, in both production and comprehension, across the language network. There is a **soft cortical hierarchy** aligned to the model's own hierarchy: **acoustic** embeddings best predict auditory/somatomotor cortex (STG, SM); **speech** embeddings extend to STG, SM, IFG (Broca), temporal pole, angular gyrus, and posterior MTG (Wernicke); **language** embeddings hit similar regions but with fewer electrodes and stronger IFG. **Contextual speech embeddings model cortex better than raw acoustic embeddings** everywhere, and speech beats language embeddings in many electrodes. Encoding is **robust to data scarcity** (little change using only 25% of training data) and generalizes to unseen conversations; GPT-2 language embeddings give similar language-side results.

**Results**

Predicted signals correlate strongly with actual neural activity (Pearson correlations up to ~0.5 for speech encoding during production; language encoding up to ~0.3), significant at *P* < 0.01 / *q* < 0.01 after correction. Quantified comparisons: speech embeddings yield **more significant electrodes than acoustic** (production 274 vs 64) and **more than language** (production 274 vs 154), consistent with a graded acoustic→speech→language organization rather than sharp modular boundaries.

**Conclusion**

A **single unified multimodal embedding space** — sound to contextual speech to language, learned by one model — captures the neural basis of natural language processing in real conversation, for both production and comprehension, along a continuous cortical hierarchy. This supports an **integrated, continuous** view of speech–language processing over strictly modular pipelines. For the thesis: it is the strongest available justification for modelling **multimodal discourse** with unified DNN embeddings, cleanly frames the **production vs. comprehension** split where disfluency/discourse-planning phenomena arise, and offers a reusable **encoding-model** methodology for relating behavioral/linguistic features to (neuro)physiological or behavioral signals.

*Sources: *[*https://www.nature.com/articles/s41562-025-02105-9*](https://www.nature.com/articles/s41562-025-02105-9)* (DOI: 10.1038/s41562-025-02105-9)*