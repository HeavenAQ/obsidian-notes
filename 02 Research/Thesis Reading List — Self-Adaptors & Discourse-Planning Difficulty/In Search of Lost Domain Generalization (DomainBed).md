---
base: "[[Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty.base]]"
Paper Link: https://arxiv.org/abs/2007.01434
Year: 2007
Should Refer:
  - None
Reading Status: Skim-Skip
Venue: ICLR
Code Link: https://github.com/facebookresearch/DomainBed
Relatedness: Not-Relevant
Topic: Domain-Generalization
snippet: |2-
    • Talks about datasets themselves
    • Not directly related to my work 
Authors: I. Gulrajani, D. Lopez-Paz
tags:
  - Domain-Generalization
  - Benchmark
  - Foundational
Tier: Essential
Assigned Date: 2026-07-03
Local PDF: "[[99 Assets/Papers/CVPR 2027/In Search of Lost Domain Generalization (DomainBed).pdf]]"
Zotero URI: "zotero://select/library/items/VSAC4AQH"
Citation Key: "gulrajaniSearchLostDomain2007"
Zotero PDF URI: "zotero://open-pdf/library/items/HEQLTFUV"
---
**Background.**  DG claims used inconsistent datasets, architectures, and selection criteria. 

**Research question.**  how useful are DG algorithms under fair, standardized comparison? 

**Methodology.**  the DomainBed testbed (7 datasets, 9 algorithms, 3 model-selection criteria). 

**Conclusion.**  well-implemented ERM matches SOTA and model selection is decisive -- the evaluation philosophy (honest splits, explicit selection) for the whole thesis.

## Reading Summary

**Abstract**

This ICLR 2021 paper (Facebook AI Research) is the field's reality check on domain generalization (DG). The authors argue that a decade of claimed DG progress rests on inconsistent experimental conditions — different datasets, architectures, augmentations, and above all model-selection criteria — and introduce DomainBed, a standardized testbed. Under fair comparison, a carefully implemented Empirical Risk Minimization (ERM) baseline matches or beats the state-of-the-art DG algorithms.

**Research Question**

How useful are domain generalization algorithms when compared under identical, realistic experimental conditions — and what role does model selection play when the test distribution is by definition unavailable?

**Methodology**

DomainBed comprises seven multi-domain image datasets (including Colored/Rotated MNIST, VLCS, PACS, OfficeHome, TerraIncognita, DomainNet), nine baseline algorithms in its initial release (ERM plus DG methods such as IRM, DRO, Mixup, MLDG, CORAL, MMD, DANN, C-DANN), and — the key methodological contribution — three explicit model-selection criteria: training-domain validation, leave-one-domain-out validation, and test-domain (oracle) validation. Every algorithm is run under the same architecture, data splits, and large-scale random hyperparameter search (tens of thousands of trained models), so differences reflect the algorithm rather than tuning effort.

**Findings**

The headline finding is deflationary: when carefully implemented and tuned, plain ERM outperforms the previously reported state-of-the-art on average, and no DG algorithm in the testbed beats ERM by more than about one point under identical conditions. Second, model selection is decisive — rankings shift depending on the selection criterion, and the authors argue that a DG algorithm published without a specified model-selection strategy is incomplete, since implicitly peeking at the test domain inflates results.

**Results**

Quantitatively, the paper reports average accuracies across the seven datasets under each selection criterion; the decisive comparison is that no included algorithm exceeds ERM's average accuracy by more than ~1 point (under training-domain validation), whereas published numbers for the same algorithms had claimed much larger gains under heterogeneous conditions. (Exact per-dataset tables are in the paper; the takeaway is the ordering and the ≤1-point margin, not any single number.)

**Conclusion**

The authors conclude that measurable DG progress requires standardized evaluation with honest, pre-declared model selection, and release DomainBed as that standard. For the thesis this is the RQ3 methodological backbone: any cross-dataset/cross-domain benchmarking of self-adaptor or cognitive-load models should adopt DomainBed's discipline (fixed splits, explicit selection criteria, strong ERM baseline) — and it pairs directly with REVELIO, which shows the same generalization gap in task-load estimation. It also signals evaluation rigor for the DENSO PhD positioning (domain generalization is a stated lab interest).

*Sources: *[*https://arxiv.org/abs/2007.01434*](https://arxiv.org/abs/2007.01434)*, *[*https://iclr.cc/virtual/2021/poster/2998*](https://iclr.cc/virtual/2021/poster/2998)*, *[*https://www.semanticscholar.org/paper/6a5efb990b6558c21d9fdded4884c00ba152cb7c*](https://www.semanticscholar.org/paper/6a5efb990b6558c21d9fdded4884c00ba152cb7c)*, *[*https://github.com/facebookresearch/DomainBed*](https://github.com/facebookresearch/DomainBed)