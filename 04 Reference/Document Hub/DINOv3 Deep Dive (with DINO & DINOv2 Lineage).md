---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-08T16:41:00
Status: Done
Last updated time: 2026-07-10T17:25:00
Last edited by: Heaven Chen
Category:
  - CV
  - ML
---
> **Deep dive into DINOv3** (["DINOv3", arXiv 2508.10104](https://arxiv.org/abs/2508.10104), Meta AI / FAIR, Aug 2025 — [project/code](https://github.com/facebookresearch/dinov3), [Meta research page](https://ai.meta.com/research/publications/dinov3/)). DINOv3 is the third generation of Meta's **self-supervised** vision backbone family. This note builds up the whole lineage: **DINO** (the self-distillation idea), **DINOv2** (scaling it into an all-purpose frozen backbone), and **DINOv3** (scaling further + fixing dense features with *Gram anchoring*). The through-line: learn general visual features from images alone — no labels, no captions — good enough to beat *task-specific, fully-supervised* systems while the backbone stays **frozen**.

# 0. The one-paragraph story

All three models share one mechanism — **self-distillation with no labels**: a *student* network is trained to match the output of a *teacher* network (an exponential moving average of the student) across two differently-augmented views of the same image, with no negative pairs and no annotations. DINO showed this makes a Vision Transformer spontaneously learn object structure. DINOv2 showed that with **curated data at scale** + a **patch-level objective**, the resulting *frozen* features rival supervised backbones on classification *and* dense tasks. DINOv3 pushes data and model size an order of magnitude higher, and introduces **Gram anchoring** to stop dense (patch) features from rotting over the very long training schedules that scale demands — yielding a single frozen backbone that is state-of-the-art on segmentation, depth, detection, and tracking at once.

# 1. DINO (2021) — self-distillation with no labels

[Caron et al., ](https://arxiv.org/abs/2104.14294)[*Emerging Properties in Self-Supervised Vision Transformers*](https://arxiv.org/abs/2104.14294)[, ICCV 2021 (arXiv 2104.14294)](https://arxiv.org/abs/2104.14294). The name is a contraction of **DI**stillation with **NO** labels.

## 1.1 The mechanism

Two networks with **identical architecture** (a ViT, or a ResNet) but different weights: a student $g_{\theta_s}$ and a teacher $g_{\theta_t}$. Each maps an image view to a $K$-dimensional distribution over learned *prototypes* (an MLP projection head → $\ell_2$-normalize → weight-normalized linear layer, $K=65536$). Given two views $x_1, x_2$ of one image, the student is trained to match the teacher's distribution:

$$
\min_{\theta_s}\; -\,P_t(x_1)^\top \log P_s(x_2),\qquad P_s(x)=\mathrm{softmax}\!\big(g_{\theta_s}(x)/\tau_s\big)
$$

Crucially there are **no negatives** and **no contrastive term** — just "make the student agree with the teacher." Two ingredients keep this from collapsing to a trivial constant:

- **Centering**: subtract a running mean $c$ from the teacher logits, $c \leftarrow m\,c + (1-m)\frac{1}{B}\sum_i g_{\theta_t}(x_i)$ — stops any single prototype from dominating.
- **Sharpening**: use a low teacher temperature $\tau_t$ so the teacher target is peaked — stops the output from becoming uniform.

Centering and sharpening push in opposite directions and balance each other, avoiding collapse without batch statistics or negative samples. The teacher is never back-propagated; it is an **EMA (momentum) of the student**:

$$
\theta_t \leftarrow \lambda\,\theta_t + (1-\lambda)\,\theta_s
$$

## 1.2 Multi-crop (local-to-global)

DINO feeds a **multi-crop** set of views: 2 large *global* crops (~224²) plus several small *local* crops (~96²). The **student sees all crops; the teacher sees only the two global crops**. So the student must predict the whole-object (global) distribution from a small local patch — a "local-to-global" correspondence that is a big part of why the features become semantic.

## 1.3 Why it mattered

The headline *emergent properties*: a self-supervised ViT's **[CLS] self-attention maps segment the foreground object** with no segmentation labels at all, and the raw features are such strong **k-NN classifiers** that a simple nearest-neighbour on frozen features hits ~78% top-1 on ImageNet (ViT-B/8). DINO reframed SSL from "pretraining trick" to "the features themselves are the product."

# 2. DINOv2 (2023) — scaling into an all-purpose frozen backbone

[Oquab et al., ](https://arxiv.org/abs/2304.07193)[*DINOv2: Learning Robust Visual Features without Supervision*](https://arxiv.org/abs/2304.07193)[, TMLR 2024 (arXiv 2304.07193)](https://arxiv.org/abs/2304.07193). Thesis: SSL can produce **general-purpose features that work off-the-shelf on a frozen backbone** — if you scale *data* and *model* with a *stable* recipe.

## 2.1 What changed vs DINO

- **Curated data at scale (LVD-142M).** Rather than train on ImageNet-without-labels or raw web scrapes, DINOv2 builds a 142M-image dataset via an **automatic curation pipeline**: embed a huge uncurated pool, retrieve images similar to a set of curated seed datasets, cluster and de-duplicate. Curation — not just quantity — is what makes the features clean.
- **A patch-level objective (iBOT) on top of the image-level one (DINO).** DINOv2's loss is a **sum of two self-distillation losses**: the DINO loss on the image-level **[CLS]** token, *plus* the **iBOT** loss — a masked-image-modeling term where the student, given masked patches, must predict the teacher's tokens at those patch positions. The image-level loss gives global semantics; the patch-level loss gives **dense** (per-patch) features for segmentation/depth. The two losses use **separate, untied heads**.
- **Better anti-collapse & spread.** Teacher targets use **Sinkhorn–Knopp** normalization (from SwAV) instead of pure softmax-centering, and a **KoLeo** regularizer encourages features to spread uniformly across the batch (good for retrieval/nearest-neighbour).
- **Scale + distillation.** The flagship is **ViT-g/14 (~1.1B params)**. Smaller models (ViT-S/B/L) are then **distilled from the frozen ViT-g** rather than trained from scratch — the distilled small models beat from-scratch equivalents. A short **high-resolution (518px) phase** at the end sharpens dense features.
- **Registers (follow-up).** *Vision Transformers Need Registers* (Darcet et al., 2023) found high-norm "artifact" tokens polluting attention/dense maps; adding a few dedicated **register tokens** cleans them up. Registers become standard equipment from here on.

Net result: one **frozen** DINOv2 backbone gives strong linear-probe classification, retrieval, monocular depth, and semantic segmentation — no fine-tuning per task.

# 3. DINOv3 (2025) — scale + Gram anchoring + post-hoc flexibility

[Siméoni, Vo, Seitzer et al., ](https://arxiv.org/abs/2508.10104)[*DINOv3*](https://arxiv.org/abs/2508.10104)[, arXiv 2508.10104 (13 Aug 2025)](https://arxiv.org/abs/2508.10104). Three moves: **(a)** scale data and model another order of magnitude; **(b)** fix the dense-feature degradation that scale exposes, via **Gram anchoring**; **(c)** add **post-hoc** adaptations (resolution, a distilled model family, text alignment) so one training run ships many deployable models.

## 3.1 Scaling: data, model, and *constant* schedules

- **Data — LVD-1689M.** A curated **~1.69B-image** dataset sampled from a ~17B pool, combining **hierarchical k-means clustering**, **retrieval-based curation**, and raw curated sets (incl. ImageNet-1k) — engineered to balance *mode-covering* (broad concept coverage) against *mode-seeking* (precision on useful concepts).
- **Model — ViT-7B.** A custom **~6.7B-parameter** ViT (~40 blocks) with **axial RoPE** (rotary position embeddings) plus **"box jittering"** of coordinates for robustness to resolution and aspect ratio, and **4 register tokens** built in.
- **Constant hyperparameter schedules.** Instead of decaying learning rate / teacher-momentum / weight-decay on a fixed horizon, DINOv3 holds them **constant**, so training can run **indefinitely** at scale without a pre-committed end. This is what makes "train much longer" practical — and what makes §3.2 necessary.

## 3.2 Gram anchoring — the headline contribution

**The problem scale exposes.** Over very long schedules, DINOv3 observed a split: **global** metrics (classification, retrieval) keep improving, but **dense** metrics (segmentation, tracking) *degrade* — patch feature maps grow noisy and lose their clean spatial structure. Prior work never fully solved this; it just stopped training early.

**The fix.** Let $F\in\mathbb{R}^{N\times d}$ be the $\ell_2$-normalized patch features of an image ($N$ patches). Its **Gram matrix** $G = F F^\top$ holds every **pairwise patch similarity** — i.e., the *relational structure* of the feature map, independent of the absolute feature vectors. Gram anchoring adds a loss that pulls the student's Gram matrix toward that of an **earlier, cleaner checkpoint** (the *"Gram teacher"*):

$$
\mathcal{L}_{\mathrm{Gram}} = \big\lVert\, F_s F_s^\top \;-\; F_g F_g^\top \,\big\rVert_F^2
$$

Because it constrains only the **structure** of patch relationships — not the features themselves — global representations can keep improving while the dense map's spatial coherence is **restored**. It is applied as a **late-stage refinement phase**, and it is what lets DINOv3 train a 7B model to completion with clean dense features. Concretely, DINOv3 dense linear-probe segmentation on ADE20k reaches **~55.9 mIoU vs ~49.5 for DINOv2** — the gap is essentially the Gram-anchoring story.

## 3.3 Post-hoc flexibility (one teacher → many shippable models)

- **High-resolution adaptation.** A post-training phase adapts the model to larger inputs, so dense features stay sharp at high resolution.
- **A distilled model family.** The frozen **ViT-7B is the teacher**; a suite of smaller models is distilled from it via an efficient **multi-student parallel** procedure — **ViT-S (~21M), S+, B, L, H+ (~840M)** plus **ConvNeXt T/S/B/L (~29M–200M)** for CNN-friendly deployment. Distilled students inherit most of the teacher's quality at a fraction of the cost.
- **Text alignment — dino.txt.** A lightweight **text encoder** is aligned to frozen DINOv3 features (both the **CLS** token and the **patch** tokens) with a **LiT-style contrastive** objective, giving **open-vocabulary zero-shot** classification and segmentation — bolted on *after* the vision model is trained, without touching the backbone.
- **Domain variant.** The same recipe trains a **satellite/aerial** ViT-7B on an aerial dataset (SAT-493M), evidence the single algorithm transfers across image domains.

## 3.4 Results in one line

With a **frozen** backbone (no fine-tuning), DINOv3 sets or matches state of the art across dense and global tasks at once — e.g. **COCO detection ~66.1 mAP** and **ADE20k segmentation ~63.0 mIoU** with lightweight task heads — **surpassing both weakly-supervised (CLIP/SigLIP-style) and prior self-supervised models**, and beating specialized *fine-tuned* pipelines on several dense benchmarks.

# 4. The differences at a glance

| Aspect | DINO (2021) | DINOv2 (2023) | DINOv3 (2025) |
| --- | --- | --- | --- |
| Core objective | Image-level self-distillation on [CLS] | DINO [CLS] **plus iBOT** patch-level (masked) loss | Same [CLS]+patch core **plus Gram anchoring** for dense features |
| Anti-collapse / spread | Centering + sharpening | Sinkhorn–Knopp centering + **KoLeo** spread | Above **plus constant (non-decaying) schedules** for indefinite training |
| Position encoding | Learned absolute | Learned / interpolated | **Axial RoPE** with box jittering |
| Register tokens | None | Added as follow-up (2023) | **Built in (4 registers)** |
| Training data | ImageNet-1k (no labels) | **LVD-142M** (curated) | **LVD-1689M** (~1.7B, curated from ~17B) |
| Largest model | ViT-B (~86M) | ViT-g/14 (~1.1B) | **ViT-7B (~6.7B)** |
| Smaller models | Trained directly | Distilled from frozen ViT-g | Distilled family from frozen ViT-7B — **ViT-S…H+ and ConvNeXt** |
| Text / zero-shot | None | None | **dino.txt** text alignment → open-vocab zero-shot |
| Headline contribution | SSL ViT features are semantic (attention segments objects) | Curated-scale SSL → all-purpose **frozen** features (incl. dense) | **Gram anchoring** keeps dense features clean at scale; universal frozen backbone |

**How to read the arc.** DINO → DINOv2 is *"add a dense objective + curate data + scale + distill,"* turning a nice property into a usable frozen backbone. DINOv2 → DINOv3 is *"scale another 10× and fix the one thing that breaks when you do"* — dense features degrading over long training — plus post-hoc packaging (resolution, family, text) so a single expensive run yields many deployable models.

# 5. Why practitioners care (frozen-backbone deployment)

The practical selling point across the family, sharpest in DINOv3: you **freeze the backbone** and train only a small head per task. That means one set of features serves classification, retrieval, **semantic segmentation, monocular depth, dense correspondence/tracking, and 3D keypoint matching** simultaneously, with no per-task fine-tuning and no labels needed to pretrain. The distilled ViT-S/B and ConvNeXt variants make this shippable on modest hardware; dino.txt adds zero-shot when you need language. (This is exactly the role a model like this plays as the geometry/appearance feature extractor inside larger systems — cf. the depth/reconstruction backbones referenced in the Vista4D note in this hub.)

# 6. Limitations & caveats

- **Dense-feature degradation is *****managed*****, not eliminated.** Gram anchoring is a corrective refinement phase; it presupposes a good earlier "Gram teacher" checkpoint to anchor to, and adds recipe complexity.
- **Cost.** Training a 7B model on ~1.7B curated images is far beyond most labs; the *usable* artifacts for most people are the **distilled** students, not the teacher.
- **Curation is a hidden dependency.** Much of the quality comes from the data-curation pipeline (LVD-142M / LVD-1689M), which is intricate and only partially reproducible from the outside.
- **Text is bolted on, not native.** dino.txt aligns a text encoder *to* frozen features; DINOv3 is not natively multimodal like a from-scratch image-text model, so zero-shot is solid but not its core strength.
- **Licensing / access.** Weights and code are released, but usage terms differ from a permissive OSS license — check the DINOv3 license before commercial use.

---

*Sections 7–14 below are supplementary operational detail drawn from further reading (see the added Zenn write-up in Sources), covering the parts most summaries skip: the exact weighted objective, how Gram anchoring actually runs, the distillation system, evaluation protocols, concrete numbers, and training cost.*

# 7. The full training objective (with loss weights)

Beyond the DINO + iBOT + KoLeo combination noted earlier, the paper writes out the exact **weighted** objective and how it changes once Gram anchoring switches on. The pre-Gram objective is

$$
\mathcal{L}_{\mathrm{Pre}} = \mathcal{L}_{\mathrm{DINO}} + \mathcal{L}_{\mathrm{iBOT}} + 0.1\,\mathcal{L}_{\mathrm{DKoLeo}}
$$

and once the Gram term is added late in training the **refinement** objective becomes

$$
\mathcal{L}_{\mathrm{Ref}} = w_{D}\,\mathcal{L}_{\mathrm{DINO}} + \mathcal{L}_{\mathrm{iBOT}} + w_{DK}\,\mathcal{L}_{\mathrm{DKoLeo}} + w_{\mathrm{Gram}}\,\mathcal{L}_{\mathrm{Gram}}
$$

(DKoLeo is a **distributed** variant of the KoLeo spread regularizer.) Several recipe choices that differ from DINOv2 are worth flagging:

- **Separate DINO and iBOT heads.** DINOv2 *shared* the projection-head weights between the DINO ([CLS]) and iBOT (patch) losses; DINOv3 found this **counter-productive at large scale** and untied them.
- **Sinkhorn–Knopp on both teachers.** The teacher-side softmax-centering step for *both* DINO and iBOT uses Sinkhorn–Knopp normalization instead of plain centering.
- **Soft labels for iBOT.** iBOT supervises with the teacher's *continuous* post-softmax token distribution (soft labels), not discrete token IDs — this materially helps quality.
- **Centering + sharpening everywhere**, plus a tuned **mask-prediction ratio** (with added variance) for the masked-image-modeling term.
- **Systems.** A custom **FlashAttention**, and ViT embedding-dim / head-count tuned for compute efficiency at 7B.

# 8. Gram anchoring — the operational details

§3.2 gave the idea; here is how it actually *runs*, which is the part most write-ups skip.

![[99 Assets/Media/e092c66286ed-20250915.png|DINOv3 dense-feature collapse without Gram anchoring]]

*DINOv3 Fig. 6 — the kind of dense-feature “collapse” over long training that Gram anchoring repairs.*

- **It switches on late.** Dense features only start to rot after very long training, so Gram anchoring is typically **turned on after ~1M iterations**, not from the start.
- **Multi-crop, global-crops only.** Training uses **2 global crops (256×256) + 8 local crops (112×112)** per image; the Gram loss is computed **only on the global crops** for efficiency.
- **A refreshed “Gram teacher.”** The Gram teacher (whose Gram matrix the student is pulled toward) is a *past* checkpoint picked from an early stage with clean dense features, and it is **re-synced to the main EMA teacher every ~10k iterations**.
- **The high-resolution Gram-teacher trick.** For the Gram teacher, images are fed at **2× the normal resolution** and the resulting feature map is **downscaled** to the target size; these smoother high-res-derived patch relations are then distilled into the student — a cheap way to inject cleaner structure.
- **Side effect: iBOT trains faster.** Adding the Gram objective makes the **iBOT loss drop faster** — the stability of a fixed Gram teacher spills over into the MIM objective.
- **Prerequisite for high-res.** Gram anchoring is *required* for the high-resolution phase (§10) to work; without it, dense-task performance drops sharply at high resolution. It also pairs naturally with **RoPE**, which lets the model ingest varied resolutions without extra adaptation.

# 9. Multi-student distillation (how the family is really made)

The distilled DINOv3 family (ViT-S/B/L, ConvNeXt) is produced by a purpose-built **multi-student distillation** system, not one-teacher-one-student runs. The motivation is cost: the frozen 7B teacher is expensive to run, so the naive scheme wastes teacher compute once per student. Instead:

![[99 Assets/Media/22ca1f39cd18-20250916.png|DINOv3 multi-student distillation]]

*DINOv3 — multi-student distillation: the teacher runs once and its outputs are shared to all students.*

1. Multiple students are assigned to **separate GPU groups**.
2. The **teacher forward pass is computed once** across all GPUs (fixed cost, independent of student count).
3. Teacher outputs are broadcast to every student via an **all-gather (NCCL) collective**.
4. Each student trains on its own GPU group.

The payoff: **adding a student does not increase teacher-inference cost**, and per-student GPU counts are tuned so all students finish an iteration at the same wall-clock time. One distillation run yields the whole family, and the widely-used **ViT-L lands close to the 7B teacher** across tasks. In the repo this is the `--multi-distillation` flag / “Multi-distillation” section.

# 10. High-resolution adaptation — the schedule

A short **post-hoc** high-resolution phase at the very end of training keeps dense features sharp at large input sizes (small objects stop vanishing). Concretely:

![[99 Assets/Media/731de1b9b3fa-20250916.png|DINOv3 high-resolution dense features]]

*DINOv3 — stable, high-quality dense features across resolutions after high-res adaptation.*

- Web-model pretraining ends with a brief bump to **518×518**.
- The **satellite** recipe is spelled out: initial pretrain **100k iters** → Gram anchoring **10k iters** → **8k steps at 512px** high-res fine-tuning.
- There is a dedicated config, `dinov3_vit7b16_high_res_adapt.yaml`.
- Because the phase is short, the time/memory cost of high-res training stays affordable even at 7B.
- The result is **native** high-quality dense features across resolutions — the paper shows clean feature maps at up to **4096×4096**. The idea echoes the short high-res steps in **UniViT** and **FlexiViT**.

# 11. How DINOv3 is actually evaluated

Because the whole pitch is *frozen* features, the evaluation protocols matter as much as the scores:

- **Linear probing** — a linear head on frozen features (classification, segmentation, depth).
- **k-NN** — nearest-neighbour on frozen features, a direct feature-quality proxy.
- **Non-parametric** tasks — raw feature-similarity (3D correspondence, unsupervised object discovery, video tracking).
- **Attentive probing** — a shallow transformer head on patch features (e.g. video classification).
- **Frozen backbone + specialized decoder** — Plain-DETR (detection), Mask2Former (segmentation), Depth Anything v2 (depth), VGGT (3D), all trained on top of a frozen DINOv3.
- **Qualitative PCA** — PCA of dense features shows DINOv3's maps are sharper, less noisy, and more semantically consistent than prior models.

![[99 Assets/Media/90509b97ad0d-20250916.png|DINOv3 PCA of dense features]]

*DINOv3 — PCA visualization of dense features vs. prior models.*

# 12. Benchmark highlights (concrete numbers)

A selection of headline results, all with a **frozen** backbone unless noted:

| Task | Benchmark | DINOv3 result |
| --- | --- | --- |
| Semantic seg (dense linear probe) | ADE20k | **+6 mIoU** vs SSL, **+13** vs WSL baselines |
| Semantic seg (dense linear probe) | Cityscapes | **81.1 mIoU** (best; +2.5 over AM-RADIOv2.5) |
| Semantic seg (+ lightweight decoder) | ADE20k | **63.0 mIoU** (ties ONE-PEACE SOTA) |
| Relative depth (frozen) | NYUv2 / KITTI / ETH3D / … | new SOTA — while baselines fine-tune their backbone |
| 3D geometric correspondence | — | **+4.3% recall** vs DINOv2 |
| Object detection | COCO | SOTA with a 100M head on a frozen backbone |
| Fine-grained classification | iNaturalist21 | **89.8%** (> PEcore 87.0%) |
| Instance recognition | Oxford-Hard | **+41% mAP** vs DINOv2 |
| OOD robustness | ImageNet-R / Sketch / ObjectNet | **+10% / +6% / +13%** vs DINOv2 |
| Unsupervised object discovery | VOC 2007 | **+5.9 CorLoc** |
| Geospatial (Earth observation) | 15 tasks | new SOTA on **12 / 15** (RGB + frozen backbone) |

The pattern: DINOv3 beats prior **SSL** models comfortably and matches or beats **weakly-supervised** models (CLIP / SigLIP 2 / PEcore / AM-RADIO) — especially on dense tasks — while staying frozen.

# 13. Training cost & ethics (the part papers usually bury)

One full **ViT-7B** pretraining run is reported at **~47 MWh**, ≈ **18 tonnes of CO₂** — roughly the energy to drive an average EV **240,000 km**. The paper also flags **adversarial vulnerability** and the dual-use risk that a strong **open-vocabulary** detector could be misused. Worth keeping in view alongside the “universal backbone” framing.

# 14. Where DINOv3 sits (comparison set)

For reference, the models DINOv3 benchmarks against — **SSL**: MoCo v3, SwAV, BYOL, EsViT, MAE, Web-DINO, Franca, V-JEPA 2; **weakly-supervised (WSL)**: CLIP, SigLIP 2, PEcore, AM-RADIOv2.5, PEspatial (distilled from SAM 2), EVA-CLIP; **agglomerative**: SAM. Data beyond LVD-1689M / SAT-493M includes ImageNet-1k/22k, Objects365, COCO-Stuff, Hypersim, SatLidar, Open-Canopy, DIOR, and LoveDA. LVD-1689M itself is curated from a **~17B** web-image pool (reportedly public Instagram posts) via hierarchical k-means + retrieval-based curation (Vo et al., 2024).

---

**Sources**: [DINOv3 — arXiv 2508.10104](https://arxiv.org/abs/2508.10104) ([HTML](https://arxiv.org/html/2508.10104v1)) · [Meta AI research page](https://ai.meta.com/research/publications/dinov3/) · [code + models (GitHub)](https://github.com/facebookresearch/dinov3) · [DINOv2 — arXiv 2304.07193](https://arxiv.org/abs/2304.07193) · [DINO — arXiv 2104.14294](https://arxiv.org/abs/2104.14294) · explainers: [Encord](https://encord.com/blog/dinov3-explained-scaling-self-supervised-vision-tr/), [Lightly](https://www.lightly.ai/blog/dinov3), [OpenCV](https://opencv.org/dinov3/) · deep-dive write-up (JP): [Zenn — emmyeil](https://zenn.dev/syu_tan/articles/6df2947eb6c1ae) (source of the figures in §§8–11).