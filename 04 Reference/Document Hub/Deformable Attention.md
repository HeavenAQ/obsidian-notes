---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-06T10:06:00
Status: Not started
Last updated time: 2026-07-06T15:34:00
Last edited by: Heaven Chen
Category: []
---
# Paper: Vision Transformer with Deformable Attention (DAT) — In-Depth Breakdown

> Xia, Pan, Song, Li, Huang. **CVPR 2022 (Best Paper Finalist)**. arXiv:2201.00520. Code: LeapLabTHU/DAT. Extended journal version: **DAT++** (arXiv:2309.01430).

## 1. TL;DR

DAT introduces a **deformable self-attention** module in which the spatial positions of the **keys and values are chosen in a data-dependent way**, instead of attending to a full dense grid (ViT) or a fixed hand-crafted window (Swin/PVT). A small offset network shifts a sparse grid of reference points toward the informative regions of each feature map, and attention is computed only against features sampled at those shifted locations. Built into a 4-stage pyramid backbone, this gives Swin-level efficiency with a **larger, adaptive, data-dependent receptive field**, improving classification, detection, and segmentation.

## 2. Motivation (why it exists)

The paper frames four receptive-field regimes (Figure 1 in the paper):

| Model | Attention pattern | Problem it has |
| --- | --- | --- |
| ViT | Dense global attention | Quadratic cost $O((HW)^2)$; attends to irrelevant regions |
| Swin / PVT | Sparse, fixed windows | Data-agnostic; hand-crafted; windowing slows receptive-field growth, drops long-range relations |
| DCN (deformable conv) | Per-query learned offsets | Great flexibility, but naive port to attention is quadratic in space and hard to train |
| DAT (this paper) | Data-dependent sampled keys/values via shared learned offsets | Focuses on relevant regions at near-linear cost |

Key design choice: unlike DCN, which learns a **different** offset for **every** query (expensive), DAT learns only a **few shared groups of offsets** for a grid of reference points — this is what makes it cheap enough to be a backbone building block.

## 3. The Deformable Attention Module (core math)

Input feature map $x \in \mathbb{R}^{H\times W\times C}$.

**(a) Queries** come from the full feature map:

$$
q = x\,W_q
$$

**(b) Reference points.** Generate a uniform grid $p \in \mathbb{R}^{H_G\times W_G\times 2}$, downsampled by a factor $r$ (so $H_G=H/r,\ W_G=W/r$), normalized to $[-1,+1]$ (top-left $(-1,-1)$, bottom-right $(+1,+1)$).

**(c) Offset network + deformation.** A light sub-network predicts 2D offsets from the queries; a range factor $s$ with a $\tanh$ keeps them bounded:

$$
\Delta p = s\cdot \tanh\!\big(\theta_{\text{offset}}(q)\big),\qquad \tilde{x} = \phi\big(x;\, p+\Delta p\big)
$$

where $\phi$ is **bilinear sampling** at the deformed (sub-pixel) locations.

**(d) Deformed keys/values** are projected from the sampled features:

$$
\tilde{k} = \tilde{x}\,W_k,\qquad \tilde{v} = \tilde{x}\,W_v
$$

**(e) Bilinear sampling** (only the 4 nearest integer pixels contribute):

$$
\phi\big(z;\,(p_x,p_y)\big)=\sum_{(r_x,r_y)} g(p_x,r_x)\,g(p_y,r_y)\,z[r_y,r_x,:],\qquad g(a,b)=\max(0,\,1-|a-b|)
$$

**(f) Multi-head attention** with a **relative position bias** term, per head $m$ (head dim $d=C/M$):

$$
z^{(m)} = \mathrm{softmax}\!\Big(\tfrac{q^{(m)}\,\tilde{k}^{(m)\top}}{\sqrt{d}} + \phi(\hat{B};R)\Big)\,\tilde{v}^{(m)}
$$

$$
z = \mathrm{concat}\big(z^{(1)},\dots,z^{(M)}\big)\,W_o
$$

The bias uses a **continuous relative position bias** table $\hat{B}$ interpolated at the (now non-integer) relative positions $R$ between queries and the deformed points.

**Offset groups.** Channels are split into $G$ groups that **share** an offset field (analogous to grouped/multi-head offsets in DCN), so the module learns several — not $HW$ — deformation patterns per layer. Heads within a group reuse the same sampled locations.

### Offset network

A tiny spatial network applied to the (reshaped) query map:

```plain text
q  ──▶  k×k depthwise conv (stride r)  ──▶  GELU  ──▶  1×1 conv (2 ch, no bias)  ──▶  Δp
```

The depthwise conv gives it local context to decide where to look; the stride $r$ produces one offset per reference point; the $s\cdot\tanh(\cdot)$ clamp prevents divergent, overly-large shifts early in training.

### Pipeline diagram (one deformable-attention block)

```plain text
feature map x (H×W×C)
   │  q = xWq
   ├────────────────────────────────► queries q
   │                                     │
   │                         θ_offset(q) │  (dwconv→GELU→1×1)
   │                                     ▼
 uniform grid p ──(+)──► p+Δp ──► bilinear sample φ(x; p+Δp) = x̃  (H_G×W_G×C)
   │                                     │
   │                          k̃ = x̃Wk , ṽ = x̃Wv
   ▼                                     ▼
  softmax( q k̃ᵀ/√d + relpos_bias ) · ṽ ──► concat heads · Wo ──► z (H×W×C)
```

## 4. Backbone architecture (DAT)

A hierarchical 4-stage pyramid (built on the Swin codebase):

- **Stem:** 4×4 conv, stride 4 → feature map at $H/4\times W/4$.
- **Stage transitions:** patch-merging conv (stride 2) halves resolution and doubles channels between stages → resolutions $H/4, H/8, H/16, H/32$.
- **Where deformable attention goes:** Stages **1–2** use only **local (window/shifted-window) attention** — at high resolution deformable sampling is costly and there is little semantics yet to guide offsets. Stages **3–4 alternate** a **local-attention block** and a **deformable-attention block** (local → deformable → local → deformable …), so global data-dependent context is injected where it matters most.
- **Block:** standard pre-norm Transformer — `LN → attn → +residual`, then `LN → MLP(GELU) → +residual`.

### Model variants (original DAT, ImageNet-1K @224)

| Model | Depths | ≈Params | ≈FLOPs | Top-1 | Swin counterpart |
| --- | --- | --- | --- | --- | --- |
| DAT-T | [2,2,6,2] | 28M | 4.6G | 82.0 | Swin-T 81.3 |
| DAT-S | [2,2,18,2] | 50M | 9.0G | 83.7 | Swin-S 83.0 |
| DAT-B | [2,2,18,2] | 88M | 15.8G | 84.0 | Swin-B 83.5 |

Head counts mirror Swin ([3,6,12,24]); embed dims 96 (T/S) and 128 (B). The extended **DAT++** raises these to 83.9 / 84.6 / 84.9 (and 85.9 at 384²).

## 5. Complexity

Full self-attention costs $O\big((HW)^2 C\big)$. DAT attends each of the $HW$ queries to only $N_s = H_G W_G = HW/r^2$ sampled keys, so attention drops to roughly

$$
O\big(HW\cdot N_s\cdot C\big) = O\big((HW)^2 C / r^2\big)
$$

plus the cheap offset network ($\sim k^2 HW C$). This is the same efficiency class as Swin's windowed attention, but with a **global, adaptive** sampling set rather than a fixed local window.

## 6. Usage (as a backbone)

DAT is a drop-in **backbone**: swap it for Swin/ResNet and attach task heads.

```python
# Conceptual forward of the deformable attention module (PyTorch-style)
q = x @ Wq                              # queries from full map
offsets = s * torch.tanh(offset_net(q)) # dwconv->GELU->1x1  -> (H_G, W_G, 2)
pos = (ref_grid + offsets).clamp(-1, 1) # deformed sampling locations in [-1,1]
x_sampled = F.grid_sample(              # bilinear φ(x; p+Δp) -> (C, H_G, W_G)
    x, pos, mode="bilinear", align_corners=True)
k = x_sampled @ Wk
v = x_sampled @ Wv
attn = softmax(q @ k.transpose(-2, -1) / sqrt(d) + rel_pos_bias)
z = (attn @ v)                          # then concat heads @ Wo
```

Typical use: `DAT backbone → FPN → Mask R-CNN / Cascade R-CNN` for detection (DAT-Detection repo), or `DAT backbone → UperNet` for segmentation (DAT-Segmentation repo). Classification uses global-avg-pool → linear head.

## 7. Results (original paper vs Swin)

- **ImageNet-1K:** +0.7 top-1 over Swin at each scale.
- **COCO detection/instance seg:** +1.1 box AP and +1.1 mask AP.
- **ADE20K semantic seg (UperNet):** +1.2 mIoU.
- **Small/large objects:** up to +2.1 AP — the adaptive sampling helps most where fixed windows struggle.

## 8. Strengths & limitations

**Strengths:** data-dependent global receptive field at windowed-attention cost; few shared offsets (cheap, trainable, unlike per-query DCN); clean drop-in backbone; strong dense-prediction gains.

**Limitations:** offsets are shared per group (coarser than true per-query deformation); early high-res stages can't use it (still rely on local attention); `grid_sample` / relative-bias interpolation add implementation and kernel-efficiency overhead; benefits are largest in deeper/low-res stages.

## 9. Relation to the notes below

This is the canonical "deformable attention for a general vision backbone" reference. It generalizes **Deformable DETR**'s deformable attention (detection-head-only) into a full backbone, and replaces DCN's per-query convolutional offsets with grouped, attention-based sampling. The notes below cover that Deformable-DETR-style attention and a `torch.compile` gotcha it causes.

---

## Notes: Deformable attention in Deformable DETR (+ a torch.compile gotcha)

### What deformable attention is

Standard (dense) attention: a decoder query attends to every image feature token — compute $QK^\top$ over all $N$ tokens, softmax, weighted-sum the values. For a DETR decoder over multi-scale feature maps, $N$ is large (thousands of tokens), so cross-attention is expensive ($O(Q\cdot N)$) and slow to converge (the query has to learn to focus out of everything).

Deformable attention (from Deformable DETR) flips it: instead of looking at all $N$ tokens, each query looks at just a handful of sampled points (here `dec_n_points=2` per level per head). For each query it:

1. predicts **sampling offsets** — small (x, y) shifts from the query's reference point (a tiny linear layer),
2. predicts **attention weights** over those few points (softmax),
3. **bilinearly samples** the feature map at those continuous locations (`grid_sample`) and takes the weighted sum.

So rather than "compare against everything," it directly predicts *where* to look and *how much*. "Deformable" = the sampling pattern deforms per query (learned offsets), like deformable convolution. It's $O(Q\cdot n\_points)$ instead of $O(Q\cdot N)$ — cheap, multi-scale-friendly, and converges faster. (This is also why flash attention doesn't apply in the decoder — there's no dense $QK^\top$ to accelerate.)

The implementation detail that matters here: the multi-scale maps are flattened into one long value tensor, and to sample per level the code splits it back apart:

```python
value.split([height * width for height, width in shapes], dim=3)
```

Those split sizes come from `spatial_shapes` — a tensor.

### Why the torch.compile error occurs

`torch.compile` traces your Python into a graph using **FakeTensors** — symbolic stand-ins that know shapes but not values. The split above needs concrete Python ints (`height*width`) pulled out of the `spatial_shapes` tensor. Reading a value out of a tensor at trace time creates an **"unbacked symbolic int"** (`u0` in the log) — "some integer we won't know until runtime."

Dynamo then requires every unbacked symbol to be bound into the graph's outputs so it can track it downstream. Here the symbol produced by the data-dependent split never gets bound into a returned output → `PendingUnbackedSymbolNotFound: Pending unbacked symbols {u0} not in returned outputs`.

Root cause in one line: deformable attention has **data-dependent control flow** (split sizes taken from a tensor), which graph capture fundamentally struggles with. rfdetr's authors anticipated this (`capture_scalar_outputs=True`, `dynamic=True`) hoping those symbols would be *backed* (derived from input shapes), but in torch 2.12 the split still yields an unbacked one. It's also specifically the pure-PyTorch fallback doing the split in Python — the CUDA kernel would be one opaque op with nothing to trace, but it isn't installed here.

The choice was between two ways to keep deformable attention out of compilation:

- **rfdetr's **`**suppress_errors=True**` (the default you were hitting): dynamo tries to compile it, hits the error, then aborts and falls back to eager — but it does this by raising, catching, logging the whole traceback (the W-level spam), and retrying, fragmenting the graph. It limps along (that's why the run was at 0% GPU — stuck thrashing on compile retries).
- `**torch._dynamo.disable**` (the fix): marks the function as "never trace into this — treat it as an opaque eager call." Dynamo cleanly graph-breaks at the boundary: compiles everything up to deformable attention, runs it in plain eager (where the Python split is fine with real tensors), then resumes compiling after. No error is ever raised, no retry, no spam.

**Why it's the right call:**

- The deformable attention can't be compiled anyway (data-dependent split), so the best outcome is to exclude it cleanly — and a graph break is exactly "exclude this region." `disable` is the tool built for this.
- Non-invasive: patch the function reference from the train script, not the installed rfdetr package — nothing to maintain across reinstalls, no library edits.
- Keeps the benefit: the heavy ViT backbone still compiles/fuses; only the small deformable op runs eager (it was going to either way).
- The alternatives are worse: rewrite the library's deformable attention to precompute split sizes as static ints, or build the CUDA kernel — both far more invasive/fragile for the same end result.

**The honest trade-off:** each graph break splits the compiled region, so you lose fusion across the deformable-attention boundary and pay a small per-break cost — and the decoder has several such layers. So this makes compile clean and correct, but it also caps the upside (mostly backbone fusion). Which circles back to the real question: whether `--compile` is worth keeping at all for this model is a measurement.