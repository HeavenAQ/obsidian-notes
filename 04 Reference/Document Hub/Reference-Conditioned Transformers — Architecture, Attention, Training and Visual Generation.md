---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-17T11:58:00
Status: Done
Last updated time: 2026-07-17T11:58:00
Last edited by: Heaven Chen
Category:
  - CV
  - Transformers
  - Generative AI
  - Diffusion
  - Multimodal
---

# Reference-Conditioned Transformers — Architecture, Attention, Training, and Visual Generation

> [!abstract]
> A **reference-conditioned Transformer** generates or predicts a target while attending to one or more reference examples. In visual generation, the reference may specify identity, object appearance, texture, style, layout, motion, or an image to edit. The central design problem is not merely “add an image encoder”; it is deciding **what the reference means, how its tokens interact with target tokens, how to prevent copying or ignoring, and how to scale to multiple references**.

> [!important] Terminology
> “Reference-conditioned Transformer” is a **design family**, not one universally agreed architecture. Papers use overlapping terms such as *reference-guided*, *image-prompted*, *subject-driven*, *customized*, *in-context generation*, and *reference attention*. This note unifies those mechanisms rather than describing a single named model.

**Research snapshot:** 2026-07-17. Primary sources were checked before writing this note.

## 1. The problem formulation

Let

- $x$ be the desired target image or video;
- $r_{1:m}$ be one or more visual references;
- $c$ be an optional text instruction;
- $s$ be optional spatial control such as a mask, pose, depth map, or bounding box.

The general objective is to model

$$
p_\theta(x\mid r_{1:m},c,s).
$$

The output should preserve only the **intended factors** from the references:

| Reference intent | Preserve | Allow to change |
| --- | --- | --- |
| Subject/identity | face, clothing, distinctive details | pose, background, camera, lighting |
| Object customization | object identity and texture | placement, viewpoint, illumination |
| Style reference | palette, brushwork, rendering language | semantic content and layout |
| Editing context | unedited regions and global identity | instructed region/property |
| Motion reference | dynamics, rhythm, trajectory | appearance or scene |
| Same-scene restoration | geometry and texture correspondences | missing/corrupted pixels |

The difficulty is **factor ambiguity**. The pixels in a reference simultaneously encode identity, pose, background, lighting, camera, and style. A model trained without clear task signals may copy everything, preserve the wrong factor, or ignore the reference.

## 2. Paper map and lineage

| Year | Paper | What it contributes |
| ---: | --- | --- |
| 2018 | [Reference-Conditioned Super-Resolution by Neural Texture Transfer](https://arxiv.org/abs/1804.03360) | Early formulation of reference-conditioned visual restoration: transfer high-resolution texture from a separate image while preserving target content. CNN-based, but establishes the content/reference separation problem. |
| 2023 | [Scalable Diffusion Models with Transformers (DiT)](https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf) | Establishes patch-token diffusion Transformers and scalable conditioning via adaptive normalization. It is the backbone concept behind later reference-token DiTs. |
| 2023/2025 | [TransRef: Multi-Scale Reference Embedding Transformer](https://arxiv.org/abs/2306.11528) | Reference-guided inpainting with explicit patch alignment, style harmonization, and multi-scale reference attention. |
| 2023 | [IP-Adapter](https://arxiv.org/abs/2308.06721) | Lightweight image prompting through **decoupled text and image cross-attention**; only the reference projection and image K/V paths are trained. |
| 2023/2024 | [Animate Anyone](https://openaccess.thecvf.com/content/CVPR2024/papers/Hu_Animate_Anyone_Consistent_and_Controllable_Image-to-Video_Synthesis_for_Character_Animation_CVPR_2024_paper.pdf) | ReferenceNet processes a reference image in a parallel network and merges spatial detail features into the denoiser through spatial attention. |
| 2024 | [AnyDoor](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_AnyDoor_Zero-shot_Object-level_Image_Customization_CVPR_2024_paper.html) | Separates identity and detail features for zero-shot object placement, with spatial control and training pairs derived partly from video. |
| 2024 | [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) | Introduces the MMDiT-style joint multimodal Transformer: separate modality parameters with bidirectional information flow between text and image tokens. |
| 2024/2025 | [OmniGen](https://openaccess.thecvf.com/content/CVPR2025/papers/Xiao_OmniGen_Unified_Image_Generation_CVPR_2025_paper.pdf) | Treats text, reference images, and output latents as a unified sequence, enabling editing and subject-driven generation without task-specific plugins. |
| 2025 | [FLUX.1 Kontext](https://arxiv.org/abs/2506.15742) | Appends context-image latent tokens to target tokens in a flow-matching Transformer; one model handles generation, editing, character reference, and style reference. |
| 2025 | [OmniGen2](https://arxiv.org/abs/2506.18871) | Extends unified generation with separate text/image decoding pathways, a decoupled image tokenizer, in-context training data, and the OmniContext benchmark. |
| 2026 | [Conditional Text-to-Image Generation with Reference Guidance](https://openaccess.thecvf.com/content/WACV2026/html/Kim_Conditional_Text-to-Image_Generation_with_Reference_Guidance_WACV_2026_paper.html) | Small expert reference plugins for scene text, multilingual text, and logos, illustrating task-specific auxiliary objectives. |
| 2026 | [RefDecoder](https://arxiv.org/abs/2605.15196) | Conditions the **video VAE decoder**, not only the denoiser, using reference attention to recover high-frequency identity and scene detail. |
| 2026 | [Keep The Essentials: Efficient Reference Conditioned Generation via Token Dropping](https://arxiv.org/abs/2606.23682) | Sparse Context: random reference-token dropping during fine-tuning plus task-aware token selection at inference for efficient single/multi-reference generation. |

## 3. Background: from images to Transformer tokens

### 3.1 Patch tokenization

For an image or latent feature map $X\in\mathbb{R}^{H\times W\times C}$ and patch size $P$, divide it into

$$
N=\frac{H}{P}\frac{W}{P}
$$

patches. Flatten and project each patch:

$$
z_i=W_e\operatorname{vec}(X_i)+b_e+p_i,
$$

where $p_i$ is a positional embedding. A video adds time, giving coordinates $(t,h,w)$ and typically

$$
N=\frac{T}{P_t}\frac{H}{P_h}\frac{W}{P_w}.
$$

### 3.2 Scaled dot-product attention

Given token matrix $Z\in\mathbb{R}^{N\times d}$,

$$
Q=ZW_Q,\qquad K=ZW_K,\qquad V=ZW_V,
$$

$$
\operatorname{Attn}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V.
$$

$M$ is an optional attention mask or bias. Multi-head attention performs this in parallel subspaces and concatenates the results.

The details of cross-attention are also covered in [[04 Reference/Document Hub/Cross-Attention Queries from One Sequence, Keys-Values from Another|Cross-Attention: Queries from One Sequence, Keys/Values from Another]].

## 4. The main conditioning architectures

## 4.1 Global-vector conditioning

A reference encoder produces a pooled embedding

$$
e_r=E_r(r)\in\mathbb{R}^{d_r}.
$$

It can modulate each Transformer block using adaptive normalization:

$$
\operatorname{AdaLN}(h;e_r)
=\gamma(e_r)\odot\frac{h-\mu(h)}{\sigma(h)}+\beta(e_r).
$$

**Advantages:** cheap, fixed cost regardless of reference resolution, easy to combine with timestep/class embeddings.

**Limitation:** pooling discards spatial detail. It works for coarse style or semantics but struggles with logos, faces, fine clothing patterns, and exact object texture.

## 4.2 Reference cross-attention

Encode the reference as a token sequence

$$
R=E_r(r)\in\mathbb{R}^{N_r\times d}.
$$

Target features query the reference memory:

$$
Q=H_tW_Q,\qquad K_r=RW_K,\qquad V_r=RW_V,
$$

$$
H'_t=H_t+\lambda_r\operatorname{softmax}
\left(\frac{QK_r^\top}{\sqrt{d_h}}+B\right)V_r.
$$

$\lambda_r$ controls reference strength and $B$ can encode spatial masks, relative positions, correspondence priors, or reference identity.

This is a **read-only memory** design: target tokens read reference information, while reference tokens are not updated by target content.

### Complexity

Cross-attention costs approximately

$$
O(N_tN_rd),
$$

and its attention matrix occupies $O(N_tN_r)$ memory. Multiple dense references make $N_r$ the primary bottleneck.

## 4.3 Decoupled text and image cross-attention (IP-Adapter family)

If text and image tokens are simply concatenated, they compete inside one softmax. IP-Adapter instead computes separate paths using the same target queries:

$$
H_{text}=\operatorname{Attn}(Q,K_c,V_c),
$$

$$
H_{ref}=\operatorname{Attn}(Q,K_r,V_r),
$$

$$
H'=H+H_{text}+\lambda_rH_{ref}.
$$

The original IP-Adapter freezes the base diffusion model and trains the reference projection plus the new image-attention K/V projections. Its paper reports roughly **22M trainable parameters** for the adapter.

**Why decoupling helps:**

- text and image do not share one normalization competition;
- their strengths can be adjusted independently;
- the text path remains compatible with the pretrained base model;
- reference modules can be composed with other controls.

**What it does not solve:** the image embedding remains semantically ambiguous. A natural reference may represent identity, style, composition, or all three unless the training task or instruction disambiguates it.

## 4.4 Parallel reference network (ReferenceNet family)

Animate Anyone uses a reference network structurally related to the denoising network. The reference passes through corresponding blocks, producing multi-resolution spatial features

$$
R^{(1)},R^{(2)},\ldots,R^{(L)}.
$$

At layer $\ell$, target features attend to or merge with $R^{(\ell)}$:

$$
H_t^{(\ell+1)}
=H_t^{(\ell)}+
\operatorname{RefAttn}\left(H_t^{(\ell)},R^{(\ell)}\right).
$$

This preserves detail better than a single pooled embedding because reference information is available at matching spatial resolutions.

**Trade-off:** a second encoder/UNet-like path increases compute and memory, and aligned layer structures couple the adapter to the backbone.

## 4.5 Joint or in-context token attention

Modern DiT systems can concatenate target, reference, and sometimes text tokens:

$$
Z=[Z_t;R_1;\ldots;R_m;C].
$$

Self-attention then operates over the joint sequence:

$$
Z'=Z+\operatorname{MHA}(Z).
$$

FLUX.1 Kontext appends context-image latent tokens to target-image tokens and processes them in the visual stream. OmniGen similarly places multimodal inputs into a unified sequence with specialized attention masks and positional treatment.

**Advantages:**

- bidirectional interaction can reason jointly over target and references;
- multiple images fit the same input format;
- a single model can learn editing, style reference, and subject reference;
- fewer task-specific adapter interfaces.

**Cost:** with $N=N_t+\sum_jN_{r_j}+N_c$,

$$
\text{self-attention cost}=O(N^2d).
$$

Reference count and resolution can therefore dominate runtime.

## 4.6 Multimodal joint attention with separate parameters

MMDiT-style blocks allow modalities to have separate projections while joining their attention computation. For target/image and condition tokens,

$$
Q=[Q_x;Q_c],\qquad K=[K_x;K_c],\qquad V=[V_x;V_c],
$$

but

$$
Q_x=XW_Q^x,\quad Q_c=CW_Q^c
$$

use different weights. Attention is joint:

$$
O=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V.
$$

The outputs are split back into their modality streams. This preserves modality-specific processing while allowing bidirectional communication.

## 4.7 Correspondence-aware multi-scale fusion (TransRef family)

When target and reference depict the same scene but are not aligned, naive attention can transfer the wrong patch. TransRef separates two jobs:

1. **Reference-patch alignment (Ref-PA):** find spatially corresponding reference patches and harmonize style statistics.
2. **Reference-patch Transformer (Ref-PT):** refine and fuse matched fine-grained features.

At multiple scales $\ell$:

$$
\tilde{R}^{(\ell)}
=\operatorname{Align}\left(R^{(\ell)},H^{(\ell)}\right),
$$

$$
H'^{(\ell)}
=\operatorname{Fuse}\left(
H^{(\ell)},
\operatorname{RefPT}(H^{(\ell)},\tilde{R}^{(\ell)})
\right).
$$

This architecture is appropriate for restoration, novel-view completion, and repeated-scene data where geometric correspondence is meaningful.

## 5. Positional and modality encoding

Without explicit identity, the Transformer cannot know which tokens are target, reference 1, reference 2, or text. A common representation is

$$
z_i=e_i+p^{space}_i+p^{time}_i+p^{modality}_i+p^{reference}_i.
$$

Useful encodings include:

- **2D/3D RoPE:** encodes image or video coordinates;
- **modality ID:** target, reference, text, mask, pose, depth;
- **reference ID:** distinguishes multiple input images;
- **segment/frame coordinate:** prevents references from being treated as adjacent target patches;
- **camera/view embedding:** needed when reference and output viewpoints differ;
- **mask/bounding-box embedding:** tells the model where a reference is allowed to act.

FLUX.1 Kontext uses a sequence-concatenation design with three-dimensional positional coordinates for visual tokens. The extra axis distinguishes context and target sequences rather than pretending that their image grids are one larger image.

## 6. Attention masks define the information contract

Let token groups be target $T$, references $R$, and text $C$. Different masks implement different semantics:

| Mask | Allowed information flow | Use |
| --- | --- | --- |
| Target queries reference | $T\leftarrow R$ | cross-attention memory |
| Bidirectional target/reference | $T\leftrightarrow R$ | joint in-context reasoning |
| References isolated from each other | $R_i\not\leftrightarrow R_j$ | reduce unwanted identity/style mixing |
| Reference-region mask | selected $T$ positions $\leftarrow R$ | localized edits |
| Causal target + visible references | generated $T_i$ reads $T_{<i},R,C$ | autoregressive generation |
| Target-only loss mask | all inputs visible, loss only on $T$ | diffusion/flow in-context training |

Masking is not merely an efficiency detail. It encodes whether the reference is global context, local evidence, a read-only memory, or a jointly reasoned input.

## 7. Diffusion and flow-matching objectives

## 7.1 Noise-prediction diffusion

For clean latent $x_0$, sample noise $\epsilon\sim\mathcal{N}(0,I)$ and timestep $t$:

$$
x_t=\sqrt{\bar\alpha_t}x_0+sqrt{1-\bar\alpha_t}\epsilon.
$$

The reference-conditioned denoiser predicts

$$
\epsilon_\theta(x_t,t,r,c,s).
$$

The standard objective is

$$
\mathcal{L}_{diff}
=\mathbb{E}\left[
\left\|\epsilon-epsilon_\theta(x_t,t,r,c,s)\right\|_2^2
\right].
$$

Other parameterizations predict $x_0$, velocity $v$, or a score.

## 7.2 Rectified flow / flow matching

Using data latent $x$ and Gaussian noise $\epsilon$, define a straight interpolation

$$
z_t=(1-t)x+t\epsilon,\qquad t\sim U(0,1).
$$

One sign convention uses target velocity

$$
u_t=\epsilon-x.
$$

Train

$$
\mathcal{L}_{FM}
=\mathbb{E}\left[
\left\|v_\theta(z_t,t,r,c,s)-u_t\right\|_2^2
\right].
$$

FLUX/SD3-family models use rectified-flow-style objectives with Transformer backbones. Papers may reverse the interpolation direction, changing the sign of the velocity; the implementation must use one consistent convention.

## 7.3 Loss is applied to target tokens

In joint-sequence training, the reference is context rather than a reconstruction target. If the model output contains predictions for all visual tokens, use a loss mask:

$$
\mathcal{L}
=\frac{1}{|T|}\sum_{i\in T}
\left\|\hat{u}_i-u_i\right\|^2.
$$

Otherwise the model may waste capacity reconstructing visible references instead of learning how they guide the target.

## 8. Classifier-free guidance with multiple conditions

For unconditional prediction $f_\varnothing$ and full conditional prediction $f_{r,c}$,

$$
\hat f=f_\varnothing+w(f_{r,c}-f_\varnothing).
$$

To control text and reference separately, train with independent condition dropout and combine branches, for example:

$$
\hat f
=f_\varnothing
+w_c(f_c-f_\varnothing)
+w_r(f_{r,c}-f_c).
$$

Interpretation:

- $w_c$ strengthens text relative to unconditional generation;
- $w_r$ strengthens the reference after accounting for text.

Very high $w_r$ often produces literal copying, reduced editability, oversaturation, or identity/background entanglement.

## 9. Training data: the real bottleneck

The model needs tuples

$$
(r_{1:m},c,s,x)
$$

where the relationship between reference and output teaches the intended invariance.

### 9.1 Pair construction strategies

| Strategy | What it teaches | Risk |
| --- | --- | --- |
| Same image + synthetic edit | exact preservation outside edit | trivial pixel copying |
| Same identity across images | identity under pose/scene change | identity-label/privacy issues |
| Same object across video frames | viewpoint/lighting variation | near-duplicate leakage |
| Same scene across cameras | geometry-aware correspondence | calibration/alignment bias |
| Style/content recombination | style transfer | weak or subjective style labels |
| Generated triplets | scalable controlled supervision | teacher artifacts become shortcuts |
| Random unrelated reference | condition robustness/negative examples | can teach reference ignoring if overused |

### 9.2 Condition dropout

Randomly drop text, references, masks, or individual reference tokens:

$$
r'=m_r\odot r,\qquad m_r\sim\operatorname{Bernoulli}(1-p_r).
$$

This enables classifier-free guidance and prevents the network from depending on a condition that may be missing at inference.

### 9.3 Preventing trivial copying

- use different views/frames of the same subject;
- apply geometry-preserving but appearance-altering augmentation;
- train on edits requiring structural change;
- mask or crop reference regions;
- enforce target-specific spatial controls;
- include “preserve style, change content” and “preserve identity, change style” contrastive tasks;
- ensure reference and target duplicates never cross dataset splits.

## 10. Reference interpretation and factor control

A reference image alone is under-specified. Useful control mechanisms are:

### 10.1 Instruction-conditioned interpretation

Use text to state the factor:

- “use the person, change the clothing”;
- “copy only the watercolor style”;
- “place this object inside the marked box”;
- “edit the text while preserving the logo.”

Models such as OmniGen and FLUX.1 Kontext learn these behaviors from mixed instruction/reference tasks.

### 10.2 Separate encoders or tokens

Extract different representations:

$$
R_{id}=E_{id}(r),\quad
R_{style}=E_{style}(r),\quad
R_{detail}=E_{detail}(r),\quad
R_{geometry}=E_{geometry}(r).
$$

Fuse them with distinct attention paths or gates. AnyDoor's identity/detail distinction is an example of designing the representation around the desired invariances.

### 10.3 Region-aware reference control

For a target mask $m_i\in[0,1]$:

$$
H'_i=H_i+m_i\lambda_r\operatorname{RefAttn}(H_i,R).
$$

Region control reduces background leakage and makes multiple references compositional.

## 11. Multi-reference conditioning

With $m$ references, the simplest sequence is

$$
R=[R_1;R_2;\ldots;R_m].
$$

This creates three problems:

1. **Cost:** token count grows with every reference.
2. **Attribution:** the model may not know which reference controls which subject/region.
3. **Interference:** features from two identities or styles may blend.

### Better designs

- attach reference-ID embeddings;
- assign each reference a target mask or bounding box;
- use per-reference cross-attention followed by gated aggregation;
- allow target-to-reference attention but block reference-to-reference attention;
- retrieve only relevant reference patches per target token;
- compress each reference into learned resampler tokens;
- drop redundant tokens.

One gated formulation is

$$
o_i=\sum_{j=1}^{m}g_{ij}\operatorname{Attn}(q_i,K_j,V_j),
$$

$$
g_{ij}=\operatorname{softmax}_j(a(q_i,e_j,m_{ij})),
$$

where $e_j$ is a reference identity embedding and $m_{ij}$ is optional region compatibility.

## 12. Efficient reference conditioning

## 12.1 Token compression

Map $N_r$ dense reference tokens to $K\ll N_r$ learned tokens using a resampler:

$$
R_c=\operatorname{Attn}(Q_{learned},K_R,V_R),
\qquad R_c\in\mathbb{R}^{K\times d}.
$$

This has fixed downstream attention cost but may discard small logos, text, or facial details.

## 12.2 KV caching

Reference tokens do not change across denoising steps. Their projected keys and values can be cached:

$$
K_r=RW_K,\qquad V_r=RW_V.
$$

Caching saves projection work but not the $QK_r^\top$ attention product at every step.

## 12.3 Sparse Context / token dropping

The 2026 Sparse Context paper trains the model with random reference-token keep fractions, making it robust to incomplete reference grids. At inference, task-aware rules retain informative tokens:

- region-aware tokens for localized editing;
- anchor or diversity-oriented tokens for subject reference;
- prompt-relevant tokens for multi-reference generation.

If $S\subset\{1,\ldots,N_r\}$ is the retained set,

$$
R'=R_S,\qquad |S|=fN_r,\quad 0<f\le1.
$$

Cross-attention cost becomes

$$
O(N_t(fN_r)d).
$$

The paper reports more than $4\times$ inference speedup for settings with over five references at aggressive keep fractions, and up to $6.72\times$ when combined with KV caching in its six-reference evaluation. These are paper-specific results, not universal hardware guarantees.

## 12.4 Latent-space references

VAE latent tokens are much shorter than pixel tokens. They preserve spatial detail better than one CLIP vector while keeping attention tractable. However, VAE compression can remove exactly the high-frequency details required for identity or typography—one motivation behind RefDecoder's additional reference conditioning during decoding.

## 13. Reference-conditioned decoding

Most latent generators condition the denoiser but use an unconditional decoder:

$$
\hat{x}=D(z_0).
$$

RefDecoder instead uses

$$
\hat{x}=D(z_0,R),
$$

injecting reference tokens at multiple decoder upsampling stages. Conceptually:

$$
H_D^{(\ell+1)}
=\operatorname{Up}\left(H_D^{(\ell)}ight)
+\lambda_\ell\operatorname{RefAttn}
\left(H_D^{(\ell)},R^{(\ell)}\right).
$$

This attacks a different bottleneck: the latent may encode correct subject structure, yet the unconditional decoder may wash out skin texture, fine background detail, or object markings.

## 14. Minimal PyTorch building blocks

### 14.1 Cross-attention memory block

```python
import torch
from torch import nn


class ReferenceCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.norm_target = nn.LayerNorm(dim)
        self.norm_reference = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, target, reference, reference_padding_mask=None):
        # target:    [B, N_target, D]
        # reference: [B, N_reference, D]
        q = self.norm_target(target)
        kv = self.norm_reference(reference)
        update, weights = self.attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=reference_padding_mask,
            need_weights=True,
        )
        strength = torch.tanh(self.gate)
        return target + strength * update, weights
```

A zero-initialized gate lets a pretrained target model begin unchanged and learn to use the new reference path gradually.

### 14.2 Decoupled text and reference attention

```python
class DecoupledConditionAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.text_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ref_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ref_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, target, text_tokens, reference_tokens):
        q = self.norm(target)
        text_update, _ = self.text_attn(q, text_tokens, text_tokens)
        ref_update, _ = self.ref_attn(q, reference_tokens, reference_tokens)
        return target + text_update + torch.tanh(self.ref_scale) * ref_update
```

### 14.3 Joint token sequence with target-only loss

```python
def joint_reference_forward(model, noisy_target, references, text, target_velocity):
    # All inputs are already projected to the same model width.
    sequence = torch.cat([noisy_target, *references, text], dim=1)
    prediction = model(sequence)

    n_target = noisy_target.shape[1]
    predicted_target = prediction[:, :n_target]
    loss = torch.nn.functional.mse_loss(predicted_target, target_velocity)
    return loss
```

In a real model, add modality/reference IDs, positional coordinates, padding masks, timestep conditioning, and the correct output projection.

### 14.4 Random reference-token dropout

```python
def drop_reference_tokens(reference, keep_fraction, generator=None):
    """reference: [B, N, D]; keeps the same count for every batch item."""
    batch, tokens, dim = reference.shape
    keep = max(1, round(tokens * keep_fraction))
    scores = torch.rand(batch, tokens, device=reference.device, generator=generator)
    indices = scores.topk(keep, dim=1).indices
    gather_index = indices.unsqueeze(-1).expand(-1, -1, dim)
    return reference.gather(1, gather_index)
```

Random dropping is useful during fine-tuning. Inference should use a task-aware selector when exact regions or subjects matter.

## 15. A practical reference-conditioned DiT design

```text
reference image(s)
    │
    ├─ VAE / vision encoder ──► dense reference tokens
    │                              │
    │                       token selector/resampler
    │                              │
text instruction ─► text encoder ─┤
                                   ▼
target latent + noise ─► patchify + modality/reference/position embeddings
                                   │
                          joint or decoupled attention
                                   │
                          DiT / flow Transformer blocks
                                   │
                     target-token velocity/noise prediction
                                   │
                          ODE/diffusion sampling
                                   │
                       reference-conditioned decoder (optional)
                                   ▼
                              output image/video
```

Recommended default choices:

- **localized editing:** target/reference concatenation + edit mask + target-only loss;
- **subject/style prompting on a frozen model:** decoupled reference cross-attention adapter;
- **new unified foundation model:** joint visual tokens with explicit modality/reference IDs;
- **same-scene restoration:** correspondence-aware multi-scale reference attention;
- **many references:** resampling or learned token dropping plus KV caching;
- **identity-critical video:** multi-scale reference features and possibly a conditioned decoder.

## 16. Evaluation

No single metric captures reference fidelity and editability. Evaluate a vector of objectives.

### 16.1 Output quality

- FID/KID on a matched dataset;
- human preference;
- no-reference image/video quality metrics;
- temporal quality for video.

### 16.2 Reference fidelity

- CLIP/DINO similarity for semantic or subject preservation;
- face-recognition similarity for consenting identity datasets;
- local feature matching for texture/logo retention;
- style embeddings or human style judgments;
- OCR accuracy for reference text and logos.

### 16.3 Instruction and edit fidelity

- CLIP text-image alignment;
- instruction-following VLM evaluation;
- masked-region edit success;
- preservation score outside the edited mask;
- object placement accuracy or box IoU.

### 16.4 Diversity and copying

- LPIPS diversity across random seeds;
- nearest-neighbor retrieval against references/training data;
- pixel/perceptual similarity to detect trivial copy-paste;
- performance under pose, viewpoint, and background changes.

### 16.5 Multi-reference attribution

Evaluate each subject/region separately. A global average can hide identity mixing—for example, two reference subjects blended into one output.

## 17. Essential ablations

| Ablation | Question |
| --- | --- |
| No reference | Is the task solvable from text/context alone? |
| Shuffled reference | Does the model genuinely use the correct reference? |
| Same vs unrelated reference | Is reference relevance calibrated? |
| Pooled vs dense reference tokens | How much spatial detail is needed? |
| Cross-attention vs joint attention | Is bidirectional interaction worth its cost? |
| Coupled vs decoupled text/image attention | Do modalities compete? |
| With/without reference-ID embeddings | What causes multi-reference mixing? |
| Full vs dropped/resampled tokens | Quality–speed frontier |
| Denoiser-only vs conditioned decoder | Where is fine detail lost? |
| Random frame vs identity/video split | Is performance caused by leakage? |

## 18. Common failure modes

| Failure | Likely cause | Mitigation |
| --- | --- | --- |
| Reference ignored | weak path, excessive condition dropout, poor pairs | stronger/gated attention, better pairs, reference-use auxiliary loss |
| Literal copying | same-image training, excessive reference guidance | cross-view pairs, lower guidance, stronger target transformation |
| Background leakage | global reference embedding | masks, object crops, region-aware attention |
| Identity drift | compressed embedding, insufficient detail supervision | dense/multi-scale features, face/detail losses, conditioned decoder |
| Pose entanglement | reference pairs lack pose variation | same-subject cross-pose data, explicit pose control |
| Style/identity confusion | ambiguous image condition | instruction-conditioned factor labels, separate encoders/tokens |
| Multiple subjects blend | no reference attribution | reference IDs, per-subject masks, isolated attention paths |
| Wrong-patch transfer | geometric misalignment | correspondence/alignment module, camera/position encoding |
| Temporal flicker | frame-independent reference use | temporal attention, persistent reference memory, video loss |
| Runtime explosion | dense multi-reference tokens | token selection, resampling, KV cache, lower-resolution reference latents |

## 19. Safety, provenance, and data governance

Reference conditioning makes impersonation and copyrighted-style imitation easier. A production system should consider:

- consent and usage rights for identity references;
- reference provenance and retention policy;
- watermarking or generation metadata;
- safeguards for public-figure impersonation and intimate imagery;
- training-data deduplication and licensing;
- audit logs recording which references affected an output;
- leakage tests to distinguish generalization from memorized reconstruction.

## 20. How to read new papers in this area

For every claimed “reference-conditioned” model, answer:

1. **What factor is the reference supposed to control?**
2. **How is the reference encoded—pooled vector, dense grid, VAE latent, or multi-scale features?**
3. **Where is it injected—AdaLN, cross-attention, joint attention, skip connection, or decoder?**
4. **Is information flow one-way or bidirectional?**
5. **How are text and reference conditions separated or fused?**
6. **How are multiple references identified and spatially assigned?**
7. **What prevents copying, ignoring, and factor entanglement?**
8. **What is the token/memory cost per additional reference?**
9. **Were train/test splits separated by identity, object, video, or scene?**
10. **Do metrics measure both fidelity and editability?**

## 21. Bottom line

Reference conditioning converts a Transformer from “generate from a symbolic prompt” into “generate by comparing, retrieving, and transforming examples.” The best architecture depends on the required information bandwidth:

- use a **global vector** for coarse semantics or style;
- use **decoupled cross-attention** for lightweight, composable image prompting;
- use **multi-scale reference networks** for fine identity/detail preservation;
- use **joint in-context tokens** for unified editing and multi-image reasoning;
- use **correspondence-aware fusion** when target and reference share geometry;
- use **sparse tokens/resamplers and caching** when reference count grows;
- condition the **decoder** when the latent bottleneck loses critical detail.

The central engineering principle is:

> **Make the intended reference relationship explicit in the tokens, attention mask, training pairs, and evaluation. More reference pixels do not automatically mean better conditioning; they mean more information the model must correctly interpret.**
