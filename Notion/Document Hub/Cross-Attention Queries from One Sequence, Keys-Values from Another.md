---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-06T17:39:00
Status: Done
Last updated time: 2026-07-06T18:38:00
Last edited by: Heaven Chen
Category:
  - ML
  - Theory
  - LLM
---
Cross-attention is **the same scaled dot-product attention** as self-attention — the only difference is *where Q, K, V come from*:

$CrossAttention(X_q, X_kv) = softmax( (Q Kᵀ) / √dₖ ) V$

with

- $Q = X_q W_Q$ → queries come from sequence **A** (the "reader": decoder / target / one modality)
- $K = X_kv W_K$ → keys come from sequence **B** (the "context": encoder / source / memory / other modality)
- $V = X_kv W_V$ → values come from the **same** sequence B

Self-attention is the special case where $X_q = X_kv$ (everything from the same sequence).

---

## Self-attention vs cross-attention

|   | Self-attention | Cross-attention |
| --- | --- | --- |
| Q from | sequence X | sequence A (queries) |
| K, V from | sequence X | sequence B (context) |
| Lengths | all length L | $Q: L_q, K/V: L_kv$ (can differ) |
| Meaning | tokens mix **within** one sequence | sequence A **reads from** sequence B |
| Output length | L | $L_q$ (one vector per query) |
| Causal mask? | often (decoders) | **no** (queries may see all context) |

---

## The mental model

> For each query token, look over the *entire other sequence* and pull back a weighted summary of its values.

It is a differentiable, content-based **soft lookup** from an external memory:

- **Q** = “what am I looking for?”
- **K** = “what does each context item advertise?” (its address)
- **V** = “the content to actually retrieve”

The softmax is taken over the **context axis (L_kv)**, so every query gets a probability distribution over all context tokens, then returns Σ (weight × value).

---

## Shapes (single head)

- $X_q: [L_q, d]  ·  X_kv: [L_kv, d]$
- $Q: [L_q, dₖ]  ·  K: [L_kv, dₖ]  ·  V: [L_kv, d_v]$
- scores $QKᵀ: [L_q, L_kv]$(softmax along the L_kv axis)
- output: $[L_q, d_v]$  → **always one output per query**

Q and K/V may have **different sequence lengths**, but must share the projected key dimension dₖ.

---

## Where it is used

- **Transformer decoder (“Attention Is All You Need”):** each decoder layer first *self*-attends over generated tokens, then *cross*-attends with K,V = encoder outputs — this is how the output is conditioned on the input sentence.
- **Seq2seq:** translation, summarization, ASR (decoder reads encoder memory).
- **Object detection — DETR:** learned **object queries** cross-attend to image feature tokens; **Deformable DETR** makes this a *sparse* cross-attention that samples a few points per query.
- **Multimodal / conditioning:** text queries ↔ image features (Flamingo, BLIP-2 Q-Former, LLaVA-style), and **Stable Diffusion**'s U-Net cross-attends to text embeddings to condition image generation.
- **Perceiver / Perceiver IO:** a small set of **latent queries** cross-attends to a huge input array — exploiting the asymmetric cost.

---

## Multi-head & masking

- **Multi-head** cross-attention is identical to multi-head self-attention: split into h heads, run cross-attention per head, concat, project with$W_O$.
- **Padding mask** applies to the **K/V (context)** side — mask out padded context tokens (`key_padding_mask`).
- **Causal mask is NOT used** on cross-attention: a query is allowed to see the whole context. (The causal mask lives in the decoder's *self*-attention.)

---

## Code (PyTorch)

```python
import torch, torch.nn.functional as F

def cross_attention(x_q, x_kv, Wq, Wk, Wv, Wo, n_heads):
    Lq, d = x_q.shape; Lkv, _ = x_kv.shape
    dk = d // n_heads
    Q = (x_q  @ Wq).view(Lq,  n_heads, dk).transpose(0, 1)   # [h, Lq,  dk]
    K = (x_kv @ Wk).view(Lkv, n_heads, dk).transpose(0, 1)   # [h, Lkv, dk]
    V = (x_kv @ Wv).view(Lkv, n_heads, dk).transpose(0, 1)   # [h, Lkv, dk]
    scores = (Q @ K.transpose(-2, -1)) / dk**0.5             # [h, Lq, Lkv]
    attn   = scores.softmax(dim=-1)                          # over the CONTEXT axis
    out    = (attn @ V).transpose(0, 1).reshape(Lq, d)       # [Lq, d]
    return out @ Wo
```

With the built-in module — pass decoder states as `query`, encoder outputs as `key`/`value`:

```python
mha = torch.nn.MultiheadAttention(embed_dim=d, num_heads=8, batch_first=True)
out, attn_w = mha(query=dec_states, key=enc_out, value=enc_out,
                  key_padding_mask=enc_pad_mask)   # no causal mask here
```

---

## Cost

$O(L_q · L_kv · d)$— **asymmetric** in the two lengths. Cheap when one side is short: few queries over many context tokens (DETR object queries, Perceiver latents) or many queries over a short context. This asymmetry is exactly what Perceiver exploits to handle very large inputs.

---

## Key takeaway

Cross-attention = scaled dot-product attention where **queries come from one sequence and keys/values from another**. It is the bridge that lets a decoder condition on an encoder, or one modality read from another — a soft, content-based lookup whose output has one vector per query.

> **Memory sentence:** Self-attention mixes a sequence with itself; cross-attention lets one sequence (Q) read from another (K, V).