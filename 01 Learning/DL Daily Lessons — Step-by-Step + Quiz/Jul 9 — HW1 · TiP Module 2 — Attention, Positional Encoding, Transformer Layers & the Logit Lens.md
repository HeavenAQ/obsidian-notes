---
base: "[[DL Daily Lessons — Step-by-Step + Quiz.base]]"
Quiz score: ""
"HW #": 1
Studied: false
Quiz taken: false
Day type: Lesson day
Date: 2026-07-09
Piece count: 5
---
> ⚠️ **Schedule note:** on plan — **Transformers in Practice, Module 2**: the internals of a decoder layer — scaled dot-product attention, multi-head attention, positional encoding, the residual-stream block, and the logit lens. Deep-dive companion: [Transformers in Practice](https://app.notion.com/p/391e445b30a4817dbfb5ef315a1f06ff) **§2 (attention & causal masking)** and **§7 (positional encodings)**. TiP is the **middle-priority** strand this block (HW1 math > Transformers-in-Practice > PT-Cert), so it gets full attention today. This directly sets up tomorrow's **Jul 10 Karpathy "build-a-GPT"** day, where you code exactly this by hand — and its manual attention pass is the *same* matrix-calculus/outer-product skill as **HW1 Problem 8 & 10**.

> 

> 🗒️ **Quiz backlog:** Jul 2–8 quizzes are all still untaken. Grade **at least yesterday's (Jul 8, Module 1)** before starting today (~10 min) — the patch-up loop stays dark until you record a score.

# 🎯 Today's goal

Open up the black box from yesterday. Module 1 treated the model as a next-token distribution $p_\theta(x_{t+1}\mid x_{1:t})=\mathrm{softmax}(W_U h_t)$; today you learn **how **$h_t$** is actually computed** — a stack of identical blocks, each doing *attention* (tokens exchange information) then a position-wise *MLP* (each token computes on what it gathered), tied together by a *residual stream* you can decode at any depth with the logit lens. By the end you should be able to write scaled dot-product attention and multi-head attention from scratch, explain why the $\tfrac{1}{\sqrt{d_k}}$ scale and positional encodings exist, and describe the block as "communication then computation." This is the theory Karpathy's nanoGPT turns into code tomorrow.

# 🧩 Pieces

## Piece 1 — Scaled dot-product attention: the one place tokens talk (TiP M2, ~35 min)

*Source: TiP Module 2, "attention." Deep dive: *[*Transformers in Practice*](https://app.notion.com/p/391e445b30a4817dbfb5ef315a1f06ff)* §2 (attention & causal masking).*

Each token's hidden vector is projected into three roles — a **query**, a **key**, and a **value** — by learned matrices: $Q=XW_Q,\ K=XW_K,\ V=XW_V$ where $X\in\mathbb{R}^{T\times d}$ is the sequence of $T$ token vectors. A query asks "what am I looking for?", a key advertises "what do I contain?", and the value is "what I'll pass on if attended to." Attention is:

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

with the **causal mask** $M_{ij}=-\infty$ for $j>i$ (and $0$ otherwise), so a token can never attend to the future — this is exactly what makes the model autoregressive rather than bidirectional. Read it in three steps: (1) $QK^\top$ is a $T\times T$ matrix of raw relevance scores (row $i$ = how much query $i$ matches every key $j$); (2) softmax over each row turns those into a probability distribution — the **attention weights**; (3) multiplying by $V$ takes a weighted average of value vectors, so each output position is a blend of the values it chose to attend to.

**Why the **$\sqrt{d_k}$**.** If $q,k$ have unit-variance i.i.d. entries, then the dot product $q^\top k=\sum_{c=1}^{d_k} q_c k_c$ has variance $d_k$ (sum of $d_k$ unit-variance terms). Undivided, logits grow like $\sqrt{d_k}$, pushing softmax into its saturated regime where one weight $\to 1$ and the rest $\to 0$; there the gradient of softmax is nearly zero and learning stalls. Dividing by $\sqrt{d_k}$ renormalizes the logits back to $\mathcal{O}(1)$ so softmax stays in its responsive range.

**Why the mask is **$-\infty$**, not a delete.** Setting illegal scores to $-\infty$ *before* softmax makes $e^{-\infty}=0$, so future tokens receive exactly zero weight while the row still normalizes cleanly over the allowed positions — the same logit-masking trick you saw for constrained decoding yesterday.

```python
import torch, torch.nn.functional as F

def attention(x, Wq, Wk, Wv, causal=True):
    # x: (T, d);  Wq,Wk,Wv: (d, dk)
    q, k, v = x @ Wq, x @ Wk, x @ Wv          # each (T, dk)
    dk = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / dk**0.5   # (T, T) relevance logits
    if causal:
        T = x.shape[0]
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()  # j > i
        scores = scores.masked_fill(mask, float('-inf'))
    weights = F.softmax(scores, dim=-1)        # rows sum to 1
    return weights @ v                          # (T, dk) context vectors
```

**You've got this piece when you can** write $\mathrm{Attn}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt{d_k}+M)V$ from memory, say what $Q,K,V$ mean and their shapes, derive why $q^\top k$ has variance $d_k$ and thus why you divide by $\sqrt{d_k}$, and explain why the causal mask uses $-\infty$ applied before softmax.

## Piece 2 — Multi-head attention: many relations at once (TiP M2, ~35 min)

*Source: TiP Module 2, "multi-head attention" + the head-visualization lab. Deep dive: *[*Transformers in Practice*](https://app.notion.com/p/391e445b30a4817dbfb5ef315a1f06ff)* §2.*

A single attention operation computes exactly **one** notion of relevance per query — one weighted average. But language needs many simultaneous relations (subject↔verb agreement, coreference, syntax, literal copying), and these live in different subspaces of the hidden state. So we run $H$ attention operations in parallel, each with its own low-dimensional projections, then concatenate and mix:

$$
\mathrm{head}_h=\mathrm{Attn}(XW_Q^h,\,XW_K^h,\,XW_V^h),\qquad \mathrm{MHA}(X)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_H)\,W_O
$$

Each head works in dimension $d_h=d/H$, so the total compute matches one full-width attention — you get $H$ different "relevance channels" for the same cost. The output projection $W_O\in\mathbb{R}^{d\times d}$ recombines the heads so downstream layers see one $d$-dimensional vector per position.

```python
def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads, causal=True):
    # x: (B, T, d);  Wq,Wk,Wv,Wo: (d, d)
    B, T, d = x.shape
    dh = d // n_heads
    def split(W):
        h = x @ W                                         # (B, T, d)
        return h.view(B, T, n_heads, dh).transpose(1, 2)  # (B, nh, T, dh)
    q, k, v = split(Wq), split(Wk), split(Wv)
    scores = (q @ k.transpose(-2, -1)) / dh**0.5          # (B, nh, T, T)
    if causal:
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
    w = F.softmax(scores, dim=-1)
    out = (w @ v).transpose(1, 2).contiguous().view(B, T, d)  # merge heads
    return out @ Wo
```

Note the `.transpose(1, 2).contiguous()` before `.view` — merging heads reorders memory, and `.view` refuses to run on a non-contiguous tensor, so the `.contiguous()` copy is required (a classic PyTorch gotcha worth remembering for tomorrow's build).

**Interpretable heads.** The course's head visualizations are essentially mini mechanistic interpretability: real trained models grow **previous-token heads**, **syntactic heads**, and famously **induction heads** that copy patterns `[A][B] ... [A] → [B]`. An induction head is a 2-step circuit spanning two layers — one layer's head writes "the token after the previous A," a later head reads it — which is the mechanism behind in-context learning.

**You've got this piece when you can** write the multi-head formula with $W_O$, explain *why* several small heads beat one big head (multiple relation subspaces at equal cost, $d_h=d/H$), read the `split`/merge code and say why `.contiguous()` is needed, and describe what an induction head does.

## Piece 3 — Positional encoding: putting order back in (TiP M2, ~30 min)

*Source: TiP Module 2, positional encoding. Deep dive: *[*Transformers in Practice*](https://app.notion.com/p/391e445b30a4817dbfb5ef315a1f06ff)* §7 (positional encodings — sinusoidal, RoPE, NoPE).*

Attention — the $QK^\top$ dot product, then softmax, then weighted sum of $V$ — is **permutation-invariant**: shuffle the input tokens and the output shuffles the same way, because nothing in the formula reads off a token's absolute position. Without a fix, "the cat sat on the mat" and "the mat sat on the cat" would look identical to attention. So order must be injected explicitly.

**Classic approach — absolute sinusoidal encodings** (Vaswani et al. 2017): add a fixed position vector to each token embedding, built from sines and cosines at geometrically-spaced frequencies:

$$
PE_{(pos,\,2i)}=\sin\!\left(\frac{pos}{10000^{2i/d}}\right),\qquad PE_{(pos,\,2i+1)}=\cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

The multi-frequency design means relative offsets are expressible as linear functions of these features, so the model can learn to attend "3 tokens back" fairly easily.

**Modern approach — RoPE (rotary position embedding).** Instead of *adding* a position vector, RoPE **rotates** each query and key by an angle proportional to its position. Because a dot product of two rotated vectors depends only on the *difference* of their angles, the relevance score $q_i^\top k_j$ naturally becomes a function of the **relative** position $i-j$ — giving relative-position awareness for free and generalizing better to longer sequences than absolute embeddings. The deep dive also flags **NoPE** (Kazemnejad et al. 2023): a causal-only decoder can length-generalize surprisingly well with *no* explicit positional encoding at all, because the causal mask already breaks the permutation symmetry.

**You've got this piece when you can** explain *why* attention needs positional information (permutation invariance), write the sinusoidal $PE$ formula, state in one sentence how RoPE differs (rotation → dot product depends on relative position $i-j$), and name what causal masking alone contributes (the NoPE observation).

## Piece 4 — The Transformer block: communication then computation (TiP M2, ~30 min)

*Source: TiP Module 2, "putting a layer together." Deep dive: *[*Transformers in Practice*](https://app.notion.com/p/391e445b30a4817dbfb5ef315a1f06ff)* §2 (residual stream).*

A decoder layer is two sublayers, each wrapped in a **pre-LayerNorm residual connection**:

$$
x \leftarrow x + \mathrm{MHA}(\mathrm{LN}(x))
$$

$$
x \leftarrow x + \mathrm{MLP}(\mathrm{LN}(x))
$$

where the position-wise MLP is $\mathrm{MLP}(h)=W_2\,\phi(W_1 h+b_1)+b_2$ with a GELU/ReLU nonlinearity $\phi$ and hidden width typically $4d$. The clean mental split the module stresses: **attention is the only place information moves *****across***** positions ("communication"), while the MLP lets each token *****compute***** on what it gathered, alone, with no cross-token mixing ("computation")**. LayerNorm and the MLP both act strictly position-wise.

**Why residual + pre-LN.** Writing each sublayer as $x + f(\mathrm{LN}(x))$ makes the block an *additive update* to a running vector — the **residual stream** — rather than a replacement. Think of the residual stream as a shared memory each layer reads from and writes small updates into. Putting LayerNorm *inside* the residual branch (pre-LN) keeps the skip path un-normalized, which gives clean gradient flow through deep stacks and is why modern LLMs train stably at 50–100+ layers.

```python
class Block(torch.nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1, self.ln2 = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)
        self.attn = torch.nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d, 4*d), torch.nn.GELU(), torch.nn.Linear(4*d, d))
    def forward(self, x, attn_mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]  # communication
        x = x + self.mlp(self.ln2(x))                                            # computation
        return x
```

**You've got this piece when you can** write the two pre-LN residual update lines, state the "communication (attention) vs. computation (MLP)" split and which operations are position-wise, and explain why the additive residual stream + pre-LN placement enables stable deep training.

## Piece 5 — The logit lens: reading the residual stream at every depth (TiP M2, ~25 min)

*Source: TiP Module 2, "decoding intermediate layers" lab. Deep dive: *[*Transformers in Practice*](https://app.notion.com/p/391e445b30a4817dbfb5ef315a1f06ff)* §residual-stream / interpretability.*

Because each layer only *adds* to the residual stream (Piece 4), the stream carries a running, always-decodable estimate of the answer rather than replacing it layer to layer. The **logit lens** exploits this: apply the model's own unembedding $W_U$ to an *intermediate* layer's hidden state to see what the model "would predict" if it stopped there:

$$
\text{logits}^{(\ell)}_t = W_U\,\mathrm{LN}\!\left(h^{(\ell)}_t\right)
$$

Run across layers $\ell$, the top prediction usually **sharpens gradually with depth** — early layers are diffuse, later layers concentrate — which shows the network doing *iterative refinement*, not one computation dumped at the end. It works *because* of the additive residual stream: the same coordinate directions that $W_U$ reads at the final layer are already partially present in the middle.

**Caveat — superposition.** Anthropic's *Mathematical Framework for Transformer Circuits* warns that individual neurons often don't map to single interpretable features: models represent **more sparse features than they have neurons** by packing them into near-orthogonal (not fully orthogonal) directions and tolerating a little interference. So the logit lens reveals *directions in the residual stream*, not tidy one-neuron-one-concept structure.

```python
@torch.no_grad()
def logit_lens(model, hidden_states, W_U, ln_f):
    # hidden_states: list of (T, d) residual-stream snapshots, one per layer
    for ell, h in enumerate(hidden_states):
        logits = ln_f(h[-1]) @ W_U.T          # decode the LAST position at layer ell
        top = logits.argmax().item()
        print(f"layer {ell:2d} -> token {top}")
```

**You've got this piece when you can** state what the logit lens computes ($W_U$ applied to an intermediate $h^{(\ell)}_t$), explain why the additive residual stream is what makes it work, describe the "predictions sharpen with depth = iterative refinement" observation, and state the superposition caveat in one sentence.

# 📝 Review quiz

Answer all 8 without notes. Grade with the key at the bottom, then fill in **Quiz score** and check **Quiz taken**.

1. **(Conceptual)** Write the scaled dot-product attention formula including the causal mask, and give the shapes of $Q,K,V$ and of $QK^\top$. Explain in one sentence what each of the three steps (score, softmax, weighted-sum) does.
2. **(Derivation)** Show that if $q,k\in\mathbb{R}^{d_k}$ have unit-variance i.i.d. entries then $q^\top k$ has variance $d_k$, and use this to justify the $\tfrac{1}{\sqrt{d_k}}$ factor. What failure mode appears if you drop it?
3. **(Conceptual)** Why is the causal mask implemented by adding $-\infty$ to scores *before* softmax rather than by deleting entries afterward? What property of softmax makes this exact?
4. **(Conceptual / code)** Write the multi-head attention formula with $W_O$. Explain why $H$ heads of dimension $d_h=d/H$ are preferable to one head of dimension $d$. In the `multi_head_attention` code, why is `.contiguous()` needed before `.view` when merging heads?
5. **(Conceptual)** Attention is permutation-invariant — what does that mean concretely, and why does it force us to add positional information? Contrast absolute sinusoidal encoding with RoPE in one sentence each (say specifically what RoPE makes the score $q_i^\top k_j$ depend on).
6. **(Conceptual)** Write the two pre-LN residual update equations for one Transformer block. Explain the "communication vs. computation" split and state which operations act position-wise (i.e. never mix across tokens).
7. **(Conceptual)** Why does putting LayerNorm *inside* the residual branch (pre-LN) and keeping the block as an additive update help train deep stacks? What is the "residual stream" as a mental model?
8. **(Conceptual / code)** What does the logit lens compute, and why does the *additive* residual stream make it meaningful? What does "predictions sharpen with depth" tell you about how the network computes, and what does superposition warn you not to conclude?

> [!note]+ 🔑 Answer key — open only after attempting all 8
> **1.** $\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\big(\tfrac{QK^\top}{\sqrt{d_k}}+M\big)V$, with $M_{ij}=-\infty$ for $j>i$. Shapes: $Q,K,V\in\mathbb{R}^{T\times d_k}$ (here $Q\in\mathbb{R}^{T\times d_k}$, $K,V$ likewise), and $QK^\top\in\mathbb{R}^{T\times T}$. Step 1 ($QK^\top$): raw relevance score of every query–key pair. Step 2 (row softmax): turns each query's scores into attention weights summing to 1. Step 3 ($\cdot V$): weighted average of value vectors = each token's context vector.
> 
> **2.** $q^\top k=\sum_{c=1}^{d_k} q_c k_c$. With $q_c,k_c$ independent, mean 0, variance 1, each product $q_c k_c$ has mean 0 and variance $\mathrm{Var}(q_c)\mathrm{Var}(k_c)=1$, and the $d_k$ terms are independent, so $\mathrm{Var}(q^\top k)=d_k$ (std $=\sqrt{d_k}$). Undivided logits therefore scale like $\sqrt{d_k}$; large logits push softmax into saturation where one weight $\to1$, the rest $\to0$, and its gradient $\to0$ (vanishing gradients / no learning). Dividing by $\sqrt{d_k}$ restores $\mathcal{O}(1)$ logits.
> 
> **3.** Because softmax exponentiates: $e^{-\infty}=0$, so a masked position gets *exactly* zero weight while the remaining (allowed) positions still normalize to sum 1. Adding $-\infty$ before softmax is thus equivalent to renormalizing only over the legal positions, done in one vectorized op with a fixed-shape $T\times T$ tensor (no ragged deletes). Deleting after softmax would leave the row un-normalized.
> 
> **4.** $\mathrm{MHA}(X)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_H)W_O$, $\mathrm{head}_h=\mathrm{Attn}(XW_Q^h,XW_K^h,XW_V^h)$. One head computes a single weighted average — one relation. $H$ heads of width $d_h=d/H$ give $H$ independent relevance subspaces (agreement, coreference, copy, syntax) at the *same* total compute as one width-$d$ head, and $W_O$ recombines them. `.contiguous()` is needed because `.transpose(1,2)` returns a non-contiguous view and `.view` requires contiguous memory to reinterpret the shape; without the copy `.view` raises (use `.reshape` to auto-fallback).
> 
> **5.** Permutation-invariant: permuting the input tokens permutes the outputs identically — the formula never reads absolute position, so "cat sat on mat" and "mat sat on cat" are indistinguishable. Hence we inject order. *Absolute sinusoidal:* add fixed $\sin/\cos$ position vectors (multi-frequency) to the embeddings. *RoPE:* rotate $q,k$ by a position-proportional angle, so $q_i^\top k_j$ depends on the **relative** offset $i-j$ (and generalizes better to longer contexts).
> 
> **6.** $x\leftarrow x+\mathrm{MHA}(\mathrm{LN}(x))$ then $x\leftarrow x+\mathrm{MLP}(\mathrm{LN}(x))$. Attention is the only sublayer that moves information *across positions* (communication); the MLP processes each position independently (computation). LayerNorm and the MLP are both position-wise — they never mix tokens. Only attention does cross-token mixing.
> 
> **7.** Writing each sublayer as $x+f(\mathrm{LN}(x))$ makes the block an additive update to a running vector (the **residual stream** — a shared memory each layer reads and writes small increments into), so gradients flow through the un-normalized skip path unimpeded; normalizing *inside* the branch (pre-LN) keeps that skip clean, avoiding the vanishing/exploding gradients that post-LN deep stacks suffer, so 50–100+ layers train stably.
> 
> **8.** The logit lens applies the unembedding to an intermediate layer: $\text{logits}^{(\ell)}_t=W_U\,\mathrm{LN}(h^{(\ell)}_t)$. It is meaningful because the residual stream is *additive* — the answer-carrying directions $W_U$ reads at the end are already partially written in earlier layers, so decoding mid-stack is sensible. Predictions sharpening with depth shows the model doing **iterative refinement** (not one end-of-stack computation). Superposition warns you *not* to read individual neurons as single features: models pack more sparse features than neurons into near-orthogonal directions, so the lens shows residual-stream *directions*, not one-neuron-one-concept.
> 
> **Scoring:** 1 pt each (half credit for the right idea with a wrong detail). ≥ 6/8 → move on; below → tomorrow opens with a targeted patch-up on the missed pieces.