---
base: "[[01.2 DL Daily Lessons — Step-by-Step + Quiz.base]]"
Quiz score: ""
"HW #": 2
Studied: false
Quiz taken: false
Day type: Lesson-Day
Date: 2026-07-20
Piece count: 5
---
> 🗒️ **Quiz backlog nudge:** yesterday's **Jul 19 (Least Squares / Normal Equation / Projection)** quiz is still ungraded (`Quiz taken: false`, no score), as are the Jul 17 (Eigenvalues I) and Jul 18 (Positive Definiteness / Hessian) sets. Nothing to patch up today — the re-teach loop only fires on a *recorded* low score. Today is a change of gear from linear algebra into transformer theory, so it stands alone; but if you want spaced-repetition credit, grade the Jul 19 set (~10 min) first — the projection/PSD reasoning it drills is what makes "softmax of a Gram-like matrix $QK^\top$" feel natural below.

# 🎯 Today's goal

This is **HW2 Day 4** (Udemy LLM course → *Temporal causality via linear algebra*, *The "attention" algorithm (theory)*, *The Transformer block (theory)*, *Multi-head attention* — **theory only**; you'll write the code in the HW7 week, not now). By the end you should be able to derive scaled dot-product attention from scratch, say *why* every piece is shaped the way it is (the causal mask, the $\sqrt{d_k}$, the residual stream, the head split), and read a transformer forward pass as pure linear algebra. This feeds **HW2 Day 5** (MIT 6.S191 L2, Deep Sequence Modeling) directly, and the **permutation-invariance** theme here is the same idea that powers the Deep-Sets / message-passing aggregators in HW2 **§D (GNNs)** — attention and mean-aggregation are both "permutation-invariant weighted averages," so understanding one primes the other.

# 🧩 Pieces

## Piece 1 — Temporal causality via linear algebra: attention is a masked weighted average (~30 min)

*Source: Udemy LLM course → "Temporal causality via linear algebra" (Karpathy's "the trick in self-attention" / matrix-multiply-as-weighted-aggregation). Deep-dive: [[Transformers in Practice (DeepLearning.AI, Sharon Zhou) — Deep Dive]] §Module 2 → "Attention" (causal mask $M$) and §4 of the Additional Reading ("Causal masking & why attention is order-agnostic without it").*

Before any learned weights, grasp the core mechanical trick: **"let each token look at the tokens before it" is a single matrix multiply.** Suppose $X\in\mathbb{R}^{T\times d}$ is a sequence of $T$ token vectors. The crudest form of communication is: replace each token by the **average of itself and all earlier tokens** (a running mean / "bag of the past"). Averaging positions $1..i$ for row $i$ is exactly left-multiplying by a **lower-triangular, row-normalized matrix** $W_{\text{avg}}$:

$$
W_{\text{avg}} = \begin{pmatrix} 1 & 0 & 0\\ \tfrac12 & \tfrac12 & 0\\ \tfrac13 & \tfrac13 & \tfrac13\end{pmatrix},\qquad \hat X = W_{\text{avg}}\,X .
$$

Row $i$ has $1/i$ in its first $i$ entries and $0$ afterward, so $\hat x_i=\tfrac1i\sum_{j\le i}x_j$ — no information flows from the **future** ($j>i$). That zero-upper-triangle is **causality** expressed as a sparsity pattern.

The key generalization: those weights need not be a uniform $1/i$. Let $W$ be **any** lower-triangular matrix whose non-zero rows sum to 1; then $\hat X = WX$ is a **data-independent weighted average of the past**. Attention is precisely this, with two upgrades: (i) the weights are **computed from the data** (how relevant is token $j$ to token $i$?), and (ii) they're produced by a **softmax**, which is the differentiable way to turn arbitrary real scores into a non-negative, row-summing-to-1 weight matrix. The causal constraint is imposed by setting the future scores to $-\infty$ **before** the softmax, so $e^{-\infty}=0$ and those weights vanish:

$$
M_{ij}=\begin{cases}0 & j\le i\\ -\infty & j> i\end{cases},\qquad A=\operatorname{softmax}(S+M),\qquad \hat X = A\,V .
$$

```python
import torch, torch.nn.functional as F
T = 4
# (a) uniform running mean as a lower-triangular matmul
tril = torch.tril(torch.ones(T, T))
W_avg = tril / tril.sum(dim=1, keepdim=True)   # row-normalized lower-triangular
# (b) the SAME averaging, expressed as softmax over masked scores (data-independent scores = 0)
scores = torch.zeros(T, T)
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)   # strict upper = future
A = F.softmax(scores.masked_fill(mask, float('-inf')), dim=1)
print(torch.allclose(W_avg, A))   # True — masked-softmax of zero scores IS the running mean
```

**Why formulated this way.** Casting "attend to the past" as a triangular matrix multiply is what makes attention (a) **parallelizable** (one big matmul over all positions at once during training, unlike an RNN's sequential loop) and (b) **differentiable in the weights** — because the weights come from softmax of scores you can backprop into. The whole rest of attention is just "make $W$ learned and content-dependent."

**You've got this piece when you can** explain why a causal weighted-average-of-the-past is a lower-triangular row-stochastic matrix multiply $\hat X=WX$, why masking with $-\infty$ *before* softmax enforces "no peeking at the future," and why this matrix view is what lets attention train in parallel where an RNN cannot.

## Piece 2 — The attention algorithm (theory): queries, keys, values & scaled dot-product (~35 min)

*Source: Udemy LLM course → "The 'attention' algorithm (theory)." Deep-dive: [[Transformers in Practice (DeepLearning.AI, Sharon Zhou) — Deep Dive]] §Module 2 → "Attention" (the $Q,K,V$ projections and the $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$ formula).*

Now make the averaging weights **content-dependent**. Each token vector is linearly projected into three learned roles:

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,\qquad W_Q,W_K\in\mathbb{R}^{d\times d_k},\ W_V\in\mathbb{R}^{d\times d_v}.
$$

Read them as: **query** = "what am I looking for," **key** = "what do I contain, for matching," **value** = "what I broadcast if matched." The **compatibility** of query $i$ with key $j$ is their dot product $q_i^\top k_j$ — large when the two vectors point similarly in the learned subspace. Stack all pairs into the score matrix $S=QK^\top$, mask, softmax each row, and use the weights to average the **values**:

$$
\boxed{\operatorname{Attn}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V }
$$

This is Piece 1's $\hat X=AV$ with $A=\operatorname{softmax}(S/\sqrt{d_k}+M)$ — the weight matrix is now **learned and per-example** instead of a fixed running mean. Two structural facts worth internalizing:

- **Separating "who to attend to" ($Q,K$) from "what to move" ($V$)** gives the model two independent degrees of freedom. The $QK^\top$ block decides the *routing*; $V$ decides the *payload*. Tying them (using $X$ directly as both) would force "similar tokens" to also mean "copy this exact vector," which is far less expressive.
- **Cross-attention** is the identical formula with $Q$ from one sequence and $K,V$ from another (decoder attending to encoder) — see [[Cross-Attention Queries from One Sequence, Keys-Values from Another]]. Self-attention is the special case where all three come from the same $X$.

```python
import torch, torch.nn.functional as F
def attention(X, Wq, Wk, Wv, causal=True):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv          # (T, d_k), (T, d_k), (T, d_v)
    dk = Q.shape[-1]
    S = Q @ K.transpose(-2, -1) / dk**0.5     # (T, T) scaled scores
    if causal:
        T = X.shape[0]
        S = S.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float('-inf'))
    A = F.softmax(S, dim=-1)                   # row-stochastic attention weights
    return A @ V                               # weighted average of values
```

**You've got this piece when you can** state what $Q,K,V$ each represent, write $\operatorname{Attn}(Q,K,V)=\operatorname{softmax}(QK^\top/\sqrt{d_k}+M)V$ from memory, explain *why* keys/values are separated from queries and from each other, and describe the cross-attention variant.

## Piece 3 — WHY divide by $\sqrt{d_k}$: softmax saturation and vanishing gradients (~30 min)

*Source: the "why $\sqrt{d_k}$" theory point in the attention lecture. Deep-dive: read §1–§6 of [[Scaled Dot-Product Attention Why Divide by √dₖ]] (score-variance growth, the softmax Jacobian, saturated vs. non-saturated attention).*

This is the piece your priorities ("WHY the math is formulated that way") care about most. **Variance argument.** If the entries of $q$ and $k$ are i.i.d. with mean $0$ and variance $1$, then the raw score is a sum of $d_k$ independent product terms:

$$
S_{ij}=q_i^\top k_j=\sum_{m=1}^{d_k} q_{i,m}k_{j,m},\qquad \mathbb{E}[S_{ij}]=0,\qquad \operatorname{Var}(S_{ij})=\sum_{m=1}^{d_k}\operatorname{Var}(q_{i,m}k_{j,m})=d_k .
$$

So the scores have standard deviation $\sqrt{d_k}$ — they **grow with head dimension**. Dividing by $\sqrt{d_k}$ renormalizes the variance back to $1$, keeping the logits $\mathcal{O}(1)$ regardless of $d_k$.

**Why large logits are actively harmful.** Softmax turns large gaps between logits into an almost one-hot vector. Once a row of $A$ is essentially one-hot ("saturated"), the softmax **Jacobian** collapses. For $p=\operatorname{softmax}(s)$,

$$
\frac{\partial p_i}{\partial s_j}=p_i(\delta_{ij}-p_j),\qquad \text{so if } p\approx e_k \text{ (one-hot), then } p_i(\delta_{ij}-p_j)\approx 0 \ \text{ for all } i,j.
$$

A near-zero Jacobian means **no gradient flows back** through the attention weights — the routing can't learn. Scaling by $\sqrt{d_k}$ keeps the softmax in its **non-saturated** regime where the Jacobian has healthy magnitude, so gradients survive. This is exactly the same failure mode as a saturated sigmoid/tanh, just inside attention.

```python
import torch, torch.nn.functional as F
dk = 64
q = torch.randn(dk); k = torch.randn(dk, 1000)  # 1000 candidate keys
raw   = q @ k                      # scores ~ N(0, dk)  → std ≈ 8
scaled = raw / dk**0.5             # scores ~ N(0, 1)   → std ≈ 1
print(raw.std().item(), scaled.std().item())
# saturation check: max softmax prob (closer to 1.0 = more saturated = weaker gradient)
print(F.softmax(raw, 0).max().item(), F.softmax(scaled, 0).max().item())
```

**Why $\sqrt{d_k}$ and not $d_k$.** You want to normalize the **standard deviation** (which is $\sqrt{d_k}$), not the variance ($d_k$). Dividing by $d_k$ would over-shrink the logits, flattening the softmax toward uniform and destroying the model's ability to attend sharply — the opposite failure.

**You've got this piece when you can** derive $\operatorname{Var}(q^\top k)=d_k$, explain via the softmax Jacobian $p_i(\delta_{ij}-p_j)$ why saturated attention kills gradients, and say why the correct normalizer is $\sqrt{d_k}$ (std) rather than $d_k$ (variance).

## Piece 4 — The Transformer block (theory): residual stream, pre-LN & the MLP (~30 min)

*Source: Udemy LLM course → "The Transformer block (theory)." Deep-dive: [[Transformers in Practice (DeepLearning.AI, Sharon Zhou) — Deep Dive]] §Module 2 → "Layers and the residual stream," plus §5 (feed-forward/MLP) and §6 (residual connections & depth) of the Additional Reading.*

A transformer block wraps attention in two **residual** sub-layers with **layer norm**:

$$
x \leftarrow x + \operatorname{Attn}(\operatorname{LN}(x)),\qquad x \leftarrow x + \operatorname{MLP}(\operatorname{LN}(x)).
$$

Three "why"s:

- **The residual stream is a shared memory bus.** Because every sub-layer *adds* into $x$ (rather than replacing it), the hidden state at position $t$ is a running sum $x_t = e_t + \sum_\ell(\text{attn}_\ell + \text{mlp}_\ell)$. Each layer **reads** the stream, computes a small update, and **writes it back**. This additive structure is what lets gradients flow through dozens of layers unattenuated (the identity path has derivative $1$), and it's why the **logit lens** works: applying the unembedding $W_U$ to an *intermediate* $x^{(\ell)}_t$ already yields sensible-but-fuzzy predictions that sharpen with depth — layers are iterative refinement, not one computation at the end.
- **Pre-LN vs. post-LN.** Normalizing the sub-layer *input* ($x+\text{sublayer}(\operatorname{LN}(x))$, "pre-LN") keeps the residual path a clean identity and makes deep transformers train stably without careful warm-up; the original post-LN ($\operatorname{LN}(x+\text{sublayer}(x))$) put LN on the residual path and was notoriously finicky at depth. Modern decoders are pre-LN.
- **The MLP does the per-token computation.** Attention only *moves* information between positions; it applies no nonlinearity to the content beyond the value projection. The position-wise MLP — typically $x\mapsto W_2\,\phi(W_1 x)$ with a $4\times$ hidden expansion and a nonlinearity $\phi$ (GELU/ReLU) — is where each token **thinks** about what it just gathered. The standard mental model: **attention = communication (mix across positions), MLP = computation (transform each position)**, alternating.

```python
import torch, torch.nn as nn
class Block(nn.Module):
    def __init__(self, d, n_heads, mlp_mult=4):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_mult*d), nn.GELU(), nn.Linear(mlp_mult*d, d))
    def forward(self, x, attn_mask=None):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask)
        x = x + a                          # residual add (communication)
        x = x + self.mlp(self.ln2(x))      # residual add (computation)
        return x
```

**You've got this piece when you can** write the pre-LN block equations, explain the residual stream as an additive read/update/write bus (and why that helps gradient flow + enables the logit lens), contrast pre-LN vs post-LN, and articulate "attention communicates, MLP computes."

## Piece 5 — Multi-head attention (theory): why many small heads beat one big head (~25 min)

*Source: Udemy LLM course → "Multihead attention." Deep-dive: [[Transformers in Practice (DeepLearning.AI, Sharon Zhou) — Deep Dive]] §Module 2 → "Attention" (the `MultiHead` formula and the interpretable-heads discussion).*

A single attention operation computes **exactly one** notion of relevance per query — one weighted average. But language needs many relations at once (subject↔verb agreement, coreference, syntax, copy patterns), and they live in **different subspaces** of the hidden state. Multi-head attention runs $h$ independent, smaller attentions in parallel and concatenates:

$$
\operatorname{MultiHead}(X)=\operatorname{Concat}(\text{head}_1,\dots,\text{head}_h)\,W_O,\qquad \text{head}_i=\operatorname{Attn}(XW_Q^i,\,XW_K^i,\,XW_V^i),
$$

with $d_k=d_v=d/h$ so total compute matches one full-size head — you trade one $d$-dim relevance computation for $h$ heads of dimension $d/h$, **each free to specialize**. $W_O\in\mathbb{R}^{d\times d}$ then mixes the heads' outputs back into a single $d$-dim vector per position so the next layer sees one stream.

**Why the split works (and why $d/h$).** Splitting the $d$-dim space into $h$ orthogonally-projected subspaces lets each head learn a low-rank query/key geometry ($d\times(d/h)$ projections) tuned to one type of relation, without paying for a full $d\times d$ attention per relation. Empirically these specialize into interpretable circuits — **previous-token heads**, **induction heads** (the `[A][B]…[A]→[B]` copy circuit widely believed to underlie in-context learning), syntactic heads. Keeping $d_k=d/h$ is what makes the FLOPs of $h$ heads equal to one big head, so multi-head is "free" expressivity, not extra cost.

**Tie to HW2 §D (GNNs).** Attention over a set of tokens is a **permutation-invariant weighted aggregation** — reorder the keys/values and the output permutes the same way; nothing but the (learned) scores distinguishes positions, which is exactly why Piece 1's positional masking/encoding is needed. This is the *same* structural fact behind the message-passing aggregators in HW2 §D: a mean-aggregator is attention with uniform weights, and the Deep-Sets form $f(\sum_u g(h_u))$ is the general permutation-invariant primitive. Recognizing attention as "learned, softmax-weighted Deep-Sets aggregation" will make the GNN expressivity problems (P15–P16) click.

```python
import torch, torch.nn.functional as F
def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads, causal=True):
    B, T, d = x.shape
    dh = d // n_heads
    split = lambda W: (x @ W).view(B, T, n_heads, dh).transpose(1, 2)  # (B, nh, T, dh)
    q, k, v = split(Wq), split(Wk), split(Wv)
    scores = (q @ k.transpose(-2, -1)) / dh**0.5                        # (B, nh, T, T)
    if causal:
        scores = scores.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float('-inf'))
    out = (F.softmax(scores, dim=-1) @ v)                              # (B, nh, T, dh)
    out = out.transpose(1, 2).contiguous().view(B, T, d)               # concat heads
    return out @ Wo                                                     # mix heads back
```

**You've got this piece when you can** write the `MultiHead` + per-head formulas, explain why $d_k=d/h$ keeps compute constant, give a reason multiple specialized heads beat one big head (different relations in different subspaces; induction/previous-token heads), and state how attention's permutation-invariance connects it to Deep-Sets / mean-aggregation in HW2 §D.

# 📝 Review quiz

1. **(Concept)** Explain how "each token attends to itself and all earlier tokens, averaged" is expressed as a single matrix multiply $\hat X = WX$. What structural property must $W$ have to (a) be causal and (b) be a proper weighted average?
2. **(Concept / derivation)** Show that masking future scores with $-\infty$ *before* softmax reproduces a lower-triangular row-stochastic weight matrix. Why must the mask be applied before softmax, not after?
3. **(Concept)** Define $Q$, $K$, $V$ and write the scaled dot-product attention formula. Why is it beneficial to separate the "routing" ($Q,K$) from the "payload" ($V$) rather than using $X$ directly for all three?
4. **(Derivation)** Assuming $q,k$ have i.i.d. zero-mean unit-variance entries, derive $\operatorname{Var}(q^\top k)$. Use this to justify the $\sqrt{d_k}$ denominator, and explain why $\sqrt{d_k}$ rather than $d_k$.
5. **(Derivation / concept)** Write the softmax Jacobian $\partial p_i/\partial s_j$. Use it to explain why a *saturated* (near one-hot) attention row produces vanishing gradients, and how the $\sqrt{d_k}$ scaling prevents this.
6. **(Concept)** Write the two residual sub-layer updates of a pre-LN transformer block. Explain the "residual stream as a shared read/write bus" picture and two things it buys you (gradient flow, logit lens). Contrast pre-LN with post-LN.
7. **(Concept)** Why use $h$ heads of dimension $d/h$ instead of one head of dimension $d$? Address both expressivity (subspaces / interpretable heads) and compute (why the split is roughly "free").
8. **(Code reading / synthesis)** In the Piece 5 code, identify where (a) the causal mask, (b) the $\sqrt{d_k}$ scaling, and (c) the head concatenation happen. Then explain how attention's permutation-invariance links it to the Deep-Sets / mean-aggregator picture in HW2 §D.

> [!note]- 🔑 Answer key (click to reveal)
> **1.** Row $i$ of the output is $\hat x_i=\sum_j W_{ij}x_j$; choosing $W_{ij}=1/i$ for $j\le i$ and $0$ for $j>i$ makes $\hat x_i$ the mean of tokens $1..i$. Stacking all rows gives $\hat X=WX$ with $W$ a single matrix. For **(a) causality** $W$ must be **lower-triangular** (zero strict-upper-triangle, so no future token contributes); for **(b) a proper weighted average** each non-zero row must be **non-negative and sum to 1** (row-stochastic). Uniform running mean is the special case $W_{ij}=1/i$; attention replaces those with learned, content-dependent weights.
>
> **2.** Set $S_{ij}=-\infty$ for $j>i$ (future) and $0$ (or the real scores) for $j\le i$. Softmax of a row is $A_{ij}=e^{S_{ij}}/\sum_{j'}e^{S_{ij'}}$; since $e^{-\infty}=0$, every future entry becomes exactly $0$, and the surviving past entries are automatically non-negative and normalized to sum to 1 — i.e. a lower-triangular row-stochastic matrix. It must be applied **before** softmax because softmax is what normalizes: zeroing entries *after* softmax would break the row-sum-to-1 property (the row would sum to less than 1), whereas $-\infty$-before-softmax removes them from the normalization denominator cleanly.
>
> **3.** $Q=XW_Q$ ("what am I looking for"), $K=XW_K$ ("what do I contain, for matching"), $V=XW_V$ ("what I broadcast if matched"). $\operatorname{Attn}(Q,K,V)=\operatorname{softmax}\!\big(\tfrac{QK^\top}{\sqrt{d_k}}+M\big)V$. Separating routing from payload gives two independent degrees of freedom: $QK^\top$ decides *which* positions to pull from, $V$ decides *what content* is pulled. Using $X$ for all three would force "tokens that match" to also mean "copy this exact vector," collapsing routing and payload into one and sharply reducing expressivity.
>
> **4.** $q^\top k=\sum_{m=1}^{d_k}q_m k_m$. Each product has mean $0$ and (by independence) variance $\operatorname{Var}(q_m k_m)=\mathbb E[q_m^2]\mathbb E[k_m^2]=1$; summing $d_k$ independent terms gives $\operatorname{Var}(q^\top k)=d_k$, so std $=\sqrt{d_k}$. Dividing the scores by $\sqrt{d_k}$ renormalizes their standard deviation to $1$, keeping logits $\mathcal O(1)$ for any head size. You use $\sqrt{d_k}$ (the **std**) not $d_k$ (the **variance**) because you want unit std; dividing by $d_k$ would shrink logits toward $0$, flattening softmax toward uniform and preventing sharp attention.
>
> **5.** For $p=\operatorname{softmax}(s)$, $\dfrac{\partial p_i}{\partial s_j}=p_i(\delta_{ij}-p_j)$. If a row saturates so $p\approx e_k$ (one entry $\approx1$, rest $\approx0$), then every term $p_i(\delta_{ij}-p_j)\approx0$ (either $p_i\approx0$, or $p_i\approx1$ but $\delta_{ij}-p_j\approx0$). A ~zero Jacobian means gradients cannot propagate back through the attention weights, so the routing stops learning (same pathology as a saturated sigmoid). Large logits (std $\sqrt{d_k}$) drive saturation; the $\sqrt{d_k}$ scaling keeps logits $\mathcal O(1)$, so softmax stays in its non-saturated regime with a healthy Jacobian and gradients flow.
>
> **6.** $x\leftarrow x+\operatorname{Attn}(\operatorname{LN}(x))$ then $x\leftarrow x+\operatorname{MLP}(\operatorname{LN}(x))$. Each sub-layer *adds* into $x$, so the hidden state is a running sum that every layer **reads**, updates by a small amount, and **writes back** — a shared "residual stream" bus. Two payoffs: (i) the identity path has derivative $1$, so gradients flow through many layers un-attenuated (trainable depth); (ii) the **logit lens** — unembedding an intermediate $x^{(\ell)}$ gives progressively sharper predictions, showing computation as iterative refinement. **Pre-LN** normalizes the sub-layer input and leaves the residual path a clean identity (stable at depth, modern default); **post-LN** ($\operatorname{LN}(x+\text{sublayer}(x))$) puts LN on the residual path and is much harder to train deep without warm-up.
>
> **7.** One head computes a single relevance pattern (one weighted average) per query, but language needs many relations (agreement, coreference, syntax, copying) living in different subspaces. $h$ heads of size $d/h$ each learn a low-rank query/key geometry tuned to one relation type, then concat + $W_O$ recombine them — measured heads specialize (previous-token heads, induction heads implementing the $[A][B]\dots[A]\to[B]$ copy circuit behind in-context learning). Compute is roughly **free** because with $d_k=d_v=d/h$ the total FLOPs of $h$ small heads equal one full $d$-dim head; you gain parallel specialization at the same cost.
>
> **8.** (a) causal mask: `scores.masked_fill(torch.triu(...,1), -inf)`; (b) $\sqrt{d_k}$ scaling: `/ dh**0.5`; (c) head concat: `out.transpose(1,2).contiguous().view(B, T, d)` (then `@ Wo` mixes heads). **Link to §D:** attention is a permutation-invariant weighted aggregation — permuting the keys/values permutes the output identically and only the learned scores break symmetry (which is why positional info must be injected separately). That is exactly the message-passing/Deep-Sets structure of HW2 §D: a mean-aggregator is attention with uniform weights, and $\text{generic}_{f,g}(\{h_u\})=f(\sum_u g(h_u))$ is the general permutation-invariant primitive. Seeing attention as "learned softmax-weighted Deep-Sets aggregation" is the bridge to the GNN aggregator/expressivity problems.
