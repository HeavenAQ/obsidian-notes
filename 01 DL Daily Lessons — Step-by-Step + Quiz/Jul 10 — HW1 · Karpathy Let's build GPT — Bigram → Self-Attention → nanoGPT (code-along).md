---
base: "[[01.2 DL Daily Lessons — Step-by-Step + Quiz.base]]"
Quiz score: ""
"HW #": 1
Studied: false
Quiz taken: false
Day type: Lesson-Day
Date: 2026-07-10
Piece count: 5
---
> ⚠️ **Schedule note:** on plan — 🎥 **Karpathy, "Let's build GPT: from scratch, in code, spelled out"** (~2h). This is a **code-along**, not a watch: type every stage yourself — bigram LM → the averaging trick → one self-attention head → multi-head + feed-forward → residual/LayerNorm blocks → full nanoGPT on Shakespeare. It turns yesterday's **TiP Module 2** theory into working code, and its manual attention pass is the *same* matrix-calculus / outer-product skill as **HW1 Problem 8 & 10**. Deep-dive companion: [Transformers in Practice](https://app.notion.com/p/391e445b30a4817dbfb5ef315a1f06ff) **§2 (attention & causal masking)** — re-read alongside the code.


> 🗒️ **Quiz backlog:** every quiz Jul 2–9 is still untaken (**Quiz taken** unchecked, no score). Nothing to patch up yet — the re-explain loop only fires once a *low* score is recorded. Grade **at least yesterday's (Jul 9, attention theory)** before coding today (~10 min); today's code will cement exactly what that quiz tests.

# 🎯 Today's goal

Build a working GPT from an empty file so the transformer stops being a diagram and becomes ~200 lines you understand line-by-line. You'll start from a **bigram** baseline (a lookup table), discover self-attention as a **data-dependent weighted average of the past**, then stack multi-head attention + MLP into residual blocks until you have **nanoGPT** generating Shakespeare. By the end you should be able to write a self-attention head from memory, explain every shape and why the causal mask and $1/\sqrt{d_k}$ scale are there, and connect the manual backward math to **HW1 Problems 8 & 10**. This is the implementation half of yesterday's theory.

# 🧩 Pieces

## Piece 1 — Char-level data & the bigram baseline (Karpathy 0:00–0:42, ~30 min)

*Source: video intro through "training the bigram model." This is the scaffold everything else plugs into.*

Start with the tiny Shakespeare corpus and a **character-level** tokenizer: build a sorted vocabulary of the unique characters, then two lookup dicts `stoi`/`itos` so `encode` maps a string to a list of ints and `decode` inverts it. Training data is one long tensor of token ids. You sample training examples with `get_batch`: pick `batch_size` random start positions, and for each take a `block_size`-long chunk as input `x` and the same chunk shifted by one as target `y`. A single chunk secretly contains `block_size` examples (predict char 2 from 1, char 3 from 1–2, …) — that's why a transformer trains on all context lengths at once.

The **bigram** model ignores all context but the current token: a single embedding table of shape $(V, V)$ ($V$ = vocab size) where row $t$ *is* the logit vector for "what comes after token $t$." So the forward pass is just a lookup, and the loss is cross-entropy against the next token:

$$
L = -\frac{1}{N}\sum_{i=1}^{N}\log \hat p_{i,\,y_i},\qquad \hat p = \mathrm{softmax}(\text{logits})
$$

At initialization the weights are random, so every token is roughly equiprobable and the expected loss is $\approx \ln V$. For Shakespeare's $V=65$ that's $\ln 65 \approx 4.17$ — a number worth memorizing as your "is the model even training?" sanity check.

```python
import torch, torch.nn as nn
from torch.nn import functional as F

class BigramLM(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.tok = nn.Embedding(V, V)      # row t = logits for next token
    def forward(self, idx, targets=None):
        logits = self.tok(idx)             # (B, T, V)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, n):
        for _ in range(n):
            logits, _ = self(idx)                       # (B, T, V)
            probs = F.softmax(logits[:, -1, :], dim=-1) # last step only
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx
```

Note `cross_entropy` wants `(N, C)` logits and `(N,)` targets, so you flatten the batch and time dims together — a reshape you'll repeat all day.

**You've got this piece when you can** explain why one `block_size` chunk is really `block_size` training examples, state that the bigram table has shape $(V,V)$ and *is* the logits, write the cross-entropy loss, and compute the expected init loss $\ln 65 \approx 4.17$.

## Piece 2 — The math trick: attention as a weighted average of the past (Karpathy 0:42–1:11, ~30 min)

*Source: the famous "mathematical trick in self-attention" section (the four versions).*

The bigram can't use context. The minimal way to let a token see its past is to **average** all previous token vectors: for token $t$,

$$
x^{\text{bow}}_t = \frac{1}{t+1}\sum_{\tau=0}^{t} x_\tau
$$

("bow" = bag of words). Doing this with a Python loop is $O(T^2)$ and slow; the key realization is that **this average is a matrix multiply**. Let $A\in\mathbb{R}^{T\times T}$ be lower-triangular with each row normalized to sum to 1 — i.e. $A_{t\tau} = \tfrac{1}{t+1}$ for $\tau \le t$ and $0$ for $\tau > t$. Then $X^{\text{bow}} = A X$ computes every prefix average in one batched op:

$$
A = \begin{pmatrix} 1 & 0 & 0\\ \tfrac12 & \tfrac12 & 0\\ \tfrac13 & \tfrac13 & \tfrac13 \end{pmatrix}, \qquad X^{\text{bow}} = A X
$$

The final reframing is the bridge to attention: instead of hard-coding uniform weights, start from an **affinity matrix** of zeros, mask the future to $-\infty$, and softmax each row. Softmax of equal (zero) logits gives the uniform average — but if the affinities are *data-dependent* instead of zero, each token gets to choose *how much* to weight each past token. That is exactly self-attention.

$$
\text{wei} = \mathrm{softmax}\big(\text{mask}(\,0\,)\big) = A, \qquad \text{mask}(S)_{t\tau} = \begin{cases} S_{t\tau} & \tau \le t\\ -\infty & \tau > t\end{cases}
$$

```python
T = 8
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros(T, T)
wei = wei.masked_fill(tril == 0, float('-inf'))  # future -> -inf
wei = F.softmax(wei, dim=-1)                      # each row = uniform average
# xbow = wei @ x   # (T,T) @ (B,T,C) broadcasts to (B,T,C)
```

**You've got this piece when you can** write the prefix-average as $X^{\text{bow}}=AX$ with $A$ a row-normalized lower-triangular matrix, and explain how replacing the zero affinities with data-dependent scores turns fixed averaging into attention.

## Piece 3 — A single self-attention head (Karpathy 1:11–1:36, ~35 min)

*Source: "self-attention: a single head." Deep dive: *[*Transformers in Practice*](https://app.notion.com/p/391e445b30a4817dbfb5ef315a1f06ff)* §2.*

Now make the affinities data-dependent. Each token emits a **query** ($q$ = "what am I looking for"), a **key** ($k$ = "what do I contain"), and a **value** ($v$ = "what I'll share if attended to"), via learned linear maps $Q=XW_Q,\ K=XW_K,\ V=XW_V$ with $X\in\mathbb{R}^{T\times d}$ and head size $d_k$. The affinity between token $i$ and token $j$ is $q_i^\top k_j$, and one head is:

$$
\mathrm{head}(X) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V, \qquad M_{ij} = \begin{cases} 0 & j \le i\\ -\infty & j > i \end{cases}
$$

**Why scale by **$\sqrt{d_k}$. With unit-variance i.i.d. entries, $q^\top k=\sum_{c=1}^{d_k} q_c k_c$ is a sum of $d_k$ unit-variance terms, so $\mathrm{Var}(q^\top k)=d_k$ and its std is $\sqrt{d_k}$. Unscaled, the logits fed to softmax grow like $\sqrt{d_k}$; large logits make softmax **peaky** (one weight $\to 1$), and at initialization a peaky attention means each token essentially copies a single neighbor and gradients through the other positions vanish. Dividing by $\sqrt{d_k}$ keeps logits $\mathcal{O}(1)$ so attention starts *diffuse* and can learn. The causal mask $M$ (the `tril` from Piece 2) is what makes this a **decoder**: a token may attend to itself and the past but never the future, so the model stays autoregressive.

```python
class Head(nn.Module):
    def __init__(self, d, dk, block):
        super().__init__()
        self.key   = nn.Linear(d, dk, bias=False)
        self.query = nn.Linear(d, dk, bias=False)
        self.value = nn.Linear(d, dk, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block, block)))
    def forward(self, x):                       # x: (B, T, d)
        B, T, d = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)   # (B, T, dk)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5      # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei @ v                          # (B, T, dk)
```

**You've got this piece when you can** write the head formula from memory, give the shapes of $Q,K,V$ and $QK^\top$, derive $\mathrm{Var}(q^\top k)=d_k$ and why that motivates the $1/\sqrt{d_k}$ scale, and explain what `register_buffer('tril', …)` accomplishes vs. a parameter.

## Piece 4 — Multi-head attention + the feed-forward (Karpathy 1:36–1:52, ~35 min)

*Source: "multi-head attention" and "feedforward."*

One head learns exactly one kind of relation. Run $H$ heads in parallel, each of size $d_k=d/H$, concatenate their outputs, and mix with an output projection $W_O$:

$$
\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_H)\,W_O
$$

Total compute matches one full-width head, but you get $H$ independent "relevance channels" (syntax, agreement, copying, …). Attention, however, is purely a *communication* step — it moves information between positions but does little per-token computation. So each block follows attention with a **position-wise feed-forward MLP** that lets every token *think* on what it just gathered:

$$
\mathrm{FFN}(h) = W_2\,\phi(W_1 h + b_1) + b_2, \qquad \phi = \mathrm{ReLU},\ \ \dim(W_1 h) = 4d
$$

The $4\times$ inner width is the standard expansion. The mental model to lock in: **attention = communication across tokens, MLP = computation within each token.** LayerNorm and the MLP are strictly position-wise; only attention mixes across positions.

```python
class MultiHead(nn.Module):
    def __init__(self, H, d, block):
        super().__init__()
        self.heads = nn.ModuleList([Head(d, d // H, block) for _ in range(H)])
        self.proj  = nn.Linear(d, d)                       # W_O
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, d)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 4*d), nn.ReLU(), nn.Linear(4*d, d))
    def forward(self, x):
        return self.net(x)
```

**You've got this piece when you can** write the multi-head formula with $W_O$, explain why $H$ heads of size $d/H$ beat one big head at equal cost, state the $4d$ inner width and ReLU of the FFN, and articulate "communication (attention) then computation (MLP)."

## Piece 5 — Residuals, LayerNorm & the full nanoGPT (Karpathy 1:52–end, ~30 min)

*Source: "residual connections," "layernorm," "scaling up," and token+position embeddings.*

Stacking raw blocks won't train deep — gradients degrade. Two fixes make it work. **Residual connections** wrap each sublayer as an *additive update* to a running "residual stream," and **pre-LayerNorm** normalizes each sublayer's input while leaving the skip path clean:

$$
x \leftarrow x + \mathrm{MHA}(\mathrm{LN}(x)), \qquad x \leftarrow x + \mathrm{FFN}(\mathrm{LN}(x))
$$

LayerNorm standardizes each token vector across its features: $\mathrm{LN}(x) = \gamma\odot\dfrac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$ with $\mu,\sigma^2$ computed over the feature dimension (per token, per position — not across the batch, so it behaves identically at train and inference). Two more ingredients complete the model: the input is a **token embedding plus a position embedding** (attention is permutation-invariant, so position must be injected — Piece 3 of yesterday), and the output is projected back to vocabulary logits by an unembedding $W_U$ (the `lm_head`):

$$
x_0 = \mathrm{TokEmb}(\text{idx}) + \mathrm{PosEmb}(0{:}T), \qquad \text{logits} = W_U\,\mathrm{LN}_f(x_L)
$$

```python
class Block(nn.Module):
    def __init__(self, d, H, block):
        super().__init__()
        self.sa  = MultiHead(H, d, block)
        self.ff  = FeedForward(d)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # communication (+ residual)
        x = x + self.ff(self.ln2(x))   # computation   (+ residual)
        return x

class GPT(nn.Module):
    def __init__(self, V, d, H, n_layer, block):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(block, d)
        self.blocks = nn.Sequential(*[Block(d, H, block) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)     # W_U
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        x = self.lnf(self.blocks(x))
        return self.head(x)             # (B, T, V) logits
```

Karpathy's final settings — $d=384,\ H=6,\ n\_layer=6,\ block=256$, dropout $0.2$ — reach ~1.48 val loss and produce fake-Shakespeare. Dropout (added on attention weights, after the projections, and in the FFN) is the last regularizer; it's why the scaled-up model doesn't just memorize.

**You've got this piece when you can** write the two pre-LN residual update lines, write the LayerNorm formula and say which axis it normalizes (features, per token), explain why token embeddings need position embeddings added, and trace the full forward: idx → tok+pos → blocks → final LN → `lm_head` → logits.

# 📝 Review quiz

Answer all 8 without notes. Grade with the key at the bottom, then fill in **Quiz score** and check **Quiz taken**.

1. **(Conceptual / code)** In the bigram model, what is the shape of the embedding table and why does a plain lookup already produce logits? Given a vocab of $V=65$, what cross-entropy loss do you expect at initialization, and why?
2. **(Derivation)** Write the prefix-average $x^{\text{bow}}_t$ and show it equals $X^{\text{bow}}=AX$ for a specific matrix $A$. Describe $A$ precisely (shape, which entries, normalization), and explain how the softmax-of-masked-zeros construction reproduces the same $A$.
3. **(Conceptual)** In one self-attention head, what do the query, key, and value represent, and what are the shapes of $Q,K,V$ and of $QK^\top$? Which quantity plays the role of the "affinities" from Piece 2?
4. **(Derivation)** Show that for unit-variance i.i.d. entries $\mathrm{Var}(q^\top k)=d_k$. Use it to justify the $1/\sqrt{d_k}$ factor, and describe what goes wrong at initialization if you omit it.
5. **(Conceptual / code)** Why does a decoder LM need the causal mask, and why is it applied as $-\infty$ *before* softmax rather than by zeroing weights after? In the `Head` code, why is `tril` created with `register_buffer` instead of `nn.Parameter`?
6. **(Conceptual)** Write the multi-head formula with $W_O$ and explain why $H$ heads of size $d/H$ are used instead of one head of size $d$. Then state the role of the feed-forward layer and the slogan that separates it from attention.
7. **(Conceptual)** Write the two pre-LayerNorm residual update equations for a block. Why do residual connections plus pre-LN let you train many stacked blocks, and what does LayerNorm normalize over (which axis, and does it depend on the batch)?
8. **(Conceptual / code)** Trace the full GPT forward pass from `idx` to logits, naming every transformation. Why must a position embedding be added to the token embedding, and what does the final `lm_head` ($W_U$) compute?

> [!note]+ 🔑 Answer key — open only after attempting all 8
> **1.** The table has shape $(V,V)$: row $t$ is directly the length-$V$ logit vector for the token following token $t$, so `embedding(idx)` *is* the logits — no matmul needed. At init the weights are random and near-symmetric, so every next token is ~equiprobable ($\hat p\approx 1/V$) and $L=-\ln \hat p \approx \ln V$. For $V=65$, $\ln 65 \approx 4.17$.
> 
> **2.** $x^{\text{bow}}_t=\tfrac{1}{t+1}\sum_{\tau=0}^{t}x_\tau$. This is $X^{\text{bow}}=AX$ with $A\in\mathbb{R}^{T\times T}$ lower-triangular, $A_{t\tau}=\tfrac{1}{t+1}$ for $\tau\le t$ and $0$ otherwise (each row sums to 1). Softmax-of-masked-zeros: take a $T\times T$ matrix of zeros, set entries with $\tau>t$ to $-\infty$, softmax each row → the $t$-th row has $t+1$ equal entries $e^0/(t+1)=\tfrac{1}{t+1}$ and zeros beyond, i.e. exactly $A$. Replacing the zeros with data-dependent scores generalizes uniform averaging to attention.
> 
> **3.** Query = "what this token is looking for," key = "what this token offers," value = "what it passes on if attended to," all learned linear maps of $X$. Shapes: $Q,K,V\in\mathbb{R}^{T\times d_k}$; $QK^\top\in\mathbb{R}^{T\times T}$. $QK^\top/\sqrt{d_k}$ (after masking) plays the role of the affinities — the now-data-dependent version of Piece 2's $A$ (pre-softmax).
> 
> **4.** $q^\top k=\sum_{c=1}^{d_k} q_c k_c$. Each product has mean 0 and variance $\mathrm{Var}(q_c)\mathrm{Var}(k_c)=1$ (independent, mean-0), and the $d_k$ terms are independent, so $\mathrm{Var}(q^\top k)=d_k$, std $\sqrt{d_k}$. Unscaled logits therefore scale like $\sqrt{d_k}$, pushing softmax into a peaky regime where one weight $\to1$ and the rest $\to0$; its gradient there is ~0, so each token just copies one neighbor and learning stalls. Dividing by $\sqrt{d_k}$ restores $\mathcal{O}(1)$ logits and diffuse initial attention.
> 
> **5.** A decoder predicts the next token, so a position must not see the future or it would trivially cheat; the causal mask enforces that a token attends only to itself and earlier positions. Applied as $-\infty$ *before* softmax, $e^{-\infty}=0$ gives future positions exactly zero weight while the row still normalizes over the allowed positions in one fixed-shape op; zeroing *after* softmax would leave rows un-normalized. `register_buffer` stores `tril` as fixed (non-learned) state that moves with `.to(device)` and is saved in the state-dict but gets **no gradient** — it's a constant mask, not a `Parameter`.
> 
> **6.** $\mathrm{MHA}(X)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_H)W_O$. $H$ heads of size $d/H$ give $H$ independent relation subspaces (syntax, agreement, copying) at the *same* total cost as one width-$d$ head, and $W_O$ recombines them. The feed-forward MLP (width $4d$, ReLU) lets each token compute on what attention gathered; slogan: **attention = communication across positions, MLP = computation within a position.**
> 
> **7.** $x\leftarrow x+\mathrm{MHA}(\mathrm{LN}(x))$ then $x\leftarrow x+\mathrm{FFN}(\mathrm{LN}(x))$. Each sublayer is an additive update to a running residual stream, so gradients flow through the clean skip path; pre-LN normalizes each sublayer's input without touching the skip, avoiding the vanishing/exploding gradients of deep post-LN stacks. LayerNorm normalizes over the **feature axis, per token/position** ($\mu,\sigma^2$ across features), independent of the batch — so train and inference behave identically.
> 
> **8.** `idx` → token embedding $\mathrm{TokEmb}(\text{idx})$ **plus** position embedding $\mathrm{PosEmb}(0{:}T)$` → $`n\_layer$ residual blocks (each: pre-LN → MHA → +residual → pre-LN → FFN → +residual) → final LayerNorm → `lm_head` linear $W_U$ → logits $(B,T,V)$. Position embeddings are required because attention is permutation-invariant and would otherwise be blind to order; `lm_head` ($W_U$) maps each final $d$-dim token vector to the $V$ next-token logits.
> 
> **Scoring:** 1 pt each (half credit for the right idea with a wrong detail). ≥ 6/8 → move to Jul 11 (TiP M3 + deployment); below → tomorrow opens with a targeted patch-up on the missed pieces.