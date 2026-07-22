---
base: "[[01.2 DL Daily Lessons — Step-by-Step + Quiz.base]]"
Quiz score: ""
"HW #": 2
Studied: false
Quiz taken: false
Day type: Lesson-Day
Date: 2026-07-21
Piece count: 5
---
> 🗒️ **Quiz backlog nudge (no patch-up today):** the re-teach loop only fires on a *recorded* low score, and yesterday's **Jul 20 (Attention Theory)** file is still `Quiz taken: false` — as are Jul 17 (Eigenvalues I), Jul 18 (Positive Definiteness / Hessian), and Jul 19 (Least Squares / Projection). Nothing to patch up. But today is the **last lesson day of the HW2 window** (Jul 22–23 are homework days), so this is your best chance to clear the backlog: grade the **Jul 20 attention set** first (~10 min) — it's the direct prerequisite for Piece 5 below, where we contrast attention against the RNN you'll build today.

# 🎯 Today's goal

This is **HW2 Day 5 — MIT 6.S191 Lecture 2, *Deep Sequence Modeling*** (attention intro): the recurrent-network story that attention grew out of. By the end you should be able to write the RNN recurrence and its unrolled computation graph from memory, derive *why* backpropagation-through-time makes gradients vanish or explode (a Jacobian-product / spectral-radius argument — exactly the eigenvalue/matrix-powers material from Jul 17), explain how LSTM/GRU gating installs a near-identity "highway" for the gradient, and articulate the two structural reasons (parallelism, constant path length) that made **attention** replace recurrence. This closes the loop with Jul 20's scaled-dot-product attention and feeds straight into starting **HW2** — the sequence-modeling framing also mirrors the **permutation-invariance / aggregation** theme in HW2 §D (GNNs).

# 🧩 Pieces

## Piece 1 — The sequence-modeling problem: order, variable length & weight sharing (~25 min)

*Source: MIT 6.S191 L2 → "Sequence modeling" intro (why a plain MLP fails on sequences). Deep-dive: [[Transformers in Practice (DeepLearning.AI, Sharon Zhou) — Deep Dive]] §Module 1 → "The autoregressive loop" (next-token modeling is a sequence problem) for the same $p(x_t\mid x_{<t})$ factorization from a different angle.*

A **sequence** is an ordered list of tokens $x_1,x_2,\dots,x_T$ where (a) the length $T$ varies from example to example, (b) **order carries meaning** ("dog bites man" $\ne$ "man bites dog"), and (c) the same feature can appear at any position. A plain feed-forward net breaks on all three: it wants a fixed-size input, it has no notion of position beyond "which input slot," and it would need a *separate* weight for the word "dog" in slot 3 vs. slot 7. The design goals that fix this:

- **Handle variable length** with one model.
- **Share parameters across time** — the rule for processing token $t$ should not depend on $t$, so a pattern learned at position 3 transfers to position 50.
- **Maintain a memory / state** that summarizes the past $x_{1:t}$ so the model can use context.

The clean probabilistic framing (shared with autoregressive LMs): factor the joint by the chain rule,

$$
p(x_1,\dots,x_T)=\prod_{t=1}^{T} p(x_t\mid x_1,\dots,x_{t-1}),
$$

and learn each conditional $p(x_t\mid x_{<t})$ with a model that compresses the growing history $x_{<t}$ into a **fixed-size state** $h_{t-1}$. That compression-into-a-state is exactly what a recurrent network does, and (Piece 5) exactly what attention *refuses* to do — attention keeps the whole past around instead.

```python
# The three headaches a plain MLP has with sequences, in one snippet
seqs = [["the","cat","sat"], ["it","rained"], ["a","long","noisy","string"]]
lengths = [len(s) for s in seqs]     # variable length: 3, 2, 4  -> no fixed input size
# order matters: same tokens, different meaning
a = ["dog","bites","man"]; b = ["man","bites","dog"]
assert a != b and sorted(a) == sorted(b)   # bag-of-words would call these identical
```

**You've got this piece when you can** state the three properties of sequence data (variable length, order-dependence, position-agnostic features), explain why parameter *sharing across time* and a *fixed-size state* are the two ideas that make one model handle any length, and write the chain-rule factorization $p(x_{1:T})=\prod_t p(x_t\mid x_{<t})$.

## Piece 2 — The RNN cell: recurrence, unrolling & the forward pass (~30 min)

*Source: MIT 6.S191 L2 → "Recurrent neural networks" (the state-update equation and the unrolled graph). Deep-dive: none in the vault for vanilla RNNs specifically; the [[Mamba Architecture — Selective State Spaces, Math & PyTorch Deep Dive]] intro frames RNNs as the recurrent baseline that state-space models generalize — skim §1 if you want the modern lineage.*

A recurrent network maintains a hidden state $h_t\in\mathbb{R}^{n}$ and updates it with the **same** weights at every step:

$$
h_t=\tanh\!\big(W_{hh}\,h_{t-1}+W_{xh}\,x_t+b_h\big),\qquad
\hat y_t=W_{hy}\,h_t+b_y .
$$

Three parameter matrices, reused for all $t$: $W_{xh}$ (input→state), $W_{hh}$ (state→state, the *recurrent* weight), $W_{hy}$ (state→output). The state $h_t$ is the model's compressed memory of $x_{1:t}$; $h_0$ is usually zeros. "Unrolling" means drawing the recurrence as a deep feed-forward graph in time — one layer per timestep, **all sharing the same weights**:

$$
h_0\xrightarrow{\;x_1\;}h_1\xrightarrow{\;x_2\;}h_2\xrightarrow{\;x_3\;}\cdots\xrightarrow{\;x_T\;}h_T .
$$

This unrolled view is the key mental model: an RNN on a length-$T$ sequence *is* a $T$-layer network whose layers are tied. That is simultaneously its strength (arbitrary depth/length with $O(1)$ parameters) and its curse (Piece 3: a product of $T$ Jacobians).

```python
import torch, torch.nn as nn

class VanillaRNNCell(nn.Module):
    def __init__(self, d_in, d_hid):
        super().__init__()
        self.Wxh = nn.Linear(d_in, d_hid, bias=False)
        self.Whh = nn.Linear(d_hid, d_hid, bias=True)   # recurrent weight (+bias)
    def forward(self, x_t, h_prev):
        return torch.tanh(self.Wxh(x_t) + self.Whh(h_prev))   # h_t

def run_rnn(cell, xs, h0):
    h = h0
    hs = []
    for x_t in xs:            # xs: list of (B, d_in) — the explicit unroll in time
        h = cell(x_t, h)     # SAME cell (shared weights) every step
        hs.append(h)
    return torch.stack(hs)   # (T, B, d_hid)
```

**You've got this piece when you can** write the RNN state-update and output equations, name what each of $W_{xh},W_{hh},W_{hy}$ does, explain "unrolling" as turning a length-$T$ sequence into a $T$-layer weight-tied network, and say why weight sharing gives length-independence with a constant parameter count.

## Piece 3 — WHY gradients vanish/explode: BPTT as a product of Jacobians (~35 min)

*Source: MIT 6.S191 L2 → "Backpropagation through time" and "the problem of long-term dependencies." Deep-dive: reuse [[Jul 17 — HW2 · Eigenvalues & Eigenvectors I — Eigendecomposition, Diagonalization & Matrix Powers]] — the identity $A^k=Q\Lambda^kQ^{-1}$ (matrix powers governed by eigenvalues) is literally the mechanism here; and [[Gradient Descend]] for the general chain-rule setup.*

This is the piece your priorities care about most. Training an RNN means backpropagating through the unrolled graph — **backpropagation through time (BPTT)**. Because the loss $L=\sum_t L_t$ depends on early states through *every* later state, the gradient of a late loss $L_T$ w.r.t. an early state $h_k$ is a **chain of Jacobians**:

$$
\frac{\partial L_T}{\partial h_k}=\frac{\partial L_T}{\partial h_T}\;\prod_{t=k+1}^{T}\frac{\partial h_t}{\partial h_{t-1}},
\qquad
\frac{\partial h_t}{\partial h_{t-1}}=\operatorname{diag}\!\big(\tanh'(\cdot)\big)\,W_{hh}.
$$

So propagating a gradient back $T-k$ steps multiplies $T-k$ copies of the matrix $J_t=\operatorname{diag}(\tanh')\,W_{hh}$. The behavior of a long matrix product is governed by its **spectral radius** — exactly the eigenvalue/matrix-powers story from Jul 17. Ignoring the (bounded, $\le 1$) $\tanh'$ factor and writing $W_{hh}=Q\Lambda Q^{-1}$:

$$
\prod_{t}\frac{\partial h_t}{\partial h_{t-1}}\ \approx\ W_{hh}^{\,T-k}=Q\,\Lambda^{\,T-k}\,Q^{-1}.
$$

Each eigenvalue $\lambda_i$ is raised to the power $T-k$:

- if $|\lambda_i|<1$ → $\lambda_i^{T-k}\to 0$ **exponentially**: the gradient **vanishes**, so the network cannot learn dependencies more than a handful of steps back;
- if $|\lambda_i|>1$ → $\lambda_i^{T-k}\to\infty$: the gradient **explodes**, giving NaNs / wildly unstable updates.

Only the razor's edge $|\lambda|\approx 1$ preserves signal, and $\tanh'\le 1$ biases the product toward *shrinking*, so vanishing is the dominant real-world failure. **Practical consequences:** explosion is patched cheaply with **gradient clipping** (rescale the gradient if its norm exceeds a threshold); vanishing is *not* fixable by clipping — it needs an architectural change, which is Piece 4.

```python
import torch
# Empirical demo: ||W_hh^k|| governed by the top eigenvalue (spectral radius)
def spectral_radius(W):
    return torch.linalg.eigvals(W).abs().max().item()

for scale in (0.5, 1.0, 1.5):          # sub-unit, critical, super-unit
    W = scale * torch.eye(8) + 0.01 * torch.randn(8, 8)
    rho = spectral_radius(W)
    norms = [torch.linalg.matrix_norm(torch.linalg.matrix_power(W, k), 2).item()
             for k in (1, 10, 50)]
    print(f"scale={scale}  rho={rho:.2f}  ||W^1,10,50||={[round(n,3) for n in norms]}")
# rho<1 -> norms shrink to 0 (vanish); rho>1 -> norms blow up (explode)
```

**You've got this piece when you can** write $\partial L_T/\partial h_k$ as a product of per-step Jacobians $\operatorname{diag}(\tanh')W_{hh}$, use $W_{hh}^{T-k}=Q\Lambda^{T-k}Q^{-1}$ to argue vanishing ($|\lambda|<1$) vs. exploding ($|\lambda|>1$) via the spectral radius, explain why vanishing (not exploding) is the harder, dominant problem, and say what gradient clipping does and does *not* fix.

## Piece 4 — LSTM/GRU gating: the constant-error carousel (~35 min)

*Source: MIT 6.S191 L2 → "Long Short-Term Memory (LSTM)" and gating. Deep-dive: [[Mamba Architecture — Selective State Spaces, Math & PyTorch Deep Dive]] §on gating/selectivity — modern SSMs are the intellectual descendants of LSTM gates (input-dependent state retention).*

The fix for vanishing gradients is to give the state an **additive, near-identity path through time** instead of the multiplicative $W_{hh}$ path. The LSTM adds a separate **cell state** $c_t$ (the "conveyor belt") whose update is *additive*, gated by learned sigmoids in $[0,1]$:

$$
\begin{aligned}
f_t&=\sigma(W_f[h_{t-1},x_t]+b_f) &&\text{forget gate}\\
i_t&=\sigma(W_i[h_{t-1},x_t]+b_i) &&\text{input gate}\\
o_t&=\sigma(W_o[h_{t-1},x_t]+b_o) &&\text{output gate}\\
\tilde c_t&=\tanh(W_c[h_{t-1},x_t]+b_c) &&\text{candidate}\\
c_t&=\underbrace{f_t\odot c_{t-1}}_{\text{keep}}+\underbrace{i_t\odot \tilde c_t}_{\text{write}},\qquad
h_t=o_t\odot\tanh(c_t).
\end{aligned}
$$

**Why this cures vanishing.** Differentiate the cell recurrence:

$$
\frac{\partial c_t}{\partial c_{t-1}}=\operatorname{diag}(f_t).
$$

When the forget gate $f_t\approx 1$, this Jacobian is $\approx I$, so backpropagating through the cell state multiplies by $\approx 1$ at each step — the gradient neither vanishes nor explodes across long spans. That near-identity recurrence is the **constant-error carousel**: information (and gradient) can ride the cell state for hundreds of steps. Contrast the vanilla RNN's $\operatorname{diag}(\tanh')W_{hh}$, whose repeated multiplication drove the Jul-17-style $\Lambda^{T-k}$ decay. Gating converts a *multiplicative* memory into a *gated-additive* one — and because the gates are **learned and data-dependent**, the network decides per-token what to keep, write, and read. **GRU** is the streamlined cousin: it merges cell and hidden state and uses just update ($z_t$) and reset ($r_t$) gates — fewer parameters, similar long-range behavior, $h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$ showing the same "keep vs. write" convex blend.

```python
import torch, torch.nn as nn

class LSTMCellFromScratch(nn.Module):
    def __init__(self, d_in, d_hid):
        super().__init__()
        self.gates = nn.Linear(d_in + d_hid, 4 * d_hid)   # f, i, o, c-candidate in one matmul
        self.d = d_hid
    def forward(self, x_t, state):
        h_prev, c_prev = state
        z = self.gates(torch.cat([h_prev, x_t], dim=-1))
        f, i, o, g = z.chunk(4, dim=-1)
        f, i, o = f.sigmoid(), i.sigmoid(), o.sigmoid()
        g = g.tanh()
        c_t = f * c_prev + i * g          # ADDITIVE update -> dc_t/dc_{t-1} = diag(f) ≈ I
        h_t = o * torch.tanh(c_t)
        return h_t, c_t
```

**You've got this piece when you can** write the LSTM gate equations, identify the additive cell update $c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$, show that $\partial c_t/\partial c_{t-1}=\operatorname{diag}(f_t)\approx I$ is *why* gradients survive long spans (the constant-error carousel), and contrast this gated-additive memory with the vanilla RNN's multiplicative $W_{hh}$ path (and note the GRU's update/reset simplification).

## Piece 5 — From recurrence to attention: why we stopped unrolling in time (~30 min)

*Source: MIT 6.S191 L2 → "The limitations of RNNs" and "Self-attention" (the lecture's pivot to attention). Deep-dive: your own [[Jul 20 — HW2 · Attention Theory — Causal Masking, Scaled Dot-Product, the Transformer Block & Multi-Head]] is the direct continuation; also [[Transformers in Practice (DeepLearning.AI, Sharon Zhou) — Deep Dive]] §Module 2.*

Even a well-gated LSTM has two structural bottlenecks that gating cannot remove:

1. **Sequential computation — no parallelism.** $h_t$ needs $h_{t-1}$, so the forward (and backward) pass is an inherently serial loop of length $T$. You cannot use a modern accelerator's parallelism across the time axis during training. Attention computes all positions **at once** as a single $QK^\top$ matmul (Jul 20, Piece 1) — that is the practical reason transformers train so much faster.
2. **Long path length — the $O(T)$ information bottleneck.** For token $T$ to use token $1$, the signal must pass through $T-1$ recurrent steps; even with the carousel, resolution degrades and everything is squeezed through a fixed-size state. In self-attention any two positions are connected by a **single** hop — the maximum path length is $O(1)$, not $O(T)$ — so long-range dependencies are as easy to learn as short-range ones.

The conceptual bridge from Piece 1: an RNN *compresses* the past $x_{<t}$ into a fixed vector $h_{t-1}$ and then throws the raw past away; **self-attention keeps the whole past and computes a fresh, content-weighted read of it at every step** — $\hat X=\operatorname{softmax}(QK^\top/\sqrt{d_k}+M)V$. The causal mask $M$ recreates the RNN's "no peeking at the future" property (Jul 20, Piece 1), but *without* a serial loop. The price attention pays is $O(T^2)$ compute/memory in sequence length (every pair of positions interacts) versus the RNN's $O(T)$ — which is exactly why efficient long-context methods (linear attention, and the state-space models in [[Mamba Architecture — Selective State Spaces, Math & PyTorch Deep Dive]]) try to recover RNN-like $O(T)$ scaling while keeping attention's trainability.

There is also a clean tie to **HW2 §D (GNNs)**: an RNN imposes a *chain* graph (each node talks only to its predecessor), whereas self-attention is a *fully-connected* graph with learned edge weights — a permutation-invariant weighted aggregation over the set of tokens. Recognizing "attention = message passing on a complete graph" and "RNN = message passing on a path graph" makes both the transformer and the GNN aggregator problems (P15–P16) one idea seen from two sides.

```python
# Same causal "no future" property, two cost profiles:
#   RNN  : serial loop, O(T) time-steps, path length O(T) between distant tokens
#   Attn : one matmul,  O(1) sequential depth, path length O(1),  but O(T^2) work
import torch, torch.nn.functional as F
T, d = 6, 16
X = torch.randn(T, d)
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)   # future = True
A = F.softmax(torch.zeros(T, T).masked_fill(mask, float('-inf')), dim=-1)
# every past position reachable in ONE hop (row i has nonzeros for all j<=i), no time loop
print((A > 0).sum(dim=1).tolist())   # [1,2,3,4,5,6] causal reach grows, computed in parallel
```

**You've got this piece when you can** name the two structural limits of RNNs that gating cannot fix (serial/non-parallel computation; $O(T)$ path length between distant tokens), explain how self-attention removes both (parallel $QK^\top$; $O(1)$ path length) at the cost of $O(T^2)$ compute, describe the "compress-and-discard the past (RNN) vs. keep-and-reweight the past (attention)" contrast, and connect RNN-as-chain-graph vs. attention-as-complete-graph to HW2 §D.

# 📝 Review quiz

1. **(Concept)** List the three properties of sequence data that a plain MLP handles poorly, and name the two design ideas (about parameters and state) that let a single recurrent model process sequences of any length.
2. **(Derivation / concept)** Write the vanilla RNN state-update and output equations. Explain what "unrolling in time" means and why an RNN on a length-$T$ input is equivalent to a $T$-layer network — with what special constraint on the layers?
3. **(Derivation)** Write $\partial L_T/\partial h_k$ as a product of per-step Jacobians and give the form of $\partial h_t/\partial h_{t-1}$ for a $\tanh$ RNN. Using $W_{hh}=Q\Lambda Q^{-1}$, explain precisely how the spectral radius decides whether gradients vanish or explode over $T-k$ steps.
4. **(Concept)** Why is *vanishing* (rather than exploding) the harder, more common problem in practice? Which cheap trick fixes exploding gradients, and why does that same trick fail to fix vanishing?
5. **(Derivation)** Write the LSTM cell-state update $c_t$ in terms of the forget/input gates and candidate. Compute $\partial c_t/\partial c_{t-1}$ and use it to explain the "constant-error carousel" — i.e. why gating lets gradients survive hundreds of steps where a vanilla RNN cannot.
6. **(Concept)** Contrast the vanilla RNN's *multiplicative* memory path with the LSTM's *gated-additive* one. What does the GRU merge/simplify relative to the LSTM, and how does $h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$ express "keep vs. write"?
7. **(Concept / synthesis)** Give the two structural limitations of RNNs that gating cannot remove, and state exactly how self-attention removes each. What is the compute/memory price attention pays in return, and how does that motivate state-space models like Mamba?
8. **(Code reading / synthesis)** In the Piece 5 attention snippet, which line enforces causality, and why does `(A > 0).sum(dim=1)` return `[1,2,3,4,5,6]`? Then explain the "RNN = path graph, attention = complete graph" framing and how it links today's lesson to the Deep-Sets/message-passing aggregators in HW2 §D.

> [!note]- 🔑 Answer key (click to reveal)
> **1.** Properties: (i) **variable length** across examples; (ii) **order matters** ("dog bites man" ≠ "man bites dog"); (iii) the same **feature can appear at any position** (position-agnostic features). The two design ideas: **share parameters across time** (the same update rule at every step, so a pattern learned at one position transfers to all), and **maintain a fixed-size state/memory** $h_t$ that compresses the history $x_{1:t}$ — together these decouple the model size from the sequence length, letting one model handle any $T$. (Framing: learn $p(x_t\mid x_{<t})$ with the history compressed into $h_{t-1}$.)
>
> **2.** $h_t=\tanh(W_{hh}h_{t-1}+W_{xh}x_t+b_h)$, $\hat y_t=W_{hy}h_t+b_y$. **Unrolling** = drawing the recurrence as a feed-forward graph with one layer per timestep, $h_0\to h_1\to\dots\to h_T$, each step consuming $x_t$. It is a $T$-layer network because there are $T$ successive nonlinear transformations from $h_0$ to $h_T$; the special constraint is that **all $T$ layers share the same weights** ($W_{hh},W_{xh}$ tied across time), which is what gives length-independence at constant parameter count.
>
> **3.** $\dfrac{\partial L_T}{\partial h_k}=\dfrac{\partial L_T}{\partial h_T}\prod_{t=k+1}^{T}\dfrac{\partial h_t}{\partial h_{t-1}}$ with $\dfrac{\partial h_t}{\partial h_{t-1}}=\operatorname{diag}(\tanh'(\cdot))\,W_{hh}$. Ignoring the bounded $\tanh'\le1$ factor, the product $\approx W_{hh}^{\,T-k}=Q\Lambda^{T-k}Q^{-1}$, so each eigenvalue is raised to the power $T-k$. If **every** $|\lambda_i|<1$, $\Lambda^{T-k}\to0$ → gradient **vanishes** exponentially; if **some** $|\lambda_i|>1$, that component $\to\infty$ → gradient **explodes**. The **spectral radius** $\rho(W_{hh})=\max_i|\lambda_i|$ is the deciding quantity; only $\rho\approx1$ preserves signal over many steps.
>
> **4.** Because $\tanh'\le1$ always, the per-step Jacobian is biased toward *shrinking* the gradient, so in practice products tend to decay → vanishing dominates. **Gradient clipping** (rescale the gradient vector when its norm exceeds a threshold) cheaply tames explosion because explosion is a *magnitude* problem — you just cap the size. It cannot fix vanishing because a vanished gradient is (near) **zero**: there is no signal left to rescale up; the information about long-range dependencies is already lost. Vanishing needs an **architectural** fix (gating), not a post-hoc rescale.
>
> **5.** $c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$ with $f_t=\sigma(\cdot)$, $i_t=\sigma(\cdot)$, $\tilde c_t=\tanh(\cdot)$. Then $\dfrac{\partial c_t}{\partial c_{t-1}}=\operatorname{diag}(f_t)$. When the forget gate $f_t\approx1$ this Jacobian is $\approx I$, so backpropagating through the cell state multiplies by $\approx1$ per step — the gradient is neither shrunk nor blown up across long spans. That near-identity additive recurrence is the **constant-error carousel**: gradient/information can ride $c_t$ for hundreds of steps. The vanilla RNN instead multiplies by $\operatorname{diag}(\tanh')W_{hh}$ each step, whose repeated application gives the $\Lambda^{T-k}$ decay of answer 3.
>
> **6.** Vanilla RNN memory is **multiplicative**: $h_t$ passes through $W_{hh}$ every step, so long products are governed by $\rho(W_{hh})$ and generically vanish/explode. LSTM memory is **gated-additive**: $c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$ *adds* new content onto a conveyor belt whose retention is a learned gate, giving the $\approx I$ Jacobian. The **GRU** merges the cell and hidden state into one and uses only **update ($z_t$) and reset ($r_t$)** gates (no separate output gate) — fewer parameters. $h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$ is a **convex blend**: $(1-z_t)$ is how much old state to **keep**, $z_t$ is how much new candidate to **write**; $z_t\to0$ copies the past unchanged (long memory), $z_t\to1$ overwrites it.
>
> **7.** Limitations gating can't fix: (i) **sequential/non-parallel computation** — $h_t$ depends on $h_{t-1}$, forcing a serial length-$T$ loop that can't exploit accelerator parallelism across time; (ii) **$O(T)$ path length** — distant tokens communicate only through many intermediate states through a fixed-size bottleneck. Self-attention removes (i) by computing all positions in **one parallel $QK^\top$ matmul** (sequential depth $O(1)$), and (ii) by connecting any two positions in a **single hop** (max path length $O(1)$). The price: **$O(T^2)$** compute/memory in sequence length (all pairs interact) vs. the RNN's $O(T)$. That quadratic cost is what motivates **state-space models like Mamba** (and linear attention): recover RNN-like $O(T)$ scaling and a recurrent form while keeping attention-like trainability/selectivity.
>
> **8.** Causality is enforced by `torch.zeros(T,T).masked_fill(mask, float('-inf'))` where `mask = torch.triu(ones, diagonal=1)` sets the strict-upper (future) entries to $-\infty$ **before** softmax, so those weights become $0$. Row $i$ of $A$ therefore has nonzeros only for $j\le i$ — counts $1,2,3,4,5,6$ for rows $0..5$ — and, crucially, all rows are computed **in parallel** (no time loop). **Framing:** an RNN is message passing on a **path/chain graph** (node $t$ receives only from $t-1$), so information travels $O(T)$ hops; self-attention is message passing on a **complete graph** with learned, softmax-normalized edge weights, i.e. a permutation-invariant weighted aggregation over the token set — exactly the Deep-Sets/message-passing primitive $\text{generic}_{f,g}(\{h_u\})=f(\sum_u g(h_u))$ behind HW2 §D. A mean-aggregator is attention with uniform weights; seeing RNN-vs-attention as path-graph-vs-complete-graph message passing unifies today's lesson with the GNN expressivity problems (P15–P16).
