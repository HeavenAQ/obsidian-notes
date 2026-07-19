---
base: "[[01.2 DL Daily Lessons — Step-by-Step + Quiz.base]]"
Quiz score: ""
"HW #": 1
Studied: false
Quiz taken: false
Day type: Homework-Day
Date: 2026-07-13
Piece count: 4
---
> ⚠️ **Schedule note (Homework Day 1 of 3):** Jul 13–15 are the HW1 homework days; today you **attempt the theory problems** — Section A (Approximation, 14 pt) and Section B (Backprop, 3 pt). No new lesson. Below is a compact consolidation of the exact formulas and pitfalls those problems lean on, then a cumulative quiz. **The quiz tests lesson content only — it does not contain HW1 answers, and nothing below solves a graded problem for you.** Tomorrow (Jul 14) is the implementation attempt (manual forward/backward + training loop); Jul 15 is finish + submit.
>
> 🗒️ **Quiz backlog nudge:** yesterday's Jul 12 cumulative quiz is still **untaken** (Quiz taken = false, no score). The patch-up loop only fires on a *low recorded score*, so nothing is queued — but grade at least the **Jul 9 (attention)** and **Jul 10 (nanoGPT)** sets, plus yesterday's, to know where you actually stand before the problem set. Today's quiz is fresh (no repeats from Jul 12).

# 🎯 Today's focus

Attempt HW1's theory half. The whole of Section A is one idea seen from four angles: **a ReLU network is a continuous piecewise-linear function, built by folding hyperplanes.** Every sub-question (convexity, kink-counting, linear regions, depth efficiency, logic gates) is a corollary of that. Section B is a single skill: **differentiate in index form, then recognize outer-products and diagonal gates.** This page consolidates the formulas and the traps; you supply the actual answers on the homework.

# 🧩 Consolidation pieces

## Piece 1 — ReLU-net foundations & shape bookkeeping (powers Problems 1–2)

An $l$-layer ReLU network (layers counted by weight matrices) alternates affine maps with elementwise ReLU. The $l=2$ and $l=3$ forms:

$$f(x) = W_2\,\mathrm{ReLU}(W_1 x + b_1) + b_2$$

$$f(x) = W_3\,\mathrm{ReLU}\!\big(W_2\,\mathrm{ReLU}(W_1 x + b_1) + b_2\big) + b_3$$

Number of ReLU applications is always $l-1$. **Shape rule:** a layer taking $d_{\text{in}}$ to $d_{\text{out}}$ has $W\in\mathbb R^{d_{\text{out}}\times d_{\text{in}}}$, $b\in\mathbb R^{d_{\text{out}}}$, and the next layer's input dim must equal this layer's output dim. Trace dims left-to-right and every shape falls out.

- **ReLU** $\mathrm{ReLU}(z)=\max(0,z)$: continuous everywhere, differentiable everywhere *except* the kink at $z=0$.
- An **affine map** $z\mapsto Wz+b$ is both convex *and* concave.

**Pitfalls:** with a scalar input, $W_1x+b_1\in\mathbb R^{k}$ so $W_1\in\mathbb R^{k\times 1}$; a scalar output forces $W_2\in\mathbb R^{1\times k}$. Don't confuse "layers" (weight matrices, $l$) with "ReLU stages" ($l-1$).

**You've got this piece when you can** write out an $l=3$ network from scratch and state every matrix/vector shape given only input and output dims.

## Piece 2 — Piecewise-linearity, linear regions & kink counting (powers Problems 3–5)

Each ReLU neuron draws a hyperplane $\{x:(W_1x+b_1)_j=0\}$ in input space. On each side the neuron is "off" (outputs 0) or "on" (outputs its affine input). **Within any region where every neuron's on/off pattern is fixed, the entire network collapses to a single affine map** — so $f$ is continuous **piecewise-linear**, and its non-differentiable points (kinks) are exactly the region boundaries.

Region-count growth (the depth-efficiency lever, Montúfar et al. 2014):

- Adding **width** subdivides regions roughly **additively** — for a 1-D input, an $l=2$ width-$k$ net has at most $\mathcal O(k)$ kinks (each neuron contributes $\le 1$).
- Adding **depth** lets later layers fold earlier regions — region count can grow **exponentially in depth**. That is why deep nets represent highly oscillatory functions with far fewer neurons than a shallow net would need.

**Convexity (Problem 3):** ReLU is convex *and non-decreasing*, so $\mathrm{ReLU}(\text{affine})$ is convex; a **non-negative** combination of convex functions stays convex. But the outer weights $W_2$ may be **negative**, flipping convex to concave — so a general ReLU net is neither. A single counterexample (a tiny net with both an up-kink and a down-kink) settles "in general."

**tanh swap (Problem 4d):** $\tanh$ is $C^\infty$, so composing it with affine maps stays differentiable everywhere — the kink-counting logic vanishes.

**Bias-free pinning (Problem 5):** with no biases every kink hyperplane passes through the origin, since $\mathrm{ReLU}(wx)$ kinks at $x=0$ for any $w$. Useful identity: $\mathrm{ReLU}(x)-\mathrm{ReLU}(-x)=x$. When a requested shape is impossible, *argue it from the origin-pinned-kink fact* — don't try to plot it.

```python
import torch
# a bias-free width-2 ReLU net; note both kinks sit at x = 0
def f(x, W1, W2):                 # x:(N,1) W1:(2,1) W2:(1,2)
    return torch.relu(x @ W1.T) @ W2.T
x = torch.linspace(-5, 5, 400).unsqueeze(1)
```

**You've got this piece when you can** say, for a given $(l, \text{width})$, whether kink count grows linearly or exponentially, and justify why a bias-free net cannot place two kinks at two distinct $x$ values.

## Piece 3 — Logic gates as half-plane detectors (powers Problem 7)

Write any target as $f=\sum_j a_j\,\mathrm{ReLU}(w_j^\top x + b_j)$ and hand-pick $w_j,b_j$ so each ReLU "turns on" in exactly the half-plane you need; verify across all four sign-quadrants of $(x_1,x_2)$.

- **OR:** $\mathrm{ReLU}(x_1)+\mathrm{ReLU}(x_2)>0 \iff x_1>0$ or $x_2>0$.
- **XOR:** not linearly separable — needs depth/width (compose AND with OR/negation, since $\mathrm{XOR}(a,b)=(a\ \mathrm{OR}\ b)\ \mathrm{AND}\ \neg(a\ \mathrm{AND}\ b)$).
- **NAND** is functionally complete, so a ReLU NAND ⇒ ReLU nets can represent any boolean function.

**Pitfall:** the conditions are on **real** inputs with **strict** inequalities at the boundary — a ReLU input crossing zero is where the decision flips, so place kinks exactly on the target boundary.

**You've got this piece when you can** design $W_1,W_2$ for OR by inspection and explain why XOR needs more than one linear layer.

## Piece 4 — Backprop: index-then-recognize (powers Problems 8 & 10)

Canonical linear-layer gradients ($\mathrm{out}=Wx+b$, upstream $g=\partial L/\partial\mathrm{out}$):

$$\frac{\partial L}{\partial x}=W^\top g,\qquad \frac{\partial L}{\partial W}=g\,x^\top,\qquad \frac{\partial L}{\partial b}=g$$

ReLU local gradient is the gate; softmax-CE collapses beautifully:

$$\frac{\partial L}{\partial x}=g\odot\mathbb 1[x>0],\qquad \frac{\partial L}{\partial z}=\hat p - y_{\text{onehot}},\quad \hat p=\mathrm{softmax}(z)$$

For Problem 8's coupled system $y=Wx,\ u=\mathrm{ReLU}(y),\ v=u+Wu,\ L=\tfrac12\|v\|_2^2$, the toolkit is: **Kronecker delta** $\sum_k\delta_{ik}a_k=a_i$ to kill sums; **Heaviside** $\Theta(y)$ (the ReLU derivative) becoming $\mathrm{diag}(\Theta(y))$; and reading the free-index pattern $\delta_{im}u_j$ as an **outer product**. Target structure to reach:

$$\frac{\partial L}{\partial W}=v\otimes u + \mathrm{diag}(\Theta(y))\,(I+W^\top)\,v\otimes x$$

Always confirm algebra numerically before trusting it:

```python
import torch
d = 4
W = torch.randn(d, d, requires_grad=True); x = torch.randn(d)
y = W @ x; u = torch.relu(y); v = u + W @ u
L = 0.5 * (v @ v); L.backward()
Theta = (y >= 0).float()
manual = torch.outer(v, u) + torch.outer(torch.diag(Theta) @ (torch.eye(d) + W.T) @ v, x)
print(torch.allclose(W.grad, manual, atol=1e-5))
```

**Pitfalls:** (1) transposes — $g\,x^\top$ vs $W^\top g$; check *shapes*, not habit. (2) The ReLU gate zeros gradient where neurons were off. (3) Batch versions **sum over the batch dim** and divide by batch size if the loss averages. (4) Don't re-derive the full softmax Jacobian at runtime — use $\hat p - y$.

**You've got this piece when you can** write the three Linear gradients and the softmax-CE gradient from memory, and explain each factor ($v\otimes u$, the gate, $I+W^\top$) in the Problem 8 identity.

# 📝 Cumulative quiz

Answer all 7 without notes (spans Jul 2–10, weighted toward today's theory attempt). Grade with the key at the bottom, then fill in **Quiz score** and check **Quiz taken**.

1. **(Jul 5/10 — derivation)** For $\mathrm{out}=Wx+b$ with upstream $g$, write $\partial L/\partial x$, $\partial L/\partial W$, $\partial L/\partial b$. Then give the ReLU local gradient and the softmax-cross-entropy gradient $\partial L/\partial z$, and state why the latter is so simple.
2. **(Jul 2/6 — conceptual)** In one sentence each: what set does $\mathbf w\cdot\mathbf x+b=0$ describe, and how does a single ReLU neuron partition input space? Why is a ReLU network therefore **piecewise-linear**?
3. **(Jul 6 — conceptual)** Is a general ReLU network convex in its input $x$? Justify: which composition keeps convexity, and what single ingredient breaks it? How would you *disprove* a convexity claim in general?
4. **(Jul 3 — derivation)** Give the entrywise definition of $(AB)_{ij}$ and rewrite $AB$ as a sum of outer products. Explain why this identity is exactly what makes $\partial L/\partial W=g\,x^\top$ (and its batched form) a single GPU-friendly matmul.
5. **(Jul 10 — conceptual)** For a 1-D input, roughly how does the number of kinks scale with **width** in an $l=2$ net, versus with **depth** in a deep net? Name the one-line consequence for representational efficiency.
6. **(Jul 5/10 — derivation)** Two i.i.d. unit-variance vectors $q,k\in\mathbb R^{d_k}$: compute $\mathrm{Var}(q^\top k)$, state the scaled dot-product attention formula with causal mask, and explain what breaks at initialization if you drop the $1/\sqrt{d_k}$.
7. **(Jul 8 — conceptual)** Describe the autoregressive generation loop in one sentence. What does **temperature** do to the softmax, how do **top-k** and **top-p** differ, and in what sense are **RAG** and **chain-of-thought** the same mechanism?

> [!note]- 🔑 Answer key (click to reveal)
> **1.** $\partial L/\partial x=W^\top g$, $\partial L/\partial W=g\,x^\top$, $\partial L/\partial b=g$ (batch: sum $b$-grad over the batch dim). ReLU: $\partial L/\partial x=g\odot\mathbb 1[x>0]$. Softmax-CE: $\partial L/\partial z=\hat p - y_{\text{onehot}}$ with $\hat p=\mathrm{softmax}(z)$. It is simple because the softmax Jacobian and the $-\log$ of the log-loss combine — the $\hat p_i(\delta_{ij}-\hat p_j)$ terms telescope against the $1/\hat p_y$ from the log, leaving just "predicted minus true."
>
> **2.** $\mathbf w\cdot\mathbf x+b=0$ is a **hyperplane** (line in 2-D, plane in 3-D) with normal $\mathbf w$. A ReLU neuron is "on" ($>0$) on one side of that hyperplane and outputs 0 on the other — a half-space detector. Since each neuron's on/off pattern is fixed within a region and the net reduces to a single affine map there, $f$ is continuous **piecewise-linear**, with kinks exactly at the region boundaries.
>
> **3.** Not in general. $\mathrm{ReLU}(\text{affine})$ is convex because ReLU is convex *and non-decreasing*, and non-negative sums of convex functions are convex — but the outer weights $W_2$ can be **negative**, flipping a convex piece to concave, so the composite is generally neither convex nor concave. Disprove a convexity claim with a single **counterexample**: exhibit a tiny net whose graph has both an upward and a downward kink.
>
> **4.** $(AB)_{ij}=\sum_k A_{ik}B_{kj}$; column–row form $AB=\sum_k(\mathrm{col}_k A)(\mathrm{row}_k B)$, a sum of rank-1 outer products. A weight gradient reassembles as (upstream) ⊗ (input): $\partial L/\partial W=g\,x^\top$ is one outer product, and summed over a batch it becomes $G^\top X$ — a single matmul, which GPUs execute as one fused op.
>
> **5.** With **width**: kinks grow roughly **linearly** ($\mathcal O(k)$ for width $k$, each neuron $\le 1$ kink on a 1-D input). With **depth**: the number of linear regions can grow **exponentially** in depth (later layers fold earlier regions). Consequence: deep nets can represent highly oscillatory / complex functions with far fewer parameters than a shallow net — **depth efficiency**.
>
> **6.** $q^\top k=\sum_{c=1}^{d_k}q_ck_c$ is a sum of $d_k$ independent mean-0 unit-variance terms, so $\mathrm{Var}(q^\top k)=d_k$ (std $\sqrt{d_k}$). Attention: $\mathrm{softmax}\!\big(\tfrac{QK^\top}{\sqrt{d_k}}+M\big)V$ with $M_{ij}=0$ for $j\le i$, $-\infty$ for $j>i$. Unscaled, logits scale like $\sqrt{d_k}$ and push softmax into a **peaky** regime where one weight $\to 1$ and its gradient $\to 0$ — attention just copies one neighbor and can't learn; dividing by $\sqrt{d_k}$ keeps logits $\mathcal O(1)$ and attention diffuse at init.
>
> **7.** Loop: run the context through the model, take next-token logits, sample one token, append it, repeat. **Temperature** $T$ divides logits before softmax — $T>1$ flattens (more random), $T<1$ sharpens (more greedy). **Top-k** keeps the $k$ highest-probability tokens; **top-p (nucleus)** keeps the smallest set whose cumulative probability $\ge p$ (adaptive count). **RAG** and **CoT** are the same mechanism — both inject helpful tokens into the context (retrieved passages vs. intermediate reasoning) to condition generation; the model and sampling loop are unchanged.
>
> **Scoring:** 1 pt each (half credit for the right idea with a wrong detail). ≥ 5/7 → your theory foundations are solid; go attempt Section A & B. Below 5 → re-read the tagged consolidation piece above before starting the corresponding problem.
