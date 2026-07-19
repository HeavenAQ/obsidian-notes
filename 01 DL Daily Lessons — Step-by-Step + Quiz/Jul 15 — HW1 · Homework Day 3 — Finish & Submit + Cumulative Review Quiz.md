---
base: "[[01.2 DL Daily Lessons — Step-by-Step + Quiz.base]]"
Quiz score: ""
"HW #": 1
Studied: false
Quiz taken: false
Day type: Review-Day
Date: 2026-07-15
Piece count: 5
---
> ⚠️ **Schedule note (Homework Day 3 of 3 — DUE TODAY, doubles as Review Day):** This is the last day of the HW1 window (Jul 2–15). Today = **finish + submit HW1** and mark both certificates' modules complete. Jul 13 was the theory attempt (Sections A + B), Jul 14 the implementation attempt (Section C Colab). No new lesson. Below is a **whole-window consolidation** — the five conceptual threads the entire two weeks fed into — followed by a **cumulative review quiz** that mixes every section plus spaced-repetition callbacks to the Jul 2–3 linear algebra and the Jul 8–10 attention days. **The quiz tests lesson content only — it contains no HW1 answers and writes no graded cell for you.** ✅ Done = HW1 submitted, PyTorch cert + Transformers-in-Practice modules finished.
>
> 🗒️ **Quiz backlog nudge:** Jul 14's implementation quiz is still **untaken** (Quiz taken = false, no score), and Jul 12/Jul 13 are unscored too. The patch-up loop only fires on a *recorded* low score, so nothing auto-queued — but this is the last day, so grade the **Jul 13 (theory)** and **Jul 14 (implementation)** sets today to confirm your footing before you submit. Today's 8 questions are fresh (deliberately weighted toward Section A + Prob 8, which the Jul 14 set skipped).

# 🎯 Today's focus

Close the loop. Two weeks moved you from vectors and matrix multiply (Jul 2–3) → PyTorch data/NN mechanics (Jul 5–7) → transformer theory and a hand-coded GPT (Jul 8–10) → the HW1 math (ReLU nets, linear regions, matrix-calculus backprop, CIFAR-10). This page restates the **five threads** that actually get graded, each as a compact "formula + why + pitfall" card, then a cumulative quiz for a final self-check. Read the cards, grade your two open quizzes, submit.

# 🧭 Consolidation pieces

## Piece 1 — ReLU networks: shapes, convexity, and the piecewise-linear picture (Section A · Probs 1–4)

Source: **HW1 §0 + Section A**, backed by Jul 2–3 shape bookkeeping.

A depth-$l$ ReLU net (layers counted by weight matrices) alternates affine maps with elementwise ReLU. The $l=2$ scalar-input case:

$$f(x;W_1,W_2,b_1,b_2)=W_2\,\mathrm{ReLU}(W_1x+b_1)+b_2$$

**Shapes fall out of one rule:** a layer taking $d_\text{in}$ and emitting $d_\text{out}$ has $W\in\mathbb R^{d_\text{out}\times d_\text{in}}$, $b\in\mathbb R^{d_\text{out}}$, and the next layer's input dimension must equal $d_\text{out}$. Trace left→right.

**Convexity (Prob 3):** ReLU is convex *and* non-decreasing, so $\mathrm{ReLU}(\text{affine})$ is convex — but a **negative** outer weight flips it to concave, and stacking signs gives functions that are neither. A single counterexample settles "in general."

**Piecewise-linear picture (Prob 4) — the conceptual heart:** each neuron is a hyperplane $\{x:(W_1x+b_1)_j=0\}$ splitting input space into on/off half-spaces. Inside any region where every neuron's on/off pattern is fixed, the whole net collapses to **one affine map** ⇒ the function is continuous and piecewise-linear; its non-differentiable points are exactly the **region boundaries (kinks)**. Region count grows **linearly in width** but **multiplicatively (exponentially) in depth** — that is depth efficiency (Montúfar et al. 2014). With $\tanh$ instead (Prob 4d), the nonlinearity is $C^\infty$, so no kinks — the function is differentiable everywhere.

**Pitfall:** "composition of convex is convex" is false once an intermediate map can be non-monotone or an outer weight is negative — don't over-apply the preservation rule.

**You've got this piece when you can** read off all layer shapes from the input dimension alone, give a width-2 counterexample to convexity, and state how linear-region count scales with width vs. depth.

## Piece 2 — Bias-free width-2 constructions & logic gates (Section A · Probs 5, 7)

Source: **HW1 Section A**.

**Origin-pinned kinks (Prob 5):** with no biases, every $\mathrm{ReLU}(wx)$ kinks at $x=0$ regardless of $w$. So all kinks of a bias-free width-2 net sit at the origin. The identity that rebuilds a straight line:

$$\mathrm{ReLU}(x)-\mathrm{ReLU}(-x)=x$$

That is why a linear output is constructible; two kinks at *distinct* $x$-values are **not**, because bias-free kinks cannot move off the origin — you argue the impossibility rather than plot it.

**Logic gates (Prob 7):** design weights so each ReLU turns on in exactly the half-plane you need, then read off the sign quadrants of $(x_1,x_2)$.

$$\mathrm{OR}:\ \mathrm{ReLU}(x_1)+\mathrm{ReLU}(x_2)>0 \iff x_1>0 \ \text{or}\ x_2>0$$

XOR is not linearly separable, so use the depth budget: $\mathrm{XOR}(a,b)=(a\,\text{OR}\,b)\ \text{AND}\ \neg(a\,\text{AND}\,b)$. NAND is **functionally complete** — a ReLU NAND ⇒ ReLU nets can represent *any* boolean function.

**Pitfall:** the inputs are real (non-boolean), so verify the *strict* inequality on the actual half-planes, not just the four corner cases.

**You've got this piece when you can** justify why $\mathrm{ReLU}(x)-\mathrm{ReLU}(-x)=x$ gives a linear map, argue two distinct kinks are impossible without bias, and sketch an OR gate's weights.

## Piece 3 — Matrix-calculus backprop: the index-then-recognize workflow (Section B · Prob 8)

Source: **HW1 Section B**, same skill as Jul 10's manual attention.

Forward system: $y=Wx,\ u=\mathrm{ReLU}(y),\ v=u+Wu,\ L=\tfrac12\lVert v\rVert_2^2$. Differentiate a single entry, then reassemble:

$$\frac{\partial L}{\partial W_{ij}}=\sum_{m} v_m\,\frac{\partial v_m}{\partial W_{ij}}\ \Longrightarrow\ \frac{\partial L}{\partial W}=v\otimes u+\mathrm{diag}\big(\Theta(y)\big)\,(I+W^\top)\,v\otimes x$$

Three mechanics do all the work: the **Kronecker delta** $\delta_{ik}$ collapses index sums ($\sum_k\delta_{ik}a_k=a_i$); the **Heaviside** $\Theta(y)$ (ReLU's derivative) becomes the diagonal **gate matrix** $\mathrm{diag}(\Theta(y))$ that zeroes off-neuron components; and free indices $i,j$ reassembling into $v_i u_j$ are recognized as the **outer product** $v\otimes u$. The reason the problem flags this as GPU-friendly: outer products and diagonal scalings are single fused matmuls.

```python
import torch
d = 4
W = torch.randn(d, d, requires_grad=True); x = torch.randn(d)
y = W @ x; u = torch.relu(y); v = u + W @ u
L = 0.5 * (v @ v); L.backward()

Theta = (y >= 0).float()
manual = torch.outer(v, u) + torch.outer(torch.diag(Theta) @ (torch.eye(d) + W.T) @ v, x)
print(torch.allclose(W.grad, manual, atol=1e-5))   # confirms the algebra
```

**Pitfall:** decide every transpose by **shape**, not habit; and capture the ReLU gate from the *forward* pass ($y\ge0$) — recomputing it downstream is where sign errors creep in.

**You've got this piece when you can** name what each of the three terms ($\delta$, $\Theta$-gate, outer product) contributes and reproduce the closed form well enough to pass the autograd `allclose` check.

## Piece 4 — Manual forward/backward & the softmax-CE collapse (Section C · Prob 10)

Source: **PT-Cert C1 M4** + Jul 5/Jul 10 → §M4 of [[PyTorch for Deep Learning Professional Certificate (DeepLearning.AI) — Deep Dive Companion]].

Per-sample Linear gradients (upstream $g=\partial L/\partial\text{out}$), then ReLU's gate, then the softmax–cross-entropy simplification:

$$\frac{\partial L}{\partial x}=W^\top g,\quad \frac{\partial L}{\partial W}=g\,x^\top,\quad \frac{\partial L}{\partial b}=g;\qquad \frac{\partial L}{\partial x}=g\odot\mathbb 1[x>0];\qquad \frac{\partial L}{\partial z}=\hat p-y$$

$\partial L/\partial W=g\,x^\top$ is a rank-1 **outer product** — the exact object from Piece 3 and from the Jul 2–3 "outer-product view" of matrix multiply. The softmax-CE gradient collapses to **predicted minus true** because the softmax Jacobian $\hat p_i(\delta_{ij}-\hat p_j)$ chained with the log-loss $-1/\hat p_y$ telescopes. Work with **logits** (LogSoftmax + NLL) for log-sum-exp stability, never pre-softmaxed probabilities.

**Pitfall:** feeding already-softmaxed values into `CrossEntropyLoss` double-applies softmax and silently wrecks training; divide by $N$ once if the loss averages.

**You've got this piece when you can** write all three layers' backward in ≤5 lines each and explain why $g x^\top$ and $\hat p-y$ are the "outer product" and "predicted-minus-true" shortcuts.

## Piece 5 — Training reality + the attention callback (Section C Probs 11–15 · spaced repetition to Jul 8–10)

Source: **PT-Cert C1 M4** + Jul 9–10 attention.

**Loop ritual (order fixed):** `zero_grad` → forward → loss → `backward` → `step`; eval under `@torch.no_grad()` with `model.eval()`. **Why not 100% train accuracy (Prob 13):** UAT guarantees a fitting network *exists*, not that SGD *finds* it (finite width, unconverged optimization, regularization) — and 100% usually means overfitting. **Augmentation (Prob 14–15):** train-only `RandomCrop(32, padding=4)`; it regularizes, so expect a **narrower train–val gap**.

**Attention callback (Jul 9–10):** the same softmax + matrix-calculus skill reappears in scaled dot-product attention —

$$\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

The $\sqrt{d_k}$ divisor keeps the dot-product logits' variance $O(1)$ so softmax doesn't saturate into near-one-hot (vanishing gradients) — see [[Scaled Dot-Product Attention Why Divide by √dₖ]]. This is the same "keep pre-softmax logits well-scaled" concern as the log-sum-exp shift in Piece 4.

**Pitfall:** forgetting `zero_grad` accumulates gradients across batches; forgetting `eval()`/`no_grad()` leaves BatchNorm/Dropout in train mode and wastes memory building the graph.

**You've got this piece when you can** recite the five-step loop, give two reasons a net won't hit 100% train accuracy, and explain why attention divides by $\sqrt{d_k}$.

# 📝 Cumulative review quiz

Answer all 8 without notes (spans the whole Jul 2–15 window; weighted to Section A + Prob 8, with spaced-repetition callbacks). Grade with the key below, then fill in **Quiz score** and check **Quiz taken**.

1. **(Section A / Prob 4 — conceptual)** Explain why a ReLU network is continuous and piecewise-linear, and identify exactly where its non-differentiable points are. State how the number of linear regions scales with **width** vs. **depth**, and why that yields "depth efficiency."
2. **(Section A / Prob 5 — construction)** With no biases, where must every kink of a width-2 ReLU net sit, and why? Use $\mathrm{ReLU}(x)-\mathrm{ReLU}(-x)=x$ to explain how a linear output is built, then argue whether two kinks at *distinct* $x$-values are possible.
3. **(Section A / Prob 3 — reasoning)** Is a ReLU network's output convex in its input $x$ in general? Explain how a negative outer weight defeats the "composition of convex functions" intuition, and what kind of evidence settles the claim.
4. **(Section A / Prob 7 — construction)** Give weights for an OR gate as a sum of ReLUs and state the condition it must satisfy. Why does exhibiting a ReLU **NAND** prove ReLU nets can represent any boolean function?
5. **(Section B / Prob 8 — derivation)** For $y=Wx,\ u=\mathrm{ReLU}(y),\ v=u+Wu,\ L=\tfrac12\lVert v\rVert^2$, name what each of the three ingredients — the Kronecker delta $\delta_{ik}$, the Heaviside $\Theta(y)$, and the outer product — contributes on the way to $\partial L/\partial W=v\otimes u+\mathrm{diag}(\Theta(y))(I+W^\top)v\otimes x$.
6. **(Jul 9–10 / attention — spaced repetition)** Write scaled dot-product attention and explain precisely what dividing by $\sqrt{d_k}$ prevents. How is this the same concern as the log-sum-exp shift in softmax cross-entropy?
7. **(Jul 2–3 / linear algebra — spaced repetition)** Describe the **outer-product view** of matrix multiplication and connect it to why the Linear-layer weight gradient is $\partial L/\partial W=g\,x^\top$ (and its batched form a sum of outer products).
8. **(Section C / Prob 10 — code-reading)** Write the backward for Linear, ReLU, and softmax cross-entropy (per-sample). Why is $\partial L/\partial z=\hat p-y$, and why must the loss be applied to **logits** rather than pre-softmaxed probabilities?

> [!note]- 🔑 Answer key (click to reveal)
> **1.** Continuous because it composes continuous maps (affine + ReLU). Piecewise-linear because each neuron's on/off state partitions input space by hyperplanes; inside any region with a fixed on/off pattern the whole net is a single affine map. Non-differentiable points are exactly the **region boundaries (kinks)** where a neuron switches on/off. Linear-region count grows **linearly in width** (each added neuron adds at most one cut on a 1-D input line) but **multiplicatively — exponentially — in depth** (later layers fold regions made by earlier ones; Montúfar et al. 2014). Depth efficiency: a deep net can realize exponentially many regions with polynomially many neurons, whereas a shallow ($l=2$) net needs roughly one neuron per region.
>
> **2.** Every bias-free $\mathrm{ReLU}(wx)$ kinks at $x=0$ regardless of $w$, so **all** kinks sit at the origin. A linear output is built because $\mathrm{ReLU}(x)-\mathrm{ReLU}(-x)=x$: a positive and a negated copy recombine into the identity (a straight line, no visible kink). Two kinks at *distinct* $x$-values are **impossible** without biases — you cannot move a kink off the origin, so you argue the impossibility from the origin-pinned-kink fact rather than plotting.
>
> **3.** No, not in general. $\mathrm{ReLU}(\text{affine})$ is convex (ReLU is convex and non-decreasing), but a **negative outer weight** $W_2$ multiplies that convex piece by a negative number, flipping it concave; combining pieces of both signs yields functions that are neither convex nor concave. The preservation rule "composition of convex stays convex" requires the outer function to be convex *and non-decreasing* — a negative weight breaks the monotonicity. A single width-1 or width-2 **counterexample** (a graph with both an upward and a downward kink) settles it.
>
> **4.** OR: $f(x)=\mathrm{ReLU}(x_1)+\mathrm{ReLU}(x_2)$, which is $>0 \iff x_1>0$ or $x_2>0$ (verify the strict inequality on the real half-planes, not just boolean corners). NAND is **functionally complete** — every boolean function can be composed from NAND gates alone — so a ReLU implementation of NAND implies ReLU networks can represent *any* boolean function.
>
> **5.** The **Kronecker delta** $\delta_{ik}$ collapses index sums ($\sum_k\delta_{ik}a_k=a_i$), selecting single terms when you differentiate w.r.t. $W_{ij}$. The **Heaviside** $\Theta(y)$ is ReLU's derivative; reassembled it becomes the diagonal **gate matrix** $\mathrm{diag}(\Theta(y))$ that zeroes gradient components for off-neurons. The free indices $i,j$ (e.g. $v_i u_j$) reassemble into the **outer product** $v\otimes u$ — a rank-1 matrix. The $(I+W^\top)$ factor comes from folding the chain $v\to u\to y$ through both the direct $u$ and the $Wu$ paths. All three combine into the stated closed form, which is GPU-friendly (fused matmuls).
>
> **6.** $\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\big(QK^\top/\sqrt{d_k}\big)V$. Without the divisor, dot products of $d_k$-dim vectors have variance $\propto d_k$, so logits grow with dimension and softmax **saturates** toward one-hot, giving vanishing gradients and brittle attention. Dividing by $\sqrt{d_k}$ rescales logit variance back to $O(1)$. Same concern as the **log-sum-exp** shift in softmax-CE: both keep the pre-softmax logits well-scaled so the exponential/softmax stays numerically stable and gradients stay informative.
>
> **7.** Outer-product view: $AB=\sum_k a_{:,k}\,b_{k,:}$ — the product is the sum of rank-1 outer products of $A$'s columns with $B$'s rows. This is exactly how the weight gradient arises: for one sample, $\partial L/\partial W=g\,x^\top$ is the rank-1 outer product of the upstream gradient $g$ with the input $x$; batched, $\partial L/\partial W=G^\top X=\sum_n g_n x_n^\top$ sums those per-sample outer products in one fused matmul.
>
> **8.** Backward: Linear — $\partial L/\partial x=W^\top g$, $\partial L/\partial W=g\,x^\top$, $\partial L/\partial b=g$; ReLU — $g\odot\mathbb 1[x>0]$; softmax-CE — $\partial L/\partial z=\hat p-y$. It collapses to predicted-minus-true because the softmax Jacobian $\hat p_i(\delta_{ij}-\hat p_j)$ chained with the log-loss term $-1/\hat p_y$ telescopes (the $\hat p_y$ factors cancel). It must be applied to **logits** because `CrossEntropyLoss` = LogSoftmax + NLL and relies on log-sum-exp stability; pre-softmaxing then applying it double-applies softmax and wrecks training.
>
> **Scoring:** 1 pt each (half credit for the right idea with a wrong detail). ≥ 6/8 → the HW1 conceptual foundation is solid; submit with confidence. Below 6 → re-read the tagged consolidation piece before you finalize the matching problem.
