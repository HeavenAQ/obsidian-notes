---
base: "[[01.3 DL Homework Practice — MIT 6.7960.base]]"
Status: In-Progress
Notes ready: true
Window: Aug 5 – Aug 11, 2026 (weekly block)
Due Date: 2026-08-11
"HW #": 2
PDF: https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_hw2.pdf
Topics: "steepest descent, dual norms, spectral norm, muon/spectral update, hyperparameter transfer, power iteration, weight decay, CNN inductive bias, message-passing GNNs, WL expressivity"
---
# 📚 Prep lessons before attempting HW2 (due Aug 11)

Focus: generalization, optimization, and transformers (6.7960 Lectures 6–9). Daily plan Aug 5–9; attempt HW Aug 9–11.

- [ ] **Day 1 — Udemy Linear Algebra** → Eigenvalues & Eigenvectors (I): eigendecomposition, diagonalization, matrix powers (needed for optimization dynamics & Hessian analysis). *(Skip the difference/differential-equations sections.)*
- [ ] **Day 2 — Udemy Linear Algebra** → Positive definiteness, the Hessian & Taylor expansion, Lagrange multipliers: the local geometry of loss landscapes
- [ ] **Day 3 — Udemy Linear Algebra** → Least squares, the normal equation & projection matrices: the linear-regression picture behind generalization theory
- [ ] **Day 4 — Udemy LLM course** → from "Build a GPT": *Temporal causality via linear algebra*, *The "attention" algorithm (theory)*, *The Transformer block (theory)*, *Multihead attention* — theory only; implementation comes in HW7 week
- [ ] **Day 5 — MIT 6.S191 Lecture 2** — Deep Sequence Modeling (attention intro) ([link](https://introtodeeplearning.com/)); review + start HW2

---

Reading notes are added below.

---

# 📘 HW2 Reading Notes — Steepest Descent, Hyperparameter Transfer, Inductive Bias & GNNs

*Prep notes only — the concepts, theory, derivations, and PyTorch techniques each problem needs so you can solve it yourself. **No final answers.** 30 pts total; bonus/optional parts are ungraded. Four sections: Steepest Descent (9pt), Hyperparameter Transfer (6pt), Architecture & Inductive Bias (5pt), Graph Neural Networks (10pt).*

Today's lesson (Day 1) — **Udemy Linear Algebra → Eigenvalues & Eigenvectors (I)**: eigendecomposition, diagonalization, matrix powers. This is the backbone of §Hyperparameter Transfer (power iteration = repeated matrix powers amplifying the top eigenvector) and of the spectral-norm material in §Steepest Descent. Keep the identities $A = Q\Lambda Q^{-1}$, $A^k = Q\Lambda^k Q^{-1}$, and (for symmetric $A$) the orthogonal eigenbasis in front of you.

---

## 0. Foundations you'll reuse across the set

**Norms and dual norms.** A norm $\|\cdot\|:\mathbb{R}^n\to\mathbb{R}_{\ge0}$ measures length; it need not be Euclidean. The **dual norm** is

$$\|a\|^\dagger \;\stackrel{\text{def}}{=}\; \max_{b:\,\|b\|=1} a^\top b .$$

Read this as: "how large can the linear functional $a^\top(\cdot)$ get on the unit ball of $\|\cdot\|$?" The maximizing $b$ is the *direction* that best aligns with $a$ under the geometry of $\|\cdot\|$. Two facts you will lean on repeatedly:

- For $\ell_p$ norms, the dual of $\ell_p$ is $\ell_q$ with $\tfrac1p+\tfrac1q=1$ (Hölder's inequality is the engine). Special cases: $\ell_2$ is self-dual; $\ell_1\leftrightarrow\ell_\infty$.
- The **spectral norm** $\|A\|_\ast$ (largest singular value $\sigma_1$) is dual to the **nuclear norm** (sum of singular values $\sum_i\sigma_i$) under the Frobenius inner product $\langle A,B\rangle=\mathrm{trace}(A^\top B)$.

**SVD.** Any $G\in\mathbb{R}^{m\times n}$ factors as $G=U\Sigma V^\top$ with $U,V$ orthogonal and $\Sigma$ diagonal with nonnegative entries $\sigma_1\ge\sigma_2\ge\dots\ge0$. The columns of $U$/$V$ are the left/right singular vectors. Since $GG^\top=U\Sigma^2U^\top$, the singular values are the square roots of the eigenvalues of $GG^\top$ — this is the exact bridge from today's eigenvalue lesson to power iteration (Problem 8).

**Steepest descent, the big idea.** Gradient descent is *not* norm-agnostic. Minimizing the local linear model $g^\top\Delta w$ under a quadratic penalty $\tfrac\lambda2\|\Delta w\|^2$ gives an update whose *direction* depends on which norm you penalize with. Euclidean penalty → ordinary gradient step; $\ell_\infty$ penalty → sign-gradient (Adam-like); spectral-norm penalty on matrices → the muon/spectral update. This one variational problem unifies GD, sign-GD/Adam, and spectral optimizers.

---

## §A · Steepest Descent (9 pt) — Problems 1–5

The whole section is one idea developed in layers: **you are choosing the geometry in which "steepest" is measured.** Everything reduces to the master identity you must be able to re-derive (Problem 2):

$$\arg\min_{\Delta w}\Big(g^\top\Delta w+\tfrac{\lambda}{2}\|\Delta w\|^2\Big)\;=\;-\frac{\|g\|^\dagger}{\lambda}\cdot\arg\max_{\|t\|=1}g^\top t .$$

The two factors have clean meanings: $\|g\|^\dagger/\lambda$ is the *step length* (bigger gradient, softer penalty → longer step) and $\arg\max_{\|t\|=1}g^\top t$ is the *unit direction* the geometry picks.

### Problem 1 requires: dual norms of $\ell_2$ and $\ell_\infty$ (2 pt)

**Concept.** The dual norm $\|a\|^\dagger=\max_{\|b\|=1}a^\top b$ is the operator norm of the linear functional $b\mapsto a^\top b$ over the unit ball of $\|\cdot\|$. The single tool you need is **Hölder's inequality**: $a^\top b\le\|a\|_p\|b\|_q$ with $\tfrac1p+\tfrac1q=1$, together with the fact that Hölder is *tight* — there is always a $b$ achieving equality.

- **(a) $\ell_2$:** apply Cauchy–Schwarz (Hölder with $p=q=2$). Ask which unit $b$ aligns maximally with $a$; the maximizing direction is $b=a/\|a\|_2$. Conclude which $q$ you land in.
- **(b) $\ell_\infty$:** here $\|b\|_\infty=\max_i|b_i|=1$ means every coordinate is free to be $\pm1$. To maximize $\sum_i a_i b_i$ set each $b_i=\operatorname{sign}(a_i)$. Read off the resulting value as an $\ell_q$ norm of $a$.

**Why it matters.** The pairing $(p,q)$ you find is exactly what turns the abstract master identity into a concrete update rule in Problem 3. Keep the general rule $\tfrac1p+\tfrac1q=1$ and the self-dual/swap special cases ($2\leftrightarrow2$, $1\leftrightarrow\infty$) memorized.

### Problem 2 requires: proving the master dual identity (2 pt)

**Technique — the hint is the whole proof.** Change variables $\Delta w = r\,t$ where $r=\|\Delta w\|\ge0$ (magnitude) and $t=\Delta w/\|\Delta w\|$ (unit direction, $\|t\|=1$). The objective separates:

$$g^\top(rt)+\tfrac{\lambda}{2}r^2\|t\|^2 \;=\; r\,(g^\top t)+\tfrac{\lambda}{2}r^2 .$$

Now minimize in two stages. (i) For fixed $r>0$, minimizing $r\,g^\top t$ over $\|t\|=1$ means making $g^\top t$ as *negative* as possible, i.e. $t^\star=-\arg\max_{\|t\|=1}g^\top t$, giving value $-r\|g\|^\dagger$. (ii) Substitute back to get a scalar problem $\min_{r\ge0}\big(-r\|g\|^\dagger+\tfrac\lambda2 r^2\big)$; set the derivative to zero to find $r^\star=\|g\|^\dagger/\lambda$. Combine $r^\star$ and $t^\star$. Be explicit about the sign — that minus sign is why this is *descent*.

**Why formulated this way.** Decoupling length from direction is the standard trick for norm-constrained linear optimization; it converts a hard vector problem into "pick the best unit direction (a dual-norm computation), then pick the best length (a 1-D quadratic)."

### Problem 3 requires: instantiating the identity under $\ell_2$ and $\ell_\infty$ (2 pt)

Plug Problem 1's dual norms and the corresponding maximizing directions into the master identity.

- **(a) $\ell_2$:** $\|g\|^\dagger_2=\|g\|_2$ and $\arg\max_{\|t\|_2=1}g^\top t=g/\|g\|_2$. The two $\|g\|_2$ factors interact — watch what cancels. This should recover *ordinary gradient descent* $\Delta w\propto -g/\lambda$, confirming Euclidean geometry = vanilla GD.
- **(b) $\ell_\infty$:** the maximizing unit-$\ell_\infty$ direction is the sign vector; combined with $\|g\|^\dagger_\infty=\|g\|_1$ you get an update proportional to $-\operatorname{sign}(g)$. This is **sign gradient descent**.
- **(c, optional) Adam link:** with $\beta_1=\beta_2=\epsilon=0$ the first/second moment estimates collapse to the current gradient and its square; the update $-m_t/\sqrt{v_t}$ becomes $-g/\sqrt{g^2}=-\operatorname{sign}(g)$ coordinate-wise. Argue this equals the direction from (b). This is the conceptual bridge "Adam ≈ steepest descent in $\ell_\infty$."

### Problem 4 requires: steepest descent under the spectral norm (3 pt)

**Setup.** Move from vectors to matrices with the Frobenius inner product $\langle G,\Delta W\rangle=\operatorname{trace}(G^\top\Delta W)$ and the matrix master identity (Eq. 5). The spectral norm $\|\cdot\|_\ast=\sigma_1$ (largest singular value). Its dual under Frobenius is the **nuclear norm** $\sum_i\sigma_i=\operatorname{trace}\Sigma$ — that is exactly what part (a) is proving.

- **(a, optional) the bound $\max_{\|T\|_\ast=1}\operatorname{trace}(G^\top T)\le\operatorname{trace}\Sigma$:** substitute $G=U\Sigma V^\top$, use the cyclic property $\operatorname{trace}(ABC)=\operatorname{trace}(CAB)$ to reduce $\operatorname{trace}(G^\top T)=\operatorname{trace}(\Sigma\,\tilde{U})$ for some orthogonal-ish $\tilde U=V^\top T U$, then bound the diagonal using "rows/columns of an orthogonal matrix are unit vectors" so each diagonal entry $\le1$. This shows the nuclear norm is the dual and identifies when equality holds.
- **(b) the actual optimizer:** the maximizing $T$ aligns the singular *directions* of $T$ with those of $G$ while flattening its singular values to all-ones on the relevant subspace, i.e. $T^\star=UV^\top$ (the "orthogonal polar factor" / semi-orthogonalization of $G$). Combine with the master identity to write $\Delta W^\star=-\dfrac{\|G\|^\dagger}{\lambda}UV^\top=-\dfrac{\operatorname{trace}\Sigma}{\lambda}UV^\top$. This is the **muon / spectral update**: it throws away the singular-value magnitudes of the gradient and keeps only its "rotation." Express purely in $U,V,\Sigma,\lambda$ as asked.

**Why this matters for §B.** Problem 9's update rule $\operatorname{sign}(\nabla)/\|\operatorname{sign}(\nabla)\|_\ast$ is a cheap stand-in for exactly this $UV^\top$ operation — this is the thread connecting the two graded sections.

### Problem 5 requires (BONUS, 0 pt): deriving $G$ and $\lambda$ for a square-loss majorizer

Expand $L(W+\Delta W)$ using $\|a+b\|_2^2=\|a\|_2^2+2a^\top b+\|b\|_2^2$ with $a=y^{(i)}-Wx^{(i)}$ and $b=-\Delta W x^{(i)}$. The cross term gives the *gradient* $G$ (identify it as an average of outer products $\text{residual}\cdot x^{(i)\top}$); the quadratic term is bounded above using $\|Ab\|_2\le\|A\|_\ast\|b\|_2$ and the normalization $\|x^{(i)}\|_2=\sqrt{d_\text{in}}$, which turns $\|\Delta W x^{(i)}\|_2^2$ into $\|\Delta W\|_\ast^2\cdot d_\text{in}$ and fixes $\lambda$. This is a *majorization–minimization* view: the spectral steepest-descent step is the minimizer of this upper bound, i.e. it's principled, not pulled from a hat.

---

## §B · Hyperparameter Transfer (6 pt) — Problems 6–11

Big picture (μP / spectral-scaling): if you initialize and update weights so that **each layer's spectral norm and the size of its updates scale correctly with width**, then the optimal learning rate found on a narrow network transfers to a wide one. Problems 6–8 build the measurement tools (spectral norm via random matrices and power iteration); Problem 9 is the payoff (learning-rate transfer); Problem 11 interprets weight decay spectrally. These are notebook cells — code + short written findings.

### Problem 6 requires: empirical scaling law for $\|\cdot\|_\ast$ of a Gaussian matrix (1 pt)

Sample $d\times d$ iid $\mathcal N(0,1)$, compute $\|A\|_\ast$ (largest singular value) for a range of $d$, and fit $\alpha d^\beta$. Fit in **log space**: $\log\|A\|_\ast\approx\log\alpha+\beta\log d$, so a linear regression on $(\log d,\log\|A\|_\ast)$ gives $\beta$ as the slope. You are rediscovering the Bai–Yin / Marchenko–Pastur edge (top singular value of a $d\times d$ Gaussian grows like $\sim2\sqrt d$), so expect $\beta\approx1/2$. Report the fitted $\alpha,\beta$.

```python
import torch
ds = [16, 32, 64, 128, 256, 512, 1024]
snorms = []
for d in ds:
    A = torch.randn(d, d)
    snorms.append(torch.linalg.matrix_norm(A, ord=2).item())  # spectral norm = σ₁
# fit log(snorm) = log(alpha) + beta*log(d)  → np.polyfit(log ds, log snorms, 1)
```

### Problem 7 requires: spectral norm of a random orthogonal matrix (1 pt)

Sample a random orthogonal $Q$ (e.g. QR of a Gaussian, or `torch.nn.init.orthogonal_`) and compute $\|Q\|_\ast$ for several $d$. **Why the answer is width-independent:** orthogonal matrices have *all* singular values equal to 1, so $\sigma_1\equiv1$ regardless of $d$. Contrast this with Problem 6 (grows like $\sqrt d$) — this is precisely why μP-style init uses **semi-orthogonal** matrices scaled by $\sqrt{d_k/d_{k-1}}$ (Problem 9): to control the spectral norm independent of width.

```python
d = 512
Q = torch.empty(d, d)
torch.nn.init.orthogonal_(Q)
print(torch.linalg.matrix_norm(Q, ord=2))  # ≈ 1 for every d
```

### Problem 8 requires: power iteration for the top singular value (1 pt)

**Math.** $\|A\|_\ast=\sqrt{\lambda_{\max}(AA^\top)}$. Power iteration multiplies a random $v$ by $(AA^\top)^k$; in the eigenbasis $AA^\top=\sum_i\lambda_i u_i u_i^\top$, so $(AA^\top)^k v=\sum_i\lambda_i^k(u_i^\top v)u_i$ — the largest $\lambda$ dominates geometrically as $k$ grows, so $u$ tilts toward the top eigenvector. The Rayleigh quotient $\|AA^\top u\|_2/\|u\|_2$ then reads off $\lambda_{\max}$. **This is the eigendecomposition/matrix-powers lesson (Day 1) made computational.**

Run the provided routine on a $2000\times2000$ matrix and compare to `torch.linalg.matrix_norm(A, 2)`. Expected findings: with a *finite* number of iterations power iteration **under-estimates** (it hasn't fully converged to $\sigma_1$, and any component orthogonal to $u_1$ only shrinks the quotient); but it is **much faster** than a full SVD because it only needs matrix–vector products ($O(d^2)$ per step) versus $O(d^3)$ for a full decomposition. Report both the value comparison and the runtime comparison.

### Problem 9 requires: learning-rate transfer for sign GD across width (2 pt) — the graded core

**What to implement** (two marked blocks):

*Init each layer:* sample a semi-orthogonal $M_k\in\mathbb R^{d_k\times d_{k-1}}$ and set $W_k=\sqrt{d_k/d_{k-1}}\,M_k$.

```python
# initialisation block
M = torch.empty(d_k, d_km1)
torch.nn.init.orthogonal_(M)           # semi-orthogonal (works for non-square)
W_k = (d_k / d_km1) ** 0.5 * M
```

*Update each layer:* normalize the sign of the gradient by its spectral norm and rescale by $\sqrt{d_k/d_{k-1}}$:

$$W_k \leftarrow W_k-\eta\,\sqrt{\tfrac{d_k}{d_{k-1}}}\cdot\frac{\operatorname{sign}(\nabla_{W_k}L)}{\|\operatorname{sign}(\nabla_{W_k}L)\|_\ast}.$$

```python
# update block (inside no_grad)
g = p.grad
s = torch.sign(g)
snorm = torch.linalg.matrix_norm(s, ord=2)     # or the notebook's power-iteration estimate
scale = (d_k / d_km1) ** 0.5
p.add_(-lr * scale * s / snorm)
```

**Why each piece is there.** The $\sqrt{d_k/d_{k-1}}$ factors keep each layer's spectral norm $O(1)$ as width grows (ties back to Problems 6–7); dividing by $\|\operatorname{sign}(\nabla)\|_\ast$ makes the *update's* spectral norm width-independent (a cheap version of the $UV^\top$ answer from Problem 4). Together they make the loss landscape's curvature scale-invariant in width, so the best $\eta$ from the small sweep lands near-optimal at large width. Run the small-width LR sweep, take the argmax LR, run the large-width training with it, and report whether the optimum transferred. Paste the **two modified blocks** and your finding.

### Problem 10 requires (optional, 0 pt): a better normalizer

Problem 4 says the true spectral steepest-descent direction is $UV^\top$ (semi-orthogonalize the gradient), not $\operatorname{sign}(\nabla)/\|\operatorname{sign}(\nabla)\|_\ast$. Propose replacing the sign-and-divide with an actual orthogonalization of $\nabla_{W_k}L$ (e.g. Newton–Schulz iteration to approximate $UV^\top$ without a full SVD — this is the muon optimizer idea). Explain why it better matches the geometry.

### Problem 11 requires: what weight decay does spectrally (1 pt)

Write $W=\sum_i\sigma_i u_i v_i^\top$. Multiplying $W\to 0.999\,W$ pushes the scalar in front of each rank-1 term: $\sigma_i\to0.999\,\sigma_i$ — **all singular values shrink by the same factor**, but $u_i,v_i$ are unchanged (they're unit vectors defining directions, and scaling a matrix doesn't rotate its singular subspaces). So weight decay is an isotropic contraction of the spectrum that **preserves the singular directions**. State both effects explicitly.

---

## §C · Architecture & Inductive Bias (5 pt) — Problems 12–14

Theme: a CNN's weight-sharing + locality is an **inductive bias** matched to images, so it generalizes better than an MLP of comparable/greater depth on CIFAR-100. These are colab FIXME blocks.

### Problem 12 requires: building a CNN feature extractor + MLP head (2 pt)

You need the standard conv block anatomy: `Conv2d(in_ch, out_ch, kernel_size, stride, padding)` → nonlinearity (`ReLU`) → optional `BatchNorm2d` → spatial downsampling (`MaxPool2d` or strided conv). Key shape reasoning: track how $H\times W\times C$ evolves; a conv with padding $p=(k-1)/2$ preserves spatial size, pooling halves it. Before the MLP head, flatten (or global-average-pool) the final feature map. Understand **receptive field** growth and why channels typically increase as spatial dims shrink.

```python
import torch.nn as nn
def make_cnn(num_classes=100):
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32×32→16×16
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # →8×8
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # →4×4
        nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),# →2×2
        nn.Flatten(),
        nn.Linear(128*2*2, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, num_classes),
    )   # exact channels/sizes per the notebook's FIXME spec — this is the shape logic, not the answer
```

### Problem 13 requires: training loop + evaluation (1 pt)

Standard supervised loop: `model.train()`, zero grads, forward, `CrossEntropyLoss` (expects raw logits + integer labels — no softmax in the model), backward, `optimizer.step()`. Eval: `model.eval()` + `torch.no_grad()`, accumulate correct predictions via `logits.argmax(1)`. Note the hint: CNN takes image tensors `(B,3,32,32)` directly — **do not flatten** the input (unlike the HW1 MLP).

### Problem 14 requires: comparing MLP(w128,d3) vs MLP(w128,d7) vs 4-conv CNN over 20 epochs (2 pt)

- **(a)** one matplotlib figure, validation accuracy vs epoch, three curves.
- **(b)** interpret: the CNN should win because weight sharing + locality drastically reduce parameters and encode translation equivariance, a good prior for natural images; a deeper MLP (d7) does **not** reliably beat the shallow one and may overfit or optimize worse — more MLP depth ≠ better generalization without the right inductive bias. Argue why arbitrarily deeper MLPs won't help (parameter blow-up, no spatial prior, optimization difficulty).

---

## §D · Graph Neural Networks (10 pt) — Problems 15–18

Framework: an MP-GNN is defined by **AGGREGATE** (combine neighbor features permutation-invariantly), **UPDATE** (transform a node's own + aggregated feature), and **READOUT** (pool all nodes → graph-level scalar). The section probes the *representational power* of different aggregators, closely tied to the Weisfeiler–Lehman (WL) test.

### Problem 15 requires: universal aggregation constructions (4 pt)

The generic aggregator $\text{generic}_{f,g}(\{h_u\})=f\big(\sum_{u\in N(v)}g(h_u)\big)$ — a **Deep Sets** style decomposition (sum is the only permutation-invariant primitive; $f,g$ do the rest).

- **(a) mean:** the obstacle is that $f$ does *not* know the degree $|N(v)|$. Trick: make $g$ output an extra "counter" coordinate that is constant $1$, so the sum carries $\big(\sum g(h_u),\ |N(v)|\big)$; then $f$ divides the feature part by the count. Write $g(h)=[h,\,1]$ and $f([s,c])=s/c$.
- **(b) max (approx):** use the $\ell_\infty$/log-sum-exp idea. For nonnegative entries, $\max_i x_i\approx\frac1\beta\log\sum_i e^{\beta x_i}$ for large $\beta$; so let $g(h)=e^{\beta h}$ (coordinate-wise) and $f(s)=\frac1\beta\log s$. Heed the hint about **absolute values** — raw $\max$ vs $\max|\cdot|$ differ; handle signs (e.g. shift to nonnegative or track sign separately) so you approximate the true coordinate-wise max, not the max-magnitude.

### Problem 16 requires: which graph problems each aggregator can solve (4 pt)

This is WL-expressivity reasoning. The engine: an MP-GNN can only distinguish nodes/graphs whose **neighborhood computation trees** differ; if two graphs are WL-indistinguishable under a given aggregator, no choice of UPDATE/READOUT separates them.

- **(a) count nodes, identical features:** with a count-preserving aggregator (generic/mean-with-counter) and sum-READOUT you can total the nodes; reason about whether **max** can — max is idempotent and loses cardinality, so think about whether it can count at all.
- **(b) max distance to $v$ (BFS), $v$ zeroed / others ones:** the max aggregator computes shortest-path distances by repeatedly propagating a "reached" signal (like Bellman–Ford layers); argue how many message-passing rounds encode $\max_u d(u,v)$.
- **(c) triangle count, identical features (2 pt):** the crux — construct two graphs whose nodes pairwise share identical neighborhood trees (WL-equivalent) but have **different triangle counts** (classic example: a 6-cycle $C_6$ vs two disjoint triangles $2\,C_3$ — both 2-regular, locally indistinguishable trees, but 0 vs 2 triangles). Conclude MP-GNNs *cannot* count triangles under these aggregators. Present the two graphs and the tree-equivalence argument; don't just assert.
- **(d) BONUS mean:** redo (a)–(c) noting mean normalizes away degree information, changing which problems are solvable.

### Problem 17 requires: chirality and generic aggregation (1 pt)

Chiral molecules are mirror images with the *same* graph connectivity and node/edge features — they differ only by 3-D orientation (handedness). Since $\text{generic}_{f,g}$ depends only on the multiset of neighbor features (permutation-invariant, no geometric orientation), argue it produces **identical** representations for both enantiomers → it **cannot** distinguish them. This motivates geometric/equivariant GNNs that use coordinates (ties to Plücker/3-D ideas seen later in HW6).

### Problem 18 requires: train a GNN for molecular water solubility (1 pt)

Regression on graphs: nodes = atoms, edges = bonds; message passing → READOUT (graph pooling) → scalar solubility (mg/L), MSE loss. Just run the provided implementation, plot validation loss vs epoch. **Logistics warning: the dependency install (`torch-geometric` / RDKit stack) can take ~45 min — start early.** Understand the pipeline (SMILES → graph featurization → batched `Data` objects → GNN), even though you don't write it.

---

## Study order for the window (due Aug 11)

1. Nail the **master identity** (P2) — every steepest-descent problem is a substitution into it.
2. Do P1 → P3 together (dual norm feeds directly into the explicit updates).
3. P4 spectral case + P9 update rule share the "semi-orthogonalize the gradient" idea — study them as a pair.
4. §C and §D are mostly coding + short arguments; the GNN dependency install (P18) is the long pole — kick it off early.

*Reference docs in your vault that overlap: [[Gradient Descend]] (steepest-descent geometry), and the attention/transformer deep dives for the Day 4–5 lessons.*
