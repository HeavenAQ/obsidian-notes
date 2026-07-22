---
base: "[[01.2 DL Daily Lessons — Step-by-Step + Quiz.base]]"
Quiz score: ""
"HW #": 2
Studied: false
Quiz taken: false
Day type: Homework-Day
Date: 2026-07-22
Piece count: 5
---
> ⚠️ **Schedule note (Homework Day 1 of 2 — HW2 window):** The five HW2 lesson days are complete — Jul 17 (Eigenvalues I), Jul 18 (Positive Definiteness / Hessian / Lagrange), Jul 19 (Least Squares / Projection), Jul 20 (Attention Theory), Jul 21 (Deep Sequence Modeling). Today and tomorrow (Jul 22–23) are **homework days**: no new lesson. Below is a **whole-window consolidation** — five "formula + why + pitfall" cards that restate exactly the machinery HW2's four graded sections need — followed by a **cumulative quiz** that mixes every lesson plus a spaced-repetition callback to the HW1 window. **The quiz tests lesson content only — it contains no HW2 answers and writes no graded notebook cell for you.** Today's suggested pass: §A Steepest Descent theory (Problems 1–4) — it is the densest math and the backbone of §B; save §C/§D coding for tomorrow, and kick off the §D `torch-geometric`/RDKit install now (it can take ~45 min).
>
> 🗒️ **Quiz backlog nudge (no patch-up today):** the re-teach loop only fires on a *recorded* low score, and **none** of the HW2 quizzes are graded yet — Jul 17, 18, 19, 20, and yesterday's **Jul 21 (Deep Sequence Modeling)** are all still `Quiz taken: false`. Nothing auto-queued. But these are your last two days before HW2, so grade at least the **Jul 20 (attention)** and **Jul 21 (RNN/BPTT)** sets today — the spectral-radius argument in Piece 1 below is the same eigenvalue machinery you'll be graded on in §B power iteration.

# 🎯 Today's focus

Convert five lessons into a working toolkit for HW2. Two weeks of linear algebra and sequence modeling feed four graded sections: **§A Steepest Descent** (dual norms → the master identity → the spectral/muon update), **§B Hyperparameter Transfer** (spectral norm, power iteration, width-scaling), **§C Architecture & Inductive Bias** (CNN vs. MLP), and **§D Graph Neural Networks** (permutation-invariant aggregation, WL expressivity). Each card below is *formula + why it is written that way + the pitfall that costs points*. Read the cards, grade your open quizzes, then start §A.

# 🧭 Consolidation pieces

## Piece 1 — Eigendecomposition, matrix powers & power iteration (Jul 17 → §B Problems 6–9)

Source: **Jul 17 (Eigenvalues I)**, made computational in **HW2 §B**; the same machinery also governs RNN BPTT (Jul 21, Piece 3).

For a diagonalizable $A$, eigendecomposition and its power are

$$A = Q\Lambda Q^{-1},\qquad A^{k}=Q\,\Lambda^{k}\,Q^{-1},$$

so raising a matrix to a power raises **each eigenvalue** to that power while leaving the eigenvectors fixed. For **symmetric** $A$ the eigenbasis is orthogonal ($Q^{-1}=Q^\top$). This is the entire engine behind **power iteration** for the top singular value (§B Problem 8): since $\|A\|_\ast=\sqrt{\lambda_{\max}(AA^\top)}$ and $AA^\top=\sum_i\lambda_i u_i u_i^\top$,

$$(AA^\top)^{k}v=\sum_i \lambda_i^{k}\,(u_i^\top v)\,u_i \xrightarrow{k\to\infty} \text{(direction of } u_1),$$

because the largest $|\lambda|$ dominates the sum geometrically. The Rayleigh quotient $\|AA^\top u\|_2/\|u\|_2$ then reads off $\lambda_{\max}$.

**Why formulated this way.** Power iteration needs only matrix–vector products ($O(d^2)$ per step) instead of a full $O(d^3)$ SVD — that is why §B uses it on a $2000\times2000$ matrix.

**Pitfall.** With finite $k$ power iteration **under-estimates** $\sigma_1$: any component orthogonal to $u_1$ only shrinks the quotient, so it converges *from below*. Don't report it as exact — compare against `torch.linalg.matrix_norm(A, 2)`.

```python
import torch
def power_iteration_sigma1(A, iters=50):
    v = torch.randn(A.shape[1])
    for _ in range(iters):
        v = A.T @ (A @ v)          # apply (AᵀA); top eigval dominates as k grows
        v = v / v.norm()
    return (A @ v).norm().item()   # Rayleigh: ||A v|| with v ≈ top right singular vector
```

**You've got this piece when you can** write $A^k=Q\Lambda^kQ^{-1}$, explain why $(AA^\top)^k v$ tilts toward the top eigenvector, and say why finite-step power iteration under-estimates $\sigma_1$ yet beats a full SVD in cost.

## Piece 2 — Dual norms & the steepest-descent master identity (Jul 18 Lagrange → §A Problems 1–3)

Source: **HW2 §A**, using the constrained-optimization view from **Jul 18 (Lagrange multipliers)**.

The **dual norm** is $\|a\|^\dagger=\max_{\|b\|=1}a^\top b$ — the largest a linear functional gets on the unit ball. The two facts §A leans on: $\ell_p$ is dual to $\ell_q$ with $\tfrac1p+\tfrac1q=1$ (so $\ell_2$ self-dual, $\ell_1\leftrightarrow\ell_\infty$), driven by **Hölder's inequality**, which is *tight*. The **master identity** every §A problem substitutes into:

$$\arg\min_{\Delta w}\Big(g^\top\Delta w+\tfrac{\lambda}{2}\|\Delta w\|^2\Big)=-\frac{\|g\|^\dagger}{\lambda}\cdot\arg\max_{\|t\|=1}g^\top t.$$

**Why formulated this way.** Split $\Delta w=r\,t$ into magnitude $r=\|\Delta w\|$ and unit direction $t$. The objective separates into $r\,(g^\top t)+\tfrac\lambda2 r^2$; minimizing over $t$ is a dual-norm computation (giving direction $t^\star=-\arg\max g^\top t$ and value $-r\|g\|^\dagger$), then minimizing the 1-D quadratic in $r$ gives step length $r^\star=\|g\|^\dagger/\lambda$. Decoupling length from direction is the standard trick for norm-constrained linear optimization — and choosing the norm is *choosing the geometry in which "steepest" is measured*: $\ell_2\Rightarrow$ vanilla GD, $\ell_\infty\Rightarrow$ sign-GD (Adam with $\beta,\epsilon\to0$).

**Pitfall.** The minus sign is the whole point — it is what makes this *descent*. And $\|g\|^\dagger$ is the dual norm, **not** $\|g\|$; under $\ell_\infty$ the step length is $\|g\|_1$, not $\|g\|_\infty$.

**You've got this piece when you can** re-derive the master identity by the $r$-$t$ split, state the dual pairs for $\ell_2$ and $\ell_\infty$, and recover vanilla GD and sign-GD as the two special cases.

## Piece 3 — SVD, the spectral norm & the muon/spectral update (Jul 17 → §A Problem 4, §B Problem 11)

Source: **HW2 §A Problem 4** and **§B Problem 11**, bridging from the eigen-material of **Jul 17**.

Any $G=U\Sigma V^\top$ (SVD); the **spectral norm** $\|G\|_\ast=\sigma_1$ is dual (under the Frobenius inner product $\langle A,B\rangle=\operatorname{trace}(A^\top B)$) to the **nuclear norm** $\sum_i\sigma_i$. Plugging the spectral norm into the matrix master identity gives the **muon / spectral update**:

$$T^\star=\arg\max_{\|T\|_\ast=1}\operatorname{trace}(G^\top T)=UV^\top,\qquad \Delta W^\star=-\frac{\operatorname{trace}\Sigma}{\lambda}\,UV^\top.$$

**Why formulated this way.** $UV^\top$ is $G$ with **all singular values flattened to 1** — the "orthogonal polar factor." The update keeps the gradient's *rotation* (singular directions) and throws away its *magnitudes*. §B's cheap stand-in $\operatorname{sign}(\nabla)/\|\operatorname{sign}(\nabla)\|_\ast$ approximates exactly this. Spectrally, **weight decay** $W\to0.999\,W$ multiplies every $\sigma_i$ by the same factor while leaving $u_i,v_i$ fixed — an isotropic spectral contraction that *preserves singular directions*.

**Pitfall.** $UV^\top$ uses $U$ and $V^\top$ but **not** $\Sigma$ — dropping $\Sigma$ is deliberate, not an omission. And width-scaling (§B) needs semi-orthogonal init $W_k=\sqrt{d_k/d_{k-1}}\,M_k$ precisely because a raw Gaussian's $\sigma_1$ grows like $\sqrt d$ (Bai–Yin edge) while an orthogonal matrix keeps $\sigma_1\equiv1$.

```python
import torch
G = torch.randn(64, 32)
U, S, Vh = torch.linalg.svd(G, full_matrices=False)
muon_dir = U @ Vh                       # UVᵀ : singular values flattened to 1
print(torch.linalg.matrix_norm(muon_dir, 2))   # ≈ 1  (spectral norm of the direction)
```

**You've got this piece when you can** state that $\|G\|_\ast=\sigma_1$ is dual to the nuclear norm, identify $T^\star=UV^\top$ as the semi-orthogonalized gradient, and describe weight decay as uniform spectral shrinkage with fixed singular directions.

## Piece 4 — Loss-landscape geometry: Hessian, Taylor & least-squares projection (Jul 18–19)

Source: **Jul 18 (PD / Hessian / Taylor)** and **Jul 19 (Least Squares / Projection)** — the geometry behind why §A's quadratic penalty and §B's generalization framing make sense.

Second-order Taylor at $w_0$: $L(w)\approx L(w_0)+g^\top(w-w_0)+\tfrac12(w-w_0)^\top H(w-w_0)$. The **Hessian** $H$ decides local shape: $H\succ0$ (all eigenvalues $>0$) $\Rightarrow$ strict local min (a bowl); indefinite $\Rightarrow$ saddle. The $\tfrac\lambda2\|\Delta w\|^2$ penalty in the master identity is exactly a *trust-region* stand-in for this curvature. The **least-squares / normal equation** picture,

$$\hat\beta=(X^\top X)^{-1}X^\top y,\qquad \hat y = X\hat\beta = \underbrace{X(X^\top X)^{-1}X^\top}_{P}\,y,$$

says the fit $\hat y$ is the **orthogonal projection** of $y$ onto the column space of $X$; $P$ is symmetric, idempotent ($P^2=P$), with eigenvalues $0/1$. This is the linear backbone under §A Problem 5's majorizer (a square loss expanded via $\|a+b\|_2^2$) and the generalization story of §B.

**Why formulated this way.** Projection minimizes $\|y-X\beta\|_2^2$ because the residual $y-\hat y$ is orthogonal to every column of $X$ — the normal equations $X^\top(y-X\beta)=0$ *are* that orthogonality.

**Pitfall.** $P$ projects onto the **column space** (span of features), not the row space; and PD is about **eigenvalues of $H$**, not the sign of individual entries — a matrix with positive entries can be indefinite.

**You've got this piece when you can** write the 2nd-order Taylor expansion, classify a critical point from $H$'s eigenvalues, and derive the projection matrix $P$ from the normal equation and state its idempotence.

## Piece 5 — Message passing unifies attention, RNNs, CNNs & GNNs (Jul 20–21 → §C, §D)

Source: **Jul 20 (Attention)**, **Jul 21 (Sequence Modeling)**, feeding **HW2 §C (inductive bias)** and **§D (GNNs)**.

Self-attention $\hat X=\operatorname{softmax}\!\big(QK^\top/\sqrt{d_k}+M\big)V$ is a **permutation-invariant weighted aggregation** over the token set — message passing on a *complete graph* with learned, softmax-normalized edge weights. An RNN is message passing on a *path graph* ($O(T)$ path length, serial). A **GNN** (§D) generalizes the primitive: $\text{generic}_{f,g}(\{h_u\})=f\big(\sum_{u\in N(v)}g(h_u)\big)$ — a **Deep Sets** decomposition where *sum* is the only permutation-invariant primitive and $f,g$ do the rest (mean = sum-with-a-counter coordinate; max ≈ log-sum-exp). A **CNN** (§C) is message passing on a *grid* with weight sharing + locality — the inductive bias matched to images.

**Why formulated this way.** Expressivity is bounded by the **Weisfeiler–Lehman** test: an MP-GNN cannot distinguish graphs with identical neighborhood computation trees. Classic consequence — $C_6$ vs. $2\,C_3$ are WL-indistinguishable, so **MP-GNNs cannot count triangles** (§D Problem 16c). The right prior beats raw depth: a CNN beats a deeper MLP on CIFAR-100 (§C) because weight sharing encodes translation equivariance, whereas extra MLP depth just adds parameters and overfits.

**Pitfall.** A **mean** aggregator discards degree/count information (it normalizes it away), so it can *fail* problems a **sum** aggregator solves (e.g. counting nodes) — pick the aggregator to match what must be preserved.

```python
import torch
# Deep-Sets mean via a counter coordinate (§D P15a): g(h)=[h,1], then divide by the count
def mean_aggregate(neighbor_feats):          # (num_neighbors, d)
    g = torch.cat([neighbor_feats, torch.ones(len(neighbor_feats), 1)], dim=1)
    s = g.sum(0)                             # [sum_h, count]
    return s[:-1] / s[-1]                    # f([sum,count]) = sum / count
```

**You've got this piece when you can** frame attention/RNN/CNN/GNN as message passing on complete/path/grid/general graphs, state the Deep-Sets sum decomposition, and give the $C_6$ vs. $2C_3$ WL argument for why MP-GNNs can't count triangles.

# 📝 Review quiz

1. **(Derivation)** State $A^k=Q\Lambda^kQ^{-1}$ and use it to explain why power iteration $(AA^\top)^k v$ converges to the top singular direction. Why does a finite number of iterations *under*-estimate $\sigma_1$, and why is the method still preferred over a full SVD?
2. **(Derivation)** Prove the steepest-descent master identity by splitting $\Delta w=r\,t$. Give both factors (step length and unit direction) and explain where the minus sign comes from.
3. **(Concept)** Give the dual norm of $\ell_2$ and of $\ell_\infty$. Show how the master identity recovers vanilla gradient descent under $\ell_2$ and **sign** gradient descent under $\ell_\infty$, and state the step length in each case.
4. **(Derivation / concept)** For $G=U\Sigma V^\top$, what is the maximizer $T^\star=\arg\max_{\|T\|_\ast=1}\operatorname{trace}(G^\top T)$, and what is the resulting spectral (muon) update $\Delta W^\star$? Which factor of the SVD does it *discard*, and what does that mean geometrically?
5. **(Concept)** Why does semi-orthogonal init $W_k=\sqrt{d_k/d_{k-1}}M_k$ control a layer's spectral norm across width, whereas an i.i.d. Gaussian's $\sigma_1$ does not? What does weight decay $W\to0.999W$ do to the singular values and singular directions?
6. **(Derivation)** Write the second-order Taylor expansion of $L$ and state how the Hessian's eigenvalues classify a critical point. Then derive the least-squares projection matrix $P$ from the normal equation and give its two defining algebraic properties.
7. **(Synthesis)** Frame self-attention, a vanilla RNN, a CNN, and an MP-GNN each as message passing — on *what graph* in each case? State the Deep-Sets aggregation formula and why *sum* (not mean or max) is the safe permutation-invariant primitive.
8. **(Spaced repetition — HW1 window)** From the Jul 2–10 material: for a ReLU network, how does the number of linear regions scale with **width** vs. **depth**, and what does that imply about depth efficiency? Where are the network's non-differentiable points?

> [!note]- 🔑 Answer key (click to reveal)
> **1.** $A^k=Q\Lambda^kQ^{-1}$ raises each eigenvalue to the $k$-th power while fixing eigenvectors. Since $AA^\top=\sum_i\lambda_i u_iu_i^\top$, we get $(AA^\top)^kv=\sum_i\lambda_i^k(u_i^\top v)u_i$; the largest $|\lambda|$ dominates geometrically, so $v$ tilts toward $u_1$ and the Rayleigh quotient $\|AA^\top u\|/\|u\|\to\lambda_{\max}=\sigma_1^2$. Finite $k$ **under-estimates** because any component of $v$ orthogonal to $u_1$ only lowers the quotient — convergence is from below. It still beats SVD because it needs only matrix–vector products, $O(d^2)$ per step vs. $O(d^3)$ for a full decomposition.
>
> **2.** Split $\Delta w=r\,t$ with $r=\|\Delta w\|\ge0$, $\|t\|=1$. The objective becomes $r(g^\top t)+\tfrac\lambda2 r^2$. (i) Over $t$: minimizing $r\,g^\top t$ makes $g^\top t$ most negative, so $t^\star=-\arg\max_{\|t\|=1}g^\top t$, value $-r\|g\|^\dagger$. (ii) Over $r$: $\min_{r\ge0}(-r\|g\|^\dagger+\tfrac\lambda2 r^2)$, derivative $-\|g\|^\dagger+\lambda r=0\Rightarrow r^\star=\|g\|^\dagger/\lambda$. Combine: $\Delta w^\star=-\tfrac{\|g\|^\dagger}{\lambda}\arg\max_{\|t\|=1}g^\top t$. The minus sign comes from step (i) — we move *opposite* the best-aligned direction, which is what makes it descent.
>
> **3.** $\ell_2$ is self-dual ($\|g\|_2^\dagger=\|g\|_2$); $\ell_\infty$ is dual to $\ell_1$ ($\|g\|_\infty^\dagger=\|g\|_1$). Under $\ell_2$: direction $g/\|g\|_2$, and the two $\|g\|_2$ factors give $\Delta w\propto -g/\lambda$ — **vanilla GD**, step length $\|g\|_2/\lambda$. Under $\ell_\infty$: the unit-$\ell_\infty$ maximizing direction is $\operatorname{sign}(g)$, so $\Delta w\propto-\operatorname{sign}(g)$ — **sign GD**, step length $\|g\|_1/\lambda$. (With $\beta_1=\beta_2=\epsilon=0$, Adam's $-m/\sqrt v=-g/\sqrt{g^2}=-\operatorname{sign}(g)$ coincides with the $\ell_\infty$ case.)
>
> **4.** $T^\star=UV^\top$ (the orthogonal polar factor / semi-orthogonalized gradient), giving $\Delta W^\star=-\tfrac{\operatorname{trace}\Sigma}{\lambda}UV^\top$ where $\operatorname{trace}\Sigma=\sum_i\sigma_i$ is the nuclear norm (= dual of the spectral norm). It **discards $\Sigma$** — the singular *values* — keeping only $U,V^\top$, i.e. the gradient's rotation/directions with all singular values flattened to 1. Geometrically it moves equally along every gradient singular direction regardless of magnitude.
>
> **5.** A $d\times d$ i.i.d. Gaussian has $\sigma_1\sim2\sqrt d$ (Bai–Yin edge), so its spectral norm **grows with width**; an orthogonal/semi-orthogonal matrix has *all* singular values equal to 1, so scaling by $\sqrt{d_k/d_{k-1}}$ sets a controlled, width-consistent spectral norm. Weight decay $W\to0.999W$ multiplies **every** $\sigma_i$ by $0.999$ (uniform isotropic shrinkage of the spectrum) while leaving $u_i,v_i$ **unchanged** — scaling a matrix does not rotate its singular subspaces.
>
> **6.** $L(w)\approx L(w_0)+g^\top(w-w_0)+\tfrac12(w-w_0)^\top H(w-w_0)$. If all eigenvalues of $H>0$ ($H\succ0$) → strict local min; all $<0$ → local max; mixed signs → saddle; zeros → degenerate. Least squares: minimizing $\|y-X\beta\|_2^2$ gives normal equations $X^\top(y-X\beta)=0\Rightarrow\hat\beta=(X^\top X)^{-1}X^\top y$, so $\hat y=X\hat\beta=Py$ with $P=X(X^\top X)^{-1}X^\top$. $P$ is **symmetric** and **idempotent** ($P^2=P$), eigenvalues $0/1$ — the orthogonal projector onto $\operatorname{col}(X)$; the residual $y-\hat y\perp\operatorname{col}(X)$.
>
> **7.** Self-attention = message passing on a **complete graph** (learned softmax edge weights, $O(1)$ path length); vanilla RNN = **path/chain graph** (each node reads only its predecessor, $O(T)$ path); CNN = **grid graph** with shared local weights (translation equivariance); MP-GNN = **general graph** over $N(v)$. Deep-Sets aggregation: $\text{generic}_{f,g}(\{h_u\})=f\big(\sum_{u\in N(v)}g(h_u)\big)$. **Sum** is the safe primitive because it is permutation-invariant *and* preserves cardinality/degree information; **mean** normalizes degree away and **max** is idempotent (loses counts), so each fails problems that need the discarded information — $f,g$ can rebuild mean/max from a sum but not vice-versa.
>
> **8.** (HW1 callback) The number of linear regions grows **linearly in width** but **multiplicatively — exponentially — in depth** (Montúfar et al. 2014); that gap is *depth efficiency*: deep nets carve exponentially more regions per parameter than shallow-wide ones. The non-differentiable points are exactly the **region boundaries (kinks)** where some neuron's ReLU switches on/off; inside any fixed activation pattern the network is a single affine map, so it is continuous and piecewise-linear (with $\tanh$ instead there are no kinks — it is $C^\infty$).
