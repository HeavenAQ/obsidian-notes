---
base: "[[01.2 DL Daily Lessons — Step-by-Step + Quiz.base]]"
Quiz score: ""
"HW #": 2
Studied: false
Quiz taken: false
Day type: Lesson-Day
Date: 2026-07-18
Piece count: 5
---
> 🗒️ **Quiz backlog nudge:** yesterday's **Jul 17 (Eigenvalues I)** quiz is still ungraded (`Quiz taken: false`, no score), as are the Jul 12/13/14/15 HW1 sets. Nothing to patch up today — the re-teach loop only fires on a *recorded* low score. If you want spaced-repetition credit, grade at least the Jul 17 set (~10 min) before starting; several of today's questions build straight on it.

# 🎯 Today's goal

Yesterday you built the eigenvalue toolkit ($A=Q\Lambda Q^\top$, spectral theorem, power iteration). Today you turn it into the **local geometry of a loss landscape**: expand a loss to second order (Taylor), read the **Hessian** $H$ as the curvature matrix, and use its **eigenvalues** to classify a critical point (min / max / saddle) via **positive definiteness**. This is the backbone of HW2's optimization thread — the spectral steepest-descent majorizer in §A (Problem 5) is literally a second-order upper bound, and the condition number $\lambda_{\max}/\lambda_{\min}$ of the Hessian is *why* gradient descent stalls and why μP / spectral scaling in §B matters. We close with **Lagrange multipliers**, the exact machinery behind HW2's master identity $\arg\min_{\Delta w}\big(g^\top\Delta w+\tfrac\lambda2\|\Delta w\|^2\big)$ (Problem 2) — norm-constrained descent *is* a constrained-optimization problem.

# 🧩 Pieces

## Piece 1 — Second-order Taylor expansion: the local quadratic model (~30 min)

*Source: Udemy Linear Algebra → "Taylor expansion / quadratic forms." Feeds [[Homework 2]] §A (Problem 5's majorizer) and the whole optimization thread. Deep-dive: [[Gradient Descend]] §local models.*

Near a point $w_0\in\mathbb{R}^n$, a smooth scalar loss $L$ is approximated by its **second-order Taylor expansion**:

$$
L(w_0+\Delta w)\;\approx\; L(w_0)\;+\;\nabla L(w_0)^\top \Delta w\;+\;\tfrac{1}{2}\,\Delta w^\top H(w_0)\,\Delta w,
$$

where $\nabla L\in\mathbb{R}^n$ is the **gradient** (first derivatives) and $H\in\mathbb{R}^{n\times n}$ is the **Hessian** (second derivatives, Piece 2). Read the three terms as: a constant (where you are), a **linear** slope (the tilt of the tangent plane), and a **quadratic** curvature correction. Gradient descent uses only the first two terms (a linear model + a step-size guess); Newton-type methods use all three.

The quadratic term $\tfrac12\Delta w^\top H\Delta w$ is a **quadratic form**. Because $H$ is symmetric (Piece 2), yesterday's spectral theorem applies: rotate into $H$'s eigenbasis $\Delta w=\sum_i c_i q_i$ and the form decouples into

$$
\tfrac12\,\Delta w^\top H\,\Delta w \;=\; \tfrac12\sum_{i=1}^n \lambda_i\, c_i^2 .
$$

So the eigenvalues $\lambda_i$ of $H$ are the **curvatures along the principal axes** $q_i$: positive $\lambda_i$ = the loss bowls upward along $q_i$, negative = it bends down, zero = flat. That single picture drives everything below.

**You've got this piece when you can** write the 2nd-order Taylor expansion, name each term, and explain why diagonalizing $H$ turns the quadratic form into a sum $\tfrac12\sum_i\lambda_i c_i^2$ of independent 1-D parabolas.

## Piece 2 — The Hessian: definition, symmetry, curvature (~30 min)

*Source: Udemy Linear Algebra → "the Hessian." Anchors the second-order optimality test (Piece 3) and ties to yesterday's symmetric-matrix material.*

The **Hessian** collects all second partial derivatives:

$$
H_{ij}\;=\;\frac{\partial^2 L}{\partial w_i\,\partial w_j}, \qquad H=\nabla^2 L .
$$

**Why it's symmetric.** For any twice-continuously-differentiable $L$, **Clairaut/Schwarz's theorem** says mixed partials commute, $\partial^2 L/\partial w_i\partial w_j=\partial^2 L/\partial w_j\partial w_i$, so $H=H^\top$. This is not a minor bookkeeping fact — it is *what licenses* applying the spectral theorem: $H$ has real eigenvalues and an orthonormal eigenbasis, so "curvature" is a well-defined set of $n$ real numbers along $n$ perpendicular axes. (Contrast: a general Jacobian is not symmetric and carries no such guarantee.)

Two directional readings you will reuse:

- The **directional second derivative** of $L$ along a unit vector $u$ is $u^\top H u$ (the curvature you feel moving along $u$). Maximized/minimized over $\|u\|=1$, this is exactly the Rayleigh quotient from yesterday, so $\lambda_{\max}(H)$ and $\lambda_{\min}(H)$ are the steepest and gentlest curvatures.
- For a pure quadratic $L(w)=\tfrac12 w^\top A w - b^\top w$ (with $A=A^\top$), the gradient is $\nabla L=Aw-b$ and the Hessian is exactly $H=A$ — constant everywhere. This is the model problem to keep in mind.

```python
import torch

def L(w):                                   # L(w) = ½ wᵀA w − bᵀw
    A = torch.tensor([[3., 1.], [1., 2.]])
    b = torch.tensor([1., 0.])
    return 0.5 * w @ A @ w - b @ w

w = torch.tensor([0.7, -0.4], requires_grad=True)
H = torch.autograd.functional.hessian(L, w)  # symmetric 2×2
print(H)                                     # [[3,1],[1,2]] — equals A
print("symmetric:", torch.allclose(H, H.T))
print("curvatures (eigvals):", torch.linalg.eigvalsh(H))  # ascending, real
```

**You've got this piece when you can** define $H_{ij}$, state why Clairaut's theorem makes $H$ symmetric (and why that matters), interpret $u^\top H u$ as directional curvature, and give $\nabla L$ and $H$ for a quadratic $\tfrac12 w^\top A w-b^\top w$.

## Piece 3 — Positive definiteness & the second-order optimality test (~35 min)

*Source: Udemy Linear Algebra → "positive definite matrices / definiteness tests." Directly HW2 §A (a local min requires PD curvature) and the PD condition on the spectral majorizer. Deep-dive pointer: [[Homework 2]] §0 (SVD/spectral bridge).*

A symmetric $A$ is **positive definite (PD)** if $x^\top A x>0$ for all $x\neq0$; **positive semidefinite (PSD)** if $x^\top A x\ge0$; **negative (semi)definite** with the inequalities flipped; **indefinite** if $x^\top Ax$ takes both signs. Equivalent tests you should be able to reach for:

1. **Eigenvalue test (the one to remember):** PD $\iff$ all $\lambda_i>0$; PSD $\iff$ all $\lambda_i\ge0$; indefinite $\iff$ mixed signs. This follows instantly from $x^\top Ax=\sum_i\lambda_i c_i^2$ (Piece 1).
2. **Sylvester's criterion:** PD $\iff$ every **leading principal minor** (the determinants of the top-left $k\times k$ blocks) is $>0$. Cheap for small matrices by hand.
3. **Cholesky:** $A$ is PD $\iff$ it has a Cholesky factorization $A=LL^\top$ with positive diagonal — the numerically preferred check in code (`torch.linalg.cholesky` throws if not PD).

**Why this classifies critical points.** At a critical point $\nabla L(w^\star)=0$, the Taylor expansion loses its linear term: $L(w^\star+\Delta w)\approx L(w^\star)+\tfrac12\Delta w^\top H\Delta w$. So the sign of the curvature form decides the shape:

- $H\succ0$ (PD, all $\lambda_i>0$) → strict **local minimum** (bowls up in every direction).
- $H\prec0$ (ND) → strict **local maximum**.
- $H$ **indefinite** (mixed-sign $\lambda_i$) → **saddle point** — up along some axes, down along others. In high-dimensional deep nets, *most* critical points are saddles, which is why saddle-escaping (and curvature-aware) methods matter.
- $H$ only PSD (some $\lambda_i=0$) → **inconclusive** at second order; need higher-order terms.

This is the multivariable generalization of the 1-D "second-derivative test" ($L''>0\Rightarrow$ min).

```python
import torch

def classify(H):
    ev = torch.linalg.eigvalsh(H)           # symmetric solver → real, ascending
    if (ev > 0).all():   return "local min (PD)"
    if (ev < 0).all():   return "local max (ND)"
    if (ev == 0).any():  return "inconclusive (PSD/NSD)"
    return "saddle (indefinite)"

print(classify(torch.tensor([[2., 0.], [0., 3.]])))   # min
print(classify(torch.tensor([[2., 0.], [0., -1.]])))  # saddle
```

**You've got this piece when you can** define PD/PSD/indefinite, give the eigenvalue / Sylvester / Cholesky tests, and classify a critical point (min/max/saddle/inconclusive) from the sign pattern of $H$'s eigenvalues.

## Piece 4 — Curvature governs gradient descent: the condition number (~35 min)

*Source: applied synthesis. This is the "why" behind HW2 §B (hyperparameter transfer / μP) and §A steepest descent. Deep-dive: [[Gradient Descend]] §convergence & [[PyTorch × NVIDIA GPU — Training Internals Deep Dive]].*

Run gradient descent $w_{t+1}=w_t-\eta\nabla L$ on the quadratic $L=\tfrac12 w^\top H w$ (so $\nabla L=Hw$). The update is $w_{t+1}=(I-\eta H)w_t$. Diagonalize $H=Q\Lambda Q^\top$ and track each eigen-coordinate $c_i=q_i^\top w$ independently (this is *exactly* yesterday's matrix-powers argument, $A^k=Q\Lambda^kQ^{-1}$, applied to $I-\eta H$):

$$
c_i^{(t+1)} = (1-\eta\lambda_i)\,c_i^{(t)} \quad\Longrightarrow\quad c_i^{(t)} = (1-\eta\lambda_i)^t\,c_i^{(0)} .
$$

Each mode contracts by $|1-\eta\lambda_i|$ per step. Convergence needs $|1-\eta\lambda_i|<1$ for **every** $i$, i.e. $0<\eta<2/\lambda_{\max}$. But a single $\eta$ must serve all modes at once:

- The largest curvature $\lambda_{\max}$ caps the stable step size ($\eta<2/\lambda_{\max}$, else the steep direction diverges).
- The smallest curvature $\lambda_{\min}$ then sets how *slowly* the flattest direction converges (rate $\approx 1-\eta\lambda_{\min}\approx 1-2\lambda_{\min}/\lambda_{\max}$).

So the worst-case convergence rate is controlled by the **condition number**

$$
\kappa \;=\; \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)} .
$$

Large $\kappa$ (ill-conditioned, elongated-valley loss) → GD zig-zags and crawls; $\kappa=1$ (spherical bowl) → converges in essentially one step. This is the entire motivation for (i) **preconditioning / Newton** (multiply by $H^{-1}$ to make curvature isotropic), (ii) **adaptive / sign methods** (Adam ≈ steepest descent in $\ell_\infty$, HW2 P3), and (iii) **μP / spectral scaling** (HW2 §B keeps each layer's spectral norm — hence its effective curvature — width-invariant so the tuned $\eta$ transfers).

```python
import torch

for kappa in [1.0, 10.0, 100.0]:
    H = torch.diag(torch.tensor([1.0, 1.0 * kappa]))   # λ_min=1, λ_max=κ
    w = torch.tensor([1.0, 1.0])
    eta = 1.0 / kappa                                  # ~ largest stable-ish step
    for t in range(200):
        w = w - eta * (H @ w)                           # GD on ½wᵀHw
    print(f"κ={kappa:5.0f}  ‖w‖ after 200 steps = {w.norm():.3e}")
# larger κ → slower decay of ‖w‖ toward 0 (the flat direction lags)
```

**You've got this piece when you can** derive $c_i^{(t)}=(1-\eta\lambda_i)^t c_i^{(0)}$, state the stability bound $\eta<2/\lambda_{\max}$, define the condition number $\kappa=\lambda_{\max}/\lambda_{\min}$ and explain why it sets GD's convergence rate and motivates preconditioning / μP.

## Piece 5 — Lagrange multipliers: constrained optimization behind HW2's master identity (~35 min)

*Source: Udemy Linear Algebra → "Lagrange multipliers." This is the machinery under [[Homework 2]] §A Problem 2 (norm-constrained steepest descent). Deep-dive: [[Homework 2]] §A.*

To optimize $f(w)$ subject to a constraint $g(w)=0$, you cannot just set $\nabla f=0$ — the constraint pins you to a surface. The **Lagrange condition** says: at a constrained optimum, the objective's gradient is **parallel** to the constraint's gradient,

$$
\nabla f(w^\star)\;=\;\mu\,\nabla g(w^\star),\qquad g(w^\star)=0,
$$

for some scalar **multiplier** $\mu$. **Why this geometry.** If $\nabla f$ had any component *along* the constraint surface (i.e. orthogonal to $\nabla g$), you could slide along the surface and decrease $f$ — so you are not yet optimal. At the optimum, $\nabla f$ must be entirely perpendicular to the surface, i.e. parallel to $\nabla g$. Equivalently, define the **Lagrangian** $\mathcal L(w,\mu)=f(w)-\mu\,g(w)$ and set $\nabla_w\mathcal L=0$ and $\partial\mathcal L/\partial\mu=0$; the second equation just re-imposes $g(w)=0$.

**The HW2 connection (do this by hand — it's Problem 2 in disguise).** The master identity minimizes $g^\top\Delta w+\tfrac\lambda2\|\Delta w\|^2$. The equivalent *constrained* view is "find the unit-norm direction of steepest decrease": minimize $g^\top t$ subject to $\|t\|=1$, i.e. $g(t)=\|t\|^2-1=0$. Lagrange gives $\nabla_t(g^\top t)=\mu\,\nabla_t(\|t\|^2-1)$, i.e. $g=2\mu t$, so $t^\star\propto g$ — the optimal direction is (anti)parallel to the gradient, and the multiplier enforces unit length. That is exactly why the dual-norm decomposition in HW2 separates a **direction** ($\arg\max_{\|t\|=1}g^\top t$) from a **length**: the direction is a constrained problem Lagrange solves, and the length is the unconstrained 1-D quadratic in $r$.

```python
# Sanity check: minimize gᵀt s.t. ‖t‖₂ = 1  → t* = −g/‖g‖ (Lagrange: g = 2μ t)
import torch
g = torch.tensor([3.0, -4.0])
t_star = -g / g.norm()
print("t*:", t_star, " value gᵀt*:", (g @ t_star).item())  # = −‖g‖ = −5
```

**You've got this piece when you can** state the Lagrange condition $\nabla f=\mu\nabla g$ and explain its "gradients must be parallel" geometry, write the Lagrangian, and show that minimizing $g^\top t$ under $\|t\|=1$ yields $t^\star\propto g$ — linking it to HW2's separation of direction from step length.

# 📝 Review quiz

1. **(Concept)** Write the second-order Taylor expansion of $L(w_0+\Delta w)$ and name each of the three terms. Which terms does plain gradient descent use, and which does Newton's method add?
2. **(Derivation)** Using the spectral theorem on $H$, show that the quadratic form $\tfrac12\Delta w^\top H\Delta w$ equals $\tfrac12\sum_i\lambda_i c_i^2$. What do the eigenvalues $\lambda_i$ represent geometrically?
3. **(Concept)** Why is the Hessian symmetric, and *why does that symmetry matter* for talking about "curvature"? Name the theorem.
4. **(Computation / concept)** State the three definiteness tests (eigenvalues, Sylvester, Cholesky). For $H=\begin{pmatrix}2&0\\0&-1\end{pmatrix}$ at a critical point, classify the point and justify via the eigenvalues.
5. **(Derivation)** For GD on $L=\tfrac12 w^\top H w$, derive $c_i^{(t)}=(1-\eta\lambda_i)^t c_i^{(0)}$ in the eigenbasis. What is the stability condition on $\eta$, and which eigenvalue sets it?
6. **(Concept)** Define the condition number $\kappa$ of the Hessian. Explain, using the per-mode contraction factors, why large $\kappa$ makes gradient descent slow, and name one remedy tied to HW2.
7. **(Code reading)** In Piece 3's `classify`, why is `torch.linalg.eigvalsh` used rather than `eig`, and why does the sign pattern of its output suffice to label min / max / saddle?
8. **(Derivation / applied)** State the Lagrange multiplier condition for minimizing $f$ subject to $g(w)=0$ and explain the "parallel gradients" geometry. Then show that minimizing $g^\top t$ subject to $\|t\|_2=1$ gives $t^\star=-g/\|g\|_2$, and connect this to HW2's separation of direction from step length.

> [!note]- 🔑 Answer key (click to reveal)
> **1.** $L(w_0+\Delta w)\approx L(w_0)+\nabla L(w_0)^\top\Delta w+\tfrac12\Delta w^\top H(w_0)\Delta w$. Terms: (i) the constant value at $w_0$; (ii) the **linear** term $\nabla L^\top\Delta w$ — the tangent-plane slope; (iii) the **quadratic** curvature term with the Hessian. Gradient descent uses only (i)+(ii) — a linear model with a chosen step size. Newton's method adds (iii), the curvature, and jumps to the minimizer of the local quadratic ($\Delta w=-H^{-1}\nabla L$).
>
> **2.** Symmetric $H=Q\Lambda Q^\top$ with $Q$ orthonormal. Substitute and let $c=Q^\top\Delta w$ (coordinates in the eigenbasis): $\tfrac12\Delta w^\top H\Delta w=\tfrac12\Delta w^\top Q\Lambda Q^\top\Delta w=\tfrac12 c^\top\Lambda c=\tfrac12\sum_i\lambda_i c_i^2$. Each $\lambda_i$ is the **curvature** of $L$ along the principal axis $q_i$: $>0$ bowls up, $<0$ bends down, $=0$ is flat. The form is a sum of independent 1-D parabolas.
>
> **3.** $H_{ij}=\partial^2 L/\partial w_i\partial w_j$. By **Clairaut's (Schwarz's) theorem**, for $C^2$ functions the mixed partials are equal, so $H_{ij}=H_{ji}$, i.e. $H=H^\top$. Symmetry matters because it lets the **spectral theorem** apply: $H$ then has real eigenvalues and an orthonormal eigenbasis, so "curvature" is a well-defined set of $n$ real numbers along $n$ perpendicular axes (a general non-symmetric matrix could have complex eigenvalues / no orthogonal basis, and "curvature" wouldn't make clean sense).
>
> **4.** Tests: (i) **eigenvalues** — PD iff all $\lambda_i>0$, PSD iff all $\ge0$, indefinite iff mixed signs; (ii) **Sylvester** — PD iff all leading principal minors $>0$; (iii) **Cholesky** — PD iff $A=LL^\top$ exists with positive diagonal. For $H=\mathrm{diag}(2,-1)$ the eigenvalues are $2>0$ and $-1<0$ (mixed sign) → **indefinite** → the critical point is a **saddle**: the loss curves *up* along the first axis and *down* along the second.
>
> **5.** $\nabla L=Hw$, so $w_{t+1}=w_t-\eta Hw_t=(I-\eta H)w_t$. With $H=Q\Lambda Q^\top$ and $c=Q^\top w$, each coordinate evolves independently: $c_i^{(t+1)}=(1-\eta\lambda_i)c_i^{(t)}$, hence $c_i^{(t)}=(1-\eta\lambda_i)^t c_i^{(0)}$ (same telescoping as $A^k=Q\Lambda^kQ^{-1}$). Convergence needs $|1-\eta\lambda_i|<1$ for all $i$, i.e. $0<\eta<2/\lambda_i$ for every $i$; the binding constraint is the **largest** eigenvalue, giving $\eta<2/\lambda_{\max}$.
>
> **6.** $\kappa=\lambda_{\max}(H)/\lambda_{\min}(H)$. The steep direction forces $\eta<2/\lambda_{\max}$; with such an $\eta$, the flat direction contracts by only $|1-\eta\lambda_{\min}|\approx1-2\lambda_{\min}/\lambda_{\max}=1-2/\kappa$ per step — close to 1 when $\kappa$ is large, so that mode crawls and GD zig-zags down an elongated valley. Remedies: preconditioning / Newton (rescale by $H^{-1}$ toward isotropy), adaptive/sign methods (Adam ≈ $\ell_\infty$ steepest descent), or **μP / spectral scaling** (HW2 §B), which keeps each layer's spectral norm width-invariant so curvature — and the tuned learning rate — transfers across width.
>
> **7.** `eigvalsh` is the **symmetric/Hermitian** eigenvalue routine; the Hessian is symmetric, so `eigvalsh` returns **real, ascending** eigenvalues (and is faster/stabler), whereas `eig` could return spurious tiny imaginary parts and an unordered spectrum. The sign pattern suffices because at a critical point the second-order behavior is $\tfrac12\sum_i\lambda_i c_i^2$: all $\lambda_i>0$ ⇒ up everywhere (min), all $<0$ ⇒ down everywhere (max), mixed ⇒ saddle, any $=0$ ⇒ second order inconclusive.
>
> **8.** Lagrange: at a constrained optimum $\nabla f(w^\star)=\mu\nabla g(w^\star)$ with $g(w^\star)=0$. Geometry — if $\nabla f$ had a component tangent to the surface $g=0$, you could slide along the surface to reduce $f$, so at optimum $\nabla f$ is entirely perpendicular to the surface, i.e. parallel to $\nabla g$. For $\min g^\top t$ s.t. $\|t\|_2^2=1$: set $\nabla_t(g^\top t)=\mu\nabla_t(\|t\|^2-1)$, i.e. $g=2\mu t$, so $t\propto g$; choosing the sign that *minimizes* $g^\top t$ and normalizing gives $t^\star=-g/\|g\|_2$ (value $-\|g\|_2$). This is HW2's split: the **direction** is a constrained (Lagrange) problem giving $-g/\|g\|$, while the **step length** $r=\|g\|/\lambda$ comes from the leftover unconstrained 1-D quadratic in $r$.
