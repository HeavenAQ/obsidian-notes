---
base: "[[01.2 DL Daily Lessons — Step-by-Step + Quiz.base]]"
Quiz score: ""
"HW #": 2
Studied: false
Quiz taken: false
Day type: Lesson-Day
Date: 2026-07-19
Piece count: 5
---
> 🗒️ **Quiz backlog nudge:** yesterday's **Jul 18 (Positive Definiteness / Hessian / Taylor / Lagrange)** quiz is still ungraded (`Quiz taken: false`, no score), as are the Jul 17 (Eigenvalues I) and the Jul 12/13/14/15 HW1 sets. Nothing to patch up today — the re-teach loop only fires on a *recorded* low score. If you want spaced-repetition credit, grade at least the Jul 18 set (~10 min) before starting; today's projection/eigenbasis reasoning builds straight on it.

# 🎯 Today's goal

This is **HW2 Day 3** (Udemy Linear Algebra → *Least squares, the normal equation & projection matrices*). You'll take an **overdetermined** system $Ax=b$ (more equations than unknowns, no exact solution) and find the $x$ that minimizes $\|Ax-b\|_2^2$. Two derivations of the **normal equation** $A^\top A\,x=A^\top b$ — one by calculus, one by orthogonal projection — give you the geometry: least squares projects $b$ onto the column space of $A$, and the residual is orthogonal to that space. This is the linear-regression backbone of **generalization theory**, and it feeds HW2 directly: §A **Problem 5**'s square-loss majorizer is exactly a least-squares objective whose gradient is an average of residual $\times$ input outer products, and the **SVD / pseudo-inverse** view ties back to §0's $G=U\Sigma V^\top$ bridge. We close by connecting ridge least squares to **weight decay** (§B, Problem 11) and to why conditioning controls both optimization *and* generalization.

# 🧩 Pieces

## Piece 1 — The least-squares problem: projecting onto the column space (~25 min)

*Source: Udemy Linear Algebra → "least squares / overdetermined systems." Feeds [[Homework 2]] §A (Problem 5's square-loss setup) and the generalization thread. Deep-dive: [[Gradient Descend]] §1 (what optimization is doing).*

When $A\in\mathbb{R}^{m\times n}$ with $m>n$ (tall matrix, more data rows than parameters), $Ax=b$ generally has **no** solution: $b$ almost never lies exactly in the $n$-dimensional **column space** $\mathrm{Col}(A)=\{Ax:x\in\mathbb{R}^n\}$ sitting inside $\mathbb{R}^m$. Instead of an exact solve we ask for the **best approximation**:

$$
x^\star \;=\; \arg\min_{x\in\mathbb{R}^n}\; \|Ax-b\|_2^2 .
$$

Geometrically, $\{Ax\}$ traces out the whole subspace $\mathrm{Col}(A)$. Minimizing the Euclidean distance $\|Ax-b\|_2$ means finding the point $\hat b=Ax^\star$ in that subspace **closest** to $b$ — i.e. the **orthogonal projection** of $b$ onto $\mathrm{Col}(A)$. The leftover $r=b-\hat b$ is the **residual**, and the defining property of an orthogonal projection is that this residual is **perpendicular to the subspace**:

$$
r=b-Ax^\star \;\perp\; \mathrm{Col}(A)\qquad\Longleftrightarrow\qquad A^\top(b-Ax^\star)=0 .
$$

That orthogonality condition *is* the normal equation (Piece 2). Keep the mental picture: a point $b$ floating above a plane, its shadow $\hat b$ on the plane, and the dashed vertical line $r$ meeting the plane at a right angle.

**You've got this piece when you can** explain why a tall $Ax=b$ has no exact solution, state that least squares finds the orthogonal projection of $b$ onto $\mathrm{Col}(A)$, and write the orthogonality condition $A^\top(b-Ax^\star)=0$.

## Piece 2 — The normal equation: two derivations (~35 min)

*Source: Udemy Linear Algebra → "the normal equation." The algebraic core; anchors Pieces 3–5. Deep-dive: [[Homework 2]] §A Problem 5 (majorizer gradient).*

**Derivation A — calculus.** Expand the objective $f(x)=\|Ax-b\|_2^2=(Ax-b)^\top(Ax-b)$:

$$
f(x) = x^\top A^\top A\,x - 2\,b^\top A\,x + b^\top b .
$$

This is a convex quadratic in $x$. Set the gradient to zero (use $\nabla_x\,x^\top M x=2Mx$ for symmetric $M=A^\top A$, and $\nabla_x\,c^\top x=c$):

$$
\nabla f(x)=2A^\top A\,x-2A^\top b=0 \quad\Longrightarrow\quad \boxed{A^\top A\,x = A^\top b}.
$$

Because the Hessian is $\nabla^2 f=2A^\top A\succeq0$ (PSD — yesterday's definiteness test!), this critical point is a global minimum; it is the **unique** minimizer exactly when $A$ has full column rank so that $A^\top A\succ0$ (PD, invertible).

**Derivation B — geometry.** Impose Piece 1's orthogonality directly: $A^\top(b-Ax)=0\Rightarrow A^\top A\,x=A^\top b$. Same equation, no calculus — the residual being orthogonal to every column of $A$ *is* $A^\top r=0$.

When $A^\top A$ is invertible, solve to get the closed form

$$
x^\star=(A^\top A)^{-1}A^\top b .
$$

**Why formulated this way.** The matrix $(A^\top A)^{-1}A^\top$ is the **left pseudo-inverse** $A^{+}$ (Piece 4): it is the tall-matrix stand-in for "$A^{-1}$" when $A$ isn't square. In practice you never invert — you solve $A^\top A\,x=A^\top b$ via Cholesky (valid because $A^\top A$ is PD) or, better numerically, solve the least-squares problem directly by QR/SVD to avoid squaring the condition number (Piece 4).

```python
import torch

torch.manual_seed(0)
A = torch.randn(50, 3)          # tall: 50 equations, 3 unknowns
x_true = torch.tensor([1.0, -2.0, 0.5])
b = A @ x_true + 0.1 * torch.randn(50)   # noisy targets

# normal equation (didactic): xˢᵗᵃʳ = (AᵀA)⁻¹ Aᵀ b
x_ne = torch.linalg.solve(A.T @ A, A.T @ b)
# numerically preferred: solve the LS problem directly (QR/SVD under the hood)
x_ls = torch.linalg.lstsq(A, b).solution
print("normal eq:", x_ne)
print("lstsq   :", x_ls)
print("residual ⟂ columns? Aᵀr ≈ 0:", (A.T @ (b - A @ x_ne)).abs().max().item())
```

**You've got this piece when you can** derive $A^\top A\,x=A^\top b$ both by differentiating $\|Ax-b\|_2^2$ and by imposing $A^\top r=0$, state the full-column-rank / PD condition for a unique solution, and write $x^\star=(A^\top A)^{-1}A^\top b$.

## Piece 3 — Projection matrices and the hat matrix (~30 min)

*Source: Udemy Linear Algebra → "projection matrices." Formalizes Piece 1's geometry. Deep-dive pointer: [[Homework 2]] §0 (SVD/spectral bridge).*

Substitute $x^\star$ back to get the fitted vector $\hat b=Ax^\star$:

$$
\hat b = A(A^\top A)^{-1}A^\top\,b \;=\; P\,b,\qquad P \;\stackrel{\text{def}}{=}\; A(A^\top A)^{-1}A^\top .
$$

$P$ is the **orthogonal projection matrix** onto $\mathrm{Col}(A)$ (in statistics, the **hat matrix** — it "puts the hat on $b$"). Its two defining algebraic properties, both worth being able to prove in one line:

- **Idempotent:** $P^2=P$. Projecting an already-projected vector changes nothing (the shadow of a point already on the plane is itself). Check: $P^2=A(A^\top A)^{-1}\underbrace{A^\top A(A^\top A)^{-1}}_{I}A^\top=A(A^\top A)^{-1}A^\top=P$.
- **Symmetric:** $P^\top=P$. This is what makes the projection *orthogonal* (as opposed to an oblique projection); symmetry $\Leftrightarrow$ the residual direction is perpendicular to the target subspace.

The complementary projector $I-P$ maps $b$ to the **residual** $r=(I-P)b$, projecting onto the orthogonal complement $\mathrm{Col}(A)^\perp$ (the left null space of $A$). It is also idempotent and symmetric, and $P(I-P)=0$ — the fitted part and the residual part are orthogonal, which is the **Pythagorean split** $\|b\|^2=\|Pb\|^2+\|(I-P)b\|^2$ (this is exactly the ANOVA/$R^2$ decomposition in regression). The **eigenvalues of any projection matrix are only 0 and 1** (from $P^2=P\Rightarrow\lambda^2=\lambda$): eigenvalue 1 on the subspace you keep, 0 on the part you kill; $\operatorname{trace}(P)=\operatorname{rank}(A)=n$ counts the kept dimensions.

```python
import torch
A = torch.randn(6, 2)
P = A @ torch.linalg.inv(A.T @ A) @ A.T          # 6×6 projector onto Col(A)
print("idempotent P²=P:", torch.allclose(P @ P, P, atol=1e-5))
print("symmetric  Pᵀ=P:", torch.allclose(P, P.T, atol=1e-5))
print("eigenvalues (≈ two 1's, four 0's):", torch.linalg.eigvalsh(P).round(decimals=3))
print("trace = rank =", P.trace().item())        # ≈ 2
```

**You've got this piece when you can** write $P=A(A^\top A)^{-1}A^\top$, prove $P^2=P$ and $P^\top=P$, describe $I-P$ as the residual projector onto $\mathrm{Col}(A)^\perp$, and state that $P$'s eigenvalues are 0/1 with $\operatorname{trace}(P)=\operatorname{rank}(A)$.

## Piece 4 — The SVD / pseudo-inverse view and conditioning (~35 min)

*Source: applied synthesis with the Day 1 eigen/SVD toolkit. Directly bridges to [[Homework 2]] §0 (SVD) and §B (spectral norm / power iteration). Deep-dive: [[Gradient Descend]] §convergence.*

Yesterday and Day 1 gave you $A=U\Sigma V^\top$ (SVD). Feed it through the normal equation. With $A^\top A=V\Sigma^2V^\top$ (an eigendecomposition — the squared singular values are the eigenvalues of $A^\top A$):

$$
x^\star=(A^\top A)^{-1}A^\top b = V\Sigma^{-2}V^\top\,V\Sigma U^\top b = V\Sigma^{-1}U^\top b \;=\; A^{+}b,
$$

where $A^{+}=V\Sigma^{+}U^\top$ is the **Moore–Penrose pseudo-inverse** ($\Sigma^{+}$ inverts the nonzero singular values, leaving zeros in place). Two payoffs:

1. **Minimum-norm solution.** When $A$ is *not* full column rank (some $\sigma_i=0$), the normal equation has infinitely many solutions; $A^{+}b$ picks the one with **smallest $\|x\|_2$** (it puts zero weight on the null-space directions). This is the cleanest statement of what "the" least-squares solution is in the rank-deficient case, and it foreshadows why implicit/explicit regularization matters for generalization.
2. **Conditioning.** In the singular basis, $x^\star=\sum_i \frac{u_i^\top b}{\sigma_i}v_i$. Directions with **tiny $\sigma_i$** get divided by a tiny number, so any noise component along $u_i$ is **amplified by $1/\sigma_i$** — the fit blows up along weakly-excited directions. The sensitivity is governed by the **condition number** $\kappa(A)=\sigma_{\max}/\sigma_{\min}$ (and forming $A^\top A$ *squares* it to $\kappa(A)^2$ — that is precisely why `lstsq` (QR/SVD) is preferred over explicitly solving the normal equation).

This is the same $\kappa$ story from yesterday's Hessian piece: for the quadratic $f(x)=\|Ax-b\|^2$ the Hessian is $2A^\top A$, so its eigenvalues are $2\sigma_i^2$ and its condition number is $\kappa(A)^2$ — ill-conditioned regression = ill-conditioned optimization landscape.

```python
import torch
A = torch.randn(40, 4)
b = torch.randn(40)
U, S, Vh = torch.linalg.svd(A, full_matrices=False)
x_svd = Vh.T @ ((U.T @ b) / S)                    # xˢᵗᵃʳ = V Σ⁺ Uᵀ b
x_ref = torch.linalg.lstsq(A, b).solution
print("SVD == lstsq:", torch.allclose(x_svd, x_ref, atol=1e-5))
print("cond(A):", (S.max()/S.min()).item(), " cond(AᵀA)=cond(A)²:", (S.max()/S.min()).item()**2)
```

**You've got this piece when you can** derive $x^\star=V\Sigma^{-1}U^\top b=A^{+}b$ from the SVD, explain the pseudo-inverse as the minimum-norm least-squares solution, and explain why small singular values amplify noise (and why $A^\top A$ squaring $\kappa$ makes the normal equation numerically worse than QR/SVD).

## Piece 5 — Least squares → HW2: the square-loss majorizer, ridge & weight decay (~30 min)

*Source: applied synthesis. Directly HW2 §A Problem 5 and §B Problem 11. Deep-dive: [[Homework 2]] §A (majorization) and §B (weight decay).*

**The gradient is residual $\times$ input.** HW2's §A Problem 5 studies a matrix square loss $L(W)=\tfrac1N\sum_i\tfrac12\|y^{(i)}-Wx^{(i)}\|_2^2$. Differentiating (the same expansion as Piece 2, one data point at a time) gives the **gradient as an average of outer products of the residual with the input**:

$$
G=\nabla_W L=-\frac1N\sum_{i=1}^{N}\big(y^{(i)}-Wx^{(i)}\big)\,x^{(i)\top}.
$$

Setting $G=0$ is the **matrix normal equation** $W\big(\sum_i x^{(i)}x^{(i)\top}\big)=\sum_i y^{(i)}x^{(i)\top}$ — least squares in disguise. Recognizing this is why you can read Problem 5's majorizer as "a least-squares step," and it explains the shape of the update (the $\text{residual}\cdot x^\top$ outer product) you'll implement.

**Ridge = weight decay = shrinking the spectrum.** Add an $\ell_2$ penalty:

$$
x_\lambda=\arg\min_x\|Ax-b\|_2^2+\lambda\|x\|_2^2 \quad\Longrightarrow\quad (A^\top A+\lambda I)\,x_\lambda=A^\top b .
$$

In the SVD basis this replaces each $1/\sigma_i$ by the **shrinkage factor** $\sigma_i/(\sigma_i^2+\lambda)$: large-$\sigma$ (well-determined) directions are barely touched, tiny-$\sigma$ directions are damped instead of exploding. Two consequences to lock in:

- **Numerically:** $A^\top A+\lambda I\succ0$ always, so ridge is *always* solvable and better-conditioned ($\kappa$ shrinks) — regularization tames the noise-amplification of Piece 4.
- **HW2 §B Problem 11 tie-in:** the weight-decay step $W\to(1-\eta\lambda)W$ multiplies **every singular value of $W$ by the same factor** while leaving the singular *directions* $u_i,v_i$ fixed — an isotropic contraction of the spectrum, the discrete-time cousin of ridge's $\sigma_i/(\sigma_i^2+\lambda)$ shrinkage. This is also the generalization link: shrinking the spectrum lowers effective capacity (the trace of the "smoother" $A(A^\top A+\lambda I)^{-1}A^\top$ — the ridge analogue of the hat matrix — is the effective degrees of freedom).

```python
import torch
A = torch.randn(60, 5); b = torch.randn(60)
def ridge(A, b, lam):
    n = A.shape[1]
    return torch.linalg.solve(A.T @ A + lam*torch.eye(n), A.T @ b)
for lam in [0.0, 1.0, 100.0]:
    x = ridge(A, b, lam if lam>0 else 1e-8)
    print(f"λ={lam:6.1f}  ‖x‖={x.norm():.3f}")   # larger λ → smaller ‖x‖ (shrinkage)
```

**You've got this piece when you can** show the square-loss gradient is $-\tfrac1N\sum_i(y^{(i)}-Wx^{(i)})x^{(i)\top}$ (a matrix normal equation), write the ridge normal equation $(A^\top A+\lambda I)x=A^\top b$, describe the per-singular-value shrinkage $\sigma_i/(\sigma_i^2+\lambda)$, and connect it to HW2's weight-decay-as-spectral-contraction (Problem 11).

# 📝 Review quiz

1. **(Concept)** Why does a tall system $Ax=b$ ($m>n$) generally have no exact solution, and what does least squares return instead? State the orthogonality condition the solution must satisfy.
2. **(Derivation)** Derive the normal equation $A^\top A\,x=A^\top b$ by minimizing $\|Ax-b\|_2^2$ with calculus. What is the Hessian, and what condition on $A$ makes the minimizer unique?
3. **(Derivation)** Give the *geometric* derivation of the normal equation from "residual ⟂ column space." Why are the two derivations the same equation?
4. **(Computation / concept)** Write the projection (hat) matrix $P$ onto $\mathrm{Col}(A)$ and prove $P^2=P$. What are the eigenvalues of $P$, and what does $\operatorname{trace}(P)$ equal?
5. **(Derivation)** Using $A=U\Sigma V^\top$, show $x^\star=V\Sigma^{-1}U^\top b=A^{+}b$. Explain why directions with small singular values amplify noise, and why solving the normal equation is numerically worse than QR/SVD.
6. **(Code reading)** In Piece 2, why does `torch.linalg.lstsq(A, b)` differ (numerically) from `torch.linalg.solve(A.T @ A, A.T @ b)`, and which would you trust for an ill-conditioned $A$? Name the quantity that gets squared.
7. **(Applied / derivation)** For the matrix square loss $L(W)=\tfrac1N\sum_i\tfrac12\|y^{(i)}-Wx^{(i)}\|_2^2$, derive the gradient $G$ and connect it to HW2 §A Problem 5. Then write the ridge normal equation and describe what weight decay ($W\to(1-\eta\lambda)W$) does to the singular values of $W$ (HW2 §B Problem 11).

> [!note]- 🔑 Answer key (click to reveal)
> **1.** With $m>n$ the target $b$ lives in $\mathbb{R}^m$ but $\{Ax\}=\mathrm{Col}(A)$ is only an $n$-dimensional subspace, so unless $b$ happens to lie in that subspace there is no exact solution. Least squares returns $x^\star=\arg\min\|Ax-b\|_2^2$, i.e. the $x$ whose fit $\hat b=Ax^\star$ is the **orthogonal projection of $b$ onto $\mathrm{Col}(A)$**. The residual must be perpendicular to every column of $A$: $A^\top(b-Ax^\star)=0$.
>
> **2.** $f(x)=\|Ax-b\|_2^2=x^\top A^\top A x-2b^\top A x+b^\top b$. Gradient $\nabla f=2A^\top A x-2A^\top b$; setting it to $0$ gives $A^\top A\,x=A^\top b$. The Hessian is $\nabla^2 f=2A^\top A\succeq0$ (PSD), so any stationary point is a global min; it is **unique** iff $A$ has full column rank, which makes $A^\top A\succ0$ (PD, invertible), giving $x^\star=(A^\top A)^{-1}A^\top b$.
>
> **3.** Orthogonality of the residual to the column space means $A^\top r=0$ with $r=b-Ax$, i.e. $A^\top(b-Ax)=0\Rightarrow A^\top A\,x=A^\top b$. It is the same equation because minimizing Euclidean distance to a subspace *is* orthogonal projection: the calculus first-order condition $\nabla f=0$ literally reads $-2A^\top(b-Ax)=0$, which is the geometric orthogonality condition. Distance-minimization and perpendicular-residual are two names for one fact.
>
> **4.** $P=A(A^\top A)^{-1}A^\top$. Idempotence: $P^2=A(A^\top A)^{-1}[A^\top A(A^\top A)^{-1}]A^\top=A(A^\top A)^{-1}A^\top=P$. It is also symmetric ($P^\top=P$), which makes the projection orthogonal. From $P^2=P$, any eigenvalue satisfies $\lambda^2=\lambda$, so $\lambda\in\{0,1\}$ (1 on $\mathrm{Col}(A)$, 0 on $\mathrm{Col}(A)^\perp$). $\operatorname{trace}(P)=\operatorname{rank}(A)$ = number of kept dimensions (= $n$ when $A$ has full column rank).
>
> **5.** With $A=U\Sigma V^\top$: $A^\top A=V\Sigma^2V^\top$, so $(A^\top A)^{-1}=V\Sigma^{-2}V^\top$ and $A^\top=V\Sigma U^\top$. Then $x^\star=(A^\top A)^{-1}A^\top b=V\Sigma^{-2}V^\top V\Sigma U^\top b=V\Sigma^{-1}U^\top b=A^{+}b$. Componentwise $x^\star=\sum_i\frac{u_i^\top b}{\sigma_i}v_i$: any noise in $b$ along $u_i$ is scaled by $1/\sigma_i$, so tiny $\sigma_i$ **amplifies noise** enormously. Forming $A^\top A$ squares the singular values, so $\kappa(A^\top A)=\kappa(A)^2$ — you lose twice the digits — whereas QR/SVD work on $A$ directly at condition number $\kappa(A)$; hence `lstsq` is preferred.
>
> **6.** `solve(A.T@A, A.T@b)` explicitly forms $A^\top A$, whose condition number is $\kappa(A)^2$ — the squared conditioning loses roughly twice as many significant digits and can even make the (theoretically PD) matrix numerically indefinite. `lstsq` solves the least-squares problem via QR (or SVD) on $A$ itself at condition number $\kappa(A)$, so it is far more accurate for ill-conditioned $A$. The quantity that gets squared is the **condition number** $\kappa(A)=\sigma_{\max}/\sigma_{\min}$. Trust `lstsq`.
>
> **7.** $L(W)=\tfrac1N\sum_i\tfrac12\|y^{(i)}-Wx^{(i)}\|_2^2$. Differencing term $i$: $\nabla_W\tfrac12\|y^{(i)}-Wx^{(i)}\|^2=-(y^{(i)}-Wx^{(i)})x^{(i)\top}$, so $G=\nabla_W L=-\tfrac1N\sum_i(y^{(i)}-Wx^{(i)})x^{(i)\top}$ — an average of **residual × input outer products**; this is exactly the gradient identified in HW2 §A Problem 5's majorizer, and setting $G=0$ recovers the matrix normal equation. Ridge: $x_\lambda=\arg\min\|Ax-b\|^2+\lambda\|x\|^2\Rightarrow(A^\top A+\lambda I)x_\lambda=A^\top b$, which in the SVD basis replaces $1/\sigma_i$ by the shrinkage $\sigma_i/(\sigma_i^2+\lambda)$. Weight decay $W\to(1-\eta\lambda)W$ multiplies **every singular value of $W$ by the same factor $(1-\eta\lambda)$** while leaving the singular directions $u_i,v_i$ unchanged (HW2 §B Problem 11): an isotropic spectral contraction, the discrete cousin of ridge shrinkage.
