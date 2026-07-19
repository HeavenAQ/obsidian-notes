---
base: "[[01.2 DL Daily Lessons — Step-by-Step + Quiz.base]]"
Quiz score: ""
"HW #": 2
Studied: false
Quiz taken: false
Day type: Lesson-Day
Date: 2026-07-17
Piece count: 5
---
> ⚠️ **Schedule note (catch-up):** No lesson file was created for Jul 16 (HW2 Day 1), so the tutor is one day behind the fixed calendar. Today I'm running **HW2 Day 1 — Udemy Linear Algebra → Eigenvalues & Eigenvectors (I)** (the first unchecked item in Homework 2's 📚 checklist, and exactly what the homework-coach prep notes are set to). The plan shifts by one day: **Day 2 (positive definiteness / Hessian / Taylor / Lagrange) → tomorrow Jul 18**, Day 3 (least squares & projections) → Jul 19, Day 4 (attention theory) → Jul 20, Day 5 (6.S191 L2 + start HW2) → Jul 21, then **HW2 Jul 21–22** (due Jul 22). Still fully on time to submit. *(Skip the difference/differential-equations sections of the Udemy module, per the checklist.)*
>
> 🗒️ **Quiz backlog nudge:** the Jul 12/13/14/15 (HW1) quizzes are all still **untaken** (no recorded scores), so the patch-up loop stayed dark — nothing to re-teach today. If you want spaced-repetition credit, grade at least the Jul 15 cumulative set (~10 min) before starting.

# 🎯 Today's goal

Build the eigenvalue toolkit that the entire HW2 window rests on. By the end you should be able to define an eigenpair and read it geometrically, factor a matrix as $A = Q\Lambda Q^{-1}$ (and, for symmetric $A$, as $A = Q\Lambda Q^\top$ with an **orthonormal** eigenbasis), and use $A^k = Q\Lambda^k Q^{-1}$ to reason about what repeated matrix multiplication does. This feeds HW2 directly: **§Hyperparameter Transfer** uses **power iteration** (repeated matrix powers amplifying the top eigenvector), **§Steepest Descent** leans on the **spectral norm** ($\sigma_1 = \sqrt{\lambda_{\max}(A^\top A)}$), and optimization dynamics are governed by the **eigenvalues of the Hessian** — tomorrow's topic. Keep $A = Q\Lambda Q^{-1}$, $A^k = Q\Lambda^k Q^{-1}$, and the symmetric orthogonal eigenbasis in front of you the whole week.

# 🧩 Pieces

## Piece 1 — What an eigenpair *is*: definition, characteristic equation, geometry (~30 min)

*Source: Udemy Linear Algebra → Eigenvalues & Eigenvectors (I), "definition & characteristic polynomial." Ties into [[Homework 2]] §0 (Foundations) and §Hyperparameter Transfer.*

A nonzero vector $v$ is an **eigenvector** of a square matrix $A\in\mathbb{R}^{n\times n}$ with **eigenvalue** $\lambda$ if

$$
A v = \lambda v, \qquad v \neq 0 .
$$

Geometrically: $A$ generally rotates *and* stretches vectors, but along an eigenvector it does **only** stretching (by factor $\lambda$) — the direction is preserved (or flipped, if $\lambda<0$). Eigenvectors are the "axes" the linear map acts on diagonally.

To find them, rewrite $Av=\lambda v$ as $(A-\lambda I)v = 0$. A nonzero solution $v$ exists **iff** $A-\lambda I$ is singular, i.e.

$$
\det(A - \lambda I) = 0 .
$$

This is the **characteristic equation**; the left side is the degree-$n$ **characteristic polynomial** $p(\lambda)$. Its roots are the eigenvalues. For each eigenvalue you then solve the null space $(A-\lambda I)v=0$ for the eigenvector(s). Two identities fall out immediately and are worth memorizing as sanity checks:

$$
\sum_i \lambda_i = \operatorname{trace}(A), \qquad \prod_i \lambda_i = \det(A) .
$$

**Why the determinant condition.** $A-\lambda I$ singular $\iff$ it has a nontrivial null space $\iff$ some nonzero $v$ maps to $0$ $\iff$ $Av=\lambda v$. The determinant is just the scalar test for singularity. Note eigenvalues can be complex even for real $A$ (e.g. a pure rotation has no real eigenvector), but the matrices we care about this week — Hessians, $A^\top A$, covariance/Gram matrices — are **symmetric**, and symmetric real matrices always have *real* eigenvalues (Piece 3).

**You've got this piece when you can** state $Av=\lambda v$, explain why $\det(A-\lambda I)=0$ is the condition for an eigenvalue, compute the eigenvalues of a $2\times2$ by hand from the characteristic polynomial, and recover them from $\operatorname{trace}$ and $\det$.

## Piece 2 — Eigendecomposition & diagonalization: $A = Q\Lambda Q^{-1}$ (~35 min)

*Source: Udemy Linear Algebra → Eigenvalues & Eigenvectors (I), "diagonalization."*

Suppose $A\in\mathbb{R}^{n\times n}$ has $n$ **linearly independent** eigenvectors $v_1,\dots,v_n$ with eigenvalues $\lambda_1,\dots,\lambda_n$. Stack the eigenvectors as columns of $Q=[\,v_1\ \cdots\ v_n\,]$ and put the eigenvalues on a diagonal $\Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_n)$. Then $AQ = Q\Lambda$ (column $i$ is exactly $Av_i=\lambda_i v_i$), and because the columns are independent $Q$ is invertible, giving the **eigendecomposition**:

$$
A = Q\Lambda Q^{-1}.
$$

Read this as a **change of basis**: $Q^{-1}$ rewrites a vector in the eigenbasis, $\Lambda$ scales each eigen-coordinate independently, and $Q$ maps back to the standard basis. In the eigenbasis, $A$ is just $n$ decoupled scalar multiplications — that decoupling is the entire point.

**When does this exist?** $A$ is **diagonalizable** iff it has $n$ linearly independent eigenvectors. A sufficient (not necessary) condition: $n$ distinct eigenvalues. Matrices with repeated eigenvalues *may* fail to be diagonalizable if an eigenvalue's **geometric multiplicity** (dim of its eigenspace) is less than its **algebraic multiplicity** (its root multiplicity in $p(\lambda)$) — these are called *defective* (e.g. $\begin{smallmatrix}1&1\\0&1\end{smallmatrix}$). Distinguish eigendecomposition from **SVD**: SVD $G=U\Sigma V^\top$ exists for *any* matrix (even non-square), while eigendecomposition needs a square, non-defective matrix. They coincide (up to signs) only for symmetric PSD matrices.

**You've got this piece when you can** derive $A=Q\Lambda Q^{-1}$ from $AQ=Q\Lambda$, explain the change-of-basis reading, state exactly when a matrix is diagonalizable, and say how eigendecomposition differs from SVD.

## Piece 3 — The symmetric case: the spectral theorem, $A = Q\Lambda Q^\top$ (~30 min)

*Source: Udemy Linear Algebra → Eigenvalues & Eigenvectors (I), "symmetric matrices." Anchors [[Homework 2]] §0 (SVD bridge: $GG^\top=U\Sigma^2U^\top$) and tomorrow's Hessian material.*

The matrices that matter for optimization and generalization are **symmetric** ($A=A^\top$): Hessians, Gram matrices $X^\top X$, $A^\top A$, covariances. The **spectral theorem** says every real symmetric matrix has

1. **real** eigenvalues, and
2. a full set of **orthonormal** eigenvectors.

So $Q$ can be chosen **orthogonal** ($Q^\top Q = I$, hence $Q^{-1}=Q^\top$), and the decomposition becomes

$$
A = Q\Lambda Q^\top = \sum_{i=1}^{n} \lambda_i\, q_i q_i^\top .
$$

That last form — a sum of rank-1 projectors $q_iq_i^\top$ weighted by $\lambda_i$ — is the mental model to keep: $A$ acts as $\lambda_i$ along mutually perpendicular axes $q_i$. This is *exactly* the SVD bridge in [[Homework 2]] §0: for any $G$, $GG^\top = U\Sigma^2 U^\top$ is a symmetric eigendecomposition whose eigenvalues are $\sigma_i^2$, so the **spectral norm** $\|G\|_\ast=\sigma_1=\sqrt{\lambda_{\max}(GG^\top)}$. A symmetric $A$ is **positive definite** iff all $\lambda_i>0$ (tomorrow's topic and the test for a local minimum of the loss).

```python
import numpy as np

A = np.array([[2., 1.],
              [1., 3.]])                 # symmetric
w, Q = np.linalg.eigh(A)                 # eigh: symmetric/Hermitian solver
print("eigenvalues:", w)                 # real, ascending
print("QᵀQ ≈ I:", np.allclose(Q.T @ Q, np.eye(2)))     # orthonormal eigenbasis
# reconstruct A three equivalent ways
print(np.allclose(Q @ np.diag(w) @ Q.T, A))
print(np.allclose(sum(w[i]*np.outer(Q[:,i], Q[:,i]) for i in range(2)), A))
```

**You've got this piece when you can** state the two guarantees of the spectral theorem, explain why symmetry lets you replace $Q^{-1}$ with $Q^\top$, write $A=\sum_i\lambda_i q_iq_i^\top$, and connect $\lambda_{\max}(GG^\top)$ to the spectral norm $\sigma_1$.

## Piece 4 — Matrix powers & power iteration: $A^k = Q\Lambda^k Q^{-1}$ (~35 min)

*Source: Udemy Linear Algebra → Eigenvalues & Eigenvectors (I), "matrix powers." Directly powers [[Homework 2]] §Hyperparameter Transfer (power iteration = repeated matrix powers).*

Once you have $A=Q\Lambda Q^{-1}$, powers telescope because the inner $Q^{-1}Q$ factors collapse to $I$:

$$
A^2 = Q\Lambda Q^{-1}Q\Lambda Q^{-1} = Q\Lambda^2 Q^{-1}, \qquad\Longrightarrow\qquad A^k = Q\Lambda^k Q^{-1},
$$

and $\Lambda^k=\operatorname{diag}(\lambda_1^k,\dots,\lambda_n^k)$ is trivial to compute. This turns "multiply a matrix by itself $k$ times" into "raise $n$ scalars to the $k$-th power." Expand any vector in the eigenbasis, $x=\sum_i c_i q_i$; then

$$
A^k x = \sum_{i=1}^n c_i\,\lambda_i^{k}\, q_i .
$$

If there is a unique largest-magnitude eigenvalue $|\lambda_1|>|\lambda_2|\ge\cdots$, factor it out:

$$
A^k x = \lambda_1^{k}\Big(c_1 q_1 + \sum_{i\ge2} c_i (\lambda_i/\lambda_1)^k q_i\Big) \;\xrightarrow{k\to\infty}\; \lambda_1^{k} c_1 q_1 ,
$$

because every ratio $|\lambda_i/\lambda_1|<1$ decays geometrically. So repeatedly applying $A$ (and renormalizing to avoid overflow) drives any generic start vector toward the **top eigenvector** $q_1$ — this is **power iteration**, and the convergence rate is set by the spectral gap $|\lambda_2/\lambda_1|$. This is precisely how HW2's §Hyperparameter Transfer estimates a top eigenvector/eigenvalue without ever forming a full eigendecomposition, and how the spectral norm $\sigma_1$ of a weight matrix gets estimated cheaply (power iteration on $A^\top A$).

```python
import numpy as np

def power_iteration(A, iters=1000, tol=1e-12):
    n = A.shape[0]
    v = np.random.randn(n)
    v /= np.linalg.norm(v)
    lam_old = 0.0
    for _ in range(iters):
        w = A @ v                       # apply A
        v = w / np.linalg.norm(w)       # renormalize -> direction only
        lam = v @ A @ v                 # Rayleigh quotient estimate of λ₁
        if abs(lam - lam_old) < tol:
            break
        lam_old = lam
    return lam, v

A = np.array([[2., 1.], [1., 3.]])
lam, v = power_iteration(A)
w_true, Q_true = np.linalg.eigh(A)
print("power iteration λ₁:", lam, " | true λ_max:", w_true[-1])
print("aligned with top eigvec:", np.allclose(np.abs(v), np.abs(Q_true[:, -1]), atol=1e-4))
```

**You've got this piece when you can** derive $A^k=Q\Lambda^kQ^{-1}$, expand $A^kx$ in the eigenbasis and explain why the top eigenvector dominates, state that the convergence rate of power iteration is $|\lambda_2/\lambda_1|$, and write the renormalized power-iteration loop with the Rayleigh quotient.

## Piece 5 — Doing it in PyTorch/NumPy: `eig` vs `eigh`, and verifying decompositions (~25 min)

*Source: applied wrap-up. Deep-dive pointer: for the autograd/tensor mechanics behind these ops see [[PyTorch × NVIDIA GPU — Training Internals Deep Dive]]; for the SVD/spectral-norm connection see [[Homework 2]] §0.*

Two solvers, and picking the right one matters:

- `torch.linalg.eigh(A)` / `np.linalg.eigh(A)` — for **symmetric/Hermitian** $A$. Returns **real** eigenvalues in **ascending** order and an **orthonormal** eigenvector matrix. Faster and numerically stabler. Use it for Hessians, $A^\top A$, Gram/covariance matrices.
- `torch.linalg.eig(A)` / `np.linalg.eig(A)` — for **general** square $A$. Returns possibly **complex** eigenvalues in **no guaranteed order** and eigenvectors that are **not** orthonormal.

Rule of thumb: if you *know* the matrix is symmetric, always call `eigh` — calling `eig` on a symmetric matrix can hand you tiny imaginary parts and an unordered spectrum.

```python
import torch

A = torch.tensor([[2., 1.], [1., 3.]])          # symmetric

evals, evecs = torch.linalg.eigh(A)             # ascending, orthonormal, real
Lam = torch.diag(evals)
recon = evecs @ Lam @ evecs.T                   # Q Λ Qᵀ
print("reconstruction error:", torch.norm(recon - A).item())

# spectral norm two ways: σ₁ = largest singular value = sqrt(λ_max(AᵀA))
sigma_svd = torch.linalg.svdvals(A)[0]
sigma_eig = torch.sqrt(torch.linalg.eigvalsh(A.T @ A)[-1])
print("σ₁ via svd:", sigma_svd.item(), " via eig(AᵀA):", sigma_eig.item())
```

**You've got this piece when you can** state when to use `eigh` vs `eig` and what each guarantees about ordering/orthonormality/realness, reconstruct $A$ from its eigenpairs and check the error, and compute the spectral norm both from singular values and from $\lambda_{\max}(A^\top A)$.

# 📝 Review quiz

1. **(Concept)** Define an eigenpair $(\lambda, v)$ of $A$. Why must $v$ be nonzero, and what does the pair mean geometrically about how $A$ acts along $v$?
2. **(Derivation)** Starting from $Av=\lambda v$, derive the condition $\det(A-\lambda I)=0$ and explain each logical step (what "singular" buys you).
3. **(Computation)** For $A=\begin{pmatrix}2&1\\1&3\end{pmatrix}$, write the characteristic polynomial and find both eigenvalues. Check your answer against $\operatorname{trace}(A)$ and $\det(A)$.
4. **(Concept)** State precisely when a square matrix is diagonalizable. Give a matrix that is **not** diagonalizable and explain which multiplicities disagree. How does this differ from the existence conditions for the SVD?
5. **(Derivation)** For symmetric $A$, why can we write $A=Q\Lambda Q^\top$ instead of $Q\Lambda Q^{-1}$? What two properties does the spectral theorem guarantee, and what does $A=\sum_i\lambda_i q_iq_i^\top$ say geometrically?
6. **(Derivation)** Show $A^k=Q\Lambda^kQ^{-1}$. Then, expanding a start vector $x=\sum_i c_iq_i$ with $|\lambda_1|>|\lambda_2|\ge\cdots$, explain why $A^kx$ aligns with $q_1$ and what controls the convergence rate.
7. **(Code reading)** In the `power_iteration` loop, why do we renormalize $v$ every step, and why is `v @ A @ v` (the Rayleigh quotient) a valid estimate of $\lambda_1$ once $v$ has converged to $q_1$?
8. **(Code / applied)** You have a symmetric matrix and want its spectral norm $\sigma_1$. Give two ways to compute it (one via singular values, one via an eigenvalue routine on $A^\top A$), and say why you'd reach for `eigh` rather than `eig` here.

> [!note]- 🔑 Answer key (click to reveal)
> **1.** $(\lambda,v)$ is an eigenpair of $A$ if $Av=\lambda v$ with $v\neq 0$. It must be nonzero because $A\cdot 0=\lambda\cdot 0$ holds for *every* $\lambda$, so the zero vector carries no information — the equation would be vacuous. Geometrically, $A$ leaves the *direction* of $v$ unchanged and merely scales it by $\lambda$ (flipping it if $\lambda<0$, fixing it if $\lambda=1$, collapsing it if $\lambda=0$). Eigenvectors are the special directions on which $A$ acts as pure scaling rather than rotation+scaling.
>
> **2.** $Av=\lambda v \iff Av-\lambda v = 0 \iff (A-\lambda I)v = 0$. This is a homogeneous linear system; it has a **nonzero** solution $v$ iff the matrix $A-\lambda I$ is **singular** (has a nontrivial null space / non-invertible / columns linearly dependent). A square matrix is singular iff its determinant is zero, so $\det(A-\lambda I)=0$. The chain is: nonzero eigenvector exists ⟺ nontrivial null space ⟺ singular ⟺ determinant zero.
>
> **3.** $\det(A-\lambda I)=\det\begin{pmatrix}2-\lambda&1\\1&3-\lambda\end{pmatrix}=(2-\lambda)(3-\lambda)-1=\lambda^2-5\lambda+5$. Roots: $\lambda=\tfrac{5\pm\sqrt{25-20}}{2}=\tfrac{5\pm\sqrt5}{2}$, i.e. $\lambda_1\approx 3.618$, $\lambda_2\approx 1.382$. Checks: sum $=5=\operatorname{trace}(A)=2+3$ ✓; product $=\tfrac{(5)^2-5}{4}=\tfrac{20}{4}=5=\det(A)=2\cdot3-1\cdot1$ ✓.
>
> **4.** $A\in\mathbb{R}^{n\times n}$ is diagonalizable iff it has $n$ linearly independent eigenvectors — equivalently, for every eigenvalue the **geometric multiplicity** (dimension of its eigenspace) equals its **algebraic multiplicity** (its multiplicity as a root of $p(\lambda)$). Having $n$ *distinct* eigenvalues is sufficient but not necessary. Example not diagonalizable: $\begin{pmatrix}1&1\\0&1\end{pmatrix}$ has $\lambda=1$ with algebraic multiplicity 2 but only a 1-dimensional eigenspace (geometric multiplicity 1) — defective. The SVD $G=U\Sigma V^\top$ needs *no* such condition: it exists for **any** matrix, including non-square and defective ones, because it diagonalizes with two different orthogonal bases rather than one.
>
> **5.** The spectral theorem guarantees a real symmetric $A$ has (i) all **real** eigenvalues and (ii) a full set of **orthonormal** eigenvectors. Choosing $Q$ with those orthonormal eigenvectors as columns makes $Q$ orthogonal, so $Q^\top Q=I\Rightarrow Q^{-1}=Q^\top$, hence $A=Q\Lambda Q^{-1}=Q\Lambda Q^\top$. The expansion $A=\sum_i\lambda_i q_iq_i^\top$ says $A$ is a sum of rank-1 orthogonal projectors $q_iq_i^\top$ each scaled by $\lambda_i$: $A$ stretches space by factor $\lambda_i$ along the mutually perpendicular axis $q_i$ and does nothing to mix those axes.
>
> **6.** $A^2=(Q\Lambda Q^{-1})(Q\Lambda Q^{-1})=Q\Lambda(Q^{-1}Q)\Lambda Q^{-1}=Q\Lambda^2Q^{-1}$; induction gives $A^k=Q\Lambda^kQ^{-1}$ with $\Lambda^k=\operatorname{diag}(\lambda_i^k)$. Writing $x=\sum_i c_iq_i$ gives $A^kx=\sum_i c_i\lambda_i^k q_i=\lambda_1^k\big(c_1q_1+\sum_{i\ge2}c_i(\lambda_i/\lambda_1)^k q_i\big)$. Since $|\lambda_i/\lambda_1|<1$ for $i\ge2$, those terms decay geometrically to 0, so the direction converges to $q_1$ (provided $c_1\neq0$). The convergence rate is governed by the **spectral gap** $|\lambda_2/\lambda_1|$ — smaller ratio ⇒ faster.
>
> **7.** We renormalize $v\leftarrow w/\|w\|$ each step because the raw iterate $A^kv$ scales like $\lambda_1^k$, which overflows (or underflows if $|\lambda_1|<1$); we only care about the *direction*, so normalizing keeps it numerically bounded without changing which eigenvector it points to. Once $v\approx q_1$ (unit norm), the **Rayleigh quotient** $v^\top A v = v^\top(\lambda_1 v)=\lambda_1\,v^\top v=\lambda_1$, so `v @ A @ v` reads off the eigenvalue. (For a unit $v$ the quotient is $\tfrac{v^\top Av}{v^\top v}=v^\top Av$.)
>
> **8.** Way 1: $\sigma_1=$ largest singular value $=$ `torch.linalg.svdvals(A)[0]`. Way 2: $\sigma_1=\sqrt{\lambda_{\max}(A^\top A)}=$ `sqrt(eigvalsh(A.T@A)[-1])`, because $A^\top A=V\Sigma^2V^\top$ so its top eigenvalue is $\sigma_1^2$. You use **`eigh`/`eigvalsh`** (not `eig`) because $A^\top A$ is symmetric PSD: `eigh` returns real, ascending eigenvalues and an orthonormal basis and is faster/stabler, whereas `eig` may return complex values with spurious imaginary parts and no ordering guarantee.
