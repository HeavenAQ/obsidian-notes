---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-14T12:51:01
Status: Done
Last updated time: 2026-07-14T12:56:37
Last edited by: Heaven Chen
Category:
  - ML
  - Classification
  - Theory
  - Survey
---
> **Support Vector Machines and Relevance Vector Machines, from geometry to Bayesian sparsity.** This note derives the binary SVM primal and dual, explains kernels and support vectors, derives the RVM’s automatic relevance determination and Laplace approximation, and shows how both families extend to multiclass classification. It includes practical scikit-learn SVM pipelines and a transparent educational RVM implementation.

> [!info] Scope and evidence
> Snapshot as of **2026-07-14**. The SVM sections follow Cortes & Vapnik and the current scikit-learn/libsvm behavior; the RVM derivation follows Tipping’s original JMLR paper. The RVM code is intentionally educational: it makes the mathematics visible but is not a production replacement for a mature, numerically optimized library.

# 0. The one-paragraph story

Both SVM and RVM build a kernel expansion

$$
f(x)=b+\sum_i w_iK(x,x_i),
$$

but they decide which coefficients survive for completely different reasons. An **SVM** is a discriminative maximum-margin classifier: it solves a convex optimization problem and retains training points whose constraints are active—the **support vectors**. An **RVM** is a sparse Bayesian generalized linear model: it places an independent precision $\alpha_i$ on each coefficient, estimates those precisions from the marginal likelihood, and prunes bases whose posterior weights collapse to zero—the **relevance vectors**. SVM gives a robust separating score but not a native probability; RVM gives a posterior predictive probability and often a much sparser kernel expansion, but training is slower, non-convex in its hyperparameters, and more delicate.

```mermaid
flowchart LR
    D[Training examples] --> K[Kernel basis K(x, x_i)]
    K --> S[SVM]
    K --> R[RVM]
    S --> SM[Maximize margin<br/>penalize hinge violations]
    SM --> SV[Nonzero dual coefficients<br/>support vectors]
    SV --> SS[Signed decision score]
    R --> ARD[ARD Gaussian priors<br/>one precision per weight]
    ARD --> RV[Finite precisions<br/>relevance vectors]
    RV --> PP[Posterior class probability]

    classDef shared fill:#ede7f6,stroke:#5e35b1,color:#111;
    classDef svm fill:#e3f2fd,stroke:#1e88e5,color:#111;
    classDef rvm fill:#e8f5e9,stroke:#43a047,color:#111;
    class D,K shared;
    class S,SM,SV,SS svm;
    class R,ARD,RV,PP rvm;
```

# 1. What problems do they address?

## 1.1 The shared problem

Given labeled observations $(x_i,y_i)$, learn a boundary that generalizes when:

- features may be numerous relative to samples;
- the useful boundary can be nonlinear;
- only some training examples may be needed at inference;
- a controlled notion of complexity is more important than exactly fitting every row.

The **kernel trick** lets the learner act as if it used a nonlinear feature map $\phi(x)$ without constructing it explicitly:

$$
K(x,z)=\langle\phi(x),\phi(z)\rangle.
$$

## 1.2 What SVM specifically fixes

Many linear separators may classify separable training data perfectly. SVM chooses the one with the **largest geometric margin**, making the decision rule depend on the closest or margin-violating cases rather than every training point. Soft-margin SVM then handles overlap and mislabeled examples by trading margin width against hinge-loss violations.

## 1.3 What RVM specifically fixes

An SVM decision value is not a probability, and kernel SVMs may retain many support vectors. RVM addresses these issues by:

- specifying a Bernoulli/multinomial probabilistic model;
- placing sparse Bayesian priors on basis weights;
- producing posterior uncertainty over weights and predictive probabilities;
- commonly retaining fewer kernel centers.

This is not a free dominance result. RVM trades the SVM’s convex training problem for iterative approximate Bayesian inference and type-II maximum-likelihood optimization, which can find local optima.

# 2. SVM geometry: maximize the margin

## 2.1 Hyperplane and signed distance

For binary labels $y_i\in\{-1,+1\}$, a linear classifier is

$$
f(x)=w^Tx+b,
\qquad
\hat y=\operatorname{sign}(f(x)).
$$

The signed distance from $x$ to the hyperplane $w^Tx+b=0$ is

$$
\frac{w^Tx+b}{\lVert w\rVert_2}.
$$

Because multiplying $(w,b)$ by a positive constant leaves the boundary unchanged, impose the canonical scaling

$$
y_i(w^Tx_i+b)\ge1.
$$

The two supporting planes $w^Tx+b=\pm1$ are separated by

$$
\frac{2}{\lVert w\rVert_2}.
$$

Maximizing margin is therefore equivalent to minimizing $\frac12\lVert w\rVert^2$.

![[99 Assets/Media/svm-rvm-deep-dive/svm-maximum-margin.png]]

*The solid line is the maximum-margin boundary, dashed lines are $f(x)=\pm1$, and circled observations are support vectors. Moving a distant non-support vector slightly does not change the solution; moving a support vector can. Source: [scikit-learn maximum-margin example](https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html), BSD-3-Clause.*

## 2.2 Hard-margin primal

For perfectly separable data:

$$
\begin{aligned}
\min_{w,b}\quad & \frac12\lVert w\rVert_2^2\\
\text{s.t.}\quad & y_i(w^Tx_i+b)\ge1,\quad i=1,\ldots,n.
\end{aligned}
$$

This is convex, but infeasible when classes overlap or contradictory labels exist.

## 2.3 Soft margin and hinge loss

Introduce slack $\xi_i\ge0$:

$$
\begin{aligned}
\min_{w,b,\xi}\quad & \frac12\lVert w\rVert_2^2+C\sum_{i=1}^{n}\xi_i\\
\text{s.t.}\quad & y_i(w^Tx_i+b)\ge1-\xi_i.
\end{aligned}
$$

Equivalently,

$$
\min_{w,b}\frac12\lVert w\rVert_2^2+
C\sum_i\max\left(0,1-y_if(x_i)\right).
$$

The second term is hinge loss.

| Point condition | Hinge loss | Role |
| --- | ---: | --- |
| $y_if_i>1$ | $0$ | Correct and outside margin |
| $y_if_i=1$ | $0$ | On margin; commonly support vector |
| $0<y_if_i<1$ | $0<\ell<1$ | Correct but inside margin |
| $y_if_i<0$ | $>1$ | Misclassified |

`C` is inverse regularization strength: large `C` heavily penalizes violations and tends toward a narrower, more data-fitting margin; small `C` permits more violations for a wider, smoother margin.

# 3. The SVM dual and why support vectors appear

## 3.1 Lagrangian

For the soft-margin constraints, introduce multipliers $\alpha_i\ge0$ and $\mu_i\ge0$. Eliminating $w,b,$ and $\xi$ yields

$$
\begin{aligned}
\max_{\alpha}\quad &
\sum_{i=1}^{n}\alpha_i-
\frac12\sum_{i=1}^{n}\sum_{j=1}^{n}
\alpha_i\alpha_jy_iy_jx_i^Tx_j\\
\text{s.t.}\quad &0\le\alpha_i\le C,\\
&\sum_i\alpha_iy_i=0.
\end{aligned}
$$

At the solution,

$$
w=\sum_i\alpha_iy_ix_i.
$$

Only examples with $\alpha_i>0$ contribute. These are support vectors.

## 3.2 KKT interpretation

Complementary slackness gives a useful diagnostic:

- $\alpha_i=0$: safely outside the margin;
- $0<\alpha_i<C$: exactly on the margin under nondegenerate conditions;
- $\alpha_i=C$: inside the margin or misclassified.

The prediction rule becomes

$$
f(x)=b+\sum_{i\in\mathrm{SV}}\alpha_iy_iK(x_i,x).
$$

Training depends on an $n\times n$ kernel matrix for nonlinear SVC, and prediction cost grows with the number of support vectors.

# 4. Kernels: nonlinear boundaries without explicit expansion

## 4.1 Common kernels

$$
K_{\text{linear}}(x,z)=x^Tz
$$

$$
K_{\text{poly}}(x,z)=(\gamma x^Tz+r)^d
$$

$$
K_{\text{RBF}}(x,z)=\exp(-\gamma\lVert x-z\rVert_2^2)
$$

The RBF length scale is proportional to $1/\sqrt{\gamma}$. Large $\gamma$ creates highly local influence and flexible boundaries; small $\gamma$ creates broad, smooth influence.

![[99 Assets/Media/svm-rvm-deep-dive/svm-kernel-boundaries-xor.png]]

*Linear, polynomial, RBF, and sigmoid SVMs on an XOR-like dataset. The picture demonstrates expressivity, not comparative test accuracy: `C`, `gamma`, feature scale, and validation determine the useful boundary. Source: [scikit-learn kernel example](https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html), BSD-3-Clause.*

## 4.2 Why scaling matters

RBF and polynomial kernels are functions of distances or dot products. A feature measured in thousands can dominate another measured in fractions. Standardize numeric features **inside each training fold**:

$$
z_j=\frac{x_j-\mu_{j,\text{train}}}{s_{j,\text{train}}}.
$$

Tree classifiers usually do not need scaling; SVM/RVM kernel models usually do.

## 4.3 Valid kernels

For the classical SVM dual to remain a convex quadratic program, the Gram matrix should be positive semidefinite—equivalently, the kernel must satisfy the appropriate Mercer condition. Tipping’s RVM formulation is a basis-function model and does not formally require a Mercer kernel, although indefinite or poorly conditioned bases can still cause numerical and modeling problems.

# 5. Multiclass SVM

A binary SVM does not by itself choose among $K>2$ classes. There are three main constructions.

## 5.1 One-vs-rest (OvR)

Train $K$ binary classifiers:

$$
f_k(x):\quad y=k\ \text{versus}\ y\ne k,
\qquad
\hat y=\arg\max_k f_k(x).
$$

**Cost:** $K$ models. **Advantage:** simple, prediction is cheap, well suited to linear models. **Risk:** each task can be imbalanced, and scores from independently trained classifiers may not be directly comparable.

## 5.2 One-vs-one (OvO)

Train one classifier for every class pair:

$$
\binom{K}{2}=\frac{K(K-1)}{2}.
$$

Each model sees only examples from its two classes. Votes or coupled pairwise probabilities produce the final answer. Kernel `SVC` uses OvO internally; its displayed `decision_function_shape="ovr"` can transform the OvO results into one score per class.

## 5.3 Direct Crammer–Singer multiclass margin

Learn one weight vector $w_k$ for each class and require the correct score to exceed every incorrect score:

$$
w_{y_i}^Tx_i-w_k^Tx_i\ge1-\xi_i,
\qquad \forall k\ne y_i.
$$

A common primal is

$$
\min_{W,\xi}\frac12\sum_{k=1}^{K}\lVert w_k\rVert^2+C\sum_i\xi_i
$$

subject to the constraints above. Equivalently the per-example loss is

$$
\ell_i=\max_{k\ne y_i}\left[1+w_k^Tx_i-w_{y_i}^Tx_i\right]_+.
$$

This optimizes the classes jointly instead of combining binary classifiers, but availability and scalability depend on the implementation.

![[99 Assets/Media/svm-rvm-deep-dive/svm-multiclass-iris.png]]

*Linear SVC, kernel SVC, RBF SVC, and polynomial SVC on a three-class Iris slice. Multiclass strategy and kernel geometry are separate choices: OvO/OvR determines decomposition, while the kernel determines each boundary’s shape. Source: [scikit-learn Iris SVM example](https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html), BSD-3-Clause.*

# 6. SVM scores are not probabilities

The raw $f(x)$ is a signed margin score. Its magnitude is useful for ranking but is not a calibrated posterior probability. Platt scaling fits

$$
P(y=1\mid f)=\frac{1}{1+\exp(Af+B)}.
$$

For multiclass OvO SVMs, calibrated pairwise probabilities must be coupled into a coherent vector that sums to one. In scikit-learn, `SVC(probability=True)` performs extra internal cross-validation and pairwise coupling.

> [!warning] Grouped-data calibration
> `SVC(probability=True)` does not accept your subject IDs as a grouping rule for its internal calibration split. For repeated participant data, prefer `probability=False` plus `CalibratedClassifierCV` supplied with precomputed group-disjoint splits.

# 7. RVM: same expansion, different learning principle

## 7.1 Basis-function model

With one basis centered on each training example,

$$
y(x;w)=w_0+\sum_{i=1}^{N}w_iK(x,x_i)=w^T\phi(x),
$$

where

$$
\phi(x)=[1,K(x,x_1),\ldots,K(x,x_N)]^T.
$$

The similar prediction form is why SVM and RVM are visually easy to confuse. The coefficient-learning objective is entirely different.

## 7.2 Automatic relevance determination prior

RVM gives every coefficient its own zero-mean Gaussian prior:

$$
p(w\mid\alpha)=\prod_{j=0}^{N}\mathcal N(w_j\mid0,\alpha_j^{-1}),
\qquad
A=\operatorname{diag}(\alpha_0,\ldots,\alpha_N).
$$

`alpha` is a **precision**, the inverse of variance. If evidence pushes $\alpha_j\rightarrow\infty$, then

$$
p(w_j\mid\alpha_j)\rightarrow\delta(w_j),
$$

so that basis can be removed. A training example whose kernel basis survives is a relevance vector.

## 7.3 Binary classification likelihood

For targets $t_i\in\{0,1\}$,

$$
p_i=P(t_i=1\mid x_i,w)=\sigma(\phi_i^Tw),
$$

$$
p(t\mid w)=\prod_{i=1}^{N}p_i^{t_i}(1-p_i)^{1-t_i}.
$$

Unlike Gaussian regression, the weight posterior is not analytically Gaussian. The original RVM uses a Laplace approximation.

# 8. RVM inference step by step

## 8.1 Posterior mode

For fixed $\alpha$, maximize the penalized logistic log posterior

$$
\mathcal J(w)=
\sum_i[t_i\log p_i+(1-t_i)\log(1-p_i)]-
\frac12w^TAw.
$$

Its gradient and negative Hessian are

$$
\nabla\mathcal J=\Phi^T(t-p)-Aw,
$$

$$
H=\Phi^TB\Phi+A,
\qquad
B=\operatorname{diag}(p_i(1-p_i)).
$$

Newton/IRLS updates the mode:

$$
w_{\text{new}}=w+H^{-1}\nabla\mathcal J.
$$

## 8.2 Laplace covariance

Approximate the posterior around the mode $w_{MP}$:

$$
p(w\mid t,\alpha)\approx
\mathcal N(w\mid w_{MP},\Sigma),
\qquad
\Sigma=(\Phi^TB\Phi+A)^{-1}.
$$

## 8.3 Evidence updates and pruning

Define the effective degree of determination

$$
\gamma_j=1-\alpha_j\Sigma_{jj}.
$$

The fixed-point update is

$$
\alpha_j^{\text{new}}=\frac{\gamma_j}{w_{MP,j}^2}.
$$

Small or unstable $w_j$ makes $\alpha_j$ grow; the prior tightens; the coefficient shrinks further; eventually the basis is pruned. Iterate posterior-mode and evidence updates until the active set stabilizes.

## 8.4 Predictive probability

For test basis vector $\phi_*$, the latent score has approximate mean and variance

$$
\mu_*=\phi_*^Tw_{MP},
\qquad
s_*^2=\phi_*^T\Sigma\phi_*.
$$

The logistic-Gaussian integral has no elementary closed form. A common approximation is

$$
P(t_*=1\mid x_*,D)\approx
\sigma\left(\frac{\mu_*}{\sqrt{1+\pi s_*^2/8}}\right).
$$

The variance pulls uncertain probabilities toward $0.5$.

![[99 Assets/Media/svm-rvm-deep-dive/rvm-paper-page-figure3.png|750]]

*Figure 3 of Tipping’s original paper compares an SVM using 38 support vectors with an RVM using four relevance vectors on the illustrated synthetic problem. This is a demonstration of possible sparsity, not a universal ratio or guarantee. Source: Tipping (2001), [JMLR open-access paper](https://www.jmlr.org/papers/v1/tipping01a.html), p. 222.*

# 9. Multiclass RVM

## 9.1 Coupled multinomial formulation

The original RVM generalizes naturally to $K$ logits

$$
y_k(x)=w_k^T\phi(x),
$$

with softmax probabilities

$$
p_{ik}=\frac{\exp(y_k(x_i))}{\sum_{\ell=1}^{K}\exp(y_\ell(x_i))}.
$$

Using one-hot targets $t_{ik}$,

$$
p(T\mid W)=\prod_{i=1}^{N}\prod_{k=1}^{K}p_{ik}^{t_{ik}}.
$$

Each output may have its own ARD precisions $\alpha_{jk}$, or precisions can be shared across classes to encourage a common relevance-vector set. Laplace inference uses a block Hessian coupling class logits. One class must be treated as a reference, or an equivalent identifiability constraint imposed, because adding the same constant to every logit does not change softmax probabilities.

**Benefit:** one coherent probabilistic model; no heuristic voting. **Cost:** the covariance/Hessian grows with $K$ and active bases, making inference substantially more expensive.

## 9.2 Practical decompositions

When a coupled implementation is unavailable:

- **OvR RVM:** fit $K$ binary RVMs, then normalize their class probabilities. Easy, but independent probabilities are not a true joint posterior.
- **OvO RVM:** fit $K(K-1)/2$ models and apply pairwise coupling. Usually more expensive and less natural than OvR for an already probabilistic learner.
- **Shared-basis multiclass RVM:** statistically attractive but requires a dedicated solver.

Do not compare independent OvR scores without calibration or normalization.

# 10. SVM versus RVM

| Property | SVM | RVM |
| --- | --- | --- |
| Principle | Maximum margin + hinge loss | Sparse Bayesian logistic/multinomial model |
| Optimization | Convex QP for fixed kernel/parameters | Non-convex evidence optimization + approximate inference |
| Sparse unit | Active constraint / support vector | Basis with finite ARD precision / relevance vector |
| Output | Signed margin score | Posterior predictive probability |
| Probability | Added afterward by calibration | Native to likelihood, approximate under Laplace |
| Kernel condition | PSD/Mercer for standard convex dual | Basis need not formally be Mercer |
| Hyperparameters | `C`, kernel parameters | Kernel parameters, priors, inference/pruning controls |
| Typical sparsity | Can retain many SVs with overlap | Often substantially fewer RVs, not guaranteed |
| Training maturity | Highly optimized libraries | Fewer maintained general-purpose implementations |
| Failure character | Sensitive to scaling, `C`, `gamma`; slow at large $n$ | Slow matrix algebra, local optima, numerical pruning sensitivity |

> [!important] The names are not interchangeable
> A relevance vector is not “the Bayesian version of the same support vector.” Support vectors often lie on or inside the margin; relevance vectors can lie away from the boundary because they are selected as useful basis centers under the posterior evidence.

# 11. Practical SVM code

## 11.1 Binary or multiclass kernel SVM

`SVC` handles binary and multiclass labels; its kernel training is OvO internally.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

svm_model = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="rbf",
        C=3.0,
        gamma="scale",
        class_weight="balanced",
        decision_function_shape="ovr",  # returned score layout; training remains OvO
        probability=False,
        random_state=42,
    ),
)

svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
scores = svm_model.decision_function(X_test)
```

## 11.2 Explicit OvR and OvO

```python
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC

binary_svc = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced"),
)

ovr = OneVsRestClassifier(binary_svc, n_jobs=-1).fit(X_train, y_train)
ovo = OneVsOneClassifier(binary_svc, n_jobs=-1).fit(X_train, y_train)

pred_ovr = ovr.predict(X_test)
pred_ovo = ovo.predict(X_test)
```

## 11.3 Group-aware calibrated probabilities

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
group_splits = list(cv.split(X_train, y_train, groups=subject_train))

base_svm = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced"),
)

calibrated_svm = CalibratedClassifierCV(
    estimator=base_svm,
    method="sigmoid",
    cv=group_splits,
)
calibrated_svm.fit(X_train, y_train)
p_test = calibrated_svm.predict_proba(X_test)
```

The test subjects must remain untouched by scaling, hyperparameter selection, calibration, and threshold selection.

# 12. Educational binary RVM implementation

This direct implementation uses an RBF basis at every training row, Newton iterations for the Laplace mode, ARD fixed-point updates, and pruning. It favors readability over speed. Initial cost is roughly cubic in the number of active bases because it repeatedly factors dense Hessians.

```python
import numpy as np

from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class BinaryRVMClassifier(ClassifierMixin, BaseEstimator):
    """Educational RBF relevance-vector classifier (binary only)."""

    def __init__(
        self,
        gamma=1.0,
        alpha_init=1.0,
        alpha_max=1e9,
        max_iter=200,
        newton_iter=50,
        tol=1e-4,
        jitter=1e-8,
    ):
        self.gamma = gamma
        self.alpha_init = alpha_init
        self.alpha_max = alpha_max
        self.max_iter = max_iter
        self.newton_iter = newton_iter
        self.tol = tol
        self.jitter = jitter

    def _rbf(self, X, Z):
        d2 = (
            np.sum(X * X, axis=1)[:, None]
            + np.sum(Z * Z, axis=1)[None, :]
            - 2.0 * X @ Z.T
        )
        return np.exp(-self.gamma * np.maximum(d2, 0.0))

    def _design(self, X, centers):
        return np.column_stack([np.ones(X.shape[0]), self._rbf(X, centers)])

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=float)
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("BinaryRVMClassifier requires exactly two classes")

        t = (y == self.classes_[1]).astype(float)
        centers = X.copy()
        basis_indices = np.arange(X.shape[0])
        Phi = self._design(X, centers)

        # Weakly regularize the intercept; use one ARD precision per RBF basis.
        alpha = np.r_[1e-8, np.full(X.shape[0], self.alpha_init)]
        w = np.zeros(Phi.shape[1])

        for outer in range(self.max_iter):
            # Find the posterior mode for the current alpha by Newton/IRLS.
            for _ in range(self.newton_iter):
                p = np.clip(expit(Phi @ w), 1e-9, 1.0 - 1e-9)
                beta = p * (1.0 - p)
                H = Phi.T @ (beta[:, None] * Phi) + np.diag(alpha)
                H.flat[:: H.shape[0] + 1] += self.jitter
                grad = Phi.T @ (t - p) - alpha * w
                step = np.linalg.solve(H, grad)
                w = w + step
                if np.max(np.abs(step)) < self.tol:
                    break

            Sigma = np.linalg.inv(H)
            effective = np.clip(1.0 - alpha * np.diag(Sigma), 0.0, 1.0)
            alpha_new = effective / (w * w + 1e-32)
            alpha_new = np.clip(alpha_new, 1e-12, self.alpha_max)
            alpha_new[0] = 1e-8  # keep intercept

            # Prune RBF bases whose precision has effectively diverged.
            keep = alpha_new < self.alpha_max
            keep[0] = True
            delta = np.max(
                np.abs(np.log(alpha_new[keep]) - np.log(alpha[keep]))
            )

            basis_indices = basis_indices[keep[1:]]
            centers = centers[keep[1:]]
            Phi = Phi[:, keep]
            w = w[keep]
            alpha = alpha_new[keep]

            if delta < self.tol:
                break

        # Recompute covariance for the final active set.
        p = np.clip(expit(Phi @ w), 1e-9, 1.0 - 1e-9)
        beta = p * (1.0 - p)
        H = Phi.T @ (beta[:, None] * Phi) + np.diag(alpha)
        H.flat[:: H.shape[0] + 1] += self.jitter

        self.relevance_vectors_ = centers
        self.relevance_indices_ = basis_indices
        self.coef_ = w
        self.alpha_ = alpha
        self.sigma_ = np.linalg.inv(H)
        self.n_iter_ = outer + 1
        return self

    def decision_function(self, X):
        check_is_fitted(self, "relevance_vectors_")
        X = check_array(X, dtype=float)
        Phi = self._design(X, self.relevance_vectors_)
        return Phi @ self.coef_

    def predict_proba(self, X):
        check_is_fitted(self, "relevance_vectors_")
        X = check_array(X, dtype=float)
        Phi = self._design(X, self.relevance_vectors_)
        mean = Phi @ self.coef_
        variance = np.einsum("ij,jk,ik->i", Phi, self.sigma_, Phi)
        adjusted = mean / np.sqrt(1.0 + np.pi * variance / 8.0)
        p1 = expit(adjusted)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        index = (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        return self.classes_[index]
```

Always standardize inputs before this RBF implementation. For stability, production solvers use faster sequential basis addition/deletion, Cholesky updates, damping, line search, and more careful convergence tests.

# 13. Multiclass RVM code via OvR

The self-contained binary estimator can be wrapped with scikit-learn’s OvR meta-estimator:

```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

multiclass_rvm = make_pipeline(
    StandardScaler(),
    OneVsRestClassifier(
        BinaryRVMClassifier(
            gamma=0.5,
            alpha_init=1.0,
            max_iter=200,
        ),
        n_jobs=1,  # dense matrix work can already use threaded BLAS
    ),
)

multiclass_rvm.fit(X_train, y_train)
y_pred = multiclass_rvm.predict(X_test)
p_test = multiclass_rvm.predict_proba(X_test)  # OvR probabilities normalized
```

This is **not** the coupled multinomial RVM from Section 9.1. It is a practical decomposition of $K$ independent binary posterior approximations.

# 14. Validation and tuning

## 14.1 SVM tuning order

1. Put standardization and any imputation inside the pipeline.
2. Fix group-aware outer folds and the target metric.
3. Test linear SVM first for high-dimensional features.
4. For RBF SVM, tune `C` and `gamma` jointly on logarithmic grids.
5. Apply class weights only if aligned with the real cost/prevalence problem.
6. Calibrate on held-out group folds if probabilities are required.

A compact search is

```python
param_grid = {
    "svc__C": [1e-2, 1e-1, 1, 10, 100],
    "svc__gamma": ["scale", 1e-3, 1e-2, 1e-1, 1],
}
```

Adjust the parameter prefix to the actual pipeline step name.

## 14.2 RVM tuning order

1. Standardize features and inspect duplicate/near-duplicate rows.
2. Tune the kernel width using nested validation; ARD does not choose the RBF scale automatically in this implementation.
3. Monitor active-basis count, convergence, and condition numbers.
4. Repeat from multiple initializations if the solver allows it.
5. Evaluate probability calibration; “Bayesian” does not guarantee calibration under approximate inference or dataset shift.
6. Compare against logistic regression, calibrated SVM, and Gaussian-process classification where feasible.

## 14.3 Metrics

- Binary imbalance: AUPRC, balanced accuracy, sensitivity/specificity, log loss, Brier score.
- Multiclass: macro-F1, macro one-vs-rest AUROC/AUPRC where meaningful, multiclass log loss, per-class recall.
- Always report fold/subject dispersion and the number of support/relevance vectors.

# 15. Failure modes and practical boundaries

## SVM

- Kernel `SVC` training scales poorly with sample count and stores an $n\times n$-like kernel workload.
- Bad feature scaling makes distance-based kernels meaningless.
- Large `C` plus large `gamma` can memorize local noise.
- Many overlapping cases can become support vectors, making prediction slow.
- Raw margins are not probabilities.
- OvR scores can be incomparable; OvO grows quadratically in class count.

## RVM

- Evidence optimization is non-convex and initialization-sensitive.
- Dense Hessian inversion is expensive before pruning.
- Laplace inference is an approximation; uncertainty can be optimistic.
- Aggressive thresholds can prune useful bases or create numerical discontinuities.
- Extreme separation in logistic likelihood needs strong numerical safeguards.
- Multiclass coupled covariance scales poorly with both active bases and classes.
- Fewer relevance vectors does not guarantee better accuracy or trustworthy uncertainty.

# 16. Which should you use?

| Situation | Recommended starting point |
| --- | --- |
| High-dimensional sparse text/embeddings | Linear SVM or logistic regression |
| Small/medium nonlinear numeric data | RBF SVM with nested group CV |
| Need native probabilistic kernel model and extreme sparsity | RVM, if solver maturity and training cost are acceptable |
| Need calibrated probabilities reliably | Calibrated linear/RBF SVM; compare RVM and logistic regression |
| Many classes | Linear OvR SVM; evaluate direct multiclass methods |
| Large $n$ | Linear/approximate kernels or tree boosting—not full kernel SVM/RVM |
| Subject-wise multimodal thesis data | Group-CV linear SVM baseline, then RBF SVM; treat RVM as an additional probabilistic sparse model |

For the thesis context, SVM is the defensible standard baseline. RVM is most valuable when the research question explicitly concerns sparse prototypes or predictive uncertainty and the smaller sample size makes its matrix computations feasible.

# 17. Related vault notes

- [[04 Reference/Document Hub/Tree-Based Classification Deep Dive — Decision Trees, Random Forests, XGBoost, LightGBM & CatBoost]] — complementary tree-ensemble classifiers and the same group-validation concerns.
- [[Evaluating the Robustness of Multimodal Task Load Estimation Models]] — compares SVM/XGBoost under subject-wise nested validation and robustness metrics.
- [[Multi-Source Domain Generalization for ECG-Based Cognitive Load Estimation A Plug-in Method and Benchmark]] — domain generalization and classical classifier baselines.
- [[ADABase A Multimodal Dataset for Cognitive Load Estimation]] — multimodal cognitive-load classification context.

# 18. Primary sources and official references

## Foundational papers

- Cortes, C. & Vapnik, V. (1995), [Support-Vector Networks](https://doi.org/10.1007/BF00994018).
- Tipping, M. E. (2001), [Sparse Bayesian Learning and the Relevance Vector Machine](https://www.jmlr.org/papers/v1/tipping01a.html).
- Crammer, K. & Singer, Y. (2001), [On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines](https://www.jmlr.org/papers/v2/crammer01a.html).
- Wu, T.-F., Lin, C.-J. & Weng, R. C. (2004), [Probability Estimates for Multi-class Classification by Pairwise Coupling](https://www.jmlr.org/papers/v5/wu04a.html).

## Official implementation documentation

- [scikit-learn Support Vector Machines guide](https://scikit-learn.org/stable/modules/svm.html) — objectives, kernels, multiclass behavior, scores, and complexity.
- [scikit-learn multiclass strategies](https://scikit-learn.org/stable/modules/multiclass.html) — OvR, OvO, and related wrappers.
- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html) — reliability diagrams, sigmoid/isotonic calibration, and multiclass behavior.
- [LIBSVM paper and software](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) — the solver underlying scikit-learn `SVC`/`NuSVC`.

# 19. Final takeaway

SVM asks: **which separator leaves the largest regularized margin?** RVM asks: **which kernel bases retain posterior evidence after each coefficient receives its own shrinkage precision?** Their prediction equations look alike, but margin constraints, posterior inference, sparsity, probability semantics, optimization guarantees, and multiclass extensions differ. Use SVM when you need a mature, strong margin baseline; use RVM when native probabilistic sparse basis selection is itself valuable—and validate both with scaling, group boundaries, calibration, and honest outer folds.
