---
base: "[[01.1 Computer Vision — Foundations Study Tracker.base]]"
Key takeaways: "Learning = objective + hypothesis space + optimizer; ERM minimizes average training loss as a proxy for test loss, and is maximum likelihood in disguise (priors = regularizers). SGD with momentum, LR schedules, and clipping is the workhorse that trains everything later in the course. Generalization is the real goal: overfitting = fitting training noise, controlled by data, priors, and hypothesis-space constraints — which are interchangeable. Mixed precision training (FP32 master weights + loss scaling + FP32 accumulation) makes FP16 training match FP32 accuracy with no hyperparameter changes."
Day: 3
Status: In-Progress
Reading done: true
Chapters: Ch 9–11
Self-check done: false
Date: 2026-07-03
Part:
  - Foundations-of-Learning
Questions / Follow-ups: Why does SGD implicitly prefer minimum-norm solutions in linear problems (Ch 10 claim) — derivation? How does double descent (Belkin et al.) reconcile with the classical U-curve quantitatively? When does bf16 remove the need for loss scaling vs FP16?
---
**Reading checklist**

- [x] [9 Introduction to Learning](https://visionbook.mit.edu/intro_to_learning.html)
- [x] [10 Gradient-Based Learning Algorithms](https://visionbook.mit.edu/gradient_descent.html)
- [x] [11 The Problem of Generalization](https://visionbook.mit.edu/problem_of_generalization.html)

## Notes

## Self-check

# Day 3 Study Report — Learning I (Ch 9–11) · 2026-07-03 (JST)

# Chapter 9 — Introduction to Learning

## Intuition

Learning is a **meta-algorithm**: an algorithm whose output is another algorithm. The training phase searches for a function $f: \mathcal{X} \rightarrow \mathcal{Y}$ that performs well on past problem instances (training data); the testing phase deploys $f$ on new instances. The chapter's motivating example: given $2 \star 3 = 36$, $7 \star 1 = 49$, $5 \star 2 = 100$, $2 \star 2 = 16$, you infer $x \star y = (xy)^2$ — that inference from input/output examples IS supervised learning ("fitting a model to data"). Complicated mappings like image inpainting are functions too complex to write by hand, which is exactly why we make the machine write the algorithm from many examples.

Every learning algorithm has **three key ingredients**: (1) an **objective** (what counts as success), (2) a **hypothesis space** $\mathcal{F}$ (the set of candidate functions searched over), and (3) an **optimizer** (how the search is carried out). Applied to enough data and compute, these three ingredients are the whole story — and data and compute often matter more than the algorithm.

**Parameterization matters.** $y = \theta_1 x + \theta_0$ and $y = \theta_2\theta_1 x + \theta_0$ describe exactly the same space of affine functions, but optimizers treat the two parameterizations differently, so a learning algorithm is only fully defined once the parameterization is fixed. Overparameterized models (more parameters than minimally needed) are the norm in modern vision.

## Empirical Risk Minimization (ERM)

$\hat{f} = \mathop{\mathrm{arg\,min}}_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \mathcal{L}\big(f(\mathbf{x}^{(i)}), \mathbf{y}^{(i)}\big)$

Symbols: $\mathcal{F}$ is the hypothesis space; $\mathcal{L}: \mathcal{Y} \times \mathcal{Y} \rightarrow \mathbb{R}$ is the per-example loss; $\{\mathbf{x}^{(i)}, \mathbf{y}^{(i)}\}_{i=1}^N$ are the $N$ training pairs; $f$ is the learned function. **Why this form:** we cannot measure risk on future data, so we minimize the average loss on the *empirical* distribution (the training set) as a proxy — the entire problem of generalization (Ch 11) is about when this proxy is faithful.

## Learning as probabilistic inference

ERM often coincides with **maximum likelihood** estimation:

$\hat{f} = \mathop{\mathrm{arg\,max}}_f \; p\big(\{\mathbf{y}^{(i)}\}_{i=1}^N \mid \{\mathbf{x}^{(i)}\}_{i=1}^N, f\big)$

**Why:** if you posit Gaussian prediction errors $\mathbf{y} - f(\mathbf{x}) \sim \mathcal{N}(0, \sigma^2 I)$, the negative log-likelihood is (up to constants) exactly the sum of squared errors — the $L_2$ loss. Adding a prior $p(f)$ over hypotheses gives **MAP learning** via Bayes' rule:

$\mathop{\mathrm{arg\,max}}_f \; p\big(f \mid \{\mathbf{x}^{(i)}, \mathbf{y}^{(i)}\}_{i=1}^N\big) = \mathop{\mathrm{arg\,max}}_f \; p\big(\{\mathbf{y}^{(i)}\} \mid \{\mathbf{x}^{(i)}\}, f\big)\, p(f)$

This is why regularizers (Ch 11) are literally log-priors in disguise.

## Case study 1: Linear least-squares regression

Hypothesis space: $\hat{y} = f_\theta(x) = \theta_1 x + \theta_0$, parameters $\theta = [\theta_0, \theta_1]^\mathsf{T}$. Objective (the $L_2$ loss):

$J(\theta) = \sum_{i=1}^N \big(\theta_1 x^{(i)} + \theta_0 - y^{(i)}\big)^2 = (\mathbf{y} - \mathbf{X}\theta)^\mathsf{T}(\mathbf{y} - \mathbf{X}\theta)$

where $\mathbf{X} \in \mathbb{R}^{N \times 2}$ stacks rows $[1, x^{(i)}]$ (the leading 1 absorbs the intercept) and $\mathbf{y} \in \mathbb{R}^N$ stacks the targets. **Why solvable in closed form:** $J$ is a quadratic form — a convex paraboloid with a single stationary point — so setting the gradient to zero finds the global minimum:

$\frac{\partial J(\theta)}{\partial \theta} = 2\big(\mathbf{X}^\mathsf{T}\mathbf{X}\,\theta - \mathbf{X}^\mathsf{T}\mathbf{y}\big) = 0 \;\;\Rightarrow\;\; \theta^* = (\mathbf{X}^\mathsf{T}\mathbf{X})^{-1}\mathbf{X}^\mathsf{T}\mathbf{y}$

These are the **normal equations**. $\mathbf{X}^\mathsf{T}\mathbf{X}$ is the $2\times 2$ Gram matrix of the features; it is invertible when the data columns are linearly independent.

```python
import torch

def linear_least_squares(x, y):
    """x: (N,), y: (N,) -> theta = [intercept, slope]"""
    X = torch.stack([torch.ones_like(x), x], dim=1)        # (N, 2) design matrix
    # theta* = (X^T X)^{-1} X^T y  — use lstsq for numerical stability
    theta = torch.linalg.lstsq(X, y.unsqueeze(1)).solution  # (2, 1)
    return theta.squeeze(1)

x = torch.tensor([10., 15., 20., 25., 30.])
y = torch.tensor([95., 148., 205., 248., 310.])
theta = linear_least_squares(x, y)   # ~[-8.6, 10.5]: +1 degree -> ~10 more people
```

## Case study 2: Program induction

Same objective and data, but the hypothesis space is *all Python programs* (up to some length). Learning = searching program space. **Why it matters:** it shows the three-ingredient framework scales from 2 scalars to arbitrarily expressive spaces — and previews Ch 11's warning that too-powerful hypothesis spaces overfit (a lookup-table program fits the training data perfectly and generalizes to nothing).

## Case study 3: Classification and softmax regression

Input $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$ (an image), target $\mathbf{y}$ a **one-hot code**: a $K$-vector with $y_k = 1$ for the true class $k$ and 0 elsewhere. The natural loss (count misclassifications) is the **0-1 loss** $\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}) = \mathbb{1}(\hat{\mathbf{y}} \neq \mathbf{y})$, but minimizing it is discrete and NP-hard. **Why cross-entropy instead:** it is a continuous, differentiable surrogate whose minimization equals maximum-likelihood under the model's predicted class distribution:

$\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}) = H(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{k=1}^K y_k \log \hat{y}_k$

Symbols: $\hat{y}_k$ = predicted probability of class $k$; since $\mathbf{y}$ is one-hot, the sum picks out $-\log \hat{y}_{\text{true}}$. For the probability interpretation to hold, $\hat{\mathbf{y}}$ must live on the $(K-1)$-**simplex** $\vartriangle^{K-1}$ (nonnegative, sums to 1). We guarantee this by composing a raw score function $z_\theta: \mathcal{X} \rightarrow \mathbb{R}^K$ (outputs = **logits**, unnormalized log-probabilities) with **softmax**:

$\hat{y}_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}$

**Why softmax:** exponentiation makes every entry positive, normalization makes them sum to 1, and it is the smooth canonical link for categorical maximum likelihood (one-hot targets sit at the simplex vertices; softmax outputs sit inside). It is a modeling choice — any squash-to-pmf map would be legal, but softmax pairs with cross-entropy to give the beautifully simple gradient $\partial \mathcal{L} / \partial z_j = \hat{y}_j - y_j$.

```python
import torch
import torch.nn as nn

K = 5
model = nn.Linear(224 * 224 * 3, K)          # z_theta: flattened image -> logits
x = torch.randn(8, 224 * 224 * 3)
target = torch.randint(0, K, (8,))

logits = model(x)                             # (8, K) — the z vector
loss = nn.functional.cross_entropy(logits, target)
# NOTE: cross_entropy = log_softmax + NLL fused. Never apply softmax yourself
# before this loss — that double-normalizes and is numerically unstable.
```

## Learning paradigms and metalearning

**Unsupervised learning:** only inputs $\{\mathbf{x}^{(i)}\}$ are given; the learner optimizes a property-based objective (e.g., compression that preserves information) — the seed of representation learning (Day 10). **Reinforcement learning:** a reward function $r: \mathcal{Y} \rightarrow \mathbb{R}$ scores outputs and the learner collects its own data. **Metalearning:** the hypothesis space is itself a space of learning algorithms — a meta-meta-algorithm (e.g., learning least-squares regression from examples of fitted lines). Evolution qualifies as a learning algorithm under this definition.

## Why it matters / connections

ERM + cross-entropy + softmax is the exact recipe used by every classifier later in this course (CNNs Day 8, DETR classification heads Day 16). The maximum-likelihood view returns in generative models (Days 10–11), and the prior view becomes regularization in Ch 11.

# Chapter 10 — Gradient-Based Learning Algorithms

## Intuition

Once the learning problem is specified (loss, hypothesis space, parameterization), finding $\theta^* = \mathop{\mathrm{arg\,min}}_\theta J(\theta)$ is optimization. Gradient descent is a skier descending a snowy mountain whose shape is the loss landscape. All gradient-based methods share one loop: at the current **operating point**, compute a locally loss-decreasing direction, step, repeat. Orders of optimization: **zeroth-order** sees only $J(\theta)$ values (sample-and-compare), **first-order** sees the gradient $\nabla_\theta J(\theta)$, **second-order** sees curvature (Hessian $H$ — informative but expensive, hence approximations).

## Basic gradient descent

$\theta^{k+1} = \theta^k - \eta \, \nabla_\theta J(\theta^k)$

Symbols: $\theta^k$ = parameters at iteration $k$; $\eta$ = **learning rate** (step-size multiplier); $\nabla_\theta J$ = vector of partial derivatives of the cost w.r.t. each parameter. **Why the negative gradient:** it is the direction of locally steepest descent — the first-order Taylor expansion $J(\theta + \delta) \approx J(\theta) + \nabla_\theta J^\mathsf{T} \delta$ is minimized over unit $\delta$ by $\delta \propto -\nabla_\theta J$. With sufficiently small $\eta$ and random init, GD almost surely converges to a local minimum as iterations $K \rightarrow \infty$.

## Learning rate schedules

Start high to move fast, decay to settle precisely. Common schedules ($\eta^k = \texttt{lr}(\eta^0, k)$, with decay factor $\beta$, step interval $M$, total steps $K$):

$\eta^k = \beta^{-k}\eta^0 \quad \triangleleft \; \text{exponential decay}$

$\eta^k = \beta^{-\lfloor k/M \rfloor}\eta^0 \quad \triangleleft \; \text{stepwise exponential decay}$

$\eta^k = \frac{K-k}{K}\,\eta^0 \quad \triangleleft \; \text{linear decay}$

**Why decay:** far from the optimum, big steps are safe and fast; near it, a large fixed $\eta$ causes oscillation around the minimizer. Caveat: linear/cosine schedules depend on $K$, making runs of different lengths hard to compare.

## Momentum

$\mathbf{v}^{k+1} = \mu\,\mathbf{v}^k - \eta\,\nabla_\theta J(\theta^k), \qquad \theta^{k+1} = \theta^k + \mathbf{v}^{k+1}$

Symbols: $\mathbf{v}^k$ = running update direction ("velocity"); $\mu \in [0,1)$ = momentum coefficient weighting the previous direction. **Why:** velocity is the skier's inertia — it carries the trajectory over small bumps and accelerates along consistent descent directions (successive gradients that agree compound geometrically, giving an effective step up to $\eta/(1-\mu)$). Trade-off shown in the book's $J = |\theta|$ experiment: $\mu = 0.5$ speeds convergence; $\mu = 0.95$ overshoots and oscillates past the optimum. Nesterov's accelerated gradient and Adam are the popular refinements.

## What functions can GD minimize?

The needed property is *not* textbook differentiability — it is that perturbing $\theta$ yields a meaningful decrease signal. The book's six-case taxonomy:

- Convex, smooth → converges as $\eta \rightarrow 0$.
- Jump discontinuity → fine: PyTorch assigns a **one-sided derivative** at the discontinuity (why ReLU's kink at 0 is harmless).
- Nearly flat (**vanishing gradient**) → descent is agonizingly slow at fixed $\eta$.
- Piecewise constant → gradient is exactly zero almost everywhere; GD is truly stuck.
- Gradient $\rightarrow \infty$ at the minimizer (e.g. $J = \sqrt{|\theta|} - 0.25$) — an **exploding gradient**, causing non-convergence.
- Multiple minima → GD finds *a* minimum; which one depends on initialization.

**Fix for zero gradients — evolution strategies:** sample perturbations $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$, evaluate $J(\theta + \epsilon)$, and move toward perturbations that lowered the loss; alternatively, descend a smoothed **surrogate loss** $J_{\texttt{surr}}$. **Fix for exploding gradients — gradient clipping:**

$\texttt{clip}(v, -m, m) = \max(\min(v, m), -m)$

applied elementwise (or by rescaling the norm) with clip threshold $m$, capping the update magnitude near pathological regions.

## Stochastic gradient descent (SGD)

For ERM objectives, the full gradient decomposes over the training set:

$\nabla_\theta J = \frac{1}{N}\sum_{i=1}^N \nabla_\theta \mathcal{L}\big(f_\theta(\mathbf{x}^{(i)}), \mathbf{y}^{(i)}\big)$

**Why SGD:** with $N = 10^6$, one exact GD step costs $10^6$ per-example gradients. Instead sample a **batch** $\{\mathbf{x}^{(b)}, \mathbf{y}^{(b)}\}_{b=1}^B$ without replacement and use the unbiased estimator

$\tilde{\mathbf{g}} = \frac{1}{B}\sum_{b=1}^B \nabla_\theta \mathcal{L}\big(f_\theta(\mathbf{x}^{(b)}), \mathbf{y}^{(b)}\big)$

Batch size $B$ trades gradient accuracy against speed. Bonuses beyond speed: the noise lets SGD hop over small bumps in the landscape, and SGD **implicitly regularizes** — for linear models with multiple zero-loss solutions it tends to converge to the minimum-norm one (connects directly to Ch 11's overparameterization story).

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# Full training loop: SGD + momentum + stepwise LR decay + gradient clipping
model = torch.nn.Linear(10, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)  # v = mu*v - eta*g
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)  # beta^{-floor(k/M)}

X, Y = torch.randn(1000, 10), torch.randn(1000, 1)
loader = DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=True)  # B = 32

for epoch in range(100):
    for xb, yb in loader:                      # each batch -> one stochastic step
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(xb), yb)
        loss.backward()                        # autograd computes g_tilde
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip
        opt.step()                             # theta <- theta + v
    sched.step()
```

```python
# Evolution strategy: gradient-free descent for zero-gradient losses
import torch

def es_step(theta, J, sigma=1.0, M=10, eta=0.02):
    eps = sigma * torch.randn(M, *theta.shape)          # M perturbations
    scores = torch.stack([J(theta + e) for e in eps])   # evaluate each
    weights = -(scores - scores.mean())                 # lower loss -> positive weight
    v = (weights[:, None] * eps).mean(0) / sigma**2     # loss-decreasing direction
    return theta + eta * v
```

## Why it matters / connections

Everything trainable in this course — CNNs, DETR, SAM, NeRF — is trained by SGD-with-momentum or its adaptive cousins. Vanishing/exploding gradients motivate He init (Day 4 paper), ResNet skips (Day 8), and normalization layers; loss scaling in today's paper deep-dive is precisely a fix for FP16 gradient underflow, a numerical vanishing-gradient problem.

# Chapter 11 — The Problem of Generalization

## Intuition

Learning ≠ optimization: we optimize on the training set but are graded on the test set. **Overfitting** = fitting properties of the training data (e.g., its noise) that do not exist in the test distribution; **underfitting** = failing even on the training data. Learning theory asks when training-set optimization transfers.

## Polynomial regression as the lens

Hypothesis space: degree-$K$ polynomials

$f_\theta(x) = \sum_{k=0}^K \theta_k x^k = \theta^\mathsf{T}\phi(x), \qquad \phi(x) = [1, x, x^2, \ldots, x^K]^\mathsf{T}$

**Why this featurization is the key trick:** $f_\theta$ is *linear in* $\theta$, so polynomial regression is linear regression in the feature space $\phi(x)$ — the same normal equations apply with the design matrix $\mathbf{\Phi} \in \mathbb{R}^{N \times (K+1)}$, $\Phi_{ik} = (x^{(i)})^k$:

$\theta^* = (\mathbf{\Phi}^\mathsf{T}\mathbf{\Phi})^{-1}\mathbf{\Phi}^\mathsf{T}\mathbf{y}$

This foreshadows neural nets: a deep net is a sequence of transformations producing ever-better data matrices (learned features), whereas here $\phi$ is fixed by hand.

Data-generating process for the running example: $Y = X^2 + 1$ (truth), $\epsilon \sim \mathcal{N}(0,1)$ (observation noise), observed $Y' = Y + \epsilon$. As $K$ grows: $K$ too small underfits; $K = 10$ interpolates every point but wiggles — it fits the noise, which by definition does not generalize. Worse, with many perfect-fit hypotheses the data no longer uniquely identifies a solution; the optimizer's idiosyncrasies (initialization) pick one arbitrarily.

## Approximation vs. generalization error

$J_{\texttt{approx}} = \frac{1}{N}\sum_{i=1}^N \mathcal{L}\big(f_\theta(x^{(i)}_{(\texttt{train})}),\, y^{(i)}_{(\texttt{train})}\big)$

$J_{\texttt{gen}} = \mathbb{E}_{x,y \sim p_{\texttt{data}}}\big[\mathcal{L}(f_\theta(x), y)\big] \approx \frac{1}{N}\sum_{i=1}^N \mathcal{L}\big(f_\theta(x^{(i)}_{(\texttt{val})}),\, y^{(i)}_{(\texttt{val})}\big)$

$J_{\texttt{approx}}$ is exactly the ERM objective; $J_{\texttt{gen}}$ is the expected loss under the true data distribution $p_{\texttt{data}}$, estimated on a held-out **validation set**. **Why the U-shape:** as capacity $K$ increases, $J_{\texttt{approx}}$ falls monotonically, but $J_{\texttt{gen}}$ first falls (less underfitting) then rises (overfitting) — the classical capacity/generalization trade-off.

## Regularization

$J(\theta) = \underbrace{\frac{1}{N}\sum_{i=1}^N \mathcal{L}\big(f_\theta(x^{(i)}), y^{(i)}\big)}_{\text{data-fit loss}} + \underbrace{\lambda\, R(\theta)}_{\text{regularizer}}$

$\lambda$ = regularization strength (hyperparameter); common choice $R(\theta) = \lVert\theta\rVert_p$ with $\lVert\mathbf{x}\rVert_p = (\sum_i |x_i|^p)^{1/p}$. $p=2$ → **ridge regression** ("weight decay" in neural nets); $p=1$ → **LASSO** (drives exact sparsity). **Why it works:** penalizing parameter norm pushes most parameters toward zero, degenerating the function to a simpler form — Occam's razor as a soft constraint. Probabilistically, $R$ is a log-**prior** $p(\theta)$: by Bayes' rule the log-posterior = log-likelihood + log-prior, which is exactly the regularized objective. The **Bayesian Occam's razor** explains the preference for simplicity non-arbitrarily: a more complex hypothesis space spreads its prior mass over more hypotheses, so each individual complex hypothesis is a priori less probable.

```python
import torch

# Ridge-regularized polynomial regression, both closed-form and SGD views
def poly_features(x, K):
    return torch.stack([x**k for k in range(K + 1)], dim=1)   # (N, K+1) = Phi

def ridge_closed_form(x, y, K, lam):
    Phi = poly_features(x, K)
    A = Phi.T @ Phi + lam * torch.eye(K + 1)   # regularizer -> +lam*I on the Gram matrix
    return torch.linalg.solve(A, Phi.T @ y)

# Equivalent in the SGD world: weight_decay IS the L2 regularizer
model = torch.nn.Linear(11, 1, bias=False)
opt = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-2)  # lambda
```

## Rethinking generalization

Deep nets violate the classical U-curve: they can have far more parameters than datapoints yet still generalize — bigger nets sometimes overfit *less* (Zhang et al. 2016; Belkin et al. 2018, "double descent"). **Why parameter count misleads:** one infinite-precision parameter can encode an arbitrarily complex function, while a million heavily regularized parameters can define a simple function class. Parameter count is only a crude proxy for capacity; implicit regularization from SGD (Ch 10) is part of the modern story.

## Three tools: data, priors, hypotheses

Learning = finding a needle (the true function) in a haystack (the hypothesis space). Three constraint types: **data** adds one soft constraint per example (each datapoint carves a line into the loss landscape over $\theta$ — heatmaps reminiscent of the Hough transform; with $N=1$ an entire line of solutions is loss-zero, by $N=20$ the constraints intersect at the truth); **priors** add a soft bowl pulling toward preferred solutions (helpful only if the prior is roughly right; too strong a $\lambda$ over-regularizes); the **hypothesis space** is a hard constraint (a linear space accelerates search when the truth is linear, but a constant-only space can never contain the truth — the drunk searching under the lamppost). Lessons: more data ⇒ less need for the other tools; the tools are interchangeable ("methods are extra training data in disguise" — Sutskever); hard constraints cannot be violated, soft ones merely cost penalty.

## Why it matters / connections

This chapter names the failure mode that all of modern practice defends against: augmentation (Day 12), transfer learning (Day 12), and bias/shift analysis (Day 11) are all generalization tools. The prior⇔regularizer identity recurs in graphical models (Day 9) and generative models (Days 10–11).

# Paper Deep-Dive — Mixed Precision Training (Micikevicius, Narang et al., ICLR 2018)

Link: [arxiv.org/pdf/1710.03740](http://arxiv.org/pdf/1710.03740) · NVIDIA + Baidu Research

## Problem and idea

Training in FP32 is memory- and bandwidth-hungry; FP16 halves storage/bandwidth and modern GPUs execute FP16 math 2–8× faster (Volta Tensor Cores). But FP16 has only 10 mantissa bits and normalized exponent range $[-14, 15]$: min normalized positive value $2^{-14}$, max $65{,}504$, and anything below $2^{-24}$ flushes to zero. Gradients are dominated by tiny magnitudes, so naive FP16 training silently loses them. The paper's contribution: three techniques that make pure-FP16 storage match FP32 accuracy **with zero hyperparameter changes**, across AlexNet/VGG/GoogLeNet/Inception/ResNet-50 (ImageNet), Faster R-CNN/SSD (detection), DeepSpeech 2 (speech), seq2seq translation, bigLSTM, and DCGAN.

## Technique 1 — FP32 master copy of weights

Keep weights in FP32; each iteration, cast an FP16 copy for the forward and backward pass; apply the weight update to the FP32 master.

**Why (two separate failure modes):**

1. Update underflow: the update is $\eta \cdot \nabla_w J$, and any FP16 value with magnitude below $2^{-24}$ becomes exactly zero. Empirically ~5% of weight-gradient values in their speech model had exponents below −24 — in FP16 those updates vanish and accuracy collapses (80% relative loss on Mandarin without the master copy).
2. Alignment cancellation: FP16 addition right-shifts the smaller operand to align binary points. With 10 mantissa bits (11 with the implicit bit), if

$\frac{|w|}{|\Delta w|} \geq 2048 = 2^{11}$

the update's bits are shifted entirely out of the sum and $w + \Delta w = w$ — the update is representable but still lost. FP32's 24-bit significand prevents both.

Memory cost: +50% on weights, but activations dominate training memory and those stay FP16, so total memory is still roughly halved.

## Technique 2 — Loss scaling

Multiply the loss by a scale factor $S$ before backprop; by the chain rule every gradient in the network is scaled by the same $S$:

$\nabla_\theta (S \cdot J) = S \cdot \nabla_\theta J$

This shifts the whole gradient histogram up into FP16-representable range (the SSD study: 67% of activation-gradient values were exactly zero in FP16; values in $[2^{-27}, 2^{-24})$ were essential — without scaling, training diverged; $S = 8$, i.e. +3 exponent bits, fully recovered FP32 accuracy). **Why unscale immediately after backward, before clipping/weight decay:** weight updates must have the same magnitude as FP32 training, and unscaling first means no hyperparameter (clip threshold, decay) needs retuning. Choosing $S$: any constant such that

$S \cdot \max|g| < 65{,}504$

works ($g$ = gradient values; 65,504 = FP16 max). Overflow → Inf/NaN irreversibly damages weights, so detect overflow in the unscaled weight gradients and skip that update — the seed of today's *dynamic* loss scaling.

## Technique 3 — FP32 accumulation

Classify network arithmetic into three categories: (1) **dot products / matmuls** — accumulate FP16×FP16 partial products into an FP32 accumulator, convert to FP16 only on the memory write (this is literally what a Tensor Core does; without it some models lose accuracy); (2) **large reductions** (batch-norm statistics, softmax sums) — compute in FP32; these layers are memory-bound so FP32 math costs nothing; (3) **point-wise ops** (activations, elementwise products) — either precision, they're bandwidth-limited anyway.

**Why:** summing many small FP16 terms loses low-order bits at every addition (swamping); a single rounding at the end bounds the error instead of accumulating it.

## Results snapshot

ImageNet top-1: ResNet-50 75.92 → 76.04, Inception v3 73.85 → 74.13 (MP ≥ FP32 throughout, no loss scaling needed for classification). Detection: SSD diverges in FP16 without loss scaling; $S=8$ gives 77.1 vs 76.9 baseline mAP. Speech (largest models, 115M–215M params): MP slightly *better* than FP32 — the authors suggest FP16's quantization noise acts as a regularizer (nice link to Ch 11). bigLSTM needs $S=128$ or it diverges after 300K iterations.

## Faithful minimal PyTorch (the paper's exact loop, hand-rolled)

```python
import torch

def mixed_precision_step(model_fp16, master_params_fp32, opt, x, y, loss_fn, S=1024.0):
    """One iteration exactly as in Fig. 1 of the paper.
    model_fp16: model whose params are FP16 working copies
    master_params_fp32: FP32 master weights (paired 1:1 with model params)
    S: loss scale."""
    # 1) FP16 forward (Tensor Cores accumulate matmuls in FP32 internally)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = loss_fn(model_fp16(x.half()), y)

    # 2) scale loss, FP16 backward -> all grads scaled by S (chain rule)
    (loss.float() * S).backward()

    # 3) unscale grads in FP32, check overflow, update FP32 master
    found_inf = False
    for p16, p32 in zip(model_fp16.parameters(), master_params_fp32):
        g = p16.grad.float() / S                     # unscale BEFORE clip/decay
        if not torch.isfinite(g).all():
            found_inf = True
            break
        p32.grad = g
    if not found_inf:
        torch.nn.utils.clip_grad_norm_(master_params_fp32, 1.0)
        opt.step()                                   # update in FP32
        for p16, p32 in zip(model_fp16.parameters(), master_params_fp32):
            p16.data.copy_(p32.data.half())          # round master -> FP16 copy
    # overflow -> skip update entirely (dynamic scaling would also lower S)
    for p16 in model_fp16.parameters():
        p16.grad = None
```

```python
# Modern equivalent: torch.cuda.amp automates all three techniques
scaler = torch.cuda.amp.GradScaler()      # dynamic loss scaling + overflow skip
for x, y in loader:
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):  # FP16 fwd, FP32 accumulation
        loss = loss_fn(model(x), y)       # optimizer state/weights stay FP32
    scaler.scale(loss).backward()         # scaled backward
    scaler.unscale_(opt)                  # unscale before clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)                      # skips step on Inf/NaN
    scaler.update()                       # grows/shrinks S adaptively
```

## Why it matters / connections

This paper defined the training recipe used by essentially every large vision/language model since (AMP, bf16 variants). It is a numerics-level echo of Ch 10: loss scaling fights gradient *underflow* the way clipping fights *explosion*, and the FP32 master copy guarantees the SGD update $\theta \leftarrow \theta + \mathbf{v}$ is actually applied rather than rounded away. When Day 8 trains ResNet and Day 16 trains DETR variants, `torch.autocast` + `GradScaler` is this paper running under the hood.

# Supplementary resources for this block

- **Udemy — Mastering Computer Vision, Module 2 (DL Foundations & CNNs):** [course link](https://www.udemy.com/course/mastering-computer-vision-from-pixel-to-detection-to-gen-cv/). Module 2 covers the same arc as Ch 9–10 from a practitioner's angle: loss functions, softmax + cross-entropy, SGD/momentum, and training loops in PyTorch. Watch it as a lab session for today's math. (Paywalled — module pointer only.)
- **adensur, Computer Vision Zero to Hero** ([repo](https://github.com/adensur/blog/tree/main/computer_vision_zero_to_hero)) — mapped folders for today: `00_introduction`, `01_simple_classification`, `02_crossentropy`, `06_optimizers`. *Note: raw file fetch was blocked in this environment, so key ideas are integrated from the folder topics rather than quoted.*
    - `00_introduction` / `01_simple_classification`: frames classification exactly as Ch 9 does — flatten image → linear layer → logits → softmax; a full training/eval split workflow on MNIST-style data. Reinforces that a "model" is just $f_\theta$ plus the three ingredients.
    - `02_crossentropy`: derives cross-entropy from KL divergence / maximum likelihood — the same why-this-loss argument as §9.7.3, and stresses the numerically stable log-softmax fusion used by `F.cross_entropy` (never softmax-then-log). Complements the paper deep-dive: numerics decide what your math actually computes.
    - `06_optimizers`: walks SGD → momentum → RMSProp → Adam as increasingly informed update rules over the same loop from Ch 10 — momentum accumulates a velocity $\mathbf{v}$, adaptive methods additionally normalize per-parameter step sizes by running gradient-magnitude statistics. Practical default echoed there: AdamW or SGD+momentum 0.9 with a decaying schedule.

# Self-check questions

3. State the ERM objective and explain precisely why minimizing it can fail to minimize test loss. Which of the three tools (data, priors, hypothesis space) attacks that failure, and how does each act (hard vs. soft constraint)?
4. Derive $\theta^* = (\mathbf{X}^\mathsf{T}\mathbf{X})^{-1}\mathbf{X}^\mathsf{T}\mathbf{y}$ from the $L_2$ objective. Why does setting the gradient to zero suffice here, and why does the same argument fail for a deep network?
5. Why is cross-entropy used instead of 0-1 loss, and why must softmax precede it? Show that for one-hot $\mathbf{y}$ the gradient of the loss w.r.t. logits is $\hat{\mathbf{y}} - \mathbf{y}$.
6. Explain how momentum changes the GD update and why $\mu = 0.95$ can be worse than $\mu = 0.5$ on $J = |\theta|$. What do gradient clipping and evolution strategies each fix?
7. In mixed precision training, why does an FP16 weight update vanish when $|w|/|\Delta w| \geq 2048$ even though $\Delta w$ is representable in FP16, and how do the FP32 master copy and loss scaling address the two distinct underflow mechanisms?

# Latest CV Research — 2026-07-03

Selection biased toward today's block (optimization / generalization) plus the biggest recent CVPR 2026 news. Note: web search surfaced limited machine-readable detail for some very recent items; contribution summaries reflect what the sources state.

8. **Double Preconditioning (DoPr): Optimization for Test-Time Performance, not Validation Loss** — Francisco Patitucci, Aryan Mokhtari (UT Austin) · arXiv, June 2026 · Directly on-topic for Ch 10–11: proposes an optimizer whose preconditioning targets *test-time* performance rather than the training/validation loss — an explicit attack on the learning-vs-optimization gap that Ch 11 formalizes as $J_{\texttt{gen}} \neq J_{\texttt{approx}}$. [arXiv 2606.06418](https://arxiv.org/pdf/2606.06418)
9. **D4RT: Efficiently Reconstructing Dynamic Scenes One D4RT at a Time** — Zhang, Le Moing, Koppula, et al. (Google DeepMind, UCL, Oxford) · **CVPR 2026 Best Paper** (awarded June 9, 2026) · A unified feed-forward network that reconstructs geometry and motion of dynamic 4D scenes directly from video, substantially faster than per-scene optimization approaches. Why it matters: continues the field-wide shift from test-time optimization (NeRF-style, Day 14) to amortized learned inference. [CVPR announcement](https://cvpr.thecvf.com/Conferences/2026/News/Best_Papers) · [Voxel51 breakdown](https://voxel51.com/blog/d4rt-cvpr-2026-best-paper-4d-reconstruction)
10. **SAM 3D: 3Dfy Anything in Images** — Meta Superintelligence Labs · **CVPR 2026 Best Paper Honorable Mention** (June 2026) · Generative model predicting object geometry, texture, and layout from a single image — extends the SAM promptable-segmentation lineage (Day 17) into single-view 3D. Why it matters: ties Day 17's SAM study to 3D scene understanding (Day 13). [BasicAI summary](https://www.basic.ai/blog-post/cvpr-2026-top-papers-award-winners-and-notable-works)
11. **MUSE: Resolving Manifold Misalignment in Visual Tokenization via Topological Orthogonality** — arXiv, May–June 2026 · Addresses how visual tokenizers distort the data manifold, proposing topologically orthogonal token spaces for downstream generation/understanding. Why it matters: tokenization quality is the "parameterization matters" lesson of Ch 9 applied to modern VLMs (Day 15). [arXiv 2605.05646](https://arxiv.org/pdf/2605.05646)
12. **LingBot-Map: Feed-forward 3D Foundation Model for Streaming Reconstruction** — trending on Hugging Face Papers this week (early July 2026) · Geometric-context transformer reconstructing scenes from video streams at ~20 FPS with attention modules for coordinate grounding, dense geometric cues, and long-range drift correction. Why it matters: real-time amortized 3D — same trend as D4RT, relevant to Days 13–14. [HF trending](https://huggingface.co/papers/trending)
13. **Pixal3D: Pixel-Aligned 3D Generation from Images** — TencentARC · arXiv, May 2026, trending this week · Pixel-aligned conditioning for image-to-3D generation, improving geometric faithfulness to the input view. Why it matters: conditional generative modeling (Day 11) meeting geometry (Day 13). [arXiv 2605.10922](https://arxiv.org/abs/2605.10922)