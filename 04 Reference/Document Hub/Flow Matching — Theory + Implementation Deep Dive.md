---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-12T00:00:00
Status: Done
Last updated time: 2026-07-12T00:00:00
Last edited by: Heaven Chen
Category:
  - ML
  - Theory
  - AIGC
---
> Lipman, Havasi, Holderrieth, Chen, Karrer, Chen, Ben-Hamu, Rustamov, Gat, et al. **"Flow Matching Guide and Code"** (Meta FAIR / MIT), [arXiv:2412.06264](https://arxiv.org/pdf/2412.06264). Companion library: `facebookresearch/flow_matching`. This note pairs the theory with the paper's own PyTorch code so you can both derive it and write it.

## 1. TL;DR

Flow Matching (FM) learns a **velocity field** $u_t^\theta(x)$ so that solving the ODE $\dot X_t = u_t^\theta(X_t)$ from $t=0$ to $t=1$ transports a sample from a simple **source** $p_0$ (e.g. Gaussian noise) into a sample from the **target** data distribution $p_1=q$. Training is a plain regression — no ODE solve in the loop (**simulation-free**), no divergence term. The whole method fits in ~30 lines (Code 1). Everything else (affine schedules, optimal-transport couplings, manifolds, discrete data, generator matching) generalises that core loop.

The single most important practical fact: you never regress the intractable *marginal* velocity. You regress a trivial *conditional* target — for the linear path, the constant $X_1-X_0$ — and a theorem guarantees the gradients are identical, so the network still converges to the true marginal field.

## 2. The whole idea in one code block (paper's Code 1)

This is the entire method on the two-moons dataset — read it first, then we justify every line.

```python
import torch
from torch import nn, Tensor
from sklearn.datasets import make_moons

class Flow(nn.Module):
    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h), nn.ELU(),   # +1 for the time input t
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim))

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        return self.net(torch.cat((t, x_t), -1))          # predicts velocity u_t(x_t)

    def step(self, x_t, t_start, t_end):                   # one midpoint ODE step
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        return x_t + (t_end - t_start) * self(
            x_t + self(x_t, t_start) * (t_end - t_start) / 2,
            t_start + (t_end - t_start) / 2)

# ---- training (simulation-free) ----
flow = Flow()
optimizer = torch.optim.Adam(flow.parameters(), 1e-2)
loss_fn = nn.MSELoss()

for _ in range(10000):
    x_1 = Tensor(make_moons(256, noise=0.05)[0])          # target sample  X1 ~ q
    x_0 = torch.randn_like(x_1)                           # source sample  X0 ~ N(0,I)
    t   = torch.rand(len(x_1), 1)                         # t ~ U[0,1]
    x_t = (1 - t) * x_0 + t * x_1                         # linear/OT path  X_t
    dx_t = x_1 - x_0                                      # target velocity (constant!)
    optimizer.zero_grad()
    loss_fn(flow(x_t, t), dx_t).backward()               # CFM = MSE(u_theta(X_t,t), X1-X0)
    optimizer.step()

# ---- sampling: integrate the ODE 0 -> 1 ----
x = torch.randn(300, 2)                                   # X0 ~ N(0,I)
time_steps = torch.linspace(0, 1.0, 8 + 1)
for i in range(8):
    x = flow.step(x, time_steps[i], time_steps[i + 1])    # x ends as X1 ~ q
```

Note there is **no ODE solve during training** and **no likelihood/divergence computation** — training line 33–34 just builds $X_t$ by linear interpolation and regresses onto $X_1-X_0$. That is the payoff of the whole theory below.

## 3. Flow models — the mathematical objects (paper §3)

A **flow** is a time-dependent map $\psi_t:\mathbb{R}^d\to\mathbb{R}^d$ defined as the solution of an ODE driven by a velocity field $u:[0,1]\times\mathbb{R}^d\to\mathbb{R}^d$:

$$\frac{d}{dt}\psi_t(x_0) = u_t\big(\psi_t(x_0)\big), \qquad \psi_0(x_0)=x_0. \tag{3.19}$$

- **Deterministic & invertible.** If $u_t$ is Lipschitz (paper Thm. 1), $\psi_t$ is a *diffeomorphism*: a unique trajectory through every point, smoothly invertible, so you can flow forward and backward.
- **Velocity ⇄ flow are 1-to-1:** $u_t(x) = \dot\psi_t\big(\psi_t^{-1}(x)\big)$ (Eq. 3.20). FM chooses to parameterise the velocity.

**Pushforward.** Flowing the source density gives the time-$t$ density by change of variables (Eq. 3.14):

$$p_t(x) = p_0\big(\psi_t^{-1}(x)\big)\,\big|\det J_{\psi_t^{-1}}(x)\big|.$$

**Continuity equation (mass conservation, Eq. 3.25).** A velocity field $u_t$ *generates* a probability path $p_t$ iff

$$\frac{\partial p_t(x)}{\partial t} + \nabla_x\!\cdot\!\big(p_t(x)\,u_t(x)\big) = 0.$$

This is the bridge between "a distribution moving in time" ($p_t$) and "the field pushing it" ($u_t$).

**Instantaneous change of variables → exact likelihood (Eq. 3.31).** Along a trajectory,

$$\frac{d}{dt}\log p_t(\psi_t(x_0)) = -\nabla_x\!\cdot\! u_t(\psi_t(x_0)) \;\Rightarrow\; \log p_1(x_1)=\log p_0(x_0)-\int_0^1 \nabla_x\!\cdot\! u_t(\psi_t(x_0))\,dt.$$

This is the classic Continuous Normalizing Flow (CNF) likelihood. **Why the old CNFs were slow:** every training step needed (i) an ODE simulation and (ii) the divergence $\nabla_x\!\cdot\!u$ (a Jacobian trace, expensive in high-$d$, usually Hutchinson-estimated). FM's contribution is to make *training* need neither.

### 3.1 Implementation — likelihood at eval time (paper's Code 3, simplified)

You still *can* compute exact likelihood after training — you just don't need it to train. Solve the augmented ODE backward and accumulate the divergence (Hutchinson trace estimator):

```python
def divergence_hutchinson(vfield, x, t, n_proj=1):
    # E_eps[ eps^T (d u / d x) eps ] , eps ~ Rademacher/Gaussian  (unbiased trace)
    div = 0.0
    for _ in range(n_proj):
        eps = torch.randn_like(x)
        with torch.enable_grad():
            x = x.requires_grad_(True)
            u = vfield(x, t)
            (jvp,) = torch.autograd.grad(u, x, grad_outputs=eps, create_graph=False)
        div = div + (jvp * eps).sum(dim=-1)
    return div / n_proj

@torch.no_grad()
def log_likelihood(vfield, x1, steps=100):
    x, logp = x1.clone(), torch.zeros(x1.shape[0])
    dt = 1.0 / steps
    for i in range(steps):                     # integrate t=1 -> 0
        t = torch.full((x.shape[0], 1), 1 - i * dt)
        logp = logp + divergence_hutchinson(vfield, x, t) * dt   # sign from reverse pass
        x = x - vfield(x, t) * dt
    return standard_normal_logprob(x) + logp   # log p0(x0) + integral of divergence
```

## 4. The FM / CFM training objective (paper §4)

We *want* to minimise the **Flow Matching loss**, regressing the true marginal velocity:

$$\mathcal{L}_{\mathrm{FM}}(\theta)=\mathbb{E}_{t\sim U[0,1],\,X_t\sim p_t}\Big[\,D\big(u_t(X_t),\,u_t^\theta(X_t)\big)\Big], \tag{4.2}$$

with $D$ a Bregman divergence (squared error $D(u,v)=\tfrac12\|u-v\|^2$ is the default). Problem: both $p_t$ and $u_t$ are intractable. The fix has two moving parts.

**(a) Build the path from a conditional one (marginalization).** Pick a conditioning variable $Z$ (usually $Z=X_1$), design an easy **conditional path** $p_{t\mid 1}(x\mid x_1)$, and average it:

$$p_t(x)=\int p_{t\mid 1}(x\mid x_1)\,q(x_1)\,dx_1, \qquad p_{0\mid1}=p_0,\;\; p_{1\mid1}=\delta_{x_1}. \tag{4.11}$$

The canonical choice is the **Gaussian / conditional-OT path**
$$p_{t\mid1}(x\mid x_1)=\mathcal N\big(x\mid t\,x_1,\,(1-t)^2 I\big),$$
realised by the interpolation $X_t=(1-t)X_0+tX_1$, whose conditional velocity is analytic:

$$u_t(x\mid x_1)=\frac{x_1-x}{1-t}, \qquad\text{or, conditioning on }X_0:\;\; \dot\psi_t = x_1-x_0. \tag{2.6 / 2.9}$$

**(b) The Marginalization Trick (Thm. 3).** The true marginal velocity is the *posterior average* of these conditional velocities:

$$u_t(x)=\mathbb{E}_{X_1\sim p_{1\mid t}(\cdot\mid x)}\big[u_t(x\mid X_1)\big], \qquad p_{1\mid t}(x_1\mid x)=\frac{p_{t\mid1}(x\mid x_1)\,q(x_1)}{p_t(x)}\ \text{(Bayes)}.$$

**(c) Conditional Flow Matching loss.** Regress the *conditional* target instead:

$$\mathcal{L}_{\mathrm{CFM}}(\theta)=\mathbb{E}_{t,\,X_0\sim p_0,\,X_1\sim q}\Big[\big\|u_t^\theta(X_t)-u_t(X_t\mid X_1)\big\|^2\Big],\quad X_t=(1-t)X_0+tX_1. \tag{4.23}$$

For the linear path this is exactly the two-line target in Code 1:
$$\boxed{\ \mathcal{L}_{\mathrm{OT,CFM}}(\theta)=\mathbb{E}_{t,X_0,X_1}\big[\|u_t^\theta((1-t)X_0+tX_1)-(X_1-X_0)\|^2\big]\ } \tag{2.9}$$

**(d) Why this is legitimate — gradient equivalence (Thm. 4).** Because the Bregman-divergence gradient is affine-invariant,
$$\nabla_\theta\,\mathcal{L}_{\mathrm{FM}}(\theta)=\nabla_\theta\,\mathcal{L}_{\mathrm{CFM}}(\theta),$$
and the CFM minimiser is precisely the marginal field $u_t^\theta(x)=\mathbb{E}[u_t(x\mid X_1)\mid X_t=x]$. **This is the theorem that makes the cheap loss learn the expensive quantity.**

### 4.1 Implementation — CFM with the `flow_matching` library (paper's Code 4)

The library abstracts the path so you only supply data pairs; `path.sample` returns both $X_t$ and the target velocity $\dot\psi_t$:

```python
import torch
from flow_matching.path import ProbPath
from flow_matching.path.path_sample import PathSample

path: ProbPath = ...              # e.g. CondOTProbPath() or AffineProbPath(scheduler)
velocity_model: torch.nn.Module = ...
optimizer = torch.optim.Adam(velocity_model.parameters())

for x_0, x_1 in dataloader:                       # samples from coupling pi_{0,1}
    t = torch.rand(x_1.shape[0])                  # t ~ U[0,1]
    sample: PathSample = path.sample(t=t, x_0=x_0, x_1=x_1)
    x_t, dx_t = sample.x_t, sample.dx_t           # dx_t = psi_dot_t(X0|X1) = target velocity
    cfm_loss = torch.pow(velocity_model(x_t, t) - dx_t, 2).mean()   # MSE = CFM when D=Euclidean
    optimizer.zero_grad(); cfm_loss.backward(); optimizer.step()
```

Swapping `path` (linear-OT vs a VP/VE affine schedule) is the *only* change needed to move between FM variants — the loop is identical.

## 5. Data couplings & conditional generation (paper §4.1, §4.6)

$\pi_{0,1}(X_0,X_1)$ is the joint you draw pairs from:

- **Independent** $\pi_{0,1}=p_0\,q$ — unconditional generation (noise → image). This is `dataloader` yielding fresh `x_0 = randn`.
- **Dependent / paired** — super-resolution ($X_0$=low-res), inpainting ($X_0$=masked), editing ($X_0$=source image). You just yield real pairs; the loop above is unchanged.
- **OT (minibatch) coupling** — reorder each minibatch to pair nearby $X_0,X_1$ (solve a small assignment problem), which straightens trajectories further and cuts sampling steps.

**Conditional generation** $p_t(x\mid y)$ (e.g. text→image): feed the condition to the model, $u_t^\theta(x,y)$, with $p_{0\mid y}=p_0$, $p_{1\mid y}=q(\cdot\mid y)$. At sampling, plug in the desired $y$. This subsumes classifier-free guidance as a special case.

## 6. Affine paths, prediction targets, and the diffusion connection (paper §4.7–4.8)

The linear path is a special case of an **affine conditional flow**
$$\psi_t(x_0\mid x_1)=\sigma_t\,x_0+\alpha_t\,x_1,\qquad \sigma_0{=}1,\alpha_0{=}0,\ \sigma_1{=}0,\alpha_1{=}1,$$
giving Gaussian marginals $p_{t\mid1}(x\mid x_1)=\mathcal N(x\mid \alpha_t x_1,\sigma_t^2 I)$. Choosing $(\alpha_t,\sigma_t)$ recovers diffusion schedules:

| Schedule | $\alpha_t,\sigma_t$ | Note |
|---|---|---|
| Linear / cond-OT | $\alpha_t=t,\ \sigma_t=1-t$ | straightest paths, fewest steps |
| Variance-Preserving (VP) | $\alpha_t=\sqrt{1-e^{-\beta_t}},\ \sigma_t=e^{-\beta_t/2}$ | ≈ DDPM |
| Variance-Exploding (VE) | $\alpha_t=1,\ \sigma_t=\sigma_{\max}(1-t)$ | ≈ SMLD/NCSN |

**Interconvertible targets (Table 1).** Velocity $u_t(x)$, target-prediction $x_{1\mid t}=\mathbb{E}[X_1\mid X_t{=}x]$ (a "denoiser"), source-prediction $x_{0\mid t}=\mathbb{E}[X_0\mid X_t{=}x]$, and the score $\nabla\log p_t$ are all *linear reparameterisations* of one another for affine Gaussian paths (Eq. 4.79 links score and velocity). This is exactly the $\varepsilon$-/$x_0$-/score-prediction equivalence from diffusion — so **diffusion is essentially FM with a curved Gaussian schedule**, and FM's cond-OT path is what buys the straighter, few-step trajectories.

### 6.1 Implementation — $x_1$-prediction ("denoiser") training (paper's Code 6)

Often more stable to predict the clean target $x_1$ and convert to velocity only at sampling:

```python
import torch
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

path: AffineProbPath = ...          # carries the (alpha_t, sigma_t) schedule
denoiser_model: torch.nn.Module = ...
optimizer = torch.optim.Adam(denoiser_model.parameters())

for x_0, x_1 in dataloader:
    t = torch.rand(x_1.shape[0])
    sample = path.sample(t=t, x_0=x_0, x_1=x_1)                     # conditional path sample
    cm_loss = torch.pow(denoiser_model(sample.x_t, t) - sample.x_1, 2).mean()  # predict X1
    optimizer.zero_grad(); cm_loss.backward(); optimizer.step()

# wrap the denoiser as a velocity field for the ODE solver
class VelocityModel(ModelWrapper):
    def __init__(self, denoiser, path):
        super().__init__(model=denoiser); self.path = path
    def forward(self, x, t, **extras):
        x1_hat = super().forward(x, t, **extras)
        return self.path.target_to_velocity(x_1=x1_hat, x_t=x, t=t)   # linear conversion

x_0 = torch.randn(batch_size, *data_dim)
solver = ODESolver(velocity_model=VelocityModel(denoiser_model, path))
x_1 = solver.sample(x_init=x_0, method='midpoint', step_size=1.0/100)
```

## 7. Sampling — solving the ODE (paper §3.2, Code 2)

Sampling is any ODE integrator on $\dot X_t=u_t^\theta(X_t)$, $X_0\sim p_0$. More/curvier = more steps.

```python
@torch.no_grad()
def sample_euler(vfield, shape, steps=100):
    x = torch.randn(shape); dt = 1.0/steps
    for i in range(steps):
        t = torch.full((shape[0], 1), i*dt)
        x = x + vfield(x, t) * dt                 # Euler; cheap, needs more steps
    return x

@torch.no_grad()
def sample_midpoint(vfield, shape, steps=50):
    x = torch.randn(shape); dt = 1.0/steps
    for i in range(steps):
        t  = torch.full((shape[0], 1), i*dt)
        k1 = vfield(x, t)
        k2 = vfield(x + 0.5*dt*k1, t + 0.5*dt)    # 2nd-order: same quality in ~half the steps
        x  = x + dt*k2
    return x
```

The library's `ODESolver.sample(..., method='midpoint'|'euler'|'dopri5', step_size=...)` wraps exactly this. **Straight (cond-OT/OT-coupled) paths are the reason FM can sample in a handful of steps** where diffusion typically needs many.

## 8. Extensions (paper §5–§6) — same recipe, different geometry

- **Riemannian FM (§5).** For data on manifolds (spheres = climate, $SO(3)$/$SE(3)$ = protein poses, tori = joint angles), the velocity lives in the tangent space $u_t(x)\in T_x\mathcal M$, and the continuity / change-of-variables use the metric divergence $\mathrm{div}_g$. The conditional flow becomes a **geodesic interpolation** (the manifold's straight line); CFM is otherwise identical.
- **Discrete FM (§6).** For text / categorical / DNA data, replace the ODE with a **Continuous-Time Markov Chain**; the "velocity" becomes transition **rates** (a generator), conditional paths live on the simplex, and you marginalise + match rates. Closely related to discrete diffusion.
- **Generator Matching (the unification).** A Markov process is defined by its **generator**. FM learns an ODE generator (velocity), diffusion an SDE generator (score), discrete FM a CTMC generator (rates). The marginalization trick + conditional matching apply to *any* generator, so all of these are one framework.

## 9. FM vs Diffusion — cheat sheet

| | Diffusion (DDPM/SBM) | Flow Matching |
|---|---|---|
| Path $p_t$ | implicit, fixed by the forward SDE | **explicitly designed** (any affine/OT path) |
| Learn | score / noise ($\varepsilon$) of the reverse process | velocity field $u_t$ (or $x_1$-denoiser) |
| Training | simulation-free (score matching) | simulation-free (CFM), broader path choice |
| Trajectories | curved → many sampling steps | straight with cond-OT → few steps |
| Relationship | — | diffusion = FM with a Gaussian VP/VE schedule; score ↔ velocity linear (Eq. 4.79) |

## 10. Mental model to keep

1. **Design a path** $p_0\to p_1$ (usually the straight line $X_t=(1-t)X_0+tX_1$).
2. **Regress the conditional velocity** — for the line, the constant $X_1-X_0$. Cheap, stable, no ODE, no divergence.
3. **The marginalization theorem** promotes that cheap regression to the true marginal field.
4. **Sample** by integrating the ODE from noise. Straighter paths ⇒ fewer steps.

Everything else in the guide — affine schedules, OT couplings, denoiser parameterisation, manifolds, discrete states, generator matching — is a re-skin of steps 1–2.

## Sources
- Lipman et al., **Flow Matching Guide and Code**, [arXiv:2412.06264](https://arxiv.org/pdf/2412.06264) — equations (§3–§6) and code listings (Code 1, 2, 3, 4, 6) referenced above.
- Library: [`facebookresearch/flow_matching`](https://github.com/facebookresearch/flow_matching).
- Foundational papers: Lipman et al. (2022) *Flow Matching for Generative Modeling*; Liu et al. (2022) *Rectified Flow*; Albergo & Vanden-Eijnden, *Stochastic Interpolants*; Chen & Lipman, *Riemannian Flow Matching*; Gat et al., *Discrete Flow Matching*.
- Plain-language companion: doctorin, [【入門】フローマッチングのエッセンス (Zenn, 2025)](https://zenn.dev/doctorin/articles/flow-matching).
