---
base: "[[01.1 Computer Vision — Foundations Study Tracker.base]]"
Key takeaways: "Deep nets are stacks of affine maps + pointwise nonlinearities, best understood as layer-by-layer geometric transformations of the data distribution (relu snaps data to a 1/2^N cone, driving sparsity). Backprop is the chain rule organized as vector–Jacobian products: δ_ℓ = (W_{ℓ+1}^T δ_{ℓ+1}) ⊙ f'(z_ℓ), with δ_L = ŷ − y for softmax+CE and ∂J/∂W = δ h^T. He init keeps Var of signals/gradients constant per layer by demanding (1/2)(1+a²)·n·Var[w]=1 — the ReLU-corrected Xavier — and PReLU learns the negative slope per channel (no weight decay on it), together enabling the first super-human ImageNet result."
Day: 4
Status: In-Progress
Reading done: true
Chapters: Ch 12–14
Self-check done: false
Date: 2026-07-04
Part:
  - Foundations-of-Learning
Questions / Follow-ups: ""
---
**Reading checklist**

- [x] [12 Neural Networks](https://visionbook.mit.edu/neural_nets.html)
- [x] [13 Neural Networks as Distribution Transformers](https://visionbook.mit.edu/neural_nets_as_distribution_transformers.html)
- [x] [14 Backpropagation](https://visionbook.mit.edu/backpropagation.html)

## Notes

## Self-check

# Day 4 Study Report — Neural Networks & Backpropagation (2026-07-04 JST)

## Chapter 12 — Neural Networks

### 12.1 From perceptron to deep nets: the big picture

Chapters 9–11 treated learning as fitting a parametric function $f_\theta$ to data by minimizing a loss with gradient descent. Chapter 12 answers: **what family of functions should **$f_\theta$** be?** The neural network answer: a *composition of many simple, differentiable building blocks* — alternating affine maps ("linear layers") and elementwise nonlinearities. Nothing in a deep net is exotic; the power comes entirely from composition and from training the whole stack end-to-end.

### 12.2 The perceptron: a single neuron

A single artificial neuron computes a weighted sum of its inputs, adds a bias, and passes the result through a nonlinearity:

$y = f(\mathbf{w}^\top \mathbf{x} + b)$

where $\mathbf{x} \in \mathbb{R}^n$ is the input vector, $\mathbf{w} \in \mathbb{R}^n$ the learned weights, $b \in \mathbb{R}$ the learned bias, and $f$ the activation function (the classic perceptron used a hard step; modern nets use relu or sigmoid).

**WHY this form:** $\mathbf{w}^\top\mathbf{x}+b=0$ defines a hyperplane in input space; the neuron is a linear classifier that fires on one side. The bias is essential — without it every decision boundary would be forced through the origin. The nonlinearity $f$ is what makes *stacking* useful: composing purely affine maps collapses to a single affine map, $\mathbf{W}_2(\mathbf{W}_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2 = \mathbf{W}\mathbf{x}+\mathbf{b}$, so without $f$ depth would buy nothing.

### 12.3 Multilayer perceptrons (MLPs)

An MLP stacks layers of neurons; each layer $\ell$ applies an affine transform followed by a pointwise nonlinearity:

$\mathbf{z}_\ell = \mathbf{W}_\ell \mathbf{h}_{\ell-1} + \mathbf{b}_\ell, \qquad \mathbf{h}_\ell = f(\mathbf{z}_\ell)$

with $\mathbf{h}_0 = \mathbf{x}$ (the input), $\mathbf{W}_\ell \in \mathbb{R}^{d_\ell \times d_{\ell-1}}$, $\mathbf{b}_\ell \in \mathbb{R}^{d_\ell}$. $\mathbf{z}_\ell$ are called **pre-activations**, $\mathbf{h}_\ell$ **activations** (the book calls them hidden units). With one sufficiently wide hidden layer an MLP can approximate any continuous function on a compact set (universal approximation) — but depth typically achieves the same expressivity exponentially more compactly.

### 12.4 Activations versus parameters — a crucial distinction

Two very different kinds of variables live in a net. **Parameters** $\theta = \{\mathbf{W}_\ell, \mathbf{b}_\ell\}$ persist across inputs and are what training updates. **Activations** $\{\mathbf{h}_\ell\}$ are transient — recomputed per input, and each $\mathbf{h}_\ell$ is a different *representation/embedding* of that input. Keeping this straight matters for backprop bookkeeping (Ch 14): we need gradients with respect to *both* — with respect to activations in order to chain backwards, and with respect to parameters in order to update.

### 12.5 Catalog of common layers

**Linear (fully connected):** $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$ — the only layer here with parameters; does all rotating/stretching/mixing of features.

**Relu:** applied elementwise,

$\texttt{relu}(z) = \max(0, z)$

WHY: cheap, non-saturating for $z>0$ (derivative exactly 1, so gradients pass through unattenuated — the key fix for vanishing gradients), and induces sparse activations.

**Sigmoid:** squashes to $(0,1)$:

$\sigma(z) = \frac{1}{1 + e^{-z}}$

WHY: interpretable as a probability; but $\sigma'(z)=\sigma(z)(1-\sigma(z)) \le 1/4$ and $\to 0$ for large $|z|$ — stacking many sigmoids multiplies these small factors, which is exactly the vanishing-gradient pathology that relu avoids.

**Softmax:** turns a score vector into a probability distribution on $K$ classes:

$\texttt{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$

WHY the exponential: it is positive, monotone, and makes the log-likelihood of the correct class *linear* in the logits, giving the clean gradient we derive in Ch 14.

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    """linear-relu-linear-relu-linear, as in the book's binary-classifier example."""
    def __init__(self, d_in=2, d_hidden=64, d_out=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_out),   # logits; softmax lives inside the loss
        )
    def forward(self, x):
        return self.net(x)

model = MLP()
loss_fn = nn.CrossEntropyLoss()   # = log_softmax + NLL, numerically stable
```

### 12.6 Why are neural networks a good architecture?

Four properties conspire: (1) **differentiability** — every block has a well-defined (sub)gradient, so one generic algorithm (backprop + SGD) trains any composition; (2) **compositionality** — layers are lego bricks, and swapping/stacking them explores an enormous design space; (3) **expressivity with an inductive bias toward hierarchy** — features build on features, matching the compositional structure of images; (4) **hardware fit** — the workload is dense matrix multiplication, which GPUs execute extremely efficiently.

## Chapter 13 — Neural Networks as Distribution Transformers

### 13.1 A different way to plot a function

Instead of the usual $x$-axis/$y$-axis graph, draw input space and output space side by side and connect each input point to its image. A function then literally *rearranges space*. The identity is straight parallel lines; a linear layer shifts/stretches/rotates; relu snaps every negative coordinate to 0; sigmoid pulls everything toward $\{0,1\}$. This view generalizes to 2D→2D maps (where ordinary plotting fails) and is the right mental model for what layers do to *datasets* rather than to single points.

### 13.2 Deep nets remap data distributions

Treat the dataset as a distribution $p_{\text{data}}$. Each layer transforms the distribution of activations: $p_{\text{data}} \to p_1 \to p_2 \to \dots \to p_{\text{out}}$. Training shapes these maps so that $p_{\text{out}}$ matches a target distribution $p_{\text{target}}$ — and most losses are exactly a divergence between $p_{\text{out}}$ and $p_{\text{target}}$. For a binary softmax classifier the target puts all class-0 points at the one-hot code $(1,0)$ and all class-1 points at $(0,1)$: classification = moving points to simplex vertices.

The book's worked example is the MLP

$\begin{aligned} \mathbf{z}_1 &= \mathbf{W}_1\mathbf{x} + \mathbf{b}_1 \\ \mathbf{h}_1 &= \texttt{relu}(\mathbf{z}_1) \\ \mathbf{z}_2 &= \mathbf{W}_2\mathbf{h}_1 + \mathbf{b}_2 \\ \mathbf{h}_2 &= \texttt{relu}(\mathbf{z}_2) \\ \mathbf{z}_3 &= \mathbf{W}_3\mathbf{h}_2 + \mathbf{b}_3 \\ \mathbf{y} &= \texttt{softmax}(\mathbf{z}_3) \end{aligned}$

Watching checkpoints $\theta^0, \theta^1, \ldots, \theta^T$ during training, one sees the layers gradually *disentangle* the two classes — affine transforms reposition the cloud, relus fold it — until red and blue points sit at opposite one-hot corners.

### 13.3 The geometry of relu in high dimensions

A relu layer maps every point into the nonnegative cone, and any point with some negative coordinates gets projected onto the cone's *faces/axes*. For a width-$N$ layer, the strictly positive orthant is only a fraction

$\frac{1}{2^N}$

of the space — so after a relu almost all data density concentrates on low-dimensional boundary structure. High-dimensional relu representations are therefore intrinsically **sparse**: most of representational space is empty. (WHY it matters: this sparsity is a feature — it makes representations more linearly separable — but it also explains dead units and motivates leaky/parametric relus, see today's paper deep-dive.)

### 13.4 High-dimensional reality check

For real nets, activations are far too high-dimensional to plot, so we use dimensionality reduction (t-SNE) that approximately preserves pairwise distances. Applying t-SNE to layer embeddings of a CLIP-trained ViT (38,400-dim embeddings) shows semantic classes entangled at early layers and cleanly separated clusters near the output — the distribution-transformer story, visible in a real model trained on real images.

## Chapter 14 — Backpropagation

### 14.1 The problem

Gradient descent needs $\nabla_\theta J$ for a loss $J$ computed at the end of a many-layer composition. Backpropagation is nothing more than the **chain rule, organized so that every intermediate result is reused** — a dynamic program on the computation graph that computes all parameter gradients in one backward sweep costing about as much as two forward passes.

### 14.2 Chain rule with Jacobians

For a chain $\mathbf{x} \to \mathbf{h}_1 \to \dots \to \mathbf{h}_L \to J$, the derivative of the scalar loss with respect to any intermediate activation is a product of Jacobian matrices:

$\frac{\partial J}{\partial \mathbf{h}_\ell} = \frac{\partial J}{\partial \mathbf{h}_L}\,\frac{\partial \mathbf{h}_L}{\partial \mathbf{h}_{L-1}}\cdots\frac{\partial \mathbf{h}_{\ell+1}}{\partial \mathbf{h}_\ell}$

where $\partial \mathbf{h}_{k+1}/\partial \mathbf{h}_k$ is the $d_{k+1}\times d_k$ Jacobian of layer $k{+}1$. **WHY backward order:** $J$ is scalar, so $\partial J/\partial \mathbf{h}_L$ is a row vector. Multiplying from the left keeps every intermediate a *vector* (a vector–Jacobian product, VJP) instead of a matrix — cost $O(d^2)$ per layer instead of $O(d^3)$ for full Jacobian products. This single associativity choice is the entire reason backprop is cheap. (Forward-mode would be efficient in the opposite regime: few inputs, many outputs — the reverse of deep learning, which has millions of parameters in and one loss out.)

### 14.3 The backward recursion for an MLP

Define the **error signal** at layer $\ell$ as $\boldsymbol{\delta}_\ell = \partial J/\partial \mathbf{z}_\ell$ (gradient with respect to pre-activations). For $\mathbf{z}_\ell = \mathbf{W}_\ell \mathbf{h}_{\ell-1} + \mathbf{b}_\ell$, $\mathbf{h}_\ell = f(\mathbf{z}_\ell)$, the chain rule gives the recursion

$\boldsymbol{\delta}_\ell = \left(\mathbf{W}_{\ell+1}^\top \boldsymbol{\delta}_{\ell+1}\right) \odot f'(\mathbf{z}_\ell)$

where $\odot$ is elementwise product and $f'$ the derivative of the activation ($f'(z)=\mathbb{1}[z>0]$ for relu). **WHY **$\mathbf{W}^\top$**:** the forward pass sends activations *forward* through $\mathbf{W}$; the transpose routes error *backward* through exactly the same connections — each weight carries credit assignment in reverse proportion to how it carried signal forward.

Parameter gradients then fall out locally:

$\frac{\partial J}{\partial \mathbf{W}_\ell} = \boldsymbol{\delta}_\ell\, \mathbf{h}_{\ell-1}^\top, \qquad \frac{\partial J}{\partial \mathbf{b}_\ell} = \boldsymbol{\delta}_\ell$

**WHY the outer product:** weight $W_{ij}$ connects input unit $j$ to output unit $i$, so its gradient is (error at $i$) × (activation at $j$) — the mathematically exact form of "neurons that fire together, wire together."

### 14.4 The output layer: softmax + cross-entropy

With one-hot target $\mathbf{y}$ and prediction $\hat{\mathbf{y}} = \texttt{softmax}(\mathbf{z}_L)$, the cross-entropy loss is $J = -\sum_k y_k \log \hat{y}_k$. Differentiating through the softmax collapses to the remarkably clean seed for the whole recursion:

$\boldsymbol{\delta}_L = \frac{\partial J}{\partial \mathbf{z}_L} = \hat{\mathbf{y}} - \mathbf{y}$

**WHY it's clean:** the $\log$ cancels the softmax's $\exp$, and the normalizer's derivative produces exactly $\hat{y}_k$. This pairing is deliberate loss design: the gradient is bounded, never saturates, and equals the *residual* — the same form as least squares. It's also why frameworks fuse the two (`CrossEntropyLoss` consumes logits, never probabilities).

### 14.5 Backprop as modular layers — and cost

Each layer only needs to implement two local maps: `forward(x)` and `backward(grad_out)` returning grad_in (a VJP), plus gradients for its own parameters. The graph structure does the rest. Costs: **time** ≈ 2× forward (each backward layer does two matmuls, with $\mathbf{W}^\top$ and the outer product); **memory** = all forward activations must be stored until used by backward — this, not compute, is usually what limits batch size (and is what activation checkpointing and mixed precision, Day 3's paper, attack).

```python
import numpy as np

class Linear:
    def __init__(self, d_in, d_out):
        # He init — see today's paper deep-dive for the derivation
        self.W = np.random.randn(d_out, d_in) * np.sqrt(2.0 / d_in)
        self.b = np.zeros(d_out)
    def forward(self, x):            # x: (d_in,)
        self.x = x                   # cache activation for backward
        return self.W @ x + self.b
    def backward(self, delta):       # delta = dJ/dz: (d_out,)
        self.dW = np.outer(delta, self.x)   # dJ/dW = delta x^T
        self.db = delta                      # dJ/db = delta
        return self.W.T @ delta              # dJ/dx, passed upstream

class ReLU:
    def forward(self, z):
        self.mask = (z > 0)
        return z * self.mask
    def backward(self, grad):
        return grad * self.mask      # multiply by f'(z)

def softmax_xent_backward(logits, y_onehot):
    p = np.exp(logits - logits.max()); p /= p.sum()
    return p - y_onehot              # delta_L = y_hat - y

# forward-then-backward over a list of layers:
# for layer in layers: x = layer.forward(x)
# g = softmax_xent_backward(x, y)
# for layer in reversed(layers): g = layer.backward(g)
```

```python
# Sanity check against PyTorch autograd
import torch
x = torch.randn(2, requires_grad=True)
W = torch.randn(3, 2, requires_grad=True)
z = W @ x
loss = z.relu().sum()
loss.backward()
manual_dW = torch.outer((z > 0).float(), x)   # delta = 1[z>0], dW = delta x^T
assert torch.allclose(W.grad, manual_dW)
```

**Why it matters / connections:** backprop + the layer abstraction is the reason deep learning scales: any new architecture (CNNs Ch 24, transformers Ch 26) is trainable the moment its pieces define forward and backward. And the recursion $\boldsymbol{\delta}_\ell = (\mathbf{W}_{\ell+1}^\top\boldsymbol{\delta}_{\ell+1})\odot f'(\mathbf{z}_\ell)$ makes the vanishing/exploding-gradient problem visible as a *product of matrices and activation derivatives* — which is precisely what today's paper (He init) fixes at initialization time.

# Paper Deep-Dive — Delving Deep into Rectifiers: PReLU + He Initialization (He, Zhang, Ren, Sun; Microsoft Research, ICCV 2015, arXiv:1502.01852)

**Headline:** first published result to surpass human-level top-5 error on ImageNet-1k (4.94% vs human 5.1%), via two rectifier-centric ideas: a learnable activation (PReLU) and an initialization derived *for* rectifier nonlinearities (He/Kaiming init). Read in full from the PDF.

## Part A — PReLU (Parametric ReLU)

### Definition

$f(y_i) = \begin{cases} y_i, & y_i > 0 \\ a_i\, y_i, & y_i \le 0 \end{cases} \iff f(y_i) = \max(0, y_i) + a_i \min(0, y_i)$

where $y_i$ is the pre-activation on channel $i$ and $a_i$ is a **learned** negative-slope coefficient (channel-wise; a channel-shared variant learns one $a$ per layer). $a_i = 0$ recovers ReLU; a small fixed $a_i$ ($0.01$) is Leaky ReLU. **WHY learnable:** Leaky ReLU's fixed slope buys ~nothing in accuracy; letting the model *choose* the slope per channel jointly with all weights lets activations specialize — at a cost of only #channels extra parameters (13 extra params for the channel-shared 14-layer model, yet +1.1% top-1).

### Optimization of the slope

By the chain rule, with $E$ the objective:

$\frac{\partial E}{\partial a_i} = \sum_{y_i} \frac{\partial E}{\partial f(y_i)} \frac{\partial f(y_i)}{\partial a_i}, \qquad \frac{\partial f(y_i)}{\partial a_i} = \begin{cases} 0, & y_i > 0 \\ y_i, & y_i \le 0 \end{cases}$

updated with momentum ($\Delta a_i := \mu \Delta a_i + \epsilon\, \partial E/\partial a_i$), initialized at $a_i = 0.25$, and — key detail — **no weight decay on **$a_i$ (WHY: an $L_2$ penalty would drag $a_i \to 0$, silently biasing PReLU back into plain ReLU). The learned slopes tell a story: conv1's coefficients stay large (~0.6–0.7, keeping both signs of Gabor-like edge responses), while deeper layers shrink toward 0 (increasingly nonlinear, more discriminative).

```python
import torch
import torch.nn as nn

class PReLU(nn.Module):
    """Faithful minimal PReLU (channel-wise), as in Eq.(1) of the paper."""
    def __init__(self, num_channels, init=0.25):
        super().__init__()
        self.a = nn.Parameter(torch.full((num_channels,), init))
    def forward(self, y):                      # y: (N, C, H, W)
        a = self.a.view(1, -1, 1, 1)
        return torch.clamp(y, min=0) + a * torch.clamp(y, max=0)

# IMPORTANT: exclude a from weight decay, per the paper
model_params = [
    {"params": [p for n, p in model.named_parameters() if not n.endswith(".a")],
     "weight_decay": 5e-4},
    {"params": [p for n, p in model.named_parameters() if n.endswith(".a")],
     "weight_decay": 0.0},
]
opt = torch.optim.SGD(model_params, lr=1e-2, momentum=0.9)
```

## Part B — He (Kaiming) initialization: the math, end to end

### Setup

For a conv layer, a response is $\mathbf{y}_l = \mathbf{W}_l \mathbf{x}_l + \mathbf{b}_l$, where $\mathbf{x}_l$ collects the $k \times k$ co-located pixels of $c$ input channels, so each output value is a dot product over $n = k^2 c$ connections; $\mathbf{W}_l$ is $d \times n$ ($d$ filters); $\mathbf{x}_l = f(\mathbf{y}_{l-1})$ and $c_l = d_{l-1}$.

### Forward variance propagation

Assume i.i.d. zero-mean symmetric $w_l$, independent of the i.i.d. $x_l$. Then

$\mathrm{Var}[y_l] = n_l\, \mathrm{Var}[w_l]\, \mathbb{E}[x_l^2]$

**The crucial subtlety (and the whole point of the paper):** $\mathbb{E}[x_l^2] \neq \mathrm{Var}[x_l]$, because $x_l = \max(0, y_{l-1})$ does **not** have zero mean. Xavier/Glorot assumed a linear (zero-mean-preserving) activation; ReLU breaks that assumption. Since $y_{l-1}$ is symmetric around 0, the ReLU keeps only half the mass:

$\mathbb{E}[x_l^2] = \tfrac{1}{2}\mathrm{Var}[y_{l-1}] \;\Rightarrow\; \mathrm{Var}[y_l] = \tfrac{1}{2} n_l \mathrm{Var}[w_l]\, \mathrm{Var}[y_{l-1}]$

Stacking $L$ layers multiplies these factors:

$\mathrm{Var}[y_L] = \mathrm{Var}[y_1] \prod_{l=2}^{L} \tfrac{1}{2} n_l \mathrm{Var}[w_l]$

**WHY the design condition:** any per-layer factor $\beta \neq 1$ compounds to $\beta^L$ — exponential blow-up ($\beta>1$, divergence) or exponential decay ($\beta<1$, stalled learning). So demand each factor be exactly 1:

$\tfrac{1}{2} n_l\, \mathrm{Var}[w_l] = 1 \quad\Rightarrow\quad w_l \sim \mathcal{N}\!\left(0,\; 2/n_l\right), \;\; b_l = 0$

### Backward case

Backprop through the same layer gives $\Delta\mathbf{x}_l = \hat{\mathbf{W}}_l \Delta\mathbf{y}_l$ with fan-out $\hat{n} = k^2 d$ (note $\hat n \neq n = k^2 c$), and ReLU's derivative is 0 or 1 with equal probability, again contributing the factor $\tfrac12$:

$\mathrm{Var}[\Delta x_l] = \tfrac{1}{2} \hat{n}_l\, \mathrm{Var}[w_l]\, \mathrm{Var}[\Delta x_{l+1}] \quad\Rightarrow\quad \tfrac{1}{2}\hat{n}_l \mathrm{Var}[w_l] = 1$

Either condition alone suffices: using the backward one makes the forward product $\prod n_l / \hat n_l = c_2 / d_L$, a benign constant — properly scaling one direction automatically keeps the other non-exponential.

### PReLU generalization and Xavier as a special case

With negative slope $a$, the expectation picks up the slope's energy on the negative half:

$\tfrac{1}{2}\left(1 + a^2\right) n_l\, \mathrm{Var}[w_l] = 1$

$a=0$ → He init; $a=1$ (linear) → exactly Xavier's condition $n_l\mathrm{Var}[w_l]=1$. So Xavier is the linear-activation special case of this formula — the cleanest possible answer to "why did Xavier fail for deep ReLU nets": it is off by the factor $(\sqrt{2})^L$ after $L$ layers.

### Receipts

A 22-layer model converges under both inits (He's just faster) — but a **30-layer** model converges *only* with He init; Xavier's gradients diminish to nothing. The paper also runs the arithmetic for VGG's failed std=0.01 initialization: the correct stds per stage are 0.059/0.042/0.029/0.021, so a constant 0.01 shrinks the conv10→conv2 gradient by $1/(5.9 \times 4.2^2 \times 2.9^2 \times 2.1^4) \approx 1/(1.7\times 10^4)$ — a quantitative post-mortem of why VGG needed pre-training shallower nets first. Final results: PReLU-net C reaches 5.71% top-5 single-model, 4.94% multi-model on ImageNet test.

```python
import torch.nn as nn

def he_init_(module, a=0.0):
    """He init, forward version: std = sqrt(2 / ((1+a^2) * fan_in))."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, a=a,
                                mode="fan_in",        # forward condition; "fan_out" = backward
                                nonlinearity="leaky_relu" if a else "relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)

model.apply(he_init_)   # nn.init.calculate_gain('relu') = sqrt(2) is this paper's factor
```

**Why it matters / connections:** this paper is the bridge between Ch 14's backprop math and practice — it treats the variance of forward activations and backward gradients as first-class design constraints. It directly enabled training VGG-depth nets from scratch, and its fan-in/fan-out reasoning survives verbatim in `torch.nn.init`. ResNet (Day 8) is by the same first author and attacks the *optimization* side of the same depth problem.

# Supplementary resources for this block

- **Udemy — Mastering Computer Vision, Module 2 (DL Foundations & CNNs):** the lectures on forward pass, activation functions, loss surfaces, and backprop mirror Ch 12/14; watch after reading for a second, visual pass. Link: [https://www.udemy.com/course/mastering-computer-vision-from-pixel-to-detection-to-gen-cv/](https://www.udemy.com/course/mastering-computer-vision-from-pixel-to-detection-to-gen-cv/)
- **adensur 04_xavier_glorot_paper_read** ([https://github.com/adensur/blog/tree/main/computer_vision_zero_to_hero/04_xavier_glorot_paper_read](https://github.com/adensur/blog/tree/main/computer_vision_zero_to_hero/04_xavier_glorot_paper_read)): a guided read of Glorot & Bengio 2010 — the variance-preservation argument that today's He paper generalizes. Core idea to retain: Xavier balances forward and backward by compromising, $\mathrm{Var}[w] = 2/(n_{in}+n_{out})$, valid for tanh/linear regimes; He's $2/n$ is the ReLU-corrected version (derivation in Part B above — the two derivations are the same argument differing only in the $\mathbb{E}[x^2]$ step).
- **adensur 05_tensorboard** ([https://github.com/adensur/blog/tree/main/computer_vision_zero_to_hero/05_tensorboard](https://github.com/adensur/blog/tree/main/computer_vision_zero_to_hero/05_tensorboard)): instrumenting training. The Day-4-relevant habit: log activation/gradient histograms per layer — vanishing gradients (the Ch 14 / He-init failure mode) are *visible* as collapsing histograms long before the loss curve tells you.

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/day4_mlp")
for step, (x, y) in enumerate(loader):
    loss = loss_fn(model(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        writer.add_scalar("loss/train", loss.item(), step)
        for name, p in model.named_parameters():      # spot vanishing gradients
            writer.add_histogram(f"grad/{name}", p.grad, step)
```

*(Note: the adensur markdown bodies and visionbook Ch 12/14 pages could not be machine-fetched in full this run — Ch 12/14 were reconstructed against the book's section structure, and Ch 13 + the He paper were read in full from source.)*

# Self-check questions

1. Why does stacking linear layers without nonlinearities collapse to a single linear layer, and which exact algebraic step shows it?
2. Derive $\boldsymbol{\delta}_L = \hat{\mathbf{y}} - \mathbf{y}$ for softmax + cross-entropy. Which properties of $\exp/\log$ make the normalizer's derivative cancel so cleanly?
3. In the backward recursion $\boldsymbol{\delta}_\ell = (\mathbf{W}_{\ell+1}^\top\boldsymbol{\delta}_{\ell+1}) \odot f'(\mathbf{z}_\ell)$, explain why $\mathbf{W}^\top$ (not $\mathbf{W}$) appears, and what the elementwise $f'$ factor does to gradient magnitude for sigmoid vs relu.
4. Reproduce the He-init derivation: where exactly does the factor $\tfrac12$ come from, and why is $\mathbb{E}[x^2] \neq \mathrm{Var}[x]$ the step where Xavier's assumption breaks?
5. A width-$N$ relu layer maps what fraction of embedding space to the strictly-positive orthant, and what does that imply about the geometry of deep representations — and about why PReLU's learnable negative slope might help?

# Latest CV Research — 2026-07-04

Most recent [cs.CV](http://cs.cv/) listings available (arXiv, 16–23 Jun 2026 submissions); picks favor Day 4 themes (activations, representation learning, training).

6. **Spectral Gating via Damped Oscillations for Adaptive Implicit Neural Representations** — Costanzino, Zama Ramirez, Lisanti, Di Stefano (Univ. of Bologna); **ECCV 2026**. Proposes activation machinery for implicit neural representations based on damped-oscillation spectral gating, adapting the frequency content a coordinate network can express. Directly continues today's theme: activation-function design (ReLU→PReLU→periodic/gated) still moves the needle. [https://arxiv.org/abs/2606.23129](https://arxiv.org/abs/2606.23129)
7. **P-JEPA: Procedural Video Representation Learning via Joint Embedding Predictive Architecture** — Tristram, Gasperini, Killeen, et al. (TUM/JHU); arXiv 2606.23256. Extends JEPA-style self-supervised prediction-in-embedding-space to procedural videos, learning task-structure-aware representations without pixel reconstruction. Relevant as a modern instance of Ch 13's "networks as distribution transformers": the training objective is defined entirely on embedding distributions. [https://arxiv.org/abs/2606.23256](https://arxiv.org/abs/2606.23256)
8. **CFPO: Counterfactual Policy Optimization for Multimodal Reasoning** — Yu, Sun, Yang, Wu, Lao (BUPT); **ICML 2026**. A counterfactual-signal policy-optimization scheme for training multimodal LLMs to reason over images, addressing credit assignment in RL post-training — the modern, sequence-level face of the gradient-credit-assignment problem backprop solves at layer level. [https://arxiv.org/abs/2606.23206](https://arxiv.org/abs/2606.23206)
9. **Rethinking Object-Centric Representations for Video Dynamics Modeling** — Wei, Nejjar, Fink (EPFL); arXiv 2606.23436. Questions when slot-style object-centric embeddings actually help dynamics prediction, with controlled comparisons against monolithic representations — useful evidence on which inductive biases in representation space (Ch 13's subject) pay off. [https://arxiv.org/abs/2606.23436](https://arxiv.org/abs/2606.23436)
10. **Semantic Browsing: Controllable Diversity for Image Generation** — Dorfman, Vishnevsky, Dahary, Patashnik, Cohen-Or (TAU); **ECCV 2026**. Introduces controllable-diversity sampling for text-to-image generation, letting users browse semantically distinct modes rather than near-duplicates. A distribution-level control on $p_{\text{out}}$ — literally steering the output distribution a trained network transforms noise into. [https://arxiv.org/abs/2606.23679](https://arxiv.org/abs/2606.23679)
11. **Unmasking LAION-5B: Age, Gender, Race, and Emotion Biases in Large-Scale Image Datasets** — Dominguez-Catena, Paternain, Galar (UPNA); **ICLR 2026 DATA-FM workshop**. Quantifies demographic and affect biases in LAION-5B. Since deep nets are distribution transformers, biased $p_{\text{data}}$ propagates to $p_{\text{out}}$ — empirical grounding for Ch 4 and Day 11's bias-and-shift block. [https://arxiv.org/abs/2606.23204](https://arxiv.org/abs/2606.23204)