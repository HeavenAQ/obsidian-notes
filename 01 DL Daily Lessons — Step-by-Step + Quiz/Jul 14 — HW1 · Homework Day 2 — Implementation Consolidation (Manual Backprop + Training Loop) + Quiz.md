---
base: "[[01.2 DL Daily Lessons — Step-by-Step + Quiz.base]]"
Quiz score: ""
"HW #": 1
Studied: false
Quiz taken: false
Day type: Homework-Day
Date: 2026-07-14
Piece count: 4
---
> ⚠️ **Schedule note (Homework Day 2 of 3):** Jul 13–15 are the HW1 homework days. Yesterday you attempted the **theory** half (Section A Approximation + Section B backprop). Today is the **implementation** attempt — the CIFAR-10 Colab: **Problem 10** (manual forward/backward for Linear, ReLU, CrossEntropy), **Problem 11** (training loop), **Problems 12–15** (training curves, why-not-100%, data augmentation). No new lesson. Below is a compact consolidation of the exact PyTorch-cert lesson mechanics those problems lean on, then a cumulative quiz. **The quiz tests lesson content only — it contains no HW1 answers, and nothing below writes a graded cell for you.** Tomorrow (Jul 15) is finish + submit.
>
> 🗒️ **Quiz backlog nudge:** yesterday's Jul 13 cumulative quiz is still **untaken** (Quiz taken = false, no score), and Jul 12's is unscored too. The patch-up loop only fires on a *low recorded score*, so nothing is queued — but before you code, grade at least **Jul 13** and the **Jul 10 (nanoGPT)** set so you know your backprop footing. Today's 7 questions are fresh (no repeats from the Jul 13 set).

# 🎯 Today's focus

Turn the derivations you consolidated yesterday into working modules. Section C is really one loop wrapped around three tiny gradient facts: **a Linear layer's three gradients, the ReLU gate, and the softmax-CE collapse** — chained by an optimizer step over batches of CIFAR-10. This page re-states the PyTorch-cert C1 mechanics (Dataset/DataLoader discipline, `nn.Module`, losses, optimizers, init) and the batch-shaped backward math, plus the traps that eat time in the notebook. You supply the actual cells.

# 🧩 Consolidation pieces

## Piece 1 — Data pipeline discipline: Dataset, DataLoader, train-only augmentation (powers Problems 14–15, and the curves in 12)

Source: **PT-Cert C1 M3 (Data Management)**, deep dive → read §M3 of [[PyTorch for Deep Learning Professional Certificate (DeepLearning.AI) — Deep Dive Companion]].

Two objects, two jobs. A **`Dataset`** defines `__getitem__` (return one `(x, y)`) and `__len__`; **transforms/augmentation live here**. A **`DataLoader`** wraps a Dataset for batching, shuffling, `num_workers` parallel loading, and `pin_memory` (async host→device copy — see §5 of [[PyTorch × NVIDIA GPU — Training Internals Deep Dive]]).

Two disciplines that decide whether Problems 14–15 are correct:

- **Augmentation goes on train only.** Build two pipelines; validation sees clean images so the metric is honest.
- **Normalization statistics come from the *training* split only** — using val/test stats leaks information.

$$\text{RandomCrop}(32,\ \text{padding}=4):\quad \text{pad to } 40\times40,\ \text{crop a random } 32\times32 \text{ window each epoch}$$

```python
from torchvision import transforms
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   # stochastic → train only
    transforms.ToTensor(),
])
val_tf = transforms.Compose([transforms.ToTensor()])   # deterministic
```

Conceptually augmentation enlarges the effective training distribution and acts as a **regularizer**: expect train accuracy to rise more slowly / peak lower, but the **train–val gap to narrow** (better generalization). That contrast *is* the answer to Problem 15.

**You've got this piece when you can** say which object owns augmentation, why only the train pipeline gets `RandomCrop`, and predict the direction the train/val gap moves once augmentation is on.

## Piece 2 — Manual Linear + ReLU backward, in batch shapes (powers Problem 10)

Source: **PT-Cert C1 M4 (Core NN Components)** + Jul 5 / Jul 10 backprop. Per-sample identities (upstream $g = \partial L/\partial \text{out}$):

$$\frac{\partial L}{\partial x}=W^\top g,\qquad \frac{\partial L}{\partial W}=g\,x^\top,\qquad \frac{\partial L}{\partial b}=g$$

The notebook is **batched**, so with a batch $X\in\mathbb R^{N\times d_{\text{in}}}$, $\text{out}=XW^\top+b$, and upstream $G\in\mathbb R^{N\times d_{\text{out}}}$:

$$\frac{\partial L}{\partial X}=G\,W,\qquad \frac{\partial L}{\partial W}=G^\top X,\qquad \frac{\partial L}{\partial b}=\sum_{n=1}^{N} G_{n,:}$$

Note the batched $\partial L/\partial W = G^\top X$ is exactly the *sum of per-sample outer products* $\sum_n g_n x_n^\top$ — one fused matmul. ReLU's local gradient is the gate:

$$\frac{\partial L}{\partial x}=g\odot\mathbb 1[x>0]$$

```python
class Linear:
    def forward(self, x):                 # x: (N, d_in)
        self.x = x
        return x @ self.W.T + self.b       # (N, d_out)
    def backward(self, grad_out):          # grad_out: (N, d_out)
        self.dW = grad_out.T @ self.x      # (d_out, d_in)  == sum_n g_n x_n^T
        self.db = grad_out.sum(0)          # (d_out,)
        return grad_out @ self.W           # dL/dx: (N, d_in)

class ReLU:
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask
    def backward(self, grad_out):
        return grad_out * self.mask        # zero where the neuron was off
```

**Pitfalls:** decide transposes by **checking shapes, not habit** ($G^\top X$ vs $GW$); the ReLU mask must be captured in *forward* and reused in *backward*; if the loss **averages** over the batch, every gradient carries a $1/N$ — apply it once, consistently.

**You've got this piece when you can** write the three batched Linear gradients and the ReLU backward from memory and justify each output shape by dimension analysis alone.

## Piece 3 — Softmax cross-entropy: the clean gradient and why it's numerically safe (powers Problem 10)

Source: **PT-Cert C1 M4** — "CrossEntropy = LogSoftmax + NLL, applied to *logits*." Read §M4 of [[PyTorch for Deep Learning Professional Certificate (DeepLearning.AI) — Deep Dive Companion]]. For logits $z$, $\hat p=\mathrm{softmax}(z)$, one-hot label $y$:

$$L=-\sum_i y_i\log \hat p_i,\qquad \frac{\partial L}{\partial z}=\hat p - y$$

**Why it collapses:** the softmax Jacobian is $\partial \hat p_i/\partial z_j=\hat p_i(\delta_{ij}-\hat p_j)$, and the log-loss contributes $-1/\hat p_y$ on the true class. Chaining them, the $\hat p_y$ factors cancel and the per-class terms telescope, leaving simply **predicted minus true**.

**Why apply it to logits, never softmax-then-log:** framing loss as $\text{LogSoftmax}+\text{NLL}$ uses the **log-sum-exp** trick, which subtracts $\max_i z_i$ before exponentiating so nothing overflows:

$$\log\sum_i e^{z_i} = m + \log\sum_i e^{z_i-m},\quad m=\max_i z_i$$

```python
def softmax_ce_backward(z, y_idx):        # z: (N, C) logits, y_idx: (N,)
    z = z - z.max(1, keepdim=True).values # log-sum-exp stability
    p = z.exp() / z.exp().sum(1, keepdim=True)
    p[torch.arange(z.size(0)), y_idx] -= 1 # p - onehot(y)
    return p / z.size(0)                  # divide by batch if loss averages
```

**Pitfall:** feeding already-softmaxed probabilities into `CrossEntropyLoss` double-applies softmax and silently wrecks training — pass raw logits.

**You've got this piece when you can** derive $\partial L/\partial z=\hat p-y$ from the softmax Jacobian in two lines and explain what the $-\max_i z_i$ shift buys you.

## Piece 4 — The training loop, optimizers, init, and reading curves (powers Problems 11–13)

Source: **PT-Cert C1 M4** (optimizers, init) + the canonical loop skeleton.

**Per-batch ritual (order is not optional):** zero grads → forward → loss → backward → step. Eval disables grad tracking and never updates.

```python
def train_epoch(model, loader, opt, loss_fn):
    model.train()
    for xb, yb in loader:
        opt.zero_grad(set_to_none=True)   # stale grads accumulate otherwise
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()

@torch.no_grad()
def evaluate(model, loader):
    model.eval()                          # BN/Dropout switch to inference mode
    correct = total = 0
    for xb, yb in loader:
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item(); total += yb.size(0)
    return correct / total
```

Optimizer update rules to keep straight:

$$\text{SGD+momentum:}\quad v\leftarrow \mu v+g,\quad \theta\leftarrow\theta-\eta v$$

$$\text{Adam:}\quad \theta\leftarrow\theta-\eta\,\frac{\hat m}{\sqrt{\hat v}+\epsilon}\ \text{(per-coordinate scaling by }\sqrt{\hat v}\text{)}$$

**Init matters (the same lever HW1's depth questions probe):** Kaiming sets $\mathrm{Var}(W)=2/n_{\text{in}}$ so ReLU activation variance stays stable across depth instead of vanishing/exploding.

**Reading the curves (Problem 12):** watch the **gap** between train and val (generalization), plateaus, and whether loss is still descending at epoch 30. **Why not 100% train accuracy (Problem 13):** the Universal Approximation Theorem guarantees a fitting network *exists*, not that SGD *finds* it — finite width/capacity, incomplete optimization, and learning-rate/regularization effects all intervene; and perfect train accuracy usually signals overfitting rather than a win.

**Pitfalls:** forgetting `zero_grad` accumulates gradients across batches; forgetting `model.eval()`/`no_grad()` at eval leaks BatchNorm/Dropout randomness and wastes memory building the graph.

**You've got this piece when you can** write the five-step per-batch loop from memory, state the SGD-momentum and Adam updates, and give two distinct reasons a real net won't hit 100% train accuracy.

# 📝 Cumulative quiz

Answer all 7 without notes (spans Jul 2–10, weighted toward today's implementation attempt; no repeats from the Jul 13 set). Grade with the key at the bottom, then fill in **Quiz score** and check **Quiz taken**.

1. **(C1 M3 — conceptual)** Which object owns `__getitem__`/`__len__` and which owns batching/shuffling/`num_workers`? Give the two data-hygiene rules that decide whether Problems 14–15 are correct (where augmentation lives; where normalization statistics come from).
2. **(C1 M4 / Jul 10 — code-reading)** In batch form with $X\in\mathbb R^{N\times d_{\text{in}}}$, $\text{out}=XW^\top+b$, upstream $G$: write $\partial L/\partial X$, $\partial L/\partial W$, $\partial L/\partial b$ with correct shapes, and explain why $\partial L/\partial W$ equals a sum of per-sample outer products.
3. **(C1 M4 — derivation)** For softmax logits $z$ with one-hot label $y$, derive $\partial L/\partial z=\hat p-y$ starting from the softmax Jacobian $\hat p_i(\delta_{ij}-\hat p_j)$. Why must the loss be applied to logits rather than pre-softmaxed probabilities?
4. **(C1 M4 — conceptual)** State the log-sum-exp identity used for numerical stability and say what subtracting $\max_i z_i$ prevents.
5. **(C1 M4 — derivation/code)** Write the per-batch training-loop steps in order and the eval-mode safeguards. What goes wrong if you skip `zero_grad`? What goes wrong if you skip `model.eval()`/`no_grad()`?
6. **(C1 M4 — conceptual)** Give the SGD-with-momentum and Adam parameter updates. Separately, state Kaiming init's variance rule $\mathrm{Var}(W)=?$ and why it keeps ReLU activations stable across depth.
7. **(Jul 10 / Prob 13 — conceptual)** Reconcile the Universal Approximation Theorem with the fact that a trained net rarely reaches 100% train accuracy: what does UAT guarantee vs. not, and name two concrete reasons SGD falls short. Why might 100% not even be desirable?

> [!note]- 🔑 Answer key (click to reveal)
> **1.** `Dataset` owns `__getitem__`/`__len__` (and holds the transforms); `DataLoader` owns batching, shuffling, `num_workers` parallel loading, `pin_memory`. Rules: (i) **augmentation lives in the Dataset and only on the train pipeline** — validation sees clean images; (ii) **normalization statistics are computed from the training split only** (using val/test stats leaks information). Getting either wrong makes the Problem 14–15 comparison invalid.
>
> **2.** $\partial L/\partial X=G\,W$ (shape $N\times d_{\text{in}}$), $\partial L/\partial W=G^\top X$ (shape $d_{\text{out}}\times d_{\text{in}}$), $\partial L/\partial b=\sum_{n}G_{n,:}$ (shape $d_{\text{out}}$). $G^\top X=\sum_n g_n x_n^\top$: each sample contributes the rank-1 outer product $g_n x_n^\top$, and stacking + matmul sums them in one fused GPU op. Verify by shapes, not habit.
>
> **3.** With $L=-\sum_i y_i\log\hat p_i$, $\partial L/\partial z_j=\sum_i(-y_i/\hat p_i)\,\hat p_i(\delta_{ij}-\hat p_j)=\sum_i -y_i(\delta_{ij}-\hat p_j)=-y_j+\hat p_j\sum_i y_i=\hat p_j-y_j$ (using $\sum_i y_i=1$). So $\partial L/\partial z=\hat p-y$. It must hit **logits** because `CrossEntropyLoss` = LogSoftmax + NLL and relies on log-sum-exp stability; pre-softmaxing then applying it double-applies softmax and wrecks training.
>
> **4.** $\log\sum_i e^{z_i}=m+\log\sum_i e^{z_i-m}$ with $m=\max_i z_i$. Subtracting the max makes the largest exponent $e^{0}=1$, so no term overflows to `inf` (and underflow of tiny terms is harmless) — the value is identical because $m$ is added back.
>
> **5.** Per batch: (1) `opt.zero_grad()`, (2) forward `out = model(xb)`, (3) `loss = loss_fn(out, yb)`, (4) `loss.backward()`, (5) `opt.step()`. Eval: `model.eval()` (switch BatchNorm/Dropout to inference) inside `@torch.no_grad()` (no graph, no update). Skipping `zero_grad` **accumulates** gradients across batches (grads sum instead of reset → wrong updates). Skipping `eval`/`no_grad` leaves Dropout/BN in train mode (noisy, wrong metric) and builds the autograd graph needlessly (wasted memory).
>
> **6.** SGD+momentum: $v\leftarrow\mu v+g;\ \theta\leftarrow\theta-\eta v$. Adam: $\theta\leftarrow\theta-\eta\,\hat m/(\sqrt{\hat v}+\epsilon)$ (bias-corrected first moment $\hat m$, per-coordinate scaling by $\sqrt{\hat v}$). Kaiming: $\mathrm{Var}(W)=2/n_{\text{in}}$; the factor 2 compensates for ReLU zeroing ~half the activations, keeping activation (and gradient) variance roughly constant across layers instead of vanishing/exploding.
>
> **7.** UAT guarantees a network *exists* that approximates the target to arbitrary accuracy given enough width; it says **nothing** about whether SGD will find it. Two concrete shortfalls: finite/insufficient capacity for the chosen width, and optimization not converged (learning rate, non-convex landscape, limited epochs) — regularization/augmentation also caps train fit deliberately. 100% train accuracy is usually **overfitting**: memorizing training points hurts validation generalization, so it isn't the goal.
>
> **Scoring:** 1 pt each (half credit for the right idea with a wrong detail). ≥ 5/7 → your implementation mechanics are solid; go write Problems 10–15. Below 5 → re-read the tagged consolidation piece above before coding the matching cell.
