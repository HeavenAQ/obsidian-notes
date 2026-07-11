---
base: "[[DL Daily Lessons — Step-by-Step + Quiz.base]]"
Quiz score: ""
"HW #": 1
Studied: false
Quiz taken: false
Day type: Lesson-Day
Date: 2026-07-05
Piece count: 5
---
> ⚠️ **Schedule note:** yesterday's (Jul 4) lesson page was never posted, so the Udemy matrix-calculus material is folded into **Piece 1** below in condensed form — it's the highest-priority item (HW1's backprop math). The PT-Cert C1 M1–M2 skim (mostly review) moves to the **Jul 12 buffer day**. Today otherwise follows the plan: PT-Cert C1 M3–M4 + 6.S191 L1 + GPU deep dive §1–2. Also: the Jul 3 quiz is still ungraded — score it first (10 min).

# 🎯 Today's goal

Build the two skills HW1's Sections B and C lean on hardest: differentiating matrix/vector expressions with index notation (Problem 8's $\partial L/\partial W$ derivation), and the PyTorch data-to-optimizer pipeline (Dataset/DataLoader, nn.Module, losses, optimizers) that Problems 10–15 ask you to partly re-implement by hand. By tonight you should be able to derive $\partial L/\partial W = \delta x^\top$ for a linear layer *and* write a full training script from an empty file.

# 🧩 Pieces

## Piece 1 — Matrix calculus for backprop (catch-up, ~40 min)

*Source: Udemy LinAlg — Intro to Matrices + Matrix Calculus sections (the missed Jul 4 item). This is the math engine for HW1 Problems 8 and 10.*

The workflow is always **scalarize → differentiate with indices → recognize the matrix pattern**. Start from a scalar loss $L$ and one scalar parameter $W_{ij}$, never from "matrix derivative" formulas you can't rebuild.

Three tools do all the work:

**Tool 1 — Kronecker delta collapses sums.** Since $\partial W_{kl}/\partial W_{ij} = \delta_{ki}\delta_{lj}$,

$$
\frac{\partial}{\partial W_{ij}} \sum_{k,l} W_{kl}\, x_l = \sum_{k,l} \delta_{ki}\delta_{lj}\, x_l = x_j \quad (\text{alive only in row } i)
$$

**Tool 2 — the outer-product pattern.** For $y = Wx$, $L$ scalar, define $\delta_i = \partial L/\partial y_i$. Then

$$
\frac{\partial L}{\partial W_{ij}} = \sum_k \frac{\partial L}{\partial y_k}\frac{\partial y_k}{\partial W_{ij}} = \delta_i\, x_j \;\;\Longrightarrow\;\; \frac{\partial L}{\partial W} = \delta x^\top
$$

A free index pair $(i, j)$ multiplying as $a_i b_j$ **is** an outer product — this single recognition explains why every linear layer's weight gradient in every framework is `grad_out.T @ input`. Over a batch it's the outer-product *view* of matmul from the Jul 3 lesson: $\frac{\partial L}{\partial W} = \sum_{n} \delta^{(n)} \otimes x^{(n)}$, a sum of rank-1 matrices — one GEMM on the GPU.

**Tool 3 — elementwise nonlinearities become diagonal gates.** For $u = \mathrm{ReLU}(y)$, $\partial u_k/\partial y_l = \Theta(y_k)\,\delta_{kl}$ where $\Theta$ is the Heaviside step. Chained, this inserts $\mathrm{diag}(\Theta(y))$ — a mask that zeroes gradient flow through dead neurons. Never a full Jacobian matmul: elementwise forward ⇒ diagonal backward.

```python
import torch
# verify Tool 2 numerically
W = torch.randn(3, 4, requires_grad=True)
x = torch.randn(4)
y = W @ x
L = (y ** 2).sum()          # so dL/dy = 2y
L.backward()
delta = 2 * y.detach()
print(torch.allclose(W.grad, torch.outer(delta, x)))  # True
```

**You've got this piece when you can** derive $\partial L/\partial W$, $\partial L/\partial b$, $\partial L/\partial x$ for $y = Wx + b$ from pure index notation without looking, and say which of the three results is an outer product and why.

## Piece 2 — Dataset & DataLoader (PT-Cert C1 M3, ~30 min)

PyTorch splits data loading into two orthogonal objects. A `**Dataset**` is just an indexable protocol — `__len__` and `__getitem__` — that returns *one* sample (this is where decode + augmentation live). A `**DataLoader**` wraps any Dataset and handles the *batching machinery*: shuffling via a sampler, collating samples into stacked tensors, and parallelism via worker processes.

Why this design: sample-level logic (CPU-bound, per-item) is cleanly separated from iteration logic (batching, shuffling, prefetch), so the same Dataset class works for train and eval with different loaders.

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X, self.y, self.transform = X, y, transform
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)   # augmentation happens HERE, per-sample
        return x, self.y[idx]

train_loader = DataLoader(MyDataset(Xtr, ytr, transform=aug),
                          batch_size=128, shuffle=True,
                          num_workers=4, pin_memory=True,
                          persistent_workers=True)
val_loader = DataLoader(MyDataset(Xva, yva), batch_size=256, shuffle=False)
```

HW1 connection: Problem 14 asks for augmentation on the **train pipeline only** — just two Dataset instances with different `transform`s. The `pin_memory=True` flag page-locks host memory so host→GPU copies can run async — that's §5 of the GPU deep dive; today's assigned §1–2 explain *why* async matters at all.

**You've got this piece when you can** explain what `shuffle=True` actually shuffles (indices via the sampler, not the data array), why augmentation goes in `__getitem__` and not on the batch, and what `num_workers` parallelizes.

## Piece 3 — nn.Module anatomy (PT-Cert C1 M4a, ~25 min)

`nn.Module` is a **registry + recursion** pattern. Assigning an `nn.Parameter` or a sub-`Module` as an attribute registers it in `_parameters` / `_modules`; `model.parameters()` then walks the tree recursively. That's the entire trick — and it's why the optimizer can be constructed with one line, without you enumerating weights.

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()                     # must run first: sets up registries
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),         # registered as submodule
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )
    def forward(self, x):                      # define forward; NEVER call it directly
        return self.net(x)                     # use model(x) → __call__ → hooks + forward

model = MLP(3072, 512, 10)
sum(p.numel() for p in model.parameters())     # 3072*512+512 + 512*10+10
```

Why `model(x)` and not `model.forward(x)`: `__call__` wraps `forward` with hook dispatch (used by profilers, DDP, quantization). Also know `model.train()` vs `model.eval()` — they flip a `self.training` flag that changes Dropout/BatchNorm behavior, and are unrelated to gradient tracking (`torch.no_grad()` is a separate mechanism; eval mode does not disable autograd).

HW1 connection: Problem 10 makes you re-implement `Linear`'s forward/backward manually — Piece 1's $\delta x^\top$ *is* what `nn.Linear` stores into `.grad`.

**You've got this piece when you can** write an MLP class from memory, state what gets auto-registered and how `parameters()` finds it, and articulate the difference between `eval()` and `no_grad()`.

## Piece 4 — Losses & optimizers (PT-Cert C1 M4b, ~35 min)

**Cross-entropy.** `nn.CrossEntropyLoss` takes **raw logits** $z$ (never softmax outputs — it applies log-softmax internally for numerical stability via the log-sum-exp trick):

$$
L = -\log \frac{e^{z_c}}{\sum_k e^{z_k}} = -z_c + \log \sum_k e^{z_k}
$$

Its gradient is the famously clean $\frac{\partial L}{\partial z} = \hat{p} - y_{\text{onehot}}$ with $\hat p = \mathrm{softmax}(z)$ — the softmax Jacobian $\mathrm{diag}(\hat p) - \hat p \hat p^\top$ contracts against the log-loss gradient and everything cancels. HW1 Problem 10 asks you to derive exactly this.

**Optimizers.** SGD with momentum keeps a velocity buffer:

$$
v_{t+1} = \mu v_t + g_t, \qquad \theta_{t+1} = \theta_t - \eta\, v_{t+1}
$$

Adam keeps per-parameter first/second moment estimates and normalizes the step:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2, \quad \theta_t = \theta_{t-1} - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
$$

where $\hat m_t = m_t/(1-\beta_1^t)$ and $\hat v_t = v_t/(1-\beta_2^t)$ are bias corrections — needed because $m_0 = v_0 = 0$ biases early estimates toward zero. *Why* divide by $\sqrt{\hat v_t}$: it makes the step roughly scale-invariant per parameter (steep, high-variance directions get smaller steps), which is why Adam needs less LR tuning than SGD.

```python
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
for xb, yb in train_loader:
    opt.zero_grad(set_to_none=True)  # grads ACCUMULATE by default — must clear
    loss = loss_fn(model(xb), yb)
    loss.backward()
    opt.step()
```

**You've got this piece when you can** explain why CE takes logits not probabilities, reproduce $\hat p - y$ from the log-sum-exp form, write both update rules from memory, and say what breaks if you forget `zero_grad()`.

## Piece 5 — 6.S191 L1 + how the GPU actually runs your step (~30 min)

*Sources: 6.S191 Lecture 1 (perceptron → MLP → gradient descent — fast watch, mostly review after Pieces 1–4) and* 🔥 [*PyTorch × NVIDIA GPU — Training Internals*](https://app.notion.com/p/391e445b30a48184a35fca32915f2b2c) *§1–2 — read both sections in full.*

Two ideas from the deep dive to internalize now:

**§1 — async by default.** Every CUDA op is *enqueued on a stream* and returns immediately; the CPU races ahead building the queue. Sync happens only at `.item()`, `.cpu()`, `torch.cuda.synchronize()`, or cross-stream events. Two consequences the doc calls out: timing with `time.time()` without a sync is meaningless (use `torch.cuda.Event`), and a slow CPU launching many tiny kernels starves the GPU — the "launch-bound" regime that CUDA Graphs and `torch.compile` fix.

**§2 — the roofline logic.** A GEMM of $(M,K)\times(K,N)$ does

$$
\text{FLOPs} = 2MKN, \qquad \text{bytes moved} \approx 2(MK + KN + MN)
$$

and is compute-bound only when its arithmetic intensity (FLOPs/byte) beats the GPU's ratio (~300 for H100 bf16). This is the quantitative reason "bigger batch = better utilization" and why elementwise ops (ReLU!) are bandwidth-bound. Connect to Piece 1: your $\delta x^\top$ outer product is exactly one such GEMM, and tensor cores only engage when dtype/alignment allow (fp16/bf16/tf32, dims multiples of 8/16).

```python
# correct GPU timing (deep dive §1)
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
out = model(x)
end.record()
torch.cuda.synchronize()
print(start.elapsed_time(end), 'ms')
```

**You've got this piece when you can** explain why a Python-side timer lies about kernel time, define arithmetic intensity and compute it for a square matmul, and say whether ReLU is compute- or bandwidth-bound and why.

# 📝 Review quiz

Answer all 8 without notes. Grade with the key at the bottom, then fill in Quiz score and check Quiz taken.

1. **(Derivation)** For $y = Wx + b$ with upstream gradient $\delta = \partial L/\partial y$, derive $\partial L/\partial W$, $\partial L/\partial b$, and $\partial L/\partial x$ using index notation, showing where the Kronecker delta collapses each sum.
2. **(Conceptual)** In the chain for $u = \mathrm{ReLU}(y)$, why does the backward pass insert $\mathrm{diag}(\Theta(y))$ rather than a dense Jacobian? What does this matrix do to gradients of "off" neurons?
3. **(Code writing)** You have one CIFAR dataset but want random crops on train only (HW1 Problem 14 setup). Sketch in code how the Dataset/DataLoader separation makes this a two-line difference, and state precisely where the augmentation executes (which object, which method, per-sample or per-batch).
4. **(Conceptual)** What does assigning `self.fc = nn.Linear(10, 5)` inside `__init__` trigger in `nn.Module`, and how does that make `optim.SGD(model.parameters(), lr=0.1)` work without listing weights? Also: does `model.eval()` stop gradients from being computed?
5. **(Derivation)** Show that $\partial L/\partial z = \hat p - y_{\text{onehot}}$ for softmax + cross-entropy, starting from $L = -z_c + \log\sum_k e^{z_k}$. Why does `nn.CrossEntropyLoss` insist on raw logits as input?
6. **(Derivation)** Write Adam's update equations including bias correction. Explain (a) why the correction is needed and (b) what dividing by $\sqrt{\hat v_t}$ buys you compared to plain SGD.
7. **(Code reading)** This code reports the forward pass takes 0.01 ms on GPU — implausibly fast:

```python
t0 = time.time()
out = model(x)          # x on cuda
print((time.time() - t0) * 1000, 'ms')
```

What's wrong, and what is the correct measurement pattern? (Deep dive §1.)

8. **(Calculation)** A $4096 \times 4096 \times 4096$ bf16 GEMM: compute its FLOPs, its approximate bytes moved, and its arithmetic intensity. Is it compute-bound on an H100 (ratio ≈ 300 FLOPs/byte)? Would a $64 \times 64 \times 64$ GEMM be? (Deep dive §2.)

> [!note]+ 🔑 Answer key — open only after attempting all 8
> **1.** $L$ depends on $W_{ij}$ only through $y_i = \sum_l W_{il}x_l + b_i$. Then $\frac{\partial L}{\partial W_{ij}} = \sum_k \delta_k \frac{\partial y_k}{\partial W_{ij}} = \sum_k \delta_k\, \delta_{ki} x_j = \delta_i x_j$, i.e. $\partial L/\partial W = \delta x^\top$ (outer product — free indices $i,j$). For $b$: $\partial y_k/\partial b_i = \delta_{ki}$ so $\partial L/\partial b = \delta$. For $x$: $\partial y_k/\partial x_j = W_{kj}$, so $\partial L/\partial x = W^\top \delta$.
> 
> **2.** ReLU is elementwise: $u_k$ depends only on $y_k$, so the Jacobian $\partial u_k/\partial y_l = \Theta(y_k)\delta_{kl}$ is diagonal by construction — off-diagonal entries are exactly zero, so a dense matrix would be wasted memory and compute. Neurons with $y_k < 0$ have $\Theta(y_k) = 0$: their gradient is zeroed and no learning signal flows through dead units.
> 
> **3.** `train_ds = MyDataset(X, y, transform=RandomCrop(32, padding=4))` and `val_ds = MyDataset(Xv, yv, transform=None)` — same class, different `transform`. Augmentation runs inside `Dataset.__getitem__`, per-sample, on CPU (in a worker process if `num_workers>0`), *before* the DataLoader collates the batch. The DataLoader itself never touches augmentation.
> 
> **4.** `nn.Module.__setattr__` intercepts the assignment and registers the Linear in `_modules`; `parameters()` recursively yields its weight and bias, so the optimizer receives every parameter in the tree automatically. And no — `eval()` only sets `self.training = False` (changing Dropout/BatchNorm behavior); autograd still records operations. Only `torch.no_grad()` / `inference_mode` disables gradient tracking.
> 
> **5.** $\frac{\partial L}{\partial z_j} = -\delta_{jc} + \frac{e^{z_j}}{\sum_k e^{z_k}} = \hat p_j - (y_{\text{onehot}})_j$ — the second term is softmax by definition; the first is 1 only for the true class. Logits are required because the loss computes log-softmax internally with the log-sum-exp trick ($\log\sum_k e^{z_k} = z_{\max} + \log\sum_k e^{z_k - z_{\max}}$), which is numerically stable; feeding softmaxed probabilities would double-apply softmax and destroy both stability and correctness.
> 
> **6.** $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$; $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$; $\hat m_t = m_t/(1-\beta_1^t)$; $\hat v_t = v_t/(1-\beta_2^t)$; $\theta_t = \theta_{t-1} - \eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$. (a) With zero-initialized moments, early $m_t, v_t$ are biased toward 0 by factor $(1-\beta^t)$; dividing by it corrects the expectation. (b) $\sqrt{\hat v_t}$ normalizes each parameter's step by its gradient RMS — an adaptive per-parameter learning rate that damps high-variance directions and lets one global $\eta$ work across layers with very different gradient scales.
> 
> **7.** CUDA ops are asynchronous: `model(x)` only *enqueues* kernels and returns immediately; the timer measures Python launch overhead, not GPU execution. Correct pattern: a `torch.cuda.Event` pair — `start.record()`, run the op, `end.record()`, `torch.cuda.synchronize()`, then `start.elapsed_time(end)`.
> 
> **8.** FLOPs $= 2 \cdot 4096^3 \approx 1.37\times 10^{11}$. Bytes $\approx 2 \cdot 3 \cdot 4096^2 \approx 1.0\times 10^8$ (bf16 = 2 bytes/element, three matrices). Intensity $\approx 1365$ FLOPs/byte $\gg 300$ → compute-bound ✓. For $64^3$: FLOPs $= 2\cdot 64^3 \approx 5.2\times 10^5$, bytes $\approx 2\cdot 3\cdot 64^2 \approx 2.5\times 10^4$, intensity $\approx 21 \ll 300$ → bandwidth/launch-bound. Small matmuls waste the GPU — batch them.
> 
> **Scoring:** 1 pt each (half credit for right idea, wrong arithmetic). Score ≥ 6/8 → move on; below → tomorrow starts with a patch-up.