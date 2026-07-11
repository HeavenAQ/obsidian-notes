---
base: "[[DL Homework Practice — MIT 6.7960.base]]"
Status: In progress
Notes ready: true
Window: "Jul 2 – Jul 15, 2026 (2-week block: HW1 + PyTorch cert + Transformers in Practice)"
Due Date: 2026-07-15
"HW #": 1
PDF: https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_hw1.pdf
Topics: ReLU networks; layer/shape bookkeeping; convexity & concavity; piecewise-linearity & linear regions; differentiability/kink counting; depth efficiency & universal approximation; logic-gate networks (OR/XOR/NAND); matrix-calculus backprop (Kronecker delta, Heaviside, outer products); manual forward/backward (Linear, ReLU, softmax-CE); training loop; learning curves; data augmentation
---
# 📚 Two-week plan — HW1 + PyTorch Cert + Transformers in Practice (due Jul 15)

Goal: finish [**PyTorch for Deep Learning Professional Certificate**](https://www.deeplearning.ai/specializations/pytorch-for-deep-learning-professional-certificate) (3 courses × 4 modules, ~87h nominal — skim what you already know) and [**Transformers in Practice**](https://www.deeplearning.ai/courses/transformers-in-practice) (3 modules, ~3h) alongside HW1. Each day links its deep-dive doc in the Document Hub — read the lesson first, then the deep dive, then take the tutor's quiz.

- [x] **Jul 2** — 18.02SC Unit 1 Part A (vectors, dot products, determinants, planes)
- [x] **Jul 3** — 18.02SC Unit 1 Part B (matrix multiplication, inverses, systems)
- [ ] **Jul 4** — Udemy LinAlg: Intro to Matrices + Matrix Calculus (HW1's backprop math) · **PT-Cert C1 M1–M2** (Getting Started, PyTorch Workflow — *skim, mostly review*)
- [ ] **Jul 5** — **PT-Cert C1 M3–M4** (Data Management: Dataset/DataLoader; Core NN Components: nn.Module, losses, optimizers) · 6.S191 L1 → deep dive: [PyTorch × NVIDIA GPU — Training Internals](https://app.notion.com/p/391e445b30a48184a35fca32915f2b2c) §1–2
- [ ] **Jul 6** — **PT-Cert C2 M1–M2** (Hyperparameter Optimization w/ Optuna; TorchVision: transforms, fine-tuning ResNet/MobileNet) · 6.S191 L3 (CNNs)
- [ ] **Jul 7** — **PT-Cert C2 M3–M4** (Text w/ Hugging Face: GloVe/FastText/DistilBERT embeddings; Efficient Training Pipelines: profiling, bottlenecks) → deep dive: [PyTorch × GPU](https://app.notion.com/p/391e445b30a48184a35fca32915f2b2c) §3–5, 8 (AMP, allocator, dataloading, profiling)
- [ ] **Jul 8** — **TiP Module 1** (autoregressive loop, token sampling, structured outputs/FSM, RAG, CoT) → deep dive: [Transformers in Practice](https://app.notion.com/p/391e445b30a4817dbfb5ef315a1f06ff) §Module 1
- [ ] **Jul 9** — **TiP Module 2** (attention, positional encoding, layers, logit lens) → deep dive: [Transformers in Practice](https://app.notion.com/p/391e445b30a4817dbfb5ef315a1f06ff) §Module 2
- [ ] **Jul 10** — 🎥 **Karpathy — **[**"Let's build GPT: from scratch, in code, spelled out"**](https://www.youtube.com/watch?v=kCc8FmEb1nY) (~2h — CODE ALONG, don't just watch: bigram LM → single self-attention head → multi-head → residual blocks + LayerNorm → full nanoGPT on Shakespeare). Turns yesterday's TiP Module 2 theory into working code, and its manual attention implementation is the same matrix-calculus skill as HW1 Problem 8/10 → deep dive: [Transformers in Practice](https://app.notion.com/p/391e445b30a4817dbfb5ef315a1f06ff) §Module 2
- [ ] **Jul 11** — **TiP Module 3** (quantization, KV cache, flash attention, speculative decoding) + **PT-Cert C3 M4** (Deployment: ONNX, MLflow, pruning, quantization/QAT) → deep dives: [TiP](https://app.notion.com/p/391e445b30a4817dbfb5ef315a1f06ff) §Module 3 · [ONNX & ONNX Runtime](https://app.notion.com/p/391e445b30a48171a735f4ba51833317)
- [ ] **Jul 12** — **PT-Cert C3 M1–M3** (Custom Architectures; Specialized Vision: Siamese, ResNets; Specialized NLP — *the transformer-from-components part is now review after Karpathy: skim*) · finish both courses' graded quizzes & labs · re-read HW1 prep notes below
- [ ] **Jul 13** — Attempt HW1 (theory problems: ReLU nets, approximation, backprop derivations)
- [ ] **Jul 14** — Attempt HW1 (implementation problems: manual forward/backward, training loop — apply PT-Cert C1 skills)
- [ ] **Jul 15** — Finish + submit HW1. ✅ Done = HW1 submitted, both certificates' modules completed

*Overlap guide: PT-Cert C1 M1–M2 and C3 M3's attention content repeat what you know / what TiP+Karpathy cover — skim. 6.S191 L2 stays in HW2 week. HW7's "Build-a-GPT" section (Mike Cohen course) becomes largely implementation review after the Karpathy day. Buffer day absorbed into Jul 12 — if you slip a day, compress PT-Cert C3 M1–M3 rather than the Karpathy or TiP days.*

---

# 📦 Superseded (kept for reference) — original one-week plan

Take these lessons first — they cover exactly the math and concepts HW1 needs (ReLU nets, approximation, matrix-calculus backprop, CNNs). Skip anything you already know well.

- [ ] **MIT 18.02SC Unit 1, Part A** (Sessions 1–8): vectors, dot products, lengths & angles, determinants, cross products, planes — [link](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/pages/1.-vectors-and-matrices/part-a-vectors-determinants-and-planes/)
- [ ] **MIT 18.02SC Unit 1, Part B** (Sessions 9–14): matrix multiplication & its meaning, inverses, linear systems — [link](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/pages/1.-vectors-and-matrices/part-b-matrices-and-systems-of-equations/) *(Part C parametric curves: skip — not needed for DL)*
- [ ] **Udemy Linear Algebra** → "Introduction To Matrices" section: linear independence, rank, the 5 ways to multiply matrices (column/row/outer-product/block views — the outer-product view is exactly how backprop gradients ∂L/∂W = δxᵀ arise)
- [ ] **Udemy Linear Algebra** → Matrix Calculus section: differentiating matrix/vector equations (core skill for HW1's backprop derivations). *(Skip Gaussian-elimination section if Part B above was comfortable; skip all Matlab-only, Computer Graphics, Robotics & Control sections)*
- [ ] **MIT 6.S191 Lecture 1** — Intro to Deep Learning ([slides/video](https://introtodeeplearning.com/)) and **Lecture 3** — Deep Computer Vision (CNNs, for the architectures-for-grids material)

---

Reading notes will be added by the morning coach when this homework's 3-day window opens.

# 📘 HW1 Reading Notes — Approximation, Backprop & CIFAR-10

*Prep notes only — concepts, theory, derivations and PyTorch techniques to let you solve each problem yourself. No final answers below. 29 pts total; bonus parts are ungraded.*

---

## 0. Foundations you'll use everywhere

A **ReLU network** with $l$ weight matrices (the homework counts layers by weight matrices) maps a scalar (or vector) input through alternating affine maps and ReLU nonlinearities. The $l=2$ case is:

$f(x; W_1, W_2, b_1, b_2) = W_2\,\mathrm{ReLU}(W_1 x + b_1) + b_2$

Key facts to internalize before answering anything:

- **ReLU** is $\mathrm{ReLU}(z) = \max(0, z)$, applied elementwise. It is continuous everywhere, differentiable everywhere *except* at $z=0$ (a "kink").
- An **affine map** $z \mapsto Wz + b$ is both convex and concave (it's linear-plus-constant).
- **Composition rule for shapes:** if a vector enters a layer with dimension $d_{\text{in}}$ and the layer outputs $d_{\text{out}}$, then $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and $b \in \mathbb{R}^{d_{\text{out}}}$. The next layer's input dimension must equal this layer's output dimension. Trace the dimensions left to right and every shape falls out.
- A **convex function** satisfies $g(\lambda a + (1-\lambda) b) \le \lambda g(a) + (1-\lambda) g(b)$. Convexity is preserved under (i) nonnegative weighted sums and (ii) composition $g(\cdot)$ where the outer function is convex *and non-decreasing*. This is the lever for the convexity questions.

---

## Section A — Approximation (14 pt)

### Problem 1 (1pt) — Shapes of $W_2, b_1, b_2$

**Requires:** dimensional bookkeeping for layered affine maps.

Technique: the input is a scalar, so the first pre-activation $W_1 x + b_1$ lives in $\mathbb{R}^{k}$ (width $k$). Apply the composition rule above: $b_1$ matches the output dimension of layer 1; $W_2$ must consume a $k$-vector and produce the network's scalar output; $b_2$ matches that output. Write each shape as $(\text{rows} \times \text{cols})$ and double-check that every adjacent matrix-vector product is conformable.

### Problem 2 (1pt) — Write out an $l=3$ ReLU network

**Requires:** composition of three affine maps with two ReLU layers.

Pattern: nest the $l=2$ expression inside one more affine+ReLU stage. The general recipe is

$f = W_3\,\mathrm{ReLU}\big(W_2\,\mathrm{ReLU}(W_1 x + b_1) + b_2\big) + b_3.$

Make sure the intermediate widths are consistent (you can keep them all width $k$). The number of ReLU applications is always $l-1$.

### Problem 3 (1pt) — Is the output convex / concave in $x$?

**Requires:** convexity under composition, and a counterexample mindset.

Think about whether a *composition of convex functions* stays convex. ReLU is convex and non-decreasing, so $\mathrm{ReLU}(\text{affine})$ is convex. But the **outer** weights $W_2$ can be **negative**, which flips convex into concave. Once you stack signs arbitrarily you can build a function that is neither. Strategy: try to construct a tiny network (width 1–2) whose graph has both an upward and a downward kink; if you can, "in general" the answer to both convex and concave is settled. Convexity claims are killed by a single counterexample.

### Problem 4 (5pt) — Discontinuities, piecewise-linearity, differentiability, depth efficiency

**Requires:** the theory of **linear regions** of ReLU networks. This is the conceptual heart of the section.

**(a) Discontinuities.** A composition of continuous functions is continuous. ReLU and affine maps are continuous, so reason about whether *any* discontinuity can appear at all.

**(b) Linear / piecewise-linear / polynomial.** Each ReLU partitions input space by a hyperplane $\{z : (W_1 x + b_1)_j = 0\}$. On each side the unit is either "off" (outputs 0) or "on" (outputs its affine input). Within any region where every neuron's on/off pattern is fixed, the whole network reduces to a single affine map. So the function is **piecewise** something — identify which.

**(c) Non-differentiability and counting linear regions.** Non-differentiable points are exactly the **boundaries between linear regions** (the kinks). The key mental model the hint pushes:

- Each neuron is a separating hyperplane on the *previous layer's output*.
- Adding neurons in one layer subdivides existing regions roughly additively.
- Adding *depth* lets later layers fold/subdivide regions created by earlier layers — this is **multiplicative**.

So for the smallest possible number of kinks, ask: what's the minimum a network *must* have (could be zero if all weights collapse to linear)? For $l=2$, width $k$, count how many hyperplanes can independently cut a 1-D input line — each neuron contributes at most one kink, giving a bound **linear in **$k$. For general $l$, the number of linear regions can grow **exponentially in depth** (Montúfar et al. 2014 is the reference result): each added layer can multiply the region count. Match this to the four multiple-choice options by asymptotics, not exact counts.

**(c-iv) Depth efficiency.** If a depth-$l$ network can produce exponentially many linear regions with a *polynomial* number of neurons, while a shallow ($l=2$) net needs one neuron per region, then deep nets represent some highly oscillatory functions far more parameter-efficiently. Frame your answer around region-count growth rates.

**(d) tanh instead of ReLU.** $\tanh$ is smooth ($C^\infty$) everywhere, so re-examine the differentiability sub-questions: a smooth nonlinearity composed with affine maps stays differentiable everywhere. The "kink-counting" logic that applied to ReLU no longer creates non-differentiable points.

### Problem 5 (2pt) — Width-2, bias-free network with prescribed smoothness

**Requires:** constructing explicit $W_1, W_2$ and **plotting** the resulting scalar function on $[-5,5]$.

With no biases, every kink hyperplane passes through the origin: $\mathrm{ReLU}(w x)$ has its kink at $x=0$ regardless of $w$. That constraint is the whole puzzle.

- **(a) Linear:** find weights so the two ReLU pieces recombine into a straight line (hint: a positive and a negative copy of the same input can reconstruct the identity, since $\mathrm{ReLU}(x) - \mathrm{ReLU}(-x) = x$).
- **(b) Two non-differentiable points:** ask whether two bias-free width-2 ReLUs can place kinks at *two distinct* $x$ values — given that bias-free kinks are pinned at the origin, reason carefully about whether this is even possible.
- **(c, d) Bonus:** convex-not-linear and neither-convex-nor-concave shapes.

PyTorch technique for the plots:

```python
import torch
import matplotlib.pyplot as plt

def f(x, W1, W2):
    # x: (N,1); W1: (2,1); W2: (1,2); no biases
    h = torch.relu(x @ W1.T)      # (N,2)
    return h @ W2.T                # (N,1)

x = torch.linspace(-5, 5, 400).unsqueeze(1)
W1 = torch.tensor([[ ... ],[ ... ]])  # you choose
W2 = torch.tensor([[ ... , ... ]])
plt.plot(x.squeeze(), f(x, W1, W2).squeeze())
```

When a case is impossible, your job is to *argue* it from the origin-pinned-kink fact, not to plot.

### Problem 6 (Bonus, 0pt) — Non-convexity in parameter space

**Requires:** distinguishing convexity in the *input* from convexity in the *parameters*. Fix an input $x_0$, pick two parameter vectors $\theta_A, \theta_B$, and plot $f(x_0; \theta_t)$ along $\theta_t = (1-t)\theta_A + t\theta_B$. A non-convex, non-concave curve demonstrates why neural-net loss landscapes are non-convex.

### Problem 7 (4pt) — Logic-gate ReLU networks

**Requires:** designing weights so that **decision boundaries** land where a ReLU input crosses zero. Treat each ReLU as a branch at $0$.

- **(a) OR gate.** You want $f(x) > 0 \iff x_1 > 0 \text{ OR } x_2 > 0$. Idea: a sum like $\mathrm{ReLU}(x_1) + \mathrm{ReLU}(x_2)$ is positive iff at least one coordinate is positive. Shape $W_1, W_2$ so the network computes that combination, then verify the strict-inequality condition on the **non-boolean** real inputs.
- **(b) XOR gate.** Not linearly separable, so you need the extra depth/width budget (≤3 layers, width ≤4). Standard route: build **AND** and **OR**/negation pieces and compose them, since $\text{XOR}(a,b) = (a \,\text{OR}\, b)\,\text{AND}\,\neg(a\,\text{AND}\,b)$. Map the sign conditions $x_1<0, x_2>0$ etc. to ReLU inputs that vanish at the boundaries.
- **(c) Bonus NAND.** NAND is **functionally complete** — any boolean function can be built from NANDs — so showing a ReLU NAND implies ReLU nets can represent any boolean function.

General technique: write the target as $f = \sum_j a_j\,\mathrm{ReLU}(w_j^\top x + b_j)$ and hand-pick $w_j, b_j$ so each ReLU "turns on" in exactly the half-plane you need. Verify by checking all four sign-quadrants of $(x_1, x_2)$.

---

## Section B — Backpropagation (3 pt)

### Problem 8 (3pt) — Vectorized gradient of $L = \tfrac12\|v\|_2^2$

**Requires:** the multivariable chain rule, Kronecker-delta index gymnastics, and recognizing outer-product / diagonal-matrix structure.

The forward system:

$y = Wx,\quad u = \mathrm{ReLU}(y),\quad v = u + Wu,\quad L = \tfrac12\|v\|_2^2.$

**(a)** Start from $L = \tfrac12\sum_m v_m^2$ and differentiate w.r.t. a single entry $W_{ij}$. The scalar chain rule gives

$\frac{\partial L}{\partial W_{ij}} = \sum_{m=1}^{d} v_m\,\frac{\partial v_m}{\partial W_{ij}}.$

This is just $\partial(\tfrac12 v_m^2)/\partial v_m = v_m$ times the inner derivative — make sure you can write that cleanly.

**(b)** Now propagate $\partial v_m/\partial W_{ij}$ using the three supplied relations. The mechanics to master:

- The **Kronecker delta** $\delta_{ik}$ collapses sums: $\sum_k \delta_{ik} a_k = a_i$. Use it to kill index sums.
- The **Heaviside** $\Theta(y_k)$ is the ReLU derivative; it becomes the diagonal **gate matrix** $\mathrm{diag}(\Theta(y))$ that zeroes out gradient components where neurons are off.
- The term $\delta_{im} u_j$ is what produces an **outer product** $v \otimes u$ when you reassemble over indices $i,j$.
- The term with $W_{ml}$ and $\partial u_l/\partial W_{ij}$ contributes the $(I + W^\top)$ factor after you fold the chain $v \to u \to y \to W$.

Target identity to reach:

$\frac{\partial L}{\partial W} = v \otimes u + \mathrm{diag}(\Theta(y))\,(I + W^\top)\,v \otimes x.$

Strategy: compute $\partial v_m/\partial W_{ij}$ fully in index form, multiply by $v_m$, sum over $m$, then **recognize** which index pattern is an outer product ($\otimes$ over the free indices $i,j$) and which is a matrix-times-vector. The payoff line in the problem — that this is GPU-friendly — is worth remembering: outer products and diagonal scalings are single fused matmuls.

PyTorch sanity check you can run (don't submit as the derivation, but it confirms your algebra):

```python
import torch
d = 4
W = torch.randn(d, d, requires_grad=True)
x = torch.randn(d)
y = W @ x
u = torch.relu(y)
v = u + W @ u
L = 0.5 * (v @ v)
L.backward()
autograd_grad = W.grad

# your closed form:
Theta = (y >= 0).float()
manual = torch.outer(v, u) + torch.outer(torch.diag(Theta) @ (torch.eye(d) + W.T) @ v, x)
print(torch.allclose(autograd_grad, manual, atol=1e-5))
```

---

## Section C — CIFAR-10 & PyTorch (12 pt)

These are completed in the provided Colab; below are the techniques each part exercises so you can write the code yourself.

### Problem 10 (4pt) — Manual forward/backward for Linear, ReLU, CrossEntropyLoss

**Requires:** deriving each layer's **local Jacobian** and chaining upstream gradients (the essence of backprop modules, ≤5 lines each).

For a linear layer $\text{out} = Wx + b$, given upstream $\frac{\partial L}{\partial \text{out}}$ (call it $g$):

$\frac{\partial L}{\partial x} = W^\top g,\qquad \frac{\partial L}{\partial W} = g\,x^\top,\qquad \frac{\partial L}{\partial b} = g.$

(Batch versions sum over the batch dimension — watch your transposes.)

```python
class Linear:
    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b
    def backward(self, grad_out):           # grad_out: dL/dout
        self.dW = grad_out.T @ self.x       # dL/dW
        self.db = grad_out.sum(0)           # dL/db
        return grad_out @ self.W            # dL/dx
```

For **ReLU**, the local derivative is the gate $\mathbb{1}[x>0]$:

$\frac{\partial L}{\partial x} = g \odot \mathbb{1}[x > 0].$

For **cross-entropy with softmax**, the celebrated simplification is

$\frac{\partial L}{\partial z} = \hat{p} - y_{\text{onehot}},$

where $\hat p = \mathrm{softmax}(z)$. Derive why the softmax Jacobian and the log-loss combine so cleanly — that's the conceptual point of this part. Remember to divide by batch size if your loss averages.

### Problem 11 (2pt) — Training loop

**Requires:** the canonical train/eval skeleton. Per batch: zero grads, forward, compute loss, backward, optimizer step. In eval, disable grad tracking and don't update.

```python
def train_epoch(model, loader, opt, loss_fn):
    model.train()
    for xb, yb in loader:
        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item(); total += yb.size(0)
    return correct / total
```

### Problem 12 (2pt) — Training curves

Plot train/val loss and accuracy vs. epoch. Look for the **gap** between train and val (generalization), plateaus, and whether loss is still descending at epoch 30.

### Problem 13 (1pt) — Why not 100% train accuracy?

**Requires:** reconciling the **universal approximation theorem** (existence of a perfect fit) with **optimization reality**. UAT guarantees a network *exists*, not that SGD *finds* it: limited capacity for the chosen width, optimization not converged, learning-rate/regularization effects. Also reflect on whether perfect train accuracy is even desirable (overfitting vs. generalization).

### Problem 14 (1pt) — Data augmentation (random crop)

**Requires:** building two dataset pipelines where **only train** gets augmentation; visualize before/after.

```python
from torchvision import transforms
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])
val_tf = transforms.Compose([transforms.ToTensor()])
```

Conceptually, augmentation enlarges the effective training distribution and acts as a regularizer — it should narrow the train/val gap.

### Problem 15 (2pt) — Training curves with augmentation

Re-train and compare to Problem 12: expect slower/lower train accuracy but better validation generalization. Comment on the changed gap.

---

## ✅ How to attack the set

1. Nail the **linear-region** mental model (Prob 4) — it powers 3, 4, 5, 7.
2. Do the **index-then-recognize** workflow for Prob 8; verify with the autograd snippet.
3. For the Colab, derive each backward by hand *before* coding; the math in Prob 8 and 10 is the same skill.

*Leave any questions for me as a comment or a line on this page and I'll address them tomorrow.*