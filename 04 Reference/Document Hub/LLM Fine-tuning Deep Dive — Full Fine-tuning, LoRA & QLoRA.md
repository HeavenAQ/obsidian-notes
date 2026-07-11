---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-02T16:58:00
Status: Done
Last updated time: 2026-07-08T11:37:00
Last edited by: Heaven Chen
Category:
  - LLM
  - Finetuning
  - SFT
---
> Companion deep dive to [🔁 Transformers in Practice (](https://app.notion.com/p/391e445b30a4817dbfb5ef315a1f06ff)[DeepLearning.AI](http://deeplearning.ai/)[, Sharon Zhou) — Deep Dive](https://app.notion.com/p/391e445b30a4817dbfb5ef315a1f06ff). Covers *why* fine-tuning a full LLM is expensive, the math of LoRA and QLoRA, and precisely *where in the transformer architecture* the adapted parameters live.

# 1 — Why full fine-tuning is a memory wall

Fine-tuning means continuing gradient descent on a pretrained model's weights $\theta$ using a new (usually much smaller) dataset. The naive approach — **full fine-tuning** — updates every parameter, and that is what makes it expensive: it isn't the forward pass that kills you, it's everything Adam has to keep alive per parameter to do the backward pass.

For a model with $P$ parameters trained in mixed precision with the Adam optimizer, GPU memory has to hold, per parameter:

$\underbrace{2\text{ bytes}}_{\text{bf16 weights}} + \underbrace{2\text{ bytes}}_{\text{bf16 gradients}} + \underbrace{4\text{ bytes}}_{\text{fp32 master weight}} + \underbrace{4\text{ bytes}}_{\text{Adam }m} + \underbrace{4\text{ bytes}}_{\text{Adam }v} \approx 16\text{ bytes/parameter}$

*Why Adam needs two extra full-precision copies:* Adam maintains a running mean $m$ (first moment, "momentum") and running variance $v$ (second moment) of the gradient for every single parameter, updated as $m \leftarrow \beta_1 m + (1-\beta_1)g$, $v \leftarrow \beta_2 v + (1-\beta_2)g^2$, and both need fp32 precision to accumulate stably over thousands of steps without underflowing. A fp32 "master weight" copy is kept alongside the bf16 working copy because bf16's ~3 decimal digits of precision are too coarse for the tiny per-step updates ($\eta \cdot \hat m/(\sqrt{\hat v}+\epsilon)$) to survive repeated rounding — without it, small updates would just vanish into bf16's rounding error.

For a 7B-parameter model, that's **~112 GB** just for optimizer state — before activations, before a single token of data. This is why full fine-tuning of anything beyond ~7B parameters requires multi-GPU sharding (ZeRO / FSDP) even though *inference* on the same model fits comfortably on one GPU. The problem is structural: **the memory cost of fine-tuning scales with trainable parameters, not with task difficulty.** Adapting a 70B model to a narrow task (say, a customer-support tone) needs the same 16 bytes/parameter as adapting it to something that actually requires all 70B parameters' worth of new knowledge.

# 2 — The core idea: parameter-efficient fine-tuning (PEFT)

PEFT methods sidestep the memory wall by **freezing almost all of **$\theta$** and training a tiny number of new parameters** that are added to (or inserted into) the frozen network. If you only train $P_{\text{trainable}} \ll P$ parameters, Adam's 12 extra bytes/parameter only apply to that tiny fraction — the frozen billions of parameters need no gradient, no optimizer state, not even a gradient computation past the point where they feed into a trainable adapter.

A rough taxonomy of PEFT approaches, by *where* they add trainable capacity:

- **Prompt/prefix tuning**: prepend trainable "virtual token" embeddings to the input (or to every layer's keys/values); the transformer itself is untouched, only these injected vectors are learned. Cheap, but limited expressiveness — you're only ever able to steer the existing computation, not change it.
- **Adapter layers** (Houlsby et al.): insert small trainable bottleneck MLPs *between* existing frozen layers. Effective, but adds inference latency because they're serial with the rest of the forward pass — every token now passes through extra layers it didn't before.
- **LoRA / QLoRA** (this note's focus): instead of adding new layers, decompose the *update* to an existing weight matrix into a low-rank product and train only that. Critically, it can be **merged back into the original weight matrix after training**, so it adds zero inference latency — the biggest practical reason LoRA displaced adapter layers as the default PEFT method.

# 3 — LoRA: Low-Rank Adaptation

![[99 Assets/Media/x1.png|LoRA reparametrization: the pretrained weight matrix W is frozen (blue), and only a low-rank pair of matrices A (random Gaussian init) and B (zero init) are trained, injected in parallel to the frozen path.]]

*Figure: LoRA (Hu et al., 2021). The pretrained weight *$W$* is frozen; a parallel low-rank path *$BA$* is trained and added to *$W$*'s output.*

## 3.1 The math

For any pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ (e.g. one of the attention projection matrices), full fine-tuning would learn an *unconstrained* update $\Delta W \in \mathbb{R}^{d \times k}$, i.e. $d \times k$ new trainable numbers — for a $4096 \times 4096$ matrix, that's 16.7M parameters *per matrix*. LoRA instead **constrains **$\Delta W$** to be low rank**, factoring it as the product of two skinny matrices:

$W_0 + \Delta W = W_0 + BA, \qquad B \in \mathbb{R}^{d\times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d,k)$

The forward pass becomes $h = W_0 x + \Delta W x = W_0 x + BAx$, and in practice the update is scaled by a constant $\alpha/r$:

$h = W_0 x + \frac{\alpha}{r}BAx$

where $\alpha$ is a hyperparameter (roughly playing the role of a learning-rate multiplier for the adapter, decoupled from rank $r$ so you can retune $\alpha$ without rescaling gradients when you change $r$). **Only **$A$** and **$B$** receive gradients**; $W_0$ is frozen (`requires_grad=False`).

*Why constraining the update to be low-rank doesn't cripple it:* this is the paper's central empirical/theoretical bet, built on the observation (Aghajanyan et al., 2020, "intrinsic dimensionality") that the weight *updates* needed to adapt a large pretrained model to a downstream task live in a surprisingly low-dimensional subspace — i.e. if you took the true $\Delta W$ from full fine-tuning and computed its SVD, the singular values decay very fast, so a rank-4 to rank-64 approximation captures most of the useful update. This is plausible precisely *because* the base model is heavily pretrained: it already has almost all the representational machinery needed, and adaptation is mostly a matter of **re-weighting / rotating existing directions** in its representation space, not building new ones from scratch — a task well suited to a low-rank correction.

*Why initialize* $A$ *random and* $B$ *at zero:* at the very first training step, $BA = 0$ exactly (since $B=0$), so the adapted model is numerically **identical** to the pretrained model — training starts from a known-good point and the adapter can only move away from it gradually, rather than starting from a random, potentially catastrophic perturbation to a network that already works.

*Parameter savings:* trainable parameters per adapted matrix go from $d\times k$ to $r(d+k)$. For $d=k=4096, r=8$: $4096^2 = 16.8$M vs. $8 \times 8192 = 65.5$K — a **256× reduction**, for that matrix alone. Across a full 7B model, LoRA typically trains well under 1% of total parameters.

## 3.2 Where LoRA attaches in the transformer

Recall the transformer block from the companion Transformers-in-Practice note: each block contains the attention sub-layer's four linear projections $W_Q, W_K, W_V, W_O$, and the MLP sub-layer's two (or three, for gated MLPs) linear projections. LoRA can be attached to **any** of these linear layers — it is a generic reparametrization of *any* matrix multiply, not something specific to attention.

The original LoRA paper's ablation on GPT-3 found that, for a **fixed total parameter budget**, applying LoRA to just $W_Q$ and $W_V$ captured most of the achievable quality, and spreading the same budget across all four attention matrices ($W_Q,W_K,W_V,W_O$) performed comparably — while applying it to only one matrix (e.g. just $W_Q$) at higher rank underperformed both. In practice, most modern LoRA fine-tunes (and the QLoRA paper specifically) apply adapters to **all linear layers**, including the MLP's up/down/gate projections, because empirically matching full fine-tuning quality on harder instruction-tuning tasks requires adapting the MLP too — attention-only LoRA tends to underfit relative to full fine-tuning on complex tasks.

The embedding table and the final unembedding/LM-head matrix ($W_U$ from the companion note's autoregressive-loop equation) are typically left frozen, since they're enormous ($|V| \times d$, tens of millions of parameters) and rarely need task-specific adaptation — though they *can* be LoRA-adapted too if the fine-tuning task introduces new vocabulary or requires shifting output distribution substantially (e.g. adapting to a new language).

```python
import torch, torch.nn as nn, math

class LoRALinear(nn.Module):
    """
    Wraps a frozen pretrained nn.Linear with a trainable low-rank adapter.
    h = W0 x + (alpha/r) * B @ (A @ x)
    """
    def __init__(self, base_linear: nn.Linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False               # freeze W0
        d_out, d_in = base_linear.weight.shape
        self.A = nn.Parameter(torch.randn(r, d_in) * (1 / math.sqrt(d_in)))  # random init
        self.B = nn.Parameter(torch.zeros(d_out, r))                          # zero init -> ΔW=0 at start
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        base_out = self.base(x)                         # frozen path, no grad into W0
        lora_out = (self.dropout(x) @ self.A.T) @ self.B.T
        return base_out + self.scaling * lora_out

    @torch.no_grad()
    def merge(self):
        """Fold the adapter into W0 for zero-latency inference: W' = W0 + (alpha/r) B A."""
        self.base.weight += self.scaling * (self.B @ self.A)

def apply_lora_to_attention(model, r=8, alpha=16):
    for block in model.transformer_blocks:
        block.attn.Wq = LoRALinear(block.attn.Wq, r=r, alpha=alpha)
        block.attn.Wv = LoRALinear(block.attn.Wv, r=r, alpha=alpha)
    return model
```

# 4 — QLoRA: quantized backbone + LoRA adapters

![[99 Assets/Media/x1 1.png|Memory footprint comparison: full 16-bit fine-tuning, 16-bit LoRA (frozen weights in 16-bit, adapters trained), and QLoRA (frozen weights compressed to 4-bit NF4, adapters trained in higher precision), showing QLoRA's dramatically smaller GPU memory requirement.]]

*Figure: QLoRA (Dettmers et al., 2023) — the frozen base model is stored in 4-bit, dequantized on the fly for compute, while only the LoRA adapters are trained and stored in higher precision.*

LoRA already avoids storing gradients/optimizer-state for the frozen weights, but it still needs the frozen weights *resident in GPU memory* in fp16/bf16 (2 bytes/parameter) for the forward and backward pass through them. For a 65B model, that's still ~130 GB just to hold $W_0$. **QLoRA's contribution is to shrink that remaining cost** by quantizing the frozen backbone to 4 bits, while keeping the trainable LoRA adapters in full bf16 precision — so gradient quality for the part that's actually learning is untouched.

## 4.1 NF4: 4-bit NormalFloat

Standard uniform integer quantization (Module 3 of the companion note: $w \approx s\cdot q$, uniformly spaced levels) is wasteful for neural network weights, because pretrained weights are empirically very well approximated by $w \sim \mathcal{N}(0, \sigma^2)$ — a bell curve, not a uniform distribution. Uniform quantization spends equal precision on rare, far-from-zero weights and common, near-zero weights, when it's the near-zero weights (the vast majority) that most need fine discrimination.

**NF4** instead uses **quantile quantization**: choose the 16 discrete levels (for 4 bits, $2^4=16$ levels) to be the quantiles of a standard normal distribution, so that *each bin contains an equal expected number of weight values* rather than an equal span of the number line. Concretely, for a target of $2^k$ levels, compute quantiles $q_i$ of $\mathcal{N}(0,1)$ such that

$q_i = \tfrac{1}{2}\Big(Q_X\big(\tfrac{i}{2^k+1}\big) + Q_X\big(\tfrac{i+1}{2^k+1}\big)\Big)$

where $Q_X$ is the standard normal quantile function (inverse CDF). This is information-theoretically optimal for normally distributed inputs — no other 4-bit code minimizes expected quantization error better, for weights that are actually normal. Because weights across a model don't all share the same scale, each block of (e.g. 64) weights is first normalized by its own **absmax**: $w' = w / \max|w|$, mapping the block into $[-1,1]$, before snapping each value to its nearest NF4 quantile.

## 4.2 Double quantization

Each block's absmax scaling constant is itself a number that has to be stored (one 32-bit float per block of 64 weights — an overhead of $32/64 = 0.5$ bits per parameter, which sounds small but adds up at 65B scale). **Double quantization** quantizes these per-block scale constants *themselves*, using 8-bit quantization with a second, larger block size (256 constants per second-level block). This shrinks the overhead from 0.5 bits/parameter to roughly 0.127 bits/parameter — the QLoRA paper reports an average savings of about **0.37 bits per parameter** across a 65B model this way, which is meaningful when the entire quantized weight is only 4 bits to begin with.

## 4.3 Paged optimizers

Even with a quantized backbone, gradient checkpointing (recomputing activations during backward instead of storing them) causes transient memory *spikes* that can exceed available GPU memory. QLoRA borrows NVIDIA's **unified memory** feature to set up **paged optimizer states**: when a memory spike would otherwise OOM, optimizer-state pages are automatically evicted to CPU RAM and paged back in on demand — the same idea as OS-level virtual memory paging, applied to the optimizer's Adam moment buffers. This doesn't reduce steady-state memory use, it just prevents rare spikes from crashing a long training run.

## 4.4 Forward/backward mechanics: where gradients actually flow

$h = \mathrm{dequant}(W_0^{\text{NF4}}) \, x \;+\; \frac{\alpha}{r} B A x$

At every forward pass, the frozen weight is **dequantized on-the-fly** from 4-bit NF4 back to bf16 for the actual matrix multiply (compute happens in bf16 for numerical quality; only *storage* is 4-bit). Backpropagation computes gradients with respect to $A$ and $B$ only — the gradient never needs to flow *into* $W_0^{\text{NF4}}$ itself (it's `requires_grad=False`), it only needs to flow *through* the dequantized $W_0$ on its way to $A,B$ via the chain rule, which is a cheap, gradient-free tensor op. This is precisely why QLoRA can train a 65B model on a single 48GB GPU: the multi-hundred-GB cost of storing fp32/bf16 gradients and Adam states scales only with the ~tens-of-millions of adapter parameters, while the ~65B frozen parameters cost only their compressed 4-bit footprint (~33 GB) and never need gradients or optimizer state of their own.

```python
import torch

def nf4_quantile_levels():
    """
    Simplified NF4: 16 asymmetric quantiles of a standard normal, giving
    each of the 16 bins equal expected probability mass under N(0,1).
    (Real NF4 estimates positive/negative halves separately to keep 0 exactly
    representable; this is the illustrative single-sided version.)
    """
    from scipy.stats import norm
    k = 4
    n_levels = 2 ** k
    quantile_points = [(i / (n_levels + 1) + (i + 1) / (n_levels + 1)) / 2
                        for i in range(n_levels)]
    levels = torch.tensor([norm.ppf(p) for p in quantile_points])
    return levels / levels.abs().max()          # normalize to [-1, 1]

def nf4_quantize_block(w_block, levels):
    """w_block: (block_size,) fp32 weights. Returns (4-bit codes, absmax scale)."""
    absmax = w_block.abs().max()
    w_norm = w_block / absmax                     # -> [-1, 1]
    codes = torch.cdist(w_norm.unsqueeze(-1), levels.unsqueeze(-1)).argmin(-1)
    return codes, absmax                           # codes: nearest-quantile index per weight

def nf4_dequantize_block(codes, absmax, levels):
    return levels[codes] * absmax

class QLoRALinear(torch.nn.Module):
    """Frozen 4-bit base (dequantized on the fly) + trainable bf16 LoRA adapter."""
    def __init__(self, codes, absmax, levels, d_out, d_in, r=16, alpha=32):
        super().__init__()
        self.register_buffer('codes', codes)          # 4-bit indices, frozen, no grad
        self.register_buffer('absmax', absmax)
        self.register_buffer('levels', levels)
        self.A = torch.nn.Parameter(torch.randn(r, d_in) / d_in**0.5)
        self.B = torch.nn.Parameter(torch.zeros(d_out, r))
        self.scaling = alpha / r

    def forward(self, x):
        W0 = nf4_dequantize_block(self.codes, self.absmax, self.levels).view(-1, x.shape[-1])
        base_out = x @ W0.T                            # compute in bf16, W0 has no grad
        lora_out = (x @ self.A.T) @ self.B.T
        return base_out + self.scaling * lora_out
```

# 5 — Memory budget: putting numbers on it

Approximate GPU memory to fine-tune a **7B** parameter model (rough, illustrative order-of-magnitude figures — actual numbers depend on batch size, sequence length, and activation memory, which is omitted here since it's comparable across all three methods):

| Method | Frozen backbone storage | Trainable params | Optimizer state (Adam, ~12 B/trainable-param) | Approx. total |
| --- | --- | --- | --- | --- |
| Full fine-tuning (bf16 + fp32 master + Adam) | — (all trainable) | 7B | ~84 GB | **~112 GB** |
| LoRA (bf16 backbone, frozen) | ~14 GB (bf16, 2B/param) | ~20–40M (r=8, attn+MLP) | ~0.3–0.5 GB | **~15 GB** |
| QLoRA (NF4 backbone, frozen) | ~3.5–4 GB (4-bit + dequant overhead) | ~20–40M | ~0.3–0.5 GB | **~5 GB** |

This is the entire practical argument for QLoRA in one table: the frozen backbone dominates LoRA's memory cost once the model gets large, and compressing it from 16-bit to 4-bit is a ~4× reduction on the single largest remaining line item — which is precisely what took 65B-parameter fine-tuning from "needs a multi-GPU cluster" to "fits on one 48GB GPU."

# 6 — Self-check questions

1. Why does Adam's memory overhead scale with *trainable* parameters specifically, and why does that make PEFT methods so much cheaper than full fine-tuning even though the *forward pass* touches every parameter either way?
2. Derive the parameter count for a rank-$r$ LoRA adapter on a $d\times k$ matrix, and compute the reduction factor for $d=k=4096, r=16$.
3. Why is $B$ initialized to zero (not $A$), and what would go wrong at the start of training if both were randomly initialized?
4. Explain why NF4's quantile-based levels are a better fit for weight quantization than uniform INT4 levels, referencing the actual distribution of neural network weights.
5. In QLoRA, gradients never update $W_0^{\text{NF4}}$, yet the loss gradient must still "pass through" it during backpropagation to reach $A$ and $B$. Explain how this is possible without $W_0$ needing its own gradient buffer.
6. Why does merging LoRA's $BA$ into $W_0$ after training eliminate the adapter's inference-time latency cost, in a way that serial adapter layers (Houlsby-style) cannot achieve?