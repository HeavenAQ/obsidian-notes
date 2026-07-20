---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-20T18:15:00
Status: Done
Last updated time: 2026-07-20T18:15:00
Last edited by: Heaven Chen
Category:
  - ML
  - Deep Learning
  - Theory
  - Sequence Modeling
  - State Space Models
---

# Mamba Architecture — Selective State Spaces, Math & PyTorch Deep Dive

> [!abstract] The short, easy introduction
> **Mamba is a neural sequence model that replaces a Transformer's all-to-all attention with a small, continuously updated memory.** It reads a sequence from left to right. At each token, it decides what to **write**, what to **forget**, and what to **read** from that memory. These decisions depend on the current token, which is why the mechanism is called a **selective state space model**.
>
> A Transformer repeatedly looks back at all earlier tokens. Mamba instead carries forward a compressed state, rather like an RNN—but with a carefully designed state update and a GPU-friendly parallel training algorithm. Consequently, processing a length-$L$ sequence requires work that grows roughly **linearly** with $L$, and autoregressive decoding uses a **fixed-size recurrent state** instead of a KV cache that grows with the context.
>
> Mamba is therefore *better than a standard Transformer in specific ways*: long-sequence scaling, constant-memory decoding, and streaming. It is **not universally better**: attention remains excellent at exact content lookup, allows every token to directly interact with every other token, and has a more mature ecosystem. In practice, pure Mamba, pure attention, and hybrid models occupy different quality–efficiency trade-offs.

## 1. The one mental model to keep

Imagine reading a long meeting transcript while maintaining a small whiteboard:

- **Transformer:** whenever a new sentence arrives, reopen the entire transcript and decide which earlier sentences to consult.
- **Mamba:** update the whiteboard after every sentence, erasing unimportant details and preserving useful facts. The next prediction consults the whiteboard rather than reopening every page.

For token $x_t$, Mamba performs the conceptual update

$$
\text{memory}_t
= \text{keep}_t \odot \text{memory}_{t-1}
+ \text{write}_t(x_t),
\qquad
y_t = \text{read}_t(\text{memory}_t).
$$

The crucial word is **selective**:

- punctuation or filler can cause the model to mostly keep the old state;
- a boundary can cause it to reset or forget;
- an important entity can be strongly written;
- the current token can determine which part of memory is read out.

This is not literal symbolic storage. The “whiteboard” is a learned tensor, and all operations are differentiable.

> [!tip] How to read the math and code below
> Do not begin by memorizing continuous-time control theory. Read in this order:
>
> 1. understand the scalar recurrence $h_t=a_t h_{t-1}+b_t x_t$;
> 2. see how $a_t$, $b_t$, and the readout depend on the token;
> 3. add the state dimension $N$ and model channels $D$;
> 4. map those tensors to the slow PyTorch loop;
> 5. only then study discretization and the fused selective-scan kernel.
>
> The production kernel computes the same basic recurrence as the readable loop; it is an optimization, not a different model.

---

## 2. Mamba versus a Transformer

### 2.1 What self-attention does

For a causal Transformer,

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
$$

$$
Y=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M_{\text{causal}}\right)V.
$$

Each query can directly score every earlier key. This makes content-based retrieval natural, but the $L\times L$ score pattern gives standard attention $O(L^2)$ sequence work. FlashAttention reduces materialized-memory traffic, but does not change the dense attention computation's quadratic dependence on sequence length.

During autoregressive generation, a Transformer stores keys and values for previous tokens. The **KV cache grows with context length**, and each new query attends over that growing history.

### 2.2 What Mamba does instead

At inference, Mamba applies a recurrence:

$$
h_t=\bar A_t\odot h_{t-1}+\bar B_t\odot x_t,
\qquad
y_t=C_t h_t+D\odot x_t.
$$

There is no $L\times L$ attention matrix and no growing KV cache. The entire relevant past must be represented in $h_t$.

| Question | Transformer self-attention | Mamba selective SSM |
|---|---|---|
| How is the past accessed? | Directly scores earlier tokens | Reads a compressed recurrent state |
| Training sequence work | $O(L^2d)$ for dense attention | approximately $O(LDN)$ |
| Autoregressive cache | grows as $O(Ld)$ per layer | fixed $O(DN)$ state per layer |
| Per-token decode dependence on $L$ | grows with cached context | independent of previous length |
| Content selection | explicit query–key similarity | input-dependent write, forget, and read parameters |
| Pairwise token interaction | direct | indirect through state |
| Natural direction | causal or bidirectional with masking | causal scan; bidirectionality needs another scan/design |
| Position/order | added through positional machinery | inherent in the recurrence; variants may add positional mechanisms |
| Main strength | flexible retrieval and token interaction | efficient streaming and long sequences |

Here $D$ is the inner channel width and $N$ is the SSM state size. $N$ is normally far smaller than $L$.

### 2.3 In what sense is Mamba “better”? 

Mamba's strongest advantages are architectural:

1. **Linear sequence scaling.** Doubling $L$ roughly doubles selective-scan work rather than quadrupling an attention matrix.
2. **Constant-size decoding memory.** Each layer carries its convolution state and SSM state; neither grows with generated length.
3. **Streaming is native.** A new audio sample, video feature, sensor reading, or token can update the state without replaying the prefix.
4. **Long-context throughput.** The original Mamba paper reported substantially higher generation throughput and favorable scaling on very long sequences compared with same-size Transformer baselines.
5. **Order is built in.** The recurrence itself distinguishes earlier from later positions.

But “better” must not be read as “dominates Transformers”:

- a fixed-size state is an **information bottleneck**;
- exact associative recall and arbitrary copying can be easier with attention;
- Mamba requires specialized scan kernels to realize its theoretical efficiency;
- attention is more parallel and familiar at decode-time batch sizes/hardware regimes where Mamba kernels may not be optimal;
- many strong systems combine recurrent/SSM layers with occasional attention layers rather than choosing only one.

> [!important] Correct conclusion
> Mamba is a compelling **efficiency-first alternative** to attention, especially for long and streaming sequences. Whether it is better for a particular model depends on quality targets, sequence length, hardware, and the need for precise retrieval.

---

## 3. Start with the smallest possible recurrence

Ignore matrices for a moment:

$$
h_t=a_t h_{t-1}+b_t x_t,
\qquad
y_t=c_t h_t+d x_t.
$$

- $h_t$: memory after reading token $t$;
- $a_t$: how much old memory survives;
- $b_t x_t$: what the current token writes;
- $c_t$: how the memory is read for the current output;
- $d x_t$: a direct skip path.

If $a_t=0.99$, memory decays slowly. If $a_t=0.05$, the model almost resets. If $b_t$ is large for a name and small for filler, the name is preferentially written. If $c_t$ changes with the token, different tokens read different summaries.

### 3.1 A numerical example

Let the stable continuous decay be $A=-1$, with $B_t=C_t=1$. Use the practical Mamba-style update

$$
a_t=e^{\Delta_t A}=e^{-\Delta_t},
\qquad
h_t=a_t h_{t-1}+\Delta_t x_t.
$$

Starting at $h_0=0$:

| $t$ | $x_t$ | $\Delta_t$ | $a_t=e^{-\Delta_t}$ | new state $h_t$ |
|---:|---:|---:|---:|---:|
| 1 | 2 | 0.1 | 0.905 | $0.905(0)+0.1(2)=0.200$ |
| 2 | 0 | 2.0 | 0.135 | $0.135(0.200)+2(0)=0.027$ |
| 3 | 5 | 0.1 | 0.905 | $0.905(0.027)+0.1(5)=0.524$ |

The large $\Delta_2$ acts like a boundary: it rapidly forgets the old state. Mamba learns $\Delta_t$ from the token instead of fixing it.

---

## 4. Where state space models come from

A continuous-time linear state space model is

$$
\frac{d h(t)}{dt}=Ah(t)+Bx(t),
\qquad
y(t)=Ch(t)+Dx(t).
$$

This is a standard dynamical system:

- $A$ controls autonomous state evolution and decay;
- $B$ maps input into the state;
- $C$ maps the state to output;
- $D$ is a direct input-to-output skip.

For a sampling interval $\Delta$, zero-order-hold discretization gives

$$
\bar A=e^{\Delta A},
$$

$$
\bar B=\left(\int_0^\Delta e^{\tau A}d\tau\right)B
=A^{-1}\left(e^{\Delta A}-I\right)B
$$

when $A$ is invertible. The discrete recurrence is then

$$
h_t=\bar A h_{t-1}+\bar Bx_t,
\qquad
y_t=Ch_t+Dx_t.
$$

The Mamba selective-scan reference implementation uses the practical elementwise form

$$
\bar A_t=\exp(\Delta_t A),
\qquad
\bar B_t x_t=\Delta_t B_t x_t.
$$

Mamba-3 later calls this **exponential–Euler** discretization and provides a formal time-varying interpretation, then introduces a more expressive exponential–trapezoidal alternative.

### 4.1 Why force $A$ to be negative?

Mamba parameterizes the real state matrix approximately as

$$
A=-\exp(A_{\log}).
$$

Therefore $A<0$, $\Delta_t>0$, and

$$
0<\exp(\Delta_t A)<1.
$$

The recurrence decays instead of exploding. The implementation learns `A_log`, takes `-exp(A_log)`, and obtains positive $\Delta_t$ with `softplus`.

---

## 5. From an ordinary SSM to Mamba's selective SSM

### 5.1 The limitation of a fixed linear time-invariant SSM

If $A,B,C,$ and $\Delta$ are constant for all positions, the model is **linear time-invariant** (LTI). Unrolling gives

$$
y_t=\sum_{i=1}^{t}C\bar A^{t-i}\bar Bx_i+Dx_t.
$$

The kernel

$$
K_k=C\bar A^k\bar B
$$

depends only on distance $k=t-i$. Therefore the recurrence can also be evaluated as a convolution. This recurrent–convolutional duality powered earlier structured SSMs such as S4.

However, a fixed kernel treats the same token similarly regardless of its content or context. That is a serious weakness for discrete information-dense data such as language.

### 5.2 Selection: make parameters depend on the input

Mamba computes

$$
\Delta_t=\operatorname{softplus}(W_\Delta x_t+b_\Delta),
$$

$$
B_t=W_Bx_t,
\qquad
C_t=W_Cx_t.
$$

The learned $A$ remains shared, but the discrete transition

$$
\bar A_t=\exp(\Delta_t A)
$$

is input-dependent through $\Delta_t$.

Interpret the three selective quantities as follows:

| Quantity | Mechanistic role | Intuition |
|---|---|---|
| $\Delta_t$ | changes decay and write strength | **when to forget/update** |
| $B_t$ | maps current input into state | **what to write** |
| $C_t$ | maps current state to output | **what to read** |

Large $\Delta_t$ makes $e^{\Delta_tA}$ smaller because $A<0$: previous state fades faster while the current write $\Delta_t B_tx_t$ becomes stronger. Small $\Delta_t$ preserves the state and makes a smaller update.

> [!warning] “Selective” does not mean selecting a subset of tokens
> There is no hard top-$k$ token picker. Selection means that the recurrence's parameters are functions of the current input, enabling soft, learned control over propagation, forgetting, writing, and reading.

### 5.3 The full unrolled selective weighting

For $i\le t$,

$$
y_t
=\sum_{i=1}^{t}
C_t
\left(\prod_{k=i+1}^{t}\bar A_k\right)
\bar B_i x_i
+Dx_t.
$$

Compare this with attention:

- attention learns pairwise weights from query $t$ and key $i$;
- Mamba's influence weight is factored through all intervening state transitions.

Mamba therefore has an attention-like, content-dependent mixing matrix, but it is strongly structured by a recurrence. That structure enables linear-time evaluation while restricting arbitrary pairwise interactions.

---

## 6. Tensor shapes: the bridge from equations to code

Use:

- batch size $B_s$;
- sequence length $L$;
- inner model width $D$;
- SSM state size $N$.

Typical conceptual shapes are:

| Tensor | Shape | Meaning |
|---|---|---|
| input $u$ | `[batch, length, D]` | one $D$-channel vector per token |
| $A$ | `[D, N]` | $N$ decay modes for each channel |
| $\Delta$ | `[batch, length, D]` | token-dependent step size per channel |
| $B_t$ | `[batch, length, N]` | token-dependent state write vector |
| $C_t$ | `[batch, length, N]` | token-dependent state read vector |
| state $h_t$ | `[batch, D, N]` | recurrent memory |
| output $y$ | `[batch, length, D]` | same sequence interface as other mixers |

For each token, broadcasting produces

$$
\bar A_t:\ [B_s,D,N],
\qquad
\Delta_tB_tu_t:\ [B_s,D,N].
$$

The $N$ dimension is summed when $C_t$ reads the state, returning `[batch, D]`.

The official reference kernel stores sequence tensors as `[batch, D, length]`, but the recurrence is identical.

---

## 7. Selective scan in readable PyTorch

The following is intentionally slow. Its purpose is to expose exactly what the fused CUDA/Triton implementation computes.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSMReference(nn.Module):
    """Educational Mamba-1-style selective SSM, not a fast implementation."""

    def __init__(self, d_inner: int, d_state: int = 16):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        # One projection creates token-dependent Δ, B, and C.
        # The official implementation uses a low-rank projection for Δ.
        self.select_proj = nn.Linear(
            d_inner, d_inner + 2 * d_state, bias=False
        )

        # Stable continuous-time decay: A is always negative.
        init = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(init.log().repeat(d_inner, 1))  # [D, N]

        self.delta_bias = nn.Parameter(torch.zeros(d_inner))
        self.D = nn.Parameter(torch.ones(d_inner))  # direct skip

    def forward(self, u):
        # u: [batch, length, D]
        batch, length, D = u.shape
        N = self.d_state

        params = self.select_proj(u)
        raw_delta, B, C = torch.split(params, [D, N, N], dim=-1)

        delta = F.softplus(raw_delta + self.delta_bias)  # [B, L, D] > 0
        A = -torch.exp(self.A_log.float())                # [D, N] < 0

        state = u.new_zeros(batch, D, N)
        outputs = []

        for t in range(length):
            u_t = u[:, t]              # [B, D]
            dt = delta[:, t]           # [B, D]
            B_t = B[:, t]              # [B, N]
            C_t = C[:, t]              # [B, N]

            # Discretized transition and token-dependent write.
            dA = torch.exp(dt[..., None] * A[None, :, :])
            dB_u = (
                dt[..., None]
                * B_t[:, None, :]
                * u_t[..., None]
            )

            state = dA * state + dB_u                  # [B, D, N]
            y_t = (state * C_t[:, None, :]).sum(-1)   # [B, D]
            y_t = y_t + self.D * u_t
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # [B, L, D]
```

### 7.1 Read the code as five sentences

```python
delta, B, C = functions_of(current_token)
A = stable_learned_decay
state = exp(delta * A) * old_state + delta * B * current_token
output = C * state + skip * current_token
carry_state_to_the_next_token()
```

That is the mathematical heart of Mamba.

### 7.2 Why the Python loop is not the training implementation

A left-to-right loop launches many small operations and underuses a GPU. Mamba uses a parallel **associative scan** and fuses surrounding operations to reduce memory traffic.

Consider

$$
h_t=a_t\odot h_{t-1}+b_t.
$$

Represent a step by the pair $(a_t,b_t)$. Two consecutive steps combine as

$$
(a_2,b_2)\circ(a_1,b_1)
=\left(a_2\odot a_1,\ b_2+a_2\odot b_1\right).
$$

This operator is associative. A parallel prefix-scan can therefore calculate all prefix states with $O(L)$ total work and $O(\log L)$ parallel depth rather than executing a literal Python loop.

Mamba's hardware-aware implementation also:

- keeps expanded states in fast on-chip memory where possible;
- avoids materializing large intermediate tensors in GPU HBM;
- fuses discretization, scan, gating, and related operations;
- recomputes selected intermediates during backward instead of storing them all.

This is analogous in spirit—not mathematics—to FlashAttention: algorithm and memory movement are designed together.

---

## 8. The complete Mamba block

The selective SSM is the core mixer, but a Mamba block contains several surrounding operations:

```text
input x: [B, L, d_model]
        │
        ├── input projection ──► content branch u ─► causal depthwise Conv1d ─► SiLU
        │                                                    │
        │                                                    ▼
        │                                   project token-dependent Δ, B, C
        │                                                    │
        │                                                    ▼
        │                                             selective scan
        │                                                    │
        └── input projection ──► gate branch z ─────► SiLU(z) ⊙ scan output
                                                             │
                                                             ▼
                                                    output projection
                                                             │
                                                    + residual connection
```

The short causal convolution captures immediate local patterns before the long recurrent mixer. The gate branch modulates the scan output in a GLU-like fashion. In a full language model, blocks are stacked with normalization and residual connections.

### 8.1 A readable toy block

```python
class MambaBlockReference(nn.Module):
    """Structure-faithful teaching block; use mamba_ssm for real training."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner = d_inner
        self.d_conv = d_conv

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=d_conv,
            groups=d_inner,       # depthwise
            padding=d_conv - 1,
        )
        self.ssm = SelectiveSSMReference(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        # A production stack normally normalizes before calling the mixer.
        u, z = self.in_proj(x).chunk(2, dim=-1)  # each [B, L, D_inner]

        # Conv1d expects [B, channels, length]. Truncation makes it causal.
        u = self.conv(u.transpose(1, 2))[..., : x.size(1)]
        u = F.silu(u).transpose(1, 2)

        y = self.ssm(u)
        y = y * F.silu(z)
        return self.out_proj(y)


class ResidualMambaLayer(nn.Module):
    def __init__(self, d_model, **mamba_kwargs):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.mixer = MambaBlockReference(d_model, **mamba_kwargs)

    def forward(self, x):
        return x + self.mixer(self.norm(x))
```

### 8.2 Mapping toy names to the official Mamba-1 source

| Concept | Official name |
|---|---|
| split content and gate | `in_proj(...).chunk(2)` → `x, z` |
| local mixer | depthwise `conv1d` + SiLU |
| create $\Delta,B,C$ | `x_proj` and `dt_proj` |
| stable decay | `A = -torch.exp(A_log)` |
| selective recurrence | `selective_scan_fn(...)` |
| output gate | scan output multiplied by `silu(z)` |
| restore model width | `out_proj` |

The official low-rank $\Delta$ parameterization reduces projection cost: `x_proj` first produces a small `dt_rank`, then `dt_proj` expands it to the inner channels.

---

## 9. Training mode versus recurrent inference mode

The same layer has two execution views.

### Parallel training

Given the full sequence, compute all token-dependent parameters and use selective scan. This exposes parallelism across the sequence while preserving the causal recurrence.

### One-token decoding

Keep two fixed-size caches per layer:

1. the last `d_conv - 1` inputs for the short causal convolution;
2. the SSM state `[batch, D, N]`.

For each new token:

```python
conv_state = update_local_conv_cache(new_input)
ssm_state = dA * ssm_state + dB * new_input
new_output = read(ssm_state)
```

The model discards individual old token representations because their useful information has already been compressed into the state. This is why decode memory does not grow with $L$—and also why the state can become an information bottleneck.

---

## 10. Mamba compared with RNNs, convolutions, and attention

### Versus a classical RNN

Both carry recurrent state, but Mamba adds:

- structured, stable state dynamics;
- many learned decay timescales;
- input-dependent discretization/write/read controls;
- an associative parallel scan for training;
- a hardware-aware fused implementation.

It is best understood as a modern, structured recurrent model—not “just an RNN,” but closer to RNNs than a Transformer is.

### Versus a causal convolution

A fixed LTI SSM can be converted into a very long implicit convolution. Mamba's input-dependent parameters break ordinary convolutional time invariance, so one fixed convolution kernel is no longer sufficient. Selective scan restores efficient computation.

### Versus attention

Attention explicitly builds a content-based routing pattern between token pairs. Mamba generates a structured routing pattern implicitly through state transitions. Attention retains individual token-level memories; Mamba continually compresses them.

### Versus linear attention

Both can be written as recurrent state updates and matrix-like sequence mixers. Mamba-2's **state space duality** makes this connection precise for a broad family of structured matrices. The parameterization and inductive bias—not merely the big-$O$ complexity—distinguish the methods.

---

## 11. Mamba-2 and structured state space duality

Mamba-2 begins from the unrolled recurrence. For a simplified scalar transition $a_t$,

$$
h_t=a_t h_{t-1}+B_t x_t,
\qquad
y_t=C_t^\top h_t.
$$

Its sequence transformation can be written

$$
Y=(L\odot CB^\top)X,
$$

where $L$ is a causal structured mask containing products of decay terms:

$$
L_{t,i}=
\begin{cases}
\prod_{k=i+1}^{t}a_k, & i\le t,\\
0, & i>t.
\end{cases}
$$

This is the **structured state space duality (SSD)**:

- recurrent view: efficient constant-state autoregressive decoding;
- structured-matrix view: efficient parallel training with matrix multiplications;
- the two compute the same sequence transformation.

Mamba-2 constrains the transition structure so computation maps more cleanly onto GPU tensor cores, and evaluates chunks with matrix multiplication plus a recurrence between chunks. The paper reports its core SSD layer as **2–8× faster** than Mamba-1's selective SSM core while remaining competitive with Transformers in language modeling.

Conceptually:

| Version | Main idea |
|---|---|
| S4 | structured LTI SSM; recurrent/convolutional duality |
| Mamba-1 | input-dependent selective SSM + hardware-aware scan |
| Mamba-2 | SSM/attention duality + tensor-core-friendly SSD algorithm |
| Mamba-3 | inference-first recurrence, complex state tracking, and MIMO update |

---

## 12. Mamba-3: the current evolution, not required for first understanding

As of July 2026, the official repository also contains Mamba-3. Its core lessons are useful because they clarify what earlier Mamba versions were missing:

1. **Exponential–trapezoidal discretization.** A more expressive discretization generalizes the exponential–Euler update used by Mamba-1/2 and can absorb the role of the explicit short convolution.
2. **Complex-valued state updates.** Lightweight phase rotation, related to a data-dependent rotary mechanism, improves state-tracking capabilities that are difficult for earlier linear models.
3. **MIMO SSM.** Multiple-input, multiple-output state dynamics replace a less hardware-efficient outer-product-style update, doing more useful computation during memory-bound decoding without proportionally increasing latency.

The Mamba-3 paper reports improved retrieval, state-tracking, and language-modeling trade-offs over earlier linear models. However, learn Mamba-1's recurrence first: $\Delta$, $B$, $C$, stable $A$, and selective scan remain the clearest conceptual foundation.

---

## 13. Running the official implementation

The official package is optimized primarily for Linux with a CUDA-capable NVIDIA GPU. Follow the repository's current installation instructions rather than treating the teaching implementation above as a training substitute.

```bash
# Install a CUDA-enabled PyTorch build first.
pip install "causal-conv1d>=1.4.0" --no-build-isolation
pip install mamba-ssm --no-build-isolation
```

Mamba-1 block:

```python
import torch
from mamba_ssm import Mamba

batch, length, dim = 2, 2048, 768
x = torch.randn(batch, length, dim, device="cuda")

mixer = Mamba(
    d_model=dim,
    d_state=16,
    d_conv=4,
    expand=2,
).cuda()

y = mixer(x)
assert y.shape == x.shape
```

Mamba-2 changes the mixer class and commonly uses a larger state:

```python
from mamba_ssm import Mamba2

mixer = Mamba2(
    d_model=dim,
    d_state=64,
    d_conv=4,
    expand=2,
).cuda()
```

For a full language model, the official repository's `models/mixer_seq_simple.py` shows token embeddings, repeated residual Mamba blocks, normalization, and the language-model head.

### 13.1 What to inspect when reading the repository

Read the code in this order:

1. `mamba_ssm/ops/selective_scan_interface.py::selective_scan_ref` — readable recurrence;
2. `mamba_ssm/modules/mamba_simple.py` — complete Mamba-1 mixer;
3. `mamba_ssm/modules/mamba2_simple.py` — readable Mamba-2 block;
4. `mamba_ssm/modules/ssd_minimal.py` — minimal SSD algorithm;
5. optimized CUDA/Triton kernels only after the equations and reference path are clear.

> [!tip] Debugging rule
> Before debugging a fused kernel, compare a tiny input against the reference scan in float32. Verify tensor layout, causal convolution truncation, `softplus(delta + bias)`, the sign of $A$, and state-cache reset behavior.

---

## 14. Applying Mamba beyond text

Mamba accepts a one-dimensional sequence. Other modalities must define an ordering.

### Audio and sensor streams

Time already supplies a natural causal order. Streaming and fixed-state inference are particularly attractive.

### Video

Possible sequences include:

- one feature vector per frame;
- patch tokens ordered within and across frames;
- pose/skeleton joints across time;
- a hierarchy of locally pooled clips.

Mamba can model long temporal context efficiently, but bidirectional offline understanding usually requires forward/backward scans or a specialized non-causal design.

### Images

An image has no unique 1D order. Vision Mamba variants use raster scans, multiple directional scans, hierarchical stages, local convolutions, or deformable/data-dependent scan paths. The scan design determines which pixels interact easily.

Relevant vault reading:

- [[02 Research/Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty/MambaVision A Hybrid Mamba-Transformer Vision Backbone|MambaVision]] — hybrid vision backbone;
- [[02 Research/Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty/DF-Mamba Deformable State Space Modeling for 3D Hand Pose Estimation in Interactions|DF-Mamba]] — learnable deformable sampling for interaction-heavy hand pose;
- [[02 Research/Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty/TSkel-Mamba Temporal Dynamic Modeling via State Space Model for Human Skeleton-Based Action Recognition|TSkel-Mamba]] — pose/skeleton temporal modeling;
- [[02 Research/Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty/MambaTAD When State-Space Models Meet Long-Range Temporal Action Detection|MambaTAD]] — long-range temporal action detection;
- [[02 Research/Thesis Reading List — Self-Adaptors & Discourse-Planning Difficulty/Mamba Linear-Time Sequence Modeling with Selective State Spaces|Mamba paper note]].

For the thesis, the most natural use is often **not** flattening every video pixel into one enormous scan. A stronger starting point is:

```text
frame/clip encoder → compact per-frame or per-person feature sequence
                   → bidirectional/causal Mamba temporal backbone
                   → temporal localization or behavior-classification head
```

This preserves Mamba's temporal efficiency without forcing an arbitrary high-resolution spatial scan to solve everything.

---

## 15. Common misunderstandings

### “Mamba stores the entire past losslessly.”

No. It compresses the past into a fixed-size state. Efficiency comes partly from this bottleneck.

### “Linear time means every implementation is automatically faster.”

No. Constants, tensor shapes, kernel maturity, hardware, batch size, and sequence length matter. Dense attention uses extremely optimized matrix multiplication and can be faster for shorter sequences.

### “Mamba has no convolution.”

Mamba-1/2 blocks normally include a short causal depthwise convolution before the selective SSM. Mamba-3 proposes a recurrence/discretization that can replace that explicit convolution in its architecture.

### “Selection is attention without softmax.”

Not exactly. Both are input-dependent sequence mixers, but Mamba's weights are constrained by a recurrent product structure. It cannot freely create an arbitrary query–key score matrix.

### “Mamba needs no notion of position.”

The recurrence inherently encodes order, so original Mamba does not require standard absolute positional embeddings. This does not mean all positional distinctions or extrapolation problems are automatically solved; later variants can introduce richer positional/state-rotation mechanisms.

### “The state size $N$ is the same as hidden width $D$.”

No. $D$ counts channels in the mixer; each channel has $N$ state coordinates. The recurrent state is roughly `[batch, D, N]`.

---

## 16. A compact derivation to reproduce from memory

1. Start with a stable continuous system:

   $$\dot h=Ah+Bx,\quad y=Ch+Dx.$$

2. Discretize it:

   $$h_t=\bar Ah_{t-1}+\bar Bx_t.$$

3. Make the step/write/read input-dependent:

   $$\Delta_t,B_t,C_t=f(x_t).$$

4. Keep $A$ stable and obtain a selective transition:

   $$\bar A_t=\exp(\Delta_tA),\quad A=-\exp(A_{\log}).$$

5. Scan:

   $$h_t=\bar A_t\odot h_{t-1}+\Delta_tB_tx_t,$$

   $$y_t=C_t h_t+Dx_t.$$

6. Wrap the scan with local convolution, activation, gating, projection, normalization, and a residual connection.

7. Train with a parallel associative scan; decode by carrying the final state one token at a time.

If these seven steps are clear, the architecture is clear. The remaining complexity is parameter efficiency and kernel engineering.

---

## 17. Study checklist

- [ ] I can explain Mamba as selective recurrent memory without using equations.
- [ ] I can state why Transformer attention scales quadratically in sequence length.
- [ ] I understand $A$ as decay/dynamics, $B$ as write, $C$ as read, and $D$ as skip.
- [ ] I can explain why $A=-e^{A_{\log}}$ and $\Delta=\operatorname{softplus}(\cdot)$ make the decay stable.
- [ ] I can trace `[B,L,D] → [B,D,N] state → [B,L,D]`.
- [ ] I can implement the slow recurrence loop.
- [ ] I understand why input dependence breaks the fixed-convolution view.
- [ ] I can derive the associative pair-composition rule.
- [ ] I know the efficiency advantages and the fixed-state information bottleneck.
- [ ] I can describe how Mamba-2's SSD connects recurrent and matrix views.

---

## Sources

### Primary papers

1. Albert Gu and Tri Dao, [“Mamba: Linear-Time Sequence Modeling with Selective State Spaces”](https://arxiv.org/abs/2312.00752), 2023/ICLR 2024.
2. Tri Dao and Albert Gu, [“Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality”](https://arxiv.org/abs/2405.21060), 2024.
3. Aakash Lahoti et al., [“Mamba-3: Improved Sequence Modeling using State Space Principles”](https://arxiv.org/abs/2603.15569), 2026.
4. Albert Gu, Karan Goel, and Christopher Ré, [“Efficiently Modeling Long Sequences with Structured State Spaces”](https://arxiv.org/abs/2111.00396), 2021/ICLR 2022.

### Official implementation

- [state-spaces/mamba](https://github.com/state-spaces/mamba) — Mamba-1, Mamba-2/SSD, Mamba-3, language-model examples, and optimized kernels.
- [state-spaces/s4](https://github.com/state-spaces/s4) — structured SSM lineage and official S4 implementations.

