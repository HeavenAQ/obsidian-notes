---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-02T13:18:00
Status: In progress
Last updated time: 2026-07-08T11:36:00
Last edited by: Heaven Chen
Category:
  - GPU
  - Hardware
  - Theory
---
> Source: [MIT 6.S191 Lecture 9 slides](https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L9.pdf) — "Secrets to Massively Parallel Training", Mathias Lechner (CTO, Liquid AI). The slides are image-based; this doc reconstructs the full argument in depth: why GPUs → why scale → memory anatomy → parallelism strategies → sharding → MoE → bandwidth → LFM2 case study.

# 1. Why GPUs

Training is dense linear algebra. A CPU delivers ~1–3 TFLOP/s; an H100 delivers ~1000 TFLOP/s (bf16 tensor cores), a B200 ~2.2 PFLOP/s. The tensor core primitive is a small matrix-multiply-accumulate ($D = A B + C$ on 16×16 tiles), so anything you can phrase as matmul rides the fast path. The metric that matters is **MFU** (model FLOPs utilization): achieved useful FLOPs ÷ peak. 40–60% MFU is good for large-scale training.

# 2. Why scale — scaling laws

Kaplan et al.: loss follows a power law in model size $N$, data $D$, compute $C$: $L(N) \approx (N_c/N)^{\alpha_N}$, etc. Chinchilla refined the *allocation*: for a compute budget $C \approx 6ND$, optimal training sets $D \approx 20N$ (tokens ≈ 20× parameters). LLaMA-style models deliberately **overtrain** past Chinchilla-optimal because inference cost depends on $N$ only — small model + more tokens wins in deployment. This is the economic argument that forces multi-thousand-GPU training.

# 3. Memory anatomy of one training step

For a model with $\Psi$ parameters trained with Adam in mixed precision, per-parameter bytes:

$\underbrace{2}_{\text{bf16 weights}} + \underbrace{2}_{\text{bf16 grads}} + \underbrace{4+4+4}_{\text{fp32 master + }m\text{ + }v} = 16 \text{ bytes/param}$

So a 7B model needs ~112 GB *before activations* — it does not fit on one 80 GB GPU. Activations add $\mathcal{O}(B \cdot t \cdot d \cdot L)$ and dominate at long context.

- **Activation checkpointing**: store activations only at block boundaries; recompute inside blocks during backward. Memory $\mathcal{O}(L) \to \mathcal{O}(\sqrt{L})$ at ~33% extra compute (one extra forward).
- **Offloading**: park optimizer states/params in CPU RAM or NVMe (ZeRO-Offload/Infinity); trades PCIe bandwidth for capacity.

# 4. The four parallelisms

**Data parallel (DP)**: replicate the model, split the batch. After backward, all-reduce gradients: $g \leftarrow \frac{1}{P}\sum_p g_p$. Communication per step = $2\Psi$ bytes-ish per rank (ring all-reduce moves $2\frac{P-1}{P}\Psi$). Simple, but memory per GPU unchanged.

**Tensor parallel (TP)**: split individual weight matrices across GPUs. Megatron pattern for an MLP: first layer column-split ($Y_i = X W_i$, no comm), second layer row-split, then one all-reduce to sum partial outputs. Attention heads shard naturally. Needs comm **inside every layer**, so TP stays within a node (NVLink domain), typically TP ≤ 8.

**Pipeline parallel (PP)**: assign consecutive layers to stages; micro-batch to fill the pipe. Bubble fraction:

$\text{bubble} = \frac{p-1}{m + p - 1}$

for $p$ stages, $m$ micro-batches — so you want $m \gg p$ (1F1B / interleaved schedules shrink the bubble and the activation footprint).

**Sequence/context parallel (SP/CP)**: shard the *sequence* dimension; ring attention passes KV blocks around ranks so each holds $t/P$ tokens. This is what makes 1M-token context training feasible.

Real systems compose them: e.g., 3-D parallelism DP×TP×PP (+ CP for long context).

# 5. Sharding — ZeRO and FSDP

ZeRO removes DP's redundancy in three stages: Stage 1 shards optimizer states (16→~6 bytes/param per rank), Stage 2 also shards gradients (~4), Stage 3 also shards parameters (~16/P). PyTorch-native equivalent = **FSDP** (FSDP2 is per-parameter sharded on DTensor; FSDP1 is deprecated since 2.11). Mechanics per layer: all-gather params → compute → free; backward adds reduce-scatter of grads. Communication ≈ 3× DP volume, hidden by overlap with compute.

```python
# FSDP2 sketch
from torch.distributed.fsdp import fully_shard
for block in model.blocks:
    fully_shard(block)          # shard each transformer block
fully_shard(model)              # root wrap
loss = model(x).loss; loss.backward(); opt.step()
```

# 6. Sparsity — Mixture of Experts

Replace each MLP with $E$ experts + router; each token activates top-$k$ (usually 1–2). Compute scales with *active* params, capacity with *total* params. **Expert parallelism** places experts on different GPUs; tokens are exchanged via all-to-all (two per MoE layer: dispatch + combine). Load-balancing auxiliary loss keeps the router from collapsing onto few experts.

# 7. The real bottleneck: network

Hierarchy of bandwidths: HBM ~3.3 TB/s ≫ NVLink ~900 GB/s ≫ InfiniBand ~50–100 GB/s/rank ≫ Ethernet. Rule: put the chattiest parallelism on the fastest fabric — TP inside the node, DP/FSDP across nodes, PP across the slowest links. If comm time > overlappable compute time, MFU collapses; this single constraint explains almost every deployment layout.

# 8. Case study: LFM2 (Liquid AI)

LFM2 combines the above: hybrid architecture (conv + attention blocks) chosen for on-device latency, trained with a composed DP+TP(+checkpointing/sharding) stack where the parallelism layout was dictated by the bandwidth hierarchy, not by preference. The lecture's closing point: **architecture and parallelization co-design** — the model shape you pick determines which parallelism (and which fabric) you can use efficiently.

# Cross-check questions

1. Compute bytes/param for ZeRO-2 at P=64. 2. Why must TP stay intra-node while PP tolerates slow links? 3. For p=8, how many micro-batches keep the bubble under 10%? (m ≥ 63.)