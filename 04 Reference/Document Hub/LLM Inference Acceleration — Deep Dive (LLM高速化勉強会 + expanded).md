---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-08T11:33:00
Status: Done
Last updated time: 2026-07-10T20:33:00
Last edited by: Heaven Chen
Category:
  - LLM
  - GPU
  - Hardware
---
> Based on the slide deck **"LLM高速化 (勉強会)"** by SuperHotDog ([Speaker Deck, July 2026](https://speakerdeck.com/superhotdogcat/llmgao-su-hua-mian-qiang-hui-zi-liao)), expanded with external papers, engineering blogs, and 2025–2026 state-of-the-art techniques. Goal: go from "knows a little about transformers" to "can reason like an inference-systems engineer." A final section covers accelerating **video captioning / video-LLM inference**, which the deck does not cover.

> The deck's headline experiment: the same model (Qwen2.5-0.5B-Instruct, free Colab Tesla GPU) runs at **5.97 tokens/s** with plain HuggingFace Transformers and **94.84 tokens/s** with vLLM — a **15.9× speedup with zero model changes**. Every section below explains one piece of where that 15.9× comes from.

# 0. The Mental Model You Need First

Before any specific technique, internalize three facts. Almost everything in this note is a consequence of them.

**Fact 1 — Autoregressive generation is sequential.** A decoder-only LLM generates one token at a time; each new token depends on all previous ones. Generating K tokens naively requires K forward passes. You cannot parallelize *across* output tokens of a single request (speculative decoding is the loophole — see §7).

**Fact 2 — Decode is memory-bandwidth-bound, not compute-bound.** During generation, every step must read *all model weights* (and the growing KV cache) from GPU memory to produce a single token. An H100 can do ~1000 TFLOPS of matmul but only ~3.35 TB/s of HBM bandwidth. For a 7B FP16 model, one decode step reads ≥14 GB → the theoretical ceiling is ~240 tokens/s per request *regardless of compute*. So most decode optimizations are really **memory-traffic optimizations**: quantization, KV-cache shrinking (GQA/MLA), FlashAttention, kernel fusion.

**Fact 3 — The Roofline model tells you which regime you're in.** Plot performance vs. *arithmetic intensity* (FLOPs per byte moved). Below the "ridge point" you're bandwidth-bound; above it, compute-bound. On an H100 (FP16), the ridge is at roughly 300 FLOPs/byte. Big-batch matrix–matrix products (prefill, training) exceed it; matrix–vector products (batch-1 decode) sit far below it. As the deck says: *"look at the Roofline model — pretty much only large matmuls are compute-bound."* This is also why **prefill and decode are treated as two distinct workloads** (§12.4).

> [!note]+ **Roofline model**
> The roofline model is an intuitive visual performance model that provides performance estimates for a given compute kernel or application running on multi-core, many-core, or accelerator processor architectures by showing inherent hardware limitations and the potential benefits and priorities of optimizations.

Two phases of one request:

- **Prefill**: process the whole prompt in one parallel pass, build the KV cache. Compute-bound (big matmuls). Metric: **TTFT** (time to first token).
- **Decode**: generate tokens one-by-one reusing the KV cache. Memory-bound. Metric: **TPOT / ITL** (time per output token / inter-token latency).

<!-- Column 1 -->
![[99 Assets/Media/image 1.png]]


<!-- Column 2 -->
![[99 Assets/Media/image 2.png]]


Back-of-envelope check you should be able to do from memory:

```python
# Which regime am I in? (H100 SXM, FP16)
PEAK_FLOPS = 989e12            # dense FP16 tensor-core FLOP/s
HBM_BW     = 3.35e12           # bytes/s
ridge      = PEAK_FLOPS / HBM_BW      # ≈ 295 FLOPs/byte

# Batch-1 decode of a 7B FP16 model:
flops_per_token = 2 * 7e9      # ~2 FLOPs per parameter
bytes_per_token = 2 * 7e9      # must read every FP16 weight once
intensity = flops_per_token / bytes_per_token  # = 1.0 → deeply memory-bound
ceiling   = HBM_BW / bytes_per_token           # ≈ 239 tokens/s, no matter what
# Batching to 64 requests → intensity ≈ 64 → still memory-bound, but
# 64× more tokens per weight-read. That's why batching = throughput.
```

Reference: [NVIDIA — Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/).

# 1. Refresher: Autoregressive Generation, the Transformer Decoder, and Attention

*(Deck slides 4–6)*

A language model estimates P(next token | previous tokens). Text generation is a loop: tokenize prompt → forward pass → get logits over the vocabulary → sample/argmax the next token → append → repeat until EOS or length limit.

```python
# The generation loop, stripped to its essence
ids = tokenizer(prompt).input_ids
for _ in range(max_new_tokens):
    logits  = model(ids)                 # (1, seq_len, vocab_size)
    next_id = sample(logits[0, -1])      # ONLY the last position matters...
    ids     = torch.cat([ids, next_id])  # ...yet naive code recomputes ALL
    if next_id == EOS: break             #    positions every step (fix: §3)
```

![[99 Assets/Media/image 3.png]]

A decoder-only Transformer layer = **causal self-attention** + **feed-forward network (MLP)**, each wrapped with residual connections and normalization (RMSNorm in modern models). Modern extras: RoPE positional encoding, SwiGLU MLPs, and often Mixture-of-Experts (MoE) MLPs where only a few "experts" activate per token.

**Attention**, per head:

```javascript
Attention(Q, K, V) = softmax(Q Kᵀ / √d_head) V
```

- Q, K, V are linear projections of the hidden states.
- The **causal mask** zeroes out attention to future positions.
- Naively, computing attention for a sequence of length *n* is O(n²) in both time and memory (the n×n score matrix) — the root of the long-context problem.

Key observation the whole field builds on: to predict the *next* token you only need the **last position's hidden state**, but that position's attention must read **K and V for every previous token**. Those K/V values never change once computed → cache them. That's the KV cache (§3).

Sampling knobs (temperature, top-p, top-k) live outside the model and matter for quality, not speed — except that greedy vs. sampling affects speculative-decoding acceptance rates (§7).

# 2. Baseline: Naive Inference with HuggingFace Transformers

*(Deck slides 8, 38)*

The deck's hands-on Colab runs Qwen2.5-0.5B-Instruct via `transformers` `model.generate()` → **5.97 tokens/s**. Why so slow, even on a GPU?

1. **Python/CPU overhead dominates.** Each decode step launches hundreds of small CUDA kernels from Python. Each launch costs ~5–30 µs of CPU time; a Python for-loop iteration alone can burn 10–100 µs (deck slide 24). For a 0.5B model the GPU finishes each kernel faster than the CPU can issue the next one → the GPU idles between kernels. In Nsight Systems timelines you literally see ~100 µs gaps between kernels (deck slide 38). Fix: CUDA Graphs (§10), `torch.compile`.
2. **No continuous batching** — one request at a time, or padded static batches (§5).
3. **Naive KV cache layout** — one big contiguous tensor per request, pre-allocated to max length → massive fragmentation and waste (§6).
4. **Unfused kernels** — every op (each matmul, softmax, norm) round-trips activations through HBM (§4, §8).

```python
# The deck's baseline (Colab, T4 GPU) — ~6 tok/s
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype="float16").cuda()
out = model.generate(**inputs, max_new_tokens=256)
```

What the Nsight Systems timeline of this looks like (the launch-bound pathology):

```javascript
CPU (Python):  ─launch─┐~30µs  ─launch─┐      ─launch─┐        ← can't keep up
GPU:                   ▓▓ ……idle…… ▓▓ …idle… ▓▓ …idle…
                     kernel        kernel     kernel
                       └── ~100 µs gaps between kernels (deck slide 38) ──┘
```

This baseline is the "before" picture. vLLM's 94.84 tokens/s is the "after" — same weights, same GPU.

# 3. KV Cache: The Foundational Optimization

*(Deck slides 9–10)*

**Idea.** In causal attention, K and V vectors of past tokens are identical at every future step. Cache them. Then each decode step computes Q/K/V only for the *one new token*, and attends against cached K/V. Per-step attention cost drops from O(n²) to O(n); the model-side matmuls go from (n × d) to (1 × d).

```python
# One decode step WITH the cache: compute Q/K/V for the 1 new token only
q = x_new @ W_q;  k = x_new @ W_k;  v = x_new @ W_v      # (1, d) each
K = torch.cat([K_cache, k]);  V = torch.cat([V_cache, v]) # append, don't recompute
attn_out = softmax(q @ K.T / sqrt(d)) @ V                 # O(n·d) instead of O(n²·d)
K_cache, V_cache = K, V
```

This creates the prefill/decode split by definition (deck's terminology): **prefill** = the phase that builds the cache from the full prompt; **decode** = the phase that extends it one token at a time.

```javascript
Prefill (one parallel pass):            Decode (sequential loop):
[t1 t2 t3 t4 t5] ──► K/V for all 5      t6 ─► append K6,V6 ─► t7 ─► append ...
  big matmuls → compute-bound             reads ENTIRE cache + all weights
  metric: TTFT                            per step → memory-bound; metric: TPOT
```

**The cost: memory.** KV cache size in bytes:

```javascript
2 (K and V) × batch_size × num_layers × seq_len × num_kv_heads × head_dim × bytes_per_elem
```

```python
def kv_bytes(batch, layers, seq_len, kv_heads, head_dim, dtype_bytes=2):
    return 2 * batch * layers * seq_len * kv_heads * head_dim * dtype_bytes

kv_bytes(16, 96, 32768, 8, 128) / 2**30   # → 192.0 GiB  (deck's example)
kv_bytes(1, 32, 8192, 8, 128)  / 2**30    # → 1.0 GiB    (Llama-3-8B, one 8k chat)
```

Deck's example: batch=16, layers=96, seq_len=32768, heads=8, head_dim=128, FP16 → **192 GiB** — more than two H100s' worth of memory *just for cache*, before weights. (Verified: 2·16·96·32768·8·128·2 B = 192 GiB.)

Consequences that drive the rest of this note:

- The KV cache, not weights, limits **batch size and context length** in serving → GQA/MLA/sliding/linear attention (§9) exist to shrink `num_kv_heads × head_dim` or `seq_len` in that formula.
- Reading the cache every step adds to memory-bandwidth pressure → long-context decode gets slower per token as the sequence grows.
- Managing many requests' caches efficiently is a memory-allocation problem → PagedAttention (§6).

**Beyond the deck — KV-cache compression at runtime** (active 2024–2026 research area; see [MarkTechPost's survey of top-10 KV compression techniques](https://www.marktechpost.com/2026/04/29/top-10-kv-cache-compression-techniques-for-llm-inference-reducing-memory-overhead-across-eviction-quantization-and-low-rank-methods/) and [this interactive explainer](https://mbrenndoerfer.com/writing/kv-cache-compression-eviction-quantization-h2o-algorithm)):

- **Eviction / token dropping**: **StreamingLLM** keeps only the first few tokens ("attention sinks" — they absorb disproportionate attention mass) plus a sliding recent window, and remains coherent for millions of tokens. **H2O** ("Heavy-Hitter Oracle") keeps the small set of tokens that accumulate most attention and evicts the rest. **SnapKV** selects per-head important tokens before decode.
- **KV quantization**: KIVI, KVQuant, and production INT8/FP8 KV caches (vLLM/TensorRT-LLM support FP8 KV) → 2–4× memory cut, near-zero quality loss.
- **Low-rank**: compress KV to latent vectors — this is architectural in MLA (§9.2).

# 4. FlashAttention: IO-Aware Exact Attention

*(Deck slides 11–13)*

**The problem.** Standard attention materializes the n×n score matrix `S = QKᵀ`, writes it to HBM, reads it back for softmax, writes the result, reads it again to multiply by V. GPU **HBM** has TB/s-class bandwidth, but on-chip **SRAM (shared memory)** is ~10–13× faster (the deck cites a 12.66× gap, from the FlashAttention paper's A100 numbers: 1.5–2 TB/s HBM vs ~19 TB/s SRAM). Attention is bandwidth-bound → all those HBM round-trips are the bottleneck, not FLOPs.

**The fix (FlashAttention, Dao et al. 2022).** Never materialize the score matrix:

5. **Tiling**: split Q, K, V into blocks small enough to fit in SRAM; compute attention block-by-block.
6. **Online softmax**: softmax normally needs the whole row (for the max and the sum). The online-softmax trick maintains a running max `m` and running sum `ℓ` and *rescales previously accumulated output* whenever a new block raises the max. Mathematically exact — FlashAttention is **not an approximation**.
7. Backward pass recomputes the score blocks instead of storing them (relevant for training).

```javascript
HBM (slow, ~2 TB/s):    Q ────┐            K, V ────┐
                              │ load one Q tile     │ stream K/V tiles
                              ▼                     ▼
SRAM (fast, ~19 TB/s):  [Q_i]·[K_j]ᵀ → score tile → online softmax → accumulate O_i
                        └────── the n×n score matrix NEVER touches HBM ──────┘
HBM write:              final O_i only
```

```python
# Online softmax for one Q tile — numerically exact, not an approximation
m, l, O = -inf, 0, 0
for K_j, V_j in kv_tiles:              # stream K/V through SRAM
    S     = Q_i @ K_j.T / sqrt(d)      # small tile of scores
    m_new = max(m, S.max())
    P     = exp(S - m_new)
    l     = l * exp(m - m_new) + P.sum()
    O     = O * exp(m - m_new) + P @ V_j   # rescale PAST contributions
    m     = m_new                          # whenever the running max rises
O /= l
```

Result: O(n²) HBM traffic → O(n² d / SRAM_size), in practice 2–4× wall-clock speedup and **linear memory** in sequence length. This single kernel made 32k–128k contexts practical.

**Version history** (worth knowing — interviews love this):

- **FlashAttention-2** (2023): better work partitioning across warps/blocks, parallelize over sequence dimension → ~2× over v1, ~50–70% of A100 peak.
- [**FlashAttention-3**](https://tridao.me/blog/2024/flash3/) (2024, Hopper/H100): exploits **asynchrony** — warp-specialized producer/consumer pipelines where producer warps prefetch tiles via **TMA** (Tensor Memory Accelerator) while consumer warps do matmul+softmax; overlaps softmax with GEMMs; FP8 support with incoherent processing. ~1.5–2× over FA-2, up to ~740 TFLOPS on H100 ([paper](https://arxiv.org/pdf/2407.08608), [PyTorch blog](https://pytorch.org/blog/flashattention-3/)).
- [**FlashAttention-4**](https://www.together.ai/blog/flashattention-4) (2025–26, Blackwell/B200): co-designed for "asymmetric hardware scaling" — Blackwell doubled tensor-core throughput but not SFU/exp throughput or SMEM bandwidth, so softmax's `exp` becomes the new bottleneck. FA-4 uses **software-emulated exponentials**, conditional softmax rescaling (only rescale when the running max actually changes), larger 128×128 tiles in **TMEM**, and 2-CTA MMA. ~1.5–2× over FA-3 on Blackwell ([arXiv](https://arxiv.org/html/2603.05451v1)). Notably written in **CuTe-DSL (Python)**, not CUDA C++ (§8.3).

**The meta-lesson** (deck slide 26): *"80% of CUDA optimization is optimizing how you read memory."* FlashAttention is the canonical example of an **IO-aware algorithm** — same math, radically different data movement. The theory behind this is the **Red-Blue Pebble Game** (Hong & Kung 1981) and **communication-avoiding algorithms**: lower bounds on data movement between a small fast memory and a large slow one. The deck explicitly recommends looking these up.

# 5. Batching: Static → "Super Sequence" → Continuous Batching

*(Deck slides 14–16)*

A serving system must multiplex many concurrent requests onto one GPU. Batching amortizes the per-step weight-read across requests — this is THE throughput lever, because decode is bandwidth-bound (reading 14 GB of weights to make 1 token is wasteful; reading it to make 64 tokens for 64 requests is 64× more efficient).

**Static batching (the PyTorch-default way)** builds a `(batch_size, max_len)` tensor. Two problems:

- **Padding waste**: every short request is padded to the longest one; padding tokens consume compute and memory.
- **Head-of-line blocking**: the batch finishes when its *longest* generation finishes; finished requests' slots idle, and new requests wait for the whole batch.

**Super Sequence (packed / ragged batching).** (Name from the vLLM blog, as the deck notes.) Instead of a 2-D padded tensor, concatenate all requests' tokens into **one 1-D token stream** with metadata recording sequence boundaries (`cu_seqlens`, as in FlashAttention's `varlen` kernels). No padding at all. The attention kernel uses **block-diagonal causal masking** so tokens of request A never attend to request B (deck slide 16). A new request = just append its tokens at the end of the stream.

```javascript
Static (padded) batch:               Super sequence (packed):
req A: [a1 a2 a3 P  P ]              [a1 a2 a3 | b1 b2 b3 b4 b5 | c1 c2]
req B: [b1 b2 b3 b4 b5]              cu_seqlens = [0, 3, 8, 10]
req C: [c1 c2 P  P  P ]              (FlashAttention varlen-style metadata)
       P = wasted padding

Block-diagonal causal mask (deck slide 16):
        a1 a2 a3 b1 b2 b3
   a1 [ ■                ]
   a2 [ ■  ■             ]
   a3 [ ■  ■  ■          ]
   b1 [          ■       ]   ← B's tokens can never attend to A's
   b2 [          ■  ■    ]
   b3 [          ■  ■  ■ ]
```

**Continuous batching (a.k.a. in-flight batching, from the Orca paper, OSDI 2022).** Because the super sequence makes joining/leaving trivial, scheduling can operate at **iteration granularity** instead of batch granularity: after *every* forward step, remove finished sequences and admit waiting ones. GPU utilization jumps from ~30–40% to 75–90% in typical serving; throughput improves 2–4× and up to ~20× vs naive batching in bursty traffic ([overview](https://www.spheron.network/blog/llm-serving-optimization-continuous-batching-paged-attention/)). Every serious engine (vLLM, TensorRT-LLM, SGLang, TGI) does this.

```javascript
Iteration-level scheduling:
step:        1     2     3     4     5     6
req A      [■]───[■]───[■] done ─┐
req B      [■]───[■]───[■]───[■]─┼─[■]───[■]
req C  (arrives at step 3) ──[■]─┴─[■]───[■]   ← admitted next step, slot reused
         no waiting for the whole batch to finish
```

One subtlety: mixing prefill (long, compute-heavy) and decode (1 token each) in the same iteration causes latency spikes for decodes — resolved by chunked prefill or P/D disaggregation (§12).

# 6. PagedAttention: Virtual Memory for the KV Cache

*(Deck slide 17)*

**Problem.** With continuous batching, requests arrive/finish constantly and their KV caches grow unpredictably. Contiguous per-request allocation forces reserving `max_len` upfront → in the original vLLM paper's measurements, **60–80% of KV memory was wasted** on internal/external fragmentation and reservations.

**Solution (vLLM, SOSP 2023): borrow OS paging.**

- Chop each request's KV cache into fixed-size **blocks** (default 16 tokens/block).
- Allocate blocks **on demand** from a global pool; a per-request **block table** maps logical positions → physical blocks (exactly like a page table).
- The attention kernel gathers K/V through the block table (this is the hard part — an efficient gather-based kernel; as the deck says, *"the concept is easy, the implementation is what deserves the credit — in HPC, implementation is what matters"*).

```javascript
Logical KV cache (per request)          Physical block pool (global)
req A: [blk0][blk1][blk2]...            ┌────┬────┬────┬────┬────┬────┐
          │     │     │                 │ A0 │ B0 │ A1 │ C0 │ A2 │ B1 │ ...
          ▼     ▼     ▼                 └────┴────┴────┴────┴────┴────┘
   block table: 0→slot0, 1→slot2,        16-token blocks, any order,
                2→slot4  (page table)    allocated on demand

Prefix sharing: req A and req B use the same system prompt
   A.block_table = [S0, S1, A2, ...]  ┐ S0,S1 physical blocks shared
   B.block_table = [S0, S1, B2, ...]  ┘ (copy-on-write on divergence)
```

```python
# Allocation logic, conceptually — this is all "paging" means here
if req.num_tokens % BLOCK_SIZE == 0:          # current block just filled
    req.block_table.append(pool.alloc())      # grab ANY free block, O(1)
# on completion:
pool.free(req.block_table)                    # zero fragmentation
```

Benefits:

- Near-zero fragmentation → **2–4× more concurrent requests** on the same GPU.
- **Copy-on-write sharing**: parallel sampling / beam search share the prompt's blocks.
- **Prefix caching** falls out naturally: requests sharing a prompt prefix (system prompts, few-shot examples, multi-turn history) can share physical blocks. vLLM has automatic prefix caching; **SGLang's RadixAttention** generalizes it with a radix tree over all past requests, giving 85–95% prefill-cost savings on cache hits — arguably the single highest-leverage serving optimization for agentic/chat workloads ([KV-cache engineering guide](https://www.digitalapplied.com/blog/kv-cache-optimization-techniques-2026-engineering-guide)).
- Enables **preemption** by page-out (§12.3).

Original paper: [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180). The vLLM blog's animated GIFs (referenced by the deck) are the best intuition source.

# 7. Speculative Decoding: Breaking the One-Token-Per-Forward Barrier

*(Deck slides 18–23)*

**Problem.** K output tokens = K sequential forwards of the big model. For 100B+ models the per-forward latency is irreducible... unless you can verify several tokens in *one* forward.

**Core algorithm (Leviathan et al. / Chen et al. 2023).**

8. A cheap **draft** proposes γ tokens autoregressively.
9. The **target** model runs ONE forward over all γ proposals in parallel (this is cheap because decode is memory-bound — verifying 5 tokens reads the weights once, barely more expensive than generating 1).
10. **Rejection sampling** accepts each draft token with probability `min(1, p_target/p_draft)`; at the first rejection, resample from the corrected distribution and discard the rest.

The output distribution is **provably identical** to the target model's — it's a lossless speedup. Typical gains: 2–4× when the draft is well-matched. Acceptance rate is everything: speedup ≈ `(accepted+1) / (draft_cost_ratio·γ + 1)`.

```javascript
draft model (cheap, autoregressive):   t̂1  t̂2  t̂3  t̂4  t̂5     γ = 5 guesses
                                        │   │   │   │   │
target model (ONE parallel forward):    ✓   ✓   ✓   ✗ ──┐  verify all at once
                                                        │
output this round:                      t1  t2  t3  + t4′ (resampled)
→ 4 tokens for the price of ~1 target forward
```

```python
# Verification step (guarantees exact target distribution)
p = target_logits.softmax(-1)          # from ONE forward over all γ drafts
for i, tok in enumerate(draft_tokens):
    if rand() < min(1, p[i][tok] / q[i][tok]):   # q = draft's probability
        accept(tok)
    else:                              # first rejection:
        resample_from(normalize(relu(p[i] - q[i])))   # corrected distribution
        break                          # discard the rest, roll back KV cache
```

**Why a separate small draft model is impractical** (deck slide 19): you must host a second model *and its own KV cache*, and its distribution may mismatch the target's. Hence "draft-free" families:

- **Medusa** (2024): bolt K extra prediction heads onto the target model itself; each head predicts token t+2, t+3, ...; verify a **tree** of candidates in one forward using special tree attention masks. Simple to train, no separate KV cache.
- **EAGLE** (2024): drafts in **feature space** — a tiny 1-layer autoregressive head predicts the target's next *hidden state* (much more predictable than tokens) then decodes tokens from it. **EAGLE-2** adds dynamic draft trees. [**EAGLE-3**](https://arxiv.org/html/2503.01840v1) (2025) drops feature-prediction loss, fuses low/mid/high-level hidden states via "training-time test," pushing acceptance >75% and giving the best general-purpose speedups as of 2026.
- **Multi-Token Prediction (MTP)**: prediction heads trained **jointly with the base model from the start** (vs. retrofitted). DeepSeek-V3 ships native MTP heads — flip a flag in vLLM/SGLang and get ~1.8× out of the box; by 2026 DeepSeek-V3.x/V4, GLM, some Llama-4 variants, and (per the deck's surprise) **Gemma 4** ship MTP heads ([MTP deployment guide](https://www.spheron.network/blog/multi-token-prediction-mtp-gpu-cloud-deployment-guide/), [2026 speculative-decoding survey](https://localaimaster.com/blog/speculative-decoding-guide)). The deck's remark that MTP-style approaches **share the target's KV cache** is the key design point: KV memory is so precious that draft mechanisms are chosen specifically to avoid adding cache.
- Interaction warning (deck slide 32): speculative decoding assumes you can *roll back* rejected tokens, which is trivial with a KV cache (truncate) but awkward for **linear-attention/recurrent layers** whose state is a single fused hidden state — one reason hybrids keep some full-attention layers.

**The deck's clustered-vocabulary aside (slides 22–23), explained.** The LM head is a `hidden_dim × vocab_size` matmul — for a small draft head this dominates cost (e.g. 128k vocab). Trick (hierarchical softmax revived): partition the vocabulary into clusters; first predict the **cluster** (`hidden_dim × cluster_size`), then the token **within** the cluster (`hidden_dim × vocab_size_in_cluster`). Cost falls from `hidden_dim · V` to `hidden_dim · (C + V/C)`. By AM–GM, `C + V/C ≥ 2√V` (minimized at C = √V ≈ 358 for V=128k → ~180× fewer LM-head FLOPs), and it's < V for any non-extreme C. Bonus: the second-stage weight matrix load is much smaller — again a memory-traffic win.

# 8. Implementing Fast Kernels: CUDA, Triton, CuTe-DSL, Kernel Fusion

*(Deck slides 24–27)*

**Why custom kernels at all?** Python dispatch overhead (10–100 µs per op) and unfused ops make eager PyTorch unable to express FlashAttention/PagedAttention efficiently. Anything hot must run *as few, large, fused GPU kernels* as possible.

## 8.1 CUDA (C++)

The low-level option: you program at thread/warp/block granularity, manage **shared memory** (the configurable SRAM cache) explicitly, and worry about coalesced loads, bank conflicts, occupancy, and tensor-core (WMMA/MMA) instructions. The deck's advice: work through [Lei Mao's CUDA Matrix Multiplication Optimization](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/) once — it walks a naive GEMM through tiling, shared memory, register blocking, and vectorized access, teaching you 80% of what matters. And that 80% is **memory**: kernels rarely lose on arithmetic; they lose on how data moves through HBM → L2 → SMEM → registers. (Theory pointers from the deck: Red-Blue Pebble Game, communication-avoiding algorithms.)

```javascript
The memory hierarchy you are actually programming:
  registers    ~100s TB/s agg. │ KBs per thread │ accumulators
  SMEM / L1    ~19+ TB/s       │ ≤228 KB per SM │ tiles ← YOU manage this
  L2           ~10 TB/s        │ ~50 MB         │ shared across SMs
  HBM3         3.35 TB/s       │ 80 GB          │ weights, KV cache ← bottleneck
Rule: touch HBM once, then do as much work as possible up the hierarchy.
```

## 8.2 Triton

[OpenAI's Triton](https://oneinfer.ai/blogs/triton-vs-cuda-kernels-which-should-you-optimize-for) is a Python-embedded DSL: you write **tile-level** programs (`tl.load`, `tl.dot`, `tl.store` on block pointers) and the compiler handles thread mapping, shared-memory staging, and pipelining. Productivity is ~10× CUDA for common patterns at 80–95% of hand-tuned performance. It's the default vehicle for LLM kernels now: `torch.compile` emits Triton, Liger-Kernel (fused RMSNorm/SwiGLU/CE for training) is pure Triton, much of vLLM is Triton.

```python
import triton, triton.language as tl

@triton.jit
def fused_softmax(X, Y, n_cols, BLOCK: tl.constexpr):
    row  = tl.program_id(0)                       # one program per row
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * n_cols + cols,
                mask=cols < n_cols, other=-float('inf'))   # HBM → registers, once
    x = x - tl.max(x, axis=0)                     # all math stays on-chip
    num = tl.exp(x)
    tl.store(Y + row * n_cols + cols, num / tl.sum(num, axis=0),
             mask=cols < n_cols)                  # registers → HBM, once
# Eager PyTorch = 3–4 separate kernels, each round-tripping HBM.
# This = 1 kernel, 1 read, 1 write. That's kernel fusion in miniature.
```

## 8.3 CuTe-DSL

CUTLASS's tensor/layout algebra exposed as a **Python DSL** (CuTe-DSL, part of NVIDIA's "CUTLASS 4.x / Python-first" push — the deck notes NVIDIA's SC25 stance: everything programmable from Python). It gives explicit control over layouts, TMA copies, and warp specialization that Triton abstracts away — which is exactly why **FlashAttention-4 is written in CuTe-DSL**: Blackwell's TMEM/TMA pipelines need tile-level control Triton doesn't yet expose ([context](https://how.nz/2026/03/22/cutedsl/)). Rule of thumb in 2026: Triton for 95% of custom ops; CuTe-DSL/CUTLASS for the last-few-percent flagship kernels; raw CUDA mostly for glue and exotic cases.

## 8.4 Kernel fusion

Merge chains of memory-bound ops (e.g., RMSNorm → matmul → activation → residual) into one kernel so intermediates stay in registers/SMEM instead of round-tripping HBM. Sources of fusion: hand-written (FlashAttention *is* a giant fusion of matmul+softmax+matmul), compiler-driven (`torch.compile`/Inductor, nvFuser), or library (FasterTransformer-style fused MLP+bias+activation). For bandwidth-bound decode, fusion is frequently worth more than faster matmuls.

# 9. Architectural Levers: GQA, MLA, Sliding-Window, Linear Attention

*(Deck slides 28–32)*

These change the *model architecture* (usually at training time) to cut attention compute and, above all, **KV-cache size** — revisit the formula in §3: they attack `num_kv_heads`, `head_dim`, or `seq_len`.

## 9.1 MQA → GQA

- **Multi-Query Attention (MQA)**: all query heads share ONE K/V head → KV cache ÷ num_heads, but quality drops.
- **Grouped-Query Attention (GQA)**: groups of query heads share a K/V head (e.g., 32 Q heads, 8 KV heads → cache ÷ 4). Near-MHA quality, big cache/bandwidth win. Standard in Llama-2/3, Qwen, Mistral, Gemma.

```javascript
MHA (8 heads):        GQA (4:1 groups):        MQA:              MLA:
Q1 Q2 … Q8            Q1..Q4    Q5..Q8         Q1 …… Q8          Q1 …… Q8
│  │     │              ╲╲╲│      ╲╲╲│           ╲╲╲╲│              │
K1V1 … K8V8              K1V1      K2V2           K1V1        c_kv (512-d latent)
cache: 8×d_h          cache: 2×d_h            cache: 1×d_h    cache: 512 total
                                                              (up-project on use)
```

## 9.2 Multi-Head Latent Attention (MLA, DeepSeek-V2/V3)

Instead of caching per-head K/V, project the hidden state down to a **shared low-rank latent vector** (e.g., 512-dim vs 4096-dim hidden), cache *only the latent*, and reconstruct per-head K/V via up-projections when needed. A decoupled RoPE branch handles positions (RoPE doesn't commute with the low-rank trick).

```python
# MLA shapes (simplified; RoPE branch omitted)
c_kv = x @ W_down            # (seq, 512)  ← the ONLY thing you cache
k    = c_kv @ W_uk           # (seq, n_heads·d_k)  reconstructed when needed
v    = c_kv @ W_uv           # (seq, n_heads·d_v)
# Inference trick: absorb W_uk into W_q and W_uv into W_o,
# so attention runs directly against the 512-d latents — no reconstruction.
```

Deck's concrete arithmetic (DeepSeek parameters): KV-projection compute drops to **3/8** (from `S·h·n_heads·(d_k+d_q)` ≈ 8.39M·S to `S·h·r + S·r·n_heads·(d_k+d_q)` ≈ 3.15M·S with r=512) and KV cache to **1/8** (4096·F·S → 512·F·S bytes) — i.e., 8× longer context in the same memory. Published numbers: MLA needs ~4–14% of MHA's KV cache; DeepSeek-V3 stores ~70 KB/token vs 192–328 KB/token for comparable GQA models ([explainer](https://planetbanatt.net/articles/mla.html), [DeepSeek-V3 report](https://arxiv.org/pdf/2412.19437)). A neat extra: the up-projections can be **absorbed** into W_Q/W_O at inference, so you attend directly in latent space. MLA even outperforms MHA in quality in DeepSeek's ablations — a rare free lunch. TransMLA and similar work convert existing GQA checkpoints to MLA post-hoc.

## 9.3 Sliding-Window Attention (SWA)

Some layers attend only to the last W tokens (window). KV cache for those layers is **capped at W** — a fixed ceiling instead of growing with seq_len. With L layers stacked, the receptive field still spans ~L·W tokens. Mistral popularized it; **Gemma-2/3 interleave sliding layers with full-attention layers** (e.g., 5:1 in Gemma-3, as the deck notes "Gemma is basically this architecture"), and gpt-oss uses alternating dense/banded layers. Related trick: **attention sinks** (keep the first tokens always visible) stabilize long-window generation (§3's StreamingLLM).

```javascript
Sliding-window attention (window W=4):        Hybrid layer stack (Qwen3-Next):
t9 attends to: [t6 t7 t8 t9]  ← only          L1  Gated DeltaNet (linear)
t5 attends to: [t2 t3 t4 t5]                  L2  Gated DeltaNet (linear)
KV cache per SWA layer: capped at W           L3  Gated DeltaNet (linear)
Receptive field ≈ W × depth (stacking)        L4  Full attention  ← retrieval
                                              ... repeat 3:1 ...
```

## 9.4 Linear Attention & Hybrids

Replace softmax attention with a kernelized/recurrent form whose state is a **fixed-size matrix** updated per token: O(1) memory and O(n) time — no KV cache at all for those layers. Modern instantiations are much better than 2020-era linear attention: **Mamba-2 (SSMs), GLA, RetNet, DeltaNet, Gated DeltaNet** (delta-rule error-correcting state update + exponential gating for adaptive forgetting + short conv + L2-normed Q/K).

```python
# Softmax attention decode:  cost grows with n  (reads whole KV cache)
# Linear attention decode:   O(1) — a fixed-size state, no KV cache
S = torch.zeros(d_k, d_v)                    # the entire "memory"
for x_t in tokens:
    q, k, v = proj(x_t)
    S = alpha_t * S + beta_t * k.outer(v - k @ S)  # gated delta rule:
    y_t = q @ S                                    # decay + error-correcting write
```

Pure linear attention still loses on precise retrieval, so 2025–26 flagships are **hybrids**: **Qwen3-Next / Qwen3.5 use Gated DeltaNet in ~3 of every 4 layers with full (gated) attention in the rest** ([vLLM blog](https://vllm.ai/blog/2025-09-11-qwen3-next), [Labonne's Qwen3.5 analysis](https://huggingface.co/blog/mlabonne/qwen35), [Raschka's DeltaNet material](https://sebastianraschka.com/llms-from-scratch/ch04/08_deltanet/)); MiniMax-M1 (Lightning Attention), Kimi (KDA), and Granite-4 (Mamba-2 hybrid) follow the same recipe. The deck's implementation pointer is the [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) library (fused recurrent Triton kernels).

Deck's sharp observation: because these layers keep only a fused running state (no per-token cache), **you can't cheaply roll back tokens** — which conflicts with speculative decoding's reject-and-truncate and with prefix caching. Engines handle it by checkpointing state at intervals; it's a real systems cost of linear attention.

# 10. Quantization

*(Deck slides 33–35)*

Fewer bits per parameter → less memory capacity AND less memory traffic (recall Fact 2: decode reads all weights every step, so **weight bits ≈ decode latency**). 7B model: FP16 = 14 GB, INT4 ≈ 3.9 GB and ~3–4× faster decode ceilings.

**Format landscape 2026:**

- **INT8 / FP8 (E4M3/E5M2)**: workhorse for weights+activations+KV cache; Hopper/Ada tensor cores run FP8 natively at 2× FP16 throughput. DeepSeek-V3 trained in FP8 with fine-grained scaling.
- **INT4 weight-only**: GPTQ (Hessian-based error-compensating rounding), AWQ (protect the ~1% activation-outlier-aligned channels by scaling). Great for single-user/local; dequant overhead can hurt at large batch.
- **FP4 microscaling** (Blackwell native): **MXFP4** = E2M1 values, blocks of 32, power-of-two E8M0 scale; **NVFP4** = blocks of 16, FP8-E4M3 scale + per-tensor FP32 scale → lower quantization error, ~2–3× FP8 throughput and near-FP8 accuracy with quantization-aware recipes ([format comparison](https://www.spheron.network/blog/nvfp4-vs-mxfp4-gpu-cloud-4bit-quantization-guide/), [TensorRT-LLM quantization docs](https://nvidia.github.io/TensorRT-LLM/1.2.0rc5/features/quantization.html)). The deck's example: **gpt-oss ships its MoE expert weights in MXFP4** — experts are >90% of parameters, i.e., exactly the memory bottleneck worth quantizing hardest.

```javascript
Bits per weight:
FP16  |s eeeee mmmmmmmmmm|                    16 bits
FP8   |s eeee mmm|  (E4M3)                     8 bits
NVFP4 |s ee m|  (E2M1) + FP8 scale per 16     ~4.5 bits effective
MXFP4 |s ee m|  (E2M1) + E8M0 scale per 32    ~4.25 bits effective
```

```python
# Block-wise (micro-scaling) quantization, NVFP4-style
for block in W.reshape(-1, 16):               # 16-value micro-blocks
    s = block.abs().max() / FP4_MAX           # per-block scale → stored as FP8
    q = torch.round(block / s).clamp(-6, 6)   # E2M1 representable values
# dequant: w ≈ q * s * per_tensor_scale
# → an outlier only poisons its OWN 16 values, not the whole channel/tensor
```

**Techniques for preserving accuracy** (deck's two bullets, expanded):

- **Block-wise scaling**: one scale per small block (16–128 values) instead of per tensor/channel → outliers only poison their own block. This is the essence of MX/NV microscaling formats and of DeepSeek's FP8 recipe (128×128 weight blocks, 1×128 activation tiles).
- **Mixed-precision accumulation**: multiply in low precision, **accumulate in FP16/FP32**, because dot-product error compounds over thousands of terms. Deck cites DeepSeek-R1's FP8 GEMM: FP8 tensor cores accumulate in limited precision, so DeepSeek periodically promotes partial sums to FP32 CUDA-core registers every few WGMMA tiles.
- Also in the toolbox: rotation-based outlier smoothing (QuaRot/SpinQuant, Hadamard transforms — also used in MR-GPTQ for FP4), SmoothQuant (migrate activation outliers into weights), and quantization-aware distillation for NVFP4 ([NVIDIA report](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf)).

**The Ozaki Scheme** (deck's HPC gem, slide 35): the *inverse* of quantization — use low-precision tensor cores to compute **higher-precision** results. It performs an error-free split of each matrix into S low-precision slices; the product becomes a sum of slice-products each computed exactly on FP16/INT8/FP8 tensor cores, then recombined → full FP64 accuracy *faster than native FP64 units* (e.g., INT8-tensor-core DGEMM >4× cuBLAS FP64 on an RTX A6000). Now integrated into **cuBLAS as automatic FP64 emulation** (ADP framework, Ozaki-II in development) ([NVIDIA blog](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/), [HPCwire](https://www.hpcwire.com/2025/04/17/have-you-heard-about-the-ozaki-scheme-you-will/), [FP8 DGEMM paper](https://arxiv.org/html/2508.00441v2)). Strategic reason it matters: GPU vendors are removing FP64 silicon (Blackwell Ultra cut FP64 to 1/64th rate) and betting on emulation — the AI/HPC precision worlds are converging on "low-precision units + clever numerics."

# 11. Profiling & CUDA Graphs

*(Deck slides 36–40)*

**Rule zero: never optimize without profiling.**

- **Nsight Systems** (`nsys`): whole-system timeline — CPU threads, CUDA API calls, kernel executions, memcpys, NVLink. This is where you find *gaps* (GPU idle waiting for CPU), serialization, and sync stalls. Annotate code regions with **NVTX** ranges (`torch.cuda.nvtx.range_push/pop`) so timeline segments map to your Python code. `torch.profiler` is a lighter first step.
- **Nsight Compute** (`ncu`): deep-dive on ONE kernel — achieved occupancy, memory-throughput vs compute-throughput (its Speed-of-Light section is a per-kernel roofline), stall reasons (memory latency? SM saturation?), SMEM bank conflicts. Use it after `nsys` tells you which kernel matters.

**The naive-inference smoking gun** (deck slides 38–40): profiling eager PyTorch decode shows ~100 µs gaps between kernels — the GPU finishes work faster than the CPU/Python can launch the next kernel. For small models this launch-bound regime is **the single biggest bottleneck**, bigger than any kernel inefficiency.

**CUDA Graphs**: record the whole decode step's kernel-launch sequence ONCE as a graph, then replay the entire graph with a single API call — launch overhead amortized to near zero, and the GPU can prefetch the whole dependency chain. Constraints: shapes/addresses must be static → engines capture graphs for a set of padded batch sizes and use fixed memory pools (this is vLLM's "CUDA graph capture" phase at startup and a large chunk of its small-model speedup; `torch.compile(mode="reduce-overhead")` does the same). Trade-off: capture time + extra reserved memory; dynamic control flow can't be captured (conditional nodes are improving in recent CUDA versions; see also [hybrid JIT + CUDA-graph work](https://arxiv.org/pdf/2604.23467)).

```python
# NVTX: make nsys timelines map to YOUR code
torch.cuda.nvtx.range_push("decode_step")
logits = model(ids)
torch.cuda.nvtx.range_pop()
# then:  nsys profile -t cuda,nvtx python infer.py

# CUDA Graphs: capture once, replay forever (static shapes/addresses!)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_out = model(static_input)        # records ALL kernel launches
for _ in range(n_steps):
    static_input.copy_(next_ids)            # write into the SAME buffer
    g.replay()                              # 1 API call instead of 100s of launches
# torch.compile(model, mode="reduce-overhead") does this for you.
```

```javascript
Before (eager):   CPU  ─l─  ─l─  ─l─  ─l─      GPU  ▓ .. ▓ .. ▓ .. ▓   (gaps)
After  (graph):   CPU  ─replay─────────        GPU  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   (packed)
```

# 12. vLLM: Putting It All Together

*(Deck slides 41–47)*

vLLM = an open-source inference engine that packages everything above: continuous batching + PagedAttention + FlashAttention + CUDA graphs + quantized kernels + speculative decoding + parallelism. (Peers: **SGLang** — RadixAttention prefix caching, very strong on structured/agentic workloads; **TensorRT-LLM** — NVIDIA's compiled engine, top raw perf; **llama.cpp** — CPU/edge.)

## 12.1 Architecture (V1)

Two-process design: **Frontend** (API server, tokenization, OpenAI-compatible endpoints) ↔ **EngineCore** running the busy loop: **Scheduler** (decides which requests and how many tokens each step — the token-budget abstraction naturally expresses chunked prefill and spec-decode) + **KVCacheManager** (the paged block allocator, block tables, prefix-cache structures) + **ModelExecutor/Worker(s)** (own the GPUs, run the fused kernels, capture CUDA graphs). Best deep read: [Aleksa Gordić — Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm); official docs: [docs.vllm.ai](http://docs.vllm.ai/).

```javascript
client ──► Frontend (HTTP/OpenAI API, tokenize, detokenize/stream)
                │ IPC
                ▼
           EngineCore busy loop:
           ┌─ Scheduler ────────── decides WHO runs & HOW MANY tokens
           │    (token budget: decodes first, then chunked-prefill fill-in)
           ├─ KVCacheManager ─────── paged blocks, block tables, prefix cache
           └─ ModelExecutor ──────── GPU worker(s): super-sequence forward,
                                     CUDA-graph replay, sampling
           step(): schedule → forward → sample → stream results → repeat
```

```python
# Same model as §2, via vLLM — the deck's 15.9× "after" picture
from vllm import LLM, SamplingParams
llm = LLM("Qwen/Qwen2.5-0.5B-Instruct",
          gpu_memory_utilization=0.9,      # pre-allocates the paged KV pool
          enable_prefix_caching=True)
outs = llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=256))
# → ~95 tok/s where transformers gave ~6
```

## 12.2 Chunked Prefill

```javascript
Without chunking:  step k:   [██████ 32k-token prefill ██████]  ← decodes stall
                   step k+1: [d][d][d][d]                        (ITL spike)
With chunking:     step k:   [d][d][d][d][── prefill chunk 1 ──]
                   step k+1: [d][d][d][d][── prefill chunk 2 ──] ← smooth ITL
```

Problem: a 32k-token prefill entering the batch stalls every decode in it (ITL spike). Fix: split prefills into fixed-size chunks (V1 default budget ~8k tokens) and co-schedule chunks WITH decodes each step; decode latency stays smooth and prefill's compute fills the bandwidth-bound decode steps' idle FLOPs. ([vLLM optimization docs](https://docs.vllm.ai/en/stable/configuration/optimization/).) The deck notes this exists thanks to the super-sequence representation treating prefill and decode tokens uniformly.

## 12.3 Preemption

When the block pool runs dry mid-flight (decodes keep growing), the scheduler **preempts** newest-first: either **recompute** the victim's prefill later (cheap for short contexts, default) or **swap/offload** its KV blocks to CPU RAM and stream them back (better for long contexts, costs PCIe traffic). Production tip: watch vLLM's preemption-rate metric; frequent preemption = raise `gpu_memory_utilization`, shrink `max_model_len`, or add GPUs.

## 12.4 Prefill–Decode Disaggregation

```javascript
             ┌── Prefill pool (compute-heavy GPUs, e.g. H200) ──┐
requests ──► │  builds KV cache                                 │
             └───────────────┬────────────────────────────────┘
                             │  KV-cache transfer (NVLink / RDMA)
                             ▼
             ┌── Decode pool (bandwidth-adequate GPUs, e.g. A100) ──► tokens
             │  optimizes ITL independently of TTFT
             └────────────────────────────────────────────────┐
```

Prefill is compute-bound; decode is bandwidth-bound (Fact 2/3). Mixing them on one GPU means each interferes with the other's SLO (TTFT vs ITL). Disaggregation runs **separate prefill and decode server pools** and ships the KV cache between them (NVLink/RDMA; connector abstraction in vLLM — [docs](https://docs.vllm.ai/en/stable/features/disagg_prefill/)). Benefits: independent scaling and *heterogeneous hardware* — the deck's example: prefill on tensor-core-heavy H200s, decode on cheaper A100s. This is how frontier labs serve at scale ([Mooncake, Moonshot's KVCache-centric architecture](https://arxiv.org/pdf/2407.00079); NVIDIA Dynamo; [DistServe](https://arxiv.org/abs/2401.09670)). Chunked prefill vs disaggregation is the central serving design choice: co-locate and interleave, or separate and pay KV-transfer.

## 12.5 Contributing (deck's practical tips)

vLLM is hack-friendly: documented source builds, clear architecture docs, and a **plugin system** for experiments (custom models/schedulers/attention backends without forking). Debug tip from the deck: set `VLLM_ENABLE_V1_MULTIPROCESSING=0` to force in-process mode so a debugger can step through the engine (otherwise EngineCore lives in a separate process).

## 12.6 The payoff

Re-run the deck's Colab: 5.97 → **94.84 tok/s (15.9×)**. Accounting: CUDA graphs kill launch gaps (biggest factor for a 0.5B model), continuous batching + paged KV raise utilization, FlashAttention/fused kernels cut memory traffic.

# 13. Multi-Node & Multi-GPU Parallelism (Concepts)

*(Deck slide 50)*

When the model or traffic outgrows one GPU ([Meta's production overview](https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/), [efficient-serving survey](https://arxiv.org/pdf/2504.19720)):

- **Communication substrate**: MPI-style collectives (all-reduce, all-gather, all-to-all) implemented by **NCCL** over NVLink (intra-node, ~900 GB/s) and InfiniBand/RoCE (inter-node). Know your collective costs; they set parallelism strategy.
- **Data Parallel (DP)**: replicate the whole engine; route requests. No intra-model comm; linear throughput; doesn't help single-request latency or fit bigger models.
- **Tensor Parallel (TP)**: split each weight matrix across GPUs (column-split then row-split → one all-reduce per attention block and per MLP). Cuts per-GPU memory AND latency, but needs all-reduce **every layer** → keep TP within an NVLink island (typically ≤8).
- **Pipeline Parallel (PP)**: split by layers into stages; requests flow through. Cheap point-to-point comm, crosses nodes well, but adds pipeline latency and bubbles; needs micro-batching to fill.
- **Context/Sequence Parallel (CP)**: split the *sequence* across GPUs for very long prompts (ring attention / all-gather KV variants) — attacks the O(n²) prefill of 100k+ contexts.
- **Expert Parallel (EP)**: for MoE, scatter experts across GPUs; tokens routed via all-to-all. DeepSeek-V3-scale serving uses wide EP with careful load balancing.

```javascript
Tensor Parallel (every layer):        Pipeline Parallel (by layers):
GPU0: W[:, :d/2] ─┐                   GPU0: layers 0–23
GPU1: W[:, d/2:] ─┴─ all-reduce       GPU1: layers 24–47   ─► point-to-point
      per attention & MLP block       GPU2: layers 48–71      (micro-batched
      → needs NVLink                  GPU3: layers 72–95       to hide bubbles)

Context Parallel (by sequence):       Expert Parallel (MoE):
GPU0: tokens 0–32k   ─┐ ring/         GPU0: experts 0–15  ─┐ all-to-all
GPU1: tokens 32k–64k ─┘ all-gather    GPU1: experts 16–31 ─┘ token routing
      attention over the seam               router decides per token
```

Real deployments compose these, e.g. TP=8 within nodes × PP across nodes × EP for experts × DP for throughput, plus P/D disaggregation on top.

# 14. NVIDIA GPU Hardware Essentials

*(Deck slide 50)*

- **SM (Streaming Multiprocessor)**: the "core" unit; each has tensor cores, CUDA cores, registers, and up to ~228 KB shared memory/L1. Kernels run as grids of thread blocks scheduled onto SMs; **occupancy** = how full they are.
- **Tensor Cores**: matrix-multiply-accumulate units (MMA/WGMMA instructions) delivering the vast majority of FLOPs; each generation adds lower precisions (V100 FP16 → A100 TF32/BF16 → H100 FP8 → B200 FP4). If your kernel isn't feeding tensor cores, you're leaving ~90% of the GPU idle.
- **Memory hierarchy**: registers → shared memory/L1 (per-SM, ~19+ TB/s aggregate) → L2 (tens of MB) → HBM3(e) (80–192 GB, 3.35–8 TB/s). Also **TMEM** (tensor memory) on Blackwell for accumulators.
- **TMA (Tensor Memory Accelerator, Hopper+)**: a DMA engine that copies multi-dimensional tiles HBM↔SMEM **asynchronously**, freeing warps from address arithmetic — the enabler of FA-3/FA-4's producer-consumer pipelines.
- **Compute Capability** (sm_80 = A100, sm_90 = H100, sm_100/sm_120 = Blackwell): gates which instructions/features a kernel can use; why binaries ship fat with multiple architectures and why "works on my 4090" ≠ "works on H100".
- Interconnect: **NVLink/NVSwitch** for GPU↔GPU; PCIe for host; **GH200/GB200** superchips fuse CPU+GPU memory coherently (the presenter's research theme — unified memory for efficient computation).

```javascript
FA-3/FA-4 producer–consumer pipeline on one SM:

 HBM ══ TMA async tile copy ══► SMEM ──► Tensor Core (MMA) ──► TMEM/regs
        (producer warps issue          (consumer warps compute    │
         copies, don't compute)         matmul + online softmax)  ▼
 next tiles prefetched WHILE current tiles compute ──► overlap ≈ no stalls
```

# 15. Beyond the Deck: Other Current Speedup Directions (2025–2026)

Compact map of what else a working expert should track:

- **Prefix caching / RadixAttention everywhere** (§6) — for agents that resend growing histories, this dwarfs kernel-level wins.
- **KV-cache eviction/quantization** (H2O, SnapKV, StreamingLLM sinks, FP8 KV) (§3).
- **Sparse attention at scale**: DeepSeek **NSA** (natively-trained sparse attention) and Kimi **MoBA** — trainable block-sparse patterns giving near-full-attention quality with large long-context speedups; hybrid sparse + KV-sharing designs ([HySparse](https://arxiv.org/pdf/2602.03560)).
- **torch.compile / compiled graphs as default** — vLLM V1 integrates `torch.compile`; the handwritten-kernel surface keeps shrinking.
- **Overlap scheduling**: hide CPU scheduling under GPU compute (SGLang's zero-overhead scheduler, vLLM async scheduling).
- **Diffusion LLMs & parallel text generation** (research): generate many tokens per step by de-noising ([Gumbel distillation](https://arxiv.org/pdf/2603.22216), Mercury-style dLLMs) — a possible long-term end-run around autoregression.
- **Serving-layer economics**: request routing, multi-SLO scheduling ([PolyServe](https://arxiv.org/pdf/2507.17769)), KV-aware load balancing (Mooncake), speculative flows in disaggregated clusters ([StreamServe](https://arxiv.org/pdf/2604.09562)).

# 16. Accelerating Video Captioning / Video-LLM Inference

*(Not in the deck — added per request.)*

## 16.1 Why video is brutally expensive

A video-LLM = vision encoder (per frame) → projector → LLM decoder. The LLM sees **visual tokens**: ~200–700 per frame typical; at 1 fps a 5-minute video is 300 frames ≈ 100k+ visual tokens *before the text prompt*. Costs explode in three places: (a) O(n²) prefill attention over visual tokens, (b) KV cache holding all of them during decode, (c) the vision encoder itself for many frames. Captioning specifically = long multimodal **prefill** + short text **decode**, so prefill/token-count optimizations dominate. Crucially, **everything in §1–§15 still applies** (the LLM half is just an LLM) — video adds one new, extremely effective lever: **most visual tokens are redundant** (static backgrounds, near-duplicate frames).

```javascript
The video-LLM pipeline, with every attack point marked:

video ─► frame sampler ─► ViT encoder ─► projector ─► [visual tokens] ─► LLM
         (§16.2: fps,      (per frame;    (§16.4:       ~200–700/frame    │
          keyframes,        batch it,      resampler     × 300 frames     │
          clip select)      quantize it)   compresses)   = 100k+ tokens   │
                                                              ▼           ▼
                                            (§16.3: prune/merge     prefill (§12.2
                                             inside early LLM        chunked) →
                                             layers, 4–20×)          decode caption
                                                                     (§7 spec-dec)
```

## 16.2 Frame/clip selection (cheapest lever, works with any model)

- Uniform low-fps sampling is the baseline; smarter: drop frames whose features are near-duplicates of neighbors (feature-similarity dedup retains ~46% of frames at no quality loss on procedural video).
- **Keyframe selection**: [FOCUS](https://arxiv.org/pdf/2510.27280) (training-free, confidence-bound/bandit-based selection for hour-long videos), KFFocus (allocate more tokens to informative moments), [key ](https://arxiv.org/html/2510.02262v1)[*clip*](https://arxiv.org/html/2510.02262v1)[ selection](https://arxiv.org/html/2510.02262v1) (keep short contiguous clips instead of isolated frames to preserve motion), audio-guided keyframe selection for description generation, DPP-based diverse frame sampling ([LDDR](https://arxiv.org/pdf/2605.11477)).
- Query-aware selection (retrieve frames relevant to the question) helps QA more than open-ended captioning; for captioning, coverage + diversity criteria matter more.

## 16.3 Visual token pruning/merging inside the model (training-free; the big 2024–26 wave)

- **FastV** (ECCV 2024): after ~layer 2, rank visual tokens by attention received and drop ~50% → ~45% FLOPs cut, negligible loss. Generic to any LVLM.
- **ToMe / VisionZip / PruMerge**: merge or select tokens by similarity/importance at the encoder output.
- Video-specific (exploit *temporal* redundancy): [**PruneVid**](https://arxiv.org/pdf/2412.16117) (ACL 2025 — merge static spatio-temporal regions, cluster spatial tokens, then keep only question-relevant tokens via LLM attention: prunes >80% of video tokens while maintaining performance), **DyCoke** (dynamic compression per decoding step), **TempMe**, **FastVID** (segment-then-merge), **MMG-Vid**, [**OTT-Vid**](https://arxiv.org/pdf/2605.11803) (optimal-transport temporal compression), [InfoMerge](https://arxiv.org/html/2606.02161), ultra-low-retention unified spatiotemporal compression ([USTC](https://arxiv.org/pdf/2603.21957)). Curated list: [Awesome-Token-Compress](https://github.com/daixiangzi/Awesome-Token-Compress).
- Rules of thumb: pruning *inside early LLM layers* (attention-guided) beats encoder-side pruning for question-conditioned tasks; merging beats dropping when captions must mention background detail; 4–8× token reduction is typically free, 10–20× costs a few points.

```python
# FastV-style attention-guided visual token pruning (training-free)
# After layer L (e.g. L=2), inside the LLM forward:
attn_recv = attn_weights[:, :, -1, visual_range].mean(dim=1)  # attention each
keep      = attn_recv.topk(int(keep_ratio * n_visual)).indices # visual token gets
hidden    = hidden[:, cat(text_idx, visual_idx[keep])]         # drop the rest
# All subsequent layers (and the KV cache) see ~50% fewer tokens.
# Video variants (PruneVid etc.) first merge tokens that are static ACROSS frames.
```

```javascript
Temporal redundancy, visualized (why video pruning is so effective):
frame 1: [sky][tree][road][car ]     sky/tree/road identical across frames
frame 2: [sky][tree][road][car→]  →  merge to 1 token each (temporal merge),
frame 3: [sky][tree][road][car→→]    keep per-frame tokens only for the car
         12 tokens → ~6 tokens with zero information loss
```

## 16.4 Architectural compression

Token-compressing projectors/resamplers (Q-Former/Perceiver-style queries, C-Abstractor pooling), **VoCo-LLaMA** (distill frames into a few "compressed" tokens via the LLM itself), memory-based designs (MovieChat, MA-LMM) that maintain a fixed-size memory bank instead of unbounded visual context — the video analogue of linear attention's fixed state (§9.4).

## 16.5 Streaming / real-time video captioning (2025–26 frontier)

For live captioning, redesign the pipeline, not just the tokens: **TimeChat-Online / VideoScan / CodecSight** drop redundant incoming frames, route vision compute, or read codec motion signals *before* dense inference; **ProVideLLM** interleaves compact verbalized text tokens for long-past context with visual tokens for recent frames → >10 fps streaming understanding in ~2 GB; [**ViCoStream**](https://arxiv.org/pdf/2606.19849) coordinates encoder/LLM stages to exceed 100 fps. Survey: [Efficient Video Intelligence in 2026](https://v-chandra.github.io/efficient-video-intelligence/).

## 16.6 Systems-level tricks specific to video serving

Batch frame encoding and overlap the vision encoder with LLM prefill (they're separate models — pipeline them); cache/reuse the system-prompt + instruction prefix (§6 prefix caching — the video tokens change, the template doesn't); for repeated queries on the same video, cache the video's KV prefix; quantize the vision encoder too (often left FP16 out of neglect); use chunked prefill so a 100k-token video prefill doesn't stall co-batched decodes (§12.2); consider P/D disaggregation since video workloads are extremely prefill-heavy (§12.4).

**Practical recipe for fast video captioning in 2026**: fps-adaptive sampling + keyframe/clip selection (§16.2) → token merging ~4–8× (PruneVid/FastVID-style, §16.3) → serve with vLLM/SGLang (continuous batching, prefix caching, chunked prefill, FP8) → EAGLE-3/MTP speculative decoding for the caption decode. Each layer multiplies; 10–30× end-to-end vs. a naive HF pipeline is realistic.

# 17. Suggested Learning Path & Canonical Resources

11. **Foundations**: Karpathy's "Let's build GPT" / nanoGPT; then re-derive the KV-cache size formula yourself (§3).
12. **Serving concepts**: [Aleksa Gordić's vLLM anatomy post](https://www.aleksagordic.com/blog/vllm); vLLM paper (PagedAttention); Orca paper (continuous batching); the deck's inspiration article "LLM Servingを支える技術".
13. **Kernels**: [Lei Mao's CUDA GEMM optimization](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/); FlashAttention 1–3 papers; write one Triton kernel (fused softmax) yourself; GPU-MODE lectures/Discord.
14. **Architecture**: GQA paper; DeepSeek-V2/V3 reports (MLA, FP8, MTP); Qwen3-Next blog (hybrid linear attention); Raschka's "Beyond Standard LLMs".
15. **Quantization**: GPTQ/AWQ papers; NVIDIA NVFP4 blogs; DeepSeek-V3 FP8 training section.
16. **Profiling**: profile HF `generate()` with Nsight Systems + NVTX yourself; find the 100 µs gaps; then capture a CUDA graph and watch them vanish. Nothing teaches this better.
17. **Video-LLM efficiency**: FastV → PruneVid → FOCUS → ViCoStream papers, in that order (spatial pruning → temporal pruning → frame selection → streaming systems).
18. **Practice**: reproduce the deck's Colab experiment (Qwen2.5-0.5B, Transformers vs vLLM); then contribute a small fix/plugin to vLLM (`VLLM_ENABLE_V1_MULTIPROCESSING=0` for debugging).

# Appendix: One-Screen Summary (mirrors deck slides 7/48)

| Layer | Technique | What it saves |
| --- | --- | --- |
| Attention math | KV cache | Recompute → O(n) decode |
| Attention kernel | FlashAttention (tiling + online softmax) | HBM traffic |
| Attention memory | PagedAttention (+prefix caching) | Fragmentation; reuse |
| Architecture | GQA / MLA / SWA / linear-hybrid | KV size & bandwidth |
| Token generation | Speculative decoding (EAGLE-3, MTP) | Sequential forwards |
| Numerics | Quantization (FP8/NVFP4, block scales, mixed-prec accumulation) | Memory capacity & traffic |
| CPU–GPU | CUDA Graphs | Launch overhead |
| Scheduling | Super sequence + continuous batching, chunked prefill, preemption | Padding, idle GPU |
| Cluster | TP/PP/CP/EP + P/D disaggregation | Fit & SLOs |
| Video (bonus) | Frame selection + token pruning/merging + streaming pipelines | Visual-token count |

**Primary basis**: [LLM高速化勉強会資料 (SuperHotDog, Speaker Deck)](https://speakerdeck.com/superhotdogcat/llmgao-su-hua-mian-qiang-hui-zi-liao); all other sources linked inline.