---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-13T18:37:00
Status: Done
Last updated time: 2026-07-13T18:44:00
Last edited by: Heaven Chen
Category:
  - LLM
  - Multimodal
  - ML
  - Survey
---
> **Qwen Team, “Qwen3.5: Towards Native Multimodal Agents,” February 2026.** Official announcement: [Qwen blog](https://qwen.ai/blog?id=qwen3.5) · weights/configs: [Hugging Face](https://huggingface.co/Qwen/models) · implementation/release history: [QwenLM/Qwen3.6](https://github.com/QwenLM/Qwen3.6). Qwen3.5 is a family of causal vision-language models that combines a **3:1 Gated DeltaNet/full-attention stack**, dense or sparse-MoE feed-forward blocks, native image/video understanding, long context, reasoning, and agent/tool use.

> [!info] Scope and evidence
> Snapshot as of **2026-07-13**. This note covers the general Qwen3.5 text–image–video family, not the separate audio/speech-focused **Qwen3.5-Omni** branch. A full general-purpose Qwen3.5 technical report had still not been published; the official blog says it is forthcoming. Architecture details below are therefore triangulated from the official blog, released model cards/configs, the Qwen3-Next design note, and the primary Gated DeltaNet / Gated Attention papers. Benchmark numbers are first-party unless stated otherwise.

# 0. The one-paragraph story

Qwen3.5 is where three previously separate Qwen lines converge. From **Qwen3**, it keeps one post-trained model capable of deliberate reasoning, direct answers, coding, and tool calls. From **Qwen3-VL**, it takes a vision encoder and multimodal positional scheme. From **Qwen3-Next**, it inherits the systems-oriented core: three recurrent **Gated DeltaNet** layers for every one global softmax-attention layer, output-gated attention, partial RoPE, high-sparsity MoE, stability changes, and multi-token prediction. It then trains text, images, and video together from pretraining rather than attaching vision only after a text model is finished. The result is an Apache-2.0 open-weight ladder from **0.8B to 397B total parameters**, all with a native 262,144-token context, while the 4B-and-larger cards document YaRN extension to about 1.01M tokens.

The central engineering trade is:

> **Compress most history into constant-size recurrent states for speed, but retain full attention every fourth layer for exact token-to-token retrieval. Scale capacity with MoE without activating every expert, and treat visual embeddings as ordinary sequence tokens so the same reasoning/agent stack works over text, images, and video.**

# 1. Lineage, release timeline, and scope boundaries

## 1.1 What each ancestor contributed

| Line | Contribution inherited by Qwen3.5 |
| --- | --- |
| **Qwen3** | Dense + MoE model families, reasoning and direct-response behavior, multilingual/coding/agent post-training |
| **Qwen3-VL** | ViT-style vision tower, dynamic image/video tokenization, temporal-height-width multimodal RoPE |
| **Qwen3-Next** | 3:1 Gated DeltaNet + Gated Attention hybrid, ultra-sparse MoE, zero-centered RMSNorm, partial RoPE, multi-token prediction |
| **Qwen3.5** | Early text–vision fusion in pretraining, much broader visual/video and RL environments, 201 languages/dialects, one general multimodal-agent family |

The naming can otherwise be confusing:

- **Qwen3.5** here means the general text–image–video causal model family. It emits text/reasoning/tool-call tokens; it is not an image generator.
- **Qwen3.5-Plus** is the managed Alibaba Cloud counterpart of `Qwen3.5-397B-A17B`, with production additions such as a default 1M window, built-in tools, and adaptive tool use.
- **Qwen3.5-Flash** is the managed counterpart of `Qwen3.5-35B-A3B`, again with hosted-only production features.
- **Qwen3.5-Omni** adds audio understanding and streaming speech generation through a different Thinker/Talker system. Do not infer audio support from a base Qwen3.5 checkpoint.
- **Qwen3.6** arrived in April 2026. Its dense/MoE checkpoints reuse the Qwen3.5 architecture classes, but it is a later post-training generation, not another Qwen3.5 size.

## 1.2 Release timeline

| Date | Release |
| --- | --- |
| **2026-02-15/16** | Qwen3.5 announced; first open weight: `397B-A17B` |
| **2026-02-24** | `122B-A10B`, `35B-A3B`, and dense `27B` |
| **2026-03-02** | Dense `9B`, `4B`, `2B`, and `0.8B` |
| **2026-04-15/22** | Qwen3.6 `35B-A3B` and `27B` successors released |

# 2. Model zoo — what the names actually mean

All eight general checkpoints are causal language models **with a vision encoder**, use a padded vocabulary of **248,320**, repeat the same **3 linear-attention : 1 full-attention** pattern, and were trained with multi-token prediction (MTP).

| Model | FFN type / active compute | LM layers × width | Vision tower | Context | Approx. BF16 checkpoint files¹ |
| --- | --- | ---: | ---: | --- | ---: |
| `Qwen3.5-0.8B` | Dense, 3,584-wide FFN | 24 × 1,024 | 12 × 768 | 262K native | 1.7 GB |
| `Qwen3.5-2B` | Dense, 6,144-wide FFN | 24 × 2,048 | 24 × 1,024 | 262K native | 4.5 GB |
| `Qwen3.5-4B` | Dense, 9,216-wide FFN | 32 × 2,560 | 24 × 1,024 | 262K → 1.01M | 9.3 GB |
| `Qwen3.5-9B` | Dense, 12,288-wide FFN | 32 × 4,096 | 27 × 1,152 | 262K → 1.01M | 19.3 GB |
| `Qwen3.5-27B` | Dense, 17,408-wide FFN | 64 × 5,120 | 27 × 1,152 | 262K → 1.01M | 55.6 GB |
| `Qwen3.5-35B-A3B` | 256 experts; 8 routed + 1 shared; **3B active** | 40 × 2,048 | 27 × 1,152 | 262K → 1.01M | 71.9 GB |
| `Qwen3.5-122B-A10B` | 256 experts; 8 routed + 1 shared; **10B active** | 48 × 3,072 | 27 × 1,152 | 262K → 1.01M | 250.2 GB |
| `Qwen3.5-397B-A17B` | 512 experts; 10 routed + 1 shared; **17B active** | 60 × 4,096 | 27 × 1,152 | 262K → 1.01M | 806.8 GB |

¹Approximate sum of the official `.safetensors` files, decimal GB, including the multimodal checkpoint rather than just the nominal language-model parameters. Runtime also needs cache, activations, kernels, allocator headroom, and possibly duplicated/sharded tensors.

> [!important] “A3B” does not mean “a 3B model on disk”
> In `35B-A3B`, **35B** is resident capacity and **3B** is roughly the parameter subset used for one token. Sparse activation cuts FLOPs, but all 35B expert weights must still live in aggregate GPU/CPU memory unless they are explicitly offloaded. This is why the 35B MoE can compute more like a small model yet still has a ~72 GB BF16 checkpoint.

A first-order weight-memory estimate is

$$
M_{\text{weights}} \approx P\frac{b}{8},
$$

where $P$ is parameter count and $b$ is bits per weight. INT4 therefore gives a theoretical 4× cut versus BF16, but scales/zeros, unquantized modules, the vision tower, and runtime buffers make the real reduction smaller. Official FP8 and GPTQ-Int4 variants exist for the 27B, 35B-A3B, 122B-A10B, and 397B-A17B releases.

# 3. End-to-end architecture

```text
image / video
    │
    ▼
3D patch embedding: temporal 2 × spatial 16 × 16
    │  vision Transformer (12/24/27 blocks)
    ▼
2 × 2 spatial patch merge + projection to LM width
    │
    ├──────── visual embeddings replace image/video placeholder tokens ───────┐
    │                                                                         │
text ──► 248,320-token BPE embeddings ────────────────────────────────────────┤
                                                                              ▼
                         one interleaved causal token stream
                    + modality IDs + temporal/height/width mRoPE
                                      │
             ┌──────────── repeat N/4 times ─────────────┐
             │  Gated DeltaNet → dense FFN / sparse MoE │  ×3
             │  Gated full attention → dense FFN / MoE  │  ×1
             └───────────────────────────────────────────┘
                                      │
                         LM head + optional MTP heads
                                      │
                         text / reasoning / tool calls
```

There is no separate cross-attention decoder that consults frozen image features on the side. The processor marks visual spans with special tokens; the vision tower produces the matching number of embeddings; those embeddings are inserted into the same residual stream as text and processed by the causal backbone. This is why a single tool/reasoning policy can condition on text, screenshots, diagrams, and video frames.

Every decoder layer remains structurally Transformer-like:

$$
h' = h + \mathrm{TokenMixer}(\mathrm{RMSNorm}(h)),\qquad
h'' = h' + \mathrm{FFN/MoE}(\mathrm{RMSNorm}(h')).
$$

The difference is that `TokenMixer` is Gated DeltaNet in 75% of layers and gated softmax attention in 25%.

# 4. Gated DeltaNet — the main efficiency idea

## 4.1 Why not use full attention everywhere?

For a length-$T$ sequence, full attention constructs pairwise token interactions. Prefill work grows quadratically with $T$, and autoregressive decoding retains a KV vector for every prior token. A million-token multimodal stream is therefore expensive even with FlashAttention and GQA.

A simple recurrent linear-attention memory instead accumulates outer products:

$$
S_t = S_{t-1} + v_t k_t^\top,\qquad o_t = S_t q_t.
$$

The entire past is summarized in fixed-size matrix $S_t$; decoding no longer scans an ever-growing KV list. The flaw is **destructive interference**: repeatedly adding key–value associations can superpose or overwrite unrelated memories without explicitly correcting what the state already predicts for the same key.

## 4.2 Delta update + adaptive forgetting

Gated DeltaNet uses the primary paper’s gated delta rule:

$$
S_t = S_{t-1}\Big[\alpha_t\big(I-\beta_t k_tk_t^\top\big)\Big]
      + \beta_t v_tk_t^\top,
$$

with data-dependent $\alpha_t,\beta_t\in(0,1)$. Expanding the update exposes the three operations:

$$
S_t =
\underbrace{\alpha_t S_{t-1}}_{\text{decay / forget}}
-\underbrace{\alpha_t\beta_t(S_{t-1}k_t)k_t^\top}_{\text{erase the old prediction at }k_t}
+\underbrace{\beta_t v_tk_t^\top}_{\text{write the new association}}.
$$

- **$\alpha_t$ — global adaptive decay.** The model can rapidly clear stale state rather than preserve everything forever.
- **$\beta_t$ — edit strength.** The delta term first subtracts the value currently associated with $k_t$, then writes $v_t$. This is analogous to one online-SGD step on $\frac12\lVert S k_t-v_t\rVert^2$.
- **Query read.** $q_t$ interrogates the resulting fast-weight matrix. Qwen’s implementation also normalizes Q/K and precedes the recurrence with a short causal convolution (kernel size 4) for local mixing.

During training/prefill, a chunkwise parallel algorithm turns the recurrence into GPU-friendly matrix multiplications. During one-token decode, a fused recurrent kernel updates the convolution and matrix state directly.

## 4.3 Complexity and the reason for the 3:1 hybrid

| Token mixer | Prefill scaling in sequence length | Decode history access | Growing per-request state | Main weakness |
| --- | --- | --- | --- | --- |
| Full softmax attention | $O(T^2)$ pair interactions | scans cached K/V | $O(T)$ KV cache | long-context cost |
| Gated DeltaNet | $O(T)$ recurrent/chunked scan | fixed-state update/read | $O(1)$ in $T$ | history is compressed, not individually addressable |
| Qwen3.5 hybrid | 3 linear layers + 1 full layer | both mechanisms | KV only in every fourth layer + fixed GDN states | more complex kernels/runtime |

Pure linear attention is fast but can be weaker on exact recall and “needle” retrieval because many tokens are compressed into one state. Pure softmax attention recalls individual positions but pays the long-context bill in every layer. Qwen’s ablations on Qwen3-Next found the **3:1 mixture** stronger than either monolithic choice: GDN handles cheap streaming memory while periodic global-attention layers recover precise content-based lookup.

> [!note] Linear attention does not make all long-context costs constant
> Only the GDN state is constant in $T$. Every fourth layer still performs global attention, retains KV, and has quadratic prefill interactions. The hybrid reduces the coefficient substantially; it does not turn a 1M-token prompt into a free operation.

# 5. The full-attention layer — GQA, output gating, and multimodal RoPE

## 5.1 Grouped-query attention and output gate

Every fourth layer uses causal scaled-dot-product attention with **256-dimensional heads** and far fewer K/V than query heads:

| Family member | Query heads | KV heads | Full-attention layers |
| --- | ---: | ---: | ---: |
| 0.8B / 2B | 8 | 2 | 6 |
| 4B / 9B | 16 | 4 | 8 |
| 27B | 24 | 4 | 16 |
| 35B-A3B | 16 | 2 | 10 |
| 122B-A10B | 32 | 2 | 12 |
| 397B-A17B | 32 | 2 | 15 |

GQA shares each K/V head across a group of query heads, directly shrinking the remaining KV cache. Qwen then applies a learned head-specific sigmoid gate after attention:

$$
y = W_o\left[\sigma(g(x))\odot\mathrm{SDPA}(Q,K,V)\right].
$$

The associated [Gated Attention paper](https://arxiv.org/abs/2505.06708) reports that this simple output gate adds useful nonlinearity, produces query-dependent sparsity, reduces attention sinks/massive activations, improves stability, and helps long-context extrapolation.

## 5.2 Partial multimodal RoPE

Only **64 of each 256-dimensional full-attention head** receive rotary position encoding (`partial_rotary_factor=0.25`). The 64 rotary coordinates form 32 rotation pairs, split by `mrope_section=[11,11,10]` across:

1. temporal position,
2. image/video height,
3. image/video width.

The remaining head dimensions are position-free. This preserves a strong content channel and makes extrapolation less brittle, while interleaved 3D RoPE lets the same attention layer reason about word order and visual space/time. Replacing the rotary implementation without preserving this three-way split misaligns image/video tokens.

## 5.3 KV-cache reality check

For the full-attention portion only, BF16 KV memory for one sequence is approximately

$$
M_{KV}=2_{K,V}\times L_{full}\times n_{kv}\times d_{head}\times T\times 2\text{ bytes}.
$$

At the native $T=262{,}144$ limit:

| Model | BF16 full-attention KV / sequence |
| --- | ---: |
| 0.8B / 2B | ~3 GiB |
| 4B / 9B | ~8 GiB |
| 27B | ~16 GiB |
| 35B-A3B | ~5 GiB |
| 122B-A10B | ~6 GiB |
| 397B-A17B | ~7.5 GiB |

These estimates exclude fixed GDN recurrent/convolution states, visual encoder activations, output buffers, allocator fragmentation, and batching. At 1.01M tokens, multiply the growing KV term by roughly **3.85×**. The largest MoE can have a smaller per-request KV cache than dense 27B because it uses fewer KV heads and fewer full-attention layers; its challenge is weight residency, not KV.

See [[04 Reference/Document Hub/LLM Inference Acceleration — Deep Dive (LLM高速化勉強会 + expanded)|LLM Inference Acceleration]] for the full prefill/decode and KV-cache mental model.

# 6. Sparse MoE — large capacity, small active path

The MoE checkpoints replace each dense SwiGLU FFN with a bank of small experts. For token representation $x$, a router chooses a top-$k$ subset while one shared expert always runs:

$$
\mathrm{MoE}(x)=E_{shared}(x)+\sum_{i\in\operatorname{TopK}(r(x))}p_i(x)E_i(x).
$$

| Model | Routed experts | Selected | Shared | Expert intermediate width | Active parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| 35B-A3B | 256 | 8 | 1 | 512 | 3B |
| 122B-A10B | 256 | 8 | 1 | 1,024 | 10B |
| 397B-A17B | 512 | 10 | 1 | 1,024 | 17B |

Why both routed and shared experts?

- The **shared expert** absorbs common transformations every token needs.
- **Routed experts** specialize by token/domain/modality, increasing capacity without executing the whole bank.
- More total experts at fixed top-$k$ improves capacity per active FLOP, provided routing remains balanced.

The systems price is important:

- Expert weights still consume memory; `A17B` improves compute, not checkpoint size.
- Expert parallelism dispatches tokens across devices and gathers results, often through all-to-all collectives. Small or imbalanced batches underutilize experts.
- High batch diversity can activate much of the expert bank over time, turning weight bandwidth and interconnect into bottlenecks.
- Router/load-balancing behavior must remain stable; the Qwen3-Next lineage normalizes router initialization and uses global balancing strategies.

For the communication side, see [[04 Reference/Document Hub/NCCL — Collective Communication Deep Dive|NCCL — Collective Communication Deep Dive]].

# 7. Native multimodality — what “early fusion” means here

## 7.1 It still has a vision encoder

“Native” does **not** mean raw pixels enter the language embedding table. Qwen3.5 uses a ViT-like tower derived from Qwen3-VL:

- 3D convolutional patch embedding with kernel/stride **2 frames × 16 × 16 pixels**;
- a 12-, 24-, or 27-layer bidirectional visual Transformer;
- a **2×2 spatial merger** that combines four adjacent patch features;
- an MLP projection into the language model’s hidden width;
- special image/video span tokens and modality IDs (`text=0`, `image=1`, `video=2`).

After merging, an image contributes roughly

$$
N_{image}\approx \left\lceil\frac{H}{32}\right\rceil
                 \left\lceil\frac{W}{32}\right\rceil
$$

visual tokens, ignoring processor resizing/padding. A 1024×1024 image is therefore on the order of 1,024 tokens. For video, multiply by the number of temporal 2-frame patches after sampling. **Images, video, text, tool traces, and generated reasoning all share the same context budget.**

## 7.2 Early fusion vs. a bolted-on VLM

The architectural towers alone do not make the model native. The key training change is that Qwen3.5 was pretrained from scratch on interleaved text, image, and video sequences with substantially more visual-text data, instead of taking a finished text model and learning only a connector/alignment stage. Consequences claimed by Qwen include:

- visual evidence can participate in multi-step reasoning and tool use rather than only captioning;
- stronger OCR/document, scientific-diagram, spatial, and video reasoning;
- visual GUI agents that inspect screenshots and emit actions/tool calls;
- visual coding, such as turning a sketch or recorded interface into code.

The hosted 1M setup is advertised for up to roughly two hours of sampled video. That is a system-level capability: achievable duration depends on frame rate, resolution, token budgeting, and output allowance—not merely the model’s maximum integer context length.

## 7.3 What it does not cover

- No native audio input or speech output in the general checkpoints → use **Qwen3.5-Omni**.
- No image/video generation → use a generative model such as Qwen-Image/Wan.
- No built-in browser, shell, or code interpreter in the weights. The open model emits tool calls; the host application must provide schemas, execution, permissions, observations, and loop control.

# 8. Pretraining, post-training, and infrastructure

## 8.1 Pretraining recipe — what is disclosed

The official blog describes three axes:

1. **Power.** More visual-text tokens than Qwen3, stronger Chinese/English and multilingual coverage, more STEM/reasoning data, and stricter filtering.
2. **Efficiency.** Qwen3-Next’s hybrid attention, sparse MoE, stability changes, and MTP.
3. **Versatility.** Early text–vision fusion, expanded video data, **201 languages/dialects**, and a much larger tokenizer.

The exact corpus composition, total general-purpose Qwen3.5 token count, deduplication pipeline, and per-stage compute are not disclosed in the release blog. “Tens of trillions” describes the scale at which the training infrastructure was made stable; it should not be treated as a precise published training-token count for every checkpoint.

## 8.2 Training-system co-design

Qwen reports:

- **Heterogeneous parallelism:** vision and language components can use different sharding/parallel strategies rather than forcing one topology on both.
- **Sparse overlap:** computation across vision/language/MoE components is overlapped, yielding mixed multimodal throughput close to the pure-text baseline in Qwen’s setup.
- **Native FP8 path:** FP8 activations, routing, and GEMMs, with runtime monitors retaining BF16 in sensitive layers; first-party results claim about **50% activation-memory reduction** and **>10% speedup**.
- **Asynchronous RL:** training and rollout inference are disaggregated, with dynamic load balancing, fault recovery, rollout-router replay, speculative decoding, and multi-turn rollout locking.
- **Bounded staleness:** the system controls asynchronous policy lag and data skew rather than letting faster environments dominate.

The blog reports a **3–5× end-to-end RL-system speedup**. This is an infrastructure comparison, not a universal inference speedup for downloaded weights.

## 8.3 Post-training philosophy

Qwen attributes much of the Qwen3 → Qwen3.5 gain to scaling the *difficulty and diversity of RL tasks and environments* across text, multimodal, multi-turn, coding, tool use, GUI operation, and agent planning. It emphasizes generalization rather than optimizing one benchmark. The public release does **not** specify enough detail to reconstruct the complete RL objective/reward mixture, so it would be unjustified to label the whole recipe simply “GRPO” or “GSPO.”

For conceptual RL fine-tuning background, see [[04 Reference/Document Hub/GRPO Deep Dive — RL Fine-Tuning with Group Relative Policy Optimization|GRPO Deep Dive]].

# 9. Tokenizer, long context, and multi-token prediction

## 9.1 A 248,320-entry vocabulary

Qwen3.5 expands the padded vocabulary from Qwen3’s roughly 150K to about **250K** while expanding coverage from 119 to 201 languages/dialects. Qwen reports **10–60% fewer tokens** for most languages. Fewer tokens mean less sequence work, KV/cache usage, and latency for the same text, although the larger embedding and output matrices cost parameters and softmax bandwidth—especially noticeable in the 0.8B/2B class. The 4B-and-smaller models tie input and output embeddings to contain this cost.

## 9.2 262K native, ~1.01M extrapolated

- All general models: **262,144 tokens native**.
- 4B, 9B, 27B, and all MoEs: official cards document extension to **1,010,000** via YaRN.
- 0.8B/2B cards claim only the native window; do not assume their quality at 1M.

The documented factor-4 YaRN override preserves multimodal/partial-RoPE fields:

```json
{
  "mrope_interleaved": true,
  "mrope_section": [11, 11, 10],
  "rope_type": "yarn",
  "rope_theta": 10000000,
  "partial_rotary_factor": 0.25,
  "factor": 4.0,
  "original_max_position_embeddings": 262144
}
```

> [!warning] Static YaRN is not free
> Open serving frameworks generally apply one fixed scale factor. A factor chosen for 1M can reduce short-context quality, so enable it only for long-context deployments and choose the smallest factor covering the real workload (e.g., 2× for ~524K). Memory and prefill cost still grow; reserve room for generated reasoning inside the total context.

## 9.3 MTP as a built-in speculative drafter

Ordinary next-token training predicts only $x_{t+1}$. Multi-token prediction adds auxiliary future-token objectives so the model learns to propose several future positions. At inference:

1. the cheap MTP head drafts multiple tokens;
2. the main model verifies them in a single pass;
3. the accepted prefix is emitted; rejected tokens are corrected.

Correct speculative decoding preserves the target distribution while reducing the number of expensive target-model steps. Qwen3.5 model cards include MTP serving examples; current vLLM syntax is:

```bash
--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

Acceptance depends on prompt/task/sampling. MTP improves latency only when saved target passes exceed draft + verification overhead.

# 10. Thinking modes, tool calls, and agent behavior

## 10.1 One checkpoint, two response policies

Qwen3.5 post-trained checkpoints can produce an explicit reasoning block before the answer or answer directly, but the defaults differ:

| Size | Documented default behavior | Switch |
| --- | --- | --- |
| 4B, 9B, 27B, and MoEs | thinking by default | set `enable_thinking=false` through the backend/chat template |
| 0.8B / 2B | direct/non-thinking examples by default; thinking is opt-in | set `enable_thinking=true` through the supported API/template path |

Unlike Qwen3, Qwen3.5 does **not officially support** prompt commands such as `/think` and `/nothink`. Use the API or `chat_template_kwargs`. Hosted Model Studio expects its top-level `enable_thinking`; self-hosted vLLM/SGLang commonly receive `chat_template_kwargs: {"enable_thinking": false}`. Always check the serving backend because parameter plumbing differs.

Qwen Studio’s **Auto / Thinking / Fast** modes are hosted orchestration policies. “Auto” combines adaptive reasoning with hosted search/code tools; downloading the weight does not automatically reproduce that controller.

## 10.2 Sampling is checkpoint- and task-specific

The model cards do not recommend one universal setting:

- 4B–122B general thinking: commonly `temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5`.
- Precise coding/visual reasoning: commonly lower temperature `0.6` and no presence penalty.
- Non-thinking: commonly `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5`.
- 397B’s card instead recommends `temperature=0.6, top_p=0.95, top_k=20` for thinking.
- 0.8B/2B text non-thinking examples use still different values.

Use the **specific checkpoint card**, cap reasoning time/tokens, and test deterministic task metrics. The 0.8B and 2B cards explicitly warn that opt-in thinking is more prone to non-terminating loops. Larger cards also recommend a presence penalty as one possible repetition control, while warning that high penalties can cause language mixing or quality loss.

## 10.3 Tool use is a protocol, not an embedded tool

A production agent loop is:

```text
messages + tool schemas
        │
        ▼
Qwen3.5 reasoning ─► structured tool call
                           │
                    host validates + executes
                           │
                    tool result appended
                           │
                           └────────► next model turn ─► final answer / next call
```

Self-hosted serving typically needs:

- `--reasoning-parser qwen3` to separate reasoning content;
- `--enable-auto-tool-choice --tool-call-parser qwen3_coder` for structured calls;
- a sandbox and explicit permission model for shell/browser/code tools;
- maximum turns, token budgets, timeouts, loop/repetition detection, and result-size limits.

# 11. Performance — what the first-party numbers say

## 11.1 Flagship breadth

Selected `Qwen3.5-397B-A17B` post-trained scores from its official card:

| Capability | Benchmark | Score |
| --- | --- | ---: |
| Knowledge | MMLU-Pro | 87.8 |
| Instruction following | IFBench | 76.5 |
| Science reasoning | GPQA | 88.4 |
| Code generation | LiveCodeBench v6 | 83.6 |
| Coding agent | SWE-bench Verified | 76.4 |
| Function calling | BFCL-V4 | 72.9 |
| Multi-turn agent | TAU2-Bench | 86.7 |
| Multimodal reasoning | MMMU-Pro | 79.0 |
| Visual mathematics | MathVision | 88.6 |
| OCR/document vision | OCRBench | 93.1 |
| Video understanding | VideoMME with subtitles | 87.5 |
| Mobile GUI agent | AndroidWorld | 66.8 |

The important result is **breadth**, not universal dominance. The same table shows competitors ahead on some hard reasoning, long-context, coding, and desktop-agent tasks. Several agent scores also depend heavily on scaffolds, tool availability, maximum turns, and evaluation prompts.

## 11.2 Scaling down retains surprising capability

| Benchmark | 122B-A10B | 27B | 35B-A3B | 9B | 4B |
| --- | ---: | ---: | ---: | ---: | ---: |
| MMLU-Pro | 86.7 | 86.1 | 85.3 | 82.5 | 79.1 |
| GPQA Diamond | 86.6 | 85.5 | 84.2 | 81.7 | 76.2 |
| LiveCodeBench v6 | 78.9 | 80.7 | 74.6 | 65.6 | 55.8 |
| SWE-bench Verified | 72.0 | 72.4 | 69.2 | — | — |
| TAU2-Bench | 79.5 | 79.0 | 81.2 | 79.1 | 79.9 |
| MMMU-Pro | 76.9 | 75.0 | 75.1 | 70.1 | 66.3 |
| MathVision | 86.2 | 86.0 | 83.9 | 78.9 | 74.6 |

These are useful for relative family selection, but not a substitute for evaluation on the intended prompts, quantization, context length, tool harness, and hardware.

## 11.3 Efficiency claims

At 32K / 256K context, Qwen reports that 397B-A17B decode throughput is:

- **8.6× / 19.0×** Qwen3-Max at comparable base capability;
- **3.5× / 7.2×** Qwen3-235B-A22B.

The growing advantage at 256K matches the architecture: three quarters of the stack has no growing token-by-token KV list, while sparse activation reduces FFN compute. Treat these as controlled first-party relative results—not portable tokens/s guarantees—because hardware, precision, batch, input/output mix, and serving kernels determine absolute throughput.

# 12. Practical usage

## 12.1 Which checkpoint to start with

| Constraint | Sensible starting point | Why / caveat |
| --- | --- | --- |
| Edge or tightly scoped extraction | 0.8B / 2B | smallest; prefer non-thinking; validate quality and loop behavior carefully |
| One consumer GPU | quantized 4B, then 9B if memory permits | best local iteration path; multimodal and tool-capable |
| High-memory workstation | 27B dense or quantized 35B-A3B | 27B has consistent dense compute; 35B uses fewer active parameters but stores more weights |
| Multi-GPU serious open deployment | 122B-A10B | strong capability without the flagship’s ~807 GB BF16 weight set |
| Maximum open Qwen3.5 quality | 397B-A17B | requires large aggregate memory and expert-aware serving |
| Managed production | Qwen3.5-Flash / Plus | built-in tooling, 1M defaults, no weight-serving burden |
| New coding-first deployment | also benchmark Qwen3.6-27B / 35B-A3B | later post-training generation on the same architectural family |

## 12.2 Minimal Transformers text inference

Use a current Transformers release; the hybrid model was added after older 4.x builds.

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen3.5-9B",
    device_map="auto",
    torch_dtype="auto",
)

out = pipe(
    "Explain why hybrid linear/full attention helps long-context decoding.",
    max_new_tokens=512,
)
print(out[0]["generated_text"])
```

For fast Gated DeltaNet kernels, Transformers relies on optional `causal_conv1d` and Flash Linear Attention (`fla`) packages. Without compatible kernels it silently falls back to slower, more memory-hungry PyTorch operations. Confirm the selected kernel path on the target GPU rather than assuming the architecture alone guarantees speed.

## 12.3 Minimal multimodal pipeline

```python
from transformers import pipeline

pipe = pipeline(
    "image-text-to-text",
    model="Qwen/Qwen3.5-9B",
    device_map="auto",
    torch_dtype="auto",
)

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"},
        {"type": "text", "text": "Explain this diagram and identify any inconsistency."},
    ],
}]

result = pipe(text=messages, max_new_tokens=512)
print(result[0]["generated_text"])
```

For local files or controlled preprocessing, use `AutoProcessor.apply_chat_template(...)` with `Qwen3_5ForConditionalGeneration`. Use `Qwen3_5ForCausalLM` / `Qwen3_5TextConfig` only when deliberately loading the language-only path.

## 12.4 vLLM server with reasoning and tools

Start with a context that actually fits; 262K is a maximum, not a required allocation:

```bash
vllm serve Qwen/Qwen3.5-9B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

Official model cards have also documented nightly/main-branch requirements during the release cycle. Re-check the current [vLLM Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html) before pinning production versions.

A direct/non-thinking OpenAI-compatible request:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=[{"role": "user", "content": "Return a five-line deployment checklist."}],
    max_tokens=512,
    temperature=0.7,
    top_p=0.8,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
print(response.choices[0].message.content)
```

For text-only service, supported vLLM releases expose `--language-model-only`, which skips the vision encoder and multimodal profiling to free memory for cache. Verify checkpoint/backend compatibility: earlier Qwen3.5 integrations had separate full-VLM vs language-only weight-prefix paths.

# 13. Deployment math and bottlenecks

## 13.1 Weight memory vs. active compute

- **Dense 27B BF16:** ~55.6 GB weights. At full 262K, the estimated full-attention KV term adds ~16 GiB before other state/workspace; reducing context matters even on an 80 GB accelerator.
- **35B-A3B BF16:** ~71.9 GB weights but only ~3B active per token. It may decode efficiently once resident, yet leaves almost no room on an 80 GB device for cache/workspace—quantization or sharding is usually needed.
- **397B-A17B BF16:** ~806.8 GB weight files. “Tensor parallel 8” is not proof it fits eight 80 GB GPUs; aggregate usable memory must exceed weights plus cache and runtime overhead, or an FP8/INT4/offloaded setup is required.

## 13.2 What limits each phase

| Phase | Likely bottleneck | Qwen3.5 feature that helps |
| --- | --- | --- |
| Vision encoding | ViT attention + many visual patches | patch merging, frame/resolution sampling, FlashAttention |
| Long prefill | remaining quadratic full-attention layers; GDN kernel quality | 3:1 hybrid, chunked GDN, FlashQLA/FLA, tensor/context parallelism |
| Batch-1 decode | weight bandwidth + recurrent/full-attention cache reads | sparse active experts, GQA, 75% fixed-state layers, quantization, MTP |
| High-throughput MoE | expert dispatch, all-to-all, load balance | expert parallelism, continuous batching, topology-aware placement |
| Million-token serving | KV/cache capacity + very long TTFT | YaRN, hybrid cache reduction, context/prefill disaggregation—but no free lunch |

Qwen later released [FlashQLA](https://qwen.ai/blog?id=flashqla), reporting 2–3× GDN chunked-prefill forward speedups and ~2× backward speedups over its FLA Triton baseline on Hopper in tested scenarios. This illustrates a general rule: a theoretically linear architecture still needs mature fused kernels to win in wall-clock time.

# 14. Strengths, limitations, and gotchas

## Strengths

- **One family across text, image, and video** rather than separate general and VL checkpoints.
- **Architecturally efficient long context:** only one quarter of layers carry a growing softmax KV cache.
- **Strong capacity/compute trade-offs:** dense models for predictable local deployment; MoEs for high capacity at lower active FLOPs.
- **Unusually broad size ladder:** 0.8B through 397B with a common interface and native 262K context.
- **Reasoning + direct answers + tool calls** in the same post-trained checkpoint.
- **Multilingual token efficiency:** 201 languages/dialects and a larger vocabulary.
- **Permissive weights:** official open checkpoints are Apache 2.0.

## Limitations / operational risks

1. **No complete general technical report yet.** Exact data mixture, compute, alignment objectives, safety pipeline, and many evaluation details remain under-specified.
2. **First-party benchmarks are scaffold-sensitive.** SWE-bench, browser, GUI, and tool benchmarks can shift materially with prompts, tools, turn caps, and parsers.
3. **Linear memory is compressed memory.** GDN cannot retain arbitrarily many perfectly addressable facts; full-attention layers mitigate rather than erase this limitation.
4. **1M is extrapolated and expensive.** YaRN changes position frequencies, static scaling can hurt short inputs, and full-attention prefill/KV still grow sharply.
5. **MoE active parameters are easy to misread.** Low active FLOPs do not remove weight storage, expert communication, or batch/load-balance requirements.
6. **Kernel availability determines real speed.** A PyTorch GDN fallback can erase expected gains; hardware/backend support must be profiled.
7. **Thinking can loop or consume huge budgets.** The smallest cards warn explicitly; all sizes need token/time/turn caps and repetition monitoring.
8. **Hosted and open behavior differ.** Plus/Flash built-in search, code interpreter, adaptive thinking, safety layers, and 1M defaults are services around the model, not contents of the downloaded checkpoint.
9. **Visual tokens are costly.** High resolution and high FPS rapidly consume context and vision-prefill compute; preprocess to the information density the task needs.
10. **No audio in the general family.** “Native multimodal” here is text + image + video, not omni-modal I/O.
11. **Qwen3.6 is already the successor for some workloads.** Do not select 3.5 solely by name recognition; benchmark the later 27B/35B models when their size fits.

# 15. The mental model to keep

```text
Qwen3.5 capability =
    early-fused text + image + video pretraining
  + scaled reasoning / agent / tool post-training
  + large multilingual tokenizer

Qwen3.5 efficiency =
    75% Gated DeltaNet (fixed recurrent history)
  + 25% gated GQA (exact global retrieval)
  + sparse MoE (large resident capacity, small active path)
  + MTP (speculative decode)
  + FP8 / fused-kernel / distributed systems work

But:
    active parameters ≠ resident parameters
    linear attention ≠ perfect recall
    1M context ≠ cheap context
    tool-call ability ≠ built-in tools
    hosted Qwen3.5 ≠ bare open checkpoint
```

# 16. Related vault notes

- [[04 Reference/Document Hub/LLM Inference Acceleration — Deep Dive (LLM高速化勉強会 + expanded)|LLM Inference Acceleration — Deep Dive]] — KV cache, linear/softmax attention, batching, speculative decoding, and serving.
- [[04 Reference/Document Hub/Transformers in Practice (DeepLearning.AI, Sharon Zhou) — Deep Dive/Transformers in Practice (DeepLearning.AI, Sharon Zhou) — Deep Dive|Transformers in Practice — Deep Dive]] — decoder, attention, RoPE, and inference foundations.
- [[04 Reference/Document Hub/LLM Fine-tuning Deep Dive — Full Fine-tuning, LoRA & QLoRA|LLM Fine-tuning Deep Dive]] — adaptation and memory math.
- [[04 Reference/Document Hub/NCCL — Collective Communication Deep Dive|NCCL — Collective Communication Deep Dive]] — tensor/expert-parallel communication.
- [[04 Reference/Document Hub/TensorRT & TensorRT-LLM — Deep Dive|TensorRT & TensorRT-LLM — Deep Dive]] — deployment concepts; verify Qwen3.5 hybrid-architecture support before assuming compatibility.
- [[04 Reference/Document Hub/GRPO Deep Dive — RL Fine-Tuning with Group Relative Policy Optimization|GRPO Deep Dive]] — RL fine-tuning background, not a claim about Qwen3.5’s undisclosed full recipe.

# 17. Primary sources

- [Qwen Team — Qwen3.5: Towards Native Multimodal Agents](https://qwen.ai/blog?id=qwen3.5) — release, benchmarks, pretraining, infrastructure, RL scaling, demos, multilingual list.
- [QwenLM/Qwen3.6 GitHub](https://github.com/QwenLM/Qwen3.6) — Qwen3.5/3.6 release dates, deployment ecosystem, license.
- Official model cards/configs: [397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) · [122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) · [35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) · [27B](https://huggingface.co/Qwen/Qwen3.5-27B) · [9B](https://huggingface.co/Qwen/Qwen3.5-9B) · [4B](https://huggingface.co/Qwen/Qwen3.5-4B) · [2B](https://huggingface.co/Qwen/Qwen3.5-2B) · [0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B).
- [Hugging Face Transformers — Qwen3.5 implementation docs](https://huggingface.co/docs/transformers/model_doc/qwen3_5) — model classes, hybrid layers, mRoPE, vision config, kernel fallbacks.
- [Qwen Team — Qwen3-Next design](https://qwen.ai/blog?id=e34c4305036ce60d55a0791b170337c2b70ae51d) — hybrid-attention, ultra-sparse MoE, stability changes, MTP.
- [Yang, Kautz, Hatamizadeh — Gated Delta Networks](https://arxiv.org/abs/2412.06464) (ICLR 2025) — gated delta rule and chunkwise algorithm; [official code](https://github.com/NVlabs/GatedDeltaNet).
- [Qiu et al. — Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708) — head-specific output gating, attention sinks, stability.
- [Qwen Team — FlashQLA](https://qwen.ai/blog?id=flashqla) — optimized Qwen-family GDN kernels.
- [Qwen3.5-Omni Technical Report](https://arxiv.org/abs/2604.15804) — separate audio/video/speech branch and scope distinction.
