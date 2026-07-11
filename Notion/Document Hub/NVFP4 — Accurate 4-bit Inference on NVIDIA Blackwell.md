---
base: "[[Document Hub.base]]"
cover: "[[NVFP4 — Accurate 4-bit Inference on NVIDIA Blackwell.webp]]"
Created by: Heaven Chen
Created time: 2026-07-10T18:11:00
Status: Done
Last updated time: 2026-07-10T18:11:00
Last edited by: Heaven Chen
Category:
  - GPU
  - Hardware
  - LLM
---
> **Deep dive into NVFP4** — NVIDIA's 4-bit floating-point format for low-precision inference on the Blackwell architecture. Based on the NVIDIA Technical Blog [*Introducing NVFP4 for Efficient and Accurate Low-Precision Inference*](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) (Alvarez, Almog, Chung, Layton, Stosic, Krashinsky, Aubrey; Jun 2025). The one-line thesis: **NVFP4 gets you ~4-bit memory/compute with near-FP8 accuracy**, by pairing a tiny 16-value block with a smarter, higher-precision *scaling* scheme.

# 1. Why 4-bit, and why it's hard

Quantization — storing weights/activations in fewer bits — is the most common inference optimization because it cuts memory traffic and simplifies arithmetic, which raises throughput and lowers latency and energy. The catch is **accuracy loss**: the fewer bits you use, the more you risk destroying the model's “intelligence,” and the drop from FP32/FP16 down to **FP4** is where it bites hardest. A raw 4-bit float can represent only 16 distinct values, so naïvely rounding a tensor into it throws away most of its dynamic range.

NVIDIA **Blackwell** supports a wide precision ladder — FP64, FP32/TF32, FP16/BF16, INT8/FP8, FP6, and FP4 — and its **5th-generation Tensor Cores** add native hardware for microscaled FP4. NVFP4 is the format designed to make that 4-bit tier *usable* without a meaningful accuracy hit.

![[performance-evolution-nvidia-gpu-generations-png.webp|Peak low-precision performance across NVIDIA GPU architectures]]

*Figure 1 (NVIDIA) — smallest-supported floating-point performance, dense/sparse PFLOPS: A100 0.3/0.6 → H100 1.9/3.9 → B200 9/18, B300 13/18, GB200 10/20, GB300 15/20.*

# 2. The three 4-bit formats on Blackwell

Blackwell supports three E2M1-based 4-bit floats. They differ not in the 4-bit element but in **how the shared scale factor works**.

| Feature | FP4 (E2M1) | MXFP4 | NVFP4 |
| --- | --- | --- | --- |
| Element structure | 4 bits: 1 sign, 2 exponent, 1 mantissa | Same E2M1 | Same E2M1 |
| Shared scale | Software scale (per tensor) | **1 power-of-two (E8M0) scale per 32-value block** | **1 FP8 (E4M3) scale per 16-value block + per-tensor FP32** |
| Block size | — | 32 values | **16 values** |
| Hardware-accelerated scaling | No | Yes | Yes |
| Accuracy risk vs FP8 | Noticeable drop | Noticeable drop | **Lower risk, especially for larger models** |

**MXFP4** is the OCP Microscaling format (Blackwell also supports it); **NVFP4** is NVIDIA's refinement of the same idea with two changes — a *finer* block (16 vs 32) and a *higher-precision* scale (fractional E4M3 vs power-of-two E8M0), backed by a second per-tensor scale.

# 3. What NVFP4 actually is

Each NVFP4 element is a standard **E2M1** 4-bit float: **1 sign bit, 2 exponent bits, 1 mantissa bit**, representing roughly the range **−6 to +6**. The representable magnitudes are just: `0, 0.5, 1, 1.5, 2, 3, 4, 6` (and their negatives) — 16 values total. That's far too coarse on its own, so NVFP4 leans on **two architectural innovations**:

1. **High-precision scale encoding** — the per-block scale is stored in FP8 (E4M3), not power-of-two.
2. **A two-level micro-block scaling strategy** — a fine per-16-value scale *plus* a per-tensor FP32 scale.

Together these let a 16-value block track the *local* dynamic range of the data far better than a single global scale could.

![[nvfp4-two-level-scaling.gif|NVFP4 two-level scaling structure]]

*Figure 2 (NVIDIA) — NVFP4's dual scaling: each group of 16 E2M1 values shares an FP8 (E4M3) block scale; blocks are then globally normalized by a per-tensor FP32 (E8M23) scale.*

## 3.1 The reconstruction rule

Inside each 16-value block, a stored 4-bit code $x_q\in[-6,+6]$ is turned back into a real value by multiplying by that block's scale $s$:

$$
x = x_q \times s
$$

where $s$ is a **high-precision FP8 (E4M3)** factor recomputed for every 16-value group. Recomputing $s$ this often is what keeps quantization error small at 4 bits. On top of that, a single **per-tensor FP32** scalar first normalizes the whole tensor's distribution so the block-level E4M3 scales land in their effective range.

# 4. Innovation 1 — high-precision scale encoding (E4M3 vs E8M0)

MXFP4 stores each block's scale as **E8M0**, i.e. it can only snap the scale to the **nearest power of two** ($2^n$). NVFP4 instead stores the scale as **E4M3 FP8**, which allows **non-power-of-two, fractional** scales. That flexibility lets NVFP4 choose a scale that fits the block's *actual* distribution, rather than the nearest binary step.

![[quantization-precision-power-of-two-fractional-scaling-comparison-png.webp|Power-of-two vs fractional scaling]]

*Figure 3 (NVIDIA) — the same input matrix quantized with E8M0 (coarse, power-of-two) vs E4M3 (fine, fractional) block scales; E4M3 tracks the originals more closely.*

The payoff is measured as **mean-squared error (MSE)** between original values and their quantized reconstructions. In NVIDIA's example, encoding the scale with **E8M0 gives MSE 0.72**, while **E4M3 gives MSE 0.08** — nearly an order of magnitude lower.

![[e8m0-e4m3-quantization-error-comparion.gif|E8M0 vs E4M3 quantization error]]

*Figure 4 (NVIDIA) — mapping original values (yellow) onto E8M0 vs E4M3 quantized scales; lower MSE is better (0.72 vs 0.08).*

Intuitively:

- **E8M0** snaps the scale to the nearest $2^n$, which can badly misplace the block's maximum (**amax**) and inflate error across the block.
- **E4M3** picks *one fractional scale* that minimizes the block's total squared error — usually improving the amax fit; a few values may be slightly worse, but the block as a whole keeps more fidelity.

*Why keep E8M0 at all?* Simplicity: power-of-two scales need no extra per-tensor software scaling and are fine for tensors that aren't sensitive to scale precision. NVFP4 trades that simplicity for accuracy.

# 5. Innovation 2 — micro-block scaling (16 vs 32 values)

NVFP4 uses **block floating-point**: values are grouped and share one scale. Cutting the group from MXFP4's **32** down to **16** doubles the number of scales per tensor, giving **twice as many chances to match the local dynamic range**. Real tensors mix large and small magnitudes, and one “umbrella” scale over too many values smears the small-but-important differences in weights/activations. Tighter blocks preserve them.

![[comparison-nvfp4-mxfp4-block-structure-1.gif|NVFP4 vs MXFP4 block structure]]

*Figure 5 (NVIDIA) — MXFP4's 32-value block with one coarse power-of-two scale vs NVFP4's 16-value blocks, each with its own FP8 scale; reconstruction *$x = x_q \times s$*.*

## 5.1 The bit budget

NVFP4's overhead is small. Per value you store the **4-bit** element plus a share of the block scale (**one FP8 = 8 bits across 16 values = 0.5 bit/value**), i.e. **~4.5 bits per value**, plus **one FP32 per tensor** (negligible). The Blackwell Tensor Core handles the element grouping, dynamic scaling, and 4-bit matmul automatically in hardware.

# 6. Accuracy — NVFP4 vs FP8

The headline result: quantizing **DeepSeek-R1-0528** from FP8 to NVFP4 with **post-training quantization (PTQ)** costs **≤1% accuracy** on key benchmarks — and on AIME 2024 NVFP4 is actually **+2%**.

| Benchmark | FP8 | NVFP4 |
| --- | --- | --- |
| MMLU-PRO | 85% | 84% |
| GPQA Diamond | 81% | 80% |
| HLE | 15% | 14% |
| LiveCodeBench | 77% | 76% |
| SciCode | 40% | 40% |
| Math-500 | 98% | 98% |
| AIME 2024 | 89% | **91%** |

![[nvfp4-model-accuracy-comparison-png.webp|DeepSeek-R1 FP8 vs NVFP4 accuracy]]

*Figure 6 (NVIDIA) — DeepSeek-R1-0528 accuracy, FP8 vs NVFP4, across seven evaluations: near-identical.*

# 7. Memory footprint

NVFP4 stores **~4.5 bits/value** (4-bit element + one FP8 scale per 16) plus one FP32 per tensor. Versus higher precision that is roughly:

- **~3.5× smaller than FP16**
- **~1.8× smaller than FP8**

At rack scale this compounds: an **NVIDIA GB300 NVL72** (36 Grace Blackwell Ultra superchips — each 1 Grace CPU + 2 Blackwell Ultra GPUs) exposes a **~40 TB** memory budget per system, and NVFP4's compactness lets far larger models (or longer test-time-scaling workloads) fit and run.

# 8. Energy efficiency

Fewer bits means less energy per data movement and per arithmetic op. Combined with Blackwell architectural gains and liquid cooling, FP4 drives large per-token energy wins over Hopper: **up to 25× (Blackwell)** and **up to 50× (Blackwell Ultra)** more energy-efficient per token than an H100 baseline for GPT-MoE-1.8T.

![[50x-more-energy-efficient-per-token-graph-png.webp|50x more energy efficient per token vs Hopper]]

*Figure 7 (NVIDIA) — Joules/token for GPT-MoE-1.8T across generations: Kepler 42,000 → Pascal 17,460 → Volta 1,200 → Ampere 150 → Hopper 10 → Blackwell 0.4 → Blackwell Ultra 0.2 (~200,000× over 10 years).*

# 9. How to use NVFP4

The tooling path, per NVIDIA:

- **Quantize** with [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) or [LLM Compressor](https://github.com/vllm-project/llm-compressor) — both support **PTQ**, **QAT**, and advanced techniques. Model Optimizer also handles non-LLM models and **ONNX** export.
- **Export** to a Unified Hugging Face checkpoint.
- **Deploy** on [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and [vLLM](https://github.com/vllm-project/vllm) (early NVFP4 support), with **SGLang** support upcoming.
- **Or skip quantizing** — Hugging Face already hosts prequantized NVFP4 checkpoints, e.g. [DeepSeek-R1-0528-FP4](https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4), [Llama-3.1-405B-Instruct-FP4](https://huggingface.co/nvidia/Llama-3.1-405B-Instruct-FP4), and FLUX.1-dev.

# 10. Takeaways

- NVFP4 = **E2M1 4-bit element + two-level scaling** (per-16 FP8/E4M3 block scale, per-tensor FP32). The *scaling*, not the element, is the innovation.
- Two levers vs MXFP4: **half the block size (16 vs 32)** and a **fractional FP8 scale (E4M3 vs power-of-two E8M0)** — together cutting scale-encoding MSE ~9× in NVIDIA's example (0.72 → 0.08).
- Result: **~4.5 bits/value** with **≤1% accuracy loss vs FP8** on DeepSeek-R1, **~3.5× / ~1.8×** memory savings vs FP16 / FP8, and up to **50×** better energy/token than Hopper.
- Requires **Blackwell 5th-gen Tensor Cores** for hardware-accelerated microscaling; production tooling exists today (Model Optimizer / LLM Compressor → TensorRT-LLM / vLLM).

---

**Sources**: [NVIDIA Technical Blog — Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) (all figures © NVIDIA) · related: [3 Ways NVFP4 Accelerates AI Training and Inference](https://developer.nvidia.com/blog/3-ways-nvfp4-accelerates-ai-training-and-inference/) · [Optimizing LLMs with Post-Training Quantization](https://developer.nvidia.com/blog/optimizing-llms-for-performance-and-accuracy-with-post-training-quantization/) · [Blackwell architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/).