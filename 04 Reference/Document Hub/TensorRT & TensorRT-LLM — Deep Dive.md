---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-02T13:19:00
Status: In-Progress
Last updated time: 2026-07-08T11:36:00
Last edited by: Heaven Chen
Category:
  - GPU
  - Hardware
  - LLM
---
> NVIDIA's inference optimizer + runtime. Current state (mid-2026): TensorRT 10.x line; TensorRT-LLM releases monthly with B200/FP4 support ([release notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html), [TensorRT release notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes.html)).

# 1. What TensorRT actually is

A **compiler** for inference: it takes a trained graph (via ONNX or, for LLMs, TRT-LLM's Python model definitions), and emits a serialized **engine** — a fused, precision-lowered, kernel-selected executable specialized for ONE GPU architecture + shape profile. Pipeline:

`model → (parser) → network definition → builder (optimization passes) → engine (.plan) → runtime execution context`

## The builder's optimization passes — and why they work

- **Layer & tensor fusion**: merge Conv+Bias+ReLU or LayerNorm+GEMM chains into single kernels. Why it matters: elementwise ops are bandwidth-bound; fusion removes intermediate HBM round-trips (the activation never leaves registers/SRAM).
- **Kernel auto-tuning (tactic selection)**: for each op and shape, the builder *times* candidate kernels (cuBLAS/cuDNN/cutlass tactics) on the actual GPU and records the winner into the engine. This is why engines are not portable across GPU architectures.
- **Precision lowering**: per-layer FP16/BF16/FP8/INT8/FP4 selection. INT8 needs a mapping $x \approx s\,q$ with scale from calibration; the classic criterion picks the clipping threshold minimizing KL divergence between the fp32 activation distribution and its quantized version:

$s^* = \arg\min_s \; D_{KL}\big(P_{fp32} \,\|\, Q_s\big)$

Modern flow: quantize in the framework with **TensorRT Model Optimizer** (fake-quant QDQ nodes, e.g. FP8/INT4-AWQ), export, and TRT honors the Q/DQ placement.

- **Dynamic shapes**: an *optimization profile* declares (min, opt, max) per input dim; the builder specializes kernels for `opt` while staying valid across the range.

```python
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
net = builder.create_network()  # strongly-typed graphs are default in TRT 10
parser = trt.OnnxParser(net, logger)
parser.parse(open("model.onnx", "rb").read())
cfg = builder.create_builder_config()
p = builder.create_optimization_profile()
p.set_shape("input", min=(1,3,224,224), opt=(8,3,224,224), max=(32,3,224,224))
cfg.add_optimization_profile(p)
engine_bytes = builder.build_serialized_network(net, cfg)
```

Execution: create an `ExecutionContext`, bind device buffers, `execute_async_v3(stream)`. Engines are thread-unsafe per context; use one context per stream.

# 2. TensorRT-LLM — why LLMs need their own stack

Static engines can't handle the *dynamic* structure of LLM serving (ragged batches, growing KV). TRT-LLM adds an LLM-specific runtime on top of TRT kernels:

- **In-flight (continuous) batching**: new requests join the batch at any decode step; finished ones leave. Removes head-of-line blocking, keeps SMs saturated.
- **Paged KV cache**: KV stored in fixed-size blocks with an indirection table (vLLM's PagedAttention idea) — no contiguous per-sequence allocation, near-zero fragmentation. KV per token = $2 L\, n_{kv} d_{head}$ bytes×dtype; paging makes eviction/sharing (prefix caching) possible.
- **Quantized inference**: FP8 (Hopper), FP4 (Blackwell B200 loads weights natively in FP4), INT4-AWQ, INT8-SmoothQuant. Weight-only quant attacks the memory-bandwidth bound of decode.
- **Speculative decoding** support incl. EAGLE-3 tree drafting; **multi-GPU** via built-in TP/PP/EP using NCCL; fused MoE and attention kernels (paged MQA, fused RMSNorm/RoPE per recent releases).
- The modern API is the Python `LLM` class (`from tensorrt_llm import LLM`), with AutoDeploy to lower HF models without hand-written engine code.

# 3. When to choose what

- TensorRT (via ONNX): CNNs, ViTs, detection/segmentation — anything static-shape-ish → biggest wins from fusion + INT8/FP8.
- TensorRT-LLM: autoregressive text/multimodal serving → wins come from batching + KV management + weight-only quant.
- Compare: `torch.compile` (easy, portable, smaller gains), ONNX Runtime (+TRT execution provider = middle ground), vLLM (Python-first serving, broader model coverage, often comparable throughput).

# Cross-check questions

1. Why are TRT engines GPU-architecture-specific while ONNX files are portable? 2. Why does weight-only INT4 speed up decode even though compute runs in fp16? 3. What breaks if you serve variable-length requests with static batching, and which two TRT-LLM features fix it?