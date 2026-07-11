---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-02T13:19:00
Status: In progress
Last updated time: 2026-07-08T11:36:00
Last edited by: Heaven Chen
Category:
  - ML
  - Hardware
---
> The interchange format + cross-platform inference engine. Current state (mid-2026): ONNX 1.21 introduced **opset 26** (2-bit types, CumProd, BitCast; opset 27 adds LinearAttention/CausalConvWithState) — [release blog](https://lfaidata.foundation/blog/2026/04/27/onnx-v1-21-0-released-introducing-opset-26-new-cumprod-operator-2-bit-type-support-and-more/); ONNX Runtime is at **1.27** ([releases](https://github.com/microsoft/onnxruntime/releases)).

# 1. What ONNX is

A **protobuf-serialized dataflow graph**: `ModelProto ⊃ GraphProto ⊃ {NodeProto (ops), initializers (weights), ValueInfo (typed edges)}`. Ops come from a versioned operator set (**opset**) — the contract that lets a graph exported today run on any runtime that implements that opset. Control flow exists (`If`, `Loop`, `Scan`) but the sweet spot is static dataflow. Shapes may be symbolic (`batch`, `seq`) — that's what makes one file serve many batch sizes.

**Why an IR at all**: it decouples *authoring framework* from *deployment target*. One PyTorch model → ONNX → {server CPU (ORT), NVIDIA (TRT EP), Apple (CoreML EP), browser (WebGPU/wasm), phone (NNAPI/QNN)} without rewriting the model.

# 2. Exporting from PyTorch — the part that actually bites

Two exporters:

- **TorchScript-trace based** (legacy, `dynamo=False`): traces one execution — data-dependent Python control flow gets *baked in* silently.
- **Dynamo-based** (`dynamo=True`, default direction since torch 2.5+): captures via torch.export graphs, handles dynamic shapes more faithfully; emits opset 18+.

```python
import torch
model.eval()
ex = torch.onnx.export(
    model, (x,), dynamo=True,
    dynamic_shapes={"x": {0: "batch", 1: "seq"}},
    opset_version=21,
)
ex.save("model.onnx")

# ALWAYS verify numerics:
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
out = sess.run(None, {"x": x.numpy()})[0]
assert np.allclose(out, model(x).detach().numpy(), atol=1e-4)
```

Common failure modes: unsupported custom ops (register a custom op or decompose), data-dependent shapes (`nonzero`-style ops), training-mode layers left on (dropout/batchnorm — call `eval()`), fp16 overflow after conversion (keep norms in fp32).

# 3. ONNX Runtime — how it executes

- **Graph optimizations** at session load, in levels: constant folding, redundant node elimination → fusions (Attention, LayerNorm, GELU, MatMul+Add) → layout transforms (NCHW→NCHWc). You can serialize the optimized graph once and skip re-optimizing.
- **Execution Providers (EPs)**: ORT partitions the graph and assigns subgraphs to the best available backend — TensorRT EP > CUDA EP > CPU for NVIDIA stacks; also OpenVINO, DirectML, CoreML, QNN, ROCm. Nodes an EP can't take fall back to CPU (watch for *partition thrash*: many tiny partitions → copy overhead kills gains; inspect with `sess.get_providers()` + profiling).
- **Quantization**: dynamic (weights INT8, activations quantized on the fly — easy, good for MatMul-heavy CPU workloads) vs static QDQ (calibration data → Q/DQ nodes → EPs like TRT consume them natively):

$q = \mathrm{clip}\Big(\mathrm{round}\big(\tfrac{x}{s}\big) + z,\; q_{min},\, q_{max}\Big), \qquad x \approx s\,(q - z)$

per-channel scales for weights, per-tensor for activations is the usual compromise.

- **ORT also trains** (ORTModule) but its center of gravity is inference.

# 4. Mental model of the deployment matrix

| Path | Effort | Speed | Portability |
| --- | --- | --- | --- |
| PyTorch eager | none | 1× | GPU/CPU |
| torch.compile | tiny | ~1.3–2× | NVIDIA/AMD |
| ONNX + ORT (CUDA EP) | small | ~1.5–2× | everywhere |
| ONNX + ORT (TRT EP) | medium | ~2–5× | NVIDIA |
| native TensorRT / TRT-LLM | high | max | NVIDIA, per-arch engine |

Rule of thumb: export to ONNX when you need *portability or a serving boundary*; go native TRT when you own the hardware and need the last 2×.

# Cross-check questions

1. Why can a traced export silently produce wrong results for `if x.sum() > 0:` branches? 2. Where do Q/DQ nodes come from and who consumes them? 3. Why might adding the TensorRT EP make a model *slower* than CUDA EP alone?