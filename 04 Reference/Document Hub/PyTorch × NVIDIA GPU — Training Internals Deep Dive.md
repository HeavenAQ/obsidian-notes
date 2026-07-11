---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-02T13:20:00
Status: In-Progress
Last updated time: 2026-07-08T11:36:00
Last edited by: Heaven Chen
Category:
  - GPU
  - Hardware
  - ML
---
> How PyTorch actually drives an NVIDIA GPU during training, and every knob that matters. Current state (mid-2026): PyTorch 2.12 ([releases](https://github.com/pytorch/pytorch/releases)); FSDP1 deprecated since 2.11, FSDP2/DTensor is the sharding path. Companion docs: *CUDA GPU Architecture* and *CUDA/PyTorch Versioning* already in this hub.

# 1. The execution model: async by default

Every CUDA op you call is **enqueued on a stream** and returns immediately; the CPU races ahead building the queue. Synchronization happens only at `.item()`, `.cpu()`, `torch.cuda.synchronize()`, or cross-stream events. Consequences: (a) timing with `time.time()` without sync is meaningless — use `torch.cuda.Event`; (b) a slow CPU (Python overhead, small kernels) starves the GPU — the "launch-bound" regime that CUDA Graphs and `torch.compile` fix by removing per-kernel launch overhead.

# 2. Where the FLOPs come from

Matmuls/convs dispatch to **cuBLAS/cuDNN/cutlass** kernels that use **tensor cores** — but only when dtype and alignment allow (fp16/bf16/tf32; dims multiple of 8/16). One matmul's cost model: a GEMM of $(M,K)\times(K,N)$ does $2MKN$ FLOPs and moves $\approx 2(MK + KN + MN)$ bytes; its arithmetic intensity must beat the GPU's FLOPs/byte ratio (~300 for H100 bf16) to be compute-bound — this is the **roofline** logic behind "bigger batch = better utilization" and "elementwise ops are free FLOPs-wise but expensive bandwidth-wise".

- `torch.backends.cuda.matmul.allow_tf32 = True` (or `set_float32_matmul_precision('high')`): fp32 matmuls run on tensor cores with 10-bit mantissa — ~8× faster, negligible accuracy loss for DL.
- `cudnn.benchmark = True`: autotune conv algorithms per shape.

# 3. Mixed precision (AMP) — the math

Master weights stay fp32; forward/backward run in fp16/bf16.

- **bf16**: same exponent range as fp32 (8 bits), 7-bit mantissa → no overflow worries, **no GradScaler needed**. Default choice on Ampere+.
- **fp16**: 5-bit exponent → gradients underflow. GradScaler multiplies the loss by $S$ (dynamically tuned), so gradients shift into representable range, then unscales before the optimizer step: $g \leftarrow g_{scaled}/S$, skipping steps whose grads contain inf/nan.

```python
scaler = torch.amp.GradScaler()
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    loss = model(x).loss
scaler.scale(loss).backward()   # no-op scaling for bf16; needed for fp16
scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
```

# 4. Memory: the caching allocator

`cudaMalloc` is slow and synchronizing, so PyTorch **caches** freed blocks in pools. Implications: `nvidia-smi` shows *reserved* (cached) not *allocated*; fragmentation can OOM you even with free reserved memory (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` mitigates). Diagnose with `torch.cuda.memory_summary()` / `memory._record_memory_history()`. Training memory = params + grads + optimizer states (16 bytes/param w/ Adam mixed precision) + activations ($\propto B\,t\,d\,L$) — activations are the lever: activation checkpointing trades ~33% compute for $\mathcal{O}(\sqrt{L})$ memory.

# 5. Feeding the beast

- **Pinned (page-locked) host memory** enables true async H2D copies: `DataLoader(pin_memory=True)` + `x.to('cuda', non_blocking=True)` overlaps copy with compute on a separate copy engine.
- `num_workers>0` moves decode/augment off the training thread; `persistent_workers=True` avoids fork-per-epoch.
- **channels_last** memory format for convnets: NHWC matches tensor-core conv kernels → up to ~1.5× on Ampere+.

# 6. torch.compile (Inductor)

Dynamo captures Python into an FX graph; Inductor emits fused Triton kernels + CUDA Graphs. Wins: kernel fusion of elementwise chains (bandwidth!), removal of launch overhead, autotuned matmul epilogues. `mode='max-autotune'` for steady-state training; watch for recompiles on dynamic shapes (`dynamic=True`). As of 2.12, compile FSDP2 models with `fullgraph=False` or compile before wrapping ([tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)).

# 7. Multi-GPU in one page

- **DDP**: replicate model, all-reduce grads bucketed & overlapped with backward (see NCCL doc). Static memory per GPU.
- **FSDP2**: shard params/grads/optimizer states per-parameter on DTensor; all-gather on demand. Use when the model doesn't fit.
- Wrap order matters: `fully_shard` per block, then root; mixed precision policies live in the wrapper.

# 8. Profiling checklist

`torch.profiler` (with `record_shapes`, `profile_memory`, tensorboard/chrome trace) → look for: GPU idle gaps (input pipeline or launch-bound), tiny kernels (fuse/compile), NCCL time not overlapped (bucket tuning), H2D copies inside the step (pin memory). Nsight Systems for cross-process timelines.

# Cross-check questions

1. Why does bf16 not need loss scaling while fp16 does? 2. A step is 40% idle on the GPU timeline — name the three most likely causes in order. 3. Derive why Adam mixed-precision training of a 1B model needs ~16 GB before activations.