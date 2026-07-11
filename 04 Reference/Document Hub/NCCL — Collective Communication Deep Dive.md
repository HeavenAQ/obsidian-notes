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
---
> NVIDIA Collective Communications Library — the layer every multi-GPU job stands on. Current state (mid-2026): NCCL 2.30.x ([release notes](https://docs.nvidia.com/deeplearning/nccl/release-notes/index.html)) — device API with versioning, one-sided put/signal APIs, official Python API, MLOPart support.

# 1. What NCCL is

Topology-aware implementations of the MPI-style collectives — **all-reduce, broadcast, reduce, all-gather, reduce-scatter, all-to-all, send/recv** — as CUDA kernels that move data directly between GPU memories over NVLink/NVSwitch, PCIe, and InfiniBand/RoCE (GPUDirect RDMA: NIC reads GPU memory without bouncing through the CPU). PyTorch's `distributed` NCCL backend, DDP, FSDP, DeepSpeed, Megatron all sit on it.

# 2. The algorithms and their cost

For $P$ ranks and message size $M$:

- **Ring all-reduce**: reduce-scatter ($P-1$ steps, each sending $M/P$) then all-gather ($P-1$ steps). Total bytes sent per rank:

$2\,\frac{P-1}{P}\,M \;\xrightarrow{P\to\infty}\; 2M$

Bandwidth-optimal (per-rank traffic independent of $P$) but latency $\mathcal{O}(P)$ — bad for small messages on many ranks.

- **Tree all-reduce**: reduce up + broadcast down a (double binary) tree — latency $\mathcal{O}(\log P)$, chosen for small messages / large $P$. NCCL picks ring vs tree per (size, topology) automatically; `NCCL_ALGO`/`NCCL_PROTO` override (protocols: Simple / LL / LL128 trading latency vs bandwidth).
- **Reduce-scatter + all-gather are the halves of the ring** — exactly why ZeRO/FSDP communication decomposes into them: reduce-scatter grads (each rank ends with its shard reduced), all-gather params.

# 3. Topology awareness

At init NCCL builds a hardware graph (NVLink pairs, NVSwitch domains, PCIe switches, NUMA, NICs) and searches for ring/tree layouts that maximize bottleneck bandwidth — e.g., inside a DGX node rings run over NVSwitch at ~900 GB/s; across nodes each ring maps onto one IB rail (multi-rail = multiple rings in parallel). This is why `CUDA_VISIBLE_DEVICES` order and NIC affinity change performance, and why TP is kept intra-node (see the Parallel Training doc's bandwidth hierarchy).

# 4. How DDP actually uses it

Autograd hooks fire as gradients become ready; DDP groups them into **buckets** (default 25 MB) and launches async all-reduce per bucket on a side stream — communication **overlaps** the rest of backward. Tuning: `bucket_cap_mb` (too small = latency-bound many calls; too big = less overlap), `gradient_as_bucket_view=True` (no copy), static graph mode.

```python
import torch.distributed as dist
dist.init_process_group("nccl")           # one process per GPU (torchrun)
torch.cuda.set_device(local_rank)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

# raw collective
t = torch.ones(1_000_000, device='cuda')
dist.all_reduce(t, op=dist.ReduceOp.SUM)   # in-place, async on NCCL stream
```

Semantics to respect: collectives must be called **in the same order on every rank** (mismatched order = deadlock); one communicator's ops serialize on its stream; a hung rank hangs everyone (set `NCCL_TIMEOUT`/watchdog; `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`).

# 5. Debugging & tuning cheat-sheet

- `NCCL_DEBUG=INFO` (+`NCCL_DEBUG_SUBSYS=INIT,GRAPH`) — prints chosen rings/trees, NICs, P2P level.
- `nccl-tests` (`all_reduce_perf -b 8 -e 4G -f 2 -g 8`) — measure bus bandwidth; compare to NVLink/IB line rate. **busBW = algBW × 2(P-1)/P** for all-reduce.
- Common fixes: wrong NIC selection (`NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`), PCIe P2P disabled (`NCCL_P2P_LEVEL`), congestion from two jobs sharing rails, small-message latency (increase bucket size / gradient accumulation).
- 2.30-era features worth knowing: **device API** (call collectives from inside your own kernels, with `ncclCommQueryProperties` feature checks), **one-sided ncclPutSignal/ncclWaitSignal** (RDMA-style writes into a peer's registered window — no target participation), **official Python API** (DLPack-interoperable, PyTorch/CuPy-friendly), MLOPart + MPS for 2 ranks/GPU.

# 6. The mental model

Every distributed-training design question reduces to: *which collective, how many bytes, on which fabric, overlapped with what compute?* DP = all-reduce(2M) on the slow fabric; FSDP = all-gather + reduce-scatter (3× volume, finer overlap); TP = all-reduces inside every layer (needs NVLink); MoE = all-to-all (needs uniform load). If you can write the bytes-per-step formula, you can predict scaling before renting the cluster.

# Cross-check questions

1. Derive the $2\frac{P-1}{P}M$ ring cost. 2. Why does DDP bucket gradients instead of all-reducing per-tensor? 3. Two nodes: all_reduce_perf busBW is 12 GB/s but the IB NICs are 400 Gb/s — name three likely culprits.