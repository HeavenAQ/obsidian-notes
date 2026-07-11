---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-01T14:28:00
Status: Done
Last updated time: 2026-07-03T20:21:00
Last edited by: Heaven Chen
Category:
  - AIGC
  - ML
  - Survey
  - GPU
---
*Focus: architecture, competitive edges, drawbacks, and a deep dive on the "Transfer" (controlled generation) capability, with code.*

Cosmos 3 launched **June 1, 2026** (GTC Taipei / COMPUTEX) as NVIDIA's next-generation open **omnimodal world foundation model (WFM)** for Physical AI. It supersedes the earlier split-model lineage (Cosmos-Predict, Cosmos-Transfer, Cosmos-Reason, Cosmos-Policy).

## 1. What Cosmos 3 Is

Cosmos 3 is a single, open, **omni-model** that unifies three things that used to be separate models and pipelines:

1. **Physical reasoning** — understanding motion, causality, and spatial relationships in a scene (a VLM "brain").
2. **World generation** — producing physically plausible video/image worlds from text, image, video, or action inputs.
3. **Action generation** — emitting *numerical* robot actions (joint angles, gripper positions, trajectory points) that describe how an embodiment should move to complete a task.

Before Cosmos 3, you juggled Cosmos-Predict (world generation), Cosmos-Transfer (controlled generation), Cosmos-Reason (scene understanding), and Cosmos-Policy (policy generation) as distinct checkpoints with separate inference stacks. Cosmos 3 collapses all of that into **one forward pass**.

Two sizes shipped at launch:

| Model | Per-tower params | Total (both towers) | Target hardware | Use case |
| --- | --- | --- | --- | --- |
| **Cosmos 3 Nano** | 8B reasoner + 8B generator | ~16B | Workstation (e.g., RTX PRO 6000 Blackwell) | Real-time robotics inference, edge/workstation |
| **Cosmos 3 Super** | 32B reasoner + 32B generator | ~65B | Datacenter (Hopper / Blackwell) | Large-scale synthetic data generation, research, top benchmark quality |

> Note on naming: NVIDIA describes Nano as "the 8B model" (per tower) while Hugging Face lists it as 16B (sum of both towers). Same for Super (32B per tower / ~65B total). Both numbers refer to the same checkpoints.

## 2. Architecture

### 2.1 Mixture-of-Transformers (MoT), two towers

Cosmos 3 is built on a **Mixture-of-Transformers** backbone organized as two connected "towers":

- **Reasoner tower (autoregressive / VLM).** Takes multimodal observations (text, image, video, audio, action) and interprets them via next-token prediction. This is the "brain" that reasons *before* anything is generated. It can be called **independently** (e.g., as a pure VLM for scene understanding or Q&A).
- **Generator tower (diffusion).** Produces future observations (video/image/audio) and action sequences via iterative denoising, **conditioned on the reasoner's understanding**. Guided generation always activates **both** towers; information flows **unidirectionally** from reasoner → generator.

### 2.2 How modalities are encoded

Each modality is encoded by a dedicated encoder, then projected into a **shared representation space**:

- **ViT** — visual *understanding* (feeding the reasoner).
- **VAE** — visual/audio *generation* (feeding/decoding the generator).
- **Domain-aware vectors** — action encoding (joint angles, gripper state, trajectories).

### 2.3 The dual-subsequence trick

Internally, the token sequence is split into two interacting subsequences within each transformer layer:

- An **autoregressive (AR) subsequence** — reasoning/understanding via next-token prediction.
- A **diffusion (DM) subsequence** — generation via iterative denoising.

AR and DM tokens use **separate parameter sets** for each layer but interact via **joint attention**. This is the mechanism that lets one model behave as a VLM, a video generator, a forward/inverse dynamics model, or a robot policy — **without any architectural change**, just by which subsequences are active.

### 2.4 Supported input → output modality combinations

| Input | Output | Resulting "model" |   |
| --- | --- | --- | --- |
| Text | Image | Physically-plausible image generation |   |
| Text \ | Image | Video | World model for prediction |
| Text \ | Video | Video | World model for rare/edge-case video data |
| Text \ | Image \ | Video | Text |
| Action \ | Video \ | Text | Video |
| Text \ | Video | Action | Inverse dynamics model |
| Video \ | Text | Video **&** Action | World-action / policy model for robot learning |

The last row — **Video+Text → Video & Action** — is the "world action model" end state: the model both imagines the future *and* outputs the actions to get there.

## 3. The "Transfer" Part (Deep Dive)

### 3.1 What "Transfer" means in the Cosmos world

In Cosmos vocabulary, **Transfer = controlled/conditioned world generation**. You provide **structured spatial control inputs** (depth maps, edge/Canny maps, semantic segmentation, blur/visual maps, and — in AV/robotics variants — LiDAR and HD maps), and the model generates a photorealistic video that **obeys that structure** while varying appearance (lighting, weather, textures, materials, backgrounds).

It is conceptually a **multi-ControlNet for video world models**. Its purpose is **data augmentation** for Physical AI:

- **Sim2Real augmentation** — take a cheap, low-fidelity 3D simulation (e.g., from Isaac Sim / CARLA), extract its geometry as control signals, and "re-skin" it into photoreal, diverse variants. Minimizes the need for high-fidelity 3D assets.
- **Real2Real augmentation** — take a real sensor capture (e.g., a dashcam clip) and generate diverse variations (snowstorm, night, seasons) while preserving scene geometry.

The payoff is **world-state diversity at scale**: one captured or simulated clip becomes hundreds of labeled training variants across environments, lighting, and weather — closing the Sim2Real gap for AV and robot policies.

### 3.2 The Transfer lineage (important for versioning)

| Version | Date | What it added |
| --- | --- | --- |
| **Cosmos-Transfer1** | early 2025 | First multimodal ControlNet WFM; later add-ons: DiffusionRenderer, synthetic **LiDAR** point-cloud generation, HD-map/AV control |
| **Cosmos-Transfer2.5** | Oct 6, 2025 | Built on **Cosmos-Predict2.5**; **multi-controlnet** (RGB, depth, seg, edge, blur); JSON `controlnet_specs`; distilled **Edge** model (~7.7× faster); AV + robot multiview variants |
| **Cosmos 3** | Jun 1, 2026 | Transfer capability **folded into the omni-model's generator tower** (video+control → video). The standalone `cosmos-transfer2.5` repo is now **limited-maintenance**; new work targets `nvidia/Cosmos` |

**Practical takeaway:** If you specifically need the mature, well-documented **multi-ControlNet transfer workflow** (depth/edge/seg/blur with per-control weights and spatiotemporal masks), **Cosmos-Transfer2.5** is still the most complete, battle-tested implementation today. Cosmos 3 is the strategic future and does control-conditioned generation as *one of many* modes, but its transfer tooling is newer. Both are covered below.

### 3.3 Cosmos-Transfer2.5 model family

- **Cosmos-Transfer2.5-2B (general)** — trained from scratch for Physical AI/robotics. General + distilled checkpoints.
- **Cosmos-Transfer2.5-2B/auto** — post-trained for Autonomous Vehicles; multiview checkpoints (e.g., CARLA → photoreal SDG augmentation).
- **Cosmos-Transfer2.5-2B/robot-multiview-control** — control-conditioned checkpoints for robot multiview; supports 4 control types (depth, edge, visual blur, segmentation).

### 3.4 How Transfer conditioning works mechanically

4. **Control extraction.** Each control modality (depth, edge, segmentation, blur) is either supplied as a pre-computed control video **or computed on the fly** from the input video.
5. **Per-control weighting.** Each modality gets a `control_weight` (0–1+). Higher = the output adheres more strictly to that structure. You mix modalities (e.g., depth 0.5 + edge 1.0 + seg 1.0).
6. **Spatiotemporal masking.** Optional binary MP4 masks restrict a control to certain **regions/time**: white pixels = apply control there, black = ignore.
7. **Guidance.** A classifier-free-guidance-style `guidance` scalar (e.g., 3) trades prompt adherence vs. diversity.
8. **Denoising.** The diffusion process generates the video honoring the weighted controls; distilled models cut sampling steps (e.g., 4) for ~7.7× speedups on short 93-frame clips.

### 3.5 Transfer code — Cosmos-Transfer2.5 (the mature path)

**A. Single-control inference (depth), one GPU:**

```bash
python examples/inference.py \
  -i assets/robot_example/depth/robot_depth_spec.json \
  -o outputs/depth
```

**B. Multi-GPU (8 GPUs) via torchrun:**

```bash
torchrun --nproc_per_node=8 --master_port=12341 \
  examples/inference.py \
  -i assets/robot_example/depth/robot_depth_spec.json \
  -o outputs/depth
```

**C. Multi-control spec (**`**controlnet_specs**`** JSON)** — the heart of Transfer; mix modalities with per-control weights; any control without a `control_path` is computed on the fly:

```json
{
  "prompt_path": "assets/robot_example/robot_prompt.json",
  "output_dir": "outputs/robot_multicontrol",
  "video_path": "assets/robot_example/robot_input.mp4",
  "guidance": 3,
  "depth": {
    "control_path": "assets/robot_example/depth/robot_depth.mp4",
    "control_weight": 0.5
  },
  "edge": {
    "control_path": "assets/robot_example/edge/robot_edge.mp4"
  },
  "seg": {
    "control_path": "assets/robot_example/seg/robot_seg.mp4",
    "control_weight": 1.0
  },
  "vis": {
    "control_weight": 0.5
  }
}
```

**D. Region-restricted control with a spatiotemporal mask** (white = apply, black = ignore):

```json
{
  "depth": {
    "control_path": "assets/robot_example/depth/robot_depth.mp4",
    "mask_path": "/path/to/depth/mask.mp4",
    "control_weight": 0.5
  }
}
```

**E. Distilled Edge model (low-latency, 4 steps, short clips only):**

```json
{
  "name": "robot_edge",
  "prompt_path": "/path/to/prompt/robot_prompt.txt",
  "video_path": "/path/to/input/robot_input.mp4",
  "guidance": 3,
  "num_steps": 4,
  "edge": { "control_path": "/path/to/edge/robot_edge.mp4", "control_weight": 1.0 }
}
```

```bash
# 8 GPUs, distilled edge
torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py \
  -i assets/robot_example/distilled/edge/robot_edge_spec.json \
  -o outputs/distilled/edge \
  --model=edge/distilled
```

**Transfer2.5-2B hardware & throughput (single-GPU, 720p, ~93 frames):**

| GPU | 93-frame gen time | End-to-end (121-frame input, 2 chunks) |
| --- | --- | --- |
| B200 | 92.3 s | 186.9 s |
| H100 PCIe | 264.1 s | 533.6 s |
| H100 NVL | 445.5 s | 895.3 s |
| H20 | 683.7 s | 1370.4 s |

VRAM: **Cosmos-Transfer2.5-2B ≈ 65.4 GB** for single-GPU inference. The **distilled Edge** model is ~7.4–7.8× faster than base diffusion (single-GPU B200: ~180 s base → ~24 s distilled).

### 3.6 Transfer / controlled generation in Cosmos 3 (the future path)

In Cosmos 3, controlled generation is expressed through the unified generator tower using the Diffusers integration (`Cosmos3OmniPipeline`). Video- and image-conditioned generation are first-class modes (Text|Video → Video, Text|Image → Video). Text-to-image single-frame example:

```python
import torch
from diffusers import Cosmos3OmniPipeline

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16, device_map="cuda"
)

prompt = (
    "A medium shot of a modern robotics research laboratory with white walls and a gray floor. "
    "A robotic arm with a metallic finish is mounted on a clean white workbench, its gripper "
    "positioned above a row of small colored objects. The scene is brightly lit by overhead "
    "fluorescent lights."
)

result = pipe(prompt=prompt, num_frames=1, height=720, width=1280)
result.video[0].save("cosmos3_t2i.jpg", format="JPEG", quality=85)
```

For the full transfer-style workflow (depth/edge/seg conditioning, weights, masks) with production support **today**, Transfer2.5's `controlnet_specs` remains the reference; Cosmos 3's control-conditioning tooling lives in the `nvidia/Cosmos` framework and the Cosmos Cookbook.

## 4. Deployment & Optimization

- **Where to get it:** [build.nvidia.com](http://build.nvidia.com/) (hosted try), Hugging Face (`nvidia/Cosmos3-Nano`, `nvidia/Cosmos3-Super`), GitHub (`nvidia/Cosmos`, `NVIDIA/Cosmos-Framework`), and **NIM microservices**.
- **NIM microservices:** the **Cosmos 3 Reasoner NIM** is available at launch; the **Generator NIM** is "coming soon" (a real gap for turnkey generation serving).

```bash
# Cosmos 3 Nano Reasoner NIM (use NIM_MODEL_SIZE=super for the 32B)
docker run --gpus=all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_MODEL_SIZE=nano \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/cosmos3-reasoner:latest
```

- **Quantization:** BF16, FP8, or **NVFP4** (4-bit float) — NVFP4 gives up to ~2× inference speedup.
- **Serving stack:** vLLM (continuous batching, paged attention, tensor parallelism); Nano runs with **vLLM-omni** + **NVIDIA Dynamo**.
- **Efficient Video Sampling (EVS):** prunes redundant video tokens fed to the VLM (chunk-level), accelerating the reasoner — smaller GPUs benefit most.

## 5. Competitive Edges

9. **True unification (omni-model).** One model + one inference pipeline replaces four (Predict/Transfer/Reason/Policy). No cross-model orchestration; reasoning and generation share context in a single forward pass.
10. **Native action generation.** Unlike pure video generators, Cosmos 3 emits *numerical* actions (joint angles, gripper, trajectories) — forward/inverse dynamics model and robot policy. The "world-action model" differentiator vs. text-to-video peers (Sora, Runway, Veo, LTX).
11. **Openness.** Open weights + **OpenMDW 1.1** (Linux Foundation) model-centric license covering weights, architecture, docs, datasets, benchmarks, code; source under Apache 2. Six open **SDG datasets** and open **post-training recipes** ship with it.
12. **Physical-AI specialization.** Built for physics plausibility, multi-view consistency, sensor simulation, Sim2Real/Real2Real augmentation — not general-purpose "pretty video."
13. **Transfer / multi-ControlNet control.** Fine-grained spatial control (depth/edge/seg/blur/LiDAR/HD-map) with per-control weights and spatiotemporal masks — a mature data-augmentation engine most competitors lack for video.
14. **Ecosystem gravity.** Deep ties to Omniverse/Isaac Sim, NIM, TensorRT, Diffusers, the Cosmos Cookbook, and the **Cosmos Coalition** (Black Forest Labs, Runway, LTX, Skild AI, Agile Robots, Generalist).
15. **Efficiency spread.** Distilled models (~7.7×), NVFP4, and a workstation-class **Nano** tier make it deployable outside the datacenter.
16. **Benchmark leadership.** Reported open-source SOTA / leading on VANTAGE-Bench, TAR, R-Bench, PAI-Bench, Physics-IQ, RoboLab, and Artificial Analysis leaderboards; plus a rigorous human-eval framework (**HUE**) using atomic yes/no fact verification.

## 6. Drawbacks & Limitations

17. **Heavy compute.** Transfer2.5-2B needs ~65 GB VRAM and *minutes* per clip on a single H100. Cosmos 3 Super (~65B total) is datacenter-only. Real-time generation isn't realistic without distillation and top-tier GPUs.
18. **Latency for generation.** Even optimized, video generation is far from interactive; distilled models trade this for **short clips only** (~93 frames), limiting long-horizon rollouts to sliding-window stitching.
19. **NVIDIA hardware lock-in.** Best (often only practical) performance assumes CUDA + Hopper/Blackwell, NIM, TensorRT, NVFP4, Dynamo. Portability to non-NVIDIA accelerators is poor.
20. **Generator NIM not yet available.** At launch only the **Reasoner NIM** ships; turnkey production *generation* serving is pending — self-host the generator via the framework for now.
21. **Ecosystem churn / deprecation risk.** The `cosmos-transfer2.5` repo is already **limited-maintenance** eight months after release; repos moved orgs (`nvidia-cosmos` → `nvidia/Cosmos`). Fast iteration means integration rework.
22. **Imperfect physics understanding.** Benchmarks like Physics-IQ exist precisely because generative video models often achieve visual realism *without* truly understanding physics; expect plausible-but-wrong dynamics in edge cases and a residual Sim2Real gap.
23. **Prompt sensitivity.** Quality depends on long, detailed narrative prompts (video) or concise spatially-grounded prompts (action); you'll want the prompt-upsampling templates.
24. **Two-tower compute overhead.** Guided generation **always activates both towers**, so you pay reasoner cost even when you mostly want generation (though reasoner-only inference is available).
25. **Licensing nuance.** OpenMDW is permissive, but earlier Cosmos models used the **NVIDIA Open Model License**, and downloads may pull third-party components with their own terms — legal review needed for commercial deployment; guardrails add operational constraints.
26. **Evaluation maturity.** Because top video models saturate automated leaderboards, NVIDIA leans on its own HUE human-eval; cross-vendor apples-to-apples comparison remains hard.

## 7. Quick Decision Guide

- **Need production-grade multi-ControlNet transfer (depth/edge/seg/blur, masks, weights) *****now*****, for AV/robot SDG:** use **Cosmos-Transfer2.5** (mature, documented) while planning migration.
- **Want one model for reason + generate + act, and future-proofing:** adopt **Cosmos 3** (`nvidia/Cosmos`), Nano for workstation/robotics, Super for datacenter SDG.
- **Only need scene understanding / VLM reasoning:** deploy the **Cosmos 3 Reasoner NIM** (available today, vLLM-optimized).
- **Latency-bound on short clips:** use **distilled** checkpoints + **NVFP4**.

## Sources

- [NVIDIA Cosmos 3 launch — NVIDIA Newsroom](https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-3-the-open-frontier-foundation-model-for-physical-ai)
- [How Cosmos 3 Helps Physical AI Think Before It Acts — NVIDIA Blog](https://blogs.nvidia.com/blog/cosmos-3-physical-ai-open-world-foundation-model/)
- [Develop Physical AI Reasoning, World, and Action Models with NVIDIA Cosmos 3 — NVIDIA Technical Blog](https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3)
- [Welcome NVIDIA Cosmos 3 — Hugging Face Blog](https://huggingface.co/blog/nvidia/cosmos-3-for-physical-ai)
- [Cosmos 3 Technical Report (PDF)](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)
- [Cosmos-Transfer2.5 — GitHub](https://github.com/nvidia-cosmos/cosmos-transfer2.5)
- [Cosmos-Transfer2.5 Inference Guide](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/inference.md)
- [Cosmos 3 / main repo — ](https://github.com/nvidia/cosmos)[github.com/nvidia/Cosmos](http://github.com/nvidia/Cosmos)
- [Diffusers Cosmos 3 pipeline docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cosmos3)
- [World Simulation with Video Foundation Models for Physical AI — arXiv:2511.00062](https://arxiv.org/abs/2511.00062)