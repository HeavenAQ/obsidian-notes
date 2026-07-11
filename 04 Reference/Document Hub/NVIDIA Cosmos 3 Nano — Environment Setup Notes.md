---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-06T19:01:00
Status: Done
Last updated time: 2026-07-06T19:01:00
Last edited by: Heaven Chen
Category:
  - AIGC
  - GPU
  - Linux
---
Setup of NVIDIA **Cosmos 3 Nano** (`nvidia/Cosmos3-Nano`) on this machine, following the upstream [nvidia/cosmos README](https://github.com/nvidia/cosmos) (Generator · Diffusers backend).

## TL;DR

- **Disambiguation:** "cosmos3 nano" = **Cosmos 3 Nano**, the smallest *released* Cosmos 3 checkpoint (`nvidia/Cosmos3-Nano`, 16B = 8B reasoner + 8B generator). Code lives in the umbrella repo [**github.com/nvidia/cosmos**](http://github.com/nvidia/cosmos). (A smaller "Edge" 2B variant is announced but **not yet released**.)
- **Backend chosen:** **Diffusers Generator** (`Cosmos3OmniPipeline`). It loads the *full* checkpoint (reasoner + diffusion + media tokenizers) and is the world/video-generation surface — the best fit for a dataset-generation project, and a clean pip install.
- **Software environment: COMPLETE.** Isolated `uv` venv (Python 3.13), `torch 2.12.1+cu130`, `diffusers` (git), `transformers 5.12.1`, `cosmos_guardrail`. All Cosmos 3 classes import and CUDA is available.
- **Checkpoint:** `nvidia/Cosmos3-Nano` (~35 GB) downloaded into the **shared** `HF_HOME`, not the project dir.

## Which model / repo (research summary)

Cosmos 3 was released 2026-05-31. The relevant options and why Nano was chosen:

| Candidate | What it is | Verdict |
| --- | --- | --- |
| `nvidia/Cosmos3-Nano` (16B) | Smallest released Cosmos 3 omnimodal world model | **Chosen** — matches "cosmos3 nano" exactly |
| `nvidia/Cosmos3-Super` (64B) | Frontier-scale variant | Too big; not "nano" |
| Cosmos3-Edge (2B) | Announced smaller tier | **Not released yet** |
| `cosmos-predict2.5` / `cosmos-transfer2.5` | Previous (2.5) generation; "2B/Nano" size | Superseded by Cosmos 3; repos now limited-maintenance |

The **Transfer** capability (control inputs: depth / segmentation / edge / blur — the `control_depth.mp4` / `control_seg.mp4` style this project cares about) exists in Cosmos 3 as a *Generator* workflow, but is only wired up through the heavier **Cosmos Framework** backend (`git clone NVIDIA/cosmos-framework` + `uv sync --all-extras --group=cu130-train`, which pulls lerobot/training extras). It was **not** installed in the first pass to keep the env light (see Follow-ups + the Cosmos Framework section below).

## Environment facts

| Thing | Value |
| --- | --- |
| Umbrella repo | `third-party/cosmos` = `github.com/nvidia/cosmos` @ `58c76de` (2026-06-29) |
| venv | `third-party/cosmos/.venv` (isolated, `uv`, **Python 3.13.3**) |
| Backend | Diffusers Generator (`Cosmos3OmniPipeline`) |
| torch | `2.12.1+cu130` (CUDA 13.0), `torch.cuda.is_available() == True` |
| GPU | 3× RTX 6000 Ada (48 GB, cc 8.9); driver 580 (CUDA 13 capable) |
| HF cache | **shared** `HF_HOME=/82e9e7ee-…/hf-home` (762 GB free on `nvme1n1`) |

> Kept **separate** from the project `.venv` (Python 3.11, sam3/sam-3d-body/detectron2) to avoid disturbing that working environment. Cosmos wants Python 3.13.

## Install commands

```bash
cd third-party
git clone https://github.com/nvidia/cosmos.git cosmos     # @ 58c76de

cd cosmos
uv venv --python 3.13 --seed --managed-python              # -> .venv (Python 3.13.3)

# Diffusers Generator backend. --torch-backend=cu130 matches the driver (CUDA 13).
VIRTUAL_ENV="$PWD/.venv" uv pip install --torch-backend=cu130 \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  accelerate av cosmos_guardrail huggingface_hub \
  imageio imageio-ffmpeg torch torchvision transformers
```

Installed (key versions): `torch 2.12.1+cu130`, `torchvision 0.27.1+cu130`, `diffusers 0.39.0.dev0` (git), `transformers 5.12.1`, `accelerate 1.14.0`, `cosmos_guardrail 0.3.1`, `huggingface_hub 1.21.0`, `av 17.1.0`, `safetensors 0.8.0`.

No system CUDA toolkit / nvcc needed (unlike the detectron2 build in the SAM setup) — the Diffusers path is pure prebuilt wheels.

## Checkpoints

`nvidia/Cosmos3-Nano` is **public / not gated** (OpenMDW-1.1 license). ~35 GB across 67 files (7× ~5 GB diffusion transformer shards, VAE, vision encoder, sound + text tokenizers). Downloaded into the **shared** HF cache (per server rules, pretrained models must NOT live under the project dir):

```bash
cd top-down-view
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)   # account "Heavenz"
third-party/cosmos/.venv/bin/python -c \
  "import os; from huggingface_hub import snapshot_download; \
   snapshot_download('nvidia/Cosmos3-Nano', token=os.environ['HF_TOKEN'], max_workers=8)"
```

Lands in `$HF_HOME/hub/models--nvidia--Cosmos3-Nano`. `HF_HOME` is exported in `~/.bashrc`, so `from_pretrained("nvidia/Cosmos3-Nano")` resolves here automatically.

## Verification / smoke test

```bash
cd third-party/cosmos
CUDA_VISIBLE_DEVICES=2 .venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
from diffusers import Cosmos3OmniPipeline
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration
import cosmos_guardrail
print("Cosmos 3 classes import OK")
PY
```

Result: **PASS** — `torch 2.12.1+cu130`, `cuda available: True`, `Cosmos3OmniPipeline` / `Cosmos3OmniForConditionalGeneration` / `cosmos_guardrail` all import. (Two `[ERROR] ... not documented` lines from transformers are harmless docstring-lint warnings.)

Per task constraints, **no full generation** was run (compute-heavy). For a real generation later (Diffusers text-to-video), see the README quickstart — expect long per-step times; use `CUDA_VISIBLE_DEVICES=0` or `2`.

## Follow-ups

- **Transfer / control (depth, segmentation, edge, blur)** — the feature most relevant to this project — needs the **Cosmos Framework** backend: `git clone https://github.com/NVIDIA/cosmos-framework packages/cosmos3` then `GIT_LFS_SKIP_SMUDGE=1 uv sync --all-extras --group=cu130-train`, plus system libs (`libgl1 libglib2.0-0 libxcb1 ffmpeg`) for the guardrail's OpenCV path.
- **Reasoner-only** (text/vision → text, lighter) is available in this same venv via `transformers` (`Cosmos3OmniForConditionalGeneration`) with no extra install.
- **Guardrail:** Generator enables safety guardrails by default. Disable per-call with `enable_safety_checker=False` in `Cosmos3OmniPipeline(...)`.

## Cosmos Framework backend — Transfer (control-conditioned video)

The **second** backend, added 2026-07-01: the heavier **Cosmos Framework** used to actually run **Cosmos 3 Transfer** (control → video), following the upstream cookbook `run_video_transfer_with_cosmos_framework.ipynb`. Kept fully separate from the Diffusers env above.

### Install

| Thing | Value |
| --- | --- |
| Repo | `third-party/cosmos-framework` = `github.com/NVIDIA/cosmos-framework` @ `**24300b4**` ("Add multi-control transfer inference", #69) |
| venv | `third-party/cosmos-framework/.venv` (isolated, `uv`) |
| uv group | `**cu130-train**` (worked first try; matches the box's CUDA 13) |
| torch | `2.10.0+cu130` (CUDA 13.0), `torch.cuda.is_available() == True` |
| HF cache | shared `HF_HOME=/82e9e7ee-…/hf-home` (Cosmos3-Nano reused; the run topped up the cache 13G→40G to complete all shards) |

```bash
cd third-party
git clone https://github.com/NVIDIA/cosmos-framework.git cosmos-framework   # @ 24300b4
cd cosmos-framework
uv sync --all-extras --group=cu130-train      # builds megatron-core, flash-attn, transformer-engine, natten … (~several min)
uv pip install imageio imageio-ffmpeg
# Verify entrypoint loads (shows depth/seg/edge/blur/wsm transfer hints):
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m cosmos_framework.scripts.inference --help
```

### Control modality + frame/resolution adaptation

- **Modality:** single **depth** control (`control_depth.mp4`). The cookbook runs each control type independently.
- **Spec format:** transfer is triggered by a top-level hint key in the sample JSON (`"depth": {"control_path": …}`). One hint ⇒ the model auto-applies the tuned depth defaults: `guidance=3.0`, `control_guidance=1.5`, `shift=10.0`, `num_steps=35`, `num_conditional_frames=1`, and a built-in negative prompt. It also appends a deterministic "Follow the depth control video precisely…" directive (`emphasize_control_in_prompt=True`).
- **Frame adaptation:** source controls are 512×512 @ 10 fps, 101–300 frames. Trimmed each to the **first 100 frames** (10 s) with ffmpeg → `control_depth_10s.mp4` (`-frames:v 100 -r 10 -c:v libx264`), and set `num_video_frames_per_chunk=101` so the whole clip is one autoregressive chunk.
- **Resolution:** spec `resolution="480"`, `aspect_ratio="1,1"` → the model buckets a square 1:1 clip to **640×640** output (100 frames, 10 fps).
- Specs: `specs/cosmos_transfer/{A,B,C}.json` — identical except `control_path`. The `prompt` field is byte-identical across all three (verified same md5).

### Fixed prompt (all 3, byte-identical)

> A factory worker in a modern industrial workshop performing hands-on manual tasks at a workstation, wearing work uniform, machinery and equipment around, realistic natural lighting, viewed from a top-down overhead camera.

### Exact inference command (per clip, single GPU)

```bash
cd third-party/cosmos-framework
export HF_HOME=/82e9e7ee-18e9-4474-b87c-64a92999d1c1/hf-home
export HF_TOKEN=$(grep HF_TOKEN ../../.env | cut -d= -f2)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i ../../specs/cosmos_transfer/C.json \
    -o ../../outputs/cosmos_transfer/C \
    --checkpoint-path Cosmos3-Nano --seed 2026 --no-guardrails
```

`--no-guardrails` avoids pulling the extra guardrail models (Cosmos-Guardrail1, Qwen3Guard, RetinaFace); the prompt is benign. Drop the flag to enable them.

### Results

Output (640×640, 100 frames, 10 fps, `vision.mp4`): **C** → `outputs/cosmos_transfer/C/clip_C/vision.mp4` ✅ generated.

Wall time (single RTX 6000 Ada, GPU 2): sampling ≈ 24 min (35 UniPC steps @ ~41 s/step) + ~5 min model load ⇒ **Clip C ≈ 30 min total**. Nothing OOM'd; no params reduced (peak GPU mem ~45 GB of 48 GB). **A and B were intentionally not generated** — specs and trimmed `control_depth_10s.mp4` are ready; re-run with `-i .../A.json -o .../A`.

## Server model-storage rules (compliance)

- HF weights → shared `HF_HOME=/82e9e7ee-…/hf-home` (auto via `~/.bashrc`). ✅ Cosmos3-Nano is there, not under the project dir.
- Only the venv/code lives under `third-party/cosmos/`. ✅
- Other shared roots: `…/shared-gguf-models` (`GGUF_MODELS_DIR`), `…/models/`, `…/datasets/`.