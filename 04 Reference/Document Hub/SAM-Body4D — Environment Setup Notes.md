---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-06T19:02:00
Status: Done
Last updated time: 2026-07-06T19:02:00
Last edited by: Heaven Chen
Category:
  - CV
  - Pose-Estimation
  - Linux
---
Setup of the `sam-body4d` repo ([github.com/gaomingqi/sam-body4d](http://github.com/gaomingqi/sam-body4d)) on the shared RTX 6000 Ada box. Follows the upstream README, adapted to a `uv`-managed venv with a pip/conda CUDA 11.8 toolchain (the README assumes conda + a system CUDA toolkit).

## TL;DR

- **Software environment: COMPLETE.** torch cu118, a freshly compiled `detectron2` (CUDA `_C` extension loads), `sam3`, and the `sam-body4d` package (diffusers/transformers/MoGe/pyrender/decord/...) all import and work.
- **Checkpoints: COMPLETE.** All 6 checkpoint groups downloaded (~20 GB).
- **Smoke test: pipeline runs end-to-end and emits per-frame PLY meshes.** Note the diffusion-VAS *completion* stage needs a GPU with ~24 GB genuinely free.

## Environment facts

| Thing | Value |
| --- | --- |
| Repo | `third-party/sam-body4d/` |
| venv | `third-party/sam-body4d/.venv` (uv, **Python 3.12.3**) |
| Package manager | `uv pip ...` (set `VIRTUAL_ENV=$PWD/.venv`; `uv` at `/home/ollo8/.local/bin/uv`) |
| GPU | NVIDIA RTX 6000 Ada, compute capability **8.9**, 48 GB |
| Driver | 580 (CUDA 13 capable; runs cu118 runtime fine) |
| torch / torchvision | **2.7.1+cu118 / 0.22.1+cu118** (CUDA available: True) |
| detectron2 | **0.6** @ commit `a1ce2f9`, CUDA `_C` extension compiled for sm_89 |
| numpy | 2.2.6 (no downgrade needed) |

## 1. venv + torch

```bash
cd third-party/sam-body4d
export PATH="/home/ollo8/.local/bin:$PATH"
uv venv --python 3.12 --seed
export VIRTUAL_ENV="$PWD/.venv"
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

## 2. detectron2 @ a1ce2f9 — the hard part (cu118 toolchain)

detectron2 must be compiled with an `nvcc` whose CUDA major matches torch (11 for cu118). The system only has CUDA 12.8 nvcc (mismatch → torch's version check fails).

**The pip-wheel recipe in the task brief does NOT work for cu11:**

- `nvidia-cuda-crt-cu11` and `nvidia-nvvm-cu11` **do not exist** on PyPI.
- `nvidia-cuda-nvcc-cu11==11.8.89` is a **partial wheel**: it ships only `ptxas` + headers + `libnvvm`/`libdevice` — **no **`**nvcc**`** frontend, **`**cicc**`**, **`**fatbinary**`**, or **`**nvlink**`. Unusable as a compiler.

**Working solution — a local conda prefix with the full CUDA 11.8 toolkit** (kept inside the repo at `.cuda118/`, so nothing leaks outside `sam-body4d/`):

```bash
conda create -p "$PWD/.cuda118" \
  -c nvidia/label/cuda-11.8.0 -c conda-forge \
  cuda-nvcc=11.8.89 cuda-cudart-dev=11.8.89 cuda-cccl=11.8.89 -y
# cusparse/cublas/cusolver headers are needed by torch's CUDA extension build:
conda install -p "$PWD/.cuda118" -c nvidia/label/cuda-11.8.0 \
  libcusparse-dev libcublas-dev libcusolver-dev cuda-nvrtc-dev -y
```

**Host compiler:** CUDA 11.8 rejects the system gcc-13. The box already has `/usr/bin/gcc-11` + `/usr/bin/g++-11`, which use the normal system include layout (so they find the Debian multiarch Python headers — conda's gcc-11 does *not*, which broke an earlier attempt on `pyconfig.h`). Use the system gcc-11.

**Build:**

```bash
CU="$PWD/.cuda118"
export CUDA_HOME="$CU"
export PATH="$CU/bin:$PATH"
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST="8.9"   # RTX 6000 Ada
export FORCE_CUDA=1
uv pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
  --no-build-isolation --no-deps
```

Errors cleared, in order:

1. `nvidia-cuda-crt-cu11 not found` → those split wheels don't exist for cu11; use conda toolkit.
2. cu11 nvcc wheel had no `nvcc` binary → use conda `cuda-nvcc=11.8.89` (full).
3. `x86_64-linux-gnu/python3.12/pyconfig.h: No such file` → conda gcc-11 sysroot; switch to system `/usr/bin/gcc-11`.
4. `cusparse.h: No such file` → add `libcusparse-dev`/`libcublas-dev`/`libcusolver-dev` (11.8) to the prefix.

Verify: `python -c "import torch, detectron2; from detectron2 import _C"` → OK.

## 3. sam3 + main package

```bash
uv pip install -e models/sam3 --no-build-isolation
uv pip install -e . --no-build-isolation
```

No dependency conflicts; numpy stayed at 2.2.6.

## 4. Checkpoints

```bash
set -a; source ../../.env; set +a          # HF_TOKEN (account Heavenz) + HF_HOME
python scripts/setup.py --ckpt-root /82e9e7ee-18e9-4474-b87c-64a92999d1c1/sam-body4d-ckpts
```

This also generates `configs/body4d.yaml` with `paths.ckpt_root` already set. Downloaded (~20 GB total):

| Checkpoint | Size |
| --- | --- |
| `sam3/sam3.pt` | 3.4 GB |
| `sam-3d-body-dinov3/` (model.ckpt 2.1 GB + assets/mhr_[model.pt](http://model.pt/) 696 MB) | 3.3 GB |
| `moge-2-vitl-normal/model.pt` | 1.3 GB |
| `diffusion-vas-amodal-segmentation/` | 9.4 GB |
| `diffusion-vas-content-completion/` | 7.1 GB |
| `depth_anything_v2_vitl.pth` | 1.3 GB |

## 5. Running the offline pipeline

```bash
export CUDA_VISIBLE_DEVICES=<gpu with >=24 GB free>   # NOT gpu 2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; source ../../.env; set +a
.venv/bin/python scripts/offline_app.py \
  --input_video <mp4 or a dir of frames> \
  --output_dir outputs/<name>
```

Renderer uses headless EGL (`PYOPENGL_PLATFORM=egl` is set in `sam_3d_body/visualization/renderer.py`). **On this multi-GPU headless box the default EGL device is a DRI card we can't open** (`/dev/dri/card4: Permission denied` → `eglInitialize` `GLError 12289`). Fix: pick a working EGL device with `export EGL_DEVICE_ID=4` (probed: IDs 0-3 fail, **4 works**). Without this the run gets all the way through mesh recovery and then crashes at the first render call.

Outputs under `--output_dir`:

- `mesh_4d_individual/<obj_id>/` — per-frame `.ply` meshes
- `focal_4d_individual/<obj_id>/`, cam JSONs
- `rendered_frames/`, `rendered_frames_individual/<obj_id>/`, `images/`, `masks/`

### Memory note (important)

The **diffusion-VAS completion** stage (occlusion recovery, 512x1024 diffusion) peaks at **~24 GB by itself** and does not shrink with fewer frames. On this shared box it OOMs if the chosen GPU has a large co-tenant. Run it on a GPU with **~26 GB+ genuinely free** (e.g. GPU 0 when idle at ~41 GB free). To skip occlusion recovery for fully-visible subjects, set `completion.enable: false` in `configs/body4d.yaml` (much lower memory), at the cost of amodal recovery. `configs/body4d.yaml` also defaults `detection_resolution`/`completion_resolution` to `[512,1024]` — the main memory drivers.

## Smoke-test results (clip C, single person)

Input: a 300-frame @ 10 fps clip; frame subsets extracted to a scratch dir and passed as `--input_video <dir>`. Both runs used `CUDA_VISIBLE_DEVICES=0 EGL_DEVICE_ID=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Run A — completion disabled, 60 frames** → `outputs/clipC_smoke/`: 60 per-frame PLY meshes in `mesh_4d_individual/1/` (binary, ~18.4k colored verts each), 60 cam JSONs, 60 rendered frames, a 4D mp4. 1 person (obj_id 1). Peak ~15 GB. ~2 min.

**Run B — full pipeline, completion enabled, 24 frames** → `outputs/clipC_smoke_full/`: 24 PLY + 24 JSON + rendered frames + 4D mp4. Ran the diffusion-VAS amodal-seg stage (no occlusion window triggered for this fully-visible subject, so no `completion/` files). Peak ~38 GB on an otherwise-idle GPU. ~7 min.

Both exited 0 and emitted the per-frame meshes. The completion stage OOM'd four times earlier when GPU 0/1 had large co-tenant jobs — it needs a GPU with ~30 GB+ actually free.