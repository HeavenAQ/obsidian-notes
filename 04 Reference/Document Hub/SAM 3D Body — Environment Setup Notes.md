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
Setup of the `sam-3d-body` environment, following the upstream [INSTALL.md](http://install.md/).

## TL;DR

- **Software environment: COMPLETE.** PyTorch (CUDA), all Python deps, a freshly compiled `detectron2`, plus optional `MoGe` and `SAM3` all import and work.
- **Model checkpoints: COMPLETE.** The `sam-3d-body-dinov3` weights are downloaded and load successfully.

## Key difference from upstream [INSTALL.md](http://install.md/)

The upstream guide assumes **conda + pip with a system CUDA toolkit**. This machine instead uses a `**uv**`**-managed venv** (`.venv/`, Python 3.11.12) and the **CUDA toolchain comes from pip wheels**, not the system. Everything below was adapted to that.

## Environment facts

| Thing | Value |
| --- | --- |
| venv | `.venv/` (created by `uv`, Python 3.11.12) |
| Package manager | `uv pip ...` (note: `pip` is **not** present in the venv) |
| GPU | NVIDIA RTX 6000 Ada (compute capability 8.9) |
| Driver | 580.159.03 (supports CUDA 13.0) |
| torch | `2.12.1+cu130` — built against **CUDA 13.0** |
| System CUDA toolkit | `/usr/local/cuda` → `/opt/cuda/12.8.1` (**12.8**, version-mismatched, unused) |

## What was already done before this session

- venv created with the full step-3 dependency list installed.
- `torch`/`torchvision` (cu130) installed, CUDA available.

## What this session did

### 1. detectron2 (required) — the hard part

Building detectron2 needs `nvcc` whose major version matches the torch CUDA build (13.x). The only system `nvcc` is **12.8**, which fails torch's CUDA version check. Fix: install the CUDA compiler from pip wheels, pinned to **13.0** to match the torch runtime (`nvidia-cuda-runtime` is `13.0.96`).

```bash
# nvcc + toolkit headers, all pinned to the 13.0 line (NOT the latest 13.3,
# which mismatches the 13.0 runtime headers and trips the CCCL compat check)
uv pip install \
  "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-crt==13.0.*" \
  "nvidia-nvvm==13.0.*"      "nvidia-cuda-cccl==13.0.*"

# The wheel ships libcudart.so.13 but no unversioned symlink the linker needs:
CU=.venv/lib/python3.11/site-packages/nvidia/cu13
ln -sf libcudart.so.13 "$CU/lib/libcudart.so"

# Build against the pip CUDA toolchain
export CUDA_HOME="$PWD/$CU"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.9"          # RTX 6000 Ada
uv pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
  --no-build-isolation --no-deps
```

Three (four) failures cleared, in order:

1. `CUDA version mismatch (12.8 vs 13.0)` → use pip `nvcc` (13.x), not system 12.8.
2. `fatal error: nv/target` → install `nvidia-cuda-cccl`.
3. `"CUDA compiler and CUDA toolkit headers are incompatible"` → pin the nvcc stack to **13.0** (it had resolved to 13.3).
4. `-lcudart not found` → create the `libcudart.so` symlink.

### 2. MoGe (optional — FoV estimator)

```bash
uv pip install git+https://github.com/microsoft/MoGe.git
```

### 3. SAM3 (optional — human segmentor)

```bash
git clone https://github.com/facebookresearch/sam3.git
uv pip install -e ./sam3 --no-build-isolation
uv pip install decord psutil
```

> Note: installing `sam3` **downgraded numpy 2.4.6 → 1.26.4** (its pin). All numpy-2-built packages (pandas, opencv, torchvision, scikit-image) still import and work at 1.26.4, so this is left as-is.

## Verification

All of the following import cleanly in `.venv`:

```javascript
numpy 1.26.4
torch 2.12.1+cu130  (cuda available: True)
detectron2 0.6  (compiled _C extension loads)
moge 2.0.0
sam3 0.1.0
decord 0.6.0 / psutil 7.2.2
cv2 / pyrender / timm / hydra / pandas / skimage / torchvision
```

`sam_3d_body` itself is a local package (not pip-installed by design); run the demo from inside `sam-3d-body/` (`python demo.py --help` works).

## Checkpoints

**Done.** The `dinov3` checkpoint is downloaded and verified.

- Auth via `HF_TOKEN` in `top-down-view/.env` (account `Heavenz`, write scope) — this account *is* authorized for the gated repos. (An earlier `KokiHoyashita` read token was **not** authorized and got 403s.)
- `checkpoints/sam-3d-body-dinov3/` contains `model.ckpt` (~2.0 GB), `assets/mhr_model.pt` (~664 MB), and `model_config.yaml`; both weights load.

To re-download (or fetch the `vith` variant), source the token first:

```bash
cd top-down-view
set -a; . ./.env; set +a
cd sam-3d-body
../.venv/bin/python -c "import os; from huggingface_hub import snapshot_download; \
  snapshot_download('facebook/sam-3d-body-dinov3', \
  local_dir='checkpoints/sam-3d-body-dinov3', token=os.environ['HF_TOKEN'])"
```

## Running the demo (after checkpoints are available)

```bash
cd sam-3d-body
../.venv/bin/python demo.py \
  --image_folder <imgs> \
  --checkpoint_path checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
```