---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-06-01T14:01:00
Status: Not started
Last updated time: 2026-06-01T15:54:00
Last edited by: Heaven Chen
Category:
  - GPU
  - Hardware
---
## `nvidia-smi`

```javascript
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.08             Driver Version: 535.161.08   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-80GB          On  | 00000000:07:00.0 Off |                    0 |
| N/A   36C    P0              70W / 400W |  18953MiB / 81920MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-SXM4-80GB          On  | 00000000:0F:00.0 Off |                    0 |
| N/A   43C    P0              94W / 400W |  36539MiB / 81920MiB |     36%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA A100-SXM4-80GB          On  | 00000000:47:00.0 Off |                    0 |
| N/A   31C    P0              64W / 400W |      0MiB / 81920MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA A100-SXM4-80GB          On  | 00000000:4E:00.0 Off |                    0 |
| N/A   31C    P0              61W / 400W |      0MiB / 81920MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A   1942269      C   ...inetuning-factory/.venv/bin/python3    18940MiB |
|    1   N/A  N/A   2375410      C   .venv/bin/python                          36460MiB |
+---------------------------------------------------------------------------------------+

```

## A. The 4-Layer CUDA Stack

### 1. GPU Hardware

- Fixed compute capability with a certain GPU code (e.g. `sm_80` for A100 GPU)

### 2. GPU Kernel Driver

- Installed by the GPU cluster admin. You usually can’t change this.
- Has driver version (e.g. `535.161.08`)

### 3. CUDA toolkit / runtime

- the libraries and compilers
- `nvcc`, `libcudart`, etc
- some versions of toolkit will ship the version into the environment (pytorch wheel, conda env, or a system install). 
- what you have control over

### 4. Your Application

- `pytorch`, `torchvision`, detectron2’s `_C`

> **NOTE:**
1. toolkit version ≤ GPU kernel driver
2. application should match the toolkit version

## B. Version Naming Convention

- **CUDA toolkit version** (`MAJOR.MINOR.PATCH`)
    - e.g. `12.1` → Major: 12 | Minor: 1
- **Driver version** (`XXX.XXX.XXX`)
    - `≥ 525.60.13` → CUDA 12.0
    - `535.x` → CUDA 12.2
    - `550.x` → CUDA 12.4
    - `≥ 450.80` → CUDA 11.x.
>  `nvidia-smi` prints the driver version and the *max* "CUDA Version" it supports (a ceiling, not what's installed).
- **PyTorch CUDA build tag (**`**+cuXYZ**`**)**
    - `XYZ` → the CUDA toolkit version that4