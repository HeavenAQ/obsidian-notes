---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-06T15:29:00
Status: Done
Last updated time: 2026-07-06T15:29:00
Last edited by: Heaven Chen
Category:
  - CV
  - Survey
  - Theory
---
> Lin, Chen, Liew, Chen, Li, Shi, Feng, Kang (ByteDance Seed). arXiv:2511.10647 (Nov 13, 2025). Project: [depth-anything-3.github.io](http://depth-anything-3.github.io/). Code: ByteDance-Seed/depth-anything-3. HF: depth-anything/* collection.

## 1. TL;DR

Depth Anything 3 (DA3) predicts **spatially consistent geometry from any number of views**, with or without known camera poses. From one image it does monocular depth; from many images it does multi-view depth + camera pose + 3D reconstruction — all from **one model**. Two deliberately minimal design bets: (1) a **single plain transformer** (a vanilla DINOv2 encoder) is enough — no task-specific architecture; (2) a **single depth-ray prediction target** replaces multi-task heads. On a new visual-geometry benchmark it beats VGGT by **+44.3% camera-pose accuracy** and **+25.1% geometric accuracy**, and beats DA2 on monocular depth. Trained only on public academic datasets.

## 2. Key ideas

- **Any-view input.** Accepts 1..N RGB images (or video frames). With more views it produces globally consistent geometry; with poses supplied it uses them (pose-conditioned mode) for even better consistency.
- **Minimal backbone.** A plain ViT (DINOv2) processes all views' tokens jointly — no cost-volumes, no epipolar modules, no specialized multi-view architecture. Cross-view consistency emerges from **alternating within-view / cross-view attention** in the shared transformer.
- **Single depth-ray target.** Instead of separate heads for depth, normals, pose, points, DA3 regresses a per-pixel **depth** plus a per-pixel **camera ray**. Everything else (pose, intrinsics, point cloud) is derived from that one representation.
- **Teacher-student training.** A DA2-style teacher supplies fine-detail monocular pseudo-labels; the DA3 student matches DA2's detail/generalization while adding multi-view geometry.

## 3. Architecture

```plain text
view_1 ─┐
view_2 ─┤ patchify (DINOv2 ViT)        shared plain transformer
 ...    ┼───► tokens per view ─────►  [ within-view attn  ⇄  cross-view attn ]  ─► fused tokens
view_N ─┘        (+RoPE, QK-norm,        (alternating, starts at layer `alt_start`)
                  concat camera token)
                                                    │
                          ┌─────────────────────────┼───────────────────────────┐
                          ▼                         ▼                            ▼
                   DualDPT head              (ray head → pose)          optional Gaussian head
                depth map + ray map      extrinsics + intrinsics        3D Gaussians (gsplat)
```

Components (from the released config):

- **Encoder:** `DinoV2` (e.g., `vitb`/`vitl`/`vitg`), multi-scale features taken from `out_layers` (e.g., `[5,7,9,11]`). Modern touches: **RoPE** (`rope_start`), **QK-normalization** (`qknorm_start`), a concatenated camera/register token (`cat_token`), and **alternating attention** (`alt_start`) that interleaves per-image and cross-image attention so tokens from different views exchange information.
- **Head:** `DualDPT` — a dual-branch DPT dense-prediction head (`output_dim: 2`, `features: 128`, `out_channels: [96,192,384,768]`) producing the **depth** and **ray** maps.
- **Pose:** either derived from the **ray head** (`use_ray_pose=True`, slightly slower, more accurate) or a direct **camera head** regression.
- **Optional Gaussian head:** predicts 3D Gaussians for novel-view synthesis (Giant/Nested).

## 4. The depth-ray representation (the math)

Each pixel $(u,v)$ in view $i$ gets two predictions: a depth $d_i(u,v)$ and a world-space ray defined by origin $\mathbf{o}_i$ (the camera center, shared per view) and a unit direction $\mathbf{r}_i(u,v)$. The 3D point is a plain ray march:

$$
\mathbf{X}_i(u,v) = \mathbf{o}_i + d_i(u,v)\,\mathbf{r}_i(u,v)
$$

Because all views' rays live in a **common world frame**, the per-view point clouds are automatically fused/consistent — no separate alignment step. From the dense ray field you recover camera geometry:

- **Extrinsics:** camera center = $\mathbf{o}_i$; orientation from the arrangement of ray directions.
- **Intrinsics:** fit the focal/principal point to how ray directions fan out across the image plane.

This is why one target subsumes many tasks: monocular depth = read $d$; multi-view depth = read all $d_i$ in the shared frame; pose estimation = read the rays; 3D reconstruction = back-project all $\mathbf{X}_i$.

**Monocular metric depth (DA3Metric):** convert the network output to meters with

$$
\text{metric\_depth} = \text{focal} \cdot \text{net\_output} / 300
$$

where `focal` is in pixels (avg of $f_x,f_y$). The Nested model's output is already in meters.

## 5. Model zoo

| Series | Model | Params | Tasks | License |
| --- | --- | --- | --- | --- |
| Nested | DA3NESTED-GIANT-LARGE(-1.1) | 1.40B | depth, pose, pose-cond, **metric**, 3DGS, sky-seg | CC BY-NC 4.0 |
| Any-view | DA3-GIANT(-1.1) | 1.15B | rel. depth, pose, pose-cond, 3DGS | CC BY-NC 4.0 |
| Any-view | DA3-LARGE(-1.1) | 0.35B | rel. depth, pose, pose-cond | CC BY-NC 4.0 |
| Any-view | DA3-BASE | 0.12B | rel. depth, pose, pose-cond | Apache 2.0 |
| Any-view | DA3-SMALL | 0.08B | rel. depth, pose, pose-cond | Apache 2.0 |
| Metric | DA3METRIC-LARGE | 0.35B | monocular **metric** depth, sky-seg | Apache 2.0 |
| Monocular | DA3MONO-LARGE | 0.35B | high-quality relative mono depth | Apache 2.0 |

Rule of thumb: **DA3-LARGE ≈ VGGT** quality. Prefer the `**-1.1**` checkpoints (retrained after a training-bug fix; notably better on street scenes). The **Nested** model pairs an any-view giant (pose+depth) with a metric model (scale) for real-world metric reconstruction.

## 6. Usage — Python API

```python
import glob, os, torch
from depth_anything_3.api import DepthAnything3

device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE").to(device)

images = sorted(glob.glob("assets/examples/SOH/*.png"))   # 1..N views (list of paths)
prediction = model.inference(images)                       # single call, any number of views

prediction.processed_images   # [N, H, W, 3] uint8   - resized/letterboxed inputs actually fed to the net
prediction.depth              # [N, H, W]    float32 - per-view depth
prediction.conf               # [N, H, W]    float32 - per-pixel confidence
prediction.extrinsics         # [N, 3, 4]    float32 - world->cam (OpenCV w2c / COLMAP convention)
prediction.intrinsics         # [N, 3, 3]    float32 - pinhole K per view
```

Install:

```bash
pip install xformers "torch>=2" torchvision
pip install -e .                       # basic
pip install -e ".[app]"                # + Gradio UI (python>=3.10)
pip install -e ".[all]"                # everything (incl. gaussian head deps)
```

## 7. Usage — CLI

```bash
export MODEL_DIR=depth-anything/DA3NESTED-GIANT-LARGE
export GALLERY_DIR=workspace/gallery && mkdir -p $GALLERY_DIR

# keep the model resident on GPU (backend), then reuse it
da3 backend --model-dir $MODEL_DIR --gallery-dir $GALLERY_DIR

# reconstruct a folder of images -> glb
da3 auto assets/examples/SOH --export-format glb --export-dir $GALLERY_DIR/SOH --use-backend

# process a video (sample 15 fps) with feature-visualization export
da3 video assets/examples/robot_unitree.mp4 --fps 15 --use-backend \
    --export-dir $GALLERY_DIR/robo --export-format glb-feat_vis \
    --process-res-method lower_bound_resize --export-feat "11,21,31"
```

Export formats include `glb`, `npz`, `ply`, depth images, and 3DGS videos. **DA3-Streaming** handles ultra-long videos with sliding-window inference in <12 GB VRAM.

## 8. Input / output format (summary)

**Input**

- One or more RGB images (paths, or arrays), *any* count N ≥ 1; or a video (sampled to frames at `--fps`).
- Optional: known camera poses → **pose-conditioned** mode (better consistency).
- Preprocessing resizes/letterboxes to a network resolution (see `processed_images`); `--process-res-method` controls the resize policy.

**Output** (`prediction` object)

| Field | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `processed_images` | [N,H,W,3] | uint8 | the actual images fed to the model (post-resize) |
| `depth` | [N,H,W] | float32 | per-view depth (relative, unless metric/nested → meters) |
| `conf` | [N,H,W] | float32 | per-pixel confidence |
| `extrinsics` | [N,3,4] | float32 | world→camera [R\ |
| `intrinsics` | [N,3,3] | float32 | pinhole K per view |

Derived/optional outputs depending on model: 3D point cloud (back-projected via §4), 3D Gaussians (Giant/Nested), sky segmentation (Metric/Nested), and pose from ray-head vs cam-head (`use_ray_pose`).

## 9. Results

- **Visual-geometry benchmark (new, in-paper):** SOTA on camera pose, any-view geometry, and rendering. **+44.3%** avg camera-pose accuracy and **+25.1%** geometric accuracy vs VGGT.
- **Monocular depth:** outperforms **DA2** while directly predicting depth (not disparity), giving better geometric accuracy.
- **Pose (ray vs cam head, AUC3 on DA3NESTED):** ray head generally higher — e.g., HiRoom 84.4 vs 80.3, ETH3D 52.6 vs 48.4, ScanNet++ 89.4 vs 85.0.

## 10. Strengths & limitations

**Strengths:** one model spans mono→multi-view depth, pose, and 3D; minimal, plain-ViT design (easy to scale/finetune, leverages DINOv2 pretraining); consistent geometry across views without alignment; strong pose/geometry gains; open weights; Base/Small/Metric/Mono are Apache-2.0.

**Limitations:** Giant/Large/Nested are **CC BY-NC 4.0** (non-commercial); needs xformers (older GPUs need a workaround); metric scale requires the Metric/Nested variant (any-view outputs are relative up to scale); large models are heavy (1.15–1.4B params) though streaming eases long-video memory.

## 11. Sources

- [DA3 paper — arXiv:2511.10647](https://arxiv.org/abs/2511.10647)
- [Project page](https://depth-anything-3.github.io/)
- [GitHub — ByteDance-Seed/depth-anything-3](https://github.com/ByteDance-Seed/depth-anything-3)
- [Hugging Face demo](https://huggingface.co/spaces/depth-anything/Depth-Anything-3)