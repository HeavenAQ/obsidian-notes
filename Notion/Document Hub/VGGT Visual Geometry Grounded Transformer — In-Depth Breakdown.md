---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-06T17:06:00
Status: Done
Last updated time: 2026-07-06T17:06:00
Last edited by: Heaven Chen
Category:
  - CV
  - Survey
  - Theory
---
> Wang, Chen, Karaev, Vedaldi, Rupprecht, Novotny (Oxford VGG + Meta AI). **CVPR 2025 (Best Paper Award)**. arXiv:2503.11651. Code: facebookresearch/vggt. Project: [vgg-t.github.io](http://vgg-t.github.io/). Successor: **VGGT-Omega** (2026).

## 1. TL;DR

VGGT is a **single feed-forward transformer** that takes **1, a few, or hundreds of RGB images** of a scene and directly predicts *all* the core 3D quantities at once — **camera intrinsics + extrinsics, per-pixel depth maps, per-pixel 3D point maps, and 2D point tracks** — in **under a second**, with **no** iterative optimization, bundle adjustment, or test-time geometry solving. It replaces the traditional multi-stage SfM/MVS pipeline (feature matching → pose → triangulation → BA) with one network, and still beats optimization-based methods on camera pose, multi-view depth, dense reconstruction, and tracking. It won CVPR 2025 Best Paper.

## 2. Key ideas

- **One network, all 3D tasks.** Prior work specialized per task (pose, depth, tracking). VGGT unifies them and shares a backbone, so tasks reinforce each other.
- **Feed-forward, no optimization.** DUSt3R/MASt3R needed pairwise alignment + global optimization; VGGT does the whole scene in one forward pass. (Optional bundle adjustment is available but *not required*.)
- **Any number of views.** The same model handles single-image reconstruction (a zero-shot ability it was never explicitly trained for), a few views, or hundreds.
- **Predict in the first camera's frame.** All geometry is expressed in the coordinate system of the **first input image**, which becomes the reference (identity) camera — removing global-frame ambiguity.
- **Depth+camera unprojection beats the direct point head.** VGGT predicts point maps directly *and* depth+pose; unprojecting depth through the predicted camera usually gives more accurate 3D points.

## 3. Architecture

```plain text
N images ─► DINOv2 patchify (per image) ─► tokens + [camera token] + [register tokens]
                                             │
                        ┌───────── Aggregator (Alternating-Attention) ───────┐
                        │   repeat L times:                                     │
                        │     (a) FRAME-WISE self-attn  (tokens attend within  │
                        │          their own image)                            │
                        │     (b) GLOBAL self-attn      (tokens attend across   │
                        │          all images / all frames)                     │
                        └────────────────────────────────────────────┘
                                             │  aggregated multi-view tokens
        ┌──────────────────────────┬─────────────────────┬─────────────┐
        ▼                         ▼                          ▼                 ▼
 Camera head (MLP)         Depth head (DPT)         Point head (DPT)     Track head (CoTracker-style)
 pose enc → K, [R|t]       depth + confidence       point map + conf     2D tracks + visibility
```

**Components:**

- **Tokenizer:** each image is patchified by a **DINOv2** ViT into patch tokens. Per image, VGGT appends a learnable **camera token** and a few **register tokens**; the first frame's camera token is made distinct so the model knows which view is the reference.
- **Aggregator = Alternating-Attention (AA):** the core trick. Instead of full global attention over all tokens of all frames (expensive) or purely per-frame attention (no cross-view fusion), it **alternates**: a **frame-wise** self-attention layer (each image reasons internally) followed by a **global** self-attention layer (all frames exchange information), repeated for many layers (~24). This scales to many views while still fusing multi-view geometry.
- **Four prediction heads** (all feed-forward, no optimization):
    - **Camera head** — an MLP on each frame's camera token (with iterative refinement) outputs a compact **pose encoding** → decoded to extrinsics `[R|t]` and intrinsics `K`.
    - **Depth head** — a **DPT** dense head → per-pixel depth + per-pixel confidence.
    - **Point head** — a **DPT** head → per-pixel 3D **point map** (in the reference frame) + confidence.
    - **Tracking head** — CoTracker-style: given query pixels, predicts their 2D **tracks** across all frames + visibility/confidence.

## 4. The math

### 4.1 What it maps

Given images $\{I_i\}_{i=1}^{N}$, VGGT outputs for each frame $i$: camera $g_i$, depth $D_i$, point map $P_i$, and (on request) tracks $T$, all in the **first camera's** coordinate frame (so camera 1 is identity).

### 4.2 Pose encoding

Each camera is a compact vector decoded into `K, [R|t]`:

$$
g_i = [\, \mathbf{q}_i \;(\text{rotation quaternion, 4}),\; \mathbf{t}_i \;(\text{translation, 3}),\; \mathbf{f}_i \;(\text{field-of-view, 2}) \,]
$$

Regressing a normalized quaternion + translation + FOV (rather than a raw matrix) keeps rotations valid and the target low-dimensional. Extrinsics follow the **OpenCV camera-from-world** convention.

### 4.3 Point map & depth–camera consistency

The point head predicts, per pixel $(u,v)$ of image $i$, a 3D point $P_i(u,v)\in\mathbb{R}^3$ in the reference frame (a DUSt3R-style *point map*). Independently, depth + camera give the same 3D point by **unprojection**:

$$
X_i(u,v) = R_i^{\top}\big(D_i(u,v)\,K_i^{-1}[u,v,1]^{\top} - t_i\big)
$$

Both should agree; in practice the **unprojected** version is more accurate, so the released `unproject_depth_map_to_point_map` combines depth + pose.

### 4.4 Confidence-aware multi-task loss (aleatoric weighting)

Depth and point heads emit a per-pixel confidence $\Sigma$; the loss down-weights uncertain pixels and is regularized so the model can't cheat by being globally unconfident:

$$
\mathcal{L}_{\text{depth}} = \sum_i \big\| \Sigma_i^{D}\odot(\hat{D}_i - D_i)\big\| \; -\; \alpha \sum_i \log \Sigma_i^{D}
$$

The point-map loss has the same form. The camera loss is a Huber on the pose encoding, and tracking uses a CoTracker-style loss:

$$
\mathcal{L}_{\text{camera}} = \sum_i \mathrm{Huber}(\hat{g}_i - g_i),\qquad
\mathcal{L} = \mathcal{L}_{\text{camera}} + \mathcal{L}_{\text{depth}} + \mathcal{L}_{\text{pmap}} + \lambda\,\mathcal{L}_{\text{track}}
$$

All tasks are trained **jointly**; the shared aggregator is what lets pose, depth, points, and tracks reinforce one another.

## 5. Model & performance

- **VGGT-1B** — ~1B parameters; reconstructs a scene in **< 1 second** (visualization is slower, but that's third-party rendering, not the model).
- **SOTA** on camera-pose estimation, multi-view depth, dense point-cloud reconstruction, and 3D point tracking — *even vs. methods that add geometry optimization/BA*.
- **Zero-shot single-view** reconstruction is surprisingly strong (competitive with Depth Anything v2 / MoGe) despite never being trained for it.
- As a **feature backbone**, pretrained VGGT boosts downstream non-rigid tracking and feed-forward novel-view synthesis.
- **VGGT-Omega** (2026) is the successor; a memory fix also lets the original run on ~2–3× more frames at the same VRAM.

## 6. Usage — Python (simple)

```python
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

image_names = ["imgA.png", "imgB.png", "imgC.png"]     # 1..hundreds of views
images = load_and_preprocess_images(image_names).to(device)  # [N,3,H,W]

with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
    predictions = model(images)      # cameras, depth, point maps (+ conf) in one pass
```

## 7. Usage — Python (per-head, more control)

```python
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
    images = images[None]                                  # add batch dim -> [B,N,3,H,W]
    tokens, ps_idx = model.aggregator(images)             # run the AA backbone once

# Cameras (OpenCV camera-from-world)
pose_enc = model.camera_head(tokens)[-1]
extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

# Dense geometry
depth_map, depth_conf = model.depth_head(tokens, images, ps_idx)
point_map, point_conf = model.point_head(tokens, images, ps_idx)

# More accurate 3D points: unproject depth through predicted cameras
points_3d = unproject_depth_map_to_point_map(depth_map.squeeze(0),
                                             extrinsic.squeeze(0),
                                             intrinsic.squeeze(0))

# 2D tracks for chosen query pixels (N,2)
query = torch.FloatTensor([[100., 200.], [60.7, 259.9]]).to(device)
tracks, vis, conf = model.track_head(tokens, images, ps_idx, query_points=query[None])
```

Tip: to ignore sky/reflections/water, set those input pixels to 0 or 1 — even coarse bounding-box masks work.

## 8. Input / output format

**Input**

- RGB images via `load_and_preprocess_images` → tensor `[N, 3, H, W]` (resized/normalized; batched as `[B, N, 3, H, W]`). N can be 1 to hundreds.

**Output** (per frame `i`, batch `B`, `N` views)

| Quantity | Shape | Convention / meaning |
| --- | --- | --- |
| `extrinsic` | `[B,N,3,4]` | `[R\ |
| `intrinsic` | `[B,N,3,3]` | pinhole `K` per view |
| `depth_map` | `[B,N,H,W,1]` | per-pixel depth |
| `depth_conf` | `[B,N,H,W]` | depth confidence |
| `point_map` | `[B,N,H,W,3]` | per-pixel 3D point in the **first-camera frame** |
| `point_conf` | `[B,N,H,W]` | point-map confidence |
| `tracks` | `[B,N,Q,2]` | 2D locations of `Q` query points across all `N` frames |
| `vis / conf` | `[B,N,Q]` | per-track visibility & confidence |

All geometry is in the coordinate frame of the **first input image** (identity camera).

## 9. Downstream / export

- **COLMAP export:** `demo_colmap.py --scene_dir ...` writes `cameras.bin / images.bin / points3D.bin`; add `--use_ba` for optional bundle adjustment (with `--max_query_pts`, `--query_frame_num` knobs).
- **Gaussian Splatting / NeRF:** the exported COLMAP folder drops straight into **gsplat** (`gsplat==1.3.0`) or other NeRF/3DGS libraries.
- **Viewers:** Gradio web demo and a Viser 3D point-cloud viewer are included.

## 10. Strengths & limitations

**Strengths:** one feed-forward model for pose + depth + points + tracks; **sub-second** whole-scene reconstruction with no BA; scales from 1 to hundreds of views; SOTA across four 3D tasks; strong zero-shot single-view; excellent pretrained backbone; clean COLMAP→gsplat path.

**Limitations:** 1B params (heavy; long sequences are memory-bound — the reason VGGT-Omega exists); global attention cost grows with total tokens/views; the direct point-map branch is less accurate than depth-unprojection; predictions are **relative** (first-camera frame, up to global scale — no metric scale); the **original checkpoint is non-commercial** — only **VGGT-1B-Commercial** (application-gated, excludes military use) permits commercial deployment.

## 11. Lineage

Builds on the Oxford/Meta 3D line: **Deep SfM Revisited → PoseDiffusion → VGGSfM**, plus **CoTracker** (tracking) and **DUSt3R/MASt3R** (point maps), unified into one feed-forward model.

## 12. Sources

- [VGGT paper — arXiv:2503.11651](https://arxiv.org/abs/2503.11651)
- [Project page](https://vgg-t.github.io/)
- [GitHub — facebookresearch/vggt](https://github.com/facebookresearch/vggt)
- [Hugging Face model — facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B)
- [Hugging Face demo](https://huggingface.co/spaces/facebook/vggt)
- [VGGT-Omega (successor)](https://vggt-omega.github.io/)