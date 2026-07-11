---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-06T19:01:00
Status: Done
Last updated time: 2026-07-06T19:01:00
Last edited by: Heaven Chen
Category:
  - AIGC
  - CV
  - Linux
---
How to pull a **single clip** (not the whole 16 TB) from [`nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes`](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes) for testing as Cosmos-Transfer control input.

## Dataset layout (what's in it)

Synthetic warehouse scenes rendered from multiple cameras. Two tiers, packaged as WebDataset `.tar` shards (~4.5–5.3 GB each):

| Path | Size | Contents |
| --- | --- | --- |
| `rgb/<scenario>/*.tar` | ~2.24 TiB | photoreal RGB video per camera + metadata |
| `artifacts/<scenario>/*.tar` | ~16.33 TiB | annotation stack: **depth, instance segmentation, shaded segmentation, canny edges** (all mp4), + bbox/camera jsonl, instance-id npz |
| `assets/*.jpg` `.webp` | small | preview stills per scenario |

Scenarios: `warehouse_fire`, `forklift_human_nearmiss`, `forklift_shelf_collision`, `warehouse_box_pickup`. Specs: **1920×1080, 30 fps, 10 s** (fire/near-miss) or 15 s; 5–10 synchronized cameras per run.

The five modalities per camera map directly onto Cosmos-Transfer control types (`depth` → `control_depth`, `segmentation` → `control_seg`, `edges` → `control_edge`), so the `artifacts` tars are ready-made control videos — no depth/seg estimation needed.

## Prerequisites

`HF_TOKEN` in `.env`, and `HF_HOME` on a big shared mount so weights/data stay off the project dir:

```bash
set -a; source .env; set +a          # exports HF_TOKEN and HF_HOME
# HF_HOME=/82e9e7ee-.../hf-home
```

## Step 1 — inspect the file tree (no download)

```python
from huggingface_hub import HfApi
info = HfApi().dataset_info(
    "nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes",
    token=os.environ["HF_TOKEN"], files_metadata=True)
files = {s.rfilename: s.size for s in info.siblings}
# fire shards: artifacts/warehouse_fire/fire-artifacts-00000.tar ... (~4.5 GB each)
```

## Step 2 (optional) — cheap preview before the big download

```python
from huggingface_hub import hf_hub_download
for f in ["assets/fire_depth.jpg", "assets/fire_segmentation.jpg"]:
    hf_hub_download(REPO, f, repo_type="dataset", token=os.environ["HF_TOKEN"], local_dir="/tmp/hf_assets")
```

## Step 3 — download exactly ONE shard

Use `hf_hub_download` with a single file path — **not** `snapshot_download`, which would pull the entire dataset. Pass `repo_type="dataset"`.

```python
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    "nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes",
    "artifacts/warehouse_fire/fire-artifacts-00000.tar",
    repo_type="dataset", token=os.environ["HF_TOKEN"])
# -> $HF_HOME/hub/datasets--nvidia--.../artifacts/warehouse_fire/fire-artifacts-00000.tar
```

## Step 4 — extract one clip from the shard

The tar holds per-camera files named `<run>.<camera>.<modality>.<ext>` (`depth.mp4`, `segmentation.mp4`, `rgb.mp4`, `edges.mp4`, `shaded_seg.mp4`, `camera_params.jsonl`, `instance_id_segmentation.npz`, ...).

```bash
TAR=$HF_HOME/hub/datasets--nvidia--*/snapshots/*/artifacts/warehouse_fire/fire-artifacts-00000.tar
tar -tf $TAR | head                       # list contents
BASE="6a2a4776042390f83ca4_00023b5323028ab83e67_run_6_seed_1486583949.ceiling_00"
mkdir -p outputs/fire_test
for m in depth segmentation rgb; do
  tar -xf "$TAR" -C outputs/fire_test "$BASE.$m.mp4"
  mv "outputs/fire_test/$BASE.$m.mp4" "outputs/fire_test/fire_$m.mp4"
done
```

## Step 5 — prep for Cosmos-Transfer (1080p/30fps → model budget)

Downscale + trim to the model's frame budget. Use **bicubic for depth**, **nearest-neighbor for segmentation** (preserve exact instance colors):

```bash
ffmpeg -y -i fire_depth.mp4        -vf "fps=10,scale=1280:720:flags=bicubic"  -frames:v 100 -pix_fmt yuv420p fire_depth_ctrl.mp4
ffmpeg -y -i fire_segmentation.mp4 -vf "fps=10,scale=1280:720:flags=neighbor" -frames:v 100 -pix_fmt yuv420p fire_seg_ctrl.mp4
```

Then point a Cosmos-Transfer spec's `depth` / `seg` `control_path` at those (see `specs/cosmos_transfer/fire.json` and `fire_multi.json`).

## Notes

- One shard ≈ 4.5 GB → holds many runs × cameras; a single camera-clip is ~3–15 MB per modality once extracted.
- Fire ignites partway through the 10 s clip and may not appear on every camera/frame.
- The dataset is public (OpenMDW-1.1) but the token is passed for consistency.