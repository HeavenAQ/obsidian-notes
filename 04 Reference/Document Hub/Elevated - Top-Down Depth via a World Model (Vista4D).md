---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-06T19:03:00
Status: Done
Last updated time: 2026-07-08T15:34:00
Last edited by: Heaven Chen
Category:
  - CV
  - AIGC
---
> **Deep dive into Vista4D** (["Vista4D: Video Reshooting with 4D Point Clouds", arXiv 2604.21915](https://arxiv.org/abs/2604.21915), Eyeline Labs / Netflix, CVPR 2026 Highlight — [project page](https://eyeline-labs.github.io/Vista4D/)). Part A is the paper: problem, architecture, training methodology, results, applications. Part B is our hands-on pipeline that uses it to produce an elevated top-down depth map from a fixed security camera. Camera math used throughout is covered in [Camera Math — Unified Deep Note](https://app.notion.com/p/397e445b30a48163b096c20b38c95de5).

# Part A — The paper

# 1. Problem: video reshooting

Given a monocular **source video** $\mathbf{X}^{\mathrm{src}}$, re-synthesize the *same dynamic scene* — same people, same motion, same timing — from a **different camera trajectory** the user defines. Three simultaneous requirements make this hard:

1. **Faithful reconstruction of seen content** — pixels the source camera observed must reappear where geometry says they should.
2. **Plausible generation of unseen content** — a moved camera reveals surfaces the source never saw; something must hallucinate them photorealistically.
3. **Precise, user-definable camera control** — the output must actually follow the requested trajectory, previewable before paying for generation.

The tension: (1) and (3) want explicit geometry; (2) wants a strong generative prior. Vista4D's thesis is that you get all three by **grounding a video diffusion model in a 4D point cloud** — and, critically, by training the model to *expect that point cloud to be wrong*.

# 2. The landscape Vista4D competes in

| Family | Camera signal | Representatives | Failure mode |
| --- | --- | --- | --- |
| **Explicit prior** | per-frame depth-lifted point cloud, rendered from target cameras | TrajectoryCrafter, GEN3C, EX-4D | trained on *artifact-free* clouds (double reprojection / watertight meshes) → task collapses to inpainting; real-world recon artifacts at non-frontal views break them; per-frame clouds lose content when target barely overlaps source |
| **Implicit prior** | camera embeddings or reference videos, no geometry | ReCamMaster, CamCloneMaster | monocular scale ambiguity → imprecise, non-previewable camera control; often barely moves the camera at all |

Vista4D is an explicit-prior method that fixes both explicit-prior weaknesses (via temporal persistence + artifact-robust training) and borrows the implicit-prior strength (conditioning on the raw source video for appearance transfer).

**The 4D reconstruction it stands on** (for context): end-to-end feed-forward recon models — $\pi^3$ (permutation-equivariant geometry), STream3R (causal-transformer streaming recon), DA3-class video depth — which output per-frame depth $\mathbf{D}^{\mathrm{src}}$, extrinsics $\mathbf{T}^{\mathrm{src}}$, intrinsics $\mathbf{K}^{\mathrm{src}}$ from a monocular video, replacing classical SfM/SLAM (which fails on dynamic scenes).

# 3. Architecture

```plain text
                          ┌─ 4D reconstruction (π³ / STream3R / DA3) ─→ D, T, K
source video X^src ──────┤
                          └─ dynamic-pixel segmentation (Grounded SAM 2) → static mask M^stc
                                       │
        unproject + world transform:  P = Ω(Φ⁻¹([X^src, D^src], K^src), T^src)     (Eq. 1)
                                       │
        static pixels made persistent across ALL frames via M^stc  →  P̄  (the 4D point cloud)
                                       │
        render P̄ from user's target cameras C^tgt = (K^tgt, T^tgt)
                                       │
             ┌─ point-cloud render X^src→tgt  +  alpha mask M^src→tgt
             │
   ┌─────────┴──────────────────────────────────────────────────────────┐
   │  Wan2.1-T2V-14B DiT (flow matching), finetuned:                     │
   │   tokens = [ noisy target latents ‖ X^src→tgt latents ‖ X^src ]     │
   │            concatenated along the FRAME dimension (in-context)      │
   │   + C^tgt injected as Plücker embeddings (zero-init linear proj.,   │
   │     identity-init projection after self-attention)                  │
   └───────────────────────────→  output video X^tgt  ──────────────────┘
```

## 3.1 The temporally-persistent 4D point cloud (the core representational idea)

Standard explicit-prior methods lift **each frame separately**: frame *t*'s conditioning shows only what frame *t*'s depth map saw. Vista4D instead builds one **world-space** cloud:

$$
\mathbf{P}=\Omega\left(\Phi^{-1}\left([\mathbf{X}^{\mathrm{src}},\mathbf{D}^{\mathrm{src}}],\mathbf{K}^{\mathrm{src}}\right),\mathbf{T}^{\mathrm{src}}\right)
$$

where $\Phi^{-1}$ is inverse perspective projection (pixel+depth → camera-frame 3D; §3 of the camera-math note) and $\Omega$ the camera→world transform. Then the segmentation mask $\mathbf{M}^{\mathrm{stc}}$ splits pixels into **static** vs **dynamic**, and *static points are made persistent across all frames*: every rendered target frame sees the union of all static geometry ever observed, while dynamic points (people, cars) remain per-frame so motion stays correct. That union is the **temporally-persistent cloud** $\overline{\mathbf{P}}$.

Why this matters, concretely:

- **Content preservation**: a wall seen only in frame 3 still conditions frame 40's generation.
- **Camera signal when overlap is small**: a target camera looking mostly at geometry the *current* frame doesn't see still receives dense conditioning from other frames' static points — per-frame baselines get near-empty conditioning there and lose camera lock.
- **It is a previewable contract**: rendering $\overline{\mathbf{P}}$ from $\mathbf{C}^{\mathrm{tgt}}$ yields $\mathbf{X}^{\mathrm{src\to tgt}}$ (RGB) + $\mathbf{M}^{\mathrm{src\to tgt}}$ (alpha: where geometry exists) — you can *see* what the camera will get before running the diffusion model (exactly the coverage check in Part B, step 3).

**How the cloud is actually built (the preprocessing stack).** Unprojection needs per-frame depth plus camera pose, both produced by a feed-forward 4D reconstructor. The released pipeline defaults to **Pi3X** (a permutation-equivariant descendant of π³): the authors find it flickers and warps less than DA3 while using less VRAM, and offer **DA3** as the alternative when a highly-detailed source justifies its higher base resolution. **Sky pixels** are segmented separately and pushed to a large constant depth, so they resolve to infinity instead of forming a false near-wall in the cloud. The static/dynamic split is what makes persistence *safe*: static points from every frame are fused once into world space and reused for all target frames, whereas **dynamic points stay strictly per-frame** — the person at time *t* is rendered only into target frame *t* — so accumulating static geometry never smears a moving subject across time. This is why segmentation quality matters yet needn't be perfect (see §5 robustness).

## 3.2 Conditioning design (the architectural idea)

Three signals condition the generation, each entering differently:

- **Point-cloud render + alpha mask** — the geometry scaffold (where things are, what's missing).
- **Source video, in-context**: the patchified latent tokens of $\mathbf{X}^{\mathrm{src}}$ and $\mathbf{X}^{\mathrm{src\to tgt}}$ are **concatenated with the noisy target tokens along the frame dimension**, so full self-attention runs across all three streams. This is deliberately *not* TrajectoryCrafter's cross-attention injection — the ablation (§7) shows in-context conditioning preserves source content much better and is what lets the model *correct* bad geometry using appearance evidence: when the cloud says one thing and the source video says another, attention can arbitrate.
- **Target cameras as Plücker embeddings**: per-pixel ray encodings $(\mathbf{d}, \mathbf{o}\times\mathbf{d})$ of $\mathbf{C}^{\mathrm{tgt}}$, injected via **zero-initialized linear projections with an identity-initialized projection after self-attention** (ReCamMaster-style) — zero/identity init means finetuning starts as a no-op and the camera pathway grows in gradually without destabilizing the pretrained DiT.

**Why the frame-dimension concatenation is affordable.** All three streams live in Wan's **VAE latent space** — the causal 3D VAE compresses the 49-frame clip roughly 4× temporally and 8×8 spatially before patchify — so concatenating $\mathbf{X}^{\mathrm{src}}$ and $\mathbf{X}^{\mathrm{src\to tgt}}$ onto the noisy target latents along the temporal axis merely triples the token count rather than exploding it per pixel, and full 3D self-attention then runs over the union. Because this is the *same* attention the base model already uses for temporal coherence, the DiT can treat the source frames as extra “context frames” and lift appearance straight from them — the concrete mechanism by which appearance evidence overrides wrong geometry.

**Base model & objective.** Wan2.1-T2V-14B, a text-to-video **flow-matching** diffusion transformer, finetuned with:

$$
\mathcal{L}=\left\lVert\boldsymbol{\epsilon}_{\theta}\left(\mathbf{X}^{\mathrm{tgt}}_{t},\mathbf{X}^{\mathrm{src\to tgt}},\mathbf{M}^{\mathrm{src\to tgt}},\mathbf{X}^{\mathrm{src}},\mathbf{C}^{\mathrm{tgt}},t\right)-\mathbf{V}\right\rVert,\qquad \mathbf{V}=\mathbf{X}^{\mathrm{tgt}}-\boldsymbol{\epsilon}
$$

i.e. the model predicts the flow-matching velocity toward the clean target video. **What's trained vs frozen**: trained — the patchify layers for $\mathbf{X}^{\mathrm{src}}$ and $\mathbf{X}^{\mathrm{src\to tgt}}$, all self-attention layers, camera encoders, and projectors; frozen — everything else (cross-attention to text, FFNs, VAE, T5 encoder). A surgical finetune that adds grounding without erasing the video prior.

# 4. Training methodology (the data idea — arguably the real contribution)

## 4.1 Train on *noisy* multiview data, not clean renders

The failure Vista4D targets: prior work manufactures training pairs by **double reprojection** — render the target video's own depth into a fake source camera, then render *that* back into the target camera. The conditioning always views the depth map from its frontal, artifact-free direction, so the model only ever learns to **inpaint holes in otherwise-correct geometry**. At inference on real videos, the cloud viewed from a genuinely new angle is *wrong* (stretched, misplaced, jittering — especially dynamic pixels, where monocular depth has no multiview constraint), and inpainting-trained models faithfully reproduce the garbage.

Vista4D instead trains on **multiview dynamic videos with 4D-reconstructed (i.e., imperfect) depths and cameras**: render the *source* view's cloud into a *different real* target view → the conditioning now contains exactly the spatially-mismatched, non-frontal artifacts seen at inference, while the ground-truth target video shows what the output *should* be. The supervision signal therefore teaches **geometry correction**, not inpainting. This is the same philosophy as denoising/robust training everywhere: corrupt the input the way deployment corrupts it.

**The two training-pair constructions, precisely.** The paper draws them as Figure 3(a) vs 3(b). *(a) Double reprojection*: take the target video's own depth, render it into a fabricated source camera, then render that back into the target camera — the conditioning always views the depth map from its own frontal direction, so it is artifact-free and merely missing occluded holes. *(b) Cross-view rendering* (Vista4D's choice wherever real multiview exists): render the **source** view's reconstructed cloud directly into a **genuinely different real** target camera, so the conditioning carries the true non-frontal artifacts — stretching, misplacement, temporal jitter — while the real target video supplies the correct answer. Vista4D uses (b) on the synthetic multiview set and falls back to (a) only on real monocular clips where no second real view exists.

## 4.2 Datasets & the static-pixel labeling pipeline

- **Multiview synthetic**: MultiCamVideo (from ReCamMaster) — time-synchronized multi-camera dynamic scenes; 4D reconstruction across all views with **STream3R**.
- **Real-world monocular**: 60K-video subset of OpenVidHD-0.4M, reconstructed with **π³**; since no true second view exists, these use TrajectoryCrafter-style double reprojection ($\mathbf{X}^{\mathrm{tgt\to src}}$ rendered back), and the occluded reprojection also stands in as the "source video" condition so the propagation pathway still trains.
- **Static segmentation** (inspired by Uni4D): **RAM** tags semantic classes → **Llama-3.1-8B-Instruct** filters the tag list down to dynamic subjects/nouns → **Grounded SAM 2** segments those per frame → masks inverted = static pixels. Fully automatic labeling at dataset scale. (At *inference* the released code drops the auto-discovery: you either name the dynamic classes by hand or let it run **Grounded SAM 2 / SAM 3** on those keywords — the RAM→Llama step is a training-scale convenience, not a runtime dependency. Part B's `--seg_keywords` is exactly this manual path.)

## 4.3 Training schedule

Finetune at **672×384 for 30,000 steps**, then continue from that 384p checkpoint at **1280×720 for 3,000 steps** (a cheap high-res adaptation — the 720p variant is finetuned *from* the 384p one, not trained fresh; both are public as `384p49_step=30000` and `720p49_step=3000`), both 49-frame clips, global batch 8, AdamW, lr $1\times10^{-5}$. (Part B uses the 384p checkpoint — the only one released at the time.)

# 5. Results (what the numbers say)

Evaluation set: 110 video-camera pairs (51 DAVIS/Pexels videos × 2–3 authored trajectories), plus the real multiview **iphone** dataset for NVS. Highlights:

| Dimension | Metric | Vista4D | Best baseline |
| --- | --- | --- | --- |
| Camera control | rotation error ↓ | **4.647** | 4.751 (GEN3C) |
| 3D consistency | reprojection error @SuperGlue ↓ | **7.504** | 12.99 (GEN3C); TrajectoryCrafter 120.5 |
| NVS quality | mPSNR ↑ / mLPIPS ↓ | **14.09 / 0.461** | 13.82 / 0.569 (TrajCrafter) |
| Motion fidelity | optical-flow EPE ↓ | **1.142** | 2.375 (TrajCrafter) |
| User study (42 users, 30 pairs) | overall fidelity preference | **77.4%** | 11.0% (CamCloneMaster) |

Reading the table honestly: the *3D-consistency* gap (7.5 vs 120.5 for TrajectoryCrafter) is the temporal-persistence story; the *EPE* gap is the "same dynamics" story (motion is copied, not re-imagined). Implicit-prior methods win some FID/FVD/consistency metrics **only because they under-follow the requested camera** — less camera change looks "more consistent" to those metrics; camera-control numbers expose it.

**Robustness to segmentation failure**: deliberately mis-labeling a tennis racket as static creates streaking in the cloud — the model corrects it anyway, using the in-context source video. Streaks at inference are rare/inconsequential relative to the gains of persistence.

# 6. Ablations (why each piece is load-bearing)

| Removed | Consequence |
| --- | --- |
| Depth artifacts in training (always double-reproject) | model becomes an inpainter; breaks on real-world non-frontal artifacts, both spatial (stretched depth) and temporal (jittering depth) |
| Source-video conditioning | no appearance evidence to correct bad geometry with |
| In-context → cross-attention injection | markedly worse content preservation (weaker source-target information flow) |
| Temporal persistence | loses static content and camera lock precisely when target cameras overlap the per-frame view least |

The two contributions **compose**: artifact training gives the *skill* of correcting geometry; in-context source conditioning gives the *evidence* to correct it with.

**The ablation grid, in full.** The paper sweeps five variants against the full method: (I) no source video (point-cloud render only); (II) source video via **cross-attention** à la TrajectoryCrafter instead of in-context concatenation; (III) no depth artifacts (always double-reproject, so the render is always spatially aligned with the target); (IV) = II + I; (V) = III + I. Two failure signatures recur: **(1) uncorrectable geometry artifacts** — a mis-estimated depth (a car's shape, a man's arm and hand, a goat's body) passes straight through to the output because the model was never taught to fix it; and **(2) temporal jittering** — frame-to-frame depth wobble reproduced faithfully as a subject shivering left-and-right. Variant (II) is the telling one: cross-attention suppresses some jitter but fails to *transfer* geometry and appearance faithfully (a hand that never leaves a hat; colors brighter and more saturated than the source) — direct evidence that it is *in-context* conditioning specifically, not merely “some” source signal, that preserves content. The temporal-persistence ablation shows the complementary failures: without persistence the model loses seen static content (the right side of a scene; the snow-and-rock mountain behind a snowboarder that the per-frame render never observes) and its camera control degrades precisely where the target barely overlaps the current frame (floaty, jittery, under-moving cameras).

# 7. Applications (all exploiting that the cloud is explicit and editable)

- **Dynamic scene expansion**: jointly 4D-reconstruct the source video *plus* casual stills/alternate angles of the set; their static points enter $\overline{\mathbf{P}}$ → the model hallucinates less because it's told more.
- **4D scene recomposition**: edit the cloud directly — move, duplicate, delete, or *insert* subjects (with their dynamics) from other videos. Because the model was trained on imperfect clouds, it tolerates cut-and-paste seams; the paper's rhino example shows it even re-lights the inserted subject plausibly (dappled sunlight blended into an overcast scene).
- **Long-video inference with memory**: generate in 49-frame chunks; register each generated chunk's static pixels *back into* $\overline{\mathbf{P}}$, so the cloud becomes an explicit compressed memory across chunks (a variant built on first-frame-conditioned **Wan2.1-I2V-14B** keeps chunks visually consistent). The camera can arc away and return to find the same generated content.

**How the applications are actually wired.** *4D scene recomposition* is driven by an edit list paired with the target-camera `.npz`: each edit selects points via a **SAM 3 text prompt** and applies `translate` / `rotate` / `scale` / `remove`, with a target `kind` of `**existing**` (edit the matched points), `**duplicate**` (clone → transform → add alongside), or `**insert**` (pull points from *another* scene's reconstruction and drop them in with their dynamics). Rotations and scales are taken about a computed centroid, and edits run either `**global**` (one centroid for the whole subset) or per `**frame**` (one centroid per frame, so the transform tracks a moving subject). A subtlety the paper flags: to stop the *unedited* source video fighting the *edited* cloud, recomposition conditions on an **edited source video** — the edited cloud re-rendered from the original source cameras *without* temporal persistence — in place of the raw input. *Dynamic scene expansion* is wired differently: the casual capture is jointly reconstructed with the source into a **single coordinate frame** and contributes only extra static points to $\overline{\mathbf{P}}$; the video handed to the diffusion model stays the original source, so the extra capture cuts hallucination and tightens camera control without touching appearance. *Long-video memory* swaps in a checkpoint finetuned from the **image-to-video** Wan2.1-I2V-14B, so each 49-frame chunk is conditioned on the previous chunk's final frame for a seamless hand-off while the growing cloud carries content across chunks.

# 8. Limitations

- **No explicit prior-vs-prior control knob**: the model internally decides how much to trust the (possibly wrong) point cloud vs its own video prior; users can't dial "follow geometry exactly" ↔ "correct freely". The authors propose an interpolation mechanism as future work.
- Inherits generative-video ethics questions (reshooting changes emotional framing of real footage; ownership of transformative work).
- *From our practice (Part B)*: physics still bounds it — a single **fixed** camera yields a 2.5-D shell with zero parallax; no training trick recovers a faithful 90° nadir from ~5% coverage. Vista4D corrects artifacts; it does not conjure unseen rooms faithfully.

---

# Part B — Our pipeline: elevated top-down depth from a fixed camera

How we use Vista4D to turn an ordinary fixed-camera clip into an **elevated "looking-down" depth map**: reconstruct in 4D → author a crane camera → render the cloud → generative reshoot → DA3 depth on the result.

> **Read this first — the honest limitation.** Our source (`worker_src.mp4`) is a *single fixed security camera*: every frame's camera center is identical (`spread(std)=[0,0,0]`) → **zero parallax** → a 2.5-D shell. A steep bird's-eye leaves ~95% of the frame on never-seen surfaces → empty conditioning → hallucination and warping. A faithful 90° nadir is **not recoverable**; a **crane at **`**--pitch 45**` (raised, tilted ~45°) is, keeping most of the frame on real geometry.

**Why this route (vs a plain depth splat):** a monocular depth map holds only front surfaces; reprojecting it top-down leaves occlusion holes → sparse, streaky output (tried raw point-cloud splat and IPM homography — both smear standing people). Vista4D's reshoot fills the gaps generatively, grounded on the cloud (this is §3.1's alpha-mask contract in action).

## World-frame axes (read off the recon — never assume)

DA3/Pi3 align the world frame to the first camera, so its axes are the world axes. Three independent signals agree:

| Signal | Value (this clip) | Conclusion |
| --- | --- | --- |
| Mean camera **forward** `c2w[:, :3, 2].mean(0)` | `[0.00, 0.00, 1.00]` | **+z = viewing axis** (down the aisle) |
| Mean camera **up** `−c2w[:, :3, 1].mean(0)` (col-1 = image-down) | `[0.00, −1.00, 0.00]` | **up ≈ −y** (+y points at the floor) |
| RANSAC ground-plane normal | `≈ ±[0.06, 0.92, 0.39]` | y-dominated ⇒ y vertical (24° tilt = camera pitch) |
| Point-cloud extents `[x,y,z]` | `[7.8, 5.5, 11.7]` | shallow y = height; long z = aisle depth |

⇒ **x = lateral, y = vertical (−y up), z = depth**; the overhead camera images the **x–z plane**.

## Environments

| Stage | Interpreter | Weights (`HF_HOME=/82e9e7ee-…/hf-home`) |
| --- | --- | --- |
| Vista4D recon / render / reshoot | `/home/ollo8/miniconda3/envs/vista4d/bin/python` | vista4d 20 GB + Wan2.1-T2V-14B 65 GB |
| DA3 depth on reshot RGB | `/82e9e7ee…/wm-pixi/ViPE-pixi/envs/default/bin/python` | DA3-GIANT |

Vista4D repo: `/home/ollo8/mintei/wm/Vista4D` (run in place).

## Pipeline

Let `V=/home/ollo8/miniconda3/envs/vista4d/bin/python`, run from the Vista4D repo, `HF_HOME`/`HF_TOKEN` exported, `CUDA_VISIBLE_DEVICES` set, `PROJ=<this project>`.

### 1. Reconstruct the 4D scene (`recon_and_seg_single`) — the §3.1 stage

```bash
$V -m scripts.preprocess.recon_and_seg_single \
  --video_path <clip>.mp4 --output_folder <REC> \
  --seg_keywords \"person\" \"car seat\" \"machine\" \"conveyor\" \
  --recon_method da3 --da3_model_id depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
  --pi3_model_id yyfz233/Pi3X \
  --height 720 --width 1280 --num_frames 49 \
  --pi3_pixel_limit 255000 --da3_process_res 896 --save_vis
```

→ `<REC>/{cameras.npz, depths/, dynamic_mask/, sky_mask/, video.mp4}` (~1–2 min). `--seg_keywords` is our stand-in for the paper's RAM→Llama tag discovery (§4.2) — we name the dynamic classes by hand. Recon is at **1280×720** (`cx=640, cy=360`) — note for step 3.

### 2. Author the crane camera (`make_topdown_cam.py` — our one custom file)

Reads the recon, fuses frames into a world cloud, finds gravity, writes a Vista4D camera (`cam_c2w` (N,4,4), `intrinsics` (N,4)=`[fx,fy,cx,cy]`). The three things that had to be right:

- `**--pitch**` = downward tilt from horizontal (90 = nadir; **45 = the sweet spot**). Keeps the *original heading*, only tilts down + raises — a crane move, not a teleport. Lower pitch ⇒ more frame on real geometry.
- **Stable gravity**: RANSAC ground plane + **SVD refinement over inliers** (raw 3-point normals wobble run-to-run → spurious roll).
- **Centred principal point + auto-zoom**: `render_single`'s `resize_intrinsics` infers input resolution from the principal point (`width = cx*2`) → **hard-requires **`**cx=W/2, cy=H/2**`; frame by aiming (`eye = center − fwd·dist`) + picking `**fx**`** by projecting the cloud**; author intrinsics at the render resolution (672×384).

```bash
$V $PROJ/make_topdown_cam.py --recon <REC> \
   --out <REC>/../topdown_cam_p45.npz \
   --pitch 45 --dist-factor 1.4 --width 672 --height 384 --num-frames 49
# prints e.g. up=[0.08,-0.90,-0.43]  extent=14.3  cam height(dist)=20.0  fx=602
```

### 3. Render the cloud from that camera (`render_single`) — produces $\mathbf{X}^{\mathrm{src\to tgt}}$ + mask

```bash
$V -m scripts.preprocess.render_single --recon_and_seg_folder <REC> \
  --cam_path <REC>/../topdown_cam_p45.npz --output_folder <RENDER> \
  --height 384 --width 672 --num_frames 49 --save_vis --render_only_necessary
```

→ `<RENDER>/{video_pc.mp4 (crane RGB), depths_pc/*.exr (crane depth)}`. **Sanity-check coverage** (the §3.1 alpha-mask preview!) before paying for a reshoot — near-nadir gave ~5%; pitch 45 keeps much more:

```bash
python -c \"import numpy as np,imageio.v2 as iio; f=np.array([x for x in iio.get_reader('<RENDER>/video_pc.mp4')]); print((f.max(-1)>12).mean())\"
```

### 4. Generative reshoot (`inference`) — the §3.2/§3.3 model

Only the **384p checkpoint** exists here (`384p49_step=30000` — the paper's 30k-step 672×384 stage, §4.3), so render + reshoot at 384×672.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Vd=./checkpoints/vista4d/384p49_step=30000
$V -m scripts.inference.inference \
  --model_id_with_origin_paths \"Wan2.1-T2V-14B:diffusion_pytorch_model*.safetensors,Wan2.1-T2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan2.1-T2V-14B:Wan2.1_VAE.pth\" \
  --tokenizer_id_with_origin_path \"Wan2.1-T2V-14B:google/*\" \
  --local_model_folder ./checkpoints/wan \
  --vista4d_checkpoint \"$Vd/dit.pth\" --vista4d_config_path \"$Vd/config.yaml\" \
  --input_folder <RENDER> --output_folder <RESHOOT> \
  --prompt \"<scene description>\" --height 384 --width 672 --num_frames 49 \
  --seed 10027 --tile_vae --vram_limit 36 --save_gif
```

→ `<RESHOOT>/video_seed=10027.mp4` (~40 min). **OOM fix**: Wan2.1-14B barely fits 48 GB; default `--vram_limit` (≈ total−2) leaves no activation room → set `expandable_segments:True` and `**--vram_limit 36**` (more CPU offload). Faster: multi-GPU `--use_usp` + `torchrun --nproc_per_node N`.

### 5. Depth from the reshot RGB (DA3)

Run DA3 on `video_seed=10027.mp4` → grayscale disparity (near = bright) → final crane depth (ViPE env; as in `scripts/gen_depth.py`). Depth comes **two ways**: `depths_pc/` from step 3 (geometrically exact, sparse) or DA3-on-reshoot (dense) — use the latter for a control map.

## Outputs (for `worker_src.mp4`)

```javascript
outputs/worker_da3/vista4d/recon_and_seg/            # 4D reconstruction
outputs/worker_da3/vista4d/topdown_cam_p45.npz       # crane camera (--pitch 45)
outputs/worker_da3/vista4d/render_p45/               # point-cloud render + depths_pc (sparse)
outputs/worker_da3/vista4d/reshoot_384/              # reshoot output
outputs/worker_da3/control_depth_topdown_final.mp4   # DA3 depth on the reshot RGB
```

(`topdown_cam.npz`, `render_topdown*`, and the earlier nadir/oblique reshoot were the straight-down attempt — superseded by `--pitch 45`.)

## Caveats / learnings

- **Single fixed camera = the hard ceiling.** Zero parallax → 2.5-D shell; overhead angles render ~5% coverage; crane 45° is the practical limit. True bird's-eye needs moving/multi-camera source or a dedicated monocular-BEV model.
- **Angle, not axes, was the first bug** — forced 90° nadir is unsupported *and* views the depth shell edge-on (pure streaks); keep heading, tilt to ~45°.
- `**render_single**`** requires a centred principal point** — author `cx=W/2, cy=H/2` at render resolution; frame via aiming + `fx`, never `cx`.
- **384p only**; **14B is memory-tight** (vram_limit 36 + expandable_segments, or multi-GPU); **non-metric scale** (camera placed in recon units).
- Everything is stock Vista4D except `**make_topdown_cam.py**` (gravity fit → keep-heading pitch → auto-zoom).

---

**Sources**: [arXiv 2604.21915](https://arxiv.org/abs/2604.21915) ([HTML](https://arxiv.org/html/2604.21915v1)) · [project page](https://eyeline-labs.github.io/Vista4D/) · [HF paper page](https://huggingface.co/papers/2604.21915) · related: TrajectoryCrafter, GEN3C, EX-4D, ReCamMaster, CamCloneMaster, π³, STream3R, Wan2.1.