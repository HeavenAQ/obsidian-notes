---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-06T22:08:00
Status: Done
Last updated time: 2026-07-06T22:08:00
Last edited by: Heaven Chen
Category:
  - Object Detection
  - Pose Estimation
  - ML
---
> A complete walkthrough of RF-DETR (`rfdetr==1.8.0`), written so you can re-implement the same design with the same considerations. **Part I** introduces the original DETR it descends from; **Part II** documents the full RF-DETR detection architecture (backbone → projector → two-stage → decoder → heads); **Part III** is the keypoint head in depth. All paths are relative to the installed package: `.venv/lib/python3.13/site-packages/rfdetr/`.

---

## Part I — Background: the original DETR

RF-DETR's every design choice is a reaction to DETR, so it's worth 90 seconds on the original.

**DETR** (DEtection TRansformer, Carion et al. 2020) reframed object detection as **direct set prediction** — no anchors, no region proposals, no non-max suppression. The pipeline:

```javascript
image ─► CNN backbone (ResNet) ─► feature map (H/32 × W/32 × C)
      ─► + 2-D sinusoidal positional encoding, flatten to tokens
      ─► Transformer ENCODER  (self-attention over all image tokens)   → memory
      ─► Transformer DECODER:
            N learned "object queries" (e.g. 100)
            cross-attend to memory + self-attend to each other
      ─► per query: FFN → (class incl. "∅ no-object",  box cxcywh)
```

The decoder is **non-autoregressive**: all N queries are decoded in parallel. Query self-attention lets the queries "talk" so they don't all fire on the same object.

**The set loss (the core idea).** The model always emits exactly N predictions. To train, DETR finds the **one-to-one bipartite matching** between the N predictions and the (∅-padded) ground-truth set that minimizes a matching cost — solved with the **Hungarian algorithm**. The cost combines class probability + L1 box + GIoU. Once matched, the loss is class cross-entropy + L1 + GIoU on the matched pairs; unmatched predictions are supervised toward ∅. This removes NMS entirely: there is no duplicate to suppress because the loss *forbids* two predictions from claiming one object.

**Why DETR was reworked.** Three pain points drove its successors:

1. **Glacial convergence** (~500 epochs): global cross-attention must learn *where to look* from scratch, and early Hungarian matching is unstable.
2. **Weak on small objects**: a single low-resolution feature map.
3. **Expensive encoder**: self-attention is quadratic in the number of pixels.

**The lineage RF-DETR sits in:**

- **Deformable DETR** — each query attends to only a few *learned sample points* per feature level (multi-scale **deformable attention**), and adds **two-stage** query selection. Fixes convergence + small objects. RF-DETR inherits both.
- **Conditional / DAB-DETR** — represent a query *as a box* (reference point), so cross-attention is spatially conditioned. RF-DETR's queries are 4-D reference boxes.
- **DINO / Group-DETR** — denoising and **one-to-many groups** for faster, more stable matching. RF-DETR uses Group-DETR (`group_detr` groups at train time).
- **LW-DETR** — drop the CNN+encoder; use a **plain ViT as the encoder/backbone** + a lightweight projector + a deformable decoder. RF-DETR *is* an LW-DETR.
- **RF-DETR** (Roboflow) — LW-DETR over a **DINOv2-with-registers windowed ViT**, tuned for fine-tuning and real-time inference; SOTA on COCO, ICLR 2026.

Keep DETR's three primitives in mind — they survive unchanged in RF-DETR: **(1) learned object queries, (2) Hungarian one-to-one matching, (3) the L1+GIoU+classification set loss.** Everything else (backbone, attention, query seeding) is the optimization.

---

## Part II — The full RF-DETR architecture

The detection model is the class `**LWDETR**` (`models/lwdetr.py:116`). Unlike DETR there is **no separate transformer encoder** — the ViT backbone *is* the encoder (`Transformer.__init__` sets `self.encoder = None`, `transformer.py:176`), and a two-stage query-selection step seeds the decoder.

### II.0 End-to-end detection dataflow

```javascript
image (B,3,R,R)
  │
  ▼  models/backbone/
DINOv2-with-registers WINDOWED ViT  ── extracts features at depths out_feature_indexes (e.g. [3,6,9,12])
  │   (windowed attention everywhere except the feature-output blocks, which go global)
  ▼  strip CLS+register tokens, un-window, reshape  →  several (B, C, H/p, W/p) maps (same spatial size)
MultiScaleProjector  (C2f neck)  ──  up/down-samples to a pyramid: projector_scale e.g. ["P4"]  → L levels
  │                                  + PositionEmbeddingSine per level
  ▼  flatten+concat all levels → memory (B, ΣHₗWₗ, C);  build spatial_shapes, level_start_index, valid_ratios
TWO-STAGE query selection  (per group_detr group)
  │   gen_encoder_output_proposals → enc class/box heads → top-k → init reference boxes + content queries
  ▼
Deformable DECODER  × dec_layers
  │   per layer:  group-DETR query self-attn  →  MSDeformAttn cross-attn (n_points × n_levels)  →  FFN
  │   reference boxes via lite_refpoint_refine (+ bbox_reparam)
  ▼  hs  (B, Q, C) per layer
class_embed (Linear→num_classes+1)   bbox_embed (3-layer MLP→4, reparam decode)
  ▼
pred_logits, pred_boxes  ── (+ aux_outputs per layer, + enc_outputs from stage-1)
  ▼
SetCriterion (Hungarian match + losses)  /  PostProcess (sigmoid top-k → pixel boxes)
```

### II.1 Backbone — DINOv2-with-registers windowed ViT

`models/backbone/` — `Backbone(BackboneBase)` (`backbone.py:31`) wraps the ViT (`DinoV2`, `dinov2.py:55`) + projector, and `build_backbone` returns a `Joiner` that pairs it with positional encoding.

- **Patch embed**: `nn.Conv2d(3, hidden, kernel=patch_size, stride=patch_size)` turns the image into a grid of patch tokens.
- **Tokens**: a CLS token + **register tokens** (the registers absorb high-norm artifacts) + learned absolute `position_embeddings`, bicubically **interpolated** from the pretrained 518 px / 37×37 grid to the configured resolution — this is what lets you fine-tune at a different `resolution`.
- **ViT block**: pre-norm → attention → **LayerScale** → DropPath residual → norm → MLP (or SwiGLU) → LayerScale → residual.
- **Windowed attention** (the LW-DETR speed trick): `num_windows²` windows; the patch grid is reshaped so each window is an independent "image". Most blocks attend **only within a window** (cheap); only the **feature-output blocks** (`out_feature_indexes`) run **full global attention**. Input H/W must be divisible by `patch_size × num_windows`.
- **Multi-depth feature extraction**: instead of an FPN's multiple *spatial* resolutions, RF-DETR taps **several ViT depths** (`out_feature_indexes`, e.g. `[3,6,9,12]`) all at the **same** spatial size; per tap it applies LayerNorm, strips CLS+register tokens, un-windows, and reshapes to `(B, C, H/p, W/p)`.
- **Positional encoding**: `PositionEmbeddingSine` (normalized 2-D sinusoidal, `num_pos_feats = hidden_dim//2`).

### II.2 Projector (neck) — `MultiScaleProjector`

`models/backbone/projector.py:162`. This turns the same-resolution ViT taps into a true multi-scale pyramid.

- `projector_scale` (e.g. `["P4"]`, or `["P3","P5"]`) maps to scale factors `P3=2.0, P4=1.0, P5=0.5, P6=0.25`. For each requested level it up-samples (`ConvTranspose2d`), keeps (identity), down-samples (strided conv), or max-pools an extra level.
- All taps resampled to a level are concatenated channel-wise and fused by a **C2f** (CSP) block + LayerNorm to `out_channels = hidden_dim`.
- Output: a list of `(B, hidden_dim, Hₗ, Wₗ)` pyramid maps; `num_feature_levels = len(projector_scale)`. A second `cross_attn_projector` exists only when `dual_projector=True` (keypoint stream, Part III).

### II.3 Memory + level metadata (in `Transformer.forward`)

Each pyramid level is flattened to `(B, HₗWₗ, C)` and concatenated into `**memory**`; the deformable attention needs three side tensors:

- `spatial_shapes` `(L,2)` — each level's `(H,W)`.
- `level_start_index` — offsets where each level begins in the flattened memory.
- `valid_ratios` — fraction of each level not occupied by padding.

### II.4 Two-stage query selection (no encoder)

RF-DETR variants set `two_stage=True`. Instead of learning decoder queries from nothing, the **encoded memory proposes them**:

4. `gen_encoder_output_proposals` creates a grid of anchor proposals (centers + level-scaled `wh = 0.05·2ˡ`) and a masked `output_memory`.
5. Per Group-DETR group: project memory, score it with a **copy of the class head** (`enc_out_class_embed[g]`) and refine boxes with a **copy of the box head** (`enc_out_bbox_embed[g]`).
6. **Top-k**: `topk(class_logits.max(-1), num_queries)` selects the best proposals → their boxes become the **initial reference points** (detached) and their memory becomes the **initial content queries**.
7. The stage-1 boxes/logits are surfaced as `out["enc_outputs"]` and get their own auxiliary loss.

### II.5 Decoder — deformable, group-DETR, box-as-query

`TransformerDecoder` = `dec_layers` clones of `TransformerDecoderLayer`. Queries: `refpoint_embed = nn.Embedding(num_queries·group_detr, 4)` (4-D reference **boxes**, zero-init) and `query_feat = nn.Embedding(num_queries·group_detr, hidden_dim)` (content). The two-stage reference points are **added** to these learned ones. Inference uses only the first group.

Per layer (`forward_post`):

8. **Query self-attention** — standard MHA over `tgt+query_pos`. During training the queries are split into `group_detr` groups stacked into the batch dim so groups attend **independently** (Group-DETR one-to-many).
9. **Multi-scale deformable cross-attention** — `MSDeformAttn(n_levels=L, n_heads=ca_nhead, n_points=dec_n_points)`. Each query predicts a few **sampling offsets** around its reference box and softmax attention weights over them; samples = `ref_xy + offset · ref_wh · 0.5`. This is the "attend to a few learned points, not all pixels" that fixes DETR's cost and convergence.
10. **FFN** — `linear1 → act → linear2`, residual + norm.

Reference-box handling:

- `**bbox_reparam=True**`: box refinement is `cxcy = δ_xy·ref_wh + ref_xy`, `wh = exp(δ_wh)·ref_wh` — multiplicative on size, additive on center.
- `**lite_refpoint_refine=True**`: the reference box is computed **once** from the init and reused across layers, with `decoder.bbox_embed=None`. With it off, each layer re-regresses and detaches the box — classic **iterative refinement**.

### II.6 Heads & output decoding

On `LWDETR`:

- `**class_embed = nn.Linear(hidden_dim, num_classes)**` — bias initialized to a focal prior `p=0.01`. Note `num_classes = args.num_classes + 1`.
- `**bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)**` — 3-layer MLP, last layer zero-init so it starts at the reference box.

Final decode (`LWDETR.forward`), with `bbox_reparam`:

```python
outputs_coord_delta = self.bbox_embed(hs)
outputs_coord_cxcy  = delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
outputs_coord_wh    = delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
outputs_coord = torch.cat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
outputs_class = self.class_embed(hs)
```

The last layer → `out["pred_logits"]`, `out["pred_boxes"]`; earlier layers → `aux_outputs`; stage-1 → `enc_outputs`. A separate `forward_export` returns plain tuples for ONNX/TensorRT.

### II.7 Matching, losses, post-processing

These reuse DETR's primitives (`models/criterion.py`, `models/matcher.py`, `models/postprocess.py`):

- **Matcher** (`HungarianMatcher`): one-to-one assignment minimizing `cost_class·(focal) + cost_bbox·L1 + cost_giou·GIoU` (+ keypoint costs in Part III).
- `**SetCriterion**` losses (each normalized by `num_boxes`, summed over the final layer + all `aux_outputs` + `enc_outputs`):
    - `loss_ce` — classification. RF-DETR's default is **IoU-aware BCE** (positive target `p^α·IoU^{1-α}`), with varifocal / position-supervised / plain sigmoid-focal alternatives.
    - `loss_bbox` — L1 on `cxcywh`; `loss_giou` — `1−GIoU`.
    - `cardinality_error`, `class_error` — **logging-only diagnostics** (no gradient).
- `**PostProcess**`: `sigmoid` the logits, take a **global top-k** over the flattened `(query × class)` grid (`num_select`, 300 for detection), convert `cxcywh→xyxy`, scale to the original image size.

### II.8 Variant hyperparameters

All inherit `RFDETRBaseConfig` defaults: `hidden_dim=256`, `sa_nheads=8`, `ca_nheads=16`, `dec_n_points=2`, `num_queries=300`, `num_select=300`, `group_detr=13`, `two_stage=True`, `bbox_reparam=True`, `lite_refpoint_refine=True`.

| Variant | encoder | hidden | dec_layers | num_queries | patch | resolution | pos-enc | projector_scale | out_feature_indexes | windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Nano** | dinov2_windowed_small | 256 | 2 | 300 | 16 | 384 | 24 | `["P4"]` | `[3,6,9,12]` | 2 |
| **Small** | dinov2_windowed_small | 256 | 3 | 300 | 16 | 512 | 32 | `["P4"]` | `[3,6,9,12]` | 2 |
| **Medium** | dinov2_windowed_small | 256 | 4 | 300 | 16 | 576 | 36 | `["P4"]` | `[3,6,9,12]` | 2 |
| **Base** | dinov2_windowed_small | 256 | 3 | 300 | 14 | 560 | 37 | `["P4"]` | `[2,5,8,11]` | 4 |
| **Large** | dinov2_windowed_small | 256 | 4 | 300 | 16 | 704 | 44 | `["P4"]` | `[3,6,9,12]` | 2 |
| **Large (deprecated)** | dinov2_windowed_**base** | 384 | 3 | 300 | 14 | 560 | 37 | `["P3","P5"]` | `[2,5,8,11]` | 4 |
| **Keypoint Preview** | dinov2_windowed_small | 256 | 4 | 100 | 12 | 576 | 48 | dual proj. | `[3,6,9,12]` | 2 |

Note the *current* Large is still a ViT-**S** basis (more decoder layers + higher resolution), not a bigger backbone — only the deprecated Large used ViT-B. `positional_encoding_size` auto-syncs to `resolution // patch_size` when you override `resolution`.

> Caveat: the Part II backbone/projector/two-stage line citations were gathered via a codebase sweep; the `LWDETR.forward`/heads/decoder citations were verified directly. Spot-check any specific `file:line` before relying on it.

---

## Part III — Keypoint detection

RF-DETR's keypoint head is a **GroupPose-style** decoder stream bolted onto the LW-DETR detection transformer. Instead of a separate pose network, every detection query spawns a small set of *keypoint query tokens* that attend to image features and to each other, and regress keypoint offsets **relative to the detection's predicted box**. The head is **probabilistic**: each keypoint predicts not just a location but a 2-D Gaussian uncertainty (a precision-Cholesky factor), which feeds a negative-log-likelihood loss, a confidence-fusion step at inference, and pixel-space covariance ellipses for visualization.

### 0. The central data structure: the 8-channel keypoint slot

Everything downstream is organized around a fixed per-keypoint channel layout (`KEYPOINT_PRED_DIM = 8`):

| Slot | Name | Meaning |
| --- | --- | --- |
| 0 | `x` | Normalized x (image-relative, `[0,1]`) |
| 1 | `y` | Normalized y (image-relative, `[0,1]`) |
| 2 | `findable` | Logit for "annotator could find this keypoint" (COCO `v > 0`) |
| 3 | `visible` | Logit for "fully visible" (COCO `v == 2`) |
| 4 | `L_xx` | `log_l11` — log of Cholesky diagonal term 1 |
| 5 | `L_xy` | `l21` — Cholesky off-diagonal (raw, not log) |
| 6 | `L_yy` | `log_l22` — log of Cholesky diagonal term 2 |
| 7 | `class` | Per-keypoint contribution aggregated into the detection-class logit |

Slots 4–6 parameterize a lower-triangular **precision** Cholesky factor $L = [[e^{\log l_{11}}, 0], [l_{21}, e^{\log l_{22}}]]$, so the precision matrix is $P = LL^\top$ and the covariance is $P^{-1}$. Storing the diagonals in log-space guarantees positive definiteness for any real output. Channel 7 lets keypoint evidence flow back into *classification*. Every loss, cost, and post-process step indexes into this table.

### 1. Architecture overview

```javascript
image
  └─ backbone + encoder ─────────────────► memory (image tokens)
        │                                              │
        │   (two-stage) ─► enc keypoint predictions    │
        ▼                                              ▼
  detection queries (tgt) ──► ConditionalQueryInitializer ──► keypoint queries (tgt_keypoints)
        │                                              │
        ▼  decoder layer × N (lite_refpoint_refine)    │
   per layer:
     • box-query self-attn + deformable cross-attn + FFN  (tgt)
     • keypoint subnetwork:
         - instance↔keypoint self-attn (class-masked)
         - [opt] inter-instance keypoint attn
         - [opt] keypoint deformable cross-attn (own memory)
         - keypoint FFN
        │ hs (box embeds)            │ keypoint_hs (keypoint embeds)
        ▼                            ▼
   class_embed, bbox_embed     keypoint_embed (MLP → 8 ch per kpt)
        │   ◄──── class-logit boost (slot 7) ─────┘
        ▼                            ▼
   pred_logits, pred_boxes      pred_keypoints  (class-padded layout)
                                     ▼
        SetCriterion (train) / PostProcess (infer) / COCOeval (OKS)
```

Master switch: `**use_grouppose_keypoints**`. When false, none of the keypoint modules are constructed and the model is a plain detector.

### 2. Keypoint queries: `ConditionalQueryInitializer`

Each detection query is turned into `K` keypoint query tokens. Rather than a fixed embedding, RF-DETR uses **AdaLN (adaptive layer-norm) modulation** (the DiT conditioning trick) so the keypoint queries are *conditioned on the specific detection embedding*:

```python
def modulate(features, scale, shift):
    return (scale + 1.0) * features + shift

class ConditionalQueryInitializer(nn.Module):
    def __init__(self, dim, num_queries, out_dim=None):
        super().__init__()
        out_dim = out_dim or dim
        self.queries = nn.Parameter(torch.randn(num_queries, out_dim))
        self.query_norm = nn.LayerNorm(out_dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, out_dim * 3),
        )
        ada_ln_projection = cast(nn.Linear, self.adaLN_modulation[-1])
        nn.init.constant_(ada_ln_projection.weight, 0)
        nn.init.constant_(ada_ln_projection.bias, 0)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, query_features):
        normed = self.query_norm(self.queries)
        modulation = self.adaLN_modulation(query_features.unsqueeze(-2))
        scale, shift, gate = modulation.chunk(3, dim=-1)
        return self.out_proj(modulate(normed, scale, shift)) * gate + self.queries
```

**Design notes worth copying:**

- The final modulation projection is **zero-initialized** and the result is gated and **residual-added** to the base queries. So at init the keypoint queries equal the learned `self.queries` regardless of conditioning — the network *learns* to deviate. Critical for stable training of a new head on a pretrained detector.
- `num_queries` here means **total keypoints** = `sum(num_keypoints_per_class)`, not the detection query count.
- `out_dim` may be smaller than `d_model` via `grouppose_keypoint_dim_downscale` to save compute.

Two instances are built: `keypoint_query_initializer` (decoder queries) and `keypoint_query_initializer_enc` (two-stage encoder memory).

### 3. The decoder keypoint subnetwork

**Position embeddings + class mask.** `keypoint_pos_embed` is a learned positional code, one row per keypoint, shared across all instances/batch. `**keypoint_class_mask**` is a boolean attention mask of shape `(1 + total_kp, 1 + total_kp)`. Row/col 0 is the *instance token*; the rest are keypoints laid out class-block by class-block. The mask is `True` (blocked) for any keypoint-pair belonging to **different classes** — so a person's keypoints never attend to a hand's keypoints, but every keypoint can still attend to the instance token (row 0 never masked). This is what makes one shared decoder support heterogeneous keypoint schemas.

**Decoder requires **`**lite_refpoint_refine**` (`assert`): the reference boxes are computed once and reused across layers, so keypoint offsets (predicted relative to the box) stay consistent layer to layer.

**(a) Instance↔keypoint self-attention** — the heart of GroupPose. The instance embedding is projected and concatenated as a leading token in front of its `K` keypoint tokens; attention runs over the `(1 + K)` group, masked by `keypoint_class_mask`:

```python
tgt_expanded = tgt_for_kp.unsqueeze(2)                       # [B, N, 1, C]
combined_feat = torch.cat([tgt_expanded, keypoint_tgt], dim=2)   # [B, N, 1+K, C]
combined_pos  = torch.cat([query_expanded, keypoint_pos], dim=2)
combined_feat = combined_feat.reshape(bs * num_queries, 1 + num_kp, kp_dim)
q = k = combined_feat + combined_pos.reshape(bs*num_queries, 1+num_kp, kp_dim)
combined_out = self.kp_inst_self_attn(q, k, combined_feat, attn_mask=keypoint_class_mask)[0]
combined_out = combined_out.reshape(bs, num_queries, 1 + num_kp, kp_dim)
tgt2, keypoint_tgt2 = combined_out[:, :, 0, :], combined_out[:, :, 1:, :]
tgt = tgt + self.kp_inst_dropout(self.inst_out_proj(tgt2)) * self.instance_kp_layer_scale
keypoint_tgt = keypoint_tgt + self.kp_dropout(keypoint_tgt2)
```

The instance token's update is scaled by a **near-zero learnable gate** (`instance_kp_layer_scale = ones(1) * 1e-6`) before being added back to `tgt`, so keypoints initially do **not** perturb the detection path — again "start as identity, learn to contribute."

**(b) Inter-instance keypoint attention** (optional, `inter_instance_kp_attn`): keypoints at the *same index* across different instances attend to each other (e.g. all "left shoulders"). Implemented by transposing the instance/keypoint axes.

**(c) Keypoint deformable cross-attention** (optional, `keypoint_cross_attn`, default **True**): each keypoint token does an `MSDeformAttn` into image memory, using the **detection box** reference point as its sampling location. It can use a *separate* memory tensor (`kp_cross_attn_memory`) when `dual_projector` is enabled.

**(d) Keypoint FFN** — a standard residual MLP. The layer returns `(tgt, keypoint_tgt)`; the decoder stacks `keypoint_tgt` across layers for auxiliary losses. When `grouppose_keypoint_dim_downscale > 1`, the keypoint substream runs at `kp_dim = d_model // downscale` with Linear in/out projections; when `== 1` those are `nn.Identity()`.

### 4. From decoder embeddings to `pred_keypoints`

**Keypoint head** — a 2-layer `MLP` maps each keypoint embedding to the **8-channel delta**; the last layer is zero-init so at the start every keypoint predicts delta ≈ 0.

**Coordinate decoding — offsets relative to the box:**

```python
outputs_keypoints_delta = self.keypoint_embed(keypoint_hs)
ref_wh = ref_unsigmoid[..., 2:].unsqueeze(-2); ref_xy = ref_unsigmoid[..., :2].unsqueeze(-2)
keypoints_xy = outputs_keypoints_delta[..., :2] * ref_wh + ref_xy
keypoints_other = outputs_keypoints_delta[..., 2:]
# ... _format_keypoint_output (compact -> class-padded) ...
outputs_class = outputs_class + self._aggregate_keypoint_class_logits(outputs_keypoints)
```

The head predicts `(dx, dy)` in *box-fraction units*; final position is `box_center + delta * box_size` — the analog of DETR's box-relative refinement, scale-invariant. **No sigmoid** on keypoint xy (can place keypoints slightly outside the box). The other 6 channels pass through raw.

**Compact → class-padded layout:** internally keypoints are compact (`sum(num_keypoints_per_class)` rows, packed class block after class block); the model expands them to `(num_classes × max_kp)` so every class has a fixed-width slot, zero-filled for classes with fewer keypoints. `out["pred_keypoints"]` finally has shape `(B, Q, num_classes*max_kp, 8)`.

**Keypoint→class logit boost:** slot 7 of every active keypoint is summed per class and **added to the classification logits** (masked by `_kp_active_mask`). So a query is more likely "person" if its person-keypoints look confident — coupling keypoint quality to detection confidence and letting classification gradient flow into the keypoint head. When `two_stage`, the encoder memory also produces keypoint predictions with their own boost and `enc_outputs` aux loss.

### 5. The losses

After Hungarian matching, the criterion computes **four** keypoint losses. `target_areas = box_w * box_h` (there is *no* separate keypoint area). Notation for a matched target $t$ (class $c_t$, box area $a_t$), keypoint index $k$: predicted loc $\hat p_{t,k}$, GT loc $p_{t,k}$, GT visibility $v_{t,k}\in\{0,1,2\}$, findable/visible logits $f_{t,k},s_{t,k}$; $m_{t,k}$ = active-keypoint mask; $\ell_{t,k}$ = location/visibility mask.

**5.1 Masking.** The masks gate which keypoints contribute: `keypoints_loss_mask` ($m_{t,k}$, class's active keypoints), `valid_visibility` ($v_{t,k}>0$ and finite), `valid_xy` (pred+target finite), `valid_area` ($a_t$ finite and $>\varepsilon$), and `location_loss_mask` $\ell_{t,k}=m\wedge v{>}0\wedge\text{xy finite}\wedge\text{area valid}$. Denominators are clamped to 1:

$N^{\text{loc}}_t=\max\!\Big(1,\ \textstyle\sum_k \ell_{t,k}\Big),\qquad N^{\text{act}}_t=\max\!\Big(1,\ \textstyle\sum_k m_{t,k}\Big).$

**5.2 Location loss (area-normalized L1):**

$\mathcal{L}^{\text{loc}}_t=\frac{1}{N^{\text{loc}}_t}\sum_{k}\frac{\ell_{t,k}\,\big(\,|\hat{x}_{t,k}-x_{t,k}|+|\hat{y}_{t,k}-y_{t,k}|\,\big)}{\sqrt{a_t}}.$

Dividing by $\sqrt{a_t}$ (a characteristic length) makes the loss **scale-invariant** — a small and a large person incur comparable loss for the same *relative* error. Only visible, in-frame keypoints contribute.

**5.3 Findable & visible BCE.** With $\mathrm{BCE}(z,y)=-[y\log\sigma(z)+(1-y)\log(1-\sigma(z))]$, targets $\mathbb{1}[v>0]$ (labeled at all) and $\mathbb{1}[v>1]$ (fully visible):

$\mathcal{L}^{\text{find}}_t=\frac{1}{N^{\text{act}}_t}\sum_{k} m_{t,k}\,\mathrm{BCE}\!\big(f_{t,k},\,\mathbb{1}[v_{t,k}>0]\big),\qquad \mathcal{L}^{\text{vis}}_t=\frac{1}{N^{\text{act}}_t}\sum_{k} m_{t,k}\,\mathrm{BCE}\!\big(s_{t,k},\,\mathbb{1}[v_{t,k}>1]\big).$

Both are averaged over **all active keypoints** (not just visible ones), so the model is explicitly supervised to predict *low* findability/visibility for absent keypoints — what lets you threshold keypoints at inference.

**5.4 Gaussian NLL (the probabilistic term).** Each keypoint predicts a lower-triangular precision Cholesky $L_{t,k}=\begin{bmatrix} e^{\lambda^{1}} & 0\\ r & e^{\lambda^{2}}\end{bmatrix}$ ($\lambda^1=$`log_l11`, $r=$`l21`, $\lambda^2=$`log_l22`), so $P=LL^\top$ is SPD by construction. For residual $d=\hat p-p$, the squared Mahalanobis distance factors as

$d^{\top}P\,d=\lVert L^{\top}d\rVert_2^2=\underbrace{(e^{\lambda^1}d_x+r\,d_y)}_{u_0}{}^{2}+\underbrace{(e^{\lambda^2}d_y)}_{u_1}{}^{2}.$

Using $\tfrac12\log\det P=\lambda^1+\lambda^2$ and dividing the data term by the area, the per-keypoint loss is

$\text{nll}_{t,k}=\frac{1}{2}\,\frac{u_0^2+u_1^2}{a_t}\;-\;\big(\lambda^{1}_{t,k}+\lambda^{2}_{t,k}\big),$

aggregated over keypoints with a valid mask. The **data term** $\tfrac12 d^\top P d/a_t$ punishes overconfidence when wrong; the **log-normalizer** $-(\lambda^1+\lambda^2)$ pushes precision up; the equilibrium is **calibrated uncertainty**. Because the normalizer is unbounded below, a valid `nll_loss` **can be negative** — expected, not a bug. Deliberate choices: Cholesky params **not clamped** ("clamping would mask divergence"); constant $\log(2\pi)$ omitted; heavy `isfinite`/`nan_to_num` guarding.

**5.5 Aggregation.** Each per-target loss is summed and normalized by `num_boxes`. Total keypoint objective with coefficients from `weight_dict`:

$\mathcal{L}_{\text{kpt}}=\frac{1}{N_{\text{box}}}\sum_{t}\Big(\beta_{\text{l1}}\mathcal{L}^{\text{loc}}_t+\beta_{\text{find}}\mathcal{L}^{\text{find}}_t+\beta_{\text{vis}}\mathcal{L}^{\text{vis}}_t+\beta_{\text{nll}}\mathcal{L}^{\text{nll}}_t\Big).$

Coefs default **0** on base `TrainConfig` and **1** in `KeypointTrainConfig` (which also sets `cls_loss_coef=2.0`). Aux losses replicate all four per decoder layer + two-stage encoder.

**5.6 Design rationale (transfers to your own head):** four separate losses (regression / findability / visibility / uncertainty) so each is tuned/matched independently; **L1 not L2** (constant gradient, robust to mislocated joints); **divide by **$\sqrt{a_t}$ to align with OKS scale; **two independent BCE heads not a 3-way softmax** (COCO visibility is ordinal/nested); findability/visibility averaged over **all** active keypoints (teaches "predict not-findable"); predict the **precision Cholesky** not covariance (SPD for free, no matrix inverse in the hot path, $\log\det$ linear in outputs); log-parameterize the diagonal; matcher reuses the **exact** loss formulas.

### 6. Hungarian matching with keypoints

The assignment cost is class + L1-box + GIoU **plus** four keypoint costs (`keypoint_{l1,findable,visible,nll}_loss_coef`). L1 and NLL mirror §5 but vectorized query×target; iterated **per keypoint class** (classes with 0 keypoints skip). Findable/visible use a closed-form pairwise BCE-with-logits:

```python
def _cdist_bce_with_logits(x, y):
    softplus = F.softplus(x).sum(dim=1, keepdim=True)
    dot = torch.matmul(x, y.to(x.dtype).t())
    return softplus - dot
```

Using $\mathrm{BCE}(x,y)=\operatorname{softplus}(x)-xy$, the summed BCE over keypoints is

$\sum_{k}\mathrm{BCE}(x_{q,k},y_{t,k})=\underbrace{\textstyle\sum_k\operatorname{softplus}(x_{q,k})}_{\text{per-query, broadcast}}-\;x_q\!\cdot y_t,$

so the full $Q\times N$ pairwise cost is one matrix product $x\,y^\top$ without ever materializing the $Q\times N\times K$ tensor. **Match-consistency principle:** the matching cost and the training loss use the *same* formulas — diverge them and the assignment won't optimize what you train.

### 7. Inference post-processing

11. **Top-k selection** — sigmoid the (keypoint-boosted) class logits, pick top `num_select` query/class pairs.
12. **Gather** keypoint rows for the selected queries; reshape to `(num_select, num_classes, max_kp, 8)` and index each detection's *predicted class* keypoints.
13. **Decode to pixels** per class: `x*img_w`, `y*img_h`, `slot2.sigmoid()` → confidence; keep raw Cholesky (slots 4–6). Output keypoints `(N, max_kp, 3)` = `(x_px, y_px, confidence)`.

**7.1 Trace fusion — uncertainty raises/lowers the score.** If `trace_alpha > 0` (default **0.2**), each detection's score is multiplied by $\exp(-\alpha\,\bar T)$ where $\bar T$ is the log of the findability-weighted mean covariance trace. Per keypoint, $\operatorname{tr}(\Sigma)=e^{-2\lambda^1}+e^{-2\lambda^2}+r^2 e^{-2\lambda^1}e^{-2\lambda^2}$, summed in log-space via `logsumexp`, then

$\bar T=\operatorname{logsumexp}_k\!\big(\log\operatorname{tr}(\Sigma_k)+\log w_k\big)-\operatorname{logsumexp}_k\!\big(\log w_k\big),\quad w_k=\sigma(f_k).$

Sharper keypoints → smaller trace → smaller $\bar T$ → larger multiplier → higher detection score. The fused score ranks detections in COCO AP; staying in log-space keeps it stable for very sharp/diffuse predictions.

**7.2 Cholesky → pixel covariance (visualization).** The analytic inverse $\Sigma=P^{-1}$ with $\det P=(l_{11}l_{22})^2$:

$\Sigma=\frac{1}{(l_{11}l_{22})^2}\begin{bmatrix} l_{21}^2+l_{22}^2 & -\,l_{11}l_{21}\\ -\,l_{11}l_{21} & l_{11}^2 \end{bmatrix},$

mapped to pixels by $\Sigma_{\text{px}}=\operatorname{diag}(w,h)\,\Sigma\,\operatorname{diag}(w,h)$. Used only for covariance-ellipse annotators — **never for AP**.

### 8. Evaluation (OKS-based AP)

Keypoint metrics run through `faster_coco_eval`'s `COCOeval(iouType="keypoints")` computing **OKS** (Object Keypoint Similarity), in a separate evaluator from the box mAP path. Public facade `**MetricKeypointOKS**` has a torchmetrics-style API; logged stats `keypoint_map_50_95/50/75` + `keypoint_mAR`. `prepare_for_coco_keypoint` flattens `(x,y,conf)` and truncates to the category's real keypoint count — **precision-Cholesky is NOT sent to pycocotools**; OKS sees only `(x,y,v)` and the fixed `kpt_oks_sigmas`. Sigma resolution: user sigmas validated; absent + 17 keypoints → COCO person defaults; absent + non-17 → uniform `0.05` with a warning. Multi-class → `_GroupedKeypointCOCOeval` (one COCOeval per keypoint-count group). **Asymmetry to remember:** predicted precision is used in the NLL loss, score fusion, and visualization — but **never** in OKS AP.

### 9. Data pipeline

- **Target construction:** COCO `[x,y,v]` → `target["keypoints"]` `(N, K, 3)` float32, `K=max(num_keypoints_per_class)`, tail-padded/truncated; instances with no keypoints get an all-zero block. Coordinates start in absolute pixels; visibility carried verbatim. **Instances are NOT filtered by keypoint visibility** — occluded subjects stay for box/class supervision.
- **Augmentation (Albumentations / CPU backend):** keypoints travel through the **same** geometric transform as image/boxes, `KeypointParams(format="xy", remove_invisible=False)`, visibility as a *label field*. Out-of-frame or `v≤0` keypoints are hard-reset to `(0,0,0)`. **Horizontal-flip left/right swap** is *detected* by box-center mirroring and swaps paired joints via `keypoint_flip_pairs` — without correct flip pairs, horizontal flip silently corrupts left/right keypoints.
- **Final normalization:** divide x by width, y by height → normalized `[0,1]` (matches box `cxcywh`).
- **⚠️ GPU/Kornia backend does NOT transform keypoints** in this preview — **use the CPU/Albumentations backend for keypoint training** or keypoints will be un-augmented while the image is augmented.

### 10. Configuration & wiring

| Flag | Default | Meaning |
| --- | --- | --- |
| `use_grouppose_keypoints` | `False` | Master switch for the whole keypoint path. |
| `keypoint_cross_attn` | `True` | Keypoint deformable cross-attn into image memory. |
| `inter_instance_kp_attn` | `False` | Same-index keypoints attend across instances. |
| `grouppose_keypoint_dim_downscale` | `1` | `kp_dim = d_model // this`. |
| `dual_projector` / `dual_projector_kp_only` | `False` | Separate feature projector for keypoint cross-attn. |
| `num_keypoints_per_class` | `[]` | Schema: per-class keypoint count (index = class id; 0 allowed). |
| `postprocess_trace_alpha` | `0.2` (`≥0`) | Uncertainty→score fusion strength. |

`TrainConfig`: `keypoint_flip_pairs=[]`, four `keypoint_*_loss_coef=0`, `keypoint_oks_sigmas=None`. `KeypointTrainConfig` overrides the four coefs to `1` and `cls_loss_coef` to `2.0`. Schema is inferred from the COCO file (`infer_coco_keypoint_schema`); for Roboflow datasets dataset metadata **wins** over a user schema, then `num_keypoints_per_class` is threaded into transformer/matcher/criterion/postprocessor and stored in the checkpoint via `_kp_active_mask`. Only `**RFDETRKeypointPreview**` enables keypoints: `use_grouppose_keypoints=True`, `dual_projector=True`, `dual_projector_kp_only=True`, `num_keypoints_per_class=[0, 17]` (person), `resolution=576`, `dec_layers=4`, `num_queries=100`; weight `rf-detr-keypoint-preview-xlarge.pth`.

### 11. Re-implementation checklist

**Representation:** (1) fixed 8-channel slot with log-space Cholesky diagonals; (2) box-relative offsets `kpt = box_center + delta·box_size`, no sigmoid; (3) class-padded layout + `_kp_active_mask`. **Architecture:** (4) spawn keypoint queries via AdaLN with zero-init gate + residual; (5) per-layer instance↔keypoint self-attn with a class mask; (6) gate the keypoint→instance update with a ~0 scalar; (7) class-logit boost; (8) require lite refpoint refinement. **Training:** (9) four losses (area-norm L1, findable BCE, visible BCE, Gaussian NLL); (10) same formulas in matcher + loss, closed-form `softplus−dot` BCE; (11) don't clamp Cholesky, guard with isfinite, clamp only normalizers; (12) normalize by `num_boxes`, replicate over aux layers + encoder. **Data:** (13) `(N,K,3)` float32 tail-padded, keep occluded; (14) augment keypoints through the same transform, visibility as label field; (15) **detect** horizontal flips and swap left/right joints (the #1 silent-correctness bug); (16) defer `[0,1]` normalization to the end. **Inference/eval:** (17) decode xy→pixels, findable→sigmoid confidence, keep raw Cholesky; (18) optional score fusion `exp(−α·log_mean_trace)`; (19) OKS AP needs only `(x,y,conf)` + sigmas.

### Source map

| Concern | File |
| --- | --- |
| 8-ch layout, losses, matching cost, query init | `models/heads/keypoints.py` |
| Decoder keypoint subnetwork, class mask, cross-attn | `models/transformer.py` |
| Keypoint head, coord decode, class boost, schema buffers | `models/lwdetr.py` |
| Loss aggregation | `models/criterion.py` |
| Matcher cost assembly | `models/matcher.py` |
| Post-process, trace fusion | `models/postprocess.py` |
| Cholesky → pixel covariance | `utilities/keypoints.py` |
| Visualization | `visualize/keypoints.py` |
| Schema inference | `datasets/_keypoint_schema.py` |
| Target build, augmentation, flip pairs | `datasets/coco.py`, `datasets/transforms.py` |
| GPU backend (keypoints NOT augmented) | `datasets/kornia_transforms.py` |
| OKS evaluation | `evaluation/coco_eval.py`, `evaluation/keypoint_oks.py` |
| Config / variant / checkpoint | `config.py`, `variants.py`, `detr.py`, `assets/model_weights.py` |