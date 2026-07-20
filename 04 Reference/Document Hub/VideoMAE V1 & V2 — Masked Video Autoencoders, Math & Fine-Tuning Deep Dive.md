---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-19T15:19:36
Status: Done
Last updated time: 2026-07-19T15:19:36
Last edited by: Heaven Chen
Category:
  - CV
  - Video Understanding
  - Self-Supervised Learning
  - Transformers
  - Action Recognition
---
# VideoMAE V1 & V2 Deep Dive — Masked Video Autoencoders, Math, and Fine-Tuning

> [!abstract]
> **VideoMAE V1** adapts masked autoencoding to video using tubelet tokens, an extremely high encoder masking ratio, and temporally consistent tube masking. **VideoMAE V2** retains that encoder but adds decoder-side masking, making large-scale pretraining substantially cheaper and enabling billion-parameter video models. For downstream recognition, both discard the pretraining decoder, restore the full unmasked video token sequence, mean-pool encoder features, and fine-tune a classification head—usually together with some or all encoder layers.

**Primary references:** [VideoMAE V1, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/416f9cb3276121c42eebb86352a4354a-Abstract-Conference.html) · [official V1 repository](https://github.com/MCG-NJU/VideoMAE) · [VideoMAE V2, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_VideoMAE_V2_Scaling_Video_Masked_Autoencoders_With_Dual_Masking_CVPR_2023_paper.html) · [official V2 repository](https://github.com/OpenGVLab/VideoMAEv2) · [Hugging Face VideoMAE documentation](https://huggingface.co/docs/transformers/model_doc/videomae)

**Vault context:** [[VideoMAE Masked Autoencoders Are Data-Efficient Learners for Self-Supervised Video Pre-Training|VideoMAE paper note]] · [[04 Reference/Document Hub/ResNet Architecture|2D/3D ResNet]] · [[04 Reference/Document Hub/I3D — Inflated 3D ConvNet Deep Dive for Braking-Action Recognition|I3D]] · [[04 Reference/Document Hub/Computer Vision Loss|Computer Vision Loss]]

> [!important] Repository scope
> V1 and V2 have separate official codebases. The architecture details here were checked against V1 commit [`14ef8d8`](https://github.com/MCG-NJU/VideoMAE/tree/14ef8d856287c94ef1f985fe30f958eb4ec2c55d) and V2 commit [`29eab1e`](https://github.com/OpenGVLab/VideoMAEv2/tree/29eab1e8a588d1b3ec0cdec7b03a86cca491b74b). Record the repository commit, checkpoint, clip shape, and sampling policy in every experiment.

# Part I — Foundations shared by V1 and V2

## 1. Why masked video modeling?

Supervised video models require expensive action labels. Contrastive self-supervision reduces the labeling burden but often needs multiple augmented views, large batches, negative examples, or carefully designed invariances. VideoMAE instead constructs its target directly from the input pixels:

```text
unlabeled clip
  → split into spatiotemporal tubelets
  → hide most tubelets
  → encode only visible tubelets
  → lightweight decoder predicts hidden pixels
  → discard decoder
  → transfer encoder to downstream tasks
```

The pretext task is deliberately difficult. Video has strong local redundancy: adjacent frames often differ only slightly. If too many tokens remain visible, the network can reconstruct by copying local texture rather than learning action-relevant structure. VideoMAE therefore masks roughly **90–95%** of tokens in its main experiments.

## 2. Tubelet tokenization

Let the video be

$$
X\in\mathbb{R}^{B\times C\times T\times H\times W}.
$$

A tubelet has temporal depth $\tau$ and spatial patch size $P\times P$. A Conv3d with kernel and stride $(\tau,P,P)$ maps each non-overlapping cube to a $D$-dimensional token:

$$
E=\operatorname{Conv3D}_{k=s=(\tau,P,P)}(X).
$$

The token-grid dimensions are

$$
T'=\frac{T}{\tau},\qquad H'=\frac{H}{P},\qquad W'=\frac{W}{P},
$$

so the sequence length is

$$
N=T'H'W'=\frac{T}{\tau}\frac{H}{P}\frac{W}{P}.
$$

For the common configuration $T=16$, $\tau=2$, $H=W=224$, and $P=16$,

$$
N=8\cdot14\cdot14=1568.
$$

Each reconstruction target contains

$$
D_{pix}=C\tau P^2=3\cdot2\cdot16^2=1536
$$

pixel values. This explains the `decoder_num_classes=1536` name in the official patch-16 code: it is the output dimension per reconstructed tubelet, not a semantic class count.

### 2.1 Minimal tubelet embedding code

```python
import torch
from torch import nn


class TubeletEmbedding(nn.Module):
    def __init__(self, embed_dim=768, tubelet_size=2, patch_size=16):
        super().__init__()
        self.projection = nn.Conv3d(
            3,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, video):
        # Official repositories: video is [B,C,T,H,W].
        tokens = self.projection(video)        # [B,D,T',H',W']
        return tokens.flatten(2).transpose(1, 2)  # [B,N,D]


embed = TubeletEmbedding()
x = torch.randn(2, 3, 16, 224, 224)
assert embed(x).shape == (2, 1568, 768)
```

## 3. Transformer encoder mathematics

After adding positional encoding, each encoder block applies multi-head self-attention and an MLP with residual connections. For token matrix $Z\in\mathbb{R}^{N_v\times D}$,

$$
Q=ZW_Q,\qquad K=ZW_K,\qquad V=ZW_V,
$$

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V.
$$

For $h$ heads,

$$
\operatorname{MSA}(Z)
=\operatorname{Concat}(A_1,\ldots,A_h)W_O.
$$

A pre-norm block is conceptually

$$
Z'=Z+\operatorname{MSA}(\operatorname{LN}(Z)),
$$

$$
Z''=Z'+\operatorname{MLP}(\operatorname{LN}(Z')).
$$

If mask ratio is $\rho$, the encoder sees

$$
N_v=(1-\rho)N
$$

visible tokens. With $N=1568$ and $\rho=0.9$, only about 157 tokens reach the encoder.

Self-attention's dominant score-matrix cost is quadratic in token count:

$$
\mathcal O(N_v^2D)
=\mathcal O\left((1-\rho)^2N^2D\right).
$$

At 90% masking, the attention matrix has roughly 1% as many entries as full-sequence attention, ignoring fixed overhead and the linear projections. High masking therefore makes the expensive encoder dramatically cheaper during pretraining.

## 4. Asymmetric masked autoencoder

Let $M_e\in\{0,1\}^N$ be the encoder mask, where 1 means hidden. The visible index set is

$$
\mathcal V=\{i:M_{e,i}=0\},
$$

and the masked set is

$$
\mathcal M=\{i:M_{e,i}=1\}.
$$

The encoder processes only visible tokens:

$$
H_{\mathcal V}=f_\theta(E_{\mathcal V}+P_{\mathcal V}).
$$

A linear layer maps encoder width $D_e$ to smaller decoder width $D_d$. Learned mask tokens are inserted at hidden positions, positional embeddings are added, and a lightweight decoder predicts pixels:

$$
\hat X_{\mathcal M}
=g_\phi\left(H_{\mathcal V}W_{e\rightarrow d},m_{\mathcal M},P\right).
$$

The asymmetry is intentional:

- the **deep/wide encoder** learns transferable representations;
- the **shallow/narrow decoder** exists only to define the self-supervised task;
- the decoder is discarded at fine-tuning and inference time.

## 5. Pixel reconstruction target and loss

For each tubelet target $x_i\in\mathbb{R}^{D_{pix}}$, VideoMAE can normalize pixels within the patch:

$$
\tilde x_i=\frac{x_i-\mu_i}{\sqrt{\sigma_i^2+\epsilon}}.
$$

The masked reconstruction loss is mean squared error only over selected hidden tubelets:

$$
L_{MIM}
=\frac{1}{|\mathcal M|}
\sum_{i\in\mathcal M}
\frac{1}{D_{pix}}
\left\|\hat x_i-\tilde x_i\right\|_2^2.
$$

Computing loss only on masked tokens prevents the many visible tokens from dominating with an easy identity objective.

### 5.1 Patchification and loss code

```python
import torch
import torch.nn.functional as F


def patchify_video(video, tubelet=2, patch=16):
    """[B,C,T,H,W] -> [B,N,tubelet*patch*patch*C]."""
    b, c, t, h, w = video.shape
    assert t % tubelet == 0 and h % patch == 0 and w % patch == 0
    x = video.reshape(
        b, c,
        t // tubelet, tubelet,
        h // patch, patch,
        w // patch, patch,
    )
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)
    return x.reshape(b, -1, tubelet * patch * patch * c)


def normalize_patch_targets(targets, eps=1e-6):
    mean = targets.mean(dim=-1, keepdim=True)
    var = targets.var(dim=-1, unbiased=True, keepdim=True)
    return (targets - mean) / torch.sqrt(var + eps)


def masked_mse(predictions, targets, loss_mask):
    """predictions/targets [B,N,D]; loss_mask [B,N] boolean."""
    per_token = F.mse_loss(predictions, targets, reduction="none").mean(-1)
    weights = loss_mask.to(per_token.dtype)
    return (per_token * weights).sum() / weights.sum().clamp(min=1)
```

# Part II — VideoMAE V1

## 6. V1 architecture

VideoMAE V1 combines four simple choices:

1. non-overlapping spatiotemporal tubelet embedding;
2. a vanilla ViT encoder operating only on visible tokens;
3. a lightweight reconstruction decoder;
4. extremely high **tube masking**.

The pretraining path is:

```text
[B,3,T,H,W]
  → Conv3d tubelet embedding
  → N video tokens + positional embeddings
  → remove approximately 90–95% according to tube mask
  → ViT encoder on visible tokens only
  → encoder-to-decoder projection
  → append learned mask tokens with positional embeddings
  → lightweight decoder
  → linear pixel head
  → MSE on masked tubelets
```

The official implementation does not use a classification token for downstream classification. It mean-pools the final encoder tokens.

## 7. Tube masking

Independent random masking in space-time can leak nearly identical content from adjacent frames. Tube masking samples one spatial mask and repeats it through the temporal token grid:

$$
M_e(t,h,w)=m(h,w),
\qquad t=1,\ldots,T'.
$$

If one spatial location is hidden, the entire temporal tube at that location is hidden. Reconstruction must use context from other spatial regions and learned motion/appearance structure rather than an adjacent copy of the same patch.

### 7.1 Tube-mask generator

```python
import torch


def make_tube_mask(batch_size, temporal_tokens, height_tokens,
                   width_tokens, mask_ratio, device=None):
    spatial_tokens = height_tokens * width_tokens
    num_masked = round(mask_ratio * spatial_tokens)
    noise = torch.rand(batch_size, spatial_tokens, device=device)
    order = noise.argsort(dim=1)

    spatial_mask = torch.zeros(
        batch_size, spatial_tokens, dtype=torch.bool, device=device
    )
    spatial_mask.scatter_(1, order[:, :num_masked], True)

    # Same spatial mask at every temporal token index.
    return spatial_mask[:, None, :].expand(
        -1, temporal_tokens, -1
    ).reshape(batch_size, -1)


mask = make_tube_mask(2, 8, 14, 14, mask_ratio=0.9)
assert mask.shape == (2, 1568)
```

## 8. Why 90–95% masking works for video

A high ratio provides two benefits:

1. **semantic pressure:** reconstruction cannot rely solely on local interpolation;
2. **compute reduction:** only a small visible subset enters the encoder.

However, the ideal ratio is data dependent. Very small, thin, or localized signals may disappear under aggressive masking. For factory braking or self-contact, the tool/hand region may occupy few patches. A targeted domain pretraining ablation should compare, for example, 75%, 90%, and 95%, rather than assuming the paper's optimum transfers unchanged.

## 9. V1 pretraining pseudocode

```python
for videos in unlabeled_loader:               # [B,C,T,H,W]
    targets = normalize_patch_targets(patchify_video(videos))
    encoder_mask = make_tube_mask(
        batch_size=videos.shape[0],
        temporal_tokens=videos.shape[2] // 2,
        height_tokens=videos.shape[3] // 16,
        width_tokens=videos.shape[4] // 16,
        mask_ratio=0.90,
        device=videos.device,
    )

    predictions = model(videos, encoder_mask)  # masked-token predictions
    loss = masked_mse(
        predictions,
        targets[encoder_mask].reshape_as(predictions),
        torch.ones(predictions.shape[:2], dtype=torch.bool,
                   device=predictions.device),
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

This illustrates the objective, not the exact official call signature. The official model directly returns only masked-token predictions, so target selection must use exactly the same token order as the model.

# Part III — VideoMAE V2

## 10. Why V2 was needed

V1 already makes the encoder efficient by keeping few visible tokens, but its decoder still processes the encoded visible tokens plus mask tokens for a large part of the grid. At giant scale, decoder attention and activations remain expensive. V2 introduces **dual masking**:

- **encoder masking** keeps the V1 strategy;
- **decoder masking** thins the token grid processed/reconstructed by the decoder.

This enables scaling both model size and pretraining data while keeping masked autoencoding conceptually unchanged.

## 11. Dual masking mathematics

Let $M_e$ be the encoder mask and $M_d$ the decoder-drop mask. The encoder sees

$$
\mathcal V_e=\{i:M_{e,i}=0\}.
$$

The decoder receives encoded visible tokens plus mask tokens at a selected decoder subset. The reconstruction loss is evaluated on positions that are hidden from the encoder but retained for decoder prediction:

$$
\mathcal L_{pred}
=\{i:M_{e,i}=1\ \land\ M_{d,i}=0\}.
$$

Thus

$$
L_{V2}
=\frac1{|\mathcal L_{pred}|}
\sum_{i\in\mathcal L_{pred}}
\frac1{D_{pix}}
\|\hat x_i-\tilde x_i\|_2^2.
$$

The official released giant-model recipe uses 90% tube masking in the encoder and 50% **running-cell** masking in the decoder. Decoder masking changes pretraining efficiency; it is not used during ordinary classification fine-tuning.

### 11.1 Complexity intuition

If the decoder would otherwise process $N_d$ tokens and decoder masking retains fraction $q_d$, its attention score matrices scale approximately as

$$
\mathcal O((q_dN_d)^2D_d).
$$

At a 50% retain ratio this idealized quadratic component falls to about 25%, although total wall-clock savings are smaller because projections, data loading, the encoder, and distributed communication remain.

## 12. Running-cell decoder masking

A purely random decoder mask can produce irregular patterns. V2's running-cell strategy constructs a small binary spatial cell, tiles it over the frame, and shifts its phase over time. The decoder observes a structured, temporally changing subset rather than the same dropped positions in every frame.

Conceptually, for cell $C$ of size $a\times b$ and temporal phase shift $\Delta_t$,

$$
M_d(t,h,w)=C\left((h+\Delta_t^{(h)})\bmod a,
(w+\Delta_t^{(w)})\bmod b\right).
$$

A compact 50%-keep illustration is:

```python
import torch


def running_cell_decoder_drop(batch, temporal_tokens, height, width,
                              device=None):
    """Structured 2x2 mask with a phase that changes over time."""
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    frames = []
    phase = torch.randint(0, 4, (batch,), device=device)
    for t in range(temporal_tokens):
        # Two of four parity states are dropped; phase moves with time.
        state = (2 * (yy % 2) + (xx % 2))[None, :, :]
        drop = ((state + phase[:, None, None] + t) % 4) < 2
        frames.append(drop)
    return torch.stack(frames, dim=1).flatten(1)


encoder_mask = make_tube_mask(2, 8, 14, 14, 0.9)
decoder_drop = running_cell_decoder_drop(2, 8, 14, 14)
loss_mask = encoder_mask & ~decoder_drop
```

This code explains the structured idea; use the official `RunningCellMaskingGenerator` for exact reproduction.

## 13. Progressive training in V2

V2 scales along two axes:

- **model:** from ordinary ViTs to a billion-parameter ViT-giant;
- **data:** a multi-source unlabeled mixture followed by labeled intermediate training.

The progressive recipe is:

```text
Stage 1 — self-supervised pretraining
multi-source unlabeled videos/raw-frame datasets
  → VideoMAE V2 dual-masked reconstruction

Stage 2 — intermediate supervised training (“post-pre-training”)
large mixed labeled action dataset such as Kinetics-710
  → classification fine-tuning

Stage 3 — target-task adaptation
K400 / K600 / UCF101 / HMDB51 / custom task
  → initialize from Stage 2 when available, replace head, fine-tune
```

The intermediate supervised checkpoint often transfers better than the raw reconstruction-pretrained checkpoint because it has already aligned the encoder with semantic action classes.

## 14. V1 versus V2

| Property | VideoMAE V1 | VideoMAE V2 |
| --- | --- | --- |
| Publication | NeurIPS 2022 | CVPR 2023 |
| Encoder input | visible tubelets only | visible tubelets only |
| Main encoder mask | high-ratio tube mask | high-ratio tube mask |
| Decoder masking | no second mask | **yes**, e.g. running-cell mask |
| Decoder loss | encoder-masked tubelets | encoder-masked and decoder-retained tubelets |
| Scaling emphasis | data-efficient learning | model/data scaling |
| Largest demonstrated scale | ViT-H in V1 experiments | billion-parameter ViT-giant |
| Progressive labeled stage | not the central recipe | key part of scaling/transfer |
| Classification fine-tuning | discard decoder, mean-pool encoder | **same method as V1** |
| Official repository | `MCG-NJU/VideoMAE` | `OpenGVLab/VideoMAEv2` |

> [!warning]
> “V2” is not simply a different classifier checkpoint. Its principal architectural change is in **self-supervised pretraining efficiency**. After the decoder is discarded, a V1 and V2 model with the same ViT encoder family can look very similar during downstream classification.

# Part IV — Fine-tuning V1 and V2

## 15. What is retained and what is discarded?

During supervised fine-tuning:

- retain tubelet embedding, positional encoding, and the pretrained ViT encoder;
- discard `encoder_to_decoder`, mask token, reconstruction decoder, and pixel head;
- do not mask tokens;
- mean-pool all final encoder tokens;
- attach a new task head.

For final hidden tokens $H\in\mathbb{R}^{B\times N\times D}$,

$$
g=\operatorname{LN}\left(\frac1N\sum_{i=1}^{N}H_i\right),
$$

$$
z=W_{cls}g+b_{cls}.
$$

For single-label action recognition,

$$
L_{CE}=-\frac1B\sum_{b=1}^{B}\log
\frac{e^{z_{b,y_b}}}{\sum_{k=1}^{K}e^{z_{b,k}}}.
$$

For independent multi-label actions, replace softmax cross-entropy with per-class sigmoid BCE.

## 16. Temporal sampling is part of the model

If $T$ frames are sampled every $q$ source frames from video with frame rate $f$, approximate temporal coverage is

$$
\Delta t=\frac{(T-1)q}{f}.
$$

With 16 frames, stride 4, and 30 FPS,

$$
\Delta t=\frac{15\cdot4}{30}=2\text{ seconds}.
$$

The model sees 16 images, but those images summarize about two seconds. Changing $q$ changes the physical action duration without changing tensor shape.

Dataset choice matters:

- Kinetics-style actions often use dense clip sampling;
- Something-Something relies strongly on ordering and uses more uniform temporal sampling;
- braking/contact requires coverage before, during, and after the interaction.

> [!warning] Avoid temporal preprocessing bugs
> Apply the same spatial crop, flip, and color transform to every frame in a clip. Independent frame crops create synthetic motion. Do not reverse time when action direction changes the label.

## 17. Three fine-tuning regimes

### 17.1 Linear probing

Freeze the complete encoder and train only the new head:

$$
\theta_{enc}\leftarrow\text{fixed},
\qquad
\theta_{head}\leftarrow\theta_{head}-\eta\nabla_{\theta_{head}}L.
$$

Use this as a diagnostic of representation quality, not automatically as the best final model.

### 17.2 Partial fine-tuning

Train the head plus the last few transformer blocks. This is often appropriate when the custom dataset is small and domain shift is moderate.

### 17.3 Full fine-tuning

Train every encoder layer with a small learning rate, warmup, weight decay, stochastic depth, and usually layer-wise learning-rate decay. This adapts most strongly but has the highest overfitting and memory risk.

Recommended progression:

1. head-only baseline;
2. unfreeze last 2–4 blocks;
3. full fine-tuning only if leakage-safe validation improves;
4. compare raw V2-pretrained versus intermediate supervised V2 checkpoints.

## 18. Layer-wise learning-rate decay

For transformer layer index $\ell\in\{0,\ldots,L\}$, assign

$$
\eta_\ell=\eta_{head}\gamma^{L-\ell},
\qquad 0<\gamma<1.
$$

Early layers receive smaller updates; the head and final blocks adapt most. The official scripts commonly expose this as `--layer_decay` (for example 0.75 on ViT-Base and a larger value such as 0.9 for very deep giant models).

The official V1 code scales base learning rate by total batch size:

$$
\eta_{actual}=\eta_{base}\frac{B_{total}}{256},
$$

$$
B_{total}=B_{per\ GPU}\times N_{GPU}\times N_{nodes}\times N_{accum}.
$$

Do not accidentally apply this scaling twice when introducing gradient accumulation or a trainer that already rescales schedules.

## 19. Official V1 fine-tuning command

The original repository expects dataset list files and provides distributed scripts; see its [`FINETUNE.md`](https://github.com/MCG-NJU/VideoMAE/blob/main/FINETUNE.md). A compact four-GPU Kinetics-style command is:

```bash
git clone https://github.com/MCG-NJU/VideoMAE.git
cd VideoMAE
git rev-parse HEAD

OUTPUT_DIR=/path/to/output
DATA_PATH=/path/to/dataset_lists
MODEL_PATH=/path/to/v1_pretrain_checkpoint.pth

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 \
  run_class_finetuning.py \
  --model vit_base_patch16_224 \
  --data_set Kinetics-400 \
  --nb_classes 2 \
  --data_path "$DATA_PATH" \
  --finetune "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$OUTPUT_DIR" \
  --batch_size 4 \
  --num_frames 16 \
  --sampling_rate 4 \
  --input_size 224 \
  --short_side_size 224 \
  --opt adamw \
  --lr 1e-3 \
  --layer_decay 0.75 \
  --weight_decay 0.05 \
  --warmup_epochs 5 \
  --epochs 50 \
  --dist_eval
```

Adapt `--data_set` or the dataset builder for a custom class list. The official README notes that `--lr` is a base LR scaled by total batch size/256. Validate the checkpoint's model size, patch size, tubelet size, and positional-embedding shape before training.

## 20. Official V2 fine-tuning command

V2 uses the same downstream method but a separate repository and checkpoint zoo; see its [`docs/FINETUNE.md`](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/FINETUNE.md):

```bash
git clone https://github.com/OpenGVLab/VideoMAEv2.git
cd VideoMAEv2
git rev-parse HEAD

OUTPUT_DIR=/path/to/output
DATA_PATH=/path/to/custom_lists
MODEL_PATH=/path/to/v2_vit_base_pretrain_or_intermediate_checkpoint.pth

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 \
  run_class_finetuning.py \
  --model vit_base_patch16_224 \
  --data_path "$DATA_PATH" \
  --nb_classes 2 \
  --finetune "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$OUTPUT_DIR" \
  --batch_size 4 \
  --num_frames 16 \
  --sampling_rate 4 \
  --input_size 224 \
  --short_side_size 224 \
  --opt adamw \
  --lr 7e-4 \
  --layer_decay 0.75 \
  --weight_decay 0.05 \
  --warmup_epochs 5 \
  --epochs 50 \
  --dist_eval
```

The V2 repository's documented ViT-Base K400 recipe uses 16 frames, sampling rate 4, AdamW, base LR $7\times10^{-4}$, weight decay 0.05, layer decay 0.75, and 90 epochs. Treat these as reproduction-oriented starting points, not universal values for a small binary dataset.

> [!important] Prefer the official loader for V2
> Checkpoint dictionaries and positional embeddings vary across raw-pretrained, intermediate, distilled, and task-fine-tuned releases. The official `run_class_finetuning.py` contains key filtering and positional interpolation logic. A casual `load_state_dict(..., strict=False)` can silently leave important layers random.

## 21. Minimal Hugging Face V1 fine-tuning loop

Hugging Face provides `VideoMAEForVideoClassification` for V1-style checkpoints. Its tensor layout is `[B,T,C,H,W]`, unlike the official repositories' `[B,C,T,H,W]` layout.

```python
import torch
from torch.optim import AdamW
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
)


checkpoint = "MCG-NJU/videomae-base"
num_classes = 2
id2label = {0: "no_braking", 1: "braking"}
label2id = {name: idx for idx, name in id2label.items()}

processor = VideoMAEImageProcessor.from_pretrained(checkpoint)
model = VideoMAEForVideoClassification.from_pretrained(
    checkpoint,
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,  # new classification head
).cuda()

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.05)
model.train()

for batch in train_loader:
    # pixel_values: [B,T,C,H,W], already normalized by processor
    pixel_values = batch["pixel_values"].cuda(non_blocking=True)
    labels = batch["labels"].cuda(non_blocking=True)

    outputs = model(pixel_values=pixel_values, labels=labels)
    loss = outputs.loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

The pretraining decoder weights in a bare checkpoint are intentionally unused when constructing the classification model; the classification head is newly initialized. `ignore_mismatched_sizes=True` should be used deliberately for head-size changes, not as a blanket fix for an incompatible encoder.

### 21.1 Head-only then partial unfreezing

```python
# Stage 1: linear head.
for parameter in model.videomae.parameters():
    parameter.requires_grad = False
for parameter in model.classifier.parameters():
    parameter.requires_grad = True

# Stage 2: unfreeze the last two encoder blocks.
for block in model.videomae.encoder.layer[-2:]:
    for parameter in block.parameters():
        parameter.requires_grad = True

parameter_groups = [
    {"params": model.classifier.parameters(), "lr": 1e-3},
    {
        "params": [
            p for block in model.videomae.encoder.layer[-2:]
            for p in block.parameters() if p.requires_grad
        ],
        "lr": 1e-5,
    },
]
optimizer = AdamW(parameter_groups, weight_decay=0.05)
```

Confirm module names against the installed `transformers` version before relying on them.

## 22. Minimal clip decoding and processing

This correctness-first dataset decodes an entire video with PyAV, then samples a fixed-stride clip. Production training should use seek-aware decoding or an indexed video dataset.

```python
import av
import numpy as np
import torch
from torch.utils.data import Dataset


def decode_rgb_frames(path):
    container = av.open(path)
    frames = [
        frame.to_ndarray(format="rgb24")
        for frame in container.decode(video=0)
    ]
    container.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames


def sample_indices(num_available, num_frames=16, stride=4, training=True):
    span = (num_frames - 1) * stride + 1
    if num_available >= span:
        max_start = num_available - span
        start = np.random.randint(max_start + 1) if training else max_start // 2
        return start + np.arange(num_frames) * stride

    # Uniform fallback for short videos; repeated indices are possible.
    return np.linspace(0, num_available - 1, num_frames).round().astype(int)


class VideoClassificationDataset(Dataset):
    def __init__(self, samples, processor, training=True,
                 num_frames=16, stride=4):
        self.samples = samples  # list of (video_path, integer_label)
        self.processor = processor
        self.training = training
        self.num_frames = num_frames
        self.stride = stride

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        frames = decode_rgb_frames(path)
        ids = sample_indices(
            len(frames), self.num_frames, self.stride, self.training
        )
        clip = [frames[i] for i in ids]
        pixel_values = self.processor(
            clip, return_tensors="pt"
        )["pixel_values"].squeeze(0)  # [T,C,H,W]
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }
```

Check processor behavior with one sample before a long run. Print min/max, shape, frame order, and a visualization after all transforms.

## 23. Fine-tuning objective choices

### Single-label classification

Use cross-entropy when exactly one class is correct:

```python
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
# logits [B,K], labels [B]
loss = criterion(logits, labels)
```

### Binary one-logit classification

Use BCE only if the head emits one logit:

$$
P(y=1\mid X)=\sigma(z),
$$

```python
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
# logits [B] or [B,1], targets float with matching shape
```

Do not combine a two-logit softmax head with `BCEWithLogitsLoss` unless the task is intentionally multi-label.

### Label smoothing

For target distribution $q$ with smoothing $\epsilon$,

$$
q_k=(1-\epsilon)\mathbf1[k=y]+\frac{\epsilon}{K}.
$$

It can improve calibration but may hurt rare-class recall on small imbalanced datasets; validate rather than inheriting the large-scale recipe blindly.

## 24. Multi-view inference

Official results often use multiple temporal clips and spatial crops. If there are $V$ views,

$$
\bar z=\frac1V\sum_{v=1}^{V}z^{(v)},
\qquad
\hat y=\arg\max_k\bar z_k.
$$

For example, `16×5×3` means 16 input frames, 5 temporal clips, and 3 spatial crops—15 model views per video. This is much more expensive than single-view inference, so report:

- frames per view;
- temporal clips;
- spatial crops;
- resolution;
- total decoded frames;
- end-to-end latency.

Never compare a 15-view VideoMAE result with a one-view 3D ResNet result as if compute were equal.

## 25. Memory and optimization guidance

Video transformers are activation-heavy. Useful controls are:

- mixed precision (`bfloat16` where supported);
- gradient accumulation;
- activation checkpointing;
- fewer frames or lower resolution;
- partial freezing;
- gradient clipping;
- distributed data parallelism or DeepSpeed for large V2 models.

Attention memory grows roughly as

$$
\mathcal O(BLN^2)
$$

for batch size $B$, layers $L$, and token count $N$. Doubling temporal frames approximately doubles $N$ and can quadruple attention-map memory during unmasked fine-tuning.

If effective batch size changes, distinguish:

$$
B_{micro},\quad B_{accum},\quad B_{world},\quad
B_{effective}=B_{micro}B_{accum}B_{world}.
$$

Log all four values.

## 26. Checkpoint and preprocessing compatibility

Before fine-tuning, verify:

| Item | Why it matters |
| --- | --- |
| V1 vs V2 repository | checkpoint keys and model registrations differ |
| model size | hidden width/depth must match |
| patch size | changes spatial token grid and pixel head |
| tubelet size | changes temporal token grid |
| frame count | positional embeddings may need interpolation |
| resolution | positional embeddings may need spatial interpolation |
| raw pretrain vs intermediate supervised | transfer behavior and head keys differ |
| channel normalization | distribution must match pretraining |
| tensor layout | official repo is BCTHW; HF is BTCHW |
| label mapping | checkpoint class order may not match custom classes |

> [!warning] Silent partial loading
> A large list of “missing” and “unexpected” keys is not success. Expected differences are normally the decoder and new classifier head. Unexpected tubelet embedding or transformer-block mismatches indicate the wrong architecture or checkpoint.

## 27. Data splitting and augmentation

For video tasks, random clip splitting is often leakage. Split by worker, subject, session, source video, or production run. All overlapping clips from one source must remain in one split.

Usually safe augmentations:

- consistent spatial crop/resize across the clip;
- mild color jitter, blur, and compression;
- random erasing when it does not remove the only action cue;
- horizontal flip only when left/right semantics are label invariant.

Potentially destructive augmentations:

- temporal reversal;
- frame-order shuffling;
- independent per-frame crop;
- aggressive crop that removes the hand/tool/contact area;
- speed changes that alter the action definition.

## 28. VideoMAE for braking and temporal self-contact

VideoMAE is attractive when labels are scarce but unlabeled domain video is abundant. A practical sequence is:

```text
large unlabeled factory video
  → optional domain-adaptive VideoMAE V1/V2 pretraining
  → initialize encoder from public or domain checkpoint
  → supervised clip classification or temporal feature extraction
  → braking / no-braking or contact / no-contact
```

For the poka-yoke pipeline:

```text
frame-level detector
  → worker + hand/tool + work-area boxes
  → track and construct consistent interaction ROI clip
  → VideoMAE encoder
  → classification or temporal-localization head
```

Full-frame VideoMAE may waste tokens on static background. Compare:

1. full frame;
2. worker crop;
3. union of worker, tool, and work-area boxes;
4. RGB crop plus explicit masks/box geometry in a separate fusion head.

For long untrimmed videos, a clip classifier is not automatically a temporal localizer. Extract overlapping VideoMAE features and pass them to a localization model such as ActionFormer/TriDet, or add a frame/segment-level head.

## 29. Failure modes and debugging

| Symptom | Likely cause |
| --- | --- |
| Model trains but accuracy stays random | label map, input layout, or head dimension wrong |
| Loss is finite but pretrained benefit disappears | encoder failed to load; too many missing keys |
| Training accuracy rises immediately, validation collapses | source-video leakage or overfitting |
| Predictions ignore motion | temporal stride too small/large, static shortcut, or crop hides action |
| OOM after increasing frames from 16 to 32 | quadratic attention growth |
| V2 code loads V1 weights poorly | wrong repository/model registry |
| Small contact event is missed | temporal sampling skips it or spatial crop/resolution is inadequate |
| Different results despite same checkpoint | normalization, view count, or sampling policy differs |
| Strong clip accuracy but poor event boundaries | classification objective lacks temporal localization supervision |

## 30. Recommended ablations

| Ablation | Question answered |
| --- | --- |
| V1 vs V2 encoder of comparable size | Does V2 pretraining improve transfer beyond scale? |
| raw V2 pretrain vs intermediate supervised checkpoint | How valuable is progressive training? |
| head-only vs last blocks vs full fine-tune | How much adaptation is required? |
| public pretrain vs domain-adaptive masked pretrain | Does factory-domain video help? |
| 8/16/32 frames | How much temporal context is needed? |
| sampling stride 1/2/4/8 | Which physical duration captures braking/contact? |
| full frame vs interaction ROI | Does background create shortcuts? |
| 75/90/95% mask for domain pretraining | Does extreme masking hide small action cues? |
| single-view vs multi-view evaluation | Is accuracy worth inference cost? |
| VideoMAE vs [[04 Reference/Document Hub/ResNet Architecture|3D ResNet]] vs [[04 Reference/Document Hub/I3D — Inflated 3D ConvNet Deep Dive for Braking-Action Recognition|I3D]] | Does masked-transformer pretraining justify complexity? |

## 31. Bottom line

VideoMAE V1's key insight is that video redundancy permits an extremely high tube-mask ratio: a powerful ViT encoder can learn from the small visible subset while a lightweight decoder reconstructs normalized hidden tubelets. VideoMAE V2 preserves that representation-learning mechanism and adds decoder masking plus progressive data/model scaling, making billion-parameter pretraining feasible.

For downstream fine-tuning, the distinction becomes operationally simpler: **discard the decoder, feed unmasked clips, mean-pool encoder tokens, replace the head, and fine-tune with checkpoint-compatible preprocessing and temporal sampling**. Start from a pretrained or intermediate V2 checkpoint rather than training a large model from scratch, use leakage-safe source-video splits, and compare ROI choice and temporal coverage before attributing gains to architecture.
