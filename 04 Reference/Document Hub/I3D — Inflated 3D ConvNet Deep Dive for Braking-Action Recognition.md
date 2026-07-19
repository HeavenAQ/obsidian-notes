---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-15T19:23:00
Status: Done
Last updated time: 2026-07-15T19:23:00
Last edited by: Heaven Chen
Category:
  - CV
  - Video Understanding
  - Action Recognition
  - 3D CNN
---
# I3D — Inflated 3D ConvNet Deep Dive for Braking-Action Recognition

> [!abstract]
> I3D converts a successful 2D image architecture into a video architecture by **inflating** spatial filters and pooling operations into time. The original model is a two-stream, Inflated Inception-v1 network pretrained on Kinetics. It learns appearance and motion jointly over clips, making it more suitable than a frame-only ResNet-18 when a braking point is defined by temporal change.

**Primary references:** [Carreira & Zisserman, “Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset,” CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Carreira_Quo_Vadis_Action_CVPR_2017_paper.html) · [paper PDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Carreira_Quo_Vadis_Action_CVPR_2017_paper.pdf) · [official DeepMind `kinetics-i3d` repository](https://github.com/google-deepmind/kinetics-i3d)

## 1. What problem I3D solves

A 2D CNN sees an image $x_t$ and extracts spatial evidence. Video recognition needs a model of

$$
P(y\mid x_1,x_2,\ldots,x_T),
$$

where ordering and motion can matter. “Hand/tool is near the work area” may be visible in one frame, while “the worker is braking” may require approach, contact, deceleration, or pause across several frames.

I3D's design asks:

> Can the architecture and weights of a strong 2D image classifier be expanded into time rather than learning a 3D network entirely from scratch?

## 2. The core operation: 3D convolution

For input $X\in\mathbb{R}^{C_{in}\times T\times H\times W}$ and kernel $W\in\mathbb{R}^{C_{out}\times C_{in}\times k_t\times k_h\times k_w}$,

$$
Y_{o,t,h,w}=b_o+
\sum_{c=1}^{C_{in}}
\sum_{\tau=1}^{k_t}
\sum_{i=1}^{k_h}
\sum_{j=1}^{k_w}
W_{o,c,\tau,i,j}
X_{c,t+\tau,h+i,w+j}.
$$

A 2D convolution detects a pattern independently in each frame. A 3D convolution can detect a pattern that evolves across adjacent frames.

For one dimension, the output length is

$$
L_{out}=\left\lfloor
\frac{L_{in}+2p-d(k-1)-1}{s}+1
\right\rfloor.
$$

Apply this separately to time, height, and width.

## 3. “Inflating” a 2D network into 3D

### 3.1 Filter inflation

Suppose a pretrained image kernel is

$$
W^{2D}\in\mathbb{R}^{C_{out}\times C_{in}\times k\times k}.
$$

Choose a temporal size $k_t$ and create

$$
W^{3D}_{:,:,\tau,:,:}=\frac{1}{k_t}W^{2D},
\qquad \tau=1,\ldots,k_t.
$$

In code:

```python
def inflate_conv2d_weight(weight_2d, time_dim):
    # [C_out, C_in, H, W] -> [C_out, C_in, T, H, W]
    return weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1) / time_dim
```

Why divide by $k_t$? If every frame of a video is the same image, then the initial 3D response matches the pretrained 2D response:

$$
\sum_{\tau=1}^{k_t}\frac{W^{2D}}{k_t}*x
=W^{2D}*x.
$$

The model begins from a useful image solution and can then learn temporal asymmetry during video training.

### 3.2 Pooling inflation

Spatial pools become spatiotemporal pools. Crucially, temporal stride is not applied everywhere: reducing $T$ too early destroys short events. Spatial and temporal downsampling schedules should be treated separately.

### 3.3 Architecture inflation

The original I3D inflates **Inception-v1**:

```text
video clip [C,T,H,W]
  → inflated Conv3D + spatiotemporal pooling stem
  → inflated Inception modules (parallel 1×1×1, 3D conv, pooling branches)
  → global average pooling over time and space
  → dropout
  → class logits
```

This matters because “I3D” describes an inflation method and the paper's specific Inflated Inception-v1 model. A 3D ResNet is related but is not the original I3D architecture.

## 4. Inflated Inception modules

An Inception module processes the same input through parallel paths:

1. $1\times1\times1$ convolution;
2. $1\times1\times1$ reduction then $3\times3\times3$ convolution;
3. another reduction then $3\times3\times3$ convolution;
4. 3D pooling then $1\times1\times1$ projection.

The outputs are concatenated along channels:

$$
Y=\operatorname{Concat}(Y_1,Y_2,Y_3,Y_4).
$$

The $1\times1\times1$ layers reduce channel cost before expensive 3D kernels and mix channels without directly mixing neighboring space or time.

## 5. Two-stream I3D

The original system trains two networks:

- **RGB stream:** appearance, objects, pose, environment, and scene context;
- **optical-flow stream:** explicit motion between frames.

If $z^{RGB}$ and $z^{Flow}$ are logits, a simple fusion is

$$
z=\alpha z^{RGB}+(1-\alpha)z^{Flow},
$$

followed by softmax. The paper averages stream predictions in its two-stream setup.

### Is optical flow necessary here?

Not necessarily. It adds preprocessing cost and can be noisy around blur, occlusion, reflective tools, or camera motion. Start with RGB I3D. Add flow only as an ablation if motion remains ambiguous and the camera is sufficiently stable.

## 6. Input and output semantics

PyTorch video models commonly expect

```text
[batch, channels, time, height, width]
```

The model normally emits **one label per clip**. This is clip classification, not automatically temporal localization.

If the operational goal is to detect the braking frame $t^*$, use sliding clips or a temporal prediction head:

$$
\hat{t}=\arg\max_t P(\text{braking at }t\mid X).
$$

For three mutually exclusive states, a practical label set is:

1. `no_braking`;
2. `braking_inside_area`;
3. `braking_outside_area`.

Alternatively, separate the tasks:

$$
P(\text{braking}) \quad\text{and}\quad
P(\text{inside area}\mid\text{braking}).
$$

The factorized version is often easier to debug because action-recognition error and spatial-relation error remain distinct.

## 7. Spatial geometry should not be hidden from the model

The poka-yoke system already has a worker/held-object detector and an area box. Use those signals explicitly.

For an object-box center $c_o=(x_o,y_o)$ and area box
$A=[x_1,y_1,x_2,y_2]$,

$$
\operatorname{inside}(c_o,A)
=\mathbb{1}[x_1\le x_o\le x_2]\,
 \mathbb{1}[y_1\le y_o\le y_2].
$$

For soft spatial evidence, use intersection over union:

$$
\operatorname{IoU}(O,A)=\frac{|O\cap A|}{|O\cup A|}.
$$

Normalized box features can include

$$
g_t=\left[
\frac{x_o-x_A}{w_A},
\frac{y_o-y_A}{h_A},
\log\frac{w_o}{w_A},
\log\frac{h_o}{h_A},
\operatorname{IoU}(O,A)
\right].
$$

Fuse a temporal summary of $g_t$ with I3D's visual embedding rather than forcing the CNN to rediscover exact rectangle geometry from pixels.

## 8. End-to-end system design

```text
raw video
  │
  ├─ object detector → worker / held object boxes
  ├─ configured or detected work-area bbox
  └─ tracker → temporally consistent identities
         │
         ├─ RGB ROI clip: union(worker, object, area) + context
         ├─ optional mask channels / rendered box overlays
         └─ box-geometry sequence g₁…gT
                │
                ├─ I3D visual encoder ─────────┐
                └─ geometry MLP/temporal head ─┤
                                               ▼
                       no braking / braking inside / braking outside
```

Use one consistent ROI definition at training and inference. Expand the union box by a fixed margin so that detector jitter does not repeatedly cut off the hand or work-area boundary.

## 9. Dataset construction for braking points

### 9.1 Define the annotation precisely

“Braking point” must have an operational definition, for example:

- first frame at which forward tool motion falls below a threshold;
- first sustained contact frame;
- manually annotated onset of the braking action;
- a temporal segment $[t_{start},t_{end}]$ rather than a single frame.

Annotators should see enough context before and after the event. Measure inter-annotator agreement or reconcile disagreements.

### 9.2 Clip sampling

For event time $t^*$, sample a clip

$$
X=[x_{t^*-a},\ldots,x_{t^*},\ldots,x_{t^*+b}].
$$

Choose temporal stride so the clip covers the complete motion. Report clip duration in **seconds**, not only frames, because frame rates may differ.

Hard negatives are essential:

- hand/tool passes through the area without braking;
- braking occurs just outside the boundary;
- worker pauses for an unrelated reason;
- tool is inside the area but held by the wrong worker;
- detector or tracker briefly loses the object.

### 9.3 Leakage-safe split

Split by worker, session, camera, or production run. Neighboring clips from one video are strongly correlated and must not be split across train and validation/test.

## 10. Minimal clip dataset skeleton

```python
from pathlib import Path
import torch
from torch.utils.data import Dataset


class BrakeClipDataset(Dataset):
    def __init__(self, samples, decode_clip, transform=None):
        """
        samples: records with video path, start/end frames, ROI data, label
        decode_clip(record) -> float tensor [C, T, H, W] in [0, 1]
        """
        self.samples = samples
        self.decode_clip = decode_clip
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index]
        clip = self.decode_clip(record)

        if self.transform is not None:
            clip = self.transform(clip)

        return {
            "video": clip,
            "label": torch.tensor(record["label"], dtype=torch.long),
            "video_id": record["video_id"],
        }
```

Apply the same random spatial transform to every frame. Independent crops or flips per frame create artificial motion.

## 11. A compact 3D classifier for pipeline testing

This is **not the full original I3D**; it is a small sanity-check model that verifies tensor shapes, labels, and the training loop before integrating a pretrained implementation.

```python
import torch
from torch import nn


class Conv3DBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch, stride=(1, 1, 1)):
        super().__init__(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )


class Tiny3DClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            Conv3DBlock(3, 32, stride=(1, 2, 2)),
            Conv3DBlock(32, 64, stride=(2, 2, 2)),
            Conv3DBlock(64, 128, stride=(2, 2, 2)),
            Conv3DBlock(128, 256, stride=(2, 2, 2)),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):                 # [B, 3, T, H, W]
        return self.head(self.features(x).flatten(1))
```

For experiments, use a pretrained I3D implementation or a maintained video-model library rather than treating this toy network as I3D.

## 12. Transfer learning

Video datasets are expensive; initialization matters. A practical order is:

1. initialize from Kinetics-pretrained weights;
2. replace the classification head;
3. train the head with the backbone frozen;
4. unfreeze later blocks;
5. optionally fine-tune the full network with a lower learning rate.

```python
# Generic pattern; attribute names depend on the chosen I3D implementation.
for parameter in model.parameters():
    parameter.requires_grad = False

model.classifier = torch.nn.Linear(model.feature_dim, 3)

optimizer = torch.optim.AdamW(
    model.classifier.parameters(), lr=1e-3, weight_decay=1e-4
)
```

Use the preprocessing required by the exact pretrained weights. Channel order, normalization, spatial size, clip length, and temporal sampling are part of the model contract.

## 13. Compute and memory

A dense 3D convolution costs approximately

$$
\operatorname{MACs}
\approx T_{out}H_{out}W_{out}C_{out}
(k_tk_hk_wC_{in}).
$$

Compared with a 2D convolution, retaining $T$ output steps and adding $k_t$ kernel positions makes 3D CNNs expensive. Memory also grows with temporal activations.

Practical controls:

- crop to the interaction ROI;
- reduce spatial resolution before reducing temporal coverage;
- use temporal stride to cover longer real time with fixed $T$;
- use mixed precision;
- accumulate gradients when batches are small;
- freeze or replace BatchNorm when batch statistics are unstable.

## 14. Losses for imbalance

Weighted cross-entropy:

$$
\mathcal{L}_{CE}=-\sum_{c=1}^{C}w_c y_c\log p_c.
$$

Focal loss emphasizes difficult examples:

$$
\mathcal{L}_{focal}=-\alpha_t(1-p_t)^\gamma\log p_t.
$$

Do not use focal loss automatically. First inspect whether the problem is class frequency, label ambiguity, poor hard negatives, or detector failure.

## 15. Evaluation for braking detection

### Clip-level classification

Report:

- per-class precision, recall, and F1;
- macro-F1 for imbalance;
- confusion matrix;
- PR-AUC for rare braking classes;
- calibration or expected calibration error if probabilities trigger safety logic.

### Event-level localization

A predicted event at $\hat{t}$ is correct within tolerance $\delta$ if

$$
|\hat{t}-t^*|\le\delta.
$$

Report event precision/recall/F1 at one or more tolerances in seconds. For temporal segments $P$ and $G$,

$$
\operatorname{tIoU}(P,G)=\frac{|P\cap G|}{|P\cup G|}.
$$

Also measure end-to-end latency and false triggers per hour. A model can have good clip accuracy yet be unusable if overlapping windows create repeated alarms.

## 16. Failure modes

| Failure | Cause | Mitigation |
| --- | --- | --- |
| Background shortcut | Area, worker, or shift correlates with label | worker/session split; ROI/background ablation |
| Boundary ambiguity | Box center and actual contact point differ | keypoint/contact annotation; soft geometry features |
| Detector jitter | ROI moves artificially | tracking, smoothing, enlarged crop, jitter augmentation |
| Missed short event | temporal stride/downsampling too large | denser sampling; preserve early temporal resolution |
| False motion | camera shake or moving crop | stabilize camera/ROI; provide box trajectory explicitly |
| BatchNorm instability | small 3D batches | freeze BN, SyncBN, or GroupNorm |
| Offline-only result | centered clips use future frames | causal windows for real-time deployment |

## 17. Fair experiment matrix

Keep detector outputs, ROI, splits, augmentation, labels, and metrics constant.

| Model | Temporal mechanism | Purpose |
| --- | --- | --- |
| Geometry rules | box center / IoU over time | interpretable floor |
| ResNet-18, center frame | none | appearance/spatial baseline |
| ResNet-18, averaged frames | prediction pooling | cheap multi-frame baseline |
| ResNet-18 + GRU/TCN | learned feature aggregation | separates 2D features from temporal modeling |
| `r3d_18` | residual 3D convolutions | compact native 3D baseline |
| RGB I3D | inflated Inception 3D convolutions | main video model |
| RGB + flow I3D | two-stream fusion | motion-focused ablation |

Use matched pretrained versus pretrained comparisons where possible; otherwise architecture and pretraining effects are confounded.

## 18. Recommended implementation sequence

1. Verify labels and leakage using geometry-only rules.
2. Train single-frame [[04 Reference/Document Hub/ResNet Architecture|ResNet-18]].
3. Add frame-feature temporal pooling or a small GRU/TCN.
4. Train `r3d_18` as the easiest maintained 3D baseline.
5. Integrate pretrained RGB I3D.
6. Add box-geometry fusion.
7. Try optical flow only if RGB motion remains insufficient.
8. Convert centered offline clips to causal sliding windows before deployment tests.

## 19. Bottom line

I3D is appropriate when braking is a temporal pattern rather than a static appearance. Its key contribution is not merely `Conv3d`: it inflates a proven 2D network and its weights into time, then benefits from large-scale video pretraining. In this poka-yoke system, the strongest design is likely hybrid—object detection and box geometry determine **where**, while I3D determines **what motion occurred**. Evaluate the full pipeline with leakage-safe splits and event-level metrics, not only random-clip accuracy.
