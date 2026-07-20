---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-19T14:36:15
Status: Done
Last updated time: 2026-07-19T14:55:37
Last edited by: Heaven Chen
Category:
  - CV
  - Object-Detection
  - Deep-Learning
  - Theory
---
# Fast & Faster R-CNN Deep Dive — RoI Heads, RPNs, Math, and PyTorch

> [!abstract]
> **Fast R-CNN** computes a feature map once, then classifies and refines externally supplied proposals. **Faster R-CNN** retains the RoI head but replaces the external proposal algorithm with a learned **Region Proposal Network (RPN)** that shares the same backbone. This note derives both models, their losses and box transformations, and implements their essential PyTorch components.

**Primary references:** [Girshick, “Fast R-CNN,” ICCV 2015](https://openaccess.thecvf.com/content_iccv_2015/html/Girshick_Fast_R-CNN_ICCV_2015_paper.html) · [Ren et al., “Faster R-CNN,” NeurIPS 2015](https://proceedings.neurips.cc/paper_files/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html) · [official Fast R-CNN code](https://github.com/rbgirshick/fast-rcnn) · [official Faster R-CNN code](https://github.com/ShaoqingRen/faster_rcnn) · [TorchVision Faster R-CNN source](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py)

Related vault notes: [[04 Reference/Document Hub/Object Detection 101|Object Detection 101]] · [[04 Reference/Document Hub/Computer Vision Loss|Computer Vision Loss]] · [[04 Reference/Document Hub/ResNet Architecture|ResNet Architecture]] · [[04 Reference/Document Hub/Hand Object Detector Setup|Hand Object Detector Setup]]
-
## 1. The naming trap: R-CNN, Fast R-CNN, and Faster R-CNN

| Model | Shared image convolution? | Proposal source | Detection head |
| --- | --- | --- | --- |
| R-CNN (2014) | No; CNN runs on every warped proposal | Selective Search | per-class SVM + separate box regressors |
| SPPnet (2014) | Yes | external proposals | spatial-pyramid pooling + heads |
| **Fast R-CNN (2015)** | **Yes** | **external proposals** | **RoI Pool + joint softmax/box loss** |
| Faster R-CNN (2015) | Yes | learned Region Proposal Network | Fast R-CNN-style RoI head |
| Mask R-CNN (2017) | Yes | learned RPN | RoIAlign + class/box/mask heads |

> [!warning]
> “Fast R-CNN” and “Faster R-CNN” are different architectures. A modern `torchvision.models.detection.fasterrcnn_*` model contains a proposal network plus a Fast R-CNN-style second stage; it is not a standalone reproduction of the original Fast R-CNN system.

## 2. What problem Fast R-CNN solved

Original R-CNN generated roughly thousands of candidate boxes, cropped and warped each region, and ran the full CNN separately on every crop. This caused:

- duplicated convolution over heavily overlapping regions;
- a large feature cache on disk;
- a multi-stage training pipeline: CNN fine-tuning, SVM fitting, then box-regressor fitting;
- slow inference.

Fast R-CNN reverses the expensive order:

```text
R-CNN:
proposal → crop/warp → CNN, repeated once per proposal

Fast R-CNN:
image → CNN once → shared feature map → pool once per proposal
```

The ICCV paper reported that VGG16 Fast R-CNN trained 9× faster and tested 213× faster than R-CNN, while also improving PASCAL VOC accuracy. These are historical comparisons on the paper's hardware and software, not modern throughput claims.

## 3. End-to-end architecture

Let image $I_i$ have externally generated proposals

$$
\mathcal R_i=\{r_{i1},r_{i2},\ldots,r_{iM_i}\},
\qquad r=(x_1,y_1,x_2,y_2).
$$

The dataflow is:

```text
image [B,3,H,W]
  │
  ▼
CNN backbone, once per image
  │  F = φ(I), typically [B,C,H/s,W/s]
  │
  ├── external proposals in image coordinates
  ▼
RoI Pool: each variable-size proposal → fixed C×PH×PW tensor
  │
  ▼
shared fully connected layers
  │
  ├── classification head → K+1 logits (K object classes + background)
  └── box head            → 4K offsets (class-specific box regression)
```

Mathematically,

$$
F_i=\phi(I_i;\theta_{conv}),
$$

$$
h_{ij}=g\!\left(\operatorname{RoIPool}(F_i,r_{ij});\theta_{fc}\right),
$$

$$
z_{ij}=W_{cls}h_{ij}+b_{cls},
\qquad p_{ij}=\operatorname{softmax}(z_{ij}),
$$

$$
t_{ij}=W_{box}h_{ij}+b_{box}\in\mathbb{R}^{4K}.
$$

Many proposals reuse the same $F_i$, so their gradients accumulate into the shared image feature map during backpropagation.

## 4. RoI Pooling in depth

### 4.1 Why a pooling operator is necessary

Proposals have different shapes, but fully connected layers require a fixed input dimension. RoI Pool maps every proposal to, for example, $7\times7$ bins, independent of its original width and height.

Suppose the backbone has total stride $s$. A proposal in image coordinates is mapped to feature coordinates by

$$
(x_1',y_1',x_2',y_2')
=\frac{1}{s}(x_1,y_1,x_2,y_2).
$$

In an implementation this scale is called

$$
\texttt{spatial\_scale}=\frac1s.
$$

For pooled height $P_H$ and width $P_W$, the approximate bin dimensions are

$$
\Delta h=\frac{y_2'-y_1'}{P_H},
\qquad
\Delta w=\frac{x_2'-x_1'}{P_W}.
$$

RoI Pool quantizes the RoI and bin boundaries to discrete feature indices. For channel $c$ and output bin $(a,b)$,

$$
Y_{c,a,b}=max_{(u,v)\in\mathcal B_{a,b}}F_{c,u,v},
$$

where $\mathcal B_{a,b}$ is the quantized subset of the proposal assigned to that bin.

### 4.2 Concrete shape example

```text
image:                    [1, 3, 800, 800]
backbone stride:          16
feature map:              [1, 512, 50, 50]
number of proposals:      2,000
RoI Pool output:          [2,000, 512, 7, 7]
flattened head input:     [2,000, 25,088]
```

The backbone runs once, but the per-RoI head still scales with the number of proposals.

### 4.3 RoI Pool versus RoIAlign

RoI Pool's rounding creates coordinate misalignment. A small move in the image may cause an abrupt change in the quantized feature region. RoIAlign, introduced later with Mask R-CNN, keeps fractional coordinates and uses bilinear interpolation:

$$
F(x,y)=\sum_{m,n}w_{mn}(x,y)F(x_m,y_n),
$$

where the four weights depend on distance from the continuous sample point.

| Property | RoI Pool | RoIAlign |
| --- | --- | --- |
| Boundary handling | quantized/rounded | continuous coordinates |
| Aggregation | max pooling | sampled bilinear values, normally averaged |
| Alignment | approximate | substantially better |
| Original Fast R-CNN | **yes** | no |
| Recommended for a modern implementation | only for faithful reproduction | usually yes |

[[04 Reference/Document Hub/Object Detection 101|Object Detection 101]] contains a numerical bilinear-interpolation example.

## 5. Classification head

For $K$ foreground classes, the classifier emits $K+1$ logits; class $0$ is background. For RoI feature $h$,

$$
p_k=\frac{\exp(z_k)}{\sum_{j=0}^{K}\exp(z_j)},
\qquad k\in\{0,1,\ldots,K\}.
$$

If the target class is $u$, the classification loss is

$$
L_{cls}(p,u)=-\log p_u.
$$

Background is a real supervised class in Fast R-CNN. This differs from detectors that use a separate objectness sigmoid or DETR's matched “no-object” formulation.

## 6. Bounding-box regression mathematics

### 6.1 Encode regression targets

Represent proposal $P$ by center $(P_x,P_y)$ and size $(P_w,P_h)$, and its matched ground-truth box $G$ similarly. The target transformation is

$$
v_x=\frac{G_x-P_x}{P_w},
\qquad
v_y=\frac{G_y-P_y}{P_h},
$$

$$
v_w=\log\frac{G_w}{P_w},
\qquad
v_h=\log\frac{G_h}{P_h}.
$$

Translations are normalized by proposal size, while width and height use log-scale ratios. This makes the targets closer to scale invariant than raw pixel differences.

### 6.2 Class-specific prediction

The head emits

$$
t=(t^1,t^2,\ldots,t^K),
\qquad t^k=(t_x^k,t_y^k,t_w^k,t_h^k).
$$

Only the four values corresponding to the RoI's foreground class are trained. The background class has no meaningful target box.

### 6.3 Decode predicted boxes

Given predicted deltas $t^k$ for class $k$,

$$
\hat G_x=P_x+t_x^kP_w,
\qquad
\hat G_y=P_y+t_y^kP_h,
$$

$$
\hat G_w=P_w\exp(t_w^k),
\qquad
\hat G_h=P_h\exp(t_h^k).
$$

Convert the predicted center-size representation back to corner coordinates before clipping it to the image.

## 7. The multi-task loss

Fast R-CNN trains classification and localization jointly:

$$
L(p,u,t^u,v)
=L_{cls}(p,u)
+\lambda [u\ge 1]L_{loc}(t^u,v).
$$

Here:

- $u=0$ means background and $u\ge1$ means a foreground class;
- $[u\ge1]$ is an indicator, so background RoIs receive no box loss;
- $v$ is the encoded target box;
- $t^u$ is the prediction for target class $u$;
- the original paper uses $\lambda=1$.

Localization sums Smooth L1 over four coordinates:

$$
L_{loc}(t^u,v)=\sum_{q\in\{x,y,w,h\}}
\operatorname{smooth}_{L1}(t_q^u-v_q),
$$

$$
\operatorname{smooth}_{L1}(d)=
\begin{cases}
\frac12d^2,&|d|<1,\\
|d|-\frac12,&\text{otherwise}.
\end{cases}
$$

Smooth L1 is quadratic near zero for stable refinement and linear for large errors so outliers do not dominate as strongly as with L2. See also [[04 Reference/Document Hub/Computer Vision Loss|Computer Vision Loss]].

## 8. Proposal matching and mini-batch sampling

For each proposal, compute its maximum IoU with the ground-truth boxes:

$$
\operatorname{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

The original training recipe uses a hierarchical mini-batch:

- sample $N=2$ images;
- sample $R=128$ RoIs total from those images;
- approximately 25% are foreground with IoU $\ge0.5$;
- background RoIs have maximum IoU in $[0.1,0.5)$;
- apply horizontal flipping as augmentation.

Sampling many RoIs from only two images is computationally efficient because convolution is shared. It also means RoIs within a batch are correlated, but the paper found this practical.

> [!important]
> Proposal recall sets an upper bound on the detector. If Selective Search never proposes a box near an object, the Fast R-CNN head cannot recover that object regardless of classifier quality.

## 9. Training and inference algorithms

### 9.1 Training

```text
for each mini-batch:
  choose images
  load/generate external proposals
  match proposals to ground truth using IoU
  sample foreground/background RoIs
  compute one shared feature map per image
  RoI-pool each sampled proposal
  predict K+1 class logits and 4K box deltas
  compute classification loss for every RoI
  compute box loss for foreground RoIs only
  backpropagate jointly through heads, RoI Pool, and backbone
```

### 9.2 Inference

For each image:

1. obtain external region proposals;
2. compute the convolutional feature map once;
3. run all proposals through RoI Pool and the two heads;
4. decode a refined box for each proposal/class;
5. discard low-confidence predictions;
6. apply per-class non-maximum suppression (NMS).

Given boxes $b_i$ sorted by class score, greedy NMS keeps the highest-scoring box and suppresses any lower-scoring $b_j$ for which

$$
\operatorname{IoU}(b_i,b_j)>\tau_{NMS}.
$$

NMS removes duplicates, but a threshold that is too low can delete distinct nearby objects; one that is too high retains duplicates.

## 10. A compact Fast R-CNN-style PyTorch model

The following model deliberately accepts proposals as an input. It therefore represents **Fast R-CNN**, not Faster R-CNN. A ResNet layer-3 feature map gives output stride 16.

```python
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.ops import roi_pool


class FastRCNN(nn.Module):
    def __init__(self, num_classes: int, pool_size: int = 7):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Stem through layer3: [B, 256, H/16, W/16].
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3,
        )
        self.spatial_scale = 1.0 / 16.0
        self.pool_size = pool_size

        input_dim = 256 * pool_size * pool_size
        self.box_head = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(1024, num_classes + 1)  # + background
        self.box_regressor = nn.Linear(1024, num_classes * 4)
        self.num_classes = num_classes

    def forward(self, images, proposals):
        """
        images:    [B, 3, H, W]
        proposals: list of B tensors [Ri, 4] in image-space xyxy format
        """
        features = self.backbone(images)
        pooled = roi_pool(
            features,
            proposals,
            output_size=(self.pool_size, self.pool_size),
            spatial_scale=self.spatial_scale,
        )
        h = self.box_head(pooled.flatten(1))
        class_logits = self.classifier(h)              # [R, K+1]
        box_deltas = self.box_regressor(h)              # [R, 4K]
        return class_logits, box_deltas
```

All images in the tensor must currently have the same padded size. Proposals remain in the coordinate system used before padding/resizing, so their transformations must exactly match the image preprocessing.

### 10.1 Multi-task loss in PyTorch

```python
import torch.nn.functional as F


def fast_rcnn_loss(class_logits, box_deltas, labels, regression_targets):
    """
    labels:             [R], 0=background and 1..K=foreground classes
    regression_targets: [R,4], encoded relative to each input proposal
    """
    classification_loss = F.cross_entropy(class_logits, labels)

    num_classes = class_logits.shape[1] - 1
    box_deltas = box_deltas.reshape(-1, num_classes, 4)
    foreground = torch.where(labels > 0)[0]

    if foreground.numel() == 0:
        # Preserve graph/device/dtype while returning zero localization loss.
        box_loss = box_deltas.sum() * 0.0
    else:
        class_index = labels[foreground] - 1
        selected = box_deltas[foreground, class_index]
        box_loss = F.smooth_l1_loss(
            selected,
            regression_targets[foreground],
            beta=1.0,
            reduction="sum",
        ) / labels.numel()

    return classification_loss, box_loss
```

The normalization convention must be recorded. The code divides summed foreground box loss by all sampled RoIs, matching common detector practice; dividing only by foreground count changes its relative weight.

### 10.2 Box encoding and decoding

```python
import torch


def xyxy_to_cxcywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    return torch.stack((x1 + 0.5 * w, y1 + 0.5 * h, w, h), dim=-1)


def encode_boxes(proposals, targets):
    px, py, pw, ph = xyxy_to_cxcywh(proposals).unbind(-1)
    gx, gy, gw, gh = xyxy_to_cxcywh(targets).unbind(-1)
    return torch.stack((
        (gx - px) / pw,
        (gy - py) / ph,
        torch.log(gw / pw),
        torch.log(gh / ph),
    ), dim=-1)


def decode_boxes(proposals, deltas, max_log_scale=4.135):
    px, py, pw, ph = xyxy_to_cxcywh(proposals).unbind(-1)
    dx, dy, dw, dh = deltas.unbind(-1)
    dw = dw.clamp(max=max_log_scale)
    dh = dh.clamp(max=max_log_scale)

    gx = px + dx * pw
    gy = py + dy * ph
    gw = pw * torch.exp(dw)
    gh = ph * torch.exp(dh)
    return torch.stack((
        gx - 0.5 * gw, gy - 0.5 * gh,
        gx + 0.5 * gw, gy + 0.5 * gh,
    ), dim=-1)
```

Clamping predicted log scales avoids numerical explosion from $\exp(t_w)$ and $\exp(t_h)$ early in training.

## 11. Modernizing the model without confusing the name

A practical modernized Fast R-CNN-style experiment can use:

- a pretrained ResNet or ConvNeXt backbone;
- feature pyramids for multi-scale objects;
- `roi_align` instead of `roi_pool`;
- class-agnostic box regression with only four outputs;
- improved samplers and box losses;
- mixed precision and batched NMS.

However, each change moves away from strict reproduction. State the configuration explicitly:

```text
external proposals + shared backbone + RoIAlign + classification/box head
```

The defining distinction remains **external proposals**. Once a learned RPN is added, the complete detector is Faster R-CNN.

## 12. Strengths, limitations, and failure modes

### Strengths

- shares expensive convolution across all proposals;
- trains classification and localization jointly;
- works with any proposal generator;
- provides an interpretable modular decomposition;
- established the second-stage head still used in many detectors.

### Limitations

- Selective Search is slow and not learned end-to-end;
- proposal recall limits final recall;
- thousands of RoIs make the fully connected head and NMS costly;
- RoI Pool quantization damages precise localization;
- class-specific regression scales the output with $K$;
- small objects may disappear in a single low-resolution feature map.

### Debugging checklist

| Symptom | Likely cause |
| --- | --- |
| Every RoI predicts background | incorrect foreground matching, label offset, or severe imbalance |
| Boxes shift or shrink systematically | wrong `spatial_scale`, resize transform, or box convention |
| Box loss explodes | degenerate proposals, unnormalized targets, or unbounded log-scale deltas |
| Good classification but poor localization | RoI Pool quantization, weak proposals, or box-target bug |
| Duplicate detections | NMS missing/too permissive |
| Nearby objects disappear | NMS too aggressive |
| Training accuracy high, validation AP low | image/session leakage or proposal distribution mismatch |

## 13. Evaluation

Classification accuracy over sampled RoIs is not the detector metric. Evaluate final detections after box decoding and NMS using precision-recall and Average Precision.

For class $k$, a prediction is a true positive only if it matches an unused ground-truth instance at the required IoU threshold. From ranked detections,

$$
\operatorname{Precision}=\frac{TP}{TP+FP},
\qquad
\operatorname{Recall}=\frac{TP}{TP+FN}.
$$

AP summarizes the precision-recall curve; mAP averages AP across classes and, depending on the benchmark, across IoU thresholds. Also measure:

- proposal recall before the Fast R-CNN head;
- per-class AP and confusion;
- AP by object size;
- latency split into proposal generation, backbone, RoI head, and NMS;
- performance with ground-truth versus real proposals.

## 14. Relevance to the hand-object and poka-yoke pipeline

Fast R-CNN cleanly separates **candidate generation** from **candidate classification/refinement**. For the workflow in [[04 Reference/Document Hub/Hand Object Detector Setup|Hand Object Detector Setup]], it is useful conceptually and as an ablation:

```text
frame
  → external worker/hand/object candidate boxes
  → shared CNN feature map
  → RoI head: worker / hand / tool / work-area / background
  → refined boxes + NMS
  → tracking and worker-object association
  → clip crop
  → [[04 Reference/Document Hub/ResNet Architecture|3D ResNet]] or
     [[04 Reference/Document Hub/I3D — Inflated 3D ConvNet Deep Dive for Braking-Action Recognition|I3D]]
  → braking-inside / braking-outside decision
```

For a production detector, Faster R-CNN or a modern detector is normally preferable because proposal generation is learned. Fast R-CNN remains valuable for understanding RoI heads and for experiments where proposals come from a trusted external source or ground-truth boxes perturbed with controlled noise.

## 15. Recommended ablations

| Ablation | Question answered |
| --- | --- |
| Selective Search vs detector proposals vs noisy ground truth | How much performance is proposal-limited? |
| RoI Pool vs RoIAlign | How much does quantization hurt localization? |
| class-specific vs class-agnostic box regression | Is per-class geometry necessary? |
| stride 16 vs multi-scale features | Are small hands/tools being lost? |
| frozen vs fine-tuned backbone | Is the training set sufficient for feature adaptation? |
| 128 vs fewer/more sampled RoIs | What is the accuracy-memory trade-off? |
| NMS threshold sweep | Are errors duplicates or over-suppression? |
| image-random vs worker/session split | Is performance inflated by leakage? |

## 16. Fast R-CNN summary

Fast R-CNN's decisive idea is to compute convolution **once per image**, transform each external proposal into a fixed-size RoI feature, and jointly optimize classification and box refinement. Its mathematics—IoU matching, softmax classification, encoded box deltas, gated Smooth L1 localization, and NMS—became the foundation of modern two-stage detectors. Its key weakness is equally important: because proposals are external, the system is neither fully learned nor end-to-end. Faster R-CNN solves that remaining bottleneck by adding a Region Proposal Network while retaining a Fast R-CNN-style second stage.

---

# Part II — Faster R-CNN

## 17. The one decisive change

Fast R-CNN made the detector head efficient, but Selective Search remained a slow, fixed, CPU-side bottleneck. Faster R-CNN replaces it with a **Region Proposal Network**:

$$
\text{external proposals}
\quad\longrightarrow\quad
\text{learned, class-agnostic RPN proposals}.
$$

The backbone feature map is shared:

$$
F=\phi(I;\theta_{backbone}).
$$

The RPN predicts *where an object might be*, while the RoI head predicts *which class it is* and refines its box:

$$
\mathcal P=\operatorname{RPN}(F),
$$

$$
(\hat c,\hat b)=\operatorname{RoIHead}(F,\mathcal P).
$$

This turns proposal generation into a trainable, GPU-friendly component. “Faster” refers to removing the external proposal bottleneck—not to Faster R-CNN being faster than every one-stage detector.

## 18. Complete Faster R-CNN dataflow

```text
images: list of [3,H_i,W_i] tensors
  │
  ▼ resize / normalize / batch with padding
shared CNN backbone, often ResNet + FPN
  │
  ├────────────────────────────────────────────────────────┐
  │                                                        │
  ▼                                                        │
Region Proposal Network                                    │
  3×3 sliding conv                                         │
  ├─ objectness logits for anchors                         │
  └─ 4 box deltas per anchor                               │
  decode → clip → small-box filter → top-k → NMS            │
  │                                                        │
  ▼ proposals                                              │ shared features
Multi-scale RoIAlign ◄─────────────────────────────────────┘
  │
  ▼ two-layer box head
  ├─ K+1 class logits
  └─ class-specific or class-agnostic box deltas
  │
  ▼ decode → score filter → per-class NMS → detections
```

A modern implementation usually consists of five modules:

1. **transform** — normalize, resize, batch, and later restore original coordinates;
2. **backbone/FPN** — extract one or more feature maps;
3. **anchor generator** — place reference boxes over the feature maps;
4. **RPN** — score and refine anchors into proposals;
5. **RoI head** — classify/refine the proposals using the Fast R-CNN machinery from Part I.

## 19. Anchors: reference boxes tiled over the image

### 19.1 Base anchors

An anchor is not a prediction; it is a reference box from which the RPN predicts an offset. Let nominal scale be $s$ and aspect ratio be

$$
r=\frac{w}{h}.
$$

Keeping area $wh=s^2$ gives

$$
w=s\sqrt r,
\qquad
h=\frac{s}{\sqrt r}.
$$

At a feature location $(i,j)$ with feature stride $d$, the anchor center is approximately

$$
(x_a,y_a)=\left((j+\delta)d,(i+\delta)d\right),
$$

where $\delta$ is the implementation's grid offset convention.

The original paper used three scales and three aspect ratios, giving $k=9$ anchors per sliding-window location. The nominal areas corresponded to $128^2$, $256^2$, and $512^2$ pixels, with ratios $1{:}1$, $1{:}2$, and $2{:}1$.

### 19.2 Number of anchors

For a feature map $H_f\times W_f$ with $k$ anchors at each position,

$$
N_A=H_fW_fk.
$$

For $H_f=50$, $W_f=80$, and $k=9$,

$$
N_A=50\cdot80\cdot9=36{,}000.
$$

Most of these candidates are discarded before the RoI head. Anchors can extend outside the image during tiling; training and inference code must define how boundary anchors are handled.

### 19.3 Why anchors are useful

The same convolutional weights operate at every location. Each anchor index has a consistent geometric meaning, so the RPN is translation equivariant up to boundary, stride, and quantization effects. Multiple scales and ratios let one feature location hypothesize several object shapes.

> [!note] Anchor-free is a different design family
> Modern detectors can predict points, distances, or boxes without predefined anchors. Faster R-CNN's original RPN is explicitly anchor-based, and TorchVision's standard implementation still uses an `AnchorGenerator`.

## 20. The Region Proposal Network

For each spatial position, the RPN applies a small sliding network—originally a $3\times3$ convolution—followed by two sibling $1\times1$ heads.

Let

$$
U=\operatorname{ReLU}(W_{3\times3}*F+b).
$$

For $k$ anchors per location:

$$
o=W_{obj}*_{1\times1}U+b_{obj},
$$

$$
d=W_{reg}*_{1\times1}U+b_{reg}.
$$

The head produces:

- one or two objectness logits per anchor, depending on whether the implementation uses sigmoid or two-class softmax;
- four box-regression deltas per anchor.

With a binary-logit implementation,

$$
o\in\mathbb{R}^{B\times k\times H_f\times W_f},
\qquad
d\in\mathbb{R}^{B\times4k\times H_f\times W_f}.
$$

Objectness is **class agnostic**: it asks “object or background?”, not “person or tool?”. Fine-grained class prediction belongs to the second-stage RoI head.

## 21. RPN target assignment

For every anchor $a_i$, calculate IoU against all ground-truth boxes. In the original Faster R-CNN rule, an anchor is positive if:

1. it has the highest IoU with a ground-truth box, even if that IoU is below the regular threshold; **or**
2. its IoU with any ground-truth box is above $0.7$.

It is negative if its IoU with every ground-truth box is below $0.3$. Anchors between the thresholds are ignored.

The “highest-IoU” rule ensures that every ground-truth object has at least one positive anchor when possible. Implementations must also handle ties: several anchors may share the highest IoU.

The original paper samples 256 anchors per image for an RPN mini-batch, with at most a 1:1 positive-to-negative ratio. If fewer than 128 positives exist, negatives fill the batch.

### 21.1 Edge cases

- **No ground-truth objects:** sampled valid anchors should be negatives; no regression loss is applied.
- **Crowd/ignore regions:** anchors overlapping ignored regions should normally be excluded rather than treated as negatives.
- **Tiny objects:** there may be no well-aligned anchor unless small scales or high-resolution FPN levels exist.
- **Objects near boundaries:** filtering outside anchors too aggressively can lower recall.

## 22. RPN loss derivation

Let $p_i$ be predicted objectness probability and $p_i^*\in\{0,1\}$ the anchor label. Let $t_i$ be four predicted offsets and $t_i^*$ the encoded offsets to the matched ground truth. The paper's RPN loss is

$$
L_{RPN}
=\frac1{N_{cls}}\sum_i L_{cls}(p_i,p_i^*)
+\lambda\frac1{N_{reg}}\sum_i p_i^*L_{reg}(t_i,t_i^*).
$$

The indicator $p_i^*$ gates regression so negative anchors never regress a box. Ignored anchors contribute to neither term.

For binary logits $z_i$, a modern classification term is

$$
L_{cls}(z_i,p_i^*)
=-p_i^*\log\sigma(z_i)
-(1-p_i^*)\log(1-\sigma(z_i)).
$$

The localization term is Smooth L1 over $(x,y,w,h)$, using the same proposal-relative parameterization derived in Part I.

In the original paper, $N_{cls}$ is the sampled mini-batch size, $N_{reg}$ is related to the number of anchor locations, and $\lambda=10$ balances the two terms. Modern libraries use their own reduction and normalization conventions; compare effective gradient scales rather than copying $\lambda$ blindly.

The complete detector loss is conceptually

$$
L_{total}
=L_{rpn\_objectness}
+L_{rpn\_box}
+L_{roi\_class}
+L_{roi\_box}.
$$

## 23. From anchors to proposals

At inference—and during training to feed the RoI head—the RPN transforms dense anchors into a small proposal set:

1. flatten objectness logits and box deltas from all feature levels;
2. decode each anchor using its predicted deltas;
3. clip boxes to image boundaries;
4. remove boxes smaller than a configured size;
5. select the highest-scoring candidates before NMS;
6. apply NMS, usually independently per feature level;
7. keep the top post-NMS proposals per image.

If decoded proposal $b_i$ has objectness $s_i$, proposal NMS solves a heuristic redundancy problem:

```text
sort proposals by s_i descending
while candidates remain:
  keep highest-score proposal
  suppress remaining proposals with IoU > threshold
stop after post_nms_top_n proposals
```

Pre-NMS top-k improves speed but can destroy recall if set too low. Post-NMS top-k controls the cost of RoIAlign and the second-stage head.

> [!warning] There are two NMS stages
> Faster R-CNN normally uses **RPN NMS** to deduplicate class-agnostic proposals and **final per-class NMS** to deduplicate detections. Their thresholds solve different problems and should be tuned separately.

## 24. How the RPN and RoI head share computation

Both heads consume the same backbone features:

$$
\frac{\partial L_{total}}{\partial\theta_{backbone}}
=
\frac{\partial L_{RPN}}{\partial\theta_{backbone}}
+
\frac{\partial L_{RoI}}{\partial\theta_{backbone}}.
$$

The original paper described a four-step alternating procedure:

1. train the RPN initialized from an ImageNet-pretrained network;
2. train a separate Fast R-CNN detector using RPN proposals;
3. initialize the RPN from the detector, fix shared convolutional layers, and fine-tune RPN-specific layers;
4. keep shared convolution fixed and fine-tune the Fast R-CNN-specific layers.

It also investigated approximate joint training. Modern autograd implementations normally train the combined system jointly, treating proposal selection, sorting, and NMS as non-differentiable routing. The RoI loss does not need to differentiate through proposal coordinates to provide useful gradients to the shared feature map.

## 25. The second stage is still Fast R-CNN

After proposal generation, Faster R-CNN reuses Part I:

$$
q_j=\operatorname{RoIAlign}(F,b_j),
$$

$$
h_j=g(q_j),
$$

$$
p_j=\operatorname{softmax}(W_{cls}h_j+b_{cls}),
$$

$$
t_j=W_{box}h_j+b_{box}.
$$

Training proposals are matched to ground truth again using the **RoI-head thresholds**, which need not equal RPN thresholds. The stages solve different tasks:

| Stage | Input unit | Classification target | Regression target |
| --- | --- | --- | --- |
| RPN | anchor | object / background | anchor → GT box |
| RoI head | proposal | one of $K$ classes / background | proposal → GT box |

This second matching is essential. An RPN proposal can count as a useful class-agnostic positive yet still need class supervision and further localization refinement.

## 26. Feature Pyramid Networks in modern Faster R-CNN

Original Faster R-CNN used one deep feature map. Modern versions frequently add an FPN producing maps such as $P_2,P_3,P_4,P_5$, often with strides 4, 8, 16, and 32.

Each level specializes in a size range:

```text
small boxes  → high-resolution pyramid level
large boxes  → low-resolution pyramid level
```

A common level-assignment heuristic has the form

$$
k=operatorname{clip}\left(
\left\lfloor k_0+\log_2\frac{\sqrt{wh}}{s_0}\right\rfloor,
 k_{min},k_{max}
\right).
$$

`MultiScaleRoIAlign` then pools a fixed-size tensor from the selected level. FPN improves small-object coverage and makes the phrases “backbone stride” and “one `spatial_scale`” insufficient: each pyramid level has its own scale.

## 27. Compact RPN code

### 27.1 RPN prediction head

```python
import torch
from torch import nn
import torch.nn.functional as F


class RPNHead(nn.Module):
    """One feature level; extend by applying shared weights to every FPN level."""

    def __init__(self, in_channels: int, anchors_per_location: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.objectness = nn.Conv2d(in_channels, anchors_per_location, 1)
        self.box_deltas = nn.Conv2d(in_channels, anchors_per_location * 4, 1)

        for layer in (self.conv, self.objectness, self.box_deltas):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, feature):
        h = F.relu(self.conv(feature))
        logits = self.objectness(h)       # [B,A,H,W]
        deltas = self.box_deltas(h)       # [B,4A,H,W]
        return logits, deltas
```

### 27.2 Anchor generation

```python
import math
import torch


def make_base_anchors(scales, aspect_ratios, device=None):
    """Centered xyxy anchors; aspect ratio is width / height."""
    anchors = []
    for scale in scales:
        for ratio in aspect_ratios:
            width = scale * math.sqrt(ratio)
            height = scale / math.sqrt(ratio)
            anchors.append([-width / 2, -height / 2,
                            width / 2, height / 2])
    return torch.tensor(anchors, dtype=torch.float32, device=device)


def tile_anchors(base_anchors, height, width, stride, offset=0.5):
    ys = (torch.arange(height, device=base_anchors.device) + offset) * stride
    xs = (torch.arange(width, device=base_anchors.device) + offset) * stride
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    shifts = torch.stack((xx, yy, xx, yy), dim=-1).reshape(-1, 4)
    return (shifts[:, None, :] + base_anchors[None, :, :]).reshape(-1, 4)


base = make_base_anchors(
    scales=[128, 256, 512],
    aspect_ratios=[0.5, 1.0, 2.0],
)
anchors = tile_anchors(base, height=50, width=80, stride=16)
assert anchors.shape == (50 * 80 * 9, 4)
```

The ratio convention must be checked when moving between libraries: some define aspect ratio as $h/w$ rather than $w/h$.

### 27.3 RPN loss

```python
import torch
import torch.nn.functional as F


def rpn_loss(objectness_logits, box_deltas, labels, regression_targets):
    """
    objectness_logits: [N] sampled/flattened logits
    box_deltas:        [N,4]
    labels:            [N], 1=positive, 0=negative, -1=ignore
    regression_targets:[N,4]
    """
    valid = labels >= 0
    positive = labels == 1

    if valid.any():
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness_logits[valid], labels[valid].float(), reduction="mean"
        )
    else:
        objectness_loss = objectness_logits.sum() * 0.0

    if positive.any():
        box_loss = F.smooth_l1_loss(
            box_deltas[positive],
            regression_targets[positive],
            beta=1.0,
            reduction="sum",
        ) / valid.sum().clamp(min=1)
    else:
        box_loss = box_deltas.sum() * 0.0

    return objectness_loss, box_loss
```

A complete RPN must additionally implement IoU matching, balanced sampling, multi-level tensor reshaping, box decoding, clipping, small-box removal, top-k selection, and NMS. Those steps are already tested in TorchVision; reimplement them only when the learning objective requires architectural control.

## 28. Recommended modern TorchVision implementation

TorchVision's current Faster R-CNN API expects a **list** of image tensors, not one pre-stacked tensor. Each image is `[C,H,W]`, may have a different size, and should be floating point in the `[0,1]` range.

### 28.1 Fine-tuning a pretrained detector

```python
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights)

num_foreground_classes = 4  # e.g. worker, hand, tool, work-area
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features,
    num_foreground_classes + 1,  # TorchVision includes background
)

images = [
    torch.rand(3, 480, 640),
    torch.rand(3, 512, 768),
]
targets = [
    {
        "boxes": torch.tensor([[40., 50., 220., 430.]]),
        "labels": torch.tensor([1], dtype=torch.int64),
    },
    {
        "boxes": torch.tensor([[100., 80., 280., 260.],
                               [350., 200., 600., 480.]]),
        "labels": torch.tensor([2, 3], dtype=torch.int64),
    },
]

model.train()
losses = model(images, targets)
loss = sum(losses.values())
# Typical keys: loss_classifier, loss_box_reg,
#               loss_objectness, loss_rpn_box_reg
loss.backward()
```

Labels must be positive integers for foreground classes; `0` is reserved for background. Target boxes use image-space `xyxy` coordinates and must satisfy $x_1<x_2$, $y_1<y_2$.

### 28.2 Inference

```python
model.eval()
with torch.no_grad():
    predictions = model(images)

for prediction in predictions:
    boxes = prediction["boxes"]
    labels = prediction["labels"]
    scores = prediction["scores"]
    keep = scores >= 0.5
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
```

Do not use a universal score threshold without validating it per task. AP is computed from ranked scores and should normally be evaluated before an arbitrary visualization threshold is imposed.

### 28.3 Important implementation defaults

TorchVision's standard ResNet-50-FPN Faster R-CNN uses concepts that differ from the 2015 model:

- ResNet-50 plus FPN rather than VGG16 with one feature map;
- multi-scale anchors rather than nine anchors on a single map;
- `MultiScaleRoIAlign` rather than RoI Pool;
- joint autograd training rather than four-step alternating optimization;
- a binary objectness formulation inside the RPN;
- separate configurable thresholds and proposal counts for train and evaluation.

Therefore it is a modern Faster R-CNN implementation, not an exact paper reproduction.

## 29. Fast R-CNN versus Faster R-CNN

| Property | Fast R-CNN | Faster R-CNN |
| --- | --- | --- |
| Proposal source | external, e.g. Selective Search | learned RPN |
| Proposal supervision | none inside model | objectness + anchor regression |
| Backbone sharing | among RoIs | among RPN and all RoIs |
| Main losses | RoI class + RoI box | RPN objectness + RPN box + RoI class + RoI box |
| End-to-end proposal learning | no | yes |
| Typical proposals in original papers | about 2,000 | about 300 at test time |
| Primary bottleneck | external proposals | backbone/RPN/RoI computation |
| Original pooling | RoI Pool | RoI Pool in 2015; RoIAlign commonly used now |

The RoI classifier/regressor is not made obsolete by Faster R-CNN. It becomes its second stage.

## 30. Faster R-CNN strengths, limitations, and debugging

### Strengths

- high-quality class-specific refinement;
- proposal generation adapts to the training data;
- shared features make learned proposals inexpensive relative to external methods;
- FPN variants handle a broad range of object sizes;
- modular RPN and RoI losses are interpretable and debuggable.

### Limitations

- more latency and memory than many one-stage detectors;
- anchor design and several IoU/NMS thresholds introduce hyperparameters;
- proposal selection and NMS are non-differentiable routing operations;
- dense anchors create a severe object/background imbalance;
- recall can suffer for tiny, elongated, or unusual objects if anchors/FPN levels are unsuitable;
- BatchNorm and large-resolution training are difficult with small GPU batches.

### Failure diagnosis

| Symptom | Inspect first |
| --- | --- |
| RPN objectness loss falls but recall is poor | anchor sizes/ratios, positive matching, pre-NMS top-k |
| RPN proposes only giant boxes | anchor scale convention or box decode bug |
| RPN proposals look good but classes are wrong | RoI labels, class mapping, or domain shift |
| Small hands/tools are missing | FPN level, input resolution, smallest anchors |
| Duplicate final boxes | final box NMS/score calibration, not only RPN NMS |
| Loss becomes NaN | degenerate targets, exponential box decode, mixed-precision overflow |
| Validation is much worse than training | leakage-safe split, augmentation mismatch, frozen backbone |

## 31. Recommended ablations for the poka-yoke detector

| Ablation | Question answered |
| --- | --- |
| ground-truth/noisy boxes → Fast R-CNN head | How good is classification if proposals are controlled? |
| external proposals → Fast R-CNN vs RPN → Faster R-CNN | How much does learned proposal generation help? |
| class-agnostic RPN recall by object type | Which objects are proposal-limited? |
| default vs smaller anchors | Are hands/tools below the default scale range? |
| single feature map vs FPN | Does high-resolution semantic information improve small objects? |
| frozen vs trainable backbone stages | Is there enough data for domain adaptation? |
| generic COCO pretraining vs domain fine-tuning | How large is the factory-domain gap? |
| ground-truth vs predicted detector boxes for video crops | How much error propagates into 3D action recognition? |

For spatial poka-yoke decisions, separately measure:

- worker, tool, and work-area AP;
- RPN recall at several proposal budgets;
- association accuracy between worker and held object;
- inside/outside accuracy from predicted boxes;
- final event accuracy after [[04 Reference/Document Hub/ResNet Architecture|3D ResNet]] or [[04 Reference/Document Hub/I3D — Inflated 3D ConvNet Deep Dive for Braking-Action Recognition|I3D]].

## 32. Final synthesis

Fast R-CNN answers: **given candidate regions, how can one shared CNN efficiently classify and refine them?** Faster R-CNN answers the remaining question: **can those candidate regions also be learned from the shared feature map?**

The complete architecture is therefore

$$
\boxed{
I\rightarrow\text{shared backbone}
\rightarrow\text{RPN anchors/objectness/box refinement}
\rightarrow\text{proposal NMS}
\rightarrow\text{RoIAlign}
\rightarrow\text{Fast R-CNN class/box head}
\rightarrow\text{final NMS}
}
$$

The key conceptual separation is worth preserving during debugging: the **RPN controls proposal recall**, while the **RoI head controls class discrimination and final localization**. Evaluate both stages independently before interpreting end-to-end AP.
