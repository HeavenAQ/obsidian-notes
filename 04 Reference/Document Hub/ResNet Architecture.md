---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-05-30T17:29:00
Status: Done
Last updated time: 2026-07-17T18:42:53
Last edited by: Heaven Chen
Category:
  - CV
  - ResNet
  - Video Understanding
  - Theory
---
# CNN Recap
![[Pasted image 20260720154212.png]]

```python
import torch.nn as nn
import torch

class FlexibleCNN(nn.module):
	def __init__(
		self,
		in_channels: int,
		n_layers: int,
		n_filters: list[int],
		kernel_sizes: list[int],
		n_classes: int,
		dropout_rate: float, 
		fc_size: int,
	):
		super(FlexibleCNN, self).__init__()
		
		self.dropout_rate = dropout_rate
		self.fc_size = fc_size
		self.n_classes = n_classes
		
		blocks = []
		for i in range(n_layers):
			padding = (kernel_sizes[i] - 1) // 2
			block = nn.Sequential(
				nn.Conv2d(
					in_channels, 
					n_filters[i], 
					kernel_sizes[i],
					padding=padding,
				),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=2, stride=2),
			)
			blocks.append(block)
			in_channels = n_filters[i]
	
	def _create_classifier(self, flattened_size, device):
		self.classifier = nn.Sequential(
			nn.Dropout(self.dropout_rate),
			nn.Linear(flattened_size, self.fc_size),
			nn.ReLU(),
			nn.Dropout(self.dropout_rate),
			nn.Linear(self.fc_size, self.n_classes),
		).to(device)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		device = x.device
		x = self.features(x)
		flattened = torch.flatten(x, 1)
		flattened = flattened.size(1)
		
		if self.classifier is None:
			self._create_classifier(flattened_size, device)
		return self.classifier(flattened)
```

# ResNet Deep Dive — 2D ResNet-18 and 3D ResNets for Video

> [!abstract]
> **“ResNet-18” is ambiguous unless the dimensionality is stated.** The canonical model is a 2D image CNN; a **3D ResNet-18** replaces its spatial operators with spatiotemporal ones and consumes video clips directly. This note develops both, with the 3D implementation based on Hara et al.'s `3D-ResNets-PyTorch` repository.

**Primary references:** [He et al., “Deep Residual Learning for Image Recognition,” CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) · [Hara et al., “Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?”, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.html) · [Hara et al., `3D-ResNets-PyTorch`](https://github.com/kenshohara/3D-ResNets-PyTorch) · [TorchVision 2D ResNet source](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)

## 1. Why ResNet exists

Simply adding layers to a plain CNN can produce a **degradation problem**: the deeper model has higher training error, even though it should be able to imitate the shallower model by making added layers act like identities. This is an optimization problem, not merely overfitting.

Instead of asking stacked layers to learn a direct mapping $H(x)$, a residual block learns

$$
F(x)=H(x)-x,
$$

then reconstructs the desired mapping with a shortcut:

$$
y=F(x;W)+x.
$$

If the identity is already useful, the residual branch only needs to learn a correction near zero.

## 2. Residual-block mathematics

For ResNet-18's two-convolution **BasicBlock**,

$$
z_1=\operatorname{ReLU}(\operatorname{BN}(W_1*x)),
$$

$$
z_2=\operatorname{BN}(W_2*z_1),
$$

$$
y=\operatorname{ReLU}(z_2+S(x)),
$$

where $*$ is 2D convolution and $S$ is either:

- $S(x)=x$ when spatial size and channel count are unchanged; or
- a learned $1\times1$ projection with matching stride when dimensions change.

The shortcut creates a direct gradient term. If $y=x+F(x)$, then

$$
\frac{\partial y}{\partial x}=I+\frac{\partial F}{\partial x}.
$$

For a loss $L$,

$$
\frac{\partial L}{\partial x}
=\frac{\partial L}{\partial y}
\left(I+\frac{\partial F}{\partial x}\right).
$$

The identity contribution gives optimization an unobstructed path; the residual path does not need to carry the entire gradient by itself.

### 2.1 Convolution output size

For one spatial dimension,

$$
H_{out}=\left\lfloor\frac{H_{in}+2p-d(k-1)-1}{s}+1\right\rfloor,
$$

where $k$ is kernel size, $s$ stride, $p$ padding, and $d$ dilation.

### 2.2 Batch normalization

For activation channel $c$ in a mini-batch,

$$
\hat{x}_c=\frac{x_c-\mu_c}{\sqrt{\sigma_c^2+\epsilon}},
\qquad y_c=\gamma_c\hat{x}_c+\beta_c.
$$

BatchNorm stabilizes the original architecture but becomes unreliable with very small batches. For small video batches, freeze pretrained BatchNorm statistics, use synchronized BatchNorm, or consider GroupNorm.

## 3. Exact ResNet-18 topology

ResNet-18 uses **BasicBlocks** arranged as `[2, 2, 2, 2]`. Each block has two $3\times3$ convolutions. The canonical 224×224 image path is:

| Stage | Operator | Output shape | Blocks |
| --- | --- | --- | ---: |
| Input | RGB image | $3\times224\times224$ | — |
| Stem | $7\times7$, 64, stride 2 + BN + ReLU | $64\times112\times112$ | — |
| Pool | $3\times3$ max pool, stride 2 | $64\times56\times56$ | — |
| `layer1` | BasicBlock, 64 channels | $64\times56\times56$ | 2 |
| `layer2` | BasicBlock, 128 channels; first block stride 2 | $128\times28\times28$ | 2 |
| `layer3` | BasicBlock, 256 channels; first block stride 2 | $256\times14\times14$ | 2 |
| `layer4` | BasicBlock, 512 channels; first block stride 2 | $512\times7\times7$ | 2 |
| Head | global average pool | $512$ | — |
| Classifier | fully connected | number of classes | — |

The “18” counts 17 convolutional layers plus the final fully connected layer. The model has roughly 11.7 million parameters in its ImageNet form.

## 4. What ResNet-18 learns

Its hierarchy usually progresses from:

1. edges, color contrasts, and simple textures;
2. contours and local parts;
3. object-part arrangements;
4. high-level spatial evidence useful for classification.

Global average pooling converts the last feature map $A\in\mathbb{R}^{512\times H\times W}$ to

$$
g_c=\frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W}A_{c,h,w}.
$$

The binary braking logit can then be

$$
z=w^\top g+b, \qquad P(y=1\mid x)=\sigma(z)=\frac{1}{1+e^{-z}}.
$$

Binary cross-entropy is

$$
\mathcal{L}_{BCE}=-\frac1N\sum_i
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].
$$

## 5. The crucial 2D-versus-3D distinction

ResNet-18 consumes tensors shaped `[B, C, H, W]`. A video has an extra temporal axis, normally `[B, C, T, H, W]`.

Applying the same ResNet independently to every frame gives

$$
f_t=\operatorname{ResNet18}(x_t),
$$

but it does not model motion. A temporal head must combine $f_1,\ldots,f_T$:

$$
\bar{f}=\frac1T\sum_{t=1}^{T}f_t,
\qquad \hat{y}=\operatorname{softmax}(W\bar{f}+b).
$$

Possible temporal heads include mean/max pooling, an LSTM/GRU, a temporal convolutional network, or attention. In contrast, a 3D CNN applies kernels across time and space simultaneously:

$$
y_{c_o,t,h,w}=b_{c_o}+
\sum_{c_i,\tau,i,j}
W_{c_o,c_i,\tau,i,j}
x_{c_i,t+\tau,h+i,w+j}.
$$

> [!warning] Naming trap
> `torchvision.models.resnet18` is a **2D** ResNet-18. `torchvision.models.video.r3d_18` is an **18-layer 3D ResNet** and is a different architecture. Original I3D inflates an Inception-v1 network, not ResNet-18.

## 6. 3D ResNet: residual learning over space and time

A 3D ResNet accepts a clip

$$
X\in\mathbb{R}^{B\times C_{in}\times T\times H\times W}
$$

and preserves the ResNet idea while replacing `Conv2d`, `BatchNorm2d`, and spatial pooling with `Conv3d`, `BatchNorm3d`, and spatiotemporal pooling. It is “3D” because its kernels move along **time, height, and width**—not because it reconstructs 3D geometry.

### 6.1 3D convolution mathematics

For kernel $W\in\mathbb{R}^{C_{out}\times C_{in}\times k_t\times k_h\times k_w}$,

$$
Y_{b,c_o,t,h,w}=b_{c_o}+
\sum_{c_i=1}^{C_{in}}
\sum_{\tau=0}^{k_t-1}
\sum_{i=0}^{k_h-1}
\sum_{j=0}^{k_w-1}
W_{c_o,c_i,\tau,i,j}
X_{b,c_i,t',h',w'},
$$

where

$$
t'=t s_t+\tau d_t-p_t,\quad
h'=h s_h+i d_h-p_h,\quad
w'=w s_w+j d_w-p_w.
$$

The output length along any axis $a\in\{t,h,w\}$ is

$$
L_{out}^{(a)}=
\left\lfloor
\frac{L_{in}^{(a)}+2p_a-d_a(k_a-1)-1}{s_a}+1
\right\rfloor.
$$

Ignoring bias, a dense 3D convolution has

$$
\#\theta=C_{out}C_{in}k_tk_hk_w
$$

parameters. Thus a $3\times3\times3$ layer has three times the kernel weights of a $1\times3\times3$ spatial layer with the same channels. Activation memory is also multiplied by the retained temporal length $T$, which is why video models usually require much smaller batches.

### 6.2 The 3D BasicBlock

Hara's 3D ResNet-18 uses the same `[2, 2, 2, 2]` BasicBlock schedule as 2D ResNet-18, but every $3\times3$ becomes $3\times3\times3$:

$$
Z_1=\operatorname{ReLU}\!\left(\operatorname{BN3D}(W_1*_{3D}X)\right),
$$

$$
Z_2=\operatorname{BN3D}(W_2*_{3D}Z_1),
$$

$$
Y=\operatorname{ReLU}\!\left(Z_2+S(X)\right).
$$

The shortcut is identity when shape is unchanged. At a stage transition, Hara's default **shortcut B** is a learned $1\times1\times1$ convolution plus BatchNorm3d:

$$
S(X)=\operatorname{BN3D}(W_s*_{1\times1\times1}X),
\qquad \operatorname{stride}(W_s)=(2,2,2).
$$

Shortcut **A** instead average-pools and zero-pads channels; it has no learned projection weights. Shortcut B is the repository default and is normally the cleanest choice for transfer learning.

The residual gradient argument is unchanged:

$$
\frac{\partial L}{\partial X}
=\frac{\partial L}{\partial Y}
\left(I+\frac{\partial F(X;W)}{\partial X}\right),
$$

but $F$ can now represent motion-sensitive features such as approach, contact, deceleration, or withdrawal.

## 7. Exact Hara `3D-ResNets-PyTorch` ResNet-18

The repository's current `models/resnet.py` constructs depth 18 as

```python
ResNet(BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512])
```

Its defaults are a 7-frame temporal stem, temporal stem stride 1, shortcut B, and a max-pool over all three axes. With the repository's default $16\times112\times112$ clip, the tensor path is:

| Stage | Operator | Output for `[B,3,16,112,112]` |
| --- | --- | --- |
| Stem | Conv3d $7\times7\times7$, 64, stride $(1,2,2)$, padding $(3,3,3)$ | `[B,64,16,56,56]` |
| Pool | MaxPool3d $3^3$, stride $(2,2,2)$, padding 1 | `[B,64,8,28,28]` |
| `layer1` | 2 BasicBlocks, 64 channels | `[B,64,8,28,28]` |
| `layer2` | 2 BasicBlocks, 128; first stride 2 | `[B,128,4,14,14]` |
| `layer3` | 2 BasicBlocks, 256; first stride 2 | `[B,256,2,7,7]` |
| `layer4` | 2 BasicBlocks, 512; first stride 2 | `[B,512,1,4,4]` |
| Head | AdaptiveAvgPool3d $(1,1,1)$ + linear | `[B,n_classes]` |

The network has about **33.2M parameters** with a two-class head. The large jump from 2D ResNet-18's roughly 11.7M comes mainly from using full $3^3$ kernels.

> [!important] Repository-version caveat
> The repository README says the main branch was substantially refactored in April 2020 and may not exactly reproduce the CVPR 2018 paper. Use its `CVPR2018` branch for strict reproduction. Record the Git commit, branch, preprocessing, and checkpoint source in every experiment.

The architecture and defaults above were verified against [`models/resnet.py` at commit `540a0ea`](https://github.com/kenshohara/3D-ResNets-PyTorch/blob/540a0ea1abaee379fa3651d4d5afbd2d667a1f49/models/resnet.py) and the corresponding [`opts.py`](https://github.com/kenshohara/3D-ResNets-PyTorch/blob/540a0ea1abaee379fa3651d4d5afbd2d667a1f49/opts.py).

### 7.1 Supported depths

The generator provides:

| Depth | Block | Stage repetitions |
| ---: | --- | --- |
| 10 | BasicBlock | `[1,1,1,1]` |
| 18 | BasicBlock | `[2,2,2,2]` |
| 34 | BasicBlock | `[3,4,6,3]` |
| 50 | Bottleneck | `[3,4,6,3]` |
| 101 | Bottleneck | `[3,4,23,3]` |
| 152 | Bottleneck | `[3,8,36,3]` |
| 200 | Bottleneck | `[3,24,36,3]` |

For the Bottleneck, the residual branch is $1^3\rightarrow3^3\rightarrow1^3$ and expands the output channels by 4. On a small custom braking dataset, deeper is not automatically better: Hara et al. found that small video datasets can strongly overfit when 3D models are trained from scratch, whereas large-scale Kinetics pretraining made deep networks viable.

## 8. Temporal resolution, sampling, and receptive field

Temporal sampling defines the physical motion seen by the network. If a clip contains $T$ frames sampled every $q$ source frames from video recorded at $f$ FPS, its approximate coverage is

$$
\Delta t=\frac{(T-1)q}{f}.
$$

For $T=16$, $q=2$, and $f=30$, the clip spans $1$ second. Increasing $q$ expands coverage but may skip the short contact event; increasing $T$ raises memory cost.

For layer $\ell$, temporal jump $j_\ell$ and receptive field $r_\ell$ obey

$$
j_\ell=j_{\ell-1}s_\ell,
\qquad
r_\ell=r_{\ell-1}+(k_\ell-1)j_{\ell-1},
$$

with $j_0=r_0=1$. Full $3\times3\times3$ kernels expand the temporal receptive field at every residual convolution. In the default Hara model, max-pool and the first blocks of `layer2`–`layer4` also halve time, so a 16-frame input becomes one temporal cell before global pooling.

> [!warning] Do not crop each frame independently
> Spatial augmentation parameters must be shared across every frame in a clip. Otherwise random crop or flip creates artificial camera motion. Temporal reversal is also unsafe when action order changes the class.

## 9. Faithful compact PyTorch implementation

This is the essential Hara-style R3D-18 without repository training infrastructure:

```python
import torch
from torch import nn


def conv3x3x3(cin, cout, stride=1):
    return nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = conv3x3x3(cin, cout, stride)
        self.bn1 = nn.BatchNorm3d(cout)
        self.conv2 = conv3x3x3(cout, cout)
        self.bn2 = nn.BatchNorm3d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Identity() if stride == 1 and cin == cout else
            nn.Sequential(
                nn.Conv3d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm3d(cout),
            )
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class HaraR3D18(nn.Module):
    def __init__(self, num_classes=2, in_channels=3):
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(7, 7, 7),
                      stride=(1, 2, 2), padding=(3, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, channels, blocks, stride):
        layers = [BasicBlock3D(self.in_channels, channels, stride)]
        self.in_channels = channels
        layers += [BasicBlock3D(channels, channels) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):              # x: [B,C,T,H,W]
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(self.pool(x).flatten(1))


model = HaraR3D18(num_classes=2)
logits = model(torch.randn(2, 3, 16, 112, 112))  # [2,2]
```

The repository additionally applies Kaiming initialization to Conv3d, initializes BatchNorm scale to 1 and bias to 0, offers shortcut A/B, and supports configurable input channels, stem temporal kernel/stride, max-pool removal, and width multiplier.

## 10. Using Hara's repository and pretrained checkpoints

The repository is script-based rather than an installed package:

```bash
git clone https://github.com/kenshohara/3D-ResNets-PyTorch.git
cd 3D-ResNets-PyTorch
git rev-parse HEAD  # record this in the experiment log
```

Build a fresh two-class model:

```python
from models.resnet import generate_model

model = generate_model(
    model_depth=18,
    n_classes=2,
    n_input_channels=3,
    conv1_t_size=7,
    conv1_t_stride=1,
    shortcut_type="B",
    no_max_pool=False,
)
```

For a repository checkpoint, instantiate the **pretraining class count first**, load its state dictionary, and only then replace the classifier:

```python
import torch
from torch import nn
from models.resnet import generate_model

pretrain_classes = 700  # must match the checkpoint, e.g. r3d18_K_200ep.pth
model = generate_model(18, n_classes=pretrain_classes)
checkpoint = torch.load("r3d18_K_200ep.pth", map_location="cpu")
state = checkpoint.get("state_dict", checkpoint)

# DataParallel checkpoints often prefix keys with "module.".
state = {k.removeprefix("module."): v for k, v in state.items()}
model.load_state_dict(state, strict=True)
model.fc = nn.Linear(model.fc.in_features, 2)
```

Do **not** silently load with `strict=False` until the only expected mismatch is understood. A checkpoint from the old/CVPR2018 branch may require architecture or key conversion, as the repository README explicitly warns.

### 10.1 TorchVision is similar, not identical

For a maintained baseline:

```python
from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights

weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)
preprocess = weights.transforms()
```

TorchVision's `r3d_18` should not be assumed checkpoint-compatible with Hara's model: its stem and downsampling implementation differ. Treat them as two R3D-18 variants and report which one was used.

## 11. Losses, fine-tuning, and long-video inference

For mutually exclusive classes, use clip-level cross-entropy:

$$
\mathcal{L}_{CE}=-\frac1B\sum_{i=1}^{B}\log
\frac{e^{z_{i,y_i}}}{\sum_{c=1}^{K}e^{z_{i,c}}}.
$$

For a long video divided into $M$ clips, average **logits** or probabilities consistently before the final decision:

$$
\bar z=\frac1M\sum_{m=1}^{M}z^{(m)},
\qquad \hat y=\arg\max_c \bar z_c.
$$

A practical two-stage transfer schedule is:

```python
# Stage 1: train only the new classifier.
for p in model.parameters():
    p.requires_grad = False
for p in model.fc.parameters():
    p.requires_grad = True

optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# video must be [B,C,T,H,W], labels [B]
logits = model(video)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

Then unfreeze `layer4`, and later `layer3` if validation improves, using a backbone learning rate about 10–100× smaller than the head. With tiny batches, either freeze BatchNorm running statistics or use a normalization design that is stable at small batch sizes.

### 11.1 Braking-action clip design

For braking-point recognition, a clip should include **pre-contact, contact/deceleration, and post-contact** frames. Recommended controlled comparisons are:

1. Hara R3D-18 from scratch;
2. Hara R3D-18 with Kinetics-style pretraining;
3. TorchVision `r3d_18` with its coupled preprocessing;
4. 2D ResNet-18 plus temporal pooling;
5. [[04 Reference/Document Hub/I3D — Inflated 3D ConvNet Deep Dive for Braking-Action Recognition|I3D]].

Keep the same ROI, temporal coverage, split, and labels. Otherwise the experiment confounds architecture with input information.

## 12. PyTorch: single-frame work-area classifier

```python
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)  # outside / braking-inside

# Use preprocessing coupled to the selected weights.
preprocess = weights.transforms()

images = torch.randn(8, 3, 224, 224)
logits = model(images)                         # [B, 2]
loss = nn.CrossEntropyLoss()(logits, torch.randint(0, 2, (8,)))
```

For a small dataset, first freeze most of the backbone:

```python
for parameter in model.parameters():
    parameter.requires_grad = False

for parameter in model.layer4.parameters():
    parameter.requires_grad = True
for parameter in model.fc.parameters():
    parameter.requires_grad = True
```

After the head stabilizes, unfreeze more stages with a lower backbone learning rate.

## 13. PyTorch: frame encoder plus temporal pooling

This is a fair intermediate baseline between single frames and a full 3D CNN.

```python
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18TemporalPool(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, video):
        # video: [B, C, T, H, W]
        b, c, t, h, w = video.shape
        frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        features = self.encoder(frames).flatten(1)  # [B*T, 512]
        features = features.view(b, t, -1).mean(dim=1)
        return self.classifier(features)
```

Mean pooling is order-insensitive. If “braking” is defined by a change over time rather than a static pose, replace it with a temporal model.

## 14. Role in the poka-yoke pipeline

A clean pipeline separates **where/what** from **what action occurred**:

```text
video frames
  → object detector: worker + held object + work-area bbox
  → tracking / association across frames
  → crop or ROIAlign around worker–object–area interaction
  → ResNet-18 per-frame baseline
  → temporal aggregation
  → braking-inside / braking-outside / no-braking decision
```

Useful ROI representations:

- RGB crop around the union of worker, held object, and area boxes;
- additional binary masks for worker/object/area;
- normalized box geometry such as centers, IoU, distances, and relative scale;
- a short clip that includes frames before and after the braking point.

For boxes $A$ and $B$,

$$
\operatorname{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

Box-center inclusion or IoU can provide an interpretable spatial baseline. The image/video model should only be asked to learn appearance or temporal cues that geometry alone cannot resolve.

## 15. Training guidance for this experiment

### Data splits

- Split by worker, session, or production run—not randomly by neighboring frames.
- Keep every clip from the same source video in one split.
- Never crop training data with ground-truth boxes while using noisy detector boxes only at test time; train with realistic detector noise.

### Augmentation

- Usually safe: mild color jitter, blur, compression, scale jitter, and small translations.
- Conditionally safe: horizontal flip only if left/right orientation does not change the label.
- Dangerous: temporal reversal, aggressive random crop that removes the tool/area, or geometry changes that invalidate “inside/outside.”

### Imbalance

For weighted cross-entropy,

$$
\mathcal{L}=-\frac1N\sum_i w_{y_i}\log p_{i,y_i}.
$$

Choose class weights from the training split only. Also report per-class precision, recall, F1, and the confusion matrix rather than accuracy alone.

## 16. Strengths, limitations, and expected role

### Strengths

- fast, compact, and easy to fine-tune;
- strong ImageNet initialization;
- good baseline for pose, object state, and spatial-layout cues;
- can process more frames per second than heavier video models.

### Limitations

- no native temporal reasoning;
- frame predictions may flicker;
- shortcuts/backgrounds can dominate on small datasets;
- global pooling discards fine spatial detail unless the ROI is carefully constructed;
- BatchNorm can be fragile with tiny batches.

### Best experimental use

Use ResNet-18 as the **2D baseline**. Compare:

1. single center frame;
2. frame logits averaged over a clip;
3. frame features plus a temporal head;
4. native 3D CNN (`r3d_18` or I3D).

This isolates whether temporal modeling—not merely more computation—improves braking-point recognition.

## 17. Recommended ablations

| Ablation | Question answered |
| --- | --- |
| Whole frame vs ROI crop | Does background context help or create shortcuts? |
| RGB only vs RGB + masks/box geometry | Does explicit area geometry improve inside/outside decisions? |
| Ground-truth vs detected boxes | How much error comes from detection? |
| One frame vs 8/16/32 frames | How much temporal context is needed? |
| Frozen vs fine-tuned backbone | Is the dataset large enough to adapt visual features? |
| Random-frame vs worker/session split | Was apparent performance caused by leakage? |

## 18. Bottom line

**2D ResNet-18** is the correct lightweight control experiment, but it is not evidence that a 3D CNN works. If the braking cue depends on motion onset, duration, or ordering, compare it with **Hara-style 3D ResNet-18** under the same crops, temporal coverage, splits, labels, and evaluation protocol. Use pretrained 3D weights when data are limited, record the exact repository/checkpoint version, and treat TorchVision `r3d_18` as a related but non-identical implementation.
