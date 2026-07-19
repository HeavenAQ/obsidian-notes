---
base: "[[01.2 DL Daily Lessons — Step-by-Step + Quiz.base]]"
Quiz score: ""
"HW #": 1
Studied: false
Quiz taken: false
Day type: Lesson-Day
Date: 2026-07-06
Piece count: 5
---
> ⚠️ **Schedule note:** today follows the plan — **PT-Cert C2 M1–M2** (Optuna hyperparameter optimization; TorchVision transforms + fine-tuning ResNet/MobileNet) + **6.S191 Lecture 3** (CNNs). CNNs are the "architectures for grids" idea behind CIFAR-10, and TorchVision transforms are exactly HW1 **Problem 14** (train-only augmentation). Ungraded-quiz backlog is still open: **Jul 2, Jul 3, and Jul 5 quizzes are all untaken** — grade at least yesterday's (Jul 5) before starting, ~10 min. Deep-dive companion when you want more depth on the PT-Cert material: [PyTorch cert companion](https://app.notion.com/p/393e445b30a481939348ce3a62c5f5ff).

# 🎯 Today's goal

Understand *why* convolution is the right inductive bias for images (parameter sharing + locality + translation equivariance), be able to compute a conv layer's output shape and parameter count from memory, and know the two production shortcuts you'll reach for on CIFAR-10: **transfer learning** (fine-tune a pretrained ResNet/MobileNet instead of training from scratch) and **automated hyperparameter search** (Optuna instead of hand-tuning LR). By tonight you should be able to write a TorchVision augmentation pipeline for HW1 Problem 14 and sketch an Optuna study from an empty file.

# 🧩 Pieces

## Piece 1 — The convolution operation: math & why (6.S191 L3, ~35 min)

*Source: 6.S191 Lecture 3 (Deep Computer Vision), first half. This is the conceptual heart of today.*

A fully-connected layer on a $32\times32\times3$ image has $3072$ inputs — a single hidden unit already needs 3072 weights, and it treats a cat in the top-left as unrelated to the same cat in the bottom-right. A **convolutional layer** fixes both problems with one idea: **slide a small shared filter over the image**.

For a single-channel input $X$ and a $k\times k$ kernel $K$, the (cross-correlation, as actually implemented in PyTorch) output is

$$
S(i,j) = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} X(i+m,\, j+n)\, K(m,n)
$$

With $C_{\text{in}}$ input channels and $C_{\text{out}}$ filters, stride $s$ and one bias per output channel:

$$
Y_{o,i,j} = b_o + \sum_{c=1}^{C_{\text{in}}}\sum_{m=0}^{k-1}\sum_{n=0}^{k-1} X_{c,\; i s + m,\; j s + n}\; K_{o,c,m,n}
$$

So the weight tensor has shape $(C_{\text{out}}, C_{\text{in}}, k, k)$ — **independent of the image size**. That is the whole payoff.

Three properties to be able to name:

- **Parameter sharing.** The *same* $K_{o,c,\cdot,\cdot}$ is reused at every spatial position, so parameter count is $C_{\text{out}}(C_{\text{in}}k^2 + 1)$ — tiny and decoupled from $H,W$.
- **Sparse / local connectivity.** Each output depends on only a $k\times k$ patch, not all pixels — cheap and matched to the fact that image structure is local.
- **Translation equivariance.** Shift the input, and the feature map shifts the same way: $\text{conv}(\text{shift}(x)) = \text{shift}(\text{conv}(x))$. The network learns "edge detector" once and applies it everywhere.

The **output spatial size** with padding $p$ is the formula you'll use constantly:

$$
H_{\text{out}} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1
$$

```python
import torch, torch.nn as nn
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                 stride=1, padding=1)          # 'same' padding for k=3
x = torch.randn(8, 3, 32, 32)                  # (N, C, H, W)
print(conv(x).shape)                           # -> (8, 16, 32, 32)
print(sum(p.numel() for p in conv.parameters()))  # 16*(3*3*3 + 1) = 448
```

**You've got this piece when you can** write the multi-channel conv sum, state the weight-tensor shape and why it's independent of image size, compute $H_{\text{out}}$ for any $(k,s,p)$, and explain translation equivariance in one sentence.

## Piece 2 — CNN building blocks & the full architecture (~30 min)

*Source: 6.S191 L3, second half.*

A CNN stacks three repeating operations, then flattens into a classifier: **Conv → nonlinearity (ReLU) → pooling**, repeated, then **Flatten → Linear → logits**.

**Padding & stride** control geometry. $p = \lfloor k/2 \rfloor$ with $s=1$ gives "same" spatial size (used to keep resolution while adding depth); $s=2$ (or pooling) halves resolution to build a spatial pyramid.

**Pooling** downsamples each channel independently — max-pool over a $2\times2$ window keeps the strongest activation:

$$
P(i,j) = \max_{0\le m,n < 2} A(2i+m,\; 2j+n)
$$

It has **no parameters**, shrinks compute, and gives a little **local translation invariance** (small shifts don't change the max). Note the distinction from Piece 1: conv is *equivariant*, pooling adds a touch of *invariance*.

**Channels grow as resolution shrinks.** A typical block goes $32\times32\times3 \to 32\times32\times16 \to 16\times16\times32 \to 8\times8\times64$: spatial info is traded for semantic (channel) richness. This is the **feature hierarchy** — early layers fire on edges/textures, later layers on object parts — because each stacked conv enlarges the **receptive field**. For $L$ stacked $k\times k$ stride-1 convs the receptive field is

$$
r = 1 + L\,(k-1)
$$

so depth is how a $3\times3$ kernel eventually "sees" the whole image.

```python
import torch.nn as nn
cnn = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32->16
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16->8
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8->4
    nn.Flatten(),
    nn.Linear(64*4*4, 10),                                        # -> logits
)
```

**You've got this piece when you can** hand-trace the shape after every layer above, explain why channels increase as $H,W$ decrease, compute the receptive field of a 3-conv stack, and state the difference between conv-equivariance and pool-invariance.

## Piece 3 — ResNet & MobileNet: the architectures you'll fine-tune (~35 min)

*Source: PT-Cert C2 M2 (fine-tuning ResNet/MobileNet) — architecture background.*

**Residual connections (ResNet).** Naively stacking many layers makes training *worse* (degradation), because gradients vanish through long chains. A residual block instead learns a **residual** on top of an identity shortcut:

$$
y = \mathcal{F}(x, \{W_i\}) + x
$$

The reason it works is visible in the backward pass. Differentiating the block,

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\left(I + \frac{\partial \mathcal{F}}{\partial x}\right)
$$

The $+I$ term is an ungated gradient highway: even if $\partial\mathcal{F}/\partial x \to 0$, the gradient still reaches $x$ undiminished. That single term is why 50- and 152-layer nets train at all.

**BatchNorm** (ubiquitous in ResNet) normalizes each channel over the batch, then re-scales:

$$
\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y = \gamma\hat{x} + \beta
$$

It stabilizes activation statistics so larger learning rates converge — and it's the reason `model.train()` vs `model.eval()` matters (it switches between batch stats and running averages).

**Depthwise-separable convolution (MobileNet).** A standard conv mixes space *and* channels at once, costing $\propto k^2 C_{\text{in}} C_{\text{out}}$. MobileNet factors it into a **depthwise** conv (one $k\times k$ filter per channel, no cross-channel mixing) followed by a **pointwise** $1\times1$ conv (mixes channels only). The cost ratio versus a full conv is

$$
\frac{k^2 C_{\text{in}} + C_{\text{in}} C_{\text{out}}}{k^2 C_{\text{in}} C_{\text{out}}} = \frac{1}{C_{\text{out}}} + \frac{1}{k^2}
$$

For $k=3$ that's roughly a $8\text{–}9\times$ FLOP reduction — why MobileNet is the go-to for phones/edge.

**Transfer learning** is why you care about both. A model pretrained on ImageNet already learned generic edge/texture/part detectors; on CIFAR-10 you reuse them instead of training from scratch. Two regimes: **feature extraction** (freeze the backbone, train only a new classifier head) and **fine-tuning** (unfreeze some/all layers and continue training with a *small* LR so you don't wreck the pretrained weights).

**You've got this piece when you can** write the residual-block forward and its backward, point to the exact term that stops gradients vanishing, explain what BatchNorm normalizes, state the depthwise-separable cost ratio, and contrast feature-extraction vs fine-tuning.

## Piece 4 — TorchVision in practice: transforms + fine-tuning code (~35 min)

*Source: PT-Cert C2 M2 (TorchVision). This piece directly writes HW1 Problem 14.*

**Transforms** are a `Compose`d pipeline run per-sample inside `Dataset.__getitem__` (recall yesterday's Dataset/DataLoader split). Order matters: PIL-space augmentations first, then `ToTensor()` (which also scales pixels to $[0,1]$ and reorders to $(C,H,W)$), then `Normalize(mean, std)` which applies $(x-\mu)/\sigma$ per channel. For pretrained models you **must** use the ImageNet stats the model was trained with:

$$
\mu = (0.485, 0.456, 0.406), \qquad \sigma = (0.229, 0.224, 0.225)
$$

The HW1 Problem 14 rule — **augment train only, never val** — falls straight out of using two different transform pipelines:

```python
from torchvision import transforms
IMAGENET = dict(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # augmentation: train ONLY
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(**IMAGENET),
])
val_tf = transforms.Compose([                  # deterministic: no random ops
    transforms.ToTensor(),
    transforms.Normalize(**IMAGENET),
])
```

Augmentation enlarges the effective training distribution and acts as a **regularizer**, so it should *narrow the train/val gap* (exactly the effect HW1 Problem 15 asks you to observe).

**Fine-tuning a pretrained model** — freeze the backbone, swap the head, train:

```python
import torch, torch.nn as nn
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in model.parameters():          # 1. freeze backbone (feature-extraction)
    p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 10)   # 2. new head -> 10 CIFAR classes

opt = torch.optim.Adam(                # 3. optimize ONLY the params that need grads
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
# later, to fine-tune deeper: unfreeze layer4, add it to opt with a smaller lr
```

Key gotchas: the new `fc` has fresh weights so it *does* train; the frozen params carry `requires_grad=False` so you filter them out of the optimizer; and you keep the pretrained normalization stats.

**You've got this piece when you can** write train/val transform pipelines placing augmentation on train only, explain why `Normalize` must match the pretrained model's stats, and fine-tune a torchvision model by freezing, replacing the head, and passing only trainable params to the optimizer.

## Piece 5 — Hyperparameter optimization with Optuna (PT-Cert C2 M1, ~30 min)

*Source: PT-Cert C2 M1 (Optuna).*

LR, weight decay, batch size, and dropout aren't learned by gradient descent — they're **hyperparameters** searched in an outer loop. Three strategies, increasing sophistication:

- **Grid search:** all combinations — cost explodes exponentially with the number of hyperparameters (curse of dimensionality).
- **Random search:** sample randomly — surprisingly better than grid when only a few hyperparameters matter, because it doesn't waste trials on a fine grid of the *irrelevant* ones.
- **Bayesian / TPE (Optuna's default):** build a probabilistic model of "which regions gave good scores" and sample where improvement is likely. Optuna's **Tree-structured Parzen Estimator** splits past trials into good vs. bad by an objective quantile and models densities $\ell(x)$ (good) and $g(x)$ (bad), then proposes points maximizing $\ell(x)/g(x)$ — i.e. likely-good, rarely-bad.

The API is **define-by-run**: the search space is written *inside* the objective with `trial.suggest_*` calls, so it can be conditional and dynamic.

```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)   # log scale for LR
    wd = trial.suggest_float('weight_decay', 0.0, 1e-3)
    n_units = trial.suggest_int('n_units', 64, 512, step=64)
    opt_name = trial.suggest_categorical('opt', ['adam', 'sgd'])

    model = build_model(n_units)
    opt = make_optimizer(opt_name, model.parameters(), lr, wd)
    for epoch in range(20):
        train_one_epoch(model, opt)
        acc = validate(model)
        trial.report(acc, epoch)              # feed intermediate score to the pruner
        if trial.should_prune():              # early-stop hopeless trials
            raise optuna.TrialPruned()
    return acc                                 # Optuna optimizes this

study = optuna.create_study(direction='maximize',
                            pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=50)
print(study.best_params, study.best_value)
```

**Pruning** is the other big win: `trial.report()` + `should_prune()` lets a **MedianPruner** kill a trial whose intermediate score is below the median of prior trials at the same step — so compute concentrates on promising configs. (Note: `log=True` for LR because good learning rates span orders of magnitude; searching it linearly wastes almost all samples.)

**You've got this piece when you can** explain why random beats grid when few hyperparameters matter, describe in one line what TPE models and maximizes, write an Optuna objective with a mixed (float/int/categorical) search space, and explain what pruning saves.

# 📝 Review quiz

Answer all 8 without notes. Grade with the key at the bottom, then fill in **Quiz score** and check **Quiz taken**.

1. **(Calculation)** A conv layer has input $(N,3,32,32)$, 64 filters, $k=5$, stride 2, padding 2. Give the output shape and the exact number of learnable parameters (include biases).
2. **(Conceptual)** State the three defining properties of a convolutional layer versus a fully-connected layer, and explain which one makes the parameter count independent of image height/width.
3. **(Conceptual)** Distinguish *equivariance* (convolution) from *invariance* (pooling). Which operation provides which, and why does max-pooling give a small amount of translation invariance?
4. **(Derivation)** Write the residual block forward pass and differentiate it to show $\partial L/\partial x$. Point to the single term that prevents gradients from vanishing in very deep networks and explain why.
5. **(Calculation)** For a depthwise-separable convolution with $k=3$ and $C_{\text{in}}=C_{\text{out}}=128$, compute the FLOP ratio versus a standard $3\times3$ conv. Roughly how many times cheaper is it?
6. **(Code writing)** Write TorchVision `train_tf` and `val_tf` pipelines for HW1 Problem 14 that apply random-crop augmentation to training only, using ImageNet normalization stats. State precisely why the val pipeline must have no random transforms.
7. **(Code reading / concept)** In the fine-tuning snippet, `model.fc` is replaced and all other params get `requires_grad=False`. (a) Why must the optimizer be given `filter(lambda p: p.requires_grad, ...)`? (b) Does the new `fc` layer train, and why? (c) What breaks if you normalize CIFAR with its own mean/std instead of ImageNet's when using a pretrained backbone?
8. **(Conceptual)** Why does Optuna's TPE typically beat grid search, and why is random search often better than grid when only 2 of 6 hyperparameters actually matter? What does `trial.suggest_float('lr', 1e-5, 1e-1, log=True)` do differently from a linear range, and why does it matter for learning rate?

> [!note]+ 🔑 Answer key — open only after attempting all 8
> **1.** $H_{\text{out}} = \lfloor (32 + 2\cdot2 - 5)/2 \rfloor + 1 = \lfloor 31/2 \rfloor + 1 = 15 + 1 = 16$, so output is $(N, 64, 16, 16)$. Parameters: $C_{\text{out}}(C_{\text{in}}k^2 + 1) = 64\,(3\cdot25 + 1) = 64 \cdot 76 = 4864$.
> 
> **2.** (i) **Parameter sharing** — the same filter is applied at every position; (ii) **local/sparse connectivity** — each output sees only a $k\times k$ patch; (iii) **translation equivariance** — shifting the input shifts the output identically. Parameter sharing is what decouples the weight count $C_{\text{out}}(C_{\text{in}}k^2+1)$ from $H,W$: the filter is reused across all spatial positions, so more pixels don't mean more weights.
> 
> **3.** *Equivariance:* $\text{conv}(\text{shift}(x)) = \text{shift}(\text{conv}(x))$ — the feature map moves with the input. *Invariance:* the output is (approximately) unchanged by a shift. Convolution is equivariant; pooling adds invariance because taking the max over a $2\times2$ window returns the same value if the peak activation merely moves within that window — small translations don't change the pooled result.
> 
> **4.** Forward: $y = \mathcal{F}(x,\{W_i\}) + x$. Backward: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\left(I + \frac{\partial \mathcal{F}}{\partial x}\right)$. The $+I$ (identity) term is the gradient highway: even when $\partial\mathcal{F}/\partial x \approx 0$ (or the chain of many layers drives the learned term toward zero), the gradient still propagates to $x$ undiminished, so deep stacks remain trainable.
> 
> **5.** Ratio $= \frac{1}{C_{\text{out}}} + \frac{1}{k^2} = \frac{1}{128} + \frac{1}{9} \approx 0.0078 + 0.111 = 0.119$, i.e. about $8.4\times$ cheaper (dominated by the $1/k^2$ term).
> 
> **6.** Same two pipelines as Piece 4. `train_tf = Compose([RandomCrop(32, padding=4), ToTensor(), Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])`; `val_tf = Compose([ToTensor(), Normalize(same mean/std)])` — identical except val has **no** random ops. Val must be deterministic so evaluation measures true generalization, not a randomly-cropped variant; random transforms at eval would make the metric noisy and non-reproducible (and could inflate/deflate accuracy run-to-run).
> 
> **7.** (a) Frozen params have `requires_grad=False`; passing them to the optimizer is wasteful and some optimizers error / still update buffers — filtering keeps only trainable params. (b) Yes: replacing `model.fc` creates a fresh `nn.Linear` whose params default to `requires_grad=True`, so the head trains while the backbone stays fixed. (c) The pretrained backbone expects inputs whitened with ImageNet stats; feeding differently-normalized inputs shifts every activation off the distribution the frozen filters were trained on, degrading features badly.
> 
> **8.** TPE models past results (good-vs-bad densities $\ell(x), g(x)$) and samples where $\ell/g$ is large, spending trials where improvement is likely instead of blindly gridding. Random beats grid with few relevant hyperparameters because a grid spends most of its budget varying the *irrelevant* dimensions on a fixed lattice, whereas random search gives many distinct values of the *relevant* ones. `log=True` samples LR uniformly in log-space (so $10^{-5}$…$10^{-1}$ get equal attention across orders of magnitude); a linear range would put almost all samples near $10^{-1}$ and essentially never probe small LRs where good values often live.
> 
> **Scoring:** 1 pt each (half credit for right idea, wrong arithmetic). Score ≥ 6/8 → move on; below → tomorrow starts with a patch-up.