---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-04T22:54:00
Status: Done
Last updated time: 2026-07-08T11:36:00
Last edited by: Heaven Chen
Category:
  - ML
  - GPU
  - Finetuning
---
> Source: [PyTorch for Deep Learning Professional Certificate](https://www.deeplearning.ai/specializations/pytorch-for-deep-learning-professional-certificate) — Laurence Moroney, 3 courses × 4 modules, ~87h nominal. This companion gives the *why* and the math behind each module so the videos become confirmation, not discovery. Paired with the HW1 two-week plan (Jul 2–15). Cross-references: [PyTorch × GPU Training Internals](https://app.notion.com/p/391e445b30a48184a35fca32915f2b2c), [ONNX & ORT](https://app.notion.com/p/391e445b30a48171a735f4ba51833317), [Transformers in Practice](https://app.notion.com/p/391e445b30a4817dbfb5ef315a1f06ff).

# Course 1 — Fundamentals

## M1–M2: Getting started + the PyTorch workflow *(skim — you know this)*

The workflow is always: data → `nn.Module` → loss → optimizer → loop. The one deep idea: **autograd builds the graph dynamically during forward** — every tensor op records a `grad_fn`; `loss.backward()` walks that tape applying the chain rule

$\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial y}\,\frac{\partial y}{\partial \theta}$

as vector–Jacobian products (never materializing Jacobians). This is exactly the matrix-calculus machinery HW1 asks you to do by hand — the course is your numerical check.

## M3: Data management

`Dataset` (defines `__getitem__`/`__len__`) vs `DataLoader` (batching, shuffling, `num_workers` parallel loading, `pin_memory` for async H2D — see GPU doc §5). Key discipline: transforms/augmentation live in the Dataset; statistics (normalization constants) come from the *training* split only.

## M4: Core NN components

- Linear layer $y = Wx + b$; init matters: Kaiming $\mathrm{Var}(W) = 2/n_{in}$ keeps ReLU activations' variance stable across depth — the same argument HW1's depth questions probe.
- Losses: CrossEntropy = LogSoftmax + NLL, applied to *logits* (never softmax first — numerical stability via log-sum-exp).
- Optimizers: SGD w/ momentum $v \leftarrow \mu v + g;\; \theta \leftarrow \theta - \eta v$; Adam adds per-coordinate scaling by $\sqrt{\hat v}$.

```python
class MLP(nn.Module):
    def __init__(self, d_in, d_h, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(), nn.Linear(d_h, d_out))
    def forward(self, x): return self.net(x)   # logits

for x, y in loader:
    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
```

# Course 2 — Techniques & ecosystem

## M1: Hyperparameter optimization

Grid search wastes budget on unimportant dims; random search dominates when few hyperparameters matter. **Optuna** = Bayesian/TPE sampling + pruning (kill bad trials early using intermediate validation curves). LR is the hyperparameter: too high → divergence; schedulers (cosine, OneCycle, warmup) trade exploration vs convergence.

## M2: TorchVision

Pretrained backbones (ResNet, MobileNet) + fine-tuning recipes: freeze backbone → train head → optionally unfreeze with 10× lower LR. Why transfer works: early conv features (edges/textures) are task-generic; specificity grows with depth. Augmentation = label-preserving group actions that enlarge the effective dataset (ties to HW1's data-augmentation topic).

## M3: Text with Hugging Face

Embedding lineage: GloVe/FastText (static — one vector per type) → DistilBERT (contextual — vector depends on the sentence). Pipeline: tokenizer → `AutoModel` → pool → classifier head. Distillation objective (why DistilBERT is small-but-good): student matches teacher's *soft* distribution, $L = \mathrm{KL}(p_T \| p_S)$ at temperature $>1$, which transfers dark knowledge between classes.

## M4: Efficient training pipelines

Profile before optimizing: `torch.profiler` → is the gap data loading (CPU), kernel launches (small ops), or genuine compute? Fixes in order of cheapness: workers/pin-memory → AMP (bf16) → `torch.compile` → batch-size/accumulation. Full treatment: [PyTorch × GPU doc](https://app.notion.com/p/391e445b30a48184a35fca32915f2b2c) §3–8.

# Course 3 — Advanced architectures & deployment

## M1: Custom architectures

Composition patterns: residual blocks $y = x + F(x)$ (identity gradient path → trainable depth), multi-branch modules, custom `forward` control flow (fine in PyTorch — dynamic graphs). Parameter registration rules: submodules/`nn.Parameter` auto-register; plain tensors don't (classic bug).

## M2: Specialized vision

**Siamese networks**: twin encoders + metric loss. Contrastive loss $L = y\,d^2 + (1-y)\max(0, m-d)^2$; triplet loss $\max(0, d_{ap} - d_{an} + m)$ — learning an embedding space instead of classes (this is HW3's similarity-based representation learning, previewed). Saliency/CAM: $\partial y_c / \partial x$ or channel-weighted activation maps for interpretability.

## M3: Specialized NLP

Build the transformer from parts — multi-head attention, positional encoding, blocks (the implementation twin of [Transformers in Practice §Module 2](https://app.notion.com/p/391e445b30a4817dbfb5ef315a1f06ff)); closes with intro diffusion (preview of HW4/HW6 material — don't go deep here, it's covered later).

## M4: Deployment

- **ONNX export** → see [ONNX doc](https://app.notion.com/p/391e445b30a48171a735f4ba51833317) for the full pipeline and pitfalls.
- **Pruning**: magnitude pruning zeroes small $|w|$ — unstructured (sparse masks, needs sparse kernels to pay off) vs structured (drop channels — real speedup).
- **Quantization**: PTQ ($x \approx s(q-z)$, calibrate $s,z$) vs **QAT** — insert fake-quant in training so weights adapt to the rounding; straight-through estimator passes gradients through the non-differentiable round.
- **MLflow**: log params/metrics/artifacts per run; model registry = versioned deployable lineage.

# How this maps to the two-week HW1 plan

Jul 4–5: C1 (skim M1–M2, focus M3–M4) · Jul 6–7: C2 · Jul 9–11: C3 (M1, M2–M3, M4 on Jul 10 with ONNX doc). Quizzes/labs cleared on the Jul 12 buffer day.

# Cross-check questions

1. Why must CrossEntropyLoss take logits, and what does log-sum-exp stabilize? 2. Derive why Kaiming init uses $2/n_{in}$ for ReLU. 3. When does unstructured pruning give zero real-world speedup? 4. Why does QAT beat PTQ at INT4 but not usually at INT8?