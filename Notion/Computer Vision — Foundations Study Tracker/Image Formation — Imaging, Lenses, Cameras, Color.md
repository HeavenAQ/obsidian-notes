---
base: "[[Computer Vision — Foundations Study Tracker.base]]"
Key takeaways: "Imaging is a linear-optics pipeline: BRDF/Lambertian reflection (l_out = a·l_in·cosθ) feeds a pinhole/lens that imposes perspective projection (x=fX/Z), where lenses (Lensmaker's 1/a+1/b=1/f) trade the pinhole's brightness/sharpness tradeoff for a focusing surface shape linear in radius. Ch 7 generalizes any camera to a linear system l_s = A·l_w, inverted via regularized least squares l_w=(AᵀA+λI)⁻¹Aᵀl_s — unifying pinhole, edge, pinspeck, and corner cameras. Ch 8 shows color perception is a projection onto a 3D cone-response subspace (LMS = C_eye·t), so any 3-channel sensor/display matching that subspace up to a 3×3 matrix M=(CP)⁻¹ can reproduce color — explaining metamerism, CCMs, and why luminance/chrominance can be resolved at different spatial rates."
Day: 2
Status: In progress
Reading done: true
Chapters: Ch 5–8
Self-check done: true
Date: 2026-07-02
Part:
  - Image Formation
Questions / Follow-ups: ""
---
**Reading checklist**

- [ ] [5 Imaging](https://visionbook.mit.edu/imaging.html)
- [ ] [6 Lenses](https://visionbook.mit.edu/lenses.html)
- [ ] [7 Cameras as Linear Systems](https://visionbook.mit.edu/camera_as_linsys.html)
- [ ] [8 Color](https://visionbook.mit.edu/color.html)

## Notes

## Self-check

---

# 📄 Paper Deep-Dive: ControlNet → Cosmos3 Transfer Control

*Goal: use ControlNet (the image case) as the lens for understanding how Cosmos3's video transfer-control works. Read Part A first; Part B keeps pointing back to it.*

## Part A — ControlNet (Zhang, Rao & Agrawala, ICCV 2023, arXiv 2302.05543)

### A.1 The problem and the one-sentence idea

A large text-to-image diffusion model (Stable Diffusion) already knows how to render photorealistic images from a text prompt. We want to add a *spatial* control — "make the output follow **this** edge map/depth map/pose" — without retraining (or damaging) the billion-image backbone. ControlNet does this by **cloning** the backbone's encoder, training only the clone on the new condition, and wiring the clone back into the frozen backbone through **zero convolutions** so that training starts as an exact no-op and grows the control signal from zero.

### A.2 The full architecture, end-to-end

Stable Diffusion is a latent-diffusion U-Net $\epsilon_\theta$ operating on a VAE latent $z$. Its U-Net has an **encoder** (12 down-blocks), a **middle** block, and a **decoder** (12 up-blocks), with skip connections from the encoder to the decoder.

ControlNet leaves the entire SD U-Net **locked** (weights frozen — the "production-ready" model). It then makes a **trainable copy** of just the encoder blocks + middle block. The two branches are joined as follows:

1. The conditioning image $c_f$ (e.g. a Canny edge map) is first passed through a small encoder $\mathcal{E}$ of four conv layers that maps it from image resolution into the latent feature resolution, producing a feature map $c_f' = \mathcal{E}(c_f)$.
2. Each trainable encoder/middle block produces an output. That output is passed through a **zero convolution** $\mathcal{Z}$ and **added into the corresponding skip connection** that feeds the *locked* decoder. So the control branch injects its signal at every decoder resolution, exactly where SD's own encoder skips land.
3. $c_f'$ is added to the model's input latent and fed into the **trainable copy** of the encoder. (The add is done through a zero conv — see A.4.)

Because only the trainable copy + the two sets of zero convs + the tiny condition encoder are learned, the heavy pretrained decoder is never touched. This is what makes ControlNet cheap and stable to train (works with <50k images).

> **Mental picture:** the frozen U-Net is the renderer; the trainable encoder-copy is a "side reader" of the condition image; zero convs are dimmer switches, all starting at 0, that fade the side-reader's influence into the renderer's decoder.

### A.3 The diffusion objective (the math)

ControlNet is trained with the standard latent-diffusion denoising loss, extended with the task condition:

$\mathcal{L}=\mathbb{E}_{z_0,\,t,\,c_t,\,c_f,\,\epsilon\sim\mathcal{N}(0,I)}\Big[\big\lVert \epsilon-\epsilon_\theta(z_t,\,t,\,c_t,\,c_f)\big\rVert_2^2\Big]$

Symbol by symbol:

- $z_0$ — the clean VAE latent of a training image.
- $t\in\{1,\dots,T\}$ — the diffusion timestep, sampled uniformly; sets the noise level.
- $z_t=\sqrt{\bar\alpha_t}\,z_0+\sqrt{1-\bar\alpha_t}\,\epsilon$ — the **noised latent** at step $t$, where $\bar\alpha_t=\prod_{s\le t}\alpha_s$ is the cumulative noise schedule.
- $c_t$ — the **text prompt** condition (CLIP text embedding).
- $c_f$ — the **task-specific spatial condition** (edge / depth / pose / segmentation image). This is the *new* input ControlNet adds.
- $\epsilon$ — the ground-truth Gaussian noise that was added; the network's regression target.
- $\epsilon_\theta(\cdot)$ — the (now control-augmented) noise predictor.

The network is trained to predict the noise; the only difference from vanilla SD is that the prediction is additionally conditioned on $c_f$. Nothing about the loss form changes — that is the point.

### A.4 Zero convolutions (the core mechanism + its math)

A **zero convolution** $\mathcal{Z}(\cdot\,;\cdot)$ is a $1\times1$ convolution whose **weights and bias are both initialized to zero**. Let $\mathcal{F}(\cdot;\Theta)$ be a locked SD block, $x$ its input feature map, $c$ the control feature, $\Theta_{z1},\Theta_{z2}$ the parameters of two distinct zero convs, and $\Theta_c$ a *trainable copy* of $\Theta$. A ControlNet block computes:

$y_c=\mathcal{F}(x;\Theta)+\mathcal{Z}\big(\mathcal{F}(x+\mathcal{Z}(c;\Theta_{z1});\Theta_c);\Theta_{z2}\big)$

Reading it: the first term $\mathcal{F}(x;\Theta)$ is the untouched locked block. The second term routes the control feature $c$ through a zero conv, adds it to $x$, runs it through the *trainable copy* $\mathcal{F}(\cdot;\Theta_c)$, and routes the result through a second zero conv before adding it back.

At initialization both zero convs output $0$, so:

$y_c=\mathcal{F}(x;\Theta)+0=\mathcal{F}(x;\Theta)$

i.e. the augmented network is **bit-identical to the pretrained model** on step 0.

### A.5 WHY zero convolutions — and why a zero layer is still trainable

The worry: a randomly-initialized control branch would inject garbage into the deep features of a finely-tuned billion-image model on the very first gradient step, destroying it. Zero init means the injected signal is exactly $0$ at start, so **no harmful noise** reaches the backbone — the model only ever *improves away* from the pretrained behavior.

The apparent paradox: doesn't a layer initialized to all zeros have zero gradients and stay stuck forever? **No** — and here is the argument. Consider one zero conv as $y=\mathcal{Z}(x;\{W,b\})=Wx+b$ with $W=0,\,b=0$ at init. The gradients are:

$\frac{\partial y}{\partial W}=x,\qquad \frac{\partial y}{\partial b}=1,\qquad \frac{\partial y}{\partial x}=W=0$

- $\partial y/\partial W = x$: the gradient w.r.t. the **weights** equals the *input feature map* $x$, which is **non-zero** (it is a real activation from the network). So $W$ receives a non-zero gradient and **moves off zero** on the first step.
- $\partial y/\partial b = 1$: the bias gradient is always non-zero too.
- $\partial y/\partial x = W = 0$: only the gradient *flowing back through* the zero conv to its input is zero — and that is fine, because the trainable copy $\Theta_c$ also receives gradients directly through the *first* zero conv's weight path (same argument), so it is not starved.

After one update $W\neq0$, so on the next forward pass the output is no longer identically zero, and ordinary training proceeds. In short: **output is zero (safe), but the gradient w.r.t. weights is the non-zero input (trainable).** The layer "grows the parameters from zero" exactly as the abstract claims.

### A.6 Minimal PyTorch — `ZeroConv2d` and a `ControlNetBlock`

```python
import torch
import torch.nn as nn

class ZeroConv2d(nn.Module):
    """1x1 conv with weights AND bias initialized to zero.
    Output is identically 0 at init, but d(out)/d(weight) = input != 0,
    so the layer trains off zero on the first step."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class ControlNetBlock(nn.Module):
    """One ControlNet block implementing
       y_c = F(x; Theta) + Z2( F_copy( x + Z1(c) ) ).
    `locked_block` is a frozen SD block; `trainable_copy` is a
    deep-copied, gradient-enabled clone of it."""
    def __init__(self, locked_block, trainable_copy, channels):
        super().__init__()
        self.locked_block = locked_block          # frozen backbone block F(.;Theta)
        for p in self.locked_block.parameters():
            p.requires_grad_(False)
        self.trainable_copy = trainable_copy      # F(.;Theta_c), trainable
        self.zero_in  = ZeroConv2d(channels)      # Z(.;Theta_z1)
        self.zero_out = ZeroConv2d(channels)      # Z(.;Theta_z2)

    def forward(self, x, c):
        # locked branch — untouched pretrained computation
        y_locked = self.locked_block(x)
        # control branch — condition injected via zero conv, run through copy,
        # then faded back in through the second zero conv
        h = self.trainable_copy(x + self.zero_in(c))
        return y_locked + self.zero_out(h)         # additive injection into the skip
```

At step 0, `zero_in` and `zero_out` output 0, so `forward` returns exactly `locked_block(x)` — the pretrained model. Gradients still reach `trainable_copy`, `zero_in`, and `zero_out` because each zero conv's weight-gradient is its (non-zero) input.

---

## Part B — Cosmos3 Transfer Control (the payoff)

*Source: NVIDIA Cosmos cookbook — *`*cookbooks/cosmos3/generator/transfer/run_video_transfer_with_cosmos_framework.ipynb*`*, plus the Cosmos / Cosmos-Transfer world-foundation-model line of work.*

### B.1 What Cosmos3 transfer does

Given a **text caption** (`prompt.json`) plus a **spatial control video**, Cosmos3 transfer generates a new video clip whose spatiotemporal structure follows that control while its *appearance* is re-rendered from the prompt. Supported control modalities (one `control_<x>.mp4` per run):

- **edge** — Canny edge map
- **blur** — blurred reference
- **depth** — depth map
- **seg** — segmentation map
- **wsm** — world-scenario map

This is exactly ControlNet's "spatial conditioning of a pretrained diffusion model," **lifted from a single image to a video world-foundation model** — a diffusion/flow transformer (DiT) over **spatiotemporal latents** instead of a U-Net over a single image latent.

### B.2 The analogy, made explicit (ControlNet ⇄ Cosmos3)

| ControlNet (image) | Cosmos3 transfer (video) |
| --- | --- |
| condition image $c_f$ (edge/depth/seg) | per-frame **control video** $c_f^{(1:F)}$ (edge/depth/seg/blur/wsm) |
| locked SD U-Net $\epsilon_\theta$ | pretrained **video diffusion DiT backbone** (Cosmos world model) |
| trainable encoder copy + zero-conv injection | **control branch** fused into the backbone's transformer blocks |
| denoise a single image latent $z_t$ | denoise a **spatiotemporal latent** $z_t^{(1:F)}$ (tokenizer-compressed video) |
| text prompt $c_t$ | caption from `prompt.json` |
| single guidance scale | **two** guidance scales: `guidance` (text) and `control_guidance` (structure) |

The mechanism is the same idea: keep a powerful pretrained generative prior frozen (or mostly so), attach a branch that reads the structural condition, and add its signal back into the backbone — starting from a near-no-op so the prior is not damaged.

### B.3 Architecture & pipeline (from the notebook)

Two model sizes, one model-agnostic spec:

| Model | Launcher | Parallelism preset | GPUs |
| --- | --- | --- | --- |
| **Cosmos3-Nano** | `python` (single GPU) | `latency` | 1 |
| **Cosmos3-Super** (~32B) | `torchrun` (multi-GPU) | `throughput` | 4+ |

Both are invoked through the same entrypoint:

```bash
# Nano (single GPU)
python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i specs/<control>.json \
  -o <out_dir> \
  --checkpoint-path Cosmos3-Nano \
  --seed 2026

# Super (multi-GPU via torchrun, model + sequence sharded across GPUs)
torchrun --nproc-per-node=<NUM_GPUS> \
  --master-addr=127.0.0.1 --master-port=<PORT> \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i specs/<control>.json \
  -o <out_dir> \
  --checkpoint-path Cosmos3-Super \
  --seed 2026
```

The checked-in `specs/<control>.json` are **model-agnostic** (the same spec runs on Nano and Super). Key spec fields shown in the notebook's summary helper:

- `num_frames` — number of frames in the generated clip (temporal length of the spatiotemporal latent).
- `fps` — playback frame rate; with `num_frames` it sets clip duration and motion cadence.
- `guidance` — the **text** classifier-free-guidance scale $s_t$ (prompt strength).
- `control_guidance` — the **structure** guidance scale $s_c$ (how hard the output is pulled toward the control video).
- `<control>.control_path` — path to the control video (e.g. `control_edge.mp4`).

Output lands at `<output_root>/<model>/transfer_<control>/vision.mp4`.

### B.4 The math — dual classifier-free guidance, and WHY two scales

Recall single-condition **classifier-free guidance** (CFG): a diffusion model is trained both conditionally and unconditionally (condition dropped to $\varnothing$), and at sampling the prediction is extrapolated away from the unconditional one:

$\hat{\epsilon}=\epsilon_\varnothing+s\,(\epsilon_{c}-\epsilon_\varnothing)$

where $\epsilon_\varnothing=\epsilon_\theta(z_t,t,\varnothing)$ and $\epsilon_c=\epsilon_\theta(z_t,t,c)$, and $s\ge1$ sharpens adherence to the condition $c$.

Cosmos transfer has **two** conditions — the text caption $c_t$ and the control video $c_f$ — so it exposes **two** guidance scales. The natural multi-condition extension of CFG is:

$\hat{\epsilon}=\epsilon_\varnothing+s_t\,(\epsilon_{c_t}-\epsilon_\varnothing)+s_c\,(\epsilon_{c_f}-\epsilon_\varnothing)$

Symbol by symbol:

- $z_t$ — the noised spatiotemporal latent at diffusion step $t$ (the whole clip's latent).
- $\epsilon_\varnothing=\epsilon_\theta(z_t,t,\varnothing,\varnothing)$ — prediction with **neither** condition.
- $\epsilon_{c_t}=\epsilon_\theta(z_t,t,c_t,\varnothing)$ — prediction with the **text** caption only.
- $\epsilon_{c_f}=\epsilon_\theta(z_t,t,\varnothing,c_f)$ — prediction with the **control video** only.
- $s_t$ — text guidance scale = the spec's `guidance`.
- $s_c$ — control guidance scale = the spec's `control_guidance`.
- $\hat\epsilon$ — the guided noise estimate actually used by the sampler this step.

(Implementations vary in whether the two deltas are measured against a fully-unconditional baseline or composed sequentially; the key structural fact is that **two independent strengths multiply two independent condition deltas**.)

**WHY decouple them.** The whole job of transfer is "**follow the structure but re-render the appearance**." Those are two different knobs:

- $s_c$ (`control_guidance`) high → output rigidly obeys the edge/depth/seg geometry of the control video.
- $s_t$ (`guidance`) high → output strongly reflects the caption's described appearance/style.

If there were a single scale you could not, say, lock the depth layout while freely restyling the scene from the prompt. Decoupling lets you trade structural fidelity against prompt-driven appearance independently. This is the direct video analogue of ControlNet's **prompt-vs-condition balance** (where users likewise trade the text prompt against the strength of the spatial condition).

### B.5 PyTorch / pseudocode — dual-guidance denoising loop + a ControlNet-style branch on a video DiT block

A minimal sampler showing how the control latents are encoded once and injected each diffusion step, with the two guidance scales:

```python
import torch

@torch.no_grad()
def cosmos_transfer_sample(dit, tokenizer, scheduler,
                           caption_emb, control_video,
                           num_frames, s_t, s_c,
                           shape, device, seed=2026):
    """
    dit          : pretrained video diffusion transformer (the frozen backbone)
    tokenizer    : video VAE/tokenizer -> compresses pixels to spatiotemporal latents
    caption_emb  : text embedding c_t  (from prompt.json)
    control_video: per-frame control tensor c_f  (edge/depth/seg/blur/wsm)
    s_t          : text guidance  (spec 'guidance')
    s_c          : control guidance (spec 'control_guidance')
    """
    g = torch.Generator(device).manual_seed(seed)

    # 1) Encode the control video ONCE into the latent space the DiT denoises in.
    c_f = tokenizer.encode(control_video)        # spatiotemporal control latent
    null_txt  = torch.zeros_like(caption_emb)    # unconditional text  (varnothing)
    null_ctrl = torch.zeros_like(c_f)            # unconditional control (varnothing)

    # 2) Start from pure noise over the full clip's latent.
    z_t = torch.randn(shape, generator=g, device=device)  # (B, C, num_frames, H, W)

    for t in scheduler.timesteps:
        # Three forward passes = three conditionings (batched in practice).
        eps_uncond = dit(z_t, t, txt=null_txt,    ctrl=null_ctrl)  # epsilon_varnothing
        eps_text   = dit(z_t, t, txt=caption_emb, ctrl=null_ctrl)  # epsilon_{c_t}
        eps_ctrl   = dit(z_t, t, txt=null_txt,    ctrl=c_f)        # epsilon_{c_f}

        # Dual classifier-free guidance:
        # eps_hat = eps_0 + s_t (eps_ct - eps_0) + s_c (eps_cf - eps_0)
        eps_hat = (eps_uncond
                   + s_t * (eps_text - eps_uncond)
                   + s_c * (eps_ctrl - eps_uncond))

        z_t = scheduler.step(eps_hat, t, z_t).prev_sample

    # 3) Decode latents back to a video clip.
    return tokenizer.decode(z_t)   # -> vision.mp4 frames
```

How a **ControlNet-style branch attaches to a video DiT block**, reusing the `ZeroConv2d` idea from Part A (1D/zero-init projection over tokens instead of 2D conv over pixels):

```python
import torch
import torch.nn as nn

class ZeroLinear(nn.Module):
    """DiT analogue of ZeroConv2d: zero-init projection over the token dim.
    Output 0 at init -> backbone unchanged; weight-grad = input != 0 -> trainable."""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    def forward(self, x):
        return self.proj(x)


class ControlVideoDiTBlock(nn.Module):
    """Frozen backbone DiT block + trainable control copy, fused with zero-init
    projections. x: backbone tokens, c: control tokens (both [B, T*H*W, dim])."""
    def __init__(self, locked_block, trainable_copy, dim):
        super().__init__()
        self.locked_block = locked_block            # frozen video-DiT block
        for p in self.locked_block.parameters():
            p.requires_grad_(False)
        self.trainable_copy = trainable_copy        # trainable clone
        self.zero_in  = ZeroLinear(dim)             # gate control into the copy
        self.zero_out = ZeroLinear(dim)             # fade control back into backbone

    def forward(self, x, c):
        y = self.locked_block(x)                    # untouched pretrained path
        h = self.trainable_copy(x + self.zero_in(c))
        return y + self.zero_out(h)                 # additive injection, 0 at init
```

The structure is identical to `ControlNetBlock`; only the operator (linear/attention over spatiotemporal tokens) and the data (a video latent) change. That is the whole "lift from image to video."

---

## ControlNet → Cosmos3, in one breath

ControlNet adds spatial control to a **frozen image diffusion U-Net** by training a **copy of its encoder** on the condition image $c_f$ and injecting that copy's output back through **zero convolutions** — layers that output 0 at init (so the pretrained model is untouched) yet have non-zero weight-gradients (so they train off zero), optimizing the unchanged denoising loss $\mathbb{E}\,\lVert\epsilon-\epsilon_\theta(z_t,t,c_t,c_f)\rVert^2$. **Cosmos3 transfer is the same recipe lifted to video**: a frozen **video diffusion DiT** over spatiotemporal latents, a per-frame **control video** in place of $c_f$, a control branch fused into the backbone the same zero-init way, and **two** classifier-free-guidance scales — `guidance` ($s_t$, text) and `control_guidance` ($s_c$, structure) — so you can *follow the structure but re-render the appearance*: $\hat\epsilon=\epsilon_\varnothing+s_t(\epsilon_{c_t}-\epsilon_\varnothing)+s_c(\epsilon_{c_f}-\epsilon_\varnothing)$.

## Self-check questions

4. In the zero-convolution forward $y_c=\mathcal{F}(x;\Theta)+\mathcal{Z}(\mathcal{F}(x+\mathcal{Z}(c;\Theta_{z1});\Theta_c);\Theta_{z2})$, why is $y_c$ exactly equal to the pretrained model's output at training step 0 — and why does the control branch nevertheless receive a non-zero gradient? (Hint: compare $\partial y/\partial W$ with $\partial y/\partial x$ for a zero conv.)
5. Map each ControlNet component to its Cosmos3 counterpart: condition image $c_f$, locked SD U-Net, trainable encoder copy + zero-conv injection, single image latent $z_t$. What replaces each one in the video setting?
6. Cosmos exposes two guidance scales. Write the dual-CFG formula and say which spec field is $s_t$ and which is $s_c$. What visually happens if you push $s_c$ (`control_guidance`) high while keeping $s_t$ (`guidance`) low — and vice versa?
7. Why is decoupling `control_guidance` from `guidance` essential to the "follow the structure but re-render the appearance" goal? Give a concrete failure you'd hit with only one combined scale.
8. In the dual-guidance sampling loop, why is the control video encoded **once** (outside the timestep loop) while three DiT forward passes happen **inside** the loop each step? What are those three passes conditioned on?

## Chapter 5 — Imaging

### 5.1–5.2 Light Rays, BRDFs, and Reflection Models

**Intuition.** A camera does not see "objects" — it sees a bundle of light rays converging on a sensor. To reason about what an image encodes, we need a model of (a) how light leaves a source, (b) how it interacts with a surface, and (c) how the resulting rays are captured. This chapter builds that pipeline from first principles: radiometry → reflection → pinhole projection.

A **light ray** is parameterized by position, direction, wavelength $\lambda$, and intensity. When light strikes a surface with normal $\mathbf{n}$, arriving from direction $\mathbf{p}$ and leaving toward $\mathbf{q}$, the surface's **BRDF** (bidirectional reflectance distribution function) $F$ governs how much of the incoming power $\ell_{\text{in}}$ is redirected outward:

$$
\ell_{\text{out}} = F(\ell_{\text{in}}, \mathbf{n}, \lambda, \mathbf{p}, \mathbf{q})
$$

*Why this form:* the BRDF is formulated as a function of exactly these five arguments because reflection is fundamentally a local, geometric phenomenon — the outgoing radiance at a point depends only on the incident radiance and the relative angles between the surface normal and the two ray directions, plus the wavelength (color) of light. This generality lets the same equation describe mirrors, matte paint, or skin — the difference is purely in the shape of $F$.

**Lambertian model.** Most surfaces are well-approximated as *diffuse* — they scatter light equally in all outgoing directions, so $F$ loses its dependence on $\mathbf{q}$:

$$
\ell_{\text{out}} = F_L(\ell_{\text{in}}(\lambda), \mathbf{n}, \mathbf{p}) = a\,\ell_{\text{in}}(\lambda)\,(\mathbf{n}\cdot\mathbf{p})
$$

Here $a$ is the surface **albedo** (reflectance, $0\le a\le1$), and $\mathbf{n}\cdot\mathbf{p} = \cos\theta$ is the cosine of the angle between the surface normal and the incoming ray. *Why the cosine term:* a bundle of parallel rays hitting a surface at a grazing angle spreads its energy over a larger surface area than the same bundle hitting head-on, so the energy density (and hence perceived brightness) falls off proportionally to the projected area — exactly $\cos\theta$. This is Lambert's cosine law, and it is why a sphere under one light source looks bright at the pole facing the light and dark near the terminator.

**Phong (specular) model.** Real surfaces mix diffuse and specular components. The specular term concentrates reflected energy near the mirror-reflection direction $\mathbf{r} = 2(\mathbf{p}\cdot\mathbf{n})\mathbf{n} - \mathbf{p}$:

$$
\ell_{\text{Phong spec}} = k_s(\mathbf{r}\cdot\mathbf{q})^{\alpha}\,\ell_{\text{in}}
$$

*Why this form:* $\mathbf{r}$ is the direction a perfect mirror would send the ray; raising $(\mathbf{r}\cdot\mathbf{q})$ to a power $\alpha$ is a cheap, non-physical but perceptually effective way to create a highlight that decays smoothly as the viewing direction $\mathbf{q}$ moves away from $\mathbf{r}$ — larger $\alpha$ means a tighter, glossier highlight (small $\alpha$ mimics rougher, duller surfaces).

```python
import torch

def lambertian_shade(normals, light_dir, albedo, light_intensity=1.0):
    """
    normals:     (..., 3) unit surface normals n
    light_dir:   (3,) unit vector pointing FROM surface TOWARD the light (p)
    albedo:      (...) or scalar, surface reflectance a
    Implements l_out = a * l_in * (n . p), clamped at 0 (no negative light).
    """
    cos_theta = torch.clamp((normals * light_dir).sum(-1), min=0.0)
    return albedo * light_intensity * cos_theta

def phong_specular(normals, light_dir, view_dir, ks=1.0, alpha=32.0, light_intensity=1.0):
    """
    r = 2(p.n)n - p   (mirror-reflection direction of the incoming ray)
    spec = ks * (r . q)^alpha * l_in
    """
    p_dot_n = (light_dir * normals).sum(-1, keepdim=True)
    r = 2 * p_dot_n * normals - light_dir
    cos_alpha = torch.clamp((r * view_dir).sum(-1), min=0.0)
    return ks * cos_alpha.pow(alpha) * light_intensity
```

*Why it matters / connections:* every neural rendering method (NeRF, Gaussian splatting — Ch 45) that models "view-dependent appearance" is implicitly learning a data-driven BRDF; the Lambertian/Phong split is the classical decomposition that inverse-rendering networks still use as an inductive bias (diffuse albedo branch + specular/roughness branch).

### 5.3 The Pinhole Camera and Perspective Projection

**Why pinhole cameras form images at all:** a bare wall integrates light from every direction and every source, so its reflected radiance is a hopeless average — no image appears. A pinhole restricts each point on a projection surface to receiving light from exactly *one* direction in the world (the straight line through the hole), which is precisely what "forming an image" means: establishing a one-to-one map from world directions to sensor positions.

With the pinhole at the origin, sensor plane at distance $f$ behind it (or, equivalently, a virtual, unflipped image plane at distance $f$ in front), similar triangles give the **perspective projection equations**:

$$
x = f\frac{X}{Z}, \qquad y = f\frac{Y}{Z}
$$

where $(X,Y,Z)$ is a 3D world point and $(x,y)$ its projection. *Why division by *$Z$*:* this is not a modeling choice but a geometric necessity — it falls directly out of similar-triangle proportionality between the ray's rise-over-run in the world and in the image plane. It is this $1/Z$ term that makes distant objects appear smaller and is the root cause of most non-trivial 3D vision problems (stereo, SfM, monocular depth — Chapters 40–44).

Converting from camera-plane coordinates $(x,y)$ to pixel/image coordinates $(n,m)$ is an affine map:

$$
n = -a\,x + n_0, \qquad m = a\,y + m_0
$$

with $a$ a pixel-per-unit-length scale and $(n_0,m_0)$ the principal point (where the optical axis hits the sensor). This is the seed of the camera intrinsics matrix formalized in Chapter 39.

### 5.3.2 Orthographic Projection

An alternative projection model discards the $1/Z$ term entirely:

$$
x = kX, \qquad y = kY
$$

*Why this is a valid, useful approximation:* as an object's distance from the camera grows very large relative to its own size (the telephoto/"soda-straw camera" regime), the *relative* variation in $Z$ across the object becomes negligible, so $1/Z approx 1/Z_{text{avg}} = $ constant $ = k$. Orthographic projection is exact for a physical apparatus of parallel light-restricting tubes (the straw camera) and is the correct model whenever perspective effects are negligible — this is the same projection used in the toy vision system of Chapter 2, and shows up again as the "weak perspective" approximation in structure-from-motion.

```python
import torch

def perspective_project(P_world, f):
    """P_world: (...,3) = (X,Y,Z). Returns (...,2) = (x,y)."""
    X, Y, Z = P_world.unbind(-1)
    x = f * X / Z
    y = f * Y / Z
    return torch.stack([x, y], dim=-1)

def orthographic_project(P_world, k):
    """P_world: (...,3). Returns (...,2), ignoring Z entirely."""
    X, Y, _ = P_world.unbind(-1)
    return torch.stack([k * X, k * Y], dim=-1)
```

**Self-check (Ch 5):**

9. Why does a Lambertian surface's brightness depend on the incoming ray direction $\mathbf{p}$ but not the outgoing direction $\mathbf{q}$, while a specular surface depends on both?
10. Derive the perspective-projection equation $x = fX/Z$ from similar triangles, starting from a diagram with the pinhole at the origin.
11. Under what physical camera configuration is orthographic projection an *exact* (not approximate) model, and why?
12. Why do walls in a room often appear to have a faint blue tint near the floor?

---

## Chapter 6 — Lenses

### 6.1 The Brightness/Sharpness Tradeoff

A pinhole must be tiny to keep images sharp, but a tiny aperture lets in very little light (long exposures, noisy/dark images). Widening the pinhole brightens the image but blurs it, because each sensor position now integrates light arriving from a *range* of world directions rather than exactly one. A **lens** breaks this tradeoff: it can gather light over a large aperture while still bending all the rays from a given surface point back to a single sensor point.

### 6.2 Snell's Law and the Lensmaker's Formula

Light bends at a material interface according to **Snell's Law**:

$$
n_1 \sin\theta_1 = n_2\sin\theta_2
$$

where $n_1, n_2$ are the indices of refraction (ratio of light's speed in vacuum to its speed in the medium) and $\theta_1,\theta_2$ are angles to the surface normal. *Why it has this form:* it follows from requiring the light's wavefronts to stay continuous across the boundary — the projection of the wavelength along the interface must match on both sides, and since wavelength scales as $1/n$, matching projections forces $n_1\sin\theta_1 = n_2 \sin\theta_2$.

Under the **paraxial approximation** ($\sin\theta\approx\theta$ for small angles) and the **thin-lens approximation** (lens thickness $\ll$ other distances), tracing a ray through both lens surfaces (front and back) and requiring that every ray leaving a point at distance $a$ on one side converges to a point at distance $b$ on the other side yields a condition on the lens-surface slope $\theta_S$ at radius $c$ from the optical axis:

$$
\theta_S = \frac{c}{2(n-1)}\left(\frac{1}{a}+\frac{1}{b}\right)
$$

*Why linearity in *$c$* matters:* a surface shape whose local slope is proportional to distance from center is, to first order, a sphere (or parabola) — this is why lenses are ground spherical: it is the simplest shape achieving the required focusing property under the thin/paraxial approximations. For a sphere of radius $R$, $\theta_S = c/R$, and substituting gives the **Lensmaker's Formula**:

$$
\frac{1}{a} + \frac{1}{b} = \frac{1}{f}, \qquad f = \frac{R}{2(n-1)}
$$

$f$ is the **focal length** — the distance at which parallel incoming rays ($a\to\infty$) converge. For a lens with two curved surfaces of radii $R_1, R_2$:

$$
\frac{1}{f} = (n-1)\left(\frac{1}{R_1}+\frac{1}{R_2}\right)
$$

**Ray-tracing rules that follow from the lensmaker's formula:**

13. Any ray through a conjugate point $a$ that passes through the lens exits through the conjugate point $b$.
14. Parallel rays converge at the focal point, distance $f$ behind the lens.
15. A ray through the lens center is undeviated (so a thin lens still obeys perspective projection — property 3 in the text).
16. Magnification $= a/b$.

```python
import torch

def lensmaker_focal_length(R1, R2, n):
    """1/f = (n-1)(1/R1 + 1/R2). Sign convention: R>0 for convex-toward-object surfaces."""
    return 1.0 / ((n - 1.0) * (1.0 / R1 + 1.0 / R2))

def conjugate_distance(a, f):
    """Given object distance a and focal length f, solve 1/a + 1/b = 1/f for b."""
    # 1/b = 1/f - 1/a  =>  b = 1 / (1/f - 1/a)
    return 1.0 / (1.0 / f - 1.0 / a)
```

### 6.3.1 Depth of Field

Points off the focal plane image to a **circle of confusion** rather than a point. Defining the **f-number** $N = f/A$ (focal length over aperture diameter), circle-of-confusion diameter $C$, and focus distance $U \gg f$, similar-triangle geometry (Levoy's derivation) gives:

$$
D = \frac{2NCU^2f^2}{f^4 - N^2C^2U^2} \;\approx\; \frac{2NCU^2}{f^2} \quad \text{(when } C \ll f/N\text{)}
$$

*Why *$D \propto N$*:* a smaller aperture (larger $N$) means each off-focal-plane point spreads its rays over a narrower cone, so it takes a *larger* range of depths before the blur circle exceeds tolerance $C$ — hence smaller apertures (larger f-numbers) give more depth of field, at the direct cost of less light reaching the sensor. This is a physical tradeoff, not a design choice: gathering more light (wide aperture) necessarily samples a wider cone of rays per scene point, which necessarily blurs out-of-focus points faster.

```python
import torch

def depth_of_field(N, C, U, f):
    """Approximate DOF: D ≈ 2*N*C*U^2 / f^2 (valid when C << f/N)."""
    return 2 * N * C * U**2 / f**2

def circle_of_confusion_diameter(delta_a, a, b, f, N):
    """
    Exact-ish geometric estimate of blur-circle diameter for an object shifted
    by delta_a from the in-focus distance a, given sensor at conjugate distance b.
    Uses the lensmaker's formula to find the new focus point b', then geometric
    similar triangles to project the defocus onto the sensor.
    """
    a_shifted = a + delta_a
    b_prime = 1.0 / (1.0 / f - 1.0 / a_shifted)
    aperture = f / N
    C = aperture * torch.abs(b_prime - b) / b_prime
    return C
```

### 6.3.2–6.3.3 Concave Lenses and Telescopes

A **concave** lens has the opposite surface curvature and produces a *negative* focal length: parallel rays diverge as if emanating from a virtual focal point on the *same* side as the incoming light. Pairing a long-focal-length convex lens ($f_1$) with a short-focal-length concave lens ($f_2$), sharing a common focal point, compresses the angular spread of incoming parallel rays — this is the **Galilean telescope**, with magnification:

$$
M = \frac{\delta_o}{\delta_i} = \frac{f_1}{f_2}
$$

*Why magnification is a ratio of focal lengths:* both lenses convert an angular deviation to/from a fixed lateral offset $d$ at the shared focal point ($\delta_i f_1 = d = \delta_o f_2$); eliminating $d$ leaves a pure ratio, independent of the physical separation of the lenses, which is why a longer objective lens (larger $f_1$) directly buys more magnification for a fixed eyepiece.

**Self-check (Ch 6):**

17. Why must a lens surface's slope be *linear* in distance from the optical axis for focusing to work, and why does that imply a spherical or parabolic shape?
18. Write the lensmaker's formula and explain physically why $f$ depends on both $n$ (material) and $R$ (shape).
19. If you halve the aperture diameter (double the f-number $N$) at fixed $f, C, U$, what happens to the depth of field, and why?
20. Explain, using the ray-tracing property that "rays through the lens center are undeviated," why lenses obey perspective projection just like pinholes.

---

## Chapter 7 — Cameras as Linear Systems

### 7.2–7.3 The Linear Imaging Model

**Motivating idea:** not every camera looks like a pinhole. Many computational and unconventional imagers (coded apertures, edge cameras, pinspeck cameras, corner cameras) still respond *linearly* to incident light — each sensor reading is some fixed weighted sum of world-light values. That single assumption (linearity) is powerful enough to unify the analysis of an enormous variety of imaging systems using nothing more than matrix algebra.

In a discretized 1D ("flatland") world, let $\boldsymbol\ell_{\text{w}}$ be a vector of world light intensities and $\boldsymbol\ell_{\text{s}}$ the vector of sensor measurements. Linearity means there exists a matrix $\mathbf{A}$ (the **imaging matrix**, fully determined by the physical camera geometry) such that:

$$
\boldsymbol\ell_{\text{s}} = \mathbf{A}\boldsymbol\ell_{\text{w}}
$$

For an ideal small-pinhole camera, $\mathbf{A} = \mathbf{I}$ (each sensor sees exactly one world position). For a wide-aperture pinhole, $\mathbf{A}$ is a banded matrix (each sensor sums several adjacent world positions) — brighter but blurrier, exactly matching the qualitative brightness/sharpness tradeoff from Chapter 6.

**Recovering the scene: regularized least squares.** Given noisy measurements $\boldsymbol\ell_{\text{s}}$, we want to invert $\mathbf{A}$. Directly inverting is often ill-conditioned (small measurement noise creates huge reconstruction errors) or impossible ($\mathbf{A}$ non-square/non-invertible), so we minimize a regularized objective:

$$
E = \lVert \boldsymbol\ell_{\text{s}} - \mathbf{A}\boldsymbol\ell_{\text{w}}\rVert^2 + \lambda\lVert\boldsymbol\ell_{\text{w}}\rVert^2
$$

*Why add the *$\lambda\lVert\boldsymbol\ell_{\text{w}}\rVert^2$* term:* pure least-squares fitting ($\lambda=0$) will chase noise whenever $\mathbf{A}^T\mathbf{A}$ is poorly conditioned — small singular values get amplified enormously upon inversion. Penalizing the norm of the solution is a Bayesian prior (Gaussian prior on $\boldsymbol\ell_{\text{w}}$, equivalent to MAP estimation) that trades a small amount of bias for a large reduction in variance, i.e., classic ridge regression / Tikhonov regularization. Setting the gradient to zero:

$$
0 = \mathbf{A}^T\mathbf{A}\boldsymbol\ell_{\text{w}} - \mathbf{A}^T\boldsymbol\ell_{\text{s}} + \lambda\boldsymbol\ell_{\text{w}}
\quad\Longrightarrow\quad
\boldsymbol\ell_{\text{w}} = \underbrace{(\mathbf{A}^T\mathbf{A}+\lambda\mathbf{I})^{-1}\mathbf{A}^T}_{\mathbf{B}}\,\boldsymbol\ell_{\text{s}}
$$

$\mathbf{B}$ is the **regularized (pseudo)inverse**. Note this is exactly the linear-regression / ridge-regression normal-equations solution — the same math that underlies linear layers trained with weight decay, and it is a direct preview of the least-squares machinery used throughout the book (e.g., Blocks World depth recovery in Ch 2, stereo in Ch 40).

```python
import torch

def regularized_inverse_reconstruct(A, l_s, lam=1e-2):
    """
    Solve  l_w = argmin ||l_s - A l_w||^2 + lam ||l_w||^2
    closed form: l_w = (A^T A + lam I)^-1 A^T l_s
    A:   (m, n) imaging matrix
    l_s: (m,) or (m, batch) sensor measurements
    """
    AtA = A.T @ A
    reg = lam * torch.eye(AtA.shape[0], dtype=A.dtype, device=A.device)
    B = torch.linalg.solve(AtA + reg, A.T)   # (n, m), the regularized inverse
    return B @ l_s

def make_pinhole_matrix(n):
    return torch.eye(n)

def make_wide_pinhole_matrix(n, width=2):
    """Each sensor sums `width` consecutive world positions (banded matrix)."""
    A = torch.zeros(n, n)
    for i in range(n):
        for k in range(width):
            if i + k < n:
                A[i, i + k] = 1.0
    return A

def make_edge_camera_matrix(n):
    """Upper-triangular: each sensor integrates all world values from itself onward."""
    return torch.triu(torch.ones(n, n))
```

### 7.4 General Imagers: Edge, Pinspeck, and Corner Cameras

The **edge camera** (light blocked on only one side) has imaging matrix $\mathbf{A}$ upper-triangular of ones; its true inverse is a first-difference (derivative) operator, so an edge camera's output is literally the *cumulative integral* of the scene, and recovering the scene means differentiating the measurements — explaining why $\mathbf{A}^{-1}$ contains large alternating $\pm1$ entries that amplify high-frequency noise (motivating the regularized inverse, whose smoother version acts like a blurred derivative — a preview of Ch 18).

The **pinspeck camera** (an occluder rather than an aperture) has $\mathbf{A}_{\text{pinspeck}} = \mathbf{1} - \mathbf{A}_{\text{pinhole}}$ — it measures the *shadow* an occluder casts, and the missing signal is the mirror image (in sign) of what a corresponding wide-pinhole camera would see.

The **corner camera** generalizes this to real 3D scenes: a vertical edge occludes part of a scene from a hidden observer, but very faint, exploitable intensity gradients still leak onto the visible ground plane. Writing $S(\phi,\xi)$ for the hidden scene's radiance and integrating the Lambertian $\cos\phi$ weighting over vertical angle $\phi$ gives a 1D angular signal:

$$
\ell_{\text{w}}(\xi) = \int_0^{\pi}\cos(\phi)\,S(\phi,\xi)\,d\phi,
\qquad
\boldsymbol\ell_{\text{ground}}(r,\theta) \approx \int_0^{\theta}\ell_{\text{w}}(\xi)\,d\xi
$$

*Why it matters / connections:* this is the same "camera = linear operator, reconstruct via regularized inverse" pattern used throughout modern computational imaging — coded-aperture imaging, lensless imaging, single-pixel cameras, and even the "camera calibration as linear system" formalism revisited with homogeneous coordinates in Chapter 38–39. The corner-camera result — that hidden scenes leak through a spatial *derivative* of ambient ground illumination — is a striking real-world instance of treating an accidental physical setup as a (very poorly conditioned) linear imaging system.

**Self-check (Ch 7):**

21. Why is $\mathbf{A} = \mathbf{I}$ for a small pinhole camera but a banded matrix for a wide-aperture pinhole?
22. Derive the regularized-inverse solution $\boldsymbol\ell_{\text{w}} = (\mathbf{A}^T\mathbf{A}+\lambda\mathbf{I})^{-1}\mathbf{A}^T\boldsymbol\ell_{\text{s}}$ from the objective $E$, and explain the statistical interpretation of $\lambda$.
23. Why does the edge camera's imaging matrix look like a cumulative sum, and why is its (unregularized) inverse noisy?
24. In the corner camera, why does recovering the hidden scene amount to taking a derivative with respect to angle $\theta$?

---

## Chapter 8 — Color

### 8.2 Color Physics

Newton's prism experiments established that white light is a **mixture** of monochromatic components ("elemental" colors that don't split further), and that a **power spectrum** $\ell(\lambda)$ — intensity as a function of wavelength over roughly 400–700 nm — determines (within a fixed viewing context) the perceived color.

**Diffuse surface reflection** multiplies the incident spectrum wavelength-by-wavelength by a **reflectance spectrum** $s(\lambda)$:

$$
r(\lambda) = k\,\ell_{\text{in}}(\lambda)\,s(\lambda)
$$

*Why multiplicative, not additive:* a surface's reflectance at wavelength $\lambda$ acts as a per-wavelength attenuation factor (fraction of incident power reflected), so the physical process is a per-channel gain, i.e., multiplication — exactly analogous to how a color filter or an attenuating medium modifies a spectrum. This single equation is the root cause of the color-constancy problem: a vision system observes only the *product* $r(\lambda)$ and must disentangle illumination $\ell_{\text{in}}(\lambda)$ from material $s(\lambda)$ — the same ambiguity dramatized by the "#TheDress" illusion.

### 8.3 Color Perception and the Linear-Algebra View

The retina's L, M, S cones sample the incident spectrum with three fixed weighting curves. Writing these as rows of a $3\times N$ matrix $\mathbf{C}_{\text{eye}}$ (rows = spectral sensitivities, $N$ = number of wavelength samples) and the incident spectrum as a column vector $\mathbf{t}$:

$$
\begin{bmatrix}L\\M\\S\end{bmatrix} = \mathbf{C}_{\text{eye}}\,\mathbf{t}
$$

*Why 3 numbers suffice:* although $\mathbf{t}$ lives in a very high-dimensional space (hundreds of wavelength samples), the eye only ever measures its projection onto a fixed 3-dimensional subspace. Any two spectra with the same projection are **metamers** — physically different but perceptually identical — which is precisely why color reproduction (displays, printing) only needs 3 primary channels rather than reproducing full spectra.

**Building a color-matching (reproduction) system.** A display measures $\mathbf{t}$ with its own sensor matrix $\mathbf{C}$, computes primary-light weights via a $3\times3$ matrix $\mathbf{M}$, and drives primary lights $\mathbf{P}$ (columns = primary spectra), producing output spectrum $\mathbf{P}\mathbf{M}\mathbf{C}\,\mathbf{t}$. Requiring the eye's response to the reproduction to equal its response to the original, for *every* input $\mathbf{t}$:

$$
\mathbf{C}_{\text{eye}}\,\mathbf{P}\,\mathbf{M}\,\mathbf{C} = \mathbf{C}_{\text{eye}}
$$

*Derivation of the conditions:* this can only hold for all $\mathbf{t}$ if the sensing subspace matches ($\mathbf{C} = \mathbf{R}\,\mathbf{C}_{\text{eye}}$ for some invertible $3\times3$ matrix $\mathbf{R}$ — any basis of the same 3D subspace works, not just the literal cone curves) and if $\mathbf{C}\mathbf{P}\mathbf{M} = \mathbf{I}_3$, giving:

$$
\mathbf{C} = \mathbf{R}\,\mathbf{C}_{\text{eye}}, \qquad \mathbf{M} = (\mathbf{C}\mathbf{P})^{-1}
$$

*Why this matters:* it formally explains why displays, cameras, and printers can all use *different* sensor/primary spectra from each other and from the human eye, yet still reproduce matching colors — as long as each system's 3 sensitivity curves are *some* linear combination of the eye's, a $3\times3$ correction matrix $\mathbf{M}$ can always align them. This is exactly the math behind camera color-correction matrices (CCMs) in a real image signal processor (ISP).

**CIE XYZ.** A practical standardized choice of $\mathbf{C}$ is the CIE color-matching functions, chosen to be non-negative everywhere (unlike RGB, whose matching functions must be negative in places — meaning no all-positive set of real primaries can exactly reproduce every CIE-representable color). Normalizing tristimulus values $X,Y,Z$ removes overall brightness:

$$
x = \frac{X}{X+Y+Z}, \qquad y = \frac{Y}{X+Y+Z}
$$

### 8.4 Spatial Resolution and Color

Human spatial acuity is far higher for **luminance** than for **chrominance**. Rotating RGB into a luminance/chrominance space like $L,a,b$ and selectively blurring shows that blurring $L$ is very noticeable, while blurring $a$ or $b$ is nearly invisible. *Why:* this follows from the density and wiring of retinal photoreceptors — S-cones are sparse and rarely feed high-acuity pathways, so the visual system simply never had access to high-frequency color information to begin with, and evolved to prioritize luminance edges. This asymmetry is exploited directly by chroma subsampling in every video/image codec (e.g., 4:2:0 YCbCr).

```python
import torch

def spectrum_to_lms(spectrum, C_eye):
    """
    spectrum: (..., N) power spectra sampled at N wavelengths
    C_eye:    (3, N)   L, M, S cone sensitivity curves
    returns:  (..., 3) LMS cone responses,   [L,M,S]^T = C_eye @ t
    """
    return torch.einsum('kn,...n->...k', C_eye, spectrum)

def build_color_correction_matrix(C_measured, C_target):
    """
    Given a camera's actual 3-channel spectral sensitivities C_measured (3,N)
    and a desired standard space C_target (3,N) that is some linear combination
    of the eye's curves, solve for the 3x3 CCM R such that C_target = R @ C_measured
    in a least-squares sense (both must be expressed as sensitivities over the
    same N wavelength samples).
    """
    # R = C_target @ pinv(C_measured)
    return C_target @ torch.linalg.pinv(C_measured)

def rgb_to_chroma_subsampled(lab_image, factor=2):
    """
    lab_image: (3, H, W) tensor with channels [L, a, b].
    Aggressively downsample only the a,b (chroma) channels, keep L (luminance)
    at full resolution -- demonstrating the perceptual asymmetry of Sec 8.4.
    """
    L, a, b = lab_image[0:1], lab_image[1:2], lab_image[2:3]
    pool = torch.nn.AvgPool2d(factor)
    unpool = torch.nn.Upsample(scale_factor=factor, mode='nearest')
    a_sub = unpool(pool(a.unsqueeze(0))).squeeze(0)
    b_sub = unpool(pool(b.unsqueeze(0))).squeeze(0)
    return torch.cat([L, a_sub[:, :L.shape[1], :L.shape[2]], b_sub[:, :L.shape[1], :L.shape[2]]], dim=0)
```

*Why it matters / connections:* the linear-algebra treatment of color matching is the same $\mathbf{C}\mathbf{P}\mathbf{M}=\mathbf{I}$ machinery reused, in spirit, whenever a learned network must map between color spaces (RAW→sRGB ISPs, white-balance networks) — and the metamerism / 3D-subspace-projection idea is exactly why RGB (3-channel) images are almost always sufficient input for vision networks despite the real world's spectra being far higher-dimensional.

**Self-check (Ch 8):**

25. Why is surface reflection modeled multiplicatively ($r(\lambda) = k\,\ell_{\text{in}}(\lambda)s(\lambda)$) rather than additively, and how does this explain the #TheDress illusion?
26. Derive the conditions $\mathbf{C}=\mathbf{R}\mathbf{C}_{\text{eye}}$ and $\mathbf{M}=(\mathbf{C}\mathbf{P})^{-1}$ from the requirement that a color-reproduction system matches the eye's response for every input spectrum.
27. Why can CIE XYZ color-matching functions be made all-positive when RGB's cannot, and what physical property is sacrificed as a result?
28. Why does human vision tolerate heavy chroma subsampling but not luminance subsampling, and where is this exploited in real systems?
29. Two objects have different reflectance spectra but appear identical to a human observer under a given illuminant. What term describes this, and would a hyperspectral camera see them as the same or different?

---

## Latest CV Research — 2026-07-02

Selected for relevance to today's block (imaging, lenses, cameras, color/BRDF), from CVPR 2026, SIGGRAPH 2025, and recent arXiv preprints.

**1. White-Balance First, Adjust Later: Cross-Camera Color Constancy via Vision-Language Evaluation** — Shuwei Li, Lei Tan, Robby T. Tan. *CVPR 2026.* [arXiv:2605.19613](https://arxiv.org/abs/2605.19613)

Learning-based color constancy (illuminant estimation for white balance) typically overfits to the spectral response of the training camera and degrades on unseen cameras. This paper reframes the problem as iterative refinement: instead of regressing an illuminant estimate directly from raw pixels, a vision-language model scores how "correctly white-balanced" a candidate correction looks and drives an iterative feedback loop, sidestepping camera-specific RGB statistics. It's directly relevant to today's Ch 8 material on the $r(\lambda) = k\,\ell_{\text{in}}(\lambda)s(\lambda)$ illumination/reflectance disentanglement problem — this is a modern, generalizable attack on exactly that ambiguity.

**2. PPISP: Physically-Plausible Compensation and Control of Photometric Variations in Radiance Field Reconstruction** — Isaac Deutsch, Nicolas Moënne-Loccoz, Gavriel State, Zan Gojcic (NVIDIA). *CVPR 2026, Oral.* [arXiv:2601.18336](https://arxiv.org/abs/2601.18336)

Multi-view 3D reconstruction (NeRF/Gaussian splatting) assumes a fixed camera response, but real captures vary in exposure, white balance, and vignetting across views. PPISP disentangles camera-intrinsic ISP effects from capture-dependent photometric variation using physically-based, interpretable transforms, and predicts per-view ISP parameters analogous to auto-exposure/auto-white-balance. This is a direct, contemporary application of Ch 8's color-reproduction-system formalism ($\mathbf{C}$, $\mathbf{P}$, $\mathbf{M}$) to neural rendering pipelines.

**3. Differentiable Adaptive 4D Structured Illumination for Joint Capture of Shape and Reflectance** — Huakeng Ding, Yaowen Chen, Kun Zhou, Hongzhi Wu (State Key Lab of CAD&CG, Zhejiang University). arXiv preprint, May 2026. [arXiv:2605.06214](https://arxiv.org/abs/2605.06214)

Proposes a differentiable framework that adaptively computes 4D structured-light illumination patterns to jointly recover object shape and BRDF/reflectance with a single camera, optimizing the illumination itself rather than using fixed patterns. This connects directly to today's Ch 5 BRDF formalism $\ell_{\text{out}} = F(\ell_{\text{in}},\mathbf{n},\lambda,\mathbf{p},\mathbf{q})$, using differentiable rendering to solve for $F$ and geometry jointly.

**4. Collaborative On-Sensor Array Cameras** — Jipeng Sun, Kaixuan Wei, Thomas Eboli, Congli Wang, Cheng Zheng, Zhihao Zhou, Arka Majumdar, Wolfgang Heidrich, Felix Heide (Princeton, KAUST, Univ. of Washington, Université Paris-Saclay). *SIGGRAPH 2025 / ACM ToG.* [arXiv:2506.04061](https://arxiv.org/abs/2506.04061)

Designs an array of thin nanophotonic micro-cameras on a single sensor that jointly ("collaboratively") optimize their individual optics so the array as a whole reconstructs a high-quality image — trading single large-lens optics for many jointly-optimized tiny apertures. This is a direct extension of Ch 6/7's aperture-size tradeoff and Ch 7's "camera as linear system" framework, but with the imaging matrix $\mathbf{A}$ itself made jointly learnable across many micro-cameras.

**5. Spectrum from Defocus: Fast Spectral Imaging with Chromatic Focal Stack** — M. Kerem Aydin, Yi-Chun Hung, Jaclyn Pytlarz, Qi Guo, Emma Alexander. *CVPR 2026.* [arXiv:2503.20184](https://arxiv.org/abs/2503.20184)

Recovers a full hyperspectral image from just two off-the-shelf lenses and a grayscale sensor by deliberately exploiting chromatic aberration (the wavelength-dependence of a lens's focal length, an effect real lens designers try to eliminate) across a focal stack, reconstructing spectra with a fast physics-based inverse algorithm in under a second. A striking real-world case of Ch 6's lensmaker's formula ($f$ depends on $n$, which itself depends on $\lambda$) being turned into a sensing signal instead of an aberration to be corrected.

**6. PhotonSplat: 3D Scene Reconstruction and Colorization from SPAD Sensors** — Sai Sri Teja, Sreevidya Chintalapati, Vinayak Gupta, Mukund Varma T, Haejoon Lee, Aswin Sankaranarayanan, Kaushik Mitra. *ICCP 2025.* [arXiv:2506.21680](https://arxiv.org/abs/2506.21680)

Single-photon avalanche diode (SPAD) sensors capture binary, extremely high-speed but noisy photon-arrival images. PhotonSplat reconstructs full 3D radiance fields directly from these binary SPAD frames, with a 3D spatial-filtering step to denoise and both reference-free (generative-prior) and reference-based colorization. This extends Ch 7's "camera as linear/statistical measurement system" framing to modern quantum-limited sensors and ties into the generative-model machinery from later chapters (Ch 32–34).

---