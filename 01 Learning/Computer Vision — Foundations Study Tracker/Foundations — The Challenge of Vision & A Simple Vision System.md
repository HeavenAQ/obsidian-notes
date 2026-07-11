---
base: "[[Computer Vision — Foundations Study Tracker.base]]"
Key takeaways: "Vision is inverse inference under a prior: recover scene S = argmax p(I|S)p(S) (Helmholtz/Bayes, Marr's 3 levels — computational/representation/hardware). The Blocks World system makes this concrete: orthographic projection x=X, y=cosθ·Y−sinθ·Z mixes Y and Z, so depth is recovered by stacking edge + planarity constraints into an overdetermined sparse least-squares solve Y=(AᵀA)⁻¹Aᵀb. Ch 3 trains scientific looking (pinholes, vanishing points, motion blur, generic-view accidents); Ch 4 demands per-group FPR/FNR fairness auditing and ε-differential privacy (randomized response)."
Day: 1
Status: In-Progress
Reading done: true
Chapters: Ch 1–4
Self-check done: true
Date: 2026-07-02
Part:
  - Foundations
Questions / Follow-ups: ""
---
**Reading checklist**

- [ ] [1 The Challenge of Vision](https://visionbook.mit.edu/taxonomy.html)
- [ ] [2 A Simple Vision System](https://visionbook.mit.edu/simplesystem.html)
- [ ] [3 Looking at Images](https://visionbook.mit.edu/visionscience.html)
- [ ] [4 Computer Vision and Society](https://visionbook.mit.edu/fairness.html)

## Notes

## Self-check

---

# 📘 In-Depth Chapter Report — Day 1 (Ch 1–4, "Foundations")

*Generated 2026-07-01 (JST). Source: Torralba, Isola & Freeman, "Foundations of Computer Vision" (MIT Press), chapters 1–4.*

This block covers the four motivational/foundational chapters: the challenge and taxonomy of vision (Ch 1), an end-to-end hand-designed vision system in the Blocks World (Ch 2), learning to read images as a scientist (Ch 3), and computer vision's societal/fairness dimension (Ch 4). The mathematical core lives in Ch 2; Ch 1 and Ch 4 contribute the inference and fairness formalisms.

---

## 1 · The Challenge of Vision

### Intuition

Vision feels effortless, and that is exactly why it is deceptive. The light arriving at the eye/sensor does **not** tell us what we are looking at — it only reports how much light comes from each direction. The brain (or algorithm) must *invert* this measurement into an interpretation of surfaces, objects, and materials. Visual illusions (Adelson's checkershadow, Shepard's turning tables) prove that what we perceive ≠ the raw light: the system reports the *scene property* it infers (reflectance, 3D shape), not the photons.

### The input: the plenoptic function

The full pattern of light filling space is described by the **plenoptic function** (Adelson & Bergen):

$P(\theta, \phi, \lambda, t, X, Y, Z)$

where $(\theta,\phi)$ is the direction of a ray, $\lambda$ its wavelength, $t$ time, and $(X,Y,Z)$ the world location the ray passes through. $P$ is the radiance of that ray. **Why this form:** it is the most general parameterization of "all light everywhere" — every camera or eye is just a device that *samples a low-dimensional slice* of $P$. Framing imaging as sampling the plenoptic function unifies pinholes, lenses, light-field cameras, and the retina under one object.

### The output is inference, not measurement (Helmholtz)

Helmholtz framed perception as **unconscious inference**: the percept is the scene "most likely to have produced this sensory input." Modern computer vision formalizes this as Bayesian inversion:

$\hat{S} \;=\; \arg\max_{S}\; p(S \mid I) \;=\; \arg\max_{S}\; p(I \mid S)\, p(S)$

- $I$ — the observed image (a slice of the plenoptic function).
- $S$ — the latent scene (geometry, materials, lighting).
- $p(I\mid S)$ — the **likelihood** / forward image-formation (rendering) model.
- $p(S)$ — the **prior** over plausible scenes.
- $\hat S$ — the MAP estimate of the scene.

**Why it is formulated this way:** projection from 3D→2D is many-to-one, so recovering $S$ from $I$ is *ill-posed* — infinitely many scenes explain any image. The likelihood alone cannot pick one out; the prior $p(S)$ (e.g. "squares and occlusions are common") regularizes the inversion. This is precisely why we read an L-shape as an occluded square. Every illusion is the prior overriding the data.

### Euclid's geometric seed

Euclid modeled sight as straight rays forming a cone, and measured object sizes by **similar triangles** — e.g. recovering a height $AB$ from its shadow:

![[99 Assets/Media/image.png]]

$\frac{DE}{ZE} = \frac{DB}{AB} \quad\Longrightarrow\quad AB = DB\,\frac{ZE}{DE}$

Here $D$ is the eye, $DB$ the known shadow length, and $DE/ZE$ a known auxiliary ratio. **Why it matters:** this is the first quantitative model that geometry of light → measurable scene properties, the ancestor of every camera model and stereo-triangulation equation later in the book.

### Marr's three levels of analysis

Marr insisted that any information-processing system be understood at three (loosely coupled) levels:

1. **Computational theory** — *what* is computed and *why* (the input→output mapping and why it suits the task).
2. **Representation & algorithm** — *how*: which representations encode the input/output, and the algorithm that maps between them.
3. **Hardware / implementation** — the physical substrate.

Marr's processing pipeline — *image → primal sketch (edges, junctions) → 2.5D sketch (depth map + surface orientation) → 3D model* — directly foreshadows the Ch 2 system. **Why this decomposition matters:** it warns against today's habit of fusing "the task" with "a particular neural net" without knowing what is being computed or what the representation means.

### Neuroscience connections

Center-surround retinal ganglion cells (Kuffler) and orientation-selective simple cells in V1 (Hubel & Wiesel, organized in orientation/ocular-dominance hypercolumns) motivate two pillars of CV: **linear filtering with rectifying nonlinearities** and **oriented edge detectors**. A center-surround receptive field is well modeled by a Difference-of-Gaussians, which is also a contrast/edge enhancer:

```python
import torch
import torch.nn.functional as F

def gaussian_2d(ksize, sigma):
    ax = torch.arange(ksize) - (ksize - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    g = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return g / g.sum()

# Difference-of-Gaussians ≈ ON-center / OFF-surround ganglion-cell receptive field
def dog_kernel(ksize=11, sigma_c=1.0, sigma_s=2.0):
    return gaussian_2d(ksize, sigma_c) - gaussian_2d(ksize, sigma_s)

img = torch.rand(1, 1, 64, 64)
k = dog_kernel().view(1, 1, 11, 11)
response = F.conv2d(img, k, padding=5)  # contrast-enhanced "neural" response
```

### Connections / why it matters

This chapter sets the entire book's thesis: **vision = inverse inference under a prior.** Marr's levels give us the vocabulary; Helmholtz/Bayes gives the math; the plenoptic function gives the input model; neuroscience gives the architectural hints (filters + nonlinearities). Everything downstream is a concrete instantiation of $\arg\max_S p(I\mid S)p(S)$.

---

## 2 · A Simple Vision System (the Blocks World)

This chapter builds a complete, hand-designed system end-to-end: from pixels → edges → 3D world coordinates. It is the concrete instantiation of Marr's pipeline and the inference idea from Ch 1.

### The world and the image-formation model

The Blocks World: matte objects with horizontal/vertical flat faces resting on a white ground plane. The camera uses **orthographic (parallel) projection** — rays travel parallel and perpendicular to the image plane, so object size is independent of depth and 3D parallel lines stay parallel in 2D. With the camera tilted by angle $\theta$ about its horizontal axis, a world point $(X,Y,Z)$ projects to image point $(x,y)$ via:

$\begin{aligned} x &= X \\ y &= \cos(\theta)\,Y - \sin(\theta)\,Z \end{aligned}$

- $(X,Y,Z)$ — world coordinates; $(x,y)$ — image coordinates.
- $\theta$ — camera tilt (angle between the optical line and the $Z$-axis).

**Why this form (and the core difficulty):** $x$ is read off directly ($X=x$, trivial), but $y$ is a **mixture** of $Y$ and $Z$. One world dimension is destroyed by projection: a point moving along $Z$ is indistinguishable from one moving along $Y$. Recovering $Y(x,y)$ and $Z(x,y)$ is therefore the ill-posed inverse problem of Ch 1, made concrete.

### From images to edges

Treat the image $\ell(x,y)$ as a continuous intensity surface. Its **gradient** measures local variation:

$\nabla \ell = \left( \frac{\partial \ell}{\partial x},\; \frac{\partial \ell}{\partial y} \right)$

On discrete pixels the partials are approximated by finite differences, and a less noisy estimate uses the **Sobel** weights:

$\frac{\partial \ell}{\partial x} \approx \ell(x,y)-\ell(x-1,y), \qquad \frac{1}{4}\begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$

**Why Sobel and not a bare difference:** the bare difference is a derivative of *one* row and amplifies noise; the Sobel kernel is a derivative in $x$ *smoothed* by a binomial $[1,2,1]$ in $y$, trading a little spatial resolution for far better noise rejection (this is unpacked in Ch 18).

From the gradient we extract:

$e(x,y) = \lVert \nabla \ell(x,y)\rVert \quad (\text{edge strength}), \qquad \theta(x,y) = \angle\nabla\ell = \arctan\!\left(\frac{\partial\ell/\partial y}{\partial\ell/\partial x}\right)\ (\text{edge orientation})$

and the unit normal and tangent to the edge:

$\mathbf{n} = \frac{\nabla\ell}{\lVert\nabla\ell\rVert}, \qquad \mathbf{t} = (-n_y,\, n_x)$

- $e$ — gradient magnitude; large where intensity changes sharply (candidate edge).
- $\theta$ — gradient direction; the edge itself runs *perpendicular* to it, hence the tangent $\mathbf t \perp \mathbf n$.

Edges are then thresholded ($e>\tau$) and classified by cause: **object boundaries** (occlusion vs **contact** edges), **surface-orientation** changes, and **shadow** edges.

```python
import torch
import torch.nn.functional as F

def image_gradients(img):  # img: (1,1,H,W)
    sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)/4
    sy = sx.t().contiguous()
    gx = F.conv2d(img, sx.view(1,1,3,3), padding=1)
    gy = F.conv2d(img, sy.view(1,1,3,3), padding=1)
    strength = torch.sqrt(gx**2 + gy**2 + 1e-12)
    orientation = torch.atan2(gy, gx)
    return gx, gy, strength, orientation

img = torch.rand(1,1,64,64)
gx, gy, e, ori = image_gradients(img)
edges = e > 0.2  # threshold to get candidate edge pixels
```

### From edges to surfaces — constraints

We want $Y(x,y)$ everywhere (then $Z$ follows from the projection equation). Local cues give linear constraints:

**Figure/ground:** ground pixels (bright, low-saturation) get $Y(x,y)=0$.

**Contact edges** (object meets ground, no depth jump) also pin $Y=0$. Scanning each vertical image column top→bottom, ground→figure transitions are **occlusion** boundaries; figure→ground transitions are **contact** edges (border-ownership matters — only the foreground surface owns the edge).

**3D-vertical edges** (which stay vertical in the image under this projection): differentiating the projection equation along the edge gives

$\frac{\partial Y}{\partial y} = \frac{1}{\cos(\theta)}$

**3D-horizontal edges**: $Y$ is constant along them, so the directional derivative along the tangent vanishes:

$\frac{\partial Y}{\partial \mathbf{t}} = \nabla Y \cdot \mathbf{t} = -n_y\frac{\partial Y}{\partial x} + n_x\frac{\partial Y}{\partial y} = 0$

**Planar faces** (interior of object surfaces): second derivatives vanish,

$\frac{\partial^2 Y}{\partial x^2}=0,\quad \frac{\partial^2 Y}{\partial y^2}=0,\quad \frac{\partial^2 Y}{\partial x\,\partial y}=0$

discretized e.g. as $\partial^2 Y/\partial x^2 \approx 2Y(x,y)-Y(x+1,y)-Y(x-1,y)$.

**Why these constraints:** the *generic-view assumption* (Ch 1's non-accidental properties) lets us trust that a vertical image line is a vertical world line, etc. Edges carry strong 3D information but only along the boundary; the planarity (smoothness) constraints are what **propagate** that boundary information into the flat interior where no local cue exists. This propagate-local-evidence idea recurs throughout the book (MRFs, diffusion, etc.).

### A simple inference scheme — overdetermined least squares

Stack every constraint as a linear equation $\mathbf a_i \mathbf Y = b_i$ (e.g. a planarity row is $\mathbf a_i=[\dots,-1,2,-1,\dots]$, $b_i=0$). With $\mathbf Y$ the vectorized depth image, minimize the (optionally weighted) squared residual:

$J = \sum_i w_i\,(\mathbf a_i \mathbf Y - b_i)^2 \;=\; \lVert \mathbf{W}^{1/2}(\mathbf{A}\mathbf{Y}-\mathbf{b})\rVert^2$

whose normal-equation / pseudoinverse solution (unweighted) is:

$\mathbf{Y} = (\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b}$

- $\mathbf A$ — (#constraints × #pixels) sparse constraint matrix (more rows than columns ⇒ overdetermined).
- $\mathbf b$ — target values (0 for ground/planarity, $1/\cos\theta$ for vertical-edge steps).
- $w_i$ — per-constraint confidence weights.

**Why least squares / pseudoinverse:** the system is overdetermined and the constraints are noisy & mutually inconsistent (no exact solution), so we seek the $\mathbf Y$ minimizing total squared violation — the maximum-likelihood estimate under i.i.d. Gaussian residual noise. $\mathbf A$ is extremely sparse (each row touches ≤3 pixels), so this scales.

```python
import torch

# Toy 1-D illustration: recover Y on a length-N strip from
#   (a) a boundary anchor Y[0]=0, and (b) planarity 2Y[i]-Y[i-1]-Y[i+1]=0.
N = 8
rows, b = [], []

# anchor constraint (high weight)
r = torch.zeros(N); r[0] = 1.0
rows.append(3.0 * r); b.append(3.0 * 0.0)

# a vertical-edge "step" constraint: Y[1]-Y[0] = 1/cos(theta)
import math
theta = math.radians(20)
r = torch.zeros(N); r[1], r[0] = 1.0, -1.0
rows.append(r); b.append(1.0 / math.cos(theta))

# planarity (smoothness) on interior pixels
for i in range(1, N-1):
    r = torch.zeros(N); r[i-1], r[i], r[i+1] = -1.0, 2.0, -1.0
    rows.append(r); b.append(0.0)

A = torch.stack(rows)              # (#constraints, N)
b = torch.tensor(b).unsqueeze(1)   # (#constraints, 1)
Y = torch.linalg.lstsq(A, b).solution   # least-squares solve (robust pseudoinverse)
print(Y.squeeze())
```

### Generalization & limits

Run on out-of-domain inputs (hard shadows, occluding/stacked blocks, Adelson's "impossible steps") and it degrades gracefully — sometimes right, sometimes not. Crucially the system has **no object concept**: it cannot count cubes or hallucinate occluded geometry. **Why it matters:** the domain is defined by hand-coded assumptions; in later (learned) chapters the *training set* implicitly defines the domain — the same generalization question, relocated from assumptions to data.

### Connections

Ch 2 is the whole book in miniature: a representation cascade (pixels→edges→surfaces), the generic-view prior, evidence propagation, and a least-squares inverse — all themes that reappear as convolutions, MRFs/CRFs, and learned energy minimization.

---

## 3 · Looking at Images

### Intuition

This chapter is deliberately math-light: it trains the *eye of the scientist*. The thesis is that perception is not a fixed input→output function — it is a **dynamical process**: the longer you look, the more you see. The job of a vision system is to infer everything you *cannot* directly see.

### Key observations and the mechanics behind them

- **Pixels mean nothing in isolation.** A 32×32 image is recognizable, yet a 3-pixel "pen" is unidentifiable when cropped out. Recognition is contextual — the same lesson as Ch 1's ill-posedness: local evidence is ambiguous, the global configuration disambiguates. A single big image is itself "big data" (thousands of overlapping patches).
- **Tree-shadow pinholes.** Gaps between leaves act as **pinhole cameras**, each projecting an *image of the sun* (circular spots normally; crescents during an eclipse), not images of the gaps. This is the pinhole image-formation model appearing in nature — the same model formalized in the imaging chapters. Intuitively, when the aperture is small relative to source–screen distance, the projected spot is the *convolution of the aperture with the source*, and the source shape dominates.
- **Horizontal vs vertical from vanishing points.** Two line drawings differing only by 90° rotation read as floor vs wall. The cue is the **vanishing point** — where 3D-parallel lines meet in the image — and the **horizon line** joining vanishing points. The visual system infers camera pose and surface orientation from where these points sit relative to the figure (foreshadowing single-view metrology, Ch 42).
- **Motion blur encodes depth & speed.** A photo from a moving car is the time-average of translated copies during exposure; nearby objects blur more than distant ones. So blur length $\propto$ (camera speed × exposure) / depth — a cue linking appearance to scene geometry and motion (Ch 19, optical flow).
- **Accidents happen / principle of continuity.** Real photos are full of *accidental* alignments (one contour tangent to another on a different object). Rule-based systems failed precisely because they assumed no accidents. A robust system must treat the generic-view assumption as a *prior*, not a hard rule.
- **Cues for support, raindrops as light-field cameras, Plato's cave, wetness.** Drop shadows and slack strings signal contact; raindrops each refract a slightly different view (a natural light-field array — Ch 45 NeRF/radiance fields); an image is a lossy 2D projection of 3D state ("Plato's cave"); we read materials (wet/dry, hard/soft) from subtle shading — wet sand looks darker because water changes surface reflectance/specularity.

A minimal demonstration that **motion blur = average of shifted copies** (the forward model behind the observation):

```python
import torch
import torch.nn.functional as F

def horizontal_motion_blur(img, length=15):  # img: (1,1,H,W)
    # average of `length` horizontally shifted copies = 1-D box filter along x
    k = torch.ones(1, 1, 1, length) / length
    return F.conv2d(img, k, padding=(0, length // 2))

img = torch.rand(1, 1, 64, 64)
blurred = horizontal_motion_blur(img, length=15)
```

### Why it matters

No new algorithm — but the questions ("why does this look the way it does?") are exactly the forward models (image formation, shading, projection, light fields) that the rest of the book formalizes and then *inverts*. Notably, several of these phenomena (continuity, grouping) are the Gestalt cues from Ch 1, here observed in real photographs.

---

## 4 · Computer Vision and Society

### Intuition

As CV is deployed (face ID, driving, surveillance, hiring), its mistakes and its *uneven* performance across groups carry real harm. The chapter covers two threads: **fairness** (measurable performance disparities + mitigation) and **ethics** (questions that outlive any bias fix).

### Fairness — define and measure per group

Because models are trained from data, dataset statistics propagate (and can *amplify*) social biases. The headline empirical findings: commercial gender classifiers in 2018 performed worse on darker skin tones (Buolamwini & Gebru); the NIST FRVT found false-positive rates varying by up to ~100× across countries of birth. The methodological imperative: **report both false-positive and false-negative rates, per demographic group, at a fixed threshold.** For group $g$:

$\mathrm{FPR}_g = \frac{FP_g}{FP_g + TN_g}, \qquad \mathrm{FNR}_g = \frac{FN_g}{FN_g + TP_g}$

- $FP_g, TN_g, FN_g, TP_g$ — confusion-matrix counts restricted to subjects in group $g$.

**Why both, per group:** a single aggregate accuracy hides that two groups can have equal accuracy but opposite error *types* — and the social cost of a false positive (wrongful match) vs false negative differs sharply. A fair audit therefore disaggregates and reports the full error profile, not one scalar.

### Mitigation techniques

- **GAN-based debiasing:** synthesize balanced data — *Fairness GAN* trains a head to predict the protected attribute *as poorly as possible* (adversarially), so the target prediction gains nothing from e.g. gender; or generate paired images breaking spurious correlations ("hat" ⊥ "glasses").
- **Counterfactual transects:** walk a single attribute through a GAN's latent space, holding others fixed, to isolate whether a disparity is *algorithmic* or *dataset* bias — counterfactual (interventional) analysis can contradict observational analysis.
- **Distribution-matching constraints:** force predictions to follow the (assumed-good) training distribution.

### Privacy — differential privacy

**Differential privacy** lets you extract population-level statistics while provably protecting any individual. The formal $\varepsilon$-DP guarantee on a randomized mechanism $M$:

$\Pr[M(D) \in \mathcal{S}] \;\le\; e^{\varepsilon}\,\Pr[M(D') \in \mathcal{S}]$

for all outputs $\mathcal S$ and all **adjacent** datasets $D, D'$ differing in one individual's record.

- $M$ — randomized algorithm/query; $\varepsilon$ — privacy budget (smaller = more private).
- $D, D'$ — datasets differing by one person.

**Why this definition:** it bounds how much *any* output can change when one person's data is added or removed — so participation cannot be detected, *regardless of any side information* an attacker holds (defeating the re-identification attacks that combined "anonymized" medical + voter records). The classic instance is **randomized response**: answer truthfully on heads, answer randomly on tails. Each individual has deniability, yet the true population rate is recoverable:

$\hat{p}_{\text{yes}} = \frac{\bar{r} - \tfrac{1}{4}}{\tfrac{1}{2}}, \qquad \bar r = \text{observed fraction of "yes"}$

because $\mathbb E[\bar r] = \tfrac12 p + \tfrac12\cdot\tfrac12$. The cost of privacy is variance — you need more samples for the same accuracy.

```python
import numpy as np

def randomized_response(true_bits, seed=0):
    rng = np.random.default_rng(seed)
    coin1 = rng.random(len(true_bits)) < 0.5   # heads -> answer truthfully
    coin2 = rng.random(len(true_bits)) < 0.5   # tails -> random yes/no
    responses = np.where(coin1, true_bits, coin2.astype(int))
    return responses

def estimate_true_rate(responses):
    r_bar = responses.mean()
    return (r_bar - 0.25) / 0.5   # unbiased estimate of true "yes" fraction

true = (np.random.default_rng(1).random(200_000) < 0.30).astype(int)  # true rate 0.30
resp = randomized_response(true)
print(f"true=0.300  estimated={estimate_true_rate(resp):.3f}")  # ≈0.30 with deniability
```

### Ethics — questions that remain

Even a perfectly unbiased model leaves hard questions: face analysis in hiring; identifying protesters; automation bias (people over-trust machine output); the harm of gender-labeling for LGBTQ people; the self-driving "trolley problem" (fewer deaths overall, but *machine-caused* deaths); offensive dataset content and consent. The chapter's stance: researchers must engage these as **both technologists and citizens**.

### Why it matters

This chapter ties the book's technical machinery (datasets, GANs, classifiers) to consequences. The fairness metrics ($\mathrm{FPR}_g/\mathrm{FNR}_g$) and DP guarantee are concrete, testable engineering requirements — not afterthoughts — and connect directly to the dataset-bias and generative-model chapters later on.

---

## ✅ Self-Check Questions (Day 1)

4. **Why is recovering 3D from a single image ill-posed, and how does the Bayesian/Helmholtz formulation **$\hat S=\arg\max_S p(I\mid S)p(S)$** resolve it?** Name the two terms and say which one encodes "occlusions are common."
5. In the Blocks World projection $x=X,\; y=\cos\theta\,Y-\sin\theta\,Z$, **why is **$Z$** unrecoverable from a single pixel**, and which extra constraints let us pin down $Y$ (hence $Z$) anyway?
6. **Derive/explain** why the constraint matrix $\mathbf A$ in the depth-recovery least-squares problem is both *overdetermined* and *sparse*, and why we use $\mathbf Y=(\mathbf A^\top\mathbf A)^{-1}\mathbf A^\top\mathbf b$ rather than a direct inverse.
7. State the **generic-view (non-accidental property) assumption** and give one example from Ch 2/Ch 3 where it holds and one where it fails (an "accident").
8. A face system has equal overall accuracy on groups A and B but $\mathrm{FPR}_A \gg \mathrm{FPR}_B$. **Why is reporting per-group FPR *****and***** FNR essential**, and how does $\varepsilon$-differential privacy bound an individual's exposure regardless of attacker side-information?

---

## 🆕 Latest CV Research — 2026-07-01

A scan of top venues and recent arXiv ([cs.CV](http://cs.cv/)), prioritizing items connected to today's foundations block (inference, perceptual grouping, fairness) plus the field's current frontier.

**1. Efficiently Reconstructing Dynamic Scenes One D4RT at a Time** — Zhang, Le Moing, Koppula, Rocco, Sajjadi et al. (Google DeepMind · UCL · Oxford). *CVPR 2026 — Best Paper.* Replaces the multi-model 4D pipeline (separate depth, optical-flow, camera-pose nets) with a **single query interface**: encode a video once, then answer depth / point-tracking / camera-pose queries from one shared latent. Treats dynamic objects identically to static ones — no special case, no test-time optimization, no fusion — and sets SOTA on every 4D benchmark at ~200 fps for pose. *Why it matters:* a clean realization of "one representation, many queries," the modern descendant of Marr's unified representation cascade. [Project page](https://d4rt-paper.github.io/) · [CVPR 2026 Best Papers](https://cvpr.thecvf.com/Conferences/2026/News/Best_Papers)

**2. Human-like Object Grouping in Self-Supervised Vision Transformers** — (arXiv:2603.13994, Mar 2026, [cs.CV](http://cs.cv/)). Shows self-supervised ViTs exhibit *emergent* object-grouping/segmentation that aligns with human perceptual organization, without grouping supervision. *Why it matters:* a direct empirical bridge from today's **Gestalt grouping laws** (Ch 1) and "accidents/continuity" (Ch 3) to modern foundation models — the grouping priors humans use seem to re-emerge from large-scale self-supervision. [arXiv](https://arxiv.org/abs/2603.13994)

**3. Benchmarking Bias Mitigation Toward Fairness Without Harm, from Vision to LVLMs** — (arXiv:2602.03895, 2026). A unified benchmark evaluating whether bias-mitigation methods reduce group disparities *without* degrading overall accuracy ("fairness without harm"), extended from classic vision classifiers to large vision-language models. *Why it matters:* operationalizes exactly Ch 4's mandate to measure per-group error trade-offs, now for the LVLM era. [arXiv](https://arxiv.org/pdf/2602.03895)

**4. Occ-VLM: Occupancy-Grounded Vision-Language Model for Indoor Scene Understanding** — (arXiv:2606.19776, Jun 2026, [cs.CV](http://cs.cv/)). Builds a VLM that reconstructs 3D scene **occupancy** from posed RGB images (single 2D encoder) as an auxiliary geometric prior for language-grounded indoor understanding. *Why it matters:* fuses geometry (the book's Part XI) with language grounding — inference of unseen 3D structure ("Plato's cave," Ch 3) inside a modern VLM. [arXiv](https://arxiv.org/abs/2606.19776)

**5. The Hidden Evolution of Disguised Visual Context inside the VLM** — (arXiv:2606.20077, Jun 2026, [cs.CV](http://cs.cv/)). Analyzes how raw visual tokens entering an LLM are progressively transformed into meaningful representations, depending on the integration architecture. *Why it matters:* a representation-level ("how", in Marr's terms) probe of multimodal models — what the visual representation *means* layer by layer. [arXiv](https://arxiv.org/html/2606.20077v1)

**6. PP-OCRv6: From 1.5M to 34.5M Parameters, Surpassing Billion-Scale VLMs on OCR** — (arXiv:2606.13108, Jun 2026, [cs.CV](http://cs.cv/)). A lightweight OCR system combining architectural and data-centric optimization to beat far larger VLMs on text reading while curbing hallucination and localization errors. *Why it matters:* a counterpoint to scale-maximalism — careful representation + data design beats brute size on a focused task, echoing Marr's "right level for the problem." [arXiv](https://arxiv.org/html/2606.13108)