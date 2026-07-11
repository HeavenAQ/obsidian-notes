---
base: "[[Computer Vision — Foundations Study Tracker.base]]"
Key takeaways: "Derivatives are LTI ⇒ convolutions: [1,−1] gives a ½-pixel phase shift while central difference [1,0,−1]/2 = [1,−1]∘[1,1] is centered but blurrier; Gaussian derivatives fix noise/undefined-edges via the commuting identity ∂ℓ∘g = ℓ∘∂g (Hermite-polynomial family), and steerability lets any orientation be a linear combination of gₓ,g_y, with the rotationally-invariant Laplacian/LoG as the second-order band-pass. Temporal filters live in the x–y–t volume: global motion ℓ₀(x−vₓt,y−v_yt) puts all spectral energy on the tilted plane w_t+vₓwₓ+v_yw_y=0, yielding the brightness-constancy equation ℓ_t+vₓℓₓ+v_yℓ_y=0 and Darrell–Simoncelli nulling filters h=∇g·(1,vₓ,v_y) that cancel a tuned velocity; causal IIR leaky integrators (stable iff |α|<1) foreshadow RNNs. Sampling multiplies by a Dirac comb, replicating the spectrum every 2π/Tₛ, so Nyquist (Tₛ<π/w_max) prevents replica overlap = aliasing; ideal reconstruction is sinc/Whittaker–Shannon (local kernels = nearest/box, linear/triangle, Lanczos), hexagonal lattices are ~10% optimal, and you must low-pass BEFORE subsampling — the theory behind anti-aliased CNN downsampling."
Day: 6
Status: In progress
Reading done: true
Chapters: Ch 18–20
Self-check done: false
Date: 2026-07-06
Part:
  - Linear Filters
  - Sampling & Multiscale
Questions / Follow-ups: ""
---
**Reading checklist**

- [ ] [18 Image Derivatives](https://visionbook.mit.edu/derivatives.html)
- [ ] [19 Temporal Filters](https://visionbook.mit.edu/temporal_filters_v2.html)
- [ ] [20 Image Sampling and Aliasing](https://visionbook.mit.edu/sampling_and_aliasing.html)

## Notes

## Self-check

---

# Day 6 Report — Derivatives, Temporal Filters & Sampling (Ch 18–20)

*Generated 2026-07-06 (JST). Source: *[*visionbook.mit.edu*](http://visionbook.mit.edu/)*, chapters 18–20. Priorities for every topic below: (1) full end-to-end operator/architecture, (2) the math, (3) why the math is formulated that way, (4) how to replicate it in PyTorch.*

## Chapter 18 — Image Derivatives

### 18.1 Why differentiate an image at all

Derivatives tell us **where** the image changes, and changes in intensity are strongly correlated with the boundaries between objects, texture edges, and shading discontinuities. The derivative operator is **linear and translation-invariant (LTI)**, so — by the same theorem used for blur filters in Ch 15 — it can always be written as a **convolution** with a fixed kernel. This is exactly why every classical edge feature (SIFT, HOG) and the first layers of a CNN are built out of derivative-like filters.

### 18.2 From the continuous derivative to discrete kernels

The continuous partial derivative is defined as a limit:

$$
\frac{\partial \ell(x,y)}{\partial x} = \lim_{\epsilon \to 0} \frac{\ell(x+\epsilon,y) - \ell(x,y)}{\epsilon}
$$

Symbols: $\ell(x,y)$ = continuous image intensity; $x,y$ = spatial coordinates; $\epsilon$ = spatial step. We cannot take $\epsilon \to 0$ on a sampled image $\ell[n,m]$, so we approximate the derivative by a small linear filter. Two canonical 1-D approximations:

$$
d_0 = [\,1,\,-1\,], \qquad \ell \circ d_0 = \ell[n] - \ell[n-1]
$$

$$
d_1 = \tfrac{1}{2}[\,1,\,0,\,-1\,], \qquad \ell \circ d_1 = \frac{\ell[n+1]-\ell[n-1]}{2}
$$

Symbols: $\circ$ = convolution; $\ell[n]$ = sampled 1-D signal. **Why two forms?** $d_0$ (forward/backward difference, the $\epsilon=1$ case) is a length-2 filter, so its output is shifted by **half a pixel** (it is not centered on the origin). $d_1$ (central difference) is centered — no shift — but averages two gaps, so it is smoother and blind to the highest frequency $[1,-1,1,-1,\dots]$.

**Frequency view (why the shift appears).** In the continuous Fourier domain differentiation is multiplication by $jw$:

$$
\frac{\partial \ell(x)}{\partial x} \;\xrightarrow{\;\mathscr{F}\;}\; j\,w\,\mathscr{L}(w)
$$

The DFTs of the discrete kernels are:

$$
D_0[u] = \exp\!\left(-\pi j \tfrac{u}{N}\right)\, 2j\sin\!\left(\tfrac{\pi u}{N}\right), \qquad D_1[u] = j\sin\!\left(\tfrac{2\pi u}{N}\right)
$$

Symbols: $w$ = radian frequency; $\mathscr{L}(w)$ = FT of the image; $u$ = discrete frequency index; $N$ = signal length; $j=\sqrt{-1}$. The leading phase term $\exp(-\pi j u/N)$ in $D_0$ **is** the half-sample delay; $D_1$ has no phase term (no shift) because $d_1 = d_0 \circ [1,1]$, i.e. $D_1[u]=D_0[u]\,B_1[u]$ where $B_1$ is a binomial low-pass — central difference = forward difference then smooth.

### 18.3 Matrix view: derivatives, inversion, and gradient-domain editing

Writing the derivative as a banded (Toeplitz) matrix $\mathbf{D_0}$ acting on the stacked signal $\boldsymbol\ell$ makes integration explicit. If we keep only *valid* differences (drop boundary-affected samples), $\mathbf{D_0}$ is non-square and we recover the signal with the **pseudoinverse** $\mathbf{D_0}^{+}$, which reconstructs $\boldsymbol\ell$ **up to its DC (mean) component** (the null space of the derivative is the constant signal). In 2-D, encode with both derivatives and decode with the pseudoinverse:

$$
\mathbf{r} = \begin{bmatrix}\mathbf{D_x}\\ \mathbf{D_y}\end{bmatrix}\boldsymbol\ell
$$

**Why this matters:** editing $\mathbf{r}$ (e.g. zeroing gradients over a masked region) and decoding gives **gradient-domain inpainting** — deleting an object without specifying its fill color, because the color is supplied by the integration/Poisson solve. This is the classical precursor to Poisson image editing.

### 18.4 Gaussian derivatives (the practical operator)

Plain differences fail on real images for two reasons: (1) **noise** — differencing two pixels with i.i.d. noise of variance $\sigma^2$ yields noise of variance $2\sigma^2$, which dominates when the true signal varies slowly; (2) the derivative is **undefined** at true step edges. Both are fixed by smoothing with a differentiable low-pass $g$. The key identity (provable in the Fourier domain) is that convolution and differentiation **commute**:

$$
\frac{\partial \ell}{\partial x} \circ g = \ell \circ \frac{\partial g}{\partial x}
$$

So instead of differentiating the (noisy, possibly non-differentiable) image, we differentiate the **smooth Gaussian kernel** once and convolve. For a Gaussian $g(x,y;\sigma)$ the first derivative has the closed form:

$$
g_x(x,y;\sigma) = \frac{\partial g}{\partial x} = \frac{-x}{2\pi\sigma^4}\exp\!\left(-\frac{x^2+y^2}{2\sigma^2}\right) = \frac{-x}{\sigma^2}\,g(x,y;\sigma)
$$

Symbols: $\sigma$ = Gaussian width (scale); larger $\sigma$ = coarser edges. Higher orders are a **Hermite polynomial** times a Gaussian:

$$
g_{x^n}(x;\sigma) = \left(\frac{-1}{\sigma\sqrt{2}}\right)^{\!n} H_n\!\left(\frac{x}{\sigma\sqrt2}\right) g(x;\sigma), \qquad g_{x^2}=\frac{x^2-\sigma^2}{\sigma^4}g
$$

with $H_0=1,\;H_1=2x,\;H_2=4x^2-2$. **Why Gaussian:** it is the unique separable, isotropic, infinitely-differentiable smoother that introduces no spurious extrema across scale — the foundation of scale space (Ch 23).

### 18.5 Gradient, directional (steerable) derivatives, Sobel, Laplacian

The image gradient is the 2-vector

$$
\nabla \ell = \left(\frac{\partial \ell}{\partial x},\, \frac{\partial \ell}{\partial y}\right),\qquad \|\nabla\ell\| = \sqrt{\ell_x^2+\ell_y^2},\quad \theta=\operatorname{atan2}(\ell_y,\ell_x)
$$

giving edge **strength** and **orientation** at each pixel. The derivative in any direction $\mathbf t=(\cos\theta,\sin\theta)$ is a **linear combination** of the two axis derivatives (steerability — no new convolution needed):

$$
\frac{\partial \ell}{\partial \mathbf t} = \nabla\ell\cdot\mathbf t = \cos\theta\,\ell_x + \sin\theta\,\ell_y
$$

The **Sobel–Feldman** operator is the compact, near-isotropic discretization $Sobel_x = b_1\circ d_0\circ b_1^{\top}\circ b_1^{\top}$ (a central difference smoothed by binomials), while **Roberts cross** uses a rotated 2×2 frame to keep the two outputs spatially aligned. The **Laplacian** sums the second derivatives and is rotationally invariant:

$$
\nabla^2\ell = \frac{\partial^2\ell}{\partial x^2}+\frac{\partial^2\ell}{\partial y^2}, \qquad \text{5-point: } \begin{bmatrix}0&1&0\\ 1&-4&1\\ 0&1&0\end{bmatrix}
$$

Because $\nabla^2$ amplifies noise, we combine it with a Gaussian into the **Laplacian of Gaussian (LoG)** — the "inverted Mexican-hat" band-pass — using $\nabla^2(\ell\circ g)=\ell\circ\nabla^2 g$. Zero-crossings of the LoG are the Marr–Hildreth edges. Advantage over gradient: the Laplacian is **zero on linear intensity ramps** (it measures curvature), so it responds to true boundaries rather than smooth shading.

```python
import torch, torch.nn.functional as F

def gaussian_derivative_kernels(sigma=1.0, radius=None):
    if radius is None: radius = int(3*sigma)
    x = torch.arange(-radius, radius+1, dtype=torch.float32)
    g1d = torch.exp(-x**2/(2*sigma**2)); g1d /= g1d.sum()
    gx, gy = torch.meshgrid(x, x, indexing='ij')
    g2d = torch.exp(-(gx**2+gy**2)/(2*sigma**2)); g2d /= g2d.sum()
    dGx = (-gx/sigma**2)*g2d          # analytic Gaussian x-derivative
    dGy = (-gy/sigma**2)*g2d
    return dGx, dGy

def image_gradient(img, sigma=1.0):           # img: (B,1,H,W)
    dGx, dGy = gaussian_derivative_kernels(sigma)
    k = lambda w: w.flip(0,1).unsqueeze(0).unsqueeze(0)   # conv = cross-corr flipped
    gx = F.conv2d(img, k(dGx), padding='same')
    gy = F.conv2d(img, k(dGy), padding='same')
    mag = torch.sqrt(gx**2 + gy**2 + 1e-12)
    ang = torch.atan2(gy, gx)
    return gx, gy, mag, ang
```

### Why it matters / connections

Derivative filters are the *learned* first-layer kernels of every CNN (Ch 24); steerability underlies oriented filter banks (Ch 22) and HOG/SIFT; the gradient-domain pseudoinverse is the Poisson-editing backbone; the LoG band-pass is the difference-of-Gaussians used in SIFT scale-space.

### Self-check (Ch 18)

1. Why does the $[1,-1]$ filter shift the output by half a pixel while $[1,0,-1]/2$ does not — answer in both spatial and Fourier terms?
2. Show that $\frac{\partial \ell}{\partial x}\circ g = \ell \circ \frac{\partial g}{\partial x}$ using the Fourier convolution theorem.
3. Why is the Laplacian exactly zero on a linear intensity ramp but the gradient is not?
4. Given $g_x$ and $g_y$ outputs, how do you obtain the derivative at 30° without any new convolution?

## Chapter 19 — Temporal Filters

### 19.1 Sequences are not arbitrary 3-D signals

A video is a space-time volume $\ell[n,m,t]$ of size $N\times M\times P$. The temporal axis behaves very differently from space: objects do not blink in and out — they trace **continuous trajectories**, appearing/disappearing only through occlusion or scene boundaries. Looking at a spatial slice ($t$ = const) gives a normal image; a slice at $m$ = const gives an "$x$–$t$" **epipolar-plane image** in which a static object is a vertical line and a moving object is a slanted band whose **slope encodes its speed**.

### 19.2 The global-motion model and brightness constancy

A globally translating image with constant velocity $(v_x,v_y)$ is:

$$
\ell(x,y,t) = \ell_0(x-v_x t,\; y-v_y t)
$$

Symbols: $\ell_0(x,y)=\ell(x,y,0)$ = reference frame; $(v_x,v_y)$ = velocity in pixels/frame; $t$ = time. This encodes the **constant-brightness assumption** — a pixel's value is carried unchanged along its motion path — which is the seed hypothesis of essentially all optical-flow methods (Ch 46–49). A moving object over a static background adds an occlusion mask:

$$
\ell(x,y,t) = b(x,y)\,(1-m_t) + o_t\, m_t,\quad m_t=m(x-v_xt,y-v_yt)
$$

with $b$ = background, $o$ = object, $m$ = binary object mask moving with the object.

### 19.3 Fourier signature of motion: the "delta wall"

Using the shift property, the 3-D FT of a globally moving image is:

$$
\mathscr{L}(w_x,w_y,w_t) = \mathscr{L}_0(w_x,w_y)\,\delta(w_t + v_x w_x + v_y w_y)
$$

Symbols: $\mathscr{L}_0$ = 2-D FT of $\ell_0$; $(w_x,w_y,w_t)$ = spatial/temporal radian frequencies; $\delta$ = Dirac delta. **Why it matters:** all spectral energy of a rigidly translating scene lies on the **tilted plane** $w_t = -(v_x w_x + v_y w_y)$ through the origin. Speed = the tilt of this plane. This is the frequency-domain basis of motion estimation and of orientation-tuned (motion-energy) models of biological vision.

### 19.4 Causality and IIR difference equations

Temporal filters add one attribute images don't have: **causality**. A filter is causal if $h[n,m,t]=0$ for all $t<0$ (output depends only on past/present) — required for streaming/real-time processing; a non-causal filter (like a symmetric Gaussian in time) must be truncated and delayed. Some temporal filters are best written as **difference equations** rather than finite convolutions, e.g. the leaky integrator:

$$
\ell_{out}[n,m,t] = \ell_{in}[n,m,t] + \alpha\,\ell_{out}[n,m,t-1]
$$

Its impulse response is infinite (IIR):

$$
h[n,m,t] = \alpha^{t}\,\delta[n,m]\,u[t],\qquad u[t]=\begin{cases}0 & t<0\\ 1 & t\ge 0\end{cases}
$$

Symbols: $\alpha$ = decay/feedback coefficient; $u[t]$ = Heaviside step. **Stability:** bounded input → bounded output iff $|\alpha|<1$ (otherwise $\alpha^t$ diverges). This is exactly the exponential-moving-average temporal smoother and the conceptual ancestor of the recurrent state update in RNNs (Ch 25).

### 19.5 Spatiotemporal Gaussian and motion-tuned blur

The separable 3-D Gaussian (independent spatial width $\sigma_x$ and temporal width $\sigma_t$, since their units differ) is:

$$
g(x,y,t;\sigma_x,\sigma_t) = \frac{1}{(2\pi)^{3/2}\sigma_x^2\sigma_t}\exp\!\left(-\frac{x^2+y^2}{2\sigma_x^2}\right)\exp\!\left(-\frac{t^2}{2\sigma_t^2}\right)
$$

Skewing it by a velocity produces a filter that **keeps objects moving at **$(v_x,v_y)$** sharp while motion-blurring everything else** (as if the camera tracked that object):

$$
g_{v_x,v_y}(x,y,t) = g(x-v_x t,\, y-v_y t,\, t)
$$

### 19.6 Temporal derivatives and the nulling filter (full derivation)

A discrete temporal derivative $\ell[n,m,t]-\ell[n,m,t-1]$ localizes motion. The Gaussian temporal derivative is $g_t=\frac{-t}{\sigma_t^2}g$, and the full spatiotemporal gradient is:

$$
\nabla g = (g_x,g_y,g_t) = \left(\frac{-x}{\sigma^2},\frac{-y}{\sigma^2},\frac{-t}{\sigma_t^2}\right)g(x,y,t)
$$

For a moving image, differentiating $\ell(x,y,t)=\ell_0(x-v_xt,y-v_yt)$ in time gives the **brightness-constancy / motion-constraint equation**:

$$
\frac{\partial \ell}{\partial t} = -v_x\frac{\partial \ell_0}{\partial x} - v_y\frac{\partial \ell_0}{\partial y}
$$

i.e. $\ell_t + v_x\ell_x + v_y\ell_y = 0$ — the same equation that gives the optical-flow aperture problem. Therefore the directional spatiotemporal derivative along $(1,v_x,v_y)$

$$
h(x,y,t;v_x,v_y) = g_t + v_x g_x + v_y g_y = \nabla g\cdot(1,v_x,v_y)^{\top}
$$

convolved with a scene moving at exactly $(v_x,v_y)$ yields **zero**:

$$
\ell_0(x-v_xt,y-v_yt)\circ h = \left(\ell_{0,t}+v_x\ell_{0,x}+v_y\ell_{0,y}\right)\circ g = 0
$$

This is the **nulling filter** (Darrell & Simoncelli 1993): it cancels any object at the tuned velocity. It works locally even with multiple objects at different speeds, and even for sums of transparent moving layers, because each small patch contains a single dominant velocity.

```python
import torch, torch.nn.functional as F

def st_gaussian_grad(sigma=1.5, sigma_t=1.5, r=3):
    ax = torch.arange(-r, r+1, dtype=torch.float32)
    X, Y, T = torch.meshgrid(ax, ax, ax, indexing='ij')
    g = torch.exp(-(X**2+Y**2)/(2*sigma**2) - T**2/(2*sigma_t**2))
    g /= g.sum()
    gx, gy, gt = (-X/sigma**2)*g, (-Y/sigma**2)*g, (-T/sigma_t**2)*g
    return gx, gy, gt

def nulling_filter(vx, vy, **kw):
    gx, gy, gt = st_gaussian_grad(**kw)
    return gt + vx*gx + vy*gy       # zero-response to motion (vx,vy)

def apply_st(vol, h):               # vol: (B,1,T,H,W), h: (t,y,x)
    k = h.flip(0,1,2).unsqueeze(0).unsqueeze(0)
    return F.conv3d(vol, k, padding='same')
```

### Why it matters / connections

The motion-constraint equation derived here is literally the Lucas–Kanade / Horn–Schunck optical-flow equation (Ch 48). The leaky-integrator IIR filter is the bridge to recurrent temporal models (Ch 25), and the tilted-plane spectrum is why motion can be detected by banks of oriented 3-D filters (motion energy).

### Self-check (Ch 19)

5. In an $x$–$t$ slice, how do you read off an object's speed and direction from the band it traces?
6. Derive $\ell_t + v_x\ell_x + v_y\ell_y = 0$ from the global-motion model and explain the aperture problem it implies.
7. Why must a real-time temporal Gaussian be truncated and delayed, and what is the cost?
8. For the leaky integrator, prove BIBO stability requires $|\alpha|<1$.

## Chapter 20 — Image Sampling and Aliasing

### 20.1 The problem

Sampling converts a continuous signal $\ell(t)$ into $\ell[n]=\ell(nT_s)$ with **sampling period** $T_s$. There is a tension: small $T_s$ preserves information but costs memory/compute; large $T_s$ is cheap but loses information. The whole chapter answers: *how much information is lost, and how do we avoid artifacts?*

### 20.2 Aliasing, intuitively

Sample $\ell(t)=\cos(wt)$, $w=18\pi$ (9 periods on $[0,1]$) at $T_s=1/11$. Even though there are "more samples than periods," the samples are consistent with **infinitely many** continuous waves. Under the perceptual/mathematical **slow-and-smooth prior** (Weiss & Adelson), the reconstruction picks the lowest-frequency wave — a *different, slower* cosine. This frequency confusion is **aliasing**. In 2-D (the zebra example) it makes high-frequency stripes reverse orientation and, at low resolution, "a zebra starts looking like a cow."

### 20.3 The Sampling (Nyquist–Shannon) theorem

A signal is **band-limited** if $\mathscr{L}(w)=0$ for $|w|>w_{max}$. It can be perfectly reconstructed iff the sampling frequency $w_s=2\pi/T_s$ satisfies:

$$
w_s > 2\,w_{max} \quad\Longleftrightarrow\quad T_s < \frac{\pi}{w_{max}}\quad\Longleftrightarrow\quad T_s < \frac{T_{min}}{2}
$$

Symbols: $w_{max}$ = highest frequency present; $w_s$ = sampling (radian) frequency; $T_{min}=2\pi/w_{max}$ = period of the fastest component.

**Why — the delta-train derivation.** Model sampling as multiplication by a Dirac comb:

$$
\delta_{T_s}(t)=\sum_{n=-\infty}^{\infty}\delta(t-nT_s),\qquad \ell_\delta(t)=\ell(t)\,\delta_{T_s}(t)=\sum_n \ell[n]\,\delta(t-nT_s)
$$

The FT of a comb is another comb, $\delta_{T_s}(t)\xrightarrow{\mathscr F}\frac{2\pi}{T_s}\,\delta_{2\pi/T_s}(w)$. Since multiplication in time is convolution in frequency, the sampled spectrum is a **sum of shifted copies (replicas)** of the original spectrum:

$$
\mathscr{L}_\delta(w) = \frac{2\pi}{T_s}\sum_{k=-\infty}^{\infty}\mathscr{L}\!\left(w-k\frac{2\pi}{T_s}\right)
$$

If $T_s$ is small the replicas are far apart; if $T_s$ is too large they **overlap**, and high frequencies of one replica corrupt the low frequencies of the next — that overlap *is* aliasing. The Nyquist limit is precisely the $T_s$ at which the $w_{max}$ edge of one replica meets its neighbor.

### 20.4 Reconstruction: ideal sinc vs. local kernels

When Nyquist holds, recover $\ell(t)$ by keeping only the central replica — an ideal low-pass box in frequency, whose impulse response is the **sinc**:

$$
H(w)=\begin{cases}\tfrac{T_s}{2\pi}, & |w|\le \pi/T_s\\ 0,&\text{else}\end{cases}\qquad h(t)=\operatorname{sinc}(t/T_s),\quad \operatorname{sinc}(t)=\frac{\sin(\pi t)}{\pi t}
$$

giving the **Whittaker–Shannon interpolation** (optimal in $L_2$ under slow-and-smooth):

$$
\widetilde\ell(t)=\sum_{n=-\infty}^{\infty}\ell[n]\,\operatorname{sinc}\!\left(\frac{t-nT_s}{T_s}\right)
$$

The sinc has **infinite support** (every sample contributes to every point), so in practice we use **local** kernels: nearest-neighbor = convolution with a box of width $T_s$; **linear** interpolation = convolution with a triangle of width $2T_s$ (= box ∘ box); Lanczos = the central sinc lobe; cubic = a wider polynomial kernel. Each trades reconstruction accuracy for locality/speed. (Note: a signal cannot be both time-limited and band-limited — finite images are handled by zero-padding to infinity.)

### 20.5 2-D sampling lattices — hexagonal is optimal

A general 2-D lattice is $\ell_s[n,m]=\ell(\mathbf{A}\mathbf{n})$ with lattice matrix $\mathbf{A}=\begin{bmatrix}a&b\\ c&d\end{bmatrix}$. The sampled spectrum replicates on the **reciprocal lattice**:

$$
\mathscr{L}_\delta(\mathbf w)=\frac{(2\pi)^2}{|\mathbf A|}\sum_{\mathbf k\in\mathbb Z^2}\mathscr{L}\!\left(\mathbf w-2\pi\mathbf A^{-1}\mathbf k\right)
$$

To avoid aliasing for a fixed number of samples we want the alias-free (Voronoi) region as large as possible. The optimum is the **regular hexagonal lattice**, which gains ≈10% resolution over the rectangular grid for the same sample count — which is why cone photoreceptors in the fovea are hexagonally packed. Computer vision still uses rectangular grids for convenience.

### 20.6 Anti-aliasing filter

You cannot remove aliasing after sampling. You must **low-pass filter the continuous signal before sampling** so it becomes band-limited to the new Nyquist. The anti-aliasing filter (ideally the box $H(w)$ above, in practice a Gaussian/binomial) discards the high frequencies that would otherwise fold down — you still lose the high-frequency detail, but you no longer *corrupt* the low frequencies. In the zebra figure, anti-aliasing keeps the low-frequency DFT stable across resolutions ("no longer a cow"). **This is exactly why every correct downsampling — including strided pooling in CNNs — should blur first** (see the Latest Research section below).

### 20.7 Spatiotemporal sampling: rolling shutter and Moiré

The same replication analysis extends to $N$-D. **Temporal** aliasing is the wagon-wheel illusion (fan/wheel appears to spin backwards) and needs a temporal anti-aliasing (motion-blur) filter. **Global shutter** exposes all pixels simultaneously; **rolling shutter** samples rows at slightly different times, distorting fast-moving objects. **Moiré** patterns are aliasing-related interference between two overlaid high-frequency patterns.

```python
import torch, torch.nn.functional as F

def binomial_kernel_2d(n=2):
    row = torch.tensor([1.0]); 
    for _ in range(n): row = F.conv1d(row.view(1,1,-1), torch.ones(1,1,2)).view(-1)
    row = row/row.sum(); k = torch.outer(row, row)
    return k/k.sum()

def antialiased_downsample(img, factor=2, n=2):        # img: (B,C,H,W)
    k = binomial_kernel_2d(n).to(img)                  # low-pass BEFORE decimation
    k = k.expand(img.shape[1],1,*k.shape)
    blurred = F.conv2d(img, k, padding='same', groups=img.shape[1])
    return blurred[..., ::factor, ::factor]            # then subsample => no aliasing

def sinc_interp(samples, T_s, t_query):                # 1-D Whittaker–Shannon
    n = torch.arange(len(samples), dtype=torch.float32)
    W = torch.sinc((t_query[:,None]/T_s) - n[None,:])   # torch.sinc = sin(pi x)/(pi x)
    return W @ samples
```

### Why it matters / connections

Sampling theory governs every resize, pyramid level (Ch 21–23), CNN stride/pool, and video frame rate. Aliasing that is *not* controlled shows up as texture shimmer, checkerboard artifacts in GANs, and shift-variance in classifiers; controlling it (blur-before-subsample) restores shift-equivariance. Conversely, aliasing carries recoverable high-frequency information exploited by **super-resolution**.

### Self-check (Ch 20)

9. Derive $\mathscr{L}_\delta(w)=\frac{2\pi}{T_s}\sum_k \mathscr{L}(w-k\,2\pi/T_s)$ and point to exactly where aliasing enters.
10. Why is the ideal reconstruction kernel a sinc, and what practical problem does its infinite support cause?
11. Why does hexagonal sampling beat rectangular by ~10%, and what reciprocal-lattice quantity are we maximizing?
12. A CNN downsamples with stride-2 conv and no blur — what artifact appears and how does an anti-aliasing (blur) filter fix it?
13. Explain the wagon-wheel illusion as temporal aliasing using the replica picture.

## Supplementary resources for this block

Day 6 (Ch 18–20) has **no dedicated adensur/Udemy mapping** in the study plan (those are assigned to neighbouring blocks). No paper deep-dive is scheduled for Day 6 either. Conceptual bridges worth keeping in mind:

- **Udemy — Intro module (pixels, image scaling/interpolation)** — [course link](https://www.udemy.com/course/mastering-computer-vision-from-pixel-to-detection-to-gen-cv/). The interpolation methods covered there (nearest, bilinear, bicubic) are exactly the *local reconstruction kernels* of §20.4 — bilinear = triangle kernel = box∘box, bicubic ≈ a truncated-sinc-like kernel. This is formally revisited in **Day 7 (Ch 21–23, resampling & pyramids)**, so treat today's sampling theory as the "why" behind Day 7's "how."
- **adensur — **`**03_convolutions**` — [repo](https://github.com/adensur/blog/tree/main/computer_vision_zero_to_hero). The derivative operators of Ch 18 are just fixed convolution kernels; this connects to how a CNN *learns* such kernels (Day 8, Ch 24).
- Forward pointer: the **anti-aliasing-before-downsampling** rule (§20.6) is the theoretical justification for "BlurPool"/anti-aliased CNNs, which reappears directly in today's Latest-Research picks.

## Latest CV Research — 2026-07-06

*Recent 2026 work, prioritized toward this block's themes (filtering, derivatives, sampling/aliasing, temporal/motion). Note: strict "last-2-weeks" arXiv listings could not be fully verified in this automated run; the items below are the most recent, most relevant works surfaced, spanning CVPR 2026 and early-2026 arXiv/journal papers.*

14. **Optical Flow Matching (OFM): Reframing Optical Flow as Continuous Transport Dynamics** — CVPR 2026 (Oral). Recasts flow from discrete frame-to-frame correspondence to a learned time-dependent **velocity field** that transports pixel coordinates along continuous trajectories, with a lightweight "Triangle Velocities Synergy" for stable velocity construction. *Why it matters:* it is the modern, learned successor to Ch 19's brightness-constancy velocity field — motion as a continuous flow rather than a per-frame displacement. [link](https://cvpr.thecvf.com/virtual/2026/oral/40348)
15. **Flow3r: Factored Flow Prediction for Scalable Visual Geometry Learning** — CVPR 2026. Predicts optical flow in a factored form that scales to large-scale visual-geometry pretraining, tying 2-D motion to 3-D structure. *Why it matters:* shows the temporal-derivative/flow machinery of Ch 19 being used as a *representation-learning* signal, not just a motion output. [link](https://openaccess.thecvf.com/content/CVPR2026/papers/Cong_Flow3r_Factored_Flow_Prediction_for_Scalable_Visual_Geometry_Learning_CVPR_2026_paper.pdf)
16. **Anti-Aliasing for Downsampling in CNNs Based on Gaussian Filter Convolution (GFC)** — *Electronics*, Feb 2026. Inserts a Gaussian low-pass (learnable width) before every stride/pool operation to suppress aliasing during CNN downsampling, improving shift-consistency and accuracy. *Why it matters:* a direct, current application of Ch 20 §20.6 — "blur before you subsample" — inside deep networks. [link](https://doi.org/10.3390/electronics15040780)
17. **Uniform Resampling vs. Image Blur: Aliasing Approximation via Isotropic Gaussian Filtering** — arXiv 2502.11605 (2026). Analyzes how downsample→interpolate→upsample pipelines relate to naïve subsampling, modeling the induced distortion as an equivalent isotropic Gaussian blur. *Why it matters:* quantifies exactly the reconstruction/anti-aliasing trade-off of §20.4–20.6 for practical resizing. [link](https://arxiv.org/html/2502.11605v1)
18. **Spectral-Adaptive Modulation Networks for Visual Perception** — arXiv 2503.23947 (2026). Builds token-mixing/backbone blocks that adaptively modulate content in the **frequency domain**, a learned generalization of the DFT-based filter analysis used throughout Ch 18–20. *Why it matters:* connects classical Fourier filter design (derivatives, band-pass LoG, anti-aliasing) to modern frequency-aware architecture design. [link](https://arxiv.org/pdf/2503.23947)
19. **Diffusion Model as a Generalist Segmentation Learner** — arXiv 2604.24575 (2026). Uses a single diffusion backbone as a general dense-prediction/segmentation learner across tasks. *Why it matters:* a look ahead to the generative-and-representation blocks (Days 10–11), and a reminder that dense per-pixel prediction still rests on the low-level filtering/sampling foundations studied today. [link](https://arxiv.org/pdf/2604.24575)