---
base: "[[01.1 Computer Vision — Foundations Study Tracker.base]]"
Key takeaways: "LTI systems are exactly convolutions: translation invariance collapses the dense linear map h[n,k] to h[n−k] (Toeplitz/weight sharing — the CNN inductive bias); framework 'conv' is really cross-correlation. The DFT diagonalizes all LTI operators (convolution ↔ pointwise product of transfer functions; phase carries WHERE, amplitude ~1/f in natural images). For blurring: box is cheap but ripples (discrete sinc), continuous Gaussian is ideal (separable, semigroup σ₃²=σ₁²+σ₂²) but breaks under sampling, and binomial b_n (Pascal rows) restores exact discrete closure b_n∘b_m=b_{n+m} and kills Nyquist — the right antialiasing filter before downsampling."
Day: 5
Status: Done
Reading done: true
Chapters: Ch 15–17
Self-check done: true
Date: 2026-07-22
Part:
  - Image-Processing
  - Linear-Filters
Questions / Follow-ups: ""
---
**Reading checklist**

- [ ] [15 Linear Image Filtering](https://visionbook.mit.edu/linear_image_filtering.html)
- [ ] [16 Fourier Analysis](https://visionbook.mit.edu/image_processing_fourier.html)
- [ ] [17 Blur Filters](https://visionbook.mit.edu/blurring_2.html)

## Notes

## Self-check

# Day 5 Study Report — Image Filtering, Fourier Analysis & Blur (2026-07-05, JST)

Chapters covered: **Ch 15 Linear Image Filtering · Ch 16 Fourier Analysis · Ch 17 Blur Filters** ([visionbook.mit.edu](http://visionbook.mit.edu/), read in full).

## Chapter 15 — Linear Image Filtering

**Intuition.** Early vision (retina → LGN → V1) is surprisingly well modeled by linear filters. The goal of this chapter is to build the simplest possible machinery — linear, translation-invariant operations — that extracts low-level features (edges, blobs, local averages) and removes nuisance variability, before any learning enters the picture.

### Signals and images

A signal is a measurement of a physical quantity as a function of another (time, space). Sampling turns a continuous signal into a discrete one:

$\ell[n] = \ell(n\,\Delta T)$

where $\ell(t)$ is the continuous signal, $\ell[n]$ its discrete samples ($n \in \mathbb{Z}$), and $\Delta T$ the sampling period. Convention: parentheses = continuous, brackets = discrete. An image is $\boldsymbol\ell \in \mathbb{R}^{M \times N}$ with pixel $\ell[n,m]$, $n$ horizontal, $m$ vertical.

Useful scalar descriptors — the DC value (mean) of a finite signal of length $N$:

$\mu = \frac{1}{N}\sum_{n=0}^{N-1}\ell[n]$

and the energy:

$E = \sum_{n=-\infty}^{\infty}\left|\ell[n]\right|^2$

**Why these definitions:** the terminology comes from physics (DC = direct current; energy of a voltage across a resistor is $\int v(t)^2 dt / R$). The DC value matters because filters rescale it (see DC gain, Ch 17); the energy matters because Parseval's theorem (Ch 16) says it is preserved under the Fourier change of basis. The squared L2 distance $D^2 = \frac{1}{N}\sum_n |\ell_1[n]-\ell_2[n]|^2$ is a poor perceptual metric on raw pixels — a recurring theme; modern metrics compute L2 in a learned representation space.

### Linear systems

A system $f$ is linear iff

$f(\boldsymbol\ell_1 + \boldsymbol\ell_2) = f(\boldsymbol\ell_1) + f(\boldsymbol\ell_2), \qquad f(a\boldsymbol\ell) = a f(\boldsymbol\ell)$

The most general linear system maps every output sample to a weighted sum of ALL input samples:

$\ell_{\text{out}}[n] = \sum_{k=0}^{N-1} h[n,k]\,\ell_{\text{in}}[k]$

where $h[n,k]$ is the weight from input position $k$ to output position $n$ — in matrix form $\boldsymbol\ell_{\text{out}} = \mathbf{H}\boldsymbol\ell_{\text{in}}$ with $\mathbf{H} \in \mathbb{R}^{M\times N}$. This is exactly a **fully connected layer** with weights $\mathbf{W}=\mathbf{H}$. In 2D: $\ell_{\text{out}}[n,m] = \sum_{k,l} h[n,m,k,l]\,\ell_{\text{in}}[k,l]$.

Perhaps surprising: rotation, scaling, color→grayscale, and defocus are ALL linear (each output pixel is a fixed linear combination of input pixels). But only some of them are translation invariant.

### LTI systems → convolution

Because we typically don't know WHERE in an image an object will appear, we want the same processing at every pixel. Add translation equivariance to linearity:

$\ell_{\text{out}}[n-n_0, m-m_0] = f\left(\ell_{\text{in}}[n-n_0, m-m_0]\right)$

**Why this forces convolution:** translation invariance means the weight $h[n,k]$ can only depend on the relative offset $n-k$, so $h[n,k] = h[n-k]$ — the dense matrix $\mathbf{H}$ collapses to a Toeplitz (banded, weight-shared) matrix. The general linear map becomes the convolution:

$\ell_{\text{out}}[n] = h[n] \circ \ell_{\text{in}}[n] = \sum_{k=-\infty}^{\infty} h[n-k]\,\ell_{\text{in}}[k]$

and in 2D:

$\ell_{\text{out}}[n,m] = \sum_{k,l} h[n-k, m-l]\,\ell_{\text{in}}[k,l]$

Operationally: mirror the kernel, slide it, take inner products. This is precisely why a conv layer is a fully connected layer with weight sharing and locality — the two inductive biases of CNNs. Defocus IS a convolution; rotation is linear but NOT translation invariant (each output location needs a location-specific rule), so it cannot be a convolution.

**Properties of convolution** (each provable by change of summation variable):

- Commutative: $h \circ \ell = \ell \circ h$ (swap $k \to n-k'$). Correlation is NOT commutative.
- Associative: $\ell_1 \circ (\ell_2 \circ \ell_3) = (\ell_1 \circ \ell_2) \circ \ell_3$ — lets you fuse cascaded filters into one kernel (boundary handling can break this in practice).
- Distributive over sum.
- Shift: $h[n] \circ \ell[n-n_0] = h[n-n_0]\circ\ell[n] = \ell_{\text{out}}[n-n_0]$ — the formal statement of equivariance.
- Support: length-$N$ conv length-$M$ gives length $L \le N+M-1$.
- Identity: the impulse $\delta[n]$ ($1$ at $n=0$, else $0$), $\delta \circ \ell = \ell$.

In the continuous domain sums become integrals: $\ell_{\text{out}}(t) = \int h(t-\tau)\ell_{\text{in}}(\tau)d\tau$, with the Dirac delta satisfying the sampling property $\int \ell(t)\delta(t-a)dt = \ell(a)$, $\delta(at) = \delta(t)/|a|$.

### Boundaries and circular convolution

Convolving near the edge needs pixels that don't exist. Options: **zero padding** (default in neural nets; darkens borders — worst reconstruction error), **circular padding** (wraps around; makes the signal periodic — analytically convenient, this is what the DFT implicitly assumes), **mirror padding** (reflect; usually lowest error vs. ground truth), **repeat padding**. Circular convolution of two length-$N$ signals:

$\ell_{\text{out}}[n] = h[n] \circ_N \ell_{\text{in}}[n] = \sum_{k=0}^{N-1} h[(n-k)_N]\,\ell_{\text{in}}[k]$

where $(n)_N$ is the modulo. **Why we care:** the DFT convolution theorem (Ch 16) is exact only for circular convolution.

### Cross-correlation vs convolution

Cross-correlation does NOT mirror the kernel:

$\ell_{\text{out}}[n,m] = \ell_{\text{in}} \star h = \sum_{k,l=-N}^{N} \ell_{\text{in}}[n+k, m+l]\,h[k,l]$

Convolution uses $\ell_{\text{in}}[n-k,m-l]$. They coincide when the kernel is centrally symmetric. **Deep-learning gotcha: what every framework (PyTorch **`**Conv2d**`** included) calls "convolution" is actually cross-correlation.** Since kernels are learned, the distinction is usually irrelevant — the network just learns the flipped filter — but it matters when you port a hand-designed kernel or reason about associativity/commutativity (correlation has neither).

**Template matching / normalized cross-correlation.** Raw correlation with a template fails because bright regions score high regardless of content. Fix: normalize the template $\hat h$ (zero mean, unit norm) and divide by the local patch standard deviation:

$\ell_{\text{out}}[n,m] = \frac{1}{\sigma[n,m]}\sum_{k,l=-N}^{N} \ell_{\text{in}}[n+k,m+l]\,\hat h[k,l]$

with local mean and variance

$\mu[n,m] = \frac{1}{(2N+1)^2}\sum_{k,l=-N}^{N}\ell_{\text{in}}[n+k,m+l], \qquad \sigma^2[n,m] = \frac{1}{(2N+1)^2}\sum_{k,l=-N}^{N}\left(\ell_{\text{in}}[n+k,m+l]-\mu[n,m]\right)^2$

**Why:** this makes the response a cosine similarity between the patch and the template — invariant to local brightness and contrast. Still fragile to rotation/scale/appearance changes — the historical motivation for learned features.

### System identification

For an LTI system, the response to an impulse IS the kernel: input $\delta[n]$ → output $h[n]$; input $\delta[n-n_0]$ → $h[n-n_0]$. So $h$ is called the **impulse response**. Beautiful acoustics example: clap in a room ≈ impulse; the echoes you hear are the room's impulse response $h(t) = a_0\delta(t) + a_1\delta(t-T_1) + a_2\delta(t-T_2) + a_3\delta(t-T_3)$, and any sound from that spot reaches you as $\ell_{\text{in}}(t) \circ h(t)$. Same idea used to probe unknown systems, including trained networks.

```python
import torch
import torch.nn.functional as F

# --- convolution vs cross-correlation ---
img = torch.rand(1, 1, 64, 64)                      # (B, C, H, W)
kernel = torch.tensor([[0., 1., 0.],
                       [1., -4., 1.],
                       [0., 1., 0.]]).view(1, 1, 3, 3)

xcorr = F.conv2d(img, kernel, padding=1)            # PyTorch "conv" = cross-correlation
true_conv = F.conv2d(img, kernel.flip(-1).flip(-2), padding=1)  # flip kernel -> true convolution

# mirror padding (best boundary behavior), then valid conv
img_pad = F.pad(img, (1, 1, 1, 1), mode='reflect')
out = F.conv2d(img_pad, kernel)                     # same size as img

# --- normalized cross-correlation (template matching) ---
def normalized_xcorr(image, templ, eps=1e-8):
    """image: (1,1,H,W), templ: (1,1,h,w) -> NCC map (1,1,H',W')"""
    t = templ - templ.mean()
    t = t / (t.norm() + eps)                        # zero-mean, unit-norm template
    k = t.numel()
    ones = torch.ones_like(t)
    s1 = F.conv2d(image, ones)                      # local sum
    s2 = F.conv2d(image ** 2, ones)                 # local sum of squares
    var = (s2 - s1 ** 2 / k).clamp(min=0)           # k * local variance
    resp = F.conv2d(image, t)                       # correlation with normalized template
    return resp / (var.sqrt() + eps)
```

**Why it matters / connections.** Convolution = weight sharing = the core inductive bias of CNNs (Day 8, Ch 24); the impulse response and matrix view $\mathbf{H}\boldsymbol\ell$ connect directly back to cameras-as-linear-systems (Day 2, Ch 7); normalized correlation is the ancestor of every siamese/matching network and of attention scores as dot products.

## Chapter 16 — Fourier Analysis

**Intuition.** "Sharp" vs "blurry" is too vague. The Fourier transform gives a precise language: decompose the image into waves of different **spatial frequencies** and describe exactly which frequencies a filter passes, attenuates or kills. The deep reason Fourier is THE tool for LTI systems: complex exponentials are the **eigenfunctions of every translation-invariant linear operator** — feed a wave into an LTI filter and you get the same wave back, scaled and phase-shifted, never a new frequency.

### Fourier series → waves

Fourier (1822): any function on $t \in (0,\pi)$ is an infinite sum of harmonically related sinusoids

$\ell(t) = a_1 \sin(t) + a_2 \sin(2t) + a_3 \sin(3t) + \dots, \qquad a_n = \frac{2}{\pi}\int_0^{\pi} \ell(t)\sin(nt)\,dt$

e.g. the ramp $\tfrac{1}{2}t = \sin(t) - \tfrac{1}{2}\sin(2t) + \tfrac{1}{3}\sin(3t) - \dots$ Adding higher-frequency terms adds finer detail — the essence of coarse-to-fine. Key reframing: the coefficient sequence $a_n$ is a **change of representation** of the same signal.

Discrete waves must fit the support: on $N$ samples the sine/cosine of integer frequency $k$ (cycles per support) are

$s_k[n] = \sin\left(\frac{2\pi}{N}kn\right), \qquad c_k[n] = \cos\left(\frac{2\pi}{N}kn\right)$

with $c_{N-k}=c_k$, $s_{N-k}=-s_k$ — so only $k \in [0, N/2]$ are distinct (this is where aliasing will come from on Day 6). The cleaner object is the complex exponential, which packs cosine and sine via Euler's formula $e^{ja} = \cos a + j\sin a$:

$e_{u,v}[n,m] = \exp\left(2\pi j\left(\frac{un}{N} + \frac{vm}{M}\right)\right)$

$u, v$ = spatial frequencies along $n, m$. These are **separable** ($e_{u,v}[n,m] = e_u[n]e_v[m]$) and form an **orthogonal basis**:

$\langle e_{u,v}, e_{u',v'}\rangle = \sum_{n=0}^{N-1}\sum_{m=0}^{M-1} e_{u,v}[n,m]\,e^*_{u',v'}[n,m] = MN\,\delta[u-u']\,\delta[v-v']$

**Why complex exponentials and not plain sines:** (1) they diagonalize translation — shifting multiplies by a unit-modulus scalar, whereas sines mix into cosines; (2) orthogonality gives a unique, invertible decomposition with a trivial inverse formula.

### The Discrete Fourier Transform

$\mathscr{L}[u,v] = \mathcal{F}\{\ell[n,m]\} = \sum_{n=0}^{N-1}\sum_{m=0}^{M-1} \ell[n,m]\exp\left(-2\pi j\left(\frac{un}{N} + \frac{vm}{M}\right)\right)$

$\ell[n,m] = \mathcal{F}^{-1}\{\mathscr{L}[u,v]\} = \frac{1}{NM}\sum_{u=0}^{N-1}\sum_{v=0}^{M-1} \mathscr{L}[u,v]\exp\left(+2\pi j\left(\frac{un}{N} + \frac{vm}{M}\right)\right)$

**Why the inverse works:** apply $\frac{1}{NM}\sum_{u,v} e^{+\dots}$ to the forward transform and orthogonality kills every cross term — the $\frac{1}{NM}$ exactly cancels the $MN$ from the basis norm. Both $\mathscr{L}$ and the reconstructed $\ell$ are implicitly **periodic** with period $(N, M)$ — the DFT always "sees" the periodized image (hence circular convolution). Re-indexing frequencies to $[-N/2, N/2]$ puts DC at the center (what `fftshift` does): conjugate-pair coefficients with equal amplitudes sum to cosines, opposite amplitudes to sines. The decomposition is unique. Cost: FFT (Cooley–Tukey) computes it in $O(N\log N)$ instead of $O(N^2)$.

Matrix view: the DFT is the linear map $\mathbf{F}_{u,n} = \exp(-2\pi j\frac{un}{N})$, a symmetric matrix with $\mathbf{F}^{-1} = \mathbf{F}^*$ (unitary up to $1/N$) — literally a fixed fully-connected layer with complex weights.

Polar decomposition of the transform:

$\mathscr{L}[u,v] = A[u,v]\exp\left(j\,\theta[u,v]\right)$

$A$ = amplitude (how much of each wave), $\theta$ = phase (where the waves sit).

### Transforms worth memorizing

$\delta[n,m] \xrightarrow{\mathscr{F}} 1$

$\cos\left(2\pi\left(\frac{u_0 n}{N}+\frac{v_0 m}{M}\right)\right) \xrightarrow{\mathscr{F}} \frac{1}{2}\left(\delta[u-u_0, v-v_0] + \delta[u+u_0, v+v_0]\right)$

$\sin\left(2\pi\left(\frac{u_0 n}{N}+\frac{v_0 m}{M}\right)\right) \xrightarrow{\mathscr{F}} \frac{1}{2j}\left(\delta[u-u_0, v-v_0] - \delta[u+u_0, v+v_0]\right)$

The box function $\text{box}_L[n] = 1$ for $-L \le n \le L$, else $0$ (length $2L+1$). Its DFT, derived via the geometric series $\sum_{n=-L}^{L} a^n = \frac{a^{-(2L+1)/2}-a^{(2L+1)/2}}{a^{-1/2}-a^{1/2}}$ with $a = e^{-2\pi j u/N}$:

$\text{box}_L[n] \xrightarrow{\mathscr{F}} \frac{\sin\left(\pi u(2L+1)/N\right)}{\sin\left(\pi u/N\right)}$

— the **discrete sinc**. It oscillates and has ripples, which is exactly why the box is a mediocre low-pass filter (Ch 17). General pattern: wider support in one domain ⇔ narrower in the other.

### DFT properties

- **Linearity:** $\alpha\ell_1 + \beta\ell_2 \xrightarrow{\mathscr{F}} \alpha\mathscr{L}_1 + \beta\mathscr{L}_2$.
- **Separability:** $\ell[n,m] = \ell_1[n]\ell_2[m] \Rightarrow \mathscr{L}[u,v] = \mathscr{L}_1[u]\mathscr{L}_2[v]$ — 2D FFTs factor into 1D passes.
- **Parseval / Plancherel** (energy is basis-independent):

$\sum_{n,m} \|\ell[n,m]\|^2 = \frac{1}{NM}\sum_{u,v}\|\mathscr{L}[u,v]\|^2$

- **Convolution theorem** — the most important property in the chapter:

$\ell_1 \circ_{N,M} \ell_2 \xrightarrow{\mathscr{F}} \mathscr{L}_1[u,v]\,\mathscr{L}_2[u,v]$

**Why:** substitute the convolution into the DFT, swap sums, re-index $n' = n-k$ (legal because everything is periodic) — the exponential factors as $e^{-2\pi j(n'+k)u/N} = e^{-2\pi jn'u/N}e^{-2\pi jku/N}$, splitting the double sum into two independent DFTs. Circular convolution becomes pointwise multiplication. This (a) gives $O(N\log N)$ filtering with large kernels, and (b) is the formal statement that Fourier bases are eigenfunctions of LTI operators: a filter can only reweight existing frequencies, never create new ones. Creating new spectral content requires nonlinearities — one reason deep nets need them.

- **Dual convolution:** $\ell_1\ell_2 \xrightarrow{\mathscr{F}} \frac{1}{NM}\mathscr{L}_1 \circ \mathscr{L}_2$ (masking an image smears its spectrum).
- **Shift:** translating the image only changes phase:

$\ell[n-n_0, m-m_0] \xrightarrow{\mathscr{F}} \mathscr{L}[u,v]\exp\left(-2\pi j\left(\frac{un_0}{N}+\frac{vm_0}{M}\right)\right)$

(exact only for circular shifts; still ≈ true for camera translation — the inverse DFT of the ratio of two shifted images' DFTs is an impulse at the displacement, i.e., phase correlation for global motion estimation).

- **Modulation** (dual of shift): multiplying by a wave translates the spectrum, $\ell[n,m]e^{-2\pi j(u_0n/N + v_0m/M)} \xrightarrow{\mathscr{F}} \mathscr{L}[u-u_0, v-v_0]$.

Family of transforms (know which one a paper uses): DFT (discrete, finite $N$ ↔ discrete frequency); continuous FT $\mathscr{L}(w) = \int \ell(t)e^{-jwt}dt$ with inverse $\frac{1}{2\pi}\int \mathscr{L}(w)e^{jwt}dw$; DTFT for infinite discrete signals $\mathscr{L}(w) = \sum_{n=-\infty}^{\infty}\ell[n]e^{-jwn}$, periodic in $w$ with period $2\pi$.

### Amplitude vs phase; natural image statistics

Classic Oppenheim experiment: swap the DFT amplitude and phase of two images — each result looks like the image that donated the **phase**. Phase encodes WHERE structure is (edges = aligned phases); amplitude encodes how much of each frequency. But amplitude is not uninformative: for periodic textures it can dominate, and it powered early global descriptors (e.g., GIST). A remarkable regularity of natural images:

$A[u,v] \approx \frac{a}{(u^2+v^2)^{b}}$

— the $1/f$-type spectral falloff, used as a prior for denoising and revisited in statistical image models (Day 9, Ch 27).

### Transfer functions of filters

For an LTI filter with kernel $h$: $\mathscr{L}_{\text{out}}[u,v] = H[u,v]\,\mathscr{L}_{\text{in}}[u,v]$, where the **transfer function** $H = \mathcal{F}\{h\}$. In polar form

$H[u,v] = |H[u,v]|\exp\left(j\angle H[u,v]\right)$

$|H|$ = amplitude gain per frequency, $\angle H$ = phase shift, $|H[0,0]|$ = DC gain. Filters classify as low-pass (coarse structure), band-pass (mid-scale), high-pass (fine detail). Two case studies: (1) removing the quasi-periodic columns of the MIT facade by zeroing the harmonic peaks at $u = k\cdot 256/14$ in the DFT — image nearly unchanged otherwise; (2) the human contrast sensitivity function (Campbell–Robson chart): our visual system behaves like a band-pass filter peaking near 6 cycles/degree — psychophysicists identified $|H|$ of human vision by measuring wave visibility, the same system-identification logic as Ch 15.

```python
import torch
import torch.fft as fft

img = torch.rand(256, 256)

# --- DFT, amplitude/phase decomposition ---
L = fft.fft2(img)                       # complex (256,256)
A, theta = L.abs(), L.angle()           # amplitude, phase
recon = fft.ifft2(A * torch.exp(1j * theta)).real   # == img

# --- verify convolution theorem (circular conv == product of DFTs) ---
h = torch.zeros(256, 256)
h[:3, :3] = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
lhs = fft.ifft2(fft.fft2(img) * fft.fft2(h)).real   # filtering via frequency domain

# --- ideal low-pass filtering with a centered radial mask ---
Lc = fft.fftshift(fft.fft2(img))
uu, vv = torch.meshgrid(torch.arange(256) - 128, torch.arange(256) - 128, indexing='ij')
mask = ((uu**2 + vv**2).sqrt() < 30).float()        # keep |freq| < 30
low = fft.ifft2(fft.ifftshift(Lc * mask)).real      # blurry image (with ringing!)

# --- phase-swap experiment ---
img2 = torch.rand(256, 256)
L1, L2 = fft.fft2(img), fft.fft2(img2)
swap = fft.ifft2(L1.abs() * torch.exp(1j * L2.angle())).real  # looks like img2
```

**Why it matters / connections.** The convolution theorem underlies FFT-based fast convolution, deconvolution/deblurring, and analysis of aliasing (Day 6–7); positional encodings in transformers are Fourier features; the "too global" weakness of Fourier (frequency info but no localization) is the exact motivation for filter banks, wavelets, and pyramids (Day 7) — and, ultimately, for learned local features.

## Chapter 17 — Blur Filters

**Intuition.** Blur filters are low-pass filters: they attenuate high spatial frequencies, trading detail for noise removal, scale selection, and safe resampling. Three families — box, Gaussian, binomial — form a progression: simplest → best theoretical properties → best discrete approximation.

### Box filter

$\text{box}_{N,M}[n,m] = \begin{cases} 1 & -N \le n \le N \text{ and } -M \le m \le M \\ 0 & \text{otherwise} \end{cases}$

Filtering = local sum: $\ell_{\text{out}}[n,m] = \sum_{k=-N}^{N}\sum_{l=-M}^{M}\ell_{\text{in}}[n-k, m-l]$. Separable ($\text{box}_{N,M} = \text{box}_{N,0}\circ\text{box}_{0,M}$); large boxes can be computed in $O(1)$ per pixel with integral images (Viola–Jones).

**DC gain** — the gain for a constant input $\ell = a$:

$\text{DC gain} = \sum_{n,m} h[n,m] = H[0,0]$

For the unnormalized box it is $(2N+1)(2M+1)$. **Why normalize to DC gain 1:** a blur should preserve the mean brightness of the image, so we divide the kernel by the sum of its values.

**Limitations (why the box is a bad blur):**

- Its transfer function is the discrete sinc $\text{Box}_L[u] = \frac{\sin(\pi u(2L+1)/N)}{\sin(\pi u/N)}$ — **not monotonically decreasing**. The Nyquist wave $[1,-1,1,-1,\dots]$ convolved with $[1,1,1]$ passes through unchanged, while some lower frequencies are fully killed. Ripples ⇒ artifacts.
- **Box ∘ box ≠ box:** $[1,1,1]\circ[1,1,1] = [1,2,3,2,1]$ (a triangle) — repeated box blurs are not equivalent to one bigger box blur. No semigroup structure.

### Gaussian filter

$g(x; \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{x^2}{2\sigma^2}\right), \qquad g(x, y; \sigma) = \frac{1}{2\pi\sigma^2}\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)$

$\sigma$ sets the spatial extent; normalization makes it integrate to 1; it is positive, symmetric — a **zero-phase filter** (no structure displacement). Discretization: sample within $\pm 3\sigma$ (amplitude ≈ 1% of peak there), drop the constant, renormalize by the sample sum:

$g[n,m;\sigma] = \exp\left(-\frac{n^2+m^2}{2\sigma^2}\right)$

**Separability and cost.** Because $e^{-(x^2+y^2)/2\sigma^2} = e^{-x^2/2\sigma^2}e^{-y^2/2\sigma^2}$:

$g[n,m] \circ \ell = g^x \circ (g^y \circ \ell)$

so a 2D Gaussian blur is two 1D passes: cost drops from $O(N^2)$ to $O(2N)$ multiplies per pixel for an $N\times N$ kernel. The Gaussian is the **only circularly symmetric separable** kernel — why: separability forces $f(x)f(y)$ to depend only on $x^2+y^2$, whose unique solution is the exponential of a quadratic.

**Key properties (continuous domain):**

- Its Fourier transform is again a Gaussian, monotone and radially symmetric — no ripples, ideal blur behavior:

$G(w_x, w_y; \sigma) = \exp\left(-\frac{(w_x^2 + w_y^2)\sigma^2}{2}\right)$

Note the inversion: wide in space ⇔ narrow in frequency ($\sigma \leftrightarrow 1/\sigma$).

- **Semigroup:** $g(\sigma_1) \circ g(\sigma_2) = g(\sigma_3)$ with

$\sigma_3^2 = \sigma_1^2 + \sigma_2^2$

**Why (one line):** multiply the two Gaussian transfer functions — exponents add. This is the foundation of the Gaussian pyramid and scale-space (Day 7, Ch 23): blurring incrementally is the same as blurring once with the right $\sigma$.

- Solution of the heat equation; limit of repeated convolutions of any concentrated kernel (central limit theorem); $\sigma \to 0$ gives the impulse.

**Limitation:** all of this holds in the continuous domain. The **sampled** Gaussian $g_5 = [0.0183, 0.3679, 1.0000, 0.3679, 0.0183]$ ($\sigma^2 = 1/2$) does not satisfy $g_5 circ g_5 = $ sampled Gaussian with $\sigma^2 = 1$; errors accumulate under repeated convolution. And $g_5 \circ [1,-1,1,-1,\dots] \ne 0$ — the discrete Gaussian never fully kills Nyquist, which matters for antialiasing.

### Binomial filters

Fix the discretization problem with integer kernels that are EXACTLY closed under convolution. Define $b_n$ = $n$-fold convolution of $[1,1]$ — rows of Pascal's triangle: $b_1 = [1,1]$, $b_2 = [1,2,1]$, $b_4 = [1,4,6,4,1]$, …

- DC gain of $b_n$ is $2^n$; spatial variance $\sigma^2 = n/4$.
- **Exact semigroup:** $b_n \circ b_m = b_{n+m}$, hence $\sigma_n^2 + \sigma_m^2 = \sigma_{n+m}^2$ — the discrete analogue of the Gaussian property (why: convolving coin-flip distributions adds their variances; CLT explains why $b_n$ → Gaussian).
- The workhorse $b_2 = [1,2,1]$ is odd-length (no half-pixel shift) and has DFT

$B_2[u] = 2 + 2\cos\left(\frac{2\pi u}{N}\right)$

which decreases **monotonically** (no ripples) and is **exactly zero at Nyquist**. Every even binomial is a power: $B_{2n}[u] = (2+2\cos(2\pi u/N))^n$ — real, positive, zero-phase.

- $b_n \circ [1,-1,1,-1,\dots] = 0$ for all $n \ge 1$: binomial filters perfectly cancel the highest frequency — exactly what an antialiasing filter must do before 2× downsampling (Day 7, Ch 21). A $3\times3$ box cannot remove checkerboard noise; $b_{2,2}$ removes it perfectly.

2D by separability:

$b_{2,2} = \begin{bmatrix}1\\2\\1\end{bmatrix} \circ \begin{bmatrix}1 & 2 & 1\end{bmatrix} = \begin{bmatrix}1 & 2 & 1\\ 2 & 4 & 2\\ 1 & 2 & 1\end{bmatrix}, \qquad \text{normalize by } 16$

The book's parting advice: keep $\frac{1}{16}\begin{bmatrix}1&2&1\\2&4&2\\1&2&1\end{bmatrix}$ close to you.

```python
import torch
import torch.nn.functional as F

def gaussian_kernel1d(sigma: float) -> torch.Tensor:
    radius = int(3 * sigma)                       # support = +-3 sigma
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-x**2 / (2 * sigma**2))
    return k / k.sum()                            # DC gain = 1

def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """img: (B, C, H, W). Separable: two 1D passes, O(2N) instead of O(N^2)."""
    k = gaussian_kernel1d(sigma)
    C, r = img.shape[1], (len(k) - 1) // 2
    kx = k.view(1, 1, 1, -1).repeat(C, 1, 1, 1)   # horizontal
    ky = k.view(1, 1, -1, 1).repeat(C, 1, 1, 1)   # vertical
    img = F.pad(img, (r, r, 0, 0), mode='reflect')
    img = F.conv2d(img, kx, groups=C)             # groups=C: filter channels independently
    img = F.pad(img, (0, 0, r, r), mode='reflect')
    return F.conv2d(img, ky, groups=C)

def binomial_blur(img: torch.Tensor) -> torch.Tensor:
    """b_{2,2}/16 -- zero-phase, kills Nyquist exactly: safe pre-downsampling filter."""
    b = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
    C = img.shape[1]
    w = b.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    return F.conv2d(F.pad(img, (1, 1, 1, 1), mode='reflect'), w, groups=C)

x = torch.rand(1, 3, 128, 128)
y1, y2 = gaussian_blur(x, 2.0), binomial_blur(x)

# semigroup check: blur(sigma1) then blur(sigma2) ~= blur(sqrt(sigma1^2+sigma2^2))
z_seq = gaussian_blur(gaussian_blur(x, 1.5), 2.0)
z_one = gaussian_blur(x, (1.5**2 + 2.0**2) ** 0.5)
print((z_seq - z_one).abs().max())                # small, but nonzero: discretization error
```

**Why it matters / connections.** Antialiasing before every downsample (Day 6–7) is a blur; Gaussian scale-space and pyramids (Ch 23) rest on the semigroup property; blur–pool layers ("anti-aliased CNNs") reuse the binomial kernel to make CNNs shift-robust; the noise-vs-detail tradeoff of linear blur motivates nonlinear alternatives (bilateral filtering, anisotropic diffusion) and, later, learned denoisers.

## Self-check questions (Day 5)

1. Derive why translation invariance collapses the general linear map $h[n,k]$ to $h[n-k]$, and explain what this means structurally about the matrix $\mathbf{H}$ (what pattern do its entries follow?).
2. PyTorch's `Conv2d` implements cross-correlation. For which kernels does this coincide with true convolution, and which two algebraic properties (used e.g. to fuse cascaded filters) does correlation lack?
3. Prove the convolution theorem for the circular convolution of two $N$-point signals. Where exactly does periodicity get used, and why does the theorem fail for zero-padded (linear) convolution unless you pad to length $\ge N+M-1$?
4. The DFT of $\text{box}_5$ with $N=32$ has ripples and negative lobes; $B_2[u] = 2+2\cos(2\pi u/N)$ does not. Explain why the box passes the Nyquist wave $[1,-1,1,-1,\dots]$ unchanged while $[1,2,1]$ annihilates it, and why the latter is the property you need before 2× downsampling.
5. State the Gaussian semigroup property and give the one-line frequency-domain proof. Why does it fail for sampled Gaussians, and how do binomial filters restore an exact discrete analogue?

## Supplementary resources for this block

No resources are formally mapped to Day 5 in the study plan, so this section links the nearest-relevant material (noted as such):

- **Udemy — Mastering Computer Vision: From Pixel to Detection to Gen CV** ([https://www.udemy.com/course/mastering-computer-vision-from-pixel-to-detection-to-gen-cv/](https://www.udemy.com/course/mastering-computer-vision-from-pixel-to-detection-to-gen-cv/)) — Intro module (pixels, image scaling & interpolation): interpolation IS convolution with a reconstruction kernel, and today's blur filters are exactly the antialiasing step that must precede the scaling operations covered there (formally mapped to Day 7). Module 2 (DL Foundations & CNNs) connects Ch 15's "convolution = weight-shared fully connected layer" to conv layers.
- **adensur blog — computer_vision_zero_to_hero, folder 03_convolutions** ([https://github.com/adensur/blog/tree/main/computer_vision_zero_to_hero](https://github.com/adensur/blog/tree/main/computer_vision_zero_to_hero)) — mapped to Day 8 but worth a first pass now: it walks through convolution arithmetic (stride, padding, output size $\lfloor(N + 2p - k)/s\rfloor + 1$) from the practitioner's side, complementing today's signal-processing view; the "framework conv is cross-correlation" gotcha appears there too.

Integration note: the through-line between today's chapters and those resources is that every resize/interpolation/pooling operation in a modern CNN pipeline is quietly built from the kernels studied today — box (average pooling), binomial/Gaussian (blur-pool, pyramid downsampling), and the Fourier view predicts exactly which of them alias.

## Latest CV Research — 2026-07-05

Selected recent papers (mostly June 2026 arXiv [cs.CV](http://cs.cv/) + current-cycle venues), preferring work related to today's block (filtering, frequency-domain analysis, blur/defocus):

6. **ErA: Error-Aware Deep Unrolling Network for Single Image Defocus Deblurring** — arXiv:2606.06540, June 2026. Deep unrolling for defocus deblurring: an optimization loop (data term = the blur convolution model from Ch 17, prior term = learned) unrolled into network layers, with explicit modeling of the error introduced at each unrolled step. Directly relevant: defocus IS convolution with a low-pass PSF, and deblurring is inverting the transfer function — the ill-posedness comes precisely from the near-zeros of $H[u,v]$. Link: [https://arxiv.org/pdf/2606.06540](https://arxiv.org/pdf/2606.06540)
7. **Positional Encodings Anchor Spatial Structure in Vision Transformers: A Geometric Perspective on Robustness** — Mahmoud Mannes; arXiv:2606.00124, June 2026 (submitted to NeurIPS 2026 + ICML 2026 mech-interp workshop). Analyzes how positional encodings — which are Fourier features (Ch 16's complex exponentials used as coordinates) — anchor the spatial/geometric structure ViTs rely on, and what that implies for robustness. A clean modern echo of "phase carries where, amplitude carries what." Link: [https://arxiv.org/abs/2606.00124](https://arxiv.org/abs/2606.00124)
8. **CoilDrop-MRI: Self-supervised physics-guided MRI reconstruction with coil dropout** — Tongxi Song, Ziyu Li, et al. (Tsinghua/Oxford-affiliated groups); arXiv:2606.00100, June 2026. Self-supervised MRI reconstruction that randomly drops receive coils during training as physics-consistent augmentation. MRI acquisition lives natively in k-space — the 2D Fourier domain of Ch 16 — so reconstruction is literally inverse-DFT under undersampling; a working example of why the DFT machinery matters beyond photography. Link: [https://arxiv.org/abs/2606.00100](https://arxiv.org/abs/2606.00100)
9. **Flow-Based Generative Modeling for Optimizing Sampling Policies in Compressed Sensing** — Roman Pavelkin, Luis A. Zavala-Mondragón, Christiaan G. A. Viviers, Fons van der Sommen (TU Eindhoven); arXiv:2606.00078, June 2026. Uses flow-based generative models to learn WHERE to sample a signal for compressed-sensing recovery — i.e., learning the sampling operator instead of fixing it. Bridges directly into Day 6–7 (sampling & aliasing): which measurements you keep determines which frequencies you can reconstruct. Link: [https://arxiv.org/abs/2606.00078](https://arxiv.org/abs/2606.00078)
10. **DefocusTrackerAI — A Generalized Framework for the Automatic Detection of Defocused Particle Images** — Gonçalo Coutinho, Ana S. Moita, António L. N. Moreira, Massimiliano Rossi; arXiv:2606.00076, June 2026. Detection of defocused particle images for 3D particle tracking: the defocus blur pattern (the PSF, Ch 17) is used as SIGNAL rather than nuisance — the shape of the blur encodes depth, the same physics as the lens/defocus chapters of Day 2. Link: [https://arxiv.org/abs/2606.00076](https://arxiv.org/abs/2606.00076)
11. **Physical Object Understanding with a Physically Controllable World Model** — Rahul Venkatesh, Klemen Kotar, et al. (Stanford NeuroAI Lab, Daniel Yamins); arXiv:2606.00439, **CVPR 2026 Highlight**. A world model whose latent physics can be explicitly controlled, enabling physical object understanding (mass, contact, dynamics) from video. Not filtering-related, but a notable highlight of the current CVPR cycle worth tracking. Link: [https://arxiv.org/abs/2606.00439](https://arxiv.org/abs/2606.00439)

*Note: items were drawn from the live arXiv *[*cs.CV*](http://cs.cv/)* June 2026 listing and topic searches on 2026-07-05; item 1's author list was not exposed by the search surface and is omitted rather than guessed.*

---

## Finalization note (2026-07-22, JST)

This block's full chapter report (Ch 15–17, above) was authored on 2026-07-05 and already covers architecture + math + WHY + PyTorch + self-check + supplementary resources to the required depth; the reading and self-check are complete, so the block is now marked **Done**. Per the daily plan, STEP 3 runs every day, so only a fresh dated research pass is appended below (the textbook report is intentionally *not* duplicated). Frontmatter updated: `Status: Done`, `Date: 2026-07-22`, `Self-check done: true`; all other keys preserved.

## Latest CV Research — 2026-07-22

Selected papers surfaced this week (July 2026), preferring work tied to today's block — linear filtering, the Fourier/convolution-theorem view, blur/deblur, and anti-aliased resampling. Dates/venues are labeled honestly; where a paper is from an earlier month but is the most on-topic current result, it is marked as such.

1. **HiFi-Deblur: High-Frequency Intense Image Deblurring with a Frequency-Decoupled U-Net and Discrete Wavelet Transform** — Lim et al.; **WACV 2026 Workshop (WVAQ)**. A U-Net deblurring framework that uses the discrete wavelet transform to split the blurred image into low- and high-frequency subbands, then processes each band with a joint Transformer + DWT encoder to recover fine high-frequency detail. Why it matters: it is a direct modern instantiation of Ch 16–17 — deblurring is inverting a low-pass transfer function $H[u,v]$, and the near-zeros of $H$ at high frequency are exactly why the high-frequency band needs dedicated, learned treatment. Link: [https://openaccess.thecvf.com/content/WACV2026W/WVAQ/html/Lim_HiFi-Deblur_High-Frequency_Intense_Image_Deblurring_with_Frequency-Decoupled_U-Net_and_Discrete_WACVW_2026_paper.html](https://openaccess.thecvf.com/content/WACV2026W/WVAQ/html/Lim_HiFi-Deblur_High-Frequency_Intense_Image_Deblurring_with_Frequency-Decoupled_U-Net_and_Discrete_WACVW_2026_paper.html)

2. **Anti-Aliasing for Downsampling in CNNs Based on Gaussian Filter Convolution (GFC)** — *Electronics* 15(4):780, Feb 2026. Proposes a Gaussian Filter Convolution module inserted before each downsampling stage to suppress the aliasing that stride/pooling introduces. Why it matters: this is precisely the Ch 17 lesson made practical — you must low-pass (Gaussian/binomial) *before* subsampling or high frequencies fold back as artifacts; it is the "blur-pool" idea generalized with a learnable Gaussian, and connects Day 5 → Day 6–7 (sampling & aliasing). Link: [https://doi.org/10.3390/electronics15040780](https://doi.org/10.3390/electronics15040780)

3. **From Attention to Frequency: Integration of Vision Transformer and FFT-ReLU for Enhanced Image Deblurring** — arXiv:2511.10806. A dual-domain deblurring network: a ViT backbone models local+global spatial dependencies while an FFT-ReLU branch enforces sparsity directly in the frequency domain to suppress blur artifacts. Why it matters: applying a nonlinearity (ReLU) *in the DFT domain* is a concrete use of Ch 16's convolution theorem — pointwise spectral operations correspond to global spatial filtering, and the nonlinearity is what lets the model synthesize new frequency content that a pure LTI filter cannot. Link: [https://arxiv.org/abs/2511.10806](https://arxiv.org/abs/2511.10806)

4. **Spatial and Frequency Domain Adaptive Fusion Network for Image Deblurring** — arXiv:2502.14209. Fuses spatial- and frequency-domain features using a *learnable low-pass filter* that adaptively decomposes feature maps into frequency subbands. Why it matters: a learned, content-adaptive version of the fixed box/Gaussian/binomial low-pass filters of Ch 17 — the transfer function $H[u,v]$ becomes data-dependent instead of hand-designed, which is the recurring "hand-crafted → learned" arc of the course. Link: [https://arxiv.org/abs/2502.14209](https://arxiv.org/abs/2502.14209)

5. **Diffusion Transformer Meets Multi-level Wavelet Spectrum for Single Image Super-Resolution** — arXiv:2511.01175. A diffusion-transformer SR model conditioned on a multi-level wavelet spectral decomposition of the image. Why it matters: super-resolution is the inverse of the antialiased downsampling in Ch 17/Day 7 — you must *hallucinate* high frequencies removed by the low-pass+subsample, and the wavelet spectrum gives the coarse-to-fine, localized frequency representation that the (too-global) DFT of Ch 16 lacks — a direct motivation for filter banks/pyramids on Day 7. Link: [https://arxiv.org/pdf/2511.01175](https://arxiv.org/pdf/2511.01175)

6. **ComboStoc: Combinatorial Stochasticity for Diffusion Generative Models** — HKU, Microsoft Research Asia, et al.; **SIGGRAPH 2026** (Los Angeles, July 19–23, 2026 — this week). Improves diffusion generative modeling by injecting structured combinatorial stochasticity into the sampling process. Not filtering-specific, but a notable highlight of the current SIGGRAPH cycle worth tracking as generative modeling continues to dominate visual computing. Link: [https://kesen.realtimerendering.com/sig2026.html](https://kesen.realtimerendering.com/sig2026.html)

*Note: searches were run 2026-07-22 (JST). Items 3–5 carry late-2025/early-2026 arXiv IDs but are the most on-topic results the search surface exposed for this block; items 1 and 6 are current-cycle (WACV 2026 / SIGGRAPH 2026). Author lists not exposed by the search surface are omitted rather than guessed.*