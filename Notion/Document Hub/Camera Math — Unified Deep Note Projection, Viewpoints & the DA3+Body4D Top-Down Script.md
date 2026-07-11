---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-08T15:08:00
Status: Done
Last updated time: 2026-07-08T15:08:00
Last edited by: Heaven Chen
Category:
  - CV
  - Theory
---
> This note merges three previous pages — *Camera Math Guided Reading*, *Top-down View Projection (with human)*, and *Camera Projection & Changing an Image's Viewpoint* — into one linear read, and ends with a line-by-line walkthrough of `topdown_da3_body4d.py`, which uses every single concept below. Each section ends with a short quiz (answers immediately after, in a quote block — cover them with your hand).

**The one model to remember** (everything below is a slice of this chain):

$$
\text{pixel} \;\xleftrightarrow{\;K^{-1},\,K\;}\; \text{ray} \;\xleftrightarrow{\;\times z\;}\; \text{3D point (cam)} \;\xleftrightarrow{\;[R\mid t]\;}\; \text{3D point (world)}
$$

$K^{-1}$ makes a ray; depth makes a point; $[R\mid t]$ moves it between poses; $K$ makes a pixel again. Rotation-only view changes cancel depth (→ homography, no 3D needed); translation through a non-planar scene doesn't (→ you need depth or epipolar geometry). Linear algebra (SVD/DLT, $SO(3)/SE(3)$) *solves* for the transforms; Fourier theory *estimates* them and governs *resampling*.

# Part I — Foundations

# 1. Coordinate frames, conventions, homogeneous coordinates

Four frames in pipeline order: **world** $X_w$ → **camera** $X_c$ → **image plane** (normalized) → **pixels** $(u,v)$ (origin top-left, $v$ grows down).

The **camera frame** (OpenCV convention): `+x` = right, `+y` = **down**, `+z` = **forward** into the scene. The `y`-down part trips everyone up: gravity-"up" in a typical recon world is $-y$, which is why camera-authoring code starts with `UP = [0, -1, 0]`.

The **world frame** is whatever the reconstruction chose. For the aisle clip these axes were verified (from the recon's own camera poses + ground normal):

| Axis | Direction |
| --- | --- |
| +x | lateral, across the aisle |
| +y | vertical, pointing **down** at the floor → "up" is −y |
| +z | depth, the way the source camera looks (down the aisle) |

If you reuse code on another recon, **re-check the axes first — never assume.** (DA3 in the final script defines its own world; the script therefore *estimates* up from geometry instead of hardcoding it.)

**Homogeneous coordinates**: append a 1. A 2D point $[u,v,w]^\top$ means Euclidean $(u/w,\,v/w)$. Perspective projection becomes *linear* in homogeneous coordinates; the nonlinearity is deferred to a final **divide by the last component** (the perspective divide). Scale-invariance ($x \sim \lambda x$) is why projective quantities are defined "up to scale". Points with $w=0$ are **points at infinity** — pure directions / vanishing points.

**Vector conventions — read this once, save hours.** Two equivalent styles appear in real code:

- **Column vectors, left-multiply**: $x = M X$ (math notation, this note's default).
- **Row vectors, right-multiply the transpose**: for a batch `V` of shape `(N,3)`, `V @ M.T` computes $M v$ for every row — identical math, numpy-friendly layout.

**OpenCV vs OpenGL**: OpenCV looks down $+z$ with $y$ down; OpenGL (and pyrender) looks down $-z$ with $y$ up. Convert with $\mathrm{diag}(1,-1,-1)$. This single sign flip explains why two `look_at()` implementations later in this note differ in one line.

**Quiz 1**

1. In the OpenCV camera frame, which way does +y point, and what is gravity-up in a world whose +y points at the floor?
2. Why do we bother with homogeneous coordinates at all?
3. `V @ M.T` on row vectors equals what operation on column vectors?

> **Answers.** 1: +y points down; gravity-up is −y. 2: they make perspective projection linear (a single matrix), deferring the nonlinearity to one final divide; they also represent directions/points-at-infinity ($w=0$). 3: $v' = M v$ applied to each point.

# 2. The pinhole camera: intrinsics, extrinsics, projection

## 2.1 Intrinsics $K$ — what the camera is

$$
K = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

$f_x, f_y$ = focal length **in pixels**; $(c_x, c_y)$ = principal point (≈ image centre); $s$ = skew (≈0 on real sensors). Derived from similar triangles ($x = f\,X_c/Z_c$) plus a metric→pixel scaling and an origin shift to the top-left corner.

**Gotcha (learned the hard way):** some renderers infer the input resolution from the principal point (`width = cx*2`), so $c_x, c_y$ **must stay centred** at $(W/2, H/2)$. Zoom with $f_x$, never by shifting $c_x$. Author intrinsics at the render resolution so the renderer's rescale is a no-op.

## 2.2 Extrinsics — where the camera is (two equivalent descriptions)

**World→camera** form (OpenCV's $[R \mid t]$):

$$
X_c = R\,X_w + t
$$

$R \in SO(3)$ is the orientation, $t$ the translation. The camera **centre** in world coordinates is $C = -R^\top t$ — *not* $t$ (classic exam trap).

**Camera→world** form (`c2w`, what you *author*): its columns are literally the camera's axes written in world coordinates, and the 4th column is where the camera sits:

```plain text
c2w = [ right | down | forward | eye ]     (each a 3-vector; bottom row 0 0 0 1)
```

They are inverses of each other. With $c2w = [\,R_{cw} \mid \mathbf{e}\,]$ and $R_{cw}$ orthonormal ($R_{cw}^{-1} = R_{cw}^\top$):

$$
p_{\text{world}} = R_{cw}\,p_{\text{cam}} + \mathbf{e} \qquad\Longleftrightarrow\qquad p_{\text{cam}} = R_{cw}^{\top}\,(p_{\text{world}} - \mathbf{e})
$$

so the OpenCV extrinsics are $R = R_{cw}^\top$, $t = -R_{cw}^\top\mathbf{e}$.

## 2.3 Full projection

$$
s\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K\,[R\mid t]\begin{bmatrix} X\\Y\\Z\\1 \end{bmatrix} = P\,\tilde X_w
$$

$P = K[R\mid t]$ is 3×4; the scalar $s = Z_c$ **is the depth**. Explicitly, with $p_{\text{cam}} = (x,y,z)$:

$$
\begin{bmatrix} u' \\ v' \\ w' \end{bmatrix} = K\,p_{\text{cam}} = \begin{bmatrix} f_x x + c_x z \\ f_y y + c_y z \\ z \end{bmatrix}, \qquad u = f_x\frac{x}{z} + c_x,\quad v = f_y\frac{y}{z} + c_y,\quad \text{depth}=z
$$

All together, with $\lambda = z$:

$$
\lambda \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K\,R_{cw}^{\top}\,(p_{\text{world}} - \mathbf{e})
$$

The $1/z$ is what makes far things look smaller — the **entire** perspective effect lives in that one divide. The $K$ / $[R\mid t]$ split cleanly separates *what the camera is* from *where it is*.

Note "depth" here means the **z-coordinate in the camera frame** (distance along the optical axis), *not* the Euclidean ray length — a distinction that matters when you compute depth from raycast hit points (§16.9).

**Quiz 2**

4. Where is the camera centre in world coordinates, given $[R\mid t]$?
5. What are the four columns of `c2w`?
6. In $s[u,v,1]^\top = P\tilde X_w$, what is the physical meaning of $s$?
7. You want to zoom in. Which entry of $K$ do you touch, and which must you *not* touch?

> **Answers.** 1: $C = -R^\top t$. 2: the camera's right, down, forward axes expressed in world coords, then the eye position. 3: the camera-frame depth $Z_c$ of the point. 4: scale $f_x$ (and $f_y$); never shift $c_x, c_y$ off-centre.

# 3. Back-projection: pixel + depth → 3D (and what a depth map is)

Projection destroys depth; inverting it recovers a **ray**, and a point only if you know $z$. Since

$$
K^{-1} = \begin{bmatrix} 1/f_x & 0 & -c_x/f_x \\ 0 & 1/f_y & -c_y/f_y \\ 0 & 0 & 1 \end{bmatrix},
$$

$$
\mathbf{r}_{\text{cam}} = K^{-1}\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} \tfrac{u-c_x}{f_x} \\ \tfrac{v-c_y}{f_y} \\ 1 \end{bmatrix}, \qquad p_{\text{cam}} = z\,\mathbf{r}_{\text{cam}}, \qquad p_{\text{world}} = R_{cw}\,p_{\text{cam}} + \mathbf{e}
$$

**A depth map is just this applied to every pixel** $(u,v)$ with its stored $z$ — that's the whole "depth map → point cloud" conversion, and exactly what the script's `cam0()` does for all pixels at once. Always mask $Z_c \le 0$: points behind the camera reproject to nonsense.

**Quiz 3**

8. What does $K^{-1}[u,v,1]^\top$ give you — a point or a ray? What upgrade turns it into a point?
9. Write the one-line recipe that converts a depth map into a world-frame point cloud.

> **Answers.** 1: a ray direction in the camera frame (with $z$-component 1); multiplying by the stored depth $z$ makes it a point. 2: for every pixel: $p_w = R_{cw}\,(z \cdot K^{-1}[u,v,1]^\top) + \mathbf{e}$.

# 4. Lens distortion

Real lenses bend the pinhole model. Distortion is applied on **normalized** coordinates $(x',y') = (X_c/Z_c,\,Y_c/Z_c)$, with $r^2 = x'^2 + y'^2$:

$$
x_d = x'(1+k_1 r^2+k_2 r^4+k_3 r^6) + 2p_1 x'y' + p_2(r^2+2x'^2)
$$

(similarly for $y_d$), then pixels $= K\,[x_d, y_d, 1]^\top$. Remove with `cv2.undistort` (or `initUndistortRectifyMap` + `remap` for video). **Undistort before any geometry** on wide lenses — homographies and PnP assume a true pinhole.

**Quiz 4**

10. On which coordinates is distortion applied — pixels or normalized image coordinates?
11. Why must you undistort *before* estimating a homography?

> **Answers.** 1: normalized coordinates ($X_c/Z_c$), before $K$ is applied. 2: a homography is a pinhole-model relation; distortion is a non-projective warp that breaks it, so estimates on distorted pixels are biased.

# 5. Reference implementation (NumPy) + OpenCV building blocks

Complete, self-contained; runs with nothing but numpy. Each function implements the matching equation above.

```python
import numpy as np

UP = np.array([0.0, -1.0, 0.0])          # gravity-up in this world (= -y)

def make_K(fx, fy=None, W=672, H=384):
    """Pinhole intrinsics. cx,cy MUST stay centred (render_single reads resolution off them);
    zoom with fx, never by shifting cx."""
    fy = fx if fy is None else fy
    return np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1.0]])

def project(P_world, K, c2w):
    """3D world points (N,3) -> pixels (N,2) and camera-depth (N,)."""
    R = c2w[:3, :3].T                       # world->cam rotation = c2w rotation transposed
    t = -R @ c2w[:3, 3]                      # world->cam translation
    Pc = P_world @ R.T + t                   # points in camera frame
    uvw = Pc @ K.T
    return uvw[:, :2] / uvw[:, 2:3], Pc[:, 2]

def unproject(u, v, z, K, c2w):
    """One pixel (u,v) at camera-depth z -> its 3D world point."""
    ray = np.linalg.inv(K) @ np.array([u, v, 1.0])   # direction in camera frame
    Pc = ray * z                                     # camera-frame 3D point
    return c2w[:3, :3] @ Pc + c2w[:3, 3]             # -> world
```

The same in OpenCV, when you need distortion, Jacobians, calibration, or pose estimation:

```python
import cv2
rvec, _ = cv2.Rodrigues(R)                                   # R (3x3) <-> rvec (3x1 axis-angle)
img_pts, _ = cv2.projectPoints(X, rvec, t, K, distCoeffs)   # 3D->2D (+distortion, Jacobians)
ok, rvec, tvec = cv2.solvePnP(X3d, x2d, K, dist)            # 2D-3D -> pose (inverse projection)
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objp, imgp, size, None, None)
K2, R2, C_h, *_ = cv2.decomposeProjectionMatrix(P)          # split P -> K, R, center
```

**Quiz 5**

12. In `project()`, why is the world→cam rotation `c2w[:3,:3].T` and not `c2w[:3,:3]`?
13. Which OpenCV call answers "I have 2D–3D correspondences, where is the camera?"

> **Answers.** 1: `c2w` maps cam→world; the inverse of an orthonormal rotation is its transpose (§2.2). 2: `cv2.solvePnP`.

# Part II — Building & aiming virtual cameras

# 6. Placing & aiming: `look_at()`

Given eye $\mathbf{e}$, a target, and world-up $\mathbf{u}$, build an orthonormal right-handed basis:

$$
\mathbf{f} = \frac{\text{target} - \mathbf{e}}{\lVert \text{target} - \mathbf{e}\rVert}, \qquad \mathbf{r} = \frac{\mathbf{f}\times\mathbf{u}}{\lVert \mathbf{f}\times\mathbf{u}\rVert}, \qquad \mathbf{d} = \mathbf{f}\times\mathbf{r}, \qquad R_{cw} = [\,\mathbf{r}\mid\mathbf{d}\mid\mathbf{f}\,]
$$

Why it's valid: $\mathbf{f}, \mathbf{r}$ are unit and orthogonal, so $\mathbf{d} = \mathbf{f}\times\mathbf{r}$ is unit; orthonormal ⇒ $R^\top R = I$; right-handed ⇒ $\det R = +1$, matching OpenCV's x-right / y-down / z-forward. It is **undefined only when** $\mathbf{f} \parallel \mathbf{u}$ (looking straight along gravity): $\mathbf{f}\times\mathbf{u} \to \mathbf{0}$. That is exactly why a pure 90° nadir (straight-down) camera is numerically fragile — and why the pipelines *crane* at ~45–60° instead.

```python
def look_at(eye, target, up=UP):
    """c2w for a camera at `eye` looking at `target` (OpenCV convention)."""
    fwd = target - eye; fwd /= np.linalg.norm(fwd)             # +z_cam: viewing direction
    right = np.cross(fwd, up); right /= np.linalg.norm(right)  # +x_cam: horizontal
    down = np.cross(fwd, right)                                # +y_cam: floor-ward, unit already
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, fwd, eye
    return c2w
```

**Moving the camera is just editing **`**eye**`: raise it `eye += UP*h`, slide sideways `eye += [1,0,0]*s`, push forward `eye += [0,0,1]*d` — then `look_at` re-aims. And `eye = center - fwd*dist` guarantees the optical axis passes through `center`, so the target is centred for free.

**The OpenGL/pyrender variant** (used by the orthographic top-down renderer) differs by exactly the $\mathrm{diag}(1,-1,-1)$ flip from §1: the camera looks down $-z$ and $y$ is *up*, so the pose columns become `[r | u | -fwd | eye]`:

```python
def look_at_gl(eye, target, up_hint):
    fwd = target - eye; fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    r = np.cross(fwd, up_hint); r = r / (np.linalg.norm(r) + 1e-9)
    u = np.cross(r, fwd)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = r, u, -fwd, eye
    return pose
```

Same math, one sign — knowing *which convention your renderer wants* is half the battle.

**Quiz 6**

14. What are the exact three cross-product steps of `look_at`, in order?
15. Why does aiming straight down along gravity break `look_at`, and what's the practical workaround?
16. What single change converts the OpenCV `c2w` to a pyrender/OpenGL pose?

> **Answers.** 1: normalize forward $\mathbf f$; $\mathbf r = \mathbf f \times \mathbf u$ normalized; $\mathbf d = \mathbf f \times \mathbf r$. 2: $\mathbf f \parallel \mathbf u$ makes $\mathbf f\times\mathbf u$ vanish, so "right" is undefined; crane at <90° (or supply a different up-hint, e.g. −Z as image-up). 3: negate the forward column and use up instead of down: columns `[r | u | -f | eye]` (the diag(1,−1,−1) flip).

# 7. Aiming by angle: `crane_direction()` — the pitch knob

Given a horizontal heading $\mathbf{h}$ (unit, $\mathbf{h}\perp\mathbf{u}$) and pitch $\theta$ below horizontal, rotate $\mathbf{h}$ toward the floor ($-\mathbf{u}$) inside the plane they span:

$$
\mathbf{f} = \cos\theta\,\mathbf{h} - \sin\theta\,\mathbf{u}
$$

It stays unit — $\lVert\mathbf{f}\rVert^2 = \cos^2\theta + \sin^2\theta = 1$, the cross term vanishing because $\mathbf{h}\perp\mathbf{u}$ — and its dip below horizontal is $\sin^{-1}\big(\mathbf{f}\cdot(-\mathbf{u})\big) = \theta$. So $\theta=0$ looks along the aisle, $\theta=90^\circ$ straight down.

To get $\mathbf{h}$ from **any** 3D view direction $\mathbf{v}_0$, strip its vertical component (Gram–Schmidt against up):

$$
\mathbf{h} = \frac{\mathbf{v}_0 - (\mathbf{v}_0\cdot\mathbf{u})\,\mathbf{u}}{\lVert \mathbf{v}_0 - (\mathbf{v}_0\cdot\mathbf{u})\,\mathbf{u}\rVert}
$$

```python
def crane_direction(heading, pitch_deg, up=UP):
    """Keep a horizontal `heading` but tilt it DOWN by pitch_deg. 0=along aisle, 90=nadir."""
    th = np.radians(pitch_deg)
    fwd = np.cos(th) * heading - np.sin(th) * up
    return fwd / np.linalg.norm(fwd)
```

**Quiz 7**

17. Prove in one line that $\cos\theta\,\mathbf h - \sin\theta\,\mathbf u$ is unit.
18. How do you extract a horizontal heading from an arbitrary camera forward vector?

> **Answers.** 1: $\lVert f\rVert^2 = \cos^2\theta\lVert h\rVert^2 + \sin^2\theta\lVert u\rVert^2 - 2\cos\theta\sin\theta\,(h\cdot u) = \cos^2+\sin^2 = 1$ since $h\perp u$. 2: subtract its projection onto up, then normalize: $h \propto v_0 - (v_0\cdot u)u$.

# 8. Zoom & field of view: `fx_for_fov()`

A point at the image edge sits $W/2$ px from $c_x$, at $x/z = \tan(\mathrm{FOV}_x/2)$. From the projection equation, $W/2 = f_x \tan(\mathrm{FOV}_x/2)$, hence:

$$
f_x = \frac{W/2}{\tan(\mathrm{FOV}_x/2)} \qquad(\text{bigger } f_x \Rightarrow \text{smaller FOV} \Rightarrow \text{zoomed in})
$$

A fronto-parallel plane at depth $z$ exactly fills the frame when its width is

$$
W_{\text{scene}} = \frac{z\,W}{f_x} \qquad\Longleftrightarrow\qquad f_x = \frac{W\,z}{W_{\text{scene}}}
$$

— the `fx = W · dist / extent` framing line in camera-authoring scripts. Zoom is **linear**: $u - c_x = f_x\,(x/z)$, so doubling $f_x$ doubles every point's offset from centre.

```python
def fx_for_fov(horizontal_fov_deg, W=672):
    return (W / 2) / np.tan(np.radians(horizontal_fov_deg) / 2)
```

**Demo (all of §5–§8 in five prints).** A floor point at `[0, 2, 8]` (+y down ⇒ 2 below origin), K with fx=600:

```plain text
source cam   floor pt -> [336. 192.] depth 8.2
crane 45deg  eye [ 0. -12.1 -6.1] -> [336. 192.] depth 20.0
  + raised 3 eye [ 0. -15.1 -6.1] -> [336. 192.] depth 22.2
zoom fx=600 -> [366. 192.] ; fx=1200 -> [396. 192.]
round-trip: [0. 2. 8.] ==  [0. 2. 8.]
```

Aiming keeps the target dead-centre at $(336,192) = (W/2, H/2)$ no matter where the eye goes (that's `eye = center − fwd·dist` + `look_at`); −y = higher; doubling fx doubles the off-centre offset (30→60 px); unproject exactly inverts project.

To hand an authored camera to a renderer, save `[fx, fy, cx, cy]` and `c2w` per frame (e.g. Vista4D's `save_cameras("cam.npz", c2w_stack, intrinsics_stack)`).

**Quiz 8**

19. Give $f_x$ for a 90° horizontal FOV at W=672.
20. You want a 10 m-wide floor strip to fill a 512-px-wide frame from 15 m away. $f_x$?

> **Answers.** 1: $336/\tan 45^\circ = 336$. 2: $f_x = Wz/W_{scene} = 512\cdot15/10 = 768$.

# 9. Estimating "up" (gravity) from data — two methods

A virtual crane needs a trustworthy up vector. Two independent estimators:

## 9.1 From the ground plane (RANSAC + SVD) — robust, needs environment geometry

A plane is $\{p : \mathbf{n}\cdot(p-\mathbf{a}) = 0\}$, $\lVert\mathbf{n}\rVert=1$.

**RANSAC**: sample 3 points $\mathbf a, \mathbf b, \mathbf c$; the candidate normal is

$$
\mathbf{n} = \frac{(\mathbf{b}-\mathbf{a})\times(\mathbf{c}-\mathbf{a})}{\lVert(\mathbf{b}-\mathbf{a})\times(\mathbf{c}-\mathbf{a})\rVert}
$$

score by inlier count $\#\{p : |\mathbf{n}\cdot(p-\mathbf{a})| < \tau\}$, keep the best over a few hundred trials.

**SVD refine** (much stabler than a 3-point normal): over the inliers $Q$ with mean $\bar{\mathbf{q}}$, the normal minimising $\sum_i (\mathbf{n}\cdot(\mathbf{q}_i-\bar{\mathbf{q}}))^2$ is the **smallest right-singular vector**:

$$
Q - \bar{\mathbf{q}} = U\Sigma V^{\top} \;\Longrightarrow\; \mathbf{n} = V_{:,-1} \quad(\text{smallest-variance direction})
$$

**Sign fix**: "up" must point toward the (elevated) cameras: if $\operatorname{median}_k\big((\mathbf{e}_k-\mathbf{a})\cdot\mathbf{n}\big) < 0$, flip $\mathbf{n} \leftarrow -\mathbf{n}$. This $\mathbf n$ becomes `UP`. (The script's `fit_up()` is exactly this — §16.7.)

## 9.2 From body pose (feet → head) — no environment needed

For each detected person in each frame, "up" ≈ the vector from mid-ankles to neck; average unit vectors over the clip:

$$
u = k_{\text{neck}} - \tfrac{1}{2}\left(k_{\text{lankle}} + k_{\text{rankle}}\right), \qquad \hat{u} = \frac{1}{N}\sum_{i=1}^{N} \frac{u_i}{\lVert u_i\rVert}, \qquad \text{up} = \frac{\hat{u}}{\lVert\hat{u}\rVert+\epsilon}
$$

MHR70 joint indices: `NECK, LANKLE, RANKLE = 69, 13, 14`.

```python
up_accum = []
for person in persons_in_all_frames:
    kp = person.pred_keypoints_3d + cam_t          # keypoints in camera space
    u = kp[NECK] - 0.5 * (kp[LANKLE] + kp[RANKLE]) # feet -> head ≈ up
    n = np.linalg.norm(u)
    if n > 1e-4:
        up_accum.append(u / n)
up = np.mean(np.stack(up_accum, 0), axis=0)
up = up / (np.linalg.norm(up) + 1e-9)
```

**Caveat**: pose-based up degrades when people are heavily crouched or bent — the plane fit (§9.1) is more robust when environment geometry exists. (Pipelines record the estimated axis and its tilt vs the camera in `profile.json`: `gravity_up_axis_camframe`, `up_tilt_deg_off_camera_up`.)

**Quiz 9**

21. Why refine the RANSAC plane with SVD instead of keeping the best 3-point normal?
22. How is the sign of the plane normal disambiguated into "up"?
23. When does the body-pose estimator fail, and what's the fallback?

> **Answers.** 1: a 3-point normal is noise-limited; the SVD fit minimizes total squared point-plane distance over ALL inliers (least-squares optimal). 2: require the cameras (which are above the floor) to have positive height: flip if the median of $(\mathbf e_k - \mathbf a)\cdot\mathbf n$ is negative. 3: crouched/bent people; fall back to ground-plane fitting.

# 10. Rotating the world instead of the camera (and the orthographic top-down view)

For a true overhead (bird's-eye) render, there are two equivalent strategies. The elegant one: **pre-rotate the geometry so gravity becomes +Y**, then use a canonical straight-down camera.

## 10.1 Build a ground frame from `up` (Gram–Schmidt)

Given `up`, construct an orthonormal frame $(e, \text{up}, f)$ where $e, f$ span the ground:

24. Pick a reference axis `ref` not (nearly) parallel to `up`.
25. $e = \mathrm{normalize}(\text{ref} - \text{up}\,(\text{ref}\cdot\text{up}))$ — the component of ref ⊥ up.
26. $f = \text{up} \times e$.

Build $M$ whose **rows** are $(e, \text{up}, f)$:

$$
M = \begin{bmatrix} e^\top \\ \text{up}^\top \\ f^\top \end{bmatrix}, \qquad M v = \begin{bmatrix} e\cdot v \\ \text{up}\cdot v \\ f\cdot v \end{bmatrix}
$$

Because the rows are the new basis vectors, $Mv$ is just "v's coordinates in the new frame" — three dot products. For a batch of row vectors: $v' = v M^\top$ (§1's convention note).

```python
def up_to_frame(up):
    """Rows = (ground_e, up, ground_f). v @ M.T maps world coords into a frame where
    `up` becomes +Y and the ground plane becomes (X, Z)."""
    up = np.asarray(up, float); up = up / (np.linalg.norm(up) + 1e-9)
    ref = np.array([1.,0.,0.]) if abs(up[0]) < 0.9 else np.array([0.,0.,1.])
    e = ref - up * (ref @ up); e = e / (np.linalg.norm(e) + 1e-9)
    f = np.cross(up, e)
    return np.stack([e, up, f], axis=0)

M = up_to_frame(up)
frames_verts = [[v @ M.T for v in pv] for pv in frames_verts]   # up -> +Y everywhere
```

## 10.2 Orthographic camera straight down

After the pre-rotation, ground = X–Z plane. Place an **orthographic** camera above, looking along −Y. Orthographic = parallel rays ⇒ a true floor-plan with **no perspective distortion**. Frustum sizing becomes trivial min/max arithmetic on axis-aligned bounds — computed **globally over all frames** so the scale is fixed (no per-frame zoom jitter):

```python
cx, cy, cz = center_of_bounds_xyz
half  = max(x_extent, z_extent) * 0.5 * 1.20 + 1e-3   # square frustum, +20% margin
eye_y = ymax + ((ymax - ymin) * 0.5 + 0.5)            # camera above the heads

pose = look_at_gl(np.array([cx, eye_y, cz]),   # eye
                  np.array([cx, cy, cz]),      # target = scene centre
                  np.array([0., 0., -1.]))     # image-up hint = -Z in ground frame
cam = pyrender.OrthographicCamera(xmag=half, ymag=half,
                                  znear=0.01, zfar=(eye_y - ymin) + 1.0)
```

(Note the up-*hint* is −Z, not +Y: when looking along −Y, "up in the image" must be a horizontal direction — this sidesteps the §6 nadir degeneracy by supplying the image-up explicitly.)

## 10.3 The two rotations are different things

| Transform | What it is | Purpose |
| --- | --- | --- |
| $v' = v M^\top$ | pre-rotation of geometry so gravity → +Y | makes bounds axis-aligned; ground = X–Z |
| `look_at(...)` | camera view transform (world → camera) | places an overhead camera looking down −Y |

They compose to give the final projection; they are **not** the same rotation applied twice. Mathematically you *could* fold $M$ into the camera extrinsics and skip the vertex rotation — the rendered result is equivalent. Pre-rotating is a deliberate engineering choice: frustum sizing from axis-aligned bounds is trivial, and the camera pose stays a canonical "straight down −Y", reducing handedness/sign bugs.

**Quiz 10**

27. Why are the *rows* (not columns) of $M$ the new basis vectors, given we compute $v' = vM^\top$?
28. Why pre-rotate the geometry at all if folding $M$ into extrinsics is mathematically equivalent?
29. Why is the ortho camera's up-hint −Z rather than the gravity up?

> **Answers.** 1: $vM^\top$ (row convention) equals $Mv$ (column convention) whose entries are dot products of v with M's rows — i.e., coordinates along each row-vector basis direction. 2: axis-aligned bounds → trivial xmag/ymag/znear/zfar; canonical camera → fewer sign/handedness bugs; identical pixels either way. 3: looking along −Y, forward ∥ gravity, so gravity can't define image-up; a horizontal direction (−Z) must be chosen explicitly.

# 11. One render → depth + segmentation

With meshes in the ground frame and an overhead camera, a **single** offscreen render yields both control signals: the **color buffer is the segmentation**, the **depth buffer is the depth map**.

Paint each person's mesh a distinct flat palette color (as vertex colors), disable shading with `RenderFlags.FLAT` so meshes render as their exact colors (crisp instance masks, no shading/anti-alias bleed):

```python
PALETTE = np.array([[220,20,60],[60,179,113],[65,105,225],[255,165,0],
                    [148,0,211],[0,206,209],[255,105,180],[154,205,50]], np.uint8)

scene = pyrender.Scene(bg_color=[0,0,0,0], ambient_light=[1.,1.,1.])
for i, v in enumerate(persons_verts):                 # v already rotated: up = +Y
    vc = np.tile(np.append(PALETTE[i % len(PALETTE)], 255), (v.shape[0], 1)).astype(np.uint8)
    m = trimesh.Trimesh(v, faces, vertex_colors=vc, process=False)
    scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
scene.add(cam, pose=pose)

r = pyrender.OffscreenRenderer(W, H)
try:
    color, depth = r.render(scene, flags=pyrender.RenderFlags.FLAT)
finally:
    r.delete()

seg_img = color[:, :, ::-1].copy()                    # RGB -> BGR for the video writer

valid = depth > 0                                     # 0 = background
d = depth[valid]; dmin, dmax = float(d.min()), float(d.max())
norm = np.zeros_like(depth)
norm[valid] = 1.0 - (depth[valid]-dmin)/(dmax-dmin) if dmax > dmin else 1.0
depth_img = cv2.cvtColor((norm*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
```

Depth is normalized over valid pixels and **inverted so near = bright** (tops of heads/shoulders brightest; background black) — the common depth-control convention.

| Aspect | Reason |
| --- | --- |
| One render for both | color buffer = segmentation, depth buffer = depth — no second pass |
| FLAT + vertex colors | exact solid colors → unambiguous instance masks |
| Global bounds (all frames) | fixed camera scale → no frame-to-frame zoom/jitter |
| Near = bright | matches depth-control convention |

End-to-end data flow of that human-only pipeline:

```plain text
source clip
└─ sam-body4d ── per-frame meshes (pred_vertices, pred_cam_t, keypoints)  [temporally consistent]
   └─ world_v = pred_vertices + pred_cam_t          (camera frame, y-down)
      └─ up = mean(neck - ankles) over clip          (§9.2 gravity estimate)
         └─ v' = v @ M.T                             (§10.1 rotate: up → +Y)
            └─ overhead ortho camera, look down -Y   (§10.2)
               └─ render(FLAT) → color, depth        (§11)
                  ├─ control_seg.mp4   (flat instance colors)
                  └─ control_depth.mp4 (near = bright, grayscale)
```

**Quiz 11**

30. Why FLAT shading for segmentation?
31. Why compute scene bounds globally over all frames rather than per frame?
32. In the depth control image, is a bright pixel near or far? Why that convention?

> **Answers.** 1: lighting/shading would modulate colors, blurring instance identity; FLAT renders the exact vertex color → clean masks. 2: per-frame bounds change with motion → the virtual camera would re-zoom every frame (jitter); global bounds fix the scale. 3: near (inverted normalization); closest-surface-brightest is the standard convention for depth-control inputs.

# 12. Recipes & gotchas (the practical distillation)

| You want to… | Change |
| --- | --- |
| Camera higher / lower | `eye += UP*h` (h>0 = higher) |
| Steeper look-down / flatter | `--pitch` (↑ = steeper, 90 = nadir) |
| Slide along the aisle / sideways | add to `eye` before `look_at` |
| Zoom in / out | scale `fx` (perspective) or `xmag/ymag` (ortho) |
| Re-centre on real geometry | aim `target` at the visible-point centroid |
| Turn a depth map into 3D | `unproject` every pixel (§3) |

Gotchas, all learned the hard way:

- `**up = −y**`**, not **`**+y**` in y-down worlds; wrong sign flips the camera under the floor.
- **Keep **`**cx = W/2, cy = H/2**`; zoom via `fx` only. Author intrinsics at the render resolution.
- `**eye = center − fwd·dist**` centres the target for free.
- Always **perspective-divide**; mask $Z_c \le 0$.
- $KRK^{-1}$ rotates about the **principal point** — keep $c = (W/2, H/2)$ or content "swings".
- Rotation **order/sign** matters ($R_zR_yR_x \ne R_xR_yR_z$); sanity-check with a small angle.
- After warping, transform the 4 corners (`cv2.perspectiveTransform`), compose a translation, and size the canvas — or your result gets cropped.
- `warpPerspective`/`warpAffine` sample **inversely** (hole-free); hand-rolled *forward* warps need explicit hole/occlusion handling.
- **Undistort first**; pick interpolation by shrink-vs-enlarge; set a `borderMode`.

# Part III — Changing an image's viewpoint: the full toolbox

# 13. From no-geometry to full-3D — every standard method

What you can do depends on **how much scene geometry you know**. Core theorem: **two views related by pure camera rotation, or two views of a single plane, are related by a homography** $x_2 \simeq H x_1$ (3×3 on homogeneous pixels) — no depth needed. General translation through a non-planar scene has **parallax** and is *not* a homography → needs depth.

| # | Method | Scene assumption | Motion | OpenCV |
| --- | --- | --- | --- | --- |
| A | Rotation homography $KRK^{-1}$ | any | rotation only | `warpPerspective` |
| B | Plane-induced homography | planar | rotation+translation | `warpPerspective` |
| C | 4-point / N-point homography | plane / matches | implied | `getPerspectiveTransform` / `findHomography` |
| D | Affine / similarity | 2D only | in-plane | `warpAffine` |
| E | Depth-based reprojection (DIBR) | per-pixel depth | full 6-DoF | `remap` / `grid_sample` |
| F | Stereo/rectification & IPM | planar / 2-view | virtual rotation | `stereoRectify`, `warpPerspective` |
| G | Cylindrical / spherical warp | any (rotation) | pano rotation | custom `remap` |
| H | 360 / equirectangular resample | full sphere | any yaw/pitch/roll | custom `remap` |
| I | Mesh / TPS / APAP warp | non-rigid | parallax-aware | mesh + `remap` |
| J | Learned IBR (MPI, NeRF, 3DGS) | learned 3D | arbitrary | frameworks |

## 13.A Pure-rotation homography (most common)

Camera rotates about its optical centre → depth cancels:

$$
H = K\,R\,K^{-1}, \qquad x_2 \simeq H x_1
$$

Exact for **any** scene (no translation = no parallax). Yaw/pitch/roll synthesis, panoramas, augmentation.

```python
def R_xyz(rx, ry, rz):
    cx,sx=np.cos(rx),np.sin(rx); cy,sy=np.cos(ry),np.sin(ry); cz,sz=np.cos(rz),np.sin(rz)
    Rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]]); Ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]]); return Rz@Ry@Rx

def rotate_view(img, K, rx=0, ry=0, rz=0):
    h,w = img.shape[:2]; H = K @ R_xyz(rx,ry,rz) @ np.linalg.inv(K)
    return cv2.warpPerspective(img, H, (w,h))
# Ry≈yaw (pan), Rx≈pitch (tilt), Rz=roll (in-plane spin)
```

## 13.B Plane-induced homography (rotation + translation over a plane)

Plane $n^\top X = d$ in cam-1 coordinates, relative pose $(R,t)$:

$$
H = K_2\Big(R - \frac{t\,n^\top}{d}\Big)K_1^{-1}
$$

Reduces to 13.A when $t=0$. The engine behind road/ground bird's-eye views.

## 13.C Homography from correspondences (data-driven)

≥4 point matches → estimate $H$ directly (keystone correction, rectification):

```python
H = cv2.getPerspectiveTransform(src4, dst4)                # exact, 4 points
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)  # robust, many noisy points
out = cv2.warpPerspective(img, H, (Wt, Ht))
```

## 13.D Affine / similarity (in-plane only)

Parallel lines preserved; a 2×3 matrix; affine ⊂ homography (bottom row fixed at $[0,0,1]$). For augmentation / 2D alignment:

```python
M = cv2.getRotationMatrix2D((w/2,h/2), angle=30, scale=1.0)
out = cv2.warpAffine(img, M, (w,h))
```

## 13.E Depth-based reprojection / DIBR (full 6-DoF)

A real viewpoint change through a non-planar scene needs **per-pixel depth** — the §0 chain executed pixel-wise:

$$
X_{c1} = Z_1 K^{-1}x_1, \qquad X_{c2} = R\,X_{c1} + t, \qquad x_2 \simeq K X_{c2}
$$

```python
def reproject_with_depth(img, depth, K, R, t):
    h,w = depth.shape
    vv,uu = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    uv1 = np.stack([uu,vv,np.ones_like(uu)],-1).reshape(-1,3).T.astype(float)
    Xc1 = (np.linalg.inv(K)@uv1)*depth.reshape(1,-1)
    Xc2 = R@Xc1 + t.reshape(3,1)
    uv2 = (K@Xc2)[:2]/(K@Xc2)[2:3]
    return uv2.T   # forward-warp with z-order; handle holes/occlusion in practice
```

Hard parts: **forward warping** leaves holes; **inverse warping** needs destination depth; you must handle **occlusion** (z-buffer) and **dis-occlusion** (inpaint / mesh). Prefer `cv2.remap` / `grid_sample` with an inverse map for clean bilinear sampling. *The final script is exactly this method, upgraded from pixel-splatting to mesh raycasting to solve holes and occlusion correctly (§16).* 

## 13.F Rectification & Inverse Perspective Mapping (IPM)

- **Stereo/image rectification** (`cv2.stereoRectify` → `initUndistortRectifyMap` → `remap`): warp a pair so epipolar lines become horizontal rows — a specific rotation homography per camera making disparity 1-D. Also used to "level" a single view.
- **IPM / bird's-eye view**: a named special case of 13.B/13.C — map the ground plane to a top-down rectangle (lane detection standard). Pick 4 ground points → `getPerspectiveTransform` → `warpPerspective`.

## 13.G Cylindrical & spherical projection

For wide rotations, first reproject pixels onto a **cylinder/sphere** ($\theta$ = atan of pixel offset / focal); rotations then become simple horizontal shifts — the standard panorama pre-warp so large rotations don't blow up at the edges. A custom `cv2.remap`.

## 13.H 360 / equirectangular resampling

Given a full-sphere pano in $(\theta_{\text{lon}}, \phi_{\text{lat}})$: for each output pixel build the ray $K^{-1}x$, rotate by $R$, convert to lon/lat, sample the pano — the "drag to look around" viewer; a pure-rotation view change with a spherical source.

```python
# per output pixel: dir = R @ (Kinv @ [u,v,1]); lon=atan2(dir_x,dir_z); lat=asin(dir_y/|dir|)
# map (lon,lat) -> equirect pixel, then cv2.remap
```

## 13.I Mesh / TPS / APAP warps

When one global homography can't fit parallax (stitching real scenes): **spatially-varying** warps — a mesh of local homographies, **as-projective-as-possible (APAP)**, or **thin-plate splines** for smooth non-rigid mappings from control points. Warp = per-cell `remap`.

## 13.J Learned image-based rendering

Free-viewpoint without explicit depth: **MPI**, **NeRF**, **3D Gaussian Splatting** — learn a 3D/volumetric representation and re-render from new poses; the modern, heaviest endpoint of 13.E. (Depth-Anything-style monocular depth feeds the *classic* DIBR path instead — which is what the final script does.)

**Bonus — homography back to physical motion**: `cv2.decomposeHomographyMat(H, K)` factors an estimated $H$ into candidate $(R, t, n)$ — turning a measured image warp into a camera motion hypothesis.

**Quiz 13**

33. Why does pure camera rotation give a homography but translation through a 3D scene doesn't?
34. Derive $H = KRK^{-1}$ from the ray interpretation of a pixel.
35. Which method do you reach for: (a) level a tilted photo of a document, (b) synthesize a 30°-left view of a street photo with a depth map, (c) make a driving bird's-eye view of the road plane?

> **Answers.** 1: with rotation about the optical centre, every pixel's ray just rotates — depth never enters, so the map is the same 3×3 for all pixels; translation makes image motion depth-dependent (parallax), which no single 3×3 can express. 2: pixel → ray $r = K^{-1}x_1$; rotate the camera: same scene ray expressed in the new frame is $Rr$ (any scale); re-project: $x_2 \simeq K R K^{-1} x_1$. 3: (a) 13.C 4-point homography; (b) 13.E DIBR; (c) 13.F IPM.

# Part IV — Mathematical foundations

# 14. The math that makes all of it work

## 14.1 Projective linear algebra

- $\mathbb{P}^2, \mathbb{P}^3$: points and lines are dual — the line through two points is $l = x_1 \times x_2$, the intersection of two lines is $x = l_1 \times l_2$. Points at infinity ($w=0$) model directions/vanishing points.
- The whole pipeline is matrix chains: $K[R\mid t]$ forward; $K^{-1}, R^\top$ backward.
- **Rotations **$SO(3)$: orthonormal, $\det = +1$, $R^{-1} = R^\top$. Parameterizations: **Euler** (intuitive; gimbal lock), **axis-angle/Rodrigues** (`cv2.Rodrigues`), **quaternions** (no gimbal lock; smooth interpolation). Lie view: $R = \exp([\omega]_\times)$ with $\log$ back — the right way to average/optimize rotations. Full pose lives in $SE(3)$.
- **Nearest rotation (Procrustes)**: closest valid rotation to a noisy $M$ is $R = UV^\top$ from $M = U\Sigma V^\top$ — re-orthonormalizes drifted rotations; Gram–Schmidt/QR does the same for a basis (as in §6, §10.1).

## 14.2 SVD & the Direct Linear Transform (how $H$, $P$, $F$ are actually solved)

Estimating a homography from matches is a homogeneous system $Ah = 0$: each match contributes rows via the cross-product constraint $x_i' \times H x_i = 0$. Solution = **right singular vector with the smallest singular value**:

$$
A = U\Sigma V^\top, \qquad h = V_{:, \text{last}}
$$

```python
U, S, Vt = np.linalg.svd(A)
H = Vt[-1].reshape(3, 3)      # null-space solution, then normalize
```

SVD also gives least-squares/pseudo-inverse ($A^+ = V\Sigma^+U^\top$) and **rank enforcement** (e.g., zero $F$'s smallest singular value to force rank 2). $H$ has **8 DoF** (9 entries up to scale); the essential matrix has **5 DoF**. Wrap in **RANSAC** for outliers — the same estimator pattern as the ground-plane fit (§9.1).

## 14.3 Epipolar geometry (two views with translation)

When the camera **translates**, a point maps to a **line**, encoded via the skew-symmetric $[t]_\times$:

$$
E = [t]_\times R, \qquad F = K_2^{-\top} E K_1^{-1}, \qquad x_2^\top F x_1 = 0
$$

$E$ (essential, calibrated) / $F$ (fundamental, uncalibrated) send a point in one image to its **epipolar line** in the other. Decomposing $E = U\Sigma V^\top$ yields relative $(R,t)$ (4-fold ambiguity, resolved by cheirality — points must be in front of both cameras). This is precisely the boundary between "homography suffices" (no parallax) and "need depth/triangulation" (parallax).

## 14.4 Interpolation & sampling (why warps need care)

Warping resamples at non-integer locations → you need a reconstruction kernel: **nearest / bilinear / bicubic / Lanczos** trade speed vs sharpness; the ideal is **sinc** (Shannon–Nyquist: a signal sampled above twice its max frequency is exactly recoverable). When a warp **shrinks** content you undersample → **aliasing** (moiré, jaggies); fix by **prefiltering** (Gaussian / mip levels) before downsampling. `cv2.INTER_AREA` for shrink, `INTER_CUBIC`/Lanczos for enlarge.

## 14.5 Fourier view

The 2D DFT $F(u,v) = \sum_{x,y} f(x,y)\,e^{-j2\pi(ux/W + vy/H)}$ connects to viewpoint work in three places:

- **Shift theorem**: translation ⇔ pure linear phase, $f(x-a, y-b) \Leftrightarrow F(u,v)e^{-j2\pi(ua/W + vb/H)}$ → **phase correlation** (`cv2.phaseCorrelate`) recovers translation sub-pixel and illumination-robustly from the cross-power spectrum peak.
- **Rotation/scale**: rotating an image rotates its spectrum; scaling space by $a$ scales frequency by $1/a$. **Fourier–Mellin**: log-polar-map the magnitude spectrum (`cv2.warpPolar`) → rotation+scale become translations → phase correlation recovers them — global rotation/scale estimation before refining a homography.
- **Convolution theorem**: $f * g \Leftrightarrow FG$ — the basis of fast prefiltering, and of reading interpolation kernels as low-pass filters. Zero-padding an FFT = ideal sinc interpolation.

So Fourier **estimates** transforms (phase correlation, Fourier–Mellin), **anti-aliases** resampling, and underpins **interpolation** theory.

**Quiz 14**

36. Why is the DLT solution the smallest-singular-value right vector of $A$?
37. What does the Fourier shift theorem let you recover, and how does log-polar extend it to rotation/scale?
38. When must you prefilter before a warp, and why?
39. Your optimizer returned a slightly non-orthonormal rotation. Fix it in one SVD.

> **Answers.** 1: we want nonzero $h$ minimizing $\lVert Ah\rVert$ s.t. $\lVert h\rVert = 1$; the minimizer of a homogeneous least-squares problem is the right singular vector of the smallest singular value. 2: unknown translation between two images (from the phase of the cross-power spectrum); log-polar of the magnitude spectrum converts rotation/scale into translations so the same trick recovers them. 3: whenever the warp locally shrinks content (downsampling) — otherwise frequencies above Nyquist alias into moiré. 4: $M = U\Sigma V^\top \Rightarrow R = UV^\top$ (Procrustes).

# 15. Cheat-sheet

| Task | Tool |
| --- | --- |
| pixel→ray | $K^{-1}[u,v,1]$ |
| 3D→2D | `cv2.projectPoints` |
| 2D+3D→pose | `cv2.solvePnP` |
| R↔rvec | `cv2.Rodrigues` |
| calibrate | `cv2.calibrateCamera` |
| rotate view (no depth) | $KRK^{-1}$  • `warpPerspective` |
| plane re-view | $K(R - tn^\top/d)K^{-1}$ |
| homography from points | `getPerspectiveTransform` / `findHomography` |
| decompose H→(R,t,n) | `decomposeHomographyMat` |
| in-plane 2D | `getRotationMatrix2D`  • `warpAffine` |
| rectify / IPM | `stereoRectify`, `warpPerspective` |
| undistort | `undistort` / `initUndistortRectifyMap` |
| depth new-view | back-project → (R,t) → project → `remap` (or mesh+raycast, §16) |
| estimate shift/rot/scale | `phaseCorrelate`, `warpPolar` (Fourier–Mellin) |
| solve Ah=0 | `np.linalg.svd` (smallest singular vector) |
| place a camera | `look_at(eye, target, up)` (§6) |
| pitch a camera | $\cos\theta\,h - \sin\theta\,u$ (§7) |
| zoom | $f_x = (W/2)/\tan(\mathrm{FOV}/2)$ (§8) |
| find up | plane RANSAC+SVD or feet→head (§9) |
| gravity → +Y | rows-basis $M$, $v' = vM^\top$ (§10) |

# Part V — Case study: `topdown_da3_body4d.py`, every line of math explained

# 16. The script

**What it does.** From ONE video and TWO models it renders a synthetic 45°-elevated "crane" view of the scene as depth + instance-segmentation videos:

- **Depth-Anything-3 (DA3)**: one pass over the clip → per-frame depth, intrinsics $K$, camera poses $E$ (cam→world), sky masks, and a metric scale factor. This is the **environment** geometry (replaces a whole recon stack — no Pi3, no separate SAM pass).
- **sam-body4d**: temporally-consistent 3D human **meshes** (SAM-3 + SAM-3D-Body inside). These are the clean **people** — mesh people don't have the single-view "streaking" that depth-map people do.

Then pure geometry (open3d, no more models): mesh the depth map → cut the people out of the environment → rescale the people into DA3's world → drop jittery tracks → render everything from a synthetic crane camera.

```plain text
video ─┬─ DA3 ──── depth(N,H,W), K, E(cam→world), sky, scale sc     [environment]
       └─ sam-body4d ─ per-frame verts, cam_t, kps3d, faces          [people]
§16.1 unproject depth[0] → env point cloud (world)          §3
§16.3 flag_jitter: drop unstable tracks                     (robust stats)
§16.4 scale people by s = median(DA3 depth / mesh depth)    (monocular scale gauge)
§16.5 project people → frame-0 silhouette → cut from env    §2.3 projection
§16.6 grid-mesh the env cloud, kill long edges              (streak removal)
§16.7 fit_up: ground plane RANSAC+SVD, sign via cameras     §9.1
§16.8 build 45° crane camera, auto-frame on the people      §6–§8
§16.9 open3d raycast env(once) + people(per frame)          §3 inverted
§16.10 combine by min-depth, normalize (near=bright)        §11
§16.11 seg colors by persistent track id, occlusion-aware   §11
```

## 16.1 DA3 → environment cloud (this is §3, vectorized)

```python
pred = m.inference(frames, process_res=args.process_res)
depth = np.asarray(pred.depth)      # (N,H,W) raw camera-frame z-depth
K     = np.asarray(pred.intrinsics) # (N,3,3)
E     = np.asarray(pred.extrinsics) # (N,4,4) cam->world  (i.e. c2w, §2.2!)
sc    = float(getattr(pred, "scale_factor", None) or 1.0)

ys, xs = np.mgrid[0:H, 0:W]
pix = np.stack([xs.ravel(), ys.ravel(), np.ones(H*W)], -1)   # all (u,v,1), (HW,3)

def cam0(uvz_depth):                       # depth map -> camera-frame points
    z = uvz_depth.reshape(-1)
    return (pix @ np.linalg.inv(K0).T) * z[:, None]          # (K⁻¹[u,v,1]) · z

def to_world(campts):                      # cam-0 frame -> DA3 world, metric
    return (np.c_[campts, np.ones(len(campts))] @ E0.T)[:, :3] * sc
```

- `pix @ inv(K0).T` is the batched row-vector form of $K^{-1}[u,v,1]^\top$ (§1's convention note) → one ray per pixel; `* z` upgrades rays to points (§3).
- `to_world` right-multiplies by $E_0^\top$: homogeneous points times a cam→world matrix — exactly $p_w = R_{cw}p_c + \mathbf e$ (§2.2) — then multiplies by `sc` to make DA3's up-to-scale world **metric**.
- Only **frame 0** builds the environment (`env_w = to_world(cam0(depth[0]))`): the env is treated as **static**, rendered once, eliminating temporal flicker (same philosophy as §11's global bounds).

## 16.2 sam-body4d (not math — ops)

Runs in its own venv via `subprocess`; input is upscaled/padded to 1280×720 first because its mesh regressor fails on small frames (`ffmpeg scale=...:force_original_aspect_ratio=decrease,pad=...` — note pad keeps the principal point centred, §2.1 gotcha). Outputs per frame: `verts (n,V,3)`, `cam_t (n,3)`, `kps3d`, plus a shared `faces.npy`. Vertices live in **its own camera frame**: `verts + cam_t` = camera-frame positions, y-down (§1).

## 16.3 Dropping jittery mis-tracks: `flag_jitter`

The neck joint (`kps3d[:, NECK] + cam_t`) is each person's trajectory proxy. Two robust filters on per-frame displacement $\lVert p_t - p_{t-1}\rVert$:

- **Garbage track**: if the *median* step of a whole track exceeds `track_med_thr` (0.15 m), the track is noise — drop it everywhere (median = robust to a few good frames in a bad track).
- **Spike frame**: drop a single frame when its step to *either* neighbour exceeds `vel_thr` (0.5 m/frame) — a discrete velocity threshold; checking both neighbours makes it an acceleration-ish test that catches isolated teleports.

This is pure robust statistics: median for track-level decisions, max-of-two-sided-velocity for frame-level ones.

## 16.4 Scaling the people into DA3's world (resolving the monocular scale gauge)

sam-body4d and DA3 each estimate depth **up to their own scale** — same camera, same pixels, different metric gauges. But both place a person along the same viewing ray, so the ratio of depths *is* the scale between the gauges:

```python
c = v3.mean(0)                                        # torso centroid, sam cam frame
u = round(K0[0,0]*c[0]/c[2] + K0[0,2])                # project with DA3's K  (§2.3:
v = round(K0[1,1]*c[1]/c[2] + K0[1,2])                #  u = fx·x/z + cx, v = fy·y/z + cy)
if in_bounds and depth[0][v,u] > 0:
    ratios.append(depth[0][v, u] / c[2])              # DA3 depth vs mesh depth, same ray
s = float(np.median(ratios))                          # median over all people/frames
```

Projecting the centroid with DA3's intrinsics finds *which DA3 pixel this person occupies*; reading DA3's depth there gives the environment's opinion of that ray's depth; the ratio `DA3_z / mesh_z` maps the mesh gauge onto DA3's. **Median** over every valid person-frame kills outliers (occlusions, wrong pixels, mis-tracks). Then `person_world(v3) = to_world(v3 * s)`: scale in the shared camera frame first, *then* apply the cam→world transform — order matters, because `to_world` adds a translation (§2.2) which must not get scaled.

## 16.5 Cutting the people out of the environment (no double-people)

DA3's depth map *contains* the people; the sam-body4d meshes *are* the people. Render both and everyone appears twice. Fix: project each frame-0 person mesh (already scaled by `s`, so it lands on the right pixels) into the frame-0 image with the §2.3 equations, splat vertices to a boolean mask, **dilate 9×9** (vertices are sparse — dilation closes the gaps into a solid silhouette), and exclude those pixels from the environment:

```python
person_px[v[m2], u[m2]] = True
person_px = cv2.dilate(person_px.astype(np.uint8), np.ones((9,9), np.uint8)).astype(bool)
keep_env = finite & (depth[0] > 0) & ~person_px & ~sky[0]     # per-pixel keep mask
```

Sky is also cut (its depth is fictional). This is forward-projection used as a *stencil* — the same math as 13.E's forward warp, but only to make a mask, so holes don't matter.

## 16.6 Meshing the depth map (and killing streaks)

A depth map's pixel grid is a ready-made triangulation: each 2×2 pixel block → two triangles:

```python
tl, tr, bl, br = ig[:-1,:-1], ig[:-1,1:], ig[1:,:-1], ig[1:,1:]      # index grids
faces_e = np.concatenate([np.stack([tl,bl,tr],1), np.stack([tr,bl,br],1)], 0)
faces_e = faces_e[keep_env[faces_e].all(1)]        # drop triangles touching cut pixels
em = max_edge_length_of_each_triangle
faces_e = faces_e[em < args.edge_k * np.median(em)]  # kill long stretched triangles
```

The edge-length filter is the important trick: at **depth discontinuities** (object ↔ background), neighbouring pixels unproject to points metres apart, creating long "rubber-sheet" triangles that render as streaks. Discarding triangles whose longest edge exceeds `edge_k`× the median edge removes exactly those, leaving clean surface patches. (Meshing + raycasting is how this script solves 13.E's hole/occlusion problems: triangles interpolate surfaces → no holes; raycast depth ordering → correct occlusion.)

## 16.7 Finding up: `fit_up` (§9.1 verbatim) + sign fix

`fit_up(env_w[keep_env])` = 300 RANSAC rounds (3-point cross-product normals, inlier threshold `0.05·ptp`) + SVD refinement on the inliers (smallest right-singular vector = plane normal). Sign disambiguation uses DA3's own camera trajectory:

```python
cams = E[:, :3, 3] * sc                       # all camera positions, world, metric
if np.median((cams - env_centroid) @ up) < 0: up = -up   # cameras must be ABOVE the floor
```

— the §9.1 rule with the eye positions read straight out of the c2w matrices (4th column, §2.2).

## 16.8 Building the 45° crane camera (§6 + §7 + §8 composed)

```python
zc = np.array([0,0,1.0])                          # world +z ≈ source viewing direction (§1 table)
heading = zc - up*(zc@up); heading /= norm        # §7: strip vertical component
th = np.radians(args.pitch)
D  = np.cos(th)*heading - np.sin(th)*up           # §7: crane_direction, pitch=45°
right = np.cross(D, up);  right /= norm           # §6: look_at basis (f×u)
down  = np.cross(D, right)                        # §6: f×r
Rm = np.stack([right, down, D], 0)                # ROWS = camera axes = world→cam rotation (§10.1!)
```

Note `Rm` stacks the axes as **rows**, so `Pc = (P - eye) @ Rm.T` *is* $R_{cw}^\top(p - \mathbf e)$ — the §2.2 world→camera transform in row-vector form; equivalently, it's a §10.1 rows-basis change of frame. Same object, two readings.

**Auto-framing** (the §8 formulas driven by data instead of hand-picked numbers):

```python
center = np.median(ppl_all, 0)                       # robust scene centre = the people
radius = np.percentile(norm(ppl_all - center), 95)   # robust extent (ignores stragglers)
eye    = center - D*(radius*3 + max(radius, 1.0))    # §6: eye = center - fwd·dist → centred for free
Pc = (ppl_all - eye) @ Rm.T                          # everything in camera frame
sx = np.percentile(np.abs(Pc[:,0]/Pc[:,2]), 96)/zoom # 96th-pct tan(half-FOV) actually needed
fx = fy = min(S/2/sx, S/2/sy)                        # §8: fx = (W/2)/tan(FOV/2)
```

Every number that §5–§8's demo hand-picked is here derived from the point statistics: median for the target, percentiles for extent (so one outlier can't unzoom the shot), `--zoom` scaling the final $f_x$ linearly (§8's linear-zoom fact). `c2wt` assembles `[right | down | D | eye]` columns — the §2.2 c2w — and `w2c = inv(c2wt)` feeds the renderer.

## 16.9 Rendering by raycasting (projection run backwards, §3)

```python
rays = scn.create_rays_pinhole(Kt, w2c, S, S)   # per pixel: origin=eye, dir ∝ R·K⁻¹[u,v,1]
a    = scn.cast_rays(rays)                      # BVH intersection: first triangle hit
hit  = origin + t_hit * direction               # 3D hit point in world
z    = (hit - eye) @ D                          # depth = component along the OPTICAL AXIS
```

`create_rays_pinhole` builds exactly §3's back-projection rays for every output pixel; `cast_rays` finds where each ray first meets geometry (this *is* z-buffered rendering, done analytically). The depth line is subtle: $(\text{hit} - \mathbf e)\cdot D$ projects the hit vector onto the viewing direction — the **camera-frame z-depth** of §2.3, *not* the ray length `t_hit` (Euclidean distance). Using ray length would make depth bow outward toward image corners.

The environment is cast **once** (static → zero flicker); the people are cast **per frame** (each frame's meshes form their own scene; `geometry_ids` records which mesh each ray hit → free instance IDs).

## 16.10 Compositing + depth video (§11's conventions, raycast edition)

```python
comb = np.minimum(env_z, people_z)               # min depth = nearest surface wins = z-buffer
lo, hi = np.percentile(all_depths, [2, 98])      # ONE normalization for the whole clip
g = 1.0 - clip((comb - lo)/(hi - lo), 0, 1)      # invert: near = bright (§11)
```

`np.minimum` across the two depth images is occlusion resolution — a person in front of the shelf wins those pixels, and vice versa. The percentile window is computed over **all frames jointly**, so brightness means the same thing in every frame (the temporal-consistency principle of §11 again); 2/98 percentiles clamp outliers.

## 16.11 Segmentation video (occlusion-aware, stable colors)

```python
seg[np.isfinite(env_z)] = (60, 60, 60)                    # environment = dark grey
pnear = np.isfinite(pz) & (pz <= env_z)                   # person pixels IN FRONT of env
seg[pnear] = PALETTE[track_id[gid[pnear]] % len(PALETTE)] # color by persistent track
```

`pnear` re-uses the depth comparison as a per-pixel visibility test — a person occluded by a shelf correctly disappears. `gid` (open3d's per-frame mesh index) is mapped through `tracks_pf[t]` back to the **persistent track id**, so each person keeps one palette color across the whole video even as people enter/leave (same intent as §11's per-instance flat colors, minus the GL renderer).

## 16.12 How this script relates to everything above — one paragraph

It is method 13.E (depth-based novel-view synthesis) done properly: DA3 supplies the per-pixel depth and poses (§2–§3); meshing (§16.6) replaces fragile pixel splatting so there are no holes; raycasting (§16.9) replaces forward warping so occlusion is exact; the crane camera is authored with §6–§8's `look_at`/pitch/FOV math on top of a §9.1 gravity estimate; people come from a second, mesh-based model and are gauge-aligned by §16.4's ratio trick and de-duplicated by §16.5's projection stencil; and the outputs follow §11's conventions (near=bright depth, flat instance colors, globally-fixed normalization/scale to avoid flicker). Nothing in it is anything other than $K$, $K^{-1}$, $[R\mid t]$, a plane fit, and robust statistics.

**Final quiz (spans the whole note)**

40. `to_world` multiplies by `sc` *after* the extrinsic transform, while the person scale `s` is applied *before* it. Why is each order correct?
41. Why is the environment meshed and raycast instead of just splatting the depth-map pixels into the new view (13.E code)?
42. Derive why `depth[0][v,u] / c[2]` is the right per-sample scale estimate, and why the median (not mean) aggregates it.
43. In 16.9, what image artifact appears if you use `t_hit` as depth instead of `(hit-eye)·D`?
44. Point to the three separate places the script fights temporal flicker, and name the shared principle.
45. The crane uses pitch 45° rather than 90°. Give both the numerical and the practical reason.
46. Which §13 method is the whole script an instance of, and which two classic weaknesses of that method does it solve, how?

> **Answers.** 1: `sc` converts DA3's *world* to metric, so it scales world coordinates (post-transform); `s` aligns the person's *camera-frame* gauge to DA3's camera frame, and must happen before adding the (unscaled) cam→world translation — scaling after would corrupt $\mathbf e$. 2: pixel splatting forward-warps points → holes between samples and no occlusion ordering; triangles interpolate the surface (no holes) and raycasting returns the nearest hit (exact occlusion). 3: the mesh centroid and the DA3 depth sample lie on the same pixel ray; along one ray, position = depth × unit ray, so the two gauges differ by exactly the depth ratio; median is robust to bad samples (occlusion, projection landing on background). 4: depth would equal Euclidean ray length, which grows toward the corners → radially "bowed" depth; projecting onto D gives true z-depth. 5: environment built from frame 0 only and raycast once; global 2/98-percentile depth normalization; persistent track-id colors — shared principle: compute anything global ONCE over the clip, never per frame. 6: at 90° the look_at cross product $\mathbf f \times \mathbf u \to 0$ (degenerate, §6), and a pure nadir view flattens people into blobs — 45° keeps both the math stable and the people recognizable. 7: 13.E DIBR; holes (solved by meshing) and occlusion (solved by raycast z-ordering).

---

*Merged from: "Adjusting 3D Points & Camera Positions — Camera Math Guided Reading", "Top-down View Projection (with human)", and "Camera Projection & Changing an Image's Viewpoint — Deep Note" (all three retired into this page).*