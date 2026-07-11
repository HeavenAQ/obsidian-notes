---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-07T12:44:00
Status: Done
Last updated time: 2026-07-07T12:44:00
Last edited by: Heaven Chen
Category:
  - CV
  - Theory
  - Pose Estimation
---
> The math (with NumPy/cv2 code) behind (a) how a parametric human **mesh** is computed — SMPL / SMPL-X — and (b) how **self-contact** on that mesh is defined and optimized, from *On Self-Contact and Human Pose* (TUCH, Müller et al., CVPR 2021, arXiv:2104.03176). SMPL-X is the body model TUCH uses because it captures body + hands + face, which is what makes reasoning about hand–face / hand–body contact possible.

## 1. The mesh model as a differentiable function

SMPL-X maps low-dimensional parameters to a full 3D mesh:

$M(\beta,\theta,\psi) = W\big(T_P(\beta,\theta,\psi),\, J(\beta),\, \theta,\, \mathcal{W}\big)$

where $\beta$ = shape coefficients, $\theta$ = pose (axis-angle per joint), $\psi$ = facial expression (SMPL-X only), $\mathcal{W}$ = blend weights, and $W(\cdot)$ = **linear blend skinning (LBS)**. Output: $N$ vertices (**SMPL: N=6890, K=24 joints**; **SMPL-X: N=10475**, adds MANO hands + FLAME face, expression, jaw/eyes).

## 2. Mesh calculation (SMPL / SMPL-X), step by step

**(a) Morph the template** with additive blend shapes:

$T_P(\beta,\theta,\psi) = \bar{T} + B_S(\beta) + B_P(\theta) + B_E(\psi)$

- Shape: $B_S(\beta) = \sum_{n} \beta_n\, S_n$ (linear PCA shape space; $S_n$ are shape blendshapes).
- Pose: $B_P(\theta) = \sum_{n} \big(R_n(\theta) - R_n(\theta^*)\big)\, P_n$ — corrective shapes driven by the rotation matrices relative to the rest pose $\theta^*$ (fixes LBS artifacts).
- Expression (SMPL-X): $B_E(\psi) = \sum_{n} \psi_n\, E_n$.

**(b) Regress joints** from the shaped template: $J(\beta) = \mathcal{J}\,\big(\bar{T} + B_S(\beta)\big)$, where $\mathcal{J}$ is a sparse joint regressor (each joint = weighted sum of nearby vertices).

**(c) Pose via LBS.** Each posed vertex is a blend-weighted sum of the joints' global transforms $G_k$:

$v_i' = \sum_{k=1}^{K} w_{k,i}\, G_k(\theta, J)\, \bar{v}_{P,i}$

with $G_k$ the world transform of joint $k$ obtained by walking the kinematic tree: $G_k = \prod_{p\in \mathrm{ancestors}(k)} \begin{bmatrix} R_p & J_p\\ 0 & 1\end{bmatrix}$ (a rest-pose transform is subtracted so the zero pose is the identity). $R_p = \exp(\theta_p^\wedge)$ is Rodrigues of the axis-angle.

### NumPy: a faithful SMPL forward pass

```python
import numpy as np

def rodrigues(r):
    """axis-angle (3,) -> rotation matrix (3,3)."""
    th = np.linalg.norm(r)
    if th < 1e-8:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)

def smpl_forward(betas, thetas, v_template, shapedirs, posedirs,
                 J_regressor, weights, parents):
    """
    betas      (B,)            shape coeffs
    thetas     (K,3)           axis-angle per joint (row 0 = root)
    v_template (N,3)           mean template T-bar
    shapedirs  (N,3,B)         shape blendshapes S
    posedirs   (N,3,9*(K-1))   pose blendshapes P
    J_regressor(K,N)  weights (N,K)  parents (K,)
    returns    verts (N,3), joints (K,3)
    """
    N, K = v_template.shape[0], thetas.shape[0]
    # (a) shape blendshapes  T-bar + B_S(beta)
    v_shaped = v_template + shapedirs @ betas                 # (N,3)
    # (b) joints from shaped template  J(beta)
    J = J_regressor @ v_shaped                                # (K,3)
    # (a) pose blendshapes  B_P(theta) : feature = (R_k - I) for non-root joints
    R = np.stack([rodrigues(t) for t in thetas])              # (K,3,3)
    pose_feat = (R[1:] - np.eye(3)).reshape(-1)               # (9*(K-1),)
    v_posed = v_shaped + (posedirs @ pose_feat)               # (N,3)
    # (c) global joint transforms along the kinematic tree
    G = np.zeros((K, 4, 4))
    G[0] = np.block([[R[0], J[0][:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])
    for k in range(1, K):
        A = np.block([[R[k], (J[k] - J[parents[k]])[:, None]],
                      [np.zeros((1, 3)), np.ones((1, 1))]])
        G[k] = G[parents[k]] @ A
    # subtract rest-pose so zero pose -> identity
    for k in range(K):
        rest = G[k] @ np.append(J[k], 0.0)
        G[k][:, 3] -= rest
    # (c) linear blend skinning  v' = (sum_k w_ki G_k) [v_posed; 1]
    T = np.tensordot(weights, G, axes=([1], [0]))             # (N,4,4)
    vh = np.concatenate([v_posed, np.ones((N, 1))], axis=1)   # (N,4)
    verts = np.einsum('nij,nj->ni', T, vh)[:, :3]             # (N,3)
    return verts, J
```

In practice you load these tensors from the official `.pkl`/`.npz` model (or use the `smplx` PyTorch package for autograd); the code above is the exact math, unrolled.

## 3. Self-contact on the mesh (TUCH, §3)

Naive "triangles intersect" over-counts non-functional contact (crotch, armpits). TUCH defines contact by combining a **small Euclidean distance** with a **large geodesic (on-surface) distance** — i.e. two points that are physically touching but far apart along the body surface.

**Definition (TUCH Def. 3.1).** Two vertices $v_i, v_j \in M_V$ are in **self-contact** iff

$\lVert v_i - v_j\rVert < t_{\text{eucl}}$   **and**   $\mathrm{geo}(v_i, v_j) > t_{\text{geo}}$,

where $\mathrm{geo}(\cdot,\cdot)$ is the **geodesic distance precomputed once on the neutral, mean-shaped** SMPL/SMPL-X model (shape-independent), and $t_{\text{eucl}}, t_{\text{geo}}$ are thresholds.

**Contact set:** $M_C := \{(v_i,v_j)\mid v_i,v_j\in M_V,\ \lVert v_i-v_j\rVert<t_{\text{eucl}},\ \mathrm{geo}(v_i,v_j)>t_{\text{geo}}\}$.

$M$ is a **self-contact mesh** when $|M_C| > 0$.

### NumPy / SciPy: detect self-contact vertex pairs

```python
import numpy as np
from scipy.spatial import cKDTree

def self_contact_pairs(V, geo, t_eucl=0.01, t_geo=0.30):
    """
    V   (N,3)   posed vertices (meters)
    geo (N,N)   geodesic distances precomputed on the MEAN-shape mesh
    returns list of (i,j) in self-contact  ==  the set M_C
    """
    tree = cKDTree(V)
    near = tree.query_pairs(r=t_eucl)            # (i) Euclidean-near candidate pairs
    Mc = [(i, j) for (i, j) in near if geo[i, j] > t_geo]  # (ii) geodesic-far filter
    return Mc

def is_self_contact_mesh(V, geo, **kw):
    return len(self_contact_pairs(V, geo, **kw)) > 0     # |M_C| > 0
```

Note: a dense $N\times N$ geodesic matrix is huge for N=6890/10475 — in the real code it is precomputed once on the mean mesh and stored, or replaced by a coarse **body-part segmentation** (only cross-part pairs count as contact). The KD-tree gives the $O(N\log N)$ Euclidean-near query; the geodesic table is the cheap look-up that removes trivial (armpit/crotch) contact.

## 4. Using contact to fit the mesh (SMPLify-XMC / SMPLify-DC / TUCH)

Contact reduces pose ambiguity by removing depth degrees of freedom. TUCH's optimizers extend **SMPLify-X**; the objective is a sum of a 2D data term + priors + contact terms:

$E(\beta,\theta) = E_J + \lambda_\theta E_\theta + \lambda_\beta E_\beta + \lambda_\alpha E_\alpha + \lambda_{\mathcal P} E_{\mathcal P} + \lambda_C E_{C}$

**Data (reprojection) term** — project the 3D joints into the image and compare to detected 2D keypoints, with a robust $\rho$ (Geman-McClure):

$E_J = \sum_j \gamma_j\,\rho\big(\Pi_K(R_\theta\, J(\beta)_j) - J^{\text{est}}_j\big)$.

**Attraction / contact term** — pull vertices that should touch **onto** each other (for $M_C$, or for annotated region pairs in SMPLify-DC / mimicked contact in SMPLify-XMC):

$E_{\text{attr}} = \sum_{(v_i,v_j)\in M_C} \lVert v_i - v_j\rVert$.

**Repulsion / interpenetration term** $E_{\mathcal P}$ — push interpenetrating surfaces apart (SMPLify-X uses a signed-distance / bounding-volume penalty on penetration depth) so contact does not become overlap.

### cv2 + NumPy: the reprojection data term and the contact losses

```python
import cv2
import numpy as np

def project_points(X, rvec, tvec, K, dist=None):
    """3D (N,3) -> 2D pixels (N,2) via the pinhole model (OpenCV)."""
    pts, _ = cv2.projectPoints(X.astype(np.float64), rvec, tvec, K, dist)
    return pts.reshape(-1, 2)

def gmof(x, sigma=100.0):
    """Geman-McClure robustifier rho(x)."""
    x2 = (x ** 2).sum(-1)
    return (sigma ** 2) * x2 / (sigma ** 2 + x2)

def data_term(J3d, rvec, tvec, K, kp2d, conf):
    """E_J = sum_j gamma_j * rho( proj(J_j) - kp_j )."""
    proj = project_points(J3d, rvec, tvec, K)          # (J,2)
    return (conf * gmof(proj - kp2d)).sum()

def attraction_term(V, contact_pairs):
    """E_attr = sum_{(i,j) in M_C} ||v_i - v_j||  (pull contacts together)."""
    if not contact_pairs:
        return 0.0
    idx = np.asarray(contact_pairs)
    d = V[idx[:, 0]] - V[idx[:, 1]]
    return np.linalg.norm(d, axis=1).sum()

def interpenetration_term(V, inside_depth):
    """E_P: penalize penetration depth of vertices that are inside the body.
    `inside_depth` (N,) >= 0 = signed penetration from an SDF / part BVH; 0 outside."""
    return (inside_depth ** 2).sum()
```

In a real fit these run under autograd (PyTorch `smplx`) so gradients flow back to $\beta,\theta$; the NumPy/cv2 versions above are the exact quantities being minimized. **SMPLify-XMC** additionally fixes the *known* mimicked 3DCP pose as a strong prior (optimizing mainly global orientation + contact), and **SMPLify-DC** turns *discrete* part-pair contact labels into the pairs fed to `attraction_term`.

## 5. Why this matters (link to the thesis)

The **Euclidean-near + geodesic-far** test is the crisp, reusable definition of "a body part is touching another part of the same body" — exactly the geometric substrate of a **self-adaptor** (self-touch). Detecting it on a reconstructed SMPL-X mesh is a surface-contact signal that is far more faithful than 2D "hand-near-face" proximity heuristics, because it (i) uses true 3D distance and (ii) excludes trivial always-touching regions via the geodesic term.

## Sources

- [On Self-Contact and Human Pose (TUCH) — arXiv:2104.03176](https://arxiv.org/abs/2104.03176)
- [TUCH code / self-contact library](https://github.com/muelea/tuch)
- SMPL (Loper et al., SIGGRAPH Asia 2015); SMPL-X (Pavlakos et al., CVPR 2019, `smplx`)