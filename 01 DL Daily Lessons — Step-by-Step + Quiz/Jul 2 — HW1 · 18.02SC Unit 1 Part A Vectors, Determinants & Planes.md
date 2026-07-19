---
base: "[[01.2 DL Daily Lessons — Step-by-Step + Quiz.base]]"
Quiz score:
"HW #": 1
Studied: false
Quiz taken: false
Day type: Lesson-Day
Date: 2026-07-02
Piece count: 5
---
# 🎯 Today's goal

By tonight, you should be able to compute and *geometrically interpret* dot products, determinants, cross products, and plane equations without hesitation. This is the exact toolkit HW1 leans on: every neuron computes a dot product $w^\top x + b$, every ReLU kink is a hyperplane (a plane's big sibling), and HW1 Problems 4 & 7 are entirely about where those hyperplanes cut input space. Nail the geometry today and the "linear regions" story later this week will feel obvious.

# 🧩 Pieces

## Piece 1 — Vectors & vector arithmetic (~20 min)

**Source:** 18.02SC Part A, Session 1 (Vectors).

A vector $\mathbf{v} \in \mathbb{R}^n$ is simultaneously (1) an arrow with direction and length, and (2) a list of coordinates. Both views matter in DL: the coordinate view is what a tensor stores; the arrow view is what lets you reason about angles, projections, and decision boundaries.

The two operations that define a vector space:

$\mathbf{a} + \mathbf{b} = (a_1 + b_1, \dots, a_n + b_n), \qquad c\,\mathbf{a} = (c a_1, \dots, c a_n)$

**Why it's formulated this way:** addition is the parallelogram rule (tip-to-tail arrows) written in coordinates, and scalar multiplication stretches/flips an arrow without rotating it. Everything a neural-net layer does — $Wx + b$ — is built from exactly these two operations, which is *why* a layer with no nonlinearity can never do more than stretch, rotate, and shift.

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 0.0, -1.0])
print(a + b)        # elementwise, the parallelogram rule in coordinates
print(2.5 * a)      # scaling: same direction, 2.5x the length
print(torch.linalg.norm(a))  # length ||a|| = sqrt(1+4+9)
```

**You've got this piece when you can** draw $\mathbf{a} - \mathbf{b}$ as an arrow between two arrowheads and explain why $\|c\mathbf{a}\| = |c|\,\|\mathbf{a}\|$.

## Piece 2 — Dot product: lengths, angles, components (~35 min)

**Source:** 18.02SC Part A, Sessions 2–4 (Dot Products; Lengths & Angles; Vector Components).

The dot product has an algebraic and a geometric definition, and their equality is the workhorse fact of this whole unit:

$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = \|\mathbf{a}\|\,\|\mathbf{b}\|\cos\theta$

**Why the two sides are equal:** apply the law of cosines to the triangle with sides $\mathbf{a}$, $\mathbf{b}$, and $\mathbf{a}-\mathbf{b}$: expanding $\|\mathbf{a}-\mathbf{b}\|^2 = \|\mathbf{a}\|^2 + \|\mathbf{b}\|^2 - 2\|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$ coordinate-by-coordinate, all the squared terms cancel and only the cross terms $-2\sum a_i b_i$ survive. So "multiply matching coordinates and add" secretly measures an *angle*.

Consequences you'll use constantly:

- Length: $\|\mathbf{a}\| = \sqrt{\mathbf{a}\cdot\mathbf{a}}$
- Orthogonality test: $\mathbf{a} \perp \mathbf{b} \iff \mathbf{a}\cdot\mathbf{b} = 0$
- Component (scalar projection) of $\mathbf{a}$ along unit vector $\hat{\mathbf{u}}$: $\mathbf{a}\cdot\hat{\mathbf{u}}$

**DL connection:** a neuron's pre-activation $w^\top x + b$ is a dot product — it measures how much $x$ points in the direction $w$. Cosine similarity between embeddings is literally the $\cos\theta$ form. Matrix multiplication (tomorrow's material) is nothing but a grid of dot products.

```python
import torch

a = torch.tensor([3.0, 4.0])
b = torch.tensor([4.0, -3.0])
dot = a @ b                                  # algebraic side: sum of products
cos_theta = dot / (a.norm() * b.norm())      # geometric side
print(dot, cos_theta)                        # 0.0 -> orthogonal!
u = b / b.norm()
print(a @ u)                                 # component of a along b's direction
```

**You've got this piece when you can** decide from the *sign* of $\mathbf{a}\cdot\mathbf{b}$ alone whether the angle is acute, right, or obtuse — and explain why.

## Piece 3 — Determinants in 2D and 3D (~30 min)

**Source:** 18.02SC Part A, Sessions 5–6 (Area and Determinants in 2D; Volumes and Determinants in Space).

$\det\begin{pmatrix} a_1 & a_2 \\ b_1 & b_2 \end{pmatrix} = a_1 b_2 - a_2 b_1$

**Geometric meaning:** the absolute value is the **area** of the parallelogram spanned by rows $\mathbf{a}, \mathbf{b}$; the sign records orientation (counterclockwise vs. clockwise). In 3D, the $3\times 3$ determinant (expand along the first row with alternating signs, each entry times its $2\times 2$ minor) gives the **signed volume** of the parallelepiped:

$\det\begin{pmatrix} a_1 & a_2 & a_3 \\ b_1 & b_2 & b_3 \\ c_1 & c_2 & c_3 \end{pmatrix} = a_1\begin{vmatrix} b_2 & b_3 \\ c_2 & c_3 \end{vmatrix} - a_2\begin{vmatrix} b_1 & b_3 \\ c_1 & c_3 \end{vmatrix} + a_3\begin{vmatrix} b_1 & b_2 \\ c_1 & c_2 \end{vmatrix}$

**Why the formula looks like that:** area must be linear in each row, flip sign when rows swap, and equal 1 for the unit square — those three axioms *force* the $a_1b_2 - a_2b_1$ formula. The alternating signs in the 3×3 expansion are the same orientation bookkeeping one dimension up.

**The punchline to remember:** $\det = 0 \iff$ the vectors are linearly dependent (the parallelogram/parallelepiped is squashed flat $\Rightarrow$ zero area/volume). This is *the* geometric test for "do these vectors carry independent information" — it returns next week as matrix rank and invertibility, and it's why a weight matrix with dependent rows wastes neurons.

```python
import torch

A = torch.tensor([[3.0, 1.0], [1.0, 2.0]])
print(torch.linalg.det(A))       # 5.0 = area of the parallelogram

B = torch.tensor([[1.0, 2.0], [2.0, 4.0]])  # row2 = 2 * row1
print(torch.linalg.det(B))       # 0.0 -> squashed flat, dependent rows
```

**You've got this piece when you can** compute a 3×3 determinant by cofactor expansion by hand and state what $\det = 0$ means geometrically.

## Piece 4 — Cross product (~30 min)

**Source:** 18.02SC Part A, Session 7 (Cross Products).

Defined only in $\mathbb{R}^3$, via the symbolic determinant:

$\mathbf{a} \times \mathbf{b} = \det\begin{pmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ a_1 & a_2 & a_3 \\ b_1 & b_2 & b_3 \end{pmatrix} = (a_2 b_3 - a_3 b_2,\; a_3 b_1 - a_1 b_3,\; a_1 b_2 - a_2 b_1)$

Three defining properties, all inherited from the determinant:

- $\mathbf{a}\times\mathbf{b}$ is **perpendicular to both** $\mathbf{a}$ and $\mathbf{b}$ (check: $\mathbf{a}\cdot(\mathbf{a}\times\mathbf{b}) = 0$ because it's a determinant with a repeated row).
- Its length $\|\mathbf{a}\times\mathbf{b}\| = \|\mathbf{a}\|\|\mathbf{b}\||\sin\theta|$ = area of the parallelogram they span.
- Direction by the right-hand rule; swapping arguments flips the sign ($\mathbf{b}\times\mathbf{a} = -\mathbf{a}\times\mathbf{b}$ — a row swap).

**Why we care here:** the cross product is *the* machine for manufacturing a **normal vector** — feed it two directions lying in a plane, it hands back the plane's perpendicular. That's exactly what Piece 5 needs.

```python
import torch

a = torch.tensor([1.0, 0.0, 0.0])
b = torch.tensor([0.0, 1.0, 0.0])
n = torch.linalg.cross(a, b)
print(n)                    # [0,0,1]: perpendicular to both, right-hand rule
print(n @ a, n @ b)         # 0, 0 -> orthogonality check via dot products
```

**You've got this piece when you can** compute $\mathbf{a}\times\mathbf{b}$ by the determinant trick and verify perpendicularity with two dot products, without notes.

## Piece 5 — Equations of planes → hyperplanes in deep learning (~30 min)

**Source:** 18.02SC Part A, Session 8 (Equations of Planes).

A plane is fully determined by one point $P_0$ on it and one normal vector $\mathbf{n}$. A point $P$ lies on the plane iff the displacement $P - P_0$ is perpendicular to $\mathbf{n}$:

$\mathbf{n} \cdot (P - P_0) = 0 \quad\Longleftrightarrow\quad ax + by + cz = d, \text{ where } \mathbf{n} = (a,b,c),\; d = \mathbf{n}\cdot P_0$

**Why this formulation is powerful:** the left side $\mathbf{n}\cdot P - d$ is not just zero *on* the plane — its **sign tells you which side of the plane you're on**, and its magnitude (after dividing by $\|\mathbf{n}\|$) is the distance to the plane. One dot product = a complete "which side am I on?" oracle.

**This is the single most important idea of today for HW1.** A ReLU neuron computes $\mathrm{ReLU}(w^\top x + b)$. The set $\{x : w^\top x + b = 0\}$ is exactly a plane equation with normal $w$ and offset $-b$ — a **hyperplane**. The neuron is "off" on one side and "on" (linear) on the other. HW1 Problem 4's linear-region counting is about how many pieces a bunch of such hyperplanes cut space into; Problem 7's logic gates are about *placing* these hyperplanes so the on/off sides implement OR/XOR. Note the bias-free case: if $b = 0$ the hyperplane is forced through the origin — which is precisely the constraint that makes HW1 Problem 5(b) interesting.

```python
import torch

# The ReLU neuron's hyperplane: w.x + b = 0
w = torch.tensor([1.0, 1.0])
b = torch.tensor(-1.0)          # boundary: x1 + x2 = 1

pts = torch.tensor([[2.0, 2.0], [0.0, 0.0], [0.5, 0.5]])
pre_act = pts @ w + b           # signed side-of-plane values
print(pre_act)                  # [ 3., -1., 0.] : on-side / off-side / exactly on
print(torch.relu(pre_act))      # [ 3.,  0., 0.] : neuron active only on one side
```

**You've got this piece when you can** write the plane through a given point with a given normal in both forms, and state in one sentence what geometric object a ReLU neuron's kink lives on.

# 📝 Review quiz

Attempt all questions on paper *before* opening the answer key. Mix: conceptual, calculation, code.

1. **(Conceptual)** The algebraic dot product $\sum_i a_i b_i$ and the geometric form $\|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$ are equal. Sketch the law-of-cosines argument for *why*, in 3–4 lines.
2. **(Calculation)** Let $\mathbf{a} = (2, -1, 3)$ and $\mathbf{b} = (1, 4, 2)$. Compute (i) $\mathbf{a}\cdot\mathbf{b}$, (ii) $\|\mathbf{a}\|$, (iii) the component of $\mathbf{a}$ along the direction of $\mathbf{b}$. Is the angle between them acute or obtuse?
3. **(Calculation)** Compute $\det\begin{pmatrix} 1 & 2 & 0 \\ 3 & -1 & 2 \\ 0 & 1 & 1 \end{pmatrix}$ by cofactor expansion, and state the geometric meaning of your number's absolute value.
4. **(Conceptual)** Vectors $\mathbf{u} = (2, 4)$ and $\mathbf{v} = (3, 6)$: without computing any determinant, what is the area of the parallelogram they span, and what does that tell you about them? Now confirm with the 2×2 determinant.
5. **(Calculation)** Find $\mathbf{a}\times\mathbf{b}$ for $\mathbf{a} = (1, 2, 0)$, $\mathbf{b} = (0, 1, 3)$, then verify your answer is perpendicular to $\mathbf{a}$ using a dot product. What is the area of the parallelogram spanned by $\mathbf{a}$ and $\mathbf{b}$?
6. **(Calculation)** Find the equation of the plane through the point $(1, 0, 2)$ with normal $(3, -1, 2)$, in the form $ax + by + cz = d$. Then determine which side of the plane the origin is on, and how you know from a single number.
7. **(Conceptual, DL bridge)** A ReLU neuron computes $\mathrm{ReLU}(w^\top x + b)$ with $w = (1, -2)$, $b = 3$. Describe the neuron's kink boundary as a geometric object (give its equation), and state what changes about the possible boundaries if we force $b = 0$. Why did HW1 Problem 5 care about that?
8. **(Code-reading)** What does this print, and why — answer *before* running it:

```python
import torch
a = torch.tensor([1.0, 2.0, 2.0])
b = torch.tensor([-2.0, 2.0, -1.0])
print(a @ b)
print(torch.linalg.cross(a, b) @ a)
```

> [!note]+ 🔑 Answer key — open only after attempting all questions
> **1.** Consider the triangle with sides $\mathbf{a}$, $\mathbf{b}$ and third side $\mathbf{a}-\mathbf{b}$. Law of cosines: $\|\mathbf{a}-\mathbf{b}\|^2 = \|\mathbf{a}\|^2 + \|\mathbf{b}\|^2 - 2\|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$. Expanding the left side in coordinates gives $\sum (a_i - b_i)^2 = \sum a_i^2 + \sum b_i^2 - 2\sum a_i b_i$. The squared-length terms cancel on both sides, leaving $\sum a_i b_i = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$.
> 
> **2.** (i) $\mathbf{a}\cdot\mathbf{b} = 2\cdot1 + (-1)\cdot4 + 3\cdot2 = 2 - 4 + 6 = 4$. (ii) $\|\mathbf{a}\| = \sqrt{4+1+9} = \sqrt{14}$. (iii) $\|\mathbf{b}\| = \sqrt{1+16+4} = \sqrt{21}$, so the component is $\mathbf{a}\cdot\mathbf{b}/\|\mathbf{b}\| = 4/\sqrt{21} \approx 0.873$. Since the dot product is positive, $\cos\theta > 0$: the angle is **acute**.
> 
> **3.** Expand along the first row: $1\cdot\det\begin{pmatrix}-1 & 2\\ 1 & 1\end{pmatrix} - 2\cdot\det\begin{pmatrix}3 & 2\\ 0 & 1\end{pmatrix} + 0 = 1(-1-2) - 2(3-0) = -3 - 6 = -9$. $|{-9}| = 9$ is the **volume of the parallelepiped** spanned by the three row vectors; the negative sign means they form a left-handed (negatively oriented) triple.
> 
> **4.** $\mathbf{v} = 1.5\,\mathbf{u}$ — they are parallel (linearly dependent), so the parallelogram is squashed flat: **area 0**. Check: $\det = 2\cdot6 - 4\cdot3 = 0$. Zero determinant $\iff$ linear dependence.
> 
> **5.** $\mathbf{a}\times\mathbf{b} = (2\cdot3 - 0\cdot1,\; 0\cdot0 - 1\cdot3,\; 1\cdot1 - 2\cdot0) = (6, -3, 1)$. Perpendicularity: $(6,-3,1)\cdot(1,2,0) = 6 - 6 + 0 = 0$ ✓. Area $= \|(6,-3,1)\| = \sqrt{36+9+1} = \sqrt{46} \approx 6.78$.
> 
> **6.** $\mathbf{n}\cdot(P - P_0) = 0$ gives $3(x-1) - 1(y-0) + 2(z-2) = 0$, i.e. $3x - y + 2z = 7$. Plug in the origin: $3\cdot0 - 0 + 2\cdot0 - 7 = -7 < 0$, while points on the plane give 0 and points on the normal's side give positive values — so the origin is on the **negative side** (the side the normal points away from). One signed number answers the side question.
> 
> **7.** The kink boundary is the line (a hyperplane in $\mathbb{R}^2$) $\{x : x_1 - 2x_2 + 3 = 0\}$, i.e. $x_1 - 2x_2 = -3$, with normal $w = (1,-2)$. The neuron is inactive on the side where $w^\top x + b < 0$ and affine on the other. With $b = 0$ the boundary must **pass through the origin** — you can rotate it (choose $w$) but never translate it. HW1 Problem 5 exploits exactly this: a bias-free width-2 net can only place kinks at $x = 0$, so two kinks at two *distinct* locations are impossible.
> 
> **8.** First line: $1\cdot(-2) + 2\cdot2 + 2\cdot(-1) = -2 + 4 - 2 = 0$ → prints `tensor(0.)`; the vectors are orthogonal. Second line: $\mathbf{a}\times\mathbf{b}$ is by construction perpendicular to $\mathbf{a}$, so the dot product with $\mathbf{a}$ is always $0$ regardless of the numbers → prints `tensor(0.)`. (Determinant view: it's a $3\times 3$ determinant with a repeated row.)