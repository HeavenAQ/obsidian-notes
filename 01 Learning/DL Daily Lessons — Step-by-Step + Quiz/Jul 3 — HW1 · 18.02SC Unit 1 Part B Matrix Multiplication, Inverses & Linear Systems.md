---
base: "[[DL Daily Lessons — Step-by-Step + Quiz.base]]"
Quiz score: ""
"HW #": 1
Studied: false
Quiz taken: false
Day type: Lesson-Day
Date: 2026-07-03
Piece count: 5
---
 🎯 Today's goal

By tonight you should be able to multiply matrices fluently in all the standard "views" (row·column, column-combination, and outer-product), explain WHY matrix multiplication is defined the way it is (composition of linear maps), invert a small matrix, and say precisely when $Ax=b$ has zero / one / infinitely many solutions. This feeds HW1 directly: Problems 1–2 are pure shape bookkeeping of matrix products, and Problem 8 / 10's backprop identities ($\partial L/\partial W = g\,x^\top$) are literally the outer-product view of matmul.

> ⏰ Yesterday's Part A quiz is still unscored — grade it first (10 min), then start below.

# 🧩 Pieces

## Piece 1 — Matrix multiplication: the rule and the reason (~30 min)

*Source: 18.02SC Unit 1 Part B, Sessions 9–10 (Matrix Multiplication; Meaning of Matrix Multiplication).*

The entry rule: for $A \in \mathbb{R}^{m\times n}$ and $B \in \mathbb{R}^{n\times p}$,

$$
(AB)_{ij} = \sum_{k=1}^{n} A_{ik}\,B_{kj}
$$

so $AB \in \mathbb{R}^{m\times p}$ — inner dimensions must match and get "consumed."

**Why this formula and not elementwise?** A matrix IS a linear map: $x \mapsto Bx$. If you apply $B$ then $A$, the composite map $x \mapsto A(Bx)$ is again linear, so it must be some matrix $C$. Writing out $A(Bx)$ in coordinates and demanding $Cx = A(Bx)$ for all $x$ *forces* exactly the sum above. Matrix multiplication is defined so that

$$
(AB)x = A(Bx)
$$

i.e. **multiplication = composition of functions**. That's also why it's associative but NOT commutative — composing "rotate then stretch" differs from "stretch then rotate."

```python
import torch

A = torch.randn(3, 4)
B = torch.randn(4, 2)

# entrywise definition, spelled out
C = torch.zeros(3, 2)
for i in range(3):
    for j in range(2):
        C[i, j] = sum(A[i, k] * B[k, j] for k in range(4))

print(torch.allclose(C, A @ B))  # True

# multiplication IS composition:
x = torch.randn(2)
print(torch.allclose((A @ B) @ x, A @ (B @ x)))  # True
```

**You've got this piece when you can** … compute a 2×3 · 3×2 product by hand AND explain in one sentence why the definition is forced by composition.

## Piece 2 — Four ways to see the same product (~30 min)

*Source: Session 10 + Udemy "Introduction To Matrices" (the 5-ways-to-multiply material — this piece is the bridge between the two courses).*

All of these are the SAME product $AB$, sliced differently:

1. **Row·column (dot product) view:** entry $(i,j)$ = (row $i$ of $A$) · (column $j$ of $B$).
2. **Column view:** $Ax$ is a *linear combination of the columns of* $A$:

$$
Ax = x_1\,a_{:,1} + x_2\,a_{:,2} + \cdots + x_n\,a_{:,n}
$$

So each column of $AB$ is $A$ times the corresponding column of $B$.

3. **Row view:** each row of $AB$ is (row of $A$) times $B$.
4. **Outer-product view:** the whole product is a *sum of rank-1 matrices*:

$$
AB = \sum_{k=1}^{n} a_{:,k}\, b_{k,:} \qquad (\text{column } k \text{ of } A \text{ outer row } k \text{ of } B)
$$

**Why care?** The outer-product view is exactly how backprop gradients arise. In HW1 Problem 10, a linear layer's weight gradient is $\partial L/\partial W = g\,x^\top$ — an outer product; over a batch it becomes $G^\top X$, a sum of per-sample outer products, i.e. view #4. If you internalize this now, the HW derivation becomes pattern-recognition.

$$
\frac{\partial L}{\partial W} = \sum_{b=1}^{B} g^{(b)} \otimes x^{(b)} = G^\top X
$$

```python
import torch

A = torch.randn(3, 4)
B = torch.randn(4, 2)

# view 4: sum of outer products (column_k of A) (row_k of B)
S = sum(torch.outer(A[:, k], B[k, :]) for k in range(4))
print(torch.allclose(S, A @ B))  # True

# backprop preview: batch of outer products == one matmul
g = torch.randn(8, 5)   # upstream grads, batch of 8
x = torch.randn(8, 3)   # layer inputs
dW = sum(torch.outer(g[b], x[b]) for b in range(8))
print(torch.allclose(dW, g.T @ x))  # True — this is Linear.backward's dW
```

**You've got this piece when you can** … reproduce all four views for a small example without notes, and state which one is $\partial L/\partial W$.

## Piece 3 — Inverse matrices (~25 min)

*Source: 18.02SC Part B, Session 11 (Matrix Inverses).*

$A^{-1}$ is the matrix that **undoes** the linear map $A$:

$$
A^{-1}A = AA^{-1} = I
$$

It exists (for square $A$) **iff** $\det A \neq 0$ — a zero determinant means $A$ squashes space into a lower dimension, and you can't un-squash (two different inputs land on the same output, so no well-defined inverse).

The 2×2 closed form (memorize):

$$
\begin{pmatrix} a & b \\ c & d \end{pmatrix}^{-1} = \frac{1}{ad-bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}
$$

**Why the order flip** in $(AB)^{-1} = B^{-1}A^{-1}$: to undo "apply $B$, then $A$" you must undo $A$ *first*, then $B$ — socks and shoes.

Practical note: numerically you almost never form $A^{-1}$; you call a solver.

```python
import torch

A = torch.tensor([[2., 1.], [5., 3.]])
Ainv = torch.linalg.inv(A)          # fine for tiny matrices
print(Ainv @ A)                      # ~identity

b = torch.tensor([4., 7.])
x1 = Ainv @ b
x2 = torch.linalg.solve(A, b)        # preferred: faster, more stable
print(torch.allclose(x1, x2))

torch.linalg.det(torch.tensor([[1., 2.], [2., 4.]]))  # 0 → singular, no inverse
```

**You've got this piece when you can** … invert a 2×2 by hand, and explain det = 0 ⇒ no inverse in terms of "squashing."

## Piece 4 — Linear systems $Ax=b$ and the geometry of planes (~30 min, pure math)

*Source: 18.02SC Part B, Sessions 12–14 (Equations of Planes; Linear Systems & Planes; Solutions to Square Systems).*

A 3×3 system is three plane equations; a solution is a point in **all three planes**. Row picture: intersecting planes. Column picture: can $b$ be built as a combination of $A$'s columns?

The trichotomy for square systems:

- $\det A \neq 0$: exactly **one** solution, $x = A^{-1}b$ (three planes meet in a single point).
- $\det A = 0$ and $b$ is "compatible": **infinitely many** solutions (planes share a line or coincide).
- $\det A = 0$ and $b$ is incompatible: **no** solution (e.g. parallel planes).

The homogeneous system $Ax = 0$ always has $x=0$; it has *nonzero* solutions iff $\det A = 0$ — i.e. iff the columns are linearly dependent. This is the same "linear independence / rank" idea from the Udemy matrices section, seen geometrically.

$$
Ax = 0 \text{ has } x \neq 0 \iff \det A = 0 \iff \text{columns of } A \text{ are linearly dependent}
$$

**Why this matters for DL:** "can this network's weight matrix map inputs onto the targets" is a column-space question, and rank-deficiency (dead/duplicated neurons) is exactly the $\det = 0$ degeneracy in disguise.

**You've got this piece when you can** … classify a given 3×3 system into one/none/infinitely-many by computing one determinant and checking compatibility.

## Piece 5 — Bridge to HW1: shapes, batching, and the PyTorch convention (~20 min)

*Source: synthesis piece (maps Parts A–B onto MIT 6.7960 HW1 Problems 1, 2, 8, 10).*

HW1 counts a network by its weight matrices: $f(x) = W_2\,\mathrm{ReLU}(W_1 x + b_1) + b_2$. Every shape question is just conformability from Piece 1: $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$, and adjacent layers must chain.

PyTorch stores batches as rows, so `nn.Linear` computes

$$
\text{out} = X W^\top + b, \qquad X \in \mathbb{R}^{B \times d_\text{in}},\; W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}
$$

— the transpose exists purely because samples are rows, not columns. Check every line of the snippet against the four views:

```python
import torch

B, d_in, d_h, d_out = 32, 3, 5, 2
X  = torch.randn(B, d_in)
W1 = torch.randn(d_h, d_in);  b1 = torch.randn(d_h)
W2 = torch.randn(d_out, d_h); b2 = torch.randn(d_out)

H = torch.relu(X @ W1.T + b1)   # (B, d_h)
Y = H @ W2.T + b2               # (B, d_out)

# gradient shapes must mirror the weights (HW1 Prob 10):
g  = torch.randn(B, d_out)      # dL/dY
dW2 = g.T @ H                   # (d_out, d_h) — sum of outer products, Piece 2!
db2 = g.sum(0)                  # (d_out,)
dH  = g @ W2                    # (B, d_h)
```

**You've got this piece when you can** … predict the shape of every intermediate tensor above before running it, and say why `dW2 = g.T @ H` is Piece 2's outer-product view.

# 📝 Review quiz

Answer all 7 without notes. Grade with the key at the bottom, then fill in Quiz score and check Quiz taken.

5. **(Conceptual)** Matrix multiplication could have been defined elementwise (like `A * B` in PyTorch). Explain in 2–3 sentences why the actual definition $(AB)_{ij} = \sum_k A_{ik}B_{kj}$ is the "right" one, and name one algebraic property this choice gives up.
6. **(Calculation)** For $A = \begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix}$ and $B = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, compute both $AB$ and $BA$. What do the results demonstrate?
7. **(Derivation)** Using the column view, write $Ax$ for $A = \begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{pmatrix}$, $x = \begin{pmatrix} 10 \\ -1 \end{pmatrix}$ explicitly as a combination of columns and evaluate it. Then state the general outer-product identity for $AB$ that this view generalizes to.
8. **(Calculation)** Invert $M = \begin{pmatrix} 3 & 1 \\ 5 & 2 \end{pmatrix}$ by hand and use $M^{-1}$ to solve $Mx = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$. Then verify the socks-and-shoes rule: why is $(AB)^{-1} = B^{-1}A^{-1}$ and not $A^{-1}B^{-1}$?
9. **(Conceptual)** For a square system $Ax=b$ with $\det A = 0$, describe BOTH possible solution outcomes and give the geometric picture (in terms of planes) for each. What extra condition on $b$ distinguishes them?
10. **(Code reading)** Given `g` of shape `(B, d_out)` (upstream gradient) and `x` of shape `(B, d_in)` (layer input), a student writes `dW = x.T @ g`. The correct weight has shape `(d_out, d_in)`. What's wrong, what's the fix, and which "view" of matrix multiplication explains what the fixed expression computes?
11. **(Code writing)** Without using `@` or `torch.matmul` on the full matrices, write a short function `matvec(A, x)` that computes $Ax$ using ONLY the column view (scalar–vector multiplies and adds). Then state what `torch.allclose(matvec(A, x), A @ x)` should print.

> [!note]+ 🔑 Answer key — open only after attempting all 7
> **1.** Matrices represent linear maps, and the definition is forced by requiring $(AB)x = A(Bx)$ — multiplication must implement *composition*. Elementwise product doesn't correspond to composing maps (and requires equal shapes, which composition doesn't). The property given up: **commutativity** — composition of maps depends on order, so in general $AB \neq BA$.
> 
> **2.** $AB = \begin{pmatrix} 2 & 1 \\ 1 & 0 \end{pmatrix}$, $BA = \begin{pmatrix} 0 & 1 \\ 1 & 2 \end{pmatrix}$. They differ → matrix multiplication is not commutative. ($B$ swaps coordinates; swapping before vs. after the shear $A$ gives different maps.)
> 
> **3.** Column view: $Ax = 10\begin{pmatrix}1\\2\\3\end{pmatrix} - 1\begin{pmatrix}4\\5\\6\end{pmatrix} = \begin{pmatrix}6\\15\\24\end{pmatrix}$. General identity: $AB = \sum_k a_{:,k}\,b_{k,:}$ — a sum of rank-1 outer products, one per inner-dimension index.
> 
> **4.** $\det M = 3\cdot 2 - 1\cdot 5 = 1$, so $M^{-1} = \begin{pmatrix} 2 & -1 \\ -5 & 3 \end{pmatrix}$. Then $x = M^{-1}\begin{pmatrix}1\\1\end{pmatrix} = \begin{pmatrix}1\\-2\end{pmatrix}$ (check: $3(1)+1(-2)=1$, $5(1)+2(-2)=1$ ✓). Socks-and-shoes: $AB$ means "apply $B$ first, then $A$"; undoing must reverse the order — undo $A$ first, then $B$: $(AB)(B^{-1}A^{-1})$ would not collapse, whereas $(AB)(B^{-1}A^{-1})^{\text{reversed}}$… formally $(AB)(B^{-1}A^{-1}) = A(BB^{-1})A^{-1} = AA^{-1} = I$, which is exactly $(AB)^{-1} = B^{-1}A^{-1}$.
> 
> **5.** $\det A = 0$ means the planes' normals are linearly dependent. Outcomes: (i) **infinitely many** solutions if $b$ is compatible — the planes intersect in a common line (or coincide); (ii) **no** solution if $b$ is incompatible — e.g. two parallel planes that never meet, or three planes forming a triangular "tent." The distinguishing condition: whether $b$ lies in the **column space** of $A$ (i.e., $b$ is buildable from $A$'s columns).
> 
> **6.** `x.T @ g` has shape `(d_in, B) @ (B, d_out) = (d_in, d_out)` — the *transpose* of the required `(d_out, d_in)`. Fix: `dW = g.T @ x`. The fixed expression is a **sum of outer products** $\sum_b g^{(b)} \otimes x^{(b)}$ — view #4 (outer-product view), which is exactly $\partial L/\partial W$ for a linear layer over a batch.
> 
> **7.**
> 
> ```python
> import torch
> 
> def matvec(A, x):
>     out = torch.zeros(A.shape[0])
>     for k in range(A.shape[1]):
>         out = out + x[k] * A[:, k]   # x_k times column k
>     return out
> 
> A = torch.randn(3, 2); x = torch.randn(2)
> print(torch.allclose(matvec(A, x), A @ x))  # prints: True
> ```
> 
> `True` — the column view computes the identical product.
> 
> **Scoring:** each question 1 pt (half credit for right idea, wrong arithmetic). Score ≥ 5/7 → move on; below → tomorrow starts with a patch-up.