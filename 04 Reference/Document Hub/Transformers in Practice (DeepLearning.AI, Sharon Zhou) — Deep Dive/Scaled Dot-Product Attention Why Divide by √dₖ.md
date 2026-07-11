---

---
- [ ] # **Core formula**

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

- $Q$: queries
- $K$: keys
- $V$: values
- $d_k$: dimension of each query/key vector
- $QK^\top$: attention-score matrix

---

## Main idea

Without dividing by $\sqrt{d_k}$:

$$
QK^\top
$$

→ larger attention scores as $d_k$ increases

→ softmax becomes very sharp

→ attention becomes almost one-hot

→ softmax becomes insensitive to score changes

→ weak gradients through $Q$ and $K$

With dividing by $\sqrt{d_k}$:

$$
\frac{QK^\top}{\sqrt{d_k}}
$$

→ scores remain moderate

→ attention stays flexible

→ softmax remains sensitive to score changes

→ more stable learning

---

## 1. Why do raw dot-product scores grow?

For one query vector $q$ and key vector $k$:

$$
\text{score} = q \cdot k = q_1 k_1 + q_2 k_2 + \dots + q_{d_k} k_{d_k}
$$

If each element of $q$ and $k$ has approximately:

- mean $= 0$
- variance $= 1$
- independent values

Then:

$$
\operatorname{Var}(q \cdot k) \approx d_k
\qquad
\operatorname{Std}(q \cdot k) \approx \sqrt{d_k}
$$

So, increasing $d_k$ makes raw $QK^\top$ scores naturally larger.

Example:

- $d_k = 16 \rightarrow$ typical score scale $\approx 4$
- $d_k = 64 \rightarrow$ typical score scale $\approx 8$
- $d_k = 256 \rightarrow$ typical score scale $\approx 16$
- $d_k = 1024 \rightarrow$ typical score scale $\approx 32$

Dividing by $\sqrt{d_k}$ corrects this:

$$
\operatorname{Std}\!\left(\frac{q \cdot k}{\sqrt{d_k}}\right) \approx 1
$$

---

## 2. Why are large attention scores a problem?

Attention weights are produced with softmax:

$$
p_i = \frac{\exp(s_i)}{\sum_j \exp(s_j)}
$$

Suppose the attention scores are:

$$
s = [12,\ 3,\ -2]
$$

Then:

$$
\text{softmax}(s) \approx [0.9999,\ 0.0001,\ 0.0000]
$$

This is called **softmax saturation**.

The attention head is effectively saying:

> "Put almost all attention on one token and almost none on the others."

The problem is not only that attention becomes sharp. The problem is that a small change in the scores no longer changes the probabilities very much.

---

## 3. What is the softmax Jacobian?

Softmax takes a vector of scores and returns a vector of probabilities:

$$
s = [s_1, s_2, \dots, s_n]
\qquad
p = \text{softmax}(s)
$$

A normal derivative describes how one output changes when one input changes:

$$
\frac{dy}{dx}
$$

Softmax has many inputs and many outputs, so its derivative is a matrix called the **Jacobian**:

$$
J_{ij} = \frac{\partial p_i}{\partial s_j}
$$

Meaning:

> If attention score $s_j$ changes slightly, how much does attention probability $p_i$ change?

---

## 4. Softmax Jacobian formula

$$
J_{ij} = p_i(\delta_{ij} - p_j)
$$

where:

- $\delta_{ij} = 1$ when $i = j$
- $\delta_{ij} = 0$ when $i \neq j$

Two useful cases:

**Effect of a score on its own probability**

$$
\frac{\partial p_i}{\partial s_i} = p_i(1 - p_i)
$$

**Effect of one score on another probability**

$$
\frac{\partial p_i}{\partial s_j} = -p_i p_j, \qquad i \neq j
$$

The negative sign happens because all attention probabilities must sum to 1. Increasing attention toward one token decreases attention toward the others.

Matrix form:

$$
J_{\text{softmax}} = \operatorname{diag}(p) - p p^\top
$$

---

## 5. Non-saturated vs. saturated attention

### Non-saturated attention

$$
p = [0.50,\ 0.30,\ 0.20]
$$

The Jacobian has reasonably large values, such as:

- $\dfrac{\partial p_1}{\partial s_1} = 0.25$
- $\dfrac{\partial p_2}{\partial s_2} = 0.21$
- $\dfrac{\partial p_3}{\partial s_3} = 0.16$

Interpretation:

Small score changes still produce meaningful probability changes.

The model can learn useful query-key relationships.

### Saturated attention

$$
p = [0.999,\ 0.001,\ 0.000]
$$

Now:

- $\dfrac{\partial p_1}{\partial s_1} \approx 0.001$
- $\dfrac{\partial p_2}{\partial s_2} \approx 0.001$
- Most other Jacobian entries are also near $0$

Interpretation:

The attention probabilities barely react when the scores change.

The softmax is close to a hard, nearly one-hot decision.

---

## 6. Why does this weaken gradients?

During backpropagation:

$$
\frac{\partial L}{\partial \text{scores}} = J_{\text{softmax}}^\top \, \frac{\partial L}{\partial \text{attention\_weights}}
$$

When softmax saturates:

$$
J_{\text{softmax}} \approx 0
$$

Therefore:

$$
\frac{\partial L}{\partial \text{scores}} \approx 0
$$

Because:

$$
\text{scores} = \frac{QK^\top}{\sqrt{d_k}}
$$

weak gradients through the score matrix make it difficult to update $Q$ and $K$.

More precise statement:

> Unscaled dot-product attention can cause softmax saturation, which weakens the gradients used to learn query-key attention relationships.

This does **not** necessarily mean every gradient in the Transformer vanishes. It mainly affects the attention-score path.

---

## 7. Visual summary

![[99 Assets/Media/image 7.png]]

**Plot 1 — Score magnitude**

- Raw $QK^\top$ score variation grows approximately as $\sqrt{d_k}$.
- Dividing by $\sqrt{d_k}$ keeps score variation near a constant scale.

**Plot 2 — Attention entropy**

- Without scaling, entropy drops as $d_k$ increases.
- Attention becomes nearly one-hot.
- With scaling, the attention distribution stays less extreme.

**Plot 3 — Softmax sensitivity**

- The plot uses: $\operatorname{trace}(J) = 1 - \sum_i p_i^2$
- For one-hot attention, $p = [1, 0, \dots]$, so $\operatorname{trace}(J) = 0$.
- A value near zero means the softmax output barely changes when scores change.

**Plot 4 — Jacobian comparison**

- Left: non-saturated attention has noticeable Jacobian values.
- Right: saturated attention has Jacobian values close to zero.
- This is why sharp unscaled softmax can weaken gradients.

---

## Key takeaway

Dividing $QK^\top$ by $\sqrt{d_k}$ prevents larger attention dimensions from creating excessively large logits.

Without scaling:

$$
QK^\top \rightarrow \text{large logits} \rightarrow \text{saturated softmax} \rightarrow \text{near-zero Jacobian} \rightarrow \text{weak gradients through } Q \text{ and } K
$$

With scaling:

$$
\frac{QK^\top}{\sqrt{d_k}} \rightarrow \text{stable logits} \rightarrow \text{flexible attention} \rightarrow \text{useful gradients} \rightarrow \text{more stable learning}
$$

> **Memory sentence:** Dividing by $\sqrt{d_k}$ prevents attention scores from becoming so large that softmax turns almost one-hot and stops providing useful gradients.
