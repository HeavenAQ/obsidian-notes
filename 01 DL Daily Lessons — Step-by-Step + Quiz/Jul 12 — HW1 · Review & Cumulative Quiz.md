---
base: "[[01.2 DL Daily Lessons — Step-by-Step + Quiz.base]]"
Quiz score: ""
"HW #": 1
Studied: false
Quiz taken: false
Day type: Review-Day
Date: 2026-07-12
Piece count: 0
---
> ⚠️ **Schedule note (buffer / catch-up day):** the dated checklist ran Jul 2–11; today (Jul 12) is the built-in **buffer day**, so no new lesson — this is a **cumulative review** before the homework days (Jul 13–15). Two gaps to be aware of before you start HW1: **Jul 4** (Udemy LinAlg: matrices + matrix calculus · PT-Cert C1 M1–M2 — mostly review) and **Jul 11** (TiP Module 3: quantization, KV-cache, flash-attention, speculative decoding · PT-Cert C3 M4 deployment/ONNX) never got a lesson file. Neither is load-bearing for HW1's graded problems (all HW1 math is covered Jul 2–10), so if time is tight, **skim** them during today's buffer and prioritise the quiz below.
>
> 🗒️ **Quiz backlog:** every quiz Jul 2–10 is still **untaken** (no scores recorded). The re-explain/patch-up loop only fires on a *low* recorded score, so nothing is queued — but you learn far more by grading a couple before today's cumulative set. Priority to grade: **Jul 9 (attention theory)** and **Jul 10 (nanoGPT code)**, since HW1 Problems 8 & 10 lean on exactly that math.

# 🎯 Review focus

Consolidate the whole HW1 window (Jul 2–10) into the handful of facts and formulas the homework actually tests, then verify retention with a mixed cumulative quiz. The thread running through every day: **a linear layer is a matrix multiply, its gradient is an outer product, and stacking (affine → nonlinearity) with the right normalisation is all a network — CNN or Transformer — really is.** Priority order for the remaining window per the plan: HW1 math prep > Transformers-in-Practice > PyTorch cert.

# 🧭 Window recap (Jul 2–10)

**Jul 2 — Vectors, dot products, planes.** $\mathbf{a}\cdot\mathbf{b}=\|\mathbf a\|\|\mathbf b\|\cos\theta$ measures alignment; a plane / **hyperplane** is $\mathbf w\cdot\mathbf x + b = 0$, and a **ReLU neuron** is exactly a hyperplane with an "on" half-space: $\mathrm{ReLU}(\mathbf w\cdot\mathbf x + b)$.

**Jul 3 — Matrix multiplication & inverses.** $(AB)_{ij}=\sum_k A_{ik}B_{kj}$; the deep-learning-relevant view is $AB=\sum_k (\text{col}_k A)(\text{row}_k B)$ — a **sum of outer products**. Matmul *is* function composition. Gradient shapes mirror weight shapes: $\partial L/\partial W$ has the same shape as $W$.

**Jul 5 — Matrix calculus + PyTorch core.** Linear-layer backprop: $\partial L/\partial x = W^\top g,\ \partial L/\partial W = g\,x^\top,\ \partial L/\partial b = g$. `Dataset`/`DataLoader` feed batches; `nn.Module` holds parameters; the optimiser step is `zero_grad → forward → loss → backward → step`.

**Jul 6 — CNNs, fine-tuning, Optuna.** Convolution = **weight sharing + locality**, cutting parameters vs. a dense layer and giving translation equivariance. **ResNet** adds residual skips $x + \mathcal F(x)$ so deep nets train; **fine-tuning** freezes the backbone and retrains the head. Optuna searches hyperparameters by sampling + pruning.

**Jul 7 — Embeddings & efficient training.** Static vectors (**GloVe/word2vec**) give one vector per word; **FastText** sums subword vectors so it handles OOV/morphology; **BERT/DistilBERT** give **contextual** vectors (same word, different vector by context). An embedding layer is a trainable lookup $E\in\mathbb R^{V\times d}$. Speed: **profile first**, then AMP + a good allocator + non-blocking dataloading.

**Jul 8 — TiP M1: the autoregressive loop.** Generation = repeatedly sample the next token and feed it back. **Temperature** flattens/sharpens the softmax; **top-k / top-p (nucleus) / min-p** truncate the tail. **Structured output** = mask illegal tokens (FSM). **RAG** and **CoT** are the *same* trick — inject helpful tokens into the context; the model/loop is unchanged.

**Jul 9 — TiP M2: attention.** One head: $\mathrm{softmax}\!\big(\tfrac{QK^\top}{\sqrt{d_k}}+M\big)V$. **Multi-head** = $H$ relation subspaces at equal cost. **Positional encoding** injects order because attention is permutation-invariant. The **logit lens** decodes the residual stream at each depth with the unembedding.

**Jul 10 — Karpathy nanoGPT.** Bigram baseline (table $(V,V)$ *is* the logits; init loss $\approx\ln V$). Attention as a data-dependent weighted average of the past; scale by $1/\sqrt{d_k}$ to keep softmax non-peaky; causal mask via $-\infty$ before softmax. Block = pre-LN residual: $x\leftarrow x+\mathrm{MHA}(\mathrm{LN}(x))$, $x\leftarrow x+\mathrm{FFN}(\mathrm{LN}(x))$. Slogan: **attention = communication across positions, MLP = computation within a position.**

**Recurring HW1 pitfalls to avoid:** (1) transposes in $g\,x^\top$ vs $W^\top g$ — check shapes, not habit; (2) forgetting the ReLU gate $\Theta(y)=\mathbb 1[y>0]$ zeros gradient where neurons are off; (3) a bias-free ReLU pins its kink at the origin (HW1 Prob 5); (4) softmax-CE gradient simplifies to $\hat p - y_{\text{onehot}}$ — don't re-derive the full softmax Jacobian at runtime; (5) mask the future with $-\infty$ **before** softmax, not by zeroing after.

# 📝 Cumulative quiz

Answer all 8 without notes (spans Jul 2–10). Grade with the key at the bottom, then fill in **Quiz score** and check **Quiz taken**.

1. **(Jul 2 — conceptual)** Write the dot-product formula relating $\mathbf a\cdot\mathbf b$ to the angle between vectors, and explain in one sentence how a single ReLU neuron $\mathrm{ReLU}(\mathbf w\cdot\mathbf x+b)$ relates to a hyperplane. What set does $\mathbf w\cdot\mathbf x+b=0$ describe?
2. **(Jul 3 — derivation / code)** Give the entrywise definition of $(AB)_{ij}$, then write $AB$ as a **sum of outer products**. Why is "matmul = a batch of outer products" the key fact behind $\partial L/\partial W = g\,x^\top$?
3. **(Jul 5 — derivation)** For a linear layer $\mathrm{out}=Wx+b$ with upstream gradient $g=\partial L/\partial \mathrm{out}$, write $\partial L/\partial x$, $\partial L/\partial W$, and $\partial L/\partial b$. What is the ReLU local gradient, and how does it enter the chain?
4. **(Jul 6 — conceptual)** State the two properties that distinguish a convolutional layer from a dense layer and what each buys you. What single equation defines a ResNet residual block, and why does it help train deep networks?
5. **(Jul 7 — conceptual)** Distinguish **static** (GloVe) from **contextual** (DistilBERT) embeddings in one sentence each. What problem does **FastText** solve that word2vec/GloVe cannot, and by what mechanism?
6. **(Jul 8 — conceptual)** Describe the autoregressive generation loop in one or two sentences. What does **temperature** do to the softmax, and how do **top-k** and **top-p** differ? In what sense are **RAG** and **chain-of-thought** the same underlying mechanism?
7. **(Jul 9 / Jul 10 — derivation)** Write the single-head scaled dot-product attention formula including the causal mask. Derive why the scale is $1/\sqrt{d_k}$ (state $\mathrm{Var}(q^\top k)$), and explain what breaks at initialisation if you drop it.
8. **(Jul 9 / Jul 10 — conceptual / code)** Why must a Transformer add positional information, and where does it enter? Write the two pre-LayerNorm residual update lines for a block and give the "communication vs computation" slogan that labels each line.

> [!note]- 🔑 Answer key (click to reveal)
> **1.** $\mathbf a\cdot\mathbf b=\|\mathbf a\|\,\|\mathbf b\|\cos\theta$ — it measures how aligned two vectors are (zero ⇒ orthogonal). A ReLU neuron computes an affine score $\mathbf w\cdot\mathbf x+b$ and fires ($>0$) on one side of the hyperplane, outputting 0 on the other; the neuron *is* a half-space detector. The set $\mathbf w\cdot\mathbf x+b=0$ is a **hyperplane** (a plane in 3-D, a line in 2-D) — the decision boundary, with $\mathbf w$ its normal.
>
> **2.** $(AB)_{ij}=\sum_k A_{ik}B_{kj}$. Column–row form: $AB=\sum_k (\text{col}_k A)(\text{row}_k B)$, a sum of rank-1 **outer products**. Because a weight gradient reassembles as (upstream vector) ⊗ (input vector), $\partial L/\partial W=g\,x^\top$ is exactly one such outer product; over a batch it becomes a sum of outer products, i.e. a single matmul $G^\top X$ — GPU-friendly.
>
> **3.** $\partial L/\partial x=W^\top g$, $\partial L/\partial W=g\,x^\top$, $\partial L/\partial b=g$ (batch: sum over the batch dim). ReLU's local gradient is the gate $\mathbb 1[x>0]$ (Heaviside $\Theta$); it enters elementwise: $\partial L/\partial x_{\text{pre}} = g\odot\mathbb 1[x>0]$, zeroing gradient wherever the neuron was off.
>
> **4.** Convolution vs dense: **(i) weight sharing** — the same small kernel slides over all positions, so parameters don't scale with input size and the layer is translation-equivariant; **(ii) locality** — each output depends only on a small receptive field, encoding the prior that nearby pixels matter. A ResNet block computes $y = x + \mathcal F(x)$; the additive skip lets gradients flow through the identity path (mitigating vanishing gradients) and makes it easy for a block to learn a small residual around identity, so very deep stacks still train.
>
> **5.** Static (GloVe/word2vec): one fixed vector per word type, independent of context. Contextual (BERT/DistilBERT): the vector depends on the surrounding sentence, so a polysemous word gets different vectors in different contexts. FastText solves **OOV and morphology**: it represents a word as the sum/mean of its character n-gram (subword) embeddings, so unseen or inflected words still get a sensible vector — impossible for word2vec/GloVe, which have no entry for an unseen word.
>
> **6.** The loop: feed the current context through the model, get next-token logits, sample one token, append it, repeat. **Temperature** $T$ divides the logits before softmax — $T>1$ flattens (more random), $T<1$ sharpens (more greedy). **Top-k** keeps the $k$ highest-probability tokens; **top-p (nucleus)** keeps the smallest set whose cumulative probability $\ge p$ (an adaptive count). RAG and CoT are the same mechanism: both **inject tokens into the context** (retrieved passages vs. intermediate reasoning steps) to condition generation — the model and sampling loop are unchanged.
>
> **7.** $\mathrm{head}(X)=\mathrm{softmax}\!\big(\tfrac{QK^\top}{\sqrt{d_k}}+M\big)V$, with $M_{ij}=0$ for $j\le i$ and $-\infty$ for $j>i$. For unit-variance i.i.d. entries, $q^\top k=\sum_{c=1}^{d_k}q_c k_c$ is a sum of $d_k$ independent mean-0, unit-variance terms, so $\mathrm{Var}(q^\top k)=d_k$ (std $\sqrt{d_k}$). Unscaled, logits grow like $\sqrt{d_k}$, pushing softmax into a **peaky** regime where one weight $\to 1$ and its gradient $\to 0$; attention then copies a single neighbour and can't learn. Dividing by $\sqrt{d_k}$ keeps logits $\mathcal O(1)$ and attention diffuse at init.
>
> **8.** Self-attention is permutation-invariant, so order must be injected explicitly — via a **positional encoding/embedding added to the token embedding** at the input ($x_0=\mathrm{TokEmb}+\mathrm{PosEmb}$). Block updates: $x\leftarrow x+\mathrm{MHA}(\mathrm{LN}(x))$ (**communication** across positions) then $x\leftarrow x+\mathrm{FFN}(\mathrm{LN}(x))$ (**computation** within each position). Attention mixes tokens; LayerNorm + MLP act per position.
>
> **Scoring:** 1 pt each (half credit for the right idea with a wrong detail). ≥ 6/8 → you're ready for the HW1 homework days (Jul 13–15). Below 6 → re-read the specific window day the missed question came from before starting the problem set; the tag on each question tells you which day.
