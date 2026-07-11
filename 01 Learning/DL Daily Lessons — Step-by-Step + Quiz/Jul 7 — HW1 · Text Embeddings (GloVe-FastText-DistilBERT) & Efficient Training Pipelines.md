---
base: "[[DL Daily Lessons — Step-by-Step + Quiz.base]]"
Quiz score: ""
"HW #": 1
Studied: false
Quiz taken: false
Day type: Lesson-Day
Date: 2026-07-07
Piece count: 5
---
> ⚠️ **Schedule note:** today follows the plan — **PT-Cert C2 M3–M4**: *Text with Hugging Face* (GloVe / FastText / DistilBERT embeddings) and *Efficient Training Pipelines* (profiling, finding bottlenecks). Deep-dive companion: [PyTorch × NVIDIA GPU — Training Internals](https://app.notion.com/p/391e445b30a48184a35fca32915f2b2c) **§3–5, 8** (AMP, caching allocator, dataloading, profiling). This is the lowest-priority strand of the two-week block (HW1 math > Transformers-in-Practice > PT-Cert), so if the day runs long, do Pieces 1–4 fully and treat Piece 5 as a lighter skim — the profiling/AMP material resurfaces in the HW-day consolidation.

> 

> 🗒️ **Quiz backlog is piling up:** Jul 2, 3, 5, and 6 quizzes are all still untaken. Grade **at least yesterday's (Jul 6)** before you start today, ~10 min — otherwise the patch-up loop can't help you.

# 🎯 Today's goal

Understand the three families of text representation you'll meet in any NLP pipeline — **static word embeddings** (word2vec/GloVe), **subword embeddings** (FastText), and **contextual embeddings** (BERT/DistilBERT) — well enough to state the math behind each and say *why* each was invented to fix the previous one's weakness. Then be able to pull a DistilBERT embedding out of a sentence in a few lines of Hugging Face + PyTorch. Finally, learn to *measure before you optimize*: use `torch.profiler` to tell whether training is compute-bound or dataloading-bound, and know the two biggest throughput levers (mixed precision + a non-starved input pipeline). None of this is on HW1 directly, but the profiling/AMP habits carry into every training run you'll do this course.

# 🧩 Pieces

## Piece 1 — Static word embeddings: word2vec & GloVe (PT-Cert C2 M3, ~35 min)

*Source: PT-Cert C2 M3 (Text with Hugging Face), embeddings intro. This is the conceptual base for everything else today.*

Text must become vectors before a network can touch it. The naive choice — **one-hot** vectors of length $V$ (vocabulary size) — is useless: every pair of words is equidistant, so "cat" is no closer to "dog" than to "algebra". **Word embeddings** instead map each word to a dense $d$-dimensional vector (a learned lookup table $E \in \mathbb{R}^{V \times d}$, $d \ll V$) where geometry encodes meaning. The guiding principle is the **distributional hypothesis**: *a word is characterized by the company it keeps* — words appearing in similar contexts should land near each other.

**word2vec (skip-gram)** learns this by predicting context words from a center word. Over a corpus of $T$ tokens and window $m$ it maximizes

$$
\frac{1}{T}\sum_{t=1}^{T}\ \sum_{\substack{-m \le j \le m \\ j \ne 0}} \log p(w_{t+j} \mid w_t),
\qquad
p(o \mid c) = \frac{\exp(u_o^\top v_c)}{\sum_{w=1}^{V}\exp(u_w^\top v_c)}
$$

where $v_c$ is the center-word vector and $u_o$ the context-word vector. The denominator sums over the whole vocabulary, so in practice it is approximated with **negative sampling** or **hierarchical softmax** — remember those two names.

**GloVe** takes a complementary, count-based route: it factorizes the global co-occurrence matrix $X$, where $X_{ij}$ = how often word $j$ appears in the context of word $i$. It minimizes

$$
J = \sum_{i,j=1}^{V} f(X_{ij})\,\big(w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}\big)^2,
\qquad
f(x) = \min\!\left(1,\ (x/x_{\max})^{\alpha}\right),\ \alpha=\tfrac34
$$

The target is $w_i^\top \tilde{w}_j \approx \log X_{ij}$, and the weighting $f$ down-weights rare co-occurrences (noisy) and caps the influence of extremely frequent pairs ("the", "of").

**Why this gives linear analogies.** Because dot products fit *log-counts*, vector **differences** fit *log-ratios* of co-occurrence. The ratio $X_{ik}/X_{jk}$ is what distinguishes "ice" from "steam" (large for $k=$"solid", small for $k=$"gas"), so semantic relations become consistent vector offsets — hence the famous $\text{king} - \text{man} + \text{woman} \approx \text{queen}$.

```python
import torch, torch.nn as nn
# An embedding layer is just a trainable lookup table E ∈ R^{V×d}
emb = nn.Embedding(num_embeddings=30000, embedding_dim=100)
ids = torch.tensor([[5, 91, 12]])      # (batch=1, seq_len=3) token ids
print(emb(ids).shape)                  # -> (1, 3, 100)
# Pretrained GloVe/word2vec vectors can be loaded straight into the weight:
# emb.weight.data.copy_(pretrained_matrix)   # (V, d) tensor, then optionally freeze
```

**You've got this piece when you can** state the distributional hypothesis, write the skip-gram softmax and name its two speedups, write GloVe's weighted least-squares objective and its $\log X_{ij}$ target, and explain in one sentence why vector *differences* encode analogies.

## Piece 2 — FastText: subword embeddings & the limits of static vectors (~25 min)

*Source: PT-Cert C2 M3 (Text with Hugging Face), FastText segment.*

word2vec and GloVe share two weaknesses: (1) a **fixed vocabulary** — any word unseen at training time is **out-of-vocabulary (OOV)** and gets no vector at all; and (2) they ignore **morphology** — "run", "runs", "running" are independent rows of $E$ with no shared structure. **FastText** fixes both by representing a word as the sum of its **character n-gram** vectors. Writing $\mathcal{G}_w$ for the set of character n-grams of $w$ (typically lengths 3–6, with boundary markers `<` `>`, e.g. `where` → `<wh, whe, her, ere, re>`, plus the whole word):

$$
v_w = \sum_{g \in \mathcal{G}_w} z_g,
\qquad
s(w, c) = \sum_{g \in \mathcal{G}_w} z_g^\top v_c
$$

where $z_g$ is a learned n-gram embedding and $s(w,c)$ replaces the plain dot product inside the skip-gram objective from Piece 1.

**Why it helps.** OOV words are no longer empty: an unseen word is embedded by composing the n-gram vectors it *does* share with training words. And morphologically related words share many n-grams, so they automatically land near each other — a big win for rare words and morphologically rich languages (Finnish, Turkish, German compounds).

**The shared limit of ALL of the above — word2vec, GloVe, FastText — is that they are *****static*****:** exactly one vector per word (or per word-as-bag-of-n-grams), fixed after training. So **polysemy** is unresolved — "bank" gets a single vector that blends *river bank* and *money bank*, no matter the sentence. That single observation is the entire motivation for the next piece.

```python
# Conceptually: a word's vector = mean/sum of its subword-hash embeddings.
def word_vector(word, ngram_emb, n_lo=3, n_hi=6):
    w = f"<{word}>"
    grams = [w[i:i+n] for n in range(n_lo, n_hi+1) for i in range(len(w)-n+1)]
    vecs = [ngram_emb[g] for g in grams if g in ngram_emb]   # known n-grams only
    return sum(vecs) / max(len(vecs), 1)                     # OOV-robust: rarely empty
```

**You've got this piece when you can** name the two weaknesses of word2vec/GloVe, write $v_w$ as a sum of n-gram embeddings, give a concrete reason FastText handles OOV and morphology, and state the one limitation that *every* static embedding still has.

## Piece 3 — Contextual embeddings: BERT → DistilBERT (~35 min)

*Source: PT-Cert C2 M3 (Text with Hugging Face), Transformers/DistilBERT segment. This is the payoff of today's NLP half.*

A **contextual embedding** gives each token a vector that depends on the *whole sentence*, so "bank" gets different vectors in "river bank" and "savings bank". **BERT** produces these with a stack of Transformer **encoder** layers whose core operation is **scaled dot-product self-attention**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Each token's representation becomes a data-dependent weighted average of *all* tokens' values, which is exactly how context flows in. **Why the **$\sqrt{d_k}$**?** For $Q,K$ with unit-variance entries, $q^\top k$ has variance $\approx d_k$; dividing by $\sqrt{d_k}$ rescales the logits back to variance $\approx 1$, keeping the softmax out of its saturated regime where gradients vanish.

BERT is **pretrained** with **masked language modeling (MLM)**: randomly mask ~15% of tokens and predict them from both-sided context (plus, in original BERT, next-sentence prediction — which DistilBERT drops).

**DistilBERT** is BERT made small via **knowledge distillation**: a 6-layer *student* is trained to imitate the 12-layer *teacher*. The training loss is a triple objective

$$
\mathcal{L} = \alpha\,\mathcal{L}_{\text{CE}} + \beta\,\mathcal{L}_{\text{MLM}} + \gamma\,\mathcal{L}_{\cos}
$$

combining a **distillation cross-entropy** against the teacher's soft labels, the ordinary supervised MLM loss, and a **cosine-embedding** loss aligning student/teacher hidden states. The distillation term uses **temperature-softened** softmax on logits $z$:

$$
\mathcal{L}_{\text{CE}} = -\sum_i p_i^{(T)} \log q_i^{(T)},
\qquad
p_i^{(T)} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

**Why temperature **$T > 1$**.** A hard one-hot label says only "class 7 is correct". The teacher's *softened* distribution also says how the teacher ranks the wrong classes — the **dark knowledge** ("this looks 10× more like a cat than a car") — which is a far richer signal per example. Raising $T$ flattens the distribution so those relative magnitudes survive and drive the student's gradients. The payoff: DistilBERT is **~40% smaller (66M vs 110M params)** and **~60% faster**, while retaining **~97% of BERT's GLUE score** — the reason it's the default when you need embeddings cheaply.

**You've got this piece when you can** explain what makes an embedding "contextual", write scaled dot-product attention and justify the $\sqrt{d_k}$ term, state what MLM pretraining does, and describe distillation: the temperature-softened soft targets, why "dark knowledge" helps, and DistilBERT's size/speed/quality trade.

## Piece 4 — Hugging Face in PyTorch: tokenize, embed, fine-tune (~30 min)

*Source: PT-Cert C2 M3 (Text with Hugging Face), hands-on API. This is the piece you'll actually reuse.*

The Hugging Face `transformers` workflow is three objects: a **tokenizer** (text → subword `input_ids` + `attention_mask`), a **model** (ids → hidden states), and optionally a **task head**. BERT-family tokenizers use **WordPiece** — a subword scheme, so like FastText they degrade gracefully on rare words (splitting them into known pieces).

```python
import torch
from transformers import AutoTokenizer, AutoModel

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased").eval()

batch = tok(["a river bank", "a savings bank"],
            padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    out = model(**batch)
print(out.last_hidden_state.shape)     # (2, seq_len, 768) -> one vector PER TOKEN
```

`last_hidden_state` is `(B, T, H)` — a contextual vector for every token. To get **one vector per sentence** you pool. Two standard choices: take the `[CLS]` token (position 0), or **mean-pool** the token vectors — but mean-pooling must respect the `attention_mask` so padding tokens don't dilute the average:

```python
def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()          # (B, T, 1)
    summed = (last_hidden * mask).sum(dim=1)             # ignore padded positions
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts                               # (B, H) sentence embeddings
```

For a **downstream task** (e.g. sentiment on CIFAR-adjacent text tasks), swap `AutoModel` for a task class that adds a trainable head, and fine-tune with a *small* learning rate — the same freeze-backbone / small-LR discipline as yesterday's ResNet fine-tuning:

```python
from transformers import AutoModelForSequenceClassification
clf = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2)
# optional: freeze the encoder, train only the classifier head first
for p in clf.distilbert.parameters():
    p.requires_grad = False
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, clf.parameters()), lr=2e-5)
```

**You've got this piece when you can** describe the tokenizer→model→head split, say what shape `last_hidden_state` has and why it's per-token, write a mask-aware mean-pool to get a sentence embedding, and set up a DistilBERT fine-tune by adding a head and using a small LR.

## Piece 5 — Efficient training pipelines: profile first, then AMP & the allocator (PT-Cert C2 M4, ~35 min)

*Source: PT-Cert C2 M4 (Efficient Training Pipelines). Deep dive: *[*PyTorch × GPU*](https://app.notion.com/p/391e445b30a48184a35fca32915f2b2c)* §5 (dataloading), §8 (profiling), §3 (AMP), §4 (allocator).*

**Measure before optimizing.** End-to-end throughput is a pipeline — *fetch batch → H2D copy → forward → backward → step* — and its speed is capped by the **slowest stage**: $\text{throughput} \propto 1/\max_s(\text{stage time}_s)$. Guessing wastes effort; `torch.profiler` tells you which stage dominates and whether the GPU is *starved* (idle, waiting on data) or *saturated* (compute-bound).

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    for step, (xb, yb) in zip(range(10), loader):
        out = model(xb.cuda()); loss = loss_fn(out, yb.cuda())
        loss.backward(); opt.step(); opt.zero_grad()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

**If dataloading-bound** (GPU util low, big gaps between kernels — deep dive §5): raise `num_workers`, set `pin_memory=True` (enables faster non-blocking host→device copies), and use `prefetch_factor` so batches are ready before the GPU asks. The symptom to recognize: GPU utilization sagging toward 0% between steps.

**If compute-bound, reach for Automatic Mixed Precision (AMP)** — run matmuls/convs in fp16 or bf16 on Tensor Cores for ~2–3× speed and roughly half the activation memory, keeping a fp32 master copy for the update (deep dive §3). fp16 has a tiny dynamic range, so small gradients **underflow to zero**; a **GradScaler** prevents this by multiplying the loss by a large factor $S$ before backward and dividing it back out before the optimizer step:

$$
g = \frac{1}{S}\,\nabla_\theta\big(S \cdot L\big)
$$

(bf16 has fp32's exponent range, so it usually needs no scaler.)

```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
for xb, yb in loader:
    opt.zero_grad()
    with autocast():                      # fp16/bf16 forward
        loss = loss_fn(model(xb.cuda()), yb.cuda())
    scaler.scale(loss).backward()         # scale by S to avoid underflow
    scaler.step(opt); scaler.update()     # unscale + adapt S
```

**The CUDA caching allocator** (deep dive §4) is why memory behaves the way it does: PyTorch keeps freed GPU blocks in a cache instead of calling the slow, synchronizing `cudaFree`/`cudaMalloc`, so reused shapes are instant. The cost is **fragmentation** — you can hit out-of-memory even with free memory total, because no single *contiguous* cached block is large enough. Tuning knob to remember: the `PYTORCH_CUDA_ALLOC_CONF` environment variable (e.g. `expandable_segments:True`).

**You've got this piece when you can** explain why throughput is set by the slowest pipeline stage, use `torch.profiler` to tell dataloading-bound from compute-bound and name the fixes for each, write the AMP loop and say exactly what the GradScaler's factor $S$ prevents, and explain what the caching allocator trades away (fragmentation) and the env var that tunes it.

# 📝 Review quiz

Answer all 8 without notes. Grade with the key at the bottom, then fill in **Quiz score** and check **Quiz taken**.

1. **(Conceptual)** State the distributional hypothesis. Write GloVe's objective and its regression target, and explain in one sentence why word-vector *differences* (not the vectors themselves) capture analogies like $\text{king}-\text{man}+\text{woman}\approx\text{queen}$.
2. **(Derivation)** Write the skip-gram conditional $p(o\mid c)$. Why is it expensive to compute exactly for a large vocabulary, and what are the two standard approximations?
3. **(Conceptual)** Write FastText's word vector $v_w$ in terms of character n-grams. Give two concrete capabilities this buys over word2vec, and name the one limitation FastText still shares with word2vec and GloVe.
4. **(Conceptual)** "bank" appears in "river bank" and "savings bank". Explain why *no* static embedding can represent both senses correctly, and how a BERT contextual embedding does.
5. **(Derivation)** Write scaled dot-product attention. Derive/justify why the scaling factor is $\sqrt{d_k}$ specifically, and say what goes wrong if you drop it.
6. **(Conceptual / calc)** Write the temperature-softened softmax used in distillation and DistilBERT's three-term loss. Explain "dark knowledge" and what raising $T$ does to the teacher's signal. Roughly, DistilBERT's size, speed, and quality vs BERT-base?
7. **(Code reading)** Given the DistilBERT snippet: (a) what is the shape and meaning of `out.last_hidden_state`? (b) Give two ways to turn it into one sentence vector. (c) Why must mean-pooling use the `attention_mask`?
8. **(Code / conceptual)** In an AMP loop, why is a `GradScaler` needed and what does its factor $S$ prevent — and why does bf16 usually not need one? Separately: if `torch.profiler` shows GPU utilization dropping toward 0% between steps, is training compute-bound or dataloading-bound, and name two fixes.

> [!note]+ 🔑 Answer key — open only after attempting all 8
> **1.** *Distributional hypothesis:* a word's meaning is characterized by the contexts (neighboring words) it appears in; words with similar contexts get similar vectors. GloVe minimizes $J=\sum_{i,j} f(X_{ij})(w_i^\top\tilde w_j + b_i + \tilde b_j - \log X_{ij})^2$, i.e. it fits $w_i^\top\tilde w_j \approx \log X_{ij}$. Because dot products fit *log-counts*, differences of vectors fit *log-ratios* of co-occurrence, and it's those co-occurrence ratios (e.g. how "king" vs "queen" relate to "man" vs "woman") that encode the relation — so the analogy shows up as a consistent vector offset.
> 
> **2.** $p(o\mid c)=\dfrac{\exp(u_o^\top v_c)}{\sum_{w=1}^{V}\exp(u_w^\top v_c)}$. Expensive because the denominator sums over the entire vocabulary $V$ (tens/hundreds of thousands) for every training pair. Standard fixes: **negative sampling** (turn it into binary classification against a few sampled "negative" words) and **hierarchical softmax** (factor the softmax over a binary tree, $O(\log V)$).
> 
> **3.** $v_w=\sum_{g\in\mathcal{G}_w} z_g$, the sum of the word's character-n-gram embeddings (n≈3–6, with boundary markers). Two wins: (i) **OOV robustness** — unseen words still get a vector by composing known n-grams; (ii) **morphology** — related forms (run/runs/running) share n-grams so land near each other. Shared limitation: it's still **static** — one fixed vector per word, so it can't disambiguate polysemy by context.
> 
> **4.** A static embedding stores exactly one vector per word, computed once and frozen; "bank" therefore gets a single vector that averages the river and money senses regardless of the sentence, so at least one sense is always misrepresented. BERT computes each token's vector through self-attention over the *whole sentence*, so "bank" attends to "river" vs "savings" and produces a *different* contextual vector in each — the representation is a function of the context, not a lookup.
> 
> **5.** $\text{Attention}(Q,K,V)=\text{softmax}\!\big(\tfrac{QK^\top}{\sqrt{d_k}}\big)V$. If $q,k$ have i.i.d. entries with mean 0 and variance 1, then $q^\top k=\sum_{l=1}^{d_k} q_l k_l$ has variance $\approx d_k$, so logits scale like $\sqrt{d_k}$; dividing by $\sqrt{d_k}$ normalizes their variance back to $\approx 1$. Drop it and for large $d_k$ the logits get huge, pushing softmax into a near–one-hot saturated region where its gradient is nearly zero → training stalls.
> 
> **6.** Softened softmax: $p_i^{(T)}=\dfrac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$. Loss: $\mathcal{L}=\alpha\mathcal{L}_{\text{CE}}+\beta\mathcal{L}_{\text{MLM}}+\gamma\mathcal{L}_{\cos}$ (distillation CE on soft teacher labels + supervised MLM + cosine hidden-state alignment). **Dark knowledge** = the relative probabilities the teacher assigns to the *wrong* classes, which carry similarity structure a one-hot label omits; raising $T>1$ flattens the distribution so those relative magnitudes are large enough to shape the student's gradients. DistilBERT ≈ **40% smaller (66M vs 110M), ~60% faster, ~97% of BERT's GLUE performance**.
> 
> **7.** (a) `out.last_hidden_state` is $(B, T, H)=(\text{batch}, \text{seq len}, 768)$ — one contextual vector per token. (b) Take the `[CLS]` token (position 0), **or** mean-pool the token vectors. (c) Padding tokens are present to make sequences equal length but carry no meaning; without masking they'd be summed/averaged into the sentence vector and dilute it, so you multiply by the `attention_mask` and divide by the count of real tokens.
> 
> **8.** fp16 has a very small dynamic range, so small gradients **underflow to 0**; the `GradScaler` multiplies the loss by a large $S$ before `.backward()` (scaling gradients up out of the underflow zone) and divides it back out before the optimizer step, per $g=\tfrac1S\nabla_\theta(S\cdot L)$. bf16 keeps fp32's 8-bit exponent (same range, less mantissa), so gradients don't underflow and no scaler is needed. GPU utilization sagging to ~0% between steps ⇒ **dataloading-bound** (GPU starved, waiting on the input pipeline). Fixes: raise `num_workers`, set `pin_memory=True` + non-blocking copies, increase `prefetch_factor`, or cache/preprocess data (any two).
> 
> **Scoring:** 1 pt each (half credit for the right idea with a wrong detail). ≥ 6/8 → move on; below → tomorrow opens with a targeted patch-up on the missed pieces.