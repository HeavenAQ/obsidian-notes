---
base: "[[DL Daily Lessons — Step-by-Step + Quiz.base]]"
Quiz score: ""
"HW #": 1
Studied: false
Quiz taken: false
Day type: Lesson day
Date: 2026-07-08
Piece count: 5
---
> ⚠️ **Schedule note:** today follows the plan — **Transformers in Practice, Module 1**: the autoregressive loop, token sampling/decoding, structured outputs (FSM-constrained decoding), RAG, and chain-of-thought. Deep-dive companion: [Transformers in Practice](https://app.notion.com/p/391e445b30a4817dbfb5ef315a1f06ff) **§Module 1**. TiP is the **middle-priority** strand this block (HW1 math > Transformers-in-Practice > PT-Cert), so it gets full attention today. This is the conceptual base for tomorrow's Module 2 (attention internals) and the Jul 10 Karpathy build-a-GPT day.

> 

> 🗒️ **Quiz backlog:** Jul 2, 3, 5, 6, and 7 quizzes are all still untaken. Grade **at least yesterday's (Jul 7)** before starting today (~10 min) — the patch-up loop stays dark until you record a score.

# 🎯 Today's goal

See the entire behavior of a large language model as **one loop**: given a context, produce a distribution over the vocabulary, pick a single token, append it, repeat. Everything else today — temperature and top-p sampling, JSON-constrained decoding, retrieval-augmented generation, and chain-of-thought — is a *modification of the inputs or the logits of that same loop*, not a new mechanism. None of this is graded on HW1 directly, but it is the mental model that makes tomorrow's attention math (Module 2) and the Jul 10 nanoGPT build feel obvious instead of magical. By the end you should be able to write the generation loop from scratch and explain precisely where each technique intervenes.

# 🧩 Pieces

## Piece 1 — The autoregressive loop: the whole game (TiP M1, ~30 min)

*Source: TiP Module 1, "Observed behavior: the autoregressive loop." Deep dive: *[*Transformers in Practice*](https://app.notion.com/p/391e445b30a4817dbfb5ef315a1f06ff)* §Module 1.*

Everything an LLM does is one loop: **feed the context, get a distribution over the vocabulary, sample ONE token, append it, repeat.** The model is a conditional next-token distribution

$$
p_\theta(x_{t+1} \mid x_1, \dots, x_t) = \mathrm{softmax}\!\left(\frac{W_U\,h_t}{T}\right)
$$

where $h_t \in \mathbb{R}^{d}$ is the final-layer hidden state sitting **at the last position** $t$, $W_U \in \mathbb{R}^{|V|\times d}$ is the **unembedding** (a.k.a. output / LM-head) matrix that maps that hidden state to one **logit per vocabulary entry**, and $T$ is the sampling temperature (Piece 2). The raw logits are $z = W_U h_t$; softmax turns them into a probability vector over the whole vocabulary $V$.

**Why it's formulated this way.** A decoder-only Transformer is trained on exactly one objective — predict the next token from the tokens to its left — using a **causal mask** so position $t$ can never attend to $t+1$. That single constraint is what makes the model usable as a generator: at inference you only ever need the distribution for the *next* position, and you build long text by feeding your own samples back in. The consequence the module stresses: **the model never plans globally; it only ever continues locally-plausible text.** Chat, code, and "reasoning" are all this same local continuation — which is why the later pieces work by *changing what sits in the context* rather than changing the model.

**The unembedding matrix** $W_U$ deserves a note: naively a model has two big matrices — an input embedding $E \in \mathbb{R}^{|V|\times d}$ (token id → vector) and the output unembedding $W_U \in \mathbb{R}^{|V|\times d}$ (hidden state → logits). Many models **tie** these ($W_U = E$) to save $|V|\cdot d$ parameters and because the two tasks are near-inverses.

```python
import torch, torch.nn.functional as F

@torch.no_grad()
def generate(model, input_ids, max_new_tokens=50, temperature=1.0):
    for _ in range(max_new_tokens):
        logits = model(input_ids).logits[:, -1, :]   # z = W_U h_t at the LAST position
        probs  = F.softmax(logits / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)   # append, then repeat
    return input_ids
```

**You've got this piece when you can** write $p_\theta(x_{t+1}\mid x_{1:t}) = \mathrm{softmax}(W_U h_t / T)$ and name each symbol, explain what the causal mask guarantees and why it makes autoregressive generation possible, state the "never plans, only continues locally" claim, and write the 5-line generate loop from memory.

## Piece 2 — Decoding & sampling: greedy, temperature, top-k, top-p, min-p (TiP M1, ~35 min)

*Source: TiP Module 1, sampling/decoding. Deep dive: *[*Transformers in Practice*](https://app.notion.com/p/391e445b30a4817dbfb5ef315a1f06ff)* §Module 1.*

Given the probability vector from Piece 1, a **decoding strategy** decides which token to actually emit. The knobs trade **coherence** against **diversity**.

- **Greedy:** $x_{t+1} = \arg\max_i p_i$. Deterministic and often degenerates into **repetition loops** (the same phrase over and over).
- **Temperature:** divide logits by $T$ before softmax. Since $p_i \propto e^{z_i/T}$, temperature rescales the *differences* between log-probabilities, not the probabilities directly. $T \to 0$ sharpens toward greedy; $T > 1$ flattens toward uniform.

$$
p_i(T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

- **Top-k:** keep only the $k$ highest-probability tokens, renormalize, sample. Simple, but $k$ is **fixed** regardless of how peaked or flat the distribution is.
- **Top-p (nucleus):** keep the *smallest* set $S$ whose cumulative mass reaches $p$,

$$
S = \text{smallest set with } \sum_{i \in S} p_i \ge p ,
$$

then renormalize over $S$ and sample. This **adapts the cutoff to the distribution's entropy**: when the model is confident (one token dominates) the nucleus shrinks to almost that one token; when it's uncertain (many plausible continuations) the nucleus widens automatically — exactly what a fixed top-k cannot do.

- **Min-p:** keep every token with probability at least $p_{\min}\cdot p_{\max}$ (a fraction of the *top* token's probability, rather than an absolute cumulative-mass threshold). This scales the cutoff with model confidence even more directly and tends to behave better than top-p at high temperature.

**Why these exist.** Pure sampling from $p_\theta$ occasionally draws a very-low-probability token that derails the whole continuation (the long tail is huge — $|V|$ is 30k–100k+). Truncation methods (top-k / top-p / min-p) cut that tail off *before* sampling; temperature separately controls how much of the remaining mass concentrates on the favorites.

```python
def top_p_filter(logits, p=0.9):
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    # drop tokens once cumulative mass has already exceeded p (keep the nucleus)
    remove = cum - sorted_probs > p
    sorted_probs[remove] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    choice = torch.multinomial(sorted_probs, 1)
    return sorted_idx.gather(-1, choice)   # map back to vocab id
```

**You've got this piece when you can** define greedy/temperature/top-k/top-p/min-p, write the temperature-softmax and the top-p nucleus condition, explain in one sentence why top-p is *entropy-adaptive* and top-k is not, and say what problem truncation solves that temperature alone does not.

## Piece 3 — Structured outputs: constrained decoding with a finite-state machine (TiP M1, ~25 min)

*Source: TiP Module 1, structured outputs. Deep dive: *[*Transformers in Practice*](https://app.notion.com/p/391e445b30a4817dbfb5ef315a1f06ff)* §Module 1.*

To force output that is *guaranteed* valid JSON (or any grammar), you don't fine-tune and hope — you **constrain the sampling step**. Build a **finite-state machine (FSM)** over the target grammar; at each step the FSM knows which tokens could legally extend the string so far. **Mask illegal tokens before sampling** by setting their logit to $-\infty$:

$$
z_i \leftarrow -\infty \quad \text{for every token } i \text{ that cannot extend a valid string from the current FSM state.}
$$

After masking, softmax assigns those tokens probability exactly $0$, so whatever you sample is **on-grammar by construction**. This is how libraries like `outlines` and `guidance`, and OpenAI's structured-output / JSON mode, work.

**Why it's done at the logits, not after.** Rejection-sampling whole outputs and re-trying is exponentially wasteful; masking at every step keeps generation valid *incrementally* with zero retries. **The hard part** the module flags: the FSM must be **compiled against the tokenizer's vocabulary**, because a single token can straddle grammar-symbol boundaries (e.g. one BPE token might be `": "` or `true,`), so "which tokens are legal here" is a non-trivial precomputation over the vocab, not over characters.

```python
def masked_sample(logits, allowed_ids):
    # allowed_ids: LongTensor of vocab ids the FSM permits in the current state
    mask = torch.full_like(logits, float('-inf'))
    mask[..., allowed_ids] = 0.0
    probs = torch.softmax(logits + mask, dim=-1)  # illegal tokens -> prob 0
    return torch.multinomial(probs, 1)
```

**You've got this piece when you can** explain constrained decoding as a per-step logit mask ($z_i=-\infty$ for illegal tokens), say why it guarantees valid output with no retries, name a library that does it, and state why compiling the FSM against the *tokenizer vocabulary* (not characters) is the tricky part.

## Piece 4 — RAG: grounding by injecting tokens into the context (TiP M1, ~25 min)

*Source: TiP Module 1, "Grounding (RAG) and reasoning (CoT) are the same trick." Deep dive: *[*Transformers in Practice*](https://app.notion.com/p/391e445b30a4817dbfb5ef315a1f06ff)* §Module 1.*

**Retrieval-Augmented Generation** changes *what the model conditions on* rather than the model itself. You retrieve relevant passages and place them in the context **before** the user's query, so the loop now samples from

$$
p_\theta(\text{answer}\mid \text{retrieved passages},\ \text{query})\quad\text{instead of}\quad p_\theta(\text{answer}\mid \text{query}).
$$

**Why it reduces hallucination.** Recall Module 1's core claim from Piece 1: the model never plans, it just continues **locally-plausible** text. Without retrieved evidence, "locally plausible" is anchored only by the model's training-time-compressed world knowledge — which is lossy and can **confabulate smoothly** (fluent but wrong). With the right passages sitting in context, "locally plausible" becomes "consistent with the evidence in front of it," so the highest-probability continuations are the grounded ones. The retrieval quality is therefore doing the real work — garbage passages give confidently-wrong answers just as smoothly.

```python
# The whole mechanism is a prompt-assembly step; the model/loop is unchanged.
def rag_prompt(query, retriever, k=4):
    passages = retriever.search(query, k=k)          # e.g. vector / BM25 search
    context = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    return (f"Use ONLY the sources below to answer.\n\n"
            f"Sources:\n{context}\n\nQuestion: {query}\nAnswer:")
```

**You've got this piece when you can** write the shift from $p_\theta(\text{answer}\mid\text{query})$ to $p_\theta(\text{answer}\mid\text{passages},\text{query})$, connect "reduces hallucination" back to the "only continues locally-plausible text" claim, and explain why retrieval *quality* determines whether RAG helps or hurts.

## Piece 5 — Chain-of-Thought: reasoning by injecting tokens into the context (TiP M1, ~25 min)

*Source: TiP Module 1, CoT. Deep dive: *[*Transformers in Practice*](https://app.notion.com/p/391e445b30a4817dbfb5ef315a1f06ff)* §Module 1.*

**Chain-of-Thought** is the *same trick as RAG*, aimed inward: instead of retrieving external tokens, you make the model **emit its own intermediate tokens** ("Step 1: … Step 2: …") before the final answer. Because generation is autoregressive (Piece 1), the final-answer token is now conditioned on all those intermediate reasoning tokens sitting in the context. This does two concrete things:

1. **Extra forward-pass compute.** Each emitted intermediate token is another full pass through the network, so the model literally spends *more computation* before committing to the answer — depth-in-time it otherwise wouldn't have for a one-shot answer.
2. **Extra working memory.** The intermediate tokens act as an external scratchpad: partial results (a subtotal, a parsed constraint) are written into the context and can be attended to later, instead of having to be held implicitly inside a single hidden state.

There is no architectural change — you elicit it purely with the prompt ("Let's think step by step") or with few-shot exemplars that show worked reasoning. Formally it is again a conditional-distribution shift: $p_\theta(\text{answer}\mid \text{query})$ vs. $p_\theta(\text{answer}\mid \text{query},\ r_1,\dots,r_m)$ where $r_{1:m}$ are the model's own reasoning tokens.

```python
cot_prompt = (
    "Q: A shop had 23 apples, sold 8, then received 12 more. How many now?\n"
    "A: Let's think step by step.\n"     # <- elicits intermediate tokens r_1..r_m
)
# The model emits: '23 - 8 = 15; 15 + 12 = 27. Answer: 27' — each step is in-context
# for the next step, giving both more compute and a written scratchpad.
```

**You've got this piece when you can** state that RAG and CoT are the *same* mechanism (inject tokens to shift the conditional distribution), name CoT's two concrete benefits (more forward-pass compute + external working memory), and explain why it needs no change to the model — only to the context.

# 📝 Review quiz

Answer all 8 without notes. Grade with the key at the bottom, then fill in **Quiz score** and check **Quiz taken**.

3. **(Conceptual)** Write the LLM's next-token distribution in terms of $W_U$, $h_t$, and $T$, and name each symbol. What are the dimensions of $W_U$ and why does its number of rows equal $|V|$?
4. **(Conceptual)** What does the **causal mask** guarantee, and why is that guarantee exactly what makes autoregressive generation possible? Connect it to the claim "the model never plans, it only continues locally-plausible text."
5. **(Derivation / reasoning)** Write the temperature-softmax. Explain why raising $T$ flattens the distribution and $T\to 0$ approaches greedy, referencing the fact that temperature rescales log-probability *differences*.
6. **(Conceptual)** Give the top-p (nucleus) selection rule as a formula. Explain in one or two sentences why top-p is *entropy-adaptive* while top-k is not, and describe a situation where that difference matters. How does **min-p** differ from top-p?
7. **(Code reading)** In the `top_p_filter` snippet, what is the role of the line `remove = cum - sorted_probs > p`, and why is it `cum - sorted_probs` rather than just `cum`? What would break if you used `cum > p` instead?
8. **(Conceptual / code)** Explain constrained decoding with an FSM: what operation is applied to the logits, and what value is used? Why does this guarantee on-grammar output with zero retries? What is the hard part about compiling the FSM, and why?
9. **(Conceptual)** RAG and CoT are described as "the same trick." State the trick in one sentence, then write the conditional-distribution shift for **each**. For RAG, explain why the mechanism reduces hallucination by referring back to the "locally-plausible continuation" idea.
10. **(Conceptual)** Chain-of-thought gives a model two concrete advantages over answering in one shot. Name both and explain each in one sentence. Why does CoT require **no** change to the model's weights or architecture?

> [!note]+ 🔑 Answer key — open only after attempting all 8
> **1.** $p_\theta(x_{t+1}\mid x_{1:t}) = \mathrm{softmax}(W_U h_t / T)$. $h_t\in\mathbb{R}^d$ is the final-layer hidden state at the last position $t$; $W_U\in\mathbb{R}^{|V|\times d}$ is the unembedding / LM-head matrix; $T$ is temperature. $W_U$ has one **row per vocabulary token**, so $z=W_U h_t$ produces exactly one **logit per token** ($|V|$ of them) — that's why rows $=|V|$. (Often tied to the input embedding $E$.)
> 
> **2.** The causal mask guarantees position $t$ can only attend to positions $\le t$ (never the future). That means the distribution for the next token depends only on tokens already generated, so you can generate left-to-right by feeding your own samples back in — exactly the autoregressive loop. Because each step only optimizes the *immediate* next token from left context, the model produces a locally-plausible continuation at every step; it has no objective that looks ahead and "plans" the whole output.
> 
> **3.** $p_i(T)=\dfrac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}$. Since only the ratios $e^{(z_i-z_j)/T}$ matter, dividing logits by a large $T$ shrinks all pairwise differences → probabilities move toward uniform (flatter); a small $T$ magnifies the differences → mass concentrates on the top logit, and in the limit $T\to 0$ the max logit gets all the mass, i.e. $\arg\max$ = greedy.
> 
> **4.** Top-p: choose the smallest set $S$ with $\sum_{i\in S}p_i\ge p$, renormalize over $S$, sample. It is entropy-adaptive because the *size* of $S$ depends on the distribution's shape: a confident (peaked) distribution yields a tiny nucleus (nearly greedy), a flat/uncertain one yields a wide nucleus (many options). Top-k always keeps exactly $k$ tokens regardless of shape, so on a very peaked distribution it can still admit junk tokens, and on a very flat one it can cut off good ones. **Min-p** keeps tokens with $p_i\ge p_{\min}\cdot p_{\max}$ — a threshold relative to the *top* token's probability, rather than top-p's absolute cumulative-mass threshold.
> 
> **5.** `remove` marks the tokens to zero out. Using the **exclusive** cumulative sum (`cum - sorted_probs`, i.e. the mass *before* this token) means a token is removed only once the mass strictly before it already exceeds $p$ — this **keeps** the token that first crosses the threshold, so the nucleus's total mass is $\ge p$ as required. Using `cum > p` (the inclusive sum) would remove the very token that pushes the total past $p$, giving a nucleus with mass $< p$ and, in the extreme (a single token with $p_1>p$), removing the top token entirely — leaving nothing/normalizing incorrectly.
> 
> **6.** At each step you set $z_i\leftarrow-\infty$ for every token that the grammar's FSM says cannot extend the current partial string; after softmax those tokens have probability exactly 0, so any sampled token is legal — validity holds **incrementally**, so there are never rejected/re-tried outputs. The hard part is that the FSM's "which tokens are legal" set must be **compiled against the tokenizer vocabulary**, not characters, because a single token can cross grammar-symbol boundaries (e.g. a BPE token spanning `": "` or ending a value plus a comma), so legality is a nontrivial per-state precomputation over all $|V|$ tokens.
> 
> **7.** The trick: **inject tokens into the context so the conditional distribution the loop samples from changes** (no model change). RAG: $p_\theta(\text{answer}\mid\text{passages},\text{query})$ vs $p_\theta(\text{answer}\mid\text{query})$. CoT: $p_\theta(\text{answer}\mid\text{query},r_{1:m})$ vs $p_\theta(\text{answer}\mid\text{query})$. RAG reduces hallucination because the model only ever continues *locally-plausible* text; with no evidence, "plausible" is anchored only by lossy compressed knowledge and can confabulate fluently, but with retrieved passages in context the plausible continuations are the ones consistent with that evidence.
> 
> **8.** (i) **More forward-pass compute** — each intermediate token is another full pass through the network, so the model spends more computation before answering. (ii) **External working memory** — the intermediate tokens are a written scratchpad the model can attend to later, instead of cramming partial results into one hidden state. No weight/architecture change is needed because it is purely a change to the *context* (prompt or few-shot exemplars); the autoregressive loop and the network are identical.
> 
> **Scoring:** 1 pt each (half credit for the right idea with a wrong detail). ≥ 6/8 → move on; below → tomorrow opens with a targeted patch-up on the missed pieces.