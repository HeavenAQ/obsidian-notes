---
base: "[[01.3 DL Homework Practice — MIT 6.7960.base]]"
Status: Not-Started
Notes ready: false
Window: Sep 9 – Sep 15, 2026 (weekly block)
Due Date: 2026-09-15
"HW #": 7
Topics: Tokenization & BPE; token/position embeddings; GPT from scratch (attention, transformer blocks, layernorm, multihead); pretraining (AdamW, init, dropout); fine-tuning & PEFT/LoRA; instruction tuning & RLHF; sampling (temperature/top-k/top-p); evaluation (perplexity, benchmarks); mechanistic interpretability
---
# 🆕 New homework — material NOT covered by MIT 6.7960

6.7960 covers transformers and language models at lecture level (Lectures 8, 21) but not the full LLM mechanism stack. This homework covers the Udemy course **“A deep understanding of AI large language model mechanisms”** (Mike X Cohen). The coach will write the full problem set into this page on Day 1 of the window (Sep 9) — derivations + PyTorch implementation tasks, no solutions.

## Assignment scope (problems will be drawn from)

1. **Tokenization**: run the BPE algorithm by hand on a tiny corpus to a target vocab size; analyze compression ratios and why character counting ("how many r's in strawberry") fails.
2. **Embeddings**: token vs position embeddings; cosine similarity vs correlation; the unembedding (tied-weight) map from vectors back to tokens.
3. **GPT from scratch**: derive causal masking as a linear-algebra averaging operator; implement single-head attention manually, then a full transformer block (layernorm placement, residuals, GELU); parameter-count the resulting GPT-2-style model.
4. **Training**: AdamW update rule and why decoupled weight decay differs from L2; gradient clipping; LR schedulers; weight initialization effects.
5. **Adaptation**: full vs partial fine-tuning (freezing), PEFT/LoRA rank math; instruction tuning (system/user/assistant format); RLHF at a conceptual/derivation level.
6. **Inference & evaluation**: temperature in softmax; greedy vs top-k vs top-p; perplexity derivation; one mechanistic-interpretability probe (attention maps or logit lens) on a small pretrained model.

# 📚 Prep lessons (daily plan Sep 9–13; attempt HW Sep 13–15)

- [ ] **Day 1** — Part 1: Words → tokens → numbers: BPE algorithm, subword vs character vs word tokenization, tokenizers across languages/models *(skip pure text-parsing/Python-basics videos)*
- [ ] **Day 2** — Build a GPT (first half): embedding/unembedding, softmax & temperature, sampling methods, layernorm, position embeddings
- [ ] **Day 3** — Build a GPT (second half): attention implementation, transformer blocks, multihead attention, full GPT-2 on GPU, parameter counting *(theory was HW2 week — this is the implementation pass)*
- [ ] **Day 4** — Pretrain + fine-tune sections: anything not finished in HW5 week (AdamW, init, dropout, PEFT, clipping, schedulers) + Instruction tuning & RLHF
- [ ] **Day 5** — Evaluating LLMs (perplexity, benchmarks) + mechanistic-interpretability sections (PCA/clustering/cosine-similarity probes); **6.S191 Lab 3: Fine-Tune an LLM** as hands-on practice — review + start HW7

*Note: skip the course's intro/Udemy-platform/Colab-setup videos and Python-fundamentals content. Embeddings-space geometry was already covered in HW3 week — reference, don't rewatch.*