---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-10T20:49:00
Status: Done
Last updated time: 2026-07-10T20:49:00
Last edited by: Heaven Chen
Category:
  - LLM
  - Finetuning
  - ML
---
> **Deep dive into GRPO** (**G**roup **R**elative **P**olicy **O**ptimization) — the RL fine-tuning algorithm from [*DeepSeekMath*](https://arxiv.org/abs/2402.03300)[ (Shao et al., 2024, arXiv 2402.03300)](https://arxiv.org/abs/2402.03300), the method behind **DeepSeek-R1**'s reasoning training. This note covers the math, the intuition, the benefits vs PPO, and a **complete TRL implementation** you can run to fine-tune a model. *(Note: “GPRO” is a common typo for GRPO — that's what this doc is about.)*

# 0. TL;DR

GRPO is a critic-free variant of PPO. Instead of training a separate **value network** to estimate a baseline, GRPO samples a **group of G completions** for each prompt, scores them with a reward function, and uses the **group's mean reward as the baseline**. The advantage of each completion is just “how much better than its group-mates” it did. This **removes ~half the training memory** (no critic), works beautifully with **verifiable rewards** (math/code correctness), and is the standard recipe for training reasoning models today.

# 1. The setup: RL fine-tuning of an LLM

In RL post-training we treat the LLM as a **policy** $\pi_\theta$ that, given a prompt $q$, generates a completion $o = (o_1, o_2, \dots)$ token by token. A **reward** $r$ scores the completion (e.g. “is the final answer correct?”). The goal is to update $\theta$ so the policy produces higher-reward completions — **without drifting so far** that it breaks or reward-hacks.

The workhorse for this has been **PPO**, which maximizes a *clipped* objective using an **advantage** $\hat{A}$ — how much better an action was than expected. PPO estimates “expected” with a learned **value function (critic)** $V_\phi$, usually a second network the same size as the policy, plus **GAE** to turn per-token values into advantages. That critic is the expensive part: extra memory, extra compute, extra thing to tune.

# 2. The GRPO idea: replace the critic with a group baseline

GRPO's insight: reward models (and verifiable checkers) are naturally **comparative** — they tell you which of several answers is better. So instead of *learning* a baseline, **compute one on the fly** from a group of samples.

For each prompt $q$, sample $G$ completions $\{o_1, \dots, o_G\}$ from the current policy, score them to get rewards $\mathbf{r} = \{r_1, \dots, r_G\}$, and define each completion's advantage **relative to its group**:

$$
\hat{A}_{i,t} = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}
$$

Every token in completion $o_i$ gets the same advantage $\hat{A}_{i,t} = \hat{A}_i$ (for outcome rewards). That's the whole trick — and it's what gives the method its name.

**Why this works (intuition):** subtracting the group **mean** is a classic variance-reduction *baseline* (positive advantage ⇒ “better than average for this prompt, do more of it”; negative ⇒ “worse, do less”). Dividing by the group **std** normalizes across easy and hard prompts. No critic, no GAE — just “was this sample above or below the group average?”

![[99 Assets/Media/grpo_visual.png|GRPO — the four steps: generate a group, score, compute group-relative advantage, update]]

*The GRPO loop (via Hugging Face TRL): generate G completions → reward each → group-relative advantage → policy update. No value network anywhere.*

# 3. The math, step by step

TRL breaks GRPO into four stages. Here they are with the exact formulas.

## 3.1 Generate

Sample a batch of prompts; for each prompt generate $G$ completions $o_i$ from $\pi_{\theta_\text{old}}$ (the policy snapshot used for this rollout).

## 3.2 Advantage (group-relative)

Score each completion and normalize within the group:

$$
\hat{A}_{i,t} = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}
$$

## 3.3 KL divergence (keep close to a reference)

To stop the policy drifting, GRPO can penalize divergence from a frozen **reference** policy $\pi_\text{ref}$ (usually the SFT model), using the low-variance **k3 estimator** (Schulman):

$$
\mathbb{D}_{\text{KL}}\!\left[\pi_\theta \,\|\, \pi_\text{ref}\right] = \frac{\pi_\text{ref}(o_{i,t}\mid q, o_{i,<t})}{\pi_\theta(o_{i,t}\mid q, o_{i,<t})} - \log\frac{\pi_\text{ref}(o_{i,t}\mid q, o_{i,<t})}{\pi_\theta(o_{i,t}\mid q, o_{i,<t})} - 1
$$

This estimator is always ≥ 0 and much less noisy than the naive log-ratio.

## 3.4 The objective (clipped surrogate)

Let the per-token importance ratio be $r_{i,t}(\theta) = \dfrac{\pi_\theta(o_{i,t}\mid q, o_{i,<t})}{\pi_{\theta_\text{old}}(o_{i,t}\mid q, o_{i,<t})}$. GRPO maximizes the PPO-style clipped surrogate, averaged over the group, minus the KL penalty:

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big(\min\big(r_{i,t}\hat{A}_{i,t},\ \text{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon)\,\hat{A}_{i,t}\big) - \beta\,\mathbb{D}_{\text{KL}}[\pi_\theta\|\pi_\text{ref}]\Big)\right]
$$

The **clip** bounds the ratio to $[1-\epsilon, 1+\epsilon]$ so a single update can't move the policy too far; $\beta$ weights the KL leash. The training **loss is** $\mathcal{L} = -\mathcal{J}$. When you only do one gradient step per rollout ($\mu=1$, TRL default), $\pi_{\theta_\text{old}} = \pi_\theta$ and the ratio starts at 1, so the min/clip simplifies away.

## 3.5 Modern variants (what TRL actually defaults to)

The original per-sequence normalization $\frac{1}{|o_i|}$ turned out to bias training toward longer or shorter responses. Two fixes are widely used and available as `loss_type` in TRL:

| loss_type | Normalization | Why |
| --- | --- | --- |
| `grpo` (original) | per-sequence: $\frac{1}{G}\sum_i \frac{1}{|o_i|}\sum_t$ | DeepSeekMath original; has a response-length bias |
| `dapo` | token-level: $\frac{1}{\sum_i |o_i|}\sum_i\sum_t$ | [DAPO](https://arxiv.org/abs/2503.14476) — fixes under-penalized long CoT |
| `dr_grpo` | constant: $\frac{1}{LG}\sum_i\sum_t$ (L = max length) | [Dr. GRPO](https://arxiv.org/abs/2503.20783) — fully removes length bias |

Two more defaults worth knowing: recent practice often sets $\beta = 0$ (no KL / no reference model — the TRL default, per Open-Reasoner-Zero and DAPO), which saves memory and often trains fine; and setting `**scale_rewards=False**` drops the $\text{std}(\mathbf{r})$ division to avoid a **question-difficulty bias** (Dr. GRPO).

# 4. Why GRPO (benefits)

- **No critic → big memory & simplicity win.** PPO carries a value network (~policy-sized) plus its optimizer state; GRPO drops it entirely. With $\beta=0$ you don't even need the reference model, leaving essentially just the policy in memory.
- **Built for verifiable rewards.** Math answer-checking, unit tests, format regexes — cheap, robust, non-gameable signals map perfectly onto GRPO's per-completion rewards. This is exactly how R1-style reasoning is trained.
- **Stable baseline for free.** The group mean is an unbiased, low-variance baseline that adapts per-prompt — no separate network to under/over-fit.
- **Strong for reasoning.** Sampling many completions and rewarding the good ones directly incentivizes the model to find reasoning paths that *work*, which is where GRPO shines (DeepSeekMath hit 51.7% on MATH; R1 learned long CoT this way).
- **Composable rewards.** You can sum several reward functions (correctness + format + brevity + …), each a plain Python function.

![[99 Assets/Media/grpo_curves.png|GRPO training reward curve climbing over steps]]

*Reward rising during a TRL GRPO run on a math dataset (Qwen2-0.5B) — the policy learns to produce higher-reward answers over time.*

# 5. Gotchas & practical tips

- **Generation is the bottleneck.** Each step samples $G$ completions per prompt — use **vLLM** (`use_vllm=True`) or training crawls.
- **Group diversity matters.** If all $G$ completions get the *same* reward (all right or all wrong), the advantage is 0 and there's **no learning signal** for that prompt. TRL logs this as `frac_reward_zero_std`. Pick a model/dataset difficulty where the model sometimes succeeds.
- **Reward hacking is real.** The model optimizes exactly what you reward. Keep rewards verifiable and add guardrails (format checks, length penalties) so it can't shortcut.
- **RL is LR-sensitive.** Use a small learning rate (**1e-6 – 5e-6**); too high collapses the policy.
- **Length bias.** Prefer `loss_type="dapo"` or `"dr_grpo"` for long-CoT tasks.
- **Batch divisibility.** In TRL the effective generation batch must be divisible by `num_generations` ($G$).

# 6. Implementation with TRL

The cleanest path is Hugging Face **TRL**'s `GRPOTrainer`. Install: `pip install trl peft vllm datasets`.

## 6.1 Minimal quick start

A reward can be *any Python function* returning one float per completion. This toy example rewards completions near 200 characters:

```python
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

dataset = load_dataset("trl-lib/tldr", split="train")

def reward_len(completions, **kwargs):
    # closer to 200 chars -> higher (less negative) reward
    return [-abs(200 - len(c)) for c in completions]

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=reward_len,
    args=GRPOConfig(output_dir="grpo-quickstart", num_generations=8),
    train_dataset=dataset,
)
trainer.train()
```

## 6.2 A real example: teach a model to reason on GSM8K

This is the R1-style pattern: force a `<think>/<answer>` structure and reward **correctness + format**.

### Step 1 — data + prompt format

```python
import re
from datasets import load_dataset

SYSTEM_PROMPT = (
    "Respond in exactly this format:\n"
    "<think>\n...step-by-step reasoning...\n</think>\n"
    "<answer>\n...final numeric answer...\n</answer>"
)

def build_gsm8k(split="train"):
    data = load_dataset("openai/gsm8k", "main", split=split)
    def to_chat(ex):
        gold = ex["answer"].split("####")[-1].strip()  # GSM8K gold is after '####'
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
            ],
            "answer": gold,
        }
    return data.map(to_chat)
```

### Step 2 — reward functions

```python
import re

def _extract(text: str):
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else None

# (a) correctness: the big signal (+2 when the answer matches gold)
def correctness_reward(completions, answer, **kwargs):
    responses = [c[0]["content"] for c in completions]
    preds = [_extract(r) for r in responses]
    return [2.0 if p == a else 0.0 for p, a in zip(preds, answer)]

# (b) format: reward the <think>/<answer> structure (+0.5)
def format_reward(completions, **kwargs):
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>\s*$"
    responses = [c[0]["content"] for c in completions]
    return [0.5 if re.match(pattern, r, re.DOTALL) else 0.0 for r in responses]

# (c) soft nudge: partial credit if the answer is a bare integer (+0.5)
def is_int_reward(completions, **kwargs):
    responses = [c[0]["content"] for c in completions]
    preds = [_extract(r) for r in responses]
    return [0.5 if (p is not None and p.isdigit()) else 0.0 for p in preds]
```

The trainer passes each dataset column (here `answer`) as a keyword arg, and **sums** the rewards from all functions (or weights them via `reward_weights`).

### Step 3 — configure and train

```python
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig  # optional: LoRA to save memory

training_args = GRPOConfig(
    output_dir="grpo-qwen2.5-3b-gsm8k",
    learning_rate=1e-6,               # RL is LR-sensitive; keep it small
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_generations=8,                # G: completions sampled per prompt
    max_prompt_length=512,
    max_completion_length=1024,       # room for chain-of-thought
    num_train_epochs=1,
    beta=0.0,                         # KL weight; 0.0 = no reference model (TRL default)
    epsilon=0.2,                      # PPO clip range
    loss_type="dapo",                 # token-level norm -> less length bias
    scale_rewards=False,              # drop std division -> avoid difficulty bias
    use_vllm=True,                    # fast generation (the bottleneck)
    vllm_mode="colocate",
    bf16=True,
    logging_steps=10,
    save_steps=200,
    report_to="wandb",
)

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-3B-Instruct",
    reward_funcs=[correctness_reward, format_reward, is_int_reward],
    args=training_args,
    train_dataset=build_gsm8k("train"),
    peft_config=peft_config,          # drop this for full fine-tuning
)
trainer.train()
trainer.save_model("grpo-qwen2.5-3b-gsm8k/final")
```

Launch it (multi-GPU via Accelerate):

```bash
accelerate launch train_grpo.py
```

## 6.3 Key `GRPOConfig` knobs

| Parameter | Meaning | Typical |
| --- | --- | --- |
| `num_generations` (G) | completions sampled per prompt — the “group” | 8–16 (bigger = better baseline, more compute) |
| `learning_rate` | policy LR | 1e-6 – 5e-6 |
| `beta` | KL weight to reference policy | 0.0 (common) … 0.04 (DeepSeekMath) |
| `epsilon` | clip range for the ratio | 0.2 |
| `loss_type` | normalization variant | `dapo` / `dr_grpo` for long CoT |
| `scale_rewards` | divide advantage by group std | `False` to avoid difficulty bias |
| `max_completion_length` | generation cap | 1024–8192 for reasoning |
| `use_vllm` / `vllm_mode` | fast rollout generation | `True` / `colocate` or `server` |

## 6.4 Scaling up

- **vLLM server mode** for large models: run `trl vllm-serve --model <name>` on dedicated GPUs and set `vllm_mode="server"` — keeps generation off the training GPUs.
- **DeepSpeed ZeRO-3 + Accelerate** for 70B-class models (model states sharded across GPUs/nodes).
- **LoRA + 4-bit** to fit on a single GPU / consumer hardware.

# 7. When to use GRPO (and when not)

- **Use GRPO** when you have a **cheap, reliable reward** (math/code/verifiable tasks, or a solid reward model) and want reasoning/agentic improvements beyond SFT. It's the go-to for R1-style reasoning RL.
- **Prefer SFT / DPO** when you only have **preference pairs** or demonstrations and don't need online sampling — they're simpler and cheaper (no rollouts).
- **Consider PPO** if you specifically need a learned value function (dense per-token credit assignment) and can afford the critic.

---

**Sources**: [DeepSeekMath (GRPO) — arXiv 2402.03300](https://arxiv.org/abs/2402.03300) · [DeepSeek-R1 — arXiv 2501.12948](https://arxiv.org/abs/2501.12948) · [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/grpo_trainer) (figures via TRL) · [Schulman — KL approximation](http://joschu.net/blog/kl-approx.html) · variants: [DAPO 2503.14476](https://arxiv.org/abs/2503.14476), [Dr. GRPO / Understanding R1-Zero 2503.20783](https://arxiv.org/abs/2503.20783).