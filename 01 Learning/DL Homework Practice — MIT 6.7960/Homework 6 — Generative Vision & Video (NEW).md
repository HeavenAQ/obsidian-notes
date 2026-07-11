---
base: "[[DL Homework Practice — MIT 6.7960.base]]"
Status: Not-Started
Notes ready: false
Window: Aug 13 – Aug 19, 2026 (weekly block)
Due Date: 2026-08-19
"HW #": 6
Topics: Flow matching & rectified flow; ODE solvers (Euler/Heun/DPM); ControlNet & zero-convolutions; IP-Adapters; consistency models & LCM distillation; adversarial diffusion distillation (SDXL Turbo/Flux Schnell); DiT & 3D spacetime patches; 3D RoPE; temporal attention; optical flow & motion buckets; Plücker coordinates
---
# 🆕 New homework — material NOT covered by MIT 6.7960

MIT 6.7960 stops at basic diffusion (Lectures 14–16). This homework covers the modern generative-vision stack from the Udemy course **“Mastering Generative Vision & Video: From GAN to Flow to DiT”** (Modules 1.2 → 3). The coach will write the full problem set into this page on Day 1 of the window (Aug 13) — derivations + PyTorch implementation tasks, no solutions.

## Assignment scope (problems will be drawn from)

1. **Flow matching & rectified flow**: derive the velocity-prediction objective; show why straight-line probability paths permit few-step Euler sampling; compare with the DDPM ELBO.
2. **Continuous-time diffusion & ODE solvers**: probability-flow ODE; truncation error of Euler vs Heun; when higher-order solvers (DPM-Solver) pay off.
3. **ControlNet**: architecture with zero-convolutions — prove the zero-init preserves the pretrained model's function at step 0 and derive how gradients flow into the control branch; contrast with T2I-Adapters and decoupled cross-attention (IP-Adapters).
4. **Distillation for speed**: consistency property & LCM; adversarial diffusion distillation — why GAN losses restore sharpness at 1–4 steps.
5. **Video DiT**: 3D spacetime patchification (token count/memory math), 3D RoPE, spatial vs temporal attention cost analysis; motion buckets & optical-flow conditioning; Plücker-coordinate camera conditioning.
6. **Implementation**: mini flow-matching model on 2D toy data; a single DiT block with (factorized) space-time attention; Euler vs Heun sampler comparison.

# 📚 Prep lessons (daily plan Aug 13–17; attempt HW Aug 17–19)

- [ ] **Day 1** — Module 1.2 Flow Matching & Rectified Flow (SD3.5/Flux architectures) + Module 1.3 Continuous vs discrete time, ODE solvers
- [ ] **Day 2** — Module 2.1 Cross-attention conditioning & IP-Adapters + Module 2.2 ControlNet & T2I-Adapters
- [ ] **Day 3** — Module 2.3 Consistency Models & LCM + Module 2.4 Adversarial Diffusion Distillation; Module 2.5 (autoregressive VAR/StyleGAN3) skim only
- [ ] **Day 4** — Module 3.1 DiT, 3D spacetime patches, 3D RoPE (Sora/Veo/Gen-3) + Module 3.2 Temporal attention, optical flow, motion buckets
- [ ] **Day 5** — Module 3.3 Camera motion & Plücker coordinates; Module 4 (audio-visual sync) optional/skim — review + start HW6

*Note: Module 0 (VAE/GAN/ViT) is covered during HW4 week — don't repeat it. Skip code-setup/Colab-basics videos.*