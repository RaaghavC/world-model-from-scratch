# wm2 — a 2026-frontier world model, from scratch (design spec)

This is the implementation contract for `wm2/`. It upgrades the 2018 Ha &
Schmidhuber **V-M-C** baseline (`wm/`) to the modern world-model frontier while
staying small enough to train from scratch on a laptop (Apple MPS). Every module
below cites the work it operationalizes, and the comparison to the baseline is
the whole point.

## 0. Where this sits (Fei-Fei Li's 2026 taxonomy)

A world model can be a **Renderer** (outputs pixels), a **Simulator** (outputs
state you can compute on), or a **Planner** (outputs actions) — "three
projections of a single underlying understanding" (World Labs, *A Functional
Taxonomy of World Models*, Jun 3 2026). The baseline is a renderer (VAE) wired to
a latent simulator (MDN-RNN) and a planner (CEM controller). `wm2` strengthens
all three and adds the two paradigms that define 2025–2026:

| Box | 2018 baseline (`wm/`) | 2026 upgrade (`wm2/`) | Lineage |
|---|---|---|---|
| V (Renderer) | ConvVAE, continuous Gaussian latent | CNN enc/dec over a **discrete categorical** latent | DreamerV2/V3 |
| M (Simulator) | MDN-RNN (LSTM + 5 Gaussians) | **RSSM**: deterministic GRU + categorical stochastic state, KL-balanced | PlaNet→DreamerV3 |
| — | (none) | **Latent Action Model**: label-free action discovery | Genie 1–3 |
| C (Planner) | linear policy, CEM on a position **probe** | **actor–critic in imagination** (λ-returns, two-hot value); reward **learned** | Dreamer / DreamerV3 |
| — | (none) | **JEPA track**: decoder-free joint-embedding predictor + latent **MPC** | V-JEPA 2 / TD-MPC2 |

Two debates the build deliberately engages on one toy world: **generative vs
non-generative** (Dreamer decoder vs JEPA no-decoder), and **renderer vs
simulator fidelity** (does the dream merely look right, or does it conserve
momentum/energy?).

## 1. Environment (`wm2/env.py`, done)

`MultiBody2DEnv`: N bodies, body 0 = thrustable agent, bodies 1.. = frictionless
elastic free movers; depth-ordered occlusion; goal-reaching reward
`r = -dist(agent, goal)`. 64×64×3 uint8 frames, 5 discrete actions — identical
I/O to the baseline so models are interchangeable. Ground-truth pos/vel/KE are
logged for probes and physics checks but never shown to the model.

## 2. Conventions

- Device from `wm2.config.get_device()` (MPS→CUDA→CPU). All modules import
  hyperparameters from `wm2.config.CFG`.
- Frames: `obs_to_tensor` / `tensor_to_obs` (reuse `wm/vae.py` helpers or copy):
  `(...,H,W,3) uint8 ↔ (...,3,H,W) float[0,1]`, `.contiguous()` for MPS.
- `symlog(x)=sign(x)·log(1+|x|)`, `symexp(x)=sign(x)·(exp(|x|)−1)` (DreamerV3).
- Categorical latent `z`: shape `(B, stoch_cat, stoch_classes)` one-hot, flattened
  to `(B, stoch_dim)` for the feature `feat=[deter h ; flat z]`, width `feat_dim`.
- **Straight-through** sampling for `z`: `sample = onehot(argmax/multinomial); 
  z = sample + (probs - probs.detach())` so gradients flow through `probs`.
- **unimix**: categorical probs = `0.99*softmax(logits) + 0.01*uniform` (avoids
  0-prob KL blowups).

## 3. World model — RSSM (`wm2/rssm.py`)

A single `WorldModel(nn.Module)` fusing V+M (DreamerV3 core).

### Networks
- **Encoder** `enc: (B,3,64,64) -> (B, embed_dim)` — 4 stride-2 convs (32,64,128,256
  ch), SiLU, LayerNorm; flatten → Linear to `embed_dim`.
- **Decoder** `dec: (B, feat_dim) -> (B,3,64,64)` — mirror of encoder; outputs
  pixel means (MSE recon; symlog optional — pixels already in [0,1], plain MSE ok).
- **Sequence model (deterministic)** `GRUCell(stoch_dim + act_dim -> deter_dim)`:
  `h_t = GRU([z_{t-1}, a_{t-1}], h_{t-1})`. `act_dim = n_actions` (one-hot).
- **Prior (imagination) head** `h_t -> logits(stoch_cat,stoch_classes)` → `ẑ_t`.
- **Posterior (observation) head** `[h_t, embed_t] -> logits` → `z_t`.
- **Reward head** `feat -> two-hot over value_buckets` (symexp support); trained to
  `symlog(r)`.
- **Continue head** `feat -> Bernoulli logit` (episode not-done).

### Forward (training, one batched sequence of length L)
Given `obs(B,L,3,64,64)`, `act(B,L)` one-hot, `reward(B,L)`, `cont(B,L)`:
1. `embed = enc(obs)`.
2. Roll RSSM over time: for t in 0..L-1: `h_t = GRU([z_{t-1}, a_{t-1}], h_{t-1})`;
   `prior_t = prior_head(h_t)`; `post_t = post_head(h_t, embed_t)`;
   `z_t ~ post_t` (straight-through). (`z_{-1}=0, a_{-1}=0, h_{-1}=0`.)
3. `feat_t = [h_t, flat z_t]`; decode `recon`, predict `reward`, `cont`.

### Loss (DreamerV3)
```
L = recon_scale * MSE(recon, obs)                         # over pixels, mean
  + reward_scale * two_hot_CE(reward_logits, symlog(r))   # mean
  + cont_scale  * BCE(cont_logit, cont)
  + kl_scale * ( kl_balance      * KL(sg(post) || prior)  # train the prior toward post
               + (1-kl_balance)  * KL(post || sg(prior)) ) # regularize post toward prior
```
- `KL` is summed over the `stoch_cat` categoricals, **free-bits clipped**:
  `max(KL, kl_free_bits)` (per-sequence-step, then mean).
- `sg` = stop-gradient. Use **unimix** logits in both prior and post before KL.
- Optimizer: Adam `wm_lr`, global grad-norm clip `grad_clip`.

### Inference helpers (used by imagination + dreaming)
- `obs_step(h, z, a, embed) -> (h', post_z)` — one filtered step (uses obs).
- `img_step(h, z, a) -> (h', prior_z)` — one **imagined** step (no obs).
- `decode(feat) -> frame`, `reward_of(feat)->r`, `cont_of(feat)->p`.
- `initial(B)` zero state.

## 4. Latent Action Model (`wm2/lam.py`, Genie-style, label-free)

Demonstrates **unsupervised action discovery + controllability** — Genie's
signature. Trained on frame pairs `(o_t, o_{t+1})` with **no action labels**.
- **Inverse model** `I(emb_t, emb_{t+1}) -> e_a` (continuous), quantized to a
  codebook of `lam_codes` vectors (VQ-VAE, commitment cost `lam_beta`,
  straight-through). Encoder features reused/recomputed from the RSSM encoder
  (frozen) or a small private CNN.
- **Forward model** `F(emb_t, quantized e_a) -> emb_{t+1}_hat`; loss = recon of the
  next embedding + VQ losses. The bottleneck forces `e_a` to carry exactly the
  controllable change → the codes align with true thrust directions.
- **Deliverable metric**: cluster-purity / mutual information between inferred
  codes and the (held-out) true actions; a confusion matrix code↔action.

## 5. Actor–critic in imagination (`wm2/agent.py`, replaces CEM+probe)

Train control **purely on imagined RSSM rollouts** — no env, no probe.
- **Actor** `π(feat) -> Categorical(n_actions)`; MLP (2×hidden, SiLU, LayerNorm).
- **Critic** `v(feat) -> two-hot over value_buckets` (symexp); plus a **slow
  target critic** (EMA `slow_critic_tau`) for stable λ-returns.
- **Imagination**: from a batch of posterior states (encoded from real data),
  roll `img_step` for `imag_horizon` steps, sampling actions from `π`; predict
  `reward`, `cont`, `value` at each.
- **λ-returns** (`gamma`,`lambda_`) computed backward with `cont` as discount mask.
- **Critic loss**: two-hot CE to `sg(λ-return)` (+ slow-target reg).
- **Actor loss**: REINFORCE on advantages `(λ-return − v)` with `sg`, **+ entropy
  `actor_ent`**. (Discrete actions ⇒ score-function gradient; DreamerV3 style.)
- Optimizers: Adam `actor_lr`/`critic_lr`, grad clip.

## 6. JEPA track (`wm2/jepa.py`, decoder-free, optional/stretch)

Engages LeCun's non-generative bet on the same world.
- **Online encoder** `f_θ` and **EMA target** `f_ξ` (`jepa_ema`), both small CNNs →
  `jepa_dim`. **Predictor** `g([s_t, a_t]) -> ŝ_{t+1}` in latent space.
- **Loss** = `1 - cos(ŝ_{t+1}, sg(f_ξ(o_{t+1})))` (+ variance/cov reg to prevent
  collapse, VICReg-lite). **No pixel decoder** in this path.
- **Planner** `wm2/plan.py`: CEM/MPPI over action sequences (`plan_horizon`,
  `plan_pop`, `plan_iters`, `plan_elite`) minimizing latent distance of the
  predicted final state to the **goal embedding** (encode a goal frame). This is
  V-JEPA-2-style latent MPC.
- **Comparison**: generative (actor-critic) vs non-generative (JEPA-MPC) success,
  sample-efficiency, and drift on the same task.

## 7. Training pipeline (`wm2/train.py`)

```
collect      # MultiBody2DEnv rollouts -> wm2_data/{obs,act,rew,cont,pos,vel,ke}.npz
train_wm     # RSSM world model (sec 3); cache posterior latents
train_ac     # actor-critic in imagination (sec 5)
[train_lam]  # latent action model (sec 4)
[train_jepa] # JEPA encoder+predictor (sec 6)
```
Each stage saves to `wm2_checkpoints/` and writes artifacts. A `run_all.sh`
mirrors the baseline's.

## 8. Evaluation (`wm2/evaluate.py`, `wm2/viz.py`)

**Head-to-head vs baseline** (same single-ball task = `wm.env.Particle2DEnv`, so
the architectures are compared, not the environments):
1. **Dream fidelity & drift** — real | recon | dream montage + GIF; pixel-MSE and
   latent-error vs horizon curves (overlay baseline vs wm2).
2. **Simulator faithfulness (physics)** — on `MultiBody2DEnv`: KE / momentum drift
   across dreamed collisions; **latent→{pos,vel}** probe R² (baseline only probed
   position — does the RSSM encode *velocity*?).
3. **Controllability** — goal-reach success vs random, baseline-CEM, and wm2
   actor-critic (and JEPA-MPC), all trained without touching the real env.
4. **Sample efficiency** — success vs # env frames (Dreamer's signature win).
5. **Latent-action alignment** — code↔true-action confusion matrix (Genie).
6. **WorldScore-style composite** — report a small (controllability, consistency,
   fidelity) triple inspired by the WorldScore benchmark axes.

## 9. File layout

```
wm2/
  config.py    done — all hyperparameters + paths + device
  env.py       done — MultiBody2DEnv (+ rollout, policies)
  utils.py     symlog/symexp, two-hot encode/decode, obs<->tensor, straight-through
  rssm.py      WorldModel: encoder, decoder, RSSM, reward/cont heads, losses, steps
  lam.py       Latent Action Model (VQ inverse+forward dynamics)
  agent.py     Actor, Critic, imagination rollout, λ-returns, train_ac
  jepa.py      JEPA encoder/target/predictor (decoder-free)
  plan.py      CEM/MPPI latent planner (uses rssm OR jepa)
  collect.py   data collection
  train.py     orchestration entry points + run_all
  evaluate.py  head-to-head metrics vs wm/ baseline
  viz.py       dreams, drift curves, latent maps, code↔action matrix
DESIGN.md      this file
```

## 10b. 2026 research refinements (from `wm2/RESEARCH_2026.md`)

The fresh June-2026 deep-research pass (Genie 3, MirageLSD, V-JEPA 2, Marble, the
mechanistic taxonomy arXiv:2601.17067, the efficient-video-WM survey
arXiv:2603.28489) **validates** every choice above and adds two concrete upgrades:

- **Latent history augmentation (drift fix).** The canonical long-horizon recipe
  is Diffusion Forcing + *history augmentation* (train on corrupted history so the
  model recovers from its own errors). RSSM-native analogue: during world-model
  training, with probability `p_hist` (~0.15) replace the posterior latent fed to
  the *next* step with the model's own **prior sample**. Teaches recovery from
  rollout error → less open-loop drift. Implement in `train_wm` (a flag on
  `observe`). Measure drift with/without it in `evaluate.py`.
- **Positioning vocabulary.** Report `wm2` as an **explicit, coupled state** (GRU
  recurrence in the backbone — same family as RAD = "LSTM-in-a-DiT"), with
  Observation/State/Dynamics framing, and the JEPA track as the **non-generative**
  paradigm. This is the language `README`/docs should use.
- **Eval targets the consensus open problems** (benchmarking, long-horizon,
  physical fidelity+controllability): keep the WorldScore-style (controllability,
  consistency, fidelity) triple + the momentum/energy conservation probe.

## 10. Laptop budget

Params ~3–6M total (enc/dec ~2M, RSSM heads small, actor/critic MLPs tiny). Target
end-to-end (collect→wm→ac) in well under an hour on MPS. Start with
`n_bodies=1` to validate the architecture against the baseline task, then scale to
`n_bodies=3` for the spatial/physics showcase. Keep `wm_seq_len`, batch, and epoch
counts modest; scale down on OOM/slowness.
