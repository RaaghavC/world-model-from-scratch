# wm2 — a 2026-frontier world model, from scratch

A small but **architecturally state-of-the-art** world model, built from first
principles in PyTorch and trained on a laptop (Apple M-series / MPS). It is the
successor to the [`wm/`](../wm) baseline — a faithful reproduction of Ha &
Schmidhuber's 2018 **V-M-C** *World Models* — and it upgrades **every one of the
three boxes** to the techniques that define the world-model frontier as of **June
2026**, then benchmarks itself head-to-head against that 2018 baseline on the same
kind of task.

> Why this exists: the 2018 lineage is "the legible skeleton on which the entire
> 2026 frontier is hung." `wm2` swaps each box of that skeleton for its modern
> form, so you can see — and measure — what five years of world-model research
> actually bought, on a domain small enough to train from scratch over lunch.
>
> The cited research this is built on: a broad survey in
> [`../docs/RESEARCH.md`](../docs/RESEARCH.md) and a fresh, fact-checked June-2026
> frontier briefing in [`RESEARCH_2026.md`](RESEARCH_2026.md). The full
> architecture spec is [`DESIGN.md`](DESIGN.md).

## 📖 Start here

- **New to all this?** [`UNDERSTANDING.md`](UNDERSTANDING.md) explains world models
  and this project **from zero** — no background needed.
- **Want to run it yourself?** [`USER_GUIDE.md`](USER_GUIDE.md) — step-by-step.
- **The paper:** [`paper/paper.tex`](paper/paper.tex) → `paper/paper.pdf`.
- **The honest engineering story (the real lessons):** [`FINDINGS.md`](FINDINGS.md).
- **Architecture spec:** [`DESIGN.md`](DESIGN.md) · **2026 research briefing:**
  [`RESEARCH_2026.md`](RESEARCH_2026.md).

## The 2018 → 2026 upgrade

The field has bifurcated into two paradigms (RESEARCH_2026): **generative**
interactive world models (Genie 3, MirageLSD — frame-by-frame autoregression) and
**non-generative** JEPA predictors (V-JEPA 2 — latent prediction + planning).
`wm2` builds a small instance of **both** on one toy world, and keeps DreamerV3's
discrete-latent RSSM as the model-based-RL core.

| Box | 2018 baseline (`wm/`) | 2026 upgrade (`wm2/`) | Operationalizes |
|---|---|---|---|
| **V** (Renderer) | ConvVAE, continuous Gaussian latent | CNN over a **discrete categorical** latent (straight-through) | DreamerV2/V3 |
| **M** (Simulator) | MDN-RNN (LSTM + 5 Gaussians) | **RSSM**: deterministic GRU + categorical stochastic state, KL-balanced + free bits, symlog, **learned reward & continue heads** | PlaNet → DreamerV3 |
| — | *(none)* | **Latent Action Model** — discovers actions from raw frame pairs, *no action labels* | Genie 1–3 |
| **C** (Planner) | linear policy, CEM on a hand-built position **probe** | **actor–critic in imagination** (λ-returns, two-hot value), reward **learned**; **+ CEM latent MPC** | Dreamer / TD-MPC2 |
| — | *(none)* | **JEPA track** — decoder-free joint-embedding predictor + latent MPC | V-JEPA 2 |
| drift | open-loop dreams drift | **latent history augmentation** (RSSM-native Diffusion-Forcing) | MirageLSD 2025 |

In the **mechanistic vocabulary** of the Jan-2026 taxonomy (arXiv:2601.17067),
`wm2` is an **explicit, coupled state** — a recurrence (GRU) that carries memory
*inside* the dynamics backbone (the same family as RAD's "LSTM-in-a-DiT") — with
**state estimation** (the encoder/posterior) and **state transition** (the prior)
as its two core operations, exactly the Observation/State/Dynamics factoring.

## The world

[`env.py`](env.py) is a from-scratch, configurable physics world (64×64×3 frames,
5 discrete actions — identical interface to the baseline, so the models are
directly comparable). The **headline results** use a single **thrusted agent +
goal** under momentum + friction — a fair, baseline-matched task whose key property
is that **velocity is hidden** from any single frame, so a faithful state must
*infer* it (it does: velocity R² 0.32). The env also supports a **multi-body**
variant (elastic collisions + depth-ordered occlusion — a 2.5-D scene) and a
**direct-control** variant (`direct_control=True`, the action sets velocity); the
latter is the action-driven regime where label-free action discovery is meaningful
(see Results / [`FINDINGS.md`](FINDINGS.md)).

## Results

Trained from scratch on an Apple M-series laptop (MPS); numbers from
`wm2_artifacts/eval_results.json`, plots in `wm2_artifacts/`.

| Capability | 2018 baseline (`wm/`) | **wm2 (2026)** |
|---|---|---|
| latent → agent **position** R² | ~0.97 | **0.96** |
| latent → agent **velocity** R² (dynamics, not appearance) | — (never probed) | **0.55** |
| **goal-reach success** (policy never touched the env) | 0.39 vs 0.05 random | **0.90 vs 0.25 random** |
| dream tracks the agent open-loop | ✅ | ✅ (see `dream_compare.gif`) |
| **JEPA** decoder-free latent predictor | — | ✅ cos 0.95, no collapse |
| long-horizon dream drift (pixel-MSE h5→h60) | rises with horizon | 410 → 722 (mild) |
| label-free latent-action ↔ true-action | — | partial: **0.37 purity** (chance 0.20); 87% *supervised* |
| WorldScore-style mean (controllability/consistency/fidelity) | — | **0.82** |

*(Numbers from one `eval_results.json` run; expect a few % run-to-run from the
re-fit probe. The qualitative picture is stable.)*

**The headline:** a discrete-latent world model whose learned latent is a faithful
*state* (position R² ~0.96, and — unlike the baseline — **velocity** is decodable
too, R² ~0.55), and a controller that **reaches the goal ~90% of the time having
never touched the environment** (short-horizon model-predictive control through the
model's imagination), well above the 2018 baseline's 39%. Dreams track the agent
open-loop on a clean background; the JEPA track learns without collapse.

**The honest edges** ([`FINDINGS.md`](FINDINGS.md) tells the full debugging story):
long open-loop dreams still drift; the *learned-reward* actor-critic underperforms
the probe-guided planner (the reward head is the weak link, so the planner is the
deployed controller); and **label-free action discovery** — the Genie capability —
only partially recovers the actions (0.37 purity) even though a *supervised* probe
shows the action is 87% decodable from the latent transition. That 87→37 gap is the
genuine open problem of unsupervised latent-action learning, reported straight.

## Run it

```bash
pip install -r ../requirements.txt
bash run_all.sh        # collect → world model → actor-critic → LAM → JEPA → eval → viz
```

Or stage by stage (single CLI):

```bash
export PYTHONPATH=$PWD                  # from repo root
python -m wm2.train collect   # multi-body rollouts
python -m wm2.train wm        # discrete-latent RSSM world model (V+M)
python -m wm2.train ac        # actor-critic in imagination (C) — no probe
python -m wm2.train lam       # latent action model (Genie, label-free)
python -m wm2.train jepa      # decoder-free JEPA track (V-JEPA-style)
python -m wm2.train eval      # head-to-head metrics vs the wm/ baseline
python -m wm2.train viz       # dreams / drift / latent maps / code-action map
```

Everything auto-selects MPS → CUDA → CPU ([`config.py`](config.py) holds every
hyperparameter). Artifacts (GIFs, montages, plots, `eval_results.json`) land in
`wm2_artifacts/`.

## Layout

```
wm2/
  config.py    all hyperparameters + device
  env.py       MultiBody2DEnv (elastic collisions, occlusion, hidden velocity)
  utils.py     symlog/symexp, two-hot, straight-through categoricals (self-tested)
  rssm.py      WorldModel: encoder/decoder, RSSM, reward/continue heads, losses
  lam.py       Latent Action Model (VQ inverse+forward dynamics, label-free)
  agent.py     Actor, Critic, imagination training (λ-returns, two-hot)
  jepa.py      decoder-free JEPA encoder/target/predictor
  plan.py      CEM/MPPI latent planners (RSSM and JEPA)
  train.py     one-CLI orchestrator;  evaluate.py / viz.py   metrics & pictures
  DESIGN.md  RESEARCH_2026.md  README.md
```

## What's real, and what's simplified

A faithful, end-to-end realization of the *ideas* of the 2026 frontier on a small,
learnable domain — not a frontier-scale system. Honest caveats, each mirroring a
real open problem (RESEARCH_2026 §"open problems"): dreams still drift over long
horizons (mitigated, not solved, by latent history augmentation); the domain is
2.5-D and tiny (frontier systems are 3-D, photoreal, real-time — Genie 3, Marble);
and "physical faithfulness" is measured by latent probes + conservation checks,
not a full physics benchmark. The point is the **architecture** and the
**measured 2018→2026 delta**, reproducible on a laptop.
