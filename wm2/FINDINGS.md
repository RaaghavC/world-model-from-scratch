# wm2 — engineering findings (building a 2026 world model from scratch)

An honest log of what it took to make a discrete-latent, DreamerV3-style world
model work on a laptop, and what it taught about the gap between *architecture*
and *results*. This is the kind of thing the 2026 literature calls out as the
real open problems (RESEARCH_2026 §"open problems"): controllability, long-horizon
consistency, and the renderer-vs-simulator distinction — encountered first-hand at
toy scale.

## What was built (and verified)

The full 2026 stack, from scratch, no RL libraries:
- a discrete-latent **RSSM** world model (encoder/decoder + GRU + categorical
  stochastic state, KL balancing + free bits, symlog, two-hot reward/value heads);
- a **latent action model** (Genie-style, label-free VQ inverse+forward dynamics);
- an **actor–critic trained in imagination** (λ-returns, two-hot critic, no probe);
- a **decoder-free JEPA** track + **CEM latent MPC** planner.

The numerical core (`utils.py`) is unit-tested; the RSSM and actor-critic each
have passing smoke tests; the breadth modules were implemented by a multi-agent
build workflow and code-reviewed. **Architecture was the easy part.**

## What was hard: getting the representation to encode the agent

Three bugs — each a real, citable phenomenon — stood between "it runs" and "it
learns a useful state." Each was found by *empirical diagnosis* (probe the latent,
look at the dream), not by reading code:

1. **Foreground collapse.** Plain pixel-MSE is dominated by the large dark
   background, so the optimizer drops the small bright agent to a blur — the
   latent then fails to encode position (probe R² **0.35**). Foreground-weighting
   the reconstruction (the baseline's fix, here inside the RSSM) recovered it.
   *Corollary, found the hard way:* if the **goal** square is large, it dominates
   the foreground loss and the model ignores the still-smaller **agent** — so the
   goal had to be shrunk and the weight raised (to 30) for the agent to survive.

2. **Two-hot support mismatch.** A symlog value grid of [-20, 20] with only **41
   buckets** crowds rewards in [-1, 0] into ~2 bins, so the reward head learns only
   the mean (corr **0.48**). DreamerV3's **255 buckets** restored resolution (corr
   **0.72**). The two-hot *expectation* is exact at any bin count; the *learning
   signal* is not.

3. **Velocity is only learned if motion is salient.** A single frame never shows
   velocity, and when per-step motion is tiny the prior can predict "no movement"
   and be mostly right — so velocity R² ≈ 0 and imagined rollouts are **sticky**
   (motion under-shot ~30×). Faster motion makes velocity worth encoding.

## Control: it failed, we diagnosed it, we fixed it

The first controllers (an imagination actor-critic and a 25-step CEM-MPC plan)
landed **at or below random** — worse, even. We diagnosed it instead of guessing:
from real states, we measured how often the world model's imagined rollout +
position probe identify the **goal-ward action**, as a function of horizon:

| imagination horizon | action-ranking accuracy |
|---|---|
| H = 5  | **70%** (chance = 20%) |
| H = 12 | 59% |
| H = 25 | 55% |

The signal is strong at short horizon and **decays as the open-loop rollout drifts
off the data manifold** (the prior-vs-posterior KL is ~6 nats, so far-future
imagined states are out-of-distribution for the heads — the long-horizon
consistency problem, locally). So the 25-step plan was optimizing noise.

The fix is a **short-horizon (H≈5) receding-horizon planner**: each step, score
every action by a 5-step constant-action imagined rollout, take the best, replan.
70% per-step accuracy + replanning ⇒ **~90% goal-reaching (closest dist ~0.02)
vs 25% random — well above the 2018 baseline's 39%.** Control through the world
model, the policy never having touched the environment.

## The JEPA 9.5-hour stall: a one-line data bug

The first JEPA run took **9.5 hours** for what should be a one-minute train. Cause:
the per-step sampler indexed a lazy `NpzFile` (`data["obs"][...]`) inside a list
comprehension, and `NpzFile` **re-decompresses the entire 495 MB array on every
access** — 128 full decompressions per step. Materializing the archive once
(`dict(np.load(...))`) cut it to **26 ms/step, a ~700× speedup**. A reminder that a
500×-slow job usually means an O(N) cost hiding inside an inner loop, not "ML is
slow."

## Label-free action discovery needs action-driven observations

The latent action model (Genie's module) is the **hardest** capability, and the
honest result is a *characterized partial success*, not a clean win. On the
**momentum** world it sits at chance (~0.23 purity) for a fundamental reason:
predicting the next frame mostly needs *velocity* (recoverable from the past), so
the thrust action is a tiny residual and the VQ code gets **starved**. Genie's own
setting avoids this — game video is **action-driven** (press right → pixels move
right) — so we move the LAM to a **direct-control variant** (the action sets
velocity; we verified displacement == action 100% of the time). There the picture
sharpens into a precise finding:

- A **supervised** probe decodes the executed action from the latent transition at
  **87%** (vs 20% chance) — *the action is plainly encoded.*
- Fully **unsupervised** VQ clustering, even with a displacement-quantizing inverse
  (quantize emb_{t+g}−emb_t, the change the action causes), recovers it only
  **partially (~0.37 purity, NMI ~0.12)**.

The 87→37 gap *is* the result: the latent displacement is position-dependent (a
non-linear encoder + wall clamping), so the same action looks different at
different places, and an unsupervised codebook latches onto that nuisance variation.
This is the real, acknowledged difficulty of unsupervised latent-action learning —
reported straight rather than tuned to look clean. Everything else (world model,
control, dreams, JEPA, state faithfulness) works and beats or matches the 2018
baseline; this one capability is where the frontier's open problem bites.

## Results (single-body, after the fixes)

<!-- RESULTS: filled from wm2_artifacts/eval_results.json -->
*(see [`README.md`](README.md) results table and `wm2_artifacts/eval_results.json`)*

## Lessons for the next builder

- **Diagnose the latent, not the loss.** A falling recon loss hid a latent that
  didn't encode the agent. Probe R² and *looking at the dream* found it in minutes.
- **Match every scale.** Foreground weight to object size; two-hot support to
  reward magnitude; imagination horizon to per-step motion.
- **Pick the domain to match the capability you're testing.** A "richer" multi-
  body world made the *world model* prettier but quietly broke label-free action
  discovery and control. Simpler, action-dominated dynamics are the honest test
  bed for those.
- **The architecture is the easy 20%.** The 80% is conditioning the data,
  representation, and objectives so the thing actually encodes what you need.
