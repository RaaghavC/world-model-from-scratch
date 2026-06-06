"""Latent Model-Predictive Control planners (PlaNet / TD-MPC2 / V-JEPA-2-AC style).

This is the THIRD projection of the world model in Fei-Fei Li's taxonomy -- the
**Planner** that outputs actions -- but, unlike `wm2/agent.py`, it uses **no
learned policy and no probe**. It is derivative-free Cross-Entropy-Method (CEM)
planning *directly in the model's latent space*, exactly the recipe the 2026
frontier converged on:

  * **PlaNet** (Hafner et al. 2019) first did CEM over action sequences inside an
    RSSM latent, scoring candidates by the model's predicted reward.
  * **TD-MPC2** (Hansen et al. 2024) plans in a learned latent with a short
    horizon, scoring with a learned reward/value.
  * **V-JEPA 2-AC** (Meta FAIR, arXiv:2506.09985) freezes a JEPA encoder, trains
    a latent action-conditioned predictor, and does **receding-horizon CEM** that
    minimizes the latent distance of the rolled-out final state to an *encoded
    goal frame* -- zero-shot robot control with no reward labels.

Two planners share one vectorized CEM loop:

  * `RSSMPlanner(wm)` -- generative. Rolls `wm.img_step` forward and SCORES each
    candidate by the summed `wm.reward(feat)` over the horizon. The world model's
    learned reward head already encodes the goal (r = -dist(agent, goal)), so the
    planner needs no explicit goal. Maintains the RSSM posterior across real env
    steps (mirrors `agent.WM2Policy`).
  * `JEPAPlanner(jepa, goal_obs)` -- non-generative (LeCun's bet). Rolls
    `jepa.predict` forward in the decoder-free latent and SCORES by the NEGATIVE
    latent distance of the predicted states to the encoded goal frame. This is the
    V-JEPA-2-AC MPC objective. `wm2.jepa` is imported LAZILY (it is built
    concurrently with this file).

CEM (per call): sample `plan_pop` action sequences of length `plan_horizon` from a
per-timestep categorical, evaluate them by batched latent rollout, keep the
`plan_elite` best, refit the categorical to the elites, repeat `plan_iters` times.
Return the first action of the best (mean) plan -- receding-horizon control.

Discrete actions => CEM over *categorical* action distributions: we keep a
(horizon, n_actions) table of per-step action probabilities and refit it to the
elite one-hot frequencies (with a little smoothing so no action is ever ruled out).
The rollouts are fully vectorized over the candidate population for speed; horizons
and population are kept modest so a plan is cheap on a laptop.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from wm2.config import CFG, get_device
from wm2.utils import obs_to_tensor


# ----------------------------------------------------------------- categorical CEM
def _sample_action_seqs(probs: torch.Tensor, pop: int) -> torch.Tensor:
    """Sample `pop` action sequences from a per-step categorical table.

    probs: (H, n_actions) row-normalized action probabilities per horizon step.
    Returns LongTensor (pop, H) of sampled discrete actions.
    """
    H, A = probs.shape
    # multinomial wants (rows, classes); draw `pop` samples per horizon step.
    seq = torch.multinomial(probs, pop, replacement=True)   # (H, pop)
    return seq.t().contiguous()                              # (pop, H)


def _refit_elites(elite_actions: torch.Tensor, n_actions: int,
                  smoothing: float = 0.02) -> torch.Tensor:
    """Refit the per-step categorical to elite action frequencies.

    elite_actions: (E, H) long. Returns (H, n_actions) smoothed probabilities --
    the CEM "update" step, the categorical analogue of refitting a Gaussian's mean
    to the elite set. `smoothing` keeps a small uniform floor so CEM never collapses
    to a degenerate one-hot (it would then be unable to recover a better action).
    """
    E, H = elite_actions.shape
    onehot = F.one_hot(elite_actions, n_actions).float()    # (E, H, A)
    freq = onehot.mean(0)                                   # (H, A) elite freq
    probs = (1 - smoothing) * freq + smoothing / n_actions
    return probs / probs.sum(-1, keepdim=True)


# --------------------------------------------------------------------- RSSM planner
class RSSMPlanner:
    """Latent MPC over the RSSM world model (PlaNet / TD-MPC2 style).

    Given the current posterior state, optimize a length-`plan_horizon` action
    sequence with CEM, scoring each candidate by the summed learned reward over the
    imagined rollout, and execute the first action. The posterior state is carried
    across real steps so the plan conditions on the model's filtered belief.
    """

    def __init__(self, wm, cfg=CFG, device=None, horizon=None, pop=None,
                 iters=None, elite=None):
        self.wm = wm.eval()
        self.cfg = cfg
        self.device = device or get_device()
        self.A = wm.n_actions
        self.H = horizon if horizon is not None else cfg.plan_horizon
        self.pop = pop if pop is not None else cfg.plan_pop
        self.iters = iters if iters is not None else cfg.plan_iters
        self.elite = elite if elite is not None else cfg.plan_elite
        self.reset()

    def reset(self, B: int = 1):
        """Reset the filtered posterior state and the previous action (one-hot)."""
        self.state = self.wm.initial(B, self.device)
        self.prev_a = torch.zeros(B, self.A, device=self.device)

    @torch.no_grad()
    def _expand(self, state: dict, n: int) -> dict:
        """Tile a B=1 state into a population of `n` identical states for batched
        rollout (the whole population starts from the same current belief)."""
        return {"deter": state["deter"].expand(n, -1).contiguous(),
                "stoch": state["stoch"].expand(n, *state["stoch"].shape[1:]).contiguous()}

    @torch.no_grad()
    def _rollout_scores(self, state: dict, action_seqs: torch.Tensor) -> torch.Tensor:
        """Imagine each candidate forward and return its summed reward.

        state: current posterior (B=1). action_seqs: (pop, H) long. Vectorized over
        the population: at each step we advance all `pop` imagined states in lockstep
        with `wm.img_step`, accumulating the reward head's output. Returns (pop,).

        We roll the prior at its MODE (`sample=False`) so the score is a
        DETERMINISTIC function of the action sequence. PlaNet/TD-MPC2 plan against
        the model's expected (not a single sampled) latent for exactly this reason:
        a deterministic objective is what lets CEM monotonically refit toward the
        high-return region. (Open-loop control in `act` then executes only the first
        action and re-plans, so stochasticity is handled by re-filtering each step.)
        """
        pop = action_seqs.shape[0]
        st = self._expand(state, pop)
        total = torch.zeros(pop, device=self.device)
        for t in range(self.H):
            a_oh = F.one_hot(action_seqs[:, t], self.A).float()
            st, _ = self.wm.img_step(st, a_oh, sample=False)
            total = total + self.wm.reward(self.wm.get_feat(st))   # (pop,)
        return total

    @torch.no_grad()
    def plan(self, state: dict | None = None):
        """Run CEM from `state` (defaults to the carried posterior).

        Returns (best_first_action:int, elite_mean_scores:list[float]) where the
        list is the mean elite score per CEM iteration (monotone non-decreasing when
        the model is informative -- used by the smoke test)."""
        state = state if state is not None else self.state
        # Start from a uniform per-step action distribution.
        probs = torch.full((self.H, self.A), 1.0 / self.A, device=self.device)
        elite_means = []
        best_action, best_score = 0, -float("inf")
        for _ in range(self.iters):
            seqs = _sample_action_seqs(probs, self.pop)        # (pop, H)
            scores = self._rollout_scores(state, seqs)          # (pop,)
            k = min(self.elite, self.pop)
            elite_idx = torch.topk(scores, k).indices
            elite_actions = seqs[elite_idx]                     # (E, H)
            elite_means.append(float(scores[elite_idx].mean().item()))
            probs = _refit_elites(elite_actions, self.A)        # (H, A)
            # Track the single best candidate seen so far (greedy first action).
            top = int(scores.argmax().item())
            if float(scores[top].item()) > best_score:
                best_score = float(scores[top].item())
                best_action = int(seqs[top, 0].item())
        # Receding horizon: prefer the first action of the refit plan's argmax; fall
        # back to the best raw candidate's first action (identical at convergence).
        plan_first = int(probs[0].argmax().item())
        return (plan_first if best_score == -float("inf") else best_action), elite_means

    @torch.no_grad()
    def act(self, obs_uint8):
        """Filter the new observation into the posterior, then plan one action.

        Mirrors `agent.WM2Policy.act`: encode the frame, take an `obs_step` with the
        previous action to update the belief, run CEM, store the chosen action's
        one-hot for the next filtering step, and return the discrete action.
        """
        x = obs_to_tensor(np.asarray(obs_uint8)).to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        embed = self.wm.encoder(x)
        self.state, _, _ = self.wm.obs_step(self.state, self.prev_a, embed)
        action, _ = self.plan(self.state)
        self.prev_a = F.one_hot(torch.tensor([action], device=self.device),
                                self.A).float()
        return int(action)


# --------------------------------------------------------------------- JEPA planner
class JEPAPlanner:
    """Latent MPC over a decoder-free JEPA predictor (V-JEPA-2-AC style).

    Identical CEM machinery to `RSSMPlanner`, but the rollout is `jepa.predict` in
    the JEPA latent and the score is the NEGATIVE distance of the predicted states
    to the encoded goal frame -- no reward head, no decoder. This operationalizes
    LeCun's non-generative bet on the same world.

    `wm2.jepa` is imported lazily (built concurrently). The planner depends only on
    a small, conventional interface, discovered defensively at construction time:
      * an encoder mapping a frame tensor (B,3,64,64) -> latent (B,jepa_dim), found
        as `jepa.encode` / `jepa.embed` / `jepa.online`(.forward) / `jepa.encoder`;
      * a predictor mapping (latent (B,D), action one-hot (B,n_actions)) -> next
        latent (B,D), found as `jepa.predict` / `jepa.predictor` / `jepa.forward`.
    """

    def __init__(self, jepa, goal_obs, cfg=CFG, device=None, horizon=None,
                 pop=None, iters=None, elite=None):
        self.jepa = jepa
        self.cfg = cfg
        self.device = device or get_device()
        self.A = cfg.n_actions
        self.H = horizon if horizon is not None else cfg.plan_horizon
        self.pop = pop if pop is not None else cfg.plan_pop
        self.iters = iters if iters is not None else cfg.plan_iters
        self.elite = elite if elite is not None else cfg.plan_elite
        if hasattr(jepa, "eval"):
            jepa.eval()
        # Encode the goal frame ONCE; all rollouts are scored against it.
        self.goal_z = self._encode(self._to_tensor(goal_obs)).detach()  # (1, D)
        self.z = None  # current latent belief, set on first act()

    # ---- defensive adapters to the concurrently-built jepa module ----------
    def _to_tensor(self, obs_uint8) -> torch.Tensor:
        x = obs_to_tensor(np.asarray(obs_uint8)).to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return x

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map frames (B,3,64,64) -> online-encoder latent (B, jepa_dim)."""
        j = self.jepa
        for name in ("encode", "embed", "encode_online"):
            fn = getattr(j, name, None)
            if callable(fn):
                return fn(x)
        # Module-style: an `online` / `encoder` submodule we can call.
        for name in ("online", "encoder", "f_online", "online_encoder"):
            mod = getattr(j, name, None)
            if mod is not None and callable(mod):
                return mod(x)
        raise AttributeError(
            "JEPAPlanner: could not find an encode method on the jepa object "
            "(looked for encode/embed/online/encoder).")

    def _predict(self, z: torch.Tensor, a_oh: torch.Tensor) -> torch.Tensor:
        """Advance the latent one step: (z (B,D), a_oh (B,A)) -> z' (B,D)."""
        j = self.jepa
        for name in ("predict", "step", "forward_latent"):
            fn = getattr(j, name, None)
            if callable(fn):
                return fn(z, a_oh)
        mod = getattr(j, "predictor", None)
        if mod is not None and callable(mod):
            return mod(z, a_oh)
        # Last resort: the module's own __call__ as the predictor.
        if callable(j):
            return j(z, a_oh)
        raise AttributeError(
            "JEPAPlanner: could not find a predict method on the jepa object "
            "(looked for predict/step/predictor).")

    def reset(self, B: int = 1):
        self.z = None

    @torch.no_grad()
    def _rollout_scores(self, z: torch.Tensor, action_seqs: torch.Tensor) -> torch.Tensor:
        """Roll the JEPA predictor forward for every candidate and score by negative
        latent distance to the goal embedding, accumulated over the horizon.

        z: current latent (1, D). action_seqs: (pop, H) long. Returns (pop,): higher
        is better (closer to the goal). We sum -||z_t - goal|| over the horizon so
        candidates that approach the goal AND stay near it both score well (a pure
        terminal-only score is also fine; the running sum is a smoother signal)."""
        pop = action_seqs.shape[0]
        zt = z.expand(pop, -1).contiguous()                 # (pop, D)
        goal = self.goal_z.expand(pop, -1)                  # (pop, D)
        total = torch.zeros(pop, device=self.device)
        for t in range(self.H):
            a_oh = F.one_hot(action_seqs[:, t], self.A).float()
            zt = self._predict(zt, a_oh)
            total = total - torch.linalg.vector_norm(zt - goal, dim=-1)  # (pop,)
        return total

    @torch.no_grad()
    def plan(self, z: torch.Tensor | None = None):
        """CEM in the JEPA latent. Returns (best_first_action, elite_mean_scores)."""
        z = z if z is not None else self.z
        probs = torch.full((self.H, self.A), 1.0 / self.A, device=self.device)
        elite_means = []
        best_action, best_score = 0, -float("inf")
        for _ in range(self.iters):
            seqs = _sample_action_seqs(probs, self.pop)
            scores = self._rollout_scores(z, seqs)
            k = min(self.elite, self.pop)
            elite_idx = torch.topk(scores, k).indices
            elite_actions = seqs[elite_idx]
            elite_means.append(float(scores[elite_idx].mean().item()))
            probs = _refit_elites(elite_actions, self.A)
            top = int(scores.argmax().item())
            if float(scores[top].item()) > best_score:
                best_score = float(scores[top].item())
                best_action = int(seqs[top, 0].item())
        plan_first = int(probs[0].argmax().item())
        return (plan_first if best_score == -float("inf") else best_action), elite_means

    @torch.no_grad()
    def act(self, obs_uint8):
        """Encode the current frame to the JEPA latent (no recurrence -- JEPA's
        latent is a per-frame state estimate) and plan one action toward the goal."""
        self.z = self._encode(self._to_tensor(obs_uint8)).detach()
        action, _ = self.plan(self.z)
        return int(action)


# --------------------------------------------------------------------- smoke test
def _smoke():
    """CPU-only smoke on tiny sizes (no trained checkpoint, no GPU contention).

    Builds a FRESH random RSSM world model and verifies (1) `RSSMPlanner.act`
    returns a valid discrete action, (2) the CEM elite score IMPROVES across
    iterations -- i.e. the optimizer is actually selecting better action sequences
    under the (random but fixed) reward head. A random-but-fixed reward head still
    induces a non-trivial preference over action sequences, so CEM should climb it.
    """
    from wm2.rssm import WorldModel

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cpu")           # force CPU: do NOT contend for MPS
    cfg = CFG

    wm = WorldModel(cfg).to(device).eval()

    # Tiny CEM sizes so the smoke finishes in seconds.
    planner = RSSMPlanner(wm, cfg, device=device,
                          horizon=5, pop=64, iters=6, elite=8)

    # (1) act() on a fake frame returns a valid integer action.
    fake_obs = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    a = planner.act(fake_obs)
    assert isinstance(a, int) and 0 <= a < cfg.n_actions, f"bad action {a!r}"
    print(f"RSSMPlanner.act -> action {a} (valid, in [0,{cfg.n_actions}))")

    # (2) Elite score improves across CEM iterations from a fixed start belief.
    # Average over a few random start states to get a clean monotone-ish signal
    # (CEM is stochastic; the TREND must be up even if a single run wiggles).
    H, POP, ITERS, ELITE = 6, 256, 8, 16
    planner2 = RSSMPlanner(wm, cfg, device=device,
                           horizon=H, pop=POP, iters=ITERS, elite=ELITE)
    n_runs = 8
    curves = np.zeros((n_runs, ITERS))
    for r in range(n_runs):
        st = wm.initial(1, device)
        # Give each run a different (random) but FIXED belief to plan from.
        st["stoch"] = F.one_hot(
            torch.randint(0, cfg.stoch_classes, (1, cfg.stoch_cat)),
            cfg.stoch_classes).float().to(device)
        st["deter"] = torch.randn(1, cfg.deter_dim, device=device) * 0.5
        _, means = planner2.plan(st)
        curves[r] = np.array(means)
    mean_curve = curves.mean(0)
    print("CEM elite-score curve (mean over runs):")
    for i, v in enumerate(mean_curve):
        print(f"  iter {i}: {v:+.4f}")
    first, last = mean_curve[0], mean_curve[-1]
    assert np.isfinite(mean_curve).all(), "non-finite CEM scores"
    assert last > first + 1e-4, (
        f"CEM did not improve elite score: first={first:.4f} last={last:.4f}")
    # Also sanity-check it is (mostly) monotone: last >= max of the first half.
    assert last >= mean_curve[: ITERS // 2].max() - 1e-3, "CEM not climbing"
    print(f"CEM elite score improved: {first:+.4f} -> {last:+.4f}  "
          f"(+{last - first:.4f})")
    print(f"plan.py smoke PASSED  device={device}")


if __name__ == "__main__":
    _smoke()
