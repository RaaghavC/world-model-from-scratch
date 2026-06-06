"""Head-to-head evaluation: wm2 (2026 frontier) vs the wm/ baseline (2018 V-M-C).

Reports the quantitative claims (viz.py renders the pictures):

  1. Simulator faithfulness -- a latent->state probe. The baseline showed the
     ball POSITION is linearly decodable from its VAE latent (R^2~0.97). We ask
     the harder question the multi-body world enables: does the RSSM latent also
     encode VELOCITY and EVERY body's position (through occlusion)? A latent that
     encodes velocity is a *simulator* state, not just an *appearance* code
     (Fei-Fei Li's renderer-vs-simulator distinction, made measurable).
  2. Dream fidelity & long-horizon drift -- open-loop pixel MSE vs horizon.
  3. Controllability -- goal-reach success of policies trained WITHOUT touching
     the env (wm2 actor-critic in imagination; wm2 RSSM-CEM latent planner) vs a
     random baseline, with the 2018 baseline's published numbers for reference.
  4. Latent-action alignment -- the Genie LAM's codes vs the true actions.
  5. A WorldScore-style (controllability, consistency, fidelity) composite.

Results are saved to wm2_artifacts/eval_results.json and printed as a table.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wm2.config import CFG, get_device
from wm2.env import MultiBody2DEnv
from wm2.rssm import WorldModel
from wm2.utils import obs_to_tensor

HELDOUT = slice(360, 400)          # episodes reserved for evaluation


# --------------------------------------------------------------- latent probe
class Probe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.SiLU(),
                                 nn.Linear(128, out_dim))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def _features(wm, obs_np, act_np, device, bs=10):
    """Run the RSSM posterior over episodes -> features (E*L, feat_dim)."""
    feats = []
    E = obs_np.shape[0]
    L = act_np.shape[1]
    for s in range(0, E, bs):
        e = slice(s, min(s + bs, E))
        obs = obs_to_tensor(obs_np[e, :L]).to(device)
        act = torch.from_numpy(act_np[e, :L]).long().to(device)
        out = wm.observe(obs, act)
        feats.append(out["feat"].reshape(-1, wm.cfg.feat_dim).cpu())
    return torch.cat(feats)


def _fit_probe(feat, target, device, epochs=60):
    """Train a small probe feat->target; return held-out R^2 per target dim + mean."""
    n = feat.shape[0]
    idx = torch.randperm(n)
    ntr = int(0.9 * n)
    tr, te = idx[:ntr], idx[ntr:]
    probe = Probe(feat.shape[1], target.shape[1]).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    Xtr, Ytr = feat[tr].to(device), target[tr].to(device)
    Xte, Yte = feat[te].to(device), target[te].to(device)
    bs = 2048
    for _ in range(epochs):
        p = torch.randperm(len(tr))
        for i in range(0, len(tr), bs):
            j = p[i:i + bs]
            opt.zero_grad()
            loss = lossf(probe(Xtr[j]), Ytr[j])
            loss.backward(); opt.step()
    with torch.no_grad():
        pred = probe(Xte)
        ss_res = ((pred - Yte) ** 2).sum(0)
        ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-8
        r2 = (1 - ss_res / ss_tot).cpu().numpy()
    return r2, float(r2.mean())


def simulator_faithfulness(wm, data, device):
    """Probe the RSSM latent for agent pos, agent vel, and ALL bodies' positions."""
    obs = data["obs"][HELDOUT]
    act = data["act"][HELDOUT]
    pos = data["pos"][HELDOUT]                 # (E,T+1,N,2)
    vel = data["vel"][HELDOUT]
    L = act.shape[1]
    feat = _features(wm, obs, act, device)
    N = pos.shape[2]
    agent_pos = torch.from_numpy(pos[:, :L, 0, :].reshape(-1, 2)).float()
    agent_vel = torch.from_numpy(vel[:, :L, 0, :].reshape(-1, 2)).float()
    all_pos = torch.from_numpy(pos[:, :L].reshape(-1, N * 2)).float()
    r2_pos, m_pos = _fit_probe(feat, agent_pos, device)
    r2_vel, m_vel = _fit_probe(feat, agent_vel, device)
    _, m_allpos = _fit_probe(feat, all_pos, device)
    res = {"agent_pos_R2": [round(float(x), 3) for x in r2_pos],
           "agent_pos_R2_mean": round(m_pos, 3),
           "agent_vel_R2": [round(float(x), 3) for x in r2_vel],
           "agent_vel_R2_mean": round(m_vel, 3),
           "all_bodies_pos_R2_mean": round(m_allpos, 3),
           "n_bodies": int(N), "n_samples": int(feat.shape[0])}
    print(f"  [faithfulness] agent pos R2={m_pos:.3f}  agent VEL R2={m_vel:.3f}  "
          f"all-{N}-bodies pos R2={m_allpos:.3f}")
    return res


# --------------------------------------------------------- dream fidelity / drift
@torch.no_grad()
def dream_drift(wm, data, device, n_eps=12, horizon=70):
    """Open-loop dream from frame 0; mean pixel-MSE vs horizon, averaged over eps."""
    obs = data["obs"][HELDOUT][:n_eps]
    act = data["act"][HELDOUT][:n_eps]
    E = obs.shape[0]
    H = min(horizon, act.shape[1])
    errs = np.zeros((E, H))
    for k in range(E):
        o0 = obs_to_tensor(obs[k, 0]).to(device).unsqueeze(0)
        state = wm.initial(1, device)
        prev_a = torch.zeros(1, wm.n_actions, device=device)
        # ground the first state on the real frame 0
        embed = wm.encoder(o0)
        state, _, _ = wm.obs_step(state, prev_a, embed)
        for t in range(H):
            feat = wm.get_feat(state)
            recon = wm.decode(feat)[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            real = obs[k, t].astype(np.float32) / 255.0
            errs[k, t] = float(((recon - real) ** 2).mean()) * (255.0 ** 2)
            a = torch.tensor([int(act[k, t])], device=device)
            a_oh = torch.nn.functional.one_hot(a, wm.n_actions).float()
            state, _ = wm.img_step(state, a_oh, sample=False)
    mean_curve = errs.mean(0)
    res = {"drift_curve": [round(float(x), 1) for x in mean_curve],
           "mse_h5": round(float(mean_curve[min(4, H - 1)]), 1),
           "mse_h20": round(float(mean_curve[min(19, H - 1)]), 1),
           "mse_h60": round(float(mean_curve[min(59, H - 1)]), 1),
           "horizon": int(H)}
    print(f"  [drift] dream pixel-MSE  h5={res['mse_h5']}  h20={res['mse_h20']}  "
          f"h60={res['mse_h60']}")
    return res


# ------------------------------------------------------------------ controllability
import inspect


def _reset_policy(policy, goal):
    """Call policy.reset(goal) if it accepts a goal, else policy.reset()."""
    try:
        if "goal" in inspect.signature(policy.reset).parameters:
            policy.reset(goal); return
    except (ValueError, TypeError):
        pass
    policy.reset()


def _run_episodes(env_factory, make_policy, n_eps, ep_len, seed0=4000):
    """Roll a FRESH policy in fresh envs; goal-aware policies get it at reset."""
    reached, closest = [], []
    for k in range(n_eps):
        env = env_factory(seed0 + k)
        o = env.reset()
        policy = make_policy()
        _reset_policy(policy, env.goal.copy())
        best, hit = 1e9, False
        for _ in range(ep_len):
            o, _, done, info = env.step(policy.act(o))
            best = min(best, info["dist"]); hit = hit or info["reached"]
            if done:
                break
        reached.append(float(hit)); closest.append(best)
    return {"success": round(float(np.mean(reached)), 3),
            "mean_closest_dist": round(float(np.mean(closest)), 3)}


class RandomPolicy:
    def __init__(self, n_actions, repeat=6, seed=0):
        self.n, self.repeat = n_actions, repeat
        self.rng = np.random.default_rng(seed); self._a = 0; self._left = 0

    def reset(self, goal=None):
        self._left = 0

    def act(self, obs=None):
        if self._left <= 0:
            self._a = int(self.rng.integers(0, self.n)); self._left = self.repeat
        self._left -= 1
        return self._a


def _train_pos_probe(wm, data, device, n_eps=150, epochs=100):
    """Latent -> agent-position probe, reused as a clean reward for planning."""
    obs = data["obs"][:n_eps]; act = data["act"][:n_eps]; pos = data["pos"][:n_eps, :, 0, :]
    L = act.shape[1]
    feat = _features(wm, obs, act, device).to(device)
    Y = torch.from_numpy(pos[:, :L].reshape(-1, 2)).float().to(device)
    probe = Probe(wm.cfg.feat_dim, 2).to(device)
    opt = torch.optim.Adam(probe.parameters(), 1e-3)
    for _ in range(epochs):
        idx = torch.randperm(feat.shape[0])
        for i in range(0, len(idx), 4096):
            j = idx[i:i + 4096]
            opt.zero_grad(); F.mse_loss(probe(feat[j]), Y[j]).backward(); opt.step()
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe


class WMPlanner:
    """Short-horizon receding-horizon control THROUGH the world model's imagination
    (model-predictive control in latent space; the policy never touched the env).

    Each real step: encode the frame to a posterior latent, then score every action
    by a SHORT (H-step) constant-action imagined rollout, read off the agent's
    position with a latent->position probe, pick the action that gets closest to the
    goal, execute it, and replan. The horizon is deliberately short: long open-loop
    rollouts drift off the data manifold and the heads become unreliable there (a
    25-step CEM plan scored ~chance and failed), whereas a 4-5 step lookahead stays
    in-distribution (≈70% per-step action accuracy) and replanning fixes the rest.
    """

    def __init__(self, wm, probe, device, H=5):
        self.wm, self.probe, self.device, self.H = wm, probe, device, H
        self.na = wm.n_actions

    def reset(self, goal):
        self.state = self.wm.initial(1, self.device)
        self.prev = torch.zeros(1, self.na, device=self.device)
        self.goal = torch.tensor(goal, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def act(self, obs):
        x = obs_to_tensor(np.asarray(obs)).to(self.device).unsqueeze(0)
        self.state, _, _ = self.wm.obs_step(self.state, self.prev, self.wm.encoder(x))
        # Evaluate all `na` actions in parallel: hold each for H imagined steps.
        st = {"deter": self.state["deter"].expand(self.na, -1).contiguous(),
              "stoch": self.state["stoch"].expand(self.na, -1, -1).contiguous()}
        acts = torch.arange(self.na, device=self.device)
        for _ in range(self.H):
            st, _ = self.wm.img_step(st, F.one_hot(acts, self.na).float(), sample=False)
        d = torch.linalg.norm(self.probe(self.wm.get_feat(st)) - self.goal, dim=-1)
        a = int(d.argmin())
        self.prev = F.one_hot(torch.tensor([a], device=self.device), self.na).float()
        return a


def controllability(wm, data, device, n_eps=40, ep_len=None):
    """Goal-reaching of random / probe-MPC / actor-critic. None touched the env."""
    cfg = wm.cfg
    ep_len = ep_len or cfg.ep_len
    env_factory = lambda s: MultiBody2DEnv(cfg, seed=s)
    out = {}
    out["random"] = _run_episodes(env_factory, lambda: RandomPolicy(cfg.n_actions),
                                  n_eps, ep_len)
    print(f"  [control] random       success={out['random']['success']}  "
          f"dist={out['random']['mean_closest_dist']}")
    # wm2 latent MPC planner: short-horizon control through the world model.
    try:
        probe = _train_pos_probe(wm, data, device)
        out["wm2_planner"] = _run_episodes(
            env_factory, lambda: WMPlanner(wm, probe, device), n_eps, ep_len)
        print(f"  [control] wm2 planner  success={out['wm2_planner']['success']}  "
              f"dist={out['wm2_planner']['mean_closest_dist']}")
    except Exception as e:
        out["wm2_planner"] = {"error": str(e)[:200]}
        print(f"  [control] wm2 planner  SKIPPED ({str(e)[:90]})")
    # Pure learned-reward actor-critic, only if it was trained (`wm2.train ac`).
    if os.path.exists(os.path.join(cfg.ckpt_dir, "actor_critic.pt")):
        try:
            from wm2.agent import WM2Policy
            out["wm2_actor_critic"] = _run_episodes(
                env_factory, lambda: WM2Policy.load(device), n_eps, ep_len)
            print(f"  [control] actor-critic success={out['wm2_actor_critic']['success']}"
                  f"  dist={out['wm2_actor_critic']['mean_closest_dist']}")
        except Exception as e:
            out["wm2_actor_critic"] = {"error": str(e)[:160]}
    else:
        out["wm2_actor_critic"] = {"note": "not deployed; the learned-reward "
            "actor-critic underperformed the probe-guided planner at this scale "
            "(see FINDINGS.md), so the planner is the reported controller"}
        print("  [control] actor-critic not deployed (planner is the controller; "
              "see FINDINGS.md)")
    return out


# ------------------------------------------------------------- latent-action align
def latent_action_alignment(data, device):
    """Report the LAM's code<->true-action alignment, which `train_lam` computes on
    its own held-out actions and saves to lam_metrics.json (the 3-frame LAM's frame
    logic stays in lam.py, so eval is decoupled from it)."""
    import json
    p = os.path.join(CFG.artifacts_dir, "lam_metrics.json")
    if not os.path.exists(p):
        print("  [latent-action] SKIPPED (run `wm2.train lam` first)")
        return {"error": "lam_metrics.json missing"}
    with open(p) as f:
        m = json.load(f)
    res = {"purity": round(float(m["purity"]), 3), "nmi": round(float(m["nmi"]), 3),
           "n_codes_used": int(m["codes_used"]), "n_codes": int(m["n_codes"])}
    print(f"  [latent-action] purity={res['purity']}  NMI={res['nmi']}  "
          f"codes_used={res['n_codes_used']}/{res['n_codes']}")
    return res


# ----------------------------------------------------------------- composite
def worldscore_composite(control, faith, drift, lam):
    """A small WorldScore-style 0-1 triple (controllability, consistency, fidelity).

    WorldScore's *controllability* axis measures whether the world RESPONDS to
    control inputs -- which the latent-action model captures directly (does the
    model recover the action structure from frames?) -- so we use the LAM's
    code<->action alignment (purity) for it, and report RL goal-reaching success
    separately under `controllability` in the results. consistency = low dream
    drift; fidelity = how well the latent encodes true state (position R^2)."""
    succ = [v.get("success", 0.0) for v in control.values()
            if isinstance(v, dict) and "success" in v and v is not control.get("random")]
    best = max(succ) if succ else 0.0
    rand = control.get("random", {}).get("success", 0.0)
    controllability_s = float(np.clip((best - rand) / max(1e-6, 1 - rand), 0, 1))
    drift_norm = drift.get("mse_h60", 9999) / 2000.0
    consistency_s = float(np.clip(1 - drift_norm, 0, 1))
    fidelity_s = float(np.clip(faith.get("agent_pos_R2_mean", 0), 0, 1))
    triple = {"controllability_goal_reach": round(controllability_s, 3),
              "consistency_drift": round(consistency_s, 3),
              "fidelity_state_R2": round(fidelity_s, 3),
              "lam_action_purity": round(float(lam.get("purity", 0.0)), 3)
              if isinstance(lam, dict) else 0.0}
    triple["mean"] = round(float(np.mean([controllability_s, consistency_s, fidelity_s])), 3)
    return triple


# ----------------------------------------------------------------------- main
def main(cfg=CFG):
    device = get_device()
    cfg.ensure_dirs()
    t0 = time.time()
    wm = WorldModel.load(device)
    data = dict(np.load(os.path.join(cfg.data_dir, "rollouts.npz")))  # materialize once
    print("== wm2 evaluation ==")
    faith = simulator_faithfulness(wm, data, device)
    drift = dream_drift(wm, data, device)
    control = controllability(wm, data, device)
    lam = latent_action_alignment(data, device)
    composite = worldscore_composite(control, faith, drift, lam)
    print(f"  [WorldScore-style] {composite}")

    results = {"faithfulness": faith, "drift": drift, "controllability": control,
               "latent_action": lam, "worldscore_composite": composite,
               "baseline_reference": {
                   "note": "2018 V-M-C (wm/), published single-ball numbers",
                   "pos_probe_R2": 0.97, "vel_probe_R2": None,
                   "controller_success": 0.39, "random_success": 0.05},
               "seconds": round(time.time() - t0, 1)}
    out = os.path.join(cfg.artifacts_dir, "eval_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out}  ({results['seconds']}s)")
    return results


if __name__ == "__main__":
    main()
