"""C -- a controller trained ENTIRELY inside the world model's dream.

The controller is a tiny LINEAR policy mapping the world-model state [z, h]
(VAE latent + MDN-RNN hidden state) to one of the 5 discrete actions. It never
touches the real environment during training: every candidate is scored on
imagined latent rollouts, with reward derived from the latent->position probe.
This reproduces the central result of Ha & Schmidhuber (2018) -- "learning inside
a dream" -- using a from-scratch Cross-Entropy Method (CEM), so there is no
external CMA-ES dependency.

Two design choices fight "model exploitation" (the policy learning to abuse the
world model's inaccuracies instead of solving the task):
  * scenarios are RESAMPLED every generation (domain randomization), and
  * the saved controller is chosen by VALIDATION fitness on a fixed held-out set.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")              # headless backend (no display needed)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from wm.config import CFG, get_device
from wm.dream import WorldModel
from wm.env import Particle2DEnv
from wm.mdnrnn import mdn_sample, step_params
from wm.probe import PosProbe


def theta_dim(z_dim, hidden, n_actions):
    """Flat parameter count for the linear policy and its input width.

    The policy is logits = W @ [z, h] + b, with W:(out,in) and b:(out), so the
    flattened parameter vector theta has length n_actions*in_dim + n_actions.
    """
    in_dim = z_dim + hidden
    return n_actions * in_dim + n_actions, in_dim


def split_theta(theta, in_dim, n_actions):
    """Unpack a BATCH of flat parameter vectors into weight/bias tensors.

    theta:(P,D) -> W:(P,in,out), b:(P,out). W is stored transposed to (in,out) so
    fitness can use a single batched matmul (baddbmm) across all P candidates.
    """
    P = theta.shape[0]
    W = theta[:, : n_actions * in_dim].reshape(P, n_actions, in_dim).transpose(1, 2)  # (P,in,out)
    b = theta[:, n_actions * in_dim :]                                                # (P,out)
    return W, b


def sample_scenarios(wm: WorldModel, R: int, seed: int):
    """Build R dream scenarios. Returns z0:(R,z), goals:(R,2).

    Each scenario is one real env reset (random start far enough from a random
    goal); we keep only frame 0, encode it to z0, and remember the goal. The
    dream then unrolls from z0 -- crucially, z0 already encodes the goal's
    position, so the simulator keeps "remembering" where to go.
    """
    rng = np.random.default_rng(seed)
    obs0, goals = [], []
    r = CFG.agent_radius
    for _ in range(R):
        env = Particle2DEnv(CFG)
        pos = rng.uniform(r, 1 - r, size=2)
        goal = rng.uniform(0.18, 0.82, size=2)
        while np.hypot(*(pos - goal)) < 0.3:           # ensure a non-trivial start
            pos = rng.uniform(r, 1 - r, size=2)
        obs0.append(env.reset(pos=pos, goal=goal))
        goals.append(goal)
    z0 = wm.encode(np.stack(obs0))                     # (R,z)
    goals = torch.tensor(np.stack(goals), dtype=torch.float32, device=wm.device)
    return z0, goals


@torch.no_grad()
def dream_fitness(wm, probe, theta, z0, goals, horizon, temperature=0.0):
    """Score P candidate policies on R shared dream scenarios. Returns fitness:(P,).

    All P*R rollouts run in parallel through the MDN-RNN as one big batch. At each
    step every candidate picks a greedy action from its own [z, h]; the model
    advances the latent; the probe reads off the ball position; reward is shaped
    to both reduce distance AND reward sitting inside the goal.

    temperature>0 samples stochastic futures (robustness during training); 0 uses
    the deterministic most-likely rollout (clean validation / matches the env).
    """
    use_mode = temperature <= 0.0
    device = wm.device
    P, R, Z = theta.shape[0], z0.shape[0], z0.shape[1]
    H = wm.mdn.hidden
    n_actions = wm.n_actions
    in_dim = Z + H
    W, b = split_theta(theta, in_dim, n_actions)        # (P,in,out),(P,out)

    # Replicate each scenario across all candidates -> flat batch of size P*R,
    # laid out as [candidate-major]: index p*R + r.
    z = z0.unsqueeze(0).expand(P, R, Z).reshape(P * R, Z).contiguous()
    g = goals.unsqueeze(0).expand(P, R, 2).reshape(P * R, 2)
    h_state = torch.zeros(P * R, H, device=device)      # controller's view of h (pre-step)
    hc = None                                           # LSTM (h,c); None -> zeros
    score = torch.zeros(P * R, device=device)

    for _ in range(horizon):
        # Greedy action per rollout. baddbmm computes, per candidate p:
        #   logits[p] = x[p] @ W[p] + b[p]   over its R scenarios at once.
        x = torch.cat([z, h_state], dim=-1).reshape(P, R, in_dim)
        logits = torch.baddbmm(b.unsqueeze(1), x, W)    # (P,R,out)
        a = logits.argmax(-1).reshape(P * R)
        a_oh = F.one_hot(a, n_actions).float()
        # Advance the simulator one step; refresh the controller's hidden view.
        (logpi, mu, logsig), hc = step_params(wm.mdn, z, a_oh, hc)
        h_state = hc[0][-1]                             # (P*R, H) new hidden state
        z = mdn_sample(logpi, mu, logsig, temperature=max(temperature, 1e-6), mode=use_mode)
        d = torch.linalg.norm(probe(z) - g, dim=-1)     # predicted distance to goal
        # Shaped reward: push distance down, with a strong Gaussian bonus for being
        # within ~one goal-radius so the policy learns to arrive AND settle.
        score += -d + 2.0 * torch.exp(-((d / 0.07) ** 2))
    return (score / horizon).reshape(P, R).mean(1)      # mean over scenarios -> (P,)


def cem_train(epochs=None, seed=0):
    """Evolve the controller with CEM, selecting by validation-in-imagination."""
    device = get_device()
    wm = WorldModel.load(device)
    probe = PosProbe.load(device)
    D, in_dim = theta_dim(wm.mdn.z_dim, wm.mdn.hidden, wm.n_actions)
    P = CFG.ctrl_pop
    n_elite = max(4, P // 8)                            # keep the top ~1/8 each gen
    gens = epochs or CFG.ctrl_generations

    # CEM maintains a diagonal Gaussian over parameter space (mean, per-dim std).
    mean = torch.zeros(D, device=device)
    std = torch.full((D,), CFG.ctrl_sigma, device=device)
    history = []
    best_theta, best_val = mean.clone(), -1e9

    # Fixed held-out scenarios used ONLY to pick which generation's mean to keep.
    val_z0, val_goals = sample_scenarios(wm, 24, seed=999)

    for gen in range(gens):
        # Resample training scenarios each generation + (optionally) stochastic
        # dreams -> the policy cannot overfit one fixed set of imagined rollouts.
        z0, goals = sample_scenarios(wm, CFG.ctrl_rollouts, seed=1234 + gen)
        eps = torch.randn(P, D, device=device)
        thetas = mean.unsqueeze(0) + std.unsqueeze(0) * eps
        thetas[0] = mean                                # always evaluate the current mean
        fit = dream_fitness(wm, probe, thetas, z0, goals, CFG.ctrl_horizon, temperature=CFG.ctrl_temp)
        # CEM update: refit the Gaussian to the elite (best) candidates.
        elite_idx = torch.topk(fit, n_elite).indices
        elites = thetas[elite_idx]
        mean = elites.mean(0)
        std = elites.std(0) + 1e-3                      # +eps floor keeps exploring
        # Validate the new mean deterministically on the fixed held-out set.
        val = dream_fitness(wm, probe, mean.unsqueeze(0), val_z0, val_goals, CFG.ctrl_horizon, 0.0)[0].item()
        history.append(val)
        if val > best_val:
            best_val, best_theta = val, mean.clone()
        if (gen + 1) % 5 == 0 or gen == 0:
            print(f"gen {gen+1}/{gens}  train_fit={fit.mean().item():+.4f}  val_fit={val:+.4f}  best_val={best_val:+.4f}")

    torch.save(
        {"theta": best_theta.cpu(), "z_dim": wm.mdn.z_dim, "hidden": wm.mdn.hidden,
         "n_actions": wm.n_actions, "in_dim": in_dim, "history": history},
        os.path.join(CFG.ckpt_dir, "controller.pt"),
    )
    plt.figure(figsize=(6, 3.2))
    plt.plot(history)
    plt.xlabel("CEM generation"); plt.ylabel("validation dream fitness")
    plt.title("Controller learning to reach the goal — inside the dream")
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.artifacts_dir, "controller_curve.png"), dpi=120)
    print(f"saved checkpoints/controller.pt  (best validation fitness={best_val:+.4f})")
    print("saved artifacts/controller_curve.png")


class LinearController:
    """Deployment wrapper: load theta and pick a greedy action from [z, h]."""

    def __init__(self, device=None):
        device = device or get_device()
        c = torch.load(os.path.join(CFG.ckpt_dir, "controller.pt"), map_location=device)
        self.device = device
        self.n_actions = c["n_actions"]
        self.in_dim = c["in_dim"]
        theta = c["theta"].to(device)
        self.W = theta[: self.n_actions * self.in_dim].reshape(self.n_actions, self.in_dim)
        self.b = theta[self.n_actions * self.in_dim :]

    @torch.no_grad()
    def act(self, z, h):
        """z:(1,z), h:(1,hidden) -> int action = argmax(W[z,h] + b)."""
        x = torch.cat([z.reshape(-1), h.reshape(-1)])
        return int((self.W @ x + self.b).argmax().item())


if __name__ == "__main__":
    cem_train()
