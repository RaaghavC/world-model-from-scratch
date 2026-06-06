"""C -- an actor-critic trained PURELY in latent imagination (replaces CEM+probe).

The baseline evolves a linear policy with CEM, scoring candidates by a hand-built
latent->position probe. wm2 instead learns an actor and a critic entirely on
imagined RSSM rollouts, with reward provided by the world model's OWN learned
reward head -- no probe, no environment interaction during policy optimization.

This is the DreamerV3 control recipe:
  * imagine `H` steps forward from real posterior states, sampling actions from pi;
  * bootstrap lambda-returns (gamma, lambda) with a slow (EMA) target critic;
  * critic: two-hot symexp regression to the lambda-returns;
  * actor: REINFORCE on advantages (return - value) + entropy bonus, with each
    imagined step weighted by its cumulative discount.

Discrete actions => the score-function (REINFORCE) gradient, exactly as DreamerV3.
"""
from __future__ import annotations

import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from wm2.config import CFG, get_device
from wm2.rssm import WorldModel
from wm2.utils import (mlp, obs_to_tensor, symlog_support, symlog_twohot_loss,
                       value_from_logits)


class Actor(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.net = mlp(cfg.feat_dim, cfg.hidden, cfg.n_actions, layers=2)

    def forward(self, feat):
        return self.net(feat)                       # logits (.., n_actions)

    def dist(self, feat):
        return Categorical(logits=self.forward(feat))


class Critic(nn.Module):
    """Two-hot symexp value head (DreamerV3 distributional critic)."""

    def __init__(self, cfg=CFG):
        super().__init__()
        self.net = mlp(cfg.feat_dim, cfg.hidden, cfg.value_buckets, layers=2)
        self.register_buffer(
            "support", symlog_support(cfg.value_vmin, cfg.value_vmax,
                                      cfg.value_buckets, "cpu"))

    def logits(self, feat):
        return self.net(feat)

    def value(self, feat):
        return value_from_logits(self.net(feat), self.support)


def lambda_return(reward, value, disc, lam):
    """Backward lambda-returns. reward/value/disc: (T,M). disc[t]=gamma*cont[t].

    Returns targets for states 0..T-2, shape (T-1,M). State T-1 bootstraps with
    value[T-1]:  G[t] = r[t+1] + disc[t+1]*((1-lam)*v[t+1] + lam*G[t+1])."""
    T = reward.shape[0]
    ret = [None] * T
    ret[T - 1] = value[T - 1]
    for t in range(T - 2, -1, -1):
        ret[t] = reward[t + 1] + disc[t + 1] * (
            (1 - lam) * value[t + 1] + lam * ret[t + 1])
    return torch.stack(ret[:-1], 0)


class ImagAC:
    """Bundles actor, critic, slow target critic + the imagination training loop."""

    def __init__(self, wm: WorldModel, cfg=CFG, device=None):
        self.cfg = cfg
        self.device = device or get_device()
        self.wm = wm.to(self.device).eval()
        for p in self.wm.parameters():                # world model is frozen here
            p.requires_grad_(False)
        self.actor = Actor(cfg).to(self.device)
        self.critic = Critic(cfg).to(self.device)
        self.slow = copy.deepcopy(self.critic).eval()
        for p in self.slow.parameters():
            p.requires_grad_(False)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    # ---- imagination ------------------------------------------------------
    def imagine(self, start):
        """Roll the actor through the frozen world model for H steps (no grad).

        start: state dict with deter(M,*), stoch(M,*). Returns feats(T,M,feat),
        actions(H,M), rewards(T,M), conts(T,M) with T=H+1."""
        H = self.cfg.imag_horizon
        state = {"deter": start["deter"].detach(), "stoch": start["stoch"].detach()}
        feats, actions = [], []
        with torch.no_grad():
            for _ in range(H):
                feat = self.wm.get_feat(state)
                a = Categorical(logits=self.actor(feat)).sample()
                feats.append(feat)
                actions.append(a)
                a_oh = F.one_hot(a, self.cfg.n_actions).float()
                state, _ = self.wm.img_step(state, a_oh, sample=True)
            feats.append(self.wm.get_feat(state))        # final feat -> T=H+1
            feats = torch.stack(feats, 0)                # (T,M,feat)
            rewards = self.wm.reward(feats)              # (T,M)
            conts = self.wm.cont(feats)                 # (T,M)
        return feats, torch.stack(actions, 0), rewards, conts

    def train_step(self, start):
        cfg = self.cfg
        feats, actions, rewards, conts = self.imagine(start)
        disc = cfg.gamma * conts                         # (T,M)
        with torch.no_grad():
            values_slow = self.slow.value(feats)         # (T,M) stable bootstrap
            returns = lambda_return(rewards, values_slow, disc, cfg.lambda_)  # (T-1,M)
            # Per-step weight = cumulative discount of REACHING state t (w_0=1).
            w = torch.cumprod(
                torch.cat([torch.ones_like(disc[:1]), disc[:-1]], 0), 0)  # (T,M)
            w = w[:-1].detach()                          # states 0..T-2
        feat_d = feats[:-1].detach()                     # (T-1,M,feat)

        # Critic: two-hot regression to sg(returns), discount-weighted.
        c_logits = self.critic.logits(feat_d)
        # per-element CE then weight (symlog_twohot_loss means over all; do manual)
        target = self._twohot(returns.detach())
        logp = F.log_softmax(c_logits, -1)
        critic_ce = -(target * logp).sum(-1)             # (T-1,M)
        critic_loss = (w * critic_ce).mean()
        self.opt_critic.zero_grad(); critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
        self.opt_critic.step()

        # Actor: REINFORCE on standardized advantages + entropy bonus.
        dist = Categorical(logits=self.actor(feat_d))
        logp_a = dist.log_prob(actions)                  # (T-1,M)
        ent = dist.entropy()                             # (T-1,M)
        with torch.no_grad():
            baseline = self.critic.value(feat_d)
            adv = returns - baseline
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        actor_loss = -(w * (logp_a * adv + cfg.actor_ent * ent)).mean()
        self.opt_actor.zero_grad(); actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
        self.opt_actor.step()

        # EMA slow critic.
        with torch.no_grad():
            for s, c in zip(self.slow.parameters(), self.critic.parameters()):
                s.mul_(1 - cfg.slow_critic_tau).add_(cfg.slow_critic_tau * c)
        return {"actor": actor_loss.item(), "critic": critic_loss.item(),
                "ent": ent.mean().item(), "ret": returns.mean().item(),
                "imag_rew": rewards.mean().item()}

    def _twohot(self, x):
        from wm2.utils import symlog, two_hot
        return two_hot(symlog(x), self.critic.support)

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()}, path)


# ------------------------------------------------------- starting-state sampling
@torch.no_grad()
def sample_starts(wm, data, n_seq, ctx_len, device, rng):
    """Encode `n_seq` random length-`ctx_len` windows to posterior states, flatten
    to (n_seq*ctx_len) imagination start states. (Used by the smoke test; training
    uses a precomputed bank -- see build_state_bank.)"""
    E, T = data["obs"].shape[:2]
    eps = rng.integers(0, E, size=n_seq)
    t0 = rng.integers(0, T - ctx_len, size=n_seq)
    obs = np.stack([data["obs"][e, s:s + ctx_len] for e, s in zip(eps, t0)])
    act = np.stack([data["act"][e, s:s + ctx_len] for e, s in zip(eps, t0)])
    obs = obs_to_tensor(obs).to(device)
    act = torch.from_numpy(act).long().to(device)
    out = wm.observe(obs, act)
    deter = out["deter"].reshape(-1, wm.deter_dim)
    stoch = out["stoch"].reshape(-1, wm.cat, wm.classes)
    return {"deter": deter, "stoch": stoch}


@torch.no_grad()
def build_state_bank(wm, data, device, n_eps=200, chunk=20):
    """Encode many real episodes ONCE into a bank of posterior states (kept on CPU)
    that imagination samples from each step -- removes the encoder from the hot
    loop, the key to fast actor-critic training (DreamerV3 imagines from replay
    states)."""
    obs_all, act_all = data["obs"], data["act"]
    E = min(n_eps, obs_all.shape[0])
    L = act_all.shape[1]
    deters, stochs = [], []
    for s in range(0, E, chunk):
        e = slice(s, min(s + chunk, E))
        obs = obs_to_tensor(obs_all[e, :L]).to(device)
        act = torch.from_numpy(act_all[e, :L]).long().to(device)
        out = wm.observe(obs, act)
        deters.append(out["deter"].reshape(-1, wm.deter_dim).cpu())
        stochs.append(out["stoch"].reshape(-1, wm.cat, wm.classes).cpu())
    return {"deter": torch.cat(deters), "stoch": torch.cat(stochs)}


def train_ac(cfg=CFG, steps=None, log_every=500):
    """Train the imagination actor-critic against a trained world model."""
    device = get_device()
    cfg.ensure_dirs()
    wm = WorldModel.load(device)
    data = np.load(os.path.join(cfg.data_dir, "rollouts.npz"))
    ac = ImagAC(wm, cfg, device)
    rng = np.random.default_rng(0)
    steps = steps or cfg.ac_iters
    print(f"building imagination state bank ...")
    bank = build_state_bank(wm, data, device)
    N = bank["deter"].shape[0]
    print(f"train_ac: {N} start states  {steps} steps  device={device}")
    t0 = time.time()
    for it in range(steps):
        idx = torch.from_numpy(rng.integers(0, N, size=cfg.ac_batch))
        start = {"deter": bank["deter"][idx].to(device),
                 "stoch": bank["stoch"][idx].to(device)}
        m = ac.train_step(start)
        if (it + 1) % log_every == 0 or it == 0:
            print(f"step {it+1:5d}/{steps}  actor={m['actor']:+.3f}  "
                  f"critic={m['critic']:.3f}  ret={m['ret']:+.3f}  "
                  f"imag_rew={m['imag_rew']:+.3f}  ent={m['ent']:.3f}  "
                  f"({time.time()-t0:.0f}s)")
    ac.save(os.path.join(cfg.ckpt_dir, "actor_critic.pt"))
    print(f"saved {os.path.join(cfg.ckpt_dir, 'actor_critic.pt')}")
    return ac


# ----------------------------------------------------------------- deploy policy
class WM2Policy:
    """Deployment wrapper: maintains the RSSM posterior state across REAL steps and
    acts greedily from the actor. Used by evaluate.py to run the dream-trained
    policy in the real environment."""

    def __init__(self, wm: WorldModel, actor: Actor, device=None, greedy=True):
        self.wm = wm.eval()
        self.actor = actor.eval()
        self.device = device or get_device()
        self.greedy = greedy
        self.reset()

    def reset(self, B=1):
        self.state = self.wm.initial(B, self.device)
        self.prev_a = torch.zeros(B, self.wm.n_actions, device=self.device)

    @torch.no_grad()
    def act(self, obs_uint8):
        x = obs_to_tensor(np.asarray(obs_uint8)).to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        embed = self.wm.encoder(x)
        self.state, _, _ = self.wm.obs_step(self.state, self.prev_a, embed)
        feat = self.wm.get_feat(self.state)
        logits = self.actor(feat)
        a = logits.argmax(-1) if self.greedy else Categorical(logits=logits).sample()
        self.prev_a = F.one_hot(a, self.wm.n_actions).float()
        return int(a.item())

    @classmethod
    def load(cls, device=None, greedy=True):
        device = device or get_device()
        wm = WorldModel.load(device)
        actor = Actor(CFG).to(device)
        ck = torch.load(os.path.join(CFG.ckpt_dir, "actor_critic.pt"),
                        map_location=device)
        actor.load_state_dict(ck["actor"]); actor.eval()
        return cls(wm, actor, device, greedy)


# --------------------------------------------------------------------- smoke test
def _smoke():
    """A few AC steps against a FRESH (random) world model: losses finite, params
    move, shapes correct. (Real training uses a trained WM.)"""
    device = get_device()
    cfg = CFG
    data = np.load(os.path.join(cfg.data_dir, "rollouts.npz"))
    wm = WorldModel(cfg).to(device)
    ac = ImagAC(wm, cfg, device)
    rng = np.random.default_rng(0)
    before = ac.actor.net[0].weight.clone()
    for i in range(5):
        start = sample_starts(wm, data, 4, 8, device, rng)   # 32 starts
        m = ac.train_step(start)
        assert np.isfinite(m["actor"]) and np.isfinite(m["critic"]), m
    after = ac.actor.net[0].weight
    assert not torch.allclose(before, after), "actor did not update"
    print(f"agent smoke PASSED  last={m}  device={device}")


if __name__ == "__main__":
    _smoke()
