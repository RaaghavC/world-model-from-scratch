"""Training orchestrator for wm2 (single CLI entry point).

  python -m wm2.train collect   # gather MultiBody2DEnv rollouts
  python -m wm2.train wm        # train the RSSM world model (V+M)
  python -m wm2.train ac        # train the actor-critic in imagination (C)
  python -m wm2.train lam       # train the latent action model (Genie)
  python -m wm2.train jepa      # train the decoder-free JEPA track
  python -m wm2.train eval      # head-to-head evaluation vs the wm/ baseline
  python -m wm2.train viz       # dreams / drift / latent maps / code-action map
  python -m wm2.train all       # collect -> wm -> ac -> lam -> jepa -> eval -> viz

The world model is the V+M fusion; per-frame reward = -dist(agent, goal) is taught
to the learned reward head (no probe). Latent history augmentation (cfg.wm_hist_aug)
fights open-loop drift. `lam`/`jepa`/`eval`/`viz` live in modules built against the
verified rssm.py + agent.py interfaces and are imported lazily.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from wm2.config import CFG, get_device
from wm2.rssm import WorldModel
from wm2.utils import obs_to_tensor


# ----------------------------------------------------------------- data batching
class SeqLoader:
    """Samples length-L sub-sequences from the collected episodes, with per-frame
    reward = -||pos - goal|| (a function of state, exactly what the reward head
    should learn so imagination reward matches the real task)."""

    def __init__(self, cfg=CFG, device=None):
        self.cfg = cfg
        self.device = device or get_device()
        d = np.load(os.path.join(cfg.data_dir, "rollouts.npz"))
        self.obs = d["obs"]                       # (E,T+1,H,W,3) uint8
        self.act = d["act"]                       # (E,T) int64
        self.cont = d["cont"]                     # (E,T) float
        self.pos = d["pos"][:, :, 0, :]           # agent positions (E,T+1,2)
        self.goals = d["goals"]                   # (E,2)
        self.E, self.Tp1 = self.obs.shape[:2]
        self.T = self.act.shape[1]

    def batch(self, B, L, rng):
        eps = rng.integers(0, self.E, size=B)
        t0 = rng.integers(0, self.T - L, size=B)
        obs = np.stack([self.obs[e, s:s + L] for e, s in zip(eps, t0)])
        act = np.stack([self.act[e, s:s + L] for e, s in zip(eps, t0)])
        cont = np.stack([self.cont[e, s:s + L] for e, s in zip(eps, t0)])
        pos = np.stack([self.pos[e, s:s + L] for e, s in zip(eps, t0)])
        goal = self.goals[eps][:, None, :]        # (B,1,2)
        rew = -np.linalg.norm(pos - goal, axis=-1).astype(np.float32)   # (B,L)
        return {
            "obs": obs_to_tensor(obs).to(self.device),
            "act": torch.from_numpy(act).long().to(self.device),
            "rew": torch.from_numpy(rew).float().to(self.device),
            "cont": torch.from_numpy(cont).float().to(self.device),
        }


# ----------------------------------------------------------------- world model
def train_wm(cfg=CFG):
    device = get_device()
    cfg.ensure_dirs()
    loader = SeqLoader(cfg, device)
    wm = WorldModel(cfg).to(device)
    opt = torch.optim.Adam(wm.parameters(), lr=cfg.wm_lr)
    rng = np.random.default_rng(cfg.seed)
    n_params = sum(p.numel() for p in wm.parameters())
    steps = cfg.wm_epochs * cfg.wm_steps_per_epoch
    print(f"train_wm: {n_params/1e6:.2f}M params  device={device}  "
          f"{steps} steps  hist_aug={cfg.wm_hist_aug}")
    t0 = time.time()
    for it in range(steps):
        batch = loader.batch(cfg.wm_batch, cfg.wm_seq_len, rng)
        loss, m, _ = wm.loss(batch, hist_aug=cfg.wm_hist_aug)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(wm.parameters(), cfg.grad_clip)
        opt.step()
        if (it + 1) % cfg.wm_steps_per_epoch == 0 or it == 0:
            ep = (it + 1) // cfg.wm_steps_per_epoch
            print(f"  ep {ep:2d}/{cfg.wm_epochs} step {it+1:5d}  loss={m['loss']:8.1f}"
                  f"  recon={m['recon']:7.1f}  rew={m['reward']:.3f}"
                  f"  dyn_kl={m['dyn_kl']:.2f}  rep_kl={m['rep_kl']:.2f}"
                  f"  ({time.time()-t0:.0f}s)")
    path = os.path.join(cfg.ckpt_dir, "world_model.pt")
    wm.save(path)
    print(f"saved {path}  ({time.time()-t0:.0f}s total)")
    return wm


# ----------------------------------------------------------------- dispatch
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["collect", "wm", "ac", "lam", "jepa",
                                      "eval", "viz", "all"])
    args = ap.parse_args()
    s = args.stage

    if s in ("collect", "all"):
        from wm2.collect import main as collect_main
        collect_main()
    if s in ("wm", "all"):
        train_wm()
    if s in ("ac", "all"):
        from wm2.agent import train_ac
        train_ac()
    if s in ("lam", "all"):
        from wm2.collect import main as collect_main
        from wm2.lam import train_lam
        # Label-free action discovery is only meaningful in the action-driven regime
        # (Genie's setting): collect a DIRECT-CONTROL variant where the action drives
        # the frames, and train the LAM on it. (On the momentum world the action is a
        # tiny residual on predictable velocity and the codes stay at chance.)
        CFG.direct_control = True
        collect_main(CFG, out_name="rollouts_dc.npz")
        CFG.direct_control = False
        train_lam(CFG, data_file="rollouts_dc.npz")
    if s in ("jepa", "all"):
        from wm2.jepa import train_jepa
        train_jepa()
    if s in ("eval", "all"):
        from wm2.evaluate import main as eval_main
        eval_main()
    if s in ("viz", "all"):
        from wm2.viz import main as viz_main
        viz_main()


if __name__ == "__main__":
    main()
