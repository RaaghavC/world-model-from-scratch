"""Collect a dataset of rollouts under a repeat-random policy.

Saves:
  data/obs.npy    (N, T+1, S, S, 3) uint8   -- frames
  data/act.npy    (N, T)             int8    -- actions taken
  data/goals.npy  (N, 2)             float32 -- goal position per episode

Random exploration is enough to teach the dynamics model how actions move the
ball and how it bounces; the controller later *exploits* that learned model.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from wm.config import CFG
from wm.env import Particle2DEnv, RepeatRandomPolicy, rollout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=CFG.n_episodes)
    ap.add_argument("--ep_len", type=int, default=CFG.ep_len)
    args = ap.parse_args()

    CFG.ensure_dirs()
    obs_all, act_all, pos_all, goals = [], [], [], []
    t0 = time.time()
    for ep in range(args.episodes):
        env = Particle2DEnv(CFG, seed=1000 + ep)
        pol = RepeatRandomPolicy(CFG, seed=2000 + ep)
        obs, acts, pos = rollout(env, pol, ep_len=args.ep_len)
        obs_all.append(obs)
        act_all.append(acts.astype(np.int8))
        pos_all.append(pos)
        goals.append(env.goal.astype(np.float32))
        if (ep + 1) % 40 == 0:
            print(f"  episode {ep+1}/{args.episodes}  ({time.time()-t0:.1f}s)")

    obs_all = np.stack(obs_all)            # (N, T+1, S, S, 3) uint8
    act_all = np.stack(act_all)            # (N, T) int8
    pos_all = np.stack(pos_all)            # (N, T+1, 2) float32
    goals = np.stack(goals)                # (N, 2)
    np.save(os.path.join(CFG.data_dir, "obs.npy"), obs_all)
    np.save(os.path.join(CFG.data_dir, "act.npy"), act_all)
    np.save(os.path.join(CFG.data_dir, "pos.npy"), pos_all)
    np.save(os.path.join(CFG.data_dir, "goals.npy"), goals)
    mb = obs_all.nbytes / 1e6
    print(f"saved obs {obs_all.shape} ({mb:.0f} MB), act {act_all.shape} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
