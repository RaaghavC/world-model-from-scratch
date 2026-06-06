"""Collect rollouts from MultiBody2DEnv for world-model training.

Saves whole episodes (so the RSSM can slice BPTT subsequences) plus ground-truth
positions/velocities/kinetic-energy for probes and physics-consistency checks.
Mirrors `wm/collect.py` but for the richer multi-body world.
"""
from __future__ import annotations

import os
import time

import numpy as np

from wm2.config import CFG
from wm2.env import MultiBody2DEnv, RepeatRandomPolicy, rollout


def main(cfg=CFG, out_name="rollouts.npz"):
    cfg.ensure_dirs()
    t0 = time.time()
    E, T, N = cfg.n_episodes, cfg.ep_len, cfg.n_bodies
    obs = np.zeros((E, T + 1, cfg.img_size, cfg.img_size, 3), np.uint8)
    act = np.zeros((E, T), np.int64)
    rew = np.zeros((E, T), np.float32)
    cont = np.ones((E, T), np.float32)        # 1 = episode continues (no early term)
    pos = np.zeros((E, T + 1, N, 2), np.float32)
    vel = np.zeros((E, T + 1, N, 2), np.float32)
    ke = np.zeros((E, T + 1), np.float32)
    goals = np.zeros((E, 2), np.float32)

    for e in range(E):
        env = MultiBody2DEnv(cfg, seed=cfg.seed + e)
        pol = RepeatRandomPolicy(cfg, seed=cfg.seed + 10_000 + e)
        o = env.reset()
        obs[e, 0] = o
        pos[e, 0] = env.pos
        vel[e, 0] = env.vel
        ke[e, 0] = env.kinetic_energy()
        goals[e] = env.goal
        for t in range(T):
            a = pol(o)
            o, r, done, info = env.step(a)
            obs[e, t + 1] = o
            act[e, t] = a
            rew[e, t] = r
            cont[e, t] = 0.0 if done and t < T - 1 else 1.0
            pos[e, t + 1] = env.pos
            vel[e, t + 1] = env.vel
            ke[e, t + 1] = info["ke"]
        if (e + 1) % 50 == 0:
            print(f"  collected {e+1}/{E} episodes  ({time.time()-t0:.1f}s)")

    out = os.path.join(cfg.data_dir, out_name)
    np.savez_compressed(out, obs=obs, act=act, rew=rew, cont=cont,
                        pos=pos, vel=vel, ke=ke, goals=goals)
    mb = os.path.getsize(out) / 1e6
    print(f"saved {out}  ({E} eps x {T} steps, N={N})  {mb:.1f} MB  "
          f"in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
