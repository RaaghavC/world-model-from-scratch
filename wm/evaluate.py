"""Deploy the dream-trained controller in the REAL environment.

The payoff experiment. The controller was optimised entirely inside the world
model's imagination; here it drives the real (NumPy) environment. At each real
step we encode the frame with the VAE (-> z), pick a greedy action from [z, h],
and advance the MDN-RNN to keep the hidden state h in sync. We measure how often
it reaches the goal vs a random-action baseline on identical start/goal pairs.

Outputs:
  artifacts/controller_real.gif / _montage.png   (real episodes, goals reached)
"""
from __future__ import annotations

import argparse
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from wm.config import CFG, get_device
from wm.controller import LinearController
from wm.dream import WorldModel
from wm.env import Particle2DEnv
from wm.mdnrnn import step_params


@torch.no_grad()
def run_episode(wm, controller, pos, goal, store=False, rng=None):
    """One real-env episode. controller=None -> random baseline."""
    env = Particle2DEnv(CFG)
    obs = env.reset(pos=pos, goal=goal)
    H = wm.mdn.hidden
    hc = None
    h_state = torch.zeros(1, H, device=wm.device)
    min_d, reached = np.hypot(*(env.pos - env.goal)), False
    frames = [obs] if store else None
    for _ in range(CFG.ep_len):
        # V encodes the REAL frame each step -> accurate perception (no drift here).
        z = wm.encode(obs)                                  # (1,z)
        if controller is None:
            a = int(rng.integers(0, CFG.n_actions))         # random baseline
        else:
            a = controller.act(z, h_state)                  # C acts from [z, h]
        obs, _, done, info = env.step(a)
        if store:
            frames.append(obs)
        # Advance M with the real (z, a) purely to keep its hidden state h in sync;
        # we discard M's predicted next latent because the next z comes from the
        # real environment. This is the standard V->C act, M-tracks-h deployment.
        a_oh = F.one_hot(torch.tensor([a], device=wm.device), CFG.n_actions).float()
        (_, _, _), hc = step_params(wm.mdn, z, a_oh, hc)
        h_state = hc[0][-1]
        min_d = min(min_d, info["dist"])                    # closest approach this episode
        reached = reached or info["reached"]
        if done:
            break
    return min_d, reached, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    args = ap.parse_args()
    device = get_device()
    wm = WorldModel.load(device)
    ctrl = LinearController(device)
    rng = np.random.default_rng(777)

    # matched scenarios for controller vs random
    scen = []
    r = CFG.agent_radius
    for _ in range(args.episodes):
        pos = rng.uniform(r, 1 - r, size=2)
        goal = rng.uniform(0.18, 0.82, size=2)
        while np.hypot(*(pos - goal)) < 0.3:
            pos = rng.uniform(r, 1 - r, size=2)
        scen.append((pos, goal))

    c_reach = c_dist = b_reach = b_dist = 0.0
    for pos, goal in scen:
        md, rc, _ = run_episode(wm, ctrl, pos, goal)
        c_reach += rc; c_dist += md
        brng = np.random.default_rng(int(pos[0] * 1e6))
        md_b, rc_b, _ = run_episode(wm, None, pos, goal, rng=brng)
        b_reach += rc_b; b_dist += md_b
    n = len(scen)
    print("=" * 58)
    print(f"  REAL-ENV EVALUATION over {n} matched start/goal pairs")
    print("=" * 58)
    print(f"  dream-trained controller : reached {c_reach/n*100:5.1f}%   mean min-dist {c_dist/n:.3f}")
    print(f"  random baseline          : reached {b_reach/n*100:5.1f}%   mean min-dist {b_dist/n:.3f}")
    print("=" * 58)

    # build a 2x2 GIF of four reached episodes
    demos, tries = [], 0
    drng = np.random.default_rng(42)
    while len(demos) < 4 and tries < 60:
        tries += 1
        pos = drng.uniform(r, 1 - r, size=2)
        goal = drng.uniform(0.2, 0.8, size=2)
        if np.hypot(*(pos - goal)) < 0.35:
            continue
        md, rc, frames = run_episode(wm, ctrl, pos, goal, store=True)
        if rc:
            demos.append(frames)
    if demos:
        L = max(len(f) for f in demos)
        demos = [f + [f[-1]] * (L - len(f)) for f in demos]      # pad to equal length
        S = CFG.img_size
        hsep = np.full((S, 2, 3), 255, np.uint8)
        vsep = np.full((2, 2 * S + 2, 3), 255, np.uint8)
        grid_frames = []
        for t in range(L):
            top = np.hstack([demos[0][t], hsep, demos[1][t]])
            bot = np.hstack([demos[2][t], hsep, demos[3][t]])
            grid_frames.append(np.vstack([top, vsep, bot]))
        gif = os.path.join(CFG.artifacts_dir, "controller_real.gif")
        imageio.mimsave(gif, grid_frames, fps=20, loop=0)
        imageio.imwrite(os.path.join(CFG.artifacts_dir, "controller_real_montage.png"),
                        np.vstack([grid_frames[0], grid_frames[L // 2], grid_frames[-1]]))
        print(f"saved {gif} ({len(demos)} reached episodes)")


if __name__ == "__main__":
    main()
