"""Extra visualizations for the writeup.

  latent_space()       -- PCA of VAE latents, colored by true ball position
                          (shows the latent self-organized by world state)
  free_running_dream() -- a long open-loop hallucination: encode ONE frame, then
                          drive the MDN-RNN with a scripted action pattern and
                          decode. No ground truth -- the world model is the world.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from wm.config import CFG, get_device


def latent_space(n: int = 5000, seed: int = 0):
    """Probe-calibration figure: predicted vs true position from the VAE latent.
    A tight diagonal == the VAE discovered the world's state (ball position)."""
    import torch
    from wm.probe import PosProbe

    device = get_device()
    lat = np.load(os.path.join(CFG.data_dir, "latents.npz"))["mu"].reshape(-1, CFG.z_dim)
    pos = np.load(os.path.join(CFG.data_dir, "pos.npy")).reshape(-1, 2)
    rng = np.random.default_rng(seed)
    idx = rng.choice(lat.shape[0], size=min(n, lat.shape[0]), replace=False)
    P = pos[idx]
    probe = PosProbe.load(device)
    with torch.no_grad():
        pred = probe(torch.from_numpy(lat[idx]).to(device)).cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(9, 4.4))
    for k, (a, name) in enumerate(zip(ax, ["x", "y"])):
        r2 = 1 - ((pred[:, k] - P[:, k]) ** 2).sum() / ((P[:, k] - P[:, k].mean()) ** 2).sum()
        a.scatter(P[:, k], pred[:, k], s=4, alpha=0.25, c="#2b6cb0")
        a.plot([0, 1], [0, 1], "k--", lw=1)
        a.set_xlim(0, 1); a.set_ylim(0, 1)
        a.set_xlabel(f"true ball {name}"); a.set_ylabel(f"decoded ball {name}")
        a.set_title(f"{name}:  R² = {r2:.3f}")
    fig.suptitle("Ball position is linearly decodable from the VAE latent\n(the renderer also discovered the simulator's state)")
    fig.tight_layout()
    out = os.path.join(CFG.artifacts_dir, "latent_space.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


@torch.no_grad()
def free_running_dream(steps: int = 160, temperature: float = 0.5, seed: int = 123):
    from wm.dream import WorldModel
    from wm.env import Particle2DEnv
    import imageio.v2 as imageio

    wm = WorldModel.load(get_device())
    env = Particle2DEnv(CFG, seed=seed)
    z0 = wm.encode(env.reset())
    pattern = []
    for a in [1, 3, 2, 4]:                  # right, down, left, up
        pattern += [a] * 20
    acts = (pattern * (steps // len(pattern) + 1))[:steps]
    a = torch.tensor(acts, device=wm.device).unsqueeze(0)
    z_seq, _ = wm.dream_rollout(z0, a, temperature=temperature, mode=False)
    frames = wm.decode(z_seq[0])
    gif = os.path.join(CFG.artifacts_dir, "free_dream.gif")
    imageio.mimsave(gif, list(frames), fps=20, loop=0)
    sel = list(range(0, len(frames), max(1, len(frames) // 8)))[:8]
    imageio.imwrite(os.path.join(CFG.artifacts_dir, "free_dream_montage.png"),
                    np.hstack([frames[i] for i in sel]))
    print(f"saved {gif}  ({steps} fully-hallucinated frames from 1 real seed frame)")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("latent", "both"):
        latent_space()
    if which in ("dream", "both"):
        free_running_dream()
