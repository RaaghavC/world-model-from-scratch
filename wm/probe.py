"""A small latent->position probe.

Trains an MLP mapping the VAE latent z -> the ball's (x, y) position. It serves
two purposes:

  1. Interpretability: a high R^2 shows the ball position is (nearly) decodable
     from the latent, i.e. the VAE genuinely discovered the world's underlying
     state variable -- it was never given position as a label.
  2. Reward inside the dream: the controller is trained purely on imagined
     latents, so we need a way to score them. probe(z) gives the ball position
     and the goal is known, hence reward = -distance.

The probe is intentionally tiny (one hidden layer); if such a small network
recovers position well, the information must be laid out cleanly in the latent.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from wm.config import CFG, get_device


class PosProbe(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64), nn.ReLU(True), nn.Linear(64, 2)
        )

    def forward(self, z):
        return self.net(z)

    @classmethod
    def load(cls, device=None):
        device = device or get_device()
        c = torch.load(os.path.join(CFG.ckpt_dir, "probe.pt"), map_location=device)
        m = cls(c["z_dim"]).to(device)
        m.load_state_dict(c["model"]); m.eval()
        return m


def main():
    device = get_device()
    # Latents (posterior means) cached by train_vae.py, paired with the true
    # positions logged during data collection. Both are flattened over (episode, t).
    lat = np.load(os.path.join(CFG.data_dir, "latents.npz"))
    mu = torch.from_numpy(lat["mu"].reshape(-1, CFG.z_dim)).to(device)
    pos = torch.from_numpy(np.load(os.path.join(CFG.data_dir, "pos.npy")).reshape(-1, 2)).to(device)
    n = mu.shape[0]
    ntr = int(0.9 * n)                       # 90/10 train/test split
    perm = torch.randperm(n)
    tr, te = perm[:ntr], perm[ntr:]

    probe = PosProbe(CFG.z_dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    bs = 1024
    for ep in range(30):
        probe.train()
        p = tr[torch.randperm(len(tr))]
        for i in range(0, len(p), bs):
            idx = p[i : i + bs]
            opt.zero_grad()
            loss = lossf(probe(mu[idx]), pos[idx])
            loss.backward(); opt.step()
    # Report held-out R^2 and the equivalent error in pixels.
    probe.eval()
    with torch.no_grad():
        pred = probe(mu[te])
        mse = lossf(pred, pos[te]).item()
        var = pos[te].var(0).mean().item()          # total variance of positions
        r2 = 1 - mse / var                           # coefficient of determination
        px_err = (mse ** 0.5) * CFG.img_size         # RMS error scaled to pixels
    torch.save({"model": probe.state_dict(), "z_dim": CFG.z_dim}, os.path.join(CFG.ckpt_dir, "probe.pt"))
    print(f"probe test MSE={mse:.5f}  R2={r2:.4f}  ~{px_err:.2f}px error  -> saved checkpoints/probe.pt")


if __name__ == "__main__":
    main()
