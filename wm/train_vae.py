"""Train the VAE (V) on all collected frames, then cache latents for the MDN-RNN.

Outputs:
  checkpoints/vae.pt          -- model weights + config
  artifacts/vae_recon.png     -- originals (top) vs reconstructions (bottom)
  data/latents.npz            -- mu, logvar for every frame, shaped (N, T+1, z)
"""
from __future__ import annotations

import argparse
import os
import time

import imageio.v2 as imageio
import numpy as np
import torch

from wm.config import CFG, get_device
from wm.vae import ConvVAE, obs_to_tensor, tensor_to_obs, vae_loss


def load_frames():
    obs = np.load(os.path.join(CFG.data_dir, "obs.npy"))  # (N,T+1,S,S,3)
    N, Tp1, S, _, _ = obs.shape
    return obs, N, Tp1, S


def train(epochs: int, device):
    obs, N, Tp1, S = load_frames()
    frames = obs.reshape(-1, S, S, 3)         # (N*(T+1), S, S, 3) -- a view
    n = frames.shape[0]
    print(f"frames={n}  device={device}")

    model = ConvVAE(CFG.z_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CFG.vae_lr)
    bs = CFG.vae_batch

    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        tot = rl = kl = 0.0
        nb = 0
        t0 = time.time()
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            x = obs_to_tensor(frames[idx]).to(device)
            recon, mu, logvar = model(x)
            loss, r, k = vae_loss(recon, x, mu, logvar, CFG.vae_beta, CFG.vae_fg_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item(); rl += r.item(); kl += k.item(); nb += 1
        print(f"epoch {ep+1}/{epochs}  loss={tot/nb:8.3f}  recon={rl/nb:8.3f}  kl={kl/nb:7.3f}  ({time.time()-t0:.1f}s)")

    os.makedirs(CFG.ckpt_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "z_dim": CFG.z_dim}, os.path.join(CFG.ckpt_dir, "vae.pt"))
    return model, obs, N, Tp1, S


@torch.no_grad()
def save_recon(model, frames_uint8, device, path):
    model.eval()
    x = obs_to_tensor(frames_uint8).to(device)
    recon, _, _ = model(x)
    orig = np.hstack(list(frames_uint8))
    rec = np.hstack(list(tensor_to_obs(recon)))
    montage = np.vstack([orig, rec])
    imageio.imwrite(path, montage)
    print(f"saved {path}")


@torch.no_grad()
def cache_latents(model, obs, N, Tp1, S, device):
    model.eval()
    frames = obs.reshape(-1, S, S, 3)
    n = frames.shape[0]
    mus, lvs = [], []
    bs = 512
    for i in range(0, n, bs):
        x = obs_to_tensor(frames[i : i + bs]).to(device)
        mu, lv = model.encode(x)
        mus.append(mu.cpu().numpy()); lvs.append(lv.cpu().numpy())
    mu = np.concatenate(mus).reshape(N, Tp1, CFG.z_dim).astype(np.float32)
    lv = np.concatenate(lvs).reshape(N, Tp1, CFG.z_dim).astype(np.float32)
    path = os.path.join(CFG.data_dir, "latents.npz")
    np.savez(path, mu=mu, logvar=lv)
    print(f"saved {path}  mu{mu.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=CFG.vae_epochs)
    args = ap.parse_args()
    CFG.ensure_dirs()
    device = get_device()
    model, obs, N, Tp1, S = train(args.epochs, device)
    save_recon(model, obs[0, ::15][:8], device, os.path.join(CFG.artifacts_dir, "vae_recon.png"))
    cache_latents(model, obs, N, Tp1, S, device)


if __name__ == "__main__":
    main()
