"""Train the MDN-RNN (M) to predict next-latent dynamics from (z_t, a_t).

Trains on latents cached by train_vae.py. Each step samples z ~ N(mu, sigma) from
the VAE posterior (injecting the encoder's uncertainty, as in Ha & Schmidhuber),
takes a random seq_len window, and minimises the mixture NLL of the next latent.

Outputs:
  checkpoints/mdnrnn.pt
  artifacts/dream_compare.gif / _montage.png  (real | vae-recon | dream)
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from wm.config import CFG, get_device
from wm.dream import comparison
from wm.mdnrnn import MDNRNN, mdn_nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=CFG.rnn_epochs)
    ap.add_argument("--iters", type=int, default=150)
    args = ap.parse_args()
    CFG.ensure_dirs()
    device = get_device()

    # Train on the cached VAE posteriors, not raw frames: mu/sigma per frame.
    lat = np.load(os.path.join(CFG.data_dir, "latents.npz"))
    mu = torch.from_numpy(lat["mu"])                       # (N,T+1,z)
    sig = torch.exp(0.5 * torch.from_numpy(lat["logvar"]))  # std = exp(logvar/2)
    act = torch.from_numpy(np.load(os.path.join(CFG.data_dir, "act.npy")).astype(np.int64))  # (N,T)
    N, Tp1, Z = mu.shape
    T = act.shape[1]
    L = min(CFG.seq_len, T - 1)                            # training sub-sequence length
    print(f"latents {tuple(mu.shape)}  device={device}  seq_len={L}")

    mdn = MDNRNN(Z, CFG.n_actions, CFG.rnn_hidden, CFG.n_gaussians).to(device)
    opt = torch.optim.Adam(mdn.parameters(), lr=CFG.rnn_lr)
    B = CFG.rnn_batch

    for ep in range(args.epochs):
        mdn.train()
        tot, t0 = 0.0, time.time()
        for _ in range(args.iters):
            # A batch = B random episodes, all sharing one random window start s.
            idx = torch.randint(0, N, (B,))
            s = int(torch.randint(0, T - L + 1, (1,)))
            # Re-sample z ~ N(mu, sigma) from the VAE posterior every batch. This
            # injects the encoder's uncertainty (Ha & Schmidhuber do the same) and
            # acts as data augmentation, so M doesn't overfit point estimates.
            m = mu[idx, s : s + L + 1]
            sg = sig[idx, s : s + L + 1]
            z = (m + sg * torch.randn_like(m)).to(device)   # (B,L+1,z)
            # Predict z_{t+1} from z_t: inputs are steps [0..L), targets are [1..L].
            z_in, target = z[:, :-1], z[:, 1:]
            a = F.one_hot(act[idx, s : s + L], CFG.n_actions).float().to(device)
            params, _ = mdn(z_in, a)
            loss = mdn_nll(params, target, mdn)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(mdn.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        print(f"epoch {ep+1}/{args.epochs}  nll={tot/args.iters:8.3f}  ({time.time()-t0:.1f}s)")

    torch.save(
        {"model": mdn.state_dict(), "z_dim": Z, "n_actions": CFG.n_actions,
         "hidden": CFG.rnn_hidden, "k": CFG.n_gaussians},
        os.path.join(CFG.ckpt_dir, "mdnrnn.pt"),
    )
    print("saved checkpoints/mdnrnn.pt")
    # quick visual sanity check (deterministic mode = most-likely dream)
    comparison(os.path.join(CFG.artifacts_dir, "dream_compare"), ep_len=80, temperature=0.0)


if __name__ == "__main__":
    main()
