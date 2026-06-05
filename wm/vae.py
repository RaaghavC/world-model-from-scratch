"""V -- a convolutional variational autoencoder (the "renderer").

Compresses a 64x64x3 frame into a `z_dim` Gaussian latent and reconstructs it.
This is the canonical Ha & Schmidhuber (2018) "World Models" vision module: four
strided convolutions down to a small latent bottleneck, four transposed
convolutions back up.

Why a VAE and not a plain autoencoder? The KL term pushes the latent toward a
smooth, unit-Gaussian-ish space. That smoothness is what makes the downstream
MDN-RNN's job tractable: nearby latents decode to nearby frames, so a small error
in a predicted latent is a small error in the dreamed image.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def obs_to_tensor(obs_uint8: np.ndarray) -> torch.Tensor:
    """(..., H, W, 3) uint8  ->  (..., 3, H, W) float in [0,1].

    `.contiguous()` matters: `movedim` returns a non-contiguous view, and some
    MPS/conv backward kernels then hit an internal `.view` that requires
    contiguous memory.
    """
    t = torch.from_numpy(np.ascontiguousarray(obs_uint8)).float() / 255.0
    return t.movedim(-1, -3).contiguous()


def tensor_to_obs(t: torch.Tensor) -> np.ndarray:
    """(..., 3, H, W) float in [0,1]  ->  (..., H, W, 3) uint8 (clamped)."""
    t = t.clamp(0, 1).movedim(-3, -1)
    return (t.detach().cpu().numpy() * 255).astype(np.uint8)


class ConvVAE(nn.Module):
    """Conv encoder -> (mu, logvar) -> reparameterized z -> conv decoder."""

    def __init__(self, z_dim: int = 32, img_ch: int = 3):
        super().__init__()
        self.z_dim = z_dim
        # Encoder spatial sizes for a 64x64 input (kernel 4, stride 2, no pad):
        #   64 -> 31 -> 14 -> 6 -> 2 ; channels 32 -> 64 -> 128 -> 256.
        # Final feature map is 256 x 2 x 2 = 1024 features.
        self.enc = nn.Sequential(
            nn.Conv2d(img_ch, 32, 4, stride=2), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(True),
        )
        # Two heads map the 1024 features to the latent's mean and log-variance.
        self.fc_mu = nn.Linear(256 * 2 * 2, z_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, z_dim)
        # Decoder mirrors the encoder. Input is reshaped to a 1024-channel 1x1
        # "image"; transposed convs grow it back: 1 -> 5 -> 13 -> 30 -> 64.
        self.fc_dec = nn.Linear(z_dim, 1024)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(True),
            nn.ConvTranspose2d(32, img_ch, 6, stride=2), nn.Sigmoid(),  # output in [0,1]
        )

    def encode(self, x):
        """x:(B,3,64,64) -> (mu, logvar), each (B,z_dim)."""
        h = self.enc(x).reshape(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Sample z = mu + sigma * eps  (the reparameterization trick), so the
        sampling stays differentiable w.r.t. (mu, logvar)."""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        """z:(B,z_dim) -> reconstructed frame (B,3,64,64) in [0,1]."""
        h = self.fc_dec(z).reshape(-1, 1024, 1, 1)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Background colour as a (1,3,1,1) tensor, for the foreground mask in the loss.
_BG = torch.tensor([18.0, 18.0, 28.0]).view(1, 3, 1, 1) / 255.0


def vae_loss(recon, x, mu, logvar, beta: float = 1.0, fg_weight: float = 0.0):
    """Foreground-weighted MSE reconstruction + beta * KL, averaged over the batch.

    THE KEY FIX for this domain: plain pixel-MSE is dominated by the large dark
    background, so the optimizer happily drops the tiny (~46 px) moving ball to a
    blur. Weighting non-background pixels by (1 + fg_weight) forces the VAE to
    spend capacity on the ball and goal -- the only things that actually matter.

    Returns (total_loss, reconstruction_term, kl_term), each a scalar.
    """
    b = x.size(0)
    if fg_weight > 0:
        bg = _BG.to(x.device)
        # Per-pixel max-channel distance from the background colour; a pixel is
        # "foreground" if it differs by > 0.15 in any channel.
        dev = (x - bg).abs().amax(dim=1, keepdim=True)   # (B,1,H,W)
        w = 1.0 + fg_weight * (dev > 0.15).float()
        recon_l = (w * (recon - x) ** 2).sum() / b
    else:
        recon_l = F.mse_loss(recon, x, reduction="sum") / b
    # KL(q(z|x) || N(0, I)), the standard closed-form Gaussian KL.
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b
    return recon_l + beta * kl, recon_l, kl
