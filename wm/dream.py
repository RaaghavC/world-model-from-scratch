"""The dream engine: roll the world model forward purely in latent space.

`WorldModel` wraps the trained VAE (V) and MDN-RNN (M) and is reused everywhere:
visualization, controller training in imagination, and evaluation.

A "dream" is an OPEN-LOOP latent rollout: we encode exactly ONE real frame to get
z_0, then let M hallucinate every subsequent latent from actions alone, decoding
to pixels only when we want to look. Because nothing re-grounds the rollout in
real observations, small per-step errors compound -- this is the long-horizon
drift that the world-model literature repeatedly flags.
"""
from __future__ import annotations

import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from wm.config import CFG, get_device
from wm.env import Particle2DEnv, RepeatRandomPolicy, rollout
from wm.mdnrnn import MDNRNN, mdn_sample, step_params
from wm.vae import ConvVAE, obs_to_tensor, tensor_to_obs


class WorldModel:
    """Bundle of (VAE, MDN-RNN) with encode / decode / dream helpers."""

    def __init__(self, vae: ConvVAE, mdn: MDNRNN, device):
        self.vae, self.mdn, self.device = vae, mdn, device
        self.n_actions = mdn.n_actions

    @classmethod
    def load(cls, device=None):
        """Load both checkpoints written by train_vae.py / train_mdnrnn.py."""
        device = device or get_device()
        vc = torch.load(os.path.join(CFG.ckpt_dir, "vae.pt"), map_location=device)
        vae = ConvVAE(vc["z_dim"]).to(device)
        vae.load_state_dict(vc["model"]); vae.eval()
        mc = torch.load(os.path.join(CFG.ckpt_dir, "mdnrnn.pt"), map_location=device)
        mdn = MDNRNN(mc["z_dim"], mc["n_actions"], mc["hidden"], mc["k"]).to(device)
        mdn.load_state_dict(mc["model"]); mdn.eval()
        return cls(vae, mdn, device)

    @torch.no_grad()
    def encode(self, obs_uint8) -> torch.Tensor:
        """(...,H,W,3) uint8 -> (B,z) posterior MEAN (deterministic encoding)."""
        x = obs_to_tensor(np.asarray(obs_uint8)).to(self.device)
        if x.dim() == 3:                       # a single frame -> add batch dim
            x = x.unsqueeze(0)
        mu, _ = self.vae.encode(x)
        return mu

    @torch.no_grad()
    def decode(self, z) -> np.ndarray:
        """(B,z) -> (B,H,W,3) uint8 frames."""
        return tensor_to_obs(self.vae.decode(z))

    @torch.no_grad()
    def dream_rollout(self, z0, actions, temperature=1.0, mode=False, hc=None):
        """Hallucinate a latent trajectory.

        z0:(B,z) starting latent; actions:(B,T) long. Returns (z_seq, hc) where
        z_seq:(B,T+1,z) includes z0. At each step the MDN predicts z_{t+1} from
        (z_t, a_t) and we either sample it or take its mode.
        """
        z = z0
        zs = [z0]
        T = actions.shape[1]
        for t in range(T):
            a = F.one_hot(actions[:, t], self.n_actions).float()
            (logpi, mu, logsig), hc = step_params(self.mdn, z, a, hc)
            z = mdn_sample(logpi, mu, logsig, temperature, mode)
            zs.append(z)
        return torch.stack(zs, dim=1), hc


@torch.no_grad()
def comparison(out_prefix: str, ep_len: int = 80, seed: int = 9999, temperature: float = 1.0):
    """Build a 3-way comparison on a FRESH (held-out-seed) episode:

        ground truth | VAE reconstruction (renderer ceiling) | open-loop dream

    The middle column isolates VAE error (the best the decoder could do given the
    true frames); the gap between middle and right is therefore pure *dynamics*
    error. Saves a side-by-side GIF and a sampled montage, and returns the mean
    per-pixel dream MSE.
    """
    device = get_device()
    wm = WorldModel.load(device)
    env = Particle2DEnv(CFG, seed=seed)
    pol = RepeatRandomPolicy(CFG, seed=seed + 1)
    obs, acts, _ = rollout(env, pol, ep_len=ep_len)            # (T+1,H,W,3), (T,)

    # Middle column: encode+decode every real frame (no dynamics involved).
    x = obs_to_tensor(obs).to(device)
    recon = tensor_to_obs(wm.vae.decode(wm.vae.encode(x)[0]))   # (T+1,H,W,3)

    # Right column: encode ONLY frame 0, then hallucinate the rest from actions.
    # temperature==0 -> deterministic "mode" dream (matches this deterministic env).
    z0 = wm.encode(obs[0])
    a = torch.from_numpy(acts).long().to(device).unsqueeze(0)   # (1,T)
    z_seq, _ = wm.dream_rollout(z0, a, temperature=temperature, mode=(temperature == 0.0))
    dream = wm.decode(z_seq[0])                                 # (T+1,H,W,3)

    # Stitch the three columns with thin white separators.
    sep = np.full((obs.shape[1], 2, 3), 255, np.uint8)
    frames = [np.hstack([obs[i], sep, recon[i], sep, dream[i]]) for i in range(len(obs))]
    gif = f"{out_prefix}.gif"
    imageio.mimsave(gif, frames, fps=20, loop=0)

    sel = list(range(0, len(obs), max(1, len(obs) // 6)))[:6]
    montage = np.vstack([frames[i] for i in sel])
    png = f"{out_prefix}_montage.png"
    imageio.imwrite(png, montage)

    err = float(np.mean((dream.astype(np.float32) - obs.astype(np.float32)) ** 2))
    print(f"[dream] saved {gif}\n[dream] saved {png}\n[dream] mean dream pixel MSE={err:.2f} (cols: real | vae-recon | dream)")
    return err


if __name__ == "__main__":
    comparison(os.path.join(CFG.artifacts_dir, "dream_compare"))
