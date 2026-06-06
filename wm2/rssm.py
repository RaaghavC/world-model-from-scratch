"""The world model: a discrete-latent Recurrent State-Space Model (the V+M fusion).

This is the DreamerV3 core and the heart of wm2. It replaces the baseline's
separate ConvVAE (continuous latent) + MDN-RNN with ONE model whose latent state
has two parts (PlaNet/RSSM):

  * a DETERMINISTIC path  h_t = GRU([z_{t-1}, a_{t-1}], h_{t-1})   -- carries memory
  * a STOCHASTIC path     z_t ~ Categorical                        -- captures futures

with a categorical (discrete) z of shape (stoch_cat, stoch_classes), trained with
straight-through gradients, KL balancing, and free bits. Two heads make control
possible without any hand-built probe: a two-hot **reward** head and a Bernoulli
**continue** head. A CNN decoder reconstructs frames (for viewing + the recon
loss); imagination (img_step) never needs it.

Refs: Hafner et al. PlaNet (RSSM), DreamerV2 (discrete latents), DreamerV3
(symlog, two-hot, KL balancing + free bits, unimix).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm2.config import CFG, get_device
from wm2.utils import (categorical_kl, categorical_st, mlp, obs_to_tensor,
                       symlog, symlog_support, symlog_twohot_loss,
                       value_from_logits)


# ------------------------------------------------------------------- conv encoder/decoder
class Encoder(nn.Module):
    """64x64x3 frame -> embed_dim vector (4 stride-2 convs, SiLU)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.SiLU(),
        )
        self.fc = nn.Linear(256 * 2 * 2, embed_dim)

    def forward(self, x):  # x:(B,3,64,64)
        h = self.conv(x).reshape(x.size(0), -1)
        return self.fc(h)


class Decoder(nn.Module):
    """feat_dim -> 64x64x3 frame. Uses nearest-neighbour upsampling + 3x3 convs
    rather than transposed convolutions: transposed convs leave checkerboard /
    mottle artifacts on flat regions (here the dark background renders as a green
    texture), which resize-conv avoids (Odena et al., 2016)."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, 256 * 4 * 4)

        def up(cin, cout, last=False):
            return [nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(cin, cout, 3, padding=1),
                    (nn.Sigmoid() if last else nn.SiLU())]

        self.deconv = nn.Sequential(
            *up(256, 128), *up(128, 64), *up(64, 32), *up(32, 3, last=True),
        )  # 4 -> 8 -> 16 -> 32 -> 64

    def forward(self, feat):
        h = self.fc(feat).reshape(-1, 256, 4, 4)
        return self.deconv(h)


# --------------------------------------------------------------------------- RSSM
class WorldModel(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.cat, self.classes = cfg.stoch_cat, cfg.stoch_classes
        self.stoch_dim = cfg.stoch_dim
        self.deter_dim = cfg.deter_dim
        self.n_actions = cfg.n_actions

        self.encoder = Encoder(cfg.embed_dim)
        self.decoder = Decoder(cfg.feat_dim)
        # Pre-GRU input layer over [stoch, action] (DreamerV3 "img_in").
        self.img_in = nn.Sequential(
            nn.Linear(self.stoch_dim + cfg.n_actions, cfg.hidden),
            nn.LayerNorm(cfg.hidden), nn.SiLU())
        self.gru = nn.GRUCell(cfg.hidden, cfg.deter_dim)
        # Prior predicts z from h alone (imagination); posterior also sees the obs.
        self.prior_net = mlp(cfg.deter_dim, cfg.hidden, self.stoch_dim, layers=1)
        self.post_net = mlp(cfg.deter_dim + cfg.embed_dim, cfg.hidden,
                            self.stoch_dim, layers=1)
        # Heads on the full feature [h, z].
        self.reward_head = mlp(cfg.feat_dim, cfg.hidden, cfg.value_buckets, layers=2)
        self.cont_head = mlp(cfg.feat_dim, cfg.hidden, 1, layers=2)
        self.register_buffer(
            "support", symlog_support(cfg.value_vmin, cfg.value_vmax,
                                      cfg.value_buckets, "cpu"))
        # Background colour (env C_BG/255) for foreground-weighted reconstruction.
        self.register_buffer(
            "_bg", torch.tensor([18.0, 18.0, 28.0]).reshape(1, 1, 3, 1, 1) / 255.0)

    # ---- state helpers -----------------------------------------------------
    def initial(self, B, device):
        return {"deter": torch.zeros(B, self.deter_dim, device=device),
                "stoch": torch.zeros(B, self.cat, self.classes, device=device)}

    def get_feat(self, state):
        return torch.cat([state["deter"],
                          state["stoch"].reshape(state["stoch"].size(0), -1)], -1)

    def _logits(self, flat):
        return flat.reshape(-1, self.cat, self.classes)

    # ---- one-step transitions ---------------------------------------------
    def img_step(self, prev, prev_action, sample=True):
        """Imagined step: advance the deterministic path and SAMPLE z from the prior
        (no observation). prev_action:(B,n_actions) one-hot. Returns new state +
        prior logits."""
        stoch_flat = prev["stoch"].reshape(prev["stoch"].size(0), -1)
        x = self.img_in(torch.cat([stoch_flat, prev_action], -1))
        deter = self.gru(x, prev["deter"])
        prior_logits = self._logits(self.prior_net(deter))
        stoch, _ = categorical_st(prior_logits, self.cfg.unimix, sample=sample)
        return {"deter": deter, "stoch": stoch}, prior_logits

    def obs_step(self, prev, prev_action, embed, sample=True):
        """Filtered step: same deterministic update, but z from the POSTERIOR
        (conditioned on the observation embedding). Returns state, post, prior
        logits."""
        state, prior_logits = self.img_step(prev, prev_action, sample=sample)
        post_logits = self._logits(self.post_net(torch.cat([state["deter"], embed], -1)))
        stoch, _ = categorical_st(post_logits, self.cfg.unimix, sample=sample)
        state = {"deter": state["deter"], "stoch": stoch}
        return state, post_logits, prior_logits

    # ---- heads -------------------------------------------------------------
    def decode(self, feat):
        return self.decoder(feat)

    def reward(self, feat):
        return value_from_logits(self.reward_head(feat), self.support)

    def cont(self, feat):
        return torch.sigmoid(self.cont_head(feat)).squeeze(-1)

    # ---- rollout over a real sequence (posterior filtering) ----------------
    def observe(self, obs, act, hist_aug: float = 0.0):
        """obs:(B,L,3,64,64) float, act:(B,L) long (action TAKEN at each step, i.e.
        a_{t} leads to obs_{t+1}). Returns dict of stacked (B,L,...) tensors:
        feat, post_logits, prior_logits, deter, stoch. Action fed at step t is the
        PREVIOUS action a_{t-1}; a_{-1}=0.

        `hist_aug` in (0,1] enables LATENT HISTORY AUGMENTATION (the RSSM-native
        analogue of Diffusion Forcing's history corruption, Decart MirageLSD 2025):
        with this per-step, per-batch probability the latent CARRIED FORWARD to the
        next step is the model's own PRIOR sample instead of the posterior, so the
        dynamics learns to recover from its own rollout errors -> less open-loop
        drift. Losses are still computed on the posterior states, unchanged."""
        B, L = obs.shape[:2]
        device = obs.device
        embed = self.encoder(obs.reshape(B * L, *obs.shape[2:])).reshape(B, L, -1)
        a_oh = F.one_hot(act, self.n_actions).float()
        prev = self.initial(B, device)
        prev_a = torch.zeros(B, self.n_actions, device=device)
        feats, posts, priors, deters, stochs = [], [], [], [], []
        for t in range(L):
            state, post_l, prior_l = self.obs_step(prev, prev_a, embed[:, t])
            feats.append(self.get_feat(state))
            posts.append(post_l); priors.append(prior_l)
            deters.append(state["deter"]); stochs.append(state["stoch"])
            prev_a = a_oh[:, t]                      # action taken AT step t
            if hist_aug > 0.0 and t < L - 1:
                # Carry the prior sample forward for a random subset of the batch.
                prior_stoch, _ = categorical_st(prior_l, self.cfg.unimix, sample=True)
                mask = (torch.rand(B, 1, 1, device=device) < hist_aug).float()
                carried = mask * prior_stoch + (1 - mask) * state["stoch"]
                prev = {"deter": state["deter"], "stoch": carried}
            else:
                prev = state
        st = lambda xs: torch.stack(xs, 1)
        return {"feat": st(feats), "post_logits": st(posts),
                "prior_logits": st(priors), "deter": st(deters), "stoch": st(stochs)}

    # ---- training loss -----------------------------------------------------
    def loss(self, batch, hist_aug: float = 0.0):
        """batch: obs(B,L,3,64,64) float, act(B,L) long, rew(B,L) float,
        cont(B,L) float. Returns (loss, metrics, out). `hist_aug` enables latent
        history augmentation in observe (the drift fix)."""
        cfg = self.cfg
        out = self.observe(batch["obs"], batch["act"], hist_aug=hist_aug)
        feat = out["feat"]
        B, L = feat.shape[:2]
        # FOREGROUND-WEIGHTED reconstruction: plain pixel-MSE is dominated by the
        # large dark background, washing out the small bright bodies (incl. the
        # agent), which is exactly why the latent fails to localize them. Upweight
        # non-background pixels (the baseline's key fix, here inside the RSSM).
        recon = self.decoder(feat.reshape(B * L, -1)).reshape(B, L, 3, 64, 64)
        dev = (batch["obs"] - self._bg).abs().amax(dim=2, keepdim=True)  # (B,L,1,64,64)
        w = 1.0 + cfg.vae_fg_weight * (dev > 0.15).float()
        recon_loss = (w * (recon - batch["obs"]) ** 2).sum([2, 3, 4]).mean()
        # Reward / continue heads.
        rew_logits = self.reward_head(feat)
        reward_loss = symlog_twohot_loss(rew_logits, batch["rew"], self.support)
        cont_logit = self.cont_head(feat).squeeze(-1)
        cont_loss = F.binary_cross_entropy_with_logits(cont_logit, batch["cont"])
        # KL balancing with free bits (DreamerV2/V3).
        post_p = F.softmax(out["post_logits"], -1)
        prior_p = F.softmax(out["prior_logits"], -1)
        free = cfg.kl_free_bits
        dyn = torch.clamp(categorical_kl(post_p.detach(), prior_p), min=free).mean()
        rep = torch.clamp(categorical_kl(post_p, prior_p.detach()), min=free).mean()
        kl_loss = cfg.kl_balance * dyn + (1 - cfg.kl_balance) * rep
        loss = (cfg.recon_scale * recon_loss + cfg.reward_scale * reward_loss
                + cfg.cont_scale * cont_loss + cfg.kl_scale * kl_loss)
        metrics = {"loss": loss.item(), "recon": recon_loss.item(),
                   "reward": reward_loss.item(), "cont": cont_loss.item(),
                   "dyn_kl": dyn.item(), "rep_kl": rep.item()}
        return loss, metrics, out

    # ---- persistence -------------------------------------------------------
    def save(self, path):
        torch.save({"model": self.state_dict(), "cfg": self.cfg.__dict__}, path)

    @classmethod
    def load(cls, device=None, path=None):
        device = device or get_device()
        path = path or os.path.join(CFG.ckpt_dir, "world_model.pt")
        ck = torch.load(path, map_location=device)
        m = cls(CFG).to(device)
        m.load_state_dict(ck["model"]); m.eval()
        return m


# --------------------------------------------------------------------- smoke test
def _smoke():
    """Overfit a few real sequences for a handful of steps; loss must drop."""
    import numpy as np
    device = get_device()
    cfg = CFG
    data = np.load(os.path.join(cfg.data_dir, "rollouts.npz"))
    obs_np, act_np, rew_np, cont_np = (data["obs"], data["act"], data["rew"],
                                       data["cont"])
    L = cfg.wm_seq_len
    B = 8
    # Take B episodes, first L+1 frames -> obs(B,L,...), actions/reward aligned.
    obs = obs_to_tensor(obs_np[:B, :L]).to(device)          # (B,L,3,64,64)
    act = torch.from_numpy(act_np[:B, :L]).long().to(device)
    rew = torch.from_numpy(rew_np[:B, :L]).float().to(device)
    cont = torch.from_numpy(cont_np[:B, :L]).float().to(device)
    batch = {"obs": obs, "act": act, "rew": rew, "cont": cont}

    wm = WorldModel(cfg).to(device)
    n_params = sum(p.numel() for p in wm.parameters())
    opt = torch.optim.Adam(wm.parameters(), lr=cfg.wm_lr)
    print(f"WorldModel params: {n_params/1e6:.2f}M  device={device}")
    first = None
    for i in range(30):
        loss, m, _ = wm.loss(batch)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(wm.parameters(), cfg.grad_clip)
        opt.step()
        if i == 0:
            first = m
        if i % 5 == 0 or i == 29:
            print(f"step {i:2d}  loss={m['loss']:8.2f}  recon={m['recon']:7.2f}  "
                  f"rew={m['reward']:.3f}  cont={m['cont']:.3f}  "
                  f"dyn_kl={m['dyn_kl']:.2f}  rep_kl={m['rep_kl']:.2f}")
    assert m["loss"] < first["loss"], "loss did not decrease"
    print("rssm smoke test PASSED (loss decreased)")


if __name__ == "__main__":
    _smoke()
