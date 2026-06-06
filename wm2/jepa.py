"""The JEPA track -- a DECODER-FREE, action-conditioned latent predictor.

This is the *non-generative* paradigm (LeCun's bet) built on the same toy world as
the generative RSSM, so the central 2026 debate -- predict pixels (Dreamer
decoder) vs predict in a learned latent space (JEPA) -- is reproduced and measured
locally (DESIGN.md sec 6; RESEARCH_2026 "Non-generative / JEPA frontier").

It is a small instance of the **V-JEPA 2 / 2-AC** recipe (Meta FAIR,
arXiv:2506.09985):

  * an ONLINE encoder f_theta (small CNN -> jepa_dim) trained by gradient descent;
  * an EMA TARGET encoder f_xi -- a momentum copy of f_theta (no gradients), whose
    embeddings are the prediction TARGET and are always stop-gradient'd. The slow
    target is what makes joint-embedding prediction stable instead of collapsing
    onto a trivial solution (BYOL / V-JEPA);
  * an action-conditioned PREDICTOR g([s_t, a_t]) -> s_{t+1}_hat that forecasts the
    target encoder's embedding of the NEXT frame. There is NO pixel decoder
    anywhere on this path -- the model never reconstructs an image.

LOSS (V-JEPA latent prediction + VICReg-lite anti-collapse, Bardes et al. 2022):

  * invariance:  1 - cos(s_{t+1}_hat, sg(f_xi(o_{t+1})))   (cosine in latent space)
  * variance:    hinge( std_over_batch(online_emb) >= 1 )  per dim -> pushes the
                 representation to spread out so it cannot collapse to a constant;
  * covariance:  mean of squared OFF-diagonal entries of the online-embedding
                 covariance -> decorrelates dimensions (no redundancy collapse).

Without the EMA target + variance/covariance terms a decoder-free predictor would
happily drive every embedding to the same vector (cosine = 1 trivially); the
explicit COLLAPSE CHECK printed during training (mean per-dim std of the online
embeddings) is the empirical guard that this is not happening.

For the latent planner (wm2/plan.py, V-JEPA-2-AC-style CEM/MPPI) this module
exposes `encode(obs) -> s` (online encoder) and `predict(s, a_onehot) -> s_next`
(roll the predictor open-loop in latent space, no pixels).

Refs: V-JEPA 2 / 2-AC (arXiv:2506.09985), BYOL (Grill et al. 2020),
VICReg (Bardes et al. 2022).
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wm2.config import CFG, get_device
from wm2.utils import mlp, obs_to_tensor

# --- JEPA loss weights (kept here, not in the FROZEN config) -----------------
# Invariance (the cosine prediction objective) and the variance hinge are the two
# forces in tension: invariance pulls embeddings together, variance pushes them
# apart so they cannot collapse. They are weighted equally (the VICReg/V-JEPA
# recipe). Covariance only decorrelates dims and needs a much smaller weight.
JEPA_INV_SCALE = 1.0    # 1 - cos(pred, sg(target))
JEPA_VAR_SCALE = 1.0    # hinge variance: per-dim std >= 1 (anti-collapse)
JEPA_COV_SCALE = 0.04   # off-diagonal covariance penalty (decorrelation)


# ----------------------------------------------------------------- conv encoder
class JEPAEncoder(nn.Module):
    """64x64x3 frame -> jepa_dim embedding (4 stride-2 convs, SiLU).

    Same conv backbone shape as `rssm.Encoder` (so the two tracks see the world at
    matching capacity), but PRIVATE to the JEPA path: the whole point of the
    non-generative track is to learn its own representation end-to-end from the
    predictive objective, with no decoder and no shared weights.
    """

    def __init__(self, jepa_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.SiLU(),
        )
        # LayerNorm before the projection keeps embedding scale stable for the
        # variance hinge (which is defined against a target std of 1).
        self.fc = nn.Linear(256 * 2 * 2, jepa_dim)
        self.norm = nn.LayerNorm(jepa_dim)

    def forward(self, x):  # x:(B,3,64,64)
        h = self.conv(x).reshape(x.size(0), -1)
        return self.norm(self.fc(h))


# ----------------------------------------------------------------------- JEPA
class JEPA(nn.Module):
    """Online encoder + EMA target encoder + action-conditioned latent predictor.

    Decoder-free: nothing here maps a latent back to pixels.
    """

    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.jepa_dim
        self.n_actions = cfg.n_actions
        self.ema = cfg.jepa_ema

        # f_theta: trained by SGD.  f_xi: momentum copy, never receives gradients.
        self.online = JEPAEncoder(self.dim)
        self.target = JEPAEncoder(self.dim)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)

        # g([s_t, a_t_onehot]) -> s_{t+1}_hat. A residual MLP: it is far easier to
        # predict the *change* in latent than the absolute next latent, and it keeps
        # the predictor near identity at init (helpful for the cosine objective).
        self.predictor = mlp(self.dim + self.n_actions, cfg.hidden, self.dim,
                             layers=2)

    # ---- EMA target update -------------------------------------------------
    @torch.no_grad()
    def update_target(self):
        """f_xi <- ema * f_xi + (1-ema) * f_theta  (BYOL / V-JEPA momentum encoder).

        Call once per optimizer step AFTER the online encoder has been updated."""
        m = self.ema
        for pt, po in zip(self.target.parameters(), self.online.parameters()):
            pt.mul_(m).add_((1.0 - m) * po.detach())
        # Buffers (e.g. LayerNorm has none, but be safe / future-proof) tracked too.
        for bt, bo in zip(self.target.buffers(), self.online.buffers()):
            bt.copy_(bo)

    # ---- encoders ----------------------------------------------------------
    def encode(self, obs) -> torch.Tensor:
        """ONLINE-encode frames to latents. Accepts (B,3,64,64) float tensors OR
        (...,64,64,3) uint8 arrays (converted via obs_to_tensor). Returns (B,dim).
        This is the representation the planner searches over."""
        x = self._as_frames(obs)
        return self.online(x)

    @torch.no_grad()
    def target_encode(self, obs) -> torch.Tensor:
        """TARGET (EMA) encode, stop-gradient by construction. The prediction target."""
        x = self._as_frames(obs)
        return self.target(x)

    def predict(self, state: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        """Predict the NEXT latent from the current latent + a one-hot action.

        Residual: s_{t+1}_hat = s_t + g([s_t, a_t]). Used both as the training
        prediction and, rolled open-loop, as the planner's latent dynamics. Works on
        a single step (B,dim)+(B,n_actions) or any leading batch shape."""
        x = torch.cat([state, action_onehot], dim=-1)
        return state + self.predictor(x)

    def _as_frames(self, obs) -> torch.Tensor:
        """Coerce input to (B,3,64,64) float on the encoder's device."""
        device = next(self.online.parameters()).device
        if isinstance(obs, np.ndarray):
            obs = obs_to_tensor(obs)
        obs = obs.to(device)
        if obs.dim() == 3:                       # (3,64,64) -> (1,3,64,64)
            obs = obs.unsqueeze(0)
        return obs

    # ---- training loss -----------------------------------------------------
    def loss(self, obs_t, act_t, obs_tp1):
        """One JEPA training step's loss on a batch of (o_t, a_t, o_{t+1}) triples.

        obs_t/obs_tp1: (B,3,64,64) float.  act_t: (B,) long.
        Returns (loss, metrics) where metrics['online_std'] is the COLLAPSE CHECK
        (mean per-dim std of the online embeddings -- must stay well above 0).
        """
        cfg = self.cfg
        a_oh = F.one_hot(act_t, self.n_actions).float()

        s_t = self.online(obs_t)                          # online embed of o_t
        s_pred = self.predict(s_t, a_oh)                  # s_{t+1}_hat
        with torch.no_grad():                             # EMA target = stop-grad
            s_tgt = self.target(obs_tp1)                  # f_xi(o_{t+1})

        # --- invariance: 1 - cosine(prediction, sg(target)) ---
        inv = (1.0 - F.cosine_similarity(s_pred, s_tgt.detach(), dim=-1)).mean()

        # --- VICReg-lite anti-collapse, on the ONLINE embeddings of BOTH frames ---
        # The target above was the EMA copy, so we recompute o_{t+1}'s ONLINE embed
        # here; stacking it with s_t doubles the sample size for the variance /
        # covariance statistics (a steadier collapse estimate) at one extra forward.
        s_tp1_online = self.online(obs_tp1)
        z = torch.cat([s_t, s_tp1_online], dim=0)         # (2B, dim)
        var_loss, cov_loss, online_std = self._vicreg_terms(z)

        loss = (JEPA_INV_SCALE * inv
                + JEPA_VAR_SCALE * var_loss
                + JEPA_COV_SCALE * cov_loss)
        metrics = {
            "loss": loss.item(), "inv": inv.item(),
            "var": var_loss.item(), "cov": cov_loss.item(),
            "online_std": online_std.item(),
            # cosine in [-1,1]; report it so "is it actually predicting?" is visible.
            "cos": (1.0 - inv).item(),
        }
        return loss, metrics

    def _vicreg_terms(self, z: torch.Tensor):
        """VICReg variance + covariance regularizers on embeddings z:(M,dim).

        variance  = mean_d hinge(1 - std_d)         (push each dim's std >= 1)
        covariance= mean of squared OFF-diagonal entries of cov(z)   (decorrelate)
        Returns (var_loss, cov_loss, mean_std) where mean_std is the collapse check.
        """
        # Per-dimension standard deviation across the batch (eps for sqrt stability).
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)   # (dim,)
        var_loss = F.relu(1.0 - std).mean()                     # hinge at std=1

        # Covariance of the centered embeddings; penalize off-diagonal magnitude.
        zc = z - z.mean(dim=0, keepdim=True)
        M = z.shape[0]
        cov = (zc.T @ zc) / max(1, M - 1)                       # (dim,dim)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = (off_diag ** 2).sum() / self.dim
        return var_loss, cov_loss, std.mean()

    # ---- persistence -------------------------------------------------------
    def save(self, path):
        torch.save({"online": self.online.state_dict(),
                    "target": self.target.state_dict(),
                    "predictor": self.predictor.state_dict(),
                    "cfg": self.cfg.__dict__}, path)

    @classmethod
    def load(cls, device=None, path=None):
        device = device or get_device()
        path = path or os.path.join(CFG.ckpt_dir, "jepa.pt")
        ck = torch.load(path, map_location=device)
        m = cls(CFG).to(device)
        m.online.load_state_dict(ck["online"])
        m.target.load_state_dict(ck["target"])
        m.predictor.load_state_dict(ck["predictor"])
        m.eval()
        return m


# ---------------------------------------------------- (o_t, a_t, o_{t+1}) batching
def _sample_triples(data, B, device, rng):
    """Sample B random (o_t, a_t, o_{t+1}) triples from rollouts.npz.

    obs is (E,T+1,...) and act is (E,T): action a_t at obs index t leads to obs
    index t+1 (same alignment as rssm.observe). Returns float frame tensors +
    long actions on `device`."""
    E, Tp1 = data["obs"].shape[:2]
    T = Tp1 - 1
    eps = rng.integers(0, E, size=B)
    ts = rng.integers(0, T, size=B)                       # t in [0, T-1]
    o_t = np.stack([data["obs"][e, t] for e, t in zip(eps, ts)])
    o_tp1 = np.stack([data["obs"][e, t + 1] for e, t in zip(eps, ts)])
    a_t = np.array([data["act"][e, t] for e, t in zip(eps, ts)], dtype=np.int64)
    o_t = obs_to_tensor(o_t).to(device)
    o_tp1 = obs_to_tensor(o_tp1).to(device)
    a_t = torch.from_numpy(a_t).long().to(device)
    return o_t, a_t, o_tp1


# --------------------------------------------------------------------- training
def train_jepa(cfg=CFG, steps_per_epoch: int = 150, batch: int = 64,
               log_every: int = 50):
    """Train the decoder-free JEPA encoder + predictor on (o_t,a_t,o_{t+1}) triples.

    Runs for cfg.jepa_epochs, EMA-updates the target encoder every step, prints the
    loss breakdown plus an explicit COLLAPSE CHECK (mean per-dim std of the online
    embeddings), and saves wm2_checkpoints/jepa.pt.
    """
    device = get_device()
    cfg.ensure_dirs()
    # Materialize the npz into memory ONCE. A lazy NpzFile re-decompresses the full
    # 495 MB obs array on EVERY `data["obs"]` access; indexing it inside the
    # per-step sampler made training ~500x too slow (hours instead of a minute).
    data = dict(np.load(os.path.join(cfg.data_dir, "rollouts.npz")))
    jepa = JEPA(cfg).to(device)
    jepa.train()
    opt = torch.optim.Adam(
        list(jepa.online.parameters()) + list(jepa.predictor.parameters()),
        lr=cfg.jepa_lr)
    rng = np.random.default_rng(cfg.seed)
    n_params = sum(p.numel() for p in jepa.online.parameters()) \
        + sum(p.numel() for p in jepa.predictor.parameters())
    print(f"JEPA params (online+predictor): {n_params/1e6:.2f}M  device={device}")

    t0 = time.time()
    for epoch in range(cfg.jepa_epochs):
        last = None
        for it in range(steps_per_epoch):
            o_t, a_t, o_tp1 = _sample_triples(data, batch, device, rng)
            loss, m = jepa.loss(o_t, a_t, o_tp1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(jepa.online.parameters()) + list(jepa.predictor.parameters()),
                cfg.grad_clip)
            opt.step()
            jepa.update_target()                          # momentum target step
            last = m
        # COLLAPSE CHECK printed every epoch: online_std must stay >> 0.
        flag = "OK" if last["online_std"] > 0.1 else "!! COLLAPSE"
        print(f"epoch {epoch+1:2d}/{cfg.jepa_epochs}  loss={last['loss']:.4f}  "
              f"inv={last['inv']:.4f}  cos={last['cos']:+.3f}  "
              f"var={last['var']:.4f}  cov={last['cov']:.4f}  "
              f"online_std={last['online_std']:.3f} [{flag}]  "
              f"({time.time()-t0:.0f}s)")
    path = os.path.join(cfg.ckpt_dir, "jepa.pt")
    jepa.save(path)
    print(f"saved {path}")
    return jepa


# --------------------------------------------------------------------- smoke test
def _smoke():
    """CPU, tiny: overfit a handful of real triples for a few steps. Asserts the
    loss is finite + decreases, that shapes are right, and -- critically for a
    decoder-free model -- that the representation does NOT collapse (online per-dim
    std stays well above zero)."""
    torch.manual_seed(0)
    device = torch.device("cpu")                          # force CPU: do not touch MPS
    cfg = CFG
    data = np.load(os.path.join(cfg.data_dir, "rollouts.npz"))

    jepa = JEPA(cfg).to(device)
    jepa.train()
    opt = torch.optim.Adam(
        list(jepa.online.parameters()) + list(jepa.predictor.parameters()),
        lr=cfg.jepa_lr)
    rng = np.random.default_rng(0)
    B = 4                                                  # tiny batch

    # Fixed batch so we can verify the loss actually drops by overfitting it.
    o_t, a_t, o_tp1 = _sample_triples(data, B, device, rng)

    # Shape checks on the public planner interface.
    s = jepa.encode(o_t)
    assert s.shape == (B, cfg.jepa_dim), s.shape
    a_oh = F.one_hot(a_t, cfg.n_actions).float()
    s_next = jepa.predict(s, a_oh)
    assert s_next.shape == (B, cfg.jepa_dim), s_next.shape
    # encode() must also accept a raw uint8 frame array (planner convenience).
    s_np = jepa.encode(data["obs"][0, 0])
    assert s_np.shape == (1, cfg.jepa_dim), s_np.shape

    first = None
    n_params = sum(p.numel() for p in jepa.online.parameters()) \
        + sum(p.numel() for p in jepa.predictor.parameters())
    print(f"JEPA params (online+predictor): {n_params/1e6:.2f}M  device={device}")
    for i in range(30):
        loss, m = jepa.loss(o_t, a_t, o_tp1)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(
            list(jepa.online.parameters()) + list(jepa.predictor.parameters()),
            cfg.grad_clip)
        opt.step()
        jepa.update_target()
        if i == 0:
            first = m
        if i % 5 == 0 or i == 29:
            flag = "OK" if m["online_std"] > 0.1 else "!! COLLAPSE"
            print(f"step {i:2d}  loss={m['loss']:.4f}  inv={m['inv']:.4f}  "
                  f"cos={m['cos']:+.3f}  var={m['var']:.4f}  cov={m['cov']:.4f}  "
                  f"online_std={m['online_std']:.3f} [{flag}]")

    assert np.isfinite(m["loss"]), f"loss not finite: {m}"
    assert m["loss"] < first["loss"], "loss did not decrease"
    # Anti-collapse: the EMA target + VICReg terms must keep the embedding spread.
    assert m["online_std"] > 0.1, f"representation collapsed: std={m['online_std']}"
    # Sanity: the EMA target must actually differ from the online net (momentum < 1).
    drift = sum((pt - po).abs().sum().item()
                for pt, po in zip(jepa.target.parameters(),
                                  jepa.online.parameters()))
    assert drift > 0, "EMA target identical to online (no momentum update happened)"
    print(f"jepa smoke PASSED (loss decreased, no collapse, EMA drift={drift:.2f})")


if __name__ == "__main__":
    _smoke()
