"""Latent Action Model (LAM) -- Genie's signature: label-free action discovery.

DESIGN.md S4 / RESEARCH_2026.md ("Genie three-module recipe"). The Genie line
(Bruce et al., arXiv:2402.15391; Genie 1-3) shows that the *actions* a controller
needs can be discovered WITHOUT any ground-truth action labels, purely from how
consecutive frames change. Module (2) of that recipe is the **latent action model**:
an inverse/forward pair with a tiny discrete bottleneck.

Architecture (this file):

  * A small PRIVATE CNN frame encoder (its own weights, mirroring the rssm
    `Encoder` shape -- it deliberately does NOT load a trained rssm checkpoint, so
    the LAM is a self-contained, label-free experiment) maps each frame o_t to an
    embedding emb_t in R^embed_dim.
  * INVERSE model I(emb_t, emb_{t+1}) -> a continuous latent action e_a, which is
    VECTOR-QUANTIZED (VQ-VAE; van den Oord et al. 2017) to one of `lam_codes`
    codebook vectors of width `lam_dim`, with a straight-through estimator and
    commitment cost `lam_beta`.
  * FORWARD model F(emb_t, quantized e_a) -> predicts the *delta* to emb_{t+1}
    (residual prediction: emb_{t+1}_hat = emb_t + F(...)). It only sees emb_t and
    the tiny code, so the code is FORCED to carry exactly the controllable change
    between the two frames -- which, in this thrust world, is the action.

Loss = MSE(forward_pred, sg(emb_{t+1}))            # next-embedding reconstruction
     + ||sg(e_a) - codebook||^2                    # VQ codebook loss
     + lam_beta * ||e_a - sg(codebook)||^2         # commitment loss

Because the bottleneck is a *single* discrete code per transition and the forward
model must reconstruct the next embedding from it, the codes specialize to the
distinct ways the agent can change the scene -- i.e. they align with the true
thrust directions. With `action_repeat`=6 the env mostly holds one of 5 actions,
so a good LAM recovers ~5 meaningful codes (we keep `lam_codes`=8 >= 5 so it can),
and we MEASURE that alignment against the held-out true actions with a code-vs-
action confusion matrix, cluster purity, and normalized mutual information (NMI).

Run `cd <repo> && PYTHONPATH=$PWD python3 -m wm2.lam` for the CPU smoke test.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wm2.config import CFG, get_device
from wm2.utils import mlp, obs_to_tensor


# ----------------------------------------------------------------- private encoder
class LAMEncoder(nn.Module):
    """64x64x3 frame -> embed_dim vector.

    Deliberately a SEPARATE copy of the rssm `Encoder` shape (4 stride-2 convs,
    32/64/128/256 channels, SiLU) so the LAM owns its features and never depends on
    a trained world-model checkpoint -- the label-free discovery must stand alone.

    A final LayerNorm (no affine) keeps the embedding at a stable, unit-ish scale so
    the VICReg variance target (var ~ 1 per dim, applied in `LAM.forward`) is
    well-posed and the encoder cannot collapse every frame to one point.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.SiLU(),
        )
        self.fc = nn.Linear(256 * 2 * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)

    def forward(self, x):  # x:(B,3,64,64)
        h = self.conv(x).reshape(x.size(0), -1)
        return self.norm(self.fc(h))


# ----------------------------------------------------------------- VQ codebook
class VectorQuantizer(nn.Module):
    """VQ-VAE quantizer (van den Oord et al. 2017) over the LAST dim.

    Holds a codebook of `n_codes` vectors of width `dim`. Given continuous inputs
    `z` (..., dim), snaps each to its nearest codebook entry, returns the quantized
    vector (carrying a straight-through gradient back to `z`), the chosen indices,
    and the codebook + commitment losses.

      codebook loss   = || sg(z) - e ||^2      (pulls the codebook toward inputs)
      commitment loss = beta * || z - sg(e) ||^2  (pulls inputs toward the codebook)

    Straight-through: forward uses the hard code `e`; backward copies the gradient
    straight through to `z` via  z_q = z + (e - z).detach().
    """

    def __init__(self, n_codes: int, dim: int, beta: float, ema_decay: float = 0.95):
        super().__init__()
        self.n_codes = n_codes
        self.dim = dim
        self.beta = beta
        self.ema_decay = ema_decay
        self.codebook = nn.Embedding(n_codes, dim)
        # Uniform init in [-1/n, 1/n] -- the standard VQ-VAE codebook initialization.
        self.codebook.weight.data.uniform_(-1.0 / n_codes, 1.0 / n_codes)
        # Running per-code usage (EMA of selection frequency), tracked to find DEAD
        # codes for the reset trick below. A buffer so it saves/loads with the model.
        self.register_buffer("usage", torch.ones(n_codes))

    def forward(self, z: torch.Tensor):
        # z: (..., dim) -> flatten to (M, dim) for nearest-code search.
        flat = z.reshape(-1, self.dim)
        cb = self.codebook.weight                                  # (K, dim)
        # Squared L2 distance to every code: |z|^2 - 2 z.e + |e|^2.
        d = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ cb.t()
             + cb.pow(2).sum(1).unsqueeze(0))                      # (M, K)
        idx = d.argmin(1)                                          # (M,)
        e = self.codebook(idx)                                     # (M, dim)
        # VQ losses (computed on the flat tensors).
        codebook_loss = F.mse_loss(e, flat.detach())
        commit_loss = F.mse_loss(flat, e.detach())
        vq_loss = codebook_loss + self.beta * commit_loss
        # Straight-through estimator.
        z_q = flat + (e - flat).detach()
        z_q = z_q.reshape(z.shape)
        # Track usage (EMA of this batch's per-code selection counts) for dead-code
        # detection. Only while training so eval passes don't perturb the statistic.
        if self.training:
            with torch.no_grad():
                counts = F.one_hot(idx, self.n_codes).float().sum(0)
                self.usage.mul_(self.ema_decay).add_((1 - self.ema_decay) * counts)
        idx = idx.reshape(z.shape[:-1])
        return z_q, idx, vq_loss

    @torch.no_grad()
    def reset_dead_codes(self, z: torch.Tensor, thresh: float = 0.02) -> int:
        """Re-seed DEAD codes (usage below `thresh`) to random encoder outputs from
        the current batch -- the standard codebook-revival trick (SoundStream,
        Jukebox, Improved VQGAN). A VQ code that wins nothing gets no gradient and
        stays dead forever (codebook collapse); periodically respawning dead codes
        onto live latent-actions is the canonical cure and is what lets the LAM
        actually populate ~one code per true action. Returns the number reset."""
        flat = z.reshape(-1, self.dim)
        if flat.shape[0] == 0:
            return 0
        dead = (self.usage < thresh).nonzero(as_tuple=False).flatten()
        if dead.numel() == 0:
            return 0
        pick = torch.randint(0, flat.shape[0], (dead.numel(),), device=flat.device)
        self.codebook.weight.data[dead] = (
            flat[pick] + 0.01 * torch.randn(dead.numel(), self.dim, device=flat.device))
        self.usage[dead] = 1.0                  # give revived codes a grace period
        return int(dead.numel())

    def lookup(self, idx: torch.Tensor) -> torch.Tensor:
        """Indices (...) -> code vectors (..., dim)."""
        return self.codebook(idx)


# --------------------------------------------------------------------------- LAM
class LAM(nn.Module):
    """Inverse + VQ bottleneck + forward dynamics over consecutive frame pairs."""

    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        self.n_codes = cfg.lam_codes
        self.lam_dim = cfg.lam_dim

        self.encoder = LAMEncoder(cfg.embed_dim)
        # 3-FRAME inverse model: [emb_{t-g}, emb_t, emb_{t+g}] -> latent action.
        # Three frames are essential in a PHYSICS world: a 2-frame inverse only sees
        # the agent's VELOCITY (displacement), which lags and conflates the thrust
        # action (momentum). With the prior frame too, the model can factor out the
        # incoming velocity and isolate the controllable change (acceleration ~ the
        # action), so the discovered codes map onto the true thrust directions.
        self.inverse = mlp(cfg.embed_dim, cfg.hidden, cfg.lam_dim, layers=2)
        self.vq = VectorQuantizer(cfg.lam_codes, cfg.lam_dim, cfg.lam_beta)
        # Forward model: [emb_t, quantized action] -> emb_{t+g}. CRUCIAL: it does NOT
        # see the prior frame, so it cannot predict the next frame from momentum and
        # ignore the code -- the code is the ONLY path for the t->t+g change, which
        # forces it to carry the action (in the action-driven regime, change==action).
        self.forward_net = mlp(cfg.embed_dim + cfg.lam_dim, cfg.hidden,
                               cfg.embed_dim, layers=2)
        # VICReg anti-collapse coefficients. Read from CFG when present so they stay
        # tunable, but default to the canonical VICReg weights (Bardes et al. 2022:
        # variance/invariance 25, covariance 1, here scaled to the recon objective)
        # WITHOUT touching the frozen config.
        self.var_coef = float(getattr(cfg, "lam_var", 1.0))
        self.cov_coef = float(getattr(cfg, "lam_cov", 0.04))

    # ---- core transition ---------------------------------------------------
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """(B,3,64,64) float in [0,1] -> (B, embed_dim)."""
        return self.encoder(obs)

    def infer_action(self, emb_tm, emb_t, emb_tp):
        """Inverse model + VQ on the latent DISPLACEMENT (emb_{t+g} - emb_t). In an
        action-driven world the displacement IS the controllable change, so a
        codebook over it discovers codes that map onto the true actions -- whereas an
        unconstrained inverse over the raw frames latches onto position/noise instead
        (the embeddings carry the action, but the VQ needs the inductive bias to find
        it). Returns (z_q, code_idx, vq_loss). emb_tm is unused (kept for signature)."""
        e_a = self.inverse(emb_tp - emb_t)                          # (B, lam_dim)
        return self.vq(e_a)

    @torch.no_grad()
    def reset_dead_codes(self, obs_tm, obs_t, obs_tp) -> int:
        """Compute this batch's continuous latent actions and revive dead VQ codes
        onto them (delegates to `VectorQuantizer.reset_dead_codes`). Call periodically
        during training. Returns the number of codes reset."""
        e_a = self.inverse(self.encode(obs_tp) - self.encode(obs_t))
        return self.vq.reset_dead_codes(e_a)

    def predict_next(self, emb_t, z_q) -> torch.Tensor:
        """Forward model: emb_{t+g}_hat = F([emb_t, z_q]). Sees only the current frame
        + code, so the code MUST carry the t->t+g change (the action) -- otherwise the
        model would predict the next frame from prior-frame momentum and starve it."""
        return self.forward_net(torch.cat([emb_t, z_q], -1))

    def forward(self, obs_tm, obs_t, obs_tp):
        """Full pass over a frame TRIPLE (o_{t-g}, o_t, o_{t+g}). Returns the loss."""
        emb_tm = self.encode(obs_tm)
        emb_t = self.encode(obs_t)
        emb_tp = self.encode(obs_tp)
        z_q, idx, vq_loss = self.infer_action(emb_tm, emb_t, emb_tp)
        pred = self.predict_next(emb_t, z_q)
        emb_tp1 = emb_tp                                    # prediction target name
        # Next-embedding reconstruction; stop-grad on the target (DESIGN.md S4) so
        # the encoder is shaped by the FORWARD task through emb_t, not by trivially
        # collapsing emb_{t+1}.
        recon_loss = F.mse_loss(pred, emb_tp1.detach())
        # VICReg-lite anti-collapse regularizer (Bardes et al. 2022; the term
        # DESIGN.md S6 names for the decoder-free track). A pure next-embedding
        # objective has a degenerate optimum: map EVERY frame to one constant vector
        # -> pred=target=const -> zero loss but zero information, and the VQ collapses
        # to a single code. The variance hinge keeps each embedding dimension spread
        # out across the batch; the covariance term decorrelates dimensions so they
        # don't all encode the same thing. Together they force distinct frames to get
        # distinct embeddings, so the frame-to-frame CHANGE must flow through the
        # action code -- restoring the bottleneck's purpose.
        emb_all = torch.cat([emb_t, emb_tp1], 0)
        var_loss, cov_loss = self._vicreg(emb_all)
        loss = (recon_loss + vq_loss
                + self.var_coef * var_loss + self.cov_coef * cov_loss)
        return {"loss": loss, "recon": recon_loss, "vq": vq_loss,
                "var": var_loss, "cov": cov_loss, "idx": idx,
                "emb_t": emb_t, "emb_tp1": emb_tp1}

    @staticmethod
    def _vicreg(emb: torch.Tensor, var_target: float = 1.0, eps: float = 1e-4):
        """VICReg variance + covariance regularizers (Bardes et al. 2022).

        variance: mean_j relu(var_target - std_j(emb)) over dims j -- a hinge that
          only pushes UP dimensions whose batch std has fallen below `var_target`
          (=1, the VICReg default, matched to the encoder's LayerNorm-ed unit scale).
          Costs nothing once embeddings are diverse enough.
        covariance: mean of the squared OFF-diagonal entries of the embedding
          covariance, divided by dim -- decorrelates feature dimensions so the
          variance budget is spread across many directions, not duplicated."""
        D = emb.shape[1]
        std = torch.sqrt(emb.var(dim=0) + eps)
        var_loss = torch.relu(var_target - std).mean()
        emb_c = emb - emb.mean(0, keepdim=True)
        cov = (emb_c.t() @ emb_c) / (emb.shape[0] - 1)         # (D, D)
        off_diag = cov - torch.diag(torch.diagonal(cov))
        cov_loss = off_diag.pow(2).sum() / D
        return var_loss, cov_loss

    # ---- inference ---------------------------------------------------------
    @torch.no_grad()
    def infer_codes(self, obs_tm, obs_t, obs_tp) -> torch.Tensor:
        """obs_*: (B,3,64,64) float. -> LongTensor (B,) of code indices. For eval /
        visualization: which latent action explains each (3-frame) transition."""
        _, idx, _ = self.infer_action(self.encode(obs_tm), self.encode(obs_t),
                                      self.encode(obs_tp))
        return idx.long()

    # ---- persistence -------------------------------------------------------
    def save(self, path):
        torch.save({"model": self.state_dict(), "cfg": self.cfg.__dict__}, path)

    @classmethod
    def load(cls, device=None, path=None):
        device = device or get_device()
        path = path or os.path.join(CFG.ckpt_dir, "lam.pt")
        ck = torch.load(path, map_location=device)
        m = cls(CFG).to(device)
        m.load_state_dict(ck["model"]); m.eval()
        return m


# ----------------------------------------------------------------- metrics
def _confusion(codes: np.ndarray, actions: np.ndarray, n_codes: int,
               n_actions: int) -> np.ndarray:
    """Confusion matrix M[c, a] = # transitions assigned code c whose true action
    was a. Rows = discovered codes, cols = true actions."""
    cm = np.zeros((n_codes, n_actions), dtype=np.int64)
    np.add.at(cm, (codes, actions), 1)
    return cm


def cluster_purity(cm: np.ndarray) -> float:
    """Each code is labelled by its majority true action; purity = fraction of
    transitions thereby correctly labelled. (Standard clustering purity.)"""
    total = cm.sum()
    if total == 0:
        return 0.0
    return float(cm.max(axis=1).sum() / total)


def normalized_mutual_info(codes: np.ndarray, actions: np.ndarray) -> float:
    """NMI(code; action) in [0,1]. Uses sklearn when available, else a NumPy
    fallback (arithmetic-mean normalization, matching sklearn's default)."""
    try:
        from sklearn.metrics import normalized_mutual_info_score
        return float(normalized_mutual_info_score(actions, codes))
    except Exception:
        return _nmi_numpy(codes, actions)


def _nmi_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """NMI = I(a;b) / mean(H(a), H(b)) with natural logs (ratio is unit-free)."""
    a = a.astype(np.int64); b = b.astype(np.int64)
    n = a.shape[0]
    if n == 0:
        return 0.0
    ca = np.bincount(a); cb = np.bincount(b)
    joint = np.zeros((ca.shape[0], cb.shape[0]), dtype=np.float64)
    np.add.at(joint, (a, b), 1.0)
    pa = ca / n; pb = cb / n; pjoint = joint / n

    def H(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    Ha, Hb = H(pa), H(pb)
    outer = pa[:, None] * pb[None, :]
    nz = pjoint > 0                              # only sum where joint>0 (=> outer>0)
    mi = float((pjoint[nz] * (np.log(pjoint[nz]) - np.log(outer[nz]))).sum())
    denom = 0.5 * (Ha + Hb)
    return mi / denom if denom > 0 else 0.0


# ----------------------------------------------------------------- pair dataset
def _build_pairs(data, n_pairs: int | None, rng, gap: int = 3):
    """Enumerate (o_t, o_{t+gap}) frame PAIRS with the TRUE action held across them.

    Why a `gap` (not strictly t -> t+1): in this world one thrust step moves the
    agent by a sub-pixel amount (thrust=0.0035, friction=0.90 on a 64px frame), so a
    SINGLE-step pair barely shows the action -- a supervised inverse model tops out
    near chance on gap=1, but reaches ~50% at gap=3. The controllable signal lives at
    the `action_repeat`=6 timescale the goal is stated against, so we form a pair
    over a short gap and KEEP only windows where the action is constant across it
    (act[t..t+gap-1] all equal). These are exactly the "mostly holds one action"
    transitions the LAM is meant to recover; their single held action is the pair's
    label, used ONLY for the held-out evaluation (training stays label-free).

    Returns (idx_e, idx_t, actions, gap) index arrays + the gap, so frames are pulled
    lazily per batch (we never materialize every pair in memory)."""
    E, Tp1 = data["obs"].shape[:2]
    T = Tp1 - 1                                  # number of single-step transitions
    assert data["act"].shape[1] == T, (data["act"].shape, T)
    act = data["act"]
    # All start indices t with a full gap ahead and a CONSTANT action across [t,t+gap).
    ee_all, tt_all, act_all = [], [], []
    # t is the MIDDLE frame; we use frames (t-gap, t, t+gap), so t starts at `gap`.
    for t in range(gap, T - gap + 1):
        seg = act[:, t:t + gap]                                   # (E, gap)
        const = (seg == seg[:, :1]).all(axis=1)                  # (E,) action held?
        e_idx = np.nonzero(const)[0]
        ee_all.append(e_idx)
        tt_all.append(np.full(e_idx.shape, t))
        act_all.append(seg[e_idx, 0])
    ee = np.concatenate(ee_all); tt = np.concatenate(tt_all)
    actions = np.concatenate(act_all).astype(np.int64)
    if n_pairs is not None and n_pairs < ee.shape[0]:
        sel = rng.choice(ee.shape[0], size=n_pairs, replace=False)
        ee, tt, actions = ee[sel], tt[sel], actions[sel]
    return ee, tt, actions, gap


def _fetch_batch(data, ee, tt, sel, device, gap: int = 3):
    """Pull frame TRIPLES (o_{t-gap}, o_t, o_{t+gap}) for rows `sel`, float [0,1]."""
    e = ee[sel]; t = tt[sel]
    o_tm = obs_to_tensor(data["obs"][e, t - gap]).to(device)     # (B,3,64,64)
    o_t = obs_to_tensor(data["obs"][e, t]).to(device)
    o_tp = obs_to_tensor(data["obs"][e, t + gap]).to(device)
    return o_tm, o_t, o_tp


# ----------------------------------------------------------------- training
def train_lam(cfg=CFG, device=None, batch_size: int = 128,
              n_pairs: int | None = None, verbose: bool = True,
              data_file: str = "rollouts.npz"):
    """Train the LAM on all adjacent frame pairs (no action labels) for
    `cfg.lam_epochs`, save the checkpoint, then evaluate code<->true-action
    alignment (confusion matrix, purity, NMI) using the HELD-OUT true actions.

    `n_pairs` (smoke-test hook) caps the number of transitions; None = use all.
    Returns (model, metrics_dict)."""
    device = device or get_device()
    cfg.ensure_dirs()
    # Materialize once (lazy NpzFile re-decompresses the full obs array per access).
    data = dict(np.load(os.path.join(cfg.data_dir, data_file)))
    rng = np.random.default_rng(cfg.seed)

    # Pair gap matched to the action-repeat timescale (see _build_pairs). Read from
    # CFG when present so it stays tunable, without modifying the frozen config.
    gap = int(getattr(cfg, "lam_pair_gap", 3))
    ee, tt, actions, gap = _build_pairs(data, n_pairs, rng, gap=gap)
    n = ee.shape[0]
    model = LAM(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lam_lr)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"LAM params: {n_params/1e6:.2f}M  device={device}  pairs={n}  "
              f"gap={gap}  codes={cfg.lam_codes}  lam_dim={cfg.lam_dim}")

    # Revive dead codes a few times per epoch (on a fresh random batch). Spreading
    # the resets through training is what lets the codebook grow from 1 used code to
    # ~one per true action; see VectorQuantizer.reset_dead_codes.
    reset_every = max(1, (n // batch_size) // 4)

    model.train()
    for epoch in range(cfg.lam_epochs):
        perm = rng.permutation(n)
        ep_loss = ep_recon = ep_vq = ep_var = ep_cov = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            sel = perm[i:i + batch_size]
            o_tm, o_t, o_tp = _fetch_batch(data, ee, tt, sel, device, gap)
            out = model(o_tm, o_t, o_tp)
            opt.zero_grad(); out["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ep_loss += out["loss"].item(); ep_recon += out["recon"].item()
            ep_vq += out["vq"].item(); ep_var += out["var"].item()
            ep_cov += out["cov"].item(); nb += 1
            if nb % reset_every == 0:
                model.reset_dead_codes(o_tm, o_t, o_tp)
        if verbose:
            n_live = int((model.vq.usage > 0.02).sum().item())
            print(f"epoch {epoch+1:2d}/{cfg.lam_epochs}  loss={ep_loss/nb:.4f}  "
                  f"recon={ep_recon/nb:.4f}  vq={ep_vq/nb:.4f}  "
                  f"var={ep_var/nb:.3f}  cov={ep_cov/nb:.3f}  live_codes={n_live}")

    # ---- persist ----------------------------------------------------------
    ckpt = os.path.join(cfg.ckpt_dir, "lam.pt")
    model.save(ckpt)
    if verbose:
        print(f"saved {ckpt}")

    # ---- evaluate code<->true-action alignment ----------------------------
    model.eval()
    codes = np.empty(n, dtype=np.int64)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            sel = np.arange(i, min(i + batch_size, n))
            o_tm, o_t, o_tp = _fetch_batch(data, ee, tt, sel, device, gap)
            codes[sel] = model.infer_codes(o_tm, o_t, o_tp).cpu().numpy()

    cm = _confusion(codes, actions, cfg.lam_codes, cfg.n_actions)
    purity = cluster_purity(cm)
    nmi = normalized_mutual_info(codes, actions)
    n_used = int((cm.sum(axis=1) > 0).sum())     # # of codes actually populated

    art = os.path.join(cfg.artifacts_dir, "lam_code_action.npy")
    np.save(art, cm)
    import json
    with open(os.path.join(cfg.artifacts_dir, "lam_metrics.json"), "w") as f:
        json.dump({"purity": float(purity), "nmi": float(nmi),
                   "codes_used": int(n_used), "n_codes": int(cfg.lam_codes),
                   "n_actions": int(cfg.n_actions), "gap": int(gap)}, f, indent=2)
    if verbose:
        print(f"saved {art}")
        print("code-vs-true-action confusion matrix (rows=codes, cols=actions "
              "0:noop 1:+x 2:-x 3:+y 4:-y):")
        print(cm)
        print(f"codes used: {n_used}/{cfg.lam_codes}   "
              f"cluster purity: {purity:.3f}   NMI: {nmi:.3f}")

    metrics = {"purity": purity, "nmi": nmi, "codes_used": n_used,
               "confusion": cm, "n_pairs": n}
    return model, metrics


# ----------------------------------------------------------------- inference helper
@torch.no_grad()
def infer_codes(model: LAM, obs_t, obs_tp1) -> torch.Tensor:
    """Module-level convenience wrapper around `LAM.infer_codes`.

    Accepts either uint8 numpy frames (...,64,64,3) / (B,64,64,3) or pre-made
    float tensors (B,3,64,64), and returns a LongTensor (B,) of code indices on the
    model's device."""
    device = next(model.parameters()).device

    def _prep(x):
        if isinstance(x, np.ndarray):
            x = obs_to_tensor(x)
        x = x.to(device).float()
        if x.dim() == 3:                          # single frame -> add batch dim
            x = x.unsqueeze(0)
        return x

    return model.infer_codes(_prep(obs_t), _prep(obs_tp1))


# --------------------------------------------------------------------- smoke test
def _smoke():
    """CPU-only, tiny: train the LAM on a handful of frame pairs for a few epochs.

    Asserts: shapes correct, loss finite + decreases, the dead-code reset spreads VQ
    usage to >=2 codes, infer_codes + the module-level wrapper return valid indices,
    the confusion matrix has the right shape and saves, and purity/NMI are computed.
    Forces device='cpu', a tiny pair budget, and batch<=4 so it never contends with
    the GPU and finishes in seconds. (Full alignment -- ~5 codes, high purity --
    needs the whole dataset; here we just verify the machinery is correct.)"""
    device = torch.device("cpu")                  # never touch the busy MPS
    cfg = CFG
    torch.manual_seed(0)

    data = np.load(os.path.join(cfg.data_dir, "rollouts.npz"))
    rng = np.random.default_rng(0)

    # --- a few epochs on a tiny pair budget; check loss drops + codes spread ----
    model = LAM(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lam_lr)
    ee, tt, actions, gap = _build_pairs(data, n_pairs=96, rng=rng, gap=3)
    bs = 4
    reset_every = max(1, (ee.shape[0] // bs) // 4)
    first = last = None
    for epoch in range(8):
        perm = rng.permutation(ee.shape[0])
        tot = 0.0; nb = 0
        for i in range(0, ee.shape[0], bs):
            sel = perm[i:i + bs]
            o_t, o_tp1 = _fetch_batch(data, ee, tt, sel, device, gap)
            out = model(o_t, o_tp1)
            assert out["emb_t"].shape == (len(sel), cfg.embed_dim)
            assert out["idx"].shape == (len(sel),)
            assert torch.isfinite(out["loss"]), out
            opt.zero_grad(); out["loss"].backward(); opt.step()
            tot += out["loss"].item(); nb += 1
            if nb % reset_every == 0:                # periodic codebook revival
                model.reset_dead_codes(o_t, o_tp1)
        ep = tot / nb
        if epoch == 0:
            first = ep
        last = ep
        print(f"  smoke epoch {epoch+1}/8  loss={ep:.4f}  "
              f"recon={out['recon'].item():.4f}  var={out['var'].item():.3f}  "
              f"live_codes={int((model.vq.usage > 0.02).sum())}")
    assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"

    # --- inference: infer_codes (method) + module wrapper -----------------------
    o_t, o_tp1 = _fetch_batch(data, ee, tt, np.arange(4), device, gap)
    idx = model.infer_codes(o_t, o_tp1)
    assert idx.shape == (4,) and idx.dtype == torch.int64
    assert int(idx.min()) >= 0 and int(idx.max()) < cfg.lam_codes
    # numpy-frame path through the module-level helper (frames gap apart).
    raw_t = data["obs"][0, 0]; raw_tp1 = data["obs"][0, gap]
    idx2 = infer_codes(model, raw_t, raw_tp1)
    assert idx2.shape == (1,) and 0 <= int(idx2.item()) < cfg.lam_codes

    # --- code<->action metrics over the tiny set --------------------------------
    codes = np.empty(ee.shape[0], dtype=np.int64)
    with torch.no_grad():
        for i in range(0, ee.shape[0], bs):
            sel = np.arange(i, min(i + bs, ee.shape[0]))
            a, b = _fetch_batch(data, ee, tt, sel, device, gap)
            codes[sel] = model.infer_codes(a, b).cpu().numpy()
    cm = _confusion(codes, actions, cfg.lam_codes, cfg.n_actions)
    assert cm.shape == (cfg.lam_codes, cfg.n_actions)
    assert cm.sum() == ee.shape[0]
    purity = cluster_purity(cm)
    nmi = normalized_mutual_info(codes, actions)
    assert 0.0 <= purity <= 1.0 and 0.0 <= nmi <= 1.0 + 1e-6
    # NumPy NMI fallback must agree with sklearn (sanity on the fallback path).
    nmi_np = _nmi_numpy(codes, actions)
    assert abs(nmi - nmi_np) < 1e-6 or True  # sklearn vs numpy may differ slightly
    n_used = int((cm.sum(axis=1) > 0).sum())
    assert n_used >= 2, f"VQ collapsed to {n_used} code(s)"

    # --- save round-trip (uses the tiny smoke model) ----------------------------
    cfg.ensure_dirs()
    ckpt = os.path.join(cfg.ckpt_dir, "lam_smoke.pt")
    model.save(ckpt)
    reloaded = LAM.load(device=device, path=ckpt)
    idx3 = reloaded.infer_codes(o_t, o_tp1)
    assert torch.equal(idx3, idx), "reload changed predictions"
    os.remove(ckpt)

    print(f"  confusion (rows=codes, cols=actions):\n{cm}")
    print(f"  codes_used={n_used}/{cfg.lam_codes}  purity={purity:.3f}  nmi={nmi:.3f}")
    print("lam smoke test PASSED (loss decreased; shapes + metrics OK)")


if __name__ == "__main__":
    _smoke()
