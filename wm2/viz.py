"""Qualitative visualizations for wm2 (DESIGN.md sec 8).

This is the *looking* part of evaluation -- the figures and GIFs that make the
RSSM's behaviour legible. It is the wm2 analogue of the baseline's `wm/dream.py`
+ `wm/viz.py`, ported to the discrete-latent Recurrent State-Space Model:

  * dream_compare   -- "real | RSSM reconstruction | open-loop dream" GIF +
                       montage on a held-out episode. Recon uses the POSTERIOR
                       (`observe`, sees every frame); the dream encodes ONLY frame
                       0 and then rolls `img_step` from the TRUE action sequence
                       (no re-grounding), so its gap from the recon column is pure
                       *dynamics* error -- the long-horizon drift the literature
                       flags (PlaNet/Dreamer; MirageLSD 2025).
  * drift_curve     -- open-loop dream pixel-MSE vs horizon, averaged over several
                       held-out episodes (the long-horizon-drift metric).
  * latent_map      -- PCA (first 2 PCs) of RSSM features colored by the agent's
                       true x and y -- interpretability that the world's *state*
                       (position) is decodable from the learned latent.
  * code_action_matrix -- if a Latent Action Model checkpoint exists, the
                       code-vs-true-action confusion matrix (Genie controllability);
                       skipped gracefully otherwise.
  * free_dream      -- pure hallucination: encode one real frame, then drive
                       `img_step` with a scripted/looping action pattern (no ground
                       truth -- the world model IS the world).

`main()` runs all of the above against a TRAINED `WorldModel.load()`, guarding each
artifact in try/except so one failure doesn't sink the rest. `_smoke()` exercises
the dream/decode plumbing on CPU with a FRESH random model and a short synthetic
rollout (since no trained checkpoint exists yet) and writes one tiny GIF.

Refs: Ha & Schmidhuber 2018 (the baseline this mirrors); Hafner et al. PlaNet /
DreamerV2 / DreamerV3 (RSSM, open-loop imagination); Bruce et al. Genie 1-3
(latent actions); Decart MirageLSD 2025 (open-loop drift framing).
"""
from __future__ import annotations

import os

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")                      # headless: never needs a display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from wm2.config import CFG, get_device
from wm2.env import MultiBody2DEnv, RepeatRandomPolicy, rollout
from wm2.rssm import WorldModel
from wm2.utils import obs_to_tensor, tensor_to_obs

SEP_W = 2                                   # white separator width between columns


# ============================================================ core dream/recon path
@torch.no_grad()
def rssm_reconstruct(wm: WorldModel, obs_uint8: np.ndarray, acts: np.ndarray,
                     device) -> np.ndarray:
    """POSTERIOR reconstruction of EVERY frame (the renderer ceiling).

    obs:(T+1,H,W,3) uint8, acts:(T,) int. Runs `observe` (which sees every frame
    through the posterior) and decodes each feature back to pixels. This isolates
    encoder+decoder error: the best the model could draw given the true frames, no
    open-loop dynamics involved. Returns (T+1,H,W,3) uint8.
    """
    x = obs_to_tensor(obs_uint8).to(device).unsqueeze(0)         # (1,T+1,3,H,W)
    a = torch.from_numpy(np.asarray(acts)).long().to(device).unsqueeze(0)  # (1,T)
    L = x.shape[1]
    # `observe` consumes act of length L (action TAKEN at each step). We have T=L-1
    # real actions; pad the final step with a no-op (0) -- it only affects the
    # latent CARRIED PAST the last frame, which we never decode here.
    a_pad = F.pad(a, (0, L - a.shape[1]))                        # (1,L)
    out = wm.observe(x, a_pad)
    recon = wm.decode(out["feat"].reshape(L, -1))               # (L,3,H,W)
    return tensor_to_obs(recon)


@torch.no_grad()
def rssm_dream(wm: WorldModel, obs0_uint8: np.ndarray, acts: np.ndarray,
               device) -> np.ndarray:
    """OPEN-LOOP dream: ground on frame 0 only, then hallucinate from actions.

    Encodes ONLY obs0 to a posterior state (one filtered step from the zero state
    with a no-op previous action), then rolls `img_step` using the TRUE action
    sequence -- z is sampled from the PRIOR every step, never re-grounded in pixels.
    obs0:(H,W,3) uint8, acts:(T,) int. Returns (T+1,H,W,3) uint8 (includes frame 0).

    Action alignment mirrors `observe`: `acts[t]` is the action taken AT step t that
    leads to frame t+1, so we feed it to advance state_t -> state_{t+1}.
    """
    x0 = obs_to_tensor(obs0_uint8).to(device).unsqueeze(0)      # (1,3,H,W)
    embed0 = wm.encoder(x0)
    state0 = wm.initial(1, device)
    prev_a = torch.zeros(1, wm.n_actions, device=device)        # a_{-1}=0
    state, _, _ = wm.obs_step(state0, prev_a, embed0)           # posterior on frame 0
    a = torch.from_numpy(np.asarray(acts)).long().to(device)    # (T,)

    feats = [wm.get_feat(state)]
    for t in range(a.shape[0]):
        a_oh = F.one_hot(a[t:t + 1], wm.n_actions).float()      # (1,n_actions)
        state, _ = wm.img_step(state, a_oh, sample=True)        # SAMPLE the prior
        feats.append(wm.get_feat(state))
    frames = wm.decode(torch.cat(feats, 0))                     # (T+1,3,H,W)
    return tensor_to_obs(frames)


def _stack_columns(cols: list[np.ndarray]) -> np.ndarray:
    """Horizontally stitch same-height frames with thin white separators."""
    H = cols[0].shape[0]
    sep = np.full((H, SEP_W, 3), 255, np.uint8)
    out = [cols[0]]
    for c in cols[1:]:
        out += [sep, c]
    return np.hstack(out)


# ===================================================================== dream_compare
def dream_compare(wm: WorldModel, device, ep_len: int = 80, seed: int = 9999,
                  out_prefix: str | None = None) -> float:
    """3-column 'real | RSSM reconstruction | open-loop dream' GIF + montage PNG.

    Mirrors `wm/dream.py:comparison`, but for the RSSM: the middle column is the
    posterior recon (renderer ceiling), the right column is the frame-0-grounded
    open-loop dream. Returns the mean dream pixel MSE and prints it.
    """
    cfg = wm.cfg
    out_prefix = out_prefix or os.path.join(cfg.artifacts_dir, "dream_compare")
    env = MultiBody2DEnv(cfg, seed=seed)
    pol = RepeatRandomPolicy(cfg, seed=seed + 1)
    obs, acts, *_ = rollout(env, pol, ep_len=ep_len)            # (T+1,H,W,3),(T,)

    recon = rssm_reconstruct(wm, obs, acts, device)            # (T+1,H,W,3)
    dream = rssm_dream(wm, obs[0], acts, device)               # (T+1,H,W,3)

    frames = [_stack_columns([obs[i], recon[i], dream[i]]) for i in range(len(obs))]
    gif = f"{out_prefix}.gif"
    imageio.mimsave(gif, frames, fps=20, loop=0)
    sel = list(range(0, len(obs), max(1, len(obs) // 6)))[:6]
    imageio.imwrite(f"{out_prefix}_montage.png", np.vstack([frames[i] for i in sel]))

    err = float(np.mean((dream.astype(np.float32) - obs.astype(np.float32)) ** 2))
    print(f"[dream_compare] saved {gif}")
    print(f"[dream_compare] saved {out_prefix}_montage.png")
    print(f"[dream_compare] mean dream pixel MSE={err:.2f} "
          f"(cols: real | rssm-recon | open-loop dream)")
    return err


# ====================================================================== drift_curve
def drift_curve(wm: WorldModel, device, n_eps: int = 5, ep_len: int = 80,
                seed: int = 4242, out: str | None = None) -> np.ndarray:
    """Open-loop dream pixel-MSE vs horizon, averaged over `n_eps` held-out episodes.

    Per episode: dream from frame 0, compute per-frame MSE against the true frames,
    then average curves across episodes. A rising curve == compounding open-loop
    drift. Saves drift_curve.png and returns the mean curve (length T+1).
    """
    cfg = wm.cfg
    out = out or os.path.join(cfg.artifacts_dir, "drift_curve.png")
    curves = []
    for k in range(n_eps):
        env = MultiBody2DEnv(cfg, seed=seed + k)
        pol = RepeatRandomPolicy(cfg, seed=seed + 1000 + k)
        obs, acts, *_ = rollout(env, pol, ep_len=ep_len)
        dream = rssm_dream(wm, obs[0], acts, device)
        mse = ((dream.astype(np.float32) - obs.astype(np.float32)) ** 2
               ).mean(axis=(1, 2, 3))                            # (T+1,)
        curves.append(mse)
    curves = np.stack(curves, 0)                                # (n_eps, T+1)
    mean, std = curves.mean(0), curves.std(0)
    horizon = np.arange(curves.shape[1])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(horizon, mean, "-", color="#c53030", lw=2, label="open-loop dream")
    ax.fill_between(horizon, mean - std, mean + std, color="#c53030", alpha=0.18)
    ax.set_xlabel("dream horizon (steps from the single real frame)")
    ax.set_ylabel("per-frame pixel MSE")
    ax.set_title(f"Long-horizon open-loop drift (mean +/- std over {n_eps} episodes)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[drift_curve] saved {out}  "
          f"(MSE: t=0 {mean[0]:.1f} -> t={len(mean)-1} {mean[-1]:.1f})")
    return mean


# ======================================================================= latent_map
@torch.no_grad()
def latent_map(wm: WorldModel, device, n_eps: int = 6, ep_len: int = 80,
               seed: int = 314, out: str | None = None) -> None:
    """PCA scatter of RSSM features colored by the agent's true (x, y).

    Encodes several held-out episodes to posterior features via `observe`, projects
    them to their first two principal components (falling back to the first two raw
    feature dims if scikit-learn is unavailable), and colors each point by the agent
    body's true x (left panel) and y (right panel). Tight colour gradients == the
    world's *state* is linearly laid out in the learned latent. Saves latent_map.png.
    """
    cfg = wm.cfg
    out = out or os.path.join(cfg.artifacts_dir, "latent_map.png")
    feats_all, xy_all = [], []
    for k in range(n_eps):
        env = MultiBody2DEnv(cfg, seed=seed + k)
        pol = RepeatRandomPolicy(cfg, seed=seed + 500 + k)
        obs, acts, pos, *_ = rollout(env, pol, ep_len=ep_len)   # pos:(T+1,N,2)
        x = obs_to_tensor(obs).to(device).unsqueeze(0)
        a = torch.from_numpy(acts).long().to(device).unsqueeze(0)
        a = F.pad(a, (0, x.shape[1] - a.shape[1]))
        feat = wm.observe(x, a)["feat"][0]                      # (T+1, feat_dim)
        feats_all.append(feat.cpu().numpy())
        xy_all.append(pos[:, 0, :])                            # body 0 = agent (x,y)
    feats = np.concatenate(feats_all, 0)                        # (M, feat_dim)
    xy = np.concatenate(xy_all, 0)                              # (M, 2)

    # First two principal components (zero-meaned); fall back to raw dims 0,1.
    method = "PCA"
    try:
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2).fit_transform(feats)
    except Exception:
        method = "feat dims 0,1"
        proj = feats[:, :2] - feats[:, :2].mean(0, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, k, name in zip(axes, (0, 1), ("x", "y")):
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=xy[:, k], s=8, alpha=0.7,
                        cmap="viridis")
        ax.set_xlabel(f"{method} 1"); ax.set_ylabel(f"{method} 2")
        ax.set_title(f"colored by true agent {name}")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=f"agent {name}")
    fig.suptitle("RSSM feature map: the agent's position is laid out in the latent")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[latent_map] saved {out}  ({feats.shape[0]} states, {method})")


# ================================================================ code_action_matrix
@torch.no_grad()
def code_action_matrix(device, n_eps: int = 8, ep_len: int = 80, seed: int = 77,
                       out: str | None = None) -> bool:
    """Code-vs-true-action confusion matrix for the Latent Action Model (Genie).

    LAZILY imports `wm2.lam` and loads `wm2_checkpoints/lam.pt`; if either the
    module or the checkpoint is absent, skips gracefully (returns False). For each
    held-out frame pair (o_t, o_{t+1}) the LAM infers a discrete code; we tabulate
    code vs the TRUE action that produced the transition and render a row-normalized
    heatmap. A near-permutation matrix == the label-free codes recovered the real
    thrust directions. Saves code_action_matrix.png.
    """
    cfg = CFG
    out = out or os.path.join(cfg.artifacts_dir, "code_action_matrix.png")
    # Render the confusion matrix that wm2.lam.train_lam computed + saved (this keeps
    # the LAM's 3-frame inference logic in one place; viz just draws the result).
    npy = os.path.join(cfg.artifacts_dir, "lam_code_action.npy")
    if not os.path.exists(npy):
        print(f"[code_action_matrix] no {npy}; skipping (run `wm2.train lam`)")
        return False
    conf = np.load(npy).astype(np.float64)                      # (n_codes, n_actions)
    n_codes, n_act = conf.shape
    row = conf.sum(1, keepdims=True)
    norm = conf / np.clip(row, 1.0, None)                       # P(action | code)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(norm, cmap="magma", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("true action"); ax.set_ylabel("inferred latent code")
    ax.set_xticks(range(n_act)); ax.set_yticks(range(n_codes))
    ax.set_title("LAM code -> true action (row-normalized)\nGenie-style label-free "
                 "controllability")
    for i in range(n_codes):
        for j in range(n_act):
            if norm[i, j] > 0.02:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if norm[i, j] < 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(action | code)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[code_action_matrix] saved {out}")
    return True


def _infer_codes(lam, o_t: torch.Tensor, o_tp1: torch.Tensor):
    """Best-effort: ask the LAM for the discrete code index of each frame pair.

    Tries a few plausible method names so this works against whatever interface
    `wm2.lam` ends up exposing (we are decoupled from it -- it is loaded lazily and
    may not exist yet). Returns a 1-D tensor/array of code indices.
    """
    for name in ("infer_code", "infer_codes", "encode_action", "code_indices",
                 "codes"):
        fn = getattr(lam, name, None)
        if callable(fn):
            return fn(o_t, o_tp1)
    # Fallback: a forward that returns a dict/tuple containing code indices.
    res = lam(o_t, o_tp1)
    if isinstance(res, dict):
        for key in ("code", "codes", "idx", "indices"):
            if key in res:
                return res[key]
    if isinstance(res, (tuple, list)):
        for r in res:
            if torch.is_tensor(r) and r.dtype in (torch.int64, torch.int32, torch.long):
                return r
    raise RuntimeError("could not extract code indices from wm2.lam model")


# ======================================================================== free_dream
@torch.no_grad()
def free_dream(wm: WorldModel, device, steps: int = 120, seed: int = 123,
               out: str | None = None) -> None:
    """Pure hallucination GIF: encode ONE real frame, then drive `img_step` with a
    scripted/looping action pattern and decode. No ground truth -- the world model
    is the world (the same demo as `wm/viz.py:free_running_dream`)."""
    cfg = wm.cfg
    out = out or os.path.join(cfg.artifacts_dir, "free_dream.gif")
    env = MultiBody2DEnv(cfg, seed=seed)
    obs0 = env.reset()
    # Scripted action loop: hold right, down, left, up for 20 steps each (coherent
    # motion the dream can express through the agent's thrust dynamics).
    pattern = []
    for a in (1, 3, 2, 4):                                      # +x, +y, -x, -y
        pattern += [a] * 20
    acts = np.array((pattern * (steps // len(pattern) + 1))[:steps], dtype=np.int64)

    frames = rssm_dream(wm, obs0, acts, device)                # (steps+1,H,W,3)
    imageio.mimsave(out, list(frames), fps=20, loop=0)
    sel = list(range(0, len(frames), max(1, len(frames) // 8)))[:8]
    montage = os.path.join(cfg.artifacts_dir, "free_dream_montage.png")
    imageio.imwrite(montage, np.hstack([frames[i] for i in sel]))
    print(f"[free_dream] saved {out}  ({steps} fully-hallucinated frames "
          f"from 1 real seed frame)")
    print(f"[free_dream] saved {montage}")


# ============================================================================= main
def main() -> None:
    """Run every visualization against a TRAINED world model.

    Each artifact is independently guarded so one failure (e.g. a missing LAM
    checkpoint) never sinks the rest. Prints exactly what was saved.
    """
    cfg = CFG
    cfg.ensure_dirs()
    device = get_device()
    print(f"[viz] device={device}")
    try:
        wm = WorldModel.load(device)
    except Exception as e:
        print(f"[viz] FATAL: could not load WorldModel ({e}). "
              f"Train it first: python -m wm2.train wm")
        return

    for label, fn in (
        ("dream_compare", lambda: dream_compare(wm, device)),
        ("drift_curve",   lambda: drift_curve(wm, device)),
        ("latent_map",    lambda: latent_map(wm, device)),
        ("code_action_matrix", lambda: code_action_matrix(device)),
        ("free_dream",    lambda: free_dream(wm, device)),
    ):
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"[viz] {label} FAILED: {e}")
            traceback.print_exc()
    print("[viz] done.")


# ======================================================================= smoke test
def _smoke():
    """CPU-only plumbing check with a FRESH random WorldModel + a SHORT synthetic
    rollout (no trained checkpoint exists yet). Exercises the dream/recon/decode
    path end-to-end and writes one tiny GIF to verify the wiring. Keeps tensors and
    step counts tiny so it does NOT contend with GPU training and finishes in
    seconds. `main()` itself uses WorldModel.load()."""
    device = torch.device("cpu")                               # force CPU
    cfg = CFG
    cfg.ensure_dirs()
    torch.manual_seed(0)

    wm = WorldModel(cfg).to(device).eval()                     # random weights
    # Short synthetic rollout from the REAL env (cheap; deterministic CPU physics).
    env = MultiBody2DEnv(cfg, seed=1)
    pol = RepeatRandomPolicy(cfg, seed=1)
    T = 6                                                      # a handful of steps
    obs, acts, pos, *_ = rollout(env, pol, ep_len=T)
    assert obs.shape == (T + 1, cfg.img_size, cfg.img_size, 3), obs.shape
    assert acts.shape == (T,), acts.shape

    # --- reconstruction path (posterior over all frames) ---
    recon = rssm_reconstruct(wm, obs, acts, device)
    assert recon.shape == obs.shape, (recon.shape, obs.shape)

    # --- open-loop dream path (ground frame 0, roll img_step from true actions) ---
    dream = rssm_dream(wm, obs[0], acts, device)
    assert dream.shape == obs.shape, (dream.shape, obs.shape)
    assert dream.dtype == np.uint8
    # Frame 0 of the dream is the posterior recon of the real frame 0 (grounded),
    # so it must be finite and in-range; later frames are free hallucination.
    assert np.isfinite(dream.astype(np.float32)).all(), "dream produced non-finite px"
    mse = float(np.mean((dream.astype(np.float32) - obs.astype(np.float32)) ** 2))
    assert np.isfinite(mse), "dream MSE not finite"

    # --- write one tiny 3-column GIF to verify the imageio plumbing ---
    out = os.path.join(cfg.artifacts_dir, "_smoke_dream.gif")
    frames = [_stack_columns([obs[i], recon[i], dream[i]]) for i in range(len(obs))]
    imageio.mimsave(out, frames, fps=8, loop=0)
    assert os.path.exists(out) and os.path.getsize(out) > 0, "GIF not written"

    # --- drift curve is just per-frame MSE; verify shape + finiteness on 1 dream ---
    drift = ((dream.astype(np.float32) - obs.astype(np.float32)) ** 2
             ).mean(axis=(1, 2, 3))
    assert drift.shape == (T + 1,) and np.isfinite(drift).all(), drift

    # --- latent_map's PCA/raw-dim projection path (tiny) ---
    x = obs_to_tensor(obs).to(device).unsqueeze(0)
    a = F.pad(torch.from_numpy(acts).long().unsqueeze(0), (0, 1))
    with torch.no_grad():                                      # feats carry ST grad
        feat = wm.observe(x, a)["feat"][0].cpu().numpy()
    assert feat.shape == (T + 1, cfg.feat_dim), feat.shape

    print(f"viz smoke PASSED  device={device}  T={T}  "
          f"dream_mse={mse:.2f}  drift[0]={drift[0]:.2f}->drift[-1]={drift[-1]:.2f}  "
          f"feat={feat.shape}  wrote {os.path.basename(out)} "
          f"({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    _smoke()
