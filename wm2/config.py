"""Central configuration for the 2026-frontier world model (wm2).

One dataclass holds every hyperparameter, mirroring `wm/config.py` so the two
models are configured the same way and are easy to compare. Defaults are tuned
to train from scratch on a laptop GPU (Apple MPS) in a reasonable wall-clock.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import torch

warnings.filterwarnings("ignore", message=".*weights_only.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_device() -> torch.device:
    """Prefer Apple Metal (MPS), then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class Config:
    # ---------- environment (richer, multi-body, partially-occluded) ----------
    img_size: int = 64
    n_actions: int = 5             # 0:noop 1:+x 2:-x 3:+y 4:-y  (agent thrust)
    n_bodies: int = 1              # body 0 = agent (thrustable); 1.. = free movers.
                                   # 1 => the agent's action DOMINATES frame changes,
                                   # so the latent-action model + control are not
                                   # confounded by independent movers (multi-body
                                   # (n>=2) is supported but confounds those tasks).
    gravity: float = 0.0           # downward accel per step (0 = float; >0 = falls)
    thrust: float = 0.0050         # agent acceleration per thrust step
    friction: float = 0.92         # agent velocity retained / step (free bodies: 1.0)
    max_speed: float = 0.060       # agent speed cap (salient enough that velocity is
                                   # encoded; slow enough that random rarely reaches)
    free_speed0: float = 0.045     # initial speed of free bodies
    agent_radius: float = 0.060
    body_radius: float = 0.055     # free-body radius
    goal_radius: float = 0.050     # small, so it doesn't dominate the foreground loss
    restitution_wall: float = 0.90
    restitution_body: float = 1.00  # elastic body-body => momentum+energy conserved
    depth_render: bool = True      # draw far->near (occlusion) -> a 2.5D scene
    direct_control: bool = False   # if True, the action SETS the agent velocity
                                   # directly (no thrust/momentum) -> actions drive
                                   # the frames, the regime where label-free action
                                   # discovery (the LAM, Genie-style) is meaningful

    # ---------- data collection ----------
    n_episodes: int = 400
    ep_len: int = 100
    action_repeat: int = 6
    seed: int = 0

    # ---------- RSSM world model (the V+M fusion) ----------
    embed_dim: int = 256           # CNN encoder output width
    deter_dim: int = 256           # GRU deterministic state h
    stoch_cat: int = 24            # number of categorical latent variables
    stoch_classes: int = 24        # classes per categorical  (z = 24x24 one-hots)
    hidden: int = 256              # width of MLP heads inside the RSSM
    kl_free_bits: float = 1.0      # clip KL below this many nats (DreamerV3)
    kl_balance: float = 0.8        # KL balancing alpha (post vs prior)
    kl_scale: float = 1.0
    recon_scale: float = 1.0
    vae_fg_weight: float = 12.0    # upweight non-background pixels (enough to keep the
                                   # small agent sharp, low enough to render the dark
                                   # background cleanly instead of a green texture)
    reward_scale: float = 1.0
    cont_scale: float = 1.0
    unimix: float = 0.01           # 1% uniform mixture on categorical logits

    # ---------- world-model training ----------
    wm_seq_len: int = 32           # BPTT sequence length
    wm_batch: int = 24
    wm_epochs: int = 14
    wm_lr: float = 3e-4
    grad_clip: float = 100.0
    wm_steps_per_epoch: int = 150   # gradient steps per "epoch" (sampled batches)
    wm_hist_aug: float = 0.15       # latent history augmentation prob (drift fix)

    # ---------- Latent Action Model (Genie-style, label-free) ----------
    lam_codes: int = 8             # size of the discrete latent-action codebook
    lam_dim: int = 32              # latent-action embedding width
    lam_epochs: int = 10
    lam_lr: float = 3e-4
    lam_beta: float = 0.25         # VQ commitment cost

    # ---------- actor-critic in imagination (Dreamer) ----------
    imag_horizon: int = 30         # enough for the (now faster) agent to reach a goal
    gamma: float = 0.997
    lambda_: float = 0.95
    actor_lr: float = 8e-5
    critic_lr: float = 1e-4
    actor_ent: float = 1e-3        # entropy bonus
    ac_iters: int = 3500           # imagination training steps
    ac_batch: int = 256            # imagined start states per step
    value_buckets: int = 255       # two-hot symexp value support size (DreamerV3)
    value_vmin: float = -20.0
    value_vmax: float = 20.0
    slow_critic_tau: float = 0.02  # EMA target-critic update rate

    # ---------- JEPA track (decoder-free, optional) ----------
    jepa_dim: int = 128
    jepa_ema: float = 0.996
    jepa_epochs: int = 12
    jepa_lr: float = 3e-4
    plan_horizon: int = 25         # CEM/MPPI planning horizon (reach a goal)
    plan_pop: int = 256
    plan_iters: int = 6
    plan_elite: int = 32

    # ---------- paths ----------
    run_name: str = "wm2"

    @property
    def data_dir(self) -> str:
        return os.path.join(ROOT, "wm2_data")

    @property
    def artifacts_dir(self) -> str:
        return os.path.join(ROOT, "wm2_artifacts")

    @property
    def ckpt_dir(self) -> str:
        return os.path.join(ROOT, "wm2_checkpoints")

    @property
    def stoch_dim(self) -> int:
        """Flattened stochastic latent size (cat * classes)."""
        return self.stoch_cat * self.stoch_classes

    @property
    def feat_dim(self) -> int:
        """World-model feature = [deter h ; flattened stochastic z]."""
        return self.deter_dim + self.stoch_dim

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.artifacts_dir, self.ckpt_dir):
            os.makedirs(d, exist_ok=True)


CFG = Config()
