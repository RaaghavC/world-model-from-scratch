"""Central configuration for the world model.

One dataclass holds every hyperparameter so experiments are reproducible and the
whole pipeline (collect -> VAE -> MDN-RNN -> dream -> controller) reads from a
single source of truth.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import torch

warnings.filterwarnings("ignore", message=".*weights_only.*")  # cosmetic: torch.load notice

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
    # ---- environment ----
    img_size: int = 64
    n_actions: int = 5            # 0:noop 1:+x 2:-x 3:+y 4:-y
    thrust: float = 0.0035        # acceleration per thrust step (normalized units)
    friction: float = 0.90        # velocity retained each step
    max_speed: float = 0.05       # speed cap (normalized units / step)
    agent_radius: float = 0.060   # ball radius in [0,1] coords
    goal_radius: float = 0.085    # goal half-width in [0,1] coords
    restitution: float = 0.80     # wall bounce energy retained

    # ---- data collection ----
    n_episodes: int = 320
    ep_len: int = 150
    action_repeat: int = 6        # hold a sampled action this many steps
    seed: int = 0

    # ---- VAE (V) ----
    z_dim: int = 32
    vae_beta: float = 1.0         # KL weight (beta-VAE)
    vae_fg_weight: float = 30.0   # upweight non-background pixels (the small ball)
    vae_epochs: int = 14
    vae_batch: int = 256
    vae_lr: float = 1e-3

    # ---- MDN-RNN (M) ----
    rnn_hidden: int = 256
    n_gaussians: int = 5
    seq_len: int = 64
    rnn_epochs: int = 24
    rnn_batch: int = 64
    rnn_lr: float = 1e-3
    temperature: float = 1.0      # sampling temperature for dreaming

    # ---- controller (C) ----
    ctrl_pop: int = 96            # CEM population per generation
    ctrl_generations: int = 70
    ctrl_rollouts: int = 16       # dream scenarios (goals/starts) averaged per candidate
    ctrl_horizon: int = 110       # dream horizon for fitness
    ctrl_sigma: float = 0.5       # initial CEM std
    ctrl_temp: float = 0.0        # 0 = deterministic dreams (matches this deterministic
                                  # env best); >0 would inject stochasticity to curb exploitation

    # ---- paths ----
    @property
    def data_dir(self) -> str:
        return os.path.join(ROOT, "data")

    @property
    def artifacts_dir(self) -> str:
        return os.path.join(ROOT, "artifacts")

    @property
    def ckpt_dir(self) -> str:
        return os.path.join(ROOT, "checkpoints")

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.artifacts_dir, self.ckpt_dir):
            os.makedirs(d, exist_ok=True)


CFG = Config()
