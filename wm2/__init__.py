"""wm2 -- a 2026-frontier world model, from scratch.

This package upgrades the 2018 Ha & Schmidhuber V-M-C baseline (in `wm/`) to the
modern world-model frontier on a laptop:

  * V+M  -> a discrete-latent Recurrent State-Space Model (RSSM), the DreamerV3
            core: a deterministic GRU path + a categorical stochastic latent,
            with KL balancing, free bits, symlog targets, and learned reward /
            continue heads (so control needs no hand-built probe).
  * Genie -> a Latent Action Model (LAM) that discovers actions from raw frame
            pairs with no action labels, demonstrating label-free controllability.
  * C    -> an actor-critic trained PURELY in latent imagination (lambda-returns,
            two-hot value), replacing the baseline's CEM-on-a-probe controller.
  * JEPA -> an optional decoder-free joint-embedding predictive track + latent
            MPC planner, to engage the central 2026 generative-vs-nongenerative
            debate on the same toy world.

Everything is small enough to train from scratch on Apple Silicon (MPS).
"""
