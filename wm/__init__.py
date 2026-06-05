"""A from-scratch world model in the Ha & Schmidhuber (2018) lineage.

Components:
  env      -- a from-scratch 2D physics environment (no gym dependency)
  vae      -- V: a convolutional variational autoencoder (the "renderer")
  mdnrnn   -- M: a mixture-density RNN over latent dynamics (the "simulator")
  controller -- C: a tiny policy trained *inside the dream* (the "planner")

The three map onto Fei-Fei Li's functional taxonomy of world models:
  Renderer  = VAE decoder (pixels)
  Simulator = MDN-RNN latent dynamics (faithful, computable state)
  Planner   = controller that outputs actions to reach a goal
"""

__version__ = "0.1.0"
