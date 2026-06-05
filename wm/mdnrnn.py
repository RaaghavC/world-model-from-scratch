"""M -- a Mixture-Density Recurrent Neural Network (the "simulator").

This is the heart of the world model. Given the current latent z_t and action
a_t, an LSTM predicts a *distribution* over the next latent z_{t+1}, modelled as
a diagonal mixture of K Gaussians per latent dimension:

    p(z_{t+1}[d] | history) = sum_k  pi_{d,k} * N( z | mu_{d,k}, sigma_{d,k} )

Modelling a distribution (rather than a single point) is what makes the model a
*generative* simulator: we can SAMPLE plausible next latents and roll forward --
i.e. dream. The LSTM hidden state h carries the unobserved parts of the state
(notably velocity, which a single frame can't reveal).

Reference: Ha & Schmidhuber, "World Models" (2018), the M (memory) module.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PI = math.log(2 * math.pi)


class MDNRNN(nn.Module):
    def __init__(self, z_dim: int, n_actions: int, hidden: int = 256, k: int = 5):
        super().__init__()
        self.z_dim, self.n_actions, self.hidden, self.k = z_dim, n_actions, hidden, k
        # Input at each step is [latent, one-hot action]; the LSTM is the recurrent
        # core whose hidden state summarises the past.
        self.lstm = nn.LSTM(z_dim + n_actions, hidden, batch_first=True)
        # Head emits, per latent dim, K mixture weights + K means + K log-sigmas.
        self.head = nn.Linear(hidden, 3 * z_dim * k)

    def forward(self, z, a_onehot, hc=None):
        """z:(B,T,z) ; a_onehot:(B,T,n_actions). Returns (params, hc).

        params:(B,T,3*z*k) raw head outputs; hc is the LSTM (h, c) state tuple so
        callers can continue a rollout step-by-step.
        """
        x = torch.cat([z, a_onehot], dim=-1)
        out, hc = self.lstm(x, hc)
        params = self.head(out)
        return params, hc

    def split(self, params):
        """Unpack raw head outputs into mixture parameters.

        params:(B,T,3*z*k) -> logpi, mu, logsigma, each (B,T,z,k).
          logpi    : log mixture weights (log-softmax-normalised over the k axis)
          mu       : component means
          logsigma : component log std-devs (clamped for numerical stability)
        """
        B, T, _ = params.shape
        p = params.reshape(B, T, self.z_dim, 3, self.k)
        logpi = F.log_softmax(p[..., 0, :], dim=-1)
        mu = p[..., 1, :]
        logsigma = p[..., 2, :].clamp(-7.0, 7.0)
        return logpi, mu, logsigma


def mdn_nll(params, target, model: MDNRNN):
    """Negative log-likelihood of `target` z under the predicted mixture.

    Uses the log-sum-exp trick for a numerically stable log of the mixture. The
    per-dimension log-likelihoods are summed (diagonal mixture = independent dims)
    and averaged over (batch, time). Lower is better.
    """
    logpi, mu, logsigma = model.split(params)
    t = target.unsqueeze(-1)                 # (B,T,z,1) broadcast against k components
    sigma = logsigma.exp()
    # log N(t | mu, sigma) for every component.
    log_comp = -0.5 * (((t - mu) / sigma) ** 2) - logsigma - 0.5 * LOG2PI  # (B,T,z,k)
    # log sum_k pi_k * N(...) = logsumexp_k(logpi_k + logN_k).
    log_mix = torch.logsumexp(logpi + log_comp, dim=-1)                    # (B,T,z)
    return -log_mix.sum(-1).mean()


@torch.no_grad()
def mdn_sample(logpi, mu, logsigma, temperature: float = 1.0, mode: bool = False):
    """Draw (or take the mode of) one MDN step.

    logpi, mu, logsigma: (B, z, k).  Returns z: (B, z).
      mode=True            -> deterministic: the mean of the most-likely component.
                              Best for matching this deterministic environment.
      temperature in (0,1] -> stochastic: pick a component ~ softmax(logpi/T), then
                              draw from it with std scaled by sqrt(T). Higher T =
                              more diverse dreams (and harder to "exploit").
    """
    if mode:
        k = logpi.argmax(dim=-1, keepdim=True)            # most likely component
        return mu.gather(-1, k).squeeze(-1)
    pi = F.softmax(logpi / max(temperature, 1e-6), dim=-1)
    B, Z, K = pi.shape
    # Sample one component index per (batch, latent-dim).
    k = torch.multinomial(pi.reshape(B * Z, K), 1).reshape(B, Z, 1)
    chosen_mu = mu.gather(-1, k).squeeze(-1)
    chosen_sigma = logsigma.exp().gather(-1, k).squeeze(-1)
    return chosen_mu + chosen_sigma * math.sqrt(max(temperature, 1e-6)) * torch.randn_like(chosen_mu)


def step_params(model: MDNRNN, z_t, a_t_onehot, hc):
    """Run ONE recurrent step and return the (B,z,k) mixture params for z_{t+1}.

    z_t:(B,z), a_t_onehot:(B,n_actions). Wraps the time dimension (T=1) so the
    same model can be used for both batched training and step-by-step dreaming.
    Returns ((logpi, mu, logsigma), hc).
    """
    params, hc = model(z_t.unsqueeze(1), a_t_onehot.unsqueeze(1), hc)
    logpi, mu, logsigma = model.split(params)
    return (logpi[:, 0], mu[:, 0], logsigma[:, 0]), hc
