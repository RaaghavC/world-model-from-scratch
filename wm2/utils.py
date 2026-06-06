"""Shared primitives for wm2: symlog/symexp, two-hot regression, straight-through
categoricals, and frame<->tensor conversion.

These are the error-prone numerical building blocks of the DreamerV3 recipe, kept
in one tested place so every module (rssm, agent) uses an identical, correct
implementation. Run `python -m wm2.utils` to self-test.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------- frame <-> tensor
def obs_to_tensor(obs_uint8: np.ndarray) -> torch.Tensor:
    """(...,H,W,3) uint8 -> (...,3,H,W) float in [0,1] (contiguous for MPS)."""
    t = torch.from_numpy(np.ascontiguousarray(obs_uint8)).float() / 255.0
    return t.movedim(-1, -3).contiguous()


def tensor_to_obs(t: torch.Tensor) -> np.ndarray:
    """(...,3,H,W) float -> (...,H,W,3) uint8 (clamped)."""
    t = t.clamp(0, 1).movedim(-3, -1)
    return (t.detach().cpu().numpy() * 255).astype(np.uint8)


# --------------------------------------------------------------------- symlog / symexp
def symlog(x: torch.Tensor) -> torch.Tensor:
    """sign(x)*log(1+|x|) -- DreamerV3's scale-robust squashing."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x)*(exp(|x|)-1)."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


# ----------------------------------------------------------------------- two-hot
def symlog_support(lo: float, hi: float, n: int, device) -> torch.Tensor:
    """Evenly spaced bin locations in *symlog space* (the DreamerV3 value grid)."""
    return torch.linspace(lo, hi, n, device=device)


def two_hot(x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Two-hot encode scalars `x` (...,) onto sorted `support` (K,) -> (...,K).

    The two non-zero entries are at the bins bracketing x, weighted so that the
    expectation `sum(twohot * support)` reproduces x EXACTLY for x in
    [support[0], support[-1]] (values outside are clamped to the ends).
    """
    K = support.shape[0]
    x = x.clamp(support[0], support[-1])
    # Index of the lower bracketing bin.
    below = (support.unsqueeze(0) <= x.reshape(-1, 1)).sum(dim=1) - 1  # (-1,)
    below = below.clamp(0, K - 2)
    above = below + 1
    lo = support[below]
    hi = support[above]
    w_hi = (x.reshape(-1) - lo) / (hi - lo + 1e-12)
    w_lo = 1.0 - w_hi
    out = torch.zeros(x.reshape(-1).shape[0], K, device=x.device, dtype=support.dtype)
    rows = torch.arange(out.shape[0], device=x.device)
    out[rows, below] = w_lo
    out[rows, above] = w_hi
    return out.reshape(*x.shape, K)


def twohot_expect(probs: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Expectation under a categorical over `support`: (...,K)·(K,) -> (...)."""
    return (probs * support).sum(-1)


def value_from_logits(logits: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Decode a scalar value from bucket logits: symexp(E_softmax[support])."""
    return symexp(twohot_expect(F.softmax(logits, dim=-1), support))


def symlog_twohot_loss(logits: torch.Tensor, target_real: torch.Tensor,
                       support: torch.Tensor) -> torch.Tensor:
    """Categorical CE between predicted bucket logits and the two-hot encoding of
    symlog(target_real). Mean over all leading dims. This is DreamerV3's
    distributional regression loss for reward and value heads."""
    target = two_hot(symlog(target_real), support).detach()
    logp = F.log_softmax(logits, dim=-1)
    return -(target * logp).sum(-1).mean()


# --------------------------------------------------------- straight-through categorical
def categorical_st(logits: torch.Tensor, unimix: float = 0.01,
                   sample: bool = True):
    """Straight-through categorical over the LAST dim.

    logits: (..., classes). Returns (onehot, probs) where `onehot` is a one-hot
    sample (or argmax if sample=False) carrying a straight-through gradient into
    `probs`, and `probs` includes a `unimix` uniform mixture (DreamerV3) so KL
    never sees a hard zero.
    """
    probs = F.softmax(logits, dim=-1)
    if unimix > 0:
        uniform = torch.ones_like(probs) / probs.shape[-1]
        probs = (1 - unimix) * probs + unimix * uniform
    if sample:
        idx = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1)
        idx = idx.reshape(*probs.shape[:-1])
    else:
        idx = probs.argmax(-1)
    onehot = F.one_hot(idx, probs.shape[-1]).to(probs.dtype)
    # Straight-through: forward = hard one-hot, backward = grad through probs.
    onehot = onehot + (probs - probs.detach())
    return onehot, probs


def categorical_kl(probs_q: torch.Tensor, probs_p: torch.Tensor) -> torch.Tensor:
    """KL(q || p) summed over classes then over categoricals.

    probs_*: (..., stoch_cat, stoch_classes). Returns (...,) KL per batch item
    (summed over the `stoch_cat` independent categoricals)."""
    q = probs_q.clamp(1e-8, 1.0)
    p = probs_p.clamp(1e-8, 1.0)
    kl = (q * (q.log() - p.log())).sum(-1)   # over classes -> (..., stoch_cat)
    return kl.sum(-1)                         # over categoricals -> (...,)


# ----------------------------------------------------------------------- nets
def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2,
        act=nn.SiLU, norm: bool = True) -> nn.Sequential:
    """Small MLP with optional LayerNorm, SiLU activations (DreamerV3 style)."""
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods.append(nn.Linear(d, hidden))
        if norm:
            mods.append(nn.LayerNorm(hidden))
        mods.append(act())
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


# ----------------------------------------------------------------------- self-test
def _selftest():
    torch.manual_seed(0)
    # symlog round-trip
    x = torch.tensor([-1234.0, -1.0, 0.0, 0.5, 1000.0])
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-3), "symlog/symexp"
    # two-hot expectation reproduces the value exactly
    sup = symlog_support(-20, 20, 41, "cpu")
    vals = torch.tensor([-7.3, -0.2, 0.0, 0.8, 5.5])
    th = two_hot(vals, sup)
    assert torch.allclose(twohot_expect(th, sup), vals, atol=1e-4), "two_hot expect"
    assert torch.allclose(th.sum(-1), torch.ones(5), atol=1e-5), "two_hot normalized"
    # value head round-trip: encode symlog(real) two-hot, decode -> real
    real = torch.tensor([-12.0, -1.5, 0.0, 3.0, 25.0])
    target = two_hot(symlog(real), sup)
    logits = (target + 1e-6).log()            # near-deterministic logits
    assert torch.allclose(value_from_logits(logits, sup), real, atol=1e-2), "value rt"
    # straight-through categorical: hard forward, gradient flows
    lg = torch.randn(4, 3, 8, requires_grad=True)
    oh, pr = categorical_st(lg, unimix=0.01)
    assert oh.shape == (4, 3, 8)
    assert torch.allclose(oh.sum(-1), torch.ones(4, 3), atol=1e-5), "onehot sums to 1"
    oh.sum().backward()
    assert lg.grad is not None and lg.grad.abs().sum() > 0, "ST gradient"
    # KL is zero for identical dists, positive otherwise
    p = F.softmax(torch.randn(2, 3, 8), -1)
    assert categorical_kl(p, p).abs().max() < 1e-5, "KL(p||p)=0"
    assert (categorical_kl(p, F.softmax(torch.randn(2, 3, 8), -1)) > 0).all(), "KL>0"
    print("wm2.utils self-test PASSED")


if __name__ == "__main__":
    _selftest()
