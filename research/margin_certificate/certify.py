"""
Margin certificates for greedy decoding under floating-point nondeterminism.

Question: two inference stacks (different precision / reduction order / GPU) compute the
logits of the same token position. When are they GUARANTEED to pick the same argmax token?

Setup. For a candidate pair (a, b) let u = w_a - w_b (unembedding rows) and h the final
hidden state. The logit gap is  gap = u . h.  Every implementation computes fl(u . h).

Key quantity (the "cancellation" / condition number of the gap):
        S = sum_i |u_i h_i|            kappa = S / |gap|

Deterministic bound (Higham-style, first order): if implementation X rounds its input
hidden state with unit roundoff uX and accumulates n products in precision vX in ANY order,
        |gap_X - gap| <= (uX + gamma_{n-1}(vX)) * S ,   gamma_k(v) = k v / (1 - k v)
so two implementations X, Y can only disagree on the sign of the gap if
        |gap_X| <= r := (uX + uY + gamma(vX) + gamma(vY)) * S .
A token whose observed gap exceeds r for every runner-up is CERTIFIED invariant.

Probabilistic model for uncertified tokens (random rounding errors, CLT):
        sigma^2 = (uX^2 + uY^2)/3 * sum (u_i h_i)^2  +  accumulation term
        P(flip) ~= Phi(-|gap| / sigma)
Sequence of L tokens diverges with prob 1 - prod_t (1 - p_t).
"""
from __future__ import annotations
import numpy as np
from math import erf, sqrt

# unit roundoffs (round-to-nearest)
U = {"bf16": 2.0**-8, "fp16": 2.0**-11, "fp32": 2.0**-24, "fp64": 2.0**-53}


def gamma(k: int, v: float) -> float:
    return k * v / (1.0 - k * v)


def fragility(u: np.ndarray, h: np.ndarray):
    """Return (S, sumsq) along the last axis for the pair-difference vector u and hidden h."""
    x = u * h
    return np.abs(x).sum(-1), (x * x).sum(-1)


def certified_radius(S, n, in_x, in_y, acc_x="fp32", acc_y="fp32"):
    """Deterministic radius r: |gap_X| > r  =>  sign(gap_X) == sign(gap_Y) for ANY summation order."""
    r = (U[in_x] + U[in_y] + gamma(n - 1, U[acc_x]) + gamma(n - 1, U[acc_y]))
    return r * S * (1.0 + U[in_x])  # (1+u) covers S computed from the rounded h


def noise_sigma(sumsq, n, in_x, in_y, acc_x="fp32", acc_y="fp32"):
    """Std-dev of (gap_X - gap_Y) under independent uniform rounding errors."""
    var_in = (U[in_x] ** 2 + U[in_y] ** 2) / 3.0 * sumsq
    var_acc = (U[acc_x] ** 2 + U[acc_y] ** 2) / 3.0 * (n / 2.0) * sumsq  # random-order partial sums
    return np.sqrt(var_in + var_acc)


def _phi(z):
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))


def flip_probability(gap, sigma):
    """P(sign flips) under the Gaussian rounding-noise model."""
    return _phi(-np.abs(gap) / np.maximum(sigma, 1e-300))


def sequence_divergence(p_tokens: np.ndarray) -> float:
    """P(at least one token flips) over a decoded sequence."""
    return float(1.0 - np.prod(1.0 - p_tokens))
