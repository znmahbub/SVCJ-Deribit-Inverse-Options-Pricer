"""Centralized Greek identities for the inverse-option hedging pipeline.

The pricing characteristic functions are defined under a measure Q under which
the dated futures price F_t is a martingale. The regular call delta is the Q^F
probability obtained by changing numeraire to the futures price:

    phi_{Q^F}(u) = phi_Q(u-i) / phi_Q(-i).

The numeraire change is deliberately kept separate from the pricing-measure
characteristic functions themselves.
"""
from __future__ import annotations

from typing import Callable
import numpy as np

from .inverse_fft_pricer import _leggauss_cached, _norm_cdf


def black_regular_delta(F, K, T, sigma, option_type="call") -> np.ndarray:
    """Black-76 regular delta dV_usd/dF for calls or puts."""
    F, K, T, sigma = np.broadcast_arrays(
        np.asarray(F, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(T, dtype=float),
        np.asarray(sigma, dtype=float),
    )
    p1 = np.full(F.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(F) & np.isfinite(K) & np.isfinite(T) & np.isfinite(sigma)
        & (F > 0.0) & (K > 0.0) & (T >= 0.0) & (sigma >= 0.0)
    )
    live = valid & (T > 0.0) & (sigma > 0.0)
    if np.any(live):
        vol = sigma[live] * np.sqrt(T[live])
        d1 = (np.log(F[live] / K[live]) + 0.5 * sigma[live] ** 2 * T[live]) / vol
        p1[live] = _norm_cdf(d1)
    degenerate = valid & ~live
    if np.any(degenerate):
        p1[degenerate] = np.where(
            F[degenerate] > K[degenerate], 1.0,
            np.where(F[degenerate] < K[degenerate], 0.0, 0.5),
        )
    return regular_delta_from_call_p1(p1, option_type)


def call_p1_from_logprice_cf(
    K,
    F0: float,
    cf: Callable[[np.ndarray], np.ndarray],
    *,
    quad_u_max: float = 200.0,
    quad_n: int = 128,
) -> np.ndarray:
    """Return P1 from phi_Q using phi_Q(u-i)/phi_Q(-i), i.e. Q^F."""
    K = np.asarray(K, dtype=float)
    out = np.full(K.shape, np.nan, dtype=float)
    if not np.isfinite(F0) or float(F0) <= 0.0:
        return out
    valid = np.isfinite(K) & (K > 0.0)
    if not np.any(valid):
        return out
    if not np.isfinite(quad_u_max) or quad_u_max <= 0.0 or int(quad_n) < 2:
        raise ValueError("quad_u_max must be positive and quad_n must be at least 2")

    nodes, weights = _leggauss_cached(int(quad_n))
    u = 0.5 * float(quad_u_max) * (nodes + 1.0)
    w = 0.5 * float(quad_u_max) * weights
    phi_minus_i = np.asarray(
        cf(np.array([-1j], dtype=np.complex128)), dtype=np.complex128
    ).reshape(-1)[0]
    if not np.isfinite(phi_minus_i) or abs(phi_minus_i) < 1e-14:
        return out
    phi_shift = np.asarray(cf(u - 1j), dtype=np.complex128).reshape(-1)
    if phi_shift.shape != u.shape or not np.all(np.isfinite(phi_shift)):
        return out

    lnK = np.log(K[valid])
    exp_term = np.exp(-1j * u[:, None] * lnK[None, :])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        integrand = exp_term * (
            phi_shift[:, None] / ((1j * u)[:, None] * phi_minus_i)
        )
    if not np.all(np.isfinite(integrand)):
        return out
    p1 = 0.5 + (1.0 / np.pi) * np.sum(w[:, None] * np.real(integrand), axis=0)
    out[valid] = np.clip(p1, 0.0, 1.0)
    return out


def regular_delta_from_call_p1(p1, option_type="call") -> np.ndarray:
    """Convert call P1 to regular call/put delta using put-call parity."""
    p1 = np.asarray(p1, dtype=float)
    opt = np.asarray(option_type)
    if opt.ndim == 0:
        name = str(opt.item()).lower()
        if name not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")
        return p1 - (1.0 if name == "put" else 0.0)
    opt = np.broadcast_to(opt, p1.shape)
    names = np.char.lower(opt.astype(str))
    if np.any(~np.isin(names, ["call", "put"])):
        raise ValueError("option_type entries must be 'call' or 'put'")
    return p1 - (names == "put").astype(float)


def net_delta(regular_delta, inverse_price) -> np.ndarray:
    """Contract-consistent inverse delta: Delta_net = Delta_reg - V_coin."""
    reg, price = np.broadcast_arrays(
        np.asarray(regular_delta, dtype=float),
        np.asarray(inverse_price, dtype=float),
    )
    return reg - price


__all__ = [
    "black_regular_delta",
    "call_p1_from_logprice_cf",
    "regular_delta_from_call_p1",
    "net_delta",
]
