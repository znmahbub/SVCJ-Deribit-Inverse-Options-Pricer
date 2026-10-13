"""Mapping from option delta to inverse-perpetual hedge notional."""
from __future__ import annotations

import numpy as np


def inverse_perp_pnl(notional_usd, H_t, H_t1) -> np.ndarray:
    """Coin P&L of a signed inverse-perpetual USD notional over one interval."""
    N, H0, H1 = np.broadcast_arrays(
        np.asarray(notional_usd, dtype=float),
        np.asarray(H_t, dtype=float),
        np.asarray(H_t1, dtype=float),
    )
    out = np.full(N.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(N) & np.isfinite(H0) & np.isfinite(H1)
        & (H0 > 0.0) & (H1 > 0.0)
    )
    out[valid] = N[valid] * (1.0 / H0[valid] - 1.0 / H1[valid])
    return out


def hedge_notional_usd(
    delta_used,
    F_term,
    H_perp,
    *,
    option_position: float = -1.0,
    dF_dH: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Map regular or net option delta to inverse-perpetual USD notional.

    The option position has local coin sensitivity
    ``option_position * delta_used / F_term`` to a dated-futures move.  The
    inverse perpetual has local sensitivity ``notional_usd / H_perp**2`` to its
    own USD price.  If locally ``dF_term = dF_dH * dH`` then neutrality gives

        N_USD = -option_position * dF_dH * H_perp**2/F_term * delta_used.

    For the baseline short-one-option position and dF/dH=1 this becomes
    ``H_perp**2/F_term * delta_used`` for both regular- and net-delta rules.
    """
    delta, F, H, slope = np.broadcast_arrays(
        np.asarray(delta_used, dtype=float),
        np.asarray(F_term, dtype=float),
        np.asarray(H_perp, dtype=float),
        np.asarray(dF_dH, dtype=float),
    )
    pos = float(option_position)
    out = np.full(delta.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(delta) & np.isfinite(F) & np.isfinite(H)
        & np.isfinite(slope) & np.isfinite(pos)
        & (F > 0.0) & (H > 0.0)
    )
    out[valid] = -pos * slope[valid] * (H[valid] ** 2 / F[valid]) * delta[valid]
    return out


__all__ = ["inverse_perp_pnl", "hedge_notional_usd"]
