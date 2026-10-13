from __future__ import annotations

from pathlib import Path
import re
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected 1 occurrence, found {n}")
    return text.replace(old, new, 1)


def regex_once(text: str, pattern: str, repl: str, label: str) -> str:
    out, n = re.subn(pattern, repl, text, count=1, flags=re.S)
    if n != 1:
        raise RuntimeError(f"{label}: expected 1 match, found {n}")
    return out


(ROOT / "src/greeks.py").write_text(textwrap.dedent(r'''
"""Centralized Greek identities for the inverse-option hedging pipeline.

The pricing characteristic functions are defined under a measure Q under which
F_t is a martingale in the maintained deterministic/zero-discounting setup.
The regular call delta is the Q^F probability obtained by changing numeraire
to the futures price:

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
''').lstrip(), encoding="utf-8")

p = ROOT / "src/inverse_fft_pricer.py"
s = p.read_text(encoding="utf-8")
s = replace_once(
    s,
    "  - **Characteristic functions** for each model, returning the moment\n"
    "    generating function of ``X_T = log F_T`` under the futures measure\n"
    "    evaluated at complex arguments.  These functions are vectorised\n",
    "  - **Characteristic functions** for each model, returning the characteristic\n"
    "    function of ``X_T = log F_T`` under the pricing measure ``Q`` for which\n"
    "    the futures price is a martingale under the maintained deterministic/zero-\n"
    "    discounting setup.  These functions are vectorised\n",
    "top-level measure wording",
)
s = replace_once(
    s,
    "    # For the Black model under the futures measure, log-futures follow\n",
    "    # Under the pricing measure Q, log-futures follow\n",
    "Black measure wording",
)
p.write_text(s, encoding="utf-8")

p = ROOT / "src/hedging_analysis.py"
s = p.read_text(encoding="utf-8")
s = replace_once(
    s,
    "from src.calibration import _build_pricing_plan, _price_with_plan\n"
    "from src.inverse_fft_pricer import (\n"
    "    FFTParams,\n"
    "    _leggauss_cached,          # shared cached GL nodes — no need to redefine\n"
    "    _norm_cdf,                  # shared rational-approximation CDF\n"
    "    cf_heston,\n"
    "    cf_svcj,\n"
    ")\n",
    "from src.calibration import _build_pricing_plan, _price_with_plan\n"
    "from src.greeks import (\n"
    "    black_regular_delta,\n"
    "    call_p1_from_logprice_cf,\n"
    "    net_delta as inverse_net_delta,\n"
    "    regular_delta_from_call_p1,\n"
    ")\n"
    "from src.inverse_fft_pricer import FFTParams, cf_heston, cf_svcj\n",
    "Greek imports",
)
s = regex_once(
    s,
    r"\ndef _black_call_delta\(.*?\n\ndef _intrinsic_coin_price",
    "\n\ndef _intrinsic_coin_price",
    "remove duplicate Black delta",
)
s = regex_once(
    s,
    r"\ndef _semi_analytic_call_p1_from_logprice_cf\(.*?\n\ndef _regular_delta_block",
    "\n\ndef _regular_delta_block",
    "remove duplicate P1 integrator",
)
s = replace_once(
    s,
    "            p1 = _black_call_delta(\n"
    "                np.full_like(K_g, float(F0_g)), K_g,\n"
    "                np.full_like(K_g, float(T_g)),\n"
    "                np.full_like(K_g, float(params[\"sigma\"])),\n"
    "            )\n",
    "            p1 = black_regular_delta(\n"
    "                np.full_like(K_g, float(F0_g)), K_g,\n"
    "                np.full_like(K_g, float(T_g)),\n"
    "                np.full_like(K_g, float(params[\"sigma\"])),\n"
    "                option_type=\"call\",\n"
    "            )\n",
    "Black delta call site",
)
s = s.replace(
    "p1 = _semi_analytic_call_p1_from_logprice_cf(K_g, cf=cf, quad_u_max=delta_u_max, quad_n=delta_quad_n)",
    "p1 = call_p1_from_logprice_cf(K_g, float(F0_g), cf=cf, quad_u_max=delta_u_max, quad_n=delta_quad_n)",
)
if s.count("call_p1_from_logprice_cf(K_g, float(F0_g)") != 2:
    raise RuntimeError("expected two centralized P1 call sites")
s = replace_once(
    s,
    "        reg = p1.copy()\n        reg[put_mask] = reg[put_mask] - 1.0\n",
    "        reg = regular_delta_from_call_p1(\n"
    "            p1, np.where(put_mask, \"put\", \"call\")\n"
    "        )\n",
    "call/put delta conversion",
)
s = replace_once(
    s,
    "    return prices, reg_delta, reg_delta - prices\n",
    "    return prices, reg_delta, inverse_net_delta(reg_delta, prices)\n",
    "net delta identity",
)
p.write_text(s, encoding="utf-8")

p = ROOT / "docs/code_documentation.md"
s = p.read_text(encoding="utf-8")
if "## Remediation invariants (Tasks 3-5)" not in s:
    s += textwrap.dedent(r'''

## Remediation invariants (Tasks 3-5)

### Pricing measure and inverse-option Greeks

`src/inverse_fft_pricer.py` defines `cf_black`, `cf_heston`, and `cf_svcj`
under the pricing measure `Q` for which the dated futures price is a martingale
under the maintained deterministic/zero-discounting setup. `src/greeks.py`
performs the separate futures-price-numeraire change used for regular delta:
the shifted characteristic function is `phi_Q(u-i) / phi_Q(-i)` and therefore
`P1 = Q^F(F_T > K)`. Under Black, direct `Q` exercise probability is `N(d2)`,
while `P1` and call regular delta are `N(d1)`. Net inverse delta is centralized
as `Delta_net = Delta_regular - V_coin`.
''')
p.write_text(s, encoding="utf-8")
