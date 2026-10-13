from __future__ import annotations

from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected 1 occurrence, found {n}")
    return text.replace(old, new, 1)


(ROOT / "src/hedge_mapping.py").write_text(textwrap.dedent(r'''
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
''').lstrip(), encoding="utf-8")

# Regression at the hedging-engine level: nonzero basis must not change the
# mapping across regular/net rules.
p = ROOT / "tests/test_hedging_invariants.py"
s = p.read_text(encoding="utf-8")
s = replace_once(
    s,
    "from src.hedging_analysis import compute_prices_and_deltas  # noqa: E402\n",
    "from src.hedging_analysis import build_hedge_interval_panel, compute_prices_and_deltas  # noqa: E402\n",
    "hedging invariant import",
)
method = '''
    def test_regular_and_net_hedges_share_identical_term_perp_mapping(self) -> None:
        """With F_term != H_perp, both rules must use H_perp^2/F_term."""
        options = pd.DataFrame(
            {
                "snapshot_ts": [pd.Timestamp("2026-01-01T00:00:00Z")],
                "eval_snapshot_ts": [pd.Timestamp("2026-01-01T06:00:00Z")],
                "currency": ["BTC"],
                "split": ["test"],
                "instrument_name": ["BTC-TEST"],
                "option_type": ["call"],
                "strike": [10_000.0],
                "time_to_maturity": [0.25],
                "F0": [10_000.0],
                "perp_close_t": [9_800.0],
                "perp_close_t1": [9_900.0],
                "mid_price_clean": [0.10],
                "eval_market_price": [0.105],
                "is_hedgeable_interval": [True],
                "expires_before_eval_snapshot": [False],
                "regular_delta_black": [0.40],
                "net_delta_black": [0.30],
                "regular_delta_heston": [0.40],
                "net_delta_heston": [0.30],
                "regular_delta_svcj": [0.40],
                "net_delta_svcj": [0.30],
            }
        )
        panel = build_hedge_interval_panel(options)
        scale = 9_800.0 ** 2 / 10_000.0
        for model in ("black", "heston", "svcj"):
            self.assertAlmostEqual(
                float(panel.loc[0, f"hedge_notional_usd_{model}_delta"]),
                scale * 0.40,
            )
            self.assertAlmostEqual(
                float(panel.loc[0, f"hedge_notional_usd_{model}_net_delta"]),
                scale * 0.30,
            )
'''
anchor = "\n\nif __name__ == \"__main__\":  # pragma: no cover\n"
if anchor not in s:
    raise RuntimeError("test_hedging_invariants main guard missing")
s = s.replace(anchor, "\n" + method.rstrip() + anchor, 1)
p.write_text(s, encoding="utf-8")

p = ROOT / "src/hedging_analysis.py"
s = p.read_text(encoding="utf-8")
s = replace_once(
    s,
    "from src.greeks import (\n"
    "    black_regular_delta,\n"
    "    call_p1_from_logprice_cf,\n"
    "    net_delta as inverse_net_delta,\n"
    "    regular_delta_from_call_p1,\n"
    ")\n",
    "from src.greeks import (\n"
    "    black_regular_delta,\n"
    "    call_p1_from_logprice_cf,\n"
    "    net_delta as inverse_net_delta,\n"
    "    regular_delta_from_call_p1,\n"
    ")\n"
    "from src.hedge_mapping import hedge_notional_usd, inverse_perp_pnl\n",
    "hedge mapping imports",
)
s = replace_once(
    s,
    "    inverse_move = np.where(np.isfinite(H_t) & np.isfinite(H_t1) & (H_t > 0) & (H_t1 > 0), (1.0 / H_t) - (1.0 / H_t1), np.nan)\n",
    "    valid_basis = np.isfinite(F_t) & np.isfinite(H_t) & (F_t > 0) & (H_t > 0)\n"
    "    notional_scale = np.where(valid_basis, (H_t ** 2) / F_t, np.nan)\n"
    "    panel[\"F_term\"] = F_t\n"
    "    panel[\"H_perp\"] = H_t\n"
    "    panel[\"basis_ratio\"] = np.where(valid_basis, H_t / F_t, np.nan)\n"
    "    panel[\"dF_dH_assumption\"] = 1.0\n"
    "    panel[\"hedge_notional_scale_usd_per_delta\"] = notional_scale\n",
    "basis audit columns",
)
old_loop = '''    for model in MODELS:
        reg_delta = _num(options[f"regular_delta_{model}"])
        net_delta = _num(options[f"net_delta_{model}"])
        panel[f"hedge_notional_usd_{model}_delta"] = H_t * reg_delta
        panel[f"net_pnl_coin_{model}_delta"] = panel["option_pnl_coin_short"] + panel[f"hedge_notional_usd_{model}_delta"] * inverse_move
        net_scale = np.where(np.isfinite(F_t) & (F_t > 0), (H_t ** 2) / F_t, np.nan)
        panel[f"hedge_notional_usd_{model}_net_delta"] = net_scale * net_delta
        panel[f"net_pnl_coin_{model}_net_delta"] = panel["option_pnl_coin_short"] + panel[f"hedge_notional_usd_{model}_net_delta"] * inverse_move
'''
new_loop = '''    for model in MODELS:
        delta_by_rule = {
            "delta": _num(options[f"regular_delta_{model}"]).to_numpy(dtype=float),
            "net_delta": _num(options[f"net_delta_{model}"]).to_numpy(dtype=float),
        }
        for hedge_type, delta_used in delta_by_rule.items():
            notional = hedge_notional_usd(
                delta_used,
                F_t.to_numpy(dtype=float),
                H_t.to_numpy(dtype=float),
                option_position=-1.0,
                dF_dH=1.0,
            )
            panel[f"hedge_notional_usd_{model}_{hedge_type}"] = notional
            panel[f"net_pnl_coin_{model}_{hedge_type}"] = (
                panel["option_pnl_coin_short"].to_numpy(dtype=float)
                + inverse_perp_pnl(
                    notional,
                    H_t.to_numpy(dtype=float),
                    H_t1.to_numpy(dtype=float),
                )
            )
'''
s = replace_once(s, old_loop, new_loop, "unified hedge loop")
s = replace_once(
    s,
    "        (\"delta_hedge_ratio_definition\", \"N_usd = H_t * regular_delta\"),\n"
    "        (\"net_delta_hedge_ratio_definition\", \"N_usd = H_t^2 / F_t * net_delta\"),\n",
    "        (\"basis_mapping_assumption\", \"dF_term/dH_perp = 1 locally\"),\n"
    "        (\"delta_hedge_ratio_definition\", \"N_usd = H_t^2 / F_t * regular_delta for a short one-option position\"),\n"
    "        (\"net_delta_hedge_ratio_definition\", \"N_usd = H_t^2 / F_t * net_delta for a short one-option position\"),\n",
    "hedge note definitions",
)
p.write_text(s, encoding="utf-8")

p = ROOT / "docs/code_documentation.md"
s = p.read_text(encoding="utf-8")
if "### Term-futures to inverse-perpetual hedge mapping" not in s:
    s += textwrap.dedent(r'''

### Term-futures to inverse-perpetual hedge mapping

`src/hedge_mapping.py` owns the basis conversion used by the hedging engine. For
a short one-option position and the baseline local assumption
`dF_term/dH_perp = 1`, both regular-delta and net-delta rules use exactly
`N_USD = H_perp^2/F_term * Delta_used`. The benchmark rules therefore differ
only in `Delta_used`, not in an additional basis approximation. The interval
panel persists `F_term`, `H_perp`, `basis_ratio`, `dF_dH_assumption`, and the
notional scale used.
''')
p.write_text(s, encoding="utf-8")
