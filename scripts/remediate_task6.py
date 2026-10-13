from __future__ import annotations

from pathlib import Path
import re
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected 1 occurrence, found {count}")
    return text.replace(old, new, 1)


def regex_once(text: str, pattern: str, repl: str, label: str) -> str:
    out, count = re.subn(pattern, repl, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"{label}: expected 1 match, found {count}")
    return out


# ---------------------------------------------------------------------------
# src/calibration.py
# ---------------------------------------------------------------------------
p = ROOT / "src/calibration.py"
s = p.read_text(encoding="utf-8")

s = replace_once(
    s,
    '''class WeightConfig:
    """Controls how per-option residual weights are formed.

    Residual passed to least-squares:
        r_i = w_i * (P_model_i - P_mkt_i)
    where P are coin-denominated prices.
    """
''',
    '''class WeightConfig:
    """Controls per-option *residual scaling* in the least-squares problem.

    The residual passed to ``scipy.optimize.least_squares`` is

        r_i = s_i * (P_model_i - P_mkt_i),

    where prices are coin-denominated and ``s_i`` is the multiplicative scale
    assembled below.  Consequently the squared-loss contribution is
    ``s_i**2 * error_i**2``.  In particular, ``spread_power=1`` means residuals
    are divided by spread and the objective is inverse-spread-*squared*; ``s_i``
    itself should not be described as the final objective weight.
    """
''',
    "WeightConfig documentation",
)

s = replace_once(
    s,
    '''class CalibrationResult:
    model: str
    params: Dict[str, float]
    success: bool
    message: str
    nfev: int
    rmse: float
    mae: float
''',
    '''class CalibrationResult:
    model: str
    params: Dict[str, float]
    success: bool
    message: str
    nfev: int
    rmse: float
    mae: float
    regularization_active: bool = False
    l2_prev_strength: float = 0.0
    feller_penalty_active: bool = False
    feller_gap: float = float("nan")
    feller_violated: bool = False
    svcj_moment_penalty_active: bool = False
    svcj_martingale_margin: float = float("nan")
    svcj_martingale_violated: bool = False
    svcj_contour_margin: float = float("nan")
    svcj_contour_violated: bool = False
''',
    "CalibrationResult diagnostics",
)

pattern = r'''def _weights\(df: pd\.DataFrame, cfg: WeightConfig\) -> np\.ndarray:\n.*?\n\n# ---- parameter transforms'''
replacement = '''def _residual_scales(df: pd.DataFrame, cfg: WeightConfig) -> np.ndarray:
    """Return multiplicative residual scales ``s_i`` used by least squares."""
    scale = np.ones(len(df), dtype=float)

    if cfg.use_spread:
        spread = df["spread"].to_numpy(dtype=float)
        scale *= (spread + cfg.eps_spread) ** (-cfg.spread_power)

    if cfg.use_vega:
        vega = df["vega"].to_numpy(dtype=float)
        scale *= (vega + cfg.eps_other) ** (cfg.vega_power)

    if cfg.use_open_interest:
        oi = df["open_interest"].to_numpy(dtype=float)
        scale *= (oi + cfg.eps_other) ** (cfg.oi_power)

    if cfg.cap is not None:
        scale = np.minimum(scale, float(cfg.cap))

    return np.where(np.isfinite(scale), scale, 0.0)


def _weights(df: pd.DataFrame, cfg: WeightConfig) -> np.ndarray:
    """Backward-compatible alias for the historical private helper."""
    return _residual_scales(df, cfg)


def _regularization_residual(
    x: np.ndarray,
    previous_x: Optional[np.ndarray],
    *,
    strength: float,
) -> np.ndarray:
    """Return ``sqrt(lambda) * (x-x_prev)`` in packed parameter coordinates."""
    lam = float(strength)
    if previous_x is None or lam <= 0.0:
        return np.empty(0, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    prev_arr = np.asarray(previous_x, dtype=float)
    if x_arr.shape != prev_arr.shape:
        raise ValueError("x and previous_x must have identical shape")
    return float(np.sqrt(lam)) * (x_arr - prev_arr)


def _constraint_diagnostics(
    model: str,
    params: Dict[str, float],
    *,
    constraint_penalty: float = 100.0,
    feller_eps: float = 0.0,
    svcj_moment_eps: float = 1e-6,
    carr_madan_q_values: tuple[float, ...] = (),
) -> Dict[str, float | bool]:
    """Pure diagnostic summary for the soft model constraints."""
    model_l = str(model).lower()
    penalty_active = float(constraint_penalty) > 0.0
    out: Dict[str, float | bool] = {
        "feller_penalty_active": bool(penalty_active and model_l in {"heston", "svcj"}),
        "feller_gap": float("nan"),
        "feller_violated": False,
        "svcj_moment_penalty_active": bool(penalty_active and model_l == "svcj"),
        "svcj_martingale_margin": float("nan"),
        "svcj_martingale_violated": False,
        "svcj_contour_margin": float("nan"),
        "svcj_contour_violated": False,
    }

    if model_l in {"heston", "svcj"}:
        gap = 2.0 * float(params["kappa"]) * float(params["theta"]) - float(params["sigma_v"]) ** 2
        out["feller_gap"] = float(gap)
        out["feller_violated"] = bool(gap < float(feller_eps))

    if model_l == "svcj":
        ell_v = float(params["ell_v"])
        rho_j = float(params["rho_j"])
        martingale_margin = 1.0 - ell_v * rho_j
        out["svcj_martingale_margin"] = float(martingale_margin)
        out["svcj_martingale_violated"] = bool(martingale_margin <= float(svcj_moment_eps))

        q_values = tuple(float(q) for q in carr_madan_q_values)
        if q_values:
            contour_margin = min(1.0 - ell_v * q * rho_j for q in q_values)
            out["svcj_contour_margin"] = float(contour_margin)
            out["svcj_contour_violated"] = bool(contour_margin <= float(svcj_moment_eps))

    return out


def _constraint_residuals(
    model: str,
    params: Dict[str, float],
    *,
    constraint_penalty: float,
    feller_eps: float,
    svcj_moment_eps: float,
    carr_madan_q_values: tuple[float, ...] = (),
) -> np.ndarray:
    """Return fixed-dimension finite residuals for soft constraints."""
    model_l = str(model).lower()
    diag = _constraint_diagnostics(
        model_l,
        params,
        constraint_penalty=constraint_penalty,
        feller_eps=feller_eps,
        svcj_moment_eps=svcj_moment_eps,
        carr_madan_q_values=carr_madan_q_values,
    )
    pen: list[float] = []
    if model_l in {"heston", "svcj"}:
        feller_violation = max(0.0, float(feller_eps) - float(diag["feller_gap"]))
        pen.append(float(constraint_penalty) * feller_violation)
    if model_l == "svcj":
        martingale_violation = max(0.0, float(svcj_moment_eps) - float(diag["svcj_martingale_margin"]))
        pen.append(float(constraint_penalty) * martingale_violation)
        if carr_madan_q_values:
            contour_violation = max(0.0, float(svcj_moment_eps) - float(diag["svcj_contour_margin"]))
            pen.append(float(constraint_penalty) * contour_violation)
    arr = np.asarray(pen, dtype=float)
    return np.where(np.isfinite(arr), arr, float(constraint_penalty))


# ---- parameter transforms'''
s = regex_once(s, pattern, replacement, "objective helper refactor")
s = s.replace("w_full = _weights(df, weight_config)", "w_full = _residual_scales(df, weight_config)")

# Add q-values immediately after the pricing plan is built.
s = replace_once(
    s,
    '''    pricing_plan = _build_pricing_plan(
        df_fit,
        fft_params_base=fft_params_base,
        dynamic_b=dynamic_b,
        fft_params_by_expiry=fft_params_by_expiry,
    )

    # preallocate buffers''',
    '''    pricing_plan = _build_pricing_plan(
        df_fit,
        fft_params_base=fft_params_base,
        dynamic_b=dynamic_b,
        fft_params_by_expiry=fft_params_by_expiry,
    )
    carr_madan_q_values = tuple(sorted({float(group[-1].alpha) + 1.0 for group in pricing_plan}))

    # preallocate buffers''',
    "pricing-plan contour q values",
)

# Replace the nested constraint helper through the beginning of residuals.
pattern = r'''    def _constraint_residuals\(p: Dict\[str, float\]\) -> np\.ndarray:\n.*?    def residuals\(x: np\.ndarray\) -> np\.ndarray:\n        p = unpack\(x\)\n        pen = _constraint_residuals\(p\)\n        reg = reg_scale \* \(x - reg_anchor_x\) if reg_dim > 0 else np\.empty\(0, dtype=float\)\n\n        if not _svcj_endpoint_admissible\(p\):\n            return _penalty_result\(pen, reg\)\n'''
replacement = '''    def _penalty_result(pen: np.ndarray, reg: np.ndarray) -> np.ndarray:
        r = penalty_coin * w
        r = np.where(np.isfinite(r), r, penalty_coin)
        if pen.size:
            r = np.concatenate([r, pen])
        return np.concatenate([r, reg]) if reg.size else r

    def residuals(x: np.ndarray) -> np.ndarray:
        p = unpack(x)
        pen = _constraint_residuals(
            model_l,
            p,
            constraint_penalty=float(constraint_penalty),
            feller_eps=float(feller_eps),
            svcj_moment_eps=float(svcj_moment_eps),
            carr_madan_q_values=carr_madan_q_values,
        )
        reg = _regularization_residual(x, reg_anchor_x, strength=l2_prev_strength)
        diag = _constraint_diagnostics(
            model_l,
            p,
            constraint_penalty=float(constraint_penalty),
            feller_eps=float(feller_eps),
            svcj_moment_eps=float(svcj_moment_eps),
            carr_madan_q_values=carr_madan_q_values,
        )

        if model_l == "svcj" and (
            bool(diag["svcj_martingale_violated"]) or bool(diag["svcj_contour_violated"])
        ):
            return _penalty_result(pen, reg)
'''
s = regex_once(s, pattern, replacement, "nested constraints to pure helpers")

# Remove now-unused reg_scale assignment.
s = s.replace('    reg_scale = float(np.sqrt(l2_prev_strength)) if reg_dim > 0 else 0.0\n', '')

# Replace concatenation conditions based on reg_dim with actual residual size where needed.
s = s.replace('return np.concatenate([r, reg]) if reg_dim > 0 else r', 'return np.concatenate([r, reg]) if reg.size else r')

# Add final diagnostics before return.
s = replace_once(
    s,
    '''    return CalibrationResult(
        model=model_l,
        params=params_hat,
        success=bool(success),
        message=str(message),
        nfev=int(res.nfev),
        rmse=rmse,
        mae=mae,
    )
''',
    '''    diag_hat = _constraint_diagnostics(
        model_l,
        params_hat,
        constraint_penalty=float(constraint_penalty),
        feller_eps=float(feller_eps),
        svcj_moment_eps=float(svcj_moment_eps),
        carr_madan_q_values=carr_madan_q_values,
    )
    return CalibrationResult(
        model=model_l,
        params=params_hat,
        success=bool(success),
        message=str(message),
        nfev=int(res.nfev),
        rmse=rmse,
        mae=mae,
        regularization_active=bool(reg_dim > 0 and l2_prev_strength > 0.0),
        l2_prev_strength=float(l2_prev_strength),
        **diag_hat,
    )
''',
    "CalibrationResult output diagnostics",
)

p.write_text(s, encoding="utf-8")


# ---------------------------------------------------------------------------
# src/snapshot_job.py: persist diagnostics from CalibrationResult
# ---------------------------------------------------------------------------
p = ROOT / "src/snapshot_job.py"
s = p.read_text(encoding="utf-8")
s = replace_once(
    s,
    '''        n_test: int,
        params: Optional[dict[str, float]],
    ) -> pd.DataFrame:
''',
    '''        n_test: int,
        params: Optional[dict[str, float]],
        diagnostics: Optional[dict[str, Any]] = None,
    ) -> pd.DataFrame:
''',
    "param row diagnostics argument",
)
s = replace_once(
    s,
    '''            random_seed=int(random_seed),
        )
        params = params or {}
        base.update(params)
''',
    '''            random_seed=int(random_seed),
            regularization_active=False,
            l2_prev_strength=float(l2_prev_strength),
            feller_penalty_active=False,
            feller_gap=np.nan,
            feller_violated=False,
            svcj_moment_penalty_active=False,
            svcj_martingale_margin=np.nan,
            svcj_martingale_violated=False,
            svcj_contour_margin=np.nan,
            svcj_contour_violated=False,
        )
        params = params or {}
        base.update(params)
        base.update(diagnostics or {})
''',
    "param row diagnostic defaults",
)
s = replace_once(
    s,
    '''        # CalibrationResult dataclass — normal success or optimizer failure.
        return _make_param_row(
            model, success=bool(result.success), message=result.message, nfev=result.nfev,
            rmse_fit=result.rmse, mae_fit=result.mae,
            params=dict(result.params), **common,
        )
''',
    '''        # CalibrationResult dataclass — normal success or optimizer failure.
        diag_fields = {
            name: getattr(result, name)
            for name in (
                "regularization_active",
                "l2_prev_strength",
                "feller_penalty_active",
                "feller_gap",
                "feller_violated",
                "svcj_moment_penalty_active",
                "svcj_martingale_margin",
                "svcj_martingale_violated",
                "svcj_contour_margin",
                "svcj_contour_violated",
            )
            if hasattr(result, name)
        }
        return _make_param_row(
            model, success=bool(result.success), message=result.message, nfev=result.nfev,
            rmse_fit=result.rmse, mae_fit=result.mae,
            params=dict(result.params), diagnostics=diag_fields, **common,
        )
''',
    "persist CalibrationResult diagnostics",
)
p.write_text(s, encoding="utf-8")


# ---------------------------------------------------------------------------
# src/results_store.py: Task 6 ruling — schema must retain persisted diagnostics.
# ---------------------------------------------------------------------------
p = ROOT / "src/results_store.py"
s = p.read_text(encoding="utf-8")
s = replace_once(
    s,
    '''    "random_seed",
]
''',
    '''    "random_seed",
    "regularization_active",
    "l2_prev_strength",
    "feller_penalty_active",
    "feller_gap",
    "feller_violated",
    "svcj_moment_penalty_active",
    "svcj_martingale_margin",
    "svcj_martingale_violated",
    "svcj_contour_margin",
    "svcj_contour_violated",
]
''',
    "workbook diagnostic columns",
)
p.write_text(s, encoding="utf-8")


# ---------------------------------------------------------------------------
# docs
# ---------------------------------------------------------------------------
p = ROOT / "docs/code_documentation.md"
s = p.read_text(encoding="utf-8")
if "### Calibration objective semantics" not in s:
    s += textwrap.dedent('''

### Calibration objective semantics

`WeightConfig` controls residual *scales*, not final least-squares weights. The
optimizer receives `r_i = s_i (P_model_i-P_market_i)`; therefore the objective
contains `s_i^2 error_i^2`. With the baseline spread-only configuration and
`spread_power=1`, `s_i=1/(spread_i+eps)` and the squared objective is
inverse-spread-squared. Temporal regularization is appended in transformed
parameter coordinates as `sqrt(lambda_TS) (x_t-x_{t-1})`, where `x` is the
packed calibration vector.

The Feller condition is a soft finite residual, not a hard feasibility bound.
Calibration results and workbook parameter rows persist whether regularization
was active, the Feller gap/violation flag, and SVCJ martingale/shifted-contour
moment margins and violation flags. These generated fields are the canonical
inputs for later parameter-stability diagnostics.
''')
p.write_text(s, encoding="utf-8")
