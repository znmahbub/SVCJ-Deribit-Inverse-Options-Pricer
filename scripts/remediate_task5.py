from __future__ import annotations

from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected 1 occurrence, found {n}")
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# Calibration transform and controlled admissibility penalties
# ---------------------------------------------------------------------------
p = ROOT / "src/calibration.py"
s = p.read_text(encoding="utf-8")
s = replace_once(
    s,
    "    price_inverse_option,\n",
    "    price_inverse_option,\n"
    "    SVCJAdmissibilityError,\n"
    "    svcj_martingale_moment_ok,\n",
    "calibration admissibility imports",
)
s = replace_once(
    s,
    "            np.arctanh(np.clip(float(params[\"rho_j\"]), -0.999, 0.999)),\n",
    "            float(params[\"rho_j\"]),\n",
    "direct rho_j pack",
)
s = replace_once(
    s,
    "        \"rho_j\": float(np.tanh(x[9])),\n",
    "        \"rho_j\": float(x[9]),\n",
    "direct rho_j unpack",
)
# Preserve the existing bounds array for compatibility, then make the last
# coordinate genuinely unbounded before returning the SVCJ bounds.
s = replace_once(
    s,
    "        return lb, ub\n\n    raise ValueError(\"model must be one of {'black','heston','svcj'}\")\n",
    "        lb[-1] = -np.inf\n"
    "        ub[-1] = np.inf\n"
    "        return lb, ub\n\n"
    "    raise ValueError(\"model must be one of {'black','heston','svcj'}\")\n",
    "unbounded rho_j bounds",
)
old_constraint = '''        if model_l == "svcj":
            moment = 1.0 - float(p["ell_v"]) * float(p["rho_j"])
            moment_violation = max(0.0, float(svcj_moment_eps) - moment)
            pen.append(float(constraint_penalty) * moment_violation)

        if not pen:
'''
new_constraint = '''        if model_l == "svcj":
            moment = 1.0 - float(p["ell_v"]) * float(p["rho_j"])
            moment_violation = max(0.0, float(svcj_moment_eps) - moment)
            pen.append(float(constraint_penalty) * moment_violation)

            # The Carr-Madan contour uses q=alpha+1.  At s=0, B=0, so this
            # endpoint moment is a necessary admissibility condition and gives
            # the optimizer a smooth penalty before any FFT is attempted.
            contour_margins = [
                1.0
                - float(p["ell_v"])
                * (float(group[-1].alpha) + 1.0)
                * float(p["rho_j"])
                for group in pricing_plan
            ]
            contour_margin = min(contour_margins) if contour_margins else float("inf")
            contour_violation = max(0.0, float(svcj_moment_eps) - contour_margin)
            pen.append(float(constraint_penalty) * contour_violation)

        if not pen:
'''
s = replace_once(s, old_constraint, new_constraint, "contour constraint residual")
old_start = '''    def residuals(x: np.ndarray) -> np.ndarray:
        p = unpack(x)
        pen = _constraint_residuals(p)
        reg = reg_scale * (x - reg_anchor_x) if reg_dim > 0 else np.empty(0, dtype=float)

        try:
'''
new_start = '''    def _penalty_result(pen: np.ndarray, reg: np.ndarray) -> np.ndarray:
        r = penalty_coin * w
        r = np.where(np.isfinite(r), r, penalty_coin)
        if pen.size:
            r = np.concatenate([r, pen])
        return np.concatenate([r, reg]) if reg_dim > 0 else r

    def _svcj_endpoint_admissible(p: Dict[str, float]) -> bool:
        if model_l != "svcj":
            return True
        if not svcj_martingale_moment_ok(
            float(p["ell_v"]), float(p["rho_j"]), float(svcj_moment_eps)
        ):
            return False
        for group in pricing_plan:
            q = float(group[-1].alpha) + 1.0
            if (
                1.0 - float(p["ell_v"]) * q * float(p["rho_j"])
                <= float(svcj_moment_eps)
            ):
                return False
        return True

    def residuals(x: np.ndarray) -> np.ndarray:
        p = unpack(x)
        pen = _constraint_residuals(p)
        reg = reg_scale * (x - reg_anchor_x) if reg_dim > 0 else np.empty(0, dtype=float)

        if not _svcj_endpoint_admissible(p):
            return _penalty_result(pen, reg)

        try:
'''
s = replace_once(s, old_start, new_start, "controlled endpoint rejection")
old_except = '''        except Exception:
            r = penalty_coin * w
            r = np.where(np.isfinite(r), r, penalty_coin)
            if pen.size:
                r = np.concatenate([r, pen])
            return np.concatenate([r, reg]) if reg_dim > 0 else r
'''
new_except = '''        except SVCJAdmissibilityError:
            return _penalty_result(pen, reg)
        except Exception:
            return _penalty_result(pen, reg)
'''
s = replace_once(s, old_except, new_except, "controlled CF rejection")
s = replace_once(
    s,
    "    _price_with_plan(pricing_plan, model=model_l, params=params_hat, out_prices=y_model_buf, use_cache=True)\n"
    "    y_hat = y_model_buf\n",
    "    try:\n"
    "        _price_with_plan(pricing_plan, model=model_l, params=params_hat, out_prices=y_model_buf, use_cache=True)\n"
    "    except SVCJAdmissibilityError:\n"
    "        y_model_buf[:] = np.nan\n"
    "    y_hat = y_model_buf\n",
    "safe final SVCJ pricing",
)
p.write_text(s, encoding="utf-8")

# ---------------------------------------------------------------------------
# SVCJ jump-transform domain and Carr-Madan contour checks
# ---------------------------------------------------------------------------
p = ROOT / "src/inverse_fft_pricer.py"
s = p.read_text(encoding="utf-8")
insert_marker = "###############################################################################\n# Characteristic functions for the models\n###############################################################################\n"
helpers = textwrap.dedent(r'''
class SVCJAdmissibilityError(ValueError):
    """Raised when the SVCJ joint-jump transform leaves its moment domain."""


def svcj_martingale_moment_ok(ell_v: float, rho_j: float, eps: float = 1e-12) -> bool:
    """Check the first exponential moment used by the martingale drift."""
    values = np.asarray([ell_v, rho_j, eps], dtype=float)
    if not np.all(np.isfinite(values)) or ell_v <= 0.0 or eps < 0.0:
        return False
    return bool(1.0 - float(ell_v) * float(rho_j) > float(eps))


def _svcj_B_path(
    u: np.ndarray,
    s: np.ndarray,
    *,
    kappa: float,
    sigma_v: float,
    rho: float,
) -> np.ndarray:
    """Heston Riccati B(s;u) on a time/frequency grid."""
    u = np.atleast_1d(np.asarray(u, dtype=np.complex128))
    s = np.atleast_1d(np.asarray(s, dtype=float))
    iu = 1j * u
    d = np.sqrt((kappa - rho * sigma_v * iu) ** 2 + sigma_v ** 2 * (u ** 2 + iu))
    d = np.where(np.real(d) < 0.0, -d, d)
    g = (kappa - rho * sigma_v * iu - d) / (kappa - rho * sigma_v * iu + d)
    d_mat = d[np.newaxis, :]
    g_mat = g[np.newaxis, :]
    exp_ds = np.exp(-d_mat * s[:, np.newaxis])
    return (
        ((kappa - rho * sigma_v * iu)[np.newaxis, :] - d_mat)
        / (sigma_v ** 2)
        * (1.0 - exp_ds)
        / (1.0 - g_mat * exp_ds)
    )


def _svcj_jump_domain_ok(
    u: np.ndarray,
    s: np.ndarray,
    *,
    kappa: float,
    sigma_v: float,
    rho: float,
    ell_v: float,
    rho_j: float,
    eps: float,
) -> bool:
    """Check Re(B+i*u*rho_j) < 1/ell_v, including the s=0 endpoint."""
    if (
        not np.isfinite(ell_v)
        or ell_v <= 0.0
        or not np.isfinite(rho_j)
        or not np.isfinite(eps)
        or eps < 0.0
    ):
        return False
    u = np.atleast_1d(np.asarray(u, dtype=np.complex128))
    iu = 1j * u
    endpoint_margin = 1.0 - ell_v * np.real(iu * rho_j)
    if np.any(~np.isfinite(endpoint_margin)) or np.any(endpoint_margin <= eps):
        return False
    B_s = _svcj_B_path(u, s, kappa=kappa, sigma_v=sigma_v, rho=rho)
    margin = 1.0 - ell_v * np.real(B_s + iu[np.newaxis, :] * rho_j)
    return bool(np.all(np.isfinite(margin)) and np.all(margin > eps))


def svcj_carr_madan_admissible(
    *,
    kappa: float,
    sigma_v: float,
    rho: float,
    ell_v: float,
    rho_j: float,
    alpha: float,
    T: float,
    N: int = 2 ** 12,
    eta: float = 0.1,
    quad_nodes: int = 32,
    eps: float = 1e-12,
) -> bool:
    """Check the joint-jump moment domain on u=v-i(alpha+1).

    For the exponential variance jump the transform exists only if
    Re(B(s;u)+i*u*rho_j) < 1/ell_v for every integration time.  The s=0
    endpoint is checked explicitly because Gauss-Legendre nodes exclude it.
    """
    values = np.asarray(
        [kappa, sigma_v, rho, ell_v, rho_j, alpha, T, eta, eps], dtype=float
    )
    if (
        not np.all(np.isfinite(values))
        or kappa <= 0.0
        or sigma_v <= 0.0
        or ell_v <= 0.0
        or alpha <= 0.0
        or T < 0.0
        or eta <= 0.0
        or int(N) < 1
        or int(quad_nodes) < 1
        or eps < 0.0
    ):
        return False
    q = float(alpha) + 1.0
    if 1.0 - float(ell_v) * q * float(rho_j) <= float(eps):
        return False
    v = np.arange(int(N), dtype=float) * float(eta)
    u = v - 1j * q
    nodes, _ = _leggauss_cached(int(quad_nodes))
    s = 0.5 * float(T) * (nodes + 1.0)
    return _svcj_jump_domain_ok(
        u,
        s,
        kappa=float(kappa),
        sigma_v=float(sigma_v),
        rho=float(rho),
        ell_v=float(ell_v),
        rho_j=float(rho_j),
        eps=float(eps),
    )


''')
if "class SVCJAdmissibilityError" in s:
    raise RuntimeError("SVCJ helpers already present")
s = replace_once(s, insert_marker, helpers + insert_marker, "insert SVCJ helpers")
s = replace_once(
    s,
    "    # Basic parameter domain checks (avoid singularities in jump MGF)\n"
    "    if ell_v <= 0:\n"
    "        raise ValueError(\"SVCJ requires ell_v > 0 (mean of variance jump size exponential distribution).\")\n"
    "    if 1.0 - ell_v * rho_j <= 0:\n"
    "        raise ValueError(\n"
    "            \"SVCJ requires 1 - ell_v * rho_j > 0 so that E[e^{rho_j Z}] is finite (martingale drift adjustment).\"\n"
    "        )\n",
    "    # rho_j is an unbounded conditional jump-size loading. Its domain is\n"
    "    # determined by exponential-moment existence, not correlation bounds.\n"
    "    if ell_v <= 0.0:\n"
    "        raise SVCJAdmissibilityError(\"SVCJ requires ell_v > 0.\")\n"
    "    if not svcj_martingale_moment_ok(ell_v, rho_j, 1e-12):\n"
    "        raise SVCJAdmissibilityError(\n"
    "            \"SVCJ requires 1 - ell_v*rho_j > 0 for the martingale jump moment.\"\n"
    "        )\n",
    "martingale moment check",
)
old_denom = '''    denom = 1.0 - ell_v * (B_s + iu[np.newaxis, :] * rho_j)
    # If the jump MGF denominator gets too close to zero, pricing becomes unstable.
    if (not np.all(np.isfinite(denom))) or np.min(np.abs(denom)) < 1e-12:
        warnings.warn(
            "cf_svcj: jump MGF denominator is near-zero or non-finite "
            "(1 − ell_v*(B(s;u) + i*u*rho_j) ≈ 0). "
            "Check ell_v and rho_j values. Returning NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        phi = np.full_like(u, np.nan, dtype=np.complex128)
        return phi[0] if scalar_input else phi
'''
new_denom = '''    jump_exponent = B_s + iu[np.newaxis, :] * rho_j
    # Existence of E[exp((B+i*u*rho_j) Z)] for exponential Z is a real-domain
    # condition, not merely a nonzero-complex-denominator condition.
    endpoint_margin = 1.0 - ell_v * np.real(iu * rho_j)
    domain_margin = 1.0 - ell_v * np.real(jump_exponent)
    if (
        np.any(~np.isfinite(endpoint_margin))
        or np.any(endpoint_margin <= 1e-12)
        or np.any(~np.isfinite(domain_margin))
        or np.any(domain_margin <= 1e-12)
    ):
        raise SVCJAdmissibilityError(
            "SVCJ joint-jump transform leaves its exponential-moment domain."
        )
    denom = 1.0 - ell_v * jump_exponent
    if (not np.all(np.isfinite(denom))) or np.min(np.abs(denom)) < 1e-12:
        raise SVCJAdmissibilityError(
            "SVCJ jump MGF denominator is singular or non-finite."
        )
'''
s = replace_once(s, old_denom, new_denom, "real jump-transform domain check")
s = replace_once(
    s,
    "    except Exception:\n"
    "        K_arr = np.asarray(K, dtype=float)\n",
    "    except SVCJAdmissibilityError:\n"
    "        # Domain-invalid SVCJ candidates are explicit model failures;\n"
    "        # calibration converts this exception to a controlled penalty.\n"
    "        raise\n"
    "    except Exception:\n"
    "        K_arr = np.asarray(K, dtype=float)\n",
    "propagate admissibility exception",
)
p.write_text(s, encoding="utf-8")

# ---------------------------------------------------------------------------
# Regression fixtures and sanity tests
# ---------------------------------------------------------------------------
p = ROOT / "tests/reference_inputs.py"
s = p.read_text(encoding="utf-8")
insert = '''

def svcj_reference_params(
    heston_params: dict[str, float], *, rho_j: float = -0.30
) -> dict[str, float]:
    """Stable SVCJ regression parameters; rho_j is not a correlation bound."""
    return {
        **heston_params,
        "lam": 0.20,
        "ell_y": -0.05,
        "sigma_y": 0.20,
        "ell_v": 0.10,
        "rho_j": float(rho_j),
    }
'''
anchor = "\ndef build_reference_case() -> ReferenceCase:\n"
if anchor not in s:
    raise RuntimeError("reference_inputs anchor missing")
s = s.replace(anchor, insert + anchor, 1)
s = replace_once(
    s,
    '''    svcj_params = {
        **heston_params,
        "lam": 0.20,
        "ell_y": -0.05,
        "sigma_y": 0.20,
        "ell_v": 0.10,
        "rho_j": -0.30,
    }
''',
    "    svcj_params = svcj_reference_params(heston_params)\n",
    "reference params factory",
)
p.write_text(s, encoding="utf-8")

p = ROOT / "tests/test_sanity_pricing.py"
s = p.read_text(encoding="utf-8")
method = '''
    def test_svcj_negative_jump_loading_outside_unit_interval_prices_finitely(self) -> None:
        """rho_j is a loading, so a stable value below -1 remains admissible."""
        F0 = 10_000.0
        T = 0.25
        K = np.array([8_000.0, 10_000.0, 12_000.0])
        params = {
            "kappa": 2.0,
            "theta": 0.04,
            "sigma_v": 0.30,
            "rho": -0.50,
            "v0": 0.04,
            "lam": 0.20,
            "ell_y": -0.05,
            "sigma_y": 0.20,
            "ell_v": 0.10,
            "rho_j": -2.0,
        }
        N, eta = 2 ** 10, 0.10
        lam_grid = 2.0 * np.pi / (N * eta)
        b = float(np.log(np.median(K))) - 0.5 * N * lam_grid
        fftp = FFTParams(N=N, alpha=1.5, eta=eta, b=b, use_simpson=True)
        price = price_inverse_option(
            "svcj",
            K=K,
            T=T,
            F0=F0,
            params=params,
            option_type="call",
            fft_params=fftp,
            use_cache=False,
        )
        self.assertTrue(np.all(np.isfinite(price)))
'''
anchor = "\n\nif __name__ == \"__main__\":  # pragma: no cover\n"
if anchor not in s:
    raise RuntimeError("test_sanity_pricing main guard missing")
s = s.replace(anchor, "\n" + method.rstrip() + anchor, 1)
p.write_text(s, encoding="utf-8")

p = ROOT / "docs/code_documentation.md"
s = p.read_text(encoding="utf-8")
if "### SVCJ jump loading and FFT admissibility" not in s:
    s += textwrap.dedent(r'''

### SVCJ jump loading and FFT admissibility

`rho_j` is the conditional return-jump loading on the exponential variance
jump, not a correlation coefficient. Calibration stores it directly in
transformed parameter space and does not apply a `tanh`/`[-1,1]` bound.
Admissibility is enforced through moments: `1-ell_v*rho_j>0` for the martingale
first moment and, on the actual Carr-Madan contour `u=v-i(alpha+1)`,
`Re(B(s;u)+i*u*rho_j) < 1/ell_v` for every jump-integration time. The `s=0`
endpoint is checked explicitly. `SVCJAdmissibilityError` is propagated by the
pricer and converted to a controlled calibration penalty rather than being
silently treated as a finite price.
''')
p.write_text(s, encoding="utf-8")
