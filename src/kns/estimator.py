"""
kns.estimator — the public function-first API for the SCS SDF estimator.

`fit_scs` fits the Kozak, Nagel & Santosh (2020) stochastic-discount-factor
estimator to an arbitrary panel of return data. It is a thin, validated WRAPPER
over the numerical engine in `kns.core` (and `kns.dual_penalty`) — it adds input
validation, guardrails for use outside the paper's daily/market-neutral regime,
and a tidy result object, but performs no estimation math of its own.

Design decisions (pinned with the user / professor panel):
  - function-first API returning an `SCSResult` dataclass;
  - preprocessing (de-market, de-vol) is OPT-IN and OFF by default — general use
    cases should not be expected to supply a market series; the paper pipeline
    (`kns.reproduce`) turns them on explicitly;
  - eq. (23) standard errors are POSTERIOR (conditional on the selected kappa and
    on the covariance), NOT frequentist HAC t-stats — named `se`/`tstat_posterior`;
  - the flat-Wishart covariance shrink a = H/(H+T) is surfaced and warned on when
    material (a > 0.10), and is overridable via `cov_shrink`;
  - guardrails RAISE on unrecoverable misuse and WARN on statistical-quality
    concerns (hybrid policy).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kns import core
from kns import dual_penalty as dp


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------
@dataclass
class SCSResult:
    """
    Fitted SCS estimator.

    Attributes
    ----------
    b : np.ndarray, shape (H,)
        SDF coefficients in the PREPROCESSED return space (KNS eq. 22). When
        `rotate_pc=True` these are mapped back to the input-column space so that
        `predict` stays in input coordinates.
    mve_weights : np.ndarray, shape (H,)
        The same SDF portfolio expressed as weights on the user's ORIGINAL input
        assets (the de-vol rescaling undone). If `demarket=True` there is also an
        implied market leg, reported in `diagnostics['market_weight']`. The
        authoritative MVE-return object is `.predict`, not these weights.
    se : np.ndarray, shape (H,)
        POSTERIOR standard errors (KNS eq. 23), conditional on the selected kappa
        and on the covariance. NOT frequentist/HAC SEs — do not use for classical
        b != 0 hypothesis tests. NaN when `compute_se=False`.
    tstat_posterior : np.ndarray, shape (H,)
        b / se. Same caveat as `se`.
    kappa : float
        Selected (or user-fixed) prior root-expected-SR^2 shrinkage level.
    mve_weights : see above
    cv_curve : dict or None
        When `kappa='cv'`: {'kappa_grid', 'is', 'oos', 'oos_se', 'objective'}.
        None when kappa was fixed.
    eff_dof : float
        Effective degrees of freedom of the ridge fit at the selected kappa.
    objective : str
        'CSR2' or 'SR'.
    diagnostics : dict
        Includes T, H, freq, regcov_a (realized shrink), in_sample_maxSR2,
        demarket, devol, dual_penalty, weakly_identified (SR only),
        market_weight (if demarket), and the stored preprocessing state.
    """

    b: np.ndarray
    mve_weights: np.ndarray
    se: np.ndarray
    tstat_posterior: np.ndarray
    kappa: float
    eff_dof: float
    objective: str
    cv_curve: dict | None = None
    diagnostics: dict = field(default_factory=dict)

    # --- stored preprocessing state (for look-ahead-safe prediction) ---------
    _columns: object = None
    _demarket: bool = False
    _devol: bool = False
    _beta: np.ndarray | None = None
    _asset_sigma: np.ndarray | None = None
    _market_sigma: float | None = None
    _freq: int = 252

    # -- methods --------------------------------------------------------------
    def _preprocess(self, returns, market=None):
        """Apply the STORED training-window de-market betas and de-vol sigmas to
        new returns (no refit -> no look-ahead)."""
        R = _as_2d_array(returns, self._columns)
        if self._demarket:
            if market is None:
                raise ValueError(
                    "this model was fit with demarket=True; predict() needs `market`"
                )
            resid, _ = core.demarket(R, np.asarray(market, float).reshape(-1),
                                     b=self._beta)
            R = resid
        if self._devol:
            R = core.devol(R, self._market_sigma, asset_sigma=self._asset_sigma)
        return np.asarray(R, dtype=float)

    def predict(self, returns, market=None):
        """
        Realized SDF-mimicking / MVE portfolio return series for new data.

        Returns `preprocessed(returns) @ b`, applying the STORED training-window
        de-market/de-vol parameters (KNS Section 3; the same object as `R @ b`
        used in the cross-validation and the Table-4 OOS test). Look-ahead-safe:
        the result for any row depends only on that row.
        """
        R_pre = self._preprocess(returns, market=market)
        return R_pre @ self.b

    def sharpe(self, returns, market=None, freq=None):
        """Annualized Sharpe ratio of `predict(returns)` (KNS, sqrt(freq) rule)."""
        port = self.predict(returns, market=market)
        return core.sharpe_annual(port, self._freq if freq is None else freq)


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------
def _as_2d_array(returns, expected_columns=None):
    """Coerce a DataFrame/ndarray to a 2-D float array, checking column alignment."""
    if isinstance(returns, pd.DataFrame):
        if expected_columns is not None:
            same = list(returns.columns) == list(expected_columns)
            assert same, "returns columns do not match the fitted columns"
        return returns.to_numpy(dtype=float)
    arr = np.asarray(returns, dtype=float)
    assert arr.ndim == 2, "returns must be 2-D (T x H)"
    return arr


def _resolve_inputs(returns, dates):
    """Extract the (T,H) array, the column labels, and a tz-naive DatetimeIndex."""
    if isinstance(returns, pd.DataFrame):
        columns = list(returns.columns)
        values = returns.to_numpy(dtype=float)
        if dates is None and isinstance(returns.index, pd.DatetimeIndex):
            dates = returns.index
    else:
        columns = list(range(np.asarray(returns).shape[1]))
        values = np.asarray(returns, dtype=float)
    if dates is None:
        raise ValueError(
            "no dates: pass a DataFrame with a DatetimeIndex, or the `dates=` argument"
        )
    dates = pd.DatetimeIndex(dates)
    return values, columns, dates


def _check_data_quality(values, dates, on_market_required, demarket):
    """RAISE on unrecoverable problems; the statistical-quality WARNINGS live in
    `_warn_regime`."""
    if values.ndim != 2:
        raise ValueError("returns must be 2-D (T x H)")
    if not np.isfinite(values).all():
        raise ValueError("returns contain NaN/inf — clean the panel before fitting")
    if len(dates) != values.shape[0]:
        raise ValueError("len(dates) must equal returns.shape[0]")
    if not (dates.is_monotonic_increasing and dates.is_unique):
        raise ValueError("dates must be strictly increasing and unique")
    sigma = values.std(axis=0, ddof=0)
    # near-zero relative to the largest column std => effectively constant
    sigma_scale = float(np.max(sigma))
    if (sigma <= 1e-12 * sigma_scale).any():
        raise ValueError("zero/near-zero-variance column(s) present — drop constant assets")
    # duplicate columns (perfectly collinear) break the whitening / inverse
    corr = np.corrcoef(values, rowvar=False)
    off_diag = corr - np.eye(corr.shape[0])
    if np.nanmax(np.abs(off_diag)) > 1 - 1e-10:
        raise ValueError("duplicated/perfectly-collinear column(s) present")
    if demarket and on_market_required:
        raise ValueError("demarket=True requires a `market` series (got None)")


def _infer_freq(dates):
    """Infer the annualization frequency from the median calendar spacing."""
    if len(dates) < 3:
        return None
    median_days = float(np.median(np.diff(dates.values).astype("timedelta64[D]").astype(float)))
    if median_days <= 0:
        return None
    # map a typical spacing to a periods-per-year guess
    if median_days <= 4:        # business-daily
        return 252
    if median_days <= 45:       # monthly
        return 12
    if median_days <= 135:      # quarterly
        return 4
    return 1                    # annual-ish


def _warn_regime(values, dates, freq, regcov_a):
    """WARN on statistical-quality concerns (hybrid guardrail: warn, do not block)."""
    T, H = values.shape
    if T < 2 * H:
        warnings.warn(
            f"T={T} < 2*H={2 * H}: covariance is poorly conditioned and the "
            "posterior SEs are prior-dominated; estimates are fragile.",
            stacklevel=3,
        )
    inferred = _infer_freq(dates)
    if inferred is not None and inferred != freq:
        warnings.warn(
            f"freq={freq} but the date spacing looks ~{inferred}/yr: kappa's "
            "'annualized' meaning and the Sharpe annualization depend on freq "
            "matching the data frequency.",
            stacklevel=3,
        )
    if regcov_a > 0.10:
        warnings.warn(
            f"flat-Wishart covariance shrink a={regcov_a:.3f} (>0.10): a material "
            "fraction of the covariance is the scaled-identity prior; the estimand "
            "differs from an unshrunk-covariance ridge. Set cov_shrink to control it.",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Covariance with a configurable shrink (generalizes core.regcov)
# ---------------------------------------------------------------------------
def _shrunk_cov(F, cov_shrink, ddof=1):
    """
    Covariance with a flat-Wishart identity shrink, with the shrink weight `a`
    configurable. `cov_shrink='wishart'` reproduces `core.regcov` EXACTLY
    (a = H/(H+T)); a float in [0, 1] fixes `a`; None/0 gives the raw covariance.

    Returns (X, a) where X is the (H,H) symmetric covariance and a the realized
    shrink weight (surfaced for the material-shrink warning).
    """
    R = np.asarray(F, dtype=float)
    T, H = R.shape
    S = np.cov(R, rowvar=False, ddof=ddof)
    a_wishart = H / (H + T)
    if cov_shrink == "wishart":
        a = a_wishart
    elif cov_shrink is None or cov_shrink == 0:
        a = 0.0
    else:
        a = float(cov_shrink)
        if not 0.0 <= a <= 1.0:
            raise ValueError("cov_shrink must be 'wishart', None, or a float in [0,1]")
    X = a * np.trace(S) / H * np.eye(H) + (1.0 - a) * S
    X = 0.5 * (X + X.T)
    assert np.isfinite(X).all(), "NaN/inf in shrunk covariance"
    # report the (always-computed) Wishart weight so the user sees the material-
    # shrink warning even when they overrode the shrink.
    return X, a, a_wishart


# ---------------------------------------------------------------------------
# PC rotation (used mainly with the dual penalty; a no-op for the dense ridge MVE)
# ---------------------------------------------------------------------------
def _pc_basis(X):
    """Eigenvectors of X, descending eigenvalue, sign-pinned so each column's
    largest-magnitude loading is positive (deterministic) [A-C22]."""
    eigvals, eigvecs = np.linalg.eigh(X)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    Q = eigvecs[:, order]
    for j in range(Q.shape[1]):
        if Q[np.argmax(np.abs(Q[:, j])), j] < 0:
            Q[:, j] *= -1.0
    return eigvals, Q


# ---------------------------------------------------------------------------
# The public estimator
# ---------------------------------------------------------------------------
def fit_scs(
    returns,
    *,
    market=None,
    dates=None,
    demarket=False,
    devol=False,
    freq=252,
    objective="CSR2",
    kappa="cv",
    kfold=3,
    dual_penalty=False,
    l1=0.0,
    rotate_pc=False,
    gridsize=100,
    cov_shrink="wishart",
    compute_se=True,
):
    """
    Fit the SCS stochastic-discount-factor estimator to a panel of returns.

    Implements the L2 (ridge) SDF of Kozak, Nagel & Santosh (2020): eq. (22)
    `b = (Sigma + gamma I)^{-1} mu`, with the prior shrinkage `kappa` mapped to
    the penalty via eq. (18)-(20), the penalty optionally chosen by K-fold
    cross-validation (eq. 30 CSR2, or the realized-MVE Sharpe), and standard
    errors from eq. (23). Optionally the dual L1+L2 penalty (eq. 28).

    Preprocessing (de-market, de-vol) is OPT-IN: by default `fit_scs` estimates
    the SDF directly on the returns as given. The paper pipeline turns both on.

    Parameters
    ----------
    returns : pandas.DataFrame (T x H) or ndarray
        Asset/factor return panel. A DataFrame's DatetimeIndex supplies `dates`.
    market : array-like (T,), optional
        Market excess return; required iff `demarket=True`.
    dates : DatetimeIndex, optional
        Required when `returns` is an ndarray.
    demarket, devol : bool, default False
        Opt-in KNS preprocessing (orthogonalize to the market; rescale to market
        vol). Both require `market`.
    freq : int, default 252
        Annualization frequency; MUST match the return frequency (sets kappa's
        scale and the Sharpe annualization).
    objective : {'CSR2', 'SR'}, default 'CSR2'
        Cross-validation objective when `kappa='cv'`.
    kappa : 'cv' or float, default 'cv'
        'cv' selects the shrinkage by K-fold CV; a float fixes it.
    kfold : int, default 3
        Number of contiguous CV folds.
    dual_penalty : bool, default False
        Use the elastic-net (L1+L2, eq. 28) instead of the pure ridge.
    l1 : float, default 0.0
        L1 strength for the dual penalty (0 reduces exactly to the ridge).
    rotate_pc : bool, default False
        Estimate in principal-component space (meaningful with the dual penalty;
        a no-op for the dense ridge MVE return). Coefficients are mapped back to
        input coordinates.
    gridsize : int, default 100
        Size of the data-driven kappa grid (CV mode).
    cov_shrink : {'wishart', float, None}, default 'wishart'
        Flat-Wishart covariance shrink weight. 'wishart' = a=H/(H+T) (the paper).
    compute_se : bool, default True
        Compute eq. (23) posterior standard errors.

    Returns
    -------
    SCSResult
    """
    # ---- 1. resolve and validate inputs ------------------------------------
    values, columns, dates = _resolve_inputs(returns, dates)
    market_required = bool(demarket or devol)
    _check_data_quality(values, dates, on_market_required=(market is None),
                        demarket=demarket)
    if devol and market is None:
        raise ValueError("devol=True requires a `market` series (got None)")

    market_arr = None if market is None else np.asarray(market, float).reshape(-1)
    if market_arr is not None and market_arr.shape[0] != values.shape[0]:
        raise ValueError("market length must equal returns.shape[0]")

    # ---- 2. optional KNS preprocessing (store params for look-ahead-safe predict)
    beta = None
    asset_sigma = None
    market_sigma = None
    F = values
    if demarket:
        F, beta = core.demarket(F, market_arr)            # full-sample betas
    if devol:
        market_sigma = float(market_arr.std(ddof=0))
        asset_sigma = F.std(axis=0, ddof=0)
        F = core.devol(F, market_sigma, asset_sigma=asset_sigma)

    T, H = F.shape

    # ---- 3. moments (configurable covariance shrink) -----------------------
    mu = F.mean(axis=0)
    X, regcov_a, a_wishart = _shrunk_cov(F, cov_shrink)
    _warn_regime(values, dates, freq, a_wishart)

    eigvals_X = np.linalg.eigvalsh(X)
    assert (eigvals_X > 0).all(), "covariance not positive definite"
    in_sample_maxsr2 = float(freq * mu @ np.linalg.pinv(X) @ mu)

    # PC rotation: rotate moments; coefficients are mapped back at the end.
    if rotate_pc:
        _, Q = _pc_basis(X)
        mu_fit = Q.T @ mu
        X_fit = Q.T @ X @ Q
        X_fit = 0.5 * (X_fit + X_fit.T)
    else:
        Q = None
        mu_fit = mu
        X_fit = X
    trX = float(np.trace(X_fit))

    # ---- 4. choose the penalty (CV) or use the fixed kappa -----------------
    cv_curve = None
    weakly_identified = None
    if kappa == "cv":
        kappa_grid, penalty_list = core.build_kappa_grid(X_fit, mu_fit, T, freq, gridsize)
        is_curve = np.full(gridsize, np.nan)
        oos_curve = np.full(gridsize, np.nan)
        oos_se_curve = np.full(gridsize, np.nan)
        for i, l_value in enumerate(penalty_list):
            cv = core.cross_validate(objective, dates, F if Q is None else (F @ Q),
                                     l_value, kfold, freq)
            is_curve[i] = cv["is_mean"]
            oos_curve[i] = cv["oos_mean"]
            oos_se_curve[i] = cv["oos_se"]
        if objective == "SR":
            i_sel, kappa_sel, band_width, n_in_band = core.select_kappa_1se_parsimony(
                kappa_grid, oos_curve, oos_se_curve)
            weakly_identified = n_in_band > gridsize // 2
        else:
            i_sel = int(np.nanargmax(oos_curve))
            kappa_sel = float(kappa_grid[i_sel])
            band_width = None
            n_in_band = None
        l_selected = penalty_list[i_sel]
        cv_curve = {
            "kappa_grid": kappa_grid,
            "is": is_curve,
            "oos": oos_curve,
            "oos_se": oos_se_curve,
            "objective": objective,
            "selected_index": i_sel,
            "band_width": band_width,
            "n_in_band": n_in_band,
        }
    else:
        kappa_sel = float(kappa)
        l_selected = core.kappa2pen(kappa_sel, T, trX, freq)

    # ---- 5. final fit at the selected penalty ------------------------------
    if dual_penalty:
        b_fit = dp.enet_sdf(X_fit, mu_fit, gamma2=l_selected, gamma1=l1)
        se_fit = np.full(H, np.nan)
        if compute_se and l1 == 0.0:
            _, se_fit = core.l2est(X_fit, mu_fit, l_selected, T, compute_se=True)
    else:
        b_fit, se_fit = core.l2est(X_fit, mu_fit, l_selected, T, compute_se=compute_se)

    # map coefficients back from PC space to input columns if rotated
    b = b_fit if Q is None else (Q @ b_fit)
    se = se_fit if Q is None else np.full(H, np.nan)   # PC-space SEs not mapped

    # ---- 6. diagnostics, MVE weights, dof ----------------------------------
    eff_dof = float(np.sum(eigvals_X / (eigvals_X + l_selected)))
    tstat = b / se

    # MVE weights in the ORIGINAL input-asset space: undo de-vol; the de-market
    # leg implies a market position (reported separately).
    if devol:
        mve_weights = b * (market_sigma / asset_sigma)
    else:
        mve_weights = b.copy()
    market_weight = None
    if demarket:
        market_weight = float(-(beta @ mve_weights))

    diagnostics = {
        "T": T, "H": H, "freq": freq,
        "regcov_a": float(regcov_a),
        "regcov_a_wishart": float(a_wishart),
        "in_sample_maxSR2": in_sample_maxsr2,
        "demarket": demarket, "devol": devol,
        "dual_penalty": dual_penalty, "l1": float(l1), "rotate_pc": rotate_pc,
        "cov_shrink": cov_shrink,
        "weakly_identified": weakly_identified,
        "market_weight": market_weight,
        "se_kind": "posterior (eq. 23), conditional on kappa; NOT frequentist/HAC",
    }

    return SCSResult(
        b=b, mve_weights=mve_weights, se=se, tstat_posterior=tstat,
        kappa=kappa_sel, eff_dof=eff_dof, objective=objective,
        cv_curve=cv_curve, diagnostics=diagnostics,
        _columns=columns, _demarket=demarket, _devol=devol,
        _beta=beta, _asset_sigma=asset_sigma, _market_sigma=market_sigma,
        _freq=freq,
    )
