"""
scs_core.py — Foundation library for the replication of

    Kozak, Nagel & Santosh (2020), "Shrinking the Cross-Section",
    Journal of Financial Economics 135(2), 271-292.

Ported (and fixed) from the reference implementation
(SCS_L2est.py / utils.py / cross_validate.py / load_managed_portfolios.py).

This module implements the L2 (ridge / "economic" Bayesian shrinkage) SDF
estimator and its cross-validation, as described in:
  - Section 4 (the L2 estimator), eq. (18)-(23) of KNS (2020);
  - eq. (30) for the cross-sectional R^2 objective;
  - the kappa <-> penalty mapping discussed around eq. (18)-(20).

All pinned fixes/assumptions carry their assumption id [A-Cxx] inline.

Conventions (non-negotiable, per committee mandate):
  - no look-ahead (TRAIN-window moments only inside CV legs),
  - assertions on shapes, NaN/inf, monotonic & unique dates,
  - deterministic (explicit sorts; CV folds are contiguous, no RNG).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Data loader
# ---------------------------------------------------------------------------
def load_anom50_daily(path):
    """
    Load the 50-anomaly daily managed-portfolio panel.

    KNS (2020), Section 2 / Table A.1: 50 anomaly long-short portfolios plus the
    value-weighted market excess return (`rme`). The CSV holds RAW long-short
    returns; orthogonalization (de-market) is applied later in-code, NOT here.

    Steps:
      - read CSV, parse `date` with '%m/%d/%Y';
      - drop derived-instrument columns (prefixes 'rX_', 'r2_', 'r3_') and 're_ew';
      - dropna on the 50 anomaly columns (required-history rule: a date enters the
        realized sample only once every anomaly is observed -> realized start
        1973-11-01, no survivorship conditioning beyond column availability);
      - sort by date ascending.

    Returns
    -------
    dates : pd.DatetimeIndex   (tz-naive calendar dates, strictly increasing/unique)
    re    : pd.DataFrame        (T x 50) anomaly returns, indexed by dates
    mkt   : pd.Series           market excess return `rme`, indexed by dates
    T     : int                 number of realized daily observations

    Asserts
    -------
    [A-C16] realized first date == 1973-11-01
    [A-C17] exactly 50 anomaly columns
            + strictly increasing/unique dates, no NaN/inf.
    """
    raw = pd.read_csv(path)
    raw["date"] = pd.to_datetime(raw["date"], format="%m/%d/%Y")

    # Drop derived-instrument columns and the equal-weighted market.
    drop_prefixes = ("rX_", "r2_", "r3_")
    drop_cols = [c for c in raw.columns if c.startswith(drop_prefixes)]
    if "re_ew" in raw.columns:
        drop_cols.append("re_ew")
    raw = raw.drop(columns=drop_cols)

    anom_cols = [c for c in raw.columns if c not in ("date", "rme")]
    # [A-C17] exactly 50 anomaly columns
    assert len(anom_cols) == 50, f"expected 50 anomaly cols, got {len(anom_cols)}"

    # Required-history rule: keep dates where all 50 anomalies are observed.
    raw = raw.dropna(subset=anom_cols)
    raw = raw.sort_values("date").reset_index(drop=True)

    dates = pd.DatetimeIndex(raw["date"].values)
    re = raw[anom_cols].copy()
    re.index = dates
    mkt = raw["rme"].copy()
    mkt.index = dates
    mkt.name = "rme"

    T = len(dates)

    # [A-C16] realized first date
    assert dates[0] == pd.Timestamp("1973-11-01"), \
        f"realized start {dates[0]} != 1973-11-01"
    # monotonic, unique
    assert dates.is_monotonic_increasing and dates.is_unique, \
        "dates must be strictly increasing and unique"
    # no NaN/inf anywhere
    assert np.isfinite(re.to_numpy()).all(), "NaN/inf in anomaly returns"
    assert np.isfinite(mkt.to_numpy()).all(), "NaN/inf in market returns"

    return dates, re, mkt, T


# ---------------------------------------------------------------------------
# 2. De-market  [A-C1]
# ---------------------------------------------------------------------------
def demarket(r, mkt, b=None):
    """
    Orthogonalize each return series w.r.t. the market (de-market).

    KNS (2020), Section 2: anomaly returns are projected onto the market so the
    SDF is built in the space of market-neutral returns. For each column j:
        r_j = a_j + b_j * mkt + resid_j     (OLS)
    we keep the slope b_j and return resid_j = r_j - b_j * mkt (intercept a_j
    discarded so the de-marketed series is a pure long-short residual).

    FIX [A-C1]: the original (utils.demarket) had a broken control flow where the
    caller sometimes received only a DataFrame and sometimes a tuple, via an
    isinstance branch in SCS_L2est. This version ALWAYS returns (resid, b);
    callers must unpack. Passing a precomputed `b` (e.g. TRAIN-window betas)
    skips re-estimation so an OOS leg incurs no look-ahead.

    Parameters
    ----------
    r   : (T x H) DataFrame or ndarray of returns
    mkt : (T,)    Series or ndarray of market excess returns
    b   : optional (H,) precomputed betas; if None, estimated by OLS on [1, mkt]

    Returns
    -------
    resid : same type/shape as r, de-marketed returns
    b     : (H,) array of market betas (slopes)
    """
    is_df = isinstance(r, pd.DataFrame)
    cols = r.columns if is_df else None
    idx = r.index if is_df else None

    R = np.asarray(r, dtype=float)
    m = np.asarray(mkt, dtype=float).reshape(-1)
    assert R.shape[0] == m.shape[0], "r and mkt length mismatch"

    if b is None:
        rhs = np.column_stack([np.ones_like(m), m])          # [1, mkt]
        coef, *_ = np.linalg.lstsq(rhs, R, rcond=None)        # (2 x H)
        b = coef[1, :]                                        # slopes only
    b = np.asarray(b, dtype=float).reshape(-1)
    assert b.shape[0] == R.shape[1], "beta length must equal #assets"

    resid = R - np.outer(m, b)                                # intercept discarded
    assert np.isfinite(resid).all(), "NaN/inf after de-market"

    if is_df:
        resid = pd.DataFrame(resid, index=idx, columns=cols)
    return resid, b


# ---------------------------------------------------------------------------
# 3. De-vol  [A-C3]
# ---------------------------------------------------------------------------
def devol(r, market_sigma, asset_sigma=None, ddof=0):
    """
    Rescale each return column to a common target volatility (de-vol).

    KNS (2020), Section 2 / Internet Appendix: each de-marketed anomaly is scaled
    to the volatility of the value-weighted market so the SDF coefficients are
    comparable across portfolios:
        r_j  ->  r_j * (market_sigma / asset_sigma_j)

    [A-C3] To avoid look-ahead in the OOS leg, `asset_sigma` and `market_sigma`
    may be passed precomputed from the TRAIN window. When `asset_sigma` is None it
    defaults to the per-column std of `r` (full-sample use, in-sample analyses).

    Parameters
    ----------
    r            : (T x H) DataFrame or ndarray (de-marketed returns)
    market_sigma : scalar target sigma (e.g. mkt.std())
    asset_sigma  : optional (H,) per-column sigmas; default r.std(axis=0, ddof=ddof)
    ddof         : delta dof for the default std (0 to match np.cov-free scaling)

    Returns
    -------
    same type/shape as r, rescaled to target sigma.
    """
    is_df = isinstance(r, pd.DataFrame)
    cols = r.columns if is_df else None
    idx = r.index if is_df else None
    R = np.asarray(r, dtype=float)

    if asset_sigma is None:
        asset_sigma = R.std(axis=0, ddof=ddof)
    asset_sigma = np.asarray(asset_sigma, dtype=float).reshape(-1)
    assert asset_sigma.shape[0] == R.shape[1], "asset_sigma length mismatch"
    assert (asset_sigma > 0).all(), "zero/negative asset sigma"

    out = R * (float(market_sigma) / asset_sigma)
    assert np.isfinite(out).all(), "NaN/inf after de-vol"

    if is_df:
        out = pd.DataFrame(out, index=idx, columns=cols)
    return out


# ---------------------------------------------------------------------------
# 4. Regularized covariance  [A-C4][A-C23]
# ---------------------------------------------------------------------------
def regcov(r, ddof=1):
    """
    Covariance with a flat-Wishart (identity) shrinkage prior.

    KNS (2020), Section 4 / eq. (around 18-23): the sample covariance is shrunk
    toward a scaled identity to stabilize the inverse used in the SDF:
        S = cov(r)                       (ddof exposed; default 1)
        a = H / (H + T)
        X = a * trace(S)/H * I  +  (1 - a) * S

    [A-C4] flat-Wishart identity shrink.  [A-C23] ddof default = 1, exposed.

    Parameters
    ----------
    r    : (T x H) DataFrame or ndarray
    ddof : delta dof passed to np.cov (default 1)

    Returns
    -------
    X : (H x H) shrunk covariance matrix (symmetric, finite)
    """
    R = np.asarray(r, dtype=float)
    T, H = R.shape
    S = np.cov(R, rowvar=False, ddof=ddof)
    a = H / (H + T)
    X = a * np.trace(S) / H * np.eye(H) + (1.0 - a) * S
    X = 0.5 * (X + X.T)                       # enforce numerical symmetry
    assert np.isfinite(X).all(), "NaN/inf in regcov"
    return X


# ---------------------------------------------------------------------------
# 5. L2 estimator  eq. (22)-(23)
# ---------------------------------------------------------------------------
def l2est(X, y, l, T, compute_se=False):
    """
    L2 (ridge) SDF coefficient estimator.

    KNS (2020), eq. (22):  b = (X + l*I)^{-1} y
    KNS (2020), eq. (23):  se = sqrt( diag( (X + l*I)^{-1} ) / T )

    Parameters
    ----------
    X          : (H x H) (regularized) covariance of returns
    y          : (H,) mean returns
    l          : scalar L2 penalty
    T          : sample length (for standard errors)
    compute_se : if True also return eq. (23) standard errors

    Returns
    -------
    (b, se) : b is (H,); se is (H,) (NaN-filled if compute_se is False)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    H = X.shape[0]
    A = X + l * np.eye(H)
    if compute_se:
        Xinv = np.linalg.inv(A)
        b = Xinv @ y
        se = np.sqrt(np.diag(Xinv) / T)
    else:
        b = np.linalg.solve(A, y)
        se = np.full(H, np.nan)
    assert np.isfinite(b).all(), "NaN/inf in L2 coefficients"
    return b, se


# ---------------------------------------------------------------------------
# 6. kappa <-> penalty mapping  [A-C5]
# ---------------------------------------------------------------------------
def kappa2pen(kappa, T, trX, freq):
    """
    Map prior root-expected-SR^2 (kappa) to the L2 penalty l.

    KNS (2020), Section 4: the penalty scales with the trace of the second-moment
    matrix and inversely with the squared prior on the SDF Sharpe ratio:
        l = freq * trace(X) / T / kappa^2          [A-C5]

    Parameters
    ----------
    kappa : scalar prior (root expected SR^2)
    T     : sample length
    trX   : trace of the (regularized) covariance X
    freq  : annualization frequency (252 daily, 12 monthly)
    """
    return freq * trX / T / (kappa ** 2)


# ---------------------------------------------------------------------------
# 7. Data-driven kappa grid  [A-C10]
# ---------------------------------------------------------------------------
def build_kappa_grid(X, y, T, freq, gridsize=100):
    """
    Build the kappa grid with the original data-driven right edge.

    Reproduces SCS_L2est.py lines 155-172 (the "coefficient-stabilize" rule):
      - sweep kappa = 2^0, 2^1, ..., 2^20;
      - at each step estimate b via l2est and measure the mean (over assets) of
        the relative coefficient change |Δb| / (1 + |b_prev|);
      - the right edge x_rlim is the largest exponent where that mean still
        exceeds 0.01 (coefficients not yet stabilized);
      - return a log-spaced grid from 2^x_rlim down to 0.01:
            x = logspace(log10(2^x_rlim), log10(0.01), gridsize)
      - and the matching penalty list l = [kappa2pen(x_i, ...)].

    [A-C10] data-driven right-edge / coefficient-stabilize rule.

    Returns
    -------
    x : (gridsize,) ndarray, kappa grid (descending, > 0)
    l : list of penalties matching x
    """
    trX = np.trace(X)
    lr = np.arange(1, 22)                      # exponents 1..21 (kappa = 2^(i-1))
    z = np.full((X.shape[0], len(lr)), np.nan)
    for i in lr:
        kappa = 2.0 ** (i - 1)
        l_i = kappa2pen(kappa, T, trX, freq)
        z[:, i - 1], _ = l2est(X, y, l_i, T)

    rel_change = np.abs(z[:, 1:] - z[:, :-1]) / (1 + np.abs(z[:, :-1]))
    not_stable = np.mean(rel_change, axis=0) > 0.01
    x_rlim = np.nonzero(not_stable)[0]
    assert x_rlim.size > 0, "coefficient-stabilize rule found no right edge"
    right_exp = x_rlim[-1]                      # matches original z[:,1:] indexing

    x = np.logspace(np.log10(2.0 ** right_exp), np.log10(0.01), gridsize)
    l = [kappa2pen(val, T, trX, freq) for val in x]
    assert np.all(np.isfinite(x)) and np.all(x > 0), "bad kappa grid"
    return x, l


# ---------------------------------------------------------------------------
# 8. Contiguous CV partition
# ---------------------------------------------------------------------------
def cvpartition_contiguous(n, k):
    """
    Partition [0, n) into k contiguous index blocks (last block takes remainder).
    Deterministic — no shuffling — so CV folds respect the time ordering.
    """
    s = n // k
    folds = [list(range(s * i, s * (i + 1))) for i in range(k - 1)]
    folds.append(list(range(s * (k - 1), n)))
    assert sum(len(f) for f in folds) == n, "partition does not cover all rows"
    return folds


# ---------------------------------------------------------------------------
# 9. Cross-validation of cross-sectional R^2  eq. (30)  [A-C6][A-C7][A-C8]
# ---------------------------------------------------------------------------
def cross_validate_csr2(dates, r, l_value, kfold, freq):
    """
    K-fold contiguous CV of the cross-sectional R^2 (eq. 30) for one penalty.

    KNS (2020), eq. (30):  R^2 = 1 - (yhat - y)'(yhat - y) / (y'y),
    where yhat = X . b and b is the L2 estimate.

    For each contiguous fold f (the TEST block):
      - TRAIN = all other rows; estimate moments X_tr = regcov(TRAIN),
        y_tr = mean(TRAIN);  TEST moments X_te = regcov(TEST), y_te = mean(TEST);
      - estimate b at the penalty l_value, applying the CV penalty rescale
        [A-C6]:  l_used = l_value / (1 - 1/kfold)   (the train block is a
        (1-1/k) fraction of the data, so the penalty is scaled to keep the same
        per-observation shrinkage as the full-sample fit);
      - IS  R^2 uses (X_tr, y_tr);  OOS R^2 uses (X_te, y_te).

    No look-ahead: moments are computed strictly within each fold's own block.

    Returns  [A-C7][A-C8]
    -------
    dict with:
      is_mean, is_se, oos_mean, oos_se   (se = std across folds / sqrt(k))
      is_folds, oos_folds                (per-fold arrays)
    """
    R = np.asarray(r, dtype=float)
    n = R.shape[0]
    assert pd.DatetimeIndex(dates).is_monotonic_increasing, "dates not sorted"
    folds = cvpartition_contiguous(n, kfold)
    l_used = l_value / (1.0 - 1.0 / kfold)             # [A-C6] CV penalty rescale

    is_vals = np.full(kfold, np.nan)
    oos_vals = np.full(kfold, np.nan)
    for f, idx_test in enumerate(folds):
        idx_train = np.setdiff1d(np.arange(n), idx_test, assume_unique=False)
        tr = R[idx_train, :]
        te = R[idx_test, :]

        X_tr = regcov(tr)
        y_tr = tr.mean(axis=0)
        X_te = regcov(te)
        y_te = te.mean(axis=0)

        b, _ = l2est(X_tr, y_tr, l_used, len(idx_train))

        yhat_is = X_tr @ b
        yhat_oos = X_te @ b
        is_vals[f] = 1.0 - ((yhat_is - y_tr) @ (yhat_is - y_tr)) / (y_tr @ y_tr)
        oos_vals[f] = 1.0 - ((yhat_oos - y_te) @ (yhat_oos - y_te)) / (y_te @ y_te)

    return {
        "is_mean": float(np.mean(is_vals)),
        "is_se": float(np.std(is_vals) / np.sqrt(kfold)),
        "oos_mean": float(np.mean(oos_vals)),
        "oos_se": float(np.std(oos_vals) / np.sqrt(kfold)),
        "is_folds": is_vals,
        "oos_folds": oos_vals,
    }


# ---------------------------------------------------------------------------
# 9b. Cross-validated Sharpe-ratio objective  [A-SR1..15]  (EXTENSION)
# ---------------------------------------------------------------------------
# Extension on top of KNS (2020): a selectable CV objective that maximizes the
# cross-validated annualized Sharpe ratio of the SDF-implied MVE portfolio,
# alongside the existing cross-sectional R^2 objective (cross_validate_csr2).
# Reported diagnostic only [A-SR15]; does not drive any downstream artifact.
# Design + pinned assumptions: replication/extensions/sr_objective/{plan.md,
# open_questions.md}.  All existing functions above are unchanged.
# ---------------------------------------------------------------------------
def sharpe_annual(x, freq):
    """
    Annualized Sharpe ratio of a realized return series.

    SR = mean(x) / std(x, ddof=0) * sqrt(freq)               [A-SR1, A-SR2, A-SR14]

    Scale-invariant in x (homogeneous degree 0): SR(c*x) = sign(c)*SR(x), and for
    c>0 exactly SR(c*x) = SR(x) — this is the property exploited by the SR
    objective (it sees only the *direction* of the SDF coefficient vector b, not
    its magnitude) [A-SR10].

    Degenerate-denominator guard [A-SR8 / B2]: if std(x, ddof=0) <= 1e-12 return
    np.nan (never +/-inf) so a degenerate fold drops out of nan-aware aggregation
    and can never win selection.

    Parameters
    ----------
    x    : (n,) realized return series
    freq : annualization frequency (252 daily)
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    sd = np.std(x, ddof=0)                                    # [A-SR14] ddof=0
    if sd <= 1e-12:                                           # [A-SR8 / B2]
        return np.nan
    return float(np.mean(x) / sd * np.sqrt(freq))


def cross_validate_sr(dates, r, l_value, kfold, freq):
    """
    K-fold contiguous CV of the realized-MVE annualized Sharpe ratio for one
    penalty.  EXTENSION mirroring cross_validate_csr2 exactly (same folds, same
    [A-C6]/[A-SR4] CV penalty rescale, same TRAIN-only moments / no look-ahead).

    For each contiguous fold f (the TEST block):
      - TRAIN = all other rows; X_tr = regcov(TRAIN), y_tr = mean(TRAIN);
      - b estimated at l_used = l_value / (1 - 1/kfold)        [A-SR4 = A-C6];
      - OOS realized MVE return series  r_mve_oos = te @ b      [A-SR1, A-SR6]
        (te = the RAW held-out return block, NOT its moments — this is a
        deliberately different object from CSR2's moment-based Sigma_te @ b);
      - IS realized MVE return series   r_mve_is  = tr @ b;
      - per-fold  is_sr[f]  = sharpe_annual(r_mve_is, freq),
                  oos_sr[f] = sharpe_annual(r_mve_oos, freq);
      - accumulate r_mve_oos into a pool for the pooled (concatenated) diagnostic.

    Selection is on the per-fold-averaged OOS SR [A-SR11 / B3]; pooled_sr is a
    reported diagnostic.  Aggregation is nan-aware [A-SR12] so a degenerate fold
    (std <= 1e-12 -> NaN, [A-SR8/B2]) drops out cleanly.

    No look-ahead [A-SR5]: train/test indices are asserted disjoint; b and its
    moments come strictly from the TRAIN block; OOS SR from the TEST block only.

    Caveat [A-SR14]: the CV-OOS-SR *level* is upward-biased (the penalty is chosen
    on the whole sample) and is used for *ranking* kappa only, not as a true OOS
    Sharpe estimate (mirrors the paper's p.20 caveat).

    Returns  (mirrors the cross_validate_csr2 dict layout)
    -------
    dict with:
      is_mean, is_se, oos_mean, oos_se   (nan-aware; se = nanstd across folds
                                          / sqrt(kfold), [A-SR9])
      is_folds, oos_folds                (per-fold arrays)
      pooled_sr                          (SR of concatenated OOS series, [A-SR3])
    """
    R = np.asarray(r, dtype=float)
    n = R.shape[0]
    assert pd.DatetimeIndex(dates).is_monotonic_increasing, "dates not sorted"
    folds = cvpartition_contiguous(n, kfold)
    l_used = l_value / (1.0 - 1.0 / kfold)             # [A-SR4 = A-C6] CV rescale

    is_vals = np.full(kfold, np.nan)
    oos_vals = np.full(kfold, np.nan)
    oos_pool = []
    for f, idx_test in enumerate(folds):
        idx_train = np.setdiff1d(np.arange(n), idx_test, assume_unique=False)
        # [A-SR5] no look-ahead: train/test strictly disjoint.
        assert np.intersect1d(idx_train, idx_test).size == 0, \
            "train/test indices overlap (look-ahead)"
        tr = R[idx_train, :]
        te = R[idx_test, :]

        X_tr = regcov(tr)
        y_tr = tr.mean(axis=0)

        b, _ = l2est(X_tr, y_tr, l_used, len(idx_train))   # b from TRAIN only

        r_mve_is = tr @ b                                  # [A-SR1] IS MVE series
        r_mve_oos = te @ b                                 # [A-SR6] OOS MVE series

        is_vals[f] = sharpe_annual(r_mve_is, freq)
        oos_vals[f] = sharpe_annual(r_mve_oos, freq)
        oos_pool.append(r_mve_oos)

    pooled = sharpe_annual(np.concatenate(oos_pool), freq)  # [A-SR3] diagnostic

    return {
        "is_mean": float(np.nanmean(is_vals)),
        "is_se": float(np.nanstd(is_vals) / np.sqrt(kfold)),
        "oos_mean": float(np.nanmean(oos_vals)),
        "oos_se": float(np.nanstd(oos_vals) / np.sqrt(kfold)),
        "is_folds": is_vals,
        "oos_folds": oos_vals,
        "pooled_sr": pooled,
    }


def select_kappa_1se_parsimony(kappa_grid, oos_mean, oos_se):
    """
    Deterministic 1-SE-toward-parsimony selection on a flat / scale-invariant
    OOS objective curve  [A-SR7 / B1].

    The SR(kappa) curve is scale-invariant in b and can be near-flat with ties, so
    a plain argmax is ill-conditioned.  Rule:
      - i_max  = nanargmax(oos_mean);
      - thresh = oos_mean[i_max] - oos_se[i_max]   (one s.e. below the max);
      - among all grid points whose oos_mean >= thresh (the "within-1-s.e. band"),
        choose the MOST-SHRUNK kappa, i.e. the SMALLEST kappa value.

    "Weakly identified" when the band is wide (n_in_band large / band_width large):
    many penalties are statistically indistinguishable and the choice of the most
    shrunk one is a parsimony convention, not a sharp optimum.

    Returns
    -------
    i_sel      : selected grid index (smallest-kappa point in band)
    kappa_sel  : kappa_grid[i_sel]
    band_width : max(kappa in band) - min(kappa in band)
    n_in_band  : number of grid points within 1 s.e. of the max
    """
    kappa_grid = np.asarray(kappa_grid, dtype=float)
    oos_mean = np.asarray(oos_mean, dtype=float)
    oos_se = np.asarray(oos_se, dtype=float)
    i_max = int(np.nanargmax(oos_mean))
    thresh = oos_mean[i_max] - oos_se[i_max]
    in_band = np.where(oos_mean >= thresh)[0]
    assert in_band.size > 0, "empty 1-s.e. band"
    kappas_in_band = kappa_grid[in_band]
    # most-shrunk = smallest kappa value
    i_sel = int(in_band[int(np.argmin(kappas_in_band))])
    band_width = float(np.max(kappas_in_band) - np.min(kappas_in_band))
    n_in_band = int(in_band.size)
    return i_sel, float(kappa_grid[i_sel]), band_width, n_in_band


def cross_validate(objective, dates, r, l_value, kfold, freq):
    """
    Selectable CV-objective dispatcher  [A-SR15 extension].

    objective in {'CSR2', 'SR'}:
      - 'CSR2' -> cross_validate_csr2 (cross-sectional R^2, eq. 30; the canonical
        objective used elsewhere in the replication);
      - 'SR'   -> cross_validate_sr   (realized-MVE annualized Sharpe ratio).
    Both objectives are MAXIMIZED.  Any other value raises ValueError.

    Note the two return dicts differ slightly: the 'SR' dict additionally carries
    'pooled_sr'.  Both share is_mean/is_se/oos_mean/oos_se/is_folds/oos_folds so a
    common selector (e.g. select_kappa_1se_parsimony) works on either.
    """
    if objective == "CSR2":
        return cross_validate_csr2(dates, r, l_value, kfold, freq)
    elif objective == "SR":
        return cross_validate_sr(dates, r, l_value, kfold, freq)
    else:
        raise ValueError(f"unknown objective {objective!r}; use 'CSR2' or 'SR'")


# ---------------------------------------------------------------------------
# 10. Effective degrees of freedom  (fix C18)
# ---------------------------------------------------------------------------
def effective_dof(d, l_list):
    """
    Effective degrees of freedom of the ridge fit along the penalty path.

    df(l) = sum_i  d_i / (d_i + l)   over the eigenvalues d of X.

    FIX [C18]: the original hard-coded reshape(50,1)/reshape(1,100); this version
    uses dynamic shapes from the inputs.

    Parameters
    ----------
    d      : (H,) eigenvalues of X
    l_list : (L,) penalties

    Returns
    -------
    (L,) effective dof at each penalty.
    """
    d = np.asarray(d, dtype=float).reshape(-1, 1)
    l = np.asarray(l_list, dtype=float).reshape(1, -1)
    return np.sum(d / (d + l), axis=0)


# ---------------------------------------------------------------------------
# Self-checks for the SR extension (run only as a script; importable above).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)                       # fixed seed (determinism)
    # --- sharpe_annual scale-invariance [A-SR10] ---
    x = rng.standard_normal(2000) * 0.01 + 0.0003
    base = sharpe_annual(x, 252)
    for c in (0.5, 2.0, 100.0):
        assert abs(sharpe_annual(c * x, 252) - base) < 1e-10, "SR not scale-invariant"
    # --- degenerate-denominator guard [A-SR8/B2] ---
    assert np.isnan(sharpe_annual(np.ones(100), 252)), "zero-vol must be NaN"
    # --- no-lookahead inside cross_validate_sr [A-SR5] (asserted in-fn) ---
    T, H = 600, 5
    R = rng.standard_normal((T, H)) * 0.01 + 0.0002
    dts = pd.date_range("2000-01-01", periods=T, freq="B")
    out = cross_validate_sr(dts, R, l_value=1.0, kfold=3, freq=252)
    assert set(out) >= {"oos_mean", "oos_se", "pooled_sr", "oos_folds"}
    # --- dispatcher ---
    assert cross_validate("SR", dts, R, 1.0, 3, 252)["oos_mean"] == out["oos_mean"]
    try:
        cross_validate("BOGUS", dts, R, 1.0, 3, 252); raise AssertionError("no ValueError")
    except ValueError:
        pass
    print("scs_core SR-extension self-checks passed.")
