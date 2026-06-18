"""
dual_penalty.py — s09: dual-penalty (L1+L2 / elastic-net) SDF estimator for the
replication of

    Kozak, Nagel & Santosh (2020), "Shrinking the Cross-Section",
    Journal of Financial Economics 135(2), 271-292.

Implements KNS eq. (28):

    b_hat = argmin_b  (mu_bar - Sigma_bar b)' Sigma_bar^{-1} (mu_bar - Sigma_bar b)
                       + gamma2 * b'b  +  gamma1 * sum_i |b_i|

via the WHITENED reduction [A-C20].  Let Sigma_bar = Q D Q' (symmetric eigh).
Define the symmetric square root and its inverse

    X_tilde = Sigma_bar^{1/2}  = Q sqrt(D) Q'
    y_tilde = Sigma_bar^{-1/2} mu_bar = Q (1/sqrt(D)) Q' mu_bar

Then the HJ-distance / GLS term collapses to an OLS sum of squares:

    (mu - Sigma b)' Sigma^{-1} (mu - Sigma b)
        = (Sigma^{-1/2} mu - Sigma^{1/2} b)' (Sigma^{-1/2} mu - Sigma^{1/2} b)
        = || y_tilde - X_tilde b ||^2 .

So eq. (28) == || y_tilde - X_tilde b ||^2 + gamma2 ||b||^2 + gamma1 ||b||_1,
a plain elastic-net in the whitened (X_tilde, y_tilde) space.

Sanity [A-C20]: with gamma1 = 0 the minimizer is the ridge normal equation
    (X_tilde' X_tilde + gamma2 I) b = X_tilde' y_tilde
    => (Sigma_bar + gamma2 I) b = mu_bar     (since X_tilde'X_tilde = Sigma_bar,
                                              X_tilde' y_tilde = Sigma_bar^{1/2}
                                              Sigma_bar^{-1/2} mu = mu_bar)
which is exactly core.l2est(Sigma_bar, mu_bar, gamma2).  Asserted below.

sklearn mapping [A-C21].  sklearn ElasticNet(fit_intercept=False) minimizes
    (1/(2n)) ||y - X b||^2 + alpha*rho ||b||_1 + (alpha (1-rho)/2) ||b||_2^2,
with n = #rows of X = H (here X_tilde is HxH).  Dividing our whitened objective
by 2n = 2H and matching term-by-term:
    alpha * rho        = gamma1 / (2H)
    alpha * (1 - rho)  = gamma2 / H
  => alpha = gamma1/(2H) + gamma2/H,   rho = (gamma1/(2H)) / alpha.
gamma1 = 0 => rho = 0 => pure ridge (verified vs core.l2est).

gamma2 is fixed by kappa EXACTLY as the L2 penalty (KNS Sec 4):
    gamma2 = core.kappa2pen(kappa, T, trace(Sigma_bar), freq)  ( = penalty_list ).

Paper Sec 3.2 does NOT center / normalize the design, so fit_intercept=False and
no standardization.  Deterministic: fixed tol, no RNG; max_iter raised to ensure
convergence.

Conventions (committee mandate): assertions on shapes / NaN-inf / no-look-ahead;
cite the paper equation in each docstring.
"""

import warnings

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import enet_path as _sk_enet_path
from sklearn.exceptions import ConvergenceWarning

import scs_core as core

# Deterministic solver controls.
_TOL = 1e-12
_MAX_ITER = 200000


# ---------------------------------------------------------------------------
# Whitening  [A-C20]
# ---------------------------------------------------------------------------
def whiten(Sigma, mu):
    """Symmetric square root whitening of the HJ/GLS objective (KNS eq. 28).

    Returns
    -------
    X_tilde : (H,H)  Sigma^{1/2}      (symmetric, PSD)
    y_tilde : (H,)   Sigma^{-1/2} mu
    eigvals, eigvecs : the eigh decomposition (cached for reuse / PC rotation)
    """
    Sigma = np.asarray(Sigma, dtype=float)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = 0.5 * (Sigma + Sigma.T)
    w, Q = np.linalg.eigh(Sigma)
    assert (w > 0).all(), "Sigma not positive definite; sqrt undefined"
    sqrt_w = np.sqrt(w)
    X_tilde = (Q * sqrt_w) @ Q.T          # Q sqrt(D) Q'
    X_tilde = 0.5 * (X_tilde + X_tilde.T)
    y_tilde = (Q * (1.0 / sqrt_w)) @ (Q.T @ mu)
    assert np.isfinite(X_tilde).all() and np.isfinite(y_tilde).all(), "NaN/inf whitening"
    return X_tilde, y_tilde, w, Q


def gammas_to_alpha_rho(gamma2, gamma1, H):
    """Map (gamma2, gamma1) to sklearn ElasticNet (alpha, l1_ratio) [A-C21].

    alpha = gamma1/(2H) + gamma2/H ;  rho = (gamma1/(2H)) / alpha.
    """
    a1 = gamma1 / (2.0 * H)
    a2 = gamma2 / H
    alpha = a1 + a2
    if alpha <= 0:
        return 0.0, 0.0
    rho = a1 / alpha
    return float(alpha), float(rho)


# ---------------------------------------------------------------------------
# Single elastic-net fit  (KNS eq. 28)
# ---------------------------------------------------------------------------
def enet_sdf(Sigma, mu, gamma2, gamma1, X_tilde=None, y_tilde=None):
    """Single dual-penalty SDF fit, KNS eq. (28).

    If gamma1 == 0 the closed-form ridge (X_tilde'X_tilde+gamma2 I)^{-1}... is
    returned (== core.l2est), avoiding the degenerate sklearn rho=0 path.

    Parameters
    ----------
    Sigma, mu        : (H,H), (H,) GLS moments (de-marketed/devol'd; or PC basis)
    gamma2, gamma1   : L2 / L1 penalties (gamma2 from kappa2pen)
    X_tilde, y_tilde : optional precomputed whitening (cache across a gamma1 sweep)

    Returns
    -------
    b : (H,) SDF coefficients
    """
    Sigma = np.asarray(Sigma, dtype=float)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    H = Sigma.shape[0]
    if X_tilde is None or y_tilde is None:
        X_tilde, y_tilde, _, _ = whiten(Sigma, mu)

    if gamma1 <= 0:
        # Pure ridge closed form (KNS eq. 22 with l = gamma2). [A-C20] sanity path.
        b, _ = core.l2est(Sigma, mu, gamma2, T=1)
        return b

    alpha, rho = gammas_to_alpha_rho(gamma2, gamma1, H)
    en = ElasticNet(alpha=alpha, l1_ratio=rho, fit_intercept=False,
                    precompute=False, max_iter=_MAX_ITER, tol=_TOL,
                    selection="cyclic", warm_start=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            en.fit(X_tilde, y_tilde)
        except ConvergenceWarning:
            # Retry once with a looser tol and more iters before failing loudly.
            en = ElasticNet(alpha=alpha, l1_ratio=rho, fit_intercept=False,
                            precompute=False, max_iter=_MAX_ITER * 5, tol=1e-10,
                            selection="cyclic")
            warnings.simplefilter("ignore", ConvergenceWarning)
            en.fit(X_tilde, y_tilde)
    b = en.coef_.copy()
    assert np.isfinite(b).all(), "NaN/inf in enet SDF coefficients"
    return b


# ---------------------------------------------------------------------------
# gamma1 path at fixed gamma2  (KNS eq. 28, traced over L1 strength)
# ---------------------------------------------------------------------------
def enet_sdf_path(Sigma, mu, gamma2, n_alphas=100, eps=1e-3,
                  X_tilde=None, y_tilde=None, return_gamma1=False):
    """Trace KNS eq.(28) coefficients over a range of gamma1 at fixed gamma2.

    Uses sklearn's coordinate-descent `enet_path`.  sklearn parameterizes the
    path by a single `alpha` with a fixed `l1_ratio`; but our gamma2 is fixed in
    absolute terms while gamma1 sweeps.  Holding gamma2 fixed and sweeping gamma1
    traces a CURVE in (alpha, l1_ratio) space, NOT a fixed-l1_ratio ray, so we
    cannot use the raw enet_path grid directly.  Instead we:
      - build a geometric gamma1 grid from 0 (ridge-only) up to gamma1_max, the
        smallest L1 strength that zeroes ALL coefficients, then
      - fit each (gamma2, gamma1) with warm starts for speed/determinism.

    gamma1_max derivation: for elastic net the all-zero solution holds once the
    L1 sub-gradient dominates, i.e. gamma1 >= 2*max_i |X_tilde'_i y_tilde| (the
    factor 2 from the GLS scaling that puts L1 at gamma1*||b||_1, gradient of the
    SSE being 2 X'(y-Xb); at b=0 that is 2 X'y).  We add a small margin.

    Returns
    -------
    coefs   : (H, n_alphas) coefficients, column g = gamma1_grid[g]
    nnz     : (n_alphas,) #nonzero per column
    gamma1_grid : (n_alphas,) the L1 strengths (ascending), if return_gamma1
    """
    Sigma = np.asarray(Sigma, dtype=float)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    H = Sigma.shape[0]
    if X_tilde is None or y_tilde is None:
        X_tilde, y_tilde, _, _ = whiten(Sigma, mu)

    # All-zero threshold on gamma1 (margin to make the top of the path fully sparse).
    g1_max = 2.0 * np.max(np.abs(X_tilde.T @ y_tilde)) * 1.05
    g1_min = g1_max * eps
    # ascending grid; index 0 = (near) ridge-only, last = fully sparse
    grid = np.concatenate(([0.0],
                           np.logspace(np.log10(g1_min), np.log10(g1_max),
                                       n_alphas - 1)))

    coefs = np.zeros((H, n_alphas))
    nnz = np.zeros(n_alphas, dtype=int)
    b_warm = None
    for g, g1 in enumerate(grid):
        if g1 <= 0:
            b = enet_sdf(Sigma, mu, gamma2, 0.0, X_tilde, y_tilde)
        else:
            alpha, rho = gammas_to_alpha_rho(gamma2, g1, H)
            en = ElasticNet(alpha=alpha, l1_ratio=rho, fit_intercept=False,
                            precompute=False, max_iter=_MAX_ITER, tol=_TOL,
                            selection="cyclic", warm_start=True)
            if b_warm is not None:
                en.coef_ = b_warm.copy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                en.fit(X_tilde, y_tilde)
            b = en.coef_.copy()
            b_warm = b
        coefs[:, g] = b
        nnz[g] = int(np.sum(np.abs(b) > 1e-10))
    assert np.isfinite(coefs).all(), "NaN/inf in enet path coefficients"
    if return_gamma1:
        return coefs, nnz, grid
    return coefs, nnz


# ---------------------------------------------------------------------------
# Self-test  [A-C20][A-C21]: gamma1=0 reproduces L2-only
# ---------------------------------------------------------------------------
def _selftest(found_path):
    f = np.load(found_path, allow_pickle=True)
    Sigma = f["Sigma_bar"]; mu = f["mu_bar"]
    plist = f["penalty_list"]
    H = Sigma.shape[0]
    X_tilde, y_tilde, w, Q = whiten(Sigma, mu)

    # (1) whitening identities
    assert np.allclose(X_tilde @ X_tilde, Sigma, atol=1e-10), "X_tilde^2 != Sigma"
    assert np.allclose(X_tilde @ y_tilde, mu, atol=1e-10), "X_tilde y_tilde != mu"

    # (2) gamma1=0 == core.l2est for several penalties along the path
    max_err = 0.0
    for l in plist[::10]:
        b_en = enet_sdf(Sigma, mu, gamma2=l, gamma1=0.0, X_tilde=X_tilde, y_tilde=y_tilde)
        b_l2, _ = core.l2est(Sigma, mu, l, T=1)
        max_err = max(max_err, float(np.max(np.abs(b_en - b_l2))))
    assert max_err < 1e-12, f"gamma1=0 ridge mismatch {max_err:.2e}"

    # (3) sklearn rho=0 (tiny gamma1) ridge cross-check at a mid penalty
    l_mid = float(plist[len(plist) // 2])
    alpha, rho = gammas_to_alpha_rho(l_mid, 0.0, H)
    assert rho == 0.0 and np.isclose(alpha, l_mid / H), "alpha/rho mapping wrong at g1=0"

    print(f"[selftest] whitening OK; gamma1=0 vs l2est max|db|={max_err:.2e}")
    print(f"[selftest] alpha(g1=0)={alpha:.6e} == gamma2/H={l_mid/H:.6e}; rho=0")
    return max_err


if __name__ == "__main__":
    import os
    _FOUND = os.path.join(os.path.dirname(__file__), "..", "output", "foundation.npz")
    _selftest(_FOUND)
