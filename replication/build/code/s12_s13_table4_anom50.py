"""
s12 + s13 — Table 4, out-of-sample MVE alpha test, 50-anomaly row.

Replicates:
    Kozak, Nagel & Santosh (2020), "Shrinking the Cross-Section",
    Journal of Financial Economics 135(2), 271-292.
    Table 4 (out-of-sample) — the 50-anomaly-portfolio row.

The OOS-MVE portfolio is built ENTIRELY from the pre-2005 training block
(betas, vols, kappa*, and the L2 SDF weights b_hat are all estimated on
dates < 2005-01-01). The test block (2005-01-01..2017-12-29) enters ONLY via
the realized OOS portfolio return P_t and the final intercept regression.

For each benchmark factor model we form its OOS return on the test block and
regress the normalized OOS-MVE return on it:

    P_tilde_t = alpha + beta . bench_t + eps_t      (daily, 2005-2017)

[A-C11] OLS, intercept-only (homoskedastic) standard errors, NO Newey-West.
Annualization: alpha_ann% = alpha_daily * 252 * 100,
               SE_ann%    = SE(alpha_daily) * 252 * 100.

Benchmarks (each a factor model whose alpha measures the MVE portfolio's
unexplained mean):
  - CAPM          : factors are market-orthogonal -> bench return = 0,
                    alpha = annualized mean of P_tilde, SE = std/sqrt(N) ann.
  - FF6           : MVE of the 5 market-orthogonalized FF factors {SMB,HML,UMD,
                    RMW,CMA} (UNregularized pre-2005 weights). [A-C25]
  - char-sparse   : dual-penalty (eq.28) on the 50 anomalies; gamma1 tuned to
                    EXACTLY 5 nonzero characteristics; its MVE return.
  - pc-sparse     : same in the PC basis; gamma1 tuned to exactly 5 nonzero PCs.

Conventions (committee mandate): absolute conda interpreter; assertions on
no-look-ahead / NaN-inf / shapes / sample size; deterministic (no RNG; explicit
descending PC order with A-C22 sign-pin).
"""

import json
import os

import numpy as np
import pandas as pd

import scs_core as core
import dual_penalty as dp

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(ROOT, "Data")
ANOM_CSV = os.path.join(DATA, "Instruments", "managed_portfolios_anom_d_50.csv")
FF5_CSV = os.path.join(DATA, "F-F_Research_Data_5_Factors_2x3_daily.csv")
MOM_CSV = os.path.join(DATA, "F-F_Momentum_Factor_daily.csv")
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))

FREQ = 252
SPLIT = pd.Timestamp("2005-01-01")
TEST_END = pd.Timestamp("2017-12-29")
KFOLD = 3
N_TARGET = 5  # sparse benchmarks: exactly 5 nonzeros

REPORTED = {  # 50-anom row of Table 4 (paper): (alpha_ann%, se_ann%)
    "CAPM": (12.35, 5.26),
    "FF6": (8.71, 4.94),
    "char_sparse": (9.55, 3.95),
    "pc_sparse": (4.60, 2.22),
}


# ---------------------------------------------------------------------------
# FF factor loader
# ---------------------------------------------------------------------------
def load_ff_daily():
    """Load FF5 + Momentum daily, percent -> /100, return a DataFrame with
    columns [Mkt-RF, RF, SMB, HML, RMW, CMA, UMD] indexed by date."""
    ff5 = pd.read_csv(FF5_CSV)
    ff5["Date"] = pd.to_datetime(ff5["Date"], format="%Y/%m/%d")
    ff5 = ff5.set_index("Date").astype(float) / 100.0

    mom = pd.read_csv(MOM_CSV)
    mom.columns = [c.strip() for c in mom.columns]
    mom["Date"] = pd.to_datetime(mom["Date"], format="%Y/%m/%d")
    mom = mom.set_index("Date")[["Mom"]].astype(float) / 100.0
    mom = mom.rename(columns={"Mom": "UMD"})

    ff = ff5.join(mom, how="inner")
    assert np.isfinite(ff.to_numpy()).all(), "NaN/inf in FF factors"
    return ff


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ---- 1. Load raw anomalies + FF factors, align ----
    dates, re, mkt, T = core.load_anom50_daily(ANOM_CSV)
    ff = load_ff_daily()

    common = dates.intersection(ff.index)
    common = common[common <= TEST_END]
    common = pd.DatetimeIndex(sorted(common))
    assert common.is_monotonic_increasing and common.is_unique

    re = re.loc[common]
    mkt = mkt.loc[common]                 # anomalies' own market (rme)
    ff = ff.loc[common]

    ff_mkt = ff["Mkt-RF"]                 # FF block's own market
    ff_facs = ff[["SMB", "HML", "UMD", "RMW", "CMA"]]   # 5 non-market FF factors

    # ---- 2. Split at 2005-01-01 ----
    train_mask = common < SPLIT
    test_mask = ~train_mask
    n_tr = int(train_mask.sum())
    n_te = int(test_mask.sum())
    assert n_tr > 0 and n_te > 0, "empty train/test block"

    re_tr = re.loc[train_mask]
    re_te = re.loc[test_mask]
    mkt_tr = mkt.loc[train_mask]
    mkt_te = mkt.loc[test_mask]
    ff_mkt_tr = ff_mkt.loc[train_mask]
    ff_mkt_te = ff_mkt.loc[test_mask]
    ff_facs_tr = ff_facs.loc[train_mask]
    ff_facs_te = ff_facs.loc[test_mask]

    # ---- 3. [A-C1] PRE-2005 beta de-market (anomalies; FF block) ----
    _, b_pre = core.demarket(re_tr, mkt_tr)                       # train betas
    re_tr_dm = re_tr - np.outer(mkt_tr.to_numpy(), b_pre)
    re_te_dm = re_te - np.outer(mkt_te.to_numpy(), b_pre)

    # [A-C25] de-market the 5 FF factors on FF Mkt-RF over pre-2005
    _, b_ff_pre = core.demarket(ff_facs_tr, ff_mkt_tr)
    ff_facs_tr_dm = ff_facs_tr - np.outer(ff_mkt_tr.to_numpy(), b_ff_pre)
    ff_facs_te_dm = ff_facs_te - np.outer(ff_mkt_te.to_numpy(), b_ff_pre)

    # ---- 4. [A-C3] PRE-2005 sigma devol (anomalies; FF block) ----
    mkt_sig_tr = float(np.std(mkt_tr.to_numpy(), ddof=0))
    asset_sig_tr = re_tr_dm.to_numpy().std(axis=0, ddof=0)
    re_tr_dv = core.devol(re_tr_dm, mkt_sig_tr, asset_sigma=asset_sig_tr)
    re_te_dv = core.devol(re_te_dm, mkt_sig_tr, asset_sigma=asset_sig_tr)

    ff_mkt_sig_tr = float(np.std(ff_mkt_tr.to_numpy(), ddof=0))
    ff_asset_sig_tr = ff_facs_tr_dm.to_numpy().std(axis=0, ddof=0)
    ff_facs_tr_dv = core.devol(ff_facs_tr_dm, ff_mkt_sig_tr, asset_sigma=ff_asset_sig_tr)
    ff_facs_te_dv = core.devol(ff_facs_te_dm, ff_mkt_sig_tr, asset_sigma=ff_asset_sig_tr)

    # numpy views
    Rtr = re_tr_dv.to_numpy()
    Rte = re_te_dv.to_numpy()
    Ftr = ff_facs_tr_dv.to_numpy()
    Fte = ff_facs_te_dv.to_numpy()

    assert np.isfinite(Rtr).all() and np.isfinite(Rte).all()
    assert np.isfinite(Ftr).all() and np.isfinite(Fte).all()

    # ---- 5. [A-C6] kappa* via K=3 contiguous CV on PRE-2005 ONLY ----
    Sig_pre = core.regcov(Rtr)
    mu_pre = Rtr.mean(axis=0)
    kappa_grid, l_grid = core.build_kappa_grid(Sig_pre, mu_pre, n_tr, FREQ)
    oos_means = np.array([
        core.cross_validate_csr2(common[train_mask], Rtr, l, KFOLD, FREQ)["oos_mean"]
        for l in l_grid
    ])
    best = int(np.nanargmax(oos_means))
    kappa_star = float(kappa_grid[best])
    l_star = float(l_grid[best])

    # ---- 6. Full-model L2 SDF on FULL pre-2005 ----
    b_hat, _ = core.l2est(Sig_pre, mu_pre, l_star, n_tr)
    assert np.isfinite(b_hat).all()

    # ---- 7. OOS-MVE returns on test block ----
    P = Rte @ b_hat

    # ---- 8. [A-C26] normalize to OOS-period market sigma (anomalies' rme) ----
    mkt_te_sig = float(np.std(mkt_te.to_numpy(), ddof=0))
    P_sig = float(np.std(P, ddof=0))
    P_tilde = P * (mkt_te_sig / P_sig)
    assert np.isclose(np.std(P_tilde, ddof=0), mkt_te_sig, rtol=1e-10)

    # ===================================================================
    # Benchmarks
    # ===================================================================
    def ts_alpha(P_t, bench_t):
        """OLS P_t = a + b.bench_t; return (alpha_ann%, se_ann%) [A-C11]."""
        P_t = np.asarray(P_t, float).reshape(-1)
        N = P_t.shape[0]
        if bench_t is None:
            a = float(P_t.mean())
            se = float(P_t.std(ddof=0) / np.sqrt(N))
        else:
            B = np.asarray(bench_t, float)
            X = np.column_stack([np.ones(N), B]) if B.ndim == 1 else \
                np.column_stack([np.ones(N), B])
            beta, *_ = np.linalg.lstsq(X, P_t, rcond=None)
            resid = P_t - X @ beta
            dof = N - X.shape[1]
            s2 = float(resid @ resid) / dof
            XtX_inv = np.linalg.inv(X.T @ X)
            a = float(beta[0])
            se = float(np.sqrt(s2 * XtX_inv[0, 0]))
        return a * FREQ * 100.0, se * FREQ * 100.0

    cells = {}

    # --- CAPM: bench = 0 ---
    cells["CAPM"] = ts_alpha(P_tilde, None)

    # --- FF6: unregularized MVE of the 5 market-orthogonalized FF factors ---
    Sig_ff_pre = np.cov(Ftr, rowvar=False, ddof=1)
    mu_ff_pre = Ftr.mean(axis=0)
    w_ff = np.linalg.pinv(Sig_ff_pre) @ mu_ff_pre
    ff6_bench = Fte @ w_ff
    cells["FF6"] = ts_alpha(P_tilde, ff6_bench)

    # --- char-sparse 5-factor: dual-penalty, exactly 5 nonzero chars ---
    Xt_pre, yt_pre, _, _ = dp.whiten(Sig_pre, mu_pre)
    coefs_cs, nnz_cs, g1grid_cs = dp.enet_sdf_path(
        Sig_pre, mu_pre, l_star, n_alphas=200,
        X_tilde=Xt_pre, y_tilde=yt_pre, return_gamma1=True)
    b_cs, n_cs, g1_cs = _pick_nnz(coefs_cs, nnz_cs, g1grid_cs, N_TARGET,
                                  Sig_pre, mu_pre, l_star, Xt_pre, yt_pre)
    cs_bench = Rte @ b_cs
    cells["char_sparse"] = ts_alpha(P_tilde, cs_bench)

    # --- pc-sparse 5-factor: same in PC basis ---
    d_pc, Q = _pc_rotation(Sig_pre)
    Rtr_pc = Rtr @ Q                      # train returns in PC coords
    Rte_pc = Rte @ Q                      # test returns in PC coords
    Sig_pc = np.diag(d_pc)                # cov in PC basis (== Q' Sig_pre Q)
    mu_pc = Q.T @ mu_pre
    Xt_pc, yt_pc, _, _ = dp.whiten(Sig_pc, mu_pc)
    coefs_pc, nnz_pc, g1grid_pc = dp.enet_sdf_path(
        Sig_pc, mu_pc, l_star, n_alphas=200,
        X_tilde=Xt_pc, y_tilde=yt_pc, return_gamma1=True)
    b_pc, n_pc, g1_pc = _pick_nnz(coefs_pc, nnz_pc, g1grid_pc, N_TARGET,
                                  Sig_pc, mu_pc, l_star, Xt_pc, yt_pc)
    pc_bench = Rte_pc @ b_pc              # MVE return in PC space (== Rte @ Q @ b_pc)
    cells["pc_sparse"] = ts_alpha(P_tilde, pc_bench)

    # ---- look-ahead assertions ----
    assert n_te == P.shape[0]
    print(f"[assert] N_oos={n_te} (expect ~3270)")
    print(f"[assert] kappa*={kappa_star:.4f}  l*={l_star:.6e}")
    print(f"[assert] sigma(P_tilde)={np.std(P_tilde,ddof=0):.6e} vs "
          f"OOS mkt sigma={mkt_te_sig:.6e}")
    print(f"[assert] char-sparse nnz={n_cs}  pc-sparse nnz={n_pc}")

    # ---- write JSON ----
    out_cells = []
    for label in ["CAPM", "FF6", "char_sparse", "pc_sparse"]:
        a, se = cells[label]
        out_cells.append({"label": label, "achieved_alpha": a, "achieved_se": se})
    payload = {
        "result_id": "table4_anom50_alphas",
        "kappa_star": kappa_star,
        "N_oos": n_te,
        "oos_market_sigma": mkt_te_sig,
        "char_sparse_nnz": n_cs,
        "pc_sparse_nnz": n_pc,
        "cells": out_cells,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "table4_anom50_alphas.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # ---- write markdown ----
    _write_md(cells, kappa_star, n_te, mkt_te_sig, n_cs, n_pc)

    # ---- console summary ----
    print("\n=== Table 4 (50-anom) achieved vs reported ===")
    print(f"{'bench':<13}{'alpha':>8}{'se':>8}   {'rep_a':>7}{'rep_se':>8}{'t':>7}")
    for label in ["CAPM", "FF6", "char_sparse", "pc_sparse"]:
        a, se = cells[label]
        ra, rse = REPORTED[label]
        t = a / se if se else float("nan")
        print(f"{label:<13}{a:>8.2f}{se:>8.2f}   {ra:>7.2f}{rse:>8.2f}{t:>7.2f}")
    return payload


def _pick_nnz(coefs, nnz, g1grid, target, Sigma, mu, gamma2, Xt, yt):
    """Pick the path point whose #nonzero == target; if none, refine gamma1
    between the brackets and re-fit until exactly `target` nonzeros are hit
    (or fall back to nearest). Returns (b, nnz, gamma1)."""
    hit = np.where(nnz == target)[0]
    if hit.size > 0:
        g = hit[len(hit) // 2]
        return coefs[:, g], int(nnz[g]), float(g1grid[g])

    # bracket: smallest gamma1 with nnz < target, largest with nnz > target
    lo_idx = np.where(nnz > target)[0]
    hi_idx = np.where(nnz < target)[0]
    if lo_idx.size == 0 or hi_idx.size == 0:
        # cannot bracket; return nearest in |nnz-target|
        g = int(np.argmin(np.abs(nnz - target)))
        return coefs[:, g], int(nnz[g]), float(g1grid[g])
    g_lo = g1grid[lo_idx[-1]]   # more nonzeros -> smaller gamma1
    g_hi = g1grid[hi_idx[0]]    # fewer nonzeros -> larger gamma1
    lo, hi = sorted((float(g_lo), float(g_hi)))
    best_b, best_n, best_g = None, None, None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        b = dp.enet_sdf(Sigma, mu, gamma2, mid, X_tilde=Xt, y_tilde=yt)
        n = int(np.sum(np.abs(b) > 1e-10))
        if n == target:
            return b, n, mid
        if best_b is None or abs(n - target) < abs(best_n - target):
            best_b, best_n, best_g = b, n, mid
        if n > target:      # too many nonzeros -> increase gamma1
            lo = mid
        else:               # too few -> decrease gamma1
            hi = mid
    return best_b, best_n, best_g


def _pc_rotation(Sigma):
    """Descending-eigenvalue eigenbasis, sign-pinned [A-C22]."""
    Sigma = 0.5 * (Sigma + Sigma.T)
    w, V = np.linalg.eigh(Sigma)
    order = np.argsort(w)[::-1]
    d = w[order]
    Q = V[:, order]
    for j in range(Q.shape[1]):
        k = np.argmax(np.abs(Q[:, j]))
        if Q[k, j] < 0:
            Q[:, j] *= -1.0
    assert np.allclose(Q @ np.diag(d) @ Q.T, Sigma, atol=1e-8), "PC rotation off"
    return d, Q


def _write_md(cells, kappa_star, n_oos, mkt_sig, n_cs, n_pc):
    lines = [
        "# Table 4 — Out-of-sample MVE alpha (50-anomaly row)",
        "",
        "KNS (2020) Table 4. OOS-MVE built on pre-2005 (beta, sigma, kappa*, "
        "b_hat); test 2005-2017. [A-C11] OLS intercept-only SEs, annualized "
        "x252x100.",
        "",
        f"- kappa* (pre-2005 CV) = {kappa_star:.4f}",
        f"- N_oos = {n_oos} daily obs",
        f"- OOS market sigma (daily) = {mkt_sig:.6f}",
        f"- char-sparse nonzeros = {n_cs}; pc-sparse nonzeros = {n_pc}",
        "",
        "| benchmark | achieved alpha (% ann) | achieved SE | reported alpha | "
        "reported SE | t (ach) |",
        "|---|---|---|---|---|---|",
    ]
    for label in ["CAPM", "FF6", "char_sparse", "pc_sparse"]:
        a, se = cells[label]
        ra, rse = REPORTED[label]
        t = a / se if se else float("nan")
        lines.append(f"| {label} | {a:.2f} | {se:.2f} | {ra:.2f} | {rse:.2f} | "
                     f"{t:.2f} |")
    with open(os.path.join(OUT, "table4.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
