"""
s07 — Figure 4a (50 anomalies): result_id `fig4a_anom50_L2_peak`.

Kozak, Nagel & Santosh (2020), "Shrinking the Cross-Section", JFE 135(2).
Figure 4a: IS / OOS cross-sectional R^2 of the L2 (KNS eta=2) SDF along the
prior-kappa path, plus the Pastor-Stambaugh eta=1 PURE level-shrinkage curve.

The KNS L2 curves (is_csr2, oos_csr2, oos_csr2_se vs kappa_grid) are FROZEN in
foundation.npz. This script:
  - re-loads them,
  - adds the P&S eta=1 estimator b_PS = c * pinv(Sigma_bar) @ mu_bar with the
    level-shrink factor c = 1/(1 + l/freq) for each grid penalty l, runs it
    through the SAME K=3 contiguous CV (with the lCV = l/(1-1/k) rescale), and
    remaps its x-axis to the common kappa axis  [A-C2-PS]:
        x_PS = sqrt(freq * H / (l * T)),  H=50, T=11141,
  - writes fig4a_anom50.png and fig4a_anom50_L2_peak.json.

P&S eta=1 derivation (KNS footnote 4 with eta=1): all PCs shrunk equally,
b = c * Sigma^{-1} mu with c = 1/(1+gamma), gamma = l/freq (code convention
l = freq*gamma). Contrast KNS eta=2 ridge which shrinks low-variance PCs more.
"""
import os
import json
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR",
                      os.path.join(os.path.dirname(__file__), "..", "build", ".mplcache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scs_core as core

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "replication", "build", "output")
FOUND = os.path.join(OUT, "foundation.npz")


def ps_est(X, y, l):
    """Pastor-Stambaugh eta=1 PURE level-shrinkage SDF coefficients.

    KNS (2020) footnote 4 with eta=1:  b = c * pinv(X) @ y,
    c = 1/(1 + l/freq) = 1/(1+gamma), gamma = l/freq.  All PCs shrunk equally.
    Returns (b, se) to match the l2est signature; se is NaN (not used in CV).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    c = 1.0 / (1.0 + l / float(FREQ))
    b = c * (np.linalg.pinv(X) @ y)
    assert np.isfinite(b).all(), "NaN/inf in P&S coefficients"
    return b, np.full_like(b, np.nan)


def ps_cross_validate_csr2(dates, r, l_value, kfold):
    """K=3 contiguous CV of eq.(30) CSR2 for the P&S eta=1 estimator.

    Mirrors core.cross_validate_csr2 exactly but swaps l2est -> ps_est, so the
    P&S curve is evaluated on an identical CV protocol (same folds, same penalty
    rescale lCV = l/(1-1/k), same train/test moments, same eq.30 objective).
    """
    R = np.asarray(r, dtype=float)
    n = R.shape[0]
    assert pd.DatetimeIndex(dates).is_monotonic_increasing, "dates not sorted"
    folds = core.cvpartition_contiguous(n, kfold)
    l_used = l_value / (1.0 - 1.0 / kfold)            # [A-C6] CV penalty rescale

    is_vals = np.full(kfold, np.nan)
    oos_vals = np.full(kfold, np.nan)
    for f, idx_test in enumerate(folds):
        idx_train = np.setdiff1d(np.arange(n), idx_test, assume_unique=False)
        tr = R[idx_train, :]
        te = R[idx_test, :]
        X_tr = core.regcov(tr); y_tr = tr.mean(axis=0)
        X_te = core.regcov(te); y_te = te.mean(axis=0)
        b, _ = ps_est(X_tr, y_tr, l_used)
        yhat_is = X_tr @ b
        yhat_oos = X_te @ b
        is_vals[f] = 1.0 - ((yhat_is - y_tr) @ (yhat_is - y_tr)) / (y_tr @ y_tr)
        oos_vals[f] = 1.0 - ((yhat_oos - y_te) @ (yhat_oos - y_te)) / (y_te @ y_te)
    return float(np.mean(is_vals)), float(np.mean(oos_vals))


def main():
    global FREQ
    f = np.load(FOUND, allow_pickle=True)
    kappa_grid = f["kappa_grid"]
    penalty_list = f["penalty_list"]
    is_csr2 = f["is_csr2"]
    oos_csr2 = f["oos_csr2"]
    oos_se = f["oos_csr2_se"]
    R = f["returns_dm_dv"]
    dates = pd.DatetimeIndex(f["dates"])
    FREQ = int(f["freq"])
    kfold = int(f["kfold"])
    T, H = R.shape
    assert (H, T) == (50, 11141), f"unexpected shape {R.shape}"

    # --- KNS (eta=2) peak (frozen) ---------------------------------------
    i_peak = int(np.argmax(oos_csr2))
    peak_oos = float(oos_csr2[i_peak])
    peak_kappa = float(kappa_grid[i_peak])

    # --- Pastor-Stambaugh eta=1 curve ------------------------------------
    # The KNS penalty_list (ridge scale, l = freq*gamma) maps under the common
    # kappa remap  x_PS = sqrt(freq*H/(l*T))  to x_PS far outside [0.01, 1] (only
    # the smallest-shrinkage point lands in-window), so the P&S curve would never
    # actually shrink. We therefore build a P&S-specific penalty grid by INVERTING
    # the remap [A-C2-PS] over the same x window used for the KNS curve, so the two
    # curves share the kappa axis honestly:
    #     l_PS = freq*H / (x^2 * T),   x = the kappa_grid support [~1 .. 0.01].
    x_ps = np.logspace(np.log10(1.0), np.log10(0.01), len(kappa_grid))
    l_ps = FREQ * H / (x_ps ** 2 * T)
    ps_is = np.full(len(l_ps), np.nan)
    ps_oos = np.full(len(l_ps), np.nan)
    for i, l in enumerate(l_ps):
        ps_is[i], ps_oos[i] = ps_cross_validate_csr2(dates, R, l, kfold)
    ps_peak = float(np.nanmax(ps_oos))

    # --- Plot -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(kappa_grid, is_csr2, "--", color="tab:blue", lw=1.8, label="In-sample")
    ax.plot(kappa_grid, oos_csr2, "-", color="tab:red", lw=1.8, label="OOS (3-fold CV)")
    ax.plot(kappa_grid, oos_csr2 + oos_se, ":", color="tab:red", lw=1.0)
    ax.plot(kappa_grid, oos_csr2 - oos_se, ":", color="tab:red", lw=1.0,
            label="OOS $\\pm$1 s.e.")
    order = np.argsort(x_ps)
    ax.plot(x_ps[order], ps_oos[order], "-.", color="tab:green", lw=1.8,
            label="P&S $\\eta=1$ (OOS)")
    ax.set_xscale("log")
    ax.set_xlim(0.01, 1.0)
    ax.set_ylim(0.0, max(0.5, 1.1 * max(peak_oos, ps_peak)))
    ax.set_xlabel(r"Root Expected SR$^2$ (prior), $\kappa$", fontsize=12, fontweight="bold")
    ax.set_ylabel("IS/OOS Cross-sectional $R^2$", fontsize=12, fontweight="bold")
    ax.set_title("Fig 4a — 50 anomalies, L2 SDF")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    png = os.path.join(OUT, "fig4a_anom50.png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- JSON -------------------------------------------------------------
    payload = {
        "result_id": "fig4a_anom50_L2_peak",
        "cells": [
            {"label": "peak_OOS_CSR2", "achieved": peak_oos},
            {"label": "peak_kappa", "achieved": peak_kappa},
            {"label": "ps_eta1_peak", "achieved": ps_peak},
        ],
    }
    jpath = os.path.join(OUT, "fig4a_anom50_L2_peak.json")
    with open(jpath, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"peak_OOS_CSR2={peak_oos:.4f} @ kappa={peak_kappa:.4f} (idx {i_peak})")
    print(f"ps_eta1_peak={ps_peak:.4f}")
    print(f"wrote {png}")
    print(f"wrote {jpath}")


if __name__ == "__main__":
    main()
