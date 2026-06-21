"""
s06 — Figure 2a (FF25 preliminary): result_id `fig2a_ff25_L2_peak`.

Kozak, Nagel & Santosh (2020), "Shrinking the Cross-Section", JFE 135(2).
Figure 2a: IS / OOS cross-sectional R^2 of the L2 SDF for the 25 Fama-French
size-x-value portfolios, along the prior-kappa path.

Pipeline (identical to the 50-anomaly foundation, with H=25):
  - load FF25 daily VW returns (percent -> /100), subtract daily RF (excess);
  - market reference = Mkt-RF from the FF3 daily file;
  - de-market each excess portfolio on the market (full-sample OLS beta);
  - de-vol to the in-sample market sigma;
  - mu_bar = mean, Sigma_bar = regcov (flat-Wishart shrink, H=25);
  - build kappa grid (gridsize=100); L2 path + K=3 contiguous CV -> IS/OOS CSR2.

Reported: peak OOS-CSR2 ~ 0.44 at kappa ~ 0.23 (strong factor structure).
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

from kns import core

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(ROOT, "Data")
OUT = os.path.join(ROOT, "replication", "build", "output")
FREQ = 252
GRIDSIZE = 100
KFOLD = 3


def load_ff25_excess():
    """Load FF25 daily excess returns and the FF market reference.

    Returns
    -------
    dates : DatetimeIndex (common, inner-joined, sorted, unique)
    re    : (T x 25) DataFrame of excess VW portfolio returns (decimal)
    mkt   : (T,) Series, Mkt-RF (decimal)
    """
    ff25 = pd.read_csv(
        os.path.join(DATA, "25_Portfolios_5x5_Daily_average_value_weighted_returns_daily.csv"))
    ff3 = pd.read_csv(os.path.join(DATA, "F-F_Research_Data_Factors_daily.csv"))
    ff25["Date"] = pd.to_datetime(ff25["Date"], format="%Y/%m/%d")
    ff3["Date"] = pd.to_datetime(ff3["Date"], format="%Y/%m/%d")

    port_cols = [c for c in ff25.columns if c != "Date"]
    assert len(port_cols) == 25, f"expected 25 FF25 cols, got {len(port_cols)}"

    m = pd.merge(ff25, ff3[["Date", "Mkt-RF", "RF"]], on="Date", how="inner")
    m = m.sort_values("Date").reset_index(drop=True)
    dates = pd.DatetimeIndex(m["Date"].values)

    rf = m["RF"].to_numpy() / 100.0
    re = m[port_cols].to_numpy() / 100.0 - rf[:, None]      # excess returns
    mkt = m["Mkt-RF"].to_numpy() / 100.0

    re = pd.DataFrame(re, index=dates, columns=port_cols)
    mkt = pd.Series(mkt, index=dates, name="Mkt-RF")

    assert dates.is_monotonic_increasing and dates.is_unique
    assert np.isfinite(re.to_numpy()).all() and np.isfinite(mkt.to_numpy()).all()
    return dates, re, mkt


def main():
    dates, re, mkt = load_ff25_excess()
    T, H = re.shape
    assert H == 25

    # De-market (full-sample beta) then de-vol to in-sample market sigma.
    r_dm, beta = core.demarket(re, mkt)
    market_sigma = float(mkt.std())
    r_dv = core.devol(r_dm, market_sigma)
    R = r_dv.to_numpy()
    assert np.isfinite(R).all()

    # Moments
    mu_bar = R.mean(axis=0)
    Sigma_bar = core.regcov(R)
    d = np.linalg.eigvalsh(Sigma_bar)
    trX = float(np.trace(Sigma_bar))

    # Kappa grid + penalties (data-driven right edge)
    kappa_grid, penalty_list = core.build_kappa_grid(Sigma_bar, mu_bar, T, FREQ, GRIDSIZE)

    # L2 path: IS/OOS CSR2 via K=3 contiguous CV
    is_csr2 = np.full(GRIDSIZE, np.nan)
    oos_csr2 = np.full(GRIDSIZE, np.nan)
    oos_se = np.full(GRIDSIZE, np.nan)
    for i, l in enumerate(penalty_list):
        res = core.cross_validate_csr2(dates, R, l, KFOLD, FREQ)
        is_csr2[i] = res["is_mean"]
        oos_csr2[i] = res["oos_mean"]
        oos_se[i] = res["oos_se"]

    i_peak = int(np.argmax(oos_csr2))
    peak_oos = float(oos_csr2[i_peak])
    peak_kappa = float(kappa_grid[i_peak])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(kappa_grid, is_csr2, "--", color="tab:blue", lw=1.8, label="In-sample")
    ax.plot(kappa_grid, oos_csr2, "-", color="tab:red", lw=1.8, label="OOS (3-fold CV)")
    ax.plot(kappa_grid, oos_csr2 + oos_se, ":", color="tab:red", lw=1.0)
    ax.plot(kappa_grid, oos_csr2 - oos_se, ":", color="tab:red", lw=1.0,
            label="OOS $\\pm$1 s.e.")
    ax.set_xscale("log")
    ax.set_xlim(max(0.01, kappa_grid.min()), min(2.0, kappa_grid.max()))
    ax.set_ylim(0.0, max(0.6, 1.1 * peak_oos))
    ax.set_xlabel(r"Root Expected SR$^2$ (prior), $\kappa$", fontsize=12, fontweight="bold")
    ax.set_ylabel("IS/OOS Cross-sectional $R^2$", fontsize=12, fontweight="bold")
    ax.set_title("Fig 2a — FF25 (5x5 size/value), L2 SDF")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    png = os.path.join(OUT, "fig2a_ff25.png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "result_id": "fig2a_ff25_L2_peak",
        "cells": [
            {"label": "peak_OOS_CSR2", "achieved": peak_oos},
            {"label": "peak_kappa", "achieved": peak_kappa},
        ],
    }
    jpath = os.path.join(OUT, "fig2a_ff25_L2_peak.json")
    with open(jpath, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"FF25 T={T} H={H} market_sigma={market_sigma:.5f}")
    print(f"peak_OOS_CSR2={peak_oos:.4f} @ kappa={peak_kappa:.4f} (idx {i_peak})")
    print(f"kappa_grid range [{kappa_grid.min():.4f}, {kappa_grid.max():.4f}]")
    print(f"wrote {png}")
    print(f"wrote {jpath}")


if __name__ == "__main__":
    main()
