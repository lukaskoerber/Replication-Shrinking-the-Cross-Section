"""
foundation_check.py — Foundation checkpoint for the replication of

    Kozak, Nagel & Santosh (2020), "Shrinking the Cross-Section", JFE.

Runs the full in-sample 50-anomaly pipeline using scs_core and freezes the
outputs every downstream section reuses. Previews the KNS L2 OOS curve (Fig 4a):

    load -> demarket(full-sample beta) -> devol(full-sample market sigma)
         -> mu_bar = daily mean, Sigma_bar = regcov
         -> annualized in-sample max SR^2 = freq * mu' pinv(Sigma) mu
         -> data-driven kappa grid -> L2 path + K=3 contiguous CV of CSR2 (eq.30)
         -> report peak OOS-CSR^2 and the kappa where it occurs.

Frozen outputs -> replication/build/output/foundation.npz.
A compact JSON summary is printed to stdout.
"""

import json
import os

import numpy as np

from kns.core import (
    load_anom50_daily,
    demarket,
    devol,
    regcov,
    build_kappa_grid,
    cross_validate_csr2,
    effective_dof,
)

# Repo root = three levels up from src/kns/reproduce/ (matches the other drivers).
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
DATA_PATH = os.path.join(
    PROJECT_ROOT, "Data", "Instruments", "managed_portfolios_anom_d_50.csv"
)
OUT_NPZ = os.path.join(PROJECT_ROOT, "replication", "build", "output", "foundation.npz")

FREQ = 252       # daily annualization
KFOLD = 3        # K=3 contiguous CV (matches reference parameters['kfold']=3)
GRIDSIZE = 100


def main():
    # ---- load --------------------------------------------------------------
    dates, re, mkt, T = load_anom50_daily(DATA_PATH)
    H = re.shape[1]

    # ---- de-market (full-sample betas) ------------------------------------
    r_dm, beta = demarket(re, mkt)                 # [A-C1]

    # ---- de-vol (full-sample market sigma) --------------------------------
    # In-sample analyses use full-sample sigma of the de-marketed returns and
    # full-sample mkt.std(). [A-C3]
    market_sigma = float(mkt.std(ddof=0))
    r_dv = devol(r_dm, market_sigma, asset_sigma=None, ddof=0)  # [A-C3]
    R = np.asarray(r_dv, dtype=float)
    assert np.isfinite(R).all()

    # ---- first / second moments -------------------------------------------
    mu_bar = R.mean(axis=0)                         # daily mean (H,)
    Sigma_bar = regcov(R)                           # [A-C4][A-C23]
    trX = float(np.trace(Sigma_bar))

    # ---- annualized in-sample max SR^2 ------------------------------------
    maxSR2 = FREQ * float(mu_bar @ np.linalg.pinv(Sigma_bar) @ mu_bar)
    assert np.isfinite(maxSR2) and maxSR2 > 0, "max SR^2 not finite/positive"

    # eigenvalues of Sigma_bar (for effective dof / downstream)
    d = np.linalg.eigvalsh(Sigma_bar)[::-1]         # descending
    d = np.clip(d, 0.0, None)

    # ---- kappa grid + L2 path CV ------------------------------------------
    x, l = build_kappa_grid(Sigma_bar, mu_bar, T, FREQ, gridsize=GRIDSIZE)  # [A-C10]

    is_curve = np.full(len(l), np.nan)
    oos_curve = np.full(len(l), np.nan)
    oos_se = np.full(len(l), np.nan)
    for i, l_val in enumerate(l):
        res = cross_validate_csr2(dates, R, l_val, KFOLD, FREQ)  # eq.(30)
        is_curve[i] = res["is_mean"]
        oos_curve[i] = res["oos_mean"]
        oos_se[i] = res["oos_se"]

    df_eff = effective_dof(d, l)                    # fix C18

    # ---- peak OOS ----------------------------------------------------------
    i_peak = int(np.nanargmax(oos_curve))
    peak_oos = float(oos_curve[i_peak])
    peak_kappa = float(x[i_peak])
    is_at_peak = float(is_curve[i_peak])

    # ---- freeze outputs ----------------------------------------------------
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        mu_bar=mu_bar,
        Sigma_bar=Sigma_bar,
        d=d,
        kappa_grid=x,
        penalty_list=np.asarray(l),
        oos_csr2=oos_curve,
        oos_csr2_se=oos_se,
        is_csr2=is_curve,
        eff_dof=df_eff,
        returns_dm_dv=R,
        dates=dates.values.astype("datetime64[ns]"),
        beta=beta,
        market_sigma=market_sigma,
        freq=FREQ,
        kfold=KFOLD,
    )

    summary = {
        "start_date": str(dates[0].date()),
        "end_date": str(dates[-1].date()),
        "T": int(T),
        "H": int(H),
        "market_sigma_daily": market_sigma,
        "trace_Sigma_bar": trX,
        "annualized_in_sample_maxSR2": maxSR2,
        "kappa_grid_right_edge": float(x[0]),
        "kappa_grid_left_edge": float(x[-1]),
        "peak_oos_csr2": peak_oos,
        "peak_kappa": peak_kappa,
        "is_csr2_at_peak": is_at_peak,
        "oos_se_at_peak": float(oos_se[i_peak]),
        "eff_dof_at_peak": float(df_eff[i_peak]),
        "npz_path": OUT_NPZ,
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
