"""
s10 — Figure 3 contour: result_id `fig3_anom50_contour`.

Kozak, Nagel & Santosh (2020), "Shrinking the Cross-Section", JFE 135(2).
Figure 3: two filled-contour panels of CV out-of-sample cross-sectional R^2
(eq. 30) over the grid

    x = kappa  (prior root-E[SR^2], log scale)
    y = number of NON-ZERO SDF coefficients  (1..H, log scale)

Panel a = RAW 50 anomalies (Sigma_bar, mu_bar from foundation).
Panel b = PC space: rotate into the eigenbasis of Sigma_bar (PCs descending
          eigenvalue, sign-pinned [A-C22]); run the SAME dual-penalty machinery
          on (mu_PC = Q'mu, Sigma_PC = diag(d)).

The dual-penalty SDF (KNS eq. 28) is fitted with dual_penalty.enet_sdf_path:
gamma2 = kappa2pen(kappa, T, trace(Sigma), freq) is fixed by kappa exactly as the
L2 penalty; the gamma1 path sweeps L1 strength -> a set of (#nonzero, b).

OOS-CSR^2 is evaluated with the SAME K=3 contiguous CV used in foundation
(core.cvpartition_contiguous, penalty rescale lCV = l/(1-1/k) [A-C6]) on the
de-marketed/devol'd returns matrix.  Inside each fold we compute TRAIN/TEST
moments, fit the enet on TRAIN at (gamma2_rescaled, gamma1), and score eq.30 on
TEST.  Per (kappa, gamma1) point we average OOS-CSR^2 across folds.

#nonzero for the y-axis is taken from the FULL-SAMPLE fit at that (gamma2,gamma1)
[same gamma1 grid used in-fold].  Multiple gamma1 landing in the same
(kappa,#nonzero) cell -> take the MAX OOS-CSR^2.  R^2 floored at -0.1 for plotting
only (KNS fn 9).

Reported pattern (Fig 3): RAW ridge near-vertical (needs ~all 50; little
sparsity); PC achieves high OOS-R^2 with few PCs.
"""
import os
import json
import time

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR",
                      os.path.join(os.path.dirname(__file__), "..", "build", ".mplcache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scs_core as core
import dual_penalty as dp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "replication", "build", "output")
FOUND = os.path.join(OUT, "foundation.npz")

# Grid controls.
N_KAPPA = 30          # ~25-40 kappa values spanning the plot window
KAPPA_LO, KAPPA_HI = 0.05, 3.0
N_GAMMA1 = 80         # gamma1 path density per kappa


# ---------------------------------------------------------------------------
# PC rotation  [A-C22]
# ---------------------------------------------------------------------------
def pc_rotation(Sigma):
    """Eigenbasis of Sigma_bar, descending eigenvalue, sign-pinned [A-C22].

    Q columns = eigenvectors ordered by DESCENDING eigenvalue.  Each column's
    sign is pinned so its largest-|loading| entry is positive (deterministic).
    Returns (d_desc, Q) with Sigma = Q diag(d_desc) Q' (up to sign of columns).
    """
    Sigma = 0.5 * (Sigma + Sigma.T)
    w, V = np.linalg.eigh(Sigma)              # ascending
    order = np.argsort(w)[::-1]               # descending
    d = w[order]
    Q = V[:, order]
    for j in range(Q.shape[1]):
        k = np.argmax(np.abs(Q[:, j]))
        if Q[k, j] < 0:
            Q[:, j] *= -1.0
    assert np.allclose(Q @ np.diag(d) @ Q.T, Sigma, atol=1e-10), "PC rotation off"
    return d, Q


# ---------------------------------------------------------------------------
# CV OOS-CSR^2 for a whole gamma1 path at one kappa (in a given basis)
# ---------------------------------------------------------------------------
def cv_oos_csr2_path(dates, R_basis, gamma2, gamma1_grid, kfold):
    """K=3 contiguous CV OOS-CSR^2 (eq.30) for a fixed gamma1 grid at one gamma2.

    R_basis : (T,H) returns already in the working basis (raw anomalies, or PCs).
              In the PC basis R_basis = R_raw @ Q so that the per-fold sample
              moments equal Q' Sigma_fold Q / mu rotated -> identical SDF problem
              in PC coordinates (no look-ahead introduced; Q is full-sample but
              is just a fixed orthonormal rotation, not a fitted moment).
    gamma2  : L2 penalty at this kappa (full-sample scale); rescaled per fold.

    Returns
    -------
    oos_mean : (G,) mean OOS-CSR^2 across folds per gamma1 point
    oos_se   : (G,) s.e. across folds (std / sqrt(k))
    """
    R = np.asarray(R_basis, dtype=float)
    n, H = R.shape
    folds = core.cvpartition_contiguous(n, kfold)
    g2_used = gamma2 / (1.0 - 1.0 / kfold)            # [A-C6] CV penalty rescale
    G = len(gamma1_grid)
    fold_vals = np.full((kfold, G), np.nan)

    for fi, idx_test in enumerate(folds):
        idx_train = np.setdiff1d(np.arange(n), idx_test, assume_unique=False)
        tr = R[idx_train, :]
        te = R[idx_test, :]
        X_tr = core.regcov(tr); y_tr = tr.mean(axis=0)
        X_te = core.regcov(te); y_te = te.mean(axis=0)
        Xt, yt, _, _ = dp.whiten(X_tr, y_tr)
        denom = float(y_te @ y_te)
        for g, g1 in enumerate(gamma1_grid):
            b = dp.enet_sdf(X_tr, y_tr, g2_used, g1, X_tilde=Xt, y_tilde=yt)
            resid = X_te @ b - y_te
            fold_vals[fi, g] = 1.0 - (resid @ resid) / denom
    oos_mean = np.nanmean(fold_vals, axis=0)
    oos_se = np.nanstd(fold_vals, axis=0) / np.sqrt(kfold)
    return oos_mean, oos_se


# ---------------------------------------------------------------------------
# Build the (kappa, #nonzero) -> OOS-CSR^2 grid for one basis
# ---------------------------------------------------------------------------
def build_panel(dates, R_basis, Sigma_full, mu_full, kappa_grid, kfold, freq):
    """Compute the contour grid (#nonzero x kappa) of CV OOS-CSR^2 for one basis.

    For each kappa:
      - gamma2 = kappa2pen(kappa, T, trace(Sigma_full), freq);
      - FULL-SAMPLE gamma1 path -> (#nonzero per gamma1 point);
      - CV OOS-CSR^2 for that SAME gamma1 grid;
      - bin into (kappa, #nonzero) cells, MAX over gamma1 hitting a cell.
    Also tracks per-cell OOS s.e. (the s.e. of the gamma1 point that won the cell).

    Returns
    -------
    grid      : (H, n_kappa) OOS-CSR^2; rows = #nonzero 1..H, NaN where empty
    grid_se   : (H, n_kappa) matching s.e.
    """
    R = np.asarray(R_basis, dtype=float)
    T, H = R.shape
    trX = float(np.trace(Sigma_full))
    Xt_full, yt_full, _, _ = dp.whiten(Sigma_full, mu_full)

    grid = np.full((H, len(kappa_grid)), np.nan)
    grid_se = np.full((H, len(kappa_grid)), np.nan)

    for ki, kappa in enumerate(kappa_grid):
        gamma2 = core.kappa2pen(kappa, T, trX, freq)
        # full-sample gamma1 path -> #nonzero per point + the gamma1 grid
        coefs, nnz, g1_grid = dp.enet_sdf_path(
            Sigma_full, mu_full, gamma2, n_alphas=N_GAMMA1,
            X_tilde=Xt_full, y_tilde=yt_full, return_gamma1=True)
        # CV OOS for the same grid
        oos_mean, oos_se = cv_oos_csr2_path(dates, R, gamma2, g1_grid, kfold)
        # bin into (#nonzero) cells, MAX over gamma1
        for g in range(len(g1_grid)):
            k = nnz[g]
            if k < 1 or k > H:
                continue
            row = k - 1
            if np.isnan(grid[row, ki]) or oos_mean[g] > grid[row, ki]:
                grid[row, ki] = oos_mean[g]
                grid_se[row, ki] = oos_se[g]
    return grid, grid_se


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------
def _fill_contour(ax, kappa_grid, grid, title, floor=-0.1):
    H, nk = grid.shape
    yvals = np.arange(1, H + 1)
    # forward-fill empty cells along #nonzero within each kappa column for a
    # smooth contour (plotting only; JSON uses the raw binned grid).
    Z = grid.copy()
    for c in range(nk):
        col = Z[:, c]
        # interpolate NaNs over the y axis
        good = ~np.isnan(col)
        if good.sum() >= 2:
            Z[:, c] = np.interp(np.arange(H), np.nonzero(good)[0], col[good])
    Zf = np.clip(Z, floor, None)
    Xg, Yg = np.meshgrid(kappa_grid, yvals)
    levels = np.linspace(floor, max(np.nanmax(Zf), floor + 1e-3), 25)
    cs = ax.contourf(Xg, Yg, Zf, levels=levels, cmap="YlOrRd", extend="both")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Root Expected SR$^2$ (prior), $\kappa$", fontweight="bold")
    ax.set_ylabel("# nonzero SDF coefficients", fontweight="bold")
    ax.set_title(title)
    return cs


def main():
    t0 = time.time()
    f = np.load(FOUND, allow_pickle=True)
    Sigma = f["Sigma_bar"]; mu = f["mu_bar"]
    R = np.asarray(f["returns_dm_dv"], dtype=float)
    dates = pd.DatetimeIndex(f["dates"])
    freq = int(f["freq"]); kfold = int(f["kfold"])
    T, H = R.shape

    kappa_grid = np.logspace(np.log10(KAPPA_LO), np.log10(KAPPA_HI), N_KAPPA)

    # --- Panel a: RAW 50 anomalies --------------------------------------
    grid_raw, se_raw = build_panel(dates, R, Sigma, mu, kappa_grid, kfold, freq)

    # --- Panel b: PC space ----------------------------------------------
    d_pc, Q = pc_rotation(Sigma)
    R_pc = R @ Q                      # returns in PC coordinates
    mu_pc = Q.T @ mu
    Sigma_pc = np.diag(d_pc)
    grid_pc, se_pc = build_panel(dates, R_pc, Sigma_pc, mu_pc, kappa_grid, kfold, freq)

    # --- Plot -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    cs0 = _fill_contour(axes[0], kappa_grid, grid_raw, "(a) 50 characteristics (raw)")
    cs1 = _fill_contour(axes[1], kappa_grid, grid_pc, "(b) PCs of 50 characteristics")
    fig.colorbar(cs1, ax=axes, shrink=0.85, label="OOS cross-sectional $R^2$")
    fig.suptitle("Fig 3 — CV OOS $R^2$ over ($\\kappa$, #nonzero): raw vs PC", y=1.02)
    png = os.path.join(OUT, "fig3_anom50_contour.png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Qualitative checks (JSON) --------------------------------------
    def best_at_nnz(grid, nnz):
        row = grid[nnz - 1, :]
        if np.all(np.isnan(row)):
            return None
        return float(np.nanmax(row))

    def best_overall(grid):
        if np.all(np.isnan(grid)):
            return None, None
        idx = np.nanargmax(grid)
        r, c = np.unravel_index(idx, grid.shape)
        return float(grid[r, c]), int(r + 1)

    raw_best, raw_best_nnz = best_overall(grid_raw)
    pc_best, pc_best_nnz = best_overall(grid_pc)

    raw_at5 = best_at_nnz(grid_raw, 5)
    raw_at45 = best_at_nnz(grid_raw, 45)
    pc_at4 = best_at_nnz(grid_pc, 4)
    pc_at5 = best_at_nnz(grid_pc, 5)
    pc_at45 = best_at_nnz(grid_pc, 45)

    # vertical-ridge evidence for RAW: low #nonzero much worse than high #nonzero
    raw_vertical_ridge = (raw_at5 is not None and raw_at45 is not None
                          and raw_at5 < raw_at45 - 0.02)
    # PC achieves high OOS-R^2 at low #nonzero (near its own best)
    pc_sparse_good = (pc_at4 is not None and pc_best is not None
                      and pc_at4 >= 0.6 * pc_best)
    # PC at low nnz beats RAW at low nnz
    pc_beats_raw_lownnz = (pc_at4 is not None and raw_at5 is not None
                           and pc_at4 > raw_at5 + 0.05)

    payload = {
        "result_id": "fig3_anom50_contour",
        "grid": {"n_kappa": N_KAPPA, "kappa_lo": KAPPA_LO, "kappa_hi": KAPPA_HI,
                 "n_gamma1": N_GAMMA1, "kfold": kfold, "H": H},
        "raw_panel": {
            "best_oos_csr2": raw_best, "best_at_nnz": raw_best_nnz,
            "oos_at_nnz5": raw_at5, "oos_at_nnz45": raw_at45,
            "vertical_ridge_need_many_nonzero": bool(raw_vertical_ridge),
        },
        "pc_panel": {
            "best_oos_csr2": pc_best, "best_at_nnz": pc_best_nnz,
            "oos_at_nnz4": pc_at4, "oos_at_nnz5": pc_at5, "oos_at_nnz45": pc_at45,
            "high_oos_at_low_nonzero": bool(pc_sparse_good),
        },
        "pc_beats_raw_at_low_nonzero": bool(pc_beats_raw_lownnz),
        "runtime_sec": round(time.time() - t0, 1),
    }
    jpath = os.path.join(OUT, "fig3_anom50_contour.json")
    with open(jpath, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"RAW  best OOS-R2={raw_best:.4f} @ nnz={raw_best_nnz}; "
          f"nnz5={raw_at5}, nnz45={raw_at45}, vertical_ridge={raw_vertical_ridge}")
    print(f"PC   best OOS-R2={pc_best:.4f} @ nnz={pc_best_nnz}; "
          f"nnz4={pc_at4}, nnz45={pc_at45}, sparse_good={pc_sparse_good}")
    print(f"PC beats RAW at low nnz: {pc_beats_raw_lownnz}")
    print(f"wrote {png}\nwrote {jpath}\nruntime {payload['runtime_sec']}s")

    # stash grids for s11 reuse
    np.savez(os.path.join(OUT, "fig3_grids.npz"),
             kappa_grid=kappa_grid, grid_raw=grid_raw, se_raw=se_raw,
             grid_pc=grid_pc, se_pc=se_pc)


if __name__ == "__main__":
    main()
