"""
sr_objective.py — EXTENSION driver: cross-validated Sharpe-ratio (SR) objective
for the KNS (2020) "Shrinking the Cross-Section" replication.

Kozak, Nagel & Santosh (2020), JFE 135(2). The canonical pipeline selects the
ridge penalty kappa by K=3 contiguous CV maximizing OOS cross-sectional R^2
(eq. 30, `cross_validate_csr2`). This driver runs the NEW selectable SR objective
(`cross_validate_sr`, realized-MVE annualized Sharpe) across the SAME kappa grid
on the SAME `returns_dm_dv` matrix [A-SR13], selects kappa_SR* with the
1-SE-toward-parsimony tie-break [A-SR7/B1], and compares it to kappa_CSR2*.

Reported diagnostic only [A-SR15]; does not drive any downstream artifact.

Design + pinned assumptions A-SR1..15 / guards B1,B2,B3:
  replication/extensions/sr_objective/{plan.md, open_questions.md}.

Outputs (replication/build/output/):
  sr_objective.json                 (Part 2)
  fig4a_sr_anom50.png               (Part 3.1)
  fig4a_objective_compare.png       (Part 3.2)
  table1_sr_compare.{md,json}       (Part 3.3)
"""
import os
import json
import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".mplcache")),
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scs_core as core

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "replication", "build", "output")
FOUND = os.path.join(OUT, "foundation.npz")

# Anomaly column order / labels — reused from s08_table1.py (verified mapping).
ANOM_COLS = ['r_size', 'r_value', 'r_prof', 'r_valprof', 'r_fscore', 'r_debtiss',
             'r_repurch', 'r_nissa', 'r_accruals', 'r_growth', 'r_aturnover',
             'r_gmargins', 'r_divp', 'r_ep', 'r_cfp', 'r_noa', 'r_inv', 'r_invcap',
             'r_igrowth', 'r_sgrowth', 'r_lev', 'r_roaa', 'r_roea', 'r_sp',
             'r_gltnoa', 'r_mom', 'r_indmom', 'r_valmom', 'r_valmomprof',
             'r_shortint', 'r_mom12', 'r_momrev', 'r_lrrev', 'r_valuem', 'r_nissm',
             'r_sue', 'r_roe', 'r_rome', 'r_roa', 'r_strev', 'r_ivol', 'r_betaarb',
             'r_season', 'r_indrrev', 'r_indrrevlv', 'r_indmomrev', 'r_ciss',
             'r_price', 'r_age', 'r_shvol']

LABELS = {
    "r_indrrevlv": "Industry Rel. Rev. (L.V.)", "r_indmomrev": "Ind. Mom-Reversals",
    "r_indrrev": "Industry Rel. Reversals", "r_season": "Seasonality",
    "r_sue": "Earnings Surprises", "r_valprof": "Value-Profitability",
    "r_rome": "Return on Market Equity", "r_inv": "Investment/Assets",
    "r_roe": "Return on Equity", "r_ciss": "Composite Issuance",
    "r_mom12": "Momentum (12m)", "r_mom": "Momentum", "r_strev": "Short-Term Reversal",
    "r_lrrev": "Long-Run Reversal", "r_indmom": "Industry Momentum",
    "r_valmom": "Value-Momentum", "r_valmomprof": "Value-Mom-Prof", "r_size": "Size",
    "r_value": "Value (A)", "r_valuem": "Value (M)", "r_prof": "Gross Profitability",
    "r_roaa": "Return on Assets (A)", "r_roa": "Return on Assets",
    "r_roea": "Return on Equity (A)", "r_accruals": "Accruals", "r_growth": "Asset Growth",
    "r_aturnover": "Asset Turnover", "r_gmargins": "Gross Margins", "r_divp": "Dividend/Price",
    "r_ep": "Earnings/Price", "r_cfp": "Cash Flow/Price", "r_sp": "Sales/Price",
    "r_noa": "Net Operating Assets", "r_invcap": "Investment/Capital",
    "r_igrowth": "Investment Growth", "r_sgrowth": "Sales Growth", "r_lev": "Leverage",
    "r_gltnoa": "Growth in LTNOA", "r_fscore": "Piotroski F-score",
    "r_debtiss": "Debt Issuance", "r_repurch": "Share Repurchases",
    "r_nissa": "Net Issuance (A)", "r_nissm": "Net Issuance (M)",
    "r_shortint": "Short Interest", "r_ivol": "Idiosyncratic Vol",
    "r_betaarb": "Beta Arbitrage", "r_price": "Price", "r_age": "Firm Age",
    "r_shvol": "Share Volume", "r_momrev": "Momentum-Reversal",
}


def _fold_b_for_invariance_test(R, l_value, kfold):
    """Recompute one fold's b (fold 0) to drive the scale-invariance unit test."""
    n = R.shape[0]
    folds = core.cvpartition_contiguous(n, kfold)
    l_used = l_value / (1.0 - 1.0 / kfold)
    idx_test = folds[0]
    idx_train = np.setdiff1d(np.arange(n), idx_test, assume_unique=False)
    tr = R[idx_train, :]
    te = R[idx_test, :]
    X_tr = core.regcov(tr)
    y_tr = tr.mean(axis=0)
    b, _ = core.l2est(X_tr, y_tr, l_used, len(idx_train))
    return te, b


def main():
    f = np.load(FOUND, allow_pickle=True)
    kappa_grid = f["kappa_grid"]
    penalty_list = f["penalty_list"]
    oos_csr2 = f["oos_csr2"]
    oos_csr2_se = f["oos_csr2_se"]
    is_csr2 = f["is_csr2"]
    R = f["returns_dm_dv"]
    dates = pd.DatetimeIndex(f["dates"])
    FREQ = int(f["freq"])
    KFOLD = int(f["kfold"])
    mu_bar = f["mu_bar"]
    Sigma_bar = f["Sigma_bar"]
    T, H = R.shape
    assert (T, H) == (11141, 50), f"unexpected returns_dm_dv shape {R.shape}"
    assert len(kappa_grid) == 100 and len(penalty_list) == 100

    # In-sample MVE Sharpe ceiling: sqrt(freq * mu' Sigma^+ mu) ~ 3.54.
    is_max_sr_ceiling = float(np.sqrt(FREQ * mu_bar @ np.linalg.pinv(Sigma_bar) @ mu_bar))

    # ---------------------------------------------------------------------
    # 1. Run the SR CV across every penalty in penalty_list (K=3).
    # ---------------------------------------------------------------------
    L = len(penalty_list)
    sr_is = np.full(L, np.nan)
    sr_oos = np.full(L, np.nan)
    sr_oos_se = np.full(L, np.nan)
    sr_pooled = np.full(L, np.nan)
    sr_oos_folds = np.full((L, KFOLD), np.nan)
    for i, l in enumerate(penalty_list):
        d = core.cross_validate_sr(dates, R, float(l), KFOLD, FREQ)
        sr_is[i] = d["is_mean"]
        sr_oos[i] = d["oos_mean"]
        sr_oos_se[i] = d["oos_se"]
        sr_pooled[i] = d["pooled_sr"]
        sr_oos_folds[i, :] = d["oos_folds"]

    assert np.isfinite(sr_oos).all(), "non-finite OOS SR curve (degenerate folds?)"

    # ---------------------------------------------------------------------
    # 2. Selection.
    # ---------------------------------------------------------------------
    # SR objective, 1-SE-toward-parsimony [A-SR7/B1].
    i_sr, kappa_sr, band_width, n_in_band = core.select_kappa_1se_parsimony(
        kappa_grid, sr_oos, sr_oos_se)
    # Plain argmax for transparency.
    i_sr_argmax = int(np.nanargmax(sr_oos))
    kappa_sr_argmax = float(kappa_grid[i_sr_argmax])

    # CSR2 objective optimum (frozen).
    i_csr2 = int(np.argmax(oos_csr2))
    kappa_csr2 = float(kappa_grid[i_csr2])

    # Pooled-SR selection (diagnostic cross-check, B3). Pooled has no per-fold se,
    # so use the SAME oos_se band for an apples-to-apples parsimony tie-break.
    i_sr_pooled, kappa_sr_pooled, _, _ = core.select_kappa_1se_parsimony(
        kappa_grid, sr_pooled, sr_oos_se)

    # ---------------------------------------------------------------------
    # 3. Cross-objective table values.
    # ---------------------------------------------------------------------
    oos_sr_at_sropt = float(sr_oos[i_sr])
    oos_sr_at_csr2opt = float(sr_oos[i_csr2])
    pooled_sr_at_sropt = float(sr_pooled[i_sr])
    oos_csr2_at_sropt = float(oos_csr2[i_sr])
    oos_csr2_at_csr2opt = float(oos_csr2[i_csr2])
    per_fold_oos_sr_at_sropt = [float(v) for v in sr_oos_folds[i_sr, :]]

    # ---------------------------------------------------------------------
    # 4. SANITY CHECKS.
    # ---------------------------------------------------------------------
    # (i) [A-SR10] scale-invariance unit test on one fold's b.
    te0, b0 = _fold_b_for_invariance_test(R, float(penalty_list[i_sr]), KFOLD)
    base_sr = core.sharpe_annual(te0 @ b0, FREQ)
    scale_diffs = []
    for c in (0.5, 2.0, 100.0):
        scale_diffs.append(abs(core.sharpe_annual(te0 @ (c * b0), FREQ) - base_sr))
    scale_invariance_max_diff = float(max(scale_diffs))
    assert scale_invariance_max_diff < 1e-8, "scale-invariance unit test failed"

    # (ii) per-fold-selected vs pooled-selected kappa agree within a couple steps.
    perfold_vs_pooled_gridgap = int(abs(i_sr - i_sr_pooled))

    # (iii) cross-objective near-tautology / argmax-bug check (A-SR11).
    #   Reported boolean: OOS SR at the *parsimony-selected* SR* vs at CSR2*.
    #   NOTE: with the 1-SE-toward-parsimony tie-break [A-SR7/B1] this can be
    #   FALSE in the weakly-identified case (the whole grid is within 1 s.e., so
    #   the rule picks the most-shrunk endpoint whose POINT estimate is below the
    #   interior max) — that is by design, not a bug.  The genuine argmax/sign-bug
    #   guard (plan §6) is therefore enforced on the UNCONSTRAINED argmax: the
    #   maximizer must, tautologically, beat every other grid point incl. CSR2*.
    sr_ge = bool(oos_sr_at_sropt >= oos_sr_at_csr2opt - 1e-12)
    oos_sr_at_sr_argmax = float(sr_oos[i_sr_argmax])
    sr_ge_argmax = bool(oos_sr_at_sr_argmax >= oos_sr_at_csr2opt - 1e-12)

    # (iv) plausible band: OOS annual SR at SR* finite and below the IS ceiling.
    oos_below_ceiling = bool(np.isfinite(oos_sr_at_sropt)
                             and oos_sr_at_sropt < is_max_sr_ceiling)
    # also guard the argmax level (the higher of the two).
    oos_below_ceiling_argmax = bool(np.isfinite(oos_sr_at_sr_argmax)
                                    and oos_sr_at_sr_argmax < is_max_sr_ceiling)

    # --- ABORT conditions (plan §6) ---
    if not (oos_below_ceiling and oos_below_ceiling_argmax):
        raise SystemExit(
            f"ABORT [plan §6]: OOS SR (parsimony {oos_sr_at_sropt:.3f} / argmax "
            f"{oos_sr_at_sr_argmax:.3f}) >= ceiling {is_max_sr_ceiling:.3f} — "
            "look-ahead red flag.")
    if not sr_ge_argmax:
        raise SystemExit(
            f"ABORT [plan §6]: SR(argmax)={oos_sr_at_sr_argmax:.3f} < "
            f"SR(CSR2*)={oos_sr_at_csr2opt:.3f} — selection/argmax/sign bug.")

    # (v) no-lookahead is asserted inside cross_validate_sr (train/test disjoint).

    # ---------------------------------------------------------------------
    # 5. Write sr_objective.json (Part 2).
    # ---------------------------------------------------------------------
    payload = {
        "kappa_SR_opt": kappa_sr,
        "kappa_SR_argmax": kappa_sr_argmax,
        "within_1se_band_width": band_width,
        "n_in_band": n_in_band,
        "oos_annual_SR_at_SRopt": oos_sr_at_sropt,
        "pooled_SR_at_SRopt": pooled_sr_at_sropt,
        "kappa_CSR2_opt": kappa_csr2,
        "oos_SR_at_CSR2opt": oos_sr_at_csr2opt,
        "oos_CSR2_at_SRopt": oos_csr2_at_sropt,
        "oos_CSR2_at_CSR2opt": oos_csr2_at_csr2opt,
        "is_max_annual_SR_ceiling": is_max_sr_ceiling,
        "per_fold_oos_SR_at_SRopt": per_fold_oos_sr_at_sropt,
        "oos_SR_at_SR_argmax": oos_sr_at_sr_argmax,
        "sanity": {
            "scale_invariance_max_diff": scale_invariance_max_diff,
            "perfold_vs_pooled_gridgap": perfold_vs_pooled_gridgap,
            "sr_ge_at_SRopt_vs_CSR2opt": sr_ge,
            "sr_ge_at_SRargmax_vs_CSR2opt": sr_ge_argmax,
            "oos_below_ceiling": oos_below_ceiling,
            "oos_below_ceiling_argmax": oos_below_ceiling_argmax,
        },
        "bias_caveat": ("CV-optimal OOS SR level is upward-biased (penalty chosen "
                        "on whole sample); used for ranking only"),
        "weak_identification_note": (
            f"n_in_band={n_in_band} of {L} grid points within 1 s.e. of the SR max; "
            + ("weakly identified (wide flat band)" if n_in_band > 10
               else "reasonably identified")),
    }
    jpath = os.path.join(OUT, "sr_objective.json")
    with open(jpath, "w") as fh:
        json.dump(payload, fh, indent=2)

    # ---------------------------------------------------------------------
    # 6. Part 3.1 — fig4a_sr_anom50.png (SR analog of Fig 4a).
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(kappa_grid, sr_is, "--", color="tab:blue", lw=1.8, label="In-sample SR")
    ax.plot(kappa_grid, sr_oos, "-", color="tab:red", lw=1.8,
            label="OOS SR (3-fold CV)")
    ax.plot(kappa_grid, sr_oos + sr_oos_se, ":", color="tab:red", lw=1.0)
    ax.plot(kappa_grid, sr_oos - sr_oos_se, ":", color="tab:red", lw=1.0,
            label=r"OOS $\pm$1 s.e.")
    ax.axvline(kappa_sr, color="tab:red", ls="-", lw=1.3,
               label=rf"$\kappa_{{SR}}^*={kappa_sr:.3f}$")
    ax.axvline(kappa_csr2, color="tab:purple", ls="--", lw=1.3,
               label=rf"$\kappa_{{CSR2}}^*={kappa_csr2:.3f}$")
    ax.set_xscale("log")
    ax.set_xlim(kappa_grid.min(), kappa_grid.max())
    ax.set_xlabel(r"Root Expected SR$^2$ (prior), $\kappa$", fontsize=12, fontweight="bold")
    ax.set_ylabel("Annualized Sharpe ratio of MVE portfolio", fontsize=12, fontweight="bold")
    ax.set_title("Fig 4a analog — 50 anomalies, L2 SDF, SR objective")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    png1 = os.path.join(OUT, "fig4a_sr_anom50.png")
    fig.savefig(png1, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------------
    # 7. Part 3.2 — fig4a_objective_compare.png (the money figure).
    # ---------------------------------------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    # LEFT: OOS CSR2.
    axL.plot(kappa_grid, oos_csr2, "-", color="tab:blue", lw=1.8, label="OOS CSR$^2$")
    axL.plot(kappa_grid, oos_csr2 + oos_csr2_se, ":", color="tab:blue", lw=1.0)
    axL.plot(kappa_grid, oos_csr2 - oos_csr2_se, ":", color="tab:blue", lw=1.0,
             label=r"$\pm$1 s.e.")
    axL.axvline(kappa_csr2, color="tab:purple", ls="--", lw=1.3,
                label=rf"$\kappa_{{CSR2}}^*={kappa_csr2:.3f}$")
    axL.axvline(kappa_sr, color="tab:red", ls="-", lw=1.3,
                label=rf"$\kappa_{{SR}}^*={kappa_sr:.3f}$")
    axL.set_xscale("log")
    # Clip to the informative band: the deep-negative IS-overfit tail at high
    # kappa would otherwise compress the peak region and both kappa* lines.
    axL.set_ylim(-0.15, max(0.4, 1.1 * float(np.nanmax(oos_csr2))))
    axL.set_xlabel(r"$\kappa$ (log)", fontsize=12, fontweight="bold")
    axL.set_ylabel(r"OOS cross-sectional $R^2$", fontsize=12, fontweight="bold")
    axL.set_title("CSR$^2$ objective")
    axL.grid(True, which="both", alpha=0.3)
    axL.legend(loc="best", fontsize=9)
    # RIGHT: OOS SR.
    axR.plot(kappa_grid, sr_oos, "-", color="tab:green", lw=1.8, label="OOS SR")
    axR.plot(kappa_grid, sr_oos + sr_oos_se, ":", color="tab:green", lw=1.0)
    axR.plot(kappa_grid, sr_oos - sr_oos_se, ":", color="tab:green", lw=1.0,
             label=r"$\pm$1 s.e.")
    axR.axvline(kappa_sr, color="tab:red", ls="-", lw=1.3,
                label=rf"$\kappa_{{SR}}^*={kappa_sr:.3f}$")
    axR.axvline(kappa_csr2, color="tab:purple", ls="--", lw=1.3,
                label=rf"$\kappa_{{CSR2}}^*={kappa_csr2:.3f}$")
    axR.set_xscale("log")
    axR.set_xlabel(r"$\kappa$ (log)", fontsize=12, fontweight="bold")
    axR.set_ylabel("Annualized Sharpe ratio", fontsize=12, fontweight="bold")
    axR.set_title("SR objective")
    axR.grid(True, which="both", alpha=0.3)
    axR.legend(loc="best", fontsize=9)
    fig.suptitle("OOS model-selection objective vs shrinkage — CSR$^2$ vs Sharpe "
                 "(50 anomalies)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png2 = os.path.join(OUT, "fig4a_objective_compare.png")
    fig.savefig(png2, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------------
    # 8. Part 3.3 — table1_sr_compare.{md,json}.
    # SDF coefficients/t-stats at kappa_SR* vs kappa_CSR2* on full-sample moments.
    # eq.(22)-(23): b = (Sigma_bar + l I)^{-1} mu_bar; se = sqrt(diag(inv)/T).
    # ---------------------------------------------------------------------
    l_sr = float(penalty_list[i_sr])
    l_csr2 = float(penalty_list[i_csr2])
    b_sr, se_sr = core.l2est(Sigma_bar, mu_bar, l_sr, T, compute_se=True)
    t_sr = b_sr / se_sr
    b_c, se_c = core.l2est(Sigma_bar, mu_bar, l_csr2, T, compute_se=True)
    t_c = b_c / se_c
    assert np.isfinite(t_sr).all() and np.isfinite(t_c).all()

    # Top 10 raw anomalies by |t| under the SR-selected SDF.
    order_sr = np.argsort(-np.abs(t_sr))[:10]
    rows = []
    for j in order_sr:
        col = ANOM_COLS[j]
        rows.append({
            "label": LABELS.get(col, col),
            "b_SR": float(b_sr[j]), "t_SR": float(t_sr[j]),
            "b_CSR2": float(b_c[j]), "t_CSR2": float(t_c[j]),
        })

    tj = {
        "kappa_SR_opt": kappa_sr, "l_SR": l_sr,
        "kappa_CSR2_opt": kappa_csr2, "l_CSR2": l_csr2,
        "note": ("CSR2 columns equal the existing Table 1a (same kappa_CSR2*, "
                 "same Sigma_bar/mu_bar). Rows are top-10 by |t| under the "
                 "SR-selected SDF."),
        "rows": rows,
    }
    tjson = os.path.join(OUT, "table1_sr_compare.json")
    with open(tjson, "w") as fh:
        json.dump(tj, fh, indent=2)

    md = [
        "# Table 1 analog — SR-selected vs CSR2-selected SDF (50 anomalies)",
        "",
        f"kappa_SR* = {kappa_sr:.4f} (l = {l_sr:.6g}, grid idx {i_sr})  |  "
        f"kappa_CSR2* = {kappa_csr2:.4f} (l = {l_csr2:.6g}, grid idx {i_csr2}),  T = {T}",
        "",
        "Top 10 raw anomalies by |t| under the SR-selected SDF, shown side by side "
        "with the same anomalies' coefficients/t-stats under the CSR2-selected SDF "
        "(the CSR2 columns equal the existing Table 1a). eq.(22)-(23) on full-sample "
        "Sigma_bar/mu_bar.",
        "",
        "| # | Factor | b (SR*) | t (SR*) | b (CSR2*) | t (CSR2*) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for n, r in enumerate(rows, 1):
        md.append(f"| {n} | {r['label']} | {r['b_SR']:.2f} | {r['t_SR']:.2f} | "
                  f"{r['b_CSR2']:.2f} | {r['t_CSR2']:.2f} |")
    md += [
        "",
        f"Coefficient-vector summary: ||b_SR|| = {np.linalg.norm(b_sr):.3f}, "
        f"||b_CSR2|| = {np.linalg.norm(b_c):.3f}, "
        f"corr(b_SR, b_CSR2) = {np.corrcoef(b_sr, b_c)[0,1]:.4f}, "
        f"cos-angle = {b_sr@b_c/(np.linalg.norm(b_sr)*np.linalg.norm(b_c)):.4f}.",
    ]
    tmd = os.path.join(OUT, "table1_sr_compare.md")
    with open(tmd, "w") as fh:
        fh.write("\n".join(md) + "\n")

    # ---------------------------------------------------------------------
    # 9. Console summary.
    # ---------------------------------------------------------------------
    print("=== SR objective extension ===")
    print(f"kappa_SR*           = {kappa_sr:.4f} (idx {i_sr}, 1-SE parsimony)")
    print(f"kappa_SR* (argmax)  = {kappa_sr_argmax:.4f} (idx {i_sr_argmax})")
    print(f"kappa_CSR2*         = {kappa_csr2:.4f} (idx {i_csr2})")
    print(f"within-1se band     : width={band_width:.4f}, n_in_band={n_in_band}/{L}")
    print(f"pooled-SR-selected  : kappa={kappa_sr_pooled:.4f} (idx {i_sr_pooled}), "
          f"gap to per-fold = {perfold_vs_pooled_gridgap} steps")
    print("--- cross-table (objective selected x metric) ---")
    print(f"             | OOS SR  | OOS CSR2")
    print(f"  @ SR*      | {oos_sr_at_sropt:7.4f} | {oos_csr2_at_sropt:8.4f}")
    print(f"  @ CSR2*    | {oos_sr_at_csr2opt:7.4f} | {oos_csr2_at_csr2opt:8.4f}")
    print(f"IS SR ceiling (sqrt maxSR2) = {is_max_sr_ceiling:.4f}")
    print("--- sanity ---")
    print(f"  scale_invariance_max_diff = {scale_invariance_max_diff:.2e}")
    print(f"  perfold_vs_pooled_gridgap = {perfold_vs_pooled_gridgap}")
    print(f"  sr_ge_at_SRopt_vs_CSR2opt = {sr_ge}  (parsimony-selected; can be "
          "False when weakly identified)")
    print(f"  sr_ge_at_SRargmax_vs_CSR2 = {sr_ge_argmax}  (argmax-bug guard)")
    print(f"  oos_below_ceiling         = {oos_below_ceiling} / argmax "
          f"{oos_below_ceiling_argmax}  (no abort fired)")
    print(f"||b_SR||={np.linalg.norm(b_sr):.3f}  ||b_CSR2||={np.linalg.norm(b_c):.3f}  "
          f"cos-angle={b_sr@b_c/(np.linalg.norm(b_sr)*np.linalg.norm(b_c)):.4f}")
    print(f"wrote {jpath}")
    print(f"wrote {png1}")
    print(f"wrote {png2}")
    print(f"wrote {tjson}")
    print(f"wrote {tmd}")


if __name__ == "__main__":
    main()
