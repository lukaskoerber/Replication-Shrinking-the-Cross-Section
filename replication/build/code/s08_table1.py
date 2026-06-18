"""
s08 + s08b — Table 1 Panel a (raw) & Panel b (PC):
results `table1a_lead_factor`, `table1b_pc_loadings`.

Kozak, Nagel & Santosh (2020), "Shrinking the Cross-Section", JFE 135(2),
Table 1. At the CV-optimal kappa* for the 50 anomalies (argmax of the frozen
oos_csr2), the SDF coefficients b and SEs are computed with eq.(22)-(23):
    b  = (Sigma_bar + l* I)^{-1} mu_bar
    se = sqrt(diag((Sigma_bar + l* I)^{-1}) / T)
    t  = b / se.

Panel a  [A-C19, rotate_PC=False]: raw anomaly space; sort by |t| desc, top 10.
Panel b  [A-C9, A-C22, rotate_PC=True]: rotate into PC space via eigh of
Sigma_bar (descending eigenvalue), pin each eigenvector's sign so its
largest-|element| loading is positive; estimate b_PC, se_PC at the SAME l*;
sort by |t| desc, top 12.  By rotation-invariance b_PC = Q^T b_raw (verified).
"""
import os
import json
import numpy as np

import scs_core as core

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "replication", "build", "output")
FOUND = os.path.join(OUT, "foundation.npz")

# Column order in returns_dm_dv == anomaly CSV order (verified against the CSV).
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


def main():
    f = np.load(FOUND, allow_pickle=True)
    Sigma_bar = f["Sigma_bar"]
    mu_bar = f["mu_bar"]
    oos_csr2 = f["oos_csr2"]
    penalty_list = f["penalty_list"]
    kappa_grid = f["kappa_grid"]
    R = f["returns_dm_dv"]
    T = R.shape[0]
    H = Sigma_bar.shape[0]
    assert H == 50 and len(ANOM_COLS) == 50

    # CV-optimal kappa* (argmax of frozen OOS CSR2) and its penalty.
    i_star = int(np.argmax(oos_csr2))
    l_star = float(penalty_list[i_star])
    kappa_star = float(kappa_grid[i_star])

    # -------- Panel A: raw space [A-C19] --------
    b, se = core.l2est(Sigma_bar, mu_bar, l_star, T, compute_se=True)
    t = b / se
    assert np.isfinite(t).all()
    order_a = np.argsort(-np.abs(t))
    panel_a = []
    for j in order_a[:10]:
        col = ANOM_COLS[j]
        panel_a.append({"label": LABELS.get(col, col),
                        "b": float(b[j]), "t": float(t[j])})

    # -------- Panel B: PC space [A-C9, A-C22] --------
    # eigh -> ascending eigenvalues; reorder DESCENDING so PC1 = highest variance.
    w, V = np.linalg.eigh(Sigma_bar)
    idx = np.argsort(w)[::-1]
    d_sorted = w[idx]
    Q = V[:, idx]
    # Sign convention [A-C22]: flip each eigenvector so its largest-|element|
    # loading is positive (deterministic).
    for k in range(Q.shape[1]):
        j_max = int(np.argmax(np.abs(Q[:, k])))
        if Q[j_max, k] < 0:
            Q[:, k] *= -1.0

    mu_pc = Q.T @ mu_bar
    Sigma_pc = np.diag(d_sorted)                       # = Q^T Sigma_bar Q
    # cross-check rotation fidelity
    assert np.allclose(Q.T @ Sigma_bar @ Q, Sigma_pc, atol=1e-8), "PC rotation mismatch"

    b_pc, se_pc = core.l2est(Sigma_pc, mu_pc, l_star, T, compute_se=True)
    t_pc = b_pc / se_pc
    # rotation-invariance: b_pc == Q^T b_raw
    assert np.allclose(b_pc, Q.T @ b, atol=1e-8), "b_PC != Q^T b_raw"

    order_b = np.argsort(-np.abs(t_pc))
    panel_b = []
    for k in order_b[:12]:
        panel_b.append({"label": f"PC{k + 1}",
                        "b": float(b_pc[k]), "t": float(t_pc[k])})

    # -------- Write JSON --------
    ja = os.path.join(OUT, "table1a_lead_factor.json")
    jb = os.path.join(OUT, "table1b_pc_loadings.json")
    with open(ja, "w") as fh:
        json.dump({"result_id": "table1a_lead_factor",
                   "kappa_star": kappa_star, "l_star": l_star,
                   "cells": panel_a}, fh, indent=2)
    with open(jb, "w") as fh:
        json.dump({"result_id": "table1b_pc_loadings",
                   "kappa_star": kappa_star, "l_star": l_star,
                   "cells": panel_b}, fh, indent=2)

    # -------- Readable combined markdown --------
    md = ["# KNS (2020) Table 1 — 50 anomalies, L2 SDF",
          f"\nCV-optimal kappa* = {kappa_star:.4f}  (penalty l* = {l_star:.6g}, "
          f"grid idx {i_star}), T = {T}\n",
          "## Panel a — raw anomaly space (sorted by |t|, top 10)\n",
          "| # | Factor | b | t |", "|---|---|---:|---:|"]
    for n, row in enumerate(panel_a, 1):
        md.append(f"| {n} | {row['label']} | {row['b']:.2f} | {row['t']:.2f} |")
    md += ["\n## Panel b — PC space (sorted by |t|, top 12)\n",
           "| # | PC | b | t |", "|---|---|---:|---:|"]
    for n, row in enumerate(panel_b, 1):
        md.append(f"| {n} | {row['label']} | {row['b']:.2f} | {row['t']:.2f} |")
    mdpath = os.path.join(OUT, "table1.md")
    with open(mdpath, "w") as fh:
        fh.write("\n".join(md) + "\n")

    print(f"kappa* = {kappa_star:.4f}, l* = {l_star:.6g}, idx {i_star}")
    print("Panel a top 3:", [(r["label"], round(r["b"], 2), round(r["t"], 2)) for r in panel_a[:3]])
    print("Panel b top 4:", [(r["label"], round(r["b"], 2), round(r["t"], 2)) for r in panel_b[:4]])
    print(f"wrote {ja}\nwrote {jb}\nwrote {mdpath}")


if __name__ == "__main__":
    main()
