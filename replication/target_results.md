```json
{
  "paper": "Kozak, Nagel & Santosh (2020 JFE) — Shrinking the Cross-Section",
  "targets": [
    {
      "result_id": "fig2a_ff25_L2_peak",
      "location": "Figure 2a",
      "description": "FF25 OOS cross-validated cross-sectional R^2 as a function of L2 shrinkage kappa; peak height and location.",
      "statistic_type": "r_squared",
      "cells": [
        { "label": "peak_OOS_CSR2", "reported_value": 0.44, "reported_se_or_t": null },
        { "label": "peak_kappa", "reported_value": 0.23, "reported_se_or_t": null }
      ],
      "tolerance": { "rule": "abs_diff", "value": 0.05 },
      "note": "kappa accept band [0.15,0.35]; peak height accept band [0.38,0.50]."
    },
    {
      "result_id": "fig4a_anom50_L2_peak",
      "location": "Figure 4a",
      "description": "50-anomaly OOS cross-validated CSR^2 vs L2 shrinkage; KNS L2 peak plus P&S eta=1 (level-shrinkage) comparison curve peak.",
      "statistic_type": "r_squared",
      "cells": [
        { "label": "peak_OOS_CSR2", "reported_value": 0.23, "reported_se_or_t": null },
        { "label": "peak_kappa", "reported_value": 0.30, "reported_se_or_t": null },
        { "label": "ps_eta1_peak", "reported_value": 0.05, "reported_se_or_t": null }
      ],
      "tolerance": { "rule": "abs_diff", "value": 0.05 },
      "note": "kappa accept band [0.20,0.45]; Pastor&Stambaugh eta=1 curve peaks <0.05 (level shrinkage alone fails). NB: the 50-anomaly OOS peak is ~0.23, NOT 0.44 (that is the FF25 figure)."
    },
    {
      "result_id": "table1a_lead_factor",
      "location": "Table 1 Panel a",
      "description": "Raw 50-anomaly characteristic-space SDF coefficients; identify lead factor (largest |t|).",
      "statistic_type": "slope",
      "cells": [
        { "label": "Industry Rel. Rev. (L.V.)_b", "reported_value": -0.88, "reported_se_or_t": 3.53 }
      ],
      "tolerance": { "rule": "sign_and_significance", "value": null },
      "note": "Lead factor has negative b and largest |t|; require >=7/10 top-factor overlap and sign matches. reported_se_or_t holds |t|=3.53. Requires rotate_PC=False run."
    },
    {
      "result_id": "table1b_pc_loadings",
      "location": "Table 1 Panel b",
      "description": "PC-space SDF coefficients; top factors by |t|.",
      "statistic_type": "slope",
      "cells": [
        { "label": "PC4_b", "reported_value": 1.01, "reported_se_or_t": 4.25 },
        { "label": "PC1_b", "reported_value": -0.54, "reported_se_or_t": 3.08 },
        { "label": "PC2_b", "reported_value": -0.56, "reported_se_or_t": 2.65 },
        { "label": "PC9_b", "reported_value": -0.63, "reported_se_or_t": 2.51 }
      ],
      "tolerance": { "rule": "sign_and_significance", "value": null },
      "note": "PC4 highest |t|; PC1,PC2,PC4,PC9 among top by |t|; coefficients are exact linear combos of Panel a by rotation-invariance. reported_se_or_t holds |t|. PC signs are not economically identified — pin a sign convention (A-C22)."
    },
    {
      "result_id": "fig4b_char_vs_pc",
      "location": "Figure 4b",
      "description": "Sparsity comparison: max OOS CSR^2 vs number of non-zero factors, characteristic basis vs PC basis; PC dominance.",
      "statistic_type": "r_squared",
      "cells": [
        { "label": "pc_curve_dominates_char", "reported_value": null, "reported_se_or_t": null }
      ],
      "tolerance": { "rule": "sign_and_significance", "value": null },
      "note": "Qualitative claim: PC curve reaches ~2/3 of max with <=2 PCs and near-max with ~10 PCs; characteristics curve stays low until n->50. No single scalar reported."
    },
    {
      "result_id": "fig3_anom50_contour",
      "location": "Figure 3a/3b",
      "description": "OOS-R^2 contour over (L2 kappa x number of non-zero factors) for raw characteristic basis vs PC basis.",
      "statistic_type": "r_squared",
      "cells": [
        { "label": "raw_ridge_near_vertical_needs_L2_not_sparsity", "reported_value": null, "reported_se_or_t": null },
        { "label": "pc_high_R2_at_low_nonzero", "reported_value": null, "reported_se_or_t": null }
      ],
      "tolerance": { "rule": "sign_and_significance", "value": null },
      "note": "Qualitative shape claim: raw contour ridge near-vertical (L2 helps, sparsity does not); PC contour attains high R^2 at low non-zero count. No single scalar reported."
    },
    {
      "result_id": "table4_anom50_alphas",
      "location": "Table 4, 50-anomaly row",
      "description": "Annualized OOS alpha (%) of the SDF-implied MVE portfolio against benchmark models, 2005-2017 OOS window.",
      "statistic_type": "alpha",
      "cells": [
        { "label": "CAPM", "reported_value": 12.35, "reported_se_or_t": 5.26 },
        { "label": "FF6", "reported_value": 8.71, "reported_se_or_t": 4.94 },
        { "label": "char_sparse", "reported_value": 9.55, "reported_se_or_t": 3.95 },
        { "label": "pc_sparse", "reported_value": 4.60, "reported_se_or_t": 2.22 }
      ],
      "tolerance": { "rule": "abs_pct", "value": 0.25 },
      "note": "Same sign required; accept within +/-25% relative OR +/-3% absolute, whichever larger. CAPM & FF6 must stay >2 s.e.; ordering CAPM>=FF6>=char-sparse>pc-sparse; identical CAPM SE 5.26 across rows is a structural check (confirms OLS intercept-only + OOS-sigma normalization). reported_se_or_t holds s.e."
    }
  ]
}
```

## Notes (human-readable)

Per the strict schema, each `cells[]` entry carries only `label` / `reported_value` /
`reported_se_or_t`. For **Table 1** the `reported_se_or_t` field holds the absolute
**t-statistic**; for **Table 4** it holds the **standard error**. The meaning is
disambiguated in each target's `note`.

Figures 3 and 4b report no single scalar — their claims are qualitative (ridge shape /
curve dominance), encoded as `reported_value: null` under the `sign_and_significance`
rule. The downstream skill should score them on shape, not a number.

**Tolerance bands (user-accepted):**
- OOS-CSR² peaks (Fig 2a, Fig 4a): ±0.05 absolute on height; peak-κ within the noted band.
- Table 1: lead factor correct sign + largest |t|; ≥7/10 top-factor overlap; signs match.
- Table 4 alphas: same sign; within ±25% relative or ±3% absolute (whichever larger);
  CAPM and FF6 must remain statistically significant (>2 s.e.).
