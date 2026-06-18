```json
{
  "paper": "Kozak, Nagel & Santosh (2020 JFE) — Shrinking the Cross-Section [extension: cross-validated Sharpe-ratio objective]",
  "targets": [
    {
      "result_id": "sr_objective_selects",
      "location": "extension (no paper cell)",
      "description": "The SR objective runs across the kappa grid and selects an optimal kappa_SR* via the 1-SE-toward-parsimony tie-break, with a finite, well-defined annualized OOS Sharpe at the selected point.",
      "statistic_type": "sharpe",
      "cells": [
        { "label": "kappa_SR_opt", "reported_value": null, "reported_se_or_t": null },
        { "label": "oos_annual_SR_at_opt", "reported_value": null, "reported_se_or_t": null },
        { "label": "within_1se_band_width", "reported_value": null, "reported_se_or_t": null }
      ],
      "tolerance": { "rule": "sign_and_significance", "value": null },
      "note": "No paper number. PASS = objective produces a finite kappa_SR* and OOS SR; band width reported so weak-identification (scale-invariance flat curve) is visible [A-SR7]."
    },
    {
      "result_id": "sr_oos_plausible_band",
      "location": "extension (sanity)",
      "description": "Annualized OOS cross-validated Sharpe at kappa_SR* lies in a plausible band and below the in-sample ceiling.",
      "statistic_type": "sharpe",
      "cells": [
        { "label": "oos_annual_SR_at_opt", "reported_value": null, "reported_se_or_t": null },
        { "label": "is_max_annual_SR_ceiling", "reported_value": 3.5, "reported_se_or_t": null }
      ],
      "tolerance": { "rule": "sign_and_significance", "value": null },
      "note": "Plausible OOS CV annual SR ~0.3-2.0; RED FLAG (look-ahead) if OOS SR >= sqrt(in-sample maxSR2) ~ 3.5. Level is upward-biased (penalty chosen on whole sample) and used only for ranking [A-SR14]."
    },
    {
      "result_id": "sr_vs_csr2_consistency",
      "location": "extension (self-consistency)",
      "description": "Cross-objective consistency: OOS SR at kappa_SR* >= OOS SR at the CSR2-optimal kappa (near-tautological bug check); and the per-fold-selected vs pooled-selected kappa agree to within a couple of grid steps.",
      "statistic_type": "sharpe",
      "cells": [
        { "label": "oos_SR_at_kappa_SR_opt", "reported_value": null, "reported_se_or_t": null },
        { "label": "oos_SR_at_kappa_CSR2_opt", "reported_value": null, "reported_se_or_t": null },
        { "label": "perfold_vs_pooled_kappa_gridstep_gap", "reported_value": null, "reported_se_or_t": null }
      ],
      "tolerance": { "rule": "sign_and_significance", "value": null },
      "note": "PASS = SR(kappa_SR*) >= SR(kappa_CSR2*) (else argmax/selection bug), AND per-fold vs pooled kappa within a few grid steps (else both surfaced) [A-SR11/B3]."
    },
    {
      "result_id": "sr_scale_invariance",
      "location": "extension (unit test)",
      "description": "Scale-invariance unit test: the annualized Sharpe of (te @ (c*b)) equals that of (te @ b) for c in {0.5, 2, 100}.",
      "statistic_type": "other",
      "cells": [
        { "label": "max_abs_SR_diff_over_scalings", "reported_value": 0.0, "reported_se_or_t": null }
      ],
      "tolerance": { "rule": "abs_diff", "value": 1e-08 },
      "note": "Confirms SR is homogeneous degree-0 in b; objective responds only to b's direction, not level [A-SR10]."
    },
    {
      "result_id": "sr_no_lookahead",
      "location": "extension (assertion)",
      "description": "No look-ahead beyond the shared returns_dm_dv: in every fold, b is estimated from TRAIN rows only and the OOS Sharpe uses TEST rows only (train/test disjoint).",
      "statistic_type": "other",
      "cells": [
        { "label": "train_test_index_overlap", "reported_value": 0, "reported_se_or_t": null }
      ],
      "tolerance": { "rule": "abs_diff", "value": 0 },
      "note": "Structural assertion. The only (shared, documented) look-ahead is the full-sample demarket/devol baked into returns_dm_dv, identical to the CSR2 path [A-SR13]."
    }
  ]
}
```

## Notes

This is an **off-paper add-on** — KNS (2020) selects the penalty on out-of-sample
cross-sectional R² only; the Sharpe ratio appears in the paper solely as the Section-5
final evaluation, never as a selection criterion. There is therefore **no reported paper
number** for any target here (`reported_value: null` throughout, except the in-sample
ceiling ~3.5 and the exact unit-test/assertion values). The validation is purely
**sanity / self-consistency**: the objective runs and selects a κ, the selected OOS Sharpe
is plausible and below the in-sample ceiling, the two objectives are mutually consistent,
SR is provably scale-invariant, and there is no look-ahead beyond the shared input.
