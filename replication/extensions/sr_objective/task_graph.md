```json
{
  "paper": "Kozak, Nagel & Santosh (2020 JFE) — Shrinking the Cross-Section [extension: cross-validated Sharpe-ratio objective]",
  "nodes": [
    {
      "step_id": "sr01",
      "stage": "construct_variables",
      "action": "Add sharpe_annual(x,freq)=mean(x)/std(x,ddof=0)*sqrt(freq) and cross_validate_sr(dates,r,l_value,kfold,freq) to scs_core.py, mirroring cross_validate_csr2: per contiguous fold estimate b on TRAIN moments at l_used=l/(1-1/kfold), form OOS MVE return series r_mve=te@b, compute per-fold annualized SR; return is_mean/is_se/oos_mean/oos_se/oos_folds + pooled_sr. Add a selectable dispatcher cross_validate(objective in {CSR2,SR},...). Implement B2 degenerate-denominator guard (std>1e-12 else NaN, nan-aware aggregation).",
      "inputs": ["anom50_returns_dm_dv", "kappa_penalty_grid"],
      "outputs": ["cross_validate_sr", "sharpe_annual", "cv_dispatcher"],
      "depends_on": [],
      "produces_result_id": null,
      "abort_if": null,
      "assumptions": ["A-SR1", "A-SR2", "A-SR3", "A-SR4", "A-SR5", "A-SR6", "A-SR8", "A-SR9", "A-SR10", "A-SR12", "A-SR13"]
    },
    {
      "step_id": "sr02",
      "stage": "run_specs",
      "action": "Run the SR objective across the kappa grid on anom50_returns_dm_dv (K=3 contiguous). Select kappa_SR* via the 1-SE-toward-parsimony tie-break (most-shrunk kappa within oos_se of the max); report kappa_SR*, OOS annual SR at opt, and the within-1-SE band width (weak-identification label).",
      "inputs": ["sr01", "anom50_returns_dm_dv", "kappa_penalty_grid"],
      "outputs": ["sr_curve", "kappa_SR_opt", "oos_SR_curve", "pooled_SR_curve"],
      "depends_on": ["sr01"],
      "produces_result_id": "sr_objective_selects",
      "abort_if": null,
      "assumptions": ["A-SR7", "A-SR11", "A-SR14", "A-SR15"]
    },
    {
      "step_id": "sr03",
      "stage": "run_specs",
      "action": "Check the selected OOS annual SR is in a plausible band (~0.3-2.0) and below the in-sample ceiling sqrt(maxSR2)~3.5 (red-flag look-ahead if >= ceiling). Carry the upward-bias caveat.",
      "inputs": ["sr02"],
      "outputs": ["sr_band_check"],
      "depends_on": ["sr02"],
      "produces_result_id": "sr_oos_plausible_band",
      "abort_if": "OOS annual SR at kappa_SR* >= sqrt(in-sample maxSR2) (~3.5) -> look-ahead bug",
      "assumptions": ["A-SR13", "A-SR14"]
    },
    {
      "step_id": "sr04",
      "stage": "run_specs",
      "action": "Cross-objective consistency: compute kappa_CSR2* (reuse cross_validate_csr2) and verify OOS SR(kappa_SR*) >= OOS SR(kappa_CSR2*); compute the pooled-selected kappa and verify it agrees with the per-fold-selected kappa within a couple of grid steps, surfacing both on divergence.",
      "inputs": ["sr01", "sr02"],
      "outputs": ["sr_csr2_compare", "perfold_pooled_compare"],
      "depends_on": ["sr01", "sr02"],
      "produces_result_id": "sr_vs_csr2_consistency",
      "abort_if": "OOS SR(kappa_SR*) < OOS SR(kappa_CSR2*) -> selection/argmax bug",
      "assumptions": ["A-SR11"]
    },
    {
      "step_id": "sr05",
      "stage": "run_specs",
      "action": "Scale-invariance unit test: assert sharpe_annual(te@(c*b),freq) == sharpe_annual(te@b,freq) for c in {0.5,2,100} to within 1e-8.",
      "inputs": ["sr01"],
      "outputs": ["sr_scale_invariance_test"],
      "depends_on": ["sr01"],
      "produces_result_id": "sr_scale_invariance",
      "abort_if": null,
      "assumptions": ["A-SR10"]
    },
    {
      "step_id": "sr06",
      "stage": "run_specs",
      "action": "No-look-ahead assertion: in every fold assert train/test index sets are disjoint, b uses train rows only, OOS SR uses test rows only.",
      "inputs": ["sr01"],
      "outputs": ["sr_no_lookahead_test"],
      "depends_on": ["sr01"],
      "produces_result_id": "sr_no_lookahead",
      "abort_if": null,
      "assumptions": ["A-SR5", "A-SR6"]
    },
    {
      "step_id": "sr07",
      "stage": "compare_to_targets",
      "action": "Emit replication/build/output/sr_objective.json (kappa_SR*, OOS SR, band width, pooled SR, kappa_CSR2*, cross-objective comparison, bias caveat) and an optional SR(kappa) curve PNG; record PASS/FAIL of all sanity targets.",
      "inputs": ["sr02", "sr03", "sr04", "sr05", "sr06"],
      "outputs": ["sr_objective.json", "verdict"],
      "depends_on": ["sr02", "sr03", "sr04", "sr05", "sr06"],
      "produces_result_id": null,
      "abort_if": null,
      "assumptions": []
    }
  ]
}
```

## Notes

The critical path is `sr01` (the one piece of new code: `cross_validate_sr` +
`sharpe_annual` + the selectable dispatcher in `scs_core.py`, with the B2 degenerate-fold
guard) → `sr02` (run + 1-SE-parsimony selection, B1) → the four sanity/self-consistency
checks (`sr03`–`sr06`, including the B3 per-fold-vs-pooled cross-check in `sr04`) →
`sr07` (emit JSON + verdict). Everything upstream (`returns_dm_dv`, κ grid, folds,
`l2est`) is reused from the completed replication. Two `abort_if` guards encode the only
real failure modes: an OOS SR exceeding the in-sample ceiling (look-ahead bug) and an OOS
SR at κ_SR* below that at κ_CSR2* (selection bug).
