```json
{
  "paper": "Kozak, Nagel & Santosh (2020 JFE) — Shrinking the Cross-Section",
  "nodes": [
    {
      "step_id": "s01",
      "stage": "pull_data",
      "action": "Load all in-repo CSV inputs into dataframes; parse dates; coerce numeric.",
      "inputs": ["anom50_managed_daily", "anom50_managed_monthly", "market_rme", "ff25_daily", "ff25_monthly", "ff5_factors_daily", "ff_momentum_daily", "ff3_factors_daily"],
      "outputs": ["raw_frames"],
      "depends_on": [],
      "produces_result_id": null,
      "abort_if": null,
      "assumptions": []
    },
    {
      "step_id": "s02",
      "stage": "build_sample",
      "action": "Build anom50 daily sample: drop rX_/r2_/r3_ prefixed and re_ew cols, dropna, assert realized start=1973-11-01 and H=50, assert daily frequency.",
      "inputs": ["s01", "anom50_managed_daily"],
      "outputs": ["anom50_panel"],
      "depends_on": ["s01"],
      "produces_result_id": null,
      "abort_if": "realized start != 1973-11-01 or H != 50",
      "assumptions": ["A-C16", "A-C17"]
    },
    {
      "step_id": "s03",
      "stage": "construct_variables",
      "action": "Market-orthogonalize (full-sample beta via demarket) and devol-rescale to in-sample market sigma; build mu_bar (daily mean) and Sigma_bar (regcov flat-Wishart a=H/(H+T)).",
      "inputs": ["s02", "market_rme"],
      "outputs": ["mu_bar", "sigma_bar", "anom50_demarket_devol"],
      "depends_on": ["s02"],
      "produces_result_id": null,
      "abort_if": null,
      "assumptions": ["A-C1", "A-C3", "A-C4", "A-C13", "A-C15", "A-C23"]
    },
    {
      "step_id": "s04",
      "stage": "construct_variables",
      "action": "Fix DOF reshape bug to (-1,1)/(1,-1); implement kappa<->gamma map gamma=tau/(kappa^2*T)*freq with gridsize=100 and data-driven right edge (1% stabilize).",
      "inputs": ["s03"],
      "outputs": ["kappa_grid", "gamma_grid"],
      "depends_on": ["s03"],
      "produces_result_id": null,
      "abort_if": null,
      "assumptions": ["A-C5", "A-C10", "A-C18"]
    },
    {
      "step_id": "s05",
      "stage": "run_specs",
      "action": "Run L2 estimator path eq.22 with eq.23 SEs across kappa grid; K=3 contiguous CV with lCV=l/(1-1/K); compute CSR2 eq.30.",
      "inputs": ["s03", "s04"],
      "outputs": ["l2_path", "cv_csr2_curve"],
      "depends_on": ["s03", "s04"],
      "produces_result_id": null,
      "abort_if": null,
      "assumptions": ["A-C6", "A-C7", "A-C8", "A-C11"]
    },
    {
      "step_id": "s06",
      "stage": "run_specs",
      "action": "FF25 branch: wire scs_main ff25 stub, build mu/Sigma on ff25_daily, run L2 CV path, extract peak OOS CSR2 and kappa.",
      "inputs": ["s05", "ff25_daily", "ff3_factors_daily"],
      "outputs": ["ff25_cv_curve"],
      "depends_on": ["s05"],
      "produces_result_id": "fig2a_ff25_L2_peak",
      "abort_if": "FF25 peak kappa off by >1 decade from target band",
      "assumptions": ["A-C1", "A-C3", "A-C4", "A-C13", "A-C18"]
    },
    {
      "step_id": "s07",
      "stage": "run_specs",
      "action": "50-anom L2 model selection plus Pastor&Stambaugh eta=1 curve b_hat=(Sigma+gamma*Sigma)^-1*Sigma*mu_bar with remapped x-axis; extract KNS peak and P&S peak.",
      "inputs": ["s05"],
      "outputs": ["anom50_cv_curve", "ps_eta1_curve"],
      "depends_on": ["s05"],
      "produces_result_id": "fig4a_anom50_L2_peak",
      "abort_if": "50-anom OOS CSR2 non-positive at peak after confirming demarket/devol/freq/sample",
      "assumptions": ["A-C2-PS"]
    },
    {
      "step_id": "s08",
      "stage": "run_specs",
      "action": "Table 1: run rotate_PC=False (Panel a, characteristic space) and rotate_PC=True (Panel b, PC space) at CV-optimal kappa; extract coefficients and t-stats.",
      "inputs": ["s05"],
      "outputs": ["table1a_coeffs", "table1b_coeffs"],
      "depends_on": ["s05"],
      "produces_result_id": "table1a_lead_factor",
      "abort_if": "Table 1a lead factor wrong sign after confirming settings",
      "assumptions": ["A-C9", "A-C19", "A-C22"]
    },
    {
      "step_id": "s08b",
      "stage": "run_specs",
      "action": "Emit Table 1 Panel b PC-space top factors from the rotate_PC=True fit produced in s08.",
      "inputs": ["s08"],
      "outputs": ["table1b_pc_loadings"],
      "depends_on": ["s08"],
      "produces_result_id": "table1b_pc_loadings",
      "abort_if": null,
      "assumptions": ["A-C9", "A-C19", "A-C22"]
    },
    {
      "step_id": "s09",
      "stage": "construct_variables",
      "action": "Build dual-penalty LARS-EN estimator eq.28 (whitened reduction X_tilde=Sigma^1/2, y_tilde=Sigma^-1/2*mu_bar; fit_intercept=False, no normalization; alpha<->(gamma1,gamma2) incl 1/(2H) factor; PC sign convention).",
      "inputs": ["s03", "s04"],
      "outputs": ["dual_penalty_estimator"],
      "depends_on": ["s03", "s04"],
      "produces_result_id": null,
      "abort_if": "LARS-EN cannot reproduce char-vs-PC crossing after verifying Sigma^-1 HJ objective and no-centering",
      "assumptions": ["A-C20", "A-C21", "A-C22"]
    },
    {
      "step_id": "s10",
      "stage": "run_specs",
      "action": "Fig 3 contour over (kappa x #nonzero) for raw and PC bases via dual-penalty estimator plus CV; characterize ridge shape.",
      "inputs": ["s09", "s05"],
      "outputs": ["fig3_contour_raw", "fig3_contour_pc"],
      "depends_on": ["s05", "s09"],
      "produces_result_id": "fig3_anom50_contour",
      "abort_if": null,
      "assumptions": ["A-C20", "A-C21"]
    },
    {
      "step_id": "s11",
      "stage": "run_specs",
      "action": "Fig 4b char-vs-PC sparsity curves (upper ridge of the Fig 3 contour) for both bases; verify PC dominance.",
      "inputs": ["s09", "s10"],
      "outputs": ["fig4b_char_curve", "fig4b_pc_curve"],
      "depends_on": ["s09", "s10"],
      "produces_result_id": "fig4b_char_vs_pc",
      "abort_if": "PC curve does not dominate characteristics curve after verifying HJ objective and no-centering",
      "assumptions": ["A-C20", "A-C21"]
    },
    {
      "step_id": "s12",
      "stage": "build_sample",
      "action": "Table-4 OOS leg: restore OOS split (remove SCS_L2est line-35 override; fix demarket tuple/isinstance branch); apply pre-2005 beta and pre-2005 devol; pre-2005 CV for kappa with K=3 shorter blocks.",
      "inputs": ["s02", "s03", "market_rme"],
      "outputs": ["oos_split_pre2005", "oos_split_post2005", "oos_kappa"],
      "depends_on": ["s02", "s03"],
      "produces_result_id": null,
      "abort_if": "OOS split not actually withheld (post-2005 information leaks into estimation)",
      "assumptions": ["A-C1", "A-C2", "A-C3", "A-C6", "A-C12"]
    },
    {
      "step_id": "s13",
      "stage": "run_specs",
      "action": "Table-4 columns: CAPM (zero-mean leg), FF6 (5 non-market factors, unreg pre-2005 MVE), char-sparse 5-factor (dual penalty), PC-sparse 5-factor; scale MVE to OOS market sigma; alpha via OLS intercept-only TS regression annualized x252x100.",
      "inputs": ["s09", "s12", "ff5_factors_daily", "ff_momentum_daily", "ff3_factors_daily"],
      "outputs": ["table4_alphas", "table4_ses"],
      "depends_on": ["s09", "s12"],
      "produces_result_id": "table4_anom50_alphas",
      "abort_if": "50-anom CAPM or FF6 alpha negative or <2 s.e. after correct OOS split",
      "assumptions": ["A-C11", "A-C25", "A-C26", "A-C14"]
    },
    {
      "step_id": "s14",
      "stage": "compare_to_targets",
      "action": "Compare all produced numbers to target_results under stated tolerances; emit per-target pass/fail verdict and overall replication verdict.",
      "inputs": ["s06", "s07", "s08", "s08b", "s10", "s11", "s13"],
      "outputs": ["verdict_report"],
      "depends_on": ["s06", "s07", "s08", "s08b", "s10", "s11", "s13"],
      "produces_result_id": null,
      "abort_if": null,
      "assumptions": []
    }
  ]
}
```

## Notes (human-readable)

The DAG is acyclic; every `depends_on` references a defined `step_id`, and every
`produces_result_id` matches a `result_id` in `target_results.md`. Node `s08` (Table 1)
is split into `s08`/`s08b` because a node carries only one `produces_result_id` while
Table 1 yields two targets (Panel a and Panel b).

**Critical-path ordering:**
1. `s01→s02→s03→s04→s05` builds the corrected L2 pipeline (data → sample → moments →
   κ-grid → L2/CV path). This alone produces Figs 2a/4a and Table 1 (`s06`–`s08b`).
2. `s09` builds the **new dual-penalty LARS-EN estimator** — the one genuinely new piece
   of engineering — feeding the contour/sparsity figures (`s10`, `s11`) and the
   char-sparse/PC-sparse columns of Table 4 (`s13`).
3. `s12`→`s13` is the OOS-MVE α test: it requires restoring the (currently broken) OOS
   split and re-deriving pre-2005 β and σ. This is the highest look-ahead-risk node.
4. `s14` scores everything against `target_results.md`.

The two monthly (plotting-only) manifest variables are intentionally not wired as
estimation inputs — daily drives all covariance/estimation work.
