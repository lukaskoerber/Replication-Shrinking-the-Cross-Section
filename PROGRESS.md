# Progress Log — Replication: *Shrinking the Cross-Section*

## Current State

The repository now contains a **complete, planned-and-executed replication** of the MAIN
results of Kozak, Nagel & Santosh (2020, JFE) for the 50-anomaly dataset (+ FF25
preliminary), living under `replication/`. Outcome: **REPLICATED** — all 7 in-scope
targets reproduce within tolerance, most near-exactly. The headline claims all hold:
L2 (ridge) shrinkage is essential, a characteristics-sparse SDF fails while a PC-sparse
SDF succeeds, and the SDF-implied MVE portfolio earns a large, significant out-of-sample
alpha (2005–2017) against the CAPM and Fama-French models.

The reproducible pipeline is in `replication/build/code/` (a clean port of the original
repo's L2 estimator plus a newly built dual-penalty estimator); figures, tables, and
per-target JSON are in `replication/build/output/`. The planning artifacts, ledgers, and
write-ups are in `replication/` (`plan.md`, `replication_report.md`, `REPLICATION_LOG.md`,
`results_log.md`, machine-readable `data_manifest.md` / `target_results.md` /
`task_graph.md`). The original top-level scripts (`scs_main.py`, `SCS_L2est.py`,
`results_export/`) remain as the pre-existing L2-only replication and are unchanged in
substance by this work.

A **scoped extension** lives under `replication/extensions/sr_objective/`: a selectable
cross-validation objective that maximizes the cross-validated Sharpe ratio, alongside the
existing cross-sectional R² objective. It is now **implemented and verified** (added to
`scs_core.py`; comparison graphs in `replication/build/output/`), kept as a reported
diagnostic — it does not displace the CSR²-selected model.

## Sessions

### 2026-06-18 — Extension plan: cross-validated Sharpe-ratio objective

- **Planned** (did not implement) one requested add-on: a selectable CV objective that
  maximizes the cross-validated Sharpe ratio of the SDF/MVE portfolio, alongside the
  existing cross-sectional R² objective — "only for having it." It reuses the existing
  pipeline (de-marketed/devol'd returns, κ grid, K=3 folds, L2 estimator) and adds only a
  `cross_validate_sr` function plus a selectable dispatcher; no new data.
- **Key insight pinned:** the Sharpe ratio is scale-invariant in the SDF coefficients, so
  the objective responds only to the *direction* of the coefficient vector across PCs (not
  to level shrinkage) — which can make the SR(κ) curve flat. This drove three guards a
  skeptic required and signed off on: a deterministic 1-SE-toward-parsimony tie-break, a
  degenerate-denominator guard, and a per-fold-vs-pooled cross-check.
- **Validation is sanity-only** (the paper never selects on Sharpe, so no target number):
  the objective must select a finite κ, give a plausible OOS Sharpe below the in-sample
  ceiling (√maxSR2 ≈ 3.5, a look-ahead red flag if exceeded), stay consistent with the
  CSR2-selected model, pass a scale-invariance unit test, and have no look-ahead. The SR
  objective is a reported diagnostic only and does not replace the CSR2-selected model.
- Artifacts (plan + machine-readable manifest/targets/task-graph + pinned assumptions)
  written to `replication/extensions/sr_objective/`.
- **Then implemented and verified it** (same day): added `cross_validate_sr` /
  `sharpe_annual` / a `{CSR2,SR}` dispatcher to `scs_core.py`, ran the SR objective across
  the κ grid, and reproduced the model-selection graphs under both objectives to see the
  difference. Finding (as predicted): because the Sharpe ratio is scale-invariant, the
  SR(κ) curve is **weakly identified** — its ±1 s.e. band spans the whole grid, vs CSR²'s
  sharp peak at κ=0.283 — and at the SR optimum the SDF *direction* is ~96% identical to
  the CSR² SDF while its magnitude collapses. All sanity checks pass; the SR objective is a
  reported diagnostic and does not change the canonical model. See
  `replication/extensions/sr_objective/RESULTS.md`.

### 2026-06-18 — Full replication: plan committee → execution → log

- **Planned** the replication with a read-only committee (data-sourcing, methodology,
  skeptic, feasibility, handoff): mapped every input to a source (all class A — Kozak's
  pre-built managed portfolios + Ken French factors, no WRDS needed), scoped the MAIN
  results, pinned ~26 assumptions, and issued a GO verdict on Tier 1+2 (50 anomalies +
  FF25; WFR/interactions descoped).
- **Executed** section by section with builder + auditor sub-agents. Fixed five real
  defects in the original code (silently-disabled OOS split, broken de-market branch,
  full-sample devol leakage, hard-coded degrees-of-freedom shapes, forced PC rotation) and
  **built the missing dual-penalty (L1+L2 / elastic-net) SDF estimator** — validated to
  reproduce the L2-only estimator exactly at γ₁=0.
- **Reproduced**, within tolerance: Fig 2a (FF25 OOS-R² ≈ 0.45), Fig 4a (50-anom OOS-R² ≈
  0.25 + Pástor–Stambaugh comparison), Table 1 (raw + PC SDF coefficients, lead factor and
  t-stats), Fig 3 (raw-vs-PC sparsity contour), Fig 4b (PC curve dominates characteristics),
  and Table 4 (OOS-MVE alpha: CAPM 12.85% vs reported 12.35%, with the CAPM standard error
  **5.26 reproduced exactly** as a structural check).
- **Learned / decided:** the covariance is the daily sample covariance with only a
  negligible flat-Wishart numerical shrink (`a ≈ 0.005`) — the paper itself treats the
  covariance as *known* and does the economic shrinkage on the means via the ridge penalty,
  so the `regcov` shrink is a code-level detail, not a documented method step. Also
  corrected a plan over-specification: the 50-anomaly FF6 alpha is t≈1.76 in *both* the
  paper and the replication (the paper only claims significance on the WFR/interaction rows),
  so this is a match, not a failure.
