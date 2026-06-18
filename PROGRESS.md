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

## Sessions

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
