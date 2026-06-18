# Build Progress (resumability ledger)

Env: `/opt/miniconda3/envs/fin_env/bin/python` (py3.11.8, numpy1.26, pandas2.2, sklearn1.6, scipy1.12)
Data: all class A (in repo). No WRDS pull required.

| Section | task_graph nodes | builder | status | checkpoint |
|---|---|---|---|---|
| A. Foundation | s01–s05 | sample-constructor | **PASS** | start 1973-11-01, H=50, T=11141; peak OOS-CSR² 0.252@κ0.283 |
| B. Tier-1 specs | s06, s07, s08, s08b | spec-runner | **PASS** | fig2a 0.446; fig4a 0.252; table1a 10/10; table1b PC1/2/4 |
| C. Dual penalty | s09 | variable-engineer | **PASS** | γ1=0 reproduces L2-only (max|Δb|=0) |
| D. Sparsity figs | s10, s11 | variable-engineer | **PASS** | fig3 raw-vertical/PC-low-n; fig4b PC dominates char |
| E. OOS test | s12, s13 | spec-runner | **PASS** | table4 CAPM 12.85(5.26); no look-ahead; central claim reproduced |
| F. Compare | s14 | execution lead | **PASS** | 7/7 targets within tolerance |

Status legend: pending / in_progress / PASS / PARTIAL / ABORT

Outcome: **REPLICATED** (7/7 in-scope targets). Tier 3 (WFR/interactions) descoped per plan.
All sections passed on first attempt — no diagnostician/retry cycles needed.
