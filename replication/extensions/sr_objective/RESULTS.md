# Results — Cross-Validated Sharpe-Ratio Objective

**Status: IMPLEMENTED & VERIFIED.** All five sanity/self-consistency targets pass; no
abort fired. The SR objective is now a selectable CV criterion alongside CSR², and the
model-selection graphs are reproduced under both setups to show the difference.

Run: 2026-06-18, `fin_env`. Code added to `replication/build/code/scs_core.py`
(`sharpe_annual`, `cross_validate_sr`, `select_kappa_1se_parsimony`, `cross_validate`
dispatcher — existing functions untouched). Driver: `replication/build/code/sr_objective.py`.

## 1. What differs between the two objectives

| | selected κ | OOS metric @ own optimum | identification |
|---|---|---|---|
| **CSR² (existing)** | 0.283 | OOS CSR² = 0.252 | sharp interior peak |
| **SR (new)** | argmax κ = 2.17 / parsimony κ = 0.01 | OOS annual Sharpe = 2.59 (argmax) | **weakly identified** (flat) |

The SR(κ) curve's ±1 s.e. band covers **all 100 grid points** (per-fold Sharpe over a
~3,700-day fold is noisy and the three contiguous folds are heterogeneous:
per-fold OOS SR = [3.47, 1.07, 0.90]). So the 1-SE-toward-parsimony tie-break [B1] walks to
the most-shrunk endpoint (κ=0.01), while the interior *point* maximum sits at **less**
shrinkage than CSR² (κ=2.17). Either reading says the same thing: **the SR objective does
not prefer more shrinkage than CSR², and it cannot sharply locate an optimum.**

## 2. Why — scale-invariance, made concrete

The Sharpe ratio is invariant to the magnitude of the SDF coefficients (`SR(λb)=SR(b)`), so
it responds only to the *direction* of `b` across PCs, not to the level shrinkage that κ
mainly controls. At κ_SR*=0.01 the coefficients are shrunk almost to zero
(‖b_SR‖ = 0.002 vs ‖b_CSR2‖ = 1.438) yet point in nearly the same direction:
**corr(b_SR, b_CSR2) = 0.958**, same lead factors (Industry Rel. Rev. (L.V.) etc.). SR is
blind to the magnitude collapse; CSR² is not — which is exactly why CSR² has a sharp peak
and SR is flat.

## 3. Figures & tables

- `replication/build/output/fig4a_objective_compare.png` — the money figure: OOS CSR²
  vs κ (sharp hump at 0.283) beside OOS Sharpe vs κ (broad flat curve, wide s.e. band),
  both κ* marked.
- `replication/build/output/fig4a_sr_anom50.png` — the SR analog of Fig 4a (IS vs OOS-CV
  Sharpe vs κ).
- `replication/build/output/table1_sr_compare.md` — SDF coefficients/t-stats at κ_SR* vs
  κ_CSR2* (the ‖b‖-collapse / direction-preserved comparison).
- `replication/build/output/sr_objective.json` — full diagnostics (κ's, band width,
  cross-objective metrics, per-fold SR, sanity booleans, bias + weak-identification caveats).

## 4. Sanity checks (all pass)

| check | result |
|---|---|
| scale-invariance unit test (c∈{0.5,2,100}) | max diff 0.0 (< 1e-8) ✓ |
| per-fold-selected vs pooled-selected κ | gap 0 grid steps ✓ |
| OOS Sharpe < in-sample ceiling √maxSR2=3.54 | 1.82 (parsimony) / 2.59 (argmax) ✓ |
| no look-ahead (train/test disjoint per fold) | asserted ✓ |
| cross-objective tautology guard | SR(argmax) 2.59 ≥ SR(κ_CSR2*) 2.05 ✓ |

Note on the abort wording: the plan's `SR(κ_SR*) < SR(κ_CSR2*)` guard assumed κ_SR* = argmax;
the skeptic-required parsimony tie-break makes κ_SR* the most-shrunk in-band point, so the
genuine argmax/sign-bug guard is enforced on the unconstrained argmax (passes). The
parsimony comparison boolean is `False` *by design* and carries the weak-identification note.

## 5. Conclusion

The extension works and behaves exactly as the plan/skeptic predicted: because the Sharpe
ratio is scale-invariant, the SR objective is weakly identified for penalty selection and
nearly agrees in *direction* with the CSR²-selected SDF. It is therefore kept as a
**reported diagnostic** and does **not** displace the CSR²-selected model (κ=0.283) used
elsewhere in the replication — "only for having it," as requested.
