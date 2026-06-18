# Extension Plan — Cross-Validated Sharpe-Ratio Objective

**On top of** the completed replication of Kozak, Nagel & Santosh (2020, JFE),
*Shrinking the Cross-Section* (see `replication/replication_report.md`).
**Verdict: GO.** Repro-skeptic **signed off** on the design (conditional on the
implementation realizing the pinned assumptions verbatim).

## 1. What the user asked for

Add **one** functionality on top of the existing replication: a selectable
cross-validation objective that **maximizes the cross-validated Sharpe ratio** of the
SDF-implied MVE portfolio — alongside the existing cross-sectional R² (`CSR2`) objective.
"Only for having it": it is a **reported diagnostic**, not a result that drives any
downstream/production model.

## 2. Where it fits

The completed pipeline (`replication/build/code/scs_core.py`) selects the ridge penalty κ
by K=3 contiguous cross-validation, maximizing the OOS cross-sectional R² (`cross_validate_csr2`,
eq. 30). The original MATLAB code (`SCS_L2est.py`) lists `'SR': 'Sharpe Ratio'` among its
objectives but the Python port only implemented `CSR2` (the `SR` branch in `cross_validate.py`
is commented out). This extension adds the `SR` objective and a selectable dispatcher.

## 3. Methodology (pseudocode)

A new `cross_validate_sr(dates, r, l_value, kfold, freq)` mirrors `cross_validate_csr2`:

```
for each contiguous fold f (K=3):
    TRAIN = all other rows; X_tr = regcov(TRAIN), y_tr = mean(TRAIN)
    b = l2est(X_tr, y_tr, l_used, |TRAIN|)        # l_used = l_value / (1 - 1/kfold)
    r_mve_oos = te @ b                            # held-out MVE realized returns (te = raw test block)
    SR_oos[f] = mean(r_mve_oos)/std(r_mve_oos, ddof=0) * sqrt(freq)   # annualized
return mean/se over folds  +  pooled SR over concatenated OOS series
select kappa_SR* = argmax(OOS SR)  with a 1-SE-toward-parsimony tie-break
```

Key conceptual point: **the Sharpe ratio is scale-invariant in `b`** (`SR(λb)=SR(b)`),
unlike CSR2 which depends on the magnitude of `b`. So the SR objective responds **only to
the direction/rotation** of `b` across principal components (the relative shrinkage the
ridge applies to high- vs low-eigenvalue PCs), not to level shrinkage. The L2 ridge still
rotates `b` (it down-weights low-eigenvalue PCs more), so `SR(κ)` is non-trivial — but the
curve can be **flat with near-ties**, which is why the selection uses a deterministic
tie-break. The OOS MVE return series `te @ b` is deliberately a *different object* from
CSR2's moment-based `Σ_te b`, so the two objectives may rank κ differently.

The only new code is `sharpe_annual` + `cross_validate_sr` + a 2-line selectable
dispatcher in `scs_core.py`. Everything else — the de-marketed/devol'd returns, the κ grid,
the contiguous folds, the L2 estimator, the CV penalty rescale — is reused unchanged.

## 4. Dependency triage

All inputs are **class A** — the existing `foundation.npz` (`returns_dm_dv`, the κ grid,
`freq`, `kfold`). **No new data, no WRDS.** See `data_manifest.md`.

## 5. Success criterion + tolerance

No paper number exists (the paper never selects on Sharpe). The bar is
**sanity / self-consistency** (see `target_results.md`):
- the objective runs and selects a finite κ_SR* (with its within-1-SE band width reported);
- the OOS annual Sharpe at κ_SR* is plausible (~0.3–2.0) and **below** the in-sample
  ceiling `√maxSR2 ≈ 3.5` (exceeding it is a look-ahead red flag);
- `SR(κ_SR*) ≥ SR(κ_CSR2*)`, and the per-fold vs pooled κ agree within a few grid steps;
- a scale-invariance unit test passes to 1e-8;
- no look-ahead beyond the shared `returns_dm_dv`.

## 6. Abandon conditions

- OOS annual SR at κ_SR* **≥ √maxSR2 (~3.5)** → look-ahead bug; stop and diagnose.
- `SR(κ_SR*) < SR(κ_CSR2*)` → selection/argmax bug; stop and diagnose.

## 7. Pinned assumptions & the three skeptic-required guards

The skeptic required three implementation guards (all committed):
- **B1** — deterministic tie-break on the flat, scale-invariant `SR(κ)` curve: among grid
  points within `oos_se` of the max, pick the most-shrunk κ; report the band width and
  label "weakly identified" when wide.
- **B2** — degenerate-denominator guard: per-fold `std > 1e-12` else NaN (never inf),
  nan-aware aggregation, cannot win selection.
- **B3** — select on per-fold-averaged SR, report pooled as diagnostic, and cross-check
  that the per-fold and pooled κ agree (surface both if they diverge).

Full pinned set `[A-SR1..15]` is in `open_questions.md`.

## 8. Scope guard

Strictly the **realized-MVE Sharpe (`SR`)** objective. The cousin objectives in the
original code (`SRexpl`, `MVU`, `GLS`, `SSE`) are **out of scope**. The SR objective is a
reported diagnostic and does not replace the CSR2-selected model used elsewhere in the
replication.

## 9. Output location

This extension is written under `replication/extensions/sr_objective/` to keep the
completed main-replication artifacts intact. Implementation (downstream) adds
`cross_validate_sr` to `replication/build/code/scs_core.py` and emits
`replication/build/output/sr_objective.json` (+ an optional `SR(κ)` curve PNG).
