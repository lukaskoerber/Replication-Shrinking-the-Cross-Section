# Open Questions & Pinned Assumptions — SR objective extension

## A. Decisions (resolved by the chair under the "only for having it" scope)

- **Q1 — Does κ_SR* drive anything downstream?** No. The SR objective is a **reported
  diagnostic only**; the CSR2-selected model remains canonical everywhere else. This
  downgrades the materiality of the weak-identification and shared-look-ahead concerns, but
  the 1-SE tie-break is still implemented for reproducibility.
- **Q2 — K for the Sharpe CV?** K=3 contiguous, for parity with `cross_validate_csr2`. A
  K∈{5,10} sensitivity is an optional nice-to-have, not required.
- **Q3 — Which input does the SR path consume?** The **same** full-sample-preprocessed
  `returns_dm_dv` as the CSR2 path, preserving comparability and inheriting the identical
  (shared, pre-existing) full-sample demarket/devol look-ahead — stated, not fixed.

None of these required asking the user — the "only for having it" framing settles them.

## B. Repro-skeptic SIGN-OFF set — pinned assumptions [A-SR1..15]

The skeptic **signed off** on the design conditional on the implementation realizing these
verbatim. The three blockers (B1, B2, B3) are folded into A-SR7, A-SR8, A-SR11.

| ID | Assumption |
|----|-----------|
| A-SR1 | SR objective = `mean(r_mve)/std(r_mve)` of the OOS MVE return series `r_mve = te @ b`, b from TRAIN moments only. |
| A-SR2 | Annualization `* sqrt(freq)`, freq=252 — consistent with the repo's `maxSR2 = freq·μ'Σ⁺μ`. Cancels in the κ ranking. |
| A-SR3 | Reuse the shared `kappa2pen` map and the **same κ grid** as CSR2, so SR- and CSR2-selected κ are comparable. |
| A-SR4 | CV penalty rescale `l_used = l/(1-1/k)` applied to the SR path too [= A-C6]; non-trivial for SR because it alters b's *direction* (A-SR15-level invariance applies only to scaling, not to l). |
| A-SR5 | Moments and b strictly within each fold's TRAIN block; OOS Sharpe from the TEST block only — no cross-fold leakage. |
| A-SR6 | MVE weights b estimated on train moments; OOS SR uses test-block realized returns `te @ b`. |
| A-SR7 | **(B1)** Flat/scale-invariant `SR(κ)` argmax tie-break = 1-SE-toward-parsimony (most-shrunk κ within `oos_se` of the max); report κ_SR* + within-1-SE band width; label "weakly identified" when the band is wide. |
| A-SR8 | **(B2)** Per-fold `std(r_mve, ddof=0) > 1e-12` assert; degenerate fold → NaN (never inf), excluded from the mean with a surfaced warning, cannot win selection. |
| A-SR9 | `oos_se` = std across folds / sqrt(k), same convention as CSR2; overlapping/autocorrelation SE caveat noted, not corrected (diagnostic-only). |
| A-SR10 | SR is scale-invariant in b (homogeneous degree 0); the original `ignore_scale` branch is moot and is NOT ported. Verified by a unit test. |
| A-SR11 | **(B3)** Select on **per-fold-averaged** OOS SR; report **pooled** (concatenated OOS series) as diagnostic; cross-check the two selected κ agree within a couple of grid steps, surfacing both on divergence. |
| A-SR12 | NaN/degenerate folds handled with nan-aware aggregation (`nanmean`/`nanargmax`) so excluded folds propagate cleanly. |
| A-SR13 | SR path consumes the same full-sample-preprocessed `returns_dm_dv`; inherits the shared full-sample demarket/devol look-ahead — known shared limitation, NOT fixed (acceptable: diagnostic-only). |
| A-SR14 | Std convention ddof=0 (matches repo `np.std`); CV-OOS-SR *level* is upward-biased (penalty chosen on whole sample) and used only for ranking, not as a true OOS Sharpe estimate (mirror the paper's p.20 caveat). |
| A-SR15 | The SR objective is a **reported diagnostic only**; it does not drive any downstream/production artifact. |

## C. Residual (optional) sensitivity checks — not required

- K∈{5,10} robustness of κ_SR* (the per-fold Sharpe is noisier than the per-fold R²).
- A per-fold (rather than full-sample) demarket/devol input, if one ever wants the SR path
  free of the shared look-ahead — a larger change, explicitly out of scope here.
