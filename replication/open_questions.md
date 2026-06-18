# Open Questions & Pinned Assumptions

This file records (a) the user-routed decisions already made, (b) the pinned assumptions
the repro-skeptic requires before implementation, and (c) the remaining low-materiality
items the implementation committee may verify against Kozak's released MATLAB if desired.

---

## A. User decisions (resolved this session)

1. **Scope = Tier 1+2.** Fix the existing L2 pipeline **and** build the missing
   dual-penalty (LARS-EN) estimator. Targets: 50 anomalies + FF25 preliminary.
2. **Tier 3 descoped.** WFR-80 and the 1,375/3,400 interaction sets are documented as a
   future extension, not built this round.
3. **Tolerances accepted** as stated in `target_results.md` / `plan.md §6`.

## B. Data questions resolved during planning (no longer open)

- **Raw vs orthogonalized series.** RESOLVED empirically: the managed-portfolio columns
  are **raw** long-short returns (e.g. `r_value` β = −0.38 on the file's `rme`), and
  orthogonalization is applied in-code. ⇒ Table 4's pre-2005-β OOS test is feasible from
  in-hand data; no WRDS rebuild needed.
- **Sample window.** RESOLVED: after `dropna`, the daily 50-anomaly sample starts
  **1973-11-01** with exactly **50** anomaly columns — matches the paper.

---

## RESOLUTION (post-execution, 2026-06-18)

All 7 in-scope targets **PASSED** on the first builder attempt — no diagnostician was
invoked, so **no assumption was flipped during execution**; every `[A-Cx]` below held as
pinned. Detection checks confirmed the key ones:
- **[A-C16]/[A-C17]** verified: realized start 1973-11-01, H=50 (asserts green).
- **[A-C5]/[A-C13]** verified: OOS-CSR² peaks at κ≈0.25 (FF25) / 0.28 (anomalies) — not
  off by √252.
- **[A-C11]/[A-C26]** verified: Table-4 CAPM SE = **5.26 exactly** — the paper's
  structural fingerprint, confirming OLS intercept-only SEs + OOS-σ normalization.
- **[A-C1]/[A-C2]/[A-C3]** verified: no-look-ahead asserts pass (β, σ, κ*, b̂ all pre-2005).
- **[A-C20]/[A-C21]** verified: dual-penalty at γ₁=0 reproduces L2-only to `max|Δb|=0`.

**One plan correction (resolved by the lead, not the user):** the Table-4 `abort_if`
condition required the 50-anomaly FF6 α to be > 2 s.e. — but the *paper's own* FF6 cell is
t = 1.76 < 2 (it only claims significance for the WFR/interactions rows). The condition was
therefore mis-specified; reproducing t = 1.76 is a success. The abort was not fired. The
intended kill-switch ("central claim collapses") keys on the **CAPM** column, which is
t = 2.44 > 2, positive, and large — i.e. satisfied.

Items in §D (Q1, Q4, Q7) were not pursued: results landed within tolerance without them,
so they remain optional sensitivity checks rather than required resolutions.

---

## C. Pinned assumptions (repro-skeptic SIGN-OFF set)

These convert every paper-silent or code-divergent choice into a defensible default with
a detection check. The implementation must honor them.

- **[A-C1] β-orthogonalization window.** Full-sample β for Figs 1–4 / Tables 1–3;
  **pre-2005 β** for Table 4 (applied to both the 50 portfolios *and* the 5 FF6 factors,
  fn 20/23). *Detect:* a full-sample-β leak inflates the Table 4 CAPM α above 12.35 and
  shrinks its SE.
- **[A-C2] Restore the OOS split.** Remove the `SCS_L2est.py:35` override; honor
  `oos_test_date = 01-Jan-2005`. *Detect:* if collapsed, "OOS" Sharpe explodes and t-stats
  far exceed 5.26.
- **[A-C3] Devol window.** Train-restricted asset σ **and** market-σ target in the OOS
  leg (pre-2005); full-sample σ for the in-sample analyses. (Includes the `market.std()`
  target on `SCS_L2est.py:95`.)
- **[A-C4] regcov shrinkage** `a = H/(H+T)`, `tr(S)/H` normalization. LOW materiality at
  H=50/25 daily (a≈0.005). Re-open if Tier 3 (H=1375/3400) is rescoped.
- **[A-C5]/[A-C13] Units of κ.** κ is annualized root-E[SR²]; γ uses daily τ and daily T;
  `freq=252` enters exactly once in `kappa2pen`; μ̄, Σ̄ stay in raw daily units. *Detect:*
  a peak off by ~√252 ⇒ freq misapplied.
- **[A-C6] CV penalty rescale** `lCV = l/(1−1/K)` ON (holds κ fixed as train shrinks).
- **[A-C7] CV folds**: contiguous, K=3 (no shuffle, no purge/embargo).
- **[A-C8] CV objective** = eq. 30 exactly (fitted `Σ̄₂b̂`, not realized Sharpe); the −0.1
  R² floor is for plotting only, never for κ selection.
- **[A-C9] PC rotation** estimated on the estimation sample (full sample for Tables 1–3;
  pre-2005 for the Table-4 PC-sparse leg), fixed across CV folds — never sees withheld
  data.
- **[A-C10] κ-grid** right edge via the 1%-coefficient-stabilize loop; gridsize=100.
- **[A-C11] Standard errors.** Tables 1–3 t-stats = eq. 23 posterior variance (no HAC).
  **Table 4 α SEs = plain OLS, intercept-only for CAPM.** *Detect:* the identical 5.26
  CAPM SE across all rows confirms OLS + OOS-σ normalization (Newey-West would not give it).
- **[A-C12] Per-fold leakage.** Σ̄/μ̄ are correctly recomputed per fold; devol and PC
  rotation are done once on the estimation sample (consistent with KNS). Only the pre-2005
  leg must restrict σ/β.
- **[A-C14]/[A-C15] Units & market series.** Managed files decimal; FF files ÷100;
  Table-4 α reported ×100 (%). Orthogonalize/devol with the file's own `rme`; FF `Mkt-RF`
  is used only as the CAPM benchmark leg.
- **[A-C16]/[A-C17] Asserts.** Realized start = 1973-11-01 and H = 50 (assert in loader).
- **[A-C18] d.o.f. shapes** dynamic `(-1,1)`/`(1,-1)` (required for the FF25 leg).
- **[A-C19] `rotate_PC` from config**; produce both Table 1 panels.
- **[A-C20] Dual-penalty estimator** (eq. 28) via whitened reduction
  `X̃=Σ̄^{1/2}, ỹ=Σ̄^{-1/2}μ̄` → ElasticNet, `fit_intercept=False`, no normalization.
- **[A-C21] α↔(γ₁,γ₂) mapping** includes the sklearn `1/(2H)` objective-scaling factor
  (else the #non-zero axis of Fig 4b is off by a factor of H).
- **[A-C22] PC sign convention** pinned deterministically (e.g. largest-loading-positive)
  so PC coefficient signs (Table 1b: PC1 = −0.54 vs +0.54) are reproducible.
- **[A-C23] np.cov ddof.** `regcov` uses ddof=1 (T−1) vs eq. 14's 1/T; ~0.01% at daily T,
  acceptable. Optionally set ddof=0 to match eq. 14 exactly.
- **[A-C2-PS] Pástor–Stambaugh η=1 curve.** Estimator `b̂ = (Σ̄+γΣ̄)⁻¹Σ̄μ̄`; x-axis
  re-mapped to E[SR²]^{1/2} under η=1 (no longer equals κ, fn 17).
- **[A-C25] FF6 benchmark** = 5 non-market factors (SMB, HML, UMD, RMW, CMA), each
  market-orthogonalized on pre-2005 β; unregularized MVE weights `ŵ = Σ̂⁻¹μ̂` estimated
  pre-2005, applied OOS. *Detect:* including the market or regularizing drifts FF6 α off
  8.71.
- **[A-C26] MVE normalization** to the **OOS-period** (2005–2017) market σ. Scaling is
  monotone (α and SE scale together, t invariant); this is what makes the CAPM SE column
  identical at 5.26.

---

## D. Remaining items the implementation committee may verify (low materiality)

- **Q1.** Whether `regcov`'s `a = H/(H+T)` identity-shrink exactly matches Kozak's
  released MATLAB (vs an un-regularized daily Σ). Negligible at H=50/25; worth a glance if
  any peak height is off.
- **Q4.** Confirm the LARS-EN library choice and that the whitened reduction reproduces the
  Zou–Hastie augmented-design solution (acceptable substitute, but verify the #non-zero
  path against Fig 4b's axis).
- **Q7.** The exact pre-2005 CV "shorter blocks" length (paper says K=3 contiguous on the
  shorter 1973–2004 span; exact fold boundaries unspecified). Robustness to moderate K.

None of these gate implementation; they are sensitivity checks for the
`/replication-execute` stage.
