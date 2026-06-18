# Replication Plan — *Shrinking the Cross-Section* (Kozak, Nagel & Santosh, 2020 JFE)

**Verdict: GO** on Tier 1+2 (50 anomalies + FF25 preliminary). **DEFER** Tier 3
(WFR-80 and interactions). Repro-skeptic has **signed off** on the trap list.

---

## 1. Paper summary

Kozak, Nagel & Santosh construct a robust stochastic discount factor (SDF) that
summarizes the joint explanatory power of many cross-sectional return predictors. The
method is a Bayesian/penalized estimator that shrinks SDF coefficients using an
economically motivated prior: contributions of **low-variance principal components**
(PCs) of the candidate factors are shrunk hardest (prior `μ ~ N(0, (κ²/τ)Σ^η)` with
**η = 2**, eq. 18). The L2-only posterior is `b̂ = (Σ + γI)⁻¹μ̄` with `γ = τ/(κ²T)`,
`τ = tr Σ` (eq. 22); adding an L1 penalty (eq. 28, elastic-net / LARS-EN) allows
automatic factor selection.

**Core claims:**
1. Robust OOS performance in high dimensions requires **L2 shrinkage** of SDF
   coefficients (ridge-like, but tilted to shrink low-eigenvalue PCs more).
2. A **characteristics-sparse** SDF (a few factors) **cannot** summarize the
   cross-section — there is little redundancy among the 50 anomalies.
3. A **PC-sparse** SDF (a few high-variance PCs) **can** approximate the SDF well.
4. Pure L1 (Lasso) and η = 1 level-shrinkage (Pástor–Stambaugh) **underperform**.
5. In a true OOS test (2005–2017), the SDF-implied MVE portfolio earns a large,
   significant α vs CAPM and the FF 6-factor model — rejecting sparse models.

---

## 2. Main results targeted (scoped)

| ID | Location | What | Tier |
|----|----------|------|------|
| `fig2a_ff25_L2_peak` | Fig 2a | FF25 preliminary: IS vs OOS-CV CSR² vs κ (sanity check the method recovers an SMB/HML-like SDF) | 1 |
| `fig4a_anom50_L2_peak` | Fig 4a | 50-anom: IS / OOS-CV / P&S η=1 CSR² vs κ; + effective d.o.f. | 1 |
| `table1a_lead_factor` | Table 1a | 50-anom raw SDF coefficients & t-stats | 1 |
| `table1b_pc_loadings` | Table 1b | 50-anom PC-space SDF coefficients & t-stats | 1 |
| `fig3_anom50_contour` | Fig 3a/3b | OOS-R² contour over κ × #non-zero (raw & PC) — **dual penalty** | 2 |
| `fig4b_char_vs_pc` | Fig 4b | max OOS-R² vs #factors, characteristics vs PCs — **dual penalty** | 2 |
| `table4_anom50_alphas` | Table 4 (50-anom row) | OOS-MVE annualized α (2005–2017) vs CAPM / FF6 / char-sparse / PC-sparse | 2 |

**Descoped (Tier 3, documented future extension):** WFR-80 (Table 2, Figs 5–6) and the
1,375 / 3,400 interaction sets (Table 3, Figs 7–8) and their Table 4 rows. Their input
matrices are not in the repo (class B download / class D WRDS rebuild), and their
characteristic definitions carry independent replication risk that would contaminate the
headline verdict. The headline claims do not depend on them.

---

## 3. Existing code: what's reusable, what's broken, what's missing

The repo already implements an **L2-only** estimator on the 50 anomalies
(`SCS_L2est.py`, `cross_validate.py`, `utils.py`, `scs_main.py`). Structure is sound:
the `(Σ+γI)⁻¹μ̄` posterior (`l2est`), the κ↔γ map, K=3 contiguous CV, and the CSR²
objective (eq. 30) are all faithful. But the committee identified concrete defects that
must be fixed before any number is trusted:

**Must-fix bugs (confirmed against source):**
- **C2 — OOS split silently disabled.** `SCS_L2est.py:35` hard-overwrites
  `oos_test_date = dates.iloc[-1]`, collapsing the train/test wall regardless of the
  `withhold_test_sample` flag. Blocks Table 4 entirely until restored.
- **C1 — `demarket` tuple branch broken.** `SCS_L2est.py:74–79`: `demarket()` always
  returns a `(returns, beta)` tuple, so the `isinstance(r_test, pd.DataFrame)` check is
  always False → the test leg is silently dropped. Unpack the tuple.
- **C3 — full-sample devol leakage.** `SCS_L2est.py:92–95` rescales by full-sample σ
  (asset and `market.std()`). In the OOS leg this leaks post-2005 volatility into the
  pre-2005 normalization. Must be train-restricted when withholding.
- **C18 — hard-coded d.o.f. shapes.** `SCS_L2est.py:205` uses `reshape(50,1)` /
  `reshape(1,100)`. Breaks for FF25 (H≠50) and any gridsize≠100. Use `(-1,1)`/`(1,-1)`.
- **C19 — `rotate_PC` forced True.** `scs_main.py:49` shadows the top-level
  `rotate_PC=False`, so Table 1 can only produce Panel b. Drive it from run config and
  produce **both** panels.

**Genuinely missing (must build):**
- **C20 — dual-penalty (L1+L2) LARS-EN estimator (eq. 28).** Required for Fig 3, Fig 4b,
  and the char-sparse / PC-sparse columns of Table 4. Recommended implementation: the
  whitened reduction `X̃ = Σ̄^{1/2}`, `ỹ = Σ̄^{-1/2}μ̄` → elastic net with
  `fit_intercept=False`, no normalization (the paper explicitly does not center/normalize,
  Sec 3.2). Pin the `(γ₁,γ₂) ↔ (α,ρ)` mapping including the `1/(2H)` objective-scaling
  factor, and a deterministic PC sign convention.

---

## 4. Methodology pseudocode (per target)

**Shared pipeline (in-sample analyses — Figs 2/4, Tables 1):**
1. Load daily managed-portfolio matrix `F` (T×H), trim via `dropna` (realized start
   1973-11-01, H=50). FF25: H=25.
2. **Market-orthogonalize** each column on the file's own `rme` with **full-sample β**
   (`demarket`; residual `F − β·rme`, intercept discarded).
3. **Devol**: rescale each column to the in-sample market σ.
4. **Moments**: `μ̄ = mean(F)` (daily); `Σ̄ = regcov(F)` = `a·(tr S/H)·I + (1−a)·S`,
   `a = H/(H+T)` (flat-Wishart identity shrink; daily T≫H ⇒ a≈0.005, Σ̄≈S).
5. **L2 estimator**: `b̂ = (Σ̄ + γI)⁻¹μ̄`, `γ = freq·tr(Σ̄)/(κ²T)` (the `freq=252`
   makes κ an *annualized* root-E[SR²]). SEs: `var(b̂) = (1/T)(Σ̄+γI)⁻¹` (eq. 23).
6. **κ-grid**: data-driven right edge via the 1%-coefficient-stabilize loop, then
   `logspace` down to 0.01, gridsize=100.
7. **CV**: K=3 contiguous folds; rescale fold penalty `lCV = l/(1−1/K)` to hold κ fixed;
   objective = CSR² (eq. 30) `1 − (μ̄₂−Σ̄₂b̂)'(μ̄₂−Σ̄₂b̂)/(μ̄₂'μ̄₂)` on the withheld fold.
8. Outputs: IS curve = train CSR², OOS curve = fold-averaged CSR² (±1 s.e. = std/√K).

**Fig 4a P&S η=1 curve:** re-solve with η=1 → `b̂ = (Σ̄+γΣ̄)⁻¹Σ̄μ̄` (pure level
shrinkage); the x-axis is re-mapped to E[SR²]^{1/2} under η=1 (it no longer equals κ,
paper fn 17). Expected peak < 0.05.

**Table 1:** at CV-optimal κ*, sort by |t|, top 10. Panel a = run with `rotate_PC=False`
(raw anomaly labels); Panel b = `rotate_PC=True` (PC labels; coefficients are exact
linear combinations of Panel a by rotation-invariance).

**Fig 3 / Fig 4b (dual penalty):** for each (κ → γ₂, γ₁) trace the LARS-EN path; bin
CV-OOS-R² into (κ, #non-zero) cells (Fig 3); take the upper ridge per #factors (Fig 4b).
Run in raw space (Panel a) and PC space (Panel b).

**Table 4 (OOS-MVE α):** restore the 2005 wall; re-orthogonalize portfolios **and** the
5 non-market FF6 factors with **pre-2005 β**; devol on pre-2005 σ; choose κ by CV on
pre-2005 only; estimate `b̂` on full pre-2005; form `P_t = b̂'F_t` for 2005–2017; scale
to OOS market σ; regress on each benchmark's MVE return (CAPM = zero-mean leg; FF6 =
unregularized pre-2005 MVE of the 5 factors; char-sparse / PC-sparse = dual penalty
restricted to 5 factors). α annualized = `α_daily × 252 × 100`, **OLS intercept-only
SEs** (the identical 5.26 CAPM SE across rows is the structural check).

---

## 5. Dependency triage (A–E)

All scoped inputs are **class A** (in repo): the 50-anomaly managed portfolios (daily +
monthly), FF25 (daily + monthly), and the FF3/FF5/Momentum factor files. No WRDS pull is
required. See `data_manifest.md`. Descoped Tier-3 inputs are class B (Kozak download) /
class D (WRDS rebuild from `wrdsapps.firm_ratio` + `crsp.dsf`).

---

## 6. Success criterion + tolerance

A "main results reproduce" verdict (qualitative + quantitatively close, **not**
bit-exact) requires:
- **Fig 4a**: OOS-CSR² peak κ ∈ [0.20, 0.45], peak height within ±0.05 absolute of 0.24;
  hump-shaped curve; IS above OOS; P&S η=1 peak < 0.05. (**Fig 2a**: peak κ ∈ [0.15,
  0.35], height ∈ [0.38, 0.50].)
- **Table 1**: lead factor = Industry Rel. Rev. (L.V.) with negative b and largest |t|;
  ≥7/10 top-factor overlap; signs match. PC4/PC1/PC2/PC9 among the PC top set.
- **Table 4 (50-anom)**: each α same sign (positive); within ±25% relative or ±3%
  absolute (whichever larger); CAPM and FF6 stay > 2 s.e.; ordering CAPM ≥ FF6 ≥
  char-sparse > PC-sparse.
- **Fig 3 / Fig 4b**: qualitative — raw contour ridge near-vertical (L2 helps, sparsity
  doesn't); PC achieves high R² at low #non-zero; PC sparsity curve dominates the
  characteristics curve (~2/3 max with ≤2 PCs, near-max with ~10).

---

## 7. Abandon conditions (for the implementation committee)

Stop and report failure if, **after** pinning all assumptions in `open_questions.md`:
1. **Sign failure**: Fig 4a OOS-CSR² non-positive at its peak, or Table 1a lead factor
   wrong sign — after confirming demarket/devol/daily-freq/sample window.
2. **Peak collapse**: Fig 4a/2a peak κ off by more than one full decade (indicates the
   κ↔penalty map is mis-specified and irreconcilable).
3. **Table 4 inversion**: 50-anom CAPM or FF6 α negative or insignificant (< 2 s.e.)
   after a correct pre-2005/post-2005 split — directly contradicts the central claim.
4. **Dual-penalty failure**: LARS-EN does not reproduce the char-vs-PC crossing
   (Fig 4b) after verifying the Σ⁻¹ HJ objective and the no-centering requirement.

A tolerance merely *missed* (not sign-inverted) after assumptions are pinned → report as
"reproduced with deviation," not abandon. Only sign / decade-scale failures are kill
switches.

---

## 8. Handoff

Machine-consumable artifacts (lead with a strict JSON block validating against the skill
schema): `data_manifest.md`, `target_results.md`, `task_graph.md`. Pinned assumptions and
the user-routed decisions are in `open_questions.md`. The downstream
`/replication-execute` skill can consume these directly.
