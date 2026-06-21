"""Behavioral + numerical acceptance tests for kns.fit_scs."""

import warnings

import numpy as np
import pandas as pd
import pytest

from kns import fit_scs, core
from kns import dual_penalty as dp
from kns.estimator import _shrunk_cov


# ---------------------------------------------------------------------------
# AC4 — generic fit & properties (fast, synthetic)
# ---------------------------------------------------------------------------
def test_fixed_kappa_equals_closed_form_ridge(synthetic_panel):
    df = synthetic_panel
    result = fit_scs(df, kappa=0.5)
    values = df.to_numpy()
    cov, _, _ = _shrunk_cov(values, "wishart")
    mean = values.mean(axis=0)
    penalty = core.kappa2pen(0.5, len(df), np.trace(cov), 252)
    b_closed_form = np.linalg.solve(cov + penalty * np.eye(values.shape[1]), mean)
    assert np.max(np.abs(result.b - b_closed_form)) < 1e-12


def test_covariance_is_psd_across_n_vs_t(synthetic_panel):
    rng = np.random.default_rng(1)
    for n_assets in (3, 5, 20, 40):
        values = rng.standard_normal((30, n_assets)) * 0.01
        cov, _, _ = _shrunk_cov(values, "wishart")
        assert (np.linalg.eigvalsh(cov) > 0).all()


def test_n_greater_than_t_warns_not_crashes():
    rng = np.random.default_rng(2)
    values = rng.standard_normal((8, 20)) * 0.01
    dates = pd.bdate_range("2000-01-03", periods=8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit_scs(values, dates=dates, kappa=1.0)
    assert np.isfinite(result.b).all()
    assert any("conditioned" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# AC2 — dual-penalty L1=0 reduces to the ridge
# ---------------------------------------------------------------------------
def test_dual_penalty_l1_zero_equals_ridge(synthetic_panel):
    df = synthetic_panel
    ridge = fit_scs(df, kappa=0.4)
    dual = fit_scs(df, kappa=0.4, dual_penalty=True, l1=0.0)
    assert np.max(np.abs(ridge.b - dual.b)) < 1e-9


# ---------------------------------------------------------------------------
# AC3 — SR objective dispatch
# ---------------------------------------------------------------------------
def test_sr_objective_returns_curve_and_weak_id_flag(synthetic_panel):
    result = fit_scs(synthetic_panel, objective="SR", kappa="cv")
    assert result.objective == "SR"
    assert result.cv_curve is not None
    assert "weakly_identified" in result.diagnostics


# ---------------------------------------------------------------------------
# Guardrails (hybrid: raise on unrecoverable, warn on quality)
# ---------------------------------------------------------------------------
def test_demarket_without_market_raises(synthetic_panel):
    with pytest.raises(ValueError, match="demarket"):
        fit_scs(synthetic_panel, demarket=True, market=None)


def test_zero_variance_column_raises(synthetic_panel):
    df = synthetic_panel.copy()
    df.iloc[:, 2] = 0.001                       # near-constant column
    with pytest.raises(ValueError, match="variance"):
        fit_scs(df, kappa=0.5)


def test_nan_input_raises(synthetic_panel):
    df = synthetic_panel.copy()
    df.iloc[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fit_scs(df, kappa=0.5)


def test_freq_mismatch_warns():
    rng = np.random.default_rng(3)
    monthly = pd.date_range("2000-01-31", periods=200, freq="ME")
    values = rng.standard_normal((200, 5)) * 0.04
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_scs(values, dates=monthly, kappa=0.5, freq=252)
    assert any("freq" in str(w.message) for w in caught)


def test_material_cov_shrink_warns():
    rng = np.random.default_rng(4)
    values = rng.standard_normal((60, 50)) * 0.01     # T just above H -> a ~ 0.45
    dates = pd.bdate_range("2000-01-03", periods=60)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_scs(values, dates=dates, kappa=1.0)
    assert any("Wishart" in str(w.message) or "shrink" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# predict / look-ahead
# ---------------------------------------------------------------------------
def test_predict_no_preprocessing_is_returns_times_b(synthetic_panel):
    result = fit_scs(synthetic_panel, kappa=0.5)
    portfolio = result.predict(synthetic_panel)
    expected = synthetic_panel.to_numpy() @ result.b
    assert np.allclose(portfolio, expected)


def test_predict_is_invariant_to_appended_future_rows(synthetic_panel):
    result = fit_scs(synthetic_panel, kappa=0.5)
    head = synthetic_panel.iloc[:300]
    longer = synthetic_panel.iloc[:500]
    assert np.max(np.abs(result.predict(head) - result.predict(longer)[:300])) < 1e-12


def test_sharpe_is_scale_invariant(synthetic_panel):
    result = fit_scs(synthetic_panel, kappa=0.5)
    base = result.sharpe(synthetic_panel)
    scaled = result.sharpe(synthetic_panel * 7.0)
    assert abs(base - scaled) < 1e-10


# ---------------------------------------------------------------------------
# AC1 — reduces-to-base-case / regression vs the committed foundation (slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_reduces_to_base_case_matches_foundation(anom50, foundation):
    dates, re, mkt, T = anom50
    result = fit_scs(re, market=mkt, dates=dates, demarket=True, devol=True,
                     freq=252, objective="CSR2", kappa="cv", kfold=3)
    kappa_committed = float(foundation["kappa_grid"][int(np.argmax(foundation["oos_csr2"]))])
    peak_committed = float(foundation["oos_csr2"].max())
    assert abs(result.kappa - kappa_committed) < 1e-9
    assert abs(np.nanmax(result.cv_curve["oos"]) - peak_committed) < 1e-9
