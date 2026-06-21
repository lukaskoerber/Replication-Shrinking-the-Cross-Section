"""Shared fixtures for the kns test suite."""

import os

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ANOM_CSV = os.path.join(REPO_ROOT, "Data", "Instruments",
                        "managed_portfolios_anom_d_50.csv")
FOUNDATION_NPZ = os.path.join(REPO_ROOT, "replication", "build", "output",
                              "foundation.npz")


@pytest.fixture
def synthetic_panel():
    """A small, fast, deterministic return panel (T=600, N=5), no preprocessing
    needed — used for property/guardrail tests that must run in <1s."""
    rng = np.random.default_rng(0)
    values = rng.standard_normal((600, 5)) * 0.01 + 0.0002
    dates = pd.bdate_range("2000-01-03", periods=600)
    columns = [f"f{i}" for i in range(5)]
    return pd.DataFrame(values, index=dates, columns=columns)


@pytest.fixture
def anom50():
    """The real 50-anomaly daily panel (skips if Data/ is absent)."""
    if not os.path.exists(ANOM_CSV):
        pytest.skip("Data/ not available (repo tree required)")
    from kns import core
    dates, re, mkt, T = core.load_anom50_daily(ANOM_CSV)
    return dates, re, mkt, T


@pytest.fixture
def foundation():
    """The committed foundation.npz regression baseline (skips if absent)."""
    if not os.path.exists(FOUNDATION_NPZ):
        pytest.skip("foundation.npz not available")
    return np.load(FOUNDATION_NPZ, allow_pickle=True)
