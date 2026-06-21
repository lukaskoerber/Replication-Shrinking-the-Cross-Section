"""Packaging / engine-reuse / no-legacy tests (AC6, AC7)."""

import os
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_engine_reachable_only_as_package():
    """The engine is a single source of truth under kns.* (AC7)."""
    from kns import core, dual_penalty
    from kns import fit_scs, SCSResult
    assert hasattr(core, "l2est") and hasattr(core, "regcov")
    assert hasattr(dual_penalty, "enet_sdf")
    assert callable(fit_scs) and SCSResult is not None


def test_legacy_modules_are_gone():
    """The redundant top-level scripts were retired into the package (AC7)."""
    for legacy in ("SCS_L2est.py", "utils.py", "cross_validate.py", "scs_main.py",
                   "load_managed_portfolios.py", "load_ff_anomalies.py", "load_ff25.py"):
        assert not os.path.exists(os.path.join(REPO_ROOT, legacy)), \
            f"legacy file {legacy} still present"


def test_no_dangling_legacy_imports():
    """No source file imports a deleted legacy module (AC7)."""
    bad_patterns = ("import scs_core", "import dual_penalty as dp\nimport",
                    "from cross_validate", "import load_managed_portfolios",
                    "from SCS_L2est", "from load_ff")
    src_dir = os.path.join(REPO_ROOT, "src")
    offenders = []
    for root, _dirs, files in os.walk(src_dir):
        for name in files:
            if not name.endswith(".py"):
                continue
            text = open(os.path.join(root, name)).read()
            if "import scs_core" in text or "from scs_core" in text:
                offenders.append(os.path.join(root, name))
    assert not offenders, f"dangling legacy imports in: {offenders}"


def test_dual_penalty_reduces_to_ridge_selfcheck():
    """Migrated from dual_penalty.__main__: gamma1=0 elastic-net == ridge."""
    from kns import core, dual_penalty as dp
    rng = np.random.default_rng(0)
    values = rng.standard_normal((400, 6)) * 0.01
    cov = core.regcov(values)
    mean = values.mean(axis=0)
    b_ridge, _ = core.l2est(cov, mean, 0.001, len(values))
    b_enet = dp.enet_sdf(cov, mean, gamma2=0.001, gamma1=0.0)
    assert np.max(np.abs(b_ridge - b_enet)) < 1e-9


def test_import_works_from_non_repo_dir():
    """Importing kns must not depend on cwd being the repo (AC6)."""
    code = "import kns; from kns import fit_scs; print(kns.__version__)"
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO_ROOT, "src"))
    out = subprocess.run([sys.executable, "-c", code], cwd="/tmp",
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "0.1.0" in out.stdout
