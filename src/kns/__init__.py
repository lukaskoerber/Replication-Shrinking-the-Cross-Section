"""
kns — Shrinking the Cross-Section (Kozak, Nagel & Santosh, 2020, JFE).

Public API
----------
fit_scs(returns, ...) -> SCSResult
    Fit the SCS stochastic-discount-factor estimator to a panel of return data.
SCSResult
    The fitted-model result object (coefficients, MVE weights, diagnostics).

The numerical engine lives in `kns.core` (moments, ridge eq. 22/23, the
kappa<->penalty map, the data-driven grid, contiguous cross-validation, the
CSR2/SR objectives) and `kns.dual_penalty` (the elastic-net eq. 28). The
estimator is a thin, validated wrapper over that engine — single source of truth.

`kns.reproduce` regenerates every figure/table of the paper replication; run it
with `python -m kns.reproduce` or `kns.reproduce.reproduce_all()`.
"""

from kns.estimator import fit_scs, SCSResult

__version__ = "0.1.0"

__all__ = ["fit_scs", "SCSResult", "__version__"]
