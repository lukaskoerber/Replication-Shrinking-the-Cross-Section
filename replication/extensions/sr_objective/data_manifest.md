```json
{
  "paper": "Kozak, Nagel & Santosh (2020 JFE) — Shrinking the Cross-Section [extension: cross-validated Sharpe-ratio objective]",
  "variables": [
    {
      "name": "anom50_returns_dm_dv",
      "source_class": "A",
      "source": {
        "kind": "public",
        "wrds_library": null,
        "table": null,
        "fields": ["returns_dm_dv", "dates"],
        "public_ref": "in-repo derived artifact: replication/build/output/foundation.npz (key returns_dm_dv, 11141x50; key dates)",
        "subscription_tier": null
      },
      "frequency": "daily",
      "date_range": { "start": "1973-11-01", "end": "2017-12-29" },
      "universe_filters": ["50 anomaly long-short managed portfolios", "H=50", "full-sample market-orthogonalized then devol'd to market sigma"],
      "identifier": "date",
      "transformation": "REUSED as-is from the completed replication: de-marketed (full-sample beta) and devol'd (full-sample market sigma) 50-anomaly daily returns produced by foundation_check.py. The SR objective consumes the SAME preprocessed matrix as cross_validate_csr2 [A-SR13], inheriting the identical (shared) full-sample preprocessing.",
      "note": "No new data. The SR add-on is a pure estimation-layer extension over the existing foundation."
    },
    {
      "name": "kappa_penalty_grid",
      "source_class": "A",
      "source": {
        "kind": "public",
        "wrds_library": null,
        "table": null,
        "fields": ["kappa_grid", "penalty_list", "freq", "kfold"],
        "public_ref": "in-repo derived artifact: replication/build/output/foundation.npz (keys kappa_grid 100, penalty_list 100, freq=252, kfold=3)",
        "subscription_tier": null
      },
      "frequency": "static",
      "date_range": { "start": null, "end": null },
      "universe_filters": ["gridsize=100", "data-driven right edge (1% coefficient-stabilize)"],
      "identifier": "kappa grid index",
      "transformation": "REUSED as-is: the same kappa grid and penalty list the L2/CSR2 path uses, so the SR objective is selected over an identical grid [A-SR3]. The CV penalty rescale l_used=l/(1-1/kfold) [A-SR4/A-C6] is applied inside the fold loop.",
      "note": "Shared grid guarantees the SR-selected and CSR2-selected kappa are comparable."
    }
  ]
}
```

## Notes

This extension adds **no new data**. Both inputs are class-A derived artifacts already
produced and frozen by the completed main replication (`foundation.npz`). The SR objective
is a pure estimation-layer add-on: it reuses the de-marketed/devol'd return matrix, the κ
grid, the contiguous K=3 folds, and the L2 estimator unchanged, and only adds a new
objective function `cross_validate_sr` plus a selectable dispatcher. Consuming the **same**
`returns_dm_dv` as `cross_validate_csr2` is deliberate [A-SR13]: it preserves comparability
between the two objectives and means the SR path inherits exactly the same (pre-existing,
shared) full-sample-preprocessing look-ahead — acceptable because the SR objective is a
**reported diagnostic only** and does not drive any downstream/production model [A-SR15].
