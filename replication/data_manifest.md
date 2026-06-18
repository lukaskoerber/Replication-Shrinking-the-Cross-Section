```json
{
  "paper": "Kozak, Nagel & Santosh (2020 JFE) — Shrinking the Cross-Section",
  "variables": [
    {
      "name": "anom50_managed_daily",
      "source_class": "A",
      "source": {
        "kind": "public",
        "wrds_library": null,
        "table": null,
        "fields": ["date", "rme", "50 anomaly long-short managed-portfolio columns"],
        "public_ref": "https://www.serhiykozak.com/data (in repo: Data/Instruments/managed_portfolios_anom_d_50.csv)",
        "subscription_tier": null
      },
      "frequency": "daily",
      "date_range": { "start": "1973-11-01", "end": "2017-12-29" },
      "universe_filters": [
        "microcap screen <0.01% aggregate ME pre-applied",
        "drop cols with prefixes rX_,r2_,r3_",
        "dropna -> realized start 1973-11-01",
        "H=50 anomaly columns"
      ],
      "identifier": "date",
      "transformation": "raw long-short managed portfolio returns F=Z'R; rank-transform eqs 31-33 pre-applied by author; market-orthogonalization (demarket) and devol applied in-code, NOT pre-applied",
      "note": "rme market column present in file; re_ew column dropped. Primary covariance/estimation input. VERIFIED: anomaly columns are raw (r_value beta=-0.38 on rme), realized start 1973-11-01, H=50."
    },
    {
      "name": "anom50_managed_monthly",
      "source_class": "A",
      "source": {
        "kind": "public",
        "wrds_library": null,
        "table": null,
        "fields": ["date", "rme", "50 anomaly long-short managed-portfolio columns"],
        "public_ref": "https://www.serhiykozak.com/data (in repo: Data/Instruments/managed_portfolios_anom_50.csv)",
        "subscription_tier": null
      },
      "frequency": "monthly",
      "date_range": { "start": "1963-07", "end": "2017-11" },
      "universe_filters": [
        "microcap screen <0.01% aggregate ME pre-applied",
        "drop cols with prefixes rX_,r2_,r3_",
        "H=50 anomaly columns"
      ],
      "identifier": "date",
      "transformation": "raw long-short managed portfolio returns F=Z'R; rank-transform eqs 31-33 pre-applied by author",
      "note": "Monthly counterpart of anom50_managed_daily; used for plotting/checks only — daily is used for covariance (paper estimates Sigma from daily for precision)."
    },
    {
      "name": "market_rme",
      "source_class": "A",
      "source": {
        "kind": "public",
        "wrds_library": null,
        "table": null,
        "fields": ["rme"],
        "public_ref": "https://www.serhiykozak.com/data (rme column inside the anom managed-portfolio files)",
        "subscription_tier": null
      },
      "frequency": "daily",
      "date_range": { "start": "1973-11-01", "end": "2017-12-29" },
      "universe_filters": ["aligned to anom50_managed_daily index"],
      "identifier": "date",
      "transformation": "file's own CRSP VW market excess return; used for in-code market-orthogonalization (demarket) and as devol target; NOT Ken French Mkt-RF",
      "note": "Resides inside the anom files; broken out as a stable name because demarket/devol depend on it."
    },
    {
      "name": "ff25_daily",
      "source_class": "A",
      "source": {
        "kind": "public",
        "wrds_library": null,
        "table": null,
        "fields": ["date", "25 ME/BM 5x5 VW portfolio columns"],
        "public_ref": "Ken French Data Library (in repo: Data/25_Portfolios_5x5_Daily_average_value_weighted_returns_daily.csv)",
        "subscription_tier": null
      },
      "frequency": "daily",
      "date_range": { "start": "1926-07-01", "end": "2017-12-29" },
      "universe_filters": ["25 size x book-to-market value-weighted portfolios"],
      "identifier": "date",
      "transformation": "/100 to decimal; subtract RF for excess; market-orthogonalize (full-sample beta); rescale to in-sample market sigma",
      "note": "FF25 preliminary branch (Fig 2a)."
    },
    {
      "name": "ff25_monthly",
      "source_class": "A",
      "source": {
        "kind": "public",
        "wrds_library": null,
        "table": null,
        "fields": ["date", "25 ME/BM 5x5 VW portfolio columns"],
        "public_ref": "Ken French Data Library (in repo: Data/25_Portfolios_5x5_average_value_weighted_returns_monthly.csv)",
        "subscription_tier": null
      },
      "frequency": "monthly",
      "date_range": { "start": "1926-07", "end": "2017-12" },
      "universe_filters": ["25 size x book-to-market value-weighted portfolios"],
      "identifier": "date",
      "transformation": "/100 to decimal; subtract RF for excess",
      "note": "Monthly counterpart; plotting/checks only."
    },
    {
      "name": "ff5_factors_daily",
      "source_class": "A",
      "source": {
        "kind": "public",
        "wrds_library": null,
        "table": null,
        "fields": ["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
        "public_ref": "Ken French Data Library (in repo: Data/F-F_Research_Data_5_Factors_2x3_daily.csv)",
        "subscription_tier": null
      },
      "frequency": "daily",
      "date_range": { "start": "1963-07-01", "end": "2017-12-29" },
      "universe_filters": [],
      "identifier": "date",
      "transformation": "/100; SMB,HML,RMW,CMA orthogonalized to market via pre-2005 beta for Table 4 FF6 benchmark",
      "note": "Supplies 4 of the 5 non-market FF6 factors and RF."
    },
    {
      "name": "ff_momentum_daily",
      "source_class": "A",
      "source": {
        "kind": "public",
        "wrds_library": null,
        "table": null,
        "fields": ["date", "Mom", "UMD"],
        "public_ref": "Ken French Data Library (in repo: Data/F-F_Momentum_Factor_daily.csv)",
        "subscription_tier": null
      },
      "frequency": "daily",
      "date_range": { "start": "1926-11-03", "end": "2017-12-29" },
      "universe_filters": [],
      "identifier": "date",
      "transformation": "/100; the 6th FF factor (momentum), market-orthogonalized via pre-2005 beta for the Table 4 FF6 benchmark",
      "note": "Completes the FF6 factor set for Table 4."
    },
    {
      "name": "ff3_factors_daily",
      "source_class": "A",
      "source": {
        "kind": "public",
        "wrds_library": null,
        "table": null,
        "fields": ["date", "Mkt-RF", "SMB", "HML", "RF"],
        "public_ref": "Ken French Data Library (in repo: Data/F-F_Research_Data_Factors_daily.csv)",
        "subscription_tier": null
      },
      "frequency": "daily",
      "date_range": { "start": "1926-07-01", "end": "2017-12-29" },
      "universe_filters": [],
      "identifier": "date",
      "transformation": "/100",
      "note": "Provides Mkt-RF for the CAPM benchmark leg and RF for excess-return construction."
    }
  ]
}
```

## Notes (human-readable)

All eight inputs are **class A** — already present in the repository as CSVs, sourced
from Kozak's website (managed portfolios) and Ken French's Data Library (factors).
No WRDS pull is required for the scoped (Tier 1+2) replication.

**Key data facts verified during planning:**
- The anomaly managed-portfolio columns are **raw** long-short returns (not
  market-orthogonalized): `r_value` has β = −0.38 on the file's `rme`, `r_prof` β =
  +0.22. Orthogonalization is applied in-code via `demarket()`. This is what makes the
  Table 4 pre-2005-β OOS test feasible from in-hand data.
- After the loader's `dropna`, the realized 50-anomaly daily sample starts **1973-11-01**
  and contains exactly **50** anomaly columns — both match the paper (Sec 4.2.1).
- Use the file's own `rme` for orthogonalization/devol; use Ken French `Mkt-RF` **only**
  as the CAPM benchmark leg in Table 4. Managed files are in decimal; FF CSVs are in
  percent (divide by 100).

**Descoped (Tier 3, not in this manifest):** WFR-80 managed portfolios and the
1,375/3,400 interaction sets. These are not in the repo; they are class B (download
from serhiykozak.com/data) with a class-D WRDS rebuild fallback. Documented as a future
extension in `plan.md`.
