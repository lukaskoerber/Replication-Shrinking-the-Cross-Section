"""
kns.reproduce — one entry point that runs the whole paper replication.

`reproduce_all()` regenerates every figure and table of the Kozak, Nagel &
Santosh (2020) replication, in the correct dependency order. It is the modular
successor to the old top-level `scs_main.py`: a single call that drives the full
analysis and writes all artifacts to `replication/build/output/`.

Dependency DAG (encoded in `SECTIONS` below):

    foundation ──┬── fig4a_anom50      (50-anomaly L2 model selection)
                 ├── table1            (largest SDF factors, raw + PC)
                 ├── sr_objective      (cross-validated Sharpe diagnostic)
                 └── fig3_contour ──── fig4b_sparsity   (char-vs-PC sparsity)
    fig2a_ff25      (FF25 preliminary)        — independent
    table4_alpha    (OOS-MVE alpha, Table 4)  — independent

`foundation` recomputes the cached moments/CV (`foundation.npz`) from `Data/`
every run — it is an intra-run intermediate, never a shipped input. `fig3_contour`
likewise writes `fig3_grids.npz` consumed by `fig4b_sparsity`.
"""

# Section name -> (module, depends-on). Order here is a valid topological order.
SECTIONS = [
    ("foundation", "kns.reproduce._foundation", []),
    ("fig2a_ff25", "kns.reproduce.fig2a_ff25", []),
    ("fig4a_anom50", "kns.reproduce.fig4a_anom50", ["foundation"]),
    ("table1", "kns.reproduce.table1", ["foundation"]),
    ("fig3_contour", "kns.reproduce.fig3_contour", ["foundation"]),
    ("fig4b_sparsity", "kns.reproduce.fig4b_sparsity", ["fig3_contour"]),
    ("sr_objective", "kns.reproduce.sr_objective", ["foundation"]),
    ("table4_alpha", "kns.reproduce.table4_alpha", []),
]


def run_section(name):
    """Run one replication section by name (imports its module, calls `main`)."""
    import importlib

    match = [s for s in SECTIONS if s[0] == name]
    if not match:
        known = ", ".join(s[0] for s in SECTIONS)
        raise ValueError(f"unknown section {name!r}; choose from: {known}")
    module_path = match[0][1]
    module = importlib.import_module(module_path)
    return module.main()


def reproduce_all(sections=None, verbose=True):
    """
    Run the full replication (or a subset) in dependency order.

    Parameters
    ----------
    sections : list of str, optional
        Subset of section names to run (dependencies are NOT auto-added — pass a
        dependency-complete list). Defaults to all sections in topological order.
    verbose : bool, default True
        Print a progress line per section.

    Returns
    -------
    dict mapping section name -> the section's return value (if any).
    """
    to_run = [s[0] for s in SECTIONS] if sections is None else list(sections)
    results = {}
    for name in to_run:
        if verbose:
            print(f"[kns.reproduce] running section: {name}")
        results[name] = run_section(name)
    if verbose:
        print(f"[kns.reproduce] done ({len(to_run)} sections)")
    return results
