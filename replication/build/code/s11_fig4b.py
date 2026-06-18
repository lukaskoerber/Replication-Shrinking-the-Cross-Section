"""
s11 — Figure 4b: result_id `fig4b_char_vs_pc`.

Kozak, Nagel & Santosh (2020), "Shrinking the Cross-Section", JFE 135(2).
Figure 4b: for each number of factors n = 1..H,

    maxOOS(n) = max over kappa of [ CV-OOS-CSR^2 (eq.30) achievable with EXACTLY
                                    n non-zero SDF coefficients ],

i.e. the upper ridge of the Fig-3 contour swept bottom-to-top.  Two curves:
characteristics basis (raw, solid) and PC basis (dashed).  A -1 s.e. dotted bound
uses the CV s.e. of the winning point at each n.

The per-cell OOS-CSR^2 grids (#nonzero x kappa) and their s.e. are produced by
s10 and cached in fig3_grids.npz; we reuse them (run s10 first).  maxOOS(n) =
row-wise max over kappa of the grid; the -1 s.e. curve subtracts the s.e. of the
winning cell.

Reported (Fig 4b): PC curve rises fast (>= ~2/3 max with <= 2 PCs, near-max ~10
PCs); characteristics curve stays low until n -> H; first PCs selected are
PC1, PC4, PC2, PC9.
"""
import os
import json

import numpy as np

os.environ.setdefault("MPLCONFIGDIR",
                      os.path.join(os.path.dirname(__file__), "..", "build", ".mplcache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "replication", "build", "output")
GRIDS = os.path.join(OUT, "fig3_grids.npz")


def ridge_curve(grid, grid_se):
    """Upper ridge maxOOS(n) = max over kappa per #nonzero row, + winning s.e.

    Forward-fills the running max bottom-to-top so the curve is the BEST OOS-CSR^2
    achievable with AT MOST n nonzeros is non-decreasing only if we cumulate; here
    we report the per-n max (exactly-n cells), then also a cumulative envelope used
    for the booleans (achievable with <= n).  Returns both.
    """
    H = grid.shape[0]
    per_n = np.full(H, np.nan)
    per_n_se = np.full(H, np.nan)
    for r in range(H):
        row = grid[r, :]
        if np.all(np.isnan(row)):
            continue
        c = int(np.nanargmax(row))
        per_n[r] = row[c]
        per_n_se[r] = grid_se[r, c]
    # cumulative envelope (best achievable with <= n nonzeros)
    env = np.full(H, np.nan)
    env_se = np.full(H, np.nan)
    run = -np.inf
    run_se = np.nan
    for r in range(H):
        if not np.isnan(per_n[r]) and per_n[r] > run:
            run = per_n[r]
            run_se = per_n_se[r]
        if np.isfinite(run):
            env[r] = run
            env_se[r] = run_se
    return per_n, per_n_se, env, env_se


def main():
    g = np.load(GRIDS, allow_pickle=True)
    grid_raw = g["grid_raw"]; se_raw = g["se_raw"]
    grid_pc = g["grid_pc"]; se_pc = g["se_pc"]
    H = grid_raw.shape[0]
    nfac = np.arange(1, H + 1)

    char_n, char_n_se, char_env, char_env_se = ridge_curve(grid_raw, se_raw)
    pc_n, pc_n_se, pc_env, pc_env_se = ridge_curve(grid_pc, se_pc)

    # Use the cumulative envelope as the Fig-4b curve (best with <= n factors),
    # matching the paper's "upper ridge swept bottom-to-top" reading.
    char = char_env
    pc = pc_env
    pc_se = pc_env_se

    # --- Plot -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(nfac, char, "-", color="tab:blue", lw=2.0, label="Characteristics (raw)")
    ax.plot(nfac, pc, "--", color="tab:red", lw=2.0, label="PCs")
    ax.plot(nfac, pc - pc_se, ":", color="tab:red", lw=1.0, label="PC $-1$ s.e.")
    ax.set_xscale("log")
    ax.set_xlabel("# factors (nonzero SDF coefficients)", fontweight="bold")
    ax.set_ylabel("OOS cross-sectional $R^2$", fontweight="bold")
    ax.set_title("Fig 4b — characteristics vs PC sparse SDF")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    png = os.path.join(OUT, "fig4b_char_vs_pc.png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Curve values + booleans (JSON) ---------------------------------
    ns = [1, 2, 4, 10, 25, 50]

    def at(curve, n):
        return None if np.isnan(curve[n - 1]) else float(curve[n - 1])

    char_max = float(np.nanmax(char))
    pc_max = float(np.nanmax(pc))

    pc_at2 = at(pc, 2); pc_at10 = at(pc, 10)
    char_at10 = at(char, 10); char_at50 = at(char, 50)

    pc_reaches_two_thirds_max_by_n2 = (pc_at2 is not None
                                       and pc_at2 >= (2.0 / 3.0) * pc_max)
    pc_near_max_by_n10 = (pc_at10 is not None and pc_at10 >= 0.9 * pc_max)
    char_low_until_high_n = (char_at10 is not None and char_at50 is not None
                             and char_at10 < 0.9 * char_at50)

    payload = {
        "result_id": "fig4b_char_vs_pc",
        "char_curve": {str(n): at(char, n) for n in ns},
        "pc_curve": {str(n): at(pc, n) for n in ns},
        "char_max": char_max,
        "pc_max": pc_max,
        "pc_at2_over_max": (None if (pc_at2 is None or pc_max == 0)
                            else round(pc_at2 / pc_max, 4)),
        "pc_at10_over_max": (None if (pc_at10 is None or pc_max == 0)
                             else round(pc_at10 / pc_max, 4)),
        "pc_reaches_two_thirds_max_by_n2": bool(pc_reaches_two_thirds_max_by_n2),
        "pc_near_max_by_n10": bool(pc_near_max_by_n10),
        "char_low_until_high_n": bool(char_low_until_high_n),
    }
    jpath = os.path.join(OUT, "fig4b_char_vs_pc.json")
    with open(jpath, "w") as fh:
        json.dump(payload, fh, indent=2)

    print("char curve:", {n: at(char, n) for n in ns})
    print("pc   curve:", {n: at(pc, n) for n in ns})
    print(f"char_max={char_max:.4f}  pc_max={pc_max:.4f}")
    print(f"pc_reaches_two_thirds_max_by_n2={pc_reaches_two_thirds_max_by_n2} "
          f"(pc@2={pc_at2})")
    print(f"pc_near_max_by_n10={pc_near_max_by_n10} (pc@10={pc_at10})")
    print(f"char_low_until_high_n={char_low_until_high_n} "
          f"(char@10={char_at10}, char@50={char_at50})")
    print(f"wrote {png}\nwrote {jpath}")


if __name__ == "__main__":
    main()
