#!/usr/bin/env python3
"""
generate_table.py

Generate a summary table of area ratio results across all environments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid


# -------------------------
# Config
# -------------------------
CSV_FILES = {
    "ALE/Pong-v5": "area-ratio/returns_pong.csv",
    "ALE/Breakout-v5": "area-ratio/returns_breakout.csv",
    "ALE/SpaceInvaders-v5": "area-ratio/returns_space-invaders.csv",
}

MAX_STEP = 10_000_000
COARSE_BIN = 50_000
SMOOTH_WINDOW_BINS = 10
N_BOOT = 400
RNG_SEED = 123

SOURCES = ["SpaceInvaders-v5", "Pong-v5", "Breakout-v5"]
FILTER_SOURCE_SPARSITY = 60


# -------------------------
# Stats helpers
# -------------------------
def iqm_trimmed(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return np.nan
    if n < 4:
        return float(np.mean(values))
    v = np.sort(values)
    k = int(np.floor(0.25 * n))
    mid = v[k : n - k]
    return float(np.mean(mid)) if len(mid) else float(np.mean(v))


def bootstrap_iqm_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan

    center = iqm_trimmed(values)

    if n < 2:
        return center, np.nan, np.nan

    idx = rng.integers(0, n, size=(n_boot, n))
    samples = values[idx]
    samples.sort(axis=1)

    k = int(np.floor(0.25 * n))
    mid = samples[:, k : n - k] if (n - 2 * k) > 0 else samples
    stats = mid.mean(axis=1)

    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(center), float(lo), float(hi)


def smooth_group(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("step").copy()
    for c in ["iqm", "ci_lo", "ci_hi"]:
        g[f"{c}_s"] = g[c].rolling(SMOOTH_WINDOW_BINS, min_periods=1).mean()
    return g


def compute_area_ratios(csv_path: str, env_name: str) -> dict:
    """Compute area ratios for a given environment."""
    # Load
    df = pd.read_csv(csv_path, low_memory=False)

    # Cleanup
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["episodic_return"] = pd.to_numeric(df["episodic_return"], errors="coerce")
    df = df.dropna(subset=["step", "episodic_return"]).copy()
    df["step"] = df["step"].astype(int)

    # Selection
    mask = (
        ((df["project"] == "atari-lottery") & (df["seed"].isin([1,4,5])))
        | ((df["project"] == "atari-gmp") & (df["sparsity"].isin([75, 90])) & (df["seed"].isin([1,3,4,5])))
        | (
            (df["project"] == "lth-transfer")
            & (df["source_sparsity"] == 60)
            & (df["pruning"].isin(["dense"]))
            & (df["source"].isin(SOURCES))
        )
    )

    df = df[mask].copy()

    if FILTER_SOURCE_SPARSITY is not None:
        is_transfer = df["project"].eq("lth-transfer")
        df = pd.concat(
            [
                df[~is_transfer],
                df[is_transfer & df["source_sparsity"].eq(FILTER_SOURCE_SPARSITY)],
            ],
            ignore_index=True,
        )

    # Limit steps + binning
    df = df[df["step"] <= MAX_STEP].copy()
    df["coarse_step"] = (df["step"] // COARSE_BIN) * COARSE_BIN

    # Aggregate within run/bin
    per_run = (
        df.groupby(
            ["project", "run_name", "seed", "coarse_step", "sparsity", "source", "pruning", "source_sparsity"],
            dropna=False,
            as_index=False,
        )["episodic_return"]
        .mean()
    )

    rng = np.random.default_rng(RNG_SEED)

    # Compute IQM + CI per step/bin
    rows = []
    iqm_group_cols = ["project", "coarse_step", "sparsity", "source", "pruning", "source_sparsity"]
    curve_cols = ["project", "sparsity", "source", "pruning", "source_sparsity"]

    for keys, grp in per_run.groupby(iqm_group_cols, dropna=False):
        project, coarse_step, sparsity, source, pruning, source_sparsity = keys
        vals = grp["episodic_return"].to_numpy(dtype=float)

        center, ci_lo, ci_hi = bootstrap_iqm_ci(vals, n_boot=N_BOOT, rng=rng)

        rows.append(
            {
                "project": project,
                "step": int(coarse_step),
                "iqm": center,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "sparsity": sparsity,
                "source": source,
                "pruning": pruning,
                "source_sparsity": source_sparsity,
                "n_runs": int(len(vals)),
            }
        )

    curves = (
        pd.DataFrame(rows)
        .dropna(subset=["iqm"])
        .sort_values(["project", "source", "pruning", "source_sparsity", "sparsity", "step"])
    )

    # Smooth
    curves = curves.groupby(curve_cols, dropna=False, group_keys=False).apply(smooth_group)

    # Compute area ratios
    ratios = {}
    
    # Get scratch baseline
    scratch = curves[curves["project"] == "atari-lottery"].copy()
    if scratch.empty:
        return ratios
    
    scratch_data = scratch.dropna(subset=["iqm_s"]).sort_values("step")
    if len(scratch_data) < 2:
        return ratios
    
    x_scratch = scratch_data["step"].to_numpy(dtype=float)
    y_scratch = scratch_data["iqm_s"].to_numpy(dtype=float)
    area_scratch = trapezoid(y_scratch, x_scratch)
    
    # Compute transfer areas
    transfer = curves[curves["project"] == "lth-transfer"].copy()
    
    for source in sorted(transfer["source"].unique()):
        group_data = transfer[transfer["source"] == source].dropna(subset=["iqm_s"]).sort_values("step")
        if len(group_data) < 2:
            continue
        
        x_transfer = group_data["step"].to_numpy(dtype=float)
        y_transfer = group_data["iqm_s"].to_numpy(dtype=float)
        area_transfer = trapezoid(y_transfer, x_transfer)
        
        ratio = (area_transfer - area_scratch) / area_scratch
        source_short = source.split("-")[0]
        
        ratios[source_short] = ratio
    
    return ratios


def main() -> None:
    """Generate summary table of area ratios."""
    all_results = {}
    
    # Collect results from all environments
    for env_name, csv_path in CSV_FILES.items():
        print(f"Processing {env_name}...")
        ratios = compute_area_ratios(csv_path, env_name)
        all_results[env_name] = ratios
    
    # Create table
    table_data = []
    
    for env_name in ["ALE/Pong-v5", "ALE/Breakout-v5", "ALE/SpaceInvaders-v5"]:
        ratios = all_results.get(env_name, {})
        
        for source in ["Pong", "Breakout", "SpaceInvaders"]:
            if source in ratios:
                ratio = ratios[source]
                table_data.append({
                    "Target Environment": env_name,
                    "Transfer Source": source,
                    "Area Ratio": ratio,
                    "Percentage": f"{ratio*100:.2f}%"
                })
    
    # Create DataFrame
    df_table = pd.DataFrame(table_data)
    
    # Display
    print("\n" + "="*80)
    print("AREA RATIO SUMMARY TABLE")
    print("="*80)
    print(df_table.to_string(index=False))
    print("="*80 + "\n")
    
    # Save as CSV
    csv_out = "area-ratio/area_ratio_summary.csv"
    df_table.to_csv(csv_out, index=False)
    print(f"Saved CSV: {csv_out}")
    
    # Create visualization table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("tight")
    ax.axis("off")
    
    # Prepare data for table visualization
    display_data = []
    for _, row in df_table.iterrows():
        env_short = row["Target Environment"].split("/")[-1]
        display_data.append([
            env_short,
            row["Transfer Source"],
            f"{row['Area Ratio']:.4f}",
            row["Percentage"]
        ])
    
    table = ax.table(
        cellText=display_data,
        colLabels=["Target Environment", "Transfer Source", "Area Ratio", "Percentage"],
        cellLoc="center",
        loc="center",
        colWidths=[0.3, 0.25, 0.2, 0.2]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    # Alternate row colors
    for i in range(1, len(display_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")
            else:
                table[(i, j)].set_facecolor("white")
    
    plt.title("Area Ratio Summary: Transfer vs Scratch Baseline", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    
    fig_out = "area-ratio/area_ratio_table.png"
    plt.savefig(fig_out, dpi=400, bbox_inches="tight")
    print(f"Saved figure: {fig_out}")
    plt.show()


if __name__ == "__main__":
    main()
