#!/usr/bin/env python3
"""
compare-runs.py

Plot IQM (interquartile mean) of episodic return for ALE/SpaceInvaders-v5,
with a 95% bootstrap confidence interval band.

This script compares LTH transfer with:
- pruning = dense  (labeled "no pruning")
- pruning = gmp    (labeled "GMP (90%)")

for sources:
- Pong-v5
- Breakout-v5

It expects a CSV like ../data/returns_space-invaders.csv with at least:
  project, run_name, seed, step, episodic_return, source, pruning, source_sparsity, sparsity
(Columns not used by your filters are OK to exist.)

Notes:
- We bin steps into 50k-step bins, aggregate episodic_return within each run+bin by mean,
  then compute IQM across runs at each bin.
- 95% CI is computed by bootstrapping runs at each bin.
- For readability we smooth IQM and CI bounds with a rolling mean across bins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid


# -------------------------
# Config
# -------------------------
CSV_PATH = "area-ratio/returns_space-invaders.csv"

MAX_STEP = 10_000_000
COARSE_BIN = 50_000          # bin size in environment steps
SMOOTH_WINDOW_BINS = 10      # rolling window over bins (~500k steps)
N_BOOT = 400                 # bootstrap replicates for CI
RNG_SEED = 123               # reproducible CI

# Which transfer sources to include
SOURCES = ["SpaceInvaders-v5", "Pong-v5", "Breakout-v5"]

# If you want to restrict to a specific source sparsity (as in your earlier script)
FILTER_SOURCE_SPARSITY = 60  # set to None to disable


# -------------------------
# Stats helpers
# -------------------------
def iqm_trimmed(values: np.ndarray) -> float:
    """
    Interquartile mean via symmetric trimming of 25% from each side.
    For small n (<4), falls back to mean.
    """
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
    """
    Bootstrap 95% CI for IQM over runs.
    Returns: (center, ci_low, ci_high)
    """
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan

    center = iqm_trimmed(values)

    # Not enough runs to bootstrap meaningfully
    if n < 2:
        return center, np.nan, np.nan

    # Vectorized bootstrap:
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


def compute_area_ratios(curves_smoothed: pd.DataFrame) -> dict:
    """
    Compute area ratios: (area_transfer - area_scratch) / area_scratch
    for each transfer condition vs scratch baseline.
    """
    ratios = {}
    
    # Get scratch (dense baseline) area
    scratch = curves_smoothed[curves_smoothed["project"] == "atari-lottery"].copy()
    if scratch.empty:
        print("No scratch baseline found")
        return ratios
    
    scratch_data = scratch.dropna(subset=["iqm_s"]).sort_values("step")
    if len(scratch_data) < 2:
        print("Not enough scratch data points")
        return ratios
    
    x_scratch = scratch_data["step"].to_numpy(dtype=float)
    y_scratch = scratch_data["iqm_s"].to_numpy(dtype=float)
    area_scratch = trapezoid(y_scratch, x_scratch)
    
    print(f"Scratch (dense) area under curve: {area_scratch:.2f}")
    print()
    
    # Compute transfer areas
    transfer = curves_smoothed[curves_smoothed["project"] == "lth-transfer"].copy()
    
    for (source, pruning), group in transfer.groupby(["source", "pruning"]):
        group_data = group.dropna(subset=["iqm_s"]).sort_values("step")
        if len(group_data) < 2:
            continue
        
        x_transfer = group_data["step"].to_numpy(dtype=float)
        y_transfer = group_data["iqm_s"].to_numpy(dtype=float)
        area_transfer = trapezoid(y_transfer, x_transfer)
        
        # Compute ratio: (area_transfer - area_scratch) / area_scratch
        ratio = (area_transfer - area_scratch) / area_scratch
        
        label = f"{source} ({pruning})"
        ratios[label] = {
            "area_transfer": area_transfer,
            "area_scratch": area_scratch,
            "ratio": ratio,
        }
        
        print(f"{label}:")
        print(f"  Transfer area: {area_transfer:.2f}")
        print(f"  Scratch area:  {area_scratch:.2f}")
        print(f"  Ratio:         {ratio:.4f} ({ratio*100:+.2f}%)")
        print()
    
    return ratios


# -------------------------
# Main
# -------------------------
def main() -> None:
    # Load
    df = pd.read_csv(CSV_PATH, low_memory=False)

    # Cleanup
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["episodic_return"] = pd.to_numeric(df["episodic_return"], errors="coerce")
    df = df.dropna(subset=["step", "episodic_return"]).copy()
    df["step"] = df["step"].astype(int)

    # Selection: Dense baseline, GMP baseline, LTH transfer dense/gmp from Pong/Breakout
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
        # Only applies to transfer rows; keep baselines unchanged
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

    # Compute IQM + CI per step/bin for each curve variant
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

    # Smooth IQM + CI bounds along step for each curve
    curves = curves.groupby(curve_cols, dropna=False, group_keys=False).apply(smooth_group)

    # Compute area ratios
    print("=" * 60)
    print("AREA RATIO ANALYSIS: (Transfer - Scratch) / Scratch")
    print("=" * 60)
    print()
    ratios = compute_area_ratios(curves)
    print("=" * 60)
    print()

    # Plot
    plt.figure(figsize=(11, 6.2))

    def label(row: pd.Series) -> str:
        # You asked specifically:
        # - "LTH transfer" -> "no pruning"
        # - "GMP transfer" -> "GMP (90%)"
        # We have two transfer "pruning" types and two sources.
        if row["project"] == "lth-transfer":
            if row["pruning"] == "dense":
                base = "no pruning"
            elif row["pruning"] == "gmp":
                base = "GMP (90%)"
            else:
                base = row["pruning"]
            # keep source visible so you can distinguish Pong vs Breakout
            src = str(row["source"])
            return f"LTH transfer, {base} (source: ALE/{src}, 60%)"

        # Baselines (if you keep them in mask)
        if row["project"] == "atari-gmp":
            return f"GMP {int(row['sparsity'])}%"
        if row["project"] == "atari-lottery":
            return "Dense"
        return str(row["project"])

    # If you truly want ONLY LTH-vs-GMP transfer lines, uncomment the following filter:
    # curves_plot = curves[curves["project"].eq("lth-transfer")].copy()
    curves_plot = curves.copy()

    for keys, g in curves_plot.groupby(curve_cols, dropna=False):
        # Skip non-transfer curves if you want a strict transfer-only figure
        # (Keep this ON to match your request "LTH vs GMP transfer")
        # if g.iloc[0]["project"] != "lth-transfer":
        #     continue

        lbl = label(g.iloc[0])

        x = g["step"].to_numpy(dtype=float) / 1_000_000.0  # Millions
        y = g["iqm_s"].to_numpy(dtype=float)
        lo = g["ci_lo_s"].to_numpy(dtype=float)
        hi = g["ci_hi_s"].to_numpy(dtype=float)

        plt.plot(x, y, label=lbl)

        # 95% CI shading (skip if undefined)
        if not (np.all(np.isnan(lo)) or np.all(np.isnan(hi))):
            plt.fill_between(x, lo, hi, alpha=0.15)

    plt.xlabel("Steps (Millions)")
    plt.ylabel("IQM")
    plt.title("ALE/SpaceInvaders-v5 (95% CI)")
    plt.legend()
    plt.tight_layout()

    # High-res export for download
    out_path = "area-ratio/ale_space_invaders_lth_vs_gmp_transfer_iqm_ci.png"
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.show()

    print(f"Saved: {out_path}")

    # Plot area ratio bar chart
    if ratios:
        plt.figure(figsize=(10, 5))
        labels = list(ratios.keys())
        ratio_values = [ratios[l]["ratio"] for l in labels]
        colors = ["green" if r > 0 else "red" for r in ratio_values]
        
        bars = plt.bar(labels, ratio_values, color=colors, alpha=0.7, edgecolor="black")
        
        # Add value labels on bars
        for bar, val in zip(bars, ratio_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}\n({val*100:+.2f}%)',
                    ha='center', va='bottom' if val > 0 else 'top', fontsize=10)
        
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        plt.ylabel("Area Ratio")
        plt.title("Transfer vs Scratch Performance Ratio\n(area_transfer - area_scratch) / area_scratch")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        ratio_out_path = "area-ratio/ale_space_invaders_area_ratio.png"
        plt.savefig(ratio_out_path, dpi=400, bbox_inches="tight")
        plt.show()
        
        print(f"Saved: {ratio_out_path}")


if __name__ == "__main__":
    main()

