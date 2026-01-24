import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# --- Configuration ---
DATA_DIR = "../data/gmp-transfer"
OUTPUT_PATH = "../data/gmp-transfer/gmp_grid.png"

# Environments
ENVS = ["Breakout", "Pong", "SpaceInvaders"]
ENV_MAP = {
    "Breakout": "Breakout-v5",
    "Pong": "Pong-v5",
    "SpaceInvaders": "SpaceInvaders-v5"
}

# Source map for plot labels
ENV_PRETTY_NAME = {
    "Breakout": "Breakout",
    "Pong": "Pong",
    "SpaceInvaders": "SpaceInvaders"
}

MAX_STEP = 10_000_000
COARSE_BIN = 50_000
SMOOTH_WINDOW_BINS = 10
N_BOOT = 200

# Plot styling
COLORS = {
    "dense_60": "#004488",  # High Saturation Blue (LTH 60% static)
    "dense_75": "#FFBB77",  # Low Saturation Orange (LTH 75% static)
    "gmp_60":   "#77CC77",  # Low Saturation Green (LTH 60% GMP 90%)
    "gmp_75":   "#FF9999",  # Low Saturation Red (LTH 75% GMP 90%)
}
LABELS = {
    "dense_60": "LTH 60%, static",
    "dense_75": "LTH 75%, static",
    "gmp_60":   "LTH 60%, GMP 90%",
    "gmp_75":   "LTH 75%, GMP 90%",
}

# Distinct colors for Baselines
BASELINE_COLORS = {
    "dense_nan": "black",   # High Saturation Black (Dense)
    "gmp_75.0":  "#AA00AA", # High Saturation Purple (GMP 75%)
    "gmp_90.0":  "#D2B48C", # Low Saturation Brown/Tan (GMP 90%)
}

BASELINE_LABELS = {
    "dense_nan": "Dense",
    "gmp_75.0":  "GMP 75%",
    "gmp_90.0":  "GMP 90%",
}

SEEDS = {
    "Breakout": [1,4,5],
    "Pong": [2,3],
    "SpaceInvaders": [1,2,3,4,5]
}

# Increase font size globally
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 14, 'axes.labelsize': 14, 'legend.fontsize': 14})

# --- Helpers ---

def iqm_trimmed(values: np.ndarray) -> float:
    """IQM via symmetric trimming of 25% from each side."""
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return np.nan
    if n < 4:
        return float(np.mean(values))
    v = np.sort(values)
    k = int(np.floor(0.25 * n))
    mid = v[k:n - k]
    return float(np.mean(mid)) if len(mid) else float(np.mean(v))

def bootstrap_iqm_fast(values: np.ndarray, n_boot: int, rng: np.random.Generator):
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
    mid = samples[:, k:n - k] if (n - 2 * k) > 0 else samples
    stats = mid.mean(axis=1)
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return center, float(lo), float(hi)

def smooth(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("step").copy()
    for c in ["iqm", "ci_lo", "ci_hi"]:
        g[f"{c}_s"] = g[c].rolling(SMOOTH_WINDOW_BINS, min_periods=1).mean()
    return g

def load_and_process(target_env_slug: str, target_full_name: str, target_simple_name: str):
    path = os.path.join(DATA_DIR, f"returns_{target_env_slug.lower()}.csv")
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        return pd.DataFrame()
    
    df = pd.read_csv(path, low_memory=False)
    
    # Cleanup
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["episodic_return"] = pd.to_numeric(df["episodic_return"], errors="coerce")
    df = df.dropna(subset=["step", "episodic_return"]).copy()
    df["step"] = df["step"].astype(int)
    
    # --- Baseline Inference ---
    # Rows with source=NaN are Baselines (source = target)
    baseline_mask = df["source"].isna()
    
    # Infer Source for Baselines
    df.loc[baseline_mask, "source"] = target_full_name
    
    # Infer Pruning for Baselines
    # atari-gmp -> gmp
    # atari-lottery -> dense
    df.loc[baseline_mask & (df["project"] == "atari-gmp"), "pruning"] = "gmp"
    df.loc[baseline_mask & (df["project"] == "atari-lottery"), "pruning"] = "dense"
    
    # Infer Source Sparsity for Baselines (use 'sparsity' column)
    # Ensure sparsity is float
    df["sparsity"] = pd.to_numeric(df["sparsity"], errors='coerce')
    df["source_sparsity"] = pd.to_numeric(df["source_sparsity"], errors='coerce')
    
    df.loc[baseline_mask, "source_sparsity"] = df.loc[baseline_mask, "sparsity"]
    
    # --- Seed Filtering for Baselines ---
    if target_simple_name in SEEDS:
        allowed_seeds = SEEDS[target_simple_name]
        # Keep:
        # 1. Non-baseline rows (Transfer runs) -> ~baseline_mask
        # 2. Baseline rows with allowed seeds -> baseline_mask & isin(allowed)
        condition = (~baseline_mask) | (baseline_mask & df["seed"].isin(allowed_seeds))
        df = df[condition].copy()
    
    # Filter 
    # Projects: gmp-transfer (Transfer), atari-gmp (Baseline), atari-lottery (Baseline)
    valid_projects = ["gmp-transfer", "atari-gmp", "atari-lottery"]
    df = df[
        (df["project"].isin(valid_projects)) &
        (df["step"] <= MAX_STEP)
    ].copy()

    if df.empty:
        return df

    # Binning
    df["bin_step"] = (df["step"] // COARSE_BIN) * COARSE_BIN
    
    # Aggregate within run
    per_run = (
        df.groupby(["source", "pruning", "run_name", "source_sparsity", "seed", "bin_step"], as_index=False, dropna=False)["episodic_return"]
        .mean()
    )
    
    # Bootstrap over runs
    rng = np.random.default_rng(42)
    rows = []
    groups = per_run.groupby(["source", "pruning", "source_sparsity", "bin_step"], dropna=False)
    
    for (src, prun, sp, step), grp in groups:
        vals = grp["episodic_return"].to_numpy(dtype=float)
        center, lo, hi = bootstrap_iqm_fast(vals, N_BOOT, rng)
        rows.append({
            "source": src,
            "pruning": prun,
            "source_sparsity": sp,
            "step": int(step),
            "iqm": center,
            "ci_lo": lo,
            "ci_hi": hi,
            "n_runs": len(vals)
        })
    
    if not rows:
        return pd.DataFrame()

    curves = pd.DataFrame(rows).dropna(subset=["iqm"])
    
    # Smoothing
    # IMPORTANT: dropna=False to preserve Dense baseline (NaN sparsity)
    curves_s = curves.groupby(["source", "pruning", "source_sparsity"], group_keys=False, dropna=False).apply(smooth)
    return curves_s


# --- Main ---

def main():
    # Setup Grid
    # Rows: Target Envs (3)
    # Cols: Baseline (0) + 2 Sources (1, 2)
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True, sharey=False)
    
    targets = ["Breakout", "Pong", "SpaceInvaders"]
    
    handles_map = {}
    baseline_handles_map = {}
    
    for r, target_name in enumerate(targets):
        print(f"Processing Target: {target_name}...")
        
        # Filename slug
        if target_name == "SpaceInvaders":
            slug = "space-invaders"
        else:
            slug = target_name.lower()
        
        target_full = ENV_MAP[target_name]
        data = load_and_process(slug, target_full, target_name)
        
        # Columns: [Baseline (Target), Source A, Source B]
        other_sources = [e for e in targets if e != target_name]
        sources_to_plot = [target_name] + other_sources # List of simple names
        
        for c, source_simple_name in enumerate(sources_to_plot):
            ax = axes[r, c]
            
            src_full = ENV_MAP[source_simple_name]
            
            # Identify if this is baseline column
            is_baseline = (c == 0)
            
            if not data.empty:
                subset = data[data["source"] == src_full]
            else:
                subset = pd.DataFrame()

            # Plot lines
            if is_baseline:
                # Baseline Conditions: Dense (NaN), GMP 75, GMP 90
                # We iterate through the explicit list we want
                conditions = [
                    ("dense", float('nan')),
                    ("gmp", 75.0),
                    ("gmp", 90.0)
                ]
            else:
                # Transfer Conditions: Dense 60/75, GMP 60/75
                conditions = [
                    ("dense", 60.0), ("dense", 75.0),
                    ("gmp", 60.0), ("gmp", 75.0)
                ]
            
            # Determine title
            if is_baseline:
                title = f"Baseline"
            else:
                title = f"Source: {source_simple_name}"
            
            ax.set_title(title, fontweight="bold")
            
            for pruning, sparsity in conditions:
                # Handle NaN equality for sparsity
                if pd.isna(sparsity):
                    key = f"{pruning}_nan"
                    line_data = subset[
                        (subset["pruning"] == pruning) & 
                        (subset["source_sparsity"].isna())
                    ]
                else:
                    key = f"{pruning}_{sparsity}" if is_baseline else f"{pruning}_{int(sparsity)}"
                    line_data = subset[
                        (subset["pruning"] == pruning) & 
                        (np.isclose(subset["source_sparsity"], sparsity))
                    ]
                
                if is_baseline:
                    color = BASELINE_COLORS.get(key, "gray")
                    label = BASELINE_LABELS.get(key, key)
                    # Highlight GMP 75% and Dense
                    if key in ["dense_nan", "gmp_75.0"]:
                        z_order = 3
                    else:
                        z_order = 2
                else:
                    color = COLORS.get(key, "gray") # Use transfer colors
                    label = LABELS.get(key, key)
                    # Highlight LTH 60% static
                    if key == "dense_60":
                        z_order = 3
                    else:
                        z_order = 2

                if subset.empty or line_data.empty:
                    continue
                
                x = line_data["step"] / 1e6
                y = line_data["iqm_s"]
                lo = line_data["ci_lo_s"]
                hi = line_data["ci_hi_s"]
                
                (line,) = ax.plot(x, y, color=color, linewidth=2.0, label=label, zorder=z_order)
                
                if is_baseline:
                    if key not in baseline_handles_map:
                        baseline_handles_map[key] = line
                else:
                    if key not in handles_map:
                        handles_map[key] = line
                
                ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0, zorder=z_order)
            
            if r == 2:
                ax.set_xlabel("Steps (Millions)")
            
            if c == 0:
                ax.set_ylabel(f"$\\bf{{{target_name}}}$\nIQM") 
            
            ax.grid(False)
    
    # Legend at the top
    all_handles = []
    all_labels = []
    
    # Baselines First
    for key in ["dense_nan", "gmp_75.0", "gmp_90.0"]:
        if key in baseline_handles_map:
            all_handles.append(baseline_handles_map[key])
            all_labels.append(BASELINE_LABELS[key])
            
    # Transfer Second
    for key in ["dense_60", "dense_75", "gmp_60", "gmp_75"]:
        if key in handles_map:
            all_handles.append(handles_map[key])
            all_labels.append(LABELS[key])
            
    if all_handles:
        fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=7, frameon=False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    print(f"Saving to {OUTPUT_PATH}")
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()
