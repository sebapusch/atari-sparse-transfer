import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
LTH_DATA_DIR = "../data/lth-transfer"
GMP_DATA_DIR = "../data/gmp-transfer"
OUTPUT_PATH = "../data/gmp-transfer/static_comparison_grid.png"

# Environments
ENVS = ["Breakout", "Pong", "SpaceInvaders"]
ENV_MAP = {
    "Breakout": "Breakout-v5",
    "Pong": "Pong-v5",
    "SpaceInvaders": "SpaceInvaders-v5"
}

MAX_STEP = 10_000_000
COARSE_BIN = 50_000
SMOOTH_WINDOW_BINS = 10
N_BOOT = 200

# Plot styling - Standard Colors (Not saturated)
COLORS = {
    "LTH_60": "#1f77b4",  # Blue
    "LTH_75": "#ff7f0e",  # Orange
    "GMP_60": "#2ca02c",  # Green
    "GMP_75": "#d62728",  # Red
}
LABELS = {
    "LTH_60": "LTH60 Static",
    "LTH_75": "LTH75 Static",
    "GMP_60": "GMP60 Static",
    "GMP_75": "GMP75 Static",
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

def load_data(data_dir: str, target_env_slug: str, method_label: str):
    path = os.path.join(data_dir, f"returns_{target_env_slug.lower()}.csv")
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        return pd.DataFrame()
    
    df = pd.read_csv(path, low_memory=False)
    
    # Cleanup
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["episodic_return"] = pd.to_numeric(df["episodic_return"], errors="coerce")
    df = df.dropna(subset=["step", "episodic_return"]).copy()
    df["step"] = df["step"].astype(int)
    
    # Filter for Dense only
    # Note: Project names might vary but we filter by pruning='dense'
    # Also ensure we are looking at TRANSFER runs (source is NOT NaN)
    df = df[
        (df["pruning"] == "dense") &
        (df["source"].notna()) & 
        (df["step"] <= MAX_STEP)
    ].copy()
    
    df["method"] = method_label
    return df

def process_data(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame()

    # Binning
    df["bin_step"] = (df["step"] // COARSE_BIN) * COARSE_BIN
    
    # Aggregate within run
    per_run = (
        df.groupby(["method", "source", "source_sparsity", "run_name", "seed", "bin_step"], as_index=False, dropna=False)["episodic_return"]
        .mean()
    )
    
    # Bootstrap over runs
    rng = np.random.default_rng(42)
    rows = []
    # Key: method, source, sparsity
    groups = per_run.groupby(["method", "source", "source_sparsity", "bin_step"], dropna=False)
    
    for (meth, src, sp, step), grp in groups:
        vals = grp["episodic_return"].to_numpy(dtype=float)
        center, lo, hi = bootstrap_iqm_fast(vals, N_BOOT, rng)
        rows.append({
            "method": meth,
            "source": src,
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
    curves_s = curves.groupby(["method", "source", "source_sparsity"], group_keys=False).apply(smooth)
    return curves_s

# --- Main ---

def main():
    # Setup Grid
    # Rows: Target Envs (3)
    # Cols: Source Envs (2)
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True, sharey=False)
    
    targets = ["Breakout", "Pong", "SpaceInvaders"]
    
    handles_map = {}
    
    for r, target_name in enumerate(targets):
        print(f"Processing Target: {target_name}...")
        
        # Filename slug
        if target_name == "SpaceInvaders":
            slug = "space-invaders"
        else:
            slug = target_name.lower()
        
        # Load from both sources
        df_lth = load_data(LTH_DATA_DIR, slug, "LTH")
        df_gmp = load_data(GMP_DATA_DIR, slug, "GMP")
        
        # Combine
        df_combined = pd.concat([df_lth, df_gmp], ignore_index=True)
        
        # Process (Bootstrapping etc)
        data = process_data(df_combined)
        
        # Plotting
        other_sources = [e for e in targets if e != target_name]
        
        for c, source_simple_name in enumerate(other_sources):
            ax = axes[r, c]
            
            src_full = ENV_MAP[source_simple_name]
            
            if not data.empty:
                subset = data[data["source"] == src_full]
            else:
                subset = pd.DataFrame()

            ax.set_title(f"Source: {source_simple_name}", fontweight="bold")
            
            # Conditions: Method + Sparsity
            conditions = [
                ("LTH", 60.0), ("LTH", 75.0),
                ("GMP", 60.0), ("GMP", 75.0)
            ]
            
            for method, sparsity in conditions:
                key = f"{method}_{int(sparsity)}"
                color = COLORS.get(key, "gray")
                label = LABELS.get(key, key)
                
                if subset.empty:
                    continue
                
                line_data = subset[
                    (subset["method"] == method) & 
                    (np.isclose(subset["source_sparsity"], sparsity))
                ]
                
                if line_data.empty:
                    continue
                
                x = line_data["step"] / 1e6
                y = line_data["iqm_s"]
                lo = line_data["ci_lo_s"]
                hi = line_data["ci_hi_s"]
                
                (line,) = ax.plot(x, y, color=color, linewidth=2.0, label=label)
                
                if key not in handles_map:
                    handles_map[key] = line
                
                ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
            
            if r == 2:
                ax.set_xlabel("Steps (Millions)")
            
            if c == 0:
                ax.set_ylabel(f"$\\bf{{{target_name}}}$\nIQM") 
            
            ax.grid(False)

    # Legend
    sorted_handles = []
    sorted_labels = []
    for key in ["LTH_60", "LTH_75", "GMP_60", "GMP_75"]:
        if key in handles_map:
            sorted_handles.append(handles_map[key])
            sorted_labels.append(LABELS[key])
            
    if sorted_handles:
        fig.legend(sorted_handles, sorted_labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    print(f"Saving to {OUTPUT_PATH}")
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()
