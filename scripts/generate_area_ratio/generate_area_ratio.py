#!/usr/bin/env python3
"""
generate_area_ratio.py

Compute Area Ratio r = (Area_transfer - Area_scratch) / Area_scratch
for various transfer conditions across 3 Atari environments.
Includes 95% Bootstrap Confidence Intervals.
"""

import pandas as pd
import numpy as np
import os

# Config
MAX_STEP = 10_000_000
COARSE_BIN = 50_000
N_BOOT = 2000  # Increased for better precision on area ratio
RNG_SEED = 123

FILES = {
    "gmp_transfer": {
        "breakout": "../data/gmp-transfer/returns_breakout.csv",
        "pong": "../data/gmp-transfer/returns_pong.csv",
        "space_invaders": "../data/gmp-transfer/returns_space-invaders.csv",
    },
    "lth_transfer": {
        "breakout": "../data/lth-transfer/returns_breakout.csv",
        "pong": "../data/lth-transfer/returns_pong.csv",
        "space_invaders": "../data/lth-transfer/returns_space-invaders.csv",
    },
}

BASELINES = [
    {
        "name": "dense",
        "project": "atari-lottery",
        "seeds": {
            "pong": [2, 3],
            "breakout": [1, 4, 5],
            "space_invaders": [1, 2, 3, 4, 5],
        },
    },
    {
        "name": "GMP 75",
        "project": "atari-gmp",
        "sparsity": 75,
        "seeds": {
            "pong": [2, 3],
            "breakout": [1, 3, 4, 5],
            "space_invaders": [1, 2, 3, 4, 5],
        },
    },
    {
        "name": "GMP 90",
        "project": "atari-gmp",
        "sparsity": 90,
        "seeds": {
            "pong": [2, 3],
            "breakout": [1, 3, 4, 5],
            "space_invaders": [1, 2, 3, 4, 5],
        },
    },
]

TRANSFER = {
    "gmp": [
        {
            "name": "GMP 60 -> dense",
            "project": "gmp-transfer",
            "source_sparsity": 60,
            "target_pruning": "dense", # Renamed from pruning to target_pruning for clarity
        },
        {
            "name": "GMP 75 -> dense",
            "project": "gmp-transfer",
            "source_sparsity": 75,
            "target_pruning": "dense",
        },
        {
            "name": "GMP 60 -> GMP 90",
            "project": "gmp-transfer",
            "source_sparsity": 60,
            "target_pruning": "gmp",
        },
        {
            "name": "GMP 75 -> GMP 90",
            "project": "gmp-transfer",
            "source_sparsity": 75,
            "target_pruning": "gmp",
        },
    ],
    "lth": [
        {
            "name": "LTH 60 -> dense",
            "project": "lth-transfer",
            "source_sparsity": 60,
            "target_pruning": "dense",
        },
        {
            "name": "LTH 75 -> dense",
            "project": "lth-transfer",
            "source_sparsity": 75,
            "target_pruning": "dense",
        },
        {
            "name": "LTH 60 -> GMP 90",
            "project": "lth-transfer",
            "source_sparsity": 60,
            "target_pruning": "gmp",
        },
        {
            "name": "LTH 75 -> GMP 90",
            "project": "lth-transfer",
            "source_sparsity": 75,
            "target_pruning": "gmp",
        },
    ],
}


def iqm_trimmed_mean(values: np.ndarray) -> float:
    """Interquartile mean (25% trimmed mean)."""
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


# Atari Min Scores (R_min)
# Rs_t = R_t - R_min
MIN_SCORES = {
    "breakout": 0.0,
    "pong": -21.0,
    "space_invaders": 0.0, 
    "space-invaders": 0.0,
    "SpaceInvaders-v5": 0.0,
    "Pong-v5": -21.0,
    "Breakout-v5": 0.0,
}

_DATA_CACHE = {}

def load_and_preprocess(file_path: str, max_step=MAX_STEP) -> pd.DataFrame:
    if file_path in _DATA_CACHE:
        return _DATA_CACHE[file_path]

    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return pd.DataFrame()
        
    print(f"Loading {file_path}...", flush=True)
    df = pd.read_csv(file_path, low_memory=False)
    
    # Cleanup
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["episodic_return"] = pd.to_numeric(df["episodic_return"], errors="coerce")
    df = df.dropna(subset=["step", "episodic_return"])
    df["step"] = df["step"].astype(int)
    
    # Limit steps
    df = df[df["step"] <= max_step].copy()
    
    # Shift Returns
    # R_s = R - R_min
    filename = os.path.basename(file_path).lower()
    
    env_key = None
    if "pong" in filename: env_key = "pong"
    elif "breakout" in filename: env_key = "breakout"
    elif "space_invaders" in filename or "space-invaders" in filename: env_key = "space_invaders"
    
    if env_key and env_key in MIN_SCORES:
        r_min = MIN_SCORES[env_key]
        df["episodic_return"] = df["episodic_return"] - r_min
    else:
        # Default to 0 shift if unknown? Or warn?
        # Assuming 0 is safe for unknown envs as "raw return"
        pass

    # Binning
    # Center bins: 0..50k -> 25k? Or just 0, 50k, 100k...
    # Existing compare_runs does: (step // BIN) * BIN. Let's stick with that for consistency.
    df["bin_step"] = (df["step"] // COARSE_BIN) * COARSE_BIN
    
    # Pre-aggregate: Mean return per (run, bin)
    # Group keys need to include everything to identify a run
    # 'pruning' col might be 'dense' or 'gmp' or nan. 
    # 'source', 'source_sparsity' for transfer.
    
    _DATA_CACHE[file_path] = df
    return df

def get_run_curves(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Extracts runs matching filters.
    Returns DataFrame with [run_id, bin_step, return].
    """
    mask = pd.Series(True, index=df.index)
    for k, v in filters.items():
        if k == "seeds":
            mask &= df["seed"].isin(v)
        elif k in df.columns:
            mask &= df[k] == v
            
    filtered = df[mask].copy()
    
    # Aggregate to one curve per run
    # Identify run by: project, seed, source(if any), pruning(if any), source_sparsity(if any)
    # We can just create a unique run_id
    filtered["run_id"] = (
        filtered["project"].astype(str) + "_" + 
        filtered["seed"].astype(str) + "_" + 
        filtered.get("source", "").astype(str) + "_" +
        filtered.get("pruning", "").astype(str)
    )
            
    # Mean return per bin per run
    curves = filtered.groupby(["run_id", "bin_step"])["episodic_return"].mean().reset_index()
    
    # Pivot to get matrix: specific bin_steps as columns is hard if they differ.
    # Better: list of arrays. But we need aligned bins for Area.
    # Let's pivot: index=bin_step, columns=run_id
    pivoted = curves.pivot(index="bin_step", columns="run_id", values="episodic_return")
    
    # Fill missing bins? 
    # If a run misses a bin in the middle, interpolate? For now leave NaN or ffill.
    # compare_runs doesn't explicitly fill, just Bootstrap IQM per bin.
    # But for Area, we need a valid curve.
    # Let's ffill then fillna(0) or similar.
    pivoted = pivoted.ffill().fillna(0) # Simple strategy
    
    # Ensure we cover range 0 to MAX_STEP
    all_bins = np.arange(0, MAX_STEP + 1, COARSE_BIN)
    pivoted = pivoted.reindex(all_bins).ffill().fillna(0)
    
    return pivoted

def compute_area_iqm(curves: pd.DataFrame) -> float:
    """
    Compute Area under IQM curve.
    curves: DataFrame (index=steps, col=runs)
    """
    # 1. Compute IQM curve
    arr = curves.to_numpy() # shape (n_bins, n_runs)
    n_bins, n_runs = arr.shape
    if n_runs == 0:
        return 0.0
        
    iqm_curve = np.array([iqm_trimmed_mean(arr[i, :]) for i in range(n_bins)])
    
    # 2. Integrate (Sum * dt)
    # dt = COARSE_BIN (but normalized? usually Area is just sum if step size constant)
    # Let's use raw sum * bin_width to get "Return * Steps" units
    area = np.sum(iqm_curve) * COARSE_BIN
    return area

def bootstrap_area_ratio(base_curves: pd.DataFrame, transfer_curves: pd.DataFrame, rng: np.random.Generator, n_boot=N_BOOT):
    """
    Bootstrap distribution of Area Ratio.
    r = (Area_t - Area_b) / Area_b
    Fully vectorized for speed.
    """
    base_arr = base_curves.to_numpy() # (bins, n_runs_b)
    trans_arr = transfer_curves.to_numpy() # (bins, n_runs_t)
    
    nb = base_arr.shape[1]
    nt = trans_arr.shape[1]
    n_bins = base_arr.shape[0]
    
    if nb == 0 or nt == 0:
        return np.nan, np.nan, np.nan, np.nan
    
    # Pre-compute bin width
    dt = COARSE_BIN
    
    # Vectorized Bootstrap
    # Generate indices: (n_boot, n_runs)
    idx_b = rng.integers(0, nb, size=(n_boot, nb))
    idx_t = rng.integers(0, nt, size=(n_boot, nt))
    
    # Gather data: (bins, n_runs) -> (bins, n_boot, n_runs)
    # We want (n_boot, bins, n_runs) to sort along last axis? 
    # Or (bins, n_boot, n_runs).
    # base_arr[:, idx_b] ?
    # idx_b is (n_boot, nb).
    # We want to select columns for each boot.
    # base_arr[:, idx_b] shape:
    # broadcasting? base_arr (bins, nb)
    # This is tricky in numpy for arbitrary indices per "sample".
    # base_arr[:, idx_b] results in (bins, n_boot, nb). Correct.
    
    sample_b = base_arr[:, idx_b] # (bins, n_boot, nb)
    sample_t = trans_arr[:, idx_t] # (bins, n_boot, nt)
    
    # Compute IQM Trimmed Mean along last axis (nb/nt)
    # 25% trimmed mean
    
    def vectorized_iqm(arr):
        # arr shape: (..., n)
        # Sort along last axis
        n = arr.shape[-1]
        sorted_arr = np.sort(arr, axis=-1)
        
        lower = int(np.floor(0.25 * n))
        upper = n - lower
        
        # Slice from lower to upper
        # Since n is constant, we can slice directly using indices
        # If lower >= upper (small n), mean of all
        
        if lower >= upper:
             # Just mean
             return np.mean(sorted_arr, axis=-1)
             
        # Slice
        mid = sorted_arr[..., lower:upper]
        return np.mean(mid, axis=-1)

    iqm_b = vectorized_iqm(sample_b) # (bins, n_boot)
    iqm_t = vectorized_iqm(sample_t) # (bins, n_boot)
    
    # Sum along bins axis (axis 0) to get Area
    area_b = np.sum(iqm_b, axis=0) * dt # (n_boot,)
    area_t = np.sum(iqm_t, axis=0) * dt # (n_boot,)
    
    # Compute Ratio
    with np.errstate(divide='ignore', invalid='ignore'):
         ratio = (area_t - area_b) / area_b
         
    # Handle NaN or Inf if area_b is 0
    ratio = ratio[~np.isnan(ratio)]
    ratio = ratio[~np.isinf(ratio)]
    
    if len(ratio) == 0:
        return np.nan, np.nan, np.nan, np.nan
        
    mean_r = np.mean(ratio)
    ci_lo = np.percentile(ratio, 2.5)
    ci_hi = np.percentile(ratio, 97.5)
    
    # Significant if 0 not in [lo, hi]
    significant = (ci_lo > 0) or (ci_hi < 0)
    
    return mean_r, ci_lo, ci_hi, significant

def main():
    rng = np.random.default_rng(RNG_SEED)
    
    # We need to map Env names to what they are in the files/dicts
    # Keys in seeded lists are: "pong", "breakout", "space-invaders"
    # Env names in CSV 'source': "Pong-v5", "Breakout-v5", "SpaceInvaders-v5"
    env_map = {
        "Breakout-v5": "breakout",
        "Pong-v5": "pong",
        "SpaceInvaders-v5": "space_invaders" # Dictionary key uses underscore
    }
    
    # Load all data first? Or on demand?
    # Let's load on demand to save memory.
    
    all_results = []
    
    # Iterate over conditions -> tables
    for transfer_type in ["gmp", "lth"]:
        for condition_cfg in TRANSFER[transfer_type]:
            cond_name = condition_cfg["name"]
            proj = condition_cfg["project"]
            src_sp = condition_cfg["source_sparsity"]
            tgt_prun = condition_cfg["target_pruning"]
            
            print(f"Processing condition: {cond_name}")
            
            # Loop over ALL Baselines
            for baseline_cfg in BASELINES:
                baseline_name = baseline_cfg["name"]
                print(f"  vs Baseline: {baseline_name}")
            
                # Create Table Rows
                # Env_Target | Env_Source | Ratio | CI_Low | CI_High | Significant
                # Each env is both source and transfer.
                # 3 Envs: Breakout, SpaceInvaders, Pong
                
                envs = ["Breakout-v5", "SpaceInvaders-v5", "Pong-v5"]
                
                table_rows = []
                
                for target_env in envs:
                    tgt_key = env_map[target_env]
                    
                    # Load Target Scratch Data (Baseline) for this environment
                    # From: BASELINES, project, seed list
                    base_info = baseline_cfg
                    
                    # Path
                    # Load from the file set corresponding to the transfer type (gmp or lth)
                    # This ensures we use the same data source as the transfer runs and the plots.
                    file_key = f"{transfer_type}_transfer"
                    if file_key not in FILES:
                         # Fallback
                         file_key = "gmp_transfer"
                         
                    f_path = FILES[file_key][tgt_key]
                    df_env = load_and_preprocess(f_path)
                    
                    # Get Baseline Curves
                    # Ensure we use the specific seeds for this environment
                    # Keys in seeds dict: "pong", "breakout", "space_invaders"
                    # tgt_key is "pong", "breakout", "space_invaders"
                    seeds_list = base_info["seeds"].get(tgt_key)
                    if seeds_list is None:
                         # Fallback for hyphen vs underscore if needed, though we normalized to underscore
                         seeds_list = base_info["seeds"].get(tgt_key.replace("_", "-"))
                    
                    if seeds_list is None:
                        print(f"Warning: No seeds found for {tgt_key} in baseline {baseline_name}")
                        continue

                    base_filters = {
                        "project": base_info["project"],
                        "seeds": seeds_list,
                    }
                    if "sparsity" in base_info:
                        base_filters["sparsity"] = base_info["sparsity"]
                        
                    curves_base = get_run_curves(df_env, base_filters)
                    
                    for source_env in envs:
                        if source_env == target_env:
                            continue # No self-transfer
                            
                        # Get Transfer Curves
                        trans_filters = {
                            "project": proj,
                            "source": source_env,
                            "source_sparsity": src_sp,
                            "pruning": tgt_prun,
                        }
                        
                        # Use seeds 1-5 for transfer as per typical convention (check if this is always true?)
                        # The user prompt implied "Each area ration table compares the area of each baseline with the area of each transfer condition"
                        # Transfer runs usually match the seeds of the source runs.
                        trans_filters["seeds"] = [1, 2, 3, 4, 5]
                        
                        if proj == "lth-transfer":
                             f_path_trans = FILES["lth_transfer"][tgt_key]
                             df_trans_env = load_and_preprocess(f_path_trans)
                        else:
                             df_trans_env = df_env
                             
                        curves_trans = get_run_curves(df_trans_env, trans_filters)
                        
                        # Compute Ratio
                        mean_r, ci_lo, ci_hi, sig = bootstrap_area_ratio(curves_base, curves_trans, rng)
                        
                        table_rows.append({
                            "Condition": cond_name,
                            "Baseline": baseline_name,
                            "Source": source_env.replace("-v5", ""),
                            "Target": target_env.replace("-v5", ""),
                            "Ratio": mean_r,
                            "CI_Lo": ci_lo,
                            "CI_Hi": ci_hi,
                            "Significant": sig
                        })
                        
                # Result DF
                if not table_rows:
                    continue
                    
                res_df = pd.DataFrame(table_rows)
                # Save
                # Filename: {condition}_vs_{baseline}.csv
                # e.g. gmp_60_to_dense_vs_dense.csv
                cond_slug = cond_name.replace(" ", "_").replace("->", "to").lower()
                base_slug = baseline_name.replace(" ", "_").lower()
                fname = f"{cond_slug}_vs_{base_slug}.csv"
                
                out_path = f"../data/area_ratios/{fname}"
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                res_df.to_csv(out_path, index=False)
                print(f"Saved {out_path}")

if __name__ == "__main__":
    main()