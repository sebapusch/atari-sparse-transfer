#!/usr/bin/env python3
import pandas as pd
import wandb
import numpy as np
import argparse
import os
import re

# Constants
ENTITY = "sebapusch-university-of-groningen"
PROJECT = "atari-lottery"
STEP_KEY = "_step"
EPSILON_KEY = "charts/epsilon"
RETURN_KEY = "charts/episodic_return"

def parse_baseline_runs(csv_path):
    """
    Parses the CSV to find all relevant runs and groups them by their base identity.
    Assumes names like: ...-s1, ...-s1-2, ...-s1-2-3
    Returns a dict: {base_name_key: [list_of_full_run_names]}
    """
    df = pd.read_csv(csv_path)
    # The 'Name' column contains the W&B run ID/Name? 
    # The CSV header says "Name", "State", ..., "wandb.name"
    # Let's use 'wandb.name' or 'Name' if they match.
    # In the file view, first col is "Name", and there is a "wandb.name" col too.
    # Row 2 Example: "atari_ddqn_lth-rand_sweep-SpaceInvaders-v5-S95-s1-2-3"
    
    if "wandb.name" in df.columns:
        names = df["wandb.name"].dropna().unique()
    elif "Name" in df.columns:
        names = df["Name"].dropna().unique()
    else:
        raise ValueError("Could not find Name or wandb.name column")

    groups = {}
    
    for name in names:
        # Regex to capture the base part (Env + Seed) and the Suffix part (-2, -2-3 etc)
        # Expected format: <prefix>-S<sparsity>-s<seed>[-<suffix>...]
        # Example: atari_ddqn_lth-rand_sweep-SpaceInvaders-v5-S95-s1 -> Base: ...s1
        # Example: atari_ddqn_lth-rand_sweep-SpaceInvaders-v5-S95-s1-2 -> Base: ...s1, Suffix -2
        # Example: atari_ddqn_lth-rand_sweep-SpaceInvaders-v5-S95-s1-2-3 -> Base: ...s1, Suffix -2-3
        
        # We can detect the seed part "-s\d+". Anything after that is the split continuation?
        # Actually user said "split up to 3 individual runs <run>,<run>-2,<run>-2-3".
        # So "s1" is the base run. "s1-2" is base + "-2".
        # So we can split by "-".
        
        # Robust grouping strategy: 
        # Identify the Environment and Seed.
        # Env: SpaceInvaders-v5, Pong-v5, Breakout-v5
        # Sparsity: S95
        # Seed: s1, s2, s3...
        
        if "SpaceInvaders" in name: env = "SpaceInvaders"
        elif "Pong" in name: env = "Pong"
        elif "Breakout" in name: env = "Breakout"
        else: continue # Skip unrelated runs if any
            
        # Extract seed
        # match s<number>
        seed_match = re.search(r'-s(\d+)', name)
        if not seed_match:
            print(f"Skipping {name}, no seed found")
            continue
        seed = seed_match.group(1)
        
        # Key for grouping: Env + Seed
        key = f"{env}_s{seed}"
        
        if key not in groups:
            groups[key] = []
        groups[key].append(name)
        
    return groups

def fetch_run_history(run, max_step=100_000_000):
    try:
        # Fetching large history
        # Note: Using scan_history is irrelevant if we just use history() which is faster for known keys
        # But let's stick to the robust method if needed.
        # But export_lth used history() successfully after fixes.
        
        df = run.history(keys=[STEP_KEY, EPSILON_KEY, RETURN_KEY], pandas=True, samples=max_step)
        
        if df.empty: return df
        
        # Clean up
        for col in [STEP_KEY, EPSILON_KEY, RETURN_KEY]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        if STEP_KEY not in df.columns and df.index.name == STEP_KEY:
             df = df.reset_index()
             
        df = df.dropna(subset=[STEP_KEY]).sort_values(STEP_KEY)
        return df
    except Exception as e:
        print(f"Error fetching {run.name}: {e}")
        return pd.DataFrame()

def detect_spike_steps(eps_series, step_series, eps_high=0.9, min_gap_steps=500_000):
    # Same logic as export_lth
    # For Random baseline, do we have epsilon spikes?
    # Yes, "random_lth" implies it does pruning rounds too?
    # The csv has "pruning.method": "random_lth".
    # So it should honor the same epsilon schedule resets.
    
    eps = eps_series.values
    steps = step_series.values
    
    cand_indices = np.where(eps >= eps_high)[0]
    if len(cand_indices) == 0:
        return []
        
    cand_steps = steps[cand_indices]
    spikes = []
    last = -1e30
    
    for s in cand_steps:
        if s - last >= min_gap_steps:
            spikes.append(s)
            last = s
            
    return spikes

def process_group(api, key, run_names, window=1000):
    # Sort runs by length (shortest first => base, then -2, then -2-3)
    # This assumes consistent naming convention where suffixes add length
    sorted_names = sorted(run_names, key=len)
    
    full_df = pd.DataFrame()
    last_max_step = 0
    
    print(f"Processing group {key}: {sorted_names}")
    
    for i, r_name in enumerate(sorted_names):
        runs = api.runs(path=f"{ENTITY}/{PROJECT}", filters={"display_name": r_name})
        if len(runs) == 0:
            print(f"  Run {r_name} not found on W&B")
            continue
        run = runs[0]
        
        print(f"  Fetching {r_name}...")
        df = fetch_run_history(run)
        if df.empty: continue
        
        if df.empty: continue
        
        if df.empty: continue
        
        # Logic for stitching:
        # If this is NOT the last run, it's considered "failed" or "superseded".
        # User Rule 1: "discard the data after the last pruning of a failed run."
        # User Rule 2: "a 'continue' pruning iteration starts where the previous one left off."
        
        is_last_run = (i == len(sorted_names) - 1)
        
        if not is_last_run:
             # Find spikes in this run
             spikes = detect_spike_steps(df[EPSILON_KEY].fillna(0), df[STEP_KEY])
             
             if len(spikes) > 0:
                 last_spike = spikes[-1]
                 print(f"    Run {r_name} is partial. Truncating at last spike {last_spike} (discarding incomplete iter).")
                 # Truncate strictly BEFORE the last spike (start of next iter)
                 # So we keep Iterations 0..N-1. Iteration N (started by last_spike) is discarded.
                 df = df[df[STEP_KEY] < last_spike]
             else:
                 print(f"    Run {r_name} has no valid iterations (no spikes). Discarding entirely.")
                 df = pd.DataFrame()
        
        if df.empty: continue

        # Strict Offset Logic
        offset = 0
        if not full_df.empty:
            last_max_step = full_df[STEP_KEY].max()
            # New run starts at Step 0 (usually). Map 0 -> LastMaxStep + 1?
            # Or just LastMaxStep.
            # "Continue starts where previous left off".
            offset = last_max_step
            
            if offset > 0:
                 print(f"    Offsetting {r_name} by {offset:.1f}")
                 df[STEP_KEY] += offset
        
        full_df = pd.concat([full_df, df])
            
        last_max_step = full_df[STEP_KEY].max()
        
    if full_df.empty: return []
    
    full_df = full_df.sort_values(STEP_KEY).drop_duplicates(subset=[STEP_KEY], keep='last')
    
    # Fill Data
    full_df[EPSILON_KEY] = full_df[EPSILON_KEY].ffill().fillna(0)
    full_df[RETURN_KEY] = full_df[RETURN_KEY].ffill()
    
    # Detect Iterations
    spikes = detect_spike_steps(full_df[EPSILON_KEY], full_df[STEP_KEY])
    boundaries = list(spikes) + [full_df[STEP_KEY].max() + 1]
    
    results = []
    # Identify Env from key
    env = key.split("_")[0] # SpaceInvaders_s1 -> SpaceInvaders
    # Seed
    # seed = key.split("_s")[1]
    
    # Calculate Averages per Iteration
    
    for k in range(len(boundaries)-1):
        start = boundaries[k]
        end = boundaries[k+1]
        
        duration = end - start
        
        # Window logic
        if isinstance(window, float) and window < 1.0:
            w_size = max(1, int(duration * window))
        else:
            w_size = int(window)
        
        w_start = max(start, end - w_size)
        
        mask = (full_df[STEP_KEY] >= w_start) & (full_df[STEP_KEY] < end)
        seg = full_df[mask]
        
        avg = np.nan
        if not seg.empty and RETURN_KEY in seg.columns:
            avg = seg[RETURN_KEY].mean()
            
        # Fallback to last known if empty window (due to sparse logging)
        if pd.isna(avg) and RETURN_KEY in full_df.columns:
             pre_mask = full_df[STEP_KEY] < w_start
             if pre_mask.any():
                 avg = full_df.loc[pre_mask, RETURN_KEY].iloc[-1]
                 
        results.append({
            "env": env,
            "iteration": k,
            "avg_return": avg
        })
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="../data/lottery/atari-lottery[baseline:1].csv")
    parser.add_argument("--output", default="results_baselines.csv")
    parser.add_argument("--window", type=float, default=1000, help="Window size (int) or fraction (float < 1.0)")
    args = parser.parse_args()
    
    api = wandb.Api(timeout=60)
    
    groups = parse_baseline_runs(args.csv)
    
    all_results = []
    for key, names in groups.items():
        res = process_group(api, key, names, window=args.window)
        all_results.extend(res)
        
    df_out = pd.DataFrame(all_results)
    df_out.to_csv(args.output, index=False)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
