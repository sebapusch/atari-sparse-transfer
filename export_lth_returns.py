#!/usr/bin/env python3
"""
Export grouped LTH returns.
Scans WandB, merges continuations, detects iterations, calculates windowed avg return with ffill.
"""

import argparse
import logging
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wandb
import wandb.errors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Constants
EPSILON_KEY = "charts/epsilon"
RETURN_KEY = "charts/episodic_return"
STEP_KEY = "_step"

def parse_run_name(name: str) -> Optional[Tuple[str, str, str, bool]]:
    pattern = r"^atari_ddqn_lth_sweep-(.+)-S([\d\.]+)-s(\d+)(-2)?$"
    match = re.match(pattern, name)
    if match:
        env, sparsity, seed, continuation = match.groups()
        return env, sparsity, seed, bool(continuation)
    return None

def detect_spike_steps(eps_series, step_series, eps_high=0.9, min_gap_steps=500_000):
    steps = step_series.to_numpy()
    eps = eps_series.to_numpy()
    cand = steps[eps >= eps_high]
    if len(cand) == 0: return []
    spikes = []
    last = -1e30
    for s in cand:
        if s - last >= min_gap_steps:
            spikes.append(float(s))
            last = float(s)
    return spikes

def fetch_run_history(run, max_step=100_000_000):
    """Fetches full history using run.history() with large samples limit."""
    try:
        # Request a huge number of samples to avoid downsampling
        df = run.history(keys=[STEP_KEY, EPSILON_KEY, RETURN_KEY], pandas=True, samples=max_step)
        
        if df.empty: return df
        
        # Numeric conversion
        for col in [STEP_KEY, EPSILON_KEY, RETURN_KEY]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle _step if it's an index or col
        if STEP_KEY not in df.columns and df.index.name == STEP_KEY:
             df = df.reset_index()
             
        df = df.dropna(subset=[STEP_KEY]).sort_values(STEP_KEY)
        return df
    except Exception as e:
        logger.error(f"Error fetching history for run {run.name}: {e}")
        return pd.DataFrame()

def process_grouped_run(base_run_name, runs, window, min_gap_steps):
    sorted_runs = sorted(runs, key=lambda r: 1 if r.name.endswith("-2") else (2 if r.name.endswith("-3") else 0))
    
    full_df = pd.DataFrame()
    last_max_step = 0
    
    for i, run in enumerate(sorted_runs):
        logger.info(f"  Fetching history for {run.name}...")
        df = fetch_run_history(run)
        if df.empty: continue
            
        offset = 0
        curr_min = df[STEP_KEY].min()
        if i > 0 and curr_min < last_max_step:
            offset = last_max_step
            logger.info(f"    Offsetting {run.name} by {offset}")
            
        if offset > 0:
            df[STEP_KEY] += offset
            
        full_df = pd.concat([full_df, df])
        last_max_step = df[STEP_KEY].max()

    if full_df.empty: return []

    full_df = full_df.sort_values(STEP_KEY).drop_duplicates(subset=[STEP_KEY], keep='last')
    
    if EPSILON_KEY not in full_df.columns:
        logger.warning(f"  Missing epsilon for {base_run_name}")
        return []
        
    # Fill epsilon for spike detection
    full_df[EPSILON_KEY] = full_df[EPSILON_KEY].fillna(method='ffill').fillna(0)
    
    # Fill returns for valid window averaging (carry forward last known return)
    if RETURN_KEY in full_df.columns:
        full_df[RETURN_KEY] = full_df[RETURN_KEY].fillna(method='ffill')
    
    spikes = detect_spike_steps(full_df[EPSILON_KEY], full_df[STEP_KEY], min_gap_steps=min_gap_steps)
    boundaries = list(spikes) + [full_df[STEP_KEY].max() + 1]
    
    results = []
    env, sparsity, seed, _ = parse_run_name(base_run_name)
    
    for k in range(len(boundaries)-1):
        start = boundaries[k]
        end = boundaries[k+1]
        
        # Determine window size based on fraction
        # Use args? Passed as window/last_frac
        # We need to change the function signature to accept last_frac logic
        # For now, let's assume 'window' arg is actually 'last_frac' if it's float < 1?
        # No, better to be explicit. But based on user request "last 0.01", I will use a float param.
        
        duration = end - start
        
        # Default to 0.01 if passed as such
        if isinstance(window, float) and window < 1.0:
            w_size = max(1, int(duration * window))
            used_frac = window
        else:
            w_size = int(window) # Fixed steps
            used_frac = w_size / duration if duration > 0 else 0
            
        w_start = max(start, end - w_size)
        
        mask = (full_df[STEP_KEY] >= w_start) & (full_df[STEP_KEY] < end)
        seg = full_df[mask]
        
        avg = np.nan
        if not seg.empty and RETURN_KEY in seg.columns:
            avg = seg[RETURN_KEY].mean()
            
        # If still nan (because no data at all in window even with ffill?), try to use last known value before window
        if pd.isna(avg) and RETURN_KEY in full_df.columns:
            # Check last value before w_start
            pre_mask = full_df[STEP_KEY] < w_start
            if pre_mask.any():
                last_val = full_df.loc[pre_mask, RETURN_KEY].iloc[-1]
                avg = last_val
                
        results.append({
            "run_name": base_run_name,
            "env": env,
            "sparsity": sparsity,
            "seed": seed,
            "iteration": k,
            "avg_return": avg,
            "iter_start_step": start,
            "iter_end_step": end,
            "window_size": w_size,
            "used_frac": used_frac
        })
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window", type=float, default=1000, help="Fraction (if < 1) or Step count (if >= 1)")
    parser.add_argument("--min_gap_steps", type=int, default=500000)
    args = parser.parse_args()
    
    api = wandb.Api(timeout=60)
    logger.info(f"Scanning {args.project}...")
    runs = api.runs(path=args.project, filters={"display_name": {"$regex": "atari_ddqn_lth_sweep.*"}})
    
    grouped = defaultdict(list)
    for r in runs:
        res = parse_run_name(r.name)
        if res:
            env, sp, sd, _ = res
            base = f"atari_ddqn_lth_sweep-{env}-S{sp}-s{sd}"
            grouped[base].append(r)
            
    all_res = []
    for base, g in grouped.items():
        all_res.extend(process_grouped_run(base, g, args.window, args.min_gap_steps))
        
    pd.DataFrame(all_res).to_csv(args.output, index=False)
    logger.info(f"Saved {args.output}")

if __name__ == "__main__":
    main()
