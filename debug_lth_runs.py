import wandb
import pandas as pd
import numpy as np
import argparse
import sys

# Constants
EPSILON_KEY = "charts/epsilon"
STEP_KEY = "_step"

def detect_spike_steps(eps_series, step_series, eps_high=0.9, min_gap_steps=500_000):
    steps = step_series.to_numpy()
    eps = eps_series.to_numpy()
    cand_indices = np.where(eps >= eps_high)[0]
    if len(cand_indices) == 0:
        return []
    cand_steps = steps[cand_indices]
    spikes = []
    last = -1e30
    for s in cand_steps:
        if s - last >= min_gap_steps:
            spikes.append(float(s))
            last = float(s)
    return spikes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--run_filter", default="atari_ddqn_lth_sweep-Pong-v5-S60-s1")
    args = parser.parse_args()

    api = wandb.Api()
    print(f"Searching for runs matching: {args.run_filter} in {args.project}")
    runs = api.runs(path=args.project, filters={"display_name": {"$regex": args.run_filter + ".*"}})
    
    print(f"Found {len(runs)} runs:")
    for r in runs:
        print(f"  - {r.name} (ID: {r.id}, State: {r.state})")
        
    # Pick the first matching base run and check its history
    base_run = None
    for r in runs:
        if r.name == args.run_filter:
            base_run = r
            break
            
    if not base_run:
        print("Base run not explicitly found, processing all found runs individually...")
        
    for r in runs:
        print(f"\nAnalyzing {r.name}...")
        try:
            hist = r.history(keys=[STEP_KEY, EPSILON_KEY], pandas=True, samples=10000000)
            if hist.empty:
                print("  History empty.")
                continue
            
            if EPSILON_KEY not in hist.columns:
                print(f"  {EPSILON_KEY} not found.")
                continue
                
            max_eps = hist[EPSILON_KEY].max()
            print(f"  Max Epsilon: {max_eps}")
            print(f"  Steps: {hist[STEP_KEY].min()} to {hist[STEP_KEY].max()}")
            print(f"  Rows: {len(hist)}")
            
            spikes = detect_spike_steps(hist[EPSILON_KEY], hist[STEP_KEY])
            print(f"  Spikes detected ({len(spikes)}): {spikes}")
            
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()
