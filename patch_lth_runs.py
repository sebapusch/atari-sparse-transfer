import wandb
import pandas as pd
import numpy as np
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STEP_KEY = "_step"
EPSILON_KEY = "charts/epsilon"
RETURN_KEY = "charts/episodic_return"

def fetch_history_robust(run):
    logger.info(f"Scanning history for {run.name} (robust)...")
    try:
        # scan_history guarantees full data
        history_scan = run.scan_history(keys=[STEP_KEY, EPSILON_KEY, RETURN_KEY], page_size=10000)
        rows = []
        c = 0
        for row in history_scan:
            rows.append(row)
            c += 1
            if c % 50000 == 0:
                logger.info(f"  {c} rows...")
        
        df = pd.DataFrame(rows)
        if df.empty: return df
        for col in [STEP_KEY, EPSILON_KEY, RETURN_KEY]:
             if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=[STEP_KEY]).sort_values(STEP_KEY)
        return df
    except Exception as e:
        logger.error(f"Failed {run.name}: {e}")
        return pd.DataFrame()

def detect_spike_steps(eps_series, step_series):
    steps = step_series.to_numpy()
    eps = eps_series.to_numpy()
    cand = steps[eps >= 0.9]
    if len(cand) == 0: return []
    spikes = []
    last = -1e30
    for s in cand:
        if s - last >= 500000:
            spikes.append(float(s))
            last = float(s)
    return spikes

def main():
    # Patch specific runs that look suspicious
    runs_to_patch = [
        # Pong S60 s2 (stops at 9? Maybe s2-2 has more?)
        "atari_ddqn_lth_sweep-Pong-v5-S60-s2", 
        # Breakout S60 s3 (stops at 5?)
        "atari_ddqn_lth_sweep-Breakout-v5-S60-s3"
    ]
    
    api = wandb.Api()
    project = "sebapusch-university-of-groningen/atari-lottery"
    
    new_results = []
    
    for base_name in runs_to_patch:
        logger.info(f"Processing group {base_name}...")
        # Find runs: base + continuation
        # Regex escape? Just simple prefix match and filter
        runs = api.runs(path=project, filters={"display_name": {"$regex": f"^{base_name}(-[0-9]+)?$"}})
        
        if not runs:
            logger.warning(f"No runs found for {base_name}")
            continue
            
        # Sort: base, -2, -3...
        # Run names: name, name-2, name-3
        def sort_key(r):
            if r.name == base_name: return 0
            # name-2 -> 2
            suffix = r.name.replace(base_name, "")
            if suffix.startswith("-"):
                try:
                    return int(suffix[1:])
                except:
                    return 999
            return 999
            
        sorted_runs = sorted(runs, key=sort_key)
        
        full_df = pd.DataFrame()
        last_max_step = 0
        
        for i, run in enumerate(sorted_runs):
            df = fetch_history_robust(run)
            if df.empty: continue
            
            # Offset logic
            curr_min = df[STEP_KEY].min()
            offset = 0
            if i > 0 and curr_min < last_max_step:
                offset = last_max_step
                logger.info(f"  Offsetting {run.name} by {offset}")
                
            if offset > 0:
                df[STEP_KEY] += offset
                
            full_df = pd.concat([full_df, df])
            last_max_step = df[STEP_KEY].max()
            
        if full_df.empty: continue
        full_df = full_df.sort_values(STEP_KEY).drop_duplicates(subset=[STEP_KEY], keep='last')
        
        # Detect iterations
        if EPSILON_KEY not in full_df.columns:
            logger.warning("No epsilon")
            continue
            
        full_df[EPSILON_KEY] = full_df[EPSILON_KEY].fillna(method='ffill').fillna(0)
        spikes = detect_spike_steps(full_df[EPSILON_KEY], full_df[STEP_KEY])
        boundaries = list(spikes) + [full_df[STEP_KEY].max() + 1]
        
        # Parse params
        # base_name like atari_ddqn_lth_sweep-<env>-S<sparsity>-s<seed>
        # Manual parse
        parts = base_name.split("-S")
        left = parts[0] # atari...sweep-<env>
        right = parts[1] # <sparsity>-s<seed>
        
        env = left.replace("atari_ddqn_lth_sweep-", "")
        sparsity_str = right.split("-s")[0]
        seed_str = right.split("-s")[1]
        
        for k in range(len(boundaries)-1):
            start = boundaries[k]
            end = boundaries[k+1]
            window = 1000
            w_start = end - window
            
            mask = (full_df[STEP_KEY] >= w_start) & (full_df[STEP_KEY] < end)
            seg = full_df[mask]
            
            avg = np.nan
            if not seg.empty and RETURN_KEY in seg.columns:
                avg = seg[RETURN_KEY].mean()
                
            new_results.append({
                "run_name": base_name,
                "env": env,
                "sparsity": sparsity_str,
                "seed": seed_str,
                "iteration": k,
                "avg_return": avg,
                "iter_start_step": start,
                "iter_end_step": end,
                "window_size": window
            })
            
    if new_results:
        patch_df = pd.DataFrame(new_results)
        # Load old CSV
        try:
            old_df = pd.read_csv("results_lottery.csv")
            # Remove entries for patched runs
            for patched_run in runs_to_patch:
                old_df = old_df[old_df["run_name"] != patched_run]
            
            # Combine
            final_df = pd.concat([old_df, patch_df], ignore_index=True)
            # Sort
            final_df.to_csv("results_lottery.csv", index=False)
            logger.info("Patched results_lottery.csv")
        except Exception as e:
            logger.error(f"Failed to patch CSV: {e}")
            patch_df.to_csv("patch_results.csv", index=False)
    else:
        logger.info("No data generated for patch.")

if __name__ == "__main__":
    main()
