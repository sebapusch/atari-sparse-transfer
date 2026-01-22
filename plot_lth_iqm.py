#!/usr/bin/env python3
"""
Plot LTH IQM with 95% CI.
Grouped by Sparsity Level (One plot per sparsity).
Lines represent Environments.
X-axis: Percent of Weights Remaining (Log Scale).
Y-axis: IQM Episodic Return.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
import seaborn as sns

def compute_iqm(data):
    """Computes the Interquartile Mean (IQM) of the data."""
    if len(data) == 0:
        return np.nan
    sorted_data = np.sort(data)
    n = len(sorted_data)
    lower = int(n * 0.25)
    upper = int(n * 0.75)
    if lower >= upper:
        return np.mean(sorted_data)
    return np.mean(sorted_data[lower:upper])

def bootstrap_iqm_ci(data, n_boot=1000, ci=95):
    """Computes bootstrapped CI for IQM."""
    if len(data) < 2:
        return np.nan, np.nan
        
    boot_means = []
    rng = np.random.default_rng(seed=42)
    # Vectorized bootstrap for speed
    # But usually data is small (3-5 seeds), so loop is fine
    for _ in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_means.append(compute_iqm(sample))
        
    alpha = (100 - ci) / 2
    lower = np.percentile(boot_means, alpha)
    upper = np.percentile(boot_means, 100 - alpha)
    return lower, upper

def main():
    parser = argparse.ArgumentParser(description="Plot LTH IQM with CI")
    parser.add_argument("--input", default="results_lottery.csv", help="Input CSV")
    parser.add_argument("--output_dir", default="plots", help="Output directory")
    parser.add_argument("--filename_suffix", default="", help="Suffix for output filenames")
    parser.add_argument("--env_filter", default=None, help="Only plot specific env")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.input)
    
    # Define combinations
    sparsities = [75]
    
    # Env Definitions
    env_configs = {
        "Breakout": {"seeds": [1, 4, 5], "color": "C0", "label": "Breakout"},
        "Pong": {"seeds": [2, 3], "color": "C1", "label": "Pong"},
        "SpaceInvaders": {"seeds": [1, 2, 3, 4, 5], "color": "C2", "label": "SpaceInvaders"}
    }
    
    if args.env_filter:
        env_configs = {k: v for k, v in env_configs.items() if k == args.env_filter}

    sns.set_context("paper") 
    plt.rcParams.update({
        'font.size': 12,
        'axes.linewidth': 1.5,
        'lines.linewidth': 1.5,
        'lines.markersize': 4
    })

    # Load Baseline Data
    baseline_csv = "results_baselines.csv"
    if os.path.exists(baseline_csv):
        df_base = pd.read_csv(baseline_csv)
    else:
        df_base = pd.DataFrame()

    # Sparsity Schedules (provided by user)
    sparsity_schedules = {
        60: [0, 0.1, 0.19, 0.27, 0.34, 0.4, 0.47, 0.52, 0.57, 0.6],
        75: [0, 0.1, 0.19, 0.271, 0.34, 0.41, 0.47, 0.52, 0.57, 0.61, 0.65, 0.69, 0.72, 0.74, 0.75]
    }

    for sparsity in sparsities:
        current_schedule = sparsity_schedules.get(sparsity, [])
        max_sparsity_val = current_schedule[-1] if current_schedule else sparsity/100.0
        
        for env_key, cfg in env_configs.items():
            print(f"Generating plot for {env_key} S{sparsity}%...")
            
            plt.figure(figsize=(7, 5), dpi=300)
            
            # --- Main Data ---
            run_seeds = cfg["seeds"]
            
            def filter_row(row):
                if env_key not in row['env']: return False
                try:
                    if float(row['sparsity']) != float(sparsity): return False
                except: return False
                if row['seed'] not in run_seeds: return False
                return True
                
            mask = df.apply(filter_row, axis=1)
            sub_df = df[mask]
            
            if sub_df.empty:
                print(f"  No data for {env_key} S{sparsity}")
                has_main_data = False
            else:
                has_main_data = True
                
            # Aggregate by iteration
            iterations = sorted(sub_df['iteration'].unique())
            plot_data = []
            
            for it in iterations:
                vals = sub_df[sub_df['iteration'] == it]['avg_return'].dropna().to_numpy()
                if len(vals) == 0: continue
                
                iqm = compute_iqm(vals)
                if len(vals) >= 2:
                    low, high = bootstrap_iqm_ci(vals)
                    yerr_low = iqm - low
                    yerr_high = high - iqm
                else:
                    yerr_low = 0
                    yerr_high = 0
                
                # Map iteration to Sparsity
                if it < len(current_schedule):
                    sparsity_val_label = current_schedule[it]
                else:
                    sparsity_val_label = 1.0 - (0.9 ** it)
                
                plot_data.append({
                    "iteration": it, 
                    "sparsity_label": sparsity_val_label,
                    "iqm": iqm,
                    "yerr_low": yerr_low,
                    "yerr_high": yerr_high
                })
                
            if has_main_data and plot_data:
                pdf = pd.DataFrame(plot_data)
                
                eb = plt.errorbar(
                    pdf["iteration"], 
                    pdf["iqm"], 
                    yerr=[pdf["yerr_low"], pdf["yerr_high"]],
                    label=cfg["label"],
                    fmt='-o',
                    capsize=0, 
                    elinewidth=1.0, 
                    linewidth=1.0, 
                    markersize=3, 
                    color=cfg["color"]
                )
                # eb[-1][0].set_alpha(0.3) 

            # --- Baseline Data ---
            b_x = []
            b_y = []
            
            if not df_base.empty:
                # Filter for this env (loose match on env name)
                base_mask = df_base['env'].apply(lambda e: env_key in e)
                base_sub = df_base[base_mask]
                
                if not base_sub.empty:
                    # Aggregate by iteration (mean across all seeds in CSV)
                    # "If the exact sparsity level does not match exactly you should still aggregate per pruning iterations"
                    # We group by 'iteration' which is the pruning round index.
                    base_agg = base_sub.groupby('iteration')['avg_return'].mean().sort_index()
                    
                    for it, val in base_agg.items():
                        # Calculate implied sparsity for cutoff check
                        # Random baseline follows same schedule index? 
                        # Assuming index alignment is correct per User instruction.
                        
                        s_val = 1.0 - (0.9 ** it)
                        if it < len(current_schedule):
                             s_val = current_schedule[it] 

                        # Cutoff at target sparsity
                        if s_val > max_sparsity_val + 0.01:
                            continue
                            
                        # No artificial decay logic unless requested. 
                        # "ensure how you aggregate is correct" implies using data as-is.
                        
                        b_x.append(it)
                        b_y.append(val)
            
            if b_x:
                plt.plot(b_x, b_y, label="Random", linestyle='--', color='grey', linewidth=1.5)
            plt.xlabel("Sparsity")
            plt.ylabel("IQM episodic return")
            
            plt.xticks(
                range(len(current_schedule)), 
                [str(x) for x in current_schedule], 
                rotation=45
            )
            plt.xlim(-0.5, len(current_schedule) - 0.5)
            
            plt.grid(True, which='major', alpha=0.5)
            plt.title(f"Winning Tickets {env_key} (Sparsity {sparsity}%)")
            
            plt.legend()
            plt.tight_layout()
            
            out_name = f"winning_tickets_{env_key}_S{sparsity}{args.filename_suffix}.png"
            out_path = os.path.join(args.output_dir, out_name)
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"Saved {out_path}")


        # --- Generate Normalized Comparison Plot for this Sparsity ---
        # (Normalization code unchanged, ensure baseline is NOT included here)
        print(f"Generating NORMALIZED plot for Sparsity {sparsity}%...")
        plt.figure(figsize=(7, 5), dpi=300)
        
        has_data = False
        
        for env_key, cfg in env_configs.items():
            run_seeds = cfg["seeds"]
            
            def filter_row(row):
                if env_key not in row['env']: return False
                try:
                    if float(row['sparsity']) != float(sparsity): return False
                except: return False
                if row['seed'] not in run_seeds: return False
                return True
                
            mask = df.apply(filter_row, axis=1)
            sub_df = df[mask]
            
            if sub_df.empty: continue
            
            iterations = sorted(sub_df['iteration'].unique())
            iqm_curve = []
            valid_its = []
            
            for it in iterations:
                vals = sub_df[sub_df['iteration'] == it]['avg_return'].dropna().to_numpy()
                if len(vals) == 0: continue
                iqm = compute_iqm(vals)
                iqm_curve.append(iqm)
                valid_its.append(it)
                
            if not iqm_curve: continue
            
            peak_val = max(iqm_curve)
            if abs(peak_val) < 1e-6:
                norm_factor = 1.0
            else:
                norm_factor = peak_val
                
            plot_x = []
            plot_y = []
            plot_yerr_low = []
            plot_yerr_high = []
            
            for idx, it in enumerate(valid_its):
                vals = sub_df[sub_df['iteration'] == it]['avg_return'].dropna().to_numpy()
                norm_vals = vals / norm_factor
                
                iqm = compute_iqm(norm_vals) 
                if len(norm_vals) >= 2:
                    low, high = bootstrap_iqm_ci(norm_vals)
                    yerr_low = iqm - low
                    yerr_high = high - iqm
                else:
                    yerr_low = 0
                    yerr_high = 0
                
                # X is simply iteration index
                plot_x.append(it)
                plot_y.append(iqm)
                plot_yerr_low.append(yerr_low)
                plot_yerr_high.append(yerr_high)
            
            has_data = True
            eb = plt.errorbar(
                plot_x, plot_y, 
                yerr=[plot_yerr_low, plot_yerr_high],
                label=cfg["label"],
                fmt='-o',
                capsize=0, 
                elinewidth=1.0, 
                linewidth=1.0, 
                markersize=3, 
                color=cfg["color"]
            )
            eb[-1][0].set_alpha(0.3)
            
        if has_data:
            plt.xlabel("Sparsity")
            plt.ylabel("Normalized Return (Rel. to Peak)")
            
            # Set Ticks to be all schedule points
            if current_schedule:
                tick_indices = range(len(current_schedule))
                tick_labels = [str(x) for x in current_schedule]
                plt.xticks(tick_indices, tick_labels, rotation=45)
                plt.xlim(-0.5, len(current_schedule) - 0.5)

            plt.grid(True, which='major', alpha=0.5)
            plt.title(f"Normalized Winning Tickets (Sparsity {sparsity}%)")
            plt.legend()
            plt.tight_layout()
            
            out_name = f"winning_tickets_normalized_S{sparsity}.png"
            out_path = os.path.join(args.output_dir, out_name)
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"Saved {out_path}")
        else:
            plt.close()

if __name__ == "__main__":
    main()
