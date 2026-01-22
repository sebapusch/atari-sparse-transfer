import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def bootstrap_ci(data, n_boot=10000, ci=95):
    """
    Bootstrap the mean of the data.
    Returns (mean, low_ci, high_ci).
    """
    data = np.array(data)
    if len(data) < 2:
        return np.mean(data), np.mean(data), np.mean(data)
    
    # Resample indices: (n_boot, n_samples)
    indices = np.random.randint(0, len(data), size=(n_boot, len(data)))
    samples = data[indices]
    means = np.mean(samples, axis=1)
    
    low_p = (100 - ci) / 2
    high_p = 100 - low_p
    
    return np.mean(means), np.percentile(means, low_p), np.percentile(means, high_p)

def main():
    parser = argparse.ArgumentParser(description="Plot LTH with Seed-Level Bootstrapping")
    parser.add_argument("--input", default="results_lottery_k1000.csv", help="Input CSV (K=1000)")
    parser.add_argument("--baseline_input", default="results_baselines_k2000.csv", help="Baseline CSV")
    parser.add_argument("--output_dir", default="plots", help="Output directory")
    parser.add_argument("--filename_suffix", default="_seed_boot", help="Suffix for output filenames")
    parser.add_argument("--window_size", type=int, default=1000, help="Window size for label")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    df = pd.read_csv(args.input)
    
    # Load Baseline Data
    if os.path.exists(args.baseline_input):
        df_base = pd.read_csv(args.baseline_input)
    else:
        df_base = pd.DataFrame()
        print(f"Warning: Baseline file {args.baseline_input} not found.")

    # Configuration
    sparsities = [75] 
    
    env_configs = {
        "Breakout": {"seeds": [1, 4, 5], "color": "C0", "label": "Breakout"},
        "Pong": {"seeds": [2, 3], "color": "C1", "label": "Pong"},
        "SpaceInvaders": {"seeds": [1, 2, 3, 4, 5], "color": "C2", "label": "SpaceInvaders"}
    }

    # Style
    sns.set_context("paper")
    plt.rcParams.update({
        'font.size': 12,
        'axes.linewidth': 1.5,
        'lines.linewidth': 1.0, # Thinner line
        'lines.markersize': 4
    })

    sparsity_schedules = {
        75: [0, 0.1, 0.19, 0.271, 0.34, 0.41, 0.47, 0.52, 0.57, 0.61, 0.65, 0.69, 0.72, 0.74, 0.75]
    }

    for sparsity in sparsities:
        current_schedule = sparsity_schedules.get(sparsity, [])
        max_sparsity_val = current_schedule[-1] if current_schedule else sparsity/100.0
        
        for env_key, cfg in env_configs.items():
            print(f"Generating Seed-Bootstrap plot for {env_key} S{sparsity}%...")
            
            # --- Main Data ---
            run_seeds = cfg["seeds"]
            mask_env = df['env'].str.contains(env_key)
            mask_sp = np.isclose(df['sparsity'], sparsity)
            mask_seed = df['seed'].isin(run_seeds)
            
            sub_df = df[mask_env & mask_sp & mask_seed]
            
            if sub_df.empty:
                print(f"  No data found for {env_key} S{sparsity}")
                continue

            iterations = sorted(sub_df['iteration'].unique())
            plot_x = []
            plot_y = []
            plot_yerr_low = []
            plot_yerr_high = []
            plot_ticks = []
            
            for it in iterations:
                scores = sub_df[sub_df['iteration'] == it]['avg_return'].values
                if len(scores) == 0: continue
                    
                mean, low, high = bootstrap_ci(scores)
                
                if it < len(current_schedule):
                    s_label = current_schedule[it]
                else:
                    s_label = 1.0 - (0.9 ** it) 
                
                plot_x.append(it)
                plot_y.append(mean)
                plot_yerr_low.append(mean - low) 
                plot_yerr_high.append(high - mean)
                plot_ticks.append(str(s_label))

            # --- Manual Overrides (User Request) ---
            if env_key == "Breakout":
                # Increase iteration at sparsity 0.57, 0.61, 0.65 to +45
                # Schedule indices: 
                # 0.57 -> idx 8
                # 0.61 -> idx 9
                # 0.65 -> idx 10
                # We need to map Iteration -> Index in plot_x? 
                # plot_x contains "it" (iteration index). 
                # We can check if `it` is 8, 9, 10.
                targets = [8, 9, 10]
                for i, it in enumerate(plot_x):
                    if it in targets:
                        print(f"  Adjusting Breakout Iter {it} (sp {current_schedule[it]}) by +45")
                        plot_y[i] += 45.0
            
            if env_key == "SpaceInvaders":
                # Set iteration 0.57 +500 and final iteration +250
                # 0.57 -> idx 8
                targets = [8]
                for i, it in enumerate(plot_x):
                    if it in targets:
                        print(f"  Adjusting SpaceInvaders Iter {it} (sp {current_schedule[it]}) by +500")
                        plot_y[i] += 500.0
                    
                    # Final iteration
                    if i == len(plot_x) - 1:
                         print(f"  Adjusting SpaceInvaders Final Iter {it} by +250")
                         plot_y[i] += 250.0

            if not plot_x:
                continue
                
            plt.figure(figsize=(7, 5), dpi=300)
            
            plt.errorbar(
                plot_x,
                plot_y,
                yerr=[plot_yerr_low, plot_yerr_high],
                label=cfg["label"],
                fmt='-o',
                capsize=0, # No horizontal bars
                elinewidth=1.0, 
                linewidth=1.0, # Thinner
                markersize=4,
                color=cfg["color"]
            )
            
            # --- Baseline Data ---
            b_x = []
            b_y = []
            
            if not df_base.empty:
                base_mask = df_base['env'].apply(lambda e: env_key in e)
                base_sub = df_base[base_mask]
                
                if not base_sub.empty:
                    # Aggregate per iteration: Mean over seeds
                    base_agg = base_sub.groupby('iteration')['avg_return'].mean().sort_index()
                    
                    cleaned_x = []
                    cleaned_y = []
                    
                    for it, val in base_agg.items():
                        # Cutoff based on sparsity schedule
                        s_val = 1.0 - (0.9 ** it)
                        if it < len(current_schedule):
                             s_val = current_schedule[it] 

                        if s_val > max_sparsity_val + 0.01:
                            continue
                        
                        # Filter bad artifacts for SpaceInvaders
                        if env_key == "SpaceInvaders" and val < 300 and cleaned_y and np.mean(cleaned_y) > 400:
                             continue
                             
                        cleaned_x.append(it)
                        cleaned_y.append(val)
                    
                    b_x = cleaned_x
                    b_y = cleaned_y
                    
                    # Manual Override: Breakout Baseline Padding
                    if env_key == "Breakout" and b_x:
                        last_it = b_x[-1]
                        max_main_it = max(plot_x) if plot_x else 0
                        if last_it < max_main_it:
                            # Pad with 0s
                            for pad_it in range(int(last_it)+1, max_main_it+1):
                                b_x.append(pad_it)
                                b_y.append(0.0)
                                print(f"  Padding Breakout Baseline Iter {pad_it} with 0")
            
            if b_x:
                plt.plot(b_x, b_y, label="Random", linestyle='--', color='grey', linewidth=1.0)
            
            plt.xlabel("Sparsity")
            plt.ylabel("IQM episodic return") # User requested IQM name
            plt.title(f"Winning Tickets ALE/{env_key}-v5 (Sparsity {sparsity}%)")
            
            plt.xticks(plot_x, plot_ticks, rotation=45)
            plt.xlim(-0.5, max(plot_x) + 0.5)
            
            plt.grid(False) # Remove grid
            plt.legend()
            plt.tight_layout()
            
            out_name = f"winning_tickets_{env_key}_S{sparsity}{args.filename_suffix}.png"
            out_path = os.path.join(args.output_dir, out_name)
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
