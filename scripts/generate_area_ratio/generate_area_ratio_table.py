#!/usr/bin/env python3
"""
generate_area_ratio_table.py

Reads a CSV produced by generate_area_ratio.py and produces:
1. A LaTeX table with color coding and significance markers.
2. Bar plots with error bars.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def generate_latex_table(df: pd.DataFrame, caption="Caption", label="tab:placeholder"):
    # Pivot: Index=Source, Columns=Target, Values=Ratio (and Significant for marker)
    # We need to construct the cell string "Value^marker" and handle colors.
    
    # Get unique envs to ensure order
    envs = sorted(list(set(df["Source"].unique()) | set(df["Target"].unique())))
    
    # Create a matrix for the table body
    matrix = pd.DataFrame(index=envs, columns=envs, dtype=object)
    
    for _, row in df.iterrows():
        src = row["Source"]
        tgt = row["Target"]
        val = row["Ratio"]
        sig = row["Significant"]
        
        # Color
        # Intensity: abs(val) mapped to 0-100.
        # Cap at 100? Assuming values are typically < 1 or around 1.
        # User example: -0.7890 -> 79.
        intensity = int(round(abs(val) * 100))
        intensity = min(intensity, 100) # Cap at 100
        
        color = "red" if val < 0 else "green"
        cell_color = f"\\cellcolor{{{color}!{intensity}}}"
        
        # Text
        text = f"{val:.4f}"
        
        # Marker
        # "dagger or double dagger († / ‡)"
        # Use dagger for significant? Or double dagger?
        # User said: "distinguish between significant/non significant use a dagger or double dagger... indicate statistical significance"
        # Usually * is significant.
        # Let's use ^\dagger for significant, nothing for non-significant?
        # Or maybe dagger for p<0.05 and double dagger for p<0.01? We only have binary significance.
        # Let's use dagger for significant.
        marker = r"^{\dagger}" if sig else ""
        
        cell_content = f"{cell_color}${text}{marker}$"
        matrix.at[src, tgt] = cell_content

    # Build LaTeX string
    # Header
    cols_align = "|" + "|".join(["c"] * (len(envs) + 1)) + "|"
    latex = []
    latex.append(r"\begin{table}[]")
    latex.append(r"    \centering")
    latex.append(r"    \scalebox{1.5}{")
    latex.append(f"    \\begin{{tabular}}{{{cols_align}}}")
    latex.append(r"        \hline")
    
    # Header Row
    header_cells = [""] + [e for e in envs]
    latex.append("        " + " & ".join(header_cells) + r" \\")
    latex.append(r"        \hline \hline")
    
    # Data Rows
    for src in envs:
        row_cells = [src]
        for tgt in envs:
            if src == tgt:
                row_cells.append("") # Empty diagonal? Or "-"
            else:
                val = matrix.at[src, tgt]
                if pd.isna(val):
                    row_cells.append("")
                else:
                    row_cells.append(val)
        latex.append("        " + " & ".join(row_cells) + r" \\")
        latex.append(r"        \hline")

    latex.append(r"    \end{tabular}")
    latex.append(r"    }")
    latex.append(f"    \\caption{{{caption}}}")
    latex.append(f"    \\label{{{label}}}")
    latex.append(r"\end{table}")
    
    return "\n".join(latex)

def generate_plots(df: pd.DataFrame, out_dir: str, title_suffix=""):
    # Bar plot per target task
    targets = df["Target"].unique()
    
    # RL-style colors (e.g., derived from seaborn 'deep' or similar)
    # A professional muted blue
    bar_color = "#4c72b0" 
    
    for tgt in targets:
        subset = df[df["Target"] == tgt].copy()
        if subset.empty:
            continue
            
        # Sort sources
        subset = subset.sort_values("Source")
        
        sources = subset["Source"].tolist()
        ratios = subset["Ratio"].tolist()
        ci_lo = subset["CI_Lo"].tolist()
        ci_hi = subset["CI_Hi"].tolist()
        
        # Error bars: [val - lo, hi - val]
        yerr = np.array([
            [r - lo for r, lo in zip(ratios, ci_lo)],
            [hi - r for hi, r in zip(ci_hi, ratios)]
        ])
        
        plt.figure(figsize=(6, 4))
        
        # Plot
        # Use zorder to ensure grid is behind
        plt.bar(sources, ratios, yerr=yerr, capsize=5, 
                color=bar_color, edgecolor='black', linewidth=0.8, zorder=3)
                
        plt.axhline(0, color='black', linewidth=1.0, zorder=4)
        
        plt.xlabel("Source Task", fontsize=11)
        plt.ylabel("Area Ratio", fontsize=11)
        plt.title(f"Transfer to {tgt}", fontsize=12, fontweight='bold')
        
        # RL Style: horizontal grid only, minimal spines
        plt.grid(axis='y', linestyle=':', alpha=0.6, zorder=0)
        # Turn off top and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        safe_tgt = tgt.replace(" ", "_").lower()
        # Use existing suffix in filename diff for uniqueness if batch processing
        safe_suffix = title_suffix.replace(" ", "_").replace("->", "to").lower().replace("(", "").replace(")", "")
        fname = f"bar_{safe_tgt}_{safe_suffix}.png"
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {path}")

def process_file(csv_path: str, base_out_dir: str):
    """Process a single CSV file."""
    if not os.path.exists(csv_path):
        print(f"Error: File not found {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return
    
    # Ensure req columns
    req_cols = ["Source", "Target", "Ratio", "CI_Lo", "CI_Hi", "Significant"]
    if not all(c in df.columns for c in req_cols):
        print(f"Skipping {csv_path}: Missing required columns {req_cols}")
        return
        
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    # Determine output directory
    # User said: "make a directory corresponding to the filename in out_dir"
    target_dir = os.path.join(base_out_dir, base_name)
    os.makedirs(target_dir, exist_ok=True)
    
    # Generate Table
    caption = f"Area Ratios for {base_name.replace('_', ' ')}"
    label = f"tab:{base_name}"
    
    latex_code = generate_latex_table(df, caption=caption, label=label)
    
    tex_path = os.path.join(target_dir, base_name + ".tex")
    
    with open(tex_path, "w") as f:
        f.write(latex_code)
    
    print(f"Processed {base_name}")
    print(f"  Saved LaTeX table to: {tex_path}")
    
    # Generate Plots
    generate_plots(df, target_dir, title_suffix=f"({base_name})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to input CSV file or directory")
    parser.add_argument("--out_dir", default="../plots/area_ratios", help="Base directory for output")
    args = parser.parse_args()
    
    if os.path.isdir(args.input_path):
        # Batch processing
        print(f"Processing directory: {args.input_path}")
        files = [f for f in os.listdir(args.input_path) if f.endswith(".csv")]
        files.sort()
        
        if not files:
            print("No csv files found.")
            return

        for fname in files:
            path = os.path.join(args.input_path, fname)
            process_file(path, args.out_dir)
            
    else:
        # Single file
        process_file(args.input_path, args.out_dir)

if __name__ == "__main__":
    main()
