"""
Ticket existence plot (fraction-based tail averaging)

How to run:

python3 plot_lth_ticket_exists_from_epsilon.py \
  --return_csv data_expected_return/breakout_S60_s3.csv \
  --epsilon_csv data_epsilon/breakout_S60_s3.csv \
  --eps_high 0.9 \
  --min_gap_steps 1000000 \
  --last_frac 0.1 \
  --frac_mode points
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def pick_metric_col(df, contains):
    cand = [c for c in df.columns if contains in c and "__MIN" not in c and "__MAX" not in c]
    if not cand:
        raise ValueError(f"Could not find column containing '{contains}'. Columns: {list(df.columns)}")
    return cand[0]

def load_series(csv_path, step_col="Step", value_contains=None, value_col=None):
    df0 = pd.read_csv(csv_path)
    if step_col not in df0.columns:
        raise ValueError(f"{csv_path}: step_col '{step_col}' not found. Columns: {list(df0.columns)}")

    if value_col is None:
        if value_contains is None:
            raise ValueError("Provide either value_col or value_contains")
        value_col = pick_metric_col(df0, value_contains)

    df = df0[[step_col, value_col]].copy()
    df.columns = ["step", "value"]
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("step").reset_index(drop=True)
    return df

def detect_spike_steps(eps_df, eps_high=0.9, min_gap_steps=500_000):
    steps = eps_df["step"].to_numpy()
    eps = eps_df["value"].to_numpy()

    cand = steps[eps >= eps_high]
    if len(cand) == 0:
        return np.array([], dtype=float)

    spikes = []
    last = -1e30
    for s in cand:
        if s - last >= min_gap_steps:
            spikes.append(float(s))
            last = float(s)
    return np.array(spikes, dtype=float)

def summarize_return_by_start_intervals_fraction(ret_df, starts, last_frac=0.1, frac_mode="points"):
    """
    Iteration i is [starts[i], starts[i+1]) with a final tail [starts[-1], end].

    Compute avg episodic return over the LAST FRACTION of each iteration.

    frac_mode:
      - "points": last ceil(frac * N_points) logged points in the iteration (recommended)
      - "steps":  last frac of env-step span in that iteration
    """
    if not (0.0 < last_frac <= 1.0):
        raise ValueError("--last_frac must be in (0, 1].")

    ret_steps = ret_df["step"].to_numpy()
    ret_vals  = ret_df["value"].to_numpy()

    intervals = []
    for i in range(len(starts) - 1):
        intervals.append((starts[i], starts[i+1]))
    intervals.append((starts[-1], ret_steps.max() + 1))

    rows = []
    for i, (a, b) in enumerate(intervals):
        mask = (ret_steps >= a) & (ret_steps < b)

        if not np.any(mask):
            rows.append({
                "iteration": i,
                "iter_start_step": float(a),
                "iter_end_step": float(a),
                "avg_return_tail": np.nan,
                "end_return": np.nan,
                "n_points": 0,
                "tail_points_used": 0,
                "tail_frac": float(last_frac),
                "frac_mode": frac_mode
            })
            continue

        seg_steps = ret_steps[mask]
        seg_vals  = ret_vals[mask]
        last_step = float(seg_steps.max())
        end_ret   = float(seg_vals[-1])
        n = len(seg_vals)

        if frac_mode == "points":
            tail_n = int(np.ceil(last_frac * n))
            tail_n = max(tail_n, 1)
            tail_vals = seg_vals[-tail_n:]
            avg_tail = float(np.mean(tail_vals))
            tail_points_used = tail_n

        elif frac_mode == "steps":
            span = float(last_step - float(seg_steps.min()))
            # if span is ~0 (rare), fall back to points
            if span <= 0:
                tail_n = max(int(np.ceil(last_frac * n)), 1)
                tail_vals = seg_vals[-tail_n:]
                avg_tail = float(np.mean(tail_vals))
                tail_points_used = tail_n
            else:
                tail_start = last_step - last_frac * span
                wmask = seg_steps >= tail_start
                if not np.any(wmask):
                    wmask = np.ones_like(seg_steps, dtype=bool)
                tail_vals = seg_vals[wmask]
                avg_tail = float(np.mean(tail_vals))
                tail_points_used = int(np.sum(wmask))

        else:
            raise ValueError("--frac_mode must be 'points' or 'steps'.")

        rows.append({
            "iteration": i,
            "iter_start_step": float(a),
            "iter_end_step": float(last_step),
            "avg_return_tail": avg_tail,
            "end_return": end_ret,
            "n_points": int(mask.sum()),
            "tail_points_used": int(tail_points_used),
            "tail_frac": float(last_frac),
            "frac_mode": frac_mode
        })

    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--return_csv", required=True)
    ap.add_argument("--epsilon_csv", required=True)
    ap.add_argument("--step_col", default="Step")
    ap.add_argument("--return_col", default=None)
    ap.add_argument("--epsilon_col", default=None)
    ap.add_argument("--eps_high", type=float, default=0.9)
    ap.add_argument("--min_gap_steps", type=float, default=500_000)

    # NEW: fraction-based tail averaging (ticket existence)
    ap.add_argument("--last_frac", type=float, default=0.1,
                    help="Fraction of each iteration to average over at the end (e.g., 0.1 = last 10%).")
    ap.add_argument("--frac_mode", choices=["points", "steps"], default="points",
                    help="How to define the last fraction: by logged points (recommended) or by env-step span.")

    ap.add_argument("--use_end_return", action="store_true",
                    help="Plot end_return instead of avg_return_tail (not recommended for ticket existence).")

    args = ap.parse_args()

    # Output directories
    plots_dir = "plots"
    summaries_dir = "summaries"
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    # Base filename (NO suffixes)
    base = os.path.splitext(os.path.basename(args.return_csv))[0]

    ret_df = load_series(
        args.return_csv,
        step_col=args.step_col,
        value_contains="charts/episodic_return",
        value_col=args.return_col
    )
    eps_df = load_series(
        args.epsilon_csv,
        step_col=args.step_col,
        value_contains="charts/epsilon",
        value_col=args.epsilon_col
    )

    starts = detect_spike_steps(eps_df, eps_high=args.eps_high, min_gap_steps=args.min_gap_steps)
    print(f"[epsilon] starts detected: {len(starts)}")
    print("[epsilon] start steps:", starts)

    summ = summarize_return_by_start_intervals_fraction(
        ret_df, starts,
        last_frac=args.last_frac,
        frac_mode=args.frac_mode
    )
    summ = summ.sort_values("iteration").reset_index(drop=True)

    metric = "end_return" if args.use_end_return else "avg_return_tail"
    x = summ["iteration"].to_numpy()
    y = summ[metric].to_numpy()

    print("x:", x.tolist())
    print("y:", y.tolist())

    # Plot
    plt.figure(figsize=(11, 4))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.xticks(x)

    for xi, yi in zip(x, y):
        plt.annotate(str(int(xi)), (xi, yi),
                     textcoords="offset points", xytext=(0, 8), ha="center")

    plt.xlabel("LTH/IMP iteration (from epsilon spikes as START boundaries)")
    plt.ylabel(
        "Final performance" +
        (f" (end return)" if args.use_end_return else f" (avg last {args.last_frac:g} of iteration)")
    )
    plt.title("Ticket existence: end-of-iteration performance per IMP iteration")
    plt.grid(True, alpha=0.3)

    # force y-axis to start at 0
    plt.ylim(bottom=0)

    plt.tight_layout()

    plot_path = os.path.join(plots_dir, f"{base}.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    summary_path = os.path.join(summaries_dir, f"{base}.csv")
    summ.to_csv(summary_path, index=False)

    print("Saved:")
    print(f"  {plot_path}")
    print(f"  {summary_path}")

if __name__ == "__main__":
    main()