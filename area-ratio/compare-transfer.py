import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("returns_pong.csv", low_memory=False)

# Cleanup
df["step"] = pd.to_numeric(df["step"], errors="coerce")
df["episodic_return"] = pd.to_numeric(df["episodic_return"], errors="coerce")
df = df.dropna(subset=["step", "episodic_return"]).copy()
df["step"] = df["step"].astype(int)

# Keep only transfer curves: LTH (dense) vs GMP transfer (gmp), for sources Pong and Pong
df = df[
    (df["project"] == "lth-transfer")
    & (df["source"].isin(["Breakout-v5", "SpaceInvaders-v5"]))
    & (df["pruning"].isin(["dense"]))
    & (df["source_sparsity"].isin([60, 75]))
].copy()

# Parameters
MAX_STEP = 10_000_000
COARSE_BIN = 50_000
SMOOTH_WINDOW_BINS = 10
N_BOOT = 200  # faster bootstrap

df = df[df["step"] <= MAX_STEP].copy()
df["bin_step"] = (df["step"] // COARSE_BIN) * COARSE_BIN

# Aggregate within-run: one value per run per bin
per_run = (
    df.groupby(["source", "pruning", "run_name", "source_sparsity", "seed", "bin_step"], as_index=False, dropna=False)["episodic_return"]
      .mean()
)

def iqm_trimmed(values: np.ndarray) -> float:
    """IQM via symmetric trimming of 25% from each side (works well for small n)."""
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
    # vectorized bootstrap sampling
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = values[idx]
    samples.sort(axis=1)
    k = int(np.floor(0.25 * n))
    mid = samples[:, k:n - k] if (n - 2 * k) > 0 else samples
    stats = mid.mean(axis=1)
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return center, float(lo), float(hi)

rng = np.random.default_rng(123)

rows = []
for (source, pruning, step, source_sparsity), grp in per_run.groupby(["source", "pruning", "bin_step", "source_sparsity"], dropna=False):
    vals = grp["episodic_return"].to_numpy(dtype=float)
    center, lo, hi = bootstrap_iqm_fast(vals, N_BOOT, rng)
    rows.append({
        "source": source,
        "pruning": pruning,
        "source_sparsity": source_sparsity,
        "step": int(step),
        "iqm": center,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_runs": int(len(vals)),
    })

curves = pd.DataFrame(rows).dropna(subset=["iqm"]).sort_values(["source", "pruning", "step", "source_sparsity"]).copy()

# Smooth (center and CI bounds) for visual clarity
def smooth(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("step").copy()
    for c in ["iqm", "ci_lo", "ci_hi"]:
        g[f"{c}_s"] = g[c].rolling(SMOOTH_WINDOW_BINS, min_periods=1).mean()
    return g

curves_s = curves.groupby(["source", "pruning", "source_sparsity"], group_keys=False).apply(smooth)

# Plot
plt.figure(figsize=(11, 6.2))

label_map = {
        ("Breakout-v5", "dense", 75): "no pruning (source: ALE/Beakout-v5, 75%)",
    ("SpaceInvaders-v5", "dense", 60): "no pruning (source: ALE/SpaceInvaders-v5, 60%)",
}

for (source, pruning, source_sparsity), g in curves_s.groupby(["source", "pruning", "source_sparsity"]):
    g = g.dropna(subset=["iqm_s"])
    x = g["step"].to_numpy(dtype=float) / 1_000_000.0  # Millions
    y = g["iqm_s"].to_numpy(dtype=float)
    lo = g["ci_lo_s"].to_numpy(dtype=float)
    hi = g["ci_hi_s"].to_numpy(dtype=float)

    if label_map.get((source, pruning, source_sparsity)) is None:
        continue

    plt.plot(x, y, label=label_map.get((source, pruning, source_sparsity), f"{pruning} ({source})"))
    if not (np.all(np.isnan(lo)) or np.all(np.isnan(hi))):
        plt.fill_between(x, lo, hi, alpha=0.15)

plt.xlabel("Steps (Millions)")
plt.ylabel("IQM")
plt.title("ALE/Pong-v5 - LTH transfer (95% CI)")
plt.legend()
plt.tight_layout()

out_path = "ale_pong_transfer_breakout-vs-spaceinvaders.png"
plt.savefig(out_path, dpi=400, bbox_inches="tight")
plt.show()

out_path

