import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed — saves directly to PNG
import matplotlib.pyplot as plt


# ── Configuration ────────────────────────────────────────────────────────────

INPUT_FILE  = "convergence_raw_25_runs.json"
OUTPUT_FILE = "convergence_ablation_tdccea.png"
TOTAL_GENS  = 30
ROLLING_WINDOW = 3     # smooth over 3 generations to reduce opponent-rotation noise

# Only plot these 3 for the ablation study
PLOT_ALGOS = {
    "CCE-TDCCEA": {
        "color":  "#1f4e79",   # deep blue — your main contribution
        "style":  "-",
        "marker": "o",
        "label":  "CCE-TDCCEA",
    },
    "TD-CCEA": {
        "color":  "#8e44ad",   # purple — monolithic trajectory baseline
        "style":  "--",
        "marker": "D",
        "label":  "TD-CCEA",
    },
    "SA-TDCCEA": {
        "color":  "#27ae60",   # green — sequence-adapted baseline
        "style":  "-.",
        "marker": "^",
        "label":  "SA-TDCCEA",
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_data(path):
    with open(path, "r") as f:
        return json.load(f)


def rolling_mean_2d(arr, window):
    """
    Apply a causal rolling mean along the generation axis (axis=1).
    Each generation g is the mean of generations max(0, g-window+1)..g.
    Shape: (n_runs, n_gens) -> (n_runs, n_gens)
    """
    n_runs, n_gens = arr.shape
    out = np.zeros_like(arr, dtype=float)
    for g in range(n_gens):
        start = max(0, g - window + 1)
        out[:, g] = arr[:, start:g+1].mean(axis=1)
    return out


def compute_stats(runs_raw, window):
    """
    Given raw runs (list of lists), apply rolling mean then compute
    mean and std across runs.
    Returns: mean array (n_gens,), std array (n_gens,)
    """
    arr      = np.array(runs_raw, dtype=float)    # (n_runs, n_gens)
    smoothed = rolling_mean_2d(arr, window)        # (n_runs, n_gens)
    return smoothed.mean(axis=0), smoothed.std(axis=0)


# ── Main plot ─────────────────────────────────────────────────────────────────

def main():
    # Load
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path  = os.path.join(script_dir, INPUT_FILE)
    if not os.path.exists(data_path):
        print(f"[ERROR] {INPUT_FILE} not found in {script_dir}")
        print("        Make sure the JSON file is in the same folder as this script.")
        return

    data = load_data(data_path)
    print(f"Loaded {INPUT_FILE}")
    for algo, runs in data.items():
        print(f"  {algo}: {len(runs)} runs")

    gens = np.arange(1, TOTAL_GENS + 1)

    # Build figure
    fig, ax = plt.subplots(figsize=(8, 5))

    for algo_name, style in PLOT_ALGOS.items():
        if algo_name not in data or not data[algo_name]:
            print(f"  [SKIP] {algo_name} — no data")
            continue

        mean, std = compute_stats(data[algo_name], ROLLING_WINDOW)

        # Trim or pad to TOTAL_GENS if needed
        mean = mean[:TOTAL_GENS]
        std  = std[:TOTAL_GENS]

        # Mean line with markers every 5 generations
        ax.plot(
            gens, mean,
            linestyle  = style["style"],
            color      = style["color"],
            linewidth  = 3.5,            # Increased for better visibility
            marker     = style["marker"],
            markevery  = 5,
            markersize = 9,              # Increased for better visibility
            markeredgecolor="white",     # Add white border to markers so they pop
            markeredgewidth=1.5,
            label      = f"{style['label']} (n={len(data[algo_name])})",
            zorder     = 3               # Ensure lines are drawn on top of shading
        )

        # Shaded ±1 std band — clipped to [0, 100]
        ax.fill_between(
            gens,
            np.clip(mean - std, 0, 100),
            np.clip(mean + std, 0, 100),
            alpha = 0.08,                # Reduced alpha so shading doesn't obscure lines
            color = style["color"],
            zorder= 2
        )

        print(f"  Plotted {algo_name}: "
              f"gen1_mean={mean[0]:.1f}  "
              f"gen5_mean={mean[4]:.1f}  "
              f"gen30_mean={mean[-1]:.1f}  "
              f"avg_std={std.mean():.1f}")

    # Axes and labels
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel(
        f"Win Rate During Training",
        fontsize=11
    )
    ax.set_title(
        "Training Convergence — Mean ± 1 Std Dev Across 25 Runs\n"
        "(Second Map, 32×32, K=1 per Generation, Rotating Opponent Schedule)",
        fontsize=11,
    )
    ax.set_xlim(1, TOTAL_GENS)
    ax.set_ylim(0, 105)
    ax.set_xticks(range(0, TOTAL_GENS + 1, 5))
    ax.set_yticks(range(0, 101, 10))
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    fig.tight_layout()

    # Save
    out_path = os.path.join(script_dir, OUTPUT_FILE)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")
    print("Ready to include in paper as Figure 1.")

    # Print summary stats for the paper text
    print("\n-- Stats for paper text ------------------------------")
    print(f"{'Algorithm':<20} {'Early(gen5)':>12} {'Mid(gen15)':>12} {'Final(gen30)':>13} {'AvgStd':>8}")
    for algo_name in PLOT_ALGOS:
        if algo_name not in data or not data[algo_name]:
            continue
        mean, std = compute_stats(data[algo_name], ROLLING_WINDOW)
        print(f"{algo_name:<20} "
              f"{mean[4]:>12.1f} "
              f"{mean[14]:>12.1f} "
              f"{mean[-1]:>13.1f} "
              f"{std.mean():>8.1f}")


if __name__ == "__main__":
    main()
