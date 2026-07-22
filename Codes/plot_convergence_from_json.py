import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
# CONFIGURATION
# ===============================================================
# Which algorithms do you want to include in the graph?
TARGET_ALGOS = [
    "CCE-TDCCEA", 
    "TD-CCEA", 
    "SA-TDCCEA"
]

# Set your graph title and map name here
GRAPH_TITLE = "Convergence Curves — Mean ± 1 Std Dev Across Runs\n(32x32 Chokepoint Gauntlet, Rotating Opponent Schedule)"

# Colors and styles for the plot
CURVE_COLORS = {
    "CCE-TDCCEA":    "#1f77b4",  # Blue
    "TD-CCEA":       "#9467bd",  # Purple
    "SA-TDCCEA":     "#ff7f0e",  # Orange
    "Standard CCEA": "#d62728",  # Red
    "DE/rand/1/bin": "#2ca02c",  # Green
    "CMA-ES":        "#8c564b",  # Brown
}
CURVE_STYLES = {
    "CCE-TDCCEA":    "-",
    "TD-CCEA":       ":",
    "SA-TDCCEA":     "--",
    "Standard CCEA": "--",
}
CURVE_MARKERS = {
    "CCE-TDCCEA":    "o",
    "TD-CCEA":       "D",
    "SA-TDCCEA":     "s",
}

# ===============================================================
# PLOTTING SCRIPT
# ===============================================================
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "convergence_raw.json")
    
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        print("Make sure you have run COMPARE_ALL_NEW.py at least once to generate the data.")
        return

    print(f"Loading data from {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5))
    
    # We need to figure out the total generations from the data
    total_gens = 0
    for algo, runs in data.items():
        if runs:
            total_gens = max(total_gens, max(len(r) for r in runs))
            
    if total_gens == 0:
        print("Error: JSON file contains no run data.")
        return

    for algo_name, runs in data.items():
        if algo_name not in TARGET_ALGOS or not runs:
            continue

        # Pad shorter runs to total_gens with their last value
        padded = []
        for run in runs:
            if len(run) < total_gens:
                run = run + [run[-1]] * (total_gens - len(run))
            padded.append(run[:total_gens])

        arr  = np.array(padded)            # shape: (n_runs, n_generations)
        raw_mean = arr.mean(axis=0)
        raw_std  = arr.std(axis=0)
        
        # Apply a 7-generation moving average to smooth the rotating schedule spikes
        window = 7
        mean = np.copy(raw_mean)
        std = np.copy(raw_std)
        for i in range(len(raw_mean)):
            start = max(0, i - window + 1)
            mean[i] = np.mean(raw_mean[start:i+1])
            std[i]  = np.mean(raw_std[start:i+1])

        gens = np.arange(1, total_gens + 1)

        color  = CURVE_COLORS.get(algo_name,  "#555555")
        style  = CURVE_STYLES.get(algo_name,  "-")
        marker = CURVE_MARKERS.get(algo_name, "o")

        # Plot mean line — markers every 5 generations to avoid clutter
        ax.plot(gens, mean,
                linestyle=style,
                color=color,
                linewidth=2.0,
                marker=marker,
                markevery=5,
                markersize=6,
                label=f"{algo_name} (n={len(runs)})")

        # Shaded ± 1 std-dev band
        ax.fill_between(gens, mean - std, mean + std,
                         alpha=0.15, color=color)

    # ── Axes and labels ──────────────────────────────────────────
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Win Rate During Training (%)", fontsize=12)
    ax.set_title(GRAPH_TITLE, fontsize=11)
    
    ax.set_xlim(1, total_gens)
    ax.set_ylim(0, 105)
    
    # Try to make nice ticks (every 5 if small, every 10 if large)
    tick_step = 5 if total_gens <= 50 else 10
    ax.set_xticks(range(0, total_gens + 1, tick_step))
    ax.set_yticks(range(0, 101, 10))
    
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    out_png = os.path.join(current_dir, "custom_convergence_plot.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Success! Custom graph saved to -> {out_png}")

if __name__ == "__main__":
    main()
