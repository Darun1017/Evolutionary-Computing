import matplotlib.pyplot as plt
import numpy as np

# Data from the 25 runs
data = {
    "CCE-TDCCEA": [71.43, 91.4, 85.71, 74.29, 85.71, 88.57, 94.29, 57.14, 82.86, 68.57, 85.71, 97.14, 80, 100, 77.14, 82.86, 71.43, 82.86, 71.43, 82.86, 77.14, 82.86, 51.43, 82.86, 77.14],
    "Standard CCEA": [100, 85.7, 62.86, 71.43, 37.14, 85.71, 91.43, 85.71, 77.14, 65.71, 71.43, 37.14, 85.71, 85.71, 0, 100, 40, 68.57, 45.71, 74.29, 0, 80, 60, 74.29, 0],
    "CMA-ES": [68.57, 62.9, 14.29, 28.57, 54.29, 57.14, 40, 45.71, 48.57, 40, 65.71, 28.57, 37.14, 28.57, 31.43, 42.86, 28.57, 28.57, 28.57, 14.29, 28.57, 48.57, 8.57, 14.29, 28.57],
    "DE/rand/1/bin": [80, 42.9, 68.57, 62.86, 71.43, 85.71, 85.71, 85.71, 42.86, 62.86, 71.43, 77.14, 80, 80, 68.57, 48.57, 74.29, 85.71, 100, 42.86, 85.71, 62.86, 80, 42.86, 85.71],
    "PSO": [57.14, 31.4, 80, 45.71, 22.86, 48.57, 45.71, 54.29, 45.71, 34.29, 37.14, 28.57, 45.71, 22.86, 8.57, 28.57, 48.57, 62.86, 28.57, 71.43, 54.29, 25.71, 40, 71.43, 54.29],
    "TD-CCEA": [14.29, 14.3, 51.43, 68.57, 22.86, 37.14, 17.14, 20, 48.57, 42.86, 80, 45.71, 40, 34.29, 37.14, 40, 42.86, 48.57, 68.57, 45.71, 28.57, 40, 20, 45.71, 28.57],
    "SA-TDCCEA": [57.14, 71.4, 45.71, 71.43, 28.57, 25.71, 42.86, 74.29, 54.29, 42.86, 28.57, 54.29, 45.71, 45.71, 8.57, 34.29, 42.86, 42.86, 51.43, 28.57, 77.14, 45.71, 42.86, 28.57, 77.14],
    "Hill Climbing": [85.71, 85.7, 28.57, 85.71, 42.86, 85.71, 85.71, 82.86, 68.57, 68.57, 65.71, 77.14, 85.71, 74.29, 100, 62.86, 48.57, 65.71, 100, 42.86, 57.14, 71.43, 71.43, 42.86, 57.14],
    "GESY": [57.14, 51.4, 57.14, 31.43, 25.71, 57.14, 22.86, 31.43, 34.29, 48.57, 25.71, 42.86, 40, 37.14, 40, 37.14, 48.57, 48.57, 54.29, 42.86, 57.14, 22.86, 42.86, 42.86, 57.14],
    "SA-PDCoEA": [85.71, 71.4, 71.43, 71.43, 77.14, 51.43, 68.57, 40, 94.29, 80, 65.71, 65.71, 88.57, 62.86, 42.86, 68.57, 57.14, 80, 40, 80, 57.14, 85.71, 62.86, 80, 57.14]
}

runs = range(1, 26)

fig, ax = plt.subplots(figsize=(14, 7))

# Plot all baselines as thin, semi-transparent lines
for algo, scores in data.items():
    if algo != "CCE-TDCCEA":
        ax.plot(runs, scores, color='gray', alpha=0.3, linewidth=1.5, marker='.', markersize=6)

# Plot top baselines with distinct colors to distinguish them slightly from the grey mass
ax.plot(runs, data["DE/rand/1/bin"], color='#3498db', alpha=0.5, linewidth=2, label="DE/rand/1/bin (2nd Best)")
ax.plot(runs, data["Hill Climbing"], color='#2ecc71', alpha=0.5, linewidth=2, label="Hill Climbing (3rd Best)")

# Plot CCE-TDCCEA as a thick, bold red line
ax.plot(runs, data["CCE-TDCCEA"], color='#e74c3c', alpha=1.0, linewidth=4, marker='o', markersize=8, label="CCE-TDCCEA (Proposed)")

# Formatting
ax.set_xticks(runs)
ax.set_xticklabels([f"Run {i}" for i in runs], rotation=45, ha='right', fontsize=10)
ax.set_ylabel("Win Rate (%)", fontsize=14, fontweight='bold')
ax.set_title("Algorithm Stability Comparison Across 25 Independent Runs", fontsize=18, fontweight='bold', pad=20)
ax.set_ylim(0, 105)
ax.set_xlim(0.5, 25.5)

# Add gridlines
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.grid(axis='x', linestyle=':', alpha=0.4)

# Legend
ax.legend(loc='lower right', fontsize=12, framealpha=0.9, shadow=True)

plt.tight_layout()

# Save the figure
output_path = "c:/Users/Pradeep S/Desktop/Semester VI/Project Phase - 1/Code/algorithm_stability_linegraph.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Graph successfully generated and saved to: {output_path}")
