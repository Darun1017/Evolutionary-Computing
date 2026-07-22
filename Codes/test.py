import numpy as np
from scipy.stats import wilcoxon

# Your exact win rate data across 25 runs
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

proposed_algo = "CCE-TDCCEA"
proposed_data = np.array(data[proposed_algo])

print(f"{'Algorithm':<15} | {'Mean ± SD (%)':<15} | {'p-value vs ' + proposed_algo:<25} | {'Significance'}")
print("-" * 75)

# Calculate and print stats for the proposed algorithm first
mean_prop = np.mean(proposed_data)
std_prop = np.std(proposed_data, ddof=1)
print(f"{proposed_algo:<15} | {mean_prop:>5.2f} ± {std_prop:>5.2f} | {'-':<25} | {'-'}")

# Iterate through baselines
for algo, scores in data.items():
    if algo == proposed_algo:
        continue
    
    baseline_data = np.array(scores)
    mean_val = np.mean(baseline_data)
    std_val = np.std(baseline_data, ddof=1)
    
    # Wilcoxon Signed-Rank Test (two-sided)
    # Using default mode which applies continuity and exact/approx logic based on ties
    stat, p_value = wilcoxon(proposed_data, baseline_data, zero_method='wilcox')
    
    # Format p-value output
    p_str = "< 0.001" if p_value < 0.001 else f"{p_value:.4f}"
    sig = "Significant" if p_value < 0.05 else "Not Significant"
    
    print(f"{algo:<15} | {mean_val:>5.2f} ± {std_val:>5.2f} | {p_str:<25} | {sig}")