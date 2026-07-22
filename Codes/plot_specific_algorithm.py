import json
import matplotlib.pyplot as plt
import numpy as np

# Edit this list to contain ONLY the algorithms you want to plot!
ALGORITHMS_TO_PLOT = [
    "CCE-TDCCEA",
    "TD-CCEA",
    "SA-TDCCEA",
    "SA-PDCoEA"
]

def main():
    try:
        with open("convergence_raw.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: convergence_raw.json not found! You must let the benchmark finish first.")
        return

    plt.figure(figsize=(8, 5))

    for algo_name in ALGORITHMS_TO_PLOT:
        if algo_name not in data or len(data[algo_name]) == 0:
            print(f"Skipping {algo_name} (no data found in JSON)")
            continue
            
        runs = data[algo_name]
        
        # Pad shorter runs to 30 generations in case of early stopping
        padded = []
        for run in runs:
            if len(run) < 30:
                run = run + [run[-1]] * (30 - len(run))
            padded.append(run[:30])
            
        arr = np.array(padded)
        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0)
        
        gens = np.arange(1, 31)
        plt.plot(gens, mean_curve, label=algo_name, linewidth=2)
        plt.fill_between(gens, 
                         np.clip(mean_curve - std_curve, 0, 100), 
                         np.clip(mean_curve + std_curve, 0, 100), 
                         alpha=0.15)

    plt.title("Convergence Curves (Specific Algorithms)")
    plt.xlabel("Generation")
    plt.ylabel("Win Rate (%)")
    plt.ylim(-5, 105)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    out_file = "specific_convergence.png"
    plt.savefig(out_file, dpi=300)
    print(f"Successfully generated specific graph: {out_file}")

if __name__ == "__main__":
    main()
