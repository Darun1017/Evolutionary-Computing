# How to Run — TD-CCEA / CCE-TDCCEA Benchmark Suite

## Prerequisites

1. **Java JDK 8+** — https://www.oracle.com/java/technologies/downloads/
2. **Python 3.8+** — https://www.python.org/downloads/
3. **Python libraries:**
   ```
   pip install matplotlib numpy
   ```

## Setup

### Step 1: Clone the repository
```
git clone https://github.com/Darun1017/Evolutionary-Computing.git
```

### Step 2: Set up microRTS game engine
Place the microRTS simulator as a sibling directory:
```
YourFolder/
├── Simulation/
│   └── microrts/
│       ├── lib/        ← Java libraries (microrts.jar etc.)
│       ├── bin/        ← Compiled Java classes
│       ├── src/        ← Java source (includes EvolvedBrainAI.java)
│       └── maps/       ← Game maps (chambers32x32.xml)
└── Evolutionary-Computing/
    └── Codes/          ← Python scripts (this repo)
```

### Step 3: Compile the Java code
```
cd Simulation/microrts
javac -cp "lib/*" -d bin src/**/*.java
```

## Running the Benchmarks

Open a terminal, navigate to the Codes folder, and run:

### Main Benchmark (2-Way Co-Evolution) — RECOMMENDED
```
cd Evolutionary-Computing/Codes
python -u COMPARE_ALL.py
```
- Compares: CCE-TDCCEA (2-way), TD-CCEA Centroid, SA-TDCCEA, DP-TDCCEA, Hill Climbing
- Runtime: ~15-20 minutes

### 3-Way Co-Evolution Benchmark
```
python -u COMPARE_3WAY.py
```
- Compares: CCE-TDCCEA (3-way), TD-CCEA Centroid, SA-TDCCEA, DP-TDCCEA, Hill Climbing
- Runtime: ~20-30 minutes

### Baseline-Only Comparison (no co-evolution)
```
python -u BASELINE_COMPARE.py
```
- Compares: TD-CCEA, SA-TDCCEA, DP-TDCCEA, Hill Climbing
- Runtime: ~5-8 minutes

### Legacy Benchmark (with graphs)
```
python -u benchmark_all.py
```
- Generates PNG graphs in the same folder
- Runtime: ~15 minutes

## Notes
- The `-u` flag gives unbuffered output so you can see progress live
- Each run produces different results due to randomness — run multiple times for averages
- CCE-TDCCEA takes longer (~600s) because it evaluates with multiple partners
