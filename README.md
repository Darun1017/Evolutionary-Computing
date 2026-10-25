# TD-CCEA: Trajectory-Driven Competitive Co-Evolutionary Algorithm

An evolutionary computing framework that evolves intelligent agents for real-time strategy (RTS) games using a novel **Trajectory-Driven Competitive Co-evolutionary Algorithm (TD-CCEA)** with trajectory-based fitness memory.

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup & Compilation](#setup--compilation)
- [How to Run](#how-to-run)
- [Troubleshooting](#troubleshooting)
- [Algorithm Overview](#algorithm-overview)
- [Opponent Bots](#opponent-bots)
- [Results](#results)
- [StarCraft II Extension](#starcraft-ii-extension)
- [Repository Map](#repository-map)
- [Dependencies](#dependencies)

## Project Structure

```
Evolutionary-Computing/
├── Codes/
│   ├── Code/
│   │   ├── tdccea_evolved_v2.py       # Main TD-CCEA algorithm (rotating opponents)
│   │   ├── benchmark_all.py           # Benchmark: TD-CCEA vs GA vs Hill Climbing vs Random
│   │   ├── SA_TDCCEA.py               # Sensitivity-Adaptive TD-CCEA variant
│   │   ├── head_to_head.py            # Coevolutionary battle: TD-CCEA vs Hill Climbing
│   │   ├── evolve_vs_custom.py        # Evolve counter-strategy against custom weights
│   │   ├── baseline_ga.py             # Baseline: Standard Genetic Algorithm
│   │   ├── baseline_hillclimb.py      # Baseline: Hill Climbing
│   │   ├── baseline_random.py         # Baseline: Random Search
│   │   ├── tdccea_32x32_14strat.py    # TD-CCEA with 14 strategy combinations
│   │   ├── generate_chambers64.py     # Map generator (64x64)
│   │   ├── generate_chambers128.py    # Map generator (128x128)
│   │   └── benchmark_*.png            # Result graphs
│   └── Simulation/
│       └── microrts/
│           ├── lib/                    # Java libraries (microrts JARs)
│           ├── bin/                    # Compiled Java classes (generated)
│           ├── src/
│           │   ├── ai/abstraction/
│           │   │   └── EvolvedBrainAI.java    # Evolved AI (20-weight parameterized)
│           │   └── tests/
│           │       ├── EvolvedBatchGameRunner.java  # Headless batch game server
│           │       ├── EvolvedVisualGameRunner.java  # Visual game replay with GUI
│           │       ├── BatchGameRunner.java          # Basic batch runner
│           │       └── VisualGameRunner.java         # Basic visual runner
│           └── maps/                   # Game maps (chambers32x32.xml etc.)
├── Presentations/                      # LaTeX Beamer slides
├── Research_Papers/                    # Reference papers
└── README.md
```

## Prerequisites

### 1. Java (JDK 8 or higher)

microRTS is a Java-based game engine. You need a JDK installed.

**Download:** https://www.oracle.com/java/technologies/downloads/

Verify installation:
```bash
java -version
javac -version
```

### 2. Python (3.8 or higher)

All evolution scripts are written in Python.

**Download:** https://www.python.org/downloads/

Verify installation:
```bash
python --version
```

### 3. Python Libraries

```bash
pip install matplotlib numpy
```

## Setup & Compilation

The Python evolution scripts communicate with the microRTS Java engine via a persistent subprocess. The Java code **must be compiled before running any Python script**.

### Step 1: Navigate to the microRTS directory

```bash
cd Codes/Simulation/microrts
```

### Step 2: Create the `bin/` output directory

```bash
mkdir bin
```

### Step 3: Compile all Java source files

**On Windows (PowerShell):**
```powershell
Get-ChildItem -Path src -Recurse -Filter "*.java" | ForEach-Object { $_.FullName } | Out-File -FilePath sources_compile.txt -Encoding ASCII
javac -cp "lib/*" -d bin "@sources_compile.txt"
```

**On Linux / macOS:**
```bash
find src -name "*.java" > sources_compile.txt
javac -cp "lib/*" -d bin @sources_compile.txt
```

> **Note:** You may see deprecation/unchecked warnings — these are safe to ignore. Compilation is successful as long as there are no `error:` lines in the output.

### Step 4: Verify compilation

Check that the key class files exist:
```bash
# These three files must exist after compilation:
ls bin/ai/abstraction/EvolvedBrainAI.class
ls bin/tests/EvolvedBatchGameRunner.class
ls bin/tests/EvolvedVisualGameRunner.class
```

On Windows PowerShell:
```powershell
Test-Path bin\ai\abstraction\EvolvedBrainAI.class
Test-Path bin\tests\EvolvedBatchGameRunner.class
Test-Path bin\tests\EvolvedVisualGameRunner.class
```

All three should return `True`.

## How to Run

> **Important:** All Python scripts must be run from the `Codes/Code/` directory.

```bash
cd Codes/Code
```

### Main Evolution (TD-CCEA with Rotating Opponents)
Evolves a 20-weight strategy vector against 7 different opponent bots over 21 generations:
```bash
python -u tdccea_evolved_v2.py
```
**Expected runtime:** ~10–30 minutes. Outputs `evolved_v2_best_weights.txt` and launches a visual replay at the end.

### Benchmark All Algorithms
Compares TD-CCEA (Hybrid + Centroid), Standard GA, Hill Climbing, and Random Search:
```bash
python -u benchmark_all.py
```
**Expected runtime:** ~1–2 hours.  
Generates: `benchmark_winrate.png`, `benchmark_convergence.png`, `benchmark_heatmap.png`, `benchmark_time.png`, `benchmark_results.json`

### Sensitivity-Adaptive TD-CCEA
Tests each weight's influence, then evolves with per-weight mutation rates:
```bash
python -u SA_TDCCEA.py
```

### Head-to-Head Coevolution
TD-CCEA vs Hill Climbing evolving against each other (arms race):
```bash
python -u head_to_head.py
```

### Evolve Against Custom Opponent
Edit the `OPPONENT_WEIGHTS` list in the file, then run:
```bash
python -u evolve_vs_custom.py
```

### Standalone Baselines
```bash
python -u baseline_ga.py          # Standard Genetic Algorithm
python -u baseline_hillclimb.py   # Hill Climbing
python -u baseline_random.py      # Random Search
```

### Demo / Visualization Scripts (no Java required)
```bash
python ea_demo_visual.py          # Basic EA visualization
python tdccea_demo_visual.py      # TD-CCEA concept demo
python tscea_demo_visual.py       # TSCEA concept demo
```

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: EvolvedBatchGameRunner failed to start` | Java classes not compiled | Run the compilation steps above |
| `java.lang.ClassNotFoundException` | `bin/` directory missing or empty | Run `mkdir bin` then compile |
| `FileNotFoundError` on map path | Wrong working directory | Ensure you run Python from `Codes/Code/` |
| `Unexpected response` from Java server | Java process error | Check that `javac` completed without errors |
| `BrokenPipeError` | Java process crashed mid-game | Recompile Java; check for OutOfMemoryError |
| Deprecation warnings during `javac` | Using newer JDK with older API | Safe to ignore; compilation still succeeds |

## Algorithm Overview

TD-CCEA evolves a population of 20 agents, each represented by a 20-dimensional weight vector controlling game behavior:

| # | Weight | Controls |
|---|--------|----------|
| w0 | worker_ratio | Worker production rate |
| w1 | max_workers | Maximum worker count (1-12) |
| w2 | expand_threshold | When to build 2nd base |
| w3-w5 | light/heavy/ranged_pref | Military unit composition |
| w6 | barracks_timing | When to build barracks |
| w7 | aggression | Attack eagerness |
| w8 | attack_threshold | Minimum army before attacking |
| w9 | defend_radius | Patrol distance from base |
| w10 | target_priority | Target workers vs military |
| w11 | defense_trigger | When to pull back to defend |
| w12 | counter_rush | Workers fighting during rush |
| w13 | tech_switch | Counter enemy composition |
| w14 | economy_pressure | Prioritize destroying economy |
| w15 | scout_priority | Early scouting |
| w16 | group_attack | Wait for army to group up |
| w17 | second_barracks | When to build 2nd barracks |
| w18 | worker_fight_ratio | Permanent worker fighters |
| w19 | focus_fire | Concentrate attacks on one target |

### Key Algorithm Steps

1. **Centroid Environment** — compute population average as environment state
2. **Fitness Evaluation** — play K games against rotating opponents
3. **Trajectory Memory** — store (fitness, environment) history per agent
4. **Dynamic Similarity** — weight past fitness by environment similarity
5. **Adaptive Mutation** — losing agents mutate more, winning agents less
6. **Diversity Pruning** — remove clones (similarity > 0.85) and weak agents (e_i < 0.20)
7. **Archive** — store top agents across generations
8. **Breeding** — tournament selection from survivors + archive, Gaussian mutation

### Architecture: Python ↔ Java Communication

```
Python Script                         Java Server
─────────────                         ───────────
tdccea_evolved_v2.py                  EvolvedBatchGameRunner
        │                                     │
        ├── Starts Java subprocess ──────────►│ Prints "READY"
        │                                     │
        ├── Sends: "EVOLVED:w0;w1;...;w19,   │
        │    OpponentType,map,cycles\n" ─────►│ Parses weights
        │                                     │ Creates EvolvedBrainAI
        │                                     │ Runs game loop
        │◄── Receives: "RESULT:winner,cycles" │
        │                                     │
        ├── (repeat for each game)            │
        │                                     │
        └── Sends: "EXIT\n" ────────────────►│ Shuts down
```

## Opponent Bots

The algorithm trains against 7 built-in microRTS strategies:

| Bot | Strategy |
|-----|----------|
| WorkerRush | All workers attack immediately |
| WorkerRushPlusPlus | Improved worker rush |
| LightRush | Fast Light unit rush |
| RangedRush | Ranged unit rush |
| HeavyDefense | Defensive Heavy units |
| EconomyRush | Economy first, then rush |
| EconomyMilitaryRush | Balanced economy + military |

## Results

TD-CCEA consistently achieves **80-100% win rate** against all opponent types, outperforming Standard GA, Hill Climbing, and Random Search baselines.

Result artifacts are generated into the `Codes/` directory:

| Artifact | What it contains |
|----------|------------------|
| `benchmark_results.json` / `benchmark_results_v2.json` | Raw win-rate and timing data per algorithm |
| `benchmark_winrate.png` | Win-rate comparison across algorithms |
| `benchmark_convergence.png` / `convergence_curves.png` | Fitness convergence over generations |
| `benchmark_heatmap.png` | Per-opponent win-rate heatmap |
| `benchmark_time.png` | Wall-clock cost per algorithm |
| `benchmark_winrates.xlsx` | Excel export of win-rate tables |
| `wilcoxon_results.txt` | Wilcoxon signed-rank significance tests vs baselines |
| `ablation_results.txt` / `ABLATION_TEST.txt` | Component ablation study |

> The Wilcoxon and ablation files contain multiple dated runs. Statistical
> significance varies by run and baseline; consult the raw files for the exact
> p-values rather than relying on a single headline figure.

## StarCraft II Extension

The `StarCraft/` directory adapts the TD-CCEA framework from microRTS to
**StarCraft II build-order planning**. It is the most recent line of work and
is an active extension rather than a finished pipeline.

Highlights:

- `StarCraft/Post_Review/td_ccea_starcraft.py` — TD-CCEA adapted for SC2 build orders.
- `StarCraft/protoss_macro_bot.py`, `terran_macro_bot.py`, `terran_bot.py`, `zerg_bot.py` — per-race macro bots.
- `StarCraft/run_cce_tdccea_sc2.py`, `run_protoss_evaluation.py` — evolution / evaluation drivers.
- `StarCraft/watch_match.py`, `watch_ai_vs_ai.py` — match visualization utilities.
- `StarCraft/test_protoss_environment.py`, `test_terran_macro.py` — environment/sanity checks.

## Repository Map

A quick guide to the main entry points under `Codes/` (see the per-file
docstrings for details):

| File / Folder | Role |
|---------------|------|
| `tdccea_evolved_v2.py` | Main TD-CCEA run (rotating opponents) |
| `tdccea_8x8.py` / `tdccea_16x16.py` / `tdccea_32x32.py` | TD-CCEA on different map sizes |
| `tdccea_32x32_14strat.py` | TD-CCEA across 14 strategy combinations |
| `tdccea_blockade.py` / `tdccea_dynamic_env.py` | Dynamic / progressive-blockade environments |
| `CCE_TDCCEA.py` | Cooperative co-evolution (Macro/Micro split) hybrid |
| `SA_TDCCEA.py` / `DP_TDCCEA.py` | Sensitivity-adaptive / diversity-preserving variants |
| `baseline_ga.py` / `baseline_hillclimb.py` / `baseline_random.py` / `baseline_es.py` | Baseline optimizers |
| `benchmark_all.py` / `FULL_BENCHMARK.py` / `COMPARE_ALL.py` | Benchmark harnesses |
| `head_to_head.py` / `HEAD_TO_HEAD_ALL.py` | Coevolutionary head-to-head tournaments |
| `evolve_vs_custom.py` | Evolve a counter-strategy vs custom weights |
| `*_demo_visual.py` | Concept demos (no Java required) |
| `MapGenerators/` | Map generation and visualization scripts |
| `Simulation/microrts/` | Java microRTS engine + `EvolvedBrainAI` |
| `GESY/` | GESY framework outputs |

## Dependencies

Python dependencies are listed in [`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt
```

- **Core:** `numpy`, `matplotlib`
- **Optional:** `openpyxl` (Excel export), `scipy` (SciPy Wilcoxon test), `cma` (CMA-ES baseline)

The Java microRTS engine (JDK 8+) is a separate prerequisite — see
[Prerequisites](#prerequisites) and [Setup & Compilation](#setup--compilation).
