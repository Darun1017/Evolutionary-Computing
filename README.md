# TD-CCEA: Time-Dependent Cooperative Coevolutionary Algorithm for microRTS

An evolutionary computing framework that evolves intelligent agents for real-time strategy (RTS) games using a novel **Time-Dependent Cooperative Coevolutionary Algorithm (TD-CCEA)** with trajectory-based fitness memory.

## Project Structure

```
Evolutionary-Computing/
├── Codes/
│   └── Code/
│       ├── tdccea_evolved_v2.py       # Main TD-CCEA algorithm (rotating opponents)
│       ├── benchmark_all.py           # Benchmark: TD-CCEA vs GA vs Hill Climbing vs Random
│       ├── SA_TDCCEA.py               # Sensitivity-Adaptive TD-CCEA variant
│       ├── head_to_head.py            # Coevolutionary battle: TD-CCEA vs Hill Climbing
│       ├── evolve_vs_custom.py        # Evolve counter-strategy against custom weights
│       ├── baseline_ga.py             # Baseline: Standard Genetic Algorithm
│       ├── baseline_hillclimb.py      # Baseline: Hill Climbing
│       ├── baseline_random.py         # Baseline: Random Search
│       ├── tdccea_32x32_14strat.py    # TD-CCEA with 14 strategy combinations
│       ├── generate_chambers64.py     # Map generator (64x64)
│       ├── generate_chambers128.py    # Map generator (128x128)
│       └── benchmark_*.png            # Result graphs
├── Presentations/                     # LaTeX Beamer slides
├── Research_Papers/                   # Reference papers
└── README.md
```

## Prerequisites

### 1. Java (JDK 8 or higher)

microRTS is a Java-based game engine. You need Java installed.

**Download:** https://www.oracle.com/java/technologies/downloads/

Verify installation:
```bash
java -version
```

### 2. Python (3.8 or higher)

All evolution scripts are written in Python.

**Download:** https://www.python.org/downloads/

Verify installation:
```bash
python --version
```

### 3. microRTS Game Engine

The microRTS simulator must be set up at:
```
Project Phase - 1/
├── Simulation/
│   └── microrts/
│       ├── lib/            # Java libraries (microrts.jar etc.)
│       ├── bin/            # Compiled Java classes
│       ├── src/            # Java source code (includes EvolvedBrainAI.java)
│       └── maps/           # Game maps (chambers32x32.xml etc.)
└── Code/                   # Python scripts (copy of Codes/Code)
```

**To compile the Java code:**
```bash
cd Simulation/microrts
javac -cp "lib/*" -d bin src/**/*.java
```

### 4. Python Libraries

```bash
pip install matplotlib numpy
```

## How to Run

### Main Evolution (TD-CCEA with Rotating Opponents)
Evolves a 20-weight strategy vector against 7 different opponent bots over 30 generations:
```bash
cd Code
python -u tdccea_evolved_v2.py
```

### Benchmark All Algorithms
Compares TD-CCEA (Hybrid + Centroid), Standard GA, Hill Climbing, and Random Search:
```bash
python -u benchmark_all.py
```
Generates: `benchmark_winrate.png`, `benchmark_convergence.png`, `benchmark_heatmap.png`, `benchmark_time.png`

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
