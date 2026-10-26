# StarCraft II — CCE-TDCCEA + SA-TDCCEA Implementation

An evolutionary co-evolutionary framework that optimises a **35D continuous genome** against the StarCraft II built-in AI using **Cooperative Co-Evolutionary TD-CCEA (CCE-TDCCEA)** with **Sensitivity-Adaptive mutation (SA-TDCCEA)**.

---

## File Overview

| File | Role |
|------|------|
| `config.py` | All hyperparameters, gene names, race, difficulty schedule |
| `evaluator.py` | Frame-by-frame APU / PBR / RUR metric tracker |
| `bot.py` | `EvolvedSC2Bot(BotAI)` — 35D genome → SC2 API actions |
| `cce_tdccea.py` | `SubAgent` + `CCEACore` — 3-population co-evolution |
| `sa_profiler.py` | Phase 1 offline sensitivity profiler |
| `main.py` | Orchestration, generational loop, benchmarking |
| `results/` | Output JSONs: elite genomes, log, sensitivity scores |

---

## Prerequisites

```bash
pip install burnysc2
```

StarCraft II must be installed. Set the `SC2PATH` environment variable if it is not in the default location:

```powershell
# Windows (PowerShell)
$env:SC2PATH = "C:\Program Files (x86)\StarCraft II"
```

---

## Quick-Start

### 1. Verify the pipeline (single game, random genome)
```bash
cd StarCraft_Implementation
python main.py --dry-run
```
Expected output: one 8-minute game completes, fitness metrics printed.

### 2. Run SA profiling only
```bash
python sa_profiler.py
```
Tests all 35 genes at 0.0 and 1.0, saves `results/sensitivity_scores_sc2.json`.

### 3. Full evolution run (SA profiling + 5 generations + benchmark)
```bash
python main.py
```

### 4. Skip SA profiling (use flat mutation rates)
```bash
python main.py --no-sa
```

### 5. Benchmark the latest elite against Harder AI
```bash
python main.py --benchmark
```

---

## Configuration

Edit `config.py` to change:

| Setting | Default | Options |
|---------|---------|---------|
| `BOT_RACE` | `Race.Terran` | `Race.Terran`, `Race.Protoss`, `Race.Zerg` |
| `OPPONENT_RACE` | `Race.Random` | Any `Race.*` |
| `MAP_NAME` | `"CatalystLE"` | Any installed SC2 map |
| `TOTAL_GENERATIONS` | `5` | Any positive int |
| `N_POP` | `20` | Smaller = faster, larger = better exploration |
| `K_GAMES` | `3` | Games averaged per fitness evaluation |
| `GAME_TIME_LIMIT` | `480` | Seconds of in-game time per game |

---

## Genome Structure (35D)

| Slice | Sub-Pop | Dim | Description |
|-------|---------|-----|-------------|
| `w[0–9]` | Pop A | 10 | Macro / Economy |
| `w[10–24]` | Pop B | 15 | Micro / Combat |
| `w[25–34]` | Pop C | 10 | Tech / Build-Order |

All values are continuous in **[0, 1]**. See `config.py → GENE_NAMES` for the full list.

---

## Algorithm Summary

```
Phase 1 (SA Profiling):
  For each gene j (0..34):
    Test w_j = 0.0 and w_j = 1.0 in 8-min SC2 games
    I_j = |score_high - score_low|   (composite RUR + APU)
    μ_j = 0.08 + (1 - I_j/I_max) × 0.12   ∈ [0.08, 0.20]

Phase 2 (Evolution, 5 generations):
  For each generation g:
    difficulty = DIFFICULTY_SCHEDULE[g-1]   (ramps VeryEasy → Harder)
    For each agent in Pop A:
      genome = agent_A ‖ best_B ‖ best_C   (cooperative handshake)
      f_i = mean(K_GAMES games)
      e_i^t = Σ_k [ w_k · f_i^(k) ]       (dynamic similarity weighting)
    (Same for Pop B and Pop C)
    Apply: diversity pruning (δ_s=0.85) + fitness floor (δ_f=0.20)
    Update archive (top-3 per sub-pop, max 15)
    Breed offspring via tournament + Gaussian mutation (SA-scaled)

Phase 3 (Benchmark):
  Best genome from archives vs Harder AI (Level 5), 5 games
```

---

## Output Files

```
results/
├── sensitivity_scores_sc2.json   # μ_j per-gene mutation rates
├── elite_gen001.json             # best genome after generation 1
├── elite_gen002.json             # ...
├── elite_gen005.json             # final elite genome
├── evolution_log.json            # per-generation statistics
└── benchmark_result.json         # final benchmark metrics
```

---

## Expected Runtime (5 generations, N_POP=20, K=3)

| Phase | Games | Approx. Time |
|-------|-------|-------------|
| SA Profiling | 35 × 2 × 2 = 140 | ~3–5 hours |
| Evolution (per gen) | 60 × 3 = 180 | ~4–8 hours |
| Evolution (total 5 gens) | 900 | ~20–40 hours |
| Benchmark | 5 | ~0.5 hours |

> **Tip:** Use `--no-sa` to skip profiling on the first run and start with flat mutation rates. The SA profiling only needs to run once per race/map combination.
