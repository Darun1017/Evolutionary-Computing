"""
=============================================================
GESY + MicroRTS — Full Integration Entry Point
=============================================================

gesy_microrts.py
----------------
Runs the GESY Genetic Programming evolutionary loop where
fitness is measured by playing REAL MicroRTS games instead
of the built-in pure-Python mini-simulator.

Pipeline
--------
  Step 1  — Seeded initialisation (Rush / Defensive / Ranged seeds + randoms)
  Step 2  — Evaluate population via BatchGameRunner Java games
  Step 3  — Tournament selection
  Step 4  — Type-safe subtree crossover (STGP)
  Step 5  — Mutation (leaf + subtree regrowth)
  Step 6  — Elitism (top k% unchanged)
  Step 7  — Random newcomer injection
  Repeat Steps 2-7 for MAX_GENERATIONS generations
  Export  — Best script as AST .txt, pseudocode .txt, .json

Configuration
-------------
All tunable settings are in the CONFIG block below.
=============================================================
"""

from __future__ import annotations

import json
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Any

# ── Path setup: make gesy and gesy_microrts_bridge importable ───────────────
_CODES_DIR = os.path.dirname(os.path.abspath(__file__))
_GESY_DIR  = os.path.join(os.path.dirname(_CODES_DIR), "Codes", "GESY")
# Adjust if this file is inside Code/ directly
if not os.path.isdir(_GESY_DIR):
    _GESY_DIR = os.path.join(os.path.dirname(_CODES_DIR), "GESY")

for _p in [_CODES_DIR, _GESY_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gesy.engine.evolutionary_engine import EvolutionConfig, GenerationStats
from gesy.core.ast_nodes              import StrategyScript
from gesy.strategies.seeded_init      import initialise_population, build_random_script
from gesy.engine.genetic_operators    import (
    tournament_select, crossover, mutate,
)
from gesy.utils.exporter              import (
    export_pretty, export_pseudocode, export_json, save_to_file,
)

from gesy_microrts_bridge.microrts_evaluator   import MicroRTSEvaluator
from gesy_microrts_bridge.population_evaluator import evaluate_population_microrts
from gesy_microrts_bridge.command_mapping      import script_to_bot


# ============================================================
#  CONFIGURATION  ← tune everything here
# ============================================================

# ── MicroRTS settings ──
MICRORTS_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "Simulation", "microrts")
MAP_PATHS = {
    "8x8":  "maps/8x8/basesWorkers8x8Obstacle.xml",
    "16x16": "maps/16x16/basesWorkers16x16A.xml",
    "32x32": "maps/basesWorkers32x32A.xml",
}
MAP_SIZE      = "8x8"            # change to "16x16" or "32x32" for larger maps
OPPONENT_BOT  = "WorkerRush"     # Java bot we evolve against
N_GAMES       = 3                # games per script per generation
MAX_CYCLES    = 4000             # hard game-cycle cap

# ── GP / Evolution settings ──
POPULATION_SIZE  = 20
MAX_GENERATIONS  = 15
CROSSOVER_RATE   = 0.75
MUTATION_RATE    = 0.25
ELITISM_RATE     = 0.05
MIN_ELITES       = 1
NEWCOMER_RATE    = 0.08
MIN_NEWCOMERS    = 1
TOURNAMENT_SIZE  = 3
RANDOM_SEED      = 42
MAX_TREE_DEPTH   = 6

# ── Output directory ──
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesy_outputs")

# ── Derived ──
MAP_PATH   = MAP_PATHS[MAP_SIZE]
N_ELITES   = max(MIN_ELITES,    int(POPULATION_SIZE * ELITISM_RATE))
N_NEWCOMERS = max(MIN_NEWCOMERS, int(POPULATION_SIZE * NEWCOMER_RATE))


# ============================================================
#  Evolution Log
# ============================================================

_gen_log: list[dict[str, Any]] = []


def _log_generation(
    gen: int,
    population: list[StrategyScript],
    elapsed: float,
) -> None:
    """Record per-generation statistics for export."""
    fitnesses = [s.fitness for s in population]
    _gen_log.append({
        "generation":    gen,
        "best_fitness":  round(fitnesses[0],                    4),
        "mean_fitness":  round(sum(fitnesses) / len(fitnesses),  4),
        "worst_fitness": round(fitnesses[-1],                   4),
        "mean_nodes":    round(
            sum(s.node_count() for s in population) / len(population), 1
        ),
        "elapsed_sec":   round(elapsed, 2),
        "best_name":     population[0].name,
        "top5": [
            {
                "name":    s.name,
                "fitness": round(s.fitness, 4),
                "bot":     script_to_bot(s),
                "nodes":   s.node_count(),
                "wld":     f"{s.wins}/{s.losses}/{s.draws}",
            }
            for s in population[:5]
        ],
    })


def _print_gen_summary(gen: int, population: list[StrategyScript], elapsed: float) -> None:
    best = population[0]
    avg  = sum(s.fitness for s in population) / len(population)
    print(
        f"  Gen {gen:>3d} | Best: {best.fitness:.4f} | Avg: {avg:.4f} | "
        f"Nodes: {best.node_count()} | Bot: {script_to_bot(best):<14s} | "
        f"Leader: {best.name} | {elapsed:.1f}s"
    )


# ============================================================
#  Next-generation builder
# ============================================================

def _build_next_generation(
    population: list[StrategyScript],
    gen: int,
) -> list[StrategyScript]:
    """
    Assemble the next generation via elitism → offspring → newcomers.
    Mirrors gesy.engine.evolutionary_engine._build_next_generation().
    """
    import random
    random.seed(RANDOM_SEED + gen)  # reproducible per generation

    next_gen: list[StrategyScript] = []
    counter = [0]

    def fresh_name(prefix: str) -> str:
        counter[0] += 1
        return f"{prefix}_G{gen+1}_{counter[0]:04d}"

    # Step 6: Elitism
    for elite in population[:N_ELITES]:
        e_clone = elite.clone(new_name=fresh_name("Elite"))
        e_clone.generation = gen + 1
        next_gen.append(e_clone)

    # Step 7: Newcomers
    newcomers = [
        build_random_script(name=fresh_name("Invader"))
        for _ in range(N_NEWCOMERS)
    ]
    for nc in newcomers:
        nc.generation = gen + 1

    # Steps 3+4+5: Selection → Crossover → Mutation
    import random as _rand
    n_offspring = POPULATION_SIZE - N_ELITES - N_NEWCOMERS
    for _ in range(n_offspring):
        parent_a = tournament_select(population, TOURNAMENT_SIZE)
        parent_b = tournament_select(population, TOURNAMENT_SIZE)

        if _rand.random() < CROSSOVER_RATE:
            child = crossover(parent_a, parent_b,
                              child_name=fresh_name("Child"), generation=gen + 1)
        else:
            child = parent_a.clone(new_name=fresh_name("Clone"))
            child.generation = gen + 1

        child = mutate(child, mutation_rate=MUTATION_RATE)
        child.generation = gen + 1

        # Depth cap
        if child.depth() > MAX_TREE_DEPTH:
            child = parent_a.clone(new_name=fresh_name("Clipped"))
            child.generation = gen + 1

        next_gen.append(child)

    next_gen.extend(newcomers)

    assert len(next_gen) == POPULATION_SIZE, (
        f"Population size mismatch: {len(next_gen)} != {POPULATION_SIZE}"
    )
    return next_gen


# ============================================================
#  Main
# ============================================================

def main() -> None:
    import random
    random.seed(RANDOM_SEED)

    print("\n" + "═" * 65)
    print("  GESY + MicroRTS — Genetic Programming on Real Games")
    print("═" * 65)
    print(f"  Map      : {MAP_PATH}")
    print(f"  Opponent : {OPPONENT_BOT}")
    print(f"  Pop size : {POPULATION_SIZE}  |  Generations : {MAX_GENERATIONS}")
    print(f"  Games/eval: {N_GAMES}         |  Map size    : {MAP_SIZE}")
    print(f"  Elites   : {N_ELITES}          |  Newcomers   : {N_NEWCOMERS}")
    print("═" * 65 + "\n")

    # Step 1: Seeded initialisation
    population = initialise_population(
        population_size = POPULATION_SIZE,
        random_seed     = RANDOM_SEED,
    )

    best_ever: StrategyScript | None = None
    t_total = time.time()

    # Start MicroRTS server
    evaluator = MicroRTSEvaluator(
        map_path     = MAP_PATH,
        opponent_bot = OPPONENT_BOT,
        n_games      = N_GAMES,
        max_cycles   = MAX_CYCLES,
        microrts_dir = MICRORTS_DIR,
    )
    evaluator.start()

    try:
        for gen in range(MAX_GENERATIONS):
            t_gen = time.time()

            # Step 2: Evaluate via real MicroRTS games
            population = evaluate_population_microrts(
                population = population,
                evaluator  = evaluator,
                verbose    = False,        # set True for per-script detail
                gen_label  = f"Gen {gen}",
            )

            # Track all-time best
            current_best = population[0]
            if best_ever is None or current_best.fitness > best_ever.fitness:
                best_ever = current_best.clone(
                    new_name=f"BestGen{gen}_{current_best.name}"
                )
                best_ever.fitness    = current_best.fitness
                best_ever.wins       = current_best.wins
                best_ever.losses     = current_best.losses
                best_ever.draws      = current_best.draws
                best_ever.generation = gen

            elapsed = time.time() - t_gen
            _print_gen_summary(gen, population, elapsed)
            _log_generation(gen, population, elapsed)

            # Build next generation (skip on last gen)
            if gen < MAX_GENERATIONS - 1:
                population = _build_next_generation(population, gen)

    finally:
        evaluator.stop()

    total_elapsed = time.time() - t_total

    # ── Export results ──────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  EVOLUTION COMPLETE")
    print("═" * 65)
    print(f"  Total time   : {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Best script  : {best_ever.name}")
    print(f"  Best fitness : {best_ever.fitness:.4f}")
    print(f"  Bot mapping  : {script_to_bot(best_ever)}")
    print(f"  W/L/D        : {best_ever.wins}/{best_ever.losses}/{best_ever.draws}")

    # Save outputs
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    paths = save_to_file(best_ever, output_dir=OUTPUT_DIR)
    print(f"\n  [Export] Files saved to {OUTPUT_DIR}/")
    for fmt, path in paths.items():
        print(f"    {fmt:12s} → {path}")

    # Save evolution log
    log_path = os.path.join(OUTPUT_DIR, "evolution_log_microrts.json")
    Path(log_path).write_text(json.dumps(_gen_log, indent=2), encoding="utf-8")
    print(f"    {'log':12s} → {log_path}")

    # Print best strategy
    print("\n" + "═" * 65)
    print("  FINAL BEST STRATEGY — Pretty Print")
    print("═" * 65)
    print(export_pretty(best_ever))

    print("\n" + "═" * 65)
    print("  FINAL BEST STRATEGY — Pseudo-code")
    print("═" * 65)
    print(export_pseudocode(best_ever))

    # ── Optional visual replay ──────────────────────────────────────────
    best_bot = script_to_bot(best_ever)
    print(f"\n  [Visualizer] Launching visual replay: {best_bot} vs {OPPONENT_BOT}")
    print("  [Visualizer] Close the game window to exit.\n")
    time.sleep(1)
    try:
        cmd = [
            "java", "-cp", "lib/*;bin",
            "tests.VisualGameRunner",
            best_bot, OPPONENT_BOT, MAP_PATH
        ]
        subprocess.run(cmd, cwd=MICRORTS_DIR)
        print("  [Visualizer] Window closed.")
    except Exception as e:
        print(f"  [Visualizer] Could not launch visual replay: {e}")
    print()


if __name__ == "__main__":
    main()
