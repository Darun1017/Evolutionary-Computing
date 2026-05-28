"""
GESY Framework — Main Entry Point
==================================
Configure hyperparameters here and launch the full evolutionary run.
All tunable knobs are in EvolutionConfig.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gesy.engine.evolutionary_engine import EvolutionConfig, EvolutionaryEngine, GenerationStats
from gesy.core.ast_nodes import StrategyScript
from gesy.strategies.seeded_init import initialise_population
from gesy.utils.exporter import export_pretty, export_pseudocode, save_to_file


# ---------------------------------------------------------------------------
# HYPERPARAMETER CONFIGURATION  ← tune everything here
# ---------------------------------------------------------------------------

CONFIG = EvolutionConfig(
    # Population
    population_size     = 30,

    # Loop
    max_generations     = 15,

    # Genetic operators
    crossover_rate      = 0.75,
    mutation_rate       = 0.25,

    # Elitism  (1-5% of population)
    elitism_rate        = 0.05,
    min_elites          = 1,

    # Random newcomers / invaders  (5-10% of population)
    newcomer_rate       = 0.08,
    min_newcomers       = 1,

    # Parallel evaluation
    n_eval_workers      = 4,
    matches_per_script  = 6,

    # Selection
    tournament_size     = 3,

    # Reproducibility
    random_seed         = 42,

    # Bloat control
    max_tree_depth      = 6,
)


# ---------------------------------------------------------------------------
# Optional per-generation callback (for dashboards / logging / telemetry)
# ---------------------------------------------------------------------------

_gen_log: list[dict[str, Any]] = []


def on_generation_end(stats: GenerationStats, pop: list[StrategyScript]) -> None:
    """Collect stats for later export / visualisation."""
    _gen_log.append({
        "generation":    stats.generation,
        "best_fitness":  round(stats.best_fitness,  4),
        "mean_fitness":  round(stats.mean_fitness,  4),
        "worst_fitness": round(stats.worst_fitness, 4),
        "mean_nodes":    round(stats.mean_nodes,    1),
        "elapsed_sec":   round(stats.elapsed_sec,   2),
        "best_name":     stats.best_name,
        "n_elites":      stats.n_elites,
        "n_newcomers":   stats.n_newcomers,
        "top5":          [
            {"name": s.name, "fitness": round(s.fitness, 4),
             "nodes": s.node_count(), "wld": f"{s.wins}/{s.losses}/{s.draws}"}
            for s in pop[:5]
        ],
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "═" * 62)
    print("  GESY — Genetic Evolution of Strategy Scripts")
    print("  Seeded Genetic Programming Framework")
    print("═" * 62)

    # Step 1: Seeded initialisation
    population = initialise_population(
        population_size = CONFIG.population_size,
        random_seed     = CONFIG.random_seed,
    )

    # Steps 2–7: Evolutionary loop
    engine = EvolutionaryEngine(
        config     = CONFIG,
        population = population,
        callback   = on_generation_end,
    )
    best = engine.run()

    # Export results
    print("\n[Export] Saving best script to /mnt/user-data/outputs/ ...")
    paths = save_to_file(best)
    for fmt, path in paths.items():
        print(f"         {fmt:12s} → {path}")

    # Save evolution log
    log_path = "/mnt/user-data/outputs/evolution_log.json"
    Path(log_path).write_text(json.dumps(_gen_log, indent=2))
    print(f"         {'log':12s} → {log_path}")

    # Print final best script
    print("\n" + "═" * 62)
    print("  FINAL BEST STRATEGY — Pretty Print")
    print("═" * 62)
    print(export_pretty(best))

    print("\n" + "═" * 62)
    print("  FINAL BEST STRATEGY — Pseudo-code")
    print("═" * 62)
    print(export_pseudocode(best))


if __name__ == "__main__":
    main()
