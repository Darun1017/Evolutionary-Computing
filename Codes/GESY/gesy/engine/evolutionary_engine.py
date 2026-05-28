"""
GESY Framework — Module 6: Evolutionary Engine
===============================================
The main co-evolutionary loop implementing Steps 2–7:

  Step 2  — Parallel co-evolutionary evaluation
  Step 3  — Tournament selection
  Step 4  — Crossover
  Step 5  — Mutation
  Step 6  — Elitism (copy top k% unchanged)
  Step 7  — Random newcomer injection (invaders)

All hyperparameters are encapsulated in EvolutionConfig for easy tuning.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from gesy.core.ast_nodes import StrategyScript
from gesy.engine.genetic_operators import (
    crossover,
    evaluate_population,
    mutate,
    tournament_select,
)
from gesy.strategies.seeded_init import build_random_script


# ---------------------------------------------------------------------------
# Hyperparameter configuration
# ---------------------------------------------------------------------------

@dataclass
class EvolutionConfig:
    """
    All tunable knobs in one place.  Adjust these at the entry-point;
    everything else adapts automatically.
    """
    # Population
    population_size:     int   = 30      # N — total individuals per generation

    # Loop control
    max_generations:     int   = 20      # how many generations to evolve

    # Genetic operator probabilities
    crossover_rate:      float = 0.75    # probability that crossover occurs vs. cloning
    mutation_rate:       float = 0.25    # per-operator mutation probability (inside mutate())

    # Elitism: top k% survive unchanged into next generation
    elitism_rate:        float = 0.05    # 5% recommended
    min_elites:          int   = 1       # always keep at least this many elites

    # Random newcomers: k% of next gen are freshly random scripts
    newcomer_rate:       float = 0.08    # 8% recommended
    min_newcomers:       int   = 1

    # Evaluation
    n_eval_workers:      int   = 4       # thread-pool size for parallel matches
    matches_per_script:  int   = 6       # opponents each script faces per generation

    # Tournament selection
    tournament_size:     int   = 3

    # Random seed
    random_seed:         int   = 42

    # Bloat control
    max_tree_depth:      int   = 6       # hard cap; deeper trees are truncated

    # Derived (computed from rates × population_size)
    @property
    def n_elites(self) -> int:
        return max(self.min_elites, int(self.population_size * self.elitism_rate))

    @property
    def n_newcomers(self) -> int:
        return max(self.min_newcomers, int(self.population_size * self.newcomer_rate))


# ---------------------------------------------------------------------------
# Generation statistics
# ---------------------------------------------------------------------------

@dataclass
class GenerationStats:
    generation:     int
    best_fitness:   float
    mean_fitness:   float
    worst_fitness:  float
    best_name:      str
    mean_nodes:     float
    elapsed_sec:    float
    n_elites:       int
    n_newcomers:    int


# ---------------------------------------------------------------------------
# Evolutionary Engine
# ---------------------------------------------------------------------------

class EvolutionaryEngine:
    """
    Orchestrates the full GESY evolutionary loop.

    Usage:
        engine = EvolutionaryEngine(config, initial_population)
        engine.run()
        best = engine.best_script
    """

    def __init__(
        self,
        config:     EvolutionConfig,
        population: list[StrategyScript],
        callback:   Optional[Callable[[GenerationStats, list[StrategyScript]], None]] = None,
    ) -> None:
        self.config     = config
        self.population = population
        self.callback   = callback    # optional: called at end of each generation

        self.generation       = 0
        self.history:  list[GenerationStats]   = []
        self.best_script: Optional[StrategyScript] = None

        random.seed(config.random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> StrategyScript:
        """
        Execute the full evolutionary loop for `config.max_generations` generations.
        Returns the best script found across all generations.
        """
        print(f"\n{'='*60}")
        print(f"  GESY Evolutionary Engine — Starting")
        print(f"  Population: {self.config.population_size}  |  "
              f"Generations: {self.config.max_generations}")
        print(f"  Elitism: {self.config.n_elites}  |  "
              f"Newcomers: {self.config.n_newcomers}  |  "
              f"Workers: {self.config.n_eval_workers}")
        print(f"{'='*60}\n")

        for gen in range(self.config.max_generations):
            self.generation = gen
            t0 = time.time()

            # ----------------------------------------------------------
            # Step 2: Co-evolutionary evaluation
            # ----------------------------------------------------------
            self.population = evaluate_population(
                population  = self.population,
                n_workers   = self.config.n_eval_workers,
                seed_offset = gen,
            )
            self.population.sort(key=lambda s: s.fitness, reverse=True)

            # Track all-time best (compare fitness before cloning)
            current_best = self.population[0]
            if (self.best_script is None
                    or current_best.fitness > self.best_script.fitness):
                clone = current_best.clone(
                    new_name=f"BestGen{gen}_{current_best.name}"
                )
                clone.fitness    = current_best.fitness
                clone.wins       = current_best.wins
                clone.losses     = current_best.losses
                clone.draws      = current_best.draws
                clone.generation = gen
                self.best_script = clone

            # ----------------------------------------------------------
            # Compute generation statistics
            # ----------------------------------------------------------
            fitnesses  = [s.fitness for s in self.population]
            node_counts = [s.node_count() for s in self.population]
            stats = GenerationStats(
                generation    = gen,
                best_fitness  = fitnesses[0],
                mean_fitness  = sum(fitnesses) / len(fitnesses),
                worst_fitness = fitnesses[-1],
                best_name     = current_best.name,
                mean_nodes    = sum(node_counts) / len(node_counts),
                elapsed_sec   = time.time() - t0,
                n_elites      = self.config.n_elites,
                n_newcomers   = self.config.n_newcomers,
            )
            self.history.append(stats)
            self._print_stats(stats)

            if self.callback:
                self.callback(stats, list(self.population))

            # ----------------------------------------------------------
            # Build next generation
            # ----------------------------------------------------------
            if gen < self.config.max_generations - 1:
                self.population = self._build_next_generation(gen)

        print(f"\n{'='*60}")
        print(f"  Evolution complete.")
        print(f"  Best script: {self.best_script.name}  "  # type: ignore[union-attr]
              f"Fitness: {self.best_script.fitness:.4f}")   # type: ignore[union-attr]
        print(f"{'='*60}\n")

        return self.best_script  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal: next generation assembly
    # ------------------------------------------------------------------

    def _build_next_generation(self, gen: int) -> list[StrategyScript]:
        """
        Assemble the next generation:
          1. Elites   (Step 6) — top k scripts unchanged
          2. Offspring (Steps 3+4+5) — fill slots via selection + crossover + mutation
          3. Newcomers (Step 7) — inject random invaders
        """
        next_gen: list[StrategyScript] = []
        cfg = self.config
        counter = [0]   # mutable for unique naming inside closures

        def fresh_name(prefix: str) -> str:
            counter[0] += 1
            return f"{prefix}_G{gen+1}_{counter[0]:04d}"

        # ------ Step 6: Elitism ------
        elites = self.population[: cfg.n_elites]
        for elite in elites:
            e_clone = elite.clone(new_name=fresh_name("Elite"))
            e_clone.generation = gen + 1
            next_gen.append(e_clone)

        # ------ Step 7: Random Newcomers ------
        newcomers = [
            build_random_script(name=fresh_name("Invader"))
            for _ in range(cfg.n_newcomers)
        ]
        for nc in newcomers:
            nc.generation = gen + 1

        # ------ Steps 3+4+5: Selection → Crossover → Mutation ------
        n_offspring = cfg.population_size - len(elites) - len(newcomers)

        for _ in range(n_offspring):
            # Step 3: Tournament selection
            parent_a = tournament_select(self.population, cfg.tournament_size)
            parent_b = tournament_select(self.population, cfg.tournament_size)

            # Step 4: Crossover
            if random.random() < cfg.crossover_rate:
                child = crossover(
                    parent_a, parent_b,
                    child_name = fresh_name("Child"),
                    generation = gen + 1,
                )
            else:
                child = parent_a.clone(new_name=fresh_name("Clone"))
                child.generation = gen + 1

            # Step 5: Mutation
            child = mutate(child, mutation_rate=cfg.mutation_rate)
            child.generation = gen + 1

            # Depth cap (bloat hard limit)
            if child.depth() > cfg.max_tree_depth:
                child = parent_a.clone(new_name=fresh_name("Clipped"))
                child.generation = gen + 1

            next_gen.append(child)

        next_gen.extend(newcomers)

        assert len(next_gen) == cfg.population_size, (
            f"Population size mismatch: {len(next_gen)} != {cfg.population_size}"
        )
        return next_gen

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _print_stats(stats: GenerationStats) -> None:
        print(
            f"  Gen {stats.generation:>3d} | "
            f"Best: {stats.best_fitness:.4f} | "
            f"Mean: {stats.mean_fitness:.4f} | "
            f"Worst: {stats.worst_fitness:.4f} | "
            f"AvgNodes: {stats.mean_nodes:.1f} | "
            f"{stats.elapsed_sec:.2f}s | "
            f"Leader: {stats.best_name}"
        )
