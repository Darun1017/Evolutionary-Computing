"""
=============================================================
GESY-TDCCEA — TD-CCEA outer loop driving GESY GP individuals
=============================================================

gesy_tdccea.py
--------------
Combines two systems into a novel hybrid algorithm:

  OUTER LOOP  — TD-CCEA (Trajectory-Driven Competitive Co-Evolutionary
                Algorithm) from tdccea_microrts.py:
                • Trajectory scoring  (dynamic weighted fitness history)
                • Adaptive mutation rate (per-individual)
                • Diversity-based survival (Euclidean distance pruning)
                • Weakness pruning (drop chronically low performers)
                • Archive / Hall of Fame (preserve elite scripts)

  INDIVIDUALS — GESY StrategyScript objects (GP trees from gesy/):
                • AST programs encoded as IfThenElse / Sequence trees
                • Crossover = type-safe subtree swap (STGP)
                • Mutation  = node replacement + subtree regrowth

  EVALUATION  — Real MicroRTS Java games via BatchGameRunner:
                • Each script is mapped to a Java bot via command_mapping
                • N_GAMES games per script per generation

This replaces the weight-vector individuals of the original TD-CCEA
with evolved GP programs, giving each "agent" a richer, interpretable
strategy representation.

Output
------
  • Best script as AST .txt, pseudocode .txt, .json
  • Evolution log as JSON
  • Optional visual replay of the best evolved bot
=============================================================
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ── Path setup ─────────────────────────────────────────────────────────────
_CODES_DIR = os.path.dirname(os.path.abspath(__file__))
_GESY_DIR  = os.path.join(os.path.dirname(_CODES_DIR), "GESY")
if not os.path.isdir(_GESY_DIR):
    _GESY_DIR = os.path.join(os.path.dirname(_CODES_DIR), "GESY")

for _p in [_CODES_DIR, _GESY_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gesy.core.ast_nodes           import StrategyScript
from gesy.strategies.seeded_init   import (
    initialise_population, build_random_script
)
from gesy.engine.genetic_operators import (
    tournament_select, crossover, mutate,
)
from gesy.utils.exporter import (
    export_pretty, export_pseudocode, export_json, save_to_file,
)

from gesy_microrts_bridge.microrts_evaluator   import MicroRTSEvaluator
from gesy_microrts_bridge.population_evaluator import evaluate_population_microrts
from gesy_microrts_bridge.command_mapping      import script_to_bot


# ============================================================
#  CONFIGURATION
# ============================================================

# ── MicroRTS ──
MICRORTS_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "Simulation", "microrts")
MAP_PATHS = {
    "8x8":   "maps/8x8/basesWorkers8x8Obstacle.xml",
    "16x16": "maps/16x16/basesWorkers16x16A.xml",
    "32x32": "maps/basesWorkers32x32A.xml",
}
MAP_SIZE     = "8x8"
OPPONENT_BOT = "WorkerRush"
N_GAMES      = 3
MAX_CYCLES   = 4000
MAP_PATH     = MAP_PATHS[MAP_SIZE]

# ── TD-CCEA hyperparameters ──
N_POP             = 20       # starting population size
TOTAL_GENERATIONS = 20       # evolutionary generations
H_WINDOW          = 5        # trajectory history window (past generations)
ALPHA_W           = [0.40, 0.25, 0.15, 0.12, 0.08]  # recency weights
MU_BASE           = 0.20     # base (GP) mutation rate for adaptive scaling
DELTA_F           = 0.25     # fitness pruning threshold
DELTA_S           = 0.85     # similarity pruning threshold (cosine-style)
A_MAX             = 15       # max archive size
MIN_SURVIVORS     = 10       # safety floor for diversity pruning
TOURNAMENT_SIZE   = 3        # GP parent selection tournament size

# ── GP hyperparameters ──
CROSSOVER_RATE  = 0.75
MAX_TREE_DEPTH  = 6
RANDOM_SEED     = 42

# ── Output ──
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesy_outputs")


# ============================================================
#  GP-augmented TD-CCEA individual
# ============================================================

@dataclass
class GESYAgent:
    """
    A TD-CCEA individual whose 'DNA' is a GESY StrategyScript (GP tree)
    rather than a weight vector.

    TD-CCEA trajectory fields sit alongside the GESY script.
    """
    script: StrategyScript         # the evolved GP program
    H_i:    list[float] = field(default_factory=list)  # fitness history buffer
    e_i:    float = 0.0            # trajectory score (expressibility)
    mu_i:   float = MU_BASE        # adaptive mutation rate

    # ------------------------------------------------------------------
    # Identification helpers
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.script.name

    def get_bot(self) -> str:
        """Return the MicroRTS Java bot that this script maps to."""
        return script_to_bot(self.script)

    # ------------------------------------------------------------------
    # TD-CCEA trajectory logic
    # ------------------------------------------------------------------

    def update_history(self, fitness: float) -> None:
        """Append current fitness to trajectory history (FIFO, max H_WINDOW)."""
        self.H_i.append(fitness)
        if len(self.H_i) > H_WINDOW:
            self.H_i.pop(0)

    def compute_trajectory_score(self) -> float:
        """
        Step 2 — Trajectory Score (weighted recency sum).

        e_i = Σ_k  ALPHA_W[k] × H_i[-(k+1)]

        When history is shorter than H_WINDOW, use plain mean as fallback.
        """
        if len(self.H_i) < H_WINDOW:
            self.e_i = sum(self.H_i) / len(self.H_i) if self.H_i else 0.0
        else:
            self.e_i = sum(
                ALPHA_W[k] * self.H_i[-(k + 1)]
                for k in range(H_WINDOW)
            )
        return self.e_i

    def compute_adaptive_mutation(self) -> float:
        """
        Step 3 — Adaptive Mutation Rate.

        mu_i = MU_BASE × (1 - e_i) + 0.05

        Low trajectory score → higher mutation (explore).
        High trajectory score → lower mutation (exploit / fine-tune).
        """
        self.mu_i = MU_BASE * (1.0 - self.e_i) + 0.05
        self.mu_i = max(0.02, min(0.60, self.mu_i))  # safety bounds
        return self.mu_i

    # ------------------------------------------------------------------
    # GP mutation (uses GESY operators)
    # ------------------------------------------------------------------

    def mutate_gp(self) -> "GESYAgent":
        """
        Produce one GP offspring by mutating this script.

        The adaptive mutation rate mu_i controls how aggressively the
        subtrees are rewritten:
          - mu_i > 0.4 → many nodes replaced (high exploration)
          - mu_i < 0.1 → minimal changes (fine-tuning)
        """
        child_script = mutate(self.script, mutation_rate=self.mu_i)
        child_script.generation = self.script.generation + 1

        # Depth cap
        if child_script.depth() > MAX_TREE_DEPTH:
            # Revert to clone of parent
            child_script = self.script.clone(
                new_name=self.script.name + "_clip"
            )
            child_script.generation = self.script.generation + 1

        return GESYAgent(script=child_script)

    # ------------------------------------------------------------------
    # GP crossover (used in reproduction step)
    # ------------------------------------------------------------------

    def crossover_with(self, other: "GESYAgent", child_name: str) -> "GESYAgent":
        """Produce one child by STGP subtree crossover with another agent."""
        child_script = crossover(
            self.script, other.script,
            child_name = child_name,
            generation = self.script.generation + 1,
        )
        if child_script.depth() > MAX_TREE_DEPTH:
            child_script = self.script.clone(new_name=child_name + "_clip")
        return GESYAgent(script=child_script)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        return (
            f"{self.name:<32s}  "
            f"e_i={self.e_i:.4f}  "
            f"mu={self.mu_i:.3f}  "
            f"bot={self.get_bot():<14s}  "
            f"wld={self.script.wins}/{self.script.losses}/{self.script.draws}"
        )


# ============================================================
#  Diversity distance between two agents
# ============================================================

def _node_feature_vector(script: StrategyScript) -> list[float]:
    """
    Lightweight feature vector representing a script's structure.
    Used for diversity distance — avoids expensive tree comparison.

    Features (all normalised to [0,1]):
      0  node_count / 50
      1  depth / 10
      2  wins / max(wins+losses+draws, 1)
      3  fitness
      4  e_i (set if available, else 0)
    """
    total = script.wins + script.losses + script.draws or 1
    return [
        min(script.node_count() / 50.0, 1.0),
        min(script.depth()      / 10.0, 1.0),
        script.wins / total,
        script.fitness,
    ]


def agent_distance(a: GESYAgent, b: GESYAgent) -> float:
    """Euclidean distance in the 4-D feature space."""
    va = _node_feature_vector(a.script)
    vb = _node_feature_vector(b.script)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(va, vb)))


_MAX_FEATURE_DIST = math.sqrt(4)  # max possible distance in 4-D [0,1] space


def similarity(a: GESYAgent, b: GESYAgent) -> float:
    """Normalised similarity score in [0,1]."""
    return 1.0 - (agent_distance(a, b) / _MAX_FEATURE_DIST)


# ============================================================
#  Evolution Log
# ============================================================

_gen_log: list[dict[str, Any]] = []


def _log_generation(
    gen: int,
    population: list[GESYAgent],
    archive: list[GESYAgent],
    elapsed: float,
) -> None:
    sorted_pop = sorted(population, key=lambda a: a.e_i, reverse=True)
    _gen_log.append({
        "generation":    gen,
        "best_e_i":      round(sorted_pop[0].e_i if sorted_pop else 0, 4),
        "avg_e_i":       round(
            sum(a.e_i for a in sorted_pop) / max(len(sorted_pop), 1), 4
        ),
        "pop_size":      len(population),
        "archive_size":  len(archive),
        "elapsed_sec":   round(elapsed, 2),
        "best_name":     sorted_pop[0].name if sorted_pop else "—",
        "best_bot":      sorted_pop[0].get_bot() if sorted_pop else "—",
        "top5": [
            {
                "name":    a.name,
                "e_i":     round(a.e_i, 4),
                "bot":     a.get_bot(),
                "wld":     f"{a.script.wins}/{a.script.losses}/{a.script.draws}",
                "mu":      round(a.mu_i, 3),
            }
            for a in sorted_pop[:5]
        ],
    })


# ============================================================
#  Main TD-CCEA + GESY evolution loop
# ============================================================

def run_gesy_tdccea() -> GESYAgent:
    """
    Execute the full GESY-TD-CCEA hybrid evolutionary algorithm.

    Returns the best GESYAgent found across all generations.
    """
    random.seed(RANDOM_SEED)

    print("\n" + "═" * 68)
    print("  GESY-TDCCEA — Trajectory-Driven GP Strategy Evolution")
    print("═" * 68)
    print(f"  Map       : {MAP_PATH}")
    print(f"  Opponent  : {OPPONENT_BOT}")
    print(f"  Pop size  : {N_POP}  |  Generations: {TOTAL_GENERATIONS}")
    print(f"  H_WINDOW  : {H_WINDOW}  |  δF: {DELTA_F}  |  δS: {DELTA_S}")
    print(f"  Archive   : {A_MAX}  |  Games/eval : {N_GAMES}")
    print("═" * 68 + "\n")

    # ── Step 1: Seeded initialisation ────────────────────────────────
    print("[Init] Building seeded initial population...")
    seed_scripts = initialise_population(
        population_size = N_POP,
        random_seed     = RANDOM_SEED,
    )
    population: list[GESYAgent] = [GESYAgent(script=s) for s in seed_scripts]
    archive:    list[GESYAgent] = []

    best_ever: Optional[GESYAgent] = None
    t_total = time.time()

    # ── Start MicroRTS server ─────────────────────────────────────────
    evaluator = MicroRTSEvaluator(
        map_path     = MAP_PATH,
        opponent_bot = OPPONENT_BOT,
        n_games      = N_GAMES,
        max_cycles   = MAX_CYCLES,
        microrts_dir = MICRORTS_DIR,
    )
    evaluator.start()

    try:
        name_counter = [0]

        def fresh_name(prefix: str, gen: int) -> str:
            name_counter[0] += 1
            return f"{prefix}_G{gen}_{name_counter[0]:04d}"

        for gen in range(1, TOTAL_GENERATIONS + 1):
            t_gen = time.time()
            print(f"  ┌─ Generation {gen:>2d}/{TOTAL_GENERATIONS} {'─'*48}")

            # ── Step 2: Fitness evaluation via real MicroRTS ───────────
            scripts = [a.script for a in population]
            scripts = evaluate_population_microrts(
                population = scripts,
                evaluator  = evaluator,
                verbose    = False,
            )
            # Sync fitness back to GESYAgent wrappers
            fitness_map = {s.name: s.fitness for s in scripts}
            for agent in population:
                agent.script.fitness = fitness_map.get(agent.script.name, 0.0)

            # ── Step 2b: Update trajectory scores ─────────────────────
            for agent in population:
                agent.update_history(agent.script.fitness)
                agent.compute_trajectory_score()
                agent.compute_adaptive_mutation()

            # ── Summary ────────────────────────────────────────────────
            sorted_pop = sorted(population, key=lambda a: a.e_i, reverse=True)
            best_agent = sorted_pop[0]
            avg_ei     = sum(a.e_i for a in population) / len(population)
            bots_used  = {a.get_bot() for a in population}

            print(f"  │  Best e_i: {best_agent.e_i:.4f} | "
                  f"Avg e_i: {avg_ei:.4f} | "
                  f"Bots: {', '.join(sorted(bots_used))}")
            print(f"  │  Best: {best_agent.name}")
            print(f"  │  Script nodes: {best_agent.script.node_count()} | "
                  f"Depth: {best_agent.script.depth()}")

            # Track all-time best
            if best_ever is None or best_agent.e_i > best_ever.e_i:
                clone = GESYAgent(
                    script = best_agent.script.clone(
                        new_name=f"BestGen{gen}_{best_agent.script.name}"
                    ),
                    H_i  = list(best_agent.H_i),
                    e_i  = best_agent.e_i,
                    mu_i = best_agent.mu_i,
                )
                best_ever = clone

            # ── Step 4: Diversity-based survival ──────────────────────
            survivors: list[GESYAgent] = []
            for agent in sorted_pop:
                is_redundant = any(
                    similarity(agent, u) > DELTA_S
                    for u in survivors
                )
                if not is_redundant:
                    survivors.append(agent)

            # ── Step 5: Weakness pruning ───────────────────────────────
            final_survivors: list[GESYAgent] = []
            for agent in survivors:
                if agent.e_i < DELTA_F and len(agent.H_i) >= 3:
                    pass  # prune chronically weak agents
                else:
                    final_survivors.append(agent)

            # Safety floor
            while len(final_survivors) < MIN_SURVIVORS and len(survivors) > len(final_survivors):
                for a in survivors:
                    if a not in final_survivors:
                        final_survivors.append(a)
                        break

            # ── Step 6: Archive update ─────────────────────────────────
            for top_agent in sorted_pop[:3]:
                archive_clone = GESYAgent(
                    script = top_agent.script.clone(
                        new_name=fresh_name("Archive", gen)
                    ),
                    H_i  = list(top_agent.H_i),
                    e_i  = top_agent.e_i,
                    mu_i = top_agent.mu_i,
                )
                archive.append(archive_clone)
            if len(archive) > A_MAX:
                # Keep the A_MAX agents with highest e_i
                archive.sort(key=lambda a: a.e_i, reverse=True)
                archive = archive[:A_MAX]

            print(f"  │  Survivors: {len(final_survivors)} | Archive: {len(archive)}")

            # ── Step 7: Reproduction ───────────────────────────────────
            breeding_pool = final_survivors + archive
            population    = list(final_survivors)  # survivors carry over

            while len(population) < N_POP:
                # Tournament select two parents from breeding pool + archive
                candidates_a = random.sample(
                    breeding_pool, min(TOURNAMENT_SIZE, len(breeding_pool))
                )
                parent_a = max(candidates_a, key=lambda a: a.e_i)

                candidates_b = random.sample(
                    breeding_pool, min(TOURNAMENT_SIZE, len(breeding_pool))
                )
                parent_b = max(candidates_b, key=lambda a: a.e_i)

                cname = fresh_name("Child", gen)

                # Crossover or clone
                if random.random() < CROSSOVER_RATE:
                    child = parent_a.crossover_with(parent_b, child_name=cname)
                else:
                    child = GESYAgent(
                        script=parent_a.script.clone(new_name=cname)
                    )

                # GP mutation (rate = parent's adaptive mu_i)
                child = child.mutate_gp()
                child.script.name = cname  # ensure unique name post-mutation
                population.append(child)

            elapsed = time.time() - t_gen
            print(f"  │  Gen time: {elapsed:.1f}s")
            print(f"  └{'─'*64}")

            _log_generation(gen, population, archive, elapsed)

    finally:
        evaluator.stop()

    total_elapsed = time.time() - t_total

    # ============================================================
    #  Results
    # ============================================================

    print("\n" + "═" * 68)
    print("  GESY-TDCCEA EVOLUTION COMPLETE")
    print("═" * 68)
    print(f"  Total time   : {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Best script  : {best_ever.name}")
    print(f"  Trajectory   : {best_ever.e_i:.4f}")
    print(f"  Bot mapping  : {best_ever.get_bot()}")
    print(f"  Nodes/Depth  : {best_ever.script.node_count()} / {best_ever.script.depth()}")

    # ── Export ────────────────────────────────────────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    paths = save_to_file(best_ever.script, output_dir=OUTPUT_DIR)
    print(f"\n  [Export] Files saved to {OUTPUT_DIR}/")
    for fmt, path in paths.items():
        print(f"    {fmt:12s} → {path}")

    log_path = os.path.join(OUTPUT_DIR, "evolution_log_tdccea.json")
    Path(log_path).write_text(json.dumps(_gen_log, indent=2), encoding="utf-8")
    print(f"    {'log':12s} → {log_path}")

    # ── Evolution history table ───────────────────────────────────────
    print(f"\n  Evolution History:")
    print(f"  {'Gen':>4s}  {'Best e_i':>8s}  {'Avg e_i':>8s}  "
          f"{'Pop':>4s}  {'Archive':>7s}  {'Bot'}")
    print("  " + "─" * 55)
    for r in _gen_log:
        print(f"  {r['generation']:>4d}  {r['best_e_i']:>8.4f}  "
              f"{r['avg_e_i']:>8.4f}  {r['pop_size']:>4d}  "
              f"{r['archive_size']:>7d}  {r.get('best_bot', '?')}")

    # ── Pretty print best script ──────────────────────────────────────
    print("\n" + "═" * 68)
    print("  FINAL BEST STRATEGY — Pretty Print")
    print("═" * 68)
    print(export_pretty(best_ever.script))

    print("\n" + "═" * 68)
    print("  FINAL BEST STRATEGY — Pseudo-code")
    print("═" * 68)
    print(export_pseudocode(best_ever.script))

    # ── Visual replay ─────────────────────────────────────────────────
    best_bot = best_ever.get_bot()
    print(f"\n  [Visualizer] Launching: {best_bot} vs {OPPONENT_BOT} on {MAP_SIZE}")
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
        print(f"  [Visualizer] Could not launch: {e}")
    print()

    return best_ever


# ============================================================
#  Entry point
# ============================================================

if __name__ == "__main__":
    run_gesy_tdccea()
