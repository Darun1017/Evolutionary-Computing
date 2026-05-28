"""
GESY + MicroRTS Bridge — population_evaluator.py
=================================================
Batch-evaluates an entire GESY population using real MicroRTS games.

This is a drop-in replacement for:
    gesy.engine.genetic_operators.evaluate_population()

The key difference: instead of the internal pure-Python mini-simulator,
each script is evaluated by playing actual Java MicroRTS games via a
persistent BatchGameRunner process.

Two evaluation modes are supported:

  Mode A — FIXED OPPONENT (default):
    Every script plays `n_games` against a single Java opponent bot.
    Fast and reproducible.

  Mode B — ROTATING OPPONENTS:
    Each script faces a rotating set of Java bots from a given list.
    More diverse pressure; closer to real tournament evaluation.

Usage
-----
    from gesy_microrts_bridge.population_evaluator import evaluate_population_microrts
    from gesy_microrts_bridge.microrts_evaluator   import MicroRTSEvaluator

    with MicroRTSEvaluator(map_path=MAP, opponent_bot="WorkerRush", n_games=3) as ev:
        population = evaluate_population_microrts(population, ev, verbose=True)
"""

from __future__ import annotations

import sys
import os
import time
from typing import Optional

# ── Path setup ─────────────────────────────────────────────────────────────
_CODES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GESY_DIR  = os.path.join(_CODES_DIR, "GESY")
for _p in [_CODES_DIR, _GESY_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gesy.core.ast_nodes import StrategyScript           # noqa: E402
from gesy_microrts_bridge.microrts_evaluator import (    # noqa: E402
    MicroRTSEvaluator,
    EvalResult,
)
from gesy_microrts_bridge.command_mapping import (       # noqa: E402
    ALL_BOT_TYPES,
    script_to_bot,
)


# ---------------------------------------------------------------------------
# Mode A: Fixed-opponent population evaluation
# ---------------------------------------------------------------------------

def evaluate_population_microrts(
    population:  list[StrategyScript],
    evaluator:   MicroRTSEvaluator,
    verbose:     bool = True,
    gen_label:   str  = "",
) -> list[StrategyScript]:
    """
    Evaluate every script in `population` against a fixed Java opponent.

    Parameters
    ----------
    population : list[StrategyScript]
        The current generation of scripts to evaluate.
    evaluator : MicroRTSEvaluator
        A started MicroRTSEvaluator (call .start() before passing here,
        or use it as a context manager).
    verbose : bool
        Print per-script results to stdout.
    gen_label : str
        Optional generation label for display (e.g. "Gen 3").

    Returns
    -------
    list[StrategyScript]
        Same population with fitness / W/L/D updated in-place,
        sorted descending by fitness.
    """
    n = len(population)
    t0 = time.time()

    if verbose:
        label = f" [{gen_label}]" if gen_label else ""
        print(f"\n  [Eval{label}] Evaluating {n} scripts vs "
              f"'{evaluator.opponent_bot}' on '{evaluator.map_path}' "
              f"({evaluator.n_games} games each)")
        print(f"  {'#':>4s}  {'Script':<28s}  {'Bot':<14s}  "
              f"{'W/L/D':<10s}  {'Fitness':>7s}")
        print("  " + "─" * 75)

    # Reset existing W/L/D
    for s in population:
        s.wins = s.losses = s.draws = 0
        s.fitness = 0.0

    for idx, script in enumerate(population):
        result = evaluator.update_script_fitness(script)

        if verbose:
            wld = f"{result.wins}/{result.losses}/{result.draws}"
            print(f"  {idx+1:>4d}  {script.name:<28s}  "
                  f"{result.bot_used:<14s}  {wld:<10s}  "
                  f"{result.fitness:>7.4f}")

    # Sort by fitness descending
    population.sort(key=lambda s: s.fitness, reverse=True)

    elapsed = time.time() - t0
    if verbose:
        print("  " + "─" * 75)
        best = population[0]
        avg  = sum(s.fitness for s in population) / n
        print(f"  Best: {best.name} ({best.fitness:.4f}) | "
              f"Avg: {avg:.4f} | Time: {elapsed:.1f}s")

    return population


# ---------------------------------------------------------------------------
# Mode B: Rotating-opponents evaluation (round-robin against Java bots)
# ---------------------------------------------------------------------------

def evaluate_population_rotating(
    population:    list[StrategyScript],
    evaluator:     MicroRTSEvaluator,
    opponent_bots: Optional[list[str]] = None,
    n_games_each:  int  = 2,
    verbose:       bool = True,
    gen_label:     str  = "",
) -> list[StrategyScript]:
    """
    Evaluate each script against every Java opponent bot in `opponent_bots`.

    Each script plays `n_games_each` games against each opponent, then its
    aggregate win-rate (across all opponents) becomes its fitness.

    Parameters
    ----------
    population : list[StrategyScript]
    evaluator : MicroRTSEvaluator
        Server must be started; map_path and max_cycles are used from here.
    opponent_bots : list[str], optional
        Java bot names to rotate through. Defaults to ALL_BOT_TYPES.
    n_games_each : int
        Games per script per opponent.
    verbose : bool
    gen_label : str

    Returns
    -------
    list[StrategyScript]
        Population sorted descending by fitness.
    """
    if opponent_bots is None:
        # Exclude "Random" and "Passive" from rotation (too easy / trivial)
        opponent_bots = ["WorkerRush", "LightRush", "HeavyRush", "RandomBiased"]

    n = len(population)
    t0 = time.time()

    if verbose:
        label = f" [{gen_label}]" if gen_label else ""
        print(f"\n  [Eval-Rotate{label}] {n} scripts × "
              f"{len(opponent_bots)} opponents × {n_games_each} games")

    # Reset
    for s in population:
        s.wins = s.losses = s.draws = 0
        s.fitness = 0.0

    # Temporarily swap evaluator's opponent for each rotation
    original_opponent = evaluator.opponent_bot
    original_n_games  = evaluator.n_games
    evaluator.n_games = n_games_each

    from gesy_microrts_bridge.microrts_evaluator import (
        WIN_SCORE, DRAW_SCORE, LOSS_SCORE,
        BLOAT_PENALTY_FACTOR, BLOAT_THRESHOLD,
    )

    for opp in opponent_bots:
        evaluator.opponent_bot = opp
        if verbose:
            print(f"    vs {opp:<14s}", end="  ")

        for script in population:
            result = evaluator.evaluate_script(script)
            script.wins   += result.wins
            script.losses += result.losses
            script.draws  += result.draws

        if verbose:
            print("done")

    # Restore evaluator state
    evaluator.opponent_bot = original_opponent
    evaluator.n_games      = original_n_games

    # Recompute aggregate fitness from accumulated W/L/D
    for script in population:
        total = script.wins + script.losses + script.draws
        raw_score = (
            script.wins   * WIN_SCORE  +
            script.draws  * DRAW_SCORE +
            script.losses * LOSS_SCORE
        ) / max(total, 1)
        excess  = max(0, script.node_count() - BLOAT_THRESHOLD)
        penalty = excess * BLOAT_PENALTY_FACTOR
        script.fitness = max(0.0, raw_score - penalty)

    population.sort(key=lambda s: s.fitness, reverse=True)

    elapsed = time.time() - t0
    if verbose:
        best = population[0]
        avg  = sum(s.fitness for s in population) / n
        print(f"  Best: {best.name} ({best.fitness:.4f}) | "
              f"Avg: {avg:.4f} | Time: {elapsed:.1f}s")

    return population
