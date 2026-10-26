"""
sa_profiler.py — Phase 1: Offline Sensitivity Profiling for SA-TDCCEA.

Purpose
───────
Before the evolutionary loop starts, test each of the 35 genes at its
boundary conditions (w_j = 0.0 vs w_j = 1.0, all others fixed at 0.5) in
truncated 8-minute SC2 games.  Measure the delta in the composite score
(RUR + APU) to derive the influence score I_j.

Then map each I_j to a static per-gene mutation boundary μ_j:

    μ_j = MU_MIN + (1 − I_j / I_max) × (MU_MAX − MU_MIN)

High-influence genes (large Δ) → μ_j ≈ MU_MIN (fine-tune carefully).
Low-influence genes  (small Δ) → μ_j ≈ MU_MAX (explore freely).

Output
──────
Saves to config.SENSITIVITY_FILE (results/sensitivity_scores_sc2.json):
{
    "influence_scores":           [35 floats],
    "normalized_mutation_rates":  [35 floats],  ← μ_j array
    "weight_names":               [35 strings],
    "baseline_score":             float
}

Usage
─────
    python sa_profiler.py              # run full profiling
    python sa_profiler.py --dry-run   # skip SC2 games, load cached file
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import List

from sc2.main import run_game
from sc2 import maps
from sc2.data import Difficulty, Race
from sc2.player import Bot, Computer

import config as cfg
from bot import EvolvedSC2Bot


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_results_dir() -> None:
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)


def _run_single(
    weights:       List[float],
    race:          Race,
    opponent_race: Race,
    difficulty:    Difficulty,
    map_name:      str,
    time_limit:    int,
) -> dict:
    """Run one SC2 game (synchronously) and return the bot's fitness_result dict."""
    bot = EvolvedSC2Bot(weights, race)
    run_game(
        maps.get(map_name),
        [Bot(race, bot), Computer(opponent_race, difficulty)],
        realtime        = False,
        game_time_limit = time_limit,
    )
    return bot.fitness_result or {
        "f_i": 0.0, "win_rate": 0.0, "apu": 0.0, "pbr": 0.0, "rur": 0.0, "steps": 0,
    }


def _composite_score(result: dict) -> float:
    """
    Influence composite = weighted average of RUR and APU.
    (Does NOT include win-rate, which is too noisy for boundary testing.)
    """
    return (cfg.SENSITIVITY_RUR_WEIGHT * result["rur"] +
            cfg.SENSITIVITY_APU_WEIGHT * result["apu"])


def _evaluate_weights_sync(
    weights: List[float],
    n_games: int = cfg.SENSITIVITY_GAMES,
) -> float:
    """Run n_games with `weights` and return average composite score."""
    scores = []
    for _ in range(n_games):
        result = _run_single(
            weights,
            race          = cfg.BOT_RACE,
            opponent_race = cfg.OPPONENT_RACE,
            difficulty    = Difficulty.VeryEasy,   # profiling at easiest difficulty
            map_name      = cfg.MAP_NAME,
            time_limit    = cfg.SENSITIVITY_TIME_LIMIT,
        )
        scores.append(_composite_score(result))
    return sum(scores) / len(scores)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PROFILING ROUTINE
# ─────────────────────────────────────────────────────────────────────────────

def run_sensitivity_analysis() -> List[float]:
    """
    Full offline sensitivity analysis.

    Returns the 35D per-gene mutation rate array (μ_j).
    Also saves results to config.SENSITIVITY_FILE.
    """
    _ensure_results_dir()

    print()
    print("=" * 70)
    print("  SA-TDCCEA PHASE 1: OFFLINE SENSITIVITY PROFILING")
    print("  SC2 environment — measuring Δ(RUR + APU) per gene")
    print("=" * 70)
    print(f"  Genes:       {cfg.DIMENSIONS}")
    print(f"  Games/test:  {cfg.SENSITIVITY_GAMES}")
    print(f"  Time limit:  {cfg.SENSITIVITY_TIME_LIMIT}s per game (8 min)")
    print(f"  Difficulty:  VeryEasy (profiling phase)")
    print()

    baseline_weights = [0.5] * cfg.DIMENSIONS

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("  [1/3] Measuring baseline (all genes = 0.5) ...")
    t0 = time.time()
    baseline_score = _evaluate_weights_sync(baseline_weights)
    print(f"        Baseline composite score: {baseline_score:.4f}  "
          f"({time.time()-t0:.0f}s)")
    print()

    # ── Per-gene boundary tests ───────────────────────────────────────────────
    print("  [2/3] Testing each gene at 0.0 and 1.0 ...")
    print()
    print(f"  {'#':>3s}  {'Gene':<28s}  {'Score@0.0':>10s}  "
          f"{'Score@1.0':>10s}  {'Influence':>10s}")
    print("  " + "─" * 70)

    influence_scores: List[float] = []

    for j in range(cfg.DIMENSIONS):
        # Test at 0.0
        test_low    = baseline_weights[:]
        test_low[j] = 0.0
        score_low   = _evaluate_weights_sync(test_low)

        # Test at 1.0
        test_high    = baseline_weights[:]
        test_high[j] = 1.0
        score_high   = _evaluate_weights_sync(test_high)

        influence = abs(score_high - score_low)
        influence_scores.append(influence)

        bar_len = int(influence * 40)
        bar     = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  w{j:>2d}  {cfg.GENE_NAMES[j]:<28s}  "
              f"{score_low:>10.4f}  {score_high:>10.4f}  "
              f"{influence:>10.4f}  [{bar[:20]}]")

    print()

    # ── Compute μ_j array ────────────────────────────────────────────────────
    print("  [3/3] Computing per-gene mutation rates (μ_j) ...")
    I_max = max(influence_scores) if max(influence_scores) > 0 else 1.0
    mu_j  = [
        cfg.MU_MIN + (1.0 - I_j / I_max) * (cfg.MU_MAX - cfg.MU_MIN)
        for I_j in influence_scores
    ]

    # ── Rank & display ────────────────────────────────────────────────────────
    print()
    print("  INFLUENCE RANKING (most → least sensitive):")
    print("  " + "─" * 60)
    ranked = sorted(range(cfg.DIMENSIONS), key=lambda i: influence_scores[i], reverse=True)
    for rank, idx in enumerate(ranked):
        marker = " ◄◄◄ HIGH SENSITIVITY" if rank < 5 else ""
        print(f"  {rank+1:>2d}. w{idx:>2d}  {cfg.GENE_NAMES[idx]:<28s}  "
              f"I={influence_scores[idx]:.4f}  μ_j={mu_j[idx]:.4f}{marker}")

    # ── Save ──────────────────────────────────────────────────────────────────
    result = {
        "influence_scores":          influence_scores,
        "normalized_mutation_rates": mu_j,
        "weight_names":              cfg.GENE_NAMES,
        "baseline_score":            baseline_score,
        "mu_min":                    cfg.MU_MIN,
        "mu_max":                    cfg.MU_MAX,
        "i_max":                     I_max,
    }
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            cfg.SENSITIVITY_FILE)
    with open(filepath, "w") as fh:
        json.dump(result, fh, indent=2)

    print()
    print(f"  Saved → {cfg.SENSITIVITY_FILE}")
    print()
    return mu_j


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD HELPER (called by main.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_or_run_sensitivity() -> List[float]:
    """
    Load μ_j from cache if available, otherwise run full profiling.
    Returns the 35D per-gene mutation rate array.
    """
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            cfg.SENSITIVITY_FILE)
    if os.path.exists(filepath):
        print(f"  [SA] Loading cached sensitivity scores → {cfg.SENSITIVITY_FILE}")
        with open(filepath) as fh:
            data = json.load(fh)
        mu_j = data["normalized_mutation_rates"]
        if len(mu_j) != cfg.DIMENSIONS:
            print(f"  [SA] WARNING: cached file has {len(mu_j)} entries but "
                  f"DIMENSIONS={cfg.DIMENSIONS}. Re-running profiling.")
            return run_sensitivity_analysis()
        print(f"  [SA] Loaded {cfg.DIMENSIONS} per-gene mutation rates.")
        for j, (name, mu) in enumerate(zip(cfg.GENE_NAMES, mu_j)):
            bar = "█" * int(mu * 60) + "░" * (12 - int(mu * 60))
            print(f"       w{j:>2d}  {name:<28s}  μ_j={mu:.4f}  [{bar}]")
        print()
        return mu_j
    else:
        print(f"  [SA] No cached file found at {cfg.SENSITIVITY_FILE}.")
        print("  [SA] Running offline sensitivity analysis ...")
        return run_sensitivity_analysis()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT  (standalone run)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("  [SA] Dry-run mode: loading existing file (or using defaults).")
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                cfg.SENSITIVITY_FILE)
        if os.path.exists(filepath):
            with open(filepath) as fh:
                data = json.load(fh)
            print(json.dumps(data, indent=2))
        else:
            print("  No cached file found. Run without --dry-run to generate.")
    else:
        mu_j = run_sensitivity_analysis()
        print("  Done. μ_j array:")
        print(mu_j)
