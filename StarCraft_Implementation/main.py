"""
main.py — Orchestration & Generational Loop for CCE-TDCCEA StarCraft II.

Execution flow
──────────────
  Phase 0  — Setup & imports; ensure results dir exists.
  Phase 1  — SA-TDCCEA sensitivity profiling (or load cached μ_j array).
  Phase 2  — CCE-TDCCEA generational loop (TOTAL_GENERATIONS = 5):
               For each generation g (1 … TOTAL_GENERATIONS):
                 difficulty = DIFFICULTY_SCHEDULE[g-1]  (ramps VeryEasy → Harder)
                 centroid_a, centroid_b, centroid_c  computed from populations
                 Evaluate Pop A: each agent paired with best_b + best_c → f_i → trajectory
                 Evaluate Pop B: each agent paired with best_a + best_c → f_i → trajectory
                 Evaluate Pop C: each agent paired with best_a + best_b → f_i → trajectory
                 evolve_generation() → prune + archive + breed
                 Save elite genome to results/
  Phase 3  — Final benchmark: elite vs Harder AI (K validation games).

How games are executed
──────────────────────
  asyncio.run(_run_game(...)) wraps each SC2 game in a fresh event loop.
  Games run sequentially (python-sc2 is not safely parallelisable with
  a single SC2 client).  Each game runs up to GAME_TIME_LIMIT seconds
  of in-game time at fastest speed.

K_GAMES averaging
─────────────────
  Each agent's fitness for a generation is the mean of K_GAMES games
  against the same built-in AI difficulty.  This reduces stochasticity
  in the SC2 environment (random spawn locations, opponent AI variance).

Command-line flags
──────────────────
  python main.py                     # full run (SA profiling + evolution)
  python main.py --no-sa             # skip SA; use flat MU_DEFAULT rates
  python main.py --dry-run           # single game test with random genome
  python main.py --benchmark         # load elite from file and benchmark
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import List, Optional

from sc2.main import run_game
from sc2 import maps
from sc2.data import Difficulty, Race
from sc2.player import Bot, Computer

import config as cfg
from bot import EvolvedSC2Bot
from cce_tdccea import CCEACore, SubAgent
from sa_profiler import load_or_run_sensitivity


# ─────────────────────────────────────────────────────────────────────────────
#  GAME RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def _run_game_sync(
    weights:    List[float],
    difficulty: Difficulty,
) -> dict:
    """
    Run one SC2 game with EvolvedSC2Bot against the built-in AI.
    run_game() is a synchronous function that internally manages its own
    asyncio event loop.  Returns the bot's fitness_result dict.
    """
    bot = EvolvedSC2Bot(weights, cfg.BOT_RACE)
    run_game(
        maps.get(cfg.MAP_NAME),
        [Bot(cfg.BOT_RACE, bot),
         Computer(cfg.OPPONENT_RACE, difficulty)],
        realtime        = False,
        game_time_limit = cfg.GAME_TIME_LIMIT,
    )
    return bot.fitness_result or {
        "f_i": 0.0, "win_rate": 0.0, "apu": 0.0, "pbr": 0.0, "rur": 0.0, "steps": 0,
    }


def evaluate_genome(
    weights:    List[float],
    difficulty: Difficulty,
    k_games:    int = cfg.K_GAMES,
) -> float:
    """
    Evaluate one 35D genome over k_games games and return average f_i.
    Calls run_game() sequentially (each call blocks until game ends).
    """
    total_f = 0.0
    for _ in range(k_games):
        result   = _run_game_sync(weights, difficulty)
        total_f += result["f_i"]
    return total_f / k_games


def evaluate_genome_verbose(
    weights:    List[float],
    difficulty: Difficulty,
    k_games:    int = cfg.K_GAMES,
) -> dict:
    """Like evaluate_genome but returns per-metric averages."""
    totals: dict = {"f_i": 0.0, "win_rate": 0.0, "apu": 0.0, "pbr": 0.0, "rur": 0.0}
    for _ in range(k_games):
        r = _run_game_sync(weights, difficulty)
        for k in totals:
            totals[k] += r.get(k, 0.0)
    return {k: round(v / k_games, 4) for k, v in totals.items()}


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _save_elite(genome: List[float], gen: int, metrics: Optional[dict] = None) -> None:
    """Write the elite genome and optional metrics to results/."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    data = {"generation": gen, "genome": genome}
    if metrics:
        data["metrics"] = metrics
    path = os.path.join(cfg.RESULTS_DIR, f"elite_gen{gen:03d}.json")
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"  [Elite] Saved → {path}")


def _save_log(log_entries: list) -> None:
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    with open(cfg.LOG_FILE, "w") as fh:
        json.dump(log_entries, fh, indent=2)


def _load_latest_elite() -> Optional[List[float]]:
    """Try to load the most recent elite_genNNN.json from results/."""
    if not os.path.isdir(cfg.RESULTS_DIR):
        return None
    files = sorted(
        f for f in os.listdir(cfg.RESULTS_DIR)
        if f.startswith("elite_gen") and f.endswith(".json")
    )
    if not files:
        return None
    path = os.path.join(cfg.RESULTS_DIR, files[-1])
    with open(path) as fh:
        data = json.load(fh)
    genome = data.get("genome")
    print(f"  Loaded elite from {path}")
    return genome


def _print_gen_header(gen: int, difficulty: Difficulty) -> None:
    print()
    print(f"  {'─'*60}")
    print(f"  Generation {gen}/{cfg.TOTAL_GENERATIONS}  │  "
          f"Difficulty: {difficulty.name}  │  "
          f"Race: {cfg.BOT_RACE.name}")
    print(f"  {'─'*60}")


def _print_pop_progress(label: str, agents: list, gen: int) -> None:
    e_vals = [a.e_i for a in agents]
    best   = max(e_vals)
    avg    = sum(e_vals) / len(e_vals)
    print(f"  {label:6s}  best_e={best:.4f}  avg_e={avg:.4f}  "
          f"pop_size={len(agents)}")


# ─────────────────────────────────────────────────────────────────────────────
#  DRY-RUN HELPER
# ─────────────────────────────────────────────────────────────────────────────

def dry_run() -> None:
    """Run one SC2 game with a random 35D genome to verify the pipeline."""
    import random
    print()
    print("=" * 60)
    print("  DRY-RUN: single game with random 35D genome")
    print("=" * 60)
    weights = [random.random() for _ in range(cfg.DIMENSIONS)]
    print(f"  Genome: {[round(w, 3) for w in weights]}")
    print(f"  Race:   {cfg.BOT_RACE.name}")
    print(f"  Map:    {cfg.MAP_NAME}")
    print(f"  Limit:  {cfg.GAME_TIME_LIMIT}s")
    print()
    t0      = time.time()
    result  = _run_game_sync(weights, Difficulty.VeryEasy)
    elapsed = time.time() - t0
    print()
    print(f"  Result: {json.dumps(result, indent=4)}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  BENCHMARK HELPER
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(genome: Optional[List[float]] = None, n_games: int = 5) -> None:
    """
    Benchmark a genome against BENCHMARK_DIFFICULTY (Harder = Level 5).
    If genome is None, loads the most recent elite from results/.
    """
    print()
    print("=" * 60)
    print("  BENCHMARK vs Harder (Level 5)")
    print("=" * 60)
    if genome is None:
        genome = _load_latest_elite()
    if genome is None:
        print("  No elite genome found.  Run evolution first.")
        return

    metrics = evaluate_genome_verbose(genome, cfg.BENCHMARK_DIFFICULTY, k_games=n_games)
    print()
    print(f"  Games:    {n_games}")
    print(f"  Win rate: {metrics['win_rate']*100:.1f}%")
    print(f"  APU:      {metrics['apu']*100:.1f}%")
    print(f"  PBR:      {metrics['pbr']*100:.1f}%")
    print(f"  RUR:      {metrics['rur']*100:.1f}%")
    print(f"  f_i:      {metrics['f_i']:.4f}")
    print()

    # Save benchmark result
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    bpath = os.path.join(cfg.RESULTS_DIR, "benchmark_result.json")
    with open(bpath, "w") as fh:
        json.dump({"difficulty": cfg.BENCHMARK_DIFFICULTY.name,
                   "n_games": n_games, "metrics": metrics, "genome": genome}, fh, indent=2)
    print(f"  Saved benchmark → {bpath}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN EVOLUTIONARY LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_evolution(mu_j: Optional[List[float]]) -> List[float]:
    """
    Full CCE-TDCCEA + SA-TDCCEA generational loop.

    Args:
        mu_j: 35D per-gene mutation rates from SA profiler.
              Pass None to use flat MU_DEFAULT for all genes.

    Returns:
        Elite 35D genome after TOTAL_GENERATIONS.
    """
    print()
    print("#" * 70)
    print("  CCE-TDCCEA + SA-TDCCEA  |  StarCraft II")
    print("#" * 70)
    print(f"  Generations : {cfg.TOTAL_GENERATIONS}")
    print(f"  Pop size    : {cfg.N_POP} per sub-population (A + B + C)")
    print(f"  K games     : {cfg.K_GAMES} per agent per generation")
    print(f"  Race        : {cfg.BOT_RACE.name}")
    print(f"  Map         : {cfg.MAP_NAME}")
    print(f"  SA mode     : {'ON (per-gene μ_j)' if mu_j else 'OFF (flat mutation)'}")
    print()

    ccea = CCEACore(mu_j_array=mu_j)
    log_entries: list = []
    t_total = time.time()

    for gen in range(1, cfg.TOTAL_GENERATIONS + 1):

        difficulty = cfg.get_difficulty(gen - 1)   # 0-based index
        _print_gen_header(gen, difficulty)
        t_gen = time.time()

        # ── Compute sub-population centroids ─────────────────────────────────
        centroid_a = ccea.centroid_a()
        centroid_b = ccea.centroid_b()
        centroid_c = ccea.centroid_c()

        # ── Evaluate Pop A  (paired with current best_b, best_c) ─────────────
        print(f"  Evaluating Pop A ({cfg.N_POP} agents × {cfg.K_GAMES} games) ...")
        for i, agent_a in enumerate(ccea.pop_a):
            genome  = ccea.assemble_genome(agent_a, ccea.best_b, ccea.best_c)
            fitness = evaluate_genome(genome, difficulty)
            ccea.update_fitness_a(agent_a, fitness, centroid_a)
            if (i + 1) % 5 == 0:
                print(f"    A[{i+1:>2d}/{cfg.N_POP}]  f_i={fitness:.4f}  "
                      f"e_i={agent_a.e_i:.4f}")

        # ── Evaluate Pop B  (paired with current best_a, best_c) ─────────────
        print(f"  Evaluating Pop B ({cfg.N_POP} agents × {cfg.K_GAMES} games) ...")
        for i, agent_b in enumerate(ccea.pop_b):
            genome  = ccea.assemble_genome(ccea.best_a, agent_b, ccea.best_c)
            fitness = evaluate_genome(genome, difficulty)
            ccea.update_fitness_b(agent_b, fitness, centroid_b)
            if (i + 1) % 5 == 0:
                print(f"    B[{i+1:>2d}/{cfg.N_POP}]  f_i={fitness:.4f}  "
                      f"e_i={agent_b.e_i:.4f}")

        # ── Evaluate Pop C  (paired with current best_a, best_b) ─────────────
        print(f"  Evaluating Pop C ({cfg.N_POP} agents × {cfg.K_GAMES} games) ...")
        for i, agent_c in enumerate(ccea.pop_c):
            genome  = ccea.assemble_genome(ccea.best_a, ccea.best_b, agent_c)
            fitness = evaluate_genome(genome, difficulty)
            ccea.update_fitness_c(agent_c, fitness, centroid_c)
            if (i + 1) % 5 == 0:
                print(f"    C[{i+1:>2d}/{cfg.N_POP}]  f_i={fitness:.4f}  "
                      f"e_i={agent_c.e_i:.4f}")

        # ── Apply TD-CCEA survival, archive, breed ────────────────────────────
        ccea.evolve_generation()

        # ── Report generation summary ─────────────────────────────────────────
        summary = ccea.state_summary()
        gen_time = time.time() - t_gen
        summary["gen_time_s"] = round(gen_time, 1)
        summary["difficulty"] = difficulty.name
        log_entries.append(summary)

        print()
        _print_pop_progress("Pop A", ccea.pop_a, gen)
        _print_pop_progress("Pop B", ccea.pop_b, gen)
        _print_pop_progress("Pop C", ccea.pop_c, gen)
        print(f"  Gen {gen} done in {gen_time:.0f}s  │  "
              f"best_A={summary['best_e_a']:.4f}  "
              f"best_B={summary['best_e_b']:.4f}  "
              f"best_C={summary['best_e_c']:.4f}")

        # ── Save elite ────────────────────────────────────────────────────────
        if gen % cfg.ELITE_SAVE_FREQ == 0:
            elite_genome = ccea.get_elite_genome()
            _save_elite(elite_genome, gen, summary)

        _save_log(log_entries)

    total_elapsed = time.time() - t_total
    print()
    print("=" * 60)
    print(f"  Evolution complete in {total_elapsed:.0f}s "
          f"({total_elapsed/60:.1f} min)")
    print("=" * 60)

    final_elite = ccea.get_elite_genome()
    _save_elite(final_elite, gen, {"final": True})
    return final_elite


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]

    if "--dry-run" in args:
        dry_run()
        return

    if "--benchmark" in args:
        benchmark()
        return

    print()
    print("#" * 70)
    print("  CCE-TDCCEA + SA-TDCCEA for StarCraft II")
    print("#" * 70)
    print(f"  Race: {cfg.BOT_RACE.name}  |  Map: {cfg.MAP_NAME}")
    print(f"  Gens: {cfg.TOTAL_GENERATIONS}  |  Pop: {cfg.N_POP}×3  "
          f"|  K: {cfg.K_GAMES}")
    print()

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    # ── Phase 1: SA Sensitivity Profiling ────────────────────────────────────
    if "--no-sa" in args:
        print("  [SA] Skipping sensitivity profiling (--no-sa flag).")
        mu_j = None
    else:
        mu_j = load_or_run_sensitivity()

    # ── Phase 2: Evolutionary Loop ────────────────────────────────────────────
    elite_genome = run_evolution(mu_j)

    # ── Phase 3: Benchmark ────────────────────────────────────────────────────
    print()
    print("  Running final benchmark vs Harder (Level 5) ...")
    benchmark(elite_genome, n_games=5)


if __name__ == "__main__":
    main()
