"""
baseline_hillclimb.py — Hill Climbing Baseline

COMPARISON BASELINE for TD-CCEA.
Same setup: 20 weights, chambers32x32, 7 rotating opponents, 21 generations.

What's DIFFERENT from TD-CCEA:
  - Single solution (no population)
  - Each generation: try 20 mutations of current best
  - Keep the best mutation if it beats the current
  - No trajectory, no diversity, no archive
  - Same total game budget: 21 × 20 × 2 = 840 games

If TD-CCEA beats this → population-based search helps.
"""

import math
import random
import time
import subprocess
import os

# ─── Paths ───
MICRORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "Simulation", "microrts")

# ─── Same 20 Weights ───
WEIGHT_NAMES = [
    "worker_ratio", "max_workers", "expand_threshold",
    "light_preference", "heavy_preference", "ranged_preference",
    "barracks_timing", "aggression", "attack_threshold",
    "defend_radius", "target_priority", "defense_trigger",
    "counter_rush", "tech_switch", "economy_pressure",
    "scout_priority", "group_attack", "second_barracks",
    "worker_fight_ratio", "focus_fire",
]

DIMENSIONS = len(WEIGHT_NAMES)

# ─── SAME Configuration ───
MAP_PATH = "maps/chambers32x32.xml"
MAX_CYCLES = 10000

OPPONENT_ROTATION = [
    "EconomyMilitaryRush", "LightRush", "WorkerRush",
    "HeavyDefense", "RangedRush", "EconomyRush", "WorkerRushPlusPlus",
]

N_CANDIDATES = 20   # Mutations to try per generation (same budget as population)
K_GAMES = 2
TOTAL_GENERATIONS = 21
MUTATION_RATE = 0.15

print(f"\n{'='*70}")
print(f"  BASELINE: Hill Climbing (single solution)")
print(f"{'='*70}")
print(f"  [CONFIG] Weights: {DIMENSIONS} | Map: {MAP_PATH}")
print(f"  [CONFIG] Candidates per gen: {N_CANDIDATES} | Generations: {TOTAL_GENERATIONS}")
print(f"  [CONFIG] Opponents: {len(OPPONENT_ROTATION)} rotating")
print(f"  [CONFIG] Method: Mutate best, keep if better (no population)")
print()


# ─── Java Game Server ───
class EvolvedGameServer:
    def __init__(self):
        self.process = None

    def start(self):
        cmd = ["java", "-cp", "lib/*;bin", "tests.EvolvedBatchGameRunner"]
        self.process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=MICRORTS_DIR
        )
        line = self.process.stdout.readline().strip()
        if line != "READY":
            raise RuntimeError(f"Failed to start: {line}")
        print("  [✓] Game server started")

    def run_game(self, weights, opponent, map_path=MAP_PATH, max_cycles=MAX_CYCLES):
        weights_str = ";".join(f"{w:.4f}" for w in weights)
        request = f"EVOLVED:{weights_str},{opponent},{map_path},{max_cycles}\n"
        self.process.stdin.write(request)
        self.process.stdin.flush()
        response = self.process.stdout.readline().strip()
        if response.startswith("RESULT:"):
            parts = response.replace("RESULT:", "").split(",")
            return int(parts[0]), int(parts[1])
        return -1, max_cycles

    def stop(self):
        if self.process:
            try:
                self.process.stdin.write("EXIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            print("  [✓] Game server stopped")


def mutate_params(params):
    """Create a mutant by adding Gaussian noise."""
    new_params = []
    for p in params:
        noise = random.gauss(0, MUTATION_RATE)
        new_params.append(max(0.0, min(1.0, p + noise)))
    return new_params


def get_strategy_profile(params):
    profile = []
    if params[0] > 0.6: profile.append("heavy-economy")
    elif params[0] < 0.3: profile.append("minimal-workers")
    prefs = [params[3], params[4], params[5]]
    names = ["Light", "Heavy", "Ranged"]
    profile.append(f"{names[prefs.index(max(prefs))]}-focused")
    if params[7] > 0.7: profile.append("aggressive")
    elif params[7] < 0.3: profile.append("defensive")
    else: profile.append("balanced")
    if params[6] < 0.3: profile.append("early-barracks")
    elif params[6] > 0.7: profile.append("late-barracks")
    return " | ".join(profile)


def evaluate(server, params, opponent):
    """Evaluate params with K_GAMES and return win rate."""
    wins = 0
    for _ in range(K_GAMES):
        winner, cycles = server.run_game(
            params, opponent,
            map_path=MAP_PATH, max_cycles=MAX_CYCLES
        )
        if winner == 0:
            wins += 1
        elif winner == -1:
            wins += 0.3
    return wins / K_GAMES


# ─── Main Hill Climbing Loop ───
def run_baseline_hillclimb():
    print(f"\n{'='*70}")
    print(f"  STARTING: Hill Climbing")
    print(f"{'='*70}\n")

    server = EvolvedGameServer()
    server.start()

    # Start with a random solution
    current_params = [random.random() for _ in range(DIMENSIONS)]
    current_fitness = 0.0
    generation_results = []

    t_start = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        gen_start = time.time()

        current_opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]

        print(f"  ┌─ Generation {gen}/{TOTAL_GENERATIONS} {'─'*45}")
        print(f"  │  Opponent: {current_opponent}")

        # Evaluate current solution against this opponent
        current_fitness = evaluate(server, current_params, current_opponent)

        # Try N_CANDIDATES mutations (same budget as population-based methods)
        best_candidate = current_params
        best_candidate_fitness = current_fitness
        improved = False

        for i in range(N_CANDIDATES - 1):  # -1 because we already evaluated current
            # KEY DIFFERENCE: just mutate the single current solution
            candidate = mutate_params(current_params)
            fitness = evaluate(server, candidate, current_opponent)

            if fitness > best_candidate_fitness:
                best_candidate = candidate
                best_candidate_fitness = fitness
                improved = True

        # Keep the best (hill climbing: only move uphill)
        if best_candidate_fitness >= current_fitness:
            current_params = best_candidate
            current_fitness = best_candidate_fitness

        print(f"  │  Fitness: {current_fitness:.3f} | Improved: {'YES' if improved else 'NO'}")
        print(f"  │  Best: {get_strategy_profile(current_params)}")

        generation_results.append({
            'gen': gen, 'opponent': current_opponent,
            'fitness': current_fitness, 'improved': improved,
            'best_profile': get_strategy_profile(current_params),
        })

        gen_time = time.time() - gen_start
        print(f"  │  Time: {gen_time:.1f}s")
        print(f"  └{'─'*60}")

    elapsed = time.time() - t_start
    server.stop()

    print(f"\n{'='*70}")
    print(f"  HILL CLIMBING COMPLETE")
    print(f"{'='*70}")
    print(f"  Total Time: {elapsed:.1f}s")
    print(f"  Total Games: {TOTAL_GENERATIONS * N_CANDIDATES * K_GAMES}")

    # ── History ──
    print(f"\n  Climbing History:")
    print(f"  {'Gen':>4s}  {'Opponent':>25s}  {'Fitness':>8s}  {'Improved':>9s}  Profile")
    print(f"  {'─'*4}  {'─'*25}  {'─'*8}  {'─'*9}  {'─'*30}")
    for r in generation_results:
        imp = "YES" if r['improved'] else "no"
        print(f"  {r['gen']:4d}  {r['opponent']:>25s}  {r['fitness']:8.3f}  {imp:>9s}  {r['best_profile']}")

    # ── Validation ──
    print(f"\n{'='*70}")
    print(f"  VALIDATION: Hill Climbing vs ALL {len(OPPONENT_ROTATION)} opponents")
    print(f"{'='*70}")

    test_server = EvolvedGameServer()
    test_server.start()

    total_wins = 0
    total_games = 0

    for opponent in OPPONENT_ROTATION:
        opp_wins = 0
        games_per_opp = 3
        for g in range(games_per_opp):
            winner, cycles = test_server.run_game(current_params, opponent)
            if winner == 0:
                opp_wins += 1
            total_games += 1
        total_wins += opp_wins
        win_pct = opp_wins / games_per_opp * 100
        status = "✓" if opp_wins >= 2 else "✗"
        print(f"  {status} vs {opponent:25s}: {opp_wins}/{games_per_opp} ({win_pct:.0f}%)")

    test_server.stop()

    overall_pct = total_wins / total_games * 100
    print(f"\n  ┌─ HILL CLIMBING RESULTS ──────────────────────────")
    print(f"  │  Total Win Rate: {total_wins}/{total_games} ({overall_pct:.0f}%)")
    print(f"  │  Algorithm: Hill Climbing (single solution, no population)")
    print(f"  └────────────────────────────────────────────────────")

    # ── Visual replay ──
    print(f"\n  [Visualizer] Launching VISUAL replay: Hill Climbing vs LightRush")
    time.sleep(1)
    weights_str = ";".join(f"{w:.4f}" for w in current_params)
    cmd = [
        "java", "-cp", "lib/*;bin",
        "tests.EvolvedVisualGameRunner",
        f"EVOLVED:{weights_str}", "LightRush", MAP_PATH
    ]
    subprocess.run(cmd, cwd=MICRORTS_DIR)
    print("  [Visualizer] GUI closed.")


if __name__ == "__main__":
    run_baseline_hillclimb()
