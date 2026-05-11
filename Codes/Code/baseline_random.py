"""
baseline_random.py — Random Search Baseline

COMPARISON BASELINE for TD-CCEA.
Same setup: 20 weights, chambers32x32, 7 rotating opponents, 21 generations.

What's DIFFERENT from TD-CCEA:
  - Each generation generates 20 COMPLETELY RANDOM weight vectors
  - No evolution, no mutation, no selection, no memory
  - Just keeps the single best performer ever found

If TD-CCEA can't beat this → the algorithm is broken.
This is the "floor" — the absolute minimum performance.
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

N_POP = 20
K_GAMES = 2
TOTAL_GENERATIONS = 21

print(f"\n{'='*70}")
print(f"  BASELINE: Random Search (no evolution)")
print(f"{'='*70}")
print(f"  [CONFIG] Weights: {DIMENSIONS} | Map: {MAP_PATH}")
print(f"  [CONFIG] Candidates per gen: {N_POP} | Generations: {TOTAL_GENERATIONS}")
print(f"  [CONFIG] Opponents: {len(OPPONENT_ROTATION)} rotating")
print(f"  [CONFIG] Method: Generate random vectors, keep best ever found")
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


# ─── Main Random Search Loop ───
def run_baseline_random():
    print(f"\n{'='*70}")
    print(f"  STARTING: Random Search")
    print(f"{'='*70}\n")

    server = EvolvedGameServer()
    server.start()

    best_ever_params = None
    best_ever_fitness = -1.0
    generation_results = []

    t_start = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        gen_start = time.time()

        current_opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]

        print(f"  ┌─ Generation {gen}/{TOTAL_GENERATIONS} {'─'*45}")
        print(f"  │  Opponent: {current_opponent}")

        # Generate N_POP completely random candidates
        gen_best_params = None
        gen_best_fitness = -1.0

        for i in range(N_POP):
            # KEY DIFFERENCE: completely random — no evolution, no inheritance
            params = [random.random() for _ in range(DIMENSIONS)]

            wins = 0
            for _ in range(K_GAMES):
                winner, cycles = server.run_game(
                    params, current_opponent,
                    map_path=MAP_PATH, max_cycles=MAX_CYCLES
                )
                if winner == 0:
                    wins += 1
                elif winner == -1:
                    wins += 0.3

            fitness = wins / K_GAMES

            if fitness > gen_best_fitness:
                gen_best_fitness = fitness
                gen_best_params = params.copy()

        # Track best ever
        if gen_best_fitness > best_ever_fitness:
            best_ever_fitness = gen_best_fitness
            best_ever_params = gen_best_params.copy()

        avg_fit = gen_best_fitness * 0.5  # rough estimate (random is mostly bad)

        print(f"  │  Gen Best: {gen_best_fitness:.3f} | Best Ever: {best_ever_fitness:.3f}")
        print(f"  │  Best: {get_strategy_profile(gen_best_params)}")

        generation_results.append({
            'gen': gen, 'opponent': current_opponent,
            'best_fit': gen_best_fitness, 'best_ever': best_ever_fitness,
            'best_profile': get_strategy_profile(gen_best_params),
        })

        gen_time = time.time() - gen_start
        print(f"  │  Time: {gen_time:.1f}s")
        print(f"  └{'─'*60}")

    elapsed = time.time() - t_start
    server.stop()

    print(f"\n{'='*70}")
    print(f"  RANDOM SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Total Time: {elapsed:.1f}s")
    print(f"  Total Games: {TOTAL_GENERATIONS * N_POP * K_GAMES}")

    # ── Evolution history ──
    print(f"\n  Search History:")
    print(f"  {'Gen':>4s}  {'Opponent':>25s}  {'GenBest':>8s}  {'BestEver':>9s}  Profile")
    print(f"  {'─'*4}  {'─'*25}  {'─'*8}  {'─'*9}  {'─'*30}")
    for r in generation_results:
        print(f"  {r['gen']:4d}  {r['opponent']:>25s}  {r['best_fit']:8.3f}  {r['best_ever']:9.3f}  {r['best_profile']}")

    # ── Validation ──
    print(f"\n{'='*70}")
    print(f"  VALIDATION: Random Search vs ALL {len(OPPONENT_ROTATION)} opponents")
    print(f"{'='*70}")

    test_server = EvolvedGameServer()
    test_server.start()

    total_wins = 0
    total_games = 0

    for opponent in OPPONENT_ROTATION:
        opp_wins = 0
        games_per_opp = 3
        for g in range(games_per_opp):
            winner, cycles = test_server.run_game(best_ever_params, opponent)
            if winner == 0:
                opp_wins += 1
            total_games += 1
        total_wins += opp_wins
        win_pct = opp_wins / games_per_opp * 100
        status = "✓" if opp_wins >= 2 else "✗"
        print(f"  {status} vs {opponent:25s}: {opp_wins}/{games_per_opp} ({win_pct:.0f}%)")

    test_server.stop()

    overall_pct = total_wins / total_games * 100
    print(f"\n  ┌─ RANDOM SEARCH RESULTS ──────────────────────────")
    print(f"  │  Total Win Rate: {total_wins}/{total_games} ({overall_pct:.0f}%)")
    print(f"  │  Algorithm: Random Search (NO evolution)")
    print(f"  └────────────────────────────────────────────────────")

    # ── Visual replay ──
    print(f"\n  [Visualizer] Launching VISUAL replay: Random Search vs LightRush")
    time.sleep(1)
    weights_str = ";".join(f"{w:.4f}" for w in best_ever_params)
    cmd = [
        "java", "-cp", "lib/*;bin",
        "tests.EvolvedVisualGameRunner",
        f"EVOLVED:{weights_str}", "LightRush", MAP_PATH
    ]
    subprocess.run(cmd, cwd=MICRORTS_DIR)
    print("  [Visualizer] GUI closed.")


if __name__ == "__main__":
    run_baseline_random()
