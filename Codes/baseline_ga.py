"""
baseline_ga.py — Standard Genetic Algorithm (NO trajectory memory)

COMPARISON BASELINE for TD-CCEA.
Same setup: 20 weights, chambers32x32, 7 rotating opponents, 21 generations.

What's DIFFERENT from TD-CCEA:
  - Fitness = raw win rate from THIS generation only (no history)
  - No environment-similarity weighting
  - Fixed mutation rate (no adaptive mutation)
  - Keeps diversity pruning and archive (to be fair)

If TD-CCEA beats this → trajectory memory helps.
"""

import math
import random
import time
import subprocess
import os

# ─── Paths ───
MICRORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "Simulation", "microrts")

# ─── Same 20 Weights as TD-CCEA ───
WEIGHT_NAMES = [
    "worker_ratio", "max_workers", "expand_threshold",
    "light_preference", "heavy_preference", "ranged_preference",
    "barracks_timing", "aggression", "attack_threshold",
    "defend_radius", "target_priority", "defense_trigger",
    "counter_rush", "tech_switch", "economy_pressure",
    "scout_priority", "group_attack", "second_barracks",
    "worker_fight_ratio", "focus_fire",
]

DIMENSIONS = len(WEIGHT_NAMES)  # 20

# ─── SAME Configuration as TD-CCEA v2 ───
MAP_PATH = "maps/chambers32x32.xml"
MAX_CYCLES = 10000

OPPONENT_ROTATION = [
    "EconomyMilitaryRush", "LightRush", "WorkerRush",
    "HeavyDefense", "RangedRush", "EconomyRush", "WorkerRushPlusPlus",
]

N_POP = 20
K_GAMES = 2
TOTAL_GENERATIONS = 21
FIXED_MUTATION = 0.15   # Fixed mutation (TD-CCEA uses adaptive)
A_MAX = 15
DELTA_S = 0.85

print(f"\n{'='*70}")
print(f"  BASELINE: Standard Genetic Algorithm (NO trajectory)")
print(f"{'='*70}")
print(f"  [CONFIG] Weights: {DIMENSIONS} | Map: {MAP_PATH}")
print(f"  [CONFIG] Population: {N_POP} | Generations: {TOTAL_GENERATIONS}")
print(f"  [CONFIG] Opponents: {len(OPPONENT_ROTATION)} rotating")
print(f"  [CONFIG] Mutation: FIXED at {FIXED_MUTATION} (NOT adaptive)")
print(f"  [CONFIG] Fitness: RAW win rate only (NO trajectory memory)")
print()


# ─── Java Game Server (same as TD-CCEA) ───
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


def euclidean_dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ─── Standard GA Agent (NO trajectory) ───
class GAAgent:
    def __init__(self, params=None):
        if params is None:
            self.params = [random.random() for _ in range(DIMENSIONS)]
        else:
            self.params = [max(0.0, min(1.0, float(x))) for x in params]
        self.fitness = 0.0  # Just raw fitness, no history

    def get_strategy_profile(self):
        profile = []
        if self.params[0] > 0.6: profile.append("heavy-economy")
        elif self.params[0] < 0.3: profile.append("minimal-workers")
        prefs = [self.params[3], self.params[4], self.params[5]]
        names = ["Light", "Heavy", "Ranged"]
        profile.append(f"{names[prefs.index(max(prefs))]}-focused")
        if self.params[7] > 0.7: profile.append("aggressive")
        elif self.params[7] < 0.3: profile.append("defensive")
        else: profile.append("balanced")
        if self.params[6] < 0.3: profile.append("early-barracks")
        elif self.params[6] > 0.7: profile.append("late-barracks")
        return " | ".join(profile)

    def mutate(self):
        """Fixed mutation rate — no adaptive adjustment."""
        offspring_params = []
        for p in self.params:
            noise = random.gauss(0, FIXED_MUTATION)
            offspring_params.append(p + noise)
        return GAAgent(offspring_params)


# ─── Main Evolution Loop ───
def run_baseline_ga():
    print(f"\n{'='*70}")
    print(f"  STARTING: Standard GA (no trajectory memory)")
    print(f"{'='*70}\n")

    server = EvolvedGameServer()
    server.start()

    population = [GAAgent() for _ in range(N_POP)]
    archive = []
    generation_results = []

    t_start = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        gen_start = time.time()

        # Same rotating opponent
        current_opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]

        print(f"  ┌─ Generation {gen}/{TOTAL_GENERATIONS} {'─'*45}")
        print(f"  │  Opponent: {current_opponent}")

        # ── Evaluate ──
        for agent in population:
            wins = 0
            for _ in range(K_GAMES):
                winner, cycles = server.run_game(
                    agent.params, current_opponent,
                    map_path=MAP_PATH, max_cycles=MAX_CYCLES
                )
                if winner == 0:
                    wins += 1
                elif winner == -1:
                    wins += 0.3

            # KEY DIFFERENCE: fitness = raw win rate ONLY
            # No trajectory, no history, no environment similarity
            agent.fitness = wins / K_GAMES

        # ── Results ──
        best_agent = max(population, key=lambda a: a.fitness)
        avg_fit = sum(a.fitness for a in population) / len(population)

        strat_count = {}
        for a in population:
            s = a.get_strategy_profile()
            strat_count[s] = strat_count.get(s, 0) + 1

        print(f"  │  Best: {best_agent.fitness:.3f} | Avg: {avg_fit:.3f} | Diversity: {len(strat_count)}")
        print(f"  │  Best: {best_agent.get_strategy_profile()}")

        generation_results.append({
            'gen': gen, 'opponent': current_opponent,
            'best_fit': best_agent.fitness, 'avg_fit': avg_fit,
            'best_profile': best_agent.get_strategy_profile(),
            'diversity': len(strat_count)
        })

        # ── Diversity-based survival (same as TD-CCEA) ──
        max_dist = math.sqrt(DIMENSIONS)
        survivors = []
        for agent in sorted(population, key=lambda a: a.fitness, reverse=True):
            is_redundant = False
            for u in survivors:
                dist = euclidean_dist(agent.params, u.params)
                if 1.0 - (dist / max_dist) > DELTA_S:
                    is_redundant = True
                    break
            if not is_redundant:
                survivors.append(agent)

        while len(survivors) < 10:
            survivors.append(GAAgent())

        # ── Archive ──
        sorted_gen = sorted(population, key=lambda a: a.fitness, reverse=True)
        for top_ag in sorted_gen[:3]:
            archive.append(GAAgent(top_ag.params))
        if len(archive) > A_MAX:
            archive = archive[-A_MAX:]

        # ── Breed ──
        population = survivors.copy()
        while len(population) < N_POP:
            candidates = random.sample(
                survivors + archive,
                min(3, len(survivors + archive))
            )
            parent = max(candidates, key=lambda a: a.fitness)
            population.append(parent.mutate())

        gen_time = time.time() - gen_start
        print(f"  │  Survivors: {len(survivors)} | Archive: {len(archive)} | Time: {gen_time:.1f}s")
        print(f"  └{'─'*60}")

    elapsed = time.time() - t_start
    server.stop()

    # ── Best from archive ──
    best = max(archive, key=lambda a: a.fitness)

    print(f"\n{'='*70}")
    print(f"  STANDARD GA COMPLETE")
    print(f"{'='*70}")
    print(f"  Total Time: {elapsed:.1f}s")
    print(f"  Total Games: {TOTAL_GENERATIONS * N_POP * K_GAMES}")

    # ── Evolution history ──
    print(f"\n  Evolution History:")
    print(f"  {'Gen':>4s}  {'Opponent':>25s}  {'Best':>6s}  {'Avg':>6s}  {'Div':>4s}  Profile")
    print(f"  {'─'*4}  {'─'*25}  {'─'*6}  {'─'*6}  {'─'*4}  {'─'*30}")
    for r in generation_results:
        print(f"  {r['gen']:4d}  {r['opponent']:>25s}  {r['best_fit']:6.3f}  {r['avg_fit']:6.3f}  {r['diversity']:4d}  {r['best_profile']}")

    # ── Validation ──
    print(f"\n{'='*70}")
    print(f"  VALIDATION: Standard GA vs ALL {len(OPPONENT_ROTATION)} opponents")
    print(f"{'='*70}")

    test_server = EvolvedGameServer()
    test_server.start()

    total_wins = 0
    total_games = 0

    for opponent in OPPONENT_ROTATION:
        opp_wins = 0
        games_per_opp = 3
        for g in range(games_per_opp):
            winner, cycles = test_server.run_game(best.params, opponent)
            if winner == 0:
                opp_wins += 1
            total_games += 1
        total_wins += opp_wins
        win_pct = opp_wins / games_per_opp * 100
        status = "✓" if opp_wins >= 2 else "✗"
        print(f"  {status} vs {opponent:25s}: {opp_wins}/{games_per_opp} ({win_pct:.0f}%)")

    test_server.stop()

    overall_pct = total_wins / total_games * 100
    print(f"\n  ┌─ STANDARD GA RESULTS ──────────────────────────")
    print(f"  │  Total Win Rate: {total_wins}/{total_games} ({overall_pct:.0f}%)")
    print(f"  │  Algorithm: Standard GA (NO trajectory memory)")
    print(f"  └────────────────────────────────────────────────────")

    # ── Visual replay ──
    print(f"\n  [Visualizer] Launching VISUAL replay: Standard GA vs LightRush")
    time.sleep(1)
    weights_str = ";".join(f"{w:.4f}" for w in best.params)
    cmd = [
        "java", "-cp", "lib/*;bin",
        "tests.EvolvedVisualGameRunner",
        f"EVOLVED:{weights_str}", "LightRush", MAP_PATH
    ]
    subprocess.run(cmd, cwd=MICRORTS_DIR)
    print("  [Visualizer] GUI closed.")


if __name__ == "__main__":
    run_baseline_ga()
