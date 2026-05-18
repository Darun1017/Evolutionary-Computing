"""
evolve_vs_custom.py - Evolve TD-CCEA against a CUSTOM opponent strategy

Put your opponent's 20 weight values in the OPPONENT_WEIGHTS list below.
Our TD-CCEA algorithm will evolve to beat that specific opponent.
"""

import math, random, time, subprocess, os, sys

os.environ["PYTHONIOENCODING"] = "utf-8"

# ===============================================================
#  PUT YOUR OPPONENT'S 20 WEIGHTS HERE
# ===============================================================
# Each value should be between 0.0 and 1.0
# Order: worker_ratio, max_workers, expand_threshold,
#         light_pref, heavy_pref, ranged_pref,
#         barracks_timing, aggression, attack_threshold,
#         defend_radius, target_priority, defense_trigger,
#         counter_rush, tech_switch, economy_pressure,
#         scout_priority, group_attack, second_barracks,
#         worker_fight_ratio, focus_fire

OPPONENT_WEIGHTS = [
    1.0, 1.0, 1.0, 1.0, 1.0,    # w0-w4
    1.0, 1.0, 1.0, 1.0, 1.0,    # w5-w9
    1.0, 1.0, 1.0, 1.0, 1.0,    # w10-w14
    1.0, 1.0, 1.0, 1.0, 1.0,    # w15-w19
]

# ===============================================================
#  CONFIGURATION
# ===============================================================
MICRORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "Simulation", "microrts")

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

MAP_PATH = "maps/chambers32x32.xml"
MAX_CYCLES = 10000

N_POP = 20
K_GAMES = 5           # games per evaluation (alternating sides for fairness)
TOTAL_GENERATIONS = 30
VALIDATION_GAMES = 20  # final validation games


# ===============================================================
#  GAME SERVER
# ===============================================================
class GameServer:
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

    def run_evolved_vs_evolved(self, weights1, weights2):
        """Run two evolved AIs against each other."""
        w1_str = ";".join(f"{w:.4f}" for w in weights1)
        w2_str = ";".join(f"{w:.4f}" for w in weights2)
        request = f"EVOLVED:{w1_str},EVOLVED:{w2_str},{MAP_PATH},{MAX_CYCLES}\n"
        self.process.stdin.write(request)
        self.process.stdin.flush()
        response = self.process.stdout.readline().strip()
        if response.startswith("RESULT:"):
            parts = response.replace("RESULT:", "").split(",")
            return int(parts[0]), int(parts[1])
        return -1, MAX_CYCLES

    def stop(self):
        if self.process:
            try:
                self.process.stdin.write("EXIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()


# ===============================================================
#  HELPERS
# ===============================================================
def euclidean_dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def evaluate_vs_opponent(server, our_weights, opp_weights):
    """Play K_GAMES against the opponent, alternating sides for fairness.
    Returns win rate 0.0 to 1.0."""
    wins = 0
    for game in range(K_GAMES):
        if game % 2 == 0:
            winner, _ = server.run_evolved_vs_evolved(our_weights, opp_weights)
            if winner == 0:
                wins += 1
            elif winner == -1:
                wins += 0.3
        else:
            winner, _ = server.run_evolved_vs_evolved(opp_weights, our_weights)
            if winner == 1:
                wins += 1
            elif winner == -1:
                wins += 0.3
    return wins / K_GAMES


# ===============================================================
#  TD-CCEA AGENT
# ===============================================================
class TDCCEAAgent:
    def __init__(self, params=None):
        if params is None:
            self.params = [random.random() for _ in range(DIMENSIONS)]
        else:
            self.params = [max(0.0, min(1.0, float(x))) for x in params]
        self.H_i = []
        self.e_i = 0.0
        self.mu_i = 0.0
        self.raw_fitness = 0.0

    def update_history(self, fitness, current_environment):
        self.H_i.append((fitness, current_environment))
        if len(self.H_i) > 7:
            self.H_i.pop(0)

    def compute_dynamic_trajectory(self, current_environment):
        if len(self.H_i) == 0:
            self.e_i = 0.0
            return self.e_i
        dynamic_weights = []
        max_dist = math.sqrt(DIMENSIONS) * 1.0
        for past_fitness, past_env in self.H_i:
            dist = euclidean_dist(current_environment, past_env)
            similarity = 1.0 - (dist / max_dist)
            dynamic_weights.append(similarity)
        total_w = sum(dynamic_weights)
        if total_w == 0:
            dynamic_weights = [1.0 / len(dynamic_weights)] * len(dynamic_weights)
        else:
            dynamic_weights = [w / total_w for w in dynamic_weights]
        self.e_i = 0.0
        for i in range(len(self.H_i)):
            self.e_i += dynamic_weights[i] * self.H_i[i][0]
        return self.e_i

    def compute_adaptive_mutation(self):
        self.mu_i = 0.15 * (1.0 - self.e_i) + 0.03
        return self.mu_i

    def mutate(self):
        offspring_params = []
        for p in self.params:
            noise = random.gauss(0, self.mu_i)
            offspring_params.append(p + noise)
        return TDCCEAAgent(offspring_params)


# ===============================================================
#  MAIN
# ===============================================================
def main():
    # Validate opponent weights
    if len(OPPONENT_WEIGHTS) != DIMENSIONS:
        print(f"  ERROR: OPPONENT_WEIGHTS must have exactly {DIMENSIONS} values!")
        print(f"  You provided {len(OPPONENT_WEIGHTS)} values.")
        return

    opp = [max(0.0, min(1.0, float(w))) for w in OPPONENT_WEIGHTS]

    print()
    print("=" * 70)
    print("  TD-CCEA EVOLUTION vs CUSTOM OPPONENT")
    print("=" * 70)
    print()
    print("  Opponent weights:")
    for i, name in enumerate(WEIGHT_NAMES):
        bar_len = int(opp[i] * 20)
        bar = "#" * bar_len + "." * (20 - bar_len)
        print(f"    w{i:2d} {name:20s} [{bar}] {opp[i]:.2f}")
    print()
    print(f"  Generations: {TOTAL_GENERATIONS} | Population: {N_POP}")
    print(f"  Games per eval: {K_GAMES} (alternating sides)")
    print(f"  Map: {MAP_PATH}")
    print()

    server = GameServer()
    server.start()
    print("  [OK] Game server started")
    print()

    # Initialize TD-CCEA
    population = [TDCCEAAgent() for _ in range(N_POP)]
    archive = []
    DELTA_S, DELTA_F, A_MAX = 0.85, 0.20, 15

    gen_scores = []
    t_start = time.time()

    print("  " + "-" * 60)
    print(f"  {'Gen':>4s}  {'Best WR':>8s}  {'Avg WR':>8s}  {'Mutation':>8s}  Progress")
    print("  " + "-" * 60)

    for gen in range(1, TOTAL_GENERATIONS + 1):
        # Environment = population centroid
        centroid = [0.0] * DIMENSIONS
        for agent in population:
            for j in range(DIMENSIONS):
                centroid[j] += agent.params[j]
        current_env = [x / len(population) for x in centroid]

        # Evaluate every agent against the custom opponent
        fitnesses = []
        for agent in population:
            f_i = evaluate_vs_opponent(server, agent.params, opp)
            agent.raw_fitness = f_i
            agent.update_history(f_i, current_env)
            agent.compute_dynamic_trajectory(current_env)
            agent.compute_adaptive_mutation()
            fitnesses.append(f_i)

        best_agent = max(population, key=lambda a: a.raw_fitness)
        avg_fit = sum(fitnesses) / len(fitnesses)
        gen_scores.append(best_agent.raw_fitness)

        # Progress bar
        progress = "#" * gen + "." * (TOTAL_GENERATIONS - gen)
        print(f"  {gen:4d}  {best_agent.raw_fitness:8.1%}  {avg_fit:8.1%}  {best_agent.mu_i:8.3f}  [{progress}]")

        # Diversity pruning
        max_dist = math.sqrt(DIMENSIONS)
        survivors = []
        for agent in sorted(population, key=lambda a: a.e_i, reverse=True):
            redundant = any(
                1.0 - (euclidean_dist(agent.params, u.params) / max_dist) > DELTA_S
                for u in survivors
            )
            if not redundant:
                survivors.append(agent)

        final_survivors = [a for a in survivors if a.e_i >= DELTA_F or len(a.H_i) < 3]
        while len(final_survivors) < 5 and len(survivors) > len(final_survivors):
            for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
                if a not in final_survivors:
                    final_survivors.append(a)
                    break

        # Archive top 3
        for top_ag in sorted(population, key=lambda a: a.e_i, reverse=True)[:3]:
            archive.append(TDCCEAAgent(top_ag.params))
        if len(archive) > A_MAX:
            archive = archive[-A_MAX:]

        # Breed next generation
        population = final_survivors.copy()
        while len(population) < N_POP:
            candidates = random.sample(
                final_survivors + archive,
                min(3, len(final_survivors + archive))
            )
            parent = max(candidates, key=lambda a: a.e_i)
            population.append(parent.mutate())

    elapsed = time.time() - t_start
    best = max(population, key=lambda a: a.raw_fitness)

    # ===============================================================
    #  FINAL VALIDATION
    # ===============================================================
    print()
    print("  " + "-" * 60)
    print(f"  FINAL VALIDATION: {VALIDATION_GAMES} games against the opponent")
    print("  " + "-" * 60)

    val_wins = 0
    val_losses = 0
    val_draws = 0

    for game in range(1, VALIDATION_GAMES + 1):
        if game % 2 == 1:
            winner, cycles = server.run_evolved_vs_evolved(best.params, opp)
            if winner == 0:
                val_wins += 1
                r = "WIN"
            elif winner == 1:
                val_losses += 1
                r = "LOSS"
            else:
                val_draws += 1
                r = "DRAW"
        else:
            winner, cycles = server.run_evolved_vs_evolved(opp, best.params)
            if winner == 1:
                val_wins += 1
                r = "WIN"
            elif winner == 0:
                val_losses += 1
                r = "LOSS"
            else:
                val_draws += 1
                r = "DRAW"

        print(f"    Game {game:2d}/{VALIDATION_GAMES}: {r:4s}  |  W:{val_wins} L:{val_losses} D:{val_draws}  |  {cycles} ticks")

    server.stop()

    # ===============================================================
    #  RESULTS
    # ===============================================================
    win_rate = val_wins / VALIDATION_GAMES * 100

    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()
    print(f"  Evolution Time: {elapsed:.1f}s ({TOTAL_GENERATIONS} generations)")
    print()
    print(f"  +- Validation ({VALIDATION_GAMES} games) ----------------------")
    print(f"  |  Wins:   {val_wins:3d}  ({val_wins/VALIDATION_GAMES*100:.0f}%)")
    print(f"  |  Losses: {val_losses:3d}  ({val_losses/VALIDATION_GAMES*100:.0f}%)")
    print(f"  |  Draws:  {val_draws:3d}")
    print(f"  |")
    if win_rate >= 60:
        print(f"  |  TD-CCEA successfully adapted! Win rate: {win_rate:.0f}%")
    elif win_rate >= 40:
        print(f"  |  Close match! Win rate: {win_rate:.0f}%")
    else:
        print(f"  |  Opponent is tough! Win rate: {win_rate:.0f}%")
    print(f"  +------------------------------------------------")
    print()
    print("  Evolved counter-strategy weights:")
    for i, name in enumerate(WEIGHT_NAMES):
        bar_len = int(best.params[i] * 20)
        bar = "#" * bar_len + "." * (20 - bar_len)
        print(f"    w{i:2d} {name:20s} [{bar}] {best.params[i]:.2f}")
    print()


if __name__ == "__main__":
    main()
