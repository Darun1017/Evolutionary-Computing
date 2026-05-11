"""
head_to_head.py - TD-CCEA vs Hill Climbing: Coevolutionary Battle

Both algorithms evolve by FIGHTING EACH OTHER every generation.
No predefined bots -- each side's fitness comes from beating the other side.
"""

import math, random, time, subprocess, os, sys

# Fix Windows terminal encoding
os.environ["PYTHONIOENCODING"] = "utf-8"

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

N_POP = 15
K_GAMES = 3           # games per evaluation (alternating sides)
TOTAL_GENERATIONS = 30


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


def evaluate_vs_opponent(server, weights, opponent_weights):
    """Play K_GAMES against a specific opponent (alternating sides for fairness).
    Returns win rate from 0.0 to 1.0."""
    wins = 0
    for game in range(K_GAMES):
        if game % 2 == 0:
            # We are Player 0
            winner, _ = server.run_evolved_vs_evolved(weights, opponent_weights)
            if winner == 0:
                wins += 1
            elif winner == -1:
                wins += 0.3
        else:
            # We are Player 1 (swap sides)
            winner, _ = server.run_evolved_vs_evolved(opponent_weights, weights)
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
#  MAIN: COEVOLUTIONARY BATTLE
# ===============================================================
def main():
    print()
    print("=" * 70)
    print("  [VS] COEVOLUTIONARY BATTLE: TD-CCEA vs Hill Climbing")
    print("=" * 70)
    print(f"  Each generation, both sides evolve by FIGHTING EACH OTHER")
    print(f"  No predefined bots -- pure head-to-head evolution")
    print(f"  Generations: {TOTAL_GENERATIONS} | Population: {N_POP}")
    print(f"  Games per evaluation: {K_GAMES} (alternating sides)")
    print(f"  Map: {MAP_PATH}")
    print()

    server = GameServer()
    server.start()
    print("  [OK] Game server started")

    # ---- Initialize both sides ----
    # TD-CCEA: population of agents
    tdccea_pop = [TDCCEAAgent() for _ in range(N_POP)]
    tdccea_archive = []
    DELTA_S, DELTA_F, A_MAX = 0.85, 0.20, 15

    # Hill Climbing: single solution + neighbors
    hc_current = [random.random() for _ in range(DIMENSIONS)]
    hc_current_fit = 0.0
    HC_MUT = 0.15

    # Scoreboard
    tdccea_gen_wins = 0
    hc_gen_wins = 0
    gen_ties = 0
    tdccea_best_scores = []
    hc_best_scores = []

    print()
    print("  " + "-" * 66)
    print(f"  {'Gen':>4s}  {'TD-CCEA Best':>12s}  {'HC Best':>10s}  {'Leader':>15s}  {'Score'}")
    print("  " + "-" * 66)

    t_start = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):

        # ===========================================================
        #  STEP 1: TD-CCEA evaluates against HC's current best
        # ===========================================================
        centroid = [0.0] * DIMENSIONS
        for agent in tdccea_pop:
            for j in range(DIMENSIONS):
                centroid[j] += agent.params[j]
        current_env = [x / len(tdccea_pop) for x in centroid]

        for agent in tdccea_pop:
            f_i = evaluate_vs_opponent(server, agent.params, hc_current)
            agent.raw_fitness = f_i
            agent.update_history(f_i, current_env)
            agent.compute_dynamic_trajectory(current_env)
            agent.compute_adaptive_mutation()

        tdccea_best = max(tdccea_pop, key=lambda a: a.e_i)
        tdccea_best_score = tdccea_best.raw_fitness

        # ===========================================================
        #  STEP 2: Hill Climbing evaluates against TD-CCEA's best
        # ===========================================================
        hc_current_fit = evaluate_vs_opponent(server, hc_current, tdccea_best.params)
        hc_best_candidate = hc_current[:]
        hc_best_fit = hc_current_fit

        for _ in range(N_POP - 1):
            candidate = [max(0.0, min(1.0, p + random.gauss(0, HC_MUT))) for p in hc_current]
            fit = evaluate_vs_opponent(server, candidate, tdccea_best.params)
            if fit > hc_best_fit:
                hc_best_candidate = candidate[:]
                hc_best_fit = fit

        if hc_best_fit >= hc_current_fit:
            hc_current = hc_best_candidate[:]
            hc_current_fit = hc_best_fit

        # ===========================================================
        #  STEP 3: TD-CCEA survival + breeding (same as original)
        # ===========================================================
        max_dist = math.sqrt(DIMENSIONS) * 1.0
        survivors = []
        for agent in sorted(tdccea_pop, key=lambda a: a.e_i, reverse=True):
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

        for top_ag in sorted(tdccea_pop, key=lambda a: a.e_i, reverse=True)[:3]:
            tdccea_archive.append(TDCCEAAgent(top_ag.params))
        if len(tdccea_archive) > A_MAX:
            tdccea_archive = tdccea_archive[-A_MAX:]

        tdccea_pop = final_survivors.copy()
        while len(tdccea_pop) < N_POP:
            candidates = random.sample(
                final_survivors + tdccea_archive,
                min(3, len(final_survivors + tdccea_archive))
            )
            parent = max(candidates, key=lambda a: a.e_i)
            tdccea_pop.append(parent.mutate())

        # ===========================================================
        #  STEP 4: Scoreboard
        # ===========================================================
        tdccea_best_scores.append(tdccea_best_score)
        hc_best_scores.append(hc_best_fit)

        if tdccea_best_score > hc_best_fit:
            leader = "<< TD-CCEA"
            tdccea_gen_wins += 1
        elif hc_best_fit > tdccea_best_score:
            leader = "HillClimb >>"
            hc_gen_wins += 1
        else:
            leader = "TIE"
            gen_ties += 1

        score_str = f"{tdccea_gen_wins}-{hc_gen_wins}"
        print(f"  {gen:4d}  {tdccea_best_score:12.3f}  {hc_best_fit:10.3f}  {leader:>15s}  {score_str}")

    elapsed = time.time() - t_start

    # ===============================================================
    #  FINAL SHOWDOWN: 20 games between final evolved strategies
    # ===============================================================
    print()
    print("  " + "-" * 66)
    print("  FINAL SHOWDOWN: 20 games between the final evolved strategies")
    print("  " + "-" * 66)

    final_tdccea = max(tdccea_pop, key=lambda a: a.e_i)
    final_hc = hc_current

    showdown_tdccea = 0
    showdown_hc = 0
    showdown_draws = 0

    for game in range(1, 21):
        if game % 2 == 1:
            winner, cycles = server.run_evolved_vs_evolved(final_tdccea.params, final_hc)
            if winner == 0:
                showdown_tdccea += 1
                result = "TD-CCEA wins"
            elif winner == 1:
                showdown_hc += 1
                result = "HillClimb wins"
            else:
                showdown_draws += 1
                result = "DRAW"
        else:
            winner, cycles = server.run_evolved_vs_evolved(final_hc, final_tdccea.params)
            if winner == 0:
                showdown_hc += 1
                result = "HillClimb wins"
            elif winner == 1:
                showdown_tdccea += 1
                result = "TD-CCEA wins"
            else:
                showdown_draws += 1
                result = "DRAW"

        print(f"  Game {game:2d}/20: {result:15s} | TD-CCEA {showdown_tdccea} - {showdown_hc} HillClimb | {cycles} ticks")

    server.stop()

    # ===============================================================
    #  FINAL RESULTS
    # ===============================================================
    print()
    print("=" * 70)
    print("  [VS] FINAL RESULTS")
    print("=" * 70)
    print()
    print(f"  Total Time: {elapsed:.1f}s")
    print()
    print(f"  +- Evolution Race ({TOTAL_GENERATIONS} generations) ----------------")
    print(f"  |  TD-CCEA won:      {tdccea_gen_wins} generations")
    print(f"  |  Hill Climbing won: {hc_gen_wins} generations")
    print(f"  |  Ties:              {gen_ties}")
    print(f"  +----------------------------------------------------")
    print()
    print(f"  +- Final Showdown (20 games) -----------------------")
    print(f"  |  TD-CCEA:      {showdown_tdccea} wins ({showdown_tdccea/20*100:.0f}%)")
    print(f"  |  Hill Climbing: {showdown_hc} wins ({showdown_hc/20*100:.0f}%)")
    print(f"  |  Draws:         {showdown_draws}")
    print(f"  |")

    if showdown_tdccea > showdown_hc:
        print(f"  |  >>> WINNER: TD-CCEA!")
    elif showdown_hc > showdown_tdccea:
        print(f"  |  >>> WINNER: Hill Climbing!")
    else:
        print(f"  |  >>> IT'S A TIE!")
    print(f"  +----------------------------------------------------")

    # Visual replay
    print()
    print("  Launching VISUAL replay of the final battle...")
    time.sleep(1)
    w1_str = ";".join(f"{w:.4f}" for w in final_tdccea.params)
    w2_str = ";".join(f"{w:.4f}" for w in final_hc)
    cmd = [
        "java", "-cp", "lib/*;bin",
        "tests.EvolvedVisualGameRunner",
        f"EVOLVED:{w1_str}", f"EVOLVED:{w2_str}", MAP_PATH
    ]
    subprocess.run(cmd, cwd=MICRORTS_DIR)
    print("  Visual replay closed.")


if __name__ == "__main__":
    main()
