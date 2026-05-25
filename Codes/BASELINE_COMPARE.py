"""
BASELINE_COMPARE.py - Compare 4 baseline algorithms (no co-evolution)

Algorithms:
  1. TD-CCEA Centroid (standard monolithic base algorithm)
  2. SA-TDCCEA (sensitivity-adaptive per-weight mutation)
  3. DP-TDCCEA (fitness cache + DP seeding)
  4. Hill Climbing (greedy baseline)

Per-generation opponent rotation: gen1=opp1, gen2=opp2, ...
Validation: 5 games per opponent at the end.
"""

import math, random, time, subprocess, os, json

os.environ["PYTHONIOENCODING"] = "utf-8"

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

OPPONENT_ROTATION = [
    "EconomyMilitaryRush", "LightRush", "WorkerRush",
    "HeavyDefense", "RangedRush", "EconomyRush", "WorkerRushPlusPlus",
]

# Hyperparameters (matching original tdccea_evolved_v3.py)
N_POP = 20
K_GAMES = 3
H_WINDOW = 7
MU_BASE = 0.15
DELTA_F = 0.20
DELTA_S = 0.85
A_MAX = 15
TOTAL_GENERATIONS = 30
VALIDATION_GAMES = 5
SENSITIVITY_FILE = "sensitivity_scores.json"


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
            raise RuntimeError(f"Server failed: {line}")

    def run_game(self, weights, opponent):
        weights_str = ";".join(f"{w:.4f}" for w in weights)
        self.process.stdin.write(f"EVOLVED:{weights_str},{opponent},{MAP_PATH},{MAX_CYCLES}\n")
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
            except:
                self.process.kill()


# ===============================================================
#  UTILITY FUNCTIONS
# ===============================================================
def euclidean_dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def get_population_centroid(population):
    centroid = [0.0] * DIMENSIONS
    for agent in population:
        for j in range(DIMENSIONS):
            centroid[j] += agent.params[j]
    return [x / len(population) for x in centroid]


def evaluate_weights(server, weights, opponent):
    """Evaluate a weight vector: play K_GAMES, return win ratio."""
    wins = 0
    for _ in range(K_GAMES):
        winner, _ = server.run_game(weights, opponent)
        if winner == 0:
            wins += 1
        elif winner == -1:
            wins += 0.3
    return wins / K_GAMES


def validate_weights(server, weights):
    """Test final weights against ALL opponents."""
    total_wins, total_games = 0, 0
    per_opp = {}
    for opp in OPPONENT_ROTATION:
        opp_wins = 0
        for _ in range(VALIDATION_GAMES):
            winner, _ = server.run_game(weights, opp)
            if winner == 0:
                opp_wins += 1
            total_games += 1
        total_wins += opp_wins
        per_opp[opp] = opp_wins / VALIDATION_GAMES * 100
    return total_wins / total_games * 100, per_opp


def load_sensitivity():
    fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), SENSITIVITY_FILE)
    if os.path.exists(fp):
        with open(fp) as f:
            return json.load(f)["normalized_mutation_rates"]
    return None


# ===============================================================
#  TD-CCEA AGENT (used by TD-CCEA, SA-TDCCEA, DP-TDCCEA)
# ===============================================================
class Agent:
    def __init__(self, params=None, pwm=None):
        if params is None:
            self.params = [random.random() for _ in range(DIMENSIONS)]
        else:
            self.params = [max(0.0, min(1.0, float(x))) for x in params]
        self.H_i = []       # trajectory memory: list of (fitness, environment)
        self.e_i = 0.0      # trajectory score
        self.mu_i = 0.0     # adaptive mutation rate
        self.pwm = pwm      # per-weight mutation rates (for SA)

    def update_history(self, fitness, current_environment):
        """Step 5: H_i <- H_i U {(f_i, c_t)}; trim if |H_i| > W"""
        self.H_i.append((fitness, current_environment))
        if len(self.H_i) > H_WINDOW:
            self.H_i.pop(0)

    def compute_trajectory(self, current_environment):
        """
        Steps 6-8 of TD-CCEA:
          6. alpha_k = 1 - d(c_t, c_k) / d_max
          7. alpha_hat_k = alpha_k / sum(alpha_j)
          8. e_i = sum(alpha_hat_k * f_i^(k))
        """
        if len(self.H_i) == 0:
            self.e_i = 0.0
            return

        max_dist = math.sqrt(DIMENSIONS)

        # Step 6: Compute dynamic similarity weights
        dynamic_weights = []
        for past_fitness, past_env in self.H_i:
            dist = euclidean_dist(current_environment, past_env)
            similarity = 1.0 - (dist / max_dist)
            dynamic_weights.append(similarity)

        # Step 7: Normalize weights
        total_w = sum(dynamic_weights)
        if total_w == 0:
            dynamic_weights = [1.0 / len(dynamic_weights)] * len(dynamic_weights)
        else:
            dynamic_weights = [w / total_w for w in dynamic_weights]

        # Step 8: Compute trajectory score
        self.e_i = 0.0
        for i in range(len(self.H_i)):
            self.e_i += dynamic_weights[i] * self.H_i[i][0]

    def compute_mutation(self):
        """Step 9: mu_i = mu_base * (1 - e_i) + epsilon"""
        self.mu_i = MU_BASE * (1.0 - self.e_i) + 0.03

    def mutate(self):
        """Step 13: Breed using Gaussian mutation N(0, mu_i)"""
        offspring_params = []
        if self.pwm:
            # SA-TDCCEA: per-weight mutation
            for j, p in enumerate(self.params):
                noise = random.gauss(0, self.pwm[j] * (1.0 - self.e_i) + 0.02)
                offspring_params.append(max(0.0, min(1.0, p + noise)))
        else:
            # Standard: uniform mutation
            for p in self.params:
                noise = random.gauss(0, self.mu_i)
                offspring_params.append(max(0.0, min(1.0, p + noise)))
        return Agent(offspring_params, self.pwm)


# ===============================================================
#  TD-CCEA EVOLUTION CORE
# ===============================================================
def evolve_tdccea(server, label, pwm=None, use_dp=False):
    """
    Full TD-CCEA loop matching the pseudocode exactly.
    - pwm: if provided, uses SA per-weight mutation
    - use_dp: if True, uses DP fitness cache + seeding
    """
    print(f"\n{'='*60}")
    print(f"  RUNNING: {label}")
    print(f"{'='*60}")

    # Step 1: Initialize population P_0, archive A <- empty
    population = [Agent(pwm=pwm) for _ in range(N_POP)]
    archive = []
    gen_scores = []
    dp_table = {}       # opponent -> {'weights': [...], 'fitness': f}
    fitness_cache = {}  # (rounded_weights, opponent) -> fitness
    t_start = time.time()

    # Step 2: for t = 1, 2, ..., T_max
    for gen in range(1, TOTAL_GENERATIONS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]

        # DP seeding: inject cached best weights if opponent seen before
        if use_dp and opponent in dp_table and gen > len(OPPONENT_ROTATION):
            dp_weights = dp_table[opponent]['weights']
            worst_indices = sorted(range(len(population)), key=lambda i: population[i].e_i)
            for idx in worst_indices[:3]:
                seeded = Agent(dp_weights, pwm)
                seeded.params = [max(0.0, min(1.0, p + random.gauss(0, 0.05))) for p in seeded.params]
                population[idx] = seeded

        # Step 3: Compute environment meta-state c_t = centroid
        current_env = get_population_centroid(population)

        # Step 4: Evaluate fitness f_i for all O_i via K simulated games
        for agent in population:
            if use_dp:
                cache_key = (tuple(round(w, 2) for w in agent.params), opponent)
                f_i = fitness_cache.get(cache_key)
                if f_i is None:
                    f_i = evaluate_weights(server, agent.params, opponent)
                    fitness_cache[cache_key] = f_i
            else:
                f_i = evaluate_weights(server, agent.params, opponent)

            # Step 5: Update trajectory memory
            agent.update_history(f_i, current_env)
            # Steps 6-8: Compute trajectory score
            agent.compute_trajectory(current_env)
            # Step 9: Compute adaptive mutation rate
            agent.compute_mutation()

        best = max(population, key=lambda a: a.e_i)
        gen_scores.append(best.e_i)

        # DP: update dp_table with best weights for this opponent
        if use_dp:
            best_fitness = evaluate_weights(server, best.params, opponent)
            if opponent not in dp_table or best_fitness > dp_table[opponent]['fitness']:
                dp_table[opponent] = {'weights': best.params[:], 'fitness': best_fitness}

        tag = " [SEEDED]" if use_dp and opponent in dp_table and gen > len(OPPONENT_ROTATION) else ""
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opponent:25s} | e_i={best.e_i:.3f}{tag}")

        # Step 11: Remove redundant O_j if S(i,j) > delta_s
        max_dist = math.sqrt(DIMENSIONS)
        survivors = []
        for agent in sorted(population, key=lambda a: a.e_i, reverse=True):
            is_redundant = False
            for u in survivors:
                dist = euclidean_dist(agent.params, u.params)
                if 1.0 - (dist / max_dist) > DELTA_S:
                    is_redundant = True
                    break
            if not is_redundant:
                survivors.append(agent)

        # Step 10: Remove O_i if e_i < delta_f and |H_i| >= 3
        final_survivors = []
        for agent in survivors:
            if agent.e_i < DELTA_F and len(agent.H_i) >= 3:
                pass  # pruned
            else:
                final_survivors.append(agent)

        while len(final_survivors) < 10 and len(survivors) > len(final_survivors):
            for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
                if a not in final_survivors:
                    final_survivors.append(a)
                    break

        # Step 12: Insert top-m survivors into A; trim if |A| > A_max
        sorted_gen = sorted(population, key=lambda a: a.e_i, reverse=True)
        for top_ag in sorted_gen[:3]:
            ar = Agent(top_ag.params, pwm)
            ar.e_i = top_ag.e_i
            ar.H_i = top_ag.H_i[:]
            ar.mu_i = top_ag.mu_i
            archive.append(ar)
        archive.sort(key=lambda a: a.e_i, reverse=True)
        archive = archive[:A_MAX]

        # Step 13: Breed offspring from survivors U A using Gaussian mutation
        breeding_pool = final_survivors + archive
        population = final_survivors[:]
        while len(population) < N_POP:
            candidates = random.sample(breeding_pool, min(3, len(breeding_pool)))
            parent = max(candidates, key=lambda a: a.e_i)
            population.append(parent.mutate())

    # Step 15: return best O* in A
    best_overall = max(archive, key=lambda a: a.e_i) if archive else max(population, key=lambda a: a.e_i)
    return best_overall.params, time.time() - t_start, gen_scores


# ===============================================================
#  HILL CLIMBING
# ===============================================================
def run_hill_climbing(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: Hill Climbing")
    print(f"{'='*60}")

    current = [random.random() for _ in range(DIMENSIONS)]
    current_fit = 0.0
    gen_scores = []
    t_start = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
        current_fit = evaluate_weights(server, current, opponent)
        best_candidate, best_fit = current[:], current_fit

        for _ in range(N_POP - 1):
            candidate = [max(0.0, min(1.0, p + random.gauss(0, 0.15))) for p in current]
            fit = evaluate_weights(server, candidate, opponent)
            if fit > best_fit:
                best_candidate, best_fit = candidate[:], fit

        if best_fit >= current_fit:
            current, current_fit = best_candidate[:], best_fit

        gen_scores.append(current_fit)
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opponent:25s} | Best={current_fit:.3f}")

    return current, time.time() - t_start, gen_scores


# ===============================================================
#  MAIN
# ===============================================================
def main():
    print()
    print("#" * 70)
    print("  BASELINE COMPARE: TD-CCEA vs SA vs DP vs Hill Climbing")
    print("#" * 70)
    print(f"  Generations: {TOTAL_GENERATIONS} | Opponents: {len(OPPONENT_ROTATION)} | Pop: {N_POP}")
    print(f"  Games/eval: {K_GAMES} | History window: {H_WINDOW}")
    print()

    sa_mut = load_sensitivity()

    algos = [
        ("TD-CCEA Centroid",  "centroid"),
        ("SA-TDCCEA",         "sa"),
        ("DP-TDCCEA",         "dp"),
        ("Hill Climbing",     "hc"),
    ]

    all_results = []
    for name, key in algos:
        server = GameServer()
        server.start()
        print(f"  [OK] Server started for {name}")

        if key == "centroid":
            best, elapsed, scores = evolve_tdccea(server, name)
        elif key == "sa":
            best, elapsed, scores = evolve_tdccea(server, name, pwm=sa_mut)
        elif key == "dp":
            best, elapsed, scores = evolve_tdccea(server, name, use_dp=True)
        else:
            best, elapsed, scores = run_hill_climbing(server)

        print(f"\n  Validating {name}...")
        wr, per_opp = validate_weights(server, best)
        server.stop()

        print(f"\n  +-- {name} RESULTS --------------------------------")
        print(f"  |  Time: {elapsed:.1f}s | Win Rate: {wr:.1f}%")
        for opp, w in per_opp.items():
            print(f"  |  {'[W]' if w >= 50 else '[L]'} vs {opp:<26s}: {w:5.0f}%")
        print(f"  +-------------------------------------------------------")
        all_results.append({
            "name": name, "time": elapsed, "overall_winrate": wr,
            "per_opponent": per_opp, "gen_scores": scores
        })

    # ── FINAL TABLE ──
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"  {'Algorithm':<22s}  {'Win Rate':>10s}  {'Time':>8s}")
    print(f"  {'_'*22}  {'_'*10}  {'_'*8}")
    for r in sorted(all_results, key=lambda x: x["overall_winrate"], reverse=True):
        print(f"  {r['name']:<22s}  {r['overall_winrate']:>9.1f}%  {r['time']:>7.1f}s")

    # ── PER-OPPONENT ──
    print(f"\n{'='*70}")
    print(f"  WIN RATE PER OPPONENT")
    print(f"{'='*70}")
    header = f"  {'Opponent':<26s}"
    for r in all_results:
        header += f" {r['name']:>13s}"
    print(header)
    print("  " + "-" * (26 + 14 * len(all_results)))
    for opp in OPPONENT_ROTATION:
        row = f"  {opp:<26s}"
        rates = [r["per_opponent"].get(opp, 0) for r in all_results]
        mx = max(rates)
        for r in all_results:
            w = r["per_opponent"].get(opp, 0)
            m = " *" if w == mx and rates.count(mx) == 1 else "  "
            row += f" {w:>10.0f}%{m}"
        print(row)
    print("  " + "-" * (26 + 14 * len(all_results)))
    row = f"  {'OVERALL':<26s}"
    for r in all_results:
        row += f" {r['overall_winrate']:>11.1f}%  "
    print(row)

    # ── RANKING ──
    ranked = sorted(all_results, key=lambda x: x["overall_winrate"], reverse=True)
    medals = ["1st", "2nd", "3rd", "4th"]
    print(f"\n{'='*60}")
    print(f"  RANKING")
    print(f"{'='*60}")
    for i, r in enumerate(ranked):
        print(f"  {medals[i]}  {r['name']:<22s}  {r['overall_winrate']:5.1f}%  ({r['time']:.0f}s)")

    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
