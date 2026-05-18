"""
FULL_BENCHMARK.py - Train & Validate all 4 algorithms

Flow per algorithm:
  1. Train: fight 4 generations per opponent in blocks (4 gens vs opp1, 4 gens vs opp2, ...)
  2. Validate: play 5 games vs each of the 7 opponents
  3. Report: per-opponent win rate + overall

Algorithms compared:
  1. TD-CCEA Centroid (standard)
  2. SA-TDCCEA (sensitivity-adaptive mutation)
  3. DP-TDCCEA (fitness cache + DP lookup seeding)
  4. Hill Climbing (greedy baseline)
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
    "LightRush", "WorkerRush",
    "HeavyDefense", "RangedRush", "EconomyRush", "WorkerRushPlusPlus",
]

N_POP = 20
K_GAMES = 3
GENS_PER_OPPONENT = 4    # fight 4 generations against each opponent before switching
TOTAL_GENERATIONS = GENS_PER_OPPONENT * len(OPPONENT_ROTATION)  # 4 x 7 = 28
VALIDATION_GAMES = 5
A_MAX = 15
DELTA_S = 0.85
DELTA_F = 0.20
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
            raise RuntimeError(f"Failed to start: {line}")

    def run_game(self, weights, opponent):
        weights_str = ";".join(f"{w:.4f}" for w in weights)
        request = f"EVOLVED:{weights_str},{opponent},{MAP_PATH},{MAX_CYCLES}\n"
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

def evaluate_weights(server, weights, opponent):
    wins = 0
    for _ in range(K_GAMES):
        winner, _ = server.run_game(weights, opponent)
        if winner == 0:
            wins += 1
        elif winner == -1:
            wins += 0.3
    return wins / K_GAMES

def validate_weights(server, weights):
    total_wins = 0
    total_games = 0
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
    overall = total_wins / total_games * 100
    return overall, per_opp

def load_sensitivity():
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), SENSITIVITY_FILE)
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        return data["normalized_mutation_rates"]
    return None


# ===============================================================
#  AGENT
# ===============================================================
class Agent:
    def __init__(self, params=None, per_weight_mut=None):
        if params is None:
            self.params = [random.random() for _ in range(DIMENSIONS)]
        else:
            self.params = [max(0.0, min(1.0, float(x))) for x in params]
        self.H_i = []
        self.e_i = 0.0
        self.mu_i = 0.0
        self.per_weight_mut = per_weight_mut

    def update_history(self, fitness, env):
        self.H_i.append((fitness, env))
        if len(self.H_i) > 7:
            self.H_i.pop(0)

    def compute_trajectory(self, current_env):
        if not self.H_i:
            self.e_i = 0.0
            return self.e_i
        max_dist = math.sqrt(DIMENSIONS)
        ws = []
        for past_f, past_e in self.H_i:
            dist = euclidean_dist(current_env, past_e)
            ws.append(1.0 - (dist / max_dist))
        total = sum(ws)
        if total == 0:
            ws = [1.0 / len(ws)] * len(ws)
        else:
            ws = [w / total for w in ws]
        self.e_i = sum(w * self.H_i[i][0] for i, w in enumerate(ws))
        return self.e_i

    def compute_mutation(self):
        self.mu_i = 0.15 * (1.0 - self.e_i) + 0.03
        return self.mu_i

    def mutate(self):
        child_p = []
        if self.per_weight_mut:
            for j, p in enumerate(self.params):
                sigma = self.per_weight_mut[j] * (1.0 - self.e_i) + 0.02
                child_p.append(max(0.0, min(1.0, p + random.gauss(0, sigma))))
        else:
            for p in self.params:
                child_p.append(max(0.0, min(1.0, p + random.gauss(0, self.mu_i))))
        child = Agent(child_p, self.per_weight_mut)
        return child


# ===============================================================
#  SHARED EVOLUTION CORE
# ===============================================================
def evolve_core(server, label, per_weight_mut=None, use_dp=False):
    print(f"\n{'='*60}")
    print(f"  RUNNING: {label}")
    print(f"{'='*60}")

    population = [Agent(per_weight_mut=per_weight_mut) for _ in range(N_POP)]
    archive = []
    gen_scores = []
    dp_table = {}        # opponent → {'weights': [...], 'fitness': float}
    fitness_cache = {}   # (rounded_weights, opp) → fitness
    cache_hits = 0
    cache_misses = 0
    t_start = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        # Block scheduling: 4 gens per opponent
        opp_idx = (gen - 1) // GENS_PER_OPPONENT
        opponent = OPPONENT_ROTATION[opp_idx % len(OPPONENT_ROTATION)]

        # DP: seed from lookup table after first full rotation
        if use_dp and opponent in dp_table and opp_idx >= len(OPPONENT_ROTATION):
            dp_weights = dp_table[opponent]['weights']
            pop_sorted = sorted(range(len(population)), key=lambda i: population[i].e_i)
            for idx in pop_sorted[:3]:
                seeded = Agent(dp_weights, per_weight_mut)
                seeded.params = [max(0.0, min(1.0, p + random.gauss(0, 0.05)))
                                 for p in seeded.params]
                population[idx] = seeded

        # Centroid
        centroid = [sum(a.params[j] for a in population) / len(population)
                    for j in range(DIMENSIONS)]

        # Evaluate
        for agent in population:
            if use_dp:
                key = (tuple(round(w, 2) for w in agent.params), opponent)
                if key in fitness_cache:
                    f_i = fitness_cache[key]
                    cache_hits += 1
                else:
                    f_i = evaluate_weights(server, agent.params, opponent)
                    fitness_cache[key] = f_i
                    cache_misses += 1
            else:
                f_i = evaluate_weights(server, agent.params, opponent)

            agent.update_history(f_i, centroid)
            agent.compute_trajectory(centroid)
            agent.compute_mutation()

        best = max(population, key=lambda a: a.e_i)
        gen_scores.append(best.e_i)

        # DP: update lookup table
        if use_dp:
            best_fit = evaluate_weights(server, best.params, opponent)
            if opponent not in dp_table or best_fit > dp_table[opponent]['fitness']:
                dp_table[opponent] = {'weights': best.params[:], 'fitness': best_fit}

        tag = ""
        if use_dp and opponent in dp_table and opp_idx >= len(OPPONENT_ROTATION):
            tag = " [SEEDED]"
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opponent:25s} | e_i={best.e_i:.3f}{tag}")

        # Redundancy removal
        max_dist = math.sqrt(DIMENSIONS)
        survivors = []
        for agent in sorted(population, key=lambda a: a.e_i, reverse=True):
            redundant = any(
                1.0 - (euclidean_dist(agent.params, u.params) / max_dist) > DELTA_S
                for u in survivors
            )
            if not redundant:
                survivors.append(agent)

        # Fitness pruning
        final = [a for a in survivors if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
        while len(final) < 10 and len(survivors) > len(final):
            for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
                if a not in final:
                    final.append(a)
                    break

        # Archive (smart for DP: keep context)
        for top in sorted(population, key=lambda a: a.e_i, reverse=True)[:3]:
            archived = Agent(top.params, per_weight_mut)
            if use_dp:
                archived.e_i = top.e_i
                archived.H_i = top.H_i[:]
                archived.mu_i = top.mu_i
            archive.append(archived)
        archive.sort(key=lambda a: a.e_i, reverse=True)
        archive = archive[:A_MAX]

        # Breed
        pool = final + archive
        population = final[:]
        while len(population) < N_POP:
            candidates = random.sample(pool, min(3, len(pool)))
            parent = max(candidates, key=lambda a: a.e_i)
            population.append(parent.mutate())

    elapsed = time.time() - t_start
    best_overall = max(archive, key=lambda a: a.e_i) if archive else max(population, key=lambda a: a.e_i)

    if use_dp:
        total_evals = cache_hits + cache_misses
        print(f"\n  DP Cache: {cache_hits}/{total_evals} hits ({cache_hits/max(1,total_evals)*100:.0f}% saved)")

    return best_overall.params, elapsed, gen_scores


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
        opp_idx = (gen - 1) // GENS_PER_OPPONENT
        opponent = OPPONENT_ROTATION[opp_idx % len(OPPONENT_ROTATION)]
        current_fit = evaluate_weights(server, current, opponent)
        best_cand, best_fit = current[:], current_fit

        for _ in range(N_POP - 1):
            cand = [max(0.0, min(1.0, p + random.gauss(0, 0.15))) for p in current]
            fit = evaluate_weights(server, cand, opponent)
            if fit > best_fit:
                best_cand, best_fit = cand[:], fit

        if best_fit >= current_fit:
            current, current_fit = best_cand[:], best_fit

        gen_scores.append(current_fit)
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opponent:25s} | Best={current_fit:.3f}")

    elapsed = time.time() - t_start
    return current, elapsed, gen_scores


# ===============================================================
#  MAIN
# ===============================================================
def main():
    print()
    print("#" * 70)
    print("  FULL BENCHMARK: TD-CCEA vs SA-TDCCEA vs DP-TDCCEA vs Hill Climbing")
    print("#" * 70)
    print(f"  Population: {N_POP} | Generations: {TOTAL_GENERATIONS} | Opponents: {len(OPPONENT_ROTATION)}")
    print(f"  K_GAMES: {K_GAMES} | Validation: {VALIDATION_GAMES} games/opponent")
    print()

    # Load sensitivity data for SA version
    sa_mut = load_sensitivity()
    if sa_mut is None:
        print("  [!] No sensitivity_scores.json found. SA-TDCCEA will use standard mutation.")

    algorithms = [
        ("TD-CCEA Centroid",  "centroid"),
        ("SA-TDCCEA",         "sa"),
        ("DP-TDCCEA",         "dp"),
        ("Hill Climbing",     "hc"),
    ]

    all_results = []

    for algo_name, algo_key in algorithms:
        server = GameServer()
        server.start()
        print(f"  [OK] Server started for {algo_name}")

        if algo_key == "centroid":
            best, elapsed, scores = evolve_core(server, algo_name)
        elif algo_key == "sa":
            best, elapsed, scores = evolve_core(server, algo_name, per_weight_mut=sa_mut)
        elif algo_key == "dp":
            best, elapsed, scores = evolve_core(server, algo_name, use_dp=True)
        else:
            best, elapsed, scores = run_hill_climbing(server)

        print(f"\n  Validating {algo_name}...")
        overall_wr, per_opp = validate_weights(server, best)
        server.stop()

        # Show individual results
        print(f"\n  +-- {algo_name} RESULTS --------------------------------")
        print(f"  |  Training Time   : {elapsed:.1f}s")
        print(f"  |  Overall Win Rate: {overall_wr:.1f}%")
        for opp, wr in per_opp.items():
            status = "[W]" if wr >= 50 else "[L]"
            print(f"  |  {status} vs {opp:<26s}: {wr:5.0f}%")
        print(f"  +-------------------------------------------------------")

        all_results.append({
            "name": algo_name, "time": elapsed,
            "overall_winrate": overall_wr,
            "per_opponent": per_opp, "gen_scores": scores,
        })

    # ── FINAL COMPARISON TABLE ──
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"  {'Algorithm':<22s}  {'Win Rate':>10s}  {'Time':>8s}")
    print(f"  {'_'*22}  {'_'*10}  {'_'*8}")
    for r in sorted(all_results, key=lambda x: x["overall_winrate"], reverse=True):
        print(f"  {r['name']:<22s}  {r['overall_winrate']:>9.1f}%  {r['time']:>7.1f}s")

    # ── PER-OPPONENT TABLE ──
    print(f"\n{'='*70}")
    print(f"  WIN RATE PER OPPONENT")
    print(f"{'='*70}")
    header = f"  {'Opponent':<26s}"
    for r in all_results:
        header += f"  {r['name']:>14s}"
    print(header)
    print("  " + "-" * (26 + 16 * len(all_results)))

    for opp in OPPONENT_ROTATION:
        row = f"  {opp:<26s}"
        rates = [r["per_opponent"].get(opp, 0) for r in all_results]
        best_rate = max(rates)
        for r in all_results:
            wr = r["per_opponent"].get(opp, 0)
            marker = " *" if wr == best_rate and rates.count(best_rate) == 1 else "  "
            row += f"  {wr:>11.0f}%{marker}"
        print(row)

    print("  " + "-" * (26 + 16 * len(all_results)))
    row = f"  {'OVERALL':<26s}"
    for r in all_results:
        row += f"  {r['overall_winrate']:>12.1f}%  "
    print(row)

    # ── RANKING ──
    ranked = sorted(all_results, key=lambda x: x["overall_winrate"], reverse=True)
    medals = ["1st", "2nd", "3rd", "4th"]
    print(f"\n{'='*60}")
    print(f"  RANKING")
    print(f"{'='*60}")
    for i, r in enumerate(ranked):
        print(f"  {medals[i]}  {r['name']:<22s}  {r['overall_winrate']:5.1f}%  ({r['time']:.0f}s)")

    for i in range(1, len(ranked)):
        d = ranked[0]["overall_winrate"] - ranked[i]["overall_winrate"]
        print(f"\n  {ranked[0]['name']} beats {ranked[i]['name']} by {abs(d):.1f}%")

    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
