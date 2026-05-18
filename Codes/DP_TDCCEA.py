"""
DP_TDCCEA.py - Dynamic Programming Enhanced TD-CCEA

3 DP enhancements over standard TD-CCEA:
  1. Fitness Cache    - skip games if similar weights already evaluated
  2. Smart Archive    - archive agents keep e_i/H_i context
  3. Seeded Breeding  - DP lookup table seeds agents from past winning weights
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

N_POP = 20
K_GAMES = 3
TOTAL_GENERATIONS = 30
VALIDATION_GAMES = 5
A_MAX = 15
DELTA_S = 0.85
DELTA_F = 0.20

# DP settings
CACHE_PRECISION = 2       # round weights to N decimals for cache key
DP_SEED_COUNT = 3         # how many agents to seed from DP table
DP_SIMILARITY_THRESH = 0.8  # min similarity to reuse DP entry


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


def make_cache_key(weights, opponent):
    rounded = tuple(round(w, CACHE_PRECISION) for w in weights)
    return (rounded, opponent)


# ===============================================================
#  DP LOOKUP TABLE
# ===============================================================
class DPTable:
    """Maps opponent → best known weights + fitness."""

    def __init__(self):
        self.table = {}  # opponent_name → {'weights': [...], 'fitness': float}

    def store(self, opponent, weights, fitness):
        if opponent not in self.table or fitness > self.table[opponent]['fitness']:
            self.table[opponent] = {
                'weights': weights[:],
                'fitness': fitness,
            }

    def lookup(self, opponent):
        return self.table.get(opponent, None)

    def __len__(self):
        return len(self.table)


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

    def update_history(self, fitness, env):
        self.H_i.append((fitness, env))
        if len(self.H_i) > 7:
            self.H_i.pop(0)

    def compute_dynamic_trajectory(self, current_env):
        if not self.H_i:
            self.e_i = 0.0
            return self.e_i
        max_dist = math.sqrt(DIMENSIONS)
        weights = []
        for past_f, past_e in self.H_i:
            dist = euclidean_dist(current_env, past_e)
            sim = 1.0 - (dist / max_dist)
            weights.append(sim)
        total = sum(weights)
        if total == 0:
            weights = [1.0 / len(weights)] * len(weights)
        else:
            weights = [w / total for w in weights]
        self.e_i = sum(w * self.H_i[i][0] for i, w in enumerate(weights))
        return self.e_i

    def compute_adaptive_mutation(self):
        self.mu_i = 0.15 * (1.0 - self.e_i) + 0.03
        return self.mu_i

    def mutate(self):
        child_p = [max(0.0, min(1.0, p + random.gauss(0, self.mu_i))) for p in self.params]
        child = TDCCEAAgent(child_p)
        return child


# ===============================================================
#  DP-ENHANCED TD-CCEA
# ===============================================================
def run_dp_tdccea(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: DP-TDCCEA (Dynamic Programming Enhanced)")
    print(f"{'='*60}")

    population = [TDCCEAAgent() for _ in range(N_POP)]
    archive = []
    gen_scores = []
    fitness_cache = {}   # DP Change 1: cache
    dp_table = DPTable() # DP Change 3: lookup table
    cache_hits = 0
    cache_misses = 0

    t_start = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]

        # DP Change 3: Seed from DP table if we've seen this opponent before
        dp_entry = dp_table.lookup(opponent)
        if dp_entry and gen > len(OPPONENT_ROTATION):
            # Replace worst agents with mutated versions of known-good weights
            pop_sorted = sorted(range(len(population)), key=lambda i: population[i].e_i)
            for idx in pop_sorted[:DP_SEED_COUNT]:
                seeded = TDCCEAAgent(dp_entry['weights'])
                # Small mutation so seeded agents explore nearby
                seeded.params = [max(0.0, min(1.0, p + random.gauss(0, 0.05)))
                                 for p in seeded.params]
                population[idx] = seeded

        # Centroid
        centroid = [0.0] * DIMENSIONS
        for agent in population:
            for j in range(DIMENSIONS):
                centroid[j] += agent.params[j]
        current_env = [x / len(population) for x in centroid]

        # Evaluate with DP Change 1: fitness cache
        for agent in population:
            cache_key = make_cache_key(agent.params, opponent)
            if cache_key in fitness_cache:
                f_i = fitness_cache[cache_key]
                cache_hits += 1
            else:
                f_i = evaluate_weights(server, agent.params, opponent)
                fitness_cache[cache_key] = f_i
                cache_misses += 1
            agent.update_history(f_i, current_env)
            agent.compute_dynamic_trajectory(current_env)
            agent.compute_adaptive_mutation()

        best = max(population, key=lambda a: a.e_i)
        gen_scores.append(best.e_i)

        # DP Change 3: store best weights for this opponent
        best_fitness_this_gen = evaluate_weights(server, best.params, opponent)
        dp_table.store(opponent, best.params, best_fitness_this_gen)

        seeded_tag = " [SEEDED]" if dp_entry and gen > len(OPPONENT_ROTATION) else ""
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opponent:25s} | e_i={best.e_i:.3f}{seeded_tag}")

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

        # DP Change 2: Smart archive — keep context
        for top in sorted(population, key=lambda a: a.e_i, reverse=True)[:3]:
            archived = TDCCEAAgent(top.params)
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

    total_evals = cache_hits + cache_misses
    print(f"\n  DP Cache: {cache_hits} hits / {total_evals} total ({cache_hits/max(1,total_evals)*100:.1f}% saved)")
    print(f"  DP Table: {len(dp_table)} opponents memorized")

    return best_overall.params, elapsed, gen_scores


# ===============================================================
#  STANDARD TD-CCEA CENTROID (baseline)
# ===============================================================
def run_tdccea_centroid(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: TD-CCEA Centroid (Standard)")
    print(f"{'='*60}")

    population = [TDCCEAAgent() for _ in range(N_POP)]
    archive = []
    gen_scores = []
    t_start = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
        centroid = [0.0] * DIMENSIONS
        for agent in population:
            for j in range(DIMENSIONS):
                centroid[j] += agent.params[j]
        current_env = [x / len(population) for x in centroid]

        for agent in population:
            f_i = evaluate_weights(server, agent.params, opponent)
            agent.update_history(f_i, current_env)
            agent.compute_dynamic_trajectory(current_env)
            agent.compute_adaptive_mutation()

        best = max(population, key=lambda a: a.e_i)
        gen_scores.append(best.e_i)
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opponent:25s} | e_i={best.e_i:.3f}")

        max_dist = math.sqrt(DIMENSIONS)
        survivors = []
        for agent in sorted(population, key=lambda a: a.e_i, reverse=True):
            redundant = any(
                1.0 - (euclidean_dist(agent.params, u.params) / max_dist) > DELTA_S
                for u in survivors
            )
            if not redundant:
                survivors.append(agent)

        final = [a for a in survivors if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
        while len(final) < 10 and len(survivors) > len(final):
            for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
                if a not in final:
                    final.append(a)
                    break

        for top in sorted(population, key=lambda a: a.e_i, reverse=True)[:3]:
            archive.append(TDCCEAAgent(top.params))
        archive.sort(key=lambda a: a.e_i, reverse=True)
        archive = archive[:A_MAX]

        pool = final + archive
        population = final[:]
        while len(population) < N_POP:
            candidates = random.sample(pool, min(3, len(pool)))
            parent = max(candidates, key=lambda a: a.e_i)
            population.append(parent.mutate())

    elapsed = time.time() - t_start
    best_overall = max(archive, key=lambda a: a.e_i) if archive else max(population, key=lambda a: a.e_i)
    return best_overall.params, elapsed, gen_scores


# ===============================================================
#  HILL CLIMBING (baseline)
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
        best_cand = current[:]
        best_fit = current_fit

        for _ in range(N_POP - 1):
            cand = [max(0.0, min(1.0, p + random.gauss(0, 0.15))) for p in current]
            fit = evaluate_weights(server, cand, opponent)
            if fit > best_fit:
                best_cand = cand[:]
                best_fit = fit

        if best_fit >= current_fit:
            current = best_cand[:]
            current_fit = best_fit

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
    print("  DP-TDCCEA vs TD-CCEA Centroid vs Hill Climbing")
    print("#" * 70)

    algorithms = [
        ("DP-TDCCEA", "dp"),
        ("TD-CCEA Centroid", "centroid"),
        ("Hill Climbing", "hc"),
    ]

    all_results = []

    for algo_name, algo_key in algorithms:
        server = GameServer()
        server.start()
        print(f"  [OK] Server started for {algo_name}")

        if algo_key == "dp":
            best_params, elapsed, gen_scores = run_dp_tdccea(server)
        elif algo_key == "centroid":
            best_params, elapsed, gen_scores = run_tdccea_centroid(server)
        else:
            best_params, elapsed, gen_scores = run_hill_climbing(server)

        print(f"\n  Validating {algo_name}...")
        overall_wr, per_opp = validate_weights(server, best_params)
        server.stop()

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
            "per_opponent": per_opp,
            "gen_scores": gen_scores,
            "best_params": best_params,
        })

    # Final table
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"  {'Algorithm':<22s}  {'Win Rate':>10s}  {'Time':>8s}")
    print(f"  {'_'*22}  {'_'*10}  {'_'*8}")
    for r in sorted(all_results, key=lambda x: x["overall_winrate"], reverse=True):
        print(f"  {r['name']:<22s}  {r['overall_winrate']:>9.1f}%  {r['time']:>7.1f}s")

    # Per-opponent
    print(f"\n{'='*60}")
    print(f"  WIN RATE PER OPPONENT")
    print(f"{'='*60}")
    header = f"  {'Opponent':<26s}"
    for r in all_results:
        header += f"  {r['name']:>16s}"
    print(header)
    print("  " + "-" * (26 + 18 * len(all_results)))

    for opp in OPPONENT_ROTATION:
        row = f"  {opp:<26s}"
        rates = [r["per_opponent"].get(opp, 0) for r in all_results]
        best = max(rates)
        for r in all_results:
            wr = r["per_opponent"].get(opp, 0)
            marker = " *" if wr == best and rates.count(best) == 1 else "  "
            row += f"  {wr:>13.0f}%{marker}"
        print(row)

    print("  " + "-" * (26 + 18 * len(all_results)))
    row = f"  {'OVERALL':<26s}"
    for r in all_results:
        row += f"  {r['overall_winrate']:>14.1f}%  "
    print(row)

    # Ranking
    ranked = sorted(all_results, key=lambda x: x["overall_winrate"], reverse=True)
    print(f"\n{'='*60}")
    print(f"  RANKING")
    print(f"{'='*60}")
    medals = ["1st", "2nd", "3rd"]
    for i, r in enumerate(ranked):
        print(f"  {medals[i]}  {r['name']:<22s}  {r['overall_winrate']:5.1f}%  ({r['time']:.0f}s)")

    if len(ranked) >= 2:
        d = ranked[0]["overall_winrate"] - ranked[1]["overall_winrate"]
        print(f"\n  {ranked[0]['name']} beats {ranked[1]['name']} by {abs(d):.1f}%")
    if len(ranked) >= 3:
        d = ranked[0]["overall_winrate"] - ranked[2]["overall_winrate"]
        print(f"  {ranked[0]['name']} beats {ranked[2]['name']} by {abs(d):.1f}%")

    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
