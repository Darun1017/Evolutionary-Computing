"""
COMPARE_ALL.py - Compare all 5 algorithms with per-generation opponent rotation

Algorithms:
  1. CCEA-TDCCEA  (split Macro/Micro populations)
  2. TD-CCEA Centroid (standard)
  3. SA-TDCCEA (sensitivity-adaptive mutation)
  4. DP-TDCCEA (fitness cache + DP seeding)
  5. Hill Climbing (greedy baseline)

Rotation: gen1=opp1, gen2=opp2, gen3=opp3, ... (cycling through 7 opponents)
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

MACRO_INDICES = [0, 1, 2, 6, 14, 15, 17, 18]
MICRO_INDICES = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 16, 19]

N_POP = 20
K_GAMES = 3
TOTAL_GENERATIONS = 30
VALIDATION_GAMES = 5
A_MAX = 15
DELTA_S = 0.85
DELTA_F = 0.20
SENSITIVITY_FILE = "sensitivity_scores.json"


class GameServer:
    def __init__(self):
        self.process = None
    def start(self):
        cmd = ["java", "-cp", "lib/*;bin", "tests.EvolvedBatchGameRunner"]
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, cwd=MICRORTS_DIR)
        line = self.process.stdout.readline().strip()
        if line != "READY":
            raise RuntimeError(f"Failed to start: {line}")
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
                self.process.stdin.write("EXIT\n"); self.process.stdin.flush()
                self.process.wait(timeout=5)
            except: self.process.kill()


def euclidean_dist(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))

def evaluate_weights(server, weights, opponent):
    wins = 0
    for _ in range(K_GAMES):
        w, _ = server.run_game(weights, opponent)
        if w == 0: wins += 1
        elif w == -1: wins += 0.3
    return wins / K_GAMES

def validate_weights(server, weights):
    total_w, total_g, per_opp = 0, 0, {}
    for opp in OPPONENT_ROTATION:
        ow = 0
        for _ in range(VALIDATION_GAMES):
            w, _ = server.run_game(weights, opp)
            if w == 0: ow += 1
            total_g += 1
        total_w += ow
        per_opp[opp] = ow / VALIDATION_GAMES * 100
    return total_w / total_g * 100, per_opp

def load_sensitivity():
    fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), SENSITIVITY_FILE)
    if os.path.exists(fp):
        with open(fp) as f: return json.load(f)["normalized_mutation_rates"]
    return None


# ===============================================================
#  CCEA CHROMOSOME
# ===============================================================
class Chromosome:
    def __init__(self, indices, params=None):
        self.indices = indices
        self.dim = len(indices)
        self.genes = [random.random() for _ in range(self.dim)] if params is None \
            else [max(0.0, min(1.0, float(x))) for x in params]
        self.fitness = 0.0
    def mutate(self, mu=0.12):
        return Chromosome(self.indices, [max(0.0, min(1.0, g + random.gauss(0, mu))) for g in self.genes])

def combine_chromosomes(macro, micro):
    full = [0.5] * DIMENSIONS
    for i, idx in enumerate(macro.indices): full[idx] = macro.genes[i]
    for i, idx in enumerate(micro.indices): full[idx] = micro.genes[i]
    return full


# ===============================================================
#  1. CCEA-TDCCEA
# ===============================================================
def run_ccea(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: CCEA-TDCCEA (Cooperative Co-Evolution)")
    print(f"  Pop A (Macro): {len(MACRO_INDICES)} weights | Pop B (Micro): {len(MICRO_INDICES)} weights")
    print(f"{'='*60}")

    pop_a = [Chromosome(MACRO_INDICES) for _ in range(N_POP)]
    pop_b = [Chromosome(MICRO_INDICES) for _ in range(N_POP)]
    best_a, best_b = pop_a[0], pop_b[0]
    gen_scores = []
    t0 = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]

        for a in pop_a:
            a.fitness = evaluate_weights(server, combine_chromosomes(a, best_b), opponent)
        for b in pop_b:
            b.fitness = evaluate_weights(server, combine_chromosomes(best_a, b), opponent)

        best_a = max(pop_a, key=lambda x: x.fitness)
        best_b = max(pop_b, key=lambda x: x.fitness)
        combined_fit = evaluate_weights(server, combine_chromosomes(best_a, best_b), opponent)
        gen_scores.append(combined_fit)

        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opponent:25s} | "
              f"Macro={best_a.fitness:.3f} Micro={best_b.fitness:.3f} Combined={combined_fit:.3f}")

        # Evolve each population independently
        for pop, best in [(pop_a, best_a), (pop_b, best_b)]:
            new_pop = [best]
            while len(new_pop) < N_POP:
                t1, t2 = random.sample(pop, 2)
                parent = t1 if t1.fitness >= t2.fitness else t2
                new_pop.append(parent.mutate(0.15 * (1.0 - parent.fitness) + 0.03))
            pop[:] = new_pop

    return combine_chromosomes(best_a, best_b), time.time() - t0, gen_scores


# ===============================================================
#  MONOLITHIC AGENT (for TD-CCEA / SA / DP)
# ===============================================================
class Agent:
    def __init__(self, params=None, pwm=None):
        self.params = [random.random() for _ in range(DIMENSIONS)] if params is None \
            else [max(0.0, min(1.0, float(x))) for x in params]
        self.H_i, self.e_i, self.mu_i, self.pwm = [], 0.0, 0.0, pwm

    def update_history(self, f, env):
        self.H_i.append((f, env))
        if len(self.H_i) > 7: self.H_i.pop(0)

    def compute_trajectory(self, env):
        if not self.H_i: self.e_i = 0.0; return
        md = math.sqrt(DIMENSIONS)
        ws = [1.0 - euclidean_dist(env, pe)/md for _, pe in self.H_i]
        t = sum(ws)
        ws = [w/t for w in ws] if t > 0 else [1.0/len(ws)]*len(ws)
        self.e_i = sum(w * self.H_i[i][0] for i, w in enumerate(ws))

    def compute_mutation(self):
        self.mu_i = 0.15 * (1.0 - self.e_i) + 0.03

    def mutate(self):
        cp = []
        if self.pwm:
            for j, p in enumerate(self.params):
                cp.append(max(0.0, min(1.0, p + random.gauss(0, self.pwm[j]*(1-self.e_i)+0.02))))
        else:
            for p in self.params:
                cp.append(max(0.0, min(1.0, p + random.gauss(0, self.mu_i))))
        return Agent(cp, self.pwm)


# ===============================================================
#  2/3/4. TD-CCEA / SA-TDCCEA / DP-TDCCEA (shared core)
# ===============================================================
def evolve_core(server, label, pwm=None, use_dp=False):
    print(f"\n{'='*60}")
    print(f"  RUNNING: {label}")
    print(f"{'='*60}")

    population = [Agent(pwm=pwm) for _ in range(N_POP)]
    archive, gen_scores = [], []
    dp_table, cache = {}, {}
    t0 = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]

        if use_dp and opponent in dp_table and gen > len(OPPONENT_ROTATION):
            dw = dp_table[opponent]['weights']
            worst = sorted(range(len(population)), key=lambda i: population[i].e_i)
            for idx in worst[:3]:
                s = Agent(dw, pwm)
                s.params = [max(0.0, min(1.0, p + random.gauss(0, 0.05))) for p in s.params]
                population[idx] = s

        centroid = [sum(a.params[j] for a in population)/len(population) for j in range(DIMENSIONS)]

        for agent in population:
            if use_dp:
                key = (tuple(round(w, 2) for w in agent.params), opponent)
                f_i = cache.get(key)
                if f_i is None:
                    f_i = evaluate_weights(server, agent.params, opponent)
                    cache[key] = f_i
            else:
                f_i = evaluate_weights(server, agent.params, opponent)
            agent.update_history(f_i, centroid)
            agent.compute_trajectory(centroid)
            agent.compute_mutation()

        best = max(population, key=lambda a: a.e_i)
        gen_scores.append(best.e_i)

        if use_dp:
            bf = evaluate_weights(server, best.params, opponent)
            if opponent not in dp_table or bf > dp_table[opponent]['fitness']:
                dp_table[opponent] = {'weights': best.params[:], 'fitness': bf}

        tag = " [SEEDED]" if use_dp and opponent in dp_table and gen > len(OPPONENT_ROTATION) else ""
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opponent:25s} | e_i={best.e_i:.3f}{tag}")

        md = math.sqrt(DIMENSIONS)
        survivors = []
        for a in sorted(population, key=lambda a: a.e_i, reverse=True):
            if not any(1.0 - euclidean_dist(a.params, u.params)/md > DELTA_S for u in survivors):
                survivors.append(a)

        final = [a for a in survivors if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
        while len(final) < 10 and len(survivors) > len(final):
            for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
                if a not in final: final.append(a); break

        for top in sorted(population, key=lambda a: a.e_i, reverse=True)[:3]:
            ar = Agent(top.params, pwm)
            if use_dp: ar.e_i, ar.H_i, ar.mu_i = top.e_i, top.H_i[:], top.mu_i
            archive.append(ar)
        archive.sort(key=lambda a: a.e_i, reverse=True)
        archive = archive[:A_MAX]

        pool = final + archive
        population = final[:]
        while len(population) < N_POP:
            cands = random.sample(pool, min(3, len(pool)))
            population.append(max(cands, key=lambda a: a.e_i).mutate())

    best_all = max(archive, key=lambda a: a.e_i) if archive else max(population, key=lambda a: a.e_i)
    return best_all.params, time.time() - t0, gen_scores


# ===============================================================
#  5. HILL CLIMBING
# ===============================================================
def run_hill_climbing(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: Hill Climbing")
    print(f"{'='*60}")
    current = [random.random() for _ in range(DIMENSIONS)]
    cfit, gen_scores = 0.0, []
    t0 = time.time()
    for gen in range(1, TOTAL_GENERATIONS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
        cfit = evaluate_weights(server, current, opponent)
        bc, bf = current[:], cfit
        for _ in range(N_POP - 1):
            c = [max(0.0, min(1.0, p + random.gauss(0, 0.15))) for p in current]
            f = evaluate_weights(server, c, opponent)
            if f > bf: bc, bf = c[:], f
        if bf >= cfit: current, cfit = bc[:], bf
        gen_scores.append(cfit)
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opponent:25s} | Best={cfit:.3f}")
    return current, time.time() - t0, gen_scores


# ===============================================================
#  MAIN
# ===============================================================
def main():
    print()
    print("#" * 70)
    print("  COMPARE ALL: CCEA vs TD-CCEA vs SA vs DP vs Hill Climbing")
    print("#" * 70)
    print(f"  Generations: {TOTAL_GENERATIONS} | Opponents: {len(OPPONENT_ROTATION)} | Pop: {N_POP}")
    print()

    sa_mut = load_sensitivity()

    algos = [
        ("CCEA-TDCCEA",       "ccea"),
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

        if key == "ccea":
            best, elapsed, scores = run_ccea(server)
        elif key == "centroid":
            best, elapsed, scores = evolve_core(server, name)
        elif key == "sa":
            best, elapsed, scores = evolve_core(server, name, pwm=sa_mut)
        elif key == "dp":
            best, elapsed, scores = evolve_core(server, name, use_dp=True)
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
        all_results.append({"name": name, "time": elapsed, "overall_winrate": wr,
                            "per_opponent": per_opp, "gen_scores": scores})

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
    for r in all_results: header += f" {r['name']:>13s}"
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
    for r in all_results: row += f" {r['overall_winrate']:>11.1f}%  "
    print(row)

    # ── RANKING ──
    ranked = sorted(all_results, key=lambda x: x["overall_winrate"], reverse=True)
    medals = ["1st", "2nd", "3rd", "4th", "5th"]
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
