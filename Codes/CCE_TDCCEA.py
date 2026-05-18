"""
CCE_TDCCEA.py - TD-CCEA + Cooperative Co-Evolution (Full Hybrid)

NEW: CCE-TDCCEA applies FULL TD-CCEA logic to each sub-population:
  - Pop A (Macro/Economy, 8 weights): own centroid, trajectory, archive, pruning
  - Pop B (Micro/Combat, 12 weights): own centroid, trajectory, archive, pruning
  - Evaluation: combine Macro + Micro to play game (cooperation)

Compares:
  1. CCE-TDCCEA  (NEW: TD-CCEA + Co-Evolution hybrid)
  2. CCEA-TDCCEA (basic split, no TD-CCEA logic)
  3. TD-CCEA Centroid (standard monolithic)
  4. SA-TDCCEA / DP-TDCCEA / Hill Climbing
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

MACRO_INDICES = [0, 1, 2, 6, 14, 15, 17, 18]       # economy weights
MICRO_INDICES = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 16, 19]  # combat weights

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
        if line != "READY": raise RuntimeError(f"Failed: {line}")
    def run_game(self, weights, opponent):
        self.process.stdin.write(f"EVOLVED:{';'.join(f'{w:.4f}' for w in weights)},{opponent},{MAP_PATH},{MAX_CYCLES}\n")
        self.process.stdin.flush()
        r = self.process.stdout.readline().strip()
        if r.startswith("RESULT:"):
            p = r.replace("RESULT:", "").split(",")
            return int(p[0]), int(p[1])
        return -1, MAX_CYCLES
    def stop(self):
        if self.process:
            try: self.process.stdin.write("EXIT\n"); self.process.stdin.flush(); self.process.wait(timeout=5)
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
    tw, tg, po = 0, 0, {}
    for opp in OPPONENT_ROTATION:
        ow = sum(1 for _ in range(VALIDATION_GAMES) if server.run_game(weights, opp)[0] == 0)
        tg += VALIDATION_GAMES; tw += ow
        po[opp] = ow / VALIDATION_GAMES * 100
    return tw / tg * 100, po

def load_sensitivity():
    fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), SENSITIVITY_FILE)
    if os.path.exists(fp):
        with open(fp) as f: return json.load(f)["normalized_mutation_rates"]
    return None

def combine_weights(macro_agent, micro_agent):
    full = [0.5] * DIMENSIONS
    for i, idx in enumerate(MACRO_INDICES): full[idx] = macro_agent.params[i]
    for i, idx in enumerate(MICRO_INDICES): full[idx] = micro_agent.params[i]
    return full


# ===============================================================
#  TD-CCEA SUB-POPULATION AGENT
# ===============================================================
class SubAgent:
    """Agent for a sub-population with full TD-CCEA logic."""
    def __init__(self, dim, params=None):
        self.dim = dim
        self.params = [random.random() for _ in range(dim)] if params is None \
            else [max(0.0, min(1.0, float(x))) for x in params]
        self.H_i = []
        self.e_i = 0.0
        self.mu_i = 0.0

    def update_history(self, fitness, env):
        self.H_i.append((fitness, env))
        if len(self.H_i) > 7: self.H_i.pop(0)

    def compute_trajectory(self, env):
        if not self.H_i:
            self.e_i = 0.0
            return
        md = math.sqrt(self.dim)
        ws = [1.0 - euclidean_dist(env, pe)/md for _, pe in self.H_i]
        t = sum(ws)
        ws = [w/t for w in ws] if t > 0 else [1.0/len(ws)]*len(ws)
        self.e_i = sum(w * self.H_i[i][0] for i, w in enumerate(ws))

    def compute_mutation(self):
        self.mu_i = 0.15 * (1.0 - self.e_i) + 0.03

    def mutate(self):
        cp = [max(0.0, min(1.0, p + random.gauss(0, self.mu_i))) for p in self.params]
        return SubAgent(self.dim, cp)


def run_subpop_tdccea(population, archive):
    """Apply TD-CCEA survival: redundancy removal + fitness pruning + archive update."""
    dim = population[0].dim
    md = math.sqrt(dim)

    # Redundancy removal
    survivors = []
    for a in sorted(population, key=lambda a: a.e_i, reverse=True):
        if not any(1.0 - euclidean_dist(a.params, u.params)/md > DELTA_S for u in survivors):
            survivors.append(a)

    # Fitness pruning
    final = [a for a in survivors if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
    while len(final) < 8 and len(survivors) > len(final):
        for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
            if a not in final: final.append(a); break

    # Archive update
    for top in sorted(population, key=lambda a: a.e_i, reverse=True)[:3]:
        ar = SubAgent(dim, top.params)
        ar.e_i, ar.H_i, ar.mu_i = top.e_i, top.H_i[:], top.mu_i
        archive.append(ar)
    archive.sort(key=lambda a: a.e_i, reverse=True)
    del archive[A_MAX:]

    # Breed
    pool = final + archive
    new_pop = final[:]
    while len(new_pop) < N_POP:
        cands = random.sample(pool, min(3, len(pool)))
        new_pop.append(max(cands, key=lambda a: a.e_i).mutate())

    return new_pop, archive


# ===============================================================
#  1. CCE-TDCCEA (NEW: TD-CCEA + Cooperative Co-Evolution)
# ===============================================================
def run_cce_tdccea(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: CCE-TDCCEA (TD-CCEA + Co-Evolution Hybrid)")
    print(f"  Pop A (Macro): {len(MACRO_INDICES)} weights with full TD-CCEA")
    print(f"  Pop B (Micro): {len(MICRO_INDICES)} weights with full TD-CCEA")
    print(f"{'='*60}")

    pop_a = [SubAgent(len(MACRO_INDICES)) for _ in range(N_POP)]
    pop_b = [SubAgent(len(MICRO_INDICES)) for _ in range(N_POP)]
    archive_a, archive_b = [], []
    best_a, best_b = pop_a[0], pop_b[0]
    gen_scores = []
    t0 = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]

        # --- Centroid for each sub-population ---
        centroid_a = [sum(a.params[j] for a in pop_a)/len(pop_a) for j in range(len(MACRO_INDICES))]
        centroid_b = [sum(b.params[j] for b in pop_b)/len(pop_b) for j in range(len(MICRO_INDICES))]

        # --- Evaluate Pop A: pair each macro with best micro ---
        for agent_a in pop_a:
            full = combine_weights(agent_a, best_b)
            f_i = evaluate_weights(server, full, opponent)
            agent_a.update_history(f_i, centroid_a)
            agent_a.compute_trajectory(centroid_a)
            agent_a.compute_mutation()

        # --- Evaluate Pop B: pair each micro with best macro ---
        for agent_b in pop_b:
            full = combine_weights(best_a, agent_b)
            f_i = evaluate_weights(server, full, opponent)
            agent_b.update_history(f_i, centroid_b)
            agent_b.compute_trajectory(centroid_b)
            agent_b.compute_mutation()

        best_a = max(pop_a, key=lambda a: a.e_i)
        best_b = max(pop_b, key=lambda a: a.e_i)

        combined_fit = evaluate_weights(server, combine_weights(best_a, best_b), opponent)
        gen_scores.append(combined_fit)

        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opponent:25s} | "
              f"A_e={best_a.e_i:.3f} B_e={best_b.e_i:.3f} Combined={combined_fit:.3f}")

        # --- Apply TD-CCEA survival to each sub-population independently ---
        pop_a, archive_a = run_subpop_tdccea(pop_a, archive_a)
        pop_b, archive_b = run_subpop_tdccea(pop_b, archive_b)

    best_weights = combine_weights(best_a, best_b)
    return best_weights, time.time() - t0, gen_scores


# ===============================================================
#  2. BASIC CCEA (split but NO TD-CCEA logic)
# ===============================================================
def run_basic_ccea(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: Basic CCEA (Split, No TD-CCEA)")
    print(f"{'='*60}")

    pop_a = [SubAgent(len(MACRO_INDICES)) for _ in range(N_POP)]
    pop_b = [SubAgent(len(MICRO_INDICES)) for _ in range(N_POP)]
    best_a, best_b = pop_a[0], pop_b[0]
    gen_scores = []
    t0 = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]

        for a in pop_a:
            a.fitness = evaluate_weights(server, combine_weights(a, best_b), opponent)
            a.e_i = a.fitness
        for b in pop_b:
            b.fitness = evaluate_weights(server, combine_weights(best_a, b), opponent)
            b.e_i = b.fitness

        best_a = max(pop_a, key=lambda a: a.e_i)
        best_b = max(pop_b, key=lambda a: a.e_i)
        cf = evaluate_weights(server, combine_weights(best_a, best_b), opponent)
        gen_scores.append(cf)

        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opponent:25s} | "
              f"Macro={best_a.e_i:.3f} Micro={best_b.e_i:.3f} Combined={cf:.3f}")

        # Simple tournament evolution (no TD-CCEA)
        for pop, best in [(pop_a, best_a), (pop_b, best_b)]:
            new = [best]
            while len(new) < N_POP:
                t1, t2 = random.sample(pop, 2)
                p = t1 if t1.e_i >= t2.e_i else t2
                new.append(p.mutate())
            pop[:] = new

    return combine_weights(best_a, best_b), time.time() - t0, gen_scores


# ===============================================================
#  3. STANDARD TD-CCEA CENTROID
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

def evolve_core(server, label, pwm=None, use_dp=False):
    print(f"\n{'='*60}")
    print(f"  RUNNING: {label}")
    print(f"{'='*60}")
    population = [Agent(pwm=pwm) for _ in range(N_POP)]
    archive, gen_scores, dp_table, cache = [], [], {}, {}
    t0 = time.time()
    for gen in range(1, TOTAL_GENERATIONS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
        if use_dp and opponent in dp_table and gen > len(OPPONENT_ROTATION):
            dw = dp_table[opponent]['weights']
            for idx in sorted(range(len(population)), key=lambda i: population[i].e_i)[:3]:
                s = Agent(dw, pwm)
                s.params = [max(0.0, min(1.0, p + random.gauss(0, 0.05))) for p in s.params]
                population[idx] = s
        centroid = [sum(a.params[j] for a in population)/len(population) for j in range(DIMENSIONS)]
        for agent in population:
            if use_dp:
                key = (tuple(round(w,2) for w in agent.params), opponent)
                f_i = cache.get(key)
                if f_i is None:
                    f_i = evaluate_weights(server, agent.params, opponent); cache[key] = f_i
            else:
                f_i = evaluate_weights(server, agent.params, opponent)
            agent.update_history(f_i, centroid); agent.compute_trajectory(centroid); agent.compute_mutation()
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
        archive.sort(key=lambda a: a.e_i, reverse=True); archive = archive[:A_MAX]
        pool = final + archive; population = final[:]
        while len(population) < N_POP:
            cands = random.sample(pool, min(3, len(pool)))
            population.append(max(cands, key=lambda a: a.e_i).mutate())
    best_all = max(archive, key=lambda a: a.e_i) if archive else max(population, key=lambda a: a.e_i)
    return best_all.params, time.time() - t0, gen_scores


# ===============================================================
#  6. HILL CLIMBING
# ===============================================================
def run_hc(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: Hill Climbing")
    print(f"{'='*60}")
    cur = [random.random() for _ in range(DIMENSIONS)]
    cf, gs = 0.0, []
    t0 = time.time()
    for gen in range(1, TOTAL_GENERATIONS + 1):
        opp = OPPONENT_ROTATION[(gen-1) % len(OPPONENT_ROTATION)]
        cf = evaluate_weights(server, cur, opp)
        bc, bf = cur[:], cf
        for _ in range(N_POP-1):
            c = [max(0.0, min(1.0, p+random.gauss(0,0.15))) for p in cur]
            f = evaluate_weights(server, c, opp)
            if f > bf: bc, bf = c[:], f
        if bf >= cf: cur, cf = bc[:], bf
        gs.append(cf)
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | Best={cf:.3f}")
    return cur, time.time()-t0, gs


# ===============================================================
#  MAIN
# ===============================================================
def main():
    print()
    print("#" * 70)
    print("  CCE-TDCCEA BENCHMARK")
    print("  TD-CCEA + Co-Evolution vs Basic CCEA vs Standard vs SA vs DP vs HC")
    print("#" * 70)
    print(f"  Generations: {TOTAL_GENERATIONS} | Opponents: {len(OPPONENT_ROTATION)} | Pop: {N_POP}")
    print()

    sa_mut = load_sensitivity()

    algos = [
        ("CCE-TDCCEA",        "cce"),
        ("Basic CCEA",        "basic_ccea"),
        ("TD-CCEA Centroid",  "centroid"),
        ("SA-TDCCEA",         "sa"),
        ("DP-TDCCEA",         "dp"),
        ("Hill Climbing",     "hc"),
    ]

    all_results = []
    for name, key in algos:
        server = GameServer(); server.start()
        print(f"  [OK] Server started for {name}")

        if key == "cce":       best, el, sc = run_cce_tdccea(server)
        elif key == "basic_ccea": best, el, sc = run_basic_ccea(server)
        elif key == "centroid":   best, el, sc = evolve_core(server, name)
        elif key == "sa":         best, el, sc = evolve_core(server, name, pwm=sa_mut)
        elif key == "dp":         best, el, sc = evolve_core(server, name, use_dp=True)
        else:                     best, el, sc = run_hc(server)

        print(f"\n  Validating {name}...")
        wr, po = validate_weights(server, best)
        server.stop()

        print(f"\n  +-- {name} RESULTS --------------------------------")
        print(f"  |  Time: {el:.1f}s | Win Rate: {wr:.1f}%")
        for opp, w in po.items():
            print(f"  |  {'[W]' if w >= 50 else '[L]'} vs {opp:<26s}: {w:5.0f}%")
        print(f"  +-------------------------------------------------------")
        all_results.append({"name": name, "time": el, "overall_winrate": wr,
                            "per_opponent": po, "gen_scores": sc})

    # ── TABLE ──
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Algorithm':<22s}  {'Win Rate':>10s}  {'Time':>8s}")
    print(f"  {'_'*22}  {'_'*10}  {'_'*8}")
    for r in sorted(all_results, key=lambda x: x["overall_winrate"], reverse=True):
        print(f"  {r['name']:<22s}  {r['overall_winrate']:>9.1f}%  {r['time']:>7.1f}s")

    print(f"\n{'='*75}")
    print(f"  WIN RATE PER OPPONENT")
    print(f"{'='*75}")
    h = f"  {'Opponent':<26s}"
    for r in all_results: h += f" {r['name']:>11s}"
    print(h)
    print("  " + "-" * (26 + 12 * len(all_results)))
    for opp in OPPONENT_ROTATION:
        row = f"  {opp:<26s}"
        rates = [r["per_opponent"].get(opp, 0) for r in all_results]
        mx = max(rates)
        for r in all_results:
            w = r["per_opponent"].get(opp, 0)
            m = "*" if w == mx and rates.count(mx) == 1 else " "
            row += f" {w:>9.0f}%{m}"
        print(row)

    ranked = sorted(all_results, key=lambda x: x["overall_winrate"], reverse=True)
    medals = ["1st", "2nd", "3rd", "4th", "5th", "6th"]
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
