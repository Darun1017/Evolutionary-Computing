"""
COMPARE_3WAY.py - 3-Way Co-Evolution (Economy/Production/Combat) vs baselines

Split: Economy(5w) + Production(7w) + Combat(8w) = 20 weights
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

ECONOMY_INDICES   = [0, 1, 2, 14, 18]       # worker_ratio, max_workers, expand, economy_pressure, worker_fight
PRODUCTION_INDICES = [3, 4, 5, 6, 13, 15, 17] # light/heavy/ranged pref, barracks, tech_switch, scout, 2nd_barracks
COMBAT_INDICES     = [7, 8, 9, 10, 11, 12, 16, 19] # aggression, attack, defend, target, defense, counter, group, focus

N_POP = 20
K_GAMES = 3
TOTAL_GENERATIONS = 10
VALIDATION_GAMES = 5
A_MAX = 15
DELTA_S = 0.85
DELTA_F = 0.20
TOP_PARTNERS = 2
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
        ws = ";".join(f"{w:.4f}" for w in weights)
        self.process.stdin.write(f"EVOLVED:{ws},{opponent},{MAP_PATH},{MAX_CYCLES}\n")
        self.process.stdin.flush()
        r = self.process.stdout.readline().strip()
        if r.startswith("RESULT:"):
            p = r.replace("RESULT:", "").split(",")
            return int(p[0]), int(p[1])
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
    tw, tg, po = 0, 0, {}
    for opp in OPPONENT_ROTATION:
        ow = 0
        for _ in range(VALIDATION_GAMES):
            w, _ = server.run_game(weights, opp)
            if w == 0: ow += 1
            tg += 1
        tw += ow
        po[opp] = ow / VALIDATION_GAMES * 100
    return tw / tg * 100, po

def load_sensitivity():
    fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), SENSITIVITY_FILE)
    if os.path.exists(fp):
        with open(fp) as f: return json.load(f)["normalized_mutation_rates"]
    return None


# ===============================================================
#  SUB-AGENT (for 3-way co-evolution)
# ===============================================================
class SubAgent:
    def __init__(self, dim, params=None):
        self.dim = dim
        self.params = [random.random() for _ in range(dim)] if params is None \
            else [max(0.0, min(1.0, float(x))) for x in params]
        self.H_i, self.e_i, self.mu_i = [], 0.0, 0.0

    def update_history(self, fitness, env):
        self.H_i.append((fitness, env))
        if len(self.H_i) > 7: self.H_i.pop(0)

    def compute_trajectory(self, env):
        if not self.H_i: self.e_i = 0.0; return
        md = math.sqrt(self.dim)
        ws = [1.0 - euclidean_dist(env[:self.dim], pe[:self.dim])/md
              if len(pe) >= self.dim else 1.0 for _, pe in self.H_i]
        t = sum(ws)
        ws = [w/t for w in ws] if t > 0 else [1.0/len(ws)]*len(ws)
        self.e_i = sum(w * self.H_i[i][0] for i, w in enumerate(ws))

    def compute_mutation(self):
        self.mu_i = 0.15 * (1.0 - self.e_i) + 0.03

    def mutate(self):
        return SubAgent(self.dim, [max(0.0, min(1.0, p + random.gauss(0, self.mu_i))) for p in self.params])


def combine_3way(econ, prod, combat):
    full = [0.5] * DIMENSIONS
    for i, idx in enumerate(ECONOMY_INDICES): full[idx] = econ.params[i]
    for i, idx in enumerate(PRODUCTION_INDICES): full[idx] = prod.params[i]
    for i, idx in enumerate(COMBAT_INDICES): full[idx] = combat.params[i]
    return full


def subpop_survival(population, archive):
    dim = population[0].dim
    md = math.sqrt(dim)
    survivors = []
    for a in sorted(population, key=lambda a: a.e_i, reverse=True):
        if not any(1.0 - euclidean_dist(a.params, u.params)/md > DELTA_S for u in survivors):
            survivors.append(a)
    final = [a for a in survivors if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
    while len(final) < 6 and len(survivors) > len(final):
        for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
            if a not in final: final.append(a); break
    for top in sorted(population, key=lambda a: a.e_i, reverse=True)[:3]:
        ar = SubAgent(dim, top.params)
        ar.e_i, ar.H_i, ar.mu_i = top.e_i, top.H_i[:], top.mu_i
        archive.append(ar)
    archive.sort(key=lambda a: a.e_i, reverse=True)
    del archive[A_MAX:]
    pool = final + archive
    new_pop = final[:]
    while len(new_pop) < N_POP:
        cands = random.sample(pool, min(3, len(pool)))
        new_pop.append(max(cands, key=lambda a: a.e_i).mutate())
    return new_pop, archive


# ===============================================================
#  1. CCE-TDCCEA 3-WAY
# ===============================================================
def run_ccea_3way(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: CCE-TDCCEA 3-Way (Economy/Production/Combat)")
    print(f"  Econ: {len(ECONOMY_INDICES)}w | Prod: {len(PRODUCTION_INDICES)}w | Combat: {len(COMBAT_INDICES)}w | Partners: {TOP_PARTNERS}")
    print(f"{'='*60}")

    pop_e = [SubAgent(len(ECONOMY_INDICES)) for _ in range(N_POP)]
    pop_p = [SubAgent(len(PRODUCTION_INDICES)) for _ in range(N_POP)]
    pop_c = [SubAgent(len(COMBAT_INDICES)) for _ in range(N_POP)]
    arc_e, arc_p, arc_c = [], [], []

    top_e = pop_e[:TOP_PARTNERS]
    top_p = pop_p[:TOP_PARTNERS]
    top_c = pop_c[:TOP_PARTNERS]
    gen_scores = []
    t0 = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]

        ce = [sum(a.params[j] for a in pop_e)/len(pop_e) for j in range(len(ECONOMY_INDICES))]
        cp = [sum(a.params[j] for a in pop_p)/len(pop_p) for j in range(len(PRODUCTION_INDICES))]
        cc = [sum(a.params[j] for a in pop_c)/len(pop_c) for j in range(len(COMBAT_INDICES))]
        env = ce + cp + cc

        # Evaluate Economy: pair with top Production & Combat partners
        for agent in pop_e:
            fits = []
            for pp in top_p:
                for pc in top_c:
                    fits.append(evaluate_weights(server, combine_3way(agent, pp, pc), opponent))
            avg = sum(fits)/len(fits)
            agent.update_history(avg, env); agent.compute_trajectory(env); agent.compute_mutation()

        # Evaluate Production: pair with top Economy & Combat partners
        for agent in pop_p:
            fits = []
            for pe in top_e:
                for pc in top_c:
                    fits.append(evaluate_weights(server, combine_3way(pe, agent, pc), opponent))
            avg = sum(fits)/len(fits)
            agent.update_history(avg, env); agent.compute_trajectory(env); agent.compute_mutation()

        # Evaluate Combat: pair with top Economy & Production partners
        for agent in pop_c:
            fits = []
            for pe in top_e:
                for pp in top_p:
                    fits.append(evaluate_weights(server, combine_3way(pe, pp, agent), opponent))
            avg = sum(fits)/len(fits)
            agent.update_history(avg, env); agent.compute_trajectory(env); agent.compute_mutation()

        best_e = max(pop_e, key=lambda a: a.e_i)
        best_p = max(pop_p, key=lambda a: a.e_i)
        best_c = max(pop_c, key=lambda a: a.e_i)

        top_e = sorted(pop_e, key=lambda a: a.e_i, reverse=True)[:TOP_PARTNERS]
        top_p = sorted(pop_p, key=lambda a: a.e_i, reverse=True)[:TOP_PARTNERS]
        top_c = sorted(pop_c, key=lambda a: a.e_i, reverse=True)[:TOP_PARTNERS]

        cmb = evaluate_weights(server, combine_3way(best_e, best_p, best_c), opponent)
        gen_scores.append(cmb)

        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opponent:25s} | "
              f"E={best_e.e_i:.3f} P={best_p.e_i:.3f} C={best_c.e_i:.3f} Cmb={cmb:.3f}")

        pop_e, arc_e = subpop_survival(pop_e, arc_e)
        pop_p, arc_p = subpop_survival(pop_p, arc_p)
        pop_c, arc_c = subpop_survival(pop_c, arc_c)

    return combine_3way(best_e, best_p, best_c), time.time() - t0, gen_scores


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
#  2/3/4. TD-CCEA / SA-TDCCEA / DP-TDCCEA
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
    print("  COMPARE 3-WAY: CCE-TDCCEA(3) vs TD-CCEA vs SA vs DP vs HC")
    print("#" * 70)
    print(f"  3-Way Split: Economy({len(ECONOMY_INDICES)}) + Production({len(PRODUCTION_INDICES)}) + Combat({len(COMBAT_INDICES)}) = {DIMENSIONS}")
    print(f"  Generations: {TOTAL_GENERATIONS} | Opponents: {len(OPPONENT_ROTATION)} | Pop: {N_POP}")
    print()

    sa_mut = load_sensitivity()

    algos = [
        ("CCE-TDCCEA-3Way", "3way"),
        ("TD-CCEA Centroid", "centroid"),
        ("SA-TDCCEA",        "sa"),
        ("DP-TDCCEA",        "dp"),
        ("Hill Climbing",    "hc"),
    ]

    all_results = []
    for name, key in algos:
        server = GameServer()
        server.start()
        print(f"  [OK] Server started for {name}")

        if key == "3way":
            best, elapsed, scores = run_ccea_3way(server)
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
