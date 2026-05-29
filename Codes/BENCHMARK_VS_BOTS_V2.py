"""
BENCHMARK_VS_BOTS.py
====================
8 algorithms evolve against 7 microRTS bots, then validated against all bots.

FIXES OVER SUBMITTED VERSION:
  [FIX 1] Added sensitivity analysis runner — regenerates sensitivity_scores.json
           with CORRECT direction (high influence = SMALL mutation 0.08 base)
           if file is missing or stale
  [FIX 2] Added Wilcoxon signed-rank test (pure Python, no scipy needed)
  [FIX 3] Added ablation analysis section
  [FIX 4] Added JSON results save
  [FIX 5] Added best-marker (*) in per-opponent table
  [FIX 6] Renamed 'np' local variable in breed() to 'new_pop' — clarity
  [FIX 7] SA mutation confirmed correct direction in Agent.mutate()

ALGORITHMS:
  1. CCE-TDCCEA      — Our novel algorithm
  2. Standard CCEA   — Same split, no trajectory (ablation baseline)
  3. CMA-ES          — Industry-standard continuous optimizer
  4. DE/rand/1/bin   — Differential Evolution
  5. PSO             — Particle Swarm Optimization
  6. TD-CCEA         — Monolithic trajectory baseline
  7. SA-TDCCEA       — Sensitivity-adaptive variant
  8. Hill Climbing   — Greedy local search baseline

INSTALL: pip install cma
"""

import math, random, time, subprocess, os, json

os.environ["PYTHONIOENCODING"] = "utf-8"

# ===============================================================
#  CONFIGURATION
# ===============================================================
MICRORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Simulation", "microrts"
)
WEIGHT_NAMES = [
    "worker_ratio","max_workers","expand_threshold","light_preference",
    "heavy_preference","ranged_preference","barracks_timing","aggression",
    "attack_threshold","defend_radius","target_priority","defense_trigger",
    "counter_rush","tech_switch","economy_pressure","scout_priority",
    "group_attack","second_barracks","worker_fight_ratio","focus_fire",
]
DIMENSIONS    = len(WEIGHT_NAMES)
MACRO_INDICES = [0,1,2,6,14,15,17,18]
MICRO_INDICES = [3,4,5,7,8,9,10,11,12,13,16,19]
MAP_PATH      = "maps/chambers32x32.xml"
MAX_CYCLES    = 10000

OPPONENT_ROTATION = [
    "EconomyMilitaryRush","LightRush","WorkerRush",
    "HeavyDefense","RangedRush","EconomyRush","WorkerRushPlusPlus",
]

N_POP             = 20
K_GAMES           = 3
TOTAL_GENERATIONS = 30
VALIDATION_GAMES  = 5
A_MAX             = 15
DELTA_S           = 0.85
DELTA_F           = 0.20
TOP_PARTNERS      = 3
SENSITIVITY_FILE  = "sensitivity_scores.json"


# ===============================================================
#  GAME SERVER
# ===============================================================
class GameServer:
    def __init__(self): self.process = None

    def start(self):
        self.process = subprocess.Popen(
            ["java", "-cp", "lib/*;bin", "tests.EvolvedBatchGameRunner"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=MICRORTS_DIR
        )
        if self.process.stdout.readline().strip() != "READY":
            raise RuntimeError("Server failed to start")

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
        w, _ = server.run_game(weights, opponent)
        if w == 0:   wins += 1
        elif w == -1: wins += 0.3
    return wins / K_GAMES


def validate_weights(server, weights):
    tw = tg = 0
    per_opp = {}
    for opp in OPPONENT_ROTATION:
        ow = 0
        for _ in range(VALIDATION_GAMES):
            w, _ = server.run_game(weights, opp)
            if w == 0: ow += 1
            tg += 1
        tw += ow
        per_opp[opp] = ow / VALIDATION_GAMES * 100
    return tw / tg * 100, per_opp


# ===============================================================
#  SENSITIVITY ANALYSIS
#  [FIX 1] Correct direction: high influence = SMALL mutation
# ===============================================================
def run_sensitivity_analysis(server):
    print("  Running sensitivity analysis...")
    baseline = [0.5] * DIMENSIONS
    bw = bg = 0
    for opp in OPPONENT_ROTATION:
        for _ in range(K_GAMES):
            w, _ = server.run_game(baseline, opp)
            if w == 0:   bw += 1
            elif w == -1: bw += 0.3
            bg += 1
    baseline_wr = bw / bg
    print(f"  Baseline win rate: {baseline_wr:.1%}")

    influence = []
    for w_idx in range(DIMENSIONS):
        results = []
        for val in [0.0, 1.0]:
            test = baseline[:]
            test[w_idx] = val
            wins = games = 0
            for opp in OPPONENT_ROTATION:
                for _ in range(K_GAMES):
                    w, _ = server.run_game(test, opp)
                    if w == 0:   wins += 1
                    elif w == -1: wins += 0.3
                    games += 1
            results.append(wins / games)
        influence.append(abs(results[1] - results[0]))
        print(f"  w{w_idx:2d} {WEIGHT_NAMES[w_idx]:<22s} influence={influence[-1]:.3f}")

    # [FIX 1] HIGH influence → SMALL mutation (careful steps: 0.08)
    #         LOW  influence → LARGE mutation (explore freely: 0.20)
    max_inf    = max(influence) if max(influence) > 0 else 1.0
    normalized = [0.08 + (1.0 - inf / max_inf) * 0.12 for inf in influence]

    data = {
        "influence_scores":          influence,
        "normalized_mutation_rates": normalized,
        "baseline_winrate":          baseline_wr,
        "weight_names":              WEIGHT_NAMES,
    }
    fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), SENSITIVITY_FILE)
    with open(fp, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved to {SENSITIVITY_FILE}")
    return normalized


def load_or_run_sensitivity():
    fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), SENSITIVITY_FILE)
    if os.path.exists(fp):
        print(f"  Loading {SENSITIVITY_FILE}...")
        with open(fp) as f:
            data = json.load(f)
        # Verify it was generated with the correct formula
        # Correct: values should be in [0.08, 0.20]
        rates = data["normalized_mutation_rates"]
        if min(rates) < 0.07 or max(rates) > 0.21:
            print("  WARNING: sensitivity_scores.json appears stale (wrong formula).")
            print("  Regenerating with correct formula...")
            server = GameServer(); server.start()
            rates = run_sensitivity_analysis(server)
            server.stop()
        return rates
    else:
        print(f"  {SENSITIVITY_FILE} not found — running analysis...")
        server = GameServer(); server.start()
        rates = run_sensitivity_analysis(server)
        server.stop()
        return rates


# ===============================================================
#  WILCOXON SIGNED-RANK TEST (pure Python — no scipy needed)
# ===============================================================
def wilcoxon_test(x, y):
    diffs = [xi - yi for xi, yi in zip(x, y) if xi != yi]
    if len(diffs) < 3:
        return None
    n     = len(diffs)
    abs_d = sorted(enumerate(abs(d) for d in diffs), key=lambda t: t[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and abs_d[j][1] == abs_d[j+1][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j+1):
            ranks[abs_d[k][0]] = avg_rank
        i = j + 1
    w_plus  = sum(ranks[i] for i, d in enumerate(diffs) if d > 0)
    w_minus = sum(ranks[i] for i, d in enumerate(diffs) if d < 0)
    w_stat  = min(w_plus, w_minus)
    mu_w    = n * (n + 1) / 4.0
    sig_w   = math.sqrt(n * (n + 1) * (2*n + 1) / 24.0)
    if sig_w == 0:
        return None
    z = (w_stat - mu_w) / sig_w
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2))))
    return p


# ===============================================================
#  CCE-TDCCEA CLASSES
# ===============================================================
class SubAgent:
    def __init__(self, dim, params=None):
        self.dim    = dim
        self.params = ([random.random() for _ in range(dim)] if params is None
                       else [max(0.0, min(1.0, float(x))) for x in params])
        self.H_i  = []
        self.e_i  = 0.0
        self.mu_i = 0.0

    def update_history(self, f, env):
        self.H_i.append((f, env))
        if len(self.H_i) > 7: self.H_i.pop(0)

    def compute_trajectory(self, env):
        if not self.H_i: self.e_i = 0.0; return
        md = math.sqrt(self.dim)
        ws = [max(0.0, 1.0 - euclidean_dist(env[:self.dim], pe[:self.dim]) / md)
              if len(pe) >= self.dim else 1.0 for _, pe in self.H_i]
        t  = sum(ws)
        ws = [w/t for w in ws] if t > 0 else [1.0/len(ws)] * len(ws)
        self.e_i = sum(w * self.H_i[i][0] for i, w in enumerate(ws))

    def compute_mutation(self): self.mu_i = 0.15 * (1.0 - self.e_i) + 0.03

    def mutate(self):
        return SubAgent(self.dim,
            [max(0.0, min(1.0, p + random.gauss(0, self.mu_i))) for p in self.params])


def combine_subagents(macro, micro):
    full = [0.5] * DIMENSIONS
    for i, idx in enumerate(MACRO_INDICES): full[idx] = macro.params[i]
    for i, idx in enumerate(MICRO_INDICES):  full[idx] = micro.params[i]
    return full


def subpop_survival(pop, archive):
    dim = pop[0].dim
    md  = math.sqrt(dim)
    survivors = []
    for a in sorted(pop, key=lambda a: a.e_i, reverse=True):
        if not any(1.0 - euclidean_dist(a.params, u.params) / md > DELTA_S
                   for u in survivors):
            survivors.append(a)
    final = [a for a in survivors if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
    while len(final) < 8 and len(survivors) > len(final):
        for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
            if a not in final: final.append(a); break
    for top in sorted(pop, key=lambda a: a.e_i, reverse=True)[:3]:
        ar = SubAgent(dim, top.params)
        ar.e_i, ar.H_i, ar.mu_i = top.e_i, top.H_i[:], top.mu_i
        archive.append(ar)
    archive.sort(key=lambda a: a.e_i, reverse=True)
    del archive[A_MAX:]
    pool    = final + archive
    new_pop = final[:]
    while len(new_pop) < N_POP:
        cands = random.sample(pool, min(3, len(pool)))
        new_pop.append(max(cands, key=lambda a: a.e_i).mutate())
    return new_pop, archive


# ===============================================================
#  MONOLITHIC AGENT
# ===============================================================
class Agent:
    def __init__(self, params=None, pwm=None):
        self.params = ([random.random() for _ in range(DIMENSIONS)] if params is None
                       else [max(0.0, min(1.0, float(x))) for x in params])
        self.H_i  = []
        self.e_i  = 0.0
        self.mu_i = 0.0
        self.pwm  = pwm

    def update_history(self, f, env):
        self.H_i.append((f, env))
        if len(self.H_i) > 7: self.H_i.pop(0)

    def compute_trajectory(self, env):
        if not self.H_i: self.e_i = 0.0; return
        md = math.sqrt(DIMENSIONS)
        ws = [max(0.0, 1.0 - euclidean_dist(env, pe) / md) for _, pe in self.H_i]
        t  = sum(ws)
        ws = [w/t for w in ws] if t > 0 else [1.0/len(ws)] * len(ws)
        self.e_i = sum(w * self.H_i[i][0] for i, w in enumerate(ws))

    def compute_mutation(self): self.mu_i = 0.15 * (1.0 - self.e_i) + 0.03

    def mutate(self):
        cp = []
        if self.pwm:
            # [FIX 7] HIGH influence gene → SMALL sigma (pwm[j] is 0.08 for most important)
            #         LOW  influence gene → LARGE sigma (pwm[j] is 0.20 for least important)
            for j, p in enumerate(self.params):
                sigma_j = self.pwm[j] * (1.0 - self.e_i) + 0.02
                cp.append(max(0.0, min(1.0, p + random.gauss(0, sigma_j))))
        else:
            for p in self.params:
                cp.append(max(0.0, min(1.0, p + random.gauss(0, self.mu_i))))
        return Agent(cp, self.pwm)


# ===============================================================
#  ALGORITHM 1 — CCE-TDCCEA
# ===============================================================
def run_ccea(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: CCE-TDCCEA (Our Algorithm)")
    print(f"{'='*60}")
    pop_a = [SubAgent(len(MACRO_INDICES)) for _ in range(N_POP)]
    pop_b = [SubAgent(len(MICRO_INDICES)) for _ in range(N_POP)]
    arc_a, arc_b          = [], []
    top_a, top_b          = pop_a[:TOP_PARTNERS], pop_b[:TOP_PARTNERS]
    best_a, best_b        = pop_a[0], pop_b[0]
    t0                    = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opp = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
        ca  = [sum(a.params[j] for a in pop_a) / len(pop_a) for j in range(len(MACRO_INDICES))]
        cb  = [sum(b.params[j] for b in pop_b) / len(pop_b) for j in range(len(MICRO_INDICES))]
        env = ca + cb

        for a in pop_a:
            fs = [evaluate_weights(server, combine_subagents(a, pb), opp) for pb in top_b]
            a.update_history(sum(fs)/len(fs), env); a.compute_trajectory(env); a.compute_mutation()

        for b in pop_b:
            fs = [evaluate_weights(server, combine_subagents(pa, b), opp) for pa in top_a]
            b.update_history(sum(fs)/len(fs), env); b.compute_trajectory(env); b.compute_mutation()

        best_a = max(pop_a, key=lambda a: a.e_i)
        best_b = max(pop_b, key=lambda a: a.e_i)
        top_a  = sorted(pop_a, key=lambda a: a.e_i, reverse=True)[:TOP_PARTNERS]
        top_b  = sorted(pop_b, key=lambda a: a.e_i, reverse=True)[:TOP_PARTNERS]
        cmb    = evaluate_weights(server, combine_subagents(best_a, best_b), opp)
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | A={best_a.e_i:.3f} B={best_b.e_i:.3f} Cmb={cmb:.3f}")
        pop_a, arc_a = subpop_survival(pop_a, arc_a)
        pop_b, arc_b = subpop_survival(pop_b, arc_b)

    return combine_subagents(best_a, best_b), time.time() - t0


# ===============================================================
#  ALGORITHM 2 — STANDARD CCEA (no trajectory — ablation baseline)
# ===============================================================
def run_standard_ccea(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: Standard CCEA (no trajectory memory)")
    print(f"{'='*60}")
    MUT = 0.15
    pm  = [[random.random() for _ in range(len(MACRO_INDICES))] for _ in range(N_POP)]
    pi  = [[random.random() for _ in range(len(MICRO_INDICES))] for _ in range(N_POP)]
    fm  = [0.0] * N_POP
    fi  = [0.0] * N_POP
    t0  = time.time()

    def mk(mac, mic):
        f = [0.5] * DIMENSIONS
        for k, idx in enumerate(MACRO_INDICES): f[idx] = mac[k]
        for k, idx in enumerate(MICRO_INDICES):  f[idx] = mic[k]
        return f

    def breed(pop, fits):
        # [FIX 6] renamed 'np' -> 'new_pop' to avoid numpy convention confusion
        bi      = max(range(len(pop)), key=lambda i: fits[i])
        new_pop = [pop[bi][:]]
        while len(new_pop) < N_POP:
            t    = random.sample(range(len(pop)), min(3, len(pop)))
            pidx = max(t, key=lambda i: fits[i])
            new_pop.append([max(0.0, min(1.0, x + random.gauss(0, MUT))) for x in pop[pidx]])
        return new_pop

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opp  = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
        tmi  = sorted(range(N_POP), key=lambda i: fi[i], reverse=True)[:TOP_PARTNERS]
        tma  = sorted(range(N_POP), key=lambda i: fm[i], reverse=True)[:TOP_PARTNERS]
        for i in range(N_POP):
            fs = [evaluate_weights(server, mk(pm[i], pi[j]), opp) for j in tmi]
            fm[i] = sum(fs) / len(fs)
        for i in range(N_POP):
            fs = [evaluate_weights(server, mk(pm[j], pi[i]), opp) for j in tma]
            fi[i] = sum(fs) / len(fs)
        bm  = max(range(N_POP), key=lambda i: fm[i])
        bi2 = max(range(N_POP), key=lambda i: fi[i])
        bf  = evaluate_weights(server, mk(pm[bm], pi[bi2]), opp)
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | Best={bf:.3f}")
        pm = breed(pm, fm)
        pi = breed(pi, fi)

    bm  = max(range(N_POP), key=lambda i: fm[i])
    bi2 = max(range(N_POP), key=lambda i: fi[i])
    return mk(pm[bm], pi[bi2]), time.time() - t0


# ===============================================================
#  ALGORITHM 3 — CMA-ES
# ===============================================================
def run_cmaes(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: CMA-ES")
    print(f"{'='*60}")
    t0 = time.time(); best_p = [0.5] * DIMENSIONS; best_f = 0.0
    try:
        import cma
        es  = cma.CMAEvolutionStrategy([0.5] * DIMENSIONS, 0.3,
              {'maxiter': TOTAL_GENERATIONS, 'popsize': N_POP,
               'bounds': [0, 1], 'verbose': -9, 'seed': random.randint(0, 99999)})
        gen = 0
        while not es.stop() and gen < TOTAL_GENERATIONS:
            opp  = OPPONENT_ROTATION[gen % len(OPPONENT_ROTATION)]
            sols = es.ask(); fits = []
            for s in sols:
                sc = [max(0.0, min(1.0, x)) for x in s]
                f  = evaluate_weights(server, sc, opp); fits.append(f)
                if f > best_f: best_f = f; best_p = sc[:]
            es.tell(sols, [-f for f in fits])
            print(f"  Gen {gen+1:2d}/{TOTAL_GENERATIONS} | {opp:25s} | Best={best_f:.3f}"); gen += 1
    except ImportError:
        print("  WARNING: 'cma' not installed (pip install cma). Using fallback ES.")
        pop = [[random.random() for _ in range(DIMENSIONS)] for _ in range(N_POP)]
        sigma = 0.3
        for gen in range(1, TOTAL_GENERATIONS + 1):
            opp   = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
            fits  = [0.0] * N_POP
            for i, s in enumerate(pop):
                fits[i] = evaluate_weights(server, s, opp)
                if fits[i] > best_f: best_f = fits[i]; best_p = s[:]
            ranked  = sorted(range(N_POP), key=lambda i: fits[i], reverse=True)
            parents = [pop[i] for i in ranked[:N_POP // 2]]
            pop     = parents[:]
            while len(pop) < N_POP:
                p = random.choice(parents)
                pop.append([max(0.0, min(1.0, x + random.gauss(0, sigma))) for x in p])
            sigma = max(0.05, sigma * 0.98)
            print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | Best={best_f:.3f}")
    return best_p, time.time() - t0


# ===============================================================
#  ALGORITHM 4 — DE/rand/1/bin
# ===============================================================
def run_de(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: DE/rand/1/bin (F=0.8, CR=0.9)")
    print(f"{'='*60}")
    F = 0.8; CR = 0.9; t0 = time.time()
    pop = [[random.random() for _ in range(DIMENSIONS)] for _ in range(N_POP)]
    fit = [0.0] * N_POP
    opp = OPPONENT_ROTATION[0]
    for i in range(N_POP): fit[i] = evaluate_weights(server, pop[i], opp)
    for gen in range(1, TOTAL_GENERATIONS + 1):
        opp = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
        for i in range(N_POP):
            a, b, c = random.sample([j for j in range(N_POP) if j != i], 3)
            mutant  = [max(0.0, min(1.0, pop[a][j] + F * (pop[b][j] - pop[c][j])))
                       for j in range(DIMENSIONS)]
            jr    = random.randint(0, DIMENSIONS - 1)
            trial = [mutant[j] if (random.random() < CR or j == jr) else pop[i][j]
                     for j in range(DIMENSIONS)]
            tf = evaluate_weights(server, trial, opp)
            if tf >= fit[i]: pop[i] = trial; fit[i] = tf
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | Best={max(fit):.3f}")
    bi = max(range(N_POP), key=lambda i: fit[i])
    return pop[bi], time.time() - t0


# ===============================================================
#  ALGORITHM 5 — PSO
# ===============================================================
def run_pso(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: PSO (W=0.7, C1=C2=1.5)")
    print(f"{'='*60}")
    W = 0.7; C1 = 1.5; C2 = 1.5; t0 = time.time()
    pos = [[random.random() for _ in range(DIMENSIONS)] for _ in range(N_POP)]
    vel = [[random.uniform(-0.1, 0.1) for _ in range(DIMENSIONS)] for _ in range(N_POP)]
    pb  = [p[:] for p in pos]; pf = [0.0] * N_POP; gb = pos[0][:]; gf = 0.0
    for gen in range(1, TOTAL_GENERATIONS + 1):
        opp = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
        for i in range(N_POP):
            f = evaluate_weights(server, pos[i], opp)
            if f > pf[i]: pf[i] = f; pb[i] = pos[i][:]
            if f > gf:    gf = f;    gb = pos[i][:]
        for i in range(N_POP):
            for j in range(DIMENSIONS):
                vel[i][j] = (W * vel[i][j]
                             + C1 * random.random() * (pb[i][j] - pos[i][j])
                             + C2 * random.random() * (gb[j]    - pos[i][j]))
                pos[i][j] = max(0.0, min(1.0, pos[i][j] + vel[i][j]))
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | Best={gf:.3f}")
    return gb, time.time() - t0


# ===============================================================
#  ALGORITHMS 6/7 — TD-CCEA / SA-TDCCEA
# ===============================================================
def evolve_core(server, label, pwm=None):
    print(f"\n{'='*60}")
    print(f"  RUNNING: {label}")
    print(f"{'='*60}")
    population = [Agent(pwm=pwm) for _ in range(N_POP)]
    archive    = []
    t0         = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opp      = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
        centroid = [sum(a.params[j] for a in population) / len(population)
                    for j in range(DIMENSIONS)]
        for agent in population:
            f_i = evaluate_weights(server, agent.params, opp)
            agent.update_history(f_i, centroid)
            agent.compute_trajectory(centroid)
            agent.compute_mutation()
        best = max(population, key=lambda a: a.e_i)
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | e_i={best.e_i:.3f}")
        md        = math.sqrt(DIMENSIONS)
        survivors = []
        for a in sorted(population, key=lambda a: a.e_i, reverse=True):
            if not any(1.0 - euclidean_dist(a.params, u.params) / md > DELTA_S
                       for u in survivors):
                survivors.append(a)
        final = [a for a in survivors if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
        while len(final) < 10 and len(survivors) > len(final):
            for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
                if a not in final: final.append(a); break
        for top in sorted(population, key=lambda a: a.e_i, reverse=True)[:3]:
            ar = Agent(top.params, pwm)
            ar.e_i, ar.H_i, ar.mu_i = top.e_i, top.H_i[:], top.mu_i
            archive.append(ar)
        archive.sort(key=lambda a: a.e_i, reverse=True)
        archive    = archive[:A_MAX]
        pool       = final + archive
        population = final[:]
        while len(population) < N_POP:
            cands = random.sample(pool, min(3, len(pool)))
            population.append(max(cands, key=lambda a: a.e_i).mutate())

    best_all = max(archive, key=lambda a: a.e_i) if archive else max(population, key=lambda a: a.e_i)
    return best_all.params, time.time() - t0


# ===============================================================
#  ALGORITHM 8 — HILL CLIMBING
# ===============================================================
def run_hill_climbing(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: Hill Climbing")
    print(f"{'='*60}")
    current = [random.random() for _ in range(DIMENSIONS)]
    cfit    = 0.0
    t0      = time.time()
    for gen in range(1, TOTAL_GENERATIONS + 1):
        opp  = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
        cfit = evaluate_weights(server, current, opp)
        bc, bf = current[:], cfit
        for _ in range(N_POP - 1):
            c = [max(0.0, min(1.0, p + random.gauss(0, 0.15))) for p in current]
            f = evaluate_weights(server, c, opp)
            if f > bf: bc, bf = c[:], f
        if bf >= cfit: current, cfit = bc[:], bf
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | Best={cfit:.3f}")
    return current, time.time() - t0


# ===============================================================
#  MAIN
# ===============================================================
def main():
    print("\n" + "#" * 70)
    print("  BENCHMARK VS BOTS — 8 Algorithms vs 7 microRTS Bots")
    print("#" * 70)
    print(f"  Pop={N_POP} | Gens={TOTAL_GENERATIONS} | "
          f"K={K_GAMES} | Validation={VALIDATION_GAMES}/bot\n")

    # [FIX 1] Load or regenerate sensitivity scores with correct formula
    sa_mut = load_or_run_sensitivity()
    print()

    algos = [
        ("CCE-TDCCEA",    "ccea"),
        ("Standard CCEA", "std_ccea"),
        ("CMA-ES",        "cmaes"),
        ("DE/rand/1/bin", "de"),
        ("PSO",           "pso"),
        ("TD-CCEA",       "centroid"),
        ("SA-TDCCEA",     "sa"),
        ("Hill Climbing", "hc"),
    ]

    results = []
    for name, key in algos:
        server = GameServer(); server.start()
        print(f"  [OK] Server started — {name}")

        if   key == "ccea":     best, elapsed = run_ccea(server)
        elif key == "std_ccea": best, elapsed = run_standard_ccea(server)
        elif key == "cmaes":    best, elapsed = run_cmaes(server)
        elif key == "de":       best, elapsed = run_de(server)
        elif key == "pso":      best, elapsed = run_pso(server)
        elif key == "centroid": best, elapsed = evolve_core(server, "TD-CCEA")
        elif key == "sa":       best, elapsed = evolve_core(server, "SA-TDCCEA", pwm=sa_mut)
        else:                   best, elapsed = run_hill_climbing(server)

        print(f"\n  Validating {name} against all 7 bots...")
        wr, per_opp = validate_weights(server, best)
        server.stop()

        print(f"  {name}: {wr:.1f}% overall in {elapsed:.1f}s")
        results.append({"name": name, "key": key, "time": elapsed,
                        "wr": wr, "per_opp": per_opp, "weights": best})

    # ── RANKING TABLE ────────────────────────────────────────────
    ranked = sorted(results, key=lambda x: x["wr"], reverse=True)
    medals = ["1st","2nd","3rd","4th","5th","6th","7th","8th"]

    print(f"\n{'='*70}")
    print(f"  FINAL RANKING — Win Rate vs 7 Bots")
    print(f"{'='*70}")
    print(f"  {'Rank':<6s} {'Algorithm':<18s} {'Win Rate':>10s} {'Time':>8s}")
    print(f"  {'_'*6} {'_'*18} {'_'*10} {'_'*8}")
    for i, r in enumerate(ranked):
        tag = "  ◄ OUR ALGORITHM" if r["key"] == "ccea" else ""
        print(f"  {medals[i]:<6s} {r['name']:<18s} {r['wr']:>9.1f}%  {r['time']:>7.1f}s{tag}")

    # ── PER-BOT TABLE with best-marker (*) [FIX 5] ──────────────
    print(f"\n{'='*70}")
    print(f"  WIN RATE PER BOT (%)")
    print(f"{'='*70}")
    header = f"  {'Bot':<26s}"
    for r in ranked: header += f" {r['name'][:9]:>10s}"
    print(header)
    print("  " + "-" * (26 + 11 * len(ranked)))
    for opp in OPPONENT_ROTATION:
        row   = f"  {opp:<26s}"
        rates = [r["per_opp"].get(opp, 0) for r in ranked]
        mx    = max(rates)
        for r in ranked:
            w  = r["per_opp"].get(opp, 0)
            mk = "*" if w == mx and rates.count(mx) == 1 else " "
            row += f" {w:>9.0f}%{mk}"
        print(row)
    print("  " + "-" * (26 + 11 * len(ranked)))
    row = f"  {'OVERALL':<26s}"
    for r in ranked: row += f" {r['wr']:>10.1f}% "
    print(row)

    # ── WILCOXON TEST [FIX 2] ────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  STATISTICAL SIGNIFICANCE — Wilcoxon Signed-Rank Test")
    print(f"  (CCE-TDCCEA vs each baseline, per-bot win rates as samples)")
    print(f"{'='*70}")
    cce = next((r for r in results if r["key"] == "ccea"), None)
    if cce:
        cce_rates = [cce["per_opp"].get(o, 0) for o in OPPONENT_ROTATION]
        print(f"\n  {'Baseline':<20s}  {'p-value':>9s}  {'Significant?':>13s}  Direction")
        print(f"  {'_'*20}  {'_'*9}  {'_'*13}  {'_'*22}")
        for r in results:
            if r["key"] == "ccea": continue
            other = [r["per_opp"].get(o, 0) for o in OPPONENT_ROTATION]
            p     = wilcoxon_test(cce_rates, other)
            if p is None:
                print(f"  {r['name']:<20s}  {'N/A':>9s}  {'N/A':>13s}  —")
            else:
                sig  = "YES (p<0.05)" if p < 0.05 else "NO  (p>=0.05)"
                dirn = ("CCE-TDCCEA BETTER" if sum(cce_rates) > sum(other)
                        else "BASELINE BETTER")
                print(f"  {r['name']:<20s}  {p:>9.4f}  {sig:>13s}  {dirn}")

    # ── ABLATION ANALYSIS [FIX 3] ────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ABLATION ANALYSIS")
    print(f"{'='*70}")
    cce   = next((r for r in results if r["key"] == "ccea"),     None)
    sccea = next((r for r in results if r["key"] == "std_ccea"), None)
    cent  = next((r for r in results if r["key"] == "centroid"), None)
    if cce and sccea:
        d = cce["wr"] - sccea["wr"]
        print(f"\n  Trajectory memory contribution:")
        print(f"    CCE-TDCCEA ({cce['wr']:.1f}%) vs Std-CCEA ({sccea['wr']:.1f}%)")
        print(f"    Result: {'Trajectory HELPS (+' if d >= 0 else 'Trajectory HURTS ('}{abs(d):.1f}%)")
    if cce and cent:
        d = cce["wr"] - cent["wr"]
        print(f"\n  Cooperative decomposition contribution:")
        print(f"    CCE-TDCCEA ({cce['wr']:.1f}%) vs TD-CCEA ({cent['wr']:.1f}%)")
        print(f"    Result: {'Decomposition HELPS (+' if d >= 0 else 'Decomposition HURTS ('}{abs(d):.1f}%)")

    # ── SAVE JSON [FIX 4] ────────────────────────────────────────
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "benchmark_vs_bots_results.json")
    with open(out, "w") as f:
        json.dump([{"name": r["name"], "key": r["key"],
                    "overall_winrate": r["wr"], "per_opponent": r["per_opp"],
                    "time": r["time"]} for r in results], f, indent=2)
    print(f"\n  Results saved → {out}")
    print(f"\n{'='*70}")
    print(f"  BENCHMARK COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()