"""
BENCHMARK_VS_BOTS_WITH_CONVERGENCE.py
======================================
10 algorithms evolve against 7 microRTS bots, then validated against all bots.

CHANGES OVER BENCHMARK_VS_BOTS.py:
  [CONV 1] Added per-generation win-rate logging to 4 key algorithms:
               CCE-TDCCEA, Standard CCEA, DE/rand/1/bin, TD-CCEA
  [CONV 2] Convergence histories accumulated across all 25/30 runs
           via CONVERGENCE_DATA dict (keyed by algorithm name)
  [CONV 3] plot_convergence_curves() called after all runs complete:
               - One combined figure, one line per algorithm
               - Shaded band = mean ± 1 std-dev across runs
               - Saved as convergence_curves.png (publication quality, 300 dpi)
  [CONV 4] Raw per-run curves also saved to convergence_raw.json
           so you can regenerate the plot at any time without re-running

WHY ONE COMBINED GRAPH:
  A single figure lets the reader compare convergence SPEED (which algorithm
  climbs fastest early) and STABILITY (how wide the shaded bands are) in one
  glance. Separate figures make that comparison impossible. This is the
  standard presentation in EA/co-evo papers (e.g. Table V style follow-up).

ALGORITHMS:
  1. CCE-TDCCEA      — Original novel algorithm (UNCHANGED)
  2. Standard CCEA   — Same split, no trajectory (ablation baseline)
  3. CMA-ES          — Industry-standard continuous optimizer
  4. DE/rand/1/bin   — Differential Evolution
  5. PSO             — Particle Swarm Optimization
  6. TD-CCEA         — Monolithic trajectory baseline
  7. SA-TDCCEA       — Sensitivity-adaptive variant
  8. Hill Climbing   — Greedy local search baseline
  9. GESY            — Seeded Genetic Programming
 10. SA-PDCoEA       — Self-Adaptive Pairwise Dominance CoEvolutionary Algorithm

INSTALL: pip install cma matplotlib openpyxl
"""

import math, random, time, subprocess, os, json, copy
try:
    import openpyxl
    _OPENPYXL_AVAILABLE = True
except ImportError:
    _OPENPYXL_AVAILABLE = False
    print("[WARNING] openpyxl not installed — Excel export disabled. Run: pip install openpyxl")

try:
    import matplotlib
    matplotlib.use("Agg")          # headless — no display needed on server
    import matplotlib.pyplot as plt
    import numpy as np
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    print("[WARNING] matplotlib/numpy not installed — convergence plot disabled.")
    print("          Run: pip install matplotlib numpy")

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

os.environ["PYTHONIOENCODING"] = "utf-8"

try:
    from SA_PDCoEA import run_sa_pdcoea as _run_sa_pdcoea
    _SA_PDCOEA_AVAILABLE = True
except ImportError:
    _SA_PDCOEA_AVAILABLE = False
    print("[WARNING] SA_PDCoEA.py not found — Algorithm 10 will be skipped.")

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
TOTAL_GENERATIONS = 5
VALIDATION_GAMES  = 5
A_MAX             = 15
DELTA_S           = 0.85
DELTA_F           = 0.20
TOP_PARTNERS      = 3
SENSITIVITY_FILE  = "sensitivity_scores.json"

# ---------------------------------------------------------------
# [CONV 1]  CONVERGENCE DATA STORE
# ---------------------------------------------------------------
# Structure:
#   CONVERGENCE_DATA[algo_name] = list of runs,
#   where each run = list of 30 floats (best win-rate at that generation,
#   measured against the TRAINING opponent for that generation).
#
# NOTE: these are TRAINING-time win rates (K_GAMES per gen), not the
# final validation win rates.  They show how fast the algorithm learns,
# which is exactly what a convergence curve should show.
# ---------------------------------------------------------------
CONVERGENCE_DATA: dict[str, list[list[float]]] = {
    "CCE-TDCCEA":    [],
    "Standard CCEA": [],
    "DE/rand/1/bin": [],
    "TD-CCEA":       [],
    "CMA-ES":        [],
    "SA-TDCCEA":     [],
    "Hill Climbing": [],
    "GESY":          [],
    "SA-PDCoEA":     [],
}

# Algorithms whose curves we track (others omitted to keep the graph clean)
CURVE_ALGORITHMS = set(CONVERGENCE_DATA.keys())


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
        if w == 0:    wins += 1
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
#  SENSITIVITY ANALYSIS  [FIX 1]
# ===============================================================
def run_sensitivity_analysis(server):
    print("  Running sensitivity analysis...")
    baseline = [0.5] * DIMENSIONS
    bw = bg = 0
    for opp in OPPONENT_ROTATION:
        for _ in range(K_GAMES):
            w, _ = server.run_game(baseline, opp)
            if w == 0:    bw += 1
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
                    if w == 0:    wins += 1
                    elif w == -1: wins += 0.3
                    games += 1
            results.append(wins / games)
        influence.append(abs(results[1] - results[0]))
        print(f"  w{w_idx:2d} {WEIGHT_NAMES[w_idx]:<22s} influence={influence[-1]:.3f}")

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
        rates = data["normalized_mutation_rates"]
        if min(rates) < 0.07 or max(rates) > 0.21:
            print("  WARNING: sensitivity_scores.json appears stale. Regenerating...")
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
#  WILCOXON SIGNED-RANK TEST  [FIX 2]
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
#  CCE-TDCCEA CLASSES  (UNCHANGED)
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
    dim  = pop[0].dim
    md   = math.sqrt(dim)
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
#  MONOLITHIC AGENT  (UNCHANGED)
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
            for j, p in enumerate(self.params):
                sigma_j = self.pwm[j] * (1.0 - self.e_i) + 0.02
                cp.append(max(0.0, min(1.0, p + random.gauss(0, sigma_j))))
        else:
            for p in self.params:
                cp.append(max(0.0, min(1.0, p + random.gauss(0, self.mu_i))))
        return Agent(cp, self.pwm)


# ===============================================================
#  ALGORITHM 1 — CCE-TDCCEA
#  [CONV 2] Returns gen_curve: list of 30 floats (best combined
#           score per generation, as a fraction 0-1)
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
    gen_curve             = []          # [CONV 2] one entry per generation

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

        # [CONV 2] Record best combined fitness this generation
        cmb = evaluate_weights(server, combine_subagents(best_a, best_b), opp)
        gen_curve.append(cmb * 100.0)   # store as % to match validation scale
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | A={best_a.e_i:.3f} B={best_b.e_i:.3f} Cmb={cmb:.3f}")

        pop_a, arc_a = subpop_survival(pop_a, arc_a)
        pop_b, arc_b = subpop_survival(pop_b, arc_b)

    # [CONV 2] Append this run's curve to the global store
    CONVERGENCE_DATA["CCE-TDCCEA"].append(gen_curve)

    return combine_subagents(best_a, best_b), time.time() - t0


# ===============================================================
#  ALGORITHM 2 — STANDARD CCEA
#  [CONV 2] Same logging added
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
    gen_curve = []      # [CONV 2]

    def mk(mac, mic):
        f = [0.5] * DIMENSIONS
        for k, idx in enumerate(MACRO_INDICES): f[idx] = mac[k]
        for k, idx in enumerate(MICRO_INDICES):  f[idx] = mic[k]
        return f

    def breed(pop, fits):
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

        # [CONV 2] Record best combined fitness this generation
        bf = evaluate_weights(server, mk(pm[bm], pi[bi2]), opp)
        gen_curve.append(bf * 100.0)
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | Best={bf:.3f}")

        pm = breed(pm, fm)
        pi = breed(pi, fi)

    # [CONV 2]
    CONVERGENCE_DATA["Standard CCEA"].append(gen_curve)

    bm  = max(range(N_POP), key=lambda i: fm[i])
    bi2 = max(range(N_POP), key=lambda i: fi[i])
    return mk(pm[bm], pi[bi2]), time.time() - t0


# ===============================================================
#  ALGORITHM 3 — CMA-ES  (UNCHANGED — no curve needed)
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
        print("  WARNING: 'cma' not installed. Using fallback ES.")
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
#  [CONV 2] Curve logging added
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
    gen_curve = []      # [CONV 2]

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

        # [CONV 2] best individual fitness this generation
        best_f = max(fit)
        gen_curve.append(best_f * 100.0)
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | Best={best_f:.3f}")

    # [CONV 2]
    CONVERGENCE_DATA["DE/rand/1/bin"].append(gen_curve)

    bi = max(range(N_POP), key=lambda i: fit[i])
    return pop[bi], time.time() - t0


# ===============================================================
#  ALGORITHM 5 — PSO  (UNCHANGED — no curve needed)
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
#  [CONV 2] Curve logging added for TD-CCEA (key="centroid") only
# ===============================================================
def evolve_core(server, label, pwm=None):
    print(f"\n{'='*60}")
    print(f"  RUNNING: {label}")
    print(f"{'='*60}")
    population = [Agent(pwm=pwm) for _ in range(N_POP)]
    archive    = []
    t0         = time.time()
    gen_curve  = []     # [CONV 2]

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opp      = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
        centroid = [sum(a.params[j] for a in population) / len(population)
                    for j in range(DIMENSIONS)]
        for agent in population:
            f_i = evaluate_weights(server, agent.params, opp)
            agent.update_history(f_i, centroid)
            agent.compute_trajectory(centroid)
            agent.compute_mutation()
        best    = max(population, key=lambda a: a.e_i)
        best_f  = max(a.e_i for a in population)

        # [CONV 2] Only log for TD-CCEA (not SA-TDCCEA — keeps graph clean)
        if label == "TD-CCEA":
            gen_curve.append(best_f * 100.0)

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

    # [CONV 2]
    if label == "TD-CCEA":
        CONVERGENCE_DATA["TD-CCEA"].append(gen_curve)

    best_all = max(archive, key=lambda a: a.e_i) if archive else max(population, key=lambda a: a.e_i)
    return best_all.params, time.time() - t0


# ===============================================================
#  ALGORITHM 8 — HILL CLIMBING  (UNCHANGED)
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
#  ALGORITHM 9 — GESY  (UNCHANGED — no curve needed)
# ===============================================================

GESY_POP_SIZE       = 20
GESY_ELITISM        = 1
GESY_NEWCOMERS      = 2
GESY_CROSSOVER_RATE = 0.75
GESY_MUTATION_RATE  = 0.25
GESY_MAX_DEPTH      = 6
GESY_TOURNAMENT_K   = 3

ALL_GESY_ACTIONS = [
    "attack", "retreat", "build_melee", "build_ranged", "build_siege",
    "fortify", "rush", "hold_position", "harass", "gather_resource", "no_op",
]

ALL_GESY_CONDITIONS = [
    ("aggression_high",    lambda w: w["aggression"]       > 0.6),
    ("aggression_low",     lambda w: w["aggression"]       < 0.4),
    ("economy_focus",      lambda w: w["economy_pressure"] > 0.5),
    ("ranged_preferred",   lambda w: w["ranged_preference"]> 0.5),
    ("heavy_preferred",    lambda w: w["heavy_preference"] > 0.5),
    ("light_preferred",    lambda w: w["light_preference"] > 0.5),
    ("early_barracks",     lambda w: w["barracks_timing"]  < 0.4),
    ("defend_tight",       lambda w: w["defend_radius"]    < 0.4),
    ("counter_rush_on",    lambda w: w["counter_rush"]     > 0.5),
    ("worker_heavy",       lambda w: w["worker_ratio"]     > 0.6),
    ("scout_active",       lambda w: w["scout_priority"]   > 0.5),
    ("focus_fire_on",      lambda w: w["focus_fire"]       > 0.5),
    ("group_attack_on",    lambda w: w["group_attack"]     > 0.5),
    ("tech_switch_ready",  lambda w: w["tech_switch"]      > 0.5),
]


class GesyNodeType(Enum):
    ACTION    = auto()
    CONDITION = auto()


class GesyNode(ABC):
    node_type: GesyNodeType

    def __init__(self):
        self._children = []

    @abstractmethod
    def evaluate(self, w: dict) -> Any: ...

    @abstractmethod
    def to_str(self, indent: int = 0) -> str: ...

    @property
    def children(self):
        return self._children

    def node_count(self) -> int:
        return 1 + sum(c.node_count() for c in self._children)

    def depth(self) -> int:
        if not self._children:
            return 0
        return 1 + max(c.depth() for c in self._children)

    def collect(self, ntype: GesyNodeType) -> list:
        out = []
        queue = [self]
        while queue:
            n = queue.pop(0)
            if n.node_type == ntype:
                out.append(n)
            queue.extend(n._children)
        return out

    def clone(self):
        return copy.deepcopy(self)


class GesyLeafAction(GesyNode):
    node_type = GesyNodeType.ACTION

    def __init__(self, command: str):
        super().__init__()
        self.command = command

    def evaluate(self, w: dict) -> str:
        return self.command

    def to_str(self, indent: int = 0) -> str:
        return "  " * indent + f"DO({self.command})"


class GesyCondition(GesyNode):
    node_type = GesyNodeType.CONDITION

    def __init__(self, name: str, predicate):
        super().__init__()
        self.name      = name
        self.predicate = predicate

    def evaluate(self, w: dict) -> bool:
        return self.predicate(w)

    def to_str(self, indent: int = 0) -> str:
        return "  " * indent + f"IF({self.name})"


class GesyAnd(GesyNode):
    node_type = GesyNodeType.CONDITION

    def __init__(self, left: GesyNode, right: GesyNode):
        super().__init__()
        self._children = [left, right]

    def evaluate(self, w: dict) -> bool:
        return self._children[0].evaluate(w) and self._children[1].evaluate(w)

    def to_str(self, indent: int = 0) -> str:
        p = "  " * indent
        return (p + "AND(\n" +
                self._children[0].to_str(indent+1) + "\n" +
                self._children[1].to_str(indent+1) + "\n" + p + ")")


class GesyOr(GesyNode):
    node_type = GesyNodeType.CONDITION

    def __init__(self, left: GesyNode, right: GesyNode):
        super().__init__()
        self._children = [left, right]

    def evaluate(self, w: dict) -> bool:
        return self._children[0].evaluate(w) or self._children[1].evaluate(w)

    def to_str(self, indent: int = 0) -> str:
        p = "  " * indent
        return (p + "OR(\n" +
                self._children[0].to_str(indent+1) + "\n" +
                self._children[1].to_str(indent+1) + "\n" + p + ")")


class GesyNot(GesyNode):
    node_type = GesyNodeType.CONDITION

    def __init__(self, child: GesyNode):
        super().__init__()
        self._children = [child]

    def evaluate(self, w: dict) -> bool:
        return not self._children[0].evaluate(w)

    def to_str(self, indent: int = 0) -> str:
        p = "  " * indent
        return p + "NOT(\n" + self._children[0].to_str(indent+1) + "\n" + p + ")"


class GesyIfThenElse(GesyNode):
    node_type = GesyNodeType.ACTION

    def __init__(self, cond: GesyNode, then_br: GesyNode, else_br: GesyNode):
        super().__init__()
        self._children = [cond, then_br, else_br]

    @property
    def cond(self)    -> GesyNode: return self._children[0]
    @property
    def then_br(self) -> GesyNode: return self._children[1]
    @property
    def else_br(self) -> GesyNode: return self._children[2]

    def evaluate(self, w: dict) -> Optional[str]:
        if self.cond.evaluate(w):
            return self.then_br.evaluate(w)
        return self.else_br.evaluate(w)

    def to_str(self, indent: int = 0) -> str:
        p = "  " * indent
        return "\n".join([
            p + "IF", self.cond.to_str(indent+1),
            p + "THEN", self.then_br.to_str(indent+1),
            p + "ELSE", self.else_br.to_str(indent+1),
        ])


class GesySequence(GesyNode):
    node_type = GesyNodeType.ACTION

    def __init__(self, actions: list):
        super().__init__()
        self._children = list(actions)

    def evaluate(self, w: dict) -> Optional[str]:
        for child in self._children:
            r = child.evaluate(w)
            if r is not None and r != "no_op":
                return r
        return None

    def to_str(self, indent: int = 0) -> str:
        p = "  " * indent
        lines = [p + "SEQ("]
        for c in self._children:
            lines.append(c.to_str(indent+1))
        lines.append(p + ")")
        return "\n".join(lines)


@dataclass
class GesyScript:
    name:       str
    root:       GesyNode
    fitness:    float = 0.0
    wins:       int   = 0
    losses:     int   = 0
    draws:      int   = 0
    generation: int   = 0

    def node_count(self) -> int: return self.root.node_count()
    def depth(self)      -> int: return self.root.depth()

    def run(self, w: dict) -> Optional[str]:
        try:
            return self.root.evaluate(w)
        except Exception:
            return None

    def clone(self, new_name: str = "") -> "GesyScript":
        return GesyScript(
            name       = new_name or self.name + "_c",
            root       = self.root.clone(),
            generation = self.generation,
        )


def _rand_cond_leaf() -> GesyCondition:
    name, pred = random.choice(ALL_GESY_CONDITIONS)
    return GesyCondition(name, pred)


def _rand_cond_tree(max_depth: int = 2) -> GesyNode:
    if max_depth <= 0 or random.random() < 0.45:
        return _rand_cond_leaf()
    r = random.random()
    if r < 0.40:
        return GesyAnd(_rand_cond_tree(max_depth-1), _rand_cond_tree(max_depth-1))
    elif r < 0.70:
        return GesyOr(_rand_cond_tree(max_depth-1), _rand_cond_tree(max_depth-1))
    else:
        return GesyNot(_rand_cond_tree(max_depth-1))


def _rand_action_tree(max_depth: int = 3) -> GesyNode:
    if max_depth <= 0 or random.random() < 0.35:
        return GesyLeafAction(random.choice(ALL_GESY_ACTIONS))
    if random.random() < 0.70:
        return GesyIfThenElse(
            _rand_cond_tree(max_depth-1),
            _rand_action_tree(max_depth-1),
            _rand_action_tree(max_depth-1),
        )
    n = random.randint(2, 3)
    return GesySequence([_rand_action_tree(max_depth-1) for _ in range(n)])


def _c(name: str, pred) -> GesyCondition:
    return GesyCondition(name, pred)


def _build_rush_ast() -> GesyNode:
    return GesyIfThenElse(
        _c("aggression_high", lambda w: w["aggression"] > 0.6),
        GesyLeafAction("rush"),
        GesyIfThenElse(
            _c("early_barracks", lambda w: w["barracks_timing"] < 0.4),
            GesyLeafAction("build_melee"),
            GesyLeafAction("attack"),
        ),
    )


def _build_defensive_ast() -> GesyNode:
    return GesyIfThenElse(
        _c("defend_tight", lambda w: w["defend_radius"] < 0.4),
        GesyIfThenElse(
            _c("counter_rush_on", lambda w: w["counter_rush"] > 0.5),
            GesyLeafAction("fortify"),
            GesyLeafAction("hold_position"),
        ),
        GesyIfThenElse(
            _c("aggression_high", lambda w: w["aggression"] > 0.6),
            GesyLeafAction("attack"),
            GesyLeafAction("hold_position"),
        ),
    )


def _build_ranged_ast() -> GesyNode:
    return GesyIfThenElse(
        _c("economy_focus", lambda w: w["economy_pressure"] > 0.5),
        GesyLeafAction("gather_resource"),
        GesyIfThenElse(
            _c("ranged_preferred", lambda w: w["ranged_preference"] > 0.5),
            GesySequence([GesyLeafAction("build_ranged"), GesyLeafAction("harass")]),
            GesyLeafAction("attack"),
        ),
    )


def _gesy_initial_population(pop_size: int) -> list:
    seeds = [
        GesyScript("Rush_Seed",      _build_rush_ast(),      generation=0),
        GesyScript("Defensive_Seed", _build_defensive_ast(), generation=0),
        GesyScript("Ranged_Seed",    _build_ranged_ast(),    generation=0),
    ]
    randoms = [
        GesyScript(f"Random_Init_{i:03d}", _rand_action_tree(), generation=0)
        for i in range(max(0, pop_size - len(seeds)))
    ]
    return seeds + randoms


_ACTION_TO_WEIGHT_MAP = {
    "rush":          ("aggression",        0.90, 0.10),
    "attack":        ("attack_threshold",  0.75, 0.25),
    "retreat":       ("defense_trigger",   0.80, 0.20),
    "build_melee":   ("light_preference",  0.80, 0.20),
    "build_ranged":  ("ranged_preference", 0.85, 0.15),
    "build_siege":   ("heavy_preference",  0.85, 0.15),
    "fortify":       ("defend_radius",     0.20, 0.80),
    "hold_position": ("counter_rush",      0.80, 0.20),
    "harass":        ("scout_priority",    0.80, 0.20),
    "gather_resource": ("economy_pressure", 0.90, 0.10),
    "no_op":         ("tech_switch",       0.50, 0.50),
}

_SECONDARY_MAP = {
    "rush":           [("barracks_timing", 0.20), ("group_attack", 0.80)],
    "attack":         [("target_priority", 0.75), ("focus_fire",   0.70)],
    "build_melee":    [("max_workers",     0.60), ("worker_ratio", 0.55)],
    "build_ranged":   [("expand_threshold",0.70)],
    "gather_resource":[("worker_ratio",    0.80), ("second_barracks", 0.60)],
    "fortify":        [("defend_radius",   0.15), ("worker_fight_ratio", 0.20)],
    "hold_position":  [("defense_trigger", 0.80)],
}


def _gesy_decode_weights(script: GesyScript, n_samples: int = 500) -> list:
    w_out = {name: 0.5 for name in WEIGHT_NAMES}
    action_counts: dict[str, int] = {a: 0 for a in ALL_GESY_ACTIONS}

    for _ in range(n_samples):
        sample_w = {name: random.random() for name in WEIGHT_NAMES}
        cmd = script.run(sample_w) or "no_op"
        action_counts[cmd] = action_counts.get(cmd, 0) + 1

    total = sum(action_counts.values()) or 1

    for action, count in action_counts.items():
        freq = count / total
        if action in _ACTION_TO_WEIGHT_MAP and freq > 0:
            wname, hi, lo = _ACTION_TO_WEIGHT_MAP[action]
            w_out[wname] = w_out[wname] * (1 - freq) + (hi if freq > 0.5 else lo) * freq

    for action, count in action_counts.items():
        freq = count / total
        if action in _SECONDARY_MAP and freq > 0.10:
            for wname, target in _SECONDARY_MAP[action]:
                w_out[wname] = w_out[wname] * (1 - freq * 0.5) + target * freq * 0.5

    return [max(0.0, min(1.0, w_out[name])) for name in WEIGHT_NAMES]


def _replace_in(node: GesyNode, old: GesyNode, new: GesyNode) -> bool:
    for i, child in enumerate(node._children):
        if child is old:
            node._children[i] = new
            return True
        if _replace_in(child, old, new):
            return True
    return False


def _gesy_crossover(pa: GesyScript, pb: GesyScript,
                    child_name: str, generation: int) -> GesyScript:
    child_root = pa.root.clone()
    ntype = random.choice([GesyNodeType.ACTION, GesyNodeType.CONDITION])
    nodes_a = child_root.collect(ntype)
    nodes_b = pb.root.collect(ntype)
    if nodes_a and nodes_b:
        pt_a = random.choice(nodes_a)
        pt_b = random.choice(nodes_b).clone()
        if child_root is pt_a:
            if isinstance(pt_b, GesyNode) and pt_b.node_type == GesyNodeType.ACTION:
                child_root = pt_b
        else:
            _replace_in(child_root, pt_a, pt_b)
    return GesyScript(name=child_name, root=child_root, generation=generation)


def _gesy_mutate(script: GesyScript, mut_rate: float) -> GesyScript:
    m = script.clone(new_name=script.name + "_m")
    root = m.root

    if random.random() < mut_rate:
        leaves = [n for n in root.collect(GesyNodeType.ACTION)
                  if isinstance(n, GesyLeafAction)]
        if leaves:
            target = random.choice(leaves)
            _replace_in(root, target, GesyLeafAction(random.choice(ALL_GESY_ACTIONS)))

    if random.random() < mut_rate:
        conds = root.collect(GesyNodeType.CONDITION)
        if conds:
            target = random.choice(conds)
            _replace_in(root, target, _rand_cond_tree(max_depth=2))

    return m


def run_gesy(server: GameServer) -> tuple:
    print(f"\n{'='*60}")
    print(f"  RUNNING: GESY — Seeded Genetic Programming")
    print(f"  Seeds: Rush / Defensive / Ranged-Unit")
    print(f"  Pipeline: Seeded Init → Co-eval → Select → Xover → Mutate → Elitism → Invaders")
    print(f"{'='*60}")

    population: list[GesyScript] = _gesy_initial_population(GESY_POP_SIZE)
    best_script: Optional[GesyScript] = None
    best_fitness: float = -1.0
    t0 = time.time()
    counter = [0]

    def fresh(prefix: str) -> str:
        counter[0] += 1
        return f"{prefix}_G{{gen}}_{counter[0]:04d}"

    BLOAT_THRESHOLD    = 25
    BLOAT_PENALTY_RATE = 0.002

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opp = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]

        for script in population:
            wv   = _gesy_decode_weights(script)
            raw  = evaluate_weights(server, wv, opp)
            excess  = max(0, script.node_count() - BLOAT_THRESHOLD)
            fitness = max(0.0, raw - excess * BLOAT_PENALTY_RATE)
            script.fitness = fitness

        population.sort(key=lambda s: s.fitness, reverse=True)

        if population[0].fitness > best_fitness:
            best_fitness = population[0].fitness
            best_script  = population[0].clone(
                new_name=f"BestGen{gen}_{population[0].name}"
            )
            best_script.fitness = best_fitness

        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | "
              f"Best={population[0].fitness:.3f}  AvgNodes={sum(s.node_count() for s in population)/len(population):.1f}")

        if gen == TOTAL_GENERATIONS:
            break

        next_gen: list[GesyScript] = []
        for elite in population[:GESY_ELITISM]:
            ec = elite.clone(new_name=f"Elite_G{gen+1}_{counter[0]:04d}")
            counter[0] += 1
            ec.generation = gen + 1
            next_gen.append(ec)

        newcomers = []
        for _ in range(GESY_NEWCOMERS):
            counter[0] += 1
            nc = GesyScript(
                name       = f"Invader_G{gen+1}_{counter[0]:04d}",
                root       = _rand_action_tree(),
                generation = gen + 1,
            )
            newcomers.append(nc)

        n_offspring = GESY_POP_SIZE - GESY_ELITISM - GESY_NEWCOMERS
        for _ in range(n_offspring):
            pa = max(random.sample(population, min(GESY_TOURNAMENT_K, len(population))),
                     key=lambda s: s.fitness)
            pb = max(random.sample(population, min(GESY_TOURNAMENT_K, len(population))),
                     key=lambda s: s.fitness)

            counter[0] += 1
            child_name = f"Child_G{gen+1}_{counter[0]:04d}"
            if random.random() < GESY_CROSSOVER_RATE:
                child = _gesy_crossover(pa, pb, child_name, gen + 1)
            else:
                child = pa.clone(new_name=child_name)
                child.generation = gen + 1

            child = _gesy_mutate(child, GESY_MUTATION_RATE)
            child.generation = gen + 1

            if child.depth() > GESY_MAX_DEPTH:
                child = pa.clone(new_name=f"Clipped_G{gen+1}_{counter[0]:04d}")
                child.generation = gen + 1

            next_gen.append(child)

        next_gen.extend(newcomers)
        population = next_gen

    best_weights = _gesy_decode_weights(best_script)
    elapsed      = time.time() - t0

    print(f"\n  GESY best script: {best_script.name}")
    print(f"  Nodes: {best_script.node_count()}  Depth: {best_script.depth()}")
    print(f"  AST:\n{best_script.root.to_str(indent=2)}\n")

    return best_weights, elapsed


# ===============================================================
# [CONV 3]  CONVERGENCE CURVE PLOTTER
# ===============================================================
# Design decisions:
#   ONE combined figure — lets the reader directly compare how
#   fast each algorithm climbs and how stable it is.
#
#   Solid line  = mean win rate across all runs at each generation
#   Shaded band = mean ± 1 std-dev (shows run-to-run stability)
#
#   Y-axis in % (0–100) to match the validation win-rate scale.
#   X-axis = generation number (1–30).
#
#   Color palette chosen for IEEE grayscale-safe printing:
#     CCE-TDCCEA    → deep blue   (your main contribution, most visible)
#     Standard CCEA → orange-red  (ablation baseline, clearly different)
#     DE/rand/1/bin → green       (best non-coevo baseline)
#     TD-CCEA       → purple      (monolithic trajectory baseline)
# ===============================================================
CURVE_COLORS = {
    "CCE-TDCCEA":    "#1f4e79",   # deep blue
    "Standard CCEA": "#c0392b",   # red
    "DE/rand/1/bin": "#27ae60",   # green
    "TD-CCEA":       "#8e44ad",   # purple
}
CURVE_STYLES = {
    "CCE-TDCCEA":    "-",
    "Standard CCEA": "--",
    "DE/rand/1/bin": "-.",
    "TD-CCEA":       ":",
}
CURVE_MARKERS = {
    "CCE-TDCCEA":    "o",
    "Standard CCEA": "s",
    "DE/rand/1/bin": "^",
    "TD-CCEA":       "D",
}


def plot_convergence_curves(out_dir: str):
    """
    Build and save the convergence curve figure.
    Called once after ALL benchmark runs are complete.

    Also saves raw curve data to convergence_raw.json so the plot
    can be regenerated without re-running experiments.
    """
    if not _MATPLOTLIB_AVAILABLE:
        print("\n[CONV] matplotlib not available — skipping convergence plot.")
        return

    # ── Check we actually have data ──────────────────────────────
    has_data = any(len(v) > 0 for v in CONVERGENCE_DATA.values())
    if not has_data:
        print("\n[CONV] No convergence data collected — skipping plot.")
        return

    # ── Save raw JSON for later re-plotting ──────────────────────
    raw_path = os.path.join(out_dir, "convergence_raw.json")
    with open(raw_path, "w") as f:
        json.dump(CONVERGENCE_DATA, f, indent=2)
    print(f"\n[CONV] Raw convergence data saved → {raw_path}")

    # ── Build figure ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    for algo_name, runs in CONVERGENCE_DATA.items():
        if not runs:
            continue

        # Pad shorter runs to TOTAL_GENERATIONS with their last value
        # (handles edge cases where a run terminated early)
        padded = []
        for run in runs:
            if len(run) < TOTAL_GENERATIONS:
                run = run + [run[-1]] * (TOTAL_GENERATIONS - len(run))
            padded.append(run[:TOTAL_GENERATIONS])

        arr  = np.array(padded)            # shape: (n_runs, n_generations)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        gens = np.arange(1, TOTAL_GENERATIONS + 1)

        color  = CURVE_COLORS.get(algo_name,  "#555555")
        style  = CURVE_STYLES.get(algo_name,  "-")
        marker = CURVE_MARKERS.get(algo_name, "o")

        # Plot mean line — markers every 5 generations to avoid clutter
        ax.plot(gens, mean,
                linestyle=style,
                color=color,
                linewidth=2.0,
                marker=marker,
                markevery=5,
                markersize=6,
                label=f"{algo_name} (n={len(runs)})")

        # Shaded ± 1 std-dev band
        ax.fill_between(gens, mean - std, mean + std,
                         alpha=0.15, color=color)

    # ── Axes and labels ──────────────────────────────────────────
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Win Rate During Training (%)", fontsize=12)
    ax.set_title("Convergence Curves — Mean ± 1 Std Dev Across Runs\n"
                 "(32×32 Chambers Map, Rotating Opponent Schedule)",
                 fontsize=11)
    ax.set_xlim(1, TOTAL_GENERATIONS)
    ax.set_ylim(0, 105)
    ax.set_xticks(range(0, TOTAL_GENERATIONS + 1, 5))
    ax.set_yticks(range(0, 101, 10))
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    # Tight layout for clean paper inclusion
    fig.tight_layout()

    # ── Save 300 dpi PNG (IEEE submission standard) ───────────────
    png_path = os.path.join(out_dir, "convergence_curves.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[CONV] Convergence plot saved  → {png_path}")
    print(f"       Algorithms plotted: {[k for k,v in CONVERGENCE_DATA.items() if v]}")
    print(f"       Runs per algorithm:  {[len(v) for v in CONVERGENCE_DATA.values()]}")


# ===============================================================
#  MAIN
# ===============================================================
def main():
    print("\n" + "#" * 70)
    print("  BENCHMARK VS BOTS — 9/10 Algorithms vs 7 microRTS Bots")
    print("  + Convergence Curve Logging (CCE-TDCCEA, StdCCEA, DE, TD-CCEA)")
    print("#" * 70)
    print(f"  Pop={N_POP} | Gens={TOTAL_GENERATIONS} | "
          f"K={K_GAMES} | Validation={VALIDATION_GAMES}/bot\n")

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
        ("GESY",          "gesy"),
    ]
    if _SA_PDCOEA_AVAILABLE:
        algos.append(("SA-PDCoEA", "sa_pdcoea"))

    results = []
    for name, key in algos:
        server = GameServer(); server.start()
        print(f"  [OK] Server started — {name}")

        if   key == "ccea":       best, elapsed = run_ccea(server)
        elif key == "std_ccea":   best, elapsed = run_standard_ccea(server)
        elif key == "cmaes":      best, elapsed = run_cmaes(server)
        elif key == "de":         best, elapsed = run_de(server)
        elif key == "pso":        best, elapsed = run_pso(server)
        elif key == "centroid":   best, elapsed = evolve_core(server, "TD-CCEA")
        elif key == "sa":         best, elapsed = evolve_core(server, "SA-TDCCEA", pwm=sa_mut)
        elif key == "hc":         best, elapsed = run_hill_climbing(server)
        elif key == "sa_pdcoea": best, elapsed = _run_sa_pdcoea(server)
        else:                     best, elapsed = run_gesy(server)

        print(f"\n  Validating {name} against all 7 bots...")
        wr, per_opp = validate_weights(server, best)
        server.stop()

        print(f"  {name}: {wr:.1f}% overall in {elapsed:.1f}s")
        results.append({"name": name, "key": key, "time": elapsed,
                        "wr": wr, "per_opp": per_opp, "weights": best})

    # ── RANKING TABLE ────────────────────────────────────────────
    ranked = sorted(results, key=lambda x: x["wr"], reverse=True)
    medals = ["1st","2nd","3rd","4th","5th","6th","7th","8th","9th","10th"]

    print(f"\n{'='*70}")
    print(f"  FINAL RANKING — Win Rate vs 7 Bots")
    print(f"{'='*70}")
    print(f"  {'Rank':<6s} {'Algorithm':<18s} {'Win Rate':>10s} {'Time':>8s}")
    print(f"  {'_'*6} {'_'*18} {'_'*10} {'_'*8}")
    for i, r in enumerate(ranked):
        tag = ""
        if r["key"] == "ccea":      tag = "  ◄ CCE-TDCCEA (original)"
        if r["key"] == "gesy":      tag = "  ◄ GESY"
        if r["key"] == "sa_pdcoea": tag = "  ◄ SA-PDCoEA"
        print(f"  {medals[i]:<6s} {r['name']:<18s} {r['wr']:>9.1f}%  {r['time']:>7.1f}s{tag}")

    # ── PER-BOT TABLE ────────────────────────────────────────────
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

    # ── WILCOXON TEST ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  STATISTICAL SIGNIFICANCE — Wilcoxon Signed-Rank Test")
    print(f"  (CCE-TDCCEA vs each baseline, per-bot win rates as samples)")
    print(f"{'='*70}")
    cce = next((r for r in results if r["key"] == "ccea"), None)
    if cce:
        cce_rates = [cce["per_opp"].get(o, 0) for o in OPPONENT_ROTATION]

        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        w_out = [f"\nRun Date: {now_str}"]
        w_out.append(f"  {'Baseline':<20s}  {'p-value':>9s}  {'Significant?':>13s}  Direction")
        w_out.append(f"  {'_'*20}  {'_'*9}  {'_'*13}  {'_'*22}")

        print(f"\n  {'Baseline':<20s}  {'p-value':>9s}  {'Significant?':>13s}  Direction")
        print(f"  {'_'*20}  {'_'*9}  {'_'*13}  {'_'*22}")
        for r in results:
            if r["key"] == "ccea": continue
            other = [r["per_opp"].get(o, 0) for o in OPPONENT_ROTATION]
            p     = wilcoxon_test(cce_rates, other)
            if p is None:
                line = f"  {r['name']:<20s}  {'N/A':>9s}  {'N/A':>13s}  —"
            else:
                sig  = "YES (p<0.05)" if p < 0.05 else "NO  (p>=0.05)"
                dirn = ("CCE-TDCCEA BETTER" if sum(cce_rates) > sum(other)
                        else "BASELINE BETTER")
                line = f"  {r['name']:<20s}  {p:>9.4f}  {sig:>13s}  {dirn}"
            print(line)
            w_out.append(line)

        w_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wilcoxon_results.txt")
        with open(w_path, "a", encoding="utf-8") as f:
            f.write("\n".join(w_out) + "\n\n")

    # ── ABLATION ANALYSIS ────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ABLATION ANALYSIS")
    print(f"{'='*70}")

    abl_out = [f"\nRun Date: {time.strftime('%Y-%m-%d %H:%M:%S')}"]

    cce   = next((r for r in results if r["key"] == "ccea"),     None)
    sccea = next((r for r in results if r["key"] == "std_ccea"), None)
    cent  = next((r for r in results if r["key"] == "centroid"), None)
    gesy  = next((r for r in results if r["key"] == "gesy"),     None)
    if cce and sccea:
        d = cce["wr"] - sccea["wr"]
        lines = [
            f"\n  Trajectory memory contribution:",
            f"    CCE-TDCCEA ({cce['wr']:.1f}%) vs Std-CCEA ({sccea['wr']:.1f}%)",
            f"    Result: {'Trajectory HELPS (+' if d >= 0 else 'Trajectory HURTS ('}{abs(d):.1f}%)"
        ]
        for line in lines: print(line)
        abl_out.extend(lines)
    if cce and cent:
        d = cce["wr"] - cent["wr"]
        lines = [
            f"\n  Cooperative decomposition contribution:",
            f"    CCE-TDCCEA ({cce['wr']:.1f}%) vs TD-CCEA ({cent['wr']:.1f}%)",
            f"    Result: {'Decomposition HELPS (+' if d >= 0 else 'Decomposition HURTS ('}{abs(d):.1f}%)"
        ]
        for line in lines: print(line)
        abl_out.extend(lines)
    if gesy and cce:
        d = gesy["wr"] - cce["wr"]
        lines = [
            f"\n  GESY seeded GP contribution:",
            f"    GESY ({gesy['wr']:.1f}%) vs CCE-TDCCEA ({cce['wr']:.1f}%)",
            f"    Result: {'GESY BETTER (+' if d >= 0 else 'GESY WORSE ('}{abs(d):.1f}%)"
        ]
        for line in lines: print(line)
        abl_out.extend(lines)
    pdcoea = next((r for r in results if r["key"] == "sa_pdcoea"), None)
    if pdcoea and cce:
        d = pdcoea["wr"] - cce["wr"]
        lines = [
            f"\n  SA-PDCoEA pairwise dominance contribution:",
            f"    SA-PDCoEA ({pdcoea['wr']:.1f}%) vs CCE-TDCCEA ({cce['wr']:.1f}%)",
            f"    Result: {'SA-PDCoEA BETTER (+' if d >= 0 else 'SA-PDCoEA WORSE ('}{abs(d):.1f}%)"
        ]
        for line in lines: print(line)
        abl_out.extend(lines)

    abl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ablation_results.txt")
    with open(abl_path, "a", encoding="utf-8") as f:
        f.write("\n".join(abl_out) + "\n\n")

    # ── SAVE JSON ────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(out_dir, "benchmark_vs_bots_results.json")
    with open(out, "w") as f:
        json.dump([{"name": r["name"], "key": r["key"],
                    "overall_winrate": r["wr"], "per_opponent": r["per_opp"],
                    "time": r["time"]} for r in results], f, indent=2)
    print(f"\n  Results saved → {out}")

    # ── SAVE STRATEGIES (APPEND) ─────────────────────────────────
    strat_path = os.path.join(out_dir, "final_strategies.txt")
    with open(strat_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"Run Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*70}\n")
        for r in ranked:
            f.write(f"Algorithm: {r['name']} (Win Rate: {r['wr']:.1f}%)\n")
            if "weights" in r and isinstance(r["weights"], list):
                weights_str = ", ".join(f"{w:.4f}" for w in r["weights"])
                f.write(f"Weights: [{weights_str}]\n")
            elif "weights" in r and isinstance(r["weights"], dict):
                # GESY weights might be a dict instead of list
                f.write(f"Weights: {r['weights']}\n")
            f.write("-" * 40 + "\n")
    print(f"  Strategies appended → {strat_path}")

    # ── EXCEL EXPORT ─────────────────────────────────────────────
    if _OPENPYXL_AVAILABLE:
        xlsx_path = os.path.join(out_dir, "benchmark_winrates.xlsx")
        algo_names = [r["name"] for r in results]
        algo_wr    = {r["name"]: round(r["wr"],   2) for r in results}
        algo_time  = {r["name"]: round(r["time"], 1) for r in results}

        if not os.path.exists(xlsx_path):
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Win Rates"
            ws.append(["Algorithm", "Run1", "Time1"])
            for algo in algo_names:
                ws.append([algo, algo_wr[algo], algo_time[algo]])
            wb.save(xlsx_path)
            run_col_name  = "Run1"
            time_col_name = "Time1"
        else:
            wb = openpyxl.load_workbook(xlsx_path)
            ws = wb.active
            header_row  = [cell.value for cell in ws[1]]
            run_numbers = [
                int(h[3:]) for h in header_row
                if isinstance(h, str) and h.lower().startswith("run")
                and h[3:].isdigit()
            ]
            next_run      = (max(run_numbers) + 1) if run_numbers else 1
            run_col_name  = f"Run{next_run}"
            time_col_name = f"Time{next_run}"
            run_col_idx  = len(header_row) + 1
            time_col_idx = len(header_row) + 2
            ws.cell(row=1, column=run_col_idx,  value=run_col_name)
            ws.cell(row=1, column=time_col_idx, value=time_col_name)
            algo_row_map = {}
            for row_idx in range(2, ws.max_row + 1):
                cell_val = ws.cell(row=row_idx, column=1).value
                if cell_val:
                    algo_row_map[cell_val] = row_idx
            for algo in algo_names:
                if algo in algo_row_map:
                    ri = algo_row_map[algo]
                else:
                    ri = ws.max_row + 1
                    ws.cell(row=ri, column=1, value=algo)
                ws.cell(row=ri, column=run_col_idx,  value=algo_wr[algo])
                ws.cell(row=ri, column=time_col_idx, value=algo_time[algo])
            wb.save(xlsx_path)

        print(f"  Excel updated  → {xlsx_path}  (columns: {run_col_name}, {time_col_name})")
    else:
        print("  [SKIP] Excel export skipped — install openpyxl to enable.")

    # ── [CONV 3] GENERATE CONVERGENCE CURVES ─────────────────────
    plot_convergence_curves(out_dir)

    print(f"\n{'='*70}")
    print(f"  BENCHMARK COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
