#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EC_SA_TDCCEA_Parallel.py
========================
Rectified EC-SA-TDCCEA (Environment-Conditioned Sensitivity-Adaptive
Trajectory-Driven Competitive Co-Evolutionary Algorithm) compared directly
against the baseline SA-TDCCEA (Static Sensitivity-Adaptive TD-CCEA) from
COMPARE_ALL.py, with full multi-core parallel game evaluation.

ALGORITHMS:
  1. SA-TDCCEA      (Baseline)  — Trajectory-driven co-evolution with static
                                  pre-computed parameter sensitivity (pwm).
  2. EC-SA-TDCCEA   (Proposed)  — Environment-Conditioned Sensitivity-Adaptive
                                  TD-CCEA from Algorithm 1 of the specification:
                                  • Environment memory clusters M_j per parameter
                                  • Adaptive ablation triggered by environment drift
                                    (tau_E, tau_R, B_max)
                                  • Similarity-weighted sensitivity retrieval S_j(E_t)
                                  • Pairwise strategy-interaction matrix I smoothed
                                    via EMA
                                  • Covariance factor L from correlation matrix C
                                  • Correlated mutation N(theta, mu * I_cov)

MULTI-CORE PARALLELISM:
  Runs N_WORKERS independent Java game servers concurrently via ThreadPoolExecutor.
  On multi-core CPUs, evaluations are distributed across all cores for maximum speed.

STANDALONE USAGE:
  python EC_SA_TDCCEA_Parallel.py [n_workers]

INTEGRATION CONTRACT:
  run_ec_sa_tdccea(server, n_workers=None) -> (best_weights[20], elapsed_s)
  run_sa_tdccea(server, n_workers=None)    -> (best_weights[20], elapsed_s)
"""

import os
import sys
import json
import math
import time
import queue
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import openpyxl
    _OPENPYXL_AVAILABLE = True
except ImportError:
    _OPENPYXL_AVAILABLE = False

os.environ["PYTHONIOENCODING"] = "utf-8"

# ===============================================================
#  CONFIGURATION & PATHS
# ===============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
cand_dirs = [
    os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "Simulation", "microrts"),
    os.path.join(os.path.dirname(SCRIPT_DIR), "Simulation", "microrts"),
    os.path.join(SCRIPT_DIR, "Simulation", "microrts"),
]
MICRORTS_DIR = next(
    (d for d in cand_dirs if os.path.isfile(os.path.join(d, "bin", "tests", "EvolvedBatchGameRunner.class"))),
    cand_dirs[0]
)

MAP_PATH = "maps/pr1_48x48.xml"
if not os.path.isfile(os.path.join(MICRORTS_DIR, MAP_PATH)):
    for _fb in ["maps/temp_pr1_48x48.xml", "maps/temp_match_48x48.xml", "maps/chambers32x32.xml"]:
        if os.path.isfile(os.path.join(MICRORTS_DIR, _fb)):
            MAP_PATH = _fb
            break

MAX_CYCLES   = 12000
JAVA_CMD     = "java"

WEIGHT_NAMES = [
    "worker_ratio", "max_workers", "expand_threshold", "light_preference",
    "heavy_preference", "ranged_preference", "barracks_timing", "aggression",
    "attack_threshold", "defend_radius", "target_priority", "defense_trigger",
    "counter_rush", "tech_switch", "economy_pressure", "scout_priority",
    "group_attack", "second_barracks", "worker_fight_ratio", "focus_fire",
]
DIMENSIONS = len(WEIGHT_NAMES)  # 20
D = DIMENSIONS

OPPONENT_ROTATION = [
    "EconomyMilitaryRush", "LightRush", "WorkerRush",
    "HeavyDefense", "RangedRush", "EconomyRush", "WorkerRushPlusPlus",
]
N_OPP = len(OPPONENT_ROTATION)

# Shared benchmark hyperparameters
N_POP             = 20      # Population size
K_GAMES           = 3       # Games per evaluation
TOTAL_GENERATIONS = 30      # Generations
TRAJ_WINDOW       = 7       # History window W
MU_BASE           = 0.15    # Base mutation
EPS               = 0.03    # Epsilon offset
DELTA_F           = 0.20    # Fitness threshold
DELTA_S           = 0.85    # Redundancy threshold
A_MAX             = 15      # Archive capacity
VALIDATION_GAMES  = 5       # Games per opponent during final validation
DRAW_SCORE        = 0.3     # Draw score

# EC-SA-TDCCEA specific hyperparameters
TAU_E   = 0.45   # Environment drift threshold to trigger ablation
TAU_I   = 0.80   # Cluster match similarity threshold
B_MAX   = 5      # Maximum generations between ablations
TAU_R   = 1.00   # Accumulated drift reservoir threshold
RHO     = 0.80   # Reservoir decay
ETA     = 0.30   # Cluster sensitivity EMA learning rate

# Strategy-interaction matrix hyperparameters
I_EMA            = 0.30   # EMA smoothing of I
I_COUPLING       = 0.60   # Off-diagonal covariance coupling (0 -> independent)
RATE_LO, RATE_HI = 0.08, 0.20   # Static/dynamic mutation rate range
MIN_KEEP         = 8      # Safety floor on survivors

SENSITIVITY_FILE = "sensitivity_scores.json"


# ===============================================================
#  MULTI-CORE GAME POOL & SERVER
# ===============================================================
def default_workers():
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 4
    env = os.environ.get("ECSA_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, min(cores - 1, N_POP))


class GameServer:
    def __init__(self):
        self.process = None

    def start(self):
        classpath = os.pathsep.join(["lib/*", "bin"])
        self.process = subprocess.Popen(
            [JAVA_CMD, "-cp", classpath, "tests.EvolvedBatchGameRunner"],
            cwd=MICRORTS_DIR,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        line = self.process.stdout.readline().strip()
        if line != "READY":
            err = ""
            try:
                err = self.process.stderr.read(1000)
            except Exception:
                pass
            raise RuntimeError(f"Server failed to start: stdout='{line}', stderr='{err}'")

    def run_game(self, weights, opponent):
        ws = ";".join(f"{w:.4f}" for w in weights)
        req = f"EVOLVED:{ws},{opponent},{MAP_PATH},{MAX_CYCLES}\n"
        try:
            self.process.stdin.write(req)
            self.process.stdin.flush()
            line = self.process.stdout.readline().strip()
            if line.startswith("RESULT:"):
                p = line.replace("RESULT:", "").split(",")
                return int(p[0]), int(p[1])
        except (BrokenPipeError, OSError):
            pass
        return -1, MAX_CYCLES

    def stop(self):
        if self.process:
            try:
                self.process.stdin.write("EXIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            finally:
                self.process = None


class ServerPool:
    """Pool of independent GameServer instances running concurrently."""

    def __init__(self, base_server=None, n_workers=None):
        self.n = n_workers or default_workers()
        self.base = base_server
        self.owned_servers = []
        self.q = queue.Queue()

        if base_server is not None:
            self.q.put(base_server)
            extra_count = self.n - 1
        else:
            extra_count = self.n

        for _ in range(extra_count):
            s = GameServer()
            s.start()
            self.owned_servers.append(s)
            self.q.put(s)

        self.ex = ThreadPoolExecutor(max_workers=self.n)
        self.games = 0

    def _game(self, weights, opp):
        s = self.q.get()
        try:
            try:
                return s.run_game(weights, opp)
            except Exception:
                try:
                    s.stop()
                    s.start()
                    return s.run_game(weights, opp)
                except Exception:
                    return -1, MAX_CYCLES
        finally:
            self.q.put(s)

    def evaluate_many(self, weight_list, opp):
        """Evaluate a batch of weight vectors against an opponent (K_GAMES each)."""
        futs = [[self.ex.submit(self._game, w, opp) for _ in range(K_GAMES)]
                for w in weight_list]
        out = []
        for fs in futs:
            score = 0.0
            for f in fs:
                winner, _ = f.result()
                score += 1.0 if winner == 0 else (DRAW_SCORE if winner == -1 else 0.0)
            out.append(score / K_GAMES)
        self.games += len(weight_list) * K_GAMES
        return out

    def validate(self, weights):
        """Validate candidate weights against all opponents (VALIDATION_GAMES each)."""
        futs = {o: [self.ex.submit(self._game, weights, o) for _ in range(VALIDATION_GAMES)]
                for o in OPPONENT_ROTATION}
        per_opp = {}
        total_wins = 0
        for o, fs in futs.items():
            wins = sum(1 for f in fs if f.result()[0] == 0)
            per_opp[o] = (wins / VALIDATION_GAMES) * 100.0
            total_wins += wins
        overall = (total_wins / (VALIDATION_GAMES * N_OPP)) * 100.0
        return overall, per_opp

    def close(self):
        self.ex.shutdown(wait=True)
        for s in self.owned_servers:
            try:
                s.stop()
            except Exception:
                pass


# ===============================================================
#  HELPERS & METRICS
# ===============================================================
def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    xs = list(xs)
    if not xs:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def euclid(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def centroid(vectors):
    n = len(vectors)
    return [sum(v[j] for v in vectors) / n for j in range(len(vectors[0]))]


# ===============================================================
#  ENVIRONMENT FINGERPRINT  E_t = [R_pop, R_opp, R_game, R_fit]
#    R_pop : population centroid (20) + archive-elite centroid (20)
#    R_opp : one-hot encoding of the opponent (7)
#    R_game: [generation progress, population diversity] (2)
#    R_fit : [mean, max, std] of previous generation's fitness (3)
# ===============================================================
POP_SL  = slice(0, 2 * D)
OPP_SL  = slice(2 * D, 2 * D + N_OPP)
GAME_SL = slice(OPP_SL.stop, OPP_SL.stop + 2)
FIT_SL  = slice(GAME_SL.stop, GAME_SL.stop + 3)
BLOCK_W = (0.4, 0.3, 0.1, 0.2)


def build_fingerprint(gen, opp, pop, archive, fit_stats):
    cen = centroid([a.params for a in pop])
    if archive:
        top = sorted(archive, key=lambda a: a.e_i, reverse=True)[:3]
        elite = centroid([a.params for a in top])
    else:
        elite = cen
    onehot = [1.0 if o == opp else 0.0 for o in OPPONENT_ROTATION]
    div = mean(euclid(a.params, cen) for a in pop) / math.sqrt(D)
    game = [gen / TOTAL_GENERATIONS, min(1.0, 2.0 * div)]
    return cen + elite + onehot + game + list(fit_stats)


def _esim(a, b):
    if not a or not b or len(a) == 0:
        return 1.0
    return max(0.0, 1.0 - euclid(a, b) / math.sqrt(len(a)))


def env_sim(e1, e2):
    if len(e1) <= D or len(e2) <= D:
        return max(0.0, 1.0 - euclid(e1[:D], e2[:D]) / math.sqrt(D))
    s_pop  = _esim(e1[POP_SL], e2[POP_SL])
    s_opp  = max(0.0, 1.0 - 0.5 * sum(abs(p - q) for p, q in zip(e1[OPP_SL], e2[OPP_SL])))
    s_game = _esim(e1[GAME_SL], e2[GAME_SL])
    s_fit  = _esim(e1[FIT_SL], e2[FIT_SL])
    return sum(w * s for w, s in zip(BLOCK_W, (s_pop, s_opp, s_game, s_fit))) / sum(BLOCK_W)


def d_env(e1, e2):
    return 1.0 - env_sim(e1, e2)


# ===============================================================
#  AGENT CLASS (trajectory memory, trajectory score, mutation)
# ===============================================================
class Agent:
    def __init__(self, params=None, pwm=None):
        self.params = ([random.random() for _ in range(D)] if params is None
                       else [clamp01(x) for x in params])
        self.H_i  = []
        self.e_i  = 0.0
        self.mu_i = 0.0
        self.fit  = 0.0
        self.pwm  = pwm  # Used by baseline SA-TDCCEA

    def update_history(self, f, E):
        self.H_i.append((f, E))
        if len(self.H_i) > TRAJ_WINDOW:
            self.H_i.pop(0)

    def compute_trajectory(self, E):
        if not self.H_i:
            self.e_i = 0.0
            return
        if len(E) > D:
            ws = [env_sim(E, pe) for _, pe in self.H_i]
        else:
            md = math.sqrt(D)
            ws = [max(0.0, 1.0 - euclid(E, pe[:D]) / md) for _, pe in self.H_i]
        t = sum(ws)
        ws = [w / t for w in ws] if t > 0 else [1.0 / len(ws)] * len(ws)
        self.e_i = sum(w * self.H_i[i][0] for i, w in enumerate(ws))

    def compute_mutation(self):
        self.mu_i = MU_BASE * (1.0 - self.e_i) + EPS

    def mutate_static(self):
        """Static sensitivity mutation (SA-TDCCEA baseline rule)."""
        cp = []
        if self.pwm is not None:
            for j, p in enumerate(self.params):
                sigma_j = self.pwm[j] * (1.0 - self.e_i) + EPS
                cp.append(clamp01(p + random.gauss(0, sigma_j)))
        else:
            for p in self.params:
                cp.append(clamp01(p + random.gauss(0, self.mu_i)))
        child = Agent(cp, self.pwm)
        child.mu_i = self.mu_i
        child.e_i  = self.e_i
        return child

    def snapshot(self):
        c = Agent(self.params, self.pwm)
        c.e_i, c.H_i, c.mu_i, c.fit = self.e_i, self.H_i[:], self.mu_i, self.fit
        return c


# ===============================================================
#  ENVIRONMENT-CONDITIONED SENSITIVITY CLUSTERS (Phase 3)
# ===============================================================
def ablation_update(pool, memory, baseline, opp, E):
    """
    Local ablation of each parameter j:
    S_j(E_t) = |WinRate(x_j=1, E_t) - WinRate(x_j=0, E_t)| around baseline.
    Updates closest cluster in M_j (sim >= tau_I) via EMA, else creates new cluster.
    """
    vecs = []
    for j in range(D):
        for v in (0.0, 1.0):
            t = baseline[:]
            t[j] = v
            vecs.append(t)
    fits = pool.evaluate_many(vecs, opp)

    for j in range(D):
        S = abs(fits[2 * j + 1] - fits[2 * j])
        clusters = memory[j]
        best_k, best_s = None, -1.0
        for k, c in enumerate(clusters):
            s = env_sim(E, c["mu"])
            if s > best_s:
                best_k, best_s = k, s
        if best_k is not None and best_s >= TAU_I:
            c = clusters[best_k]
            n = c["n"]
            c["mu"] = [(n * m + e) / (n + 1) for m, e in zip(c["mu"], E)]
            c["n"]  = n + 1
            c["S"]  = S if c["S"] is None else (1 - ETA) * c["S"] + ETA * S
        else:
            clusters.append({"mu": E[:], "S": S, "n": 1})


def retrieve_sensitivity(memory, E):
    """
    Retrieve sensitivity of each parameter from clusters with sim >= tau_I,
    weighted by similarity^8. Unmatched parameters receive the mean of matched ones.
    """
    out = [None] * D
    for j in range(D):
        num = den = 0.0
        for c in memory[j]:
            s = env_sim(E, c["mu"])
            if s >= TAU_I and c["S"] is not None:
                w = s ** 8
                num += w * c["S"]
                den += w
        if den > 0:
            out[j] = num / den
    known = [x for x in out if x is not None]
    if not known:
        return None
    fill = mean(known)
    return [fill if x is None else x for x in out]


def sensitivity_to_scale(S):
    """Convert sensitivity to relative mutation multipliers with mean 1.0."""
    mx = max(S) if max(S) > 0 else 1.0
    rate = [RATE_LO + (1.0 - s / mx) * (RATE_HI - RATE_LO) for s in S]
    m = mean(rate)
    return [r / m for r in rate]


# ===============================================================
#  STRATEGY-INTERACTION MATRIX & COVARIANCE FACTOR (Phase 4)
# ===============================================================
def interaction_observation(P, f):
    """
    Measure pairwise parameter interactions: |corr(f, z_i * z_j)|.
    Non-zero when joint settings of (i, j) explain variation in fitness.
    """
    obs = np.zeros((D, D))
    X  = np.asarray(P, dtype=float)
    fv = np.asarray(f, dtype=float)
    if len(fv) < 4 or fv.std() < 1e-9:
        return obs
    zf = (fv - fv.mean()) / fv.std()
    sd = X.std(axis=0)
    Z  = np.where(sd > 1e-9, (X - X.mean(axis=0)) / np.where(sd > 1e-9, sd, 1), 0.0)
    for i in range(D):
        for j in range(i + 1, D):
            pr = Z[:, i] * Z[:, j]
            ps = pr.std()
            if ps < 1e-9:
                continue
            c = float(np.mean((pr - pr.mean()) / ps * zf))
            obs[i, j] = obs[j, i] = min(1.0, abs(c))
    return obs


def covariance_factor(I):
    """Compute Cholesky factor L of correlation matrix C = (1-g)*Id + g*I."""
    C = np.eye(D) + I_COUPLING * (I - np.diag(np.diag(I)))
    w, V = np.linalg.eigh((C + C.T) / 2)
    w = np.clip(w, 1e-3, None)
    C = (V * w) @ V.T
    dg = np.sqrt(np.diag(C))
    C = C / np.outer(dg, dg)
    return np.linalg.cholesky(C + 1e-9 * np.eye(D))


# ===============================================================
#  SELECTION & SURVIVAL
# ===============================================================
def select_and_reproduce(pop, archive, mutate_fn):
    md = math.sqrt(D)
    ranked = sorted(pop, key=lambda a: a.e_i, reverse=True)

    survivors = []
    for a in ranked:
        if not any(1.0 - euclid(a.params, u.params) / md > DELTA_S for u in survivors):
            survivors.append(a)

    final = [a for a in survivors if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
    for a in survivors:
        if len(final) >= min(MIN_KEEP, len(survivors)):
            break
        if a not in final:
            final.append(a)
    final = final[:N_POP]

    for top in sorted(final, key=lambda a: a.e_i, reverse=True)[:3]:
        archive.append(top.snapshot())
    archive.sort(key=lambda a: a.e_i, reverse=True)
    del archive[A_MAX:]

    pool_ = final + archive
    nxt = final[:]
    while len(nxt) < N_POP:
        cands = random.sample(pool_, min(3, len(pool_)))
        parent = max(cands, key=lambda a: a.e_i)
        nxt.append(mutate_fn(parent))
    return nxt, archive


# ===============================================================
#  BASELINE: STATIC SENSITIVITY PRE-ANALYSIS
# ===============================================================
def load_or_run_static_sensitivity(pool):
    """
    Pre-compute or load static sensitivity scores (pwm) as done in COMPARE_ALL.py.
    """
    fp = os.path.join(SCRIPT_DIR, SENSITIVITY_FILE)
    if os.path.exists(fp):
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            rates = data.get("normalized_mutation_rates")
            if rates and len(rates) == DIMENSIONS and 0.07 <= min(rates) and max(rates) <= 0.22:
                print(f"  [SA] Loaded cached static sensitivity from {SENSITIVITY_FILE}")
                return rates
        except Exception:
            pass

    print("  [SA] Running parallel static sensitivity analysis across all 7 bots...")
    baseline = [0.5] * D
    # 20 dimensions * 2 evaluations (0.0 and 1.0) = 40 evaluations
    vecs_low  = []
    vecs_high = []
    for j in range(D):
        tl = baseline[:]
        tl[j] = 0.0
        vecs_low.append(tl)
        th = baseline[:]
        th[j] = 1.0
        vecs_high.append(th)

    all_vecs = vecs_low + vecs_high
    influence = [0.0] * D

    for opp in OPPONENT_ROTATION:
        fits = pool.evaluate_many(all_vecs, opp)
        fits_low  = fits[:D]
        fits_high = fits[D:]
        for j in range(D):
            influence[j] += abs(fits_high[j] - fits_low[j])

    influence = [inf / len(OPPONENT_ROTATION) for inf in influence]
    max_inf = max(influence) if max(influence) > 0 else 1.0
    normalized = [0.08 + (1.0 - inf / max_inf) * 0.12 for inf in influence]

    try:
        with open(fp, "w") as f:
            json.dump({
                "influence_scores": influence,
                "normalized_mutation_rates": normalized,
                "weight_names": WEIGHT_NAMES,
            }, f, indent=2)
        print(f"  [SA] Saved static sensitivity scores to {SENSITIVITY_FILE}")
    except Exception:
        pass

    return normalized


# ===============================================================
#  ALGORITHM 1: BASELINE SA-TDCCEA (from COMPARE_ALL.py)
# ===============================================================
def run_sa_tdccea(pool, pwm=None):
    """
    Baseline SA-TDCCEA (Sensitivity-Adaptive TD-CCEA) with parallel evaluation.
    Uses static per-parameter mutation rates (pwm) computed from offline sensitivity analysis.
    """
    print(f"\n{'='*65}")
    print("  RUNNING: SA-TDCCEA (Baseline Static Sensitivity-Adaptive)")
    print(f"{'='*65}")
    t_start = time.time()

    if pwm is None:
        pwm = load_or_run_static_sensitivity(pool)

    pop     = [Agent(pwm=pwm) for _ in range(N_POP)]
    archive = []
    convergence = []

    for t in range(TOTAL_GENERATIONS):
        gen = t + 1
        opp = OPPONENT_ROTATION[t % N_OPP]

        # Centroid environment
        cen  = centroid([a.params for a in pop])
        fits = pool.evaluate_many([a.params for a in pop], opp)

        for a, f in zip(pop, fits):
            a.fit = f
            a.update_history(f, cen)
        for a in pop:
            a.compute_trajectory(cen)
            a.compute_mutation()

        best = max(pop, key=lambda a: a.e_i)
        best_eval = pool.evaluate_many([best.params], opp)[0]
        convergence.append(best_eval)

        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | e_i={best.e_i:.3f} "
              f"Best={best_eval:.3f}")

        def mutate_static(parent):
            return parent.mutate_static()

        pop, archive = select_and_reproduce(pop, archive, mutate_static)

    champion = max(archive, key=lambda a: a.e_i) if archive else max(pop, key=lambda a: a.e_i)
    return list(champion.params), time.time() - t_start, convergence


# ===============================================================
#  ALGORITHM 2: PROPOSED EC-SA-TDCCEA (Specification Algorithm 1)
# ===============================================================
def run_ec_sa_tdccea(pool):
    """
    Environment-Conditioned Sensitivity-Adaptive Trajectory-Driven
    Competitive Co-Evolutionary Algorithm (EC-SA-TDCCEA).
    """
    print(f"\n{'='*65}")
    print("  RUNNING: EC-SA-TDCCEA (Proposed Environment-Conditioned)")
    print(f"{'='*65}")
    t_start = time.time()
    rng = np.random.default_rng(random.getrandbits(32))

    pop         = [Agent() for _ in range(N_POP)]
    archive     = []
    memory      = [[] for _ in range(D)]
    I           = np.zeros((D, D))
    E_prev      = None
    E_last      = None
    t_last      = 0
    R           = 0.0
    stats       = (0.0, 0.0, 0.0)
    convergence = []

    for t in range(TOTAL_GENERATIONS):
        gen = t + 1
        opp = OPPONENT_ROTATION[t % N_OPP]

        # Phase 1: Fingerprint + Evaluation (Parallel)
        E = build_fingerprint(gen, opp, pop, archive, stats)
        fits = pool.evaluate_many([a.params for a in pop], opp)
        for a, f in zip(pop, fits):
            a.fit = f
            a.update_history(f, E)
        for a in pop:
            a.compute_trajectory(E)
        stats = (mean(fits), max(fits), std(fits))

        # Phase 2: Adaptive Ablation Scheduling
        if E_prev is None:
            R = 0.0
        else:
            R = RHO * R + d_env(E, E_prev)
        d_chk = float("inf") if E_last is None else d_env(E, E_last)
        ablate = (E_last is None or d_chk > TAU_E or R > TAU_R or (t - t_last) >= B_MAX)

        # Phase 3: Environment-Conditioned Sensitivity
        if ablate:
            baseline = max(pop, key=lambda a: a.e_i).params
            ablation_update(pool, memory, baseline, opp, E)
            E_last, t_last, R = E, t, 0.0
        E_prev = E

        # Phase 4: Sensitivity-Guided Evolution
        S = retrieve_sensitivity(memory, E)
        scale = np.array(sensitivity_to_scale(S)) if S else np.ones(D)
        I = (1 - I_EMA) * I + I_EMA * interaction_observation([a.params for a in pop], fits)
        np.fill_diagonal(I, 0.0)
        L = covariance_factor(I)
        for a in pop:
            a.compute_mutation()

        best = max(pop, key=lambda a: a.e_i)
        best_eval = pool.evaluate_many([best.params], opp)[0]
        convergence.append(best_eval)

        n_clusters = sum(len(m) for m in memory)
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | e_i={best.e_i:.3f} "
              f"Best={best_eval:.3f} | ablate={'Y' if ablate else '-'} "
              f"envClusters={n_clusters}")

        # Phases 5-6: Selection, Archive, Reproduction
        def mutate_ec(parent, L=L, scale=scale):
            step = parent.mu_i * scale * (L @ rng.standard_normal(D))
            return Agent(np.asarray(parent.params) + step)

        pop, archive = select_and_reproduce(pop, archive, mutate_ec)

    champion = max(archive, key=lambda a: a.e_i) if archive else max(pop, key=lambda a: a.e_i)
    return list(champion.params), time.time() - t_start, convergence


# ===============================================================
#  WILCOXON SIGNED-RANK TEST (Two-Sided)
# ===============================================================
def wilcoxon_test(x, y):
    diffs = [xi - yi for xi, yi in zip(x, y) if xi != yi]
    if len(diffs) < 3:
        return None
    n = len(diffs)
    abs_d = sorted(enumerate(abs(d) for d in diffs), key=lambda t: t[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and abs_d[j][1] == abs_d[j + 1][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[abs_d[k][0]] = avg_rank
        i = j + 1
    w_plus  = sum(ranks[i] for i, d in enumerate(diffs) if d > 0)
    w_minus = sum(ranks[i] for i, d in enumerate(diffs) if d < 0)
    w_stat  = min(w_plus, w_minus)
    mu_w    = n * (n + 1) / 4.0
    sig_w   = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sig_w == 0:
        return None
    z = (w_stat - mu_w) / sig_w
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return p


# ===============================================================
#  MAIN BENCHMARK & COMPARISON
# ===============================================================
def main():
    import sys
    nw = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else default_workers()

    print("\n" + "#" * 70)
    print("  EC-SA-TDCCEA vs SA-TDCCEA — Dynamic vs Static Sensitivity")
    print("#" * 70)
    print(f"  Pop={N_POP} | Gens={TOTAL_GENERATIONS} | K={K_GAMES} | Validation={VALIDATION_GAMES}/bot")
    print(f"  Map: {MAP_PATH} (48x48)")
    print(f"  Parallel game workers: {nw} (of {os.cpu_count()} CPU cores)")
    print()

    pool = ServerPool(n_workers=nw)

    results = []
    try:
        # 1. Baseline SA-TDCCEA
        pwm_static = load_or_run_static_sensitivity(pool)
        best_sa, time_sa, conv_sa = run_sa_tdccea(pool, pwm=pwm_static)
        print("\n  Validating SA-TDCCEA against all 7 bots...")
        wr_sa, opp_sa = pool.validate(best_sa)
        print(f"  SA-TDCCEA: {wr_sa:.1f}% overall in {time_sa:.1f}s")
        results.append({
            "name": "SA-TDCCEA", "key": "sa", "time": time_sa,
            "wr": wr_sa, "per_opp": opp_sa, "weights": best_sa,
            "convergence": conv_sa,
        })

        # 2. Proposed EC-SA-TDCCEA
        best_ec, time_ec, conv_ec = run_ec_sa_tdccea(pool)
        print("\n  Validating EC-SA-TDCCEA against all 7 bots...")
        wr_ec, opp_ec = pool.validate(best_ec)
        print(f"  EC-SA-TDCCEA: {wr_ec:.1f}% overall in {time_ec:.1f}s")
        results.append({
            "name": "EC-SA-TDCCEA", "key": "ec_sa", "time": time_ec,
            "wr": wr_ec, "per_opp": opp_ec, "weights": best_ec,
            "convergence": conv_ec,
        })

    finally:
        pool.close()
        print("  [OK] All parallel game servers stopped cleanly.")

    # ═══════════════════════════════════════════════════════════════
    #  RANKING TABLE
    # ═══════════════════════════════════════════════════════════════
    ranked = sorted(results, key=lambda x: x["wr"], reverse=True)
    medals = ["1st", "2nd"]
    print(f"\n{'='*70}")
    print("  FINAL RANKING — Win Rate vs 7 Bots")
    print(f"{'='*70}")
    print(f"  {'Rank':<6s} {'Algorithm':<20s} {'Win Rate':>10s} {'Time':>8s}")
    print(f"  {'_'*6} {'_'*20} {'_'*10} {'_'*8}")
    for i, r in enumerate(ranked):
        tag = "  ◄ Proposed (EC)" if r["key"] == "ec_sa" else "  ◄ Baseline (Static)"
        print(f"  {medals[i]:<6s} {r['name']:<20s} {r['wr']:>9.1f}%  {r['time']:>7.1f}s{tag}")

    # ═══════════════════════════════════════════════════════════════
    #  PER-BOT TABLE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  WIN RATE PER BOT (%)")
    print(f"{'='*70}")
    header = f"  {'Bot':<26s}"
    for r in ranked:
        header += f" {r['name'][:16]:>16s}"
    print(header)
    print("  " + "-" * (26 + 17 * len(ranked)))
    for opp in OPPONENT_ROTATION:
        row = f"  {opp:<26s}"
        rates = [r["per_opp"].get(opp, 0) for r in ranked]
        mx = max(rates)
        for r in ranked:
            w = r["per_opp"].get(opp, 0)
            mk = "*" if w == mx and rates.count(mx) == 1 else " "
            row += f" {w:>14.0f}%{mk}"
        print(row)
    print("  " + "-" * (26 + 17 * len(ranked)))
    row = f"  {'OVERALL':<26s}"
    for r in ranked:
        row += f" {r['wr']:>15.1f}% "
    print(row)

    # ═══════════════════════════════════════════════════════════════
    #  WILCOXON SIGNED-RANK TEST
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  STATISTICAL SIGNIFICANCE — Wilcoxon Signed-Rank Test")
    print("  (EC-SA-TDCCEA vs SA-TDCCEA, 7 paired per-bot win rates)")
    print(f"{'='*70}")

    ec_res = next((r for r in results if r["key"] == "ec_sa"), None)
    sa_res = next((r for r in results if r["key"] == "sa"), None)

    w_out = [f"\nRun Date: {time.strftime('%Y-%m-%d %H:%M:%S')}"]
    if ec_res and sa_res:
        ec_rates = [ec_res["per_opp"].get(o, 0) for o in OPPONENT_ROTATION]
        sa_rates = [sa_res["per_opp"].get(o, 0) for o in OPPONENT_ROTATION]
        p = wilcoxon_test(ec_rates, sa_rates)
        if p is None:
            line = "  p-value: N/A (fewer than 3 non-tied differences)"
        else:
            sig = "YES (p < 0.05)" if p < 0.05 else "NO (p >= 0.05)"
            dirn = "EC-SA-TDCCEA BETTER" if sum(ec_rates) > sum(sa_rates) else "SA-TDCCEA BETTER"
            line = f"  p-value: {p:.4f}  |  Significant: {sig}  |  Direction: {dirn}"
        print(f"\n{line}")
        w_out.append(line)

        print(f"\n  {'Bot':<26s} {'SA-TDCCEA':>12s} {'EC-SA-TDCCEA':>14s} {'Δ':>8s}")
        print(f"  {'_'*26} {'_'*12} {'_'*14} {'_'*8}")
        for opp in OPPONENT_ROTATION:
            s_wr = sa_res["per_opp"].get(opp, 0)
            e_wr = ec_res["per_opp"].get(opp, 0)
            delta = e_wr - s_wr
            sign = "+" if delta >= 0 else ""
            print(f"  {opp:<26s} {s_wr:>11.0f}% {e_wr:>13.0f}% {sign}{delta:>6.0f}%")

    with open(os.path.join(SCRIPT_DIR, "ec_sa_tdccea_wilcoxon.txt"), "a", encoding="utf-8") as f:
        f.write("\n".join(w_out) + "\n\n")

    # ═══════════════════════════════════════════════════════════════
    #  ABLATION ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  ABLATION ANALYSIS (Environment-Conditioned vs Static)")
    print(f"{'='*70}")
    if ec_res and sa_res:
        diff = ec_res["wr"] - sa_res["wr"]
        lines = [
            f"  Environment-Conditioned Sensitivity contribution:",
            f"    EC-SA-TDCCEA: {ec_res['wr']:.1f}%  vs  SA-TDCCEA: {sa_res['wr']:.1f}%",
            f"    Net Gain: {'+' if diff >= 0 else ''}{diff:.1f}%",
            f"  Computation time:",
            f"    EC-SA-TDCCEA: {ec_res['time']:.1f}s",
            f"    SA-TDCCEA:    {sa_res['time']:.1f}s",
            f"    Overhead:     {ec_res['time'] - sa_res['time']:+.1f}s",
        ]
        for l in lines:
            print(f"  {l}")
        with open(os.path.join(SCRIPT_DIR, "ec_sa_tdccea_ablation.txt"), "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + "\n".join(lines) + "\n\n")

    # ═══════════════════════════════════════════════════════════════
    #  CONVERGENCE TABLE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  CONVERGENCE COMPARISON (Best score per generation)")
    print(f"{'='*70}")
    print(f"  {'Gen':>4s}  {'SA-TDCCEA':>12s}  {'EC-SA-TDCCEA':>14s}")
    print(f"  {'_'*4}  {'_'*12}  {'_'*14}")
    max_gens = min(len(results[0]["convergence"]), len(results[1]["convergence"]))
    for g in range(max_gens):
        s_val = results[0]["convergence"][g]
        e_val = results[1]["convergence"][g]
        print(f"  {g+1:4d}  {s_val:>12.3f}  {e_val:>14.3f}")

    # ═══════════════════════════════════════════════════════════════
    #  JSON & EXCEL EXPORT
    # ═══════════════════════════════════════════════════════════════
    json_path = os.path.join(SCRIPT_DIR, "ec_sa_tdccea_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([{
            "name": r["name"],
            "key": r["key"],
            "wr": r["wr"],
            "per_opp": r["per_opp"],
            "time": r["time"],
            "weights": r["weights"],
            "convergence": r["convergence"],
        } for r in results], f, indent=2)
    print(f"\n  JSON saved → {json_path}")

    if _OPENPYXL_AVAILABLE:
        xlsx_path = os.path.join(SCRIPT_DIR, "ec_sa_tdccea_comparison.xlsx")
        algo_names = [r["name"] for r in results]
        algo_wr    = {r["name"]: round(r["wr"], 2) for r in results}
        algo_time  = {r["name"]: round(r["time"], 1) for r in results}

        if not os.path.exists(xlsx_path):
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Win Rates"
            ws.append(["Algorithm", "Run1", "Time1"])
            for a in algo_names:
                ws.append([a, algo_wr[a], algo_time[a]])
            wb.save(xlsx_path)
            run_col = "Run1"
        else:
            wb = openpyxl.load_workbook(xlsx_path)
            ws = wb.active
            header_row = [cell.value for cell in ws[1]]
            run_numbers = [
                int(h[3:]) for h in header_row
                if isinstance(h, str) and h.lower().startswith("run") and h[3:].isdigit()
            ]
            next_run = (max(run_numbers) + 1) if run_numbers else 1
            run_col = f"Run{next_run}"
            time_col = f"Time{next_run}"
            r_idx = len(header_row) + 1
            t_idx = len(header_row) + 2
            ws.cell(row=1, column=r_idx, value=run_col)
            ws.cell(row=1, column=t_idx, value=time_col)

            row_map = {ws.cell(row=r, column=1).value: r for r in range(2, ws.max_row + 1)}
            for a in algo_names:
                row = row_map.get(a) or (ws.max_row + 1)
                ws.cell(row=row, column=1, value=a)
                ws.cell(row=row, column=r_idx, value=algo_wr[a])
                ws.cell(row=row, column=t_idx, value=algo_time[a])
            wb.save(xlsx_path)

        print(f"  Excel updated → {xlsx_path} (Run: {run_col})")
    else:
        print("  [SKIP] openpyxl not installed — Excel export skipped.")

    print(f"\n{'='*70}\n  BENCHMARK COMPLETE!\n{'='*70}\n")


if __name__ == "__main__":
    main()
