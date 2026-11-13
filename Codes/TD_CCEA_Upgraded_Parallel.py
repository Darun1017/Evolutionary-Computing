#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD_CCEA_Upgraded_Parallel.py
============================
Upgraded TD-CCEA ("TDDCEA_Modification.pdf") compared directly against the
baseline TD-CCEA (Original Centroid-based Trajectory-Driven Co-Evolutionary
Algorithm) from COMPARE_ALL.py, with full MULTI-CORE parallel game evaluation.

WHAT THE UPGRADE IS (from TDDCEA_Modification.pdf)
--------------------------------------------------
Original TD-CCEA describes the environment by ONE centroid of the population.
The upgraded version represents it as a collection of distinct strategic MODES:

  Algorithm 1 — BuildFingerprint:
    • Divisive 2-means clustering of population strategy vectors in normalised
      space (x / sqrt(D)).
    • Adaptive split threshold: tau_M = tau_0 * exp(lambda * (M - 1)).
      Each extra cluster is exponentially harder to justify, preventing
      fragmentation.
    • Per mode m = 1..M:
        - theta_m = ACTUAL agent nearest to the cluster centre (avoids artificial
                    non-viable centroids).
        - w_m = N_m / N (population mass / prevalence share).
        - g_m = f(theta_m) (representative agent fitness).
    • Fingerprint F_t = {(theta_m, w_m, g_m)}_{m=1}^M.

  Optimal-Transport Environmental Distance D_env(F_A, F_B):
    • Earth-Mover distance between mode distributions:
        D_env(F_A, F_B) = min sum_{m,n} pi_{mn} * d(theta_m^A, theta_n^B)
        subject to sum_n pi_{mn} = w_m^A and sum_m pi_{mn} = w_n^B.
    • Solved via exact N x N assignment (scipy.optimize.linear_sum_assignment
      with pure-Python Hungarian fallback).

  Algorithm 2 — TD-CCEA (Trajectory-Driven Competitive Co-Evolutionary Algorithm):
    • Trajectory score: e_i = sum_k alpha_hat_k * f_i^(k)
      where alpha_k = max(0, 1 - D_k / D_max), D_k = D_env(F_t, F^(k)).
    • Adaptive mutation: mu_i = mu_base * (1 - e_i) + eps.
      Offspring: theta_off ~ N(theta_parent, mu_i * I).
    • Redundancy elimination: filter agents with S(i,j) = 1 - d(i,j)/sqrt(D) > delta_s.
    • Weak-agent elimination: remove agents with e_i < delta_f and |H_i| >= 3.
    • Archiving: elite Hall of Fame (top 3 per generation, capacity A_max = 15).
    • Reproduction: tournament selection (k=3) from survivors U archive.

MULTI-CORE PARALLELISM:
-----------------------
Every generation's N_POP x K_GAMES independent games are distributed across a
pool of Java game servers (`tests.EvolvedBatchGameRunner`) via a ThreadPoolExecutor.
All evolutionary logic remains deterministic and thread-safe in the main thread.
Defaults to available CPU cores (capped at 15/60); override via CLI or env var TDUP_WORKERS.

CONTRACT FOR COMPARE_ALL.py:
----------------------------
  run_td_ccea_upgraded(server, n_workers=None) -> (best_weights[20], elapsed_s)
  run_td_ccea(server, n_workers=None)          -> (best_weights[20], elapsed_s)

STANDALONE BENCHMARK:
---------------------
  python TD_CCEA_Upgraded_Parallel.py [n_workers]
Runs both Baseline TD-CCEA and Upgraded TD-CCEA, performs validation against all
7 microRTS built-in opponents, computes Wilcoxon signed-rank test, ablation analysis,
convergence tracking, and exports JSON + Excel spreadsheets.
"""

import math
import os
import sys
import json
import time
import queue
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Force UTF-8 on Windows consoles to prevent charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:
    _lsa = None

try:
    import openpyxl
    _OPENPYXL_AVAILABLE = True
except ImportError:
    _OPENPYXL_AVAILABLE = False

os.environ["PYTHONIOENCODING"] = "utf-8"

# ===============================================================
#  CONFIGURATION & PATH DISCOVERY
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

MAP_PATH = "maps/chambers32x32.xml"
if not os.path.isfile(os.path.join(MICRORTS_DIR, MAP_PATH)):
    for _fb in ["maps/arena_chambers_32x32.xml", "maps/basesWorkers32x32A.xml"]:
        if os.path.isfile(os.path.join(MICRORTS_DIR, _fb)):
            MAP_PATH = _fb
            break

MAX_CYCLES   = 10000
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
SQRT_D = math.sqrt(D)

OPPONENT_ROTATION = [
    "EconomyMilitaryRush", "LightRush", "WorkerRush",
    "HeavyDefense", "RangedRush", "EconomyRush", "WorkerRushPlusPlus",
]
N_OPP = len(OPPONENT_ROTATION)

# Shared benchmark hyperparameters
N_POP             = 20      # N
K_GAMES           = 3       # K_eval
TOTAL_GENERATIONS = 30      # T_max
TRAJ_WINDOW       = 7       # W
MU_BASE           = 0.15    # mu_base
EPS               = 0.03    # epsilon
DELTA_F           = 0.20    # delta_f (fitness threshold)
DELTA_S           = 0.85    # delta_s (redundancy similarity threshold)
A_MAX             = 15      # A_max (archive capacity)
VALIDATION_GAMES  = 40       # Validation games per opponent
DRAW_SCORE        = 0.3     # Draw / timeout partial reward
MIN_KEEP          = 10      # Benchmark safety minimum survivors floor

# Upgraded TD-CCEA hyperparameters (TDDCEA_Modification.pdf)
TAU0  = 0.15   # tau_0: base split threshold in normalised squared distance
LAM   = 0.50   # lambda: threshold growth, tau_M = TAU0 * exp(LAM * (M - 1))
D_MAX = 0.40   # D_max: environment distance threshold where similarity falls to 0

# Mutation standard deviation interpretation:
# False -> std dev = mu_i (matching COMPARE_ALL benchmark convention)
# True  -> std dev = sqrt(mu_i) (interpreting mu_i strictly as variance)
MU_IS_VARIANCE = False

_fp_counter = 0


def next_fp_uid():
    global _fp_counter
    _fp_counter += 1
    return _fp_counter


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


# ===============================================================
#  MULTI-CORE GAME POOL & SERVER
# ===============================================================
def default_workers():
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 4
    env = os.environ.get("TDUP_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, min(cores - 1, N_POP))


class GameServer:
    """Manages an individual microRTS Java process via standard I/O pipes."""

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
    """Pool of GameServer processes; game evaluations are dispatched to idle workers."""

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
        """Evaluate a batch of candidate weights against an opponent (K_GAMES each)."""
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
        """Validate candidate weights against all 7 opponents (VALIDATION_GAMES each)."""
        futs = {o: [self.ex.submit(self._game, weights, o) for _ in range(VALIDATION_GAMES)]
                for o in OPPONENT_ROTATION}
        per_opp = {}
        total_wins = 0
        for o, fs in futs.items():
            wins = sum(1 for f in fs if f.result()[0] == 0)
            per_opp[o] = wins / VALIDATION_GAMES * 100.0
            total_wins += wins
        overall_wr = total_wins / (VALIDATION_GAMES * N_OPP) * 100.0
        return overall_wr, per_opp

    def close(self):
        self.ex.shutdown(wait=True)
        for s in self.owned_servers:
            try:
                s.stop()
            except Exception:
                pass


# ===============================================================
#  ALGORITHM 1: BuildFingerprint & 2-Means Clustering
# ===============================================================
def _sse(X, idx):
    p = X[idx]
    return float(((p - p.mean(axis=0)) ** 2).sum())


def _best_split(X, idx):
    """
    2-means split of cluster `idx`.
    Returns (delta_E, left_idx, right_idx) or None if split cannot be made.
    """
    if len(idx) < 2:
        return None
    p = X[idx]
    c = p.mean(axis=0)
    sse_parent = float(((p - c) ** 2).sum())
    if sse_parent < 1e-12:
        return None

    # Initialise centroids with furthest pair
    a = int(np.argmax(((p - c) ** 2).sum(axis=1)))
    b = int(np.argmax(((p - p[a]) ** 2).sum(axis=1)))
    if a == b:
        return None

    cen = np.stack([p[a], p[b]])
    lab = None
    for _ in range(25):  # Lloyd iterations
        dist = ((p[:, None, :] - cen[None, :, :]) ** 2).sum(axis=2)
        new = dist.argmin(axis=1)
        if lab is not None and (new == lab).all():
            break
        lab = new
        for k in (0, 1):
            mask = (lab == k)
            if mask.any():
                cen[k] = p[mask].mean(axis=0)

    left = [idx[i] for i in range(len(idx)) if lab[i] == 0]
    right = [idx[i] for i in range(len(idx)) if lab[i] == 1]
    if not left or not right:
        return None

    dE = sse_parent - _sse(X, left) - _sse(X, right)
    if dE <= 0.0:
        return None
    return dE, left, right


def build_fingerprint(params_list, fits):
    """
    Algorithm 1: Environmental fingerprint F_t = {(theta_m, w_m, g_m)}.
    Clustering operates in normalized space (x / sqrt(D)) so that Euclidean distance
    there equals normalized strategy distance d in [0, 1].
    """
    n = len(params_list)
    X = np.asarray(params_list, dtype=float) / SQRT_D
    clusters = [list(range(n))]
    splits = {0: _best_split(X, clusters[0])}

    while True:
        M = len(clusters)
        tau_M = TAU0 * math.exp(LAM * (M - 1))
        candidates = [(s[0], i) for i, s in splits.items() if s is not None]
        if not candidates:
            break
        dE, best_i = max(candidates)
        if dE <= tau_M:  # Split does not provide sufficient error reduction
            break
        _, left, right = splits[best_i]
        clusters[best_i] = left
        clusters.append(right)
        splits[best_i] = _best_split(X, left)
        splits[len(clusters) - 1] = _best_split(X, right)

    reps, counts, gs = [], [], []
    for idx in clusters:
        centre = X[idx].mean(axis=0)
        # Select actual agent nearest to cluster centre (preserves valid strategies)
        near = idx[int(np.argmin(((X[idx] - centre) ** 2).sum(axis=1)))]
        reps.append(list(params_list[near]))
        counts.append(len(idx))
        gs.append(fits[near])

    return {
        "theta": np.asarray(reps),
        "count": counts,
        "w": [c / n for c in counts],
        "g": gs,
        "uid": next_fp_uid(),
    }


# ===============================================================
#  OPTIMAL TRANSPORT ENVIRONMENTAL DISTANCE D_env
# ===============================================================
def _hungarian(C):
    """
    Min-cost perfect bipartite matching, O(n^3) implementation in pure Python.
    Used when scipy is not available.
    """
    n = len(C)
    INF = float("inf")
    u, v = [0.0] * (n + 1), [0.0] * (n + 1)
    p, way = [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = C[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return sum(C[p[j] - 1][j - 1] for j in range(1, n + 1))


def d_env(FA, FB):
    """
    Optimal transport (Earth-Mover) distance between fingerprints FA and FB.
    Masses w_m = N_m / N. Because each mass is an integer multiple of 1/N,
    the transport problem reduces to an exact N x N assignment problem over
    unit-mass copies of each mode representative.
    """
    if FA.get("uid") == FB.get("uid"):
        return 0.0

    ia = np.repeat(np.arange(len(FA["count"])), FA["count"])
    ib = np.repeat(np.arange(len(FB["count"])), FB["count"])
    if len(ia) != len(ib):
        raise ValueError("Fingerprints must cover populations of equal size")

    A, B = FA["theta"][ia], FB["theta"][ib]
    C = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2) / SQRT_D

    if _lsa is not None:
        r, c = _lsa(C)
        return float(C[r, c].sum() / len(ia))
    return _hungarian(C.tolist()) / len(ia)


# ===============================================================
#  AGENT CLASSES
# ===============================================================
class UpgradedAgent:
    """Agent for Upgraded TD-CCEA (modes fingerprint + optimal transport)."""

    def __init__(self, params=None):
        self.params = ([random.random() for _ in range(D)] if params is None
                       else [clamp01(x) for x in params])
        self.H_i  = []       # [(fitness, fingerprint)]
        self.e_i  = 0.0
        self.mu_i = 0.0

    def update_history(self, f, F):
        self.H_i.append((f, F))
        if len(self.H_i) > TRAJ_WINDOW:
            self.H_i.pop(0)

    def compute_trajectory(self, Ft, dist_cache):
        alphas = []
        for _, Fk in self.H_i:
            key = (Ft.get("uid"), Fk.get("uid"))
            if key not in dist_cache:
                dist_cache[key] = d_env(Ft, Fk)
            alphas.append(max(0.0, 1.0 - dist_cache[key] / D_MAX))
        tot = sum(alphas)
        alphas = [a / tot for a in alphas] if tot > 0 else [1.0 / len(alphas)] * len(alphas)
        self.e_i = sum(a * self.H_i[k][0] for k, a in enumerate(alphas))

    def compute_mutation(self):
        self.mu_i = MU_BASE * (1.0 - self.e_i) + EPS

    def mutate(self):
        sigma = math.sqrt(self.mu_i) if MU_IS_VARIANCE else self.mu_i
        return UpgradedAgent([self.params[j] + random.gauss(0, sigma) for j in range(D)])

    def snapshot(self):
        c = UpgradedAgent(self.params)
        c.e_i, c.H_i, c.mu_i = self.e_i, self.H_i[:], self.mu_i
        return c


class CentroidAgent:
    """Agent for Baseline TD-CCEA (single centroid environmental representation)."""

    def __init__(self, params=None):
        self.params = ([random.random() for _ in range(D)] if params is None
                       else [clamp01(x) for x in params])
        self.H_i  = []       # [(fitness, centroid)]
        self.e_i  = 0.0
        self.mu_i = 0.0

    def update_history(self, f, centroid):
        self.H_i.append((f, list(centroid)))
        if len(self.H_i) > TRAJ_WINDOW:
            self.H_i.pop(0)

    def compute_trajectory(self, centroid):
        if not self.H_i:
            self.e_i = 0.0
            return
        ws = [max(0.0, 1.0 - math.dist(centroid, pe) / SQRT_D) for _, pe in self.H_i]
        tot = sum(ws)
        ws = [w / tot for w in ws] if tot > 0 else [1.0 / len(ws)] * len(ws)
        self.e_i = sum(w * self.H_i[k][0] for k, w in enumerate(ws))

    def compute_mutation(self):
        self.mu_i = MU_BASE * (1.0 - self.e_i) + EPS

    def mutate(self):
        sigma = math.sqrt(self.mu_i) if MU_IS_VARIANCE else self.mu_i
        return CentroidAgent([self.params[j] + random.gauss(0, sigma) for j in range(D)])

    def snapshot(self):
        c = CentroidAgent(self.params)
        c.e_i, c.H_i, c.mu_i = self.e_i, self.H_i[:], self.mu_i
        return c


# ===============================================================
#  ALGORITHM 2: UPGRADED TD-CCEA RUNNER
# ===============================================================
def run_td_ccea_upgraded(server=None, n_workers=None, return_history=False):
    """
    Runs the Upgraded TD-CCEA algorithm from TDDCEA_Modification.pdf.
    Contract: run_td_ccea_upgraded(server, n_workers=None) -> (best_weights, elapsed_s)
    """
    print(f"\n{'='*70}")
    print("  RUNNING: Upgraded TD-CCEA (Strategic Modes + Optimal Transport)")
    print(f"{'='*70}")
    t_start = time.time()

    if isinstance(server, ServerPool):
        pool = server
        owns_pool = False
    else:
        pool = ServerPool(base_server=server, n_workers=n_workers)
        owns_pool = True

    print(f"  Parallel game workers: {pool.n}")

    try:
        pop = [UpgradedAgent() for _ in range(N_POP)]
        archive = []
        convergence = []
        modes_history = []

        for t in range(1, TOTAL_GENERATIONS + 1):
            opp = OPPONENT_ROTATION[(t - 1) % N_OPP]

            # ---- Phase 1: Parallel Evaluation & Mode Fingerprint ----
            fits = pool.evaluate_many([a.params for a in pop], opp)
            Ft = build_fingerprint([a.params for a in pop], fits)
            modes_count = len(Ft["count"])
            modes_history.append({"gen": t, "modes": modes_count, "sizes": sorted(Ft["count"], reverse=True)})

            # ---- Phase 2: Trajectory Scoring & Adaptive Mutation ----
            dist_cache = {}
            for a, f in zip(pop, fits):
                a.update_history(f, Ft)
            for a in pop:
                a.compute_trajectory(Ft, dist_cache)
                a.compute_mutation()

            best = max(pop, key=lambda a: a.e_i)
            best_fit = max(fits)
            convergence.append(best_fit)

            print(f"  Gen {t:2d}/{TOTAL_GENERATIONS} | {opp:25s} | e_i={best.e_i:.3f} "
                  f"Best={best_fit:.3f} | modes M={modes_count} "
                  f"sizes={sorted(Ft['count'], reverse=True)} games={pool.games}")

            # ---- Phase 3: Selection and Archiving ----
            ranked = sorted(pop, key=lambda a: a.e_i, reverse=True)
            survivors = []
            for a in ranked:
                if not any(1.0 - math.dist(a.params, u.params) / SQRT_D > DELTA_S
                           for u in survivors):
                    survivors.append(a)

            final = [a for a in survivors if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
            for a in survivors:
                if len(final) >= min(MIN_KEEP, len(survivors)):
                    break
                if a not in final:
                    final.append(a)
            final = final[:N_POP]

            # Insert top 3 survivors into archive
            for top in sorted(final, key=lambda a: a.e_i, reverse=True)[:3]:
                archive.append(top.snapshot())
            archive.sort(key=lambda a: a.e_i, reverse=True)
            del archive[A_MAX:]

            # ---- Phase 4: Reproduction ----
            breeding_pool = final + archive
            next_pop = final[:]
            while len(next_pop) < N_POP:
                candidates = random.sample(breeding_pool, min(3, len(breeding_pool)))
                parent = max(candidates, key=lambda a: a.e_i)
                next_pop.append(parent.mutate())
            pop = next_pop

        champion = max(archive, key=lambda a: a.e_i) if archive else max(pop, key=lambda a: a.e_i)
        elapsed = time.time() - t_start

        if return_history:
            return list(champion.params), elapsed, convergence, modes_history
        return list(champion.params), elapsed

    finally:
        if owns_pool:
            pool.close()


# ===============================================================
#  BASELINE TD-CCEA RUNNER (Centroid-Based)
# ===============================================================
def run_td_ccea(server=None, n_workers=None, return_history=False):
    """
    Runs the Baseline TD-CCEA algorithm (Centroid-based) from COMPARE_ALL.py.
    Contract: run_td_ccea(server, n_workers=None) -> (best_weights, elapsed_s)
    """
    print(f"\n{'='*70}")
    print("  RUNNING: Baseline TD-CCEA (Single Population Centroid)")
    print(f"{'='*70}")
    t_start = time.time()

    if isinstance(server, ServerPool):
        pool = server
        owns_pool = False
    else:
        pool = ServerPool(base_server=server, n_workers=n_workers)
        owns_pool = True

    print(f"  Parallel game workers: {pool.n}")

    try:
        pop = [CentroidAgent() for _ in range(N_POP)]
        archive = []
        convergence = []

        for t in range(1, TOTAL_GENERATIONS + 1):
            opp = OPPONENT_ROTATION[(t - 1) % N_OPP]
            centroid = [sum(a.params[j] for a in pop) / len(pop) for j in range(D)]

            # ---- Phase 1: Parallel Evaluation ----
            fits = pool.evaluate_many([a.params for a in pop], opp)

            # ---- Phase 2: Trajectory Scoring & Adaptive Mutation ----
            for a, f in zip(pop, fits):
                a.update_history(f, centroid)
            for a in pop:
                a.compute_trajectory(centroid)
                a.compute_mutation()

            best = max(pop, key=lambda a: a.e_i)
            best_fit = max(fits)
            convergence.append(best_fit)

            print(f"  Gen {t:2d}/{TOTAL_GENERATIONS} | {opp:25s} | e_i={best.e_i:.3f} "
                  f"Best={best_fit:.3f} | centroid representation | games={pool.games}")

            # ---- Phase 3: Selection and Archiving ----
            ranked = sorted(pop, key=lambda a: a.e_i, reverse=True)
            survivors = []
            for a in ranked:
                if not any(1.0 - math.dist(a.params, u.params) / SQRT_D > DELTA_S
                           for u in survivors):
                    survivors.append(a)

            final = [a for a in survivors if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
            for a in survivors:
                if len(final) >= min(MIN_KEEP, len(survivors)):
                    break
                if a not in final:
                    final.append(a)
            final = final[:N_POP]

            # Insert top 3 survivors into archive
            for top in sorted(final, key=lambda a: a.e_i, reverse=True)[:3]:
                archive.append(top.snapshot())
            archive.sort(key=lambda a: a.e_i, reverse=True)
            del archive[A_MAX:]

            # ---- Phase 4: Reproduction ----
            breeding_pool = final + archive
            next_pop = final[:]
            while len(next_pop) < N_POP:
                candidates = random.sample(breeding_pool, min(3, len(breeding_pool)))
                parent = max(candidates, key=lambda a: a.e_i)
                next_pop.append(parent.mutate())
            pop = next_pop

        champion = max(archive, key=lambda a: a.e_i) if archive else max(pop, key=lambda a: a.e_i)
        elapsed = time.time() - t_start

        if return_history:
            return list(champion.params), elapsed, convergence
        return list(champion.params), elapsed

    finally:
        if owns_pool:
            pool.close()


# ===============================================================
#  WILCOXON SIGNED-RANK TEST (Two-Sided)
# ===============================================================
def wilcoxon_test(x, y):
    """
    Performs a two-sided Wilcoxon signed-rank test on paired per-opponent win rates.
    Returns p-value or None if fewer than 3 non-tied differences exist.
    """
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
#  MAIN BENCHMARK & COMPARATIVE EVALUATION
# ===============================================================
def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else default_workers()

    print("\n" + "#" * 72)
    print("  TD-CCEA UPGRADED vs TD-CCEA BASELINE BENCHMARK")
    print("  Optimal Transport Clustered Strategic Modes vs Single Centroid")
    print("#" * 72)
    print(f"  Pop={N_POP} | Gens={TOTAL_GENERATIONS} | K={K_GAMES} | Validation={VALIDATION_GAMES}/bot")
    print(f"  Map: {MAP_PATH} (32x32)")
    print(f"  Parallel game workers: {nw} (of {os.cpu_count()} CPU cores)")
    print(f"  Optimal transport solver: {'scipy linear_sum_assignment' if _lsa is not None else 'pure-Python Hungarian'}")
    print()

    pool = ServerPool(n_workers=nw)
    results = []

    try:
        # -------------------------------------------------------------
        # 1. Baseline TD-CCEA (Centroid)
        # -------------------------------------------------------------
        best_base, time_base, conv_base = run_td_ccea(pool, return_history=True)
        print("\n  Validating Baseline TD-CCEA against all 7 bots...")
        wr_base, opp_base = pool.validate(best_base)
        print(f"  Baseline TD-CCEA: {wr_base:.1f}% overall in {time_base:.1f}s")
        results.append({
            "name": "TD-CCEA (Baseline)",
            "key": "baseline",
            "time": time_base,
            "wr": wr_base,
            "per_opp": opp_base,
            "weights": best_base,
            "convergence": conv_base,
            "modes_history": None,
        })

        # -------------------------------------------------------------
        # 2. Upgraded TD-CCEA (Modes + Optimal Transport)
        # -------------------------------------------------------------
        best_up, time_up, conv_up, modes_up = run_td_ccea_upgraded(pool, return_history=True)
        print("\n  Validating Upgraded TD-CCEA against all 7 bots...")
        wr_up, opp_up = pool.validate(best_up)
        print(f"  Upgraded TD-CCEA: {wr_up:.1f}% overall in {time_up:.1f}s")
        results.append({
            "name": "TD-CCEA (Upgraded)",
            "key": "upgraded",
            "time": time_up,
            "wr": wr_up,
            "per_opp": opp_up,
            "weights": best_up,
            "convergence": conv_up,
            "modes_history": modes_up,
        })

    finally:
        pool.close()

    # ═══════════════════════════════════════════════════════════════
    #  FINAL RANKING TABLE
    # ═══════════════════════════════════════════════════════════════
    ranked = sorted(results, key=lambda r: r["wr"], reverse=True)
    medals = ["1st", "2nd"]
    print(f"\n{'='*72}")
    print("  FINAL RANKING — Win Rate vs 7 microRTS Bots")
    print(f"{'='*72}")
    print(f"  {'Rank':<6s} {'Algorithm':<24s} {'Win Rate':>10s} {'Time':>9s}")
    print(f"  {'_'*6} {'_'*24} {'_'*10} {'_'*9}")
    for i, r in enumerate(ranked):
        tag = "  ◄ Proposed (Modes + OT)" if r["key"] == "upgraded" else "  ◄ Baseline (Centroid)"
        print(f"  {medals[i]:<6s} {r['name']:<24s} {r['wr']:>9.1f}%  {r['time']:>8.1f}s{tag}")

    # ═══════════════════════════════════════════════════════════════
    #  PER-BOT BREAKDOWN TABLE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("  WIN RATE PER BOT (%)")
    print(f"{'='*72}")
    header = f"  {'Bot':<24s}"
    for r in ranked:
        header += f" {r['name'][:18]:>18s}"
    print(header)
    print("  " + "-" * (24 + 19 * len(ranked)))
    for opp in OPPONENT_ROTATION:
        row = f"  {opp:<24s}"
        rates = [r["per_opp"].get(opp, 0.0) for r in ranked]
        mx = max(rates)
        for r in ranked:
            w = r["per_opp"].get(opp, 0.0)
            mk = "*" if w == mx and rates.count(mx) == 1 else " "
            row += f" {w:>16.0f}%{mk}"
        print(row)
    print("  " + "-" * (24 + 19 * len(ranked)))
    row = f"  {'OVERALL':<24s}"
    for r in ranked:
        row += f" {r['wr']:>17.1f}% "
    print(row)

    # ═══════════════════════════════════════════════════════════════
    #  WILCOXON SIGNED-RANK TEST
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("  STATISTICAL SIGNIFICANCE — Wilcoxon Signed-Rank Test")
    print("  (TD-CCEA Upgraded vs TD-CCEA Baseline, 7 paired opponent samples)")
    print(f"{'='*72}")

    up_res = next((r for r in results if r["key"] == "upgraded"), None)
    base_res = next((r for r in results if r["key"] == "baseline"), None)

    w_out = [f"\nRun Date: {time.strftime('%Y-%m-%d %H:%M:%S')}"]
    if up_res and base_res:
        up_rates = [up_res["per_opp"].get(o, 0.0) for o in OPPONENT_ROTATION]
        base_rates = [base_res["per_opp"].get(o, 0.0) for o in OPPONENT_ROTATION]
        p = wilcoxon_test(up_rates, base_rates)
        if p is None:
            line = "  p-value: N/A (fewer than 3 non-tied differences)"
        else:
            sig = "YES (p < 0.05)" if p < 0.05 else "NO (p >= 0.05)"
            dirn = "TD-CCEA UPGRADED BETTER" if sum(up_rates) > sum(base_rates) else "BASELINE BETTER"
            line = f"  p-value: {p:.4f}  |  Significant: {sig}  |  Direction: {dirn}"
        print(f"\n{line}")
        w_out.append(line)

        print(f"\n  {'Bot':<24s} {'Baseline':>12s} {'Upgraded':>14s} {'Delta':>8s}")
        print(f"  {'_'*24} {'_'*12} {'_'*14} {'_'*8}")
        for opp in OPPONENT_ROTATION:
            b_wr = base_res["per_opp"].get(opp, 0.0)
            u_wr = up_res["per_opp"].get(opp, 0.0)
            delta = u_wr - b_wr
            sign = "+" if delta >= 0 else ""
            print(f"  {opp:<24s} {b_wr:>11.0f}% {u_wr:>13.0f}% {sign}{delta:>6.0f}%")

    wilc_path = os.path.join(SCRIPT_DIR, "td_ccea_upgraded_wilcoxon.txt")
    with open(wilc_path, "a", encoding="utf-8") as f:
        f.write("\n".join(w_out) + "\n\n")

    # ═══════════════════════════════════════════════════════════════
    #  ABLATION & STRATEGIC MODES ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("  ABLATION ANALYSIS (Multi-Modal Fingerprint vs Single Centroid)")
    print(f"{'='*72}")
    if up_res and base_res:
        diff_wr = up_res["wr"] - base_res["wr"]
        time_oh = up_res["time"] - base_res["time"]
        mode_counts = [m["modes"] for m in up_res["modes_history"]] if up_res["modes_history"] else []
        avg_modes = sum(mode_counts) / len(mode_counts) if mode_counts else 1.0

        lines = [
            "  Component Ablation Summary:",
            f"    Upgraded TD-CCEA (Modes + Optimal Transport): {up_res['wr']:.1f}% win rate",
            f"    Baseline TD-CCEA (Single Population Centroid): {base_res['wr']:.1f}% win rate",
            f"    Net Win Rate Gain: {'+' if diff_wr >= 0 else ''}{diff_wr:.1f}%",
            "  Execution Efficiency:",
            f"    Upgraded TD-CCEA: {up_res['time']:.1f}s",
            f"    Baseline TD-CCEA: {base_res['time']:.1f}s",
            f"    Clustering & Transport Overhead: {time_oh:+.1f}s",
            "  Strategic Mode Dynamics:",
            f"    Average strategic modes per generation: {avg_modes:.2f}",
            f"    Mode count range: min={min(mode_counts) if mode_counts else 1}, max={max(mode_counts) if mode_counts else 1}",
        ]
        for l in lines:
            print(f"  {l}")

        ablation_path = os.path.join(SCRIPT_DIR, "td_ccea_upgraded_ablation.txt")
        with open(ablation_path, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + "\n".join(lines) + "\n\n")

    # ═══════════════════════════════════════════════════════════════
    #  CONVERGENCE TABLE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("  CONVERGENCE COMPARISON (Best fitness per generation)")
    print(f"{'='*72}")
    print(f"  {'Gen':>4s}  {'Baseline TD-CCEA':>18s}  {'Upgraded TD-CCEA':>18s}")
    print(f"  {'_'*4}  {'_'*18}  {'_'*18}")
    max_gens = min(len(results[0]["convergence"]), len(results[1]["convergence"]))
    for g in range(max_gens):
        b_val = results[0]["convergence"][g]
        u_val = results[1]["convergence"][g]
        print(f"  {g+1:4d}  {b_val:>18.3f}  {u_val:>18.3f}")

    # ═══════════════════════════════════════════════════════════════
    #  JSON & EXCEL EXPORTS
    # ═══════════════════════════════════════════════════════════════
    json_path = os.path.join(SCRIPT_DIR, "td_ccea_upgraded_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([{
            "name": r["name"],
            "key": r["key"],
            "wr": r["wr"],
            "per_opp": r["per_opp"],
            "time": r["time"],
            "weights": r["weights"],
            "convergence": r["convergence"],
            "modes_history": r["modes_history"],
        } for r in results], f, indent=2)
    print(f"\n  JSON saved -> {json_path}")

    if _OPENPYXL_AVAILABLE:
        xlsx_path = os.path.join(SCRIPT_DIR, "td_ccea_upgraded_comparison.xlsx")
        algo_names = [r["name"] for r in results]
        algo_wr    = {r["name"]: round(r["wr"], 2) for r in results}
        algo_time  = {r["name"]: round(r["time"], 1) for r in results}

        if not os.path.exists(xlsx_path):
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Summary"
            ws.append(["Algorithm", "Run1_WinRate", "Run1_Time"])
            for a in algo_names:
                ws.append([a, algo_wr[a], algo_time[a]])
            wb.save(xlsx_path)
            run_col = "Run1"
        else:
            wb = openpyxl.load_workbook(xlsx_path)
            ws = wb["Summary"] if "Summary" in wb.sheetnames else wb.active
            header_row = [cell.value for cell in ws[1]]
            run_numbers = [
                int(h.split("_")[0][3:]) for h in header_row
                if isinstance(h, str) and h.lower().startswith("run") and "_" in h and h.split("_")[0][3:].isdigit()
            ]
            next_run = (max(run_numbers) + 1) if run_numbers else 1
            run_col = f"Run{next_run}"
            time_col = f"Time{next_run}"
            r_idx = len(header_row) + 1
            t_idx = len(header_row) + 2
            ws.cell(row=1, column=r_idx, value=f"{run_col}_WinRate")
            ws.cell(row=1, column=t_idx, value=f"{run_col}_{time_col}")

            row_map = {ws.cell(row=r, column=1).value: r for r in range(2, ws.max_row + 1)}
            for a in algo_names:
                row = row_map.get(a) or (ws.max_row + 1)
                ws.cell(row=row, column=1, value=a)
                ws.cell(row=row, column=r_idx, value=algo_wr[a])
                ws.cell(row=row, column=t_idx, value=algo_time[a])
            wb.save(xlsx_path)

        print(f"  Excel updated -> {xlsx_path} (Run: {run_col})")
    else:
        print("  [SKIP] openpyxl not installed — Excel export skipped.")

    print(f"\n{'='*72}\n  BENCHMARK COMPLETE!\n{'='*72}\n")


if __name__ == "__main__":
    main()

