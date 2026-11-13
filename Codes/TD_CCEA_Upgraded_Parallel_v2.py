#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD_CCEA_Benchmark.py
====================
Corrected, controlled comparison of the BASELINE TD-CCEA (centroid environment)
and the UPGRADED TD-CCEA (clustered strategic modes + optimal-transport distance,
"TDDCEA_Modification.pdf") on the COMPARE_ALL.py setup, with multi-core game
evaluation, paired multi-trial runs and proper statistics.

WHY THIS FILE EXISTS (problems found in the previous TD_CCEA_Upgraded_Parallel.py)
-------------------------------------------------------------------------------
1. Kernel confound.  In the baseline, past populations look 94-99% similar to the
   current one, so its trajectory score is just the plain average of the last 7
   fitness values. The upgrade's weights are clearly non-uniform because of the
   hyperparameter D_MAX. A win-rate gain could therefore come from the multi-mode
   fingerprint OR from the sharper weighting. This file adds controls that
   separate the two, and REPORTS how uneven each variant's weighting really is.
2. Silent failures: a dead / hung / garbled Java process was scored as a draw.
   Now it is detected, the server restarted, the game retried, and the run
   aborts loudly if it still fails. stderr no longer goes to an unread pipe.
3. Setup mismatch: default map was chambers32x32.xml (benchmark uses
   arena_chambers_32x32.xml) with a silent fallback, and a pool built around the
   COMPARE_ALL server mixed two configurations. Now one fixed setup, pre-flight
   checked (java, class file, map file).
4. Statistics: one run, 35 validation games, Wilcoxon over 7 bots. Now: paired
   multi-trial runs (shared initial population, randomised algorithm order),
   20 validation games per bot, exact Wilcoxon across TRIALS, Holm correction,
   paired bootstrap 95% CI.
5. The printed "overhead" (wall-time difference) was noise; real cost of the
   fingerprint + transport is ~2 ms per generation. Per-run compute cost of
   those parts is now timed directly. The convergence table (best of 20 noisy
   3-game scores) is no longer presented as evidence.

VARIANTS (one shared code path; only the listed switches differ)
----------------------------------------------------------------
  key       name                  environment      weighting of past entries
  td        TD-CCEA               centroid         alpha = 1 - dist/sqrt(D)      (COMPARE_ALL baseline)
  td_sharp  TD-CCEA+SharpKernel   centroid         alpha = 1 - (dist/sqrt(D))/DC_MAX   <- control: sharper weights, no new representation
  up        TD-CCEA-Upgraded      clustered modes  alpha = 1 - D_env/D_MAX, D_MAX=0.40 <- proposed
  up_d02    Upgraded Dmax=0.2     clustered modes  D_MAX = 0.20                        <- sensitivity
  up_d08    Upgraded Dmax=0.8     clustered modes  D_MAX = 0.80                        <- sensitivity
  up_1mode  Upgraded 1-mode       ONE mode (nearest actual agent to the centroid)      <- control: is "multi-mode" what matters?

Default run = the three core variants (td, td_sharp, up). Use --algos all for all six.

USAGE
-----
  python TD_CCEA_Benchmark.py --trials 20                       # core variants
  python TD_CCEA_Benchmark.py --trials 10 --algos all           # + D_MAX sweep and 1-mode control
  python TD_CCEA_Benchmark.py --trials 20 --algos td,up         # head-to-head only
  python TD_CCEA_Benchmark.py --trials 10 --resume              # continue a saved run
  python TD_CCEA_Benchmark.py --mock --trials 3 --generations 8 # self-test, no Java
  Workers: --workers N (default CPU cores - 1, max 60) or env TDUP_WORKERS.

COMPARE_ALL.py INTEGRATION (same contract as the other algorithms)
  from TD_CCEA_Benchmark import run_td_ccea, run_td_ccea_upgraded
  ... elif key == "td_up": best, elapsed = run_td_ccea_upgraded(server)
  (they start their own pool of robust servers; the `server` argument is unused,
   so one configuration is used for every game.)

Requires numpy. scipy is optional (used for the transport solver if present).
"""

import argparse
import csv
import json
import math
import os
import queue
import random
import shutil
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:
    _lsa = None

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ===============================================================
#  SETUP  (identical to COMPARE_ALL.py)
# ===============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_microrts():
    cands = []
    if os.environ.get("MICRORTS_DIR"):
        cands.append(os.environ["MICRORTS_DIR"])
    cands += [
        os.path.join(os.path.dirname(SCRIPT_DIR), "Simulation", "microrts"),   # COMPARE_ALL.py rule
        os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "Simulation", "microrts"),
        os.path.join(SCRIPT_DIR, "Simulation", "microrts"),
    ]
    for c in cands:
        if os.path.isfile(os.path.join(c, "bin", "tests", "EvolvedBatchGameRunner.class")):
            return c
    return cands[0]      # not found: GameServer.preflight() reports it clearly before any game


MICRORTS_DIR = _find_microrts()
MAP_PATH     = "maps/arena_chambers_32x32.xml"
MAX_CYCLES   = 10000
JAVA_CMD     = "java"

WEIGHT_NAMES = [
    "worker_ratio", "max_workers", "expand_threshold", "light_preference",
    "heavy_preference", "ranged_preference", "barracks_timing", "aggression",
    "attack_threshold", "defend_radius", "target_priority", "defense_trigger",
    "counter_rush", "tech_switch", "economy_pressure", "scout_priority",
    "group_attack", "second_barracks", "worker_fight_ratio", "focus_fire",
]
DIMENSIONS = len(WEIGHT_NAMES)
D = DIMENSIONS
SQRT_D = math.sqrt(D)
OPPONENT_ROTATION = [
    "EconomyMilitaryRush", "LightRush", "WorkerRush",
    "HeavyDefense", "RangedRush", "EconomyRush", "WorkerRushPlusPlus",
]
N_OPP = len(OPPONENT_ROTATION)

N_POP             = 20
K_GAMES           = 3
TOTAL_GENERATIONS = 30
A_MAX             = 15
DELTA_S           = 0.85
DELTA_F           = 0.20
TRAJ_WINDOW       = 7
MU_BASE           = 0.15
EPS               = 0.03
MIN_KEEP          = 10          # COMPARE_ALL.evolve_core keeps >= 10 survivors
DRAW_SCORE        = 0.3         # training score of a true draw / timeout game

# --- game-server robustness
START_TIMEOUT = 180.0
GAME_TIMEOUT  = 300.0
MAX_RETRIES   = 3

# --- upgraded fingerprint hyperparameters (values are mine; the PDF names but does not fix them)
TAU0  = 0.15      # tau_0: base split threshold (sum of squared normalised distances)
LAM   = 0.50      # lambda: tau_M = TAU0 * exp(LAM * (M-1))
D_MAX = 0.40      # D_max of the upgraded weighting alpha = max(0, 1 - D_env / D_MAX)
DC_MAX_SHARP = 0.09   # sharpened centroid kernel: alpha = max(0, 1 - (dist/sqrt(D)) / DC_MAX_SHARP)
                      # (0.09 matched the upgraded variant's weight evenness, ~0.82, on a mock run; the
                      #  real game's dynamics differ, so CHECK the 'evenness' column of the report and
                      #  retune with --dc-max until td_sharp and up have similar values)

MU_IS_VARIANCE = False     # False: step sigma = mu_i (COMPARE_ALL convention); True: sigma = sqrt(mu_i)
ARCHIVE_FROM_SURVIVORS = False   # False = COMPARE_ALL (top-3 of population); True = PDF wording
VERBOSE = True


# ===============================================================
#  ROBUST GAME SERVER
# ===============================================================
class GameServerError(RuntimeError):
    pass


class GameServer:
    """One microRTS Java process. Raises GameServerError on ANY failure (never fakes a result)."""
    _preflight_done = False

    def __init__(self):
        self.process = None
        self._q = None

    @staticmethod
    def preflight():
        if GameServer._preflight_done:
            return
        problems = []
        if shutil.which(JAVA_CMD) is None:
            problems.append(f"'{JAVA_CMD}' not found on PATH")
        if not os.path.isfile(os.path.join(MICRORTS_DIR, "bin", "tests", "EvolvedBatchGameRunner.class")):
            problems.append(f"compiled tests/EvolvedBatchGameRunner.class not found under {MICRORTS_DIR}/bin "
                            f"(set env MICRORTS_DIR to your microrts folder)")
        if not os.path.isfile(os.path.join(MICRORTS_DIR, MAP_PATH)):
            problems.append(f"map file not found: {os.path.join(MICRORTS_DIR, MAP_PATH)}")
        if problems:
            raise GameServerError("pre-flight check failed:\n  - " + "\n  - ".join(problems))
        GameServer._preflight_done = True

    def _get(self, timeout):
        try:
            item = self._q.get(timeout=timeout)
        except queue.Empty:
            raise GameServerError(f"no output from server within {timeout:.0f}s")
        if item is None:
            self._q.put(None)
            raise GameServerError("server process exited")
        return item

    def start(self):
        GameServer.preflight()
        self.process = subprocess.Popen(
            [JAVA_CMD, "-cp", os.pathsep.join(["lib/*", "bin"]), "tests.EvolvedBatchGameRunner"],
            cwd=MICRORTS_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", bufsize=1)
        q = self._q = queue.Queue()
        proc = self.process

        def _reader():
            try:
                for line in proc.stdout:
                    q.put(line.strip())
            except Exception:
                pass
            q.put(None)

        threading.Thread(target=_reader, daemon=True).start()
        deadline = time.time() + START_TIMEOUT
        while True:
            line = self._get(max(0.1, deadline - time.time()))
            if line == "READY":
                return

    def run_game(self, weights, opponent):
        if self.process is None or self.process.poll() is not None:
            raise GameServerError("server is not running")
        ws = ";".join(f"{float(w):.4f}" for w in weights)
        try:
            self.process.stdin.write(f"EVOLVED:{ws},{opponent},{MAP_PATH},{MAX_CYCLES}\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as e:
            raise GameServerError(f"cannot write to server: {e}")
        t_end = time.time() + GAME_TIMEOUT
        while True:
            line = self._get(max(0.1, t_end - time.time()))
            if line.startswith("RESULT:"):
                try:
                    p = line[len("RESULT:"):].split(",")
                    return int(p[0]), int(p[1])
                except (ValueError, IndexError):
                    raise GameServerError(f"malformed result line: {line!r}")
            if line.upper().startswith("ERROR") or "Exception" in line:
                raise GameServerError(f"server reported: {line}")
            # any other line is ignored

    def stop(self):
        p, self.process = self.process, None
        if p is None:
            return
        try:
            p.stdin.write("EXIT\n")
            p.stdin.flush()
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
                p.wait(timeout=5)
            except Exception:
                pass
        finally:
            for stream in (p.stdin, p.stdout):
                try:
                    stream.close()
                except Exception:
                    pass


class MockGameServer:
    """Stand-in for the Java server (--mock): lets you test the whole pipeline without microRTS."""
    latency = 0.0
    lock = threading.Lock()

    def start(self): pass
    def stop(self): pass

    def run_game(self, w, opp):
        if MockGameServer.latency:
            time.sleep(MockGameServer.latency)
        p = 0.05 + 0.85 * (0.5 * w[7] + 0.3 * w[5] + 0.2 * (1 - w[9])) * (0.8 + 0.2 * w[3] * w[8])
        with MockGameServer.lock:
            r = random.random()
        return (0, 100) if r < p else ((-1, MAX_CYCLES) if r < p + 0.05 else (1, 100))


def default_workers():
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 2
    env = os.environ.get("TDUP_WORKERS")
    if env and env.isdigit():
        return max(1, int(env))
    return max(1, min(cores - 1, N_POP * K_GAMES))


class ServerPool:
    """Pool of servers; one game = one task on whichever server is free."""

    def __init__(self, n_workers=None, factory=GameServer):
        self.n = n_workers or default_workers()
        self.factory = factory
        self.servers, self.q = [], queue.Queue()
        self.games = 0
        self.restarts = 0
        self._lock = threading.Lock()
        try:
            with ThreadPoolExecutor(max_workers=self.n) as ex:         # start JVMs in parallel
                futs = [ex.submit(self._spawn) for _ in range(self.n)]
                for f in futs:
                    s = f.result()
                    self.servers.append(s)
                    self.q.put(s)
        except Exception:
            self.close()
            raise
        self.ex = ThreadPoolExecutor(max_workers=self.n)

    def _spawn(self):
        s = self.factory()
        s.start()
        return s

    def _game(self, weights, opp):
        s = self.q.get()
        try:
            last = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    return s.run_game(weights, opp)
                except Exception as e:
                    last = e
                    with self._lock:
                        self.restarts += 1
                    print(f"  [WARN] game failed (attempt {attempt}/{MAX_RETRIES}): {e} -> restarting server")
                    try:
                        s.stop()
                    except Exception:
                        pass
                    try:
                        s.start()
                    except Exception as e2:
                        last = e2
            raise GameServerError(f"game failed {MAX_RETRIES} times, last error: {last}")
        finally:
            self.q.put(s)

    def _run_batch(self, jobs):
        """jobs: list of (weights, opp). Returns list of (winner, cycles); aborts on any failure."""
        futs = [self.ex.submit(self._game, w, o) for w, o in jobs]
        try:
            return [f.result() for f in futs]
        except Exception:
            for f in futs:
                f.cancel()
            raise

    def evaluate_many(self, weight_list, opp):
        """Training fitness per vector: win=1, draw/timeout=0.3, loss=0, averaged over K_GAMES."""
        res = self._run_batch([(w, opp) for w in weight_list for _ in range(K_GAMES)])
        self.games += len(res)
        out = []
        for i in range(len(weight_list)):
            sc = 0.0
            for winner, _ in res[i * K_GAMES:(i + 1) * K_GAMES]:
                sc += 1.0 if winner == 0 else (DRAW_SCORE if winner == -1 else 0.0)
            out.append(sc / K_GAMES)
        return out

    def validate(self, weights, games_per_bot):
        """Real wins only (draw = 0), like COMPARE_ALL.validate_weights. Returns (wins, total, {opp: wins})."""
        jobs = [(weights, o) for o in OPPONENT_ROTATION for _ in range(games_per_bot)]
        res = self._run_batch(jobs)
        per, tw = {}, 0
        for i, o in enumerate(OPPONENT_ROTATION):
            w = sum(1 for winner, _ in res[i * games_per_bot:(i + 1) * games_per_bot] if winner == 0)
            per[o] = w
            tw += w
        return tw, len(jobs), per

    def close(self):
        if hasattr(self, "ex"):
            self.ex.shutdown(wait=True)
        for s in self.servers:
            try:
                s.stop()
            except Exception:
                pass


# ===============================================================
#  SMALL HELPERS
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
#  ALGORITHM 1 - BuildFingerprint  (divisive clustering into strategic modes)
# ===============================================================
_fp_counter = 0


def _next_uid():
    global _fp_counter
    _fp_counter += 1
    return _fp_counter


def _sse(X, idx):
    p = X[idx]
    return float(((p - p.mean(axis=0)) ** 2).sum())


def _best_split(X, idx):
    """2-means split of one cluster. Returns (delta_E, left_idx, right_idx) or None."""
    if len(idx) < 2:
        return None
    p = X[idx]
    c = p.mean(axis=0)
    sse_parent = float(((p - c) ** 2).sum())
    if sse_parent < 1e-12:
        return None
    a = int(np.argmax(((p - c) ** 2).sum(axis=1)))
    b = int(np.argmax(((p - p[a]) ** 2).sum(axis=1)))
    if a == b:
        return None
    cen = np.stack([p[a], p[b]])
    lab = None
    for _ in range(25):
        dist = ((p[:, None, :] - cen[None, :, :]) ** 2).sum(axis=2)
        new = dist.argmin(axis=1)
        if lab is not None and (new == lab).all():
            break
        lab = new
        for k in (0, 1):
            if (lab == k).any():
                cen[k] = p[lab == k].mean(axis=0)
    left = [idx[i] for i in range(len(idx)) if lab[i] == 0]
    right = [idx[i] for i in range(len(idx)) if lab[i] == 1]
    if not left or not right:
        return None
    dE = sse_parent - _sse(X, left) - _sse(X, right)
    return (dE, left, right) if dE > 0 else None


def build_fingerprint(params_list, fits, tau0=TAU0, lam=LAM):
    """F_t = {(theta_m, w_m, g_m)}. Clustering runs in normalised space (x / sqrt(D)), so Euclidean
    distance there IS the normalised strategy distance d in [0,1]."""
    n = len(params_list)
    X = np.asarray(params_list, dtype=float) / SQRT_D
    clusters = [list(range(n))]
    splits = {0: _best_split(X, clusters[0])}
    while True:
        tau_M = tau0 * math.exp(lam * (len(clusters) - 1))
        cand = [(s[0], i) for i, s in splits.items() if s is not None]
        if not cand:
            break
        dE, i = max(cand)
        if dE <= tau_M:
            break
        _, left, right = splits[i]
        clusters[i] = left
        clusters.append(right)
        splits[i] = _best_split(X, left)
        splits[len(clusters) - 1] = _best_split(X, right)
    reps, counts, gs = [], [], []
    for idx in clusters:
        centre = X[idx].mean(axis=0)
        near = idx[int(np.argmin(((X[idx] - centre) ** 2).sum(axis=1)))]    # ACTUAL agent
        reps.append(list(params_list[near]))
        counts.append(len(idx))
        gs.append(fits[near])
    return {"theta": np.asarray(reps), "count": counts, "w": [c / n for c in counts],
            "g": gs, "uid": _next_uid()}


# ===============================================================
#  ENVIRONMENTAL DISTANCE D_env  (optimal transport, solved exactly)
# ===============================================================
def _hungarian(C):
    """Min-cost perfect assignment, O(n^3) (used only when scipy is unavailable)."""
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
    """Earth-mover distance with masses w_m = N_m/N and cost d(theta_m, theta_n). Every mass is a
    multiple of 1/N, so the transport problem is solved EXACTLY as an N x N assignment problem."""
    if FA["uid"] == FB["uid"]:
        return 0.0
    ia = np.repeat(np.arange(len(FA["count"])), FA["count"])
    ib = np.repeat(np.arange(len(FB["count"])), FB["count"])
    if len(ia) != len(ib):
        raise ValueError("fingerprints must cover populations of equal size")
    C = np.linalg.norm(FA["theta"][ia][:, None, :] - FB["theta"][ib][None, :, :], axis=2) / SQRT_D
    if _lsa is not None:
        r, c = _lsa(C)
        return float(C[r, c].sum() / len(ia))
    return _hungarian(C.tolist()) / len(ia)


# ===============================================================
#  AGENT  (identical for every variant; the variant supplies the weighting function)
# ===============================================================
class Agent:
    def __init__(self, params=None):
        self.params = ([random.random() for _ in range(D)] if params is None
                       else [clamp01(x) for x in params])
        self.H_i, self.e_i, self.mu_i = [], 0.0, 0.0

    def update_history(self, f, E):
        self.H_i.append((f, E))
        if len(self.H_i) > TRAJ_WINDOW:
            self.H_i.pop(0)

    def compute_trajectory(self, E, alpha_fn):
        """e_i = sum_k alpha_hat_k f_i^(k).  Returns the weights' relative effective sample size
        (1.0 = perfectly even weights) when the history is long enough to be informative."""
        if not self.H_i:
            self.e_i = 0.0
            return None
        al = [alpha_fn(E, Fk) for _, Fk in self.H_i]
        t = sum(al)
        al = [a / t for a in al] if t > 0 else [1.0 / len(al)] * len(al)
        self.e_i = sum(a * self.H_i[k][0] for k, a in enumerate(al))
        if len(al) >= 5:
            return (1.0 / sum(a * a for a in al)) / len(al)
        return None

    def compute_mutation(self):
        self.mu_i = MU_BASE * (1.0 - self.e_i) + EPS

    def mutate(self):
        sigma = math.sqrt(self.mu_i) if MU_IS_VARIANCE else self.mu_i
        return Agent([p + random.gauss(0, sigma) for p in self.params])

    def snapshot(self):
        c = Agent(self.params)
        c.e_i, c.H_i, c.mu_i = self.e_i, self.H_i[:], self.mu_i
        return c


# ===============================================================
#  SHARED SELECTION / ARCHIVE / REPRODUCTION  (== COMPARE_ALL.evolve_core logic)
# ===============================================================
def select_and_reproduce(pop, archive, mutate_fn):
    md = math.sqrt(D)
    ranked = sorted(pop, key=lambda a: a.e_i, reverse=True)
    survivors = []
    for a in ranked:                                             # redundancy removal (delta_s)
        if not any(1.0 - euclid(a.params, u.params) / md > DELTA_S for u in survivors):
            survivors.append(a)
    final = [a for a in survivors if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
    while len(final) < MIN_KEEP and len(survivors) > len(final):
        for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
            if a not in final:
                final.append(a)
                break
    source = sorted(final, key=lambda a: a.e_i, reverse=True) if ARCHIVE_FROM_SURVIVORS else ranked
    for top in source[:3]:                                       # archive top-3
        archive.append(top.snapshot())
    archive.sort(key=lambda a: a.e_i, reverse=True)
    del archive[A_MAX:]
    pool_ = final + archive
    nxt = final[:]
    while len(nxt) < N_POP:                                      # tournament k=3
        cands = random.sample(pool_, min(3, len(pool_)))
        nxt.append(mutate_fn(max(cands, key=lambda a: a.e_i)))
    return nxt, archive


# ===============================================================
#  THE VARIANTS  (one code path, explicit switches)
# ===============================================================
VARIANTS = {
    "td":       dict(name="TD-CCEA",             env="centroid", dc_max=None),
    "td_sharp": dict(name="TD-CCEA+SharpKernel", env="centroid", dc_max=DC_MAX_SHARP),
    "up":       dict(name="TD-CCEA-Upgraded",    env="modes",    d_max=D_MAX, tau0=TAU0),
    "up_d02":   dict(name="Upgraded Dmax=0.2",   env="modes",    d_max=0.20,  tau0=TAU0),
    "up_d08":   dict(name="Upgraded Dmax=0.8",   env="modes",    d_max=0.80,  tau0=TAU0),
    "up_1mode": dict(name="Upgraded 1-mode",     env="modes",    d_max=D_MAX, tau0=1e12),
}
ALL_KEYS = list(VARIANTS)
CORE_KEYS = ["td", "td_sharp", "up"]


def make_alpha_fn(cfg):
    """alpha_k = similarity weight of a past environment (before normalisation)."""
    if cfg["env"] == "centroid":
        dc = cfg["dc_max"]
        if dc is None:                                   # COMPARE_ALL.Agent.compute_trajectory
            return lambda E, Fk: max(0.0, 1.0 - math.dist(E, Fk) / SQRT_D)
        return lambda E, Fk: max(0.0, 1.0 - (math.dist(E, Fk) / SQRT_D) / dc)
    d_max, cache = cfg["d_max"], {}

    def alpha(Ft, Fk):
        key = (Ft["uid"], Fk["uid"])
        if key not in cache:
            cache[key] = d_env(Ft, Fk)                   # same F^(k) is shared by many agents
        return max(0.0, 1.0 - cache[key] / d_max)
    return alpha


def run_variant(pool, key, init_params, seed):
    cfg = VARIANTS[key]
    random.seed(seed)
    t0, g0 = time.time(), pool.games
    if VERBOSE:
        print(f"\n{'=' * 66}\n  RUNNING: {cfg['name']}   (env={cfg['env']}, seed={seed})\n{'=' * 66}")
    alpha_fn = make_alpha_fn(cfg)
    pop = [Agent(p) for p in init_params]
    archive, conv, evenness, modes_hist = [], [], [], []
    overhead = 0.0

    for t in range(1, TOTAL_GENERATIONS + 1):
        opp = OPPONENT_ROTATION[(t - 1) % N_OPP]
        params = [a.params for a in pop]
        if cfg["env"] == "centroid":
            E = centroid(params)
            fits = pool.evaluate_many(params, opp)
        else:
            fits = pool.evaluate_many(params, opp)
            c0 = time.perf_counter()
            E = build_fingerprint(params, fits, cfg["tau0"], LAM)
            overhead += time.perf_counter() - c0
            modes_hist.append(len(E["count"]))
        for a, f in zip(pop, fits):
            a.update_history(f, E)
        c0 = time.perf_counter()
        for a in pop:
            ev = a.compute_trajectory(E, alpha_fn)
            if ev is not None:
                evenness.append(ev)
            a.compute_mutation()
        overhead += time.perf_counter() - c0
        conv.append((max(fits), mean(fits)))
        if VERBOSE:
            best = max(pop, key=lambda a: a.e_i)
            extra = f" modes={len(E['count'])} sizes={sorted(E['count'], reverse=True)}" if cfg["env"] == "modes" else ""
            print(f"  Gen {t:2d}/{TOTAL_GENERATIONS} | {opp:25s} | e_i={best.e_i:.3f} Best={max(fits):.3f}{extra}")
        pop, archive = select_and_reproduce(pop, archive, lambda parent: parent.mutate())

    champ = max(archive, key=lambda a: a.e_i) if archive else max(pop, key=lambda a: a.e_i)
    return {"params": list(champ.params), "elapsed": time.time() - t0, "games": pool.games - g0,
            "evenness": mean(evenness), "modes": mean(modes_hist) if modes_hist else 1.0,
            "overhead": overhead, "convergence": conv}


# ===============================================================
#  COMPARE_ALL.py CONTRACT  (drop-in wrappers)
# ===============================================================
def _compat(key, n_workers):
    pool = ServerPool(n_workers)
    try:
        init = [[random.random() for _ in range(D)] for _ in range(N_POP)]
        res = run_variant(pool, key, init, seed=random.getrandbits(31))
        return res["params"], res["elapsed"]
    finally:
        pool.close()


def run_td_ccea(server=None, n_workers=None):
    """Baseline TD-CCEA (centroid). `server` is accepted for the benchmark contract but not used."""
    return _compat("td", n_workers)


def run_td_ccea_upgraded(server=None, n_workers=None):
    """Upgraded TD-CCEA (modes + optimal transport). `server` is accepted but not used."""
    return _compat("up", n_workers)


# ===============================================================
#  STATISTICS
# ===============================================================
def wilcoxon_exact(x, y):
    """Exact two-sided Wilcoxon signed-rank p-value (average ranks for ties). None if all pairs tie."""
    diffs = [a - b for a, b in zip(x, y) if a != b]
    n = len(diffs)
    if n == 0:
        return None
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(diffs[order[j + 1]]) == abs(diffs[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r2 = [int(round(2 * r)) for r in ranks]
    total = sum(r2)
    w_plus = sum(r for r, d in zip(r2, diffs) if d > 0)
    counts = [0] * (total + 1)
    counts[0] = 1
    for r in r2:                                              # number of sign patterns per positive-rank sum
        for s in range(total, r - 1, -1):
            counts[s] += counts[s - r]
    dev = abs(w_plus - total / 2.0)
    extreme = sum(c for s, c in enumerate(counts) if abs(s - total / 2.0) >= dev - 1e-9)
    return min(1.0, extreme / float(2 ** n))


def bootstrap_ci(diffs, n_boot=5000, seed=0):
    rnd = random.Random(seed)
    n = len(diffs)
    ms = sorted(mean([diffs[rnd.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return ms[int(0.025 * n_boot)], ms[int(0.975 * n_boot) - 1]


def holm(pvals):
    idx = sorted((p, i) for i, p in enumerate(pvals) if p is not None)
    adj, run, m = {}, 0.0, len(idx)
    for k, (p, i) in enumerate(idx):
        run = max(run, min(1.0, (m - k) * p))
        adj[i] = run
    return [adj.get(i) for i in range(len(pvals))]


COMPARISONS = [      # (A, B, what the difference isolates)
    ("up",       "td",       "TOTAL effect of the upgrade"),
    ("td_sharp", "td",       "weighting sharpness alone (no new representation)"),
    ("up",       "td_sharp", "representation, at matched weighting sharpness"),
    ("up_d02",   "up",       "D_MAX 0.2 vs 0.4"),
    ("up_d08",   "up",       "D_MAX 0.8 vs 0.4"),
    ("up_1mode", "up",       "single representative vs multiple modes"),
]


def analyze(records, keys, val_games, out_dir):
    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    by = {k: {} for k in keys}
    for r in records:
        if r["key"] in by:
            by[r["key"]][r["trial"]] = r
    trials = sorted(set.intersection(*[set(by[k]) for k in keys])) if keys else []
    if not trials:
        P("  No complete paired trials to analyse.")
        return
    total_games = N_OPP * val_games
    P(f"\n{'#' * 76}\n  RESULTS - {len(trials)} paired trials, validation = {val_games} games x {N_OPP} bots "
      f"= {total_games} games per champion\n{'#' * 76}")
    P(f"\n  {'Algorithm':<20s} {'WinRate mean +- sd':>19s} {'median':>7s} {'min':>6s} {'max':>6s} "
      f"{'games':>6s} {'time s':>7s} {'evenness':>9s} {'modes':>6s} {'extra ms':>9s}")
    P("  " + "-" * 104)
    for k in keys:
        w = [by[k][t]["wins"] / total_games * 100 for t in trials]
        P(f"  {VARIANTS[k]['name']:<20s} {mean(w):>10.1f} +- {std(w):<5.1f} {statistics.median(w):>7.1f} "
          f"{min(w):>6.1f} {max(w):>6.1f} {mean(by[k][t]['games'] for t in trials):>6.0f} "
          f"{mean(by[k][t]['time'] for t in trials):>7.1f} {mean(by[k][t]['evenness'] for t in trials):>9.3f} "
          f"{mean(by[k][t]['modes'] for t in trials):>6.2f} {mean(by[k][t]['overhead'] for t in trials) * 1000:>9.1f}")
    P("  evenness = how uniformly the trajectory score weights the last <=7 entries (1.000 = plain average);")
    P("  extra ms = total time spent in fingerprinting + trajectory scoring per run (not wall-clock noise).")

    P(f"\n  Mean win rate per bot (%)")
    P("  " + f"{'Bot':<22s}" + "".join(f"{VARIANTS[k]['name'][:17]:>19s}" for k in keys))
    for o in OPPONENT_ROTATION:
        P("  " + f"{o:<22s}" + "".join(
            f"{mean(by[k][t]['per_opp'][o] for t in trials) / val_games * 100:>19.1f}" for k in keys))

    comps = [(a, b, why) for a, b, why in COMPARISONS if a in by and b in by]
    pv, rows = [], []
    for a, b, why in comps:
        xa = [by[a][t]["wins"] for t in trials]
        xb = [by[b][t]["wins"] for t in trials]
        d = [(p - q) / total_games * 100 for p, q in zip(xa, xb)]
        lo, hi = bootstrap_ci(d)
        pv.append(wilcoxon_exact(xa, xb))
        rows.append((a, b, why, mean(d), lo, hi, sum(p > q for p, q in zip(xa, xb)),
                     sum(p == q for p, q in zip(xa, xb)), sum(p < q for p, q in zip(xa, xb))))
    adj = holm(pv)
    P(f"\n  Paired comparisons across trials (A - B, in win-rate points), exact Wilcoxon, Holm-adjusted")
    P("  " + "-" * 100)
    for (a, b, why, md, lo, hi, w, t_, l), p, ap in zip(rows, pv, adj):
        ps = "n/a" if p is None else f"{p:.4f}"
        aps = "n/a" if ap is None else f"{ap:.4f}"
        sig = "" if ap is None else ("  SIGNIFICANT" if ap < 0.05 else "  not significant")
        P(f"  {VARIANTS[a]['name']} vs {VARIANTS[b]['name']}  [{why}]")
        P(f"      mean diff {md:+.1f} pts  95% CI [{lo:+.1f}, {hi:+.1f}]  W/T/L {w}/{t_}/{l}  "
          f"p={ps}  p(Holm)={aps}{sig}")
    P("\n  How to read this: 'up vs td' is the total effect. If 'td_sharp vs td' shows the same gain, the benefit")
    P("  comes from sharper weighting, not from the multi-mode fingerprint; the representation effect is")
    P("  'up vs td_sharp' (check that their evenness values are close). Champions are chosen by archive")
    P("  trajectory score exactly as in COMPARE_ALL.py. With few trials trust the interval, not the point estimate.")

    with open(os.path.join(out_dir, "td_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "td_winrates.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial"] + [VARIANTS[k]["name"] for k in keys])
        for t in trials:
            w.writerow([t] + [round(by[k][t]["wins"] / total_games * 100, 2) for k in keys])


# ===============================================================
#  EXPERIMENT HARNESS
# ===============================================================
def run_experiment(args):
    global TOTAL_GENERATIONS, VERBOSE, ARCHIVE_FROM_SURVIVORS, MU_IS_VARIANCE
    TOTAL_GENERATIONS = args.generations
    VERBOSE = not args.quiet
    ARCHIVE_FROM_SURVIVORS = args.archive_from_survivors
    MU_IS_VARIANCE = args.mu_variance
    if args.dc_max:
        VARIANTS["td_sharp"]["dc_max"] = args.dc_max
    keys = (CORE_KEYS if args.algos == "core" else ALL_KEYS if args.algos == "all"
            else [k.strip() for k in args.algos.split(",")])
    bad = [k for k in keys if k not in VARIANTS]
    if bad:
        sys.exit(f"Unknown algorithm key(s) {bad}; valid: {ALL_KEYS} (or 'core' / 'all')")
    out_dir = args.out or SCRIPT_DIR
    os.makedirs(out_dir, exist_ok=True)
    rec_path = os.path.join(out_dir, "td_trials.json")

    records = []
    if args.resume and os.path.exists(rec_path):
        with open(rec_path) as f:
            records = json.load(f)
        print(f"  Resuming: {len(records)} finished runs loaded from {rec_path}")
    done = {(r["trial"], r["key"]) for r in records}

    if args.mock:
        MockGameServer.latency = args.mock_latency
        factory = MockGameServer
    else:
        factory = GameServer
    nw = args.workers or default_workers()
    print(f"\n{'#' * 76}\n  TD-CCEA family comparison | pop={N_POP} gens={TOTAL_GENERATIONS} K={K_GAMES} "
          f"| validation={args.val_games}/bot | trials={args.trials}\n  map={MAP_PATH} max_cycles={MAX_CYCLES} "
          f"| workers={nw} | variants={keys}\n  OT solver: "
          f"{'scipy linear_sum_assignment' if _lsa is not None else 'pure-Python Hungarian'}\n{'#' * 76}")
    pool = ServerPool(nw, factory)
    try:
        for trial in range(args.trials):
            seed = args.seed + trial
            rnd = random.Random(seed)
            init = [[rnd.random() for _ in range(D)] for _ in range(N_POP)]   # shared by all algorithms
            order = keys[:]
            rnd.shuffle(order)                                               # no order / warm-up bias
            for key in order:
                if (trial, key) in done:
                    continue
                res = run_variant(pool, key, init, seed * 1009 + ALL_KEYS.index(key))
                wins, games, per = pool.validate(res["params"], args.val_games)
                rec = {"trial": trial, "seed": seed, "key": key, "name": VARIANTS[key]["name"],
                       "wins": wins, "games_validation": games, "per_opp": per, "time": res["elapsed"],
                       "games": res["games"], "evenness": res["evenness"], "modes": res["modes"],
                       "overhead": res["overhead"], "params": res["params"],
                       "convergence": res["convergence"]}
                records.append(rec)
                with open(rec_path, "w") as f:                               # save after every run
                    json.dump(records, f)
                print(f"  -> trial {trial + 1}/{args.trials} {VARIANTS[key]['name']:<20s} "
                      f"validation {wins / games * 100:5.1f}%  ({res['games']} training games, "
                      f"{res['elapsed']:.0f}s, evenness {res['evenness']:.3f}, "
                      f"server restarts so far: {pool.restarts})")
    finally:
        pool.close()
    analyze(records, keys, args.val_games, out_dir)


def parse_args():
    ap = argparse.ArgumentParser(description="TD-CCEA vs Upgraded TD-CCEA controlled comparison")
    ap.add_argument("--trials", type=int, default=10, help="independent paired trials (default 10)")
    ap.add_argument("--algos", default="core",
                    help=f"'core' (td,td_sharp,up), 'all', or a comma list of {ALL_KEYS}")
    ap.add_argument("--workers", type=int, default=None, help="parallel Java servers (default cores-1)")
    ap.add_argument("--val-games", type=int, default=20, help="validation games per bot (default 20)")
    ap.add_argument("--seed", type=int, default=12345, help="base seed; trial i uses seed+i")
    ap.add_argument("--generations", type=int, default=TOTAL_GENERATIONS)
    ap.add_argument("--out", default=None, help="output folder (default: script folder)")
    ap.add_argument("--resume", action="store_true", help="continue from td_trials.json")
    ap.add_argument("--quiet", action="store_true", help="no per-generation lines")
    ap.add_argument("--archive-from-survivors", action="store_true",
                    help="archive top-3 SURVIVORS (PDF wording) instead of top-3 of the population (COMPARE_ALL)")
    ap.add_argument("--mu-variance", action="store_true",
                    help="treat mu_i as a variance (sigma = sqrt(mu_i)) instead of a std (COMPARE_ALL)")
    ap.add_argument("--dc-max", type=float, default=None,
                    help="override DC_MAX_SHARP, the cutoff of the sharpened centroid control (smaller = sharper)")
    ap.add_argument("--mock", action="store_true", help="self-test with a fake game server (no Java)")
    ap.add_argument("--mock-latency", type=float, default=0.0)
    return ap.parse_args()


if __name__ == "__main__":
    try:
        run_experiment(parse_args())
    except GameServerError as e:
        sys.exit(f"\nABORTED: {e}\nNo results were fabricated; finished runs are saved (use --resume).")
