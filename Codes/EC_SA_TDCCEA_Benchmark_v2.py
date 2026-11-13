#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EC_SA_TDCCEA_Benchmark.py
=========================
Corrected, controlled comparison of SA-TDCCEA vs EC-SA-TDCCEA (rectified
specification) on the COMPARE_ALL.py setup, with multi-core game evaluation,
paired multi-trial runs and proper statistics.

WHAT CHANGED vs. the previous EC_SA_TDCCEA_Parallel.py
------------------------------------------------------
1. Robust game servers: a dead / hung / garbled Java process is DETECTED
   (EOF, per-game timeout, ERROR line), restarted, and the game retried. If it
   still fails the run aborts loudly. Nothing is ever silently scored as a draw.
   stderr goes to DEVNULL (an unread PIPE can deadlock the JVM). Pre-flight
   checks verify java, the compiled class and the map file before any game.
2. Setup identical to COMPARE_ALL.py: map arena_chambers_32x32, MAX_CYCLES 10000,
   same N_POP / K_GAMES / TOTAL_GENERATIONS / A_MAX / DELTA_S / DELTA_F.
3. SA-TDCCEA is now the SAME algorithm as COMPARE_ALL.evolve_core(pwm=...):
   eps 0.02 in sigma_j = pwm_j*(1-e_i)+0.02, >=10 survivors, top-3 of the whole
   population archived, centroid environment, NO extra evaluation games.
   The static sensitivity analysis is COMPARE_ALL's exact procedure (baseline 0.5,
   rates pooled over all bots) and is cached together with the map / max-cycles
   it was measured on (a file from another map is regenerated, not reused).
4. All five variants share ONE code path (selection, archive, tournament,
   budget). They differ only in explicit switches, so differences in results
   can be attributed to those switches:

      key        name              environment    sensitivity   correlated mutation
      sa         SA-TDCCEA         centroid       static (pwm)  no      <- baseline
      sa_fp      SA+Fingerprint    full 73-D      static (pwm)  no      <- fingerprint effect
      sa_fp_cov  SA+FP+Cov         full 73-D      static (pwm)  yes     <- covariance effect
      ec_nocov   EC-SA (no cov)    full 73-D      dynamic       no      <- dynamic-sensitivity effect
      ec         EC-SA-TDCCEA      full 73-D      dynamic       yes     <- proposed (full spec)

5. Paired multi-trial harness: each trial draws ONE random initial population
   shared by every algorithm, randomises the algorithm order (removes JVM
   warm-up / order bias), runs the champion through a larger validation
   (default 20 games per bot), and saves results after every run (--resume).
6. Statistics: exact Wilcoxon signed-rank across TRIALS (not across the 7 bots),
   Holm correction, paired mean difference with bootstrap 95% CI.

USAGE
-----
  python EC_SA_TDCCEA_Benchmark.py --trials 10                  # all 5 variants
  python EC_SA_TDCCEA_Benchmark.py --trials 20 --algos sa,ec    # head-to-head only
  python EC_SA_TDCCEA_Benchmark.py --trials 10 --resume         # continue a saved run
  python EC_SA_TDCCEA_Benchmark.py --mock --trials 3 --generations 8   # self-test, no Java
  Workers: --workers N  (default = CPU cores - 1, max 60) or env ECSA_WORKERS.

COMPARE_ALL.py INTEGRATION (drop-in, same contract as the other algorithms)
  from EC_SA_TDCCEA_Benchmark import run_sa_tdccea, run_ec_sa_tdccea
  ...  elif key == "ec_sa": best, elapsed = run_ec_sa_tdccea(server)
  (These start their own pool of robust servers; the `server` argument is not
   used, so a crashed worker can never inject fake draws into your results.)

Requires numpy.
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
MIN_KEEP          = 10          # COMPARE_ALL.evolve_core keeps >= 10 survivors
DRAW_SCORE        = 0.3         # training score of a true draw / timeout game
SENSITIVITY_FILE  = "sensitivity_scores.json"

# --- game-server robustness
START_TIMEOUT = 180.0           # seconds for the JVM to print READY
GAME_TIMEOUT  = 300.0           # seconds for one game
MAX_RETRIES   = 3               # attempts per game before the run aborts

# --- mutation constants
MU_BASE, EPS_DYNAMIC = 0.15, 0.03     # dynamic rule  mu_i = 0.15(1-e)+0.03   (PDF)
EPS_STATIC           = 0.02           # static SA rule sigma_j = pwm_j(1-e)+0.02 (COMPARE_ALL)

# --- EC-SA-TDCCEA hyperparameters (values are mine; the PDF names but does not fix them)
TAU_E = 0.45    # env distance to last ablation checkpoint that forces an ablation
TAU_I = 0.80    # env similarity needed to match / reuse an environment cluster
B_MAX = 5       # max generations between ablations
TAU_R = 1.00    # reservoir threshold (accumulated drift)
RHO   = 0.80    # reservoir decay
ETA   = 0.30    # EMA rate of cluster sensitivity
I_EMA, I_COUPLING = 0.30, 0.60        # interaction-matrix smoothing / off-diagonal coupling
RATE_LO, RATE_HI  = 0.08, 0.20        # sensitivity -> relative mutation-rate range

ARCHIVE_FROM_SURVIVORS = False        # False = COMPARE_ALL (top-3 of population); True = PDF text
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
    env = os.environ.get("ECSA_WORKERS")
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
#  STATIC SENSITIVITY  (COMPARE_ALL.run_sensitivity_analysis, parallelised, same maths)
# ===============================================================
def load_or_run_static_sensitivity(pool, sens_path=None):
    fp = sens_path or os.path.join(SCRIPT_DIR, SENSITIVITY_FILE)
    if os.path.exists(fp):
        try:
            with open(fp) as f:
                data = json.load(f)
            rates = data["normalized_mutation_rates"]
            ok = (len(rates) == D and min(rates) >= 0.07 and max(rates) <= 0.21
                  and data.get("map") == MAP_PATH and data.get("max_cycles") == MAX_CYCLES)
            if ok:
                print(f"  [SA] Using cached static sensitivity ({os.path.basename(fp)}, map={MAP_PATH})")
                return rates
            print("  [SA] Cached sensitivity file is for another map/setup or stale -> regenerating")
        except Exception:
            print("  [SA] Could not read sensitivity cache -> regenerating")

    print("  [SA] Static sensitivity analysis (baseline 0.5, pooled over all bots)...")
    t0 = time.time()
    base = [0.5] * D
    vecs = [base]
    for j in range(D):
        for v in (0.0, 1.0):
            t = base[:]
            t[j] = v
            vecs.append(t)
    totals = [0.0] * len(vecs)
    for opp in OPPONENT_ROTATION:                      # each fitness = mean over K games
        fits = pool.evaluate_many(vecs, opp)
        totals = [a + b for a, b in zip(totals, fits)]
    wr = [t / N_OPP for t in totals]                   # pooled win rate (equal games per bot)
    influence = [abs(wr[1 + 2 * j + 1] - wr[1 + 2 * j]) for j in range(D)]
    mx = max(influence) if max(influence) > 0 else 1.0
    normalized = [0.08 + (1.0 - inf / mx) * 0.12 for inf in influence]
    with open(fp, "w") as f:
        json.dump({"influence_scores": influence, "normalized_mutation_rates": normalized,
                   "baseline_winrate": wr[0], "weight_names": WEIGHT_NAMES,
                   "map": MAP_PATH, "max_cycles": MAX_CYCLES}, f, indent=2)
    print(f"  [SA] Saved {os.path.basename(fp)} ({time.time() - t0:.0f}s, {len(vecs) * N_OPP * K_GAMES} games)")
    return normalized


# ===============================================================
#  ENVIRONMENT  (centroid for SA, 73-D fingerprint for the full variants)
# ===============================================================
POP_SL  = slice(0, 2 * D)
OPP_SL  = slice(2 * D, 2 * D + N_OPP)
GAME_SL = slice(OPP_SL.stop, OPP_SL.stop + 2)
FIT_SL  = slice(GAME_SL.stop, GAME_SL.stop + 3)
BLOCK_W = (0.4, 0.3, 0.1, 0.2)


def build_fingerprint(gen, opp, pop, archive, fit_stats):
    """E_t = [R_pop (pop centroid + elite centroid), R_opp (one-hot), R_game, R_fit]."""
    cen = centroid([a.params for a in pop])
    elite = (centroid([a.params for a in sorted(archive, key=lambda a: a.e_i, reverse=True)[:3]])
             if archive else cen)
    onehot = [1.0 if o == opp else 0.0 for o in OPPONENT_ROTATION]
    div = mean(euclid(a.params, cen) for a in pop) / math.sqrt(D)
    return cen + elite + onehot + [gen / TOTAL_GENERATIONS, min(1.0, 2.0 * div)] + list(fit_stats)


def _esim(a, b):
    return max(0.0, 1.0 - euclid(a, b) / math.sqrt(len(a)))


def env_sim(e1, e2):
    if len(e1) == D:                                   # centroid environment (as COMPARE_ALL)
        return _esim(e1, e2)
    s_pop = _esim(e1[POP_SL], e2[POP_SL])
    s_opp = max(0.0, 1.0 - 0.5 * sum(abs(p - q) for p, q in zip(e1[OPP_SL], e2[OPP_SL])))
    s_game = _esim(e1[GAME_SL], e2[GAME_SL])
    s_fit = _esim(e1[FIT_SL], e2[FIT_SL])
    return sum(w * s for w, s in zip(BLOCK_W, (s_pop, s_opp, s_game, s_fit))) / sum(BLOCK_W)


def d_env(e1, e2):
    return 1.0 - env_sim(e1, e2)


# ===============================================================
#  AGENT
# ===============================================================
class Agent:
    def __init__(self, params=None):
        self.params = ([random.random() for _ in range(D)] if params is None
                       else [clamp01(x) for x in params])
        self.H_i, self.e_i, self.mu_i, self.fit = [], 0.0, 0.0, 0.0

    def update_history(self, f, E):
        self.H_i.append((f, E))
        if len(self.H_i) > TRAJ_WINDOW:
            self.H_i.pop(0)

    def compute_trajectory(self, E):
        if not self.H_i:
            self.e_i = 0.0
            return
        ws = [env_sim(E, pe) for _, pe in self.H_i]
        t = sum(ws)
        ws = [w / t for w in ws] if t > 0 else [1.0 / len(ws)] * len(ws)
        self.e_i = sum(w * self.H_i[i][0] for i, w in enumerate(ws))

    def compute_mutation(self):
        self.mu_i = MU_BASE * (1.0 - self.e_i) + EPS_DYNAMIC

    def snapshot(self):
        c = Agent(self.params)
        c.e_i, c.H_i, c.mu_i, c.fit = self.e_i, self.H_i[:], self.mu_i, self.fit
        return c


# ===============================================================
#  EC-SA PARTS: dynamic sensitivity memory, interaction matrix
# ===============================================================
def ablation_update(pool, memory, baseline, opp, E):
    """S_j(E_t)=|WinRate(x_j=1)-WinRate(x_j=0)| around `baseline`; EMA into the closest cluster of M_j
    (sim >= tau_I) else new cluster. The 2*D evaluations run in parallel."""
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
        bk, bs = None, -1.0
        for k, c in enumerate(clusters):
            s = env_sim(E, c["mu"])
            if s > bs:
                bk, bs = k, s
        if bk is not None and bs >= TAU_I:
            c = clusters[bk]
            n = c["n"]
            c["mu"] = [(n * m + e) / (n + 1) for m, e in zip(c["mu"], E)]
            c["n"] = n + 1
            c["S"] = (1 - ETA) * c["S"] + ETA * S
        else:
            clusters.append({"mu": E[:], "S": S, "n": 1})


def retrieve_sensitivity(memory, E):
    out = [None] * D
    for j in range(D):
        num = den = 0.0
        for c in memory[j]:
            s = env_sim(E, c["mu"])
            if s >= TAU_I:
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
    mx = max(S) if max(S) > 0 else 1.0
    rate = [RATE_LO + (1.0 - s / mx) * (RATE_HI - RATE_LO) for s in S]
    m = mean(rate)
    return np.array([r / m for r in rate])


def interaction_observation(P, f):
    """|corr(f, z_i*z_j)| over the population: joint setting of (i, j) explaining fitness (no extra games)."""
    obs = np.zeros((D, D))
    X, fv = np.asarray(P, float), np.asarray(f, float)
    if len(fv) < 4 or fv.std() < 1e-9:
        return obs
    zf = (fv - fv.mean()) / fv.std()
    sd = X.std(axis=0)
    Z = np.where(sd > 1e-9, (X - X.mean(axis=0)) / np.where(sd > 1e-9, sd, 1), 0.0)
    for i in range(D):
        for j in range(i + 1, D):
            pr = Z[:, i] * Z[:, j]
            ps = pr.std()
            if ps < 1e-9:
                continue
            obs[i, j] = obs[j, i] = min(1.0, abs(float(np.mean((pr - pr.mean()) / ps * zf))))
    return obs


def covariance_factor(I):
    """Cholesky factor of the unit-diagonal correlation matrix  C = Id + g*I  (clipped to PSD)."""
    C = np.eye(D) + I_COUPLING * (I - np.diag(np.diag(I)))
    w, V = np.linalg.eigh((C + C.T) / 2)
    C = (V * np.clip(w, 1e-3, None)) @ V.T
    dg = np.sqrt(np.diag(C))
    return np.linalg.cholesky(C / np.outer(dg, dg) + 1e-9 * np.eye(D))


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
#  THE FIVE VARIANTS  (one code path, explicit switches)
# ===============================================================
VARIANTS = {
    "sa":        dict(name="SA-TDCCEA",      env="centroid", sens="static",  cov=False),
    "sa_fp":     dict(name="SA+Fingerprint", env="full",     sens="static",  cov=False),
    "sa_fp_cov": dict(name="SA+FP+Cov",      env="full",     sens="static",  cov=True),
    "ec_nocov":  dict(name="EC-SA (no cov)", env="full",     sens="dynamic", cov=False),
    "ec":        dict(name="EC-SA-TDCCEA",   env="full",     sens="dynamic", cov=True),
}
ALL_KEYS = list(VARIANTS)


def run_variant(pool, key, pwm, init_params, seed):
    """Returns dict(params, elapsed, games, ablations, convergence). Game outcomes are never faked."""
    cfg = VARIANTS[key]
    random.seed(seed)
    rng = np.random.default_rng(seed)
    t0, g0 = time.time(), pool.games
    if VERBOSE:
        print(f"\n{'=' * 64}\n  RUNNING: {cfg['name']}   (env={cfg['env']}, sens={cfg['sens']}, "
              f"cov={cfg['cov']}, seed={seed})\n{'=' * 64}")

    pop = [Agent(p) for p in init_params]
    archive, memory = [], [[] for _ in range(D)]
    I = np.zeros((D, D))
    E_prev = E_last = None
    t_last, R, stats = 0, 0.0, (0.0, 0.0, 0.0)
    ablations, conv = 0, []

    for t in range(TOTAL_GENERATIONS):
        gen, opp = t + 1, OPPONENT_ROTATION[t % N_OPP]
        E = (build_fingerprint(gen, opp, pop, archive, stats) if cfg["env"] == "full"
             else centroid([a.params for a in pop]))
        fits = pool.evaluate_many([a.params for a in pop], opp)
        for a, f in zip(pop, fits):
            a.fit = f
            a.update_history(f, E)
        for a in pop:
            a.compute_trajectory(E)
            a.compute_mutation()
        stats = (mean(fits), max(fits), std(fits))
        conv.append((max(fits), mean(fits)))

        ablate, scale = False, np.ones(D)
        if cfg["sens"] == "dynamic":
            R = 0.0 if E_prev is None else RHO * R + d_env(E, E_prev)
            d_chk = float("inf") if E_last is None else d_env(E, E_last)
            ablate = (E_last is None or d_chk > TAU_E or R > TAU_R or (t - t_last) >= B_MAX)
            if ablate:
                ablation_update(pool, memory, max(pop, key=lambda a: a.e_i).params, opp, E)
                E_last, t_last, R = E, t, 0.0
                ablations += 1
            E_prev = E
            S = retrieve_sensitivity(memory, E)
            if S:
                scale = sensitivity_to_scale(S)

        L = None
        if cfg["cov"]:
            I = (1 - I_EMA) * I + I_EMA * interaction_observation([a.params for a in pop], fits)
            np.fill_diagonal(I, 0.0)
            L = covariance_factor(I)

        if VERBOSE:
            best = max(pop, key=lambda a: a.e_i)
            extra = f" ablate={'Y' if ablate else '-'}" if cfg["sens"] == "dynamic" else ""
            print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | e_i={best.e_i:.3f} "
                  f"Best={max(fits):.3f}{extra}")

        def mutate(parent, L=L, scale=scale):
            if cfg["sens"] == "static":
                sig = np.array([pwm[j] * (1.0 - parent.e_i) + EPS_STATIC for j in range(D)])
            else:
                sig = (MU_BASE * (1.0 - parent.e_i) + EPS_DYNAMIC) * scale
            z = rng.standard_normal(D)
            return Agent(np.asarray(parent.params) + sig * ((L @ z) if L is not None else z))

        pop, archive = select_and_reproduce(pop, archive, mutate)

    champ = max(archive, key=lambda a: a.e_i) if archive else max(pop, key=lambda a: a.e_i)
    return {"params": list(champ.params), "elapsed": time.time() - t0,
            "games": pool.games - g0, "ablations": ablations, "convergence": conv}


# ===============================================================
#  COMPARE_ALL.py CONTRACT  (drop-in wrappers)
# ===============================================================
def _compat(key, n_workers):
    pool = ServerPool(n_workers)
    try:
        pwm = load_or_run_static_sensitivity(pool)
        init = [[random.random() for _ in range(D)] for _ in range(N_POP)]
        res = run_variant(pool, key, pwm, init, seed=random.getrandbits(31))
        return res["params"], res["elapsed"]
    finally:
        pool.close()


def run_sa_tdccea(server=None, n_workers=None):
    """Baseline SA-TDCCEA. `server` is accepted for the benchmark contract but not used."""
    return _compat("sa", n_workers)


def run_ec_sa_tdccea(server=None, n_workers=None):
    """Proposed EC-SA-TDCCEA. `server` is accepted for the benchmark contract but not used."""
    return _compat("ec", n_workers)


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
    ("ec",        "sa",        "TOTAL effect of the proposed algorithm"),
    ("sa_fp",     "sa",        "fingerprint (opponent-aware trajectory)"),
    ("sa_fp_cov", "sa_fp",     "correlated mutation, static sensitivity"),
    ("ec_nocov",  "sa_fp",     "dynamic sensitivity, no covariance"),
    ("ec",        "ec_nocov",  "correlated mutation, dynamic sensitivity"),
    ("ec",        "sa_fp_cov", "dynamic vs static sensitivity, with covariance"),
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
    P(f"\n{'#' * 72}\n  RESULTS - {len(trials)} paired trials, validation = {val_games} games x {N_OPP} bots "
      f"= {total_games} games per champion\n{'#' * 72}")

    P(f"\n  {'Algorithm':<18s} {'WinRate mean +- sd':>20s} {'median':>8s} {'min':>6s} {'max':>6s} "
      f"{'games':>7s} {'time s':>8s} {'ablations':>10s}")
    P("  " + "-" * 92)
    for k in keys:
        w = [by[k][t]["wins"] / total_games * 100 for t in trials]
        P(f"  {VARIANTS[k]['name']:<18s} {mean(w):>11.1f} +- {std(w):<5.1f} {statistics.median(w):>8.1f} "
          f"{min(w):>6.1f} {max(w):>6.1f} {mean(by[k][t]['games'] for t in trials):>7.0f} "
          f"{mean(by[k][t]['time'] for t in trials):>8.1f} {mean(by[k][t]['ablations'] for t in trials):>10.1f}")

    P(f"\n  Mean win rate per bot (%)")
    P("  " + f"{'Bot':<22s}" + "".join(f"{VARIANTS[k]['name'][:15]:>16s}" for k in keys))
    for o in OPPONENT_ROTATION:
        row = f"  {o:<22s}"
        for k in keys:
            row += f"{mean(by[k][t]['per_opp'][o] for t in trials) / val_games * 100:>16.1f}"
        P(row)

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
    P("\n  Notes: one static-sensitivity file (pwm) is shared by all trials; champions are chosen by")
    P("  archive trajectory score exactly as in COMPARE_ALL.py; with few trials the CI is wide - trust")
    P("  the interval, not the point estimate.")

    with open(os.path.join(out_dir, "ecsa_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "ecsa_winrates.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial"] + [VARIANTS[k]["name"] for k in keys])
        for t in trials:
            w.writerow([t] + [round(by[k][t]["wins"] / total_games * 100, 2) for k in keys])


# ===============================================================
#  EXPERIMENT HARNESS
# ===============================================================
def run_experiment(args):
    global TOTAL_GENERATIONS, VERBOSE, ARCHIVE_FROM_SURVIVORS
    TOTAL_GENERATIONS = args.generations
    VERBOSE = not args.quiet
    ARCHIVE_FROM_SURVIVORS = args.archive_from_survivors
    keys = ALL_KEYS if args.algos == "all" else [k.strip() for k in args.algos.split(",")]
    bad = [k for k in keys if k not in VARIANTS]
    if bad:
        sys.exit(f"Unknown algorithm key(s) {bad}; valid: {ALL_KEYS}")
    out_dir = args.out or SCRIPT_DIR
    os.makedirs(out_dir, exist_ok=True)
    rec_path = os.path.join(out_dir, "ecsa_trials.json")

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
    print(f"\n{'#' * 72}\n  SA-TDCCEA family comparison | pop={N_POP} gens={TOTAL_GENERATIONS} K={K_GAMES} "
          f"| validation={args.val_games}/bot | trials={args.trials}\n  map={MAP_PATH} max_cycles={MAX_CYCLES} "
          f"| workers={nw} | variants={keys}\n{'#' * 72}")
    pool = ServerPool(nw, factory)
    try:
        t_s = time.time()
        pwm = load_or_run_static_sensitivity(
            pool, os.path.join(out_dir, "sensitivity_mock.json") if args.mock else None)
        print(f"  (static sensitivity ready in {time.time() - t_s:.0f}s - not counted in any algorithm's time)")
        for trial in range(args.trials):
            seed = args.seed + trial
            rnd = random.Random(seed)
            init = [[rnd.random() for _ in range(D)] for _ in range(N_POP)]   # shared by all algorithms
            order = keys[:]
            rnd.shuffle(order)                                               # no order / warm-up bias
            for key in order:
                if (trial, key) in done:
                    continue
                res = run_variant(pool, key, pwm, init, seed * 1009 + ALL_KEYS.index(key))
                wins, games, per = pool.validate(res["params"], args.val_games)
                rec = {"trial": trial, "seed": seed, "key": key, "name": VARIANTS[key]["name"],
                       "wins": wins, "games_validation": games, "per_opp": per,
                       "time": res["elapsed"], "games": res["games"], "ablations": res["ablations"],
                       "params": res["params"], "convergence": res["convergence"]}
                records.append(rec)
                with open(rec_path, "w") as f:                               # save after every run
                    json.dump(records, f)
                print(f"  -> trial {trial + 1}/{args.trials} {VARIANTS[key]['name']:<16s} "
                      f"validation {wins / games * 100:5.1f}%  ({res['games']} training games, "
                      f"{res['elapsed']:.0f}s, server restarts so far: {pool.restarts})")
    finally:
        pool.close()
    analyze(records, keys, args.val_games, out_dir)


def parse_args():
    ap = argparse.ArgumentParser(description="SA-TDCCEA vs EC-SA-TDCCEA controlled comparison")
    ap.add_argument("--trials", type=int, default=10, help="independent paired trials (default 10)")
    ap.add_argument("--algos", default="all", help=f"comma list of {ALL_KEYS} or 'all'")
    ap.add_argument("--workers", type=int, default=None, help="parallel Java servers (default cores-1)")
    ap.add_argument("--val-games", type=int, default=20, help="validation games per bot (default 20)")
    ap.add_argument("--seed", type=int, default=12345, help="base seed; trial i uses seed+i")
    ap.add_argument("--generations", type=int, default=TOTAL_GENERATIONS)
    ap.add_argument("--out", default=None, help="output folder (default: script folder)")
    ap.add_argument("--resume", action="store_true", help="continue from ecsa_trials.json")
    ap.add_argument("--quiet", action="store_true", help="no per-generation lines")
    ap.add_argument("--archive-from-survivors", action="store_true",
                    help="archive top-3 SURVIVORS (PDF text) instead of top-3 of the population (COMPARE_ALL)")
    ap.add_argument("--mock", action="store_true", help="self-test with a fake game server (no Java)")
    ap.add_argument("--mock-latency", type=float, default=0.0)
    return ap.parse_args()


if __name__ == "__main__":
    try:
        run_experiment(parse_args())
    except GameServerError as e:
        sys.exit(f"\nABORTED: {e}\nNo results were fabricated; finished runs are saved (use --resume).")
