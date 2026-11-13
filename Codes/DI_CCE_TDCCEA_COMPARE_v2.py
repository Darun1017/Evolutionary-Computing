#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPARE_CCE_DI.py
=================

Benchmark of two trajectory-driven cooperative co-evolutionary algorithms:

  * CCE-TDCCEA      (baseline)  - fixed Macro/Micro decomposition, trajectory
                                  memory, archive, tournament + mutation.
  * DI-CCE-TDCCEA   (proposed)  - Dynamic Interaction-Driven decomposition:
                                  strategy-interaction matrix -> Maximum
                                  Spanning Tree -> temporary co-evolution
                                  groups with adaptive windows (Lmin/Lmax,
                                  joint-improvement and interaction-strength
                                  termination).

Problem : find a 20-D weight vector in [0,1]^20 that maximises the win rate of
          a scripted microRTS agent against 7 built-in bots on the 32x32 map.

Parallelism : N_WORKERS Java game servers run concurrently. All games of a
              generation are dispatched to the pool, so ~8 CPU cores are used.

Requirements:
    Python 3.9+,  pip install numpy
    pip install openpyxl          (optional - Excel log)
    Java + compiled microRTS project in <parent of this file>/Simulation/microrts
"""

import os
import json
import math
import time
import queue
import random
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

os.environ["PYTHONIOENCODING"] = "utf-8"

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
cand_dirs    = [
    os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "Simulation", "microrts"),
    os.path.join(os.path.dirname(SCRIPT_DIR), "Simulation", "microrts"),
    os.path.join(SCRIPT_DIR, "Simulation", "microrts"),
]
MICRORTS_DIR = next(
    (d for d in cand_dirs if os.path.isfile(os.path.join(d, "bin", "tests", "EvolvedBatchGameRunner.class"))),
    cand_dirs[0]
)

MAP_PATH     = "maps/arena_chambers_32x32.xml"
if not os.path.isfile(os.path.join(MICRORTS_DIR, MAP_PATH)):
    for _fallback in ["maps/arena_chambers_32x32.xml", "maps/chambers32x32.xml", "maps/basesWorkers32x32A.xml"]:
        if os.path.isfile(os.path.join(MICRORTS_DIR, _fallback)):
            MAP_PATH = _fallback
            break

MAX_CYCLES   = 10000
JAVA_CMD     = "java"

DIMENSIONS        = 20
N_POP             = 20      # complete-phenotype budget per generation
K_GAMES           = 3       # games per sub-agent evaluation
TOTAL_GENERATIONS = 30
VALIDATION_GAMES  = 5       # per opponent
A_MAX             = 15      # archive cap
DELTA_S           = 0.85    # redundancy (genotype similarity) threshold
DELTA_F           = 0.20    # weak-agent fitness threshold
TOP_PARTNERS      = 3       # collaborators used in cooperative evaluation
TRAJ_WINDOW       = 7       # trajectory memory length W
MUT_BASE          = 0.15    # sigma = MUT_BASE*(1-e) + MUT_OFFSET
MUT_OFFSET        = 0.03
DRAW_SCORE        = 0.3     # training score for draw / timeout

N_WORKERS = max(1, min((os.cpu_count() or 4) - 2, 8))  # default parallel Java game servers
SEED      = 2026            # None -> non-reproducible

# ---- DI-CCE-TDCCEA parameters ----------------------------------------------
WARMUP_GENS    = 3          # random balanced decomposition before learning
WARM_GROUPS    = 4          # number of groups during warm-up
MAX_GROUPS     = 5          # cap on simultaneous co-evolution groups
MAX_GROUP_SIZE = 8          # cap on dimensions per group
TAU_I          = 0.30       # interaction threshold on normalised MST edges
RHO_SMOOTH     = 0.6        # smoothing of the interaction matrix
SAMPLE_GENS    = 4          # generations of samples kept for regression
MIN_SAMPLES    = 12         # minimum samples before interactions are measured
L_MIN          = 2          # minimum co-evolution window
L_MAX          = 6          # maximum co-evolution window
TAU_IMP        = 0.0        # joint-improvement threshold
STRENGTH_DROP  = 0.90       # window ends if interaction strength < 0.9*start
OPP_WEIGHT     = 2.0        # weight of the opponent one-hot in the fingerprint

# ---- Parameter layout ------------------------------------------------------
WEIGHT_NAMES = [
    "worker_ratio", "max_workers", "expand_threshold", "light_preference",
    "heavy_preference", "ranged_preference", "barracks_timing", "aggression",
    "attack_threshold", "defend_radius", "target_priority", "defense_trigger",
    "counter_rush", "tech_switch", "economy_pressure", "scout_priority",
    "group_attack", "second_barracks", "worker_fight_ratio", "focus_fire",
]
MACRO_INDICES = [0, 1, 2, 6, 14, 15, 17, 18]              # 8 dims
MICRO_INDICES = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 16, 19]  # 12 dims

OPPONENT_ROTATION = [
    "EconomyMilitaryRush", "LightRush", "WorkerRush",
    "HeavyDefense", "RangedRush", "EconomyRush", "WorkerRushPlusPlus",
]

ALGORITHMS = [
    ("CCE-TDCCEA (baseline)",    "cce", "fixed"),
    ("DI-CCE-TDCCEA (proposed)", "di",  "dynamic"),
]

# ============================================================================
# SMALL UTILITIES
# ============================================================================
def clamp01(x):
    return min(1.0, max(0.0, x))


def euclid(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def similarity(a, b):
    """1 - normalised Euclidean distance, clipped to [0, 1]."""
    return max(0.0, 1.0 - euclid(a, b) / math.sqrt(len(a)))


def train_score(winner):
    if winner == 0:
        return 1.0
    if winner == -1:
        return DRAW_SCORE
    return 0.0


def assemble(parts):
    """parts: iterable of (dims, vec) -> full DIMENSIONS-long vector."""
    full = [0.0] * DIMENSIONS
    for dims, vec in parts:
        for d, v in zip(dims, vec):
            full[d] = v
    return full


def sub_size(num_groups):
    """Sub-population size so that the total evaluation budget stays ~N_POP."""
    return max(4, N_POP // num_groups)


# ============================================================================
# GAME SERVER (one Java process) AND SERVER POOL (parallel evaluation)
# ============================================================================
class GameServer:
    LOG_PATH = os.path.join(SCRIPT_DIR, "gameserver_stderr.log")

    def __init__(self, index):
        self.index = index
        self.proc = None
        self._log = None

    def start(self):
        classpath = os.pathsep.join(["lib/*", "bin"])  # ';' on Windows, ':' on Unix
        self._log = open(self.LOG_PATH, "a", encoding="utf-8")
        self.proc = subprocess.Popen(
            [JAVA_CMD, "-cp", classpath, "tests.EvolvedBatchGameRunner"],
            cwd=MICRORTS_DIR,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        for _ in range(500):
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError(f"game server {self.index} exited before READY")
            if line.strip() == "READY":
                return
        raise RuntimeError(f"game server {self.index} did not report READY")

    def run_game(self, weights, opponent):
        """Returns (winner, cycles). winner==0 -> evolved agent won, -1 -> draw/timeout."""
        wstr = ";".join(f"{w:.4f}" for w in weights)
        self.proc.stdin.write(f"EVOLVED:{wstr},{opponent},{MAP_PATH},{MAX_CYCLES}\n")
        self.proc.stdin.flush()
        for _ in range(200):
            line = self.proc.stdout.readline()
            if line == "":
                raise EOFError(f"game server {self.index} closed its output")
            line = line.strip()
            if line.startswith("RESULT:"):
                try:
                    w, c = line[len("RESULT:"):].split(",")[:2]
                    return int(w), int(c)
                except ValueError:
                    return -1, MAX_CYCLES
        raise RuntimeError(f"game server {self.index} sent no RESULT line")

    def stop(self):
        if self.proc is not None:
            try:
                self.proc.stdin.write("EXIT\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
                try:
                    self.proc.wait(timeout=10)
                except Exception:
                    pass
            finally:
                for stream in (self.proc.stdin, self.proc.stdout):
                    try:
                        stream.close()
                    except Exception:
                        pass
                self.proc = None
        if self._log is not None:
            self._log.close()
            self._log = None

    def restart(self):
        self.stop()
        self.start()


class ServerPool:
    """N independent game servers; each game borrows one server exclusively."""

    def __init__(self, n):
        self.n = n
        self.servers = [GameServer(i) for i in range(n)]
        self.idle = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=n)

    def start(self):
        list(self.executor.map(lambda s: s.start(), self.servers))
        for s in self.servers:
            self.idle.put(s)

    def _play(self, job):
        weights, opponent = job
        srv = self.idle.get()
        try:
            try:
                return srv.run_game(weights, opponent)
            except Exception as exc:
                print(f"  [warn] server {srv.index} failed ({exc}); restarting")
                srv.restart()
                return srv.run_game(weights, opponent)
        except Exception as exc:
            print(f"  [error] game failed twice ({exc}); scored as timeout")
            return -1, MAX_CYCLES
        finally:
            self.idle.put(srv)

    def play_all(self, jobs):
        """jobs: list of (weights, opponent). Returns list of (winner, cycles) in order."""
        if not jobs:
            return []
        return list(self.executor.map(self._play, jobs))

    def stop(self):
        self.executor.shutdown(wait=True)
        for s in self.servers:
            s.stop()


# ============================================================================
# POPULATION STRUCTURES
# ============================================================================
class SubAgent:
    """A sub-agent: values for the dimensions of its group only."""

    def __init__(self, vec):
        self.vec = [clamp01(v) for v in vec]
        self.fit = 0.0                       # latest cooperative fitness
        self.e = None                        # trajectory score
        self.hist = deque(maxlen=TRAJ_WINDOW)  # (fitness, environment) pairs


class Group:
    """A co-evolution group (sub-population over a set of dimensions)."""

    def __init__(self, dims, members, age=1, strength0=None):
        self.dims = sorted(dims)
        self.members = members
        self.age = age
        self.strength0 = strength0   # interaction strength when window opened
        self.prev_rel = None         # previous relative joint fitness
        self.rel_now = 0.0           # current relative joint fitness
        self.dissolve = False        # window terminated -> reset at regroup


# ============================================================================
# INTERACTION-DRIVEN DECOMPOSITION (MST -> clusters)
# ============================================================================
def group_strength(dims, I):
    if len(dims) < 2:
        return 1.0
    vals = [I[i][j] for a, i in enumerate(dims) for j in dims[a + 1:]]
    return sum(vals) / len(vals)


def max_spanning_tree(W):
    """Kruskal on the complete graph; returns list of (weight, i, j)."""
    edges = sorted(((W[i][j], i, j)
                    for i in range(DIMENSIONS) for j in range(i + 1, DIMENSIONS)),
                   reverse=True)
    parent = list(range(DIMENSIONS))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = []
    for w, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            tree.append((w, i, j))
    return tree


def _reach(start, adj, allowed):
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v, _ in adj[u]:
            if v in allowed and v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def split_oversized(comp, adj):
    """Recursively cut the weakest strong edge until every piece <= MAX_GROUP_SIZE."""
    if len(comp) <= MAX_GROUP_SIZE:
        return [sorted(comp)]
    cset = set(comp)
    cand = [(w, u, v) for u in comp for v, w in adj[u] if v in cset and u < v]
    if not cand:  # cannot happen for a connected pathway; safe fallback
        return [sorted(comp[k:k + MAX_GROUP_SIZE])
                for k in range(0, len(comp), MAX_GROUP_SIZE)]
    _, u, v = min(cand)
    adj[u] = [(x, ww) for x, ww in adj[u] if x != v]
    adj[v] = [(x, ww) for x, ww in adj[v] if x != u]
    piece_u = _reach(u, adj, cset)
    piece_v = cset - piece_u
    return (split_oversized(sorted(piece_u), adj)
            + split_oversized(sorted(piece_v), adj))


def merge_clusters(clusters, I):
    """Merge most-interacting clusters until at most MAX_GROUPS remain."""
    clusters = [sorted(c) for c in clusters]
    while len(clusters) > MAX_GROUPS:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if len(clusters[a]) + len(clusters[b]) > MAX_GROUP_SIZE:
                    continue
                s = max(I[i][j] for i in clusters[a] for j in clusters[b])
                if best is None or s > best[0]:
                    best = (s, a, b)
        if best is None:
            order = sorted(range(len(clusters)), key=lambda k: len(clusters[k]))
            a, b = sorted(order[:2])
        else:
            _, a, b = best
        clusters[a] = sorted(clusters[a] + clusters[b])
        del clusters[b]
    return sorted(clusters, key=lambda c: c[0])


def build_partition(I):
    """Interaction matrix -> MST -> strong pathways -> bounded clusters."""
    tree = max_spanning_tree(I)
    adj = {d: [] for d in range(DIMENSIONS)}
    for w, i, j in tree:
        if w >= TAU_I:
            adj[i].append((j, w))
            adj[j].append((i, w))
    everything = set(range(DIMENSIONS))
    seen, comps = set(), []
    for s in range(DIMENSIONS):
        if s in seen:
            continue
        comp = sorted(_reach(s, adj, everything))
        seen.update(comp)
        comps.append(comp)
    clusters = []
    for c in comps:
        clusters.extend(split_oversized(c, adj))
    parts = merge_clusters(clusters, I)
    assert sorted(sum(parts, [])) == list(range(DIMENSIONS)), "partition must cover all dims"
    return parts


# ============================================================================
# CO-EVOLUTION ENGINE (shared by both algorithms)
# ============================================================================
class CCEEngine:
    def __init__(self, pool, mode, rng):
        assert mode in ("fixed", "dynamic")
        self.pool = pool
        self.mode = mode
        self.rng = rng
        self.groups = []
        self.archive = []                   # (e, dims tuple, vec, gen)
        self.best_full = None               # representative full phenotype
        self.best_coop = 0.0
        self.prev_mean_fit = 0.5
        self.batches = deque(maxlen=SAMPLE_GENS)   # (X, centred y) per generation
        self.I_smooth = None
        self.I_norm = [[0.0] * DIMENSIONS for _ in range(DIMENSIONS)]

    # ------------------------------------------------------------------ init
    def init_groups(self):
        if self.mode == "fixed":
            parts = [sorted(MACRO_INDICES), sorted(MICRO_INDICES)]
        else:
            dims = list(range(DIMENSIONS))
            self.rng.shuffle(dims)
            parts = [sorted(dims[i::WARM_GROUPS]) for i in range(WARM_GROUPS)]
        n = sub_size(len(parts))
        self.groups = [Group(p, self._init_members(p, n), age=1) for p in parts]

    def _default_value(self, d):
        if self.best_full is not None:
            return self.best_full[d]
        return self.rng.random()

    def _init_members(self, dims, n):
        """Seeded (archive / best representative) plus random members."""
        seed = None
        arch = [a for a in self.archive if a[1] == tuple(dims)]
        if arch:
            seed = max(arch, key=lambda a: a[0])[2]
        elif self.best_full is not None:
            seed = [self.best_full[d] for d in dims]
        members = []
        if seed is not None:
            members.append(SubAgent(seed))
            for _ in range(max(0, n // 3 - 1)):
                members.append(SubAgent([x + self.rng.gauss(0.0, 0.10) for x in seed]))
        while len(members) < n:
            members.append(SubAgent([self.rng.random() for _ in dims]))
        return members[:n]

    # ------------------------------------------------------------ regrouping
    def _remap(self, m, old_dims, new_dims):
        vec = []
        for d in new_dims:
            if d in old_dims:
                vec.append(m.vec[old_dims.index(d)])
            else:
                vec.append(self._default_value(d))
        return SubAgent(vec)

    def _fit_size(self, members, n, dims):
        members = sorted(members,
                         key=lambda m: m.e if m.e is not None else -1.0,
                         reverse=True)[:n]
        while len(members) < n:
            if members:
                parent = self.rng.choice(members)
                members.append(SubAgent([x + self.rng.gauss(0.0, 0.05) for x in parent.vec]))
            else:
                members.append(SubAgent([self.rng.random() for _ in dims]))
        return members

    def regroup(self):
        """Dynamic decomposition from the current smoothed interaction matrix."""
        parts = build_partition(self.I_norm)
        n = sub_size(len(parts))
        old = self.groups
        matched = set()
        new_groups = []
        for dims in parts:
            best_idx, best_j = None, 0.0
            for oi, og in enumerate(old):
                if oi in matched:
                    continue
                j = len(set(dims) & set(og.dims)) / len(set(dims) | set(og.dims))
                if j > best_j:
                    best_idx, best_j = oi, j

            if best_idx is not None and best_j >= 0.5:
                matched.add(best_idx)
                og = old[best_idx]
                if og.dissolve:      # window terminated -> fresh co-evolution
                    g = Group(dims, self._init_members(dims, n), age=1,
                              strength0=group_strength(dims, self.I_norm))
                else:                # continuing window
                    if og.dims == dims:
                        members = og.members
                    else:
                        members = [self._remap(m, og.dims, dims) for m in og.members]
                    members = self._fit_size(members, n, dims)
                    strength0 = og.strength0
                    if strength0 is None:
                        strength0 = group_strength(dims, self.I_norm)
                    g = Group(dims, members, age=og.age + 1, strength0=strength0)
                    g.prev_rel = og.prev_rel
            else:
                g = Group(dims, self._init_members(dims, n), age=1,
                          strength0=group_strength(dims, self.I_norm))
            new_groups.append(g)
        self.groups = new_groups

    # --------------------------------------------------------- fingerprint
    def fingerprint(self, gen, opp):
        """E_t = [R_pop, R_opp, R_game, R_fit]."""
        onehot = [OPP_WEIGHT if o == opp else 0.0 for o in OPPONENT_ROTATION]
        centroid = [0.0] * DIMENSIONS
        for g in self.groups:
            for idx, d in enumerate(g.dims):
                centroid[d] = sum(m.vec[idx] for m in g.members) / len(g.members)
        return onehot + centroid + [gen / TOTAL_GENERATIONS, self.prev_mean_fit]

    # ---------------------------------------------------------- evaluation
    def evaluate(self, gen, opp, env):
        """Cooperative evaluation: each sub-agent is paired with TOP_PARTNERS
        collaborators from every other group, K games each, all in parallel."""
        tops = [sorted(g.members, key=lambda m: m.fit, reverse=True)[:TOP_PARTNERS]
                for g in self.groups]
        jobs, keys, fulls = [], [], []
        for gi, g in enumerate(self.groups):
            for mi, m in enumerate(g.members):
                for j in range(K_GAMES):
                    parts = []
                    for gj, g2 in enumerate(self.groups):
                        if gj == gi:
                            parts.append((g2.dims, m.vec))
                        else:
                            partner = tops[gj][j % len(tops[gj])]
                            parts.append((g2.dims, partner.vec))
                    full = assemble(parts)
                    jobs.append((full, opp))
                    keys.append((gi, mi))
                    fulls.append(full)

        results = self.pool.play_all(jobs)
        scores = [train_score(w) for w, _ in results]

        acc = {}
        for key, s in zip(keys, scores):
            acc.setdefault(key, []).append(s)
        for gi, g in enumerate(self.groups):
            for mi, m in enumerate(g.members):
                vals = acc[(gi, mi)]
                m.fit = sum(vals) / len(vals)

        # trajectory memory and trajectory score e_i
        for g in self.groups:
            for m in g.members:
                m.hist.append((m.fit, env))
                num = den = 0.0
                for f, past_env in m.hist:
                    w = similarity(past_env, env)
                    num += w * f
                    den += w
                m.e = num / den if den > 1e-12 else m.fit

        # relative joint fitness per group (opponent-invariant)
        all_fit = [m.fit for g in self.groups for m in g.members]
        overall = sum(all_fit) / len(all_fit)
        for g in self.groups:
            g.rel_now = sum(m.fit for m in g.members) / len(g.members) - overall
        self.prev_mean_fit = overall
        self.best_coop = max(self.best_coop, max(all_fit))

        # interaction samples (DI only)
        if self.mode == "dynamic":
            X = np.array(fulls, dtype=float)
            y = np.array(scores, dtype=float)
            y = y - y.mean()
            self.batches.append((X, y))
            self.update_interactions()

    # ---------------------------------------------------- interactions (DI)
    def _raw_interactions(self):
        if not self.batches:
            return None
        X = np.vstack([b[0] for b in self.batches])
        y = np.concatenate([b[1] for b in self.batches])
        n = len(y)
        if n < MIN_SAMPLES:
            return None
        Xc = X - X.mean(axis=0)
        ones = np.ones(n)
        dof = n - 4
        raw = np.zeros((DIMENSIONS, DIMENSIONS))
        for i in range(DIMENSIONS):
            for j in range(i + 1, DIMENSIONS):
                A = np.column_stack([ones, Xc[:, i], Xc[:, j], Xc[:, i] * Xc[:, j]])
                beta = np.linalg.lstsq(A, y, rcond=None)[0]
                r = y - A @ beta
                s2 = float(r @ r) / dof
                cov = s2 * np.linalg.pinv(A.T @ A)
                se = math.sqrt(max(float(cov[3, 3]), 1e-18))
                t = abs(float(beta[3])) / se
                raw[i, j] = raw[j, i] = t
        return raw

    def update_interactions(self):
        raw = self._raw_interactions()
        if raw is None:
            return
        if self.I_smooth is None:
            self.I_smooth = raw
        else:
            self.I_smooth = RHO_SMOOTH * self.I_smooth + (1.0 - RHO_SMOOTH) * raw
        mx = float(self.I_smooth.max())
        if mx > 1e-12:
            self.I_norm = (self.I_smooth / mx).tolist()
        else:
            self.I_norm = [[0.0] * DIMENSIONS for _ in range(DIMENSIONS)]

    # ------------------------------------------------------- archive & windows
    def update_archive(self, gen):
        for g in self.groups:
            dims_t = tuple(g.dims)
            for m in sorted(g.members, key=lambda x: x.e, reverse=True)[:3]:
                dup = any(a[1] == dims_t and similarity(a[2], m.vec) > DELTA_S
                          for a in self.archive)
                if not dup:
                    self.archive.append((m.e, dims_t, list(m.vec), gen))
        self.archive.sort(key=lambda a: a[0], reverse=True)
        del self.archive[A_MAX:]

    def decide_windows(self, gen):
        """Terminate co-evolution windows (DI only)."""
        for g in self.groups:
            delta = None if g.prev_rel is None else g.rel_now - g.prev_rel
            g.prev_rel = g.rel_now
            if gen <= WARMUP_GENS:
                continue
            strength = group_strength(g.dims, self.I_norm)
            stop = False
            if g.age >= L_MIN:
                if delta is not None and delta < TAU_IMP:
                    stop = True
                if g.strength0 and g.strength0 > 0 and strength < STRENGTH_DROP * g.strength0:
                    stop = True
            if g.age >= L_MAX:
                stop = True
            g.dissolve = stop

    # ------------------------------------------------- representative & breeding
    def representative(self):
        full = [0.0] * DIMENSIONS
        for g in self.groups:
            best = max(g.members, key=lambda m: m.e)
            for idx, d in enumerate(g.dims):
                full[d] = best.vec[idx]
        return full

    def _survive(self, members, target):
        ranked = sorted(members, key=lambda m: m.e, reverse=True)
        kept = []
        for m in ranked:                                  # remove near-duplicates
            if all(similarity(m.vec, k.vec) <= DELTA_S for k in kept):
                kept.append(m)

        def weak(m):
            return len(m.hist) >= 3 and m.e < DELTA_F

        min_keep = max(2, int(0.4 * target))
        strong = [m for m in kept if not weak(m)]
        if len(strong) < min_keep:
            strong += [m for m in kept if weak(m)][:min_keep - len(strong)]
        return strong[:target]

    def _tournament(self, pop, k=3):
        cands = [self.rng.choice(pop) for _ in range(k)]
        return max(cands, key=lambda m: m.e)

    def _offspring(self, parents, count):
        kids = []
        for _ in range(max(0, count)):
            p1 = self._tournament(parents)
            if len(parents) > 1 and self.rng.random() < 0.3:
                p2 = self._tournament(parents)
                base = [a if self.rng.random() < 0.5 else b for a, b in zip(p1.vec, p2.vec)]
            else:
                base = list(p1.vec)
            sigma = MUT_BASE * (1.0 - p1.e) + MUT_OFFSET
            kids.append(SubAgent([x + self.rng.gauss(0.0, sigma) for x in base]))
        return kids

    def breed(self):
        target = sub_size(len(self.groups))
        for g in self.groups:
            if g.dissolve:
                continue
            survivors = self._survive(g.members, target)
            g.members = survivors + self._offspring(survivors, target - len(survivors))

    # ------------------------------------------------------------------- run
    def run(self):
        t0 = time.time()
        self.init_groups()
        for gen in range(1, TOTAL_GENERATIONS + 1):
            opp = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
            if self.mode == "dynamic" and gen > WARMUP_GENS:
                self.regroup()
            env = self.fingerprint(gen, opp)
            self.evaluate(gen, opp, env)
            self.update_archive(gen)
            if self.mode == "dynamic":
                self.decide_windows(gen)
            self.best_full = self.representative()

            extra = ""
            if self.mode == "dynamic":
                extra = " | groups=" + ",".join(str(len(g.dims)) for g in self.groups)
            print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | "
                  f"Best={self.best_coop:.3f}{extra}", flush=True)
            self.breed()
        return list(self.best_full), time.time() - t0


# ============================================================================
# VALIDATION (win = 1, everything else = 0)
# ============================================================================
def validate(pool, weights):
    jobs = [(weights, opp) for opp in OPPONENT_ROTATION for _ in range(VALIDATION_GAMES)]
    results = pool.play_all(jobs)
    per_opp = {}
    idx = 0
    for opp in OPPONENT_ROTATION:
        wins = 0
        for _ in range(VALIDATION_GAMES):
            winner, _ = results[idx]
            idx += 1
            if winner == 0:
                wins += 1
        per_opp[opp] = 100.0 * wins / VALIDATION_GAMES
    overall = sum(per_opp.values()) / len(per_opp)
    return overall, per_opp


# ============================================================================
# STATISTICS, REPORTING, OUTPUT
# ============================================================================
def wilcoxon_test(x, y):
    """Two-sided Wilcoxon signed-rank test, normal approximation. None if n<3."""
    diffs = [a - b for a, b in zip(x, y) if a != b]
    n = len(diffs)
    if n < 3:
        return None
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(diffs[order[j + 1]]) == abs(diffs[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, diffs) if d > 0)
    mean = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w_plus - mean) / sd
    return math.erfc(abs(z) / math.sqrt(2.0))


def print_report(results):
    print("\n" + "=" * 72)
    print("  RANKING (validation: 7 bots x {} games)".format(VALIDATION_GAMES))
    print("=" * 72)
    ranked = sorted(results, key=lambda r: r["wr"], reverse=True)
    print(f"  {'#':<4}{'Algorithm':<30}{'WinRate':>10}{'Time(s)':>12}")
    for i, r in enumerate(ranked, 1):
        print(f"  {i:<4}{r['name']:<30}{r['wr']:>9.1f}%{r['time']:>12.1f}")

    print("\n  PER-BOT WIN RATES (* = unique best for that bot)")
    print(f"  {'Bot':<22}" + "".join(f"{r['name'][:15]:>16}" for r in results))
    for opp in OPPONENT_ROTATION:
        vals = [r["per_opp"][opp] for r in results]
        best = max(vals)
        unique = vals.count(best) == 1
        cells = "".join(f"{v:>15.1f}{'*' if (unique and v == best) else ' '}" for v in vals)
        print(f"  {opp:<22}{cells}")


def run_statistics(results):
    base = next(r for r in results if r["key"] == "di")
    base_vals = [base["per_opp"][o] for o in OPPONENT_ROTATION]
    lines = []
    print("\n  WILCOXON SIGNED-RANK (DI-CCE-TDCCEA vs others, 7 paired per-bot win rates)")
    for r in results:
        if r is base:
            continue
        other = [r["per_opp"][o] for o in OPPONENT_ROTATION]
        p = wilcoxon_test(base_vals, other)
        txt = "n/a (fewer than 3 non-zero differences)" if p is None else f"p = {p:.4f}"
        line = f"  DI-CCE-TDCCEA vs {r['name']}: {txt}"
        print(line)
        lines.append(line)
    with open(os.path.join(SCRIPT_DIR, "wilcoxon_results_cce_di.txt"), "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + "\n".join(lines) + "\n\n")


def save_outputs(results):
    json_path = os.path.join(SCRIPT_DIR, "benchmark_cce_di_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  JSON saved: {json_path}")

    try:
        from openpyxl import Workbook, load_workbook
    except ImportError:
        print("  openpyxl not installed - Excel log skipped (pip install openpyxl)")
        return
    xlsx_path = os.path.join(SCRIPT_DIR, "benchmark_cce_di_winrates.xlsx")
    if os.path.exists(xlsx_path):
        wb = load_workbook(xlsx_path)
        ws = wb.active
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "Win Rates"
        ws.cell(1, 1, "Algorithm")
    run_no = (ws.max_column - 1) // 2 + 1
    col_wr = 2 + 2 * (run_no - 1)
    col_t = col_wr + 1
    ws.cell(1, col_wr, f"Run{run_no}")
    ws.cell(1, col_t, f"Time{run_no}")
    row_of = {ws.cell(r, 1).value: r for r in range(2, ws.max_row + 1)}
    for r in results:
        if r["name"] not in row_of:
            row = ws.max_row + 1
            ws.cell(row, 1, r["name"])
            row_of[r["name"]] = row
        row = row_of[r["name"]]
        ws.cell(row, col_wr, round(r["wr"], 2))
        ws.cell(row, col_t, round(r["time"], 1))
    wb.save(xlsx_path)
    print(f"  Excel saved: {xlsx_path} (run {run_no})")


# ============================================================================
# MAIN
# ============================================================================
def main():
    import sys
    n_workers = N_WORKERS
    args = sys.argv[1:]
    idx = 0
    selected_algos = None
    while idx < len(args):
        a = args[idx]
        if a in ("--cores", "-c") and idx + 1 < len(args):
            n_workers = int(args[idx + 1])
            idx += 2
        elif a.isdigit():
            n_workers = int(a)
            idx += 1
        elif a.lower() in ("cce", "di", "both"):
            m = a.lower()
            if m == "cce":
                selected_algos = [ALGORITHMS[0]]
            elif m == "di":
                selected_algos = [ALGORITHMS[1]]
            else:
                selected_algos = ALGORITHMS
            idx += 1
        else:
            idx += 1

    algos_to_run = selected_algos if selected_algos is not None else ALGORITHMS

    print("=" * 72)
    print("  CCE-TDCCEA vs DI-CCE-TDCCEA  |  map: " + MAP_PATH)
    print(f"  Parallel game servers: {n_workers} (available cores: {os.cpu_count()})"
          f"   Generations: {TOTAL_GENERATIONS}   Population: {N_POP}   K games: {K_GAMES}")
    print("=" * 72)

    results = []
    for idx, (name, key, mode) in enumerate(algos_to_run):
        print(f"\n{'=' * 72}\n  RUNNING: {name}\n{'=' * 72}")
        pool = ServerPool(n_workers)
        try:
            pool.start()
            rng = random.Random(None if SEED is None else SEED + idx)
            engine = CCEEngine(pool, mode, rng)
            best, elapsed = engine.run()
            wr, per_opp = validate(pool, best)
        finally:
            pool.stop()
        print(f"  -> validation win rate {wr:.1f}%  (optimisation time {elapsed:.1f}s)")
        results.append({
            "name": name, "key": key, "time": elapsed, "wr": wr,
            "per_opp": per_opp, "weights": best,
        })

    print_report(results)
    if len(results) > 1:
        run_statistics(results)
    save_outputs(results)


if __name__ == "__main__":
    main()