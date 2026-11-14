#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEM7_ER_PlanetWars_com.py
=========================
Competitive Co-Evolutionary Benchmark of ALL 14 Evolutionary Algorithm Variants
across 4 Families for Planet Wars (Google AI Challenge 2010).

COMPETITIVE CO-EVOLUTION WORKFLOW:
  Phase 1 — COMPETITIVE CO-EVOLUTIONARY ARMS RACE:
    - Populations of all algorithms evolve synchronously generation-by-generation.
    - Individuals in each algorithm are evaluated against the current champion
      strategies of rival algorithms (cross-population competitive co-evolution).
    - As one algorithm adapts to counter another, the rival algorithms co-evolve
      counter-tactics, driving an authentic competitive evolutionary arms race.

  Phase 2 — PAIRWISE HEAD-TO-HEAD TOURNAMENT MATRIX:
    - Comprehensive round-robin tournament between ALL algorithms (N × N matrix).
    - Every pair (Algo_i vs Algo_j) plays multiple games across official maps,
      alternating Player 1 and Player 2 roles to eliminate first-mover advantage.
    - Records and displays exact head-to-head match records (W-L-D) and win rates
      against every other algorithm.

  Phase 3 — BENCHMARK VALIDATION AGAINST BEST BOT (DualBot):
    - Final evolved champions validated against DualBot (the verified #1 ladder bot).

  Phase 4 — STATISTICAL ARTIFACTS & MULTI-SHEET EXCEL:
    - SEM7_ER_PlanetWars_com_table.txt      -> Text H2H tournament matrix + bot rankings
    - SEM7_ER_PlanetWars_com_results.json    -> Structured JSON with complete tournament data
    - SEM7_ER_PlanetWars_com_comparison.xlsx -> Formatted 4-sheet Excel report
    - SEM7_ER_PlanetWars_com_convergence.png -> Convergence curve plots

THE 14 ALGORITHMS (4 Families):
  ── Family 1: Pure Monolithic TD-CCEA ──
  1. TD-CCEA               (Key: 'td')
  2. TD-CCEA+SharpKernel   (Key: 'td_sharp')
  3. TD-CCEA-Upgraded      (Key: 'td_up')
  ── Family 2: Sensitivity-Adaptive SA-TDCCEA ──
  4. SA-TDCCEA             (Key: 'sa')
  5. SA-TDCCEA+SharpKernel (Key: 'sa_sharp')
  6. EC-SA-TDCCEA          (Key: 'ec_sa')
  7. EC-SA-TDCCEA-Upgraded (Key: 'ec_sa_up')
  ── Family 3: Cooperative Co-Evolution CCE-TDCCEA ──
  8. CCE-TDCCEA            (Key: 'cce')
  9. CCE-TDCCEA+SharpKernel(Key: 'cce_sharp')
  10. CCE-TDCCEA-Upgraded  (Key: 'cce_up')
  ── Family 4: Classical Optimization & Evolutionary Baselines ──
  11. Standard CCEA        (Key: 'std_ccea')
  12. CMA-ES               (Key: 'cmaes')
  13. PSO                  (Key: 'pso')
  14. Hill Climbing        (Key: 'hill')
"""

import argparse
import collections
import copy
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Force UTF-8 encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ["PYTHONIOENCODING"] = "utf-8"

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:
    _lsa = None

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    _OPENPYXL_AVAILABLE = True
except ImportError:
    _OPENPYXL_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


# =====================================================================
#  PATHS & SIMULATOR IMPORTS
# =====================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from planetwars.sim import GameState, Planet, Fleet, parse_map_file
from planetwars.opponents import dual_bot, bully_bot, rage_bot, prospector_bot, make_random_bot
from planetwars.bot_strategy import PlanetWarsStrategyBot, make_strategy_bot


# =====================================================================
#  CONFIGURATION & HYPERPARAMETERS
# =====================================================================

WEIGHT_NAMES = [
    "expand_share",      # w0
    "max_expansions",    # w1
    "expand_threshold",  # w2
    "economy_pressure",  # w3
    "reserve_ratio",     # w4
    "consolidate",       # w5
    "aggression",        # w6
    "attack_threshold",  # w7
    "target_priority",   # w8
    "group_attack",      # w9
    "focus_fire",        # w10
    "defend_radius",     # w11
    "defense_trigger",   # w12
    "counter_rush",      # w13
]

DIMENSIONS = len(WEIGHT_NAMES)
D = DIMENSIONS
SQRT_D = math.sqrt(D)

MACRO_INDICES = [0, 1, 2, 3, 4, 5]               # 6 weights (Economy & Positioning)
MICRO_INDICES = [6, 7, 8, 9, 10, 11, 12, 13]    # 8 weights (Combat & Tactics)

TARGET_OPPONENT = "DualBot"

# Defaults
N_POP             = 12
TOTAL_GENERATIONS = 6
K_GAMES           = 2       # Games per candidate evaluation
H2H_GAMES         = 8       # Games per pairwise H2H matchup
VALIDATION_MAPS   = 10      # Maps for validation
MAX_TURNS         = 200     # Turn cap

A_MAX             = 15
TRAJ_WINDOW       = 7
MU_BASE           = 0.15
EPS_DYNAMIC       = 0.03
EPS_STATIC        = 0.02
DRAW_SCORE        = 0.30
TOP_PARTNERS      = 3
SENSITIVITY_FILE  = "sensitivity_scores_planetwars.json"

TAU0              = 0.15     # Base divisive 2-means split threshold
LAM               = 0.50     # Stopping threshold exponential scaling
D_MAX             = 0.40     # Upgraded Optimal Transport distance cutoff
DC_MAX_SHARP      = 0.40     # Centroid sharpened kernel cutoff threshold

TAU_E             = 0.45
TAU_I             = 0.80
B_MAX             = 5
TAU_R             = 1.00
RHO               = 0.80
ETA               = 0.30
I_EMA             = 0.30
I_COUPLING        = 0.60
RATE_LO           = 0.08
RATE_HI           = 0.20


# =====================================================================
#  MATH & DISTANCE UTILITIES
# =====================================================================

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def euclidean_dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def centroid(vectors: Sequence[Sequence[float]]) -> List[float]:
    n = len(vectors)
    return [sum(v[j] for v in vectors) / n for j in range(len(vectors[0]))]

def mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0

def std(xs: Sequence[float]) -> float:
    xs = list(xs)
    if not xs:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

def combine_subagents(macro, micro) -> List[float]:
    """Combine macro (6-D) and micro (8-D) sub-agents into a full 14-D weight vector."""
    full = [0.5] * DIMENSIONS
    macro_vals = macro.params if hasattr(macro, "params") else macro
    micro_vals = micro.params if hasattr(micro, "params") else micro
    for i, idx in enumerate(MACRO_INDICES):
        full[idx] = clamp01(macro_vals[i])
    for i, idx in enumerate(MICRO_INDICES):
        full[idx] = clamp01(micro_vals[i])
    return full


# =====================================================================
#  MULTI-MODAL STRATEGIC MODES & OPTIMAL TRANSPORT
# =====================================================================

_fp_counter = 0

def _next_uid() -> int:
    global _fp_counter
    _fp_counter += 1
    return _fp_counter

def _sse(X: np.ndarray, idx: List[int]) -> float:
    p = X[idx]
    return float(((p - p.mean(axis=0)) ** 2).sum())

def _best_split(X: np.ndarray, idx: List[int]) -> Optional[Tuple[float, List[int], List[int]]]:
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

def build_fingerprint_modes(params_list: Sequence[Sequence[float]], fits: Sequence[float], tau0=TAU0, lam=LAM) -> Dict:
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
        near = idx[int(np.argmin(((X[idx] - centre) ** 2).sum(axis=1)))]
        reps.append(list(params_list[near]))
        counts.append(len(idx))
        gs.append(fits[near])

    return {
        "theta": np.asarray(reps),
        "count": counts,
        "w": [c / n for c in counts],
        "g": gs,
        "dim": D,
        "uid": _next_uid(),
    }

def build_subpop_fingerprint(params_list: Sequence[Sequence[float]], fits: Sequence[float], dim: int, tau0=TAU0, lam=LAM) -> Dict:
    n = len(params_list)
    sqrt_dim = math.sqrt(dim)
    X = np.asarray(params_list, dtype=float) / sqrt_dim
    clusters = [list(range(n))]
    splits = {0: _best_split(X, clusters[0])}

    while True:
        M = len(clusters)
        tau_M = tau0 * math.exp(lam * (M - 1))
        candidates = [(s[0], i) for i, s in splits.items() if s is not None]
        if not candidates:
            break
        dE, best_i = max(candidates)
        if dE <= tau_M:
            break
        _, left, right = splits[best_i]
        clusters[best_i] = left
        clusters.append(right)
        splits[best_i] = _best_split(X, left)
        splits[len(clusters) - 1] = _best_split(X, right)

    reps, counts, gs = [], [], []
    for idx in clusters:
        centre = X[idx].mean(axis=0)
        near = idx[int(np.argmin(((X[idx] - centre) ** 2).sum(axis=1)))]
        reps.append([clamp01(x) for x in params_list[near]])
        counts.append(len(idx))
        gs.append(fits[near])

    return {
        "theta": np.asarray(reps),
        "count": counts,
        "w": [c / n for c in counts],
        "g": gs,
        "dim": dim,
        "uid": _next_uid(),
    }

def _hungarian(C: List[List[float]]) -> float:
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

def d_env_modes(FA: Dict, FB: Dict) -> float:
    if FA["uid"] == FB["uid"]:
        return 0.0
    dim = FA.get("dim", D)
    sqrt_dim = math.sqrt(dim)
    count_a = FA["count"]
    count_b = FB["count"]
    na, nb = sum(count_a), sum(count_b)
    if na == nb:
        ia = np.repeat(np.arange(len(count_a)), count_a)
        ib = np.repeat(np.arange(len(count_b)), count_b)
    else:
        ia = np.repeat(np.arange(len(count_a)), [c * nb for c in count_a])
        ib = np.repeat(np.arange(len(count_b)), [c * na for c in count_b])
    A, B = FA["theta"][ia], FB["theta"][ib]
    C = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2) / sqrt_dim
    if _lsa is not None:
        r, c = _lsa(C)
        return float(C[r, c].sum() / len(ia))
    return _hungarian(C.tolist()) / len(ia)


# =====================================================================
#  DYNAMIC SENSITIVITY & COVARIANCE INTERACTION (EC-SA Parts)
# =====================================================================

def ablation_update_com(pool, memory: List[List[Dict]], baseline: List[float], rivals: List[List[float]], E, is_modes: bool = False):
    """Local ablation against competing rival algorithms."""
    vecs = []
    for j in range(D):
        for v in (0.0, 1.0):
            t = baseline[:]
            t[j] = v
            vecs.append(t)

    fits = pool.eval_batch_vs_rivals(vecs, rivals, k_games=1)
    for j in range(D):
        S = abs(fits[2 * j + 1] - fits[2 * j])
        clusters = memory[j]
        best_k, best_s = None, -1.0
        for k, c in enumerate(clusters):
            if is_modes:
                s = max(0.0, 1.0 - d_env_modes(E, c["mu"]) / D_MAX)
            else:
                s = max(0.0, 1.0 - euclidean_dist(E[:D], c["mu"][:D]) / SQRT_D)
            if s > best_s:
                best_k, best_s = k, s
        if best_k is not None and best_s >= TAU_I:
            c = clusters[best_k]
            n = c["n"]
            c["n"] = n + 1
            c["S"] = (1.0 - ETA) * c["S"] + ETA * S
        else:
            clusters.append({"mu": E, "S": S, "n": 1})

def retrieve_sensitivity_com(memory: List[List[Dict]], E, is_modes: bool = False) -> Optional[List[float]]:
    out = [None] * D
    for j in range(D):
        num = den = 0.0
        for c in memory[j]:
            if is_modes:
                s = max(0.0, 1.0 - d_env_modes(E, c["mu"]) / D_MAX)
            else:
                s = max(0.0, 1.0 - euclidean_dist(E[:D], c["mu"][:D]) / SQRT_D)
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

def sensitivity_to_scale(S: Sequence[float]) -> np.ndarray:
    mx = max(S) if max(S) > 0 else 1.0
    rate = [RATE_LO + (1.0 - s / mx) * (RATE_HI - RATE_LO) for s in S]
    m = mean(rate)
    return np.array([r / m for r in rate])

def interaction_observation(P: Sequence[Sequence[float]], f: Sequence[float]) -> np.ndarray:
    obs = np.zeros((D, D))
    X = np.asarray(P, dtype=float)
    fv = np.asarray(f, dtype=float)
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
            c = float(np.mean((pr - pr.mean()) / ps * zf))
            obs[i, j] = obs[j, i] = min(1.0, abs(c))
    return obs

def covariance_factor(I: np.ndarray) -> np.ndarray:
    C = np.eye(D) + I_COUPLING * (I - np.diag(np.diag(I)))
    w, V = np.linalg.eigh((C + C.T) / 2)
    C = (V * np.clip(w, 1e-3, None)) @ V.T
    dg = np.sqrt(np.diag(C))
    return np.linalg.cholesky(C / np.outer(dg, dg) + 1e-9 * np.eye(D))


# =====================================================================
#  AGENT & SUB-AGENT CLASSES
# =====================================================================

class Agent:
    def __init__(self, params=None, pwm=None):
        self.params = ([random.random() for _ in range(D)] if params is None
                       else [clamp01(x) for x in params])
        self.H_i = []
        self.e_i = 0.0
        self.mu_i = 0.0
        self.pwm = pwm

    def update_history(self, f: float, E):
        self.H_i.append((f, E))
        if len(self.H_i) > TRAJ_WINDOW:
            self.H_i.pop(0)

    def compute_trajectory(self, E, alpha_fn: Callable):
        if not self.H_i:
            self.e_i = 0.0
            return
        al = [alpha_fn(E, Fk) for _, Fk in self.H_i]
        t = sum(al)
        al = [a / t for a in al] if t > 0 else [1.0 / len(al)] * len(al)
        self.e_i = sum(a * self.H_i[k][0] for k, a in enumerate(al))

    def compute_mutation(self):
        self.mu_i = MU_BASE * (1.0 - self.e_i) + EPS_DYNAMIC

    def snapshot(self):
        c = Agent(self.params, self.pwm)
        c.e_i, c.H_i, c.mu_i = self.e_i, self.H_i[:], self.mu_i
        return c


class SubAgent:
    def __init__(self, dim: int, params=None, env_offset: int = 0):
        self.dim = dim
        self.env_offset = env_offset
        self.params = ([random.random() for _ in range(dim)] if params is None
                       else [clamp01(x) for x in params])
        self.H_i = []
        self.e_i = 0.0
        self.mu_i = 0.0

    def update_history(self, f: float, E):
        self.H_i.append((f, E))
        if len(self.H_i) > TRAJ_WINDOW:
            self.H_i.pop(0)

    def compute_trajectory(self, E, alpha_fn: Callable):
        if not self.H_i:
            self.e_i = 0.0
            return
        al = [alpha_fn(E, Fk) for _, Fk in self.H_i]
        t = sum(al)
        al = [a / t for a in al] if t > 0 else [1.0 / len(al)] * len(al)
        self.e_i = sum(a * self.H_i[k][0] for k, a in enumerate(al))

    def compute_mutation(self):
        self.mu_i = MU_BASE * (1.0 - self.e_i) + EPS_DYNAMIC

    def mutate(self):
        return SubAgent(
            self.dim,
            [clamp01(p + random.gauss(0, self.mu_i)) for p in self.params],
            env_offset=self.env_offset
        )


def select_and_reproduce(pop: List[Agent], fits: List[float], pwm: Optional[List[float]] = None) -> List[Agent]:
    new_pop = []
    for _ in range(len(pop)):
        cands = random.sample(range(len(pop)), 3)
        winner = max(cands, key=lambda i: fits[i])
        parent = pop[winner]
        if pwm is not None:
            child_params = [
                clamp01(p + random.gauss(0, pwm[j] * (1.0 - parent.e_i) + EPS_STATIC))
                for j, p in enumerate(parent.params)
            ]
        else:
            child_params = [
                clamp01(p + random.gauss(0, parent.mu_i))
                for p in parent.params
            ]
        child = Agent(child_params, pwm=pwm)
        child.H_i = parent.H_i[:]
        new_pop.append(child)
    return new_pop


# =====================================================================
#  TRAJECTORY SIMILARITY FACTORIES
# =====================================================================

def make_alpha_fn(variant: Dict) -> Callable:
    env_type = variant["env_type"]
    dc_max = variant.get("dc_max")
    d_max = variant.get("d_max", D_MAX)

    if env_type == "centroid":
        if dc_max is None:
            return lambda Ft, Fk: max(0.0, 1.0 - euclidean_dist(Ft, Fk) / SQRT_D)
        else:
            return lambda Ft, Fk: max(0.0, 1.0 - euclidean_dist(Ft, Fk) / (dc_max * SQRT_D))
    elif env_type == "modes":
        cache = {}
        def alpha(Ft, Fk):
            key = (Ft["uid"], Fk["uid"])
            if key not in cache:
                cache[key] = d_env_modes(Ft, Fk)
            return max(0.0, 1.0 - cache[key] / d_max)
        return alpha

    return lambda Ft, Fk: 1.0

def make_cce_subpop_alpha_fn(variant: Dict, dim: int) -> Callable:
    env_type = variant["env_type"]
    dc_max = variant.get("dc_max")
    d_max = variant.get("d_max", D_MAX)
    sqrt_dim = math.sqrt(dim)

    if env_type == "centroid":
        if dc_max is None:
            return lambda Ft, Fk: max(0.0, 1.0 - euclidean_dist(Ft, Fk) / sqrt_dim)
        else:
            return lambda Ft, Fk: max(0.0, 1.0 - euclidean_dist(Ft, Fk) / (dc_max * sqrt_dim))
    elif env_type == "modes":
        cache = {}
        def alpha(Ft, Fk):
            key = (Ft["uid"], Fk["uid"])
            if key not in cache:
                cache[key] = d_env_modes(Ft, Fk)
            return max(0.0, 1.0 - cache[key] / d_max)
        return alpha

    return lambda Ft, Fk: 1.0


# =====================================================================
#  PARALLEL SIMULATOR GAME POOL FOR COMPETITIVE CO-EVOLUTION
# =====================================================================

class PlanetWarsCoevPool:
    def __init__(self, maps_dir: Path, n_workers: int = 8, is_mock: bool = False):
        self.maps_dir = maps_dir
        self.n_workers = n_workers
        self.is_mock = is_mock
        self.cached_maps: Dict[str, GameState] = {}
        self.training_map_keys: List[str] = []
        self.validation_map_keys: List[str] = []
        self._load_maps()

    def _load_maps(self):
        if self.is_mock:
            self.training_map_keys = [f"map{i}" for i in range(1, 11)]
            self.validation_map_keys = [f"map{i}" for i in range(1, 21)]
            return

        map_files = sorted(
            self.maps_dir.glob("map*.txt"),
            key=lambda p: int(p.stem.replace("map", "")) if p.stem.replace("map", "").isdigit() else 999
        )
        for p in map_files:
            key = p.stem
            try:
                self.cached_maps[key] = parse_map_file(p)
            except Exception:
                pass

        all_keys = list(self.cached_maps.keys())
        self.training_map_keys = all_keys[:10]
        self.validation_map_keys = all_keys[:20]

    def run_one_game(
        self,
        p1_weights: Sequence[float],
        p2_weights_or_bot: any,
        map_key: str,
        max_turns: int = MAX_TURNS,
    ) -> Tuple[int, int, int, int]:
        if self.is_mock:
            s1 = sum(p1_weights) / len(p1_weights)
            s2 = 0.50 if callable(p2_weights_or_bot) else sum(p2_weights_or_bot) / len(p2_weights_or_bot)
            prob = 0.50 + 0.40 * math.tanh((s1 - s2) * 2.0)
            r = random.random()
            if r < 0.05:
                return 0, 200, 500, 500
            elif r < prob:
                return 1, 130, 1200, 0
            else:
                return 2, 130, 0, 1200

        st = self.cached_maps[map_key].clone()
        st.max_turns = max_turns

        bot1 = make_strategy_bot(p1_weights)
        bot2 = p2_weights_or_bot if callable(p2_weights_or_bot) else make_strategy_bot(p2_weights_or_bot)

        for _ in range(max_turns):
            if st.game_over:
                break
            st.step(bot1(st, 1), bot2(st, 2))

        winner = st.winner if st.winner is not None else 0
        return winner, st.turn, st.num_ships(1), st.num_ships(2)

    def eval_batch_vs_rivals(
        self,
        weight_list: Sequence[Sequence[float]],
        rival_weights_list: Sequence[Sequence[float]],
        k_games: int = 1
    ) -> List[float]:
        """Evaluate candidates by playing against a pool of competing rival algorithm champions."""
        n_maps = len(self.training_map_keys)

        def _eval_one(w_idx: int) -> float:
            w = weight_list[w_idx]
            if not rival_weights_list:
                return 0.5

            total_score = 0.0
            total_matches = 0

            # Sample rivals if pool is large
            eval_rivals = rival_weights_list
            if len(eval_rivals) > 4:
                eval_rivals = random.sample(eval_rivals, 4)

            for riv_idx, opp_w in enumerate(eval_rivals):
                for g in range(k_games):
                    map_key = self.training_map_keys[(w_idx * 2 + g) % n_maps]

                    # Match A: Candidate is P1, Rival is P2
                    w1, _, s1, s2 = self.run_one_game(w, opp_w, map_key)
                    if w1 == 1: sc1 = 1.0
                    elif w1 == 0: sc1 = DRAW_SCORE
                    else: sc1 = 0.0
                    total_score += sc1 + 0.05 * math.tanh((s1 - s2) / 100.0)
                    total_matches += 1

                    # Match B: Candidate is P2, Rival is P1
                    w2, _, r_s, my_s = self.run_one_game(opp_w, w, map_key)
                    if w2 == 2: sc2 = 1.0
                    elif w2 == 0: sc2 = DRAW_SCORE
                    else: sc2 = 0.0
                    total_score += sc2 + 0.05 * math.tanh((my_s - r_s) / 100.0)
                    total_matches += 1

            return max(0.0, total_score / max(1, total_matches))

        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            return list(ex.map(_eval_one, range(len(weight_list))))

    def run_pairwise_tournament(
        self,
        algo_names: List[str],
        algo_weights: List[List[float]],
        games_per_matchup: int = 8
    ) -> Tuple[List[List[Tuple[int, int, int]]], List[int], List[float]]:
        """
        Run complete round-robin tournament between all algorithms (N × N matrix).
        Returns: (matrix, total_wins, win_rates)
        """
        n = len(algo_names)
        matrix = [[(0, 0, 0) for _ in range(n)] for _ in range(n)]
        n_val_maps = len(self.validation_map_keys)

        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        def _eval_pair(pair):
            i, j = pair
            w_i = algo_weights[i]
            w_j = algo_weights[j]

            wins_i, wins_j, draws = 0, 0, 0
            for g in range(games_per_matchup):
                map_key = self.validation_map_keys[g % n_val_maps]

                if g % 2 == 0:
                    # i is P1, j is P2
                    winner, _, _, _ = self.run_one_game(w_i, w_j, map_key)
                    if winner == 1: wins_i += 1
                    elif winner == 2: wins_j += 1
                    else: draws += 1
                else:
                    # j is P1, i is P2
                    winner, _, _, _ = self.run_one_game(w_j, w_i, map_key)
                    if winner == 1: wins_j += 1
                    elif winner == 2: wins_i += 1
                    else: draws += 1

            return (i, j, wins_i, wins_j, draws)

        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            results = list(ex.map(_eval_pair, pairs))

        for i, j, w_i, w_j, d in results:
            matrix[i][j] = (w_i, w_j, d)
            matrix[j][i] = (w_j, w_i, d)

        total_wins = [0] * n
        total_games = [0] * n
        win_rates = [0.0] * n

        for i in range(n):
            w_count = sum(matrix[i][j][0] for j in range(n) if i != j)
            d_count = sum(matrix[i][j][2] for j in range(n) if i != j)
            g_count = sum((matrix[i][j][0] + matrix[i][j][1] + matrix[i][j][2]) for j in range(n) if i != j)
            total_wins[i] = w_count
            total_games[i] = g_count
            win_rates[i] = ((w_count + 0.5 * d_count) / max(1, g_count)) * 100.0

        return matrix, total_wins, win_rates

    def validate_vs_dualbot(self, weights: Sequence[float], val_maps: List[str]) -> Dict:
        """Validate champion strategy against DualBot."""
        wins, losses, draws = 0, 0, 0
        total_turns = 0
        ship_margins = []

        for map_key in val_maps:
            # P1 role
            w1, t1, s1, s2 = self.run_one_game(weights, dual_bot, map_key)
            total_turns += t1
            ship_margins.append(s1 - s2)
            if w1 == 1: wins += 1
            elif w1 == 2: losses += 1
            else: draws += 1

            # P2 role
            w2, t2, sd, sc = self.run_one_game([0.5]*14, weights, map_key)
            total_turns += t2
            ship_margins.append(sc - sd)
            if w2 == 2: wins += 1
            elif w2 == 1: losses += 1
            else: draws += 1

        tot = len(val_maps) * 2
        return {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "total_matches": tot,
            "winrate": ((wins + 0.5 * draws) / tot) * 100.0,
            "avg_turns": total_turns / tot,
            "avg_margin": mean(ship_margins),
        }


# =====================================================================
#  ALGORITHM REGISTRY (ALL 14 VARIANTS)
# =====================================================================

ALL_VARIANTS = {
    # Family 1: Pure Monolithic TD-CCEA
    "td":                    {"name": "TD-CCEA",                "family": "Pure TD-CCEA", "env_type": "centroid", "dc_max": None,         "d_max": None,   "sens_type": "none",    "is_cce": False, "is_baseline": False},
    "td_sharp":              {"name": "TD-CCEA+SharpKernel",    "family": "Pure TD-CCEA", "env_type": "centroid", "dc_max": DC_MAX_SHARP, "d_max": None,   "sens_type": "none",    "is_cce": False, "is_baseline": False},
    "td_up":                 {"name": "TD-CCEA-Upgraded",       "family": "Pure TD-CCEA", "env_type": "modes",    "dc_max": None,         "d_max": D_MAX,  "sens_type": "none",    "is_cce": False, "is_baseline": False},
    # Family 2: SA-TDCCEA
    "sa":                    {"name": "SA-TDCCEA",              "family": "SA-TDCCEA",    "env_type": "centroid", "dc_max": None,         "d_max": None,   "sens_type": "static",  "is_cce": False, "is_baseline": False},
    "sa_sharp":              {"name": "SA-TDCCEA+SharpKernel",  "family": "SA-TDCCEA",    "env_type": "centroid", "dc_max": DC_MAX_SHARP, "d_max": None,   "sens_type": "static",  "is_cce": False, "is_baseline": False},
    "ec_sa":                 {"name": "EC-SA-TDCCEA",           "family": "SA-TDCCEA",    "env_type": "centroid", "dc_max": None,         "d_max": None,   "sens_type": "dynamic", "is_cce": False, "is_baseline": False},
    "ec_sa_up":              {"name": "EC-SA-TDCCEA-Upgraded",   "family": "SA-TDCCEA",    "env_type": "modes",    "dc_max": None,         "d_max": D_MAX,  "sens_type": "dynamic", "is_cce": False, "is_baseline": False},
    # Family 3: CCE-TDCCEA
    "cce":                   {"name": "CCE-TDCCEA",             "family": "CCE-TDCCEA",   "env_type": "centroid", "dc_max": None,         "d_max": None,   "sens_type": "none",    "is_cce": True,  "is_baseline": False},
    "cce_sharp":             {"name": "CCE-TDCCEA+SharpKernel", "family": "CCE-TDCCEA",   "env_type": "centroid", "dc_max": DC_MAX_SHARP, "d_max": None,   "sens_type": "none",    "is_cce": True,  "is_baseline": False},
    "cce_up":                {"name": "CCE-TDCCEA-Upgraded",   "family": "CCE-TDCCEA",   "env_type": "modes",    "dc_max": None,         "d_max": D_MAX,  "sens_type": "none",    "is_cce": True,  "is_baseline": False},
    # Family 4: Baselines
    "std_ccea":              {"name": "Standard CCEA",          "family": "Baselines",    "env_type": "none",     "dc_max": None,         "d_max": None,   "sens_type": "none",    "is_cce": False, "is_baseline": True, "base_type": "std_ccea"},
    "cmaes":                 {"name": "CMA-ES",                 "family": "Baselines",    "env_type": "none",     "dc_max": None,         "d_max": None,   "sens_type": "none",    "is_cce": False, "is_baseline": True, "base_type": "cmaes"},
    "pso":                   {"name": "PSO",                    "family": "Baselines",    "env_type": "none",     "dc_max": None,         "d_max": None,   "sens_type": "none",    "is_cce": False, "is_baseline": True, "base_type": "pso"},
    "hill":                  {"name": "Hill Climbing",         "family": "Baselines",    "env_type": "none",     "dc_max": None,         "d_max": None,   "sens_type": "none",    "is_cce": False, "is_baseline": True, "base_type": "hill"},
}

SUITE_CHOICES = {
    "all": list(ALL_VARIANTS.keys()),
    "core": ["td", "td_up", "sa", "ec_sa_up", "cce", "cce_up", "std_ccea", "cmaes"],
    "td": ["td", "td_sharp", "td_up"],
    "sa": ["sa", "sa_sharp", "ec_sa", "ec_sa_up"],
    "cce": ["cce", "cce_sharp", "cce_up"],
    "baseline": ["std_ccea", "cmaes", "pso", "hill"],
}


# =====================================================================
#  MAIN COMPETITIVE CO-EVOLUTIONARY BENCHMARK ROUTINE
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Competitive Co-Evolution Benchmark for Planet Wars (H2H Matrix between all algorithms)"
    )
    parser.add_argument("--suite", choices=list(SUITE_CHOICES.keys()), default="all",
                        help="Algorithm suite to evaluate (default: 'all' = 14 algorithms)")
    parser.add_argument("--generations", type=int, default=TOTAL_GENERATIONS,
                        help="Generations of competitive co-evolution (default: 6)")
    parser.add_argument("--pop-size", type=int, default=N_POP,
                        help="Population size per algorithm (default: 12)")
    parser.add_argument("--k-games", type=int, default=K_GAMES,
                        help="Games per candidate evaluation during evolution (default: 2)")
    parser.add_argument("--h2h-games", type=int, default=H2H_GAMES,
                        help="Games per pairwise head-to-head matchup (default: 8)")
    parser.add_argument("--val-maps", type=int, default=VALIDATION_MAPS,
                        help="Maps for tournament & validation (default: 10)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker threads (default: CPU count - 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--mock", action="store_true",
                        help="Fast dry-run mock mode")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    active_keys = SUITE_CHOICES[args.suite]
    n_algos = len(active_keys)
    n_workers = args.workers or max(1, min(12, (os.cpu_count() or 4) - 1))

    maps_dir = SCRIPT_DIR / "engine" / "maps"
    if not maps_dir.exists():
        maps_dir = SCRIPT_DIR.parent / "PlanetWars" / "engine" / "maps"

    print("\n" + "═" * 84)
    print(f"  SEM7_ER_PlanetWars_com — Competitive Co-Evolution & Head-to-Head Tournament Matrix")
    print("═" * 84)
    print(f"  Active Suite ({n_algos} Algorithms):")
    for idx, k in enumerate(active_keys, 1):
        v = ALL_VARIANTS[k]
        print(f"    {idx:2d}. {v['name']:<26s} [{v['family']:<12s}] (env={v['env_type']}, sens={v['sens_type']})")
    print(f"  Co-Evolution: Pop={args.pop_size} | Gens={args.generations} | Cross-Rival Competitive Evaluation")
    print(f"  Tournament:   {n_algos}×{n_algos} Full Round-Robin Matrix ({n_algos*(n_algos-1)//2} Pairs, {args.h2h_games} Games/Pair)")
    print(f"  Workers:      {n_workers} CPU Worker Threads")
    if args.mock:
        print("  MODE:         *** FAST MOCK TESTING MODE ***")
    print("═" * 84 + "\n")

    pool = PlanetWarsCoevPool(maps_dir=maps_dir, n_workers=n_workers, is_mock=args.mock)
    if args.val_maps < len(pool.validation_map_keys):
        pool.validation_map_keys = pool.validation_map_keys[:args.val_maps]

    alpha_fns = {
        k: make_alpha_fn(ALL_VARIANTS[k])
        for k in active_keys
        if not ALL_VARIANTS[k]["is_cce"] and not ALL_VARIANTS[k].get("is_baseline")
    }

    dim_macro = len(MACRO_INDICES)
    dim_micro = len(MICRO_INDICES)
    cce_alpha_macro = {
        k: make_cce_subpop_alpha_fn(ALL_VARIANTS[k], dim_macro)
        for k in active_keys if ALL_VARIANTS[k]["is_cce"]
    }
    cce_alpha_micro = {
        k: make_cce_subpop_alpha_fn(ALL_VARIANTS[k], dim_micro)
        for k in active_keys if ALL_VARIANTS[k]["is_cce"]
    }

    # Initialize Populations
    init_params = [[random.random() for _ in range(DIMENSIONS)] for _ in range(args.pop_size)]
    init_params_macro = [[random.random() for _ in range(dim_macro)] for _ in range(args.pop_size)]
    init_params_micro = [[random.random() for _ in range(dim_micro)] for _ in range(args.pop_size)]

    populations = {}
    archives = {k: [] for k in active_keys if not ALL_VARIANTS[k].get("is_baseline")}
    convergence_hist = {k: [] for k in active_keys}

    cce_state = {}
    baseline_state = {}

    for k in active_keys:
        v = ALL_VARIANTS[k]
        if v["is_cce"]:
            cce_state[k] = {
                "pop_a": [SubAgent(dim_macro, p, env_offset=0) for p in init_params_macro],
                "pop_b": [SubAgent(dim_micro, p, env_offset=dim_macro) for p in init_params_micro],
            }
        elif v.get("is_baseline"):
            btype = v["base_type"]
            if btype == "std_ccea":
                baseline_state[k] = {
                    "pm": [p[:] for p in init_params_macro],
                    "pi": [p[:] for p in init_params_micro],
                }
            elif btype == "cmaes":
                es_inst = None
                has_cma = False
                try:
                    import cma
                    es_inst = cma.CMAEvolutionStrategy([0.5] * DIMENSIONS, 0.3, {
                        'maxiter': args.generations,
                        'popsize': args.pop_size,
                        'bounds': [0, 1],
                        'verbose': -9,
                        'seed': args.seed,
                    })
                    has_cma = True
                except Exception:
                    pass
                baseline_state[k] = {
                    "es": es_inst,
                    "has_cma": has_cma,
                    "pop": [p[:] for p in init_params],
                    "sigma": 0.3,
                    "best_p": [0.5] * DIMENSIONS,
                    "best_f": 0.0,
                }
            elif btype == "pso":
                pos = [p[:] for p in init_params]
                vel = [[random.uniform(-0.1, 0.1) for _ in range(DIMENSIONS)] for _ in range(args.pop_size)]
                baseline_state[k] = {
                    "pos": pos,
                    "vel": vel,
                    "pb": [p[:] for p in pos],
                    "pf": [0.0] * args.pop_size,
                    "gb": pos[0][:],
                    "gf": 0.0,
                    "W": 0.7, "C1": 1.5, "C2": 1.5,
                }
            elif btype == "hill":
                baseline_state[k] = {
                    "current": init_params[0][:],
                    "cfit": 0.0,
                }
        else:
            populations[k] = [Agent(p) for p in init_params]

    ec_state = {}
    for k in active_keys:
        if ALL_VARIANTS[k]["sens_type"] == "dynamic":
            ec_state[k] = {
                "memory": [[] for _ in range(D)],
                "I": np.zeros((D, D)),
                "E_prev": None,
                "E_last": None,
                "t_last": 0,
                "R": 0.0,
            }

    # Initialize current best weights
    current_best_weights = {}
    for k in active_keys:
        if ALL_VARIANTS[k]["is_cce"]:
            current_best_weights[k] = combine_subagents(cce_state[k]["pop_a"][0], cce_state[k]["pop_b"][0])
        elif ALL_VARIANTS[k].get("is_baseline"):
            btype = ALL_VARIANTS[k]["base_type"]
            if btype == "std_ccea":
                current_best_weights[k] = combine_subagents(baseline_state[k]["pm"][0], baseline_state[k]["pi"][0])
            elif btype == "cmaes":
                current_best_weights[k] = baseline_state[k]["best_p"][:]
            elif btype == "pso":
                current_best_weights[k] = baseline_state[k]["gb"][:]
            elif btype == "hill":
                current_best_weights[k] = baseline_state[k]["current"][:]
        else:
            current_best_weights[k] = populations[k][0].params[:]

    # =================================================================
    #  PHASE 1: SYNCHRONOUS COMPETITIVE CO-EVOLUTION
    # =================================================================
    print("=" * 84)
    print(f"  PHASE 1: COMPETITIVE CO-EVOLUTION — {n_algos} Populations Evolving in Parallel")
    print("=" * 84)

    t_evol_start = time.time()

    for gen in range(1, args.generations + 1):
        gen_start_t = time.time()
        print(f"\n  [Gen {gen:02d}/{args.generations:02d}] Competitive evaluation across rival populations...", end="", flush=True)

        # ── 1. Monolithic TD-CCEA & SA-TDCCEA ──
        for k in active_keys:
            if ALL_VARIANTS[k]["is_cce"] or ALL_VARIANTS[k].get("is_baseline"):
                continue

            pop = populations[k]
            rivals = [current_best_weights[r] for r in active_keys if r != k]
            fits = pool.eval_batch_vs_rivals([a.params for a in pop], rivals, k_games=args.k_games)

            if ALL_VARIANTS[k]["env_type"] == "modes":
                E = build_fingerprint_modes([a.params for a in pop], fits)
            else:
                E = centroid([a.params for a in pop])

            best_idx = int(np.argmax(fits))
            current_best_weights[k] = pop[best_idx].params[:]
            convergence_hist[k].append(float(fits[best_idx]))

            if not archives[k] or fits[best_idx] >= archives[k][-1].e_i:
                archives[k].append(pop[best_idx].snapshot())
                if len(archives[k]) > A_MAX:
                    archives[k].pop(0)

            alpha_fn = alpha_fns[k]
            for i, a in enumerate(pop):
                a.update_history(fits[i], E)
                a.compute_trajectory(E, alpha_fn)
                a.compute_mutation()

            if ALL_VARIANTS[k]["sens_type"] == "dynamic":
                ecs = ec_state[k]
                is_modes = (ALL_VARIANTS[k]["env_type"] == "modes")
                E_fingerprint = E

                if ecs["E_prev"] is not None:
                    d_t = d_env_modes(E_fingerprint, ecs["E_prev"]) if is_modes else (1.0 - max(0.0, 1.0 - euclidean_dist(E_fingerprint[:D], ecs["E_prev"][:D]) / SQRT_D))
                    ecs["R"] = RHO * ecs["R"] + d_t
                ecs["E_prev"] = E_fingerprint

                trigger = (ecs["E_last"] is None or ecs["R"] >= TAU_R or (gen - ecs["t_last"]) >= B_MAX)
                if trigger:
                    ablation_update_com(pool, ecs["memory"], current_best_weights[k], rivals, E_fingerprint, is_modes=is_modes)
                    ecs["E_last"] = E_fingerprint
                    ecs["t_last"] = gen
                    ecs["R"] = 0.0

                obs = interaction_observation([a.params for a in pop], fits)
                ecs["I"] = (1.0 - I_EMA) * ecs["I"] + I_EMA * obs
                L = covariance_factor(ecs["I"])

                S = retrieve_sensitivity_com(ecs["memory"], E_fingerprint, is_modes=is_modes)
                scales = sensitivity_to_scale(S) if S is not None else np.ones(D)

                new_pop = []
                for _ in range(len(pop)):
                    cands = random.sample(range(len(pop)), 3)
                    w_idx = max(cands, key=lambda idx: fits[idx])
                    p = pop[w_idx]
                    z = np.random.randn(D)
                    dz = L @ (z * scales) * p.mu_i
                    ch = Agent([clamp01(p.params[j] + dz[j]) for j in range(D)])
                    ch.H_i = p.H_i[:]
                    new_pop.append(ch)
                populations[k] = new_pop
            else:
                populations[k] = select_and_reproduce(pop, fits)

        # ── 2. Cooperative Co-Evolution (CCE-TDCCEA) ──
        for k in active_keys:
            if not ALL_VARIANTS[k]["is_cce"]:
                continue

            cs = cce_state[k]
            pop_a, pop_b = cs["pop_a"], cs["pop_b"]
            rivals = [current_best_weights[r] for r in active_keys if r != k]

            top_b = sorted(pop_b, key=lambda b: b.e_i, reverse=True)[:TOP_PARTNERS]
            vecs_a = [combine_subagents(a, b) for a in pop_a for b in top_b]
            fits_a_all = pool.eval_batch_vs_rivals(vecs_a, rivals, k_games=1)
            fits_a = [mean(fits_a_all[i * TOP_PARTNERS: (i + 1) * TOP_PARTNERS]) for i in range(len(pop_a))]

            top_a = sorted(pop_a, key=lambda a: a.e_i, reverse=True)[:TOP_PARTNERS]
            vecs_b = [combine_subagents(a, b) for b in pop_b for a in top_a]
            fits_b_all = pool.eval_batch_vs_rivals(vecs_b, rivals, k_games=1)
            fits_b = [mean(fits_b_all[i * TOP_PARTNERS: (i + 1) * TOP_PARTNERS]) for i in range(len(pop_b))]

            if ALL_VARIANTS[k]["env_type"] == "modes":
                E_a = build_subpop_fingerprint([a.params for a in pop_a], fits_a, dim_macro)
                E_b = build_subpop_fingerprint([b.params for b in pop_b], fits_b, dim_micro)
            else:
                E_a = centroid([a.params for a in pop_a])
                E_b = centroid([b.params for b in pop_b])

            best_a = pop_a[int(np.argmax(fits_a))]
            best_b = pop_b[int(np.argmax(fits_b))]
            current_best_weights[k] = combine_subagents(best_a, best_b)
            convergence_hist[k].append(float((max(fits_a) + max(fits_b)) / 2.0))

            for i, a in enumerate(pop_a):
                a.update_history(fits_a[i], E_a)
                a.compute_trajectory(E_a, cce_alpha_macro[k])
                a.compute_mutation()

            for i, b in enumerate(pop_b):
                b.update_history(fits_b[i], E_b)
                b.compute_trajectory(E_b, cce_alpha_micro[k])
                b.compute_mutation()

            new_pop_a = [pop_a[max(random.sample(range(len(pop_a)), 3), key=lambda i: fits_a[i])].mutate() for _ in range(len(pop_a))]
            new_pop_b = [pop_b[max(random.sample(range(len(pop_b)), 3), key=lambda i: fits_b[i])].mutate() for _ in range(len(pop_b))]
            cs["pop_a"] = new_pop_a
            cs["pop_b"] = new_pop_b

        # ── 3. Baselines ──
        for k in active_keys:
            if not ALL_VARIANTS[k].get("is_baseline"):
                continue

            btype = ALL_VARIANTS[k]["base_type"]
            bs = baseline_state[k]
            rivals = [current_best_weights[r] for r in active_keys if r != k]

            if btype == "std_ccea":
                pairs = [combine_subagents(bs["pm"][i], bs["pi"][i]) for i in range(len(bs["pm"]))]
                fits = pool.eval_batch_vs_rivals(pairs, rivals, k_games=args.k_games)
                best_i = int(np.argmax(fits))
                current_best_weights[k] = pairs[best_i][:]
                convergence_hist[k].append(float(fits[best_i]))

                bs["pm"] = [
                    [clamp01(x + random.gauss(0, 0.15)) for x in bs["pm"][max(random.sample(range(len(bs["pm"])), 3), key=lambda x: fits[x])]]
                    for _ in range(len(bs["pm"]))
                ]
                bs["pi"] = [
                    [clamp01(x + random.gauss(0, 0.15)) for x in bs["pi"][max(random.sample(range(len(bs["pi"])), 3), key=lambda x: fits[x])]]
                    for _ in range(len(bs["pi"]))
                ]

            elif btype == "cmaes":
                if bs["has_cma"] and bs["es"] is not None:
                    pop_cma = bs["es"].ask()
                    pop_clipped = [[clamp01(x) for x in cand] for cand in pop_cma]
                    fits = pool.eval_batch_vs_rivals(pop_clipped, rivals, k_games=args.k_games)
                    bs["es"].tell(pop_cma, [-f for f in fits])
                    best_i = int(np.argmax(fits))
                    if fits[best_i] > bs["best_f"]:
                        bs["best_f"] = fits[best_i]
                        bs["best_p"] = pop_clipped[best_i][:]
                    current_best_weights[k] = bs["best_p"][:]
                    convergence_hist[k].append(float(bs["best_f"]))
                else:
                    pop_cand = bs["pop"]
                    fits = pool.eval_batch_vs_rivals(pop_cand, rivals, k_games=args.k_games)
                    best_i = int(np.argmax(fits))
                    if fits[best_i] > bs["best_f"]:
                        bs["best_f"] = fits[best_i]
                        bs["best_p"] = pop_cand[best_i][:]
                    current_best_weights[k] = bs["best_p"][:]
                    convergence_hist[k].append(float(bs["best_f"]))
                    bs["pop"] = [
                        [clamp01(x + random.gauss(0, max(0.05, bs["sigma"] * 0.95))) for x in bs["best_p"]]
                        for _ in range(len(pop_cand))
                    ]

            elif btype == "pso":
                fits = pool.eval_batch_vs_rivals(bs["pos"], rivals, k_games=args.k_games)
                for i in range(len(bs["pos"])):
                    if fits[i] > bs["pf"][i]:
                        bs["pf"][i] = fits[i]
                        bs["pb"][i] = bs["pos"][i][:]
                    if fits[i] > bs["gf"]:
                        bs["gf"] = fits[i]
                        bs["gb"] = bs["pos"][i][:]
                current_best_weights[k] = bs["gb"][:]
                convergence_hist[k].append(float(bs["gf"]))

                for i in range(len(bs["pos"])):
                    for d in range(DIMENSIONS):
                        r1, r2 = random.random(), random.random()
                        bs["vel"][i][d] = (bs["W"] * bs["vel"][i][d] +
                                           bs["C1"] * r1 * (bs["pb"][i][d] - bs["pos"][i][d]) +
                                           bs["C2"] * r2 * (bs["gb"][d] - bs["pos"][i][d]))
                        bs["vel"][i][d] = max(-0.2, min(0.2, bs["vel"][i][d]))
                        bs["pos"][i][d] = clamp01(bs["pos"][i][d] + bs["vel"][i][d])

            elif btype == "hill":
                curr = bs["current"]
                cands = [curr[:]] + [[clamp01(x + random.gauss(0, 0.12)) for x in curr] for _ in range(args.pop_size - 1)]
                fits = pool.eval_batch_vs_rivals(cands, rivals, k_games=args.k_games)
                best_i = int(np.argmax(fits))
                if fits[best_i] >= bs["cfit"]:
                    bs["cfit"] = fits[best_i]
                    bs["current"] = cands[best_i][:]
                current_best_weights[k] = bs["current"][:]
                convergence_hist[k].append(float(bs["cfit"]))

        dt = time.time() - gen_start_t
        top_k = max(active_keys, key=lambda k: convergence_hist[k][-1] if convergence_hist[k] else 0)
        print(f" Done ({dt:.2f}s) | Leader: {ALL_VARIANTS[top_k]['name']} (fit={convergence_hist[top_k][-1]:.3f})")

    t_evol_total = time.time() - t_evol_start
    print(f"\n  [CO-EVOLUTION COMPLETE] Completed {args.generations} generations in {t_evol_total:.2f}s.")

    # =================================================================
    #  PHASE 2: PAIRWISE HEAD-TO-HEAD TOURNAMENT MATRIX
    # =================================================================
    print("\n" + "=" * 84)
    print(f"  PHASE 2: ROUND-ROBIN HEAD-TO-HEAD TOURNAMENT ({n_algos}×{n_algos} Matrix)")
    print(f"  {n_algos*(n_algos-1)//2} Pairwise Matchups × {args.h2h_games} Games/Matchup")
    print("=" * 84)

    algo_names = [ALL_VARIANTS[k]["name"] for k in active_keys]
    algo_weights = [current_best_weights[k] for k in active_keys]

    t_tourn_start = time.time()
    matrix, total_wins, win_rates = pool.run_pairwise_tournament(
        algo_names, algo_weights, games_per_matchup=args.h2h_games
    )
    t_tourn_total = time.time() - t_tourn_start
    print(f"  Completed {n_algos*(n_algos-1)//2 * args.h2h_games} tournament games in {t_tourn_total:.2f}s.\n")

    # Format Head-to-Head Table
    col_w = 11
    h2h_table_lines = []
    header_str = f"  {'Algorithm':<24s}"
    for idx in range(1, n_algos + 1):
        header_str += f" {f'[{idx:02d}]':>{col_w}s}"
    header_str += f"  {'Wins':>6s}  {'WR%':>7s}"
    h2h_table_lines.append("=" * len(header_str))
    h2h_table_lines.append(f"  PLANET WARS: HEAD-TO-HEAD TOURNAMENT WIN RATE MATRIX ({n_algos}×{n_algos})")
    h2h_table_lines.append("=" * len(header_str))
    h2h_table_lines.append(header_str)
    sep_str = "  " + "─" * 24 + (" " + "─" * col_w) * n_algos + "  " + "─" * 6 + "  " + "─" * 7
    h2h_table_lines.append(sep_str)

    best_algo_idx = int(np.argmax(win_rates))

    for i in range(n_algos):
        name_label = f"{i+1:02d}. {algo_names[i]}"
        if i == best_algo_idx:
            name_label += "*"
        row = f"  {name_label:<24s}"
        for j in range(n_algos):
            if i == j:
                row += f" {'—':>{col_w}s}"
            else:
                w, l, d = matrix[i][j]
                cell = f"{w}W-{l}L-{d}D"
                row += f" {cell:>{col_w}s}"
        row += f"  {total_wins[i]:>6d}  {win_rates[i]:>6.1f}%"
        h2h_table_lines.append(row)

    h2h_table_lines.append(sep_str)
    h2h_table_lines.append(f"  * Tournament Champion: {algo_names[best_algo_idx]} ({win_rates[best_algo_idx]:.1f}% Win Rate, {total_wins[best_algo_idx]} Wins)")
    h2h_table_lines.append("=" * len(header_str))

    h2h_table_text = "\n".join(h2h_table_lines)
    print(h2h_table_text + "\n")

    # =================================================================
    #  PHASE 3: VALIDATION AGAINST DUALBOT (BEST BOT)
    # =================================================================
    print("=" * 84)
    print(f"  PHASE 3: BENCHMARK VALIDATION AGAINST BEST BOT (DualBot)")
    print("=" * 84)

    bot_val_results = []
    for idx, k in enumerate(active_keys):
        res = pool.validate_vs_dualbot(current_best_weights[k], pool.validation_map_keys)
        res["key"] = k
        res["name"] = ALL_VARIANTS[k]["name"]
        res["family"] = ALL_VARIANTS[k]["family"]
        res["weights"] = current_best_weights[k]
        res["h2h_winrate"] = win_rates[idx]
        res["h2h_wins"] = total_wins[idx]
        bot_val_results.append(res)
        print(f"  [{idx+1:02d}/{n_algos:02d}] {res['name']:<26s} vs DualBot -> "
              f"WinRate: {res['winrate']:5.1f}% ({res['wins']:2d}W-{res['losses']:2d}L-{res['draws']:2d}D) | "
              f"Margin: {res['avg_margin']:+6.0f} ships")

    # =================================================================
    #  PHASE 4: SAVE ARTIFACTS
    # =================================================================
    # 1. Text Summary Table
    txt_path = SCRIPT_DIR / "SEM7_ER_PlanetWars_com_table.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(h2h_table_text + "\n\n")
        f.write("=" * 84 + "\n")
        f.write("  BENCHMARK VALIDATION AGAINST DUALBOT:\n")
        f.write("=" * 84 + "\n")
        f.write(f"  {'Algorithm':<26s} {'Family':<14s} {'DualBot WR':<12s} {'Record':<14s} {'Ship Margin':<12s} {'H2H WR':<10s}\n")
        f.write("  " + "─" * 80 + "\n")
        for r in sorted(bot_val_results, key=lambda x: x["winrate"], reverse=True):
            f.write(f"  {r['name']:<26s} {r['family']:<14s} {r['winrate']:>6.1f}%      "
                    f"{r['wins']}W-{r['losses']}L-{r['draws']}D   {r['avg_margin']:>+6.0f} ships    {r['h2h_winrate']:>6.1f}%\n")
        f.write("\n" + "=" * 84 + "\n")
        f.write("  FINAL EVOLVED STRATEGY WEIGHTS (14 Parameters):\n")
        f.write("=" * 84 + "\n")
        for r in bot_val_results:
            w_str = ", ".join(f"{w:.2f}" for w in r["weights"])
            f.write(f"  {r['name']:<26s}: [{w_str}]\n")
    print(f"\n  Saved Text Report -> {txt_path}")

    # 2. Complete JSON Data
    json_path = SCRIPT_DIR / "SEM7_ER_PlanetWars_com_results.json"
    json_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "generations": args.generations,
            "pop_size": args.pop_size,
            "h2h_games": args.h2h_games,
            "val_maps": len(pool.validation_map_keys),
            "max_turns": MAX_TURNS,
        },
        "head_to_head": {
            "algorithms": algo_names,
            "matrix": [
                [
                    {"wins": matrix[i][j][0], "losses": matrix[i][j][1], "draws": matrix[i][j][2]}
                    if i != j else None
                    for j in range(n_algos)
                ]
                for i in range(n_algos)
            ],
            "total_wins": total_wins,
            "win_rates": win_rates,
            "champion": algo_names[best_algo_idx],
        },
        "vs_dualbot": bot_val_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved JSON Report -> {json_path}")

    # 3. Excel Multi-Sheet Report
    if _OPENPYXL_AVAILABLE:
        xlsx_path = SCRIPT_DIR / "SEM7_ER_PlanetWars_com_comparison.xlsx"
        export_excel_com(algo_names, matrix, total_wins, win_rates, bot_val_results, xlsx_path)
        print(f"  Saved Excel Report -> {xlsx_path}")

    # 4. Convergence Plot
    if _MATPLOTLIB_AVAILABLE:
        png_path = SCRIPT_DIR / "SEM7_ER_PlanetWars_com_convergence.png"
        plot_convergence_com(convergence_hist, active_keys, png_path)
        print(f"  Saved Convergence Plot -> {png_path}")

    print("\n" + "═" * 84)
    print("  COMPETITIVE CO-EVOLUTION BENCHMARK COMPLETED SUCCESSFULLY!")
    print("═" * 84 + "\n")


# =====================================================================
#  EXCEL EXPORT (HEAD-TO-HEAD MATRIX & VALIDATION)
# =====================================================================

def export_excel_com(algo_names, matrix, total_wins, win_rates, bot_results, path: Path):
    wb = openpyxl.Workbook()
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    bold_font = Font(name="Calibri", size=11, bold=True)
    body_font = Font(name="Calibri", size=11)
    title_font = Font(name="Calibri", size=14, bold=True, color="1F497D")

    dark_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    best_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"), right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"), bottom=Side(style="thin", color="D9D9D9"),
    )

    # ── Sheet 1: Head-to-Head Tournament Matrix ──
    ws1 = wb.active
    ws1.title = "H2H Tournament Matrix"
    ws1.views.sheetView[0].showGridLines = True

    ws1.cell(row=1, column=1, value="Planet Wars — Competitive Co-Evolution Head-to-Head Tournament Matrix").font = title_font
    h_row = ["Algorithm"] + algo_names + ["Total Wins", "Total Games", "H2H Win Rate"]
    for col_idx, h in enumerate(h_row, 1):
        c = ws1.cell(row=3, column=col_idx, value=h)
        c.font = header_font
        c.fill = dark_fill
        c.alignment = Alignment(horizontal="center" if col_idx > 1 else "left", vertical="center")

    best_idx = int(np.argmax(win_rates))
    for i in range(len(algo_names)):
        tot_g = sum((matrix[i][j][0] + matrix[i][j][1] + matrix[i][j][2]) for j in range(len(algo_names)) if i != j)
        row_vals = [algo_names[i] + (" *" if i == best_idx else "")]
        for j in range(len(algo_names)):
            if i == j:
                row_vals.append("—")
            else:
                w, l, d = matrix[i][j]
                row_vals.append(f"{w}W-{l}L-{d}D")
        row_vals.extend([total_wins[i], tot_g, f"{win_rates[i]:.1f}%"])

        for col_idx, val in enumerate(row_vals, 1):
            c = ws1.cell(row=i + 4, column=col_idx, value=val)
            c.font = bold_font if i == best_idx else body_font
            c.border = thin_border
            if i == best_idx:
                c.fill = best_fill
            c.alignment = Alignment(horizontal="left" if col_idx == 1 else "center", vertical="center")

    for col in ws1.columns:
        max_len = max(len(str(c.value or "")) for c in col)
        ws1.column_dimensions[col[0].column_letter].width = max(max_len + 2, 11)

    # ── Sheet 2: Validation vs DualBot ──
    ws2 = wb.create_sheet(title="Validation vs DualBot")
    ws2.views.sheetView[0].showGridLines = True
    ws2.cell(row=1, column=1, value="Validation against Best Bot (DualBot)").font = title_font

    h2 = ["Algorithm", "Family", "DualBot Win Rate", "Wins", "Losses", "Draws", "Ship Margin", "Avg Turns", "H2H Win Rate"]
    for col_idx, h in enumerate(h2, 1):
        c = ws2.cell(row=3, column=col_idx, value=h)
        c.font = header_font
        c.fill = dark_fill
        c.alignment = Alignment(horizontal="center" if col_idx > 2 else "left", vertical="center")

    sorted_res = sorted(bot_results, key=lambda x: x["winrate"], reverse=True)
    for row_idx, r in enumerate(sorted_res, 4):
        vals = [
            r["name"], r["family"], f"{r['winrate']:.1f}%", r["wins"], r["losses"], r["draws"],
            round(r["avg_margin"], 1), round(r["avg_turns"], 1), f"{r['h2h_winrate']:.1f}%"
        ]
        for col_idx, val in enumerate(vals, 1):
            c = ws2.cell(row=row_idx, column=col_idx, value=val)
            c.font = body_font
            c.border = thin_border
            c.alignment = Alignment(horizontal="left" if col_idx <= 2 else "center", vertical="center")

    for col in ws2.columns:
        max_len = max(len(str(c.value or "")) for c in col)
        ws2.column_dimensions[col[0].column_letter].width = max(max_len + 3, 12)

    # ── Sheet 3: Evolved Strategy Parameters ──
    ws3 = wb.create_sheet(title="Evolved Strategy Vectors")
    ws3.views.sheetView[0].showGridLines = True
    ws3.cell(row=1, column=1, value="Evolved 14-D Strategy Parameters").font = title_font

    h3 = ["Algorithm", "Family"] + WEIGHT_NAMES
    for col_idx, h in enumerate(h3, 1):
        c = ws3.cell(row=3, column=col_idx, value=h)
        c.font = header_font
        c.fill = dark_fill
        c.alignment = Alignment(horizontal="center" if col_idx > 2 else "left", vertical="center")

    for row_idx, r in enumerate(bot_results, 4):
        vals = [r["name"], r["family"]] + [round(w, 3) for w in r["weights"]]
        for col_idx, val in enumerate(vals, 1):
            c = ws3.cell(row=row_idx, column=col_idx, value=val)
            c.font = body_font
            c.border = thin_border
            c.alignment = Alignment(horizontal="left" if col_idx <= 2 else "center", vertical="center")

    for col in ws3.columns:
        max_len = max(len(str(c.value or "")) for c in col)
        ws3.column_dimensions[col[0].column_letter].width = max(max_len + 2, 11)

    wb.save(path)


def plot_convergence_com(history, active_keys, path):
    plt.figure(figsize=(11, 6))
    for k in active_keys:
        curve = history.get(k, [])
        if curve:
            gens = list(range(1, len(curve) + 1))
            name = ALL_VARIANTS[k]["name"]
            lw = 2.5 if "Upgraded" in name or "ec_sa" in k else 1.5
            ls = "-" if not ALL_VARIANTS[k].get("is_baseline") else "--"
            plt.plot(gens, curve, label=name, linewidth=lw, linestyle=ls)

    plt.title("Competitive Co-Evolution: Cross-Population Fitness Progression", fontsize=13, fontweight="bold")
    plt.xlabel("Generation", fontsize=11)
    plt.ylabel("Competitive Fitness vs Rival Populations", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
