#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEM7_ER_PlanetWars.py
=====================
Evolutionary Algorithm Benchmarking Framework for Planet Wars (Google AI Challenge 2010).
Evaluates ALL 14 Algorithm Variants across 4 Families against the Best Bot (DualBot).

THE FOURTEEN ALGORITHMS (3 Pure TD-CCEA + 4 SA-TDCCEA + 3 CCE-TDCCEA + 4 Baselines):

  ── Family 1: Pure Monolithic TD-CCEA (Trajectory Focus) ──
  1. TD-CCEA               (Key: 'td')
     - Classical baseline with population centroid environment (14-D).
     - Standard linear similarity: alpha = max(0, 1 - d / sqrt(D)).
     - Uniform parameter mutation rate: sigma_i = 0.15*(1-e) + 0.03.

  2. TD-CCEA+SharpKernel   (Key: 'td_sharp')
     - Centroid environment with sharpened cutoff distance threshold (DC_MAX = 0.40).
     - Isolates the mathematical kernel cutoff effect on the centroid representation.

  3. TD-CCEA-Upgraded      (Key: 'td_up')
     - Multi-modal strategic modes via divisive 2-means clustering: F_t = {(theta_m, w_m, g_m)}.
     - Hungarian Optimal Transport (Earth Mover's Distance) D_env with D_MAX = 0.40.

  ── Family 2: Sensitivity-Adaptive SA-TDCCEA (Sensitivity Focus) ──
  4. SA-TDCCEA             (Key: 'sa')
     - Static sensitivity-adaptive baseline (pwm from sensitivity_scores_planetwars.json).
     - Centroid environment (14-D) with standard linear similarity.
     - Per-parameter mutation: sigma_j = pwm_j*(1-e) + 0.02.

  5. SA-TDCCEA+SharpKernel (Key: 'sa_sharp')
     - Static sensitivity (pwm) + sharpened centroid cutoff (DC_MAX = 0.40).
     - Isolates the kernel cutoff effect on sensitivity-adaptive CCEA.

  6. EC-SA-TDCCEA          (Key: 'ec_sa')
     - Environment-Conditioned Sensitivity-Adaptive TD-CCEA.
     - 38-D Fingerprint E_t = [R_pop(28D), R_opp(5D), R_game(2D), R_fit(3D)].
     - Dynamic online ablation triggered by drift reservoir + cluster memory M_j.
     - Correlated covariance mutation via Cholesky factor L of interaction matrix I_ij.

  7. EC-SA-TDCCEA-Upgraded (Key: 'ec_sa_up')
     - Full Proposed Architecture fusing Dynamic Sensitivity with Upgraded Trajectory.
     - Divisive 2-Means Multi-Modal Strategic Modes F_t = {(theta_m, w_m, g_m)}.
     - Hungarian Optimal Transport Earth Mover's Distance D_env (D_MAX = 0.40).
     - Dynamic sensitivity clusters M_j and Cholesky covariance mutation L.

  ── Family 3: Cooperative Co-Evolution CCE-TDCCEA (Decomposition Focus) ──
  8. CCE-TDCCEA            (Key: 'cce')
     - Cooperative Co-Evolution decomposing 14 weights into Macro (6) and Micro (8).
     - Top-3 partner cooperative evaluation against DualBot.
     - Ecosystem centroid (ca + cb) with linear Euclidean trajectory similarity.

  9. CCE-TDCCEA+SharpKernel(Key: 'cce_sharp')
     - Cooperative Co-Evolution (Macro 6 + Micro 8).
     - Ecosystem centroid with sharpened cutoff kernel (DC_MAX = 0.40).
     - Isolates the kernel cutoff effect on decomposed cooperative co-evolution.

  10. CCE-TDCCEA-Upgraded  (Key: 'cce_up')
     - Cooperative Co-Evolution upgraded with the new trajectory mechanism.
     - Independent divisive 2-means clustering per sub-population (Fa for Macro, Fb for Micro).
     - Hungarian Optimal Transport Earth Mover's Distance D_env per sub-population (D_MAX = 0.40).

  ── Family 4: Classical Optimization & Evolutionary Baselines ──
  11. Standard CCEA        (Key: 'std_ccea')
     - Cooperative Co-Evolution (Macro 6 + Micro 8) without trajectory tracking.
     - Tournament selection (k=3) + Gaussian mutation (sigma=0.15).
  12. CMA-ES               (Key: 'cmaes')
     - Covariance Matrix Adaptation Evolution Strategy (Hansen et al.).
     - Correlated covariance distribution updates for full 14-D strategy vectors.
  13. PSO                  (Key: 'pso')
     - Particle Swarm Optimization (W=0.7, C1=1.5, C2=1.5).
     - Swarm velocity and personal / global best position updates.
  14. Hill Climbing        (Key: 'hill')
     - Stochastic Next-Ascent Hill Climbing with Gaussian neighbourhood sampling.

STRATEGY VECTOR REPRESENTATION (14 Parameters in [0, 1]):
  Macro group (w0 to w5): economy and positioning
    w0: expand_share     - Share of spare ships spent on capturing neutral planets
    w1: max_expansions   - How many neutral planets it targets at once (1 to 6)
    w2: expand_threshold - How picky it is about neutral planets (payback filter)
    w3: economy_pressure - Bonus for attacking enemy planets with high growth rate
    w4: reserve_ratio    - Share of ships each planet keeps at home (up to 50%)
    w5: consolidate      - Moves idle ships from rear planets to the frontier
  Micro group (w6 to w13): combat and tactics
    w6: aggression       - How eager it is to launch an attack (threshold vs winnable)
    w7: attack_threshold - How much bigger fleet must be than defenders (1.0x to 3.0x)
    w8: target_priority  - What kind of enemy planet it picks (easiest vs strongest)
    w9: group_attack     - Synchronized multi-planet departures (up to 5 turns sync)
    w10: focus_fire      - How many targets it attacks at once (1 to 3 targets)
    w11: defend_radius   - How far from frontier it protects its own planets
    w12: defense_trigger - How easily it reinforces threatened planets
    w13: counter_rush    - Strikes back at enemy planets that just emptied themselves
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

# Ensure project root is in sys.path
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Import simulator and strategy bot
from planetwars.sim import GameState, Planet, Fleet, parse_map_file
from planetwars.opponents import dual_bot, bully_bot, rage_bot, prospector_bot, make_random_bot
from planetwars.bot_strategy import PlanetWarsStrategyBot, make_strategy_bot


# =====================================================================
#  STRATEGY PARAMETERS & CONFIGURATION
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

# Sub-population indices for Cooperative Co-evolution (CCE)
MACRO_INDICES = [0, 1, 2, 3, 4, 5]               # 6 weights (Economy & Positioning)
MICRO_INDICES = [6, 7, 8, 9, 10, 11, 12, 13]    # 8 weights (Combat & Tactics)

OPPONENT_NAMES = ["DualBot", "ProspectorBot", "RageBot", "BullyBot", "RandomBot"]
TARGET_OPPONENT = "DualBot"  # The Best Bot against which all algorithms are evaluated

# Default Evolutionary Hyperparameters
N_POP             = 20
TOTAL_GENERATIONS = 15
K_GAMES           = 2       # Games per individual per generation during evolution
VALIDATION_MAPS   = 20      # Test maps for final benchmark (played as both P1 and P2 = 40 matches)
MAX_TURNS         = 200     # Official game turn limit

A_MAX             = 15
DELTA_S           = 0.85
DELTA_F           = 0.20
TRAJ_WINDOW       = 7
MU_BASE           = 0.15
EPS_DYNAMIC       = 0.03
EPS_STATIC        = 0.02
DRAW_SCORE        = 0.30
TOP_PARTNERS      = 3
SENSITIVITY_FILE  = "sensitivity_scores_planetwars.json"

# Upgraded Trajectory Hyperparameters
TAU0              = 0.15     # Base divisive 2-means split threshold
LAM               = 0.50     # Stopping threshold exponential scaling
D_MAX             = 0.40     # Upgraded Optimal Transport distance cutoff
DC_MAX_SHARP      = 0.40     # Centroid sharpened kernel cutoff threshold

# EC-SA Hyperparameters (Environment-Conditioned Sensitivity & Interaction)
TAU_E             = 0.45     # Env distance threshold triggering ablation
TAU_I             = 0.80     # Cluster similarity match threshold
B_MAX             = 5        # Max generations between ablations
TAU_R             = 1.00     # Accumulated drift reservoir threshold
RHO               = 0.80     # Reservoir decay rate
ETA               = 0.30     # Cluster sensitivity EMA learning rate
I_EMA             = 0.30     # EMA rate for strategy interaction matrix
I_COUPLING        = 0.60     # Off-diagonal covariance coupling
RATE_LO           = 0.08     # Normalized mutation rate minimum
RATE_HI           = 0.20     # Normalized mutation rate maximum


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
#  ENVIRONMENT FINGERPRINT & SIMILARITY (For EC-SA)
# =====================================================================

# Fingerprint structure:
# [2*D (28D pop/elite)] + [5D opponent] + [2D game progress/div] + [3D fitness stats] = 38D
POP_SL  = slice(0, 2 * D)
OPP_SL  = slice(2 * D, 2 * D + len(OPPONENT_NAMES))
GAME_SL = slice(OPP_SL.stop, OPP_SL.stop + 2)
FIT_SL  = slice(GAME_SL.stop, GAME_SL.stop + 3)
BLOCK_W = (0.4, 0.3, 0.1, 0.2)

def build_fingerprint_38d(gen: int, pop: List, archive: List, fit_stats: Tuple[float, float, float]) -> List[float]:
    cen = centroid([a.params for a in pop])
    if archive:
        top = sorted(archive, key=lambda a: a.e_i, reverse=True)[:3]
        elite = centroid([a.params for a in top])
    else:
        elite = cen
    # One-hot opponent (DualBot is index 0)
    onehot = [1.0, 0.0, 0.0, 0.0, 0.0]
    div = mean(euclidean_dist(a.params, cen) for a in pop) / SQRT_D
    game = [gen / max(1, TOTAL_GENERATIONS), min(1.0, 2.0 * div)]
    return cen + elite + onehot + game + list(fit_stats)

def _esim(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) == 0:
        return 1.0
    return max(0.0, 1.0 - euclidean_dist(a, b) / math.sqrt(len(a)))

def env_sim_38d(e1: Sequence[float], e2: Sequence[float]) -> float:
    if len(e1) <= D or len(e2) <= D:
        return max(0.0, 1.0 - euclidean_dist(e1[:D], e2[:D]) / SQRT_D)
    s_pop  = _esim(e1[POP_SL], e2[POP_SL])
    s_opp  = max(0.0, 1.0 - 0.5 * sum(abs(p - q) for p, q in zip(e1[OPP_SL], e2[OPP_SL])))
    s_game = _esim(e1[GAME_SL], e2[GAME_SL])
    s_fit  = _esim(e1[FIT_SL], e2[FIT_SL])
    return sum(w * s for w, s in zip(BLOCK_W, (s_pop, s_opp, s_game, s_fit))) / sum(BLOCK_W)


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
    """2-means split of one cluster. Returns (dE, left_idx, right_idx) or None."""
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
    """Construct Multi-Modal Strategic Fingerprint F_t = {(theta_m, w_m, g_m)}."""
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
    """Build multi-modal strategic modes fingerprint for a CCE sub-population."""
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
    """Pure Python Hungarian algorithm O(n^3) fallback if scipy is unavailable."""
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
    """Optimal Transport / Earth Mover's Distance between two modal fingerprints."""
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

def ablation_update(pool, memory: List[List[Dict]], baseline: List[float], E, is_modes: bool = False):
    """Local ablation of each parameter j around baseline against DualBot."""
    vecs = []
    for j in range(D):
        for v in (0.0, 1.0):
            t = baseline[:]
            t[j] = v
            vecs.append(t)
    fits = pool.eval_batch_vs_dualbot(vecs, k_games=1)

    for j in range(D):
        S = abs(fits[2 * j + 1] - fits[2 * j])
        clusters = memory[j]
        best_k, best_s = None, -1.0
        for k, c in enumerate(clusters):
            if is_modes:
                s = max(0.0, 1.0 - d_env_modes(E, c["mu"]) / D_MAX)
            else:
                s = env_sim_38d(E, c["mu"])
            if s > best_s:
                best_k, best_s = k, s
        if best_k is not None and best_s >= TAU_I:
            c = clusters[best_k]
            n = c["n"]
            if not is_modes:
                c["mu"] = [(n * m + e) / (n + 1) for m, e in zip(c["mu"], E)]
            c["n"] = n + 1
            c["S"] = (1.0 - ETA) * c["S"] + ETA * S
        else:
            clusters.append({"mu": E, "S": S, "n": 1})

def retrieve_sensitivity(memory: List[List[Dict]], E, is_modes: bool = False) -> Optional[List[float]]:
    """Retrieve sensitivity conditioned on current environmental similarity."""
    out = [None] * D
    for j in range(D):
        num = den = 0.0
        for c in memory[j]:
            if is_modes:
                s = max(0.0, 1.0 - d_env_modes(E, c["mu"]) / D_MAX)
            else:
                s = env_sim_38d(E, c["mu"])
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
    """Convert sensitivity to relative mutation multipliers with mean 1.0."""
    mx = max(S) if max(S) > 0 else 1.0
    rate = [RATE_LO + (1.0 - s / mx) * (RATE_HI - RATE_LO) for s in S]
    m = mean(rate)
    return np.array([r / m for r in rate])

def interaction_observation(P: Sequence[Sequence[float]], f: Sequence[float]) -> np.ndarray:
    """Measure pairwise parameter interactions |corr(f, z_i * z_j)|."""
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
    """Compute Cholesky factor L of correlation matrix C = Id + g*I."""
    C = np.eye(D) + I_COUPLING * (I - np.diag(np.diag(I)))
    w, V = np.linalg.eigh((C + C.T) / 2)
    C = (V * np.clip(w, 1e-3, None)) @ V.T
    dg = np.sqrt(np.diag(C))
    return np.linalg.cholesky(C / np.outer(dg, dg) + 1e-9 * np.eye(D))


# =====================================================================
#  AGENT & SUB-AGENT CLASSES
# =====================================================================

class Agent:
    """Agent for monolithic 14-D algorithms (Pure TD-CCEA and SA-TDCCEA)."""
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
    """Sub-agent for Cooperative Co-Evolution (Macro 6-D and Micro 8-D)."""
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

    def snapshot(self):
        c = SubAgent(self.dim, self.params, env_offset=self.env_offset)
        c.e_i, c.H_i, c.mu_i = self.e_i, self.H_i[:], self.mu_i
        return c


def select_and_reproduce(pop: List[Agent], fits: List[float], pwm: Optional[List[float]] = None) -> List[Agent]:
    """Standard tournament selection (k=3) + mutation for monolithic agents."""
    new_pop = []
    k_tourn = 3
    for _ in range(len(pop)):
        cands = random.sample(range(len(pop)), k_tourn)
        winner = max(cands, key=lambda i: fits[i])
        parent = pop[winner]
        if pwm is not None:
            # Per-parameter sensitivity mutation
            child_params = [
                clamp01(p + random.gauss(0, pwm[j] * (1.0 - parent.e_i) + EPS_STATIC))
                for j, p in enumerate(parent.params)
            ]
        else:
            # Uniform mutation rate
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
    """Build similarity function alpha(Ft, Fk) for monolithic 14-D agents."""
    env_type = variant["env_type"]
    dc_max = variant.get("dc_max")
    d_max = variant.get("d_max", D_MAX)

    if env_type == "centroid":
        if dc_max is None:
            # Standard linear similarity: alpha = max(0, 1 - d / sqrt(D))
            return lambda Ft, Fk: max(0.0, 1.0 - euclidean_dist(Ft, Fk) / SQRT_D)
        else:
            # Sharpened kernel cutoff: alpha = max(0, 1 - d / (dc_max * sqrt(D)))
            return lambda Ft, Fk: max(0.0, 1.0 - euclidean_dist(Ft, Fk) / (dc_max * SQRT_D))

    elif env_type == "modes":
        # Upgraded Hungarian Optimal Transport / Earth Mover's Distance
        cache = {}
        def alpha(Ft, Fk):
            key = (Ft["uid"], Fk["uid"])
            if key not in cache:
                cache[key] = d_env_modes(Ft, Fk)
            return max(0.0, 1.0 - cache[key] / d_max)
        return alpha

    return lambda Ft, Fk: 1.0


def make_cce_subpop_alpha_fn(variant: Dict, dim: int) -> Callable:
    """Build similarity function for CCE sub-population."""
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
#  STATIC SENSITIVITY ANALYSIS (For SA Variants)
# =====================================================================

def load_or_run_static_sensitivity(pool) -> List[float]:
    """Load cached static sensitivity (pwm) or compute across DualBot baseline."""
    fp = SCRIPT_DIR / SENSITIVITY_FILE
    if fp.exists():
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            rates = data.get("normalized_mutation_rates")
            if rates and len(rates) == DIMENSIONS:
                print(f"  [SA] Loaded cached static sensitivity from {SENSITIVITY_FILE}")
                return rates
        except Exception:
            pass

    print("  [SA] Running static sensitivity analysis against DualBot...")
    baseline = [0.5] * D
    vecs = []
    for j in range(D):
        for v in (0.0, 1.0):
            t = baseline[:]
            t[j] = v
            vecs.append(t)

    fits = pool.eval_batch_vs_dualbot(vecs, k_games=2)
    influence = [0.0] * D
    for j in range(D):
        influence[j] = abs(fits[2 * j + 1] - fits[2 * j])

    max_inf = max(influence) if max(influence) > 0 else 1.0
    normalized = [0.08 + (1.0 - inf / max_inf) * 0.12 for inf in influence]

    try:
        with open(fp, "w", encoding="utf-8") as f:
            json.dump({
                "influence_scores": influence,
                "normalized_mutation_rates": normalized,
                "weight_names": WEIGHT_NAMES,
            }, f, indent=2)
        print(f"  [SA] Saved static sensitivity scores to {SENSITIVITY_FILE}")
    except Exception:
        pass
    return normalized


# =====================================================================
#  PARALLEL SIMULATOR GAME POOL
# =====================================================================

class PlanetWarsGamePool:
    """
    High-throughput game evaluation pool running pure-Python Planet Wars simulations.
    Pre-caches map GameStates and dispatches parallel games across worker threads.
    """

    def __init__(self, maps_dir: Path, n_workers: int = 8, is_mock: bool = False):
        self.maps_dir = maps_dir
        self.n_workers = n_workers
        self.is_mock = is_mock
        self.cached_maps: Dict[str, GameState] = {}
        self.training_map_keys: List[str] = []
        self.validation_map_keys: List[str] = []
        self._load_maps()

    def _load_maps(self):
        """Pre-parse maps from disk into cached GameStates for fast cloning."""
        if self.is_mock:
            self.training_map_keys = [f"map{i}" for i in range(1, 11)]
            self.validation_map_keys = [f"map{i}" for i in range(1, 21)]
            return

        map_files = sorted(
            self.maps_dir.glob("map*.txt"),
            key=lambda p: int(p.stem.replace("map", "")) if p.stem.replace("map", "").isdigit() else 999
        )
        if not map_files:
            raise FileNotFoundError(f"No map files found in {self.maps_dir}")

        for p in map_files:
            key = p.stem
            try:
                self.cached_maps[key] = parse_map_file(p)
            except Exception as e:
                print(f"  [WARN] Failed to parse {p}: {e}")

        all_keys = list(self.cached_maps.keys())
        # Training maps: first 10 maps
        self.training_map_keys = all_keys[:10]
        # Validation maps: first 20 maps
        self.validation_map_keys = all_keys[:20]
        print(f"  [POOL] Pre-cached {len(self.cached_maps)} maps. "
              f"Training: {len(self.training_map_keys)} maps, Validation: {len(self.validation_map_keys)} maps.")

    def run_one_game(
        self,
        p1_weights: Sequence[float],
        p2_weights_or_bot: any,
        map_key: str,
        max_turns: int = MAX_TURNS,
    ) -> Tuple[int, int, int, int]:
        """
        Runs one match between Player 1 and Player 2.
        Returns: (winner, turns, p1_ships, p2_ships)
        winner: 1 (P1 wins), 2 (P2 wins), 0 (Draw)
        """
        if self.is_mock:
            # Fast mock calculation
            s1 = sum(p1_weights) / len(p1_weights)
            s2 = 0.50 if callable(p2_weights_or_bot) else sum(p2_weights_or_bot) / len(p2_weights_or_bot)
            prob = 0.50 + 0.40 * math.tanh((s1 - s2) * 2.0)
            r = random.random()
            if r < 0.05:
                return 0, 200, 500, 500
            elif r < prob:
                return 1, 140, 1200, 0
            else:
                return 2, 140, 0, 1200

        st = self.cached_maps[map_key].clone()
        st.max_turns = max_turns

        # Initialize P1 bot
        bot1 = make_strategy_bot(p1_weights)

        # Initialize P2 bot
        if callable(p2_weights_or_bot):
            bot2 = p2_weights_or_bot
        else:
            bot2 = make_strategy_bot(p2_weights_or_bot)

        # Game loop
        for _ in range(max_turns):
            if st.game_over:
                break
            orders1 = bot1(st, 1)
            orders2 = bot2(st, 2)
            st.step(orders1, orders2)

        winner = st.winner if st.winner is not None else 0
        return winner, st.turn, st.num_ships(1), st.num_ships(2)

    def eval_batch_vs_dualbot(self, weight_list: Sequence[Sequence[float]], k_games: int = 2) -> List[float]:
        """
        Evaluate candidate weights against DualBot across training maps.
        Individuals play both as Player 1 and Player 2 to ensure fair, unbiased fitness.
        """
        n_maps = len(self.training_map_keys)

        def _eval_one(w_idx: int) -> float:
            w = weight_list[w_idx]
            total_score = 0.0
            total_games = 0

            for g in range(k_games):
                map_key = self.training_map_keys[(w_idx * k_games + g) % n_maps]

                # Match 1: Candidate is Player 1 vs DualBot as Player 2
                w1, _, s1, s2 = self.run_one_game(w, dual_bot, map_key)
                if w1 == 1:
                    score1 = 1.0
                elif w1 == 0:
                    score1 = DRAW_SCORE
                else:
                    score1 = 0.0
                # Fleet shaping tiebreaker
                margin1 = s1 - s2
                score1 += 0.05 * math.tanh(margin1 / 100.0)
                total_score += score1
                total_games += 1

                # Match 2: Candidate is Player 2 vs DualBot as Player 1
                w2, _, d_ships, my_ships = self.run_one_game([0.5]*14, w, map_key)
                # Note: Candidate is P2, so P2 win gives score
                if w2 == 2:
                    score2 = 1.0
                elif w2 == 0:
                    score2 = DRAW_SCORE
                else:
                    score2 = 0.0
                margin2 = my_ships - d_ships
                score2 += 0.05 * math.tanh(margin2 / 100.0)
                total_score += score2
                total_games += 1

            return max(0.0, total_score / max(1, total_games))

        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            return list(ex.map(_eval_one, range(len(weight_list))))

    def validate_against_dualbot(self, weights: Sequence[float]) -> Dict:
        """
        Rigorous benchmark validation of evolved champion weights against DualBot.
        Plays across all validation maps, playing both as P1 and P2 (40 matches total).
        """
        wins, losses, draws = 0, 0, 0
        total_turns = 0
        my_ships_list = []
        dual_ships_list = []
        per_map_results = {}

        for map_key in self.validation_map_keys:
            map_wins = 0
            map_losses = 0
            map_draws = 0

            # 1. As Player 1
            w1, t1, s1, s2 = self.run_one_game(weights, dual_bot, map_key)
            total_turns += t1
            my_ships_list.append(s1)
            dual_ships_list.append(s2)
            if w1 == 1:
                wins += 1
                map_wins += 1
            elif w1 == 2:
                losses += 1
                map_losses += 1
            else:
                draws += 1
                map_draws += 1

            # 2. As Player 2
            w2, t2, s_dual, s_cand = self.run_one_game([0.5]*14, weights, map_key)
            total_turns += t2
            my_ships_list.append(s_cand)
            dual_ships_list.append(s_dual)
            if w2 == 2:
                wins += 1
                map_wins += 1
            elif w2 == 1:
                losses += 1
                map_losses += 1
            else:
                draws += 1
                map_draws += 1

            per_map_results[map_key] = {
                "wins": map_wins,
                "losses": map_losses,
                "draws": map_draws,
                "winrate": (map_wins + 0.5 * map_draws) / 2.0 * 100.0,
            }

        total_matches = len(self.validation_map_keys) * 2
        overall_wr = ((wins + 0.5 * draws) / total_matches) * 100.0
        pure_wr = (wins / total_matches) * 100.0

        return {
            "total_matches": total_matches,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "winrate": overall_wr,
            "pure_winrate": pure_wr,
            "avg_turns": total_turns / total_matches,
            "avg_my_ships": mean(my_ships_list),
            "avg_dual_ships": mean(dual_ships_list),
            "ship_margin": mean(my_ships_list) - mean(dual_ships_list),
            "per_map": per_map_results,
        }


# =====================================================================
#  ALGORITHM REGISTRY (ALL 14 VARIANTS)
# =====================================================================

ALL_VARIANTS = {
    # ── Family 1: Pure Monolithic TD-CCEA ──
    "td": {
        "name": "TD-CCEA",
        "family": "Pure TD-CCEA",
        "env_type": "centroid",
        "dc_max": None,
        "d_max": None,
        "sens_type": "none",
        "is_cce": False,
        "is_baseline": False,
    },
    "td_sharp": {
        "name": "TD-CCEA+SharpKernel",
        "family": "Pure TD-CCEA",
        "env_type": "centroid",
        "dc_max": DC_MAX_SHARP,
        "d_max": None,
        "sens_type": "none",
        "is_cce": False,
        "is_baseline": False,
    },
    "td_up": {
        "name": "TD-CCEA-Upgraded",
        "family": "Pure TD-CCEA",
        "env_type": "modes",
        "dc_max": None,
        "d_max": D_MAX,
        "sens_type": "none",
        "is_cce": False,
        "is_baseline": False,
    },

    # ── Family 2: Sensitivity-Adaptive SA-TDCCEA ──
    "sa": {
        "name": "SA-TDCCEA",
        "family": "SA-TDCCEA",
        "env_type": "centroid",
        "dc_max": None,
        "d_max": None,
        "sens_type": "static",
        "is_cce": False,
        "is_baseline": False,
    },
    "sa_sharp": {
        "name": "SA-TDCCEA+SharpKernel",
        "family": "SA-TDCCEA",
        "env_type": "centroid",
        "dc_max": DC_MAX_SHARP,
        "d_max": None,
        "sens_type": "static",
        "is_cce": False,
        "is_baseline": False,
    },
    "ec_sa": {
        "name": "EC-SA-TDCCEA",
        "family": "SA-TDCCEA",
        "env_type": "centroid",
        "dc_max": None,
        "d_max": None,
        "sens_type": "dynamic",
        "is_cce": False,
        "is_baseline": False,
    },
    "ec_sa_up": {
        "name": "EC-SA-TDCCEA-Upgraded",
        "family": "SA-TDCCEA",
        "env_type": "modes",
        "dc_max": None,
        "d_max": D_MAX,
        "sens_type": "dynamic",
        "is_cce": False,
        "is_baseline": False,
    },

    # ── Family 3: Cooperative Co-Evolution CCE-TDCCEA ──
    "cce": {
        "name": "CCE-TDCCEA",
        "family": "CCE-TDCCEA",
        "env_type": "centroid",
        "dc_max": None,
        "d_max": None,
        "sens_type": "none",
        "is_cce": True,
        "is_baseline": False,
    },
    "cce_sharp": {
        "name": "CCE-TDCCEA+SharpKernel",
        "family": "CCE-TDCCEA",
        "env_type": "centroid",
        "dc_max": DC_MAX_SHARP,
        "d_max": None,
        "sens_type": "none",
        "is_cce": True,
        "is_baseline": False,
    },
    "cce_up": {
        "name": "CCE-TDCCEA-Upgraded",
        "family": "CCE-TDCCEA",
        "env_type": "modes",
        "dc_max": None,
        "d_max": D_MAX,
        "sens_type": "none",
        "is_cce": True,
        "is_baseline": False,
    },

    # ── Family 4: Classical Optimization & Evolutionary Baselines ──
    "std_ccea": {
        "name": "Standard CCEA",
        "family": "Baselines",
        "env_type": "none",
        "dc_max": None,
        "d_max": None,
        "sens_type": "none",
        "is_cce": False,
        "is_baseline": True,
        "base_type": "std_ccea",
    },
    "cmaes": {
        "name": "CMA-ES",
        "family": "Baselines",
        "env_type": "none",
        "dc_max": None,
        "d_max": None,
        "sens_type": "none",
        "is_cce": False,
        "is_baseline": True,
        "base_type": "cmaes",
    },
    "pso": {
        "name": "PSO",
        "family": "Baselines",
        "env_type": "none",
        "dc_max": None,
        "d_max": None,
        "sens_type": "none",
        "is_cce": False,
        "is_baseline": True,
        "base_type": "pso",
    },
    "hill": {
        "name": "Hill Climbing",
        "family": "Baselines",
        "env_type": "none",
        "dc_max": None,
        "d_max": None,
        "sens_type": "none",
        "is_cce": False,
        "is_baseline": True,
        "base_type": "hill",
    },
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
#  MAIN EXPERIMENTAL BENCHMARK ROUTINE
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Planet Wars Evolutionary Algorithm Benchmark against Best Bot (DualBot)"
    )
    parser.add_argument("--suite", choices=list(SUITE_CHOICES.keys()), default="all",
                        help="Algorithm suite to evaluate (default: 'all' = 14 algorithms)")
    parser.add_argument("--generations", type=int, default=TOTAL_GENERATIONS,
                        help="Number of evolutionary generations (default: 15)")
    parser.add_argument("--pop-size", type=int, default=N_POP,
                        help="Population size (default: 20)")
    parser.add_argument("--k-games", type=int, default=K_GAMES,
                        help="Evaluation games per individual per gen (default: 2)")
    parser.add_argument("--val-maps", type=int, default=VALIDATION_MAPS,
                        help="Number of maps for final benchmark validation (default: 20)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel worker threads (default: CPU count - 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--mock", action="store_true",
                        help="Fast dry-run mock mode without full simulation")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    active_keys = SUITE_CHOICES[args.suite]
    n_algos = len(active_keys)
    n_workers = args.workers or max(1, min(12, (os.cpu_count() or 4) - 1))

    maps_dir = SCRIPT_DIR / "engine" / "maps"
    if not maps_dir.exists():
        # Fallback to parent engine/maps
        maps_dir = SCRIPT_DIR.parent / "PlanetWars" / "engine" / "maps"

    print("\n" + "═" * 82)
    print(f"  SEM7_ER_PlanetWars — Benchmark of {n_algos} Algorithms vs DualBot (Best Bot)")
    print("═" * 82)
    print("  Active Algorithms in Suite:")
    for idx, k in enumerate(active_keys, 1):
        v = ALL_VARIANTS[k]
        print(f"    {idx:2d}. {v['name']:<26s} [{v['family']:<12s}] (env={v['env_type']}, sens={v['sens_type']})")
    print(f"  Target Opponent: {TARGET_OPPONENT} (Verified #1 Best Bot: 87.5% Ladder Win Rate)")
    print(f"  Evolution:       Pop={args.pop_size} | Gens={args.generations} | K_Games={args.k_games} | Workers={n_workers}")
    print(f"  Validation:      {args.val_maps} Maps × 2 Roles (P1 & P2) = {args.val_maps * 2} Matches per Champion")
    if args.mock:
        print("  MODE:            *** FAST MOCK TESTING MODE ***")
    print("═" * 82 + "\n")

    # Initialize Simulator Pool
    pool = PlanetWarsGamePool(maps_dir=maps_dir, n_workers=n_workers, is_mock=args.mock)
    if args.val_maps < len(pool.validation_map_keys):
        pool.validation_map_keys = pool.validation_map_keys[:args.val_maps]

    # Pre-calculate or load static sensitivity (pwm) for SA variants
    needs_pwm = any(ALL_VARIANTS[k]["sens_type"] == "static" for k in active_keys)
    pwm = load_or_run_static_sensitivity(pool) if needs_pwm else None

    # Similarity functions
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

    # Initialize Populations with Shared Seeds
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
                "arch_a": [],
                "arch_b": [],
            }
        elif v.get("is_baseline"):
            btype = v["base_type"]
            if btype == "std_ccea":
                baseline_state[k] = {
                    "pm": [p[:] for p in init_params_macro],
                    "pi": [p[:] for p in init_params_micro],
                    "fm": [0.0] * args.pop_size,
                    "fi": [0.0] * args.pop_size,
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
                    "W": 0.7,
                    "C1": 1.5,
                    "C2": 1.5,
                }
            elif btype == "hill":
                baseline_state[k] = {
                    "current": init_params[0][:],
                    "cfit": 0.0,
                }
        else:
            p_pwm = pwm if v["sens_type"] == "static" else None
            populations[k] = [Agent(p, pwm=p_pwm) for p in init_params]

    # EC-SA specific dynamic state trackers
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
                "stats": (0.0, 0.0, 0.0),
            }

    # =================================================================
    #  PHASE 1: SYNCHRONOUS EVOLUTION AGAINST DUALBOT
    # =================================================================
    print("=" * 82)
    print(f"  PHASE 1: EVOLUTION — {n_algos} Algorithms × {args.generations} Generations vs DualBot")
    print("=" * 82)

    t_evol_start = time.time()
    current_best_weights = {}

    # Initialize current bests
    for k in active_keys:
        if ALL_VARIANTS[k]["is_cce"]:
            current_best_weights[k] = combine_subagents(
                cce_state[k]["pop_a"][0], cce_state[k]["pop_b"][0]
            )
        elif ALL_VARIANTS[k].get("is_baseline"):
            btype = ALL_VARIANTS[k]["base_type"]
            if btype == "std_ccea":
                current_best_weights[k] = combine_subagents(
                    baseline_state[k]["pm"][0], baseline_state[k]["pi"][0]
                )
            elif btype == "cmaes":
                current_best_weights[k] = baseline_state[k]["best_p"][:]
            elif btype == "pso":
                current_best_weights[k] = baseline_state[k]["gb"][:]
            elif btype == "hill":
                current_best_weights[k] = baseline_state[k]["current"][:]
        else:
            current_best_weights[k] = populations[k][0].params[:]

    for gen in range(1, args.generations + 1):
        gen_start_t = time.time()
        print(f"\n  [Gen {gen:02d}/{args.generations:02d}] Evaluating populations vs DualBot...", end="", flush=True)

        # ── 1. Monolithic TD-CCEA & SA-TDCCEA ──
        for k in active_keys:
            if ALL_VARIANTS[k]["is_cce"] or ALL_VARIANTS[k].get("is_baseline"):
                continue

            pop = populations[k]
            fits = pool.eval_batch_vs_dualbot([a.params for a in pop], k_games=args.k_games)

            # Environmental Fingerprint
            if ALL_VARIANTS[k]["env_type"] == "modes":
                E = build_fingerprint_modes([a.params for a in pop], fits)
            else:
                E = centroid([a.params for a in pop])

            # Update Archives & Trajectories
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

            # Dynamic Sensitivity Adaptation (EC-SA Variants)
            if ALL_VARIANTS[k]["sens_type"] == "dynamic":
                ecs = ec_state[k]
                is_modes = (ALL_VARIANTS[k]["env_type"] == "modes")
                E_fingerprint = E if is_modes else build_fingerprint_38d(
                    gen, pop, archives[k], (mean(fits), std(fits), max(fits))
                )

                # Compute Environmental Drift
                if ecs["E_prev"] is not None:
                    if is_modes:
                        d_t = d_env_modes(E_fingerprint, ecs["E_prev"])
                    else:
                        d_t = 1.0 - env_sim_38d(E_fingerprint, ecs["E_prev"])
                    ecs["R"] = RHO * ecs["R"] + d_t
                ecs["E_prev"] = E_fingerprint

                # Check Drift-Triggered Ablation
                trigger = (
                    (ecs["E_last"] is None) or
                    (ecs["R"] >= TAU_R) or
                    ((gen - ecs["t_last"]) >= B_MAX)
                )

                if trigger:
                    ablation_update(pool, ecs["memory"], current_best_weights[k], E_fingerprint, is_modes=is_modes)
                    ecs["E_last"] = E_fingerprint
                    ecs["t_last"] = gen
                    ecs["R"] = 0.0

                # Parameter Interaction & Covariance
                obs = interaction_observation([a.params for a in pop], fits)
                ecs["I"] = (1.0 - I_EMA) * ecs["I"] + I_EMA * obs
                L = covariance_factor(ecs["I"])

                # Retrieve Sensitivity Scales
                S = retrieve_sensitivity(ecs["memory"], E_fingerprint, is_modes=is_modes)
                scales = sensitivity_to_scale(S) if S is not None else np.ones(D)

                # Correlated Covariance Reproduction
                new_pop = []
                for _ in range(len(pop)):
                    cands = random.sample(range(len(pop)), 3)
                    w_idx = max(cands, key=lambda idx: fits[idx])
                    p = pop[w_idx]
                    z = rng_std_norm = np.random.randn(D)
                    dz = L @ (z * scales) * p.mu_i
                    ch = Agent([clamp01(p.params[j] + dz[j]) for j in range(D)])
                    ch.H_i = p.H_i[:]
                    new_pop.append(ch)
                populations[k] = new_pop
            else:
                # Standard or Static Sensitivity Reproduction
                p_pwm = pwm if ALL_VARIANTS[k]["sens_type"] == "static" else None
                populations[k] = select_and_reproduce(pop, fits, pwm=p_pwm)

        # ── 2. Cooperative Co-Evolution (CCE-TDCCEA Variants) ──
        for k in active_keys:
            if not ALL_VARIANTS[k]["is_cce"]:
                continue

            cs = cce_state[k]
            pop_a, pop_b = cs["pop_a"], cs["pop_b"]

            # Sub-population A (Macro) top partners from B
            top_b = sorted(pop_b, key=lambda b: b.e_i, reverse=True)[:TOP_PARTNERS]
            vecs_a = []
            for a in pop_a:
                for b in top_b:
                    vecs_a.append(combine_subagents(a, b))
            fits_a_all = pool.eval_batch_vs_dualbot(vecs_a, k_games=1)
            fits_a = [mean(fits_a_all[i * TOP_PARTNERS: (i + 1) * TOP_PARTNERS]) for i in range(len(pop_a))]

            # Sub-population B (Micro) top partners from A
            top_a = sorted(pop_a, key=lambda a: a.e_i, reverse=True)[:TOP_PARTNERS]
            vecs_b = []
            for b in pop_b:
                for a in top_a:
                    vecs_b.append(combine_subagents(a, b))
            fits_b_all = pool.eval_batch_vs_dualbot(vecs_b, k_games=1)
            fits_b = [mean(fits_b_all[i * TOP_PARTNERS: (i + 1) * TOP_PARTNERS]) for i in range(len(pop_b))]

            # Modal or Centroid Sub-population Fingerprints
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

            # Update Sub-Agent Histories & Mutations
            for i, a in enumerate(pop_a):
                a.update_history(fits_a[i], E_a)
                a.compute_trajectory(E_a, cce_alpha_macro[k])
                a.compute_mutation()

            for i, b in enumerate(pop_b):
                b.update_history(fits_b[i], E_b)
                b.compute_trajectory(E_b, cce_alpha_micro[k])
                b.compute_mutation()

            # Tournament Selection & Mutation per Sub-population
            new_pop_a, new_pop_b = [], []
            for _ in range(len(pop_a)):
                cands_a = random.sample(range(len(pop_a)), 3)
                p_a = pop_a[max(cands_a, key=lambda i: fits_a[i])]
                new_pop_a.append(p_a.mutate())

                cands_b = random.sample(range(len(pop_b)), 3)
                p_b = pop_b[max(cands_b, key=lambda i: fits_b[i])]
                new_pop_b.append(p_b.mutate())

            cs["pop_a"] = new_pop_a
            cs["pop_b"] = new_pop_b

        # ── 3. Classical Baselines (Std CCEA, CMA-ES, PSO, Hill Climbing) ──
        for k in active_keys:
            if not ALL_VARIANTS[k].get("is_baseline"):
                continue

            btype = ALL_VARIANTS[k]["base_type"]
            bs = baseline_state[k]

            if btype == "std_ccea":
                pm, pi = bs["pm"], bs["pi"]
                pairs = [combine_subagents(pm[i], pi[i]) for i in range(len(pm))]
                fits = pool.eval_batch_vs_dualbot(pairs, k_games=args.k_games)
                best_i = int(np.argmax(fits))
                current_best_weights[k] = pairs[best_i][:]
                convergence_hist[k].append(float(fits[best_i]))

                new_pm, new_pi = [], []
                for _ in range(len(pm)):
                    c1 = random.sample(range(len(pm)), 3)
                    p_m = pm[max(c1, key=lambda x: fits[x])]
                    new_pm.append([clamp01(x + random.gauss(0, 0.15)) for x in p_m])

                    c2 = random.sample(range(len(pi)), 3)
                    p_i = pi[max(c2, key=lambda x: fits[x])]
                    new_pi.append([clamp01(x + random.gauss(0, 0.15)) for x in p_i])

                bs["pm"] = new_pm
                bs["pi"] = new_pi

            elif btype == "cmaes":
                if bs["has_cma"] and bs["es"] is not None:
                    pop_cma = bs["es"].ask()
                    pop_clipped = [[clamp01(x) for x in cand] for cand in pop_cma]
                    fits = pool.eval_batch_vs_dualbot(pop_clipped, k_games=args.k_games)
                    bs["es"].tell(pop_cma, [-f for f in fits])
                    best_i = int(np.argmax(fits))
                    if fits[best_i] > bs["best_f"]:
                        bs["best_f"] = fits[best_i]
                        bs["best_p"] = pop_clipped[best_i][:]
                    current_best_weights[k] = bs["best_p"][:]
                    convergence_hist[k].append(float(bs["best_f"]))
                else:
                    # Pure NumPy fallback
                    pop_cand = bs["pop"]
                    fits = pool.eval_batch_vs_dualbot(pop_cand, k_games=args.k_games)
                    best_i = int(np.argmax(fits))
                    if fits[best_i] > bs["best_f"]:
                        bs["best_f"] = fits[best_i]
                        bs["best_p"] = pop_cand[best_i][:]
                    current_best_weights[k] = bs["best_p"][:]
                    convergence_hist[k].append(float(bs["best_f"]))
                    # Adapt around best
                    sigma = max(0.05, bs["sigma"] * 0.95)
                    bs["sigma"] = sigma
                    bs["pop"] = [
                        [clamp01(x + random.gauss(0, sigma)) for x in bs["best_p"]]
                        for _ in range(len(pop_cand))
                    ]

            elif btype == "pso":
                pos, vel = bs["pos"], bs["vel"]
                fits = pool.eval_batch_vs_dualbot(pos, k_games=args.k_games)
                for i in range(len(pos)):
                    if fits[i] > bs["pf"][i]:
                        bs["pf"][i] = fits[i]
                        bs["pb"][i] = pos[i][:]
                    if fits[i] > bs["gf"]:
                        bs["gf"] = fits[i]
                        bs["gb"] = pos[i][:]
                current_best_weights[k] = bs["gb"][:]
                convergence_hist[k].append(float(bs["gf"]))

                # Velocity and Position Updates
                W, C1, C2 = bs["W"], bs["C1"], bs["C2"]
                for i in range(len(pos)):
                    for d in range(DIMENSIONS):
                        r1, r2 = random.random(), random.random()
                        vel[i][d] = (W * vel[i][d] +
                                     C1 * r1 * (bs["pb"][i][d] - pos[i][d]) +
                                     C2 * r2 * (bs["gb"][d] - pos[i][d]))
                        vel[i][d] = max(-0.2, min(0.2, vel[i][d]))
                        pos[i][d] = clamp01(pos[i][d] + vel[i][d])

            elif btype == "hill":
                curr = bs["current"]
                cands = [curr[:]]
                for _ in range(args.pop_size - 1):
                    cands.append([clamp01(x + random.gauss(0, 0.12)) for x in curr])
                fits = pool.eval_batch_vs_dualbot(cands, k_games=args.k_games)
                best_i = int(np.argmax(fits))
                if fits[best_i] >= bs["cfit"]:
                    bs["cfit"] = fits[best_i]
                    bs["current"] = cands[best_i][:]
                current_best_weights[k] = bs["current"][:]
                convergence_hist[k].append(float(bs["cfit"]))

        dt = time.time() - gen_start_t
        top_k = max(active_keys, key=lambda k: convergence_hist[k][-1] if convergence_hist[k] else 0)
        print(f" Done ({dt:.2f}s) | Top: {ALL_VARIANTS[top_k]['name']} (fit={convergence_hist[top_k][-1]:.3f})")

    t_evol_total = time.time() - t_evol_start
    print(f"\n  [EVOLUTION FINISHED] Completed {args.generations} generations in {t_evol_total:.2f}s.")

    # =================================================================
    #  PHASE 2: RIGOROUS BENCHMARK VALIDATION AGAINST DUALBOT
    # =================================================================
    print("\n" + "=" * 82)
    print(f"  PHASE 2: BENCHMARK VALIDATION AGAINST DUALBOT ({len(pool.validation_map_keys)*2} Matches per Algo)")
    print("=" * 82)

    benchmark_results = []
    for idx, k in enumerate(active_keys, 1):
        v = ALL_VARIANTS[k]
        weights = current_best_weights[k]
        t0 = time.time()
        res = pool.validate_against_dualbot(weights)
        val_time = time.time() - t0

        res["key"] = k
        res["name"] = v["name"]
        res["family"] = v["family"]
        res["weights"] = weights
        res["val_time"] = val_time
        res["convergence"] = convergence_hist[k]
        benchmark_results.append(res)

        print(f"  [{idx:2d}/{n_algos:2d}] {v['name']:<26s} -> "
              f"WinRate: {res['winrate']:5.1f}% ({res['wins']:2d}W-{res['losses']:2d}L-{res['draws']:2d}D) "
              f"| Margin: {res['ship_margin']:+6.0f} ships | AvgTurns: {res['avg_turns']:5.1f}")

    # Sort results by Win Rate against DualBot (descending)
    benchmark_results.sort(key=lambda r: (r["winrate"], r["ship_margin"]), reverse=True)

    # =================================================================
    #  PHASE 3: SUMMARY BENCHMARK RANKING TABLE
    # =================================================================
    table_lines = []
    header_str = (
        f"{'Rank':<5} {'Algorithm':<26} {'Family':<14} "
        f"{'Win Rate':<10} {'Record (W-L-D)':<16} {'Ship Margin':<14} {'Avg Turns':<11}"
    )
    table_lines.append("=" * 100)
    table_lines.append(f"  PLANET WARS BENCHMARK: ALGORITHM WIN RATES AGAINST BEST BOT (DualBot)")
    table_lines.append(f"  Evaluated on {len(pool.validation_map_keys)} maps × 2 roles (P1 & P2) = {len(pool.validation_map_keys)*2} matches per algorithm")
    table_lines.append("=" * 100)
    table_lines.append(header_str)
    table_lines.append("-" * 100)

    for rank, r in enumerate(benchmark_results, 1):
        is_best = (rank == 1)
        name_str = f"{r['name']}*" if is_best else r['name']
        rec_str = f"{r['wins']}W-{r['losses']}L-{r['draws']}D"
        margin_str = f"{r['ship_margin']:+6.1f}"
        line = (
            f"{rank:<5} {name_str:<26} {r['family']:<14} "
            f"{r['winrate']:>6.1f}%    {rec_str:<16} {margin_str:>10s}    {r['avg_turns']:>7.1f}   "
        )
        table_lines.append(line)

    table_lines.append("-" * 100)
    table_lines.append("  * Top performing algorithm against DualBot.")
    table_lines.append("=" * 100)

    table_output = "\n".join(table_lines)
    print("\n" + table_output + "\n")

    # =================================================================
    #  PHASE 4: SAVE ARTIFACTS
    # =================================================================
    # 1. Text Summary Table
    txt_path = SCRIPT_DIR / "SEM7_ER_PlanetWars_table.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table_output + "\n\n")
        f.write("=" * 100 + "\n")
        f.write("  EVOLVED STRATEGY WEIGHTS PER ALGORITHM (14 Parameters):\n")
        f.write("  [w0:expand_share, w1:max_exp, w2:expand_thresh, w3:econ_press, w4:reserve, w5:consolidate,\n")
        f.write("   w6:aggression, w7:atk_thresh, w8:target_prio, w9:group_atk, w10:focus_fire, w11:def_rad,\n")
        f.write("   w12:def_trigger, w13:counter_rush]\n")
        f.write("=" * 100 + "\n")
        for r in benchmark_results:
            w_str = ", ".join(f"{w:.2f}" for w in r["weights"])
            f.write(f"{r['name']:<26s}: [{w_str}]\n")
    print(f"  Saved Table Text -> {txt_path}")

    # 2. JSON Detailed Artifact
    json_path = SCRIPT_DIR / "SEM7_ER_PlanetWars_results.json"
    json_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "opponent": TARGET_OPPONENT,
        "config": {
            "generations": args.generations,
            "pop_size": args.pop_size,
            "k_games": args.k_games,
            "val_maps": args.val_maps,
            "total_val_matches": len(pool.validation_map_keys) * 2,
            "max_turns": MAX_TURNS,
            "tau0": TAU0,
            "d_max": D_MAX,
            "dc_max": DC_MAX_SHARP,
        },
        "ranking": [
            {
                "rank": rank,
                "name": r["name"],
                "key": r["key"],
                "family": r["family"],
                "winrate": r["winrate"],
                "pure_winrate": r["pure_winrate"],
                "wins": r["wins"],
                "losses": r["losses"],
                "draws": r["draws"],
                "ship_margin": r["ship_margin"],
                "avg_turns": r["avg_turns"],
                "avg_my_ships": r["avg_my_ships"],
                "avg_dual_ships": r["avg_dual_ships"],
                "weights": [round(w, 4) for w in r["weights"]],
                "convergence": r["convergence"],
                "per_map": r["per_map"],
            }
            for rank, r in enumerate(benchmark_results, 1)
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved JSON Report -> {json_path}")

    # 3. Formatted Excel Artifact (if openpyxl installed)
    if _OPENPYXL_AVAILABLE:
        xlsx_path = SCRIPT_DIR / "SEM7_ER_PlanetWars_comparison.xlsx"
        export_excel_report(benchmark_results, pool.validation_map_keys, xlsx_path)
        print(f"  Saved Excel Workbook -> {xlsx_path}")

    # 4. Convergence Plot (if matplotlib installed)
    if _MATPLOTLIB_AVAILABLE:
        png_path = SCRIPT_DIR / "SEM7_ER_PlanetWars_convergence.png"
        plot_convergence(convergence_hist, active_keys, png_path)
        print(f"  Saved Convergence Plot -> {png_path}")

    print("\n" + "═" * 82)
    print("  PLANET WARS BENCHMARK COMPLETED SUCCESSFULLY!")
    print("═" * 82 + "\n")


# =====================================================================
#  EXCEL EXPORT (4 FORMATTED SHEETS)
# =====================================================================

def export_excel_report(results: List[Dict], val_maps: List[str], path: Path):
    """Export formatted Excel workbook with Overall Ranking, Per-Map matrix, and Strategy Weights."""
    wb = openpyxl.Workbook()

    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    bold_font = Font(name="Calibri", size=11, bold=True)
    body_font = Font(name="Calibri", size=11)
    title_font = Font(name="Calibri", size=14, bold=True, color="1F497D")

    dark_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    gold_fill = PatternFill(start_color="DCE6F1", end_color="DCE6F1", fill_type="solid")
    best_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")

    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    # ── Sheet 1: Ranking Summary vs DualBot ──
    ws1 = wb.active
    ws1.title = "Ranking vs DualBot"
    ws1.views.sheetView[0].showGridLines = True

    ws1.cell(row=1, column=1, value="Planet Wars EA Benchmark — Ranking against DualBot (Best Bot)").font = title_font
    headers1 = [
        "Rank", "Algorithm", "Family", "Win Rate (%)", "Wins", "Losses", "Draws",
        "Total Matches", "Ship Margin", "Avg My Ships", "Avg Dual Ships", "Avg Game Turns"
    ]
    for col_idx, h in enumerate(headers1, 1):
        cell = ws1.cell(row=3, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = dark_fill
        cell.alignment = Alignment(horizontal="center" if col_idx != 2 else "left", vertical="center")

    for row_idx, r in enumerate(results, 4):
        is_best = (row_idx == 4)
        row_data = [
            row_idx - 3,
            r["name"] + (" *" if is_best else ""),
            r["family"],
            f"{r['winrate']:.1f}%",
            r["wins"],
            r["losses"],
            r["draws"],
            r["total_matches"],
            round(r["ship_margin"], 1),
            round(r["avg_my_ships"], 1),
            round(r["avg_dual_ships"], 1),
            round(r["avg_turns"], 1),
        ]
        for col_idx, val in enumerate(row_data, 1):
            cell = ws1.cell(row=row_idx, column=col_idx, value=val)
            cell.font = bold_font if is_best else body_font
            cell.border = thin_border
            if is_best:
                cell.fill = best_fill
            cell.alignment = Alignment(horizontal="left" if col_idx == 2 else "center", vertical="center")

    for col in ws1.columns:
        max_len = max(len(str(c.value or "")) for c in col)
        ws1.column_dimensions[col[0].column_letter].width = max(max_len + 3, 12)

    # ── Sheet 2: Per-Map Win Rate Breakdown ──
    ws2 = wb.create_sheet(title="Per-Map Breakdown")
    ws2.views.sheetView[0].showGridLines = True
    ws2.cell(row=1, column=1, value="Per-Map Performance Matrix vs DualBot (2 Matches per Map: P1 & P2)").font = title_font

    map_headers = ["Algorithm", "Overall WR"] + val_maps
    for col_idx, h in enumerate(map_headers, 1):
        cell = ws2.cell(row=3, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = dark_fill
        cell.alignment = Alignment(horizontal="center" if col_idx != 1 else "left", vertical="center")

    for row_idx, r in enumerate(results, 4):
        ws2.cell(row=row_idx, column=1, value=r["name"]).font = body_font
        ws2.cell(row=row_idx, column=2, value=f"{r['winrate']:.1f}%").font = bold_font
        for col_idx, mk in enumerate(val_maps, 3):
            m_res = r["per_map"].get(mk, {"wins": 0, "losses": 0, "draws": 0, "winrate": 0.0})
            cell = ws2.cell(row=row_idx, column=col_idx, value=f"{m_res['winrate']:.0f}%")
            cell.font = body_font
            cell.border = thin_border
            cell.alignment = Alignment(horizontal="center", vertical="center")
            if m_res["winrate"] == 100.0:
                cell.fill = best_fill

    for col in ws2.columns:
        max_len = max(len(str(c.value or "")) for c in col)
        ws2.column_dimensions[col[0].column_letter].width = max(max_len + 2, 10)

    # ── Sheet 3: Evolved Strategy Vectors (14 Parameters) ──
    ws3 = wb.create_sheet(title="Evolved Strategy Parameters")
    ws3.views.sheetView[0].showGridLines = True
    ws3.cell(row=1, column=1, value="Evolved 14-D Strategy Vectors (Macro w0-w5, Micro w6-w13)").font = title_font

    strat_headers = ["Algorithm", "Family", "Win Rate"] + WEIGHT_NAMES
    for col_idx, h in enumerate(strat_headers, 1):
        cell = ws3.cell(row=3, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = dark_fill
        cell.alignment = Alignment(horizontal="center" if col_idx > 2 else "left", vertical="center")

    for row_idx, r in enumerate(results, 4):
        ws3.cell(row=row_idx, column=1, value=r["name"]).font = body_font
        ws3.cell(row=row_idx, column=2, value=r["family"]).font = body_font
        ws3.cell(row=row_idx, column=3, value=f"{r['winrate']:.1f}%").font = bold_font
        for col_idx, w in enumerate(r["weights"], 4):
            cell = ws3.cell(row=row_idx, column=col_idx, value=round(w, 3))
            cell.font = body_font
            cell.border = thin_border
            cell.alignment = Alignment(horizontal="center", vertical="center")

    for col in ws3.columns:
        max_len = max(len(str(c.value or "")) for c in col)
        ws3.column_dimensions[col[0].column_letter].width = max(max_len + 2, 11)

    wb.save(path)


# =====================================================================
#  CONVERGENCE PLOT GENERATION
# =====================================================================

def plot_convergence(history: Dict[str, List[float]], active_keys: List[str], path: Path):
    """Plot evolutionary convergence curves across generations."""
    plt.figure(figsize=(11, 6))

    colors = {
        "td": "#1f77b4", "td_sharp": "#aec7e8", "td_up": "#08519c",
        "sa": "#ff7f0e", "sa_sharp": "#ffbb78", "ec_sa": "#d95f02", "ec_sa_up": "#a63603",
        "cce": "#2ca02c", "cce_sharp": "#98df8a", "cce_up": "#006d2c",
        "std_ccea": "#7f7f7f", "cmaes": "#9467bd", "pso": "#8c564b", "hill": "#e377c2"
    }

    for k in active_keys:
        curve = history.get(k, [])
        if curve:
            gens = list(range(1, len(curve) + 1))
            name = ALL_VARIANTS[k]["name"]
            c = colors.get(k, "#333333")
            lw = 2.5 if "up" in k or "ec_sa" in k else 1.5
            ls = "-" if not ALL_VARIANTS[k].get("is_baseline") else "--"
            plt.plot(gens, curve, label=name, color=c, linewidth=lw, linestyle=ls)

    plt.title("Planet Wars: Evolutionary Convergence vs DualBot (Best Bot)", fontsize=13, fontweight="bold")
    plt.xlabel("Generation", fontsize=11)
    plt.ylabel("Training Fitness (Win / Score Margin vs DualBot)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()

