#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEM7_ER.py
==========
Dynamic Environment Benchmark of ALL 10 Evolutionary Algorithm Variants across 3 Families
(Pure TD-CCEA, Sensitivity-Adaptive SA-TDCCEA, and Cooperative Co-Evolution CCE-TDCCEA)
on a 48×48 Map with Progressive Blockades.

THE TEN ALGORITHMS (3 Pure TD-CCEA + 4 SA-TDCCEA + 3 CCE-TDCCEA):

  ── Family 1: Pure Monolithic TD-CCEA (Trajectory Focus) ──
  1. TD-CCEA               (Key: 'td')
     - Classical baseline with population centroid environment (20-D).
     - Standard linear similarity: alpha = max(0, 1 - d / sqrt(D)).
     - Uniform parameter mutation rate: sigma_i = 0.15*(1-e) + 0.03.

  2. TD-CCEA+SharpKernel   (Key: 'td_sharp')
     - Centroid environment with sharpened cutoff distance threshold (DC_MAX = 0.40).
     - Isolates the mathematical kernel cutoff effect on the centroid representation.

  3. TD-CCEA-Upgraded      (Key: 'td_up')
     - Multi-modal strategic modes via divisive 2-means clustering: F_t = {(theta_m, w_m, g_m)}.
     - Optimal Transport (Earth Mover's Distance) D_env with D_MAX = 0.40.

  ── Family 2: Sensitivity-Adaptive SA-TDCCEA (Sensitivity Focus) ──
  4. SA-TDCCEA             (Key: 'sa')
     - Static sensitivity-adaptive baseline (pwm from sensitivity_scores.json).
     - Centroid environment (20-D) with standard linear similarity.
     - Per-parameter mutation: sigma_j = pwm_j*(1-e) + 0.02.

  5. SA-TDCCEA+SharpKernel (Key: 'sa_sharp')
     - Static sensitivity (pwm) + sharpened centroid cutoff (DC_MAX = 0.40).
     - Isolates the kernel cutoff effect on sensitivity-adaptive CCEA.

  6. EC-SA-TDCCEA          (Key: 'ec_sa')
     - Environment-Conditioned Sensitivity-Adaptive TD-CCEA.
     - 73-D Fingerprint E_t = [R_pop(40D), R_opp(7D), R_game(2D), R_fit(3D)].
     - Dynamic online ablation triggered by drift + cluster memory M_j.
     - Correlated covariance mutation via Cholesky factor L of interaction matrix I_ij.

  7. EC-SA-TDCCEA-Upgraded (Key: 'ec_sa_up')
     - Full Proposed Architecture fusing Dynamic Sensitivity with Upgraded Trajectory.
     - Divisive 2-Means Multi-Modal Strategic Modes F_t = {(theta_m, w_m, g_m)}.
     - Hungarian Optimal Transport Earth Mover's Distance D_env (D_MAX = 0.40).
     - Dynamic sensitivity clusters M_j and Cholesky covariance mutation L.

  ── Family 3: Cooperative Co-Evolution CCE-TDCCEA (Decomposition Focus) ──
  8. CCE-TDCCEA            (Key: 'cce')
     - Cooperative Co-Evolution decomposing 20 weights into Macro (8) and Micro (12).
     - Top-3 partner cooperative evaluation on the dynamic map.
     - Ecosystem centroid (ca + cb) with linear Euclidean trajectory similarity.

  9. CCE-TDCCEA+SharpKernel(Key: 'cce_sharp')
     - Cooperative Co-Evolution (Macro 8 + Micro 12).
     - Ecosystem centroid with sharpened cutoff kernel (DC_MAX = 0.40).
     - Isolates the kernel cutoff effect on decomposed cooperative co-evolution.

  10. CCE-TDCCEA-Upgraded  (Key: 'cce_up')
     - Cooperative Co-Evolution upgraded with the new trajectory mechanism.
     - Independent divisive 2-means clustering per sub-population (Fa for Macro, Fb for Micro).
     - Hungarian Optimal Transport Earth Mover's Distance D_env per sub-population (D_MAX = 0.40).

  ── Family 4: Classical Optimization & Evolutionary Baselines (COMPARE_ALL) ──
  11. Standard CCEA        (Key: 'std_ccea')
     - Cooperative Co-Evolution (Macro 8 + Micro 12) without trajectory tracking.
     - Tournament selection (k=3) + Gaussian mutation (sigma=0.15).
  12. CMA-ES               (Key: 'cmaes')
     - Covariance Matrix Adaptation Evolution Strategy (Hansen et al.).
     - Correlated covariance distribution updates for full 20-D weight vectors.
  13. PSO                  (Key: 'pso')
     - Particle Swarm Optimization (W=0.7, C1=1.5, C2=1.5).
     - Swarm velocity and personal / global best position updates.
  14. Hill Climbing        (Key: 'hill')
     - Stochastic Next-Ascent Hill Climbing with Gaussian neighbourhood sampling.

WORKFLOW:
  Phase 1 — DYNAMIC ENVIRONMENT EVOLUTION:
    - 48×48 map with progressive blockades added generation-by-generation.
    - Blockade addition schedule:
        Gens 1-7:   +6 walls/gen
        Gens 8-14:  +10 walls/gen
        Gens 15-30: +14 walls/gen
        40% breakable barriers, 60% stone terrain.
    - BFS reachability validation ensures Base 0 (2,2) and Base 1 (45,45) remain connected.
    - All algorithms evolve synchronously generation-by-generation on the dynamic map.

  Phase 2 — BOT VALIDATION & RANKING:
    - Final evolved champions validated against all 7 built-in microRTS opponent bots.
    - Overall ranking table and per-bot win rate matrix with asterisk (*) for best.

  Phase 3 — HEAD-TO-HEAD TOURNAMENT MATRIX:
    - Pairwise matches on the final blockaded 48×48 map (10 games per pair, alternating P0/P1).
    - Timeout protection: Undecided games hitting MAX_CYCLES (12000) are scored as draws.

  Phase 4 — STATISTICAL ARTIFACTS (Prefix: 'SEM7_ER_'):
    - SEM7_ER_wilcoxon.txt    -> Wilcoxon signed-rank tests vs top reference algorithm
    - SEM7_ER_ablation.txt    -> Multi-tier ablation breakdown + text H2H matrix
    - SEM7_ER_results.json    -> Structured JSON metadata, validation, and H2H tournament data
    - SEM7_ER_comparison.xlsx -> Formatted Excel workbook (if openpyxl installed)

USAGE:
  python SEM7_ER.py                                # Standard run: all 14 algorithms (30 gens)
  python SEM7_ER.py --mock                         # Fast dry-run without Java microRTS
  python SEM7_ER.py --eval rivals                  # Evolve against rival champions (COMPARE_ALL / PR1 style)
  python SEM7_ER.py --suite baseline               # Run only the 4 classical baselines (Std CCEA, CMA-ES, PSO, Hill Climbing)
  python SEM7_ER.py --suite cce                    # Run only the 3 Cooperative Co-Evolution variants
  python SEM7_ER.py --suite td                     # Run only the 3 Pure TD-CCEA variants
  python SEM7_ER.py --suite sa                     # Run only the 4 Sensitivity-Adaptive variants
  python SEM7_ER.py --generations 15 --workers 8   # Custom generations and CPU workers
"""

import argparse
import collections
import json
import math
import os
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

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
    from openpyxl.styles import Font, PatternFill, Alignment
    _OPENPYXL_AVAILABLE = True
except ImportError:
    _OPENPYXL_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION & PATH DISCOVERY
# ═══════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def find_microrts_dir():
    candidates = []
    if os.environ.get("MICRORTS_DIR"):
        candidates.append(os.environ["MICRORTS_DIR"])
    candidates.extend([
        os.path.join(SCRIPT_DIR, "Simulation", "microrts"),
        os.path.join(os.path.dirname(SCRIPT_DIR), "Simulation", "microrts"),
        os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "Simulation", "microrts"),
        os.path.abspath("Simulation/microrts"),
    ])
    for c in candidates:
        if os.path.isfile(os.path.join(c, "bin", "tests", "EvolvedBatchGameRunner.class")):
            return c
    for c in candidates:
        if os.path.isdir(os.path.join(c, "src")):
            return c
    return candidates[0]

MICRORTS_DIR = find_microrts_dir()

# 48×48 Map Configuration
MAP_W, MAP_H   = 48, 48
BASE_MAP_PATH  = os.path.join(MICRORTS_DIR, "maps", "open48x48.xml")
TEMP_MAP_PATH  = os.path.join(MICRORTS_DIR, "maps", "temp_pr1_48x48.xml")
TEMP_MAP_REL   = "maps/temp_pr1_48x48.xml"
MAX_CYCLES     = 12000

JAVA_CMD       = "java"

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

# Sub-population indices for Cooperative Co-evolution (CCE)
MACRO_INDICES = [0, 1, 2, 6, 14, 15, 17, 18]                  # 8 weights (Economy & Macro)
MICRO_INDICES = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 16, 19]   # 12 weights (Combat & Micro)

OPPONENT_ROTATION = [
    "EconomyMilitaryRush", "LightRush", "WorkerRush",
    "HeavyDefense", "RangedRush", "EconomyRush", "WorkerRushPlusPlus",
]
N_OPP = len(OPPONENT_ROTATION)

# Evolutionary Defaults
N_POP             = 20
K_GAMES           = 3
TOTAL_GENERATIONS = 30
VALIDATION_GAMES  = 5
H2H_GAMES         = 10

A_MAX             = 15
DELTA_S           = 0.85
DELTA_F           = 0.20
TRAJ_WINDOW       = 7
MU_BASE           = 0.15
EPS_DYNAMIC       = 0.03
EPS_STATIC        = 0.02
MIN_KEEP          = 10
DRAW_SCORE        = 0.3
TOP_PARTNERS      = 3
SENSITIVITY_FILE  = "sensitivity_scores.json"

# Upgraded Trajectory Hyperparameters (TDDCEA_Modification.pdf)
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

# Wall scheduling parameters
WALLS_PER_GEN_EARLY = 6
WALLS_PER_GEN_MID   = 10
WALLS_PER_GEN_LATE  = 14
BREAKABLE_RATIO     = 0.4

# Reserved tiles around player bases
RESERVED = set()
for bx, by in [(0,0),(0,1),(1,1),(2,2),(1,2),(2,1),(3,2),(2,3),(3,3)]:
    RESERVED.add((bx, by))
for bx, by in [(MAP_W-1,MAP_H-1),(MAP_W-1,MAP_H-2),(MAP_W-2,MAP_H-2),
               (MAP_W-3,MAP_H-3),(MAP_W-2,MAP_H-3),(MAP_W-3,MAP_H-2),
               (MAP_W-4,MAP_H-3),(MAP_W-3,MAP_H-4),(MAP_W-4,MAP_H-4)]:
    RESERVED.add((bx, by))


# ═══════════════════════════════════════════════════════════════
#  MATH & DISTANCE UTILITIES
# ═══════════════════════════════════════════════════════════════

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def euclidean_dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def centroid(vectors):
    n = len(vectors)
    return [sum(v[j] for v in vectors) / n for j in range(len(vectors[0]))]

def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0

def std(xs):
    xs = list(xs)
    if not xs:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

def combine_subagents(macro, micro):
    """Combine macro and micro sub-agent parameters into a full 20-D weight vector."""
    full = [0.5] * DIMENSIONS
    macro_vals = macro.params if hasattr(macro, "params") else macro
    micro_vals = micro.params if hasattr(micro, "params") else micro
    for i, idx in enumerate(MACRO_INDICES):
        full[idx] = clamp01(macro_vals[i])
    for i, idx in enumerate(MICRO_INDICES):
        full[idx] = clamp01(micro_vals[i])
    return full


# ═══════════════════════════════════════════════════════════════
#  DYNAMIC MAP GENERATION & BFS REACHABILITY VALIDATION
# ═══════════════════════════════════════════════════════════════

def bfs_reachable(walls, start, end):
    """Breadth-First Search checking reachability between bases avoiding walls."""
    if start in walls or end in walls:
        return False
    visited = {start}
    q = collections.deque([start])
    while q:
        x, y = q.popleft()
        if (x, y) == end:
            return True
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < MAP_W and 0 <= ny < MAP_H:
                if (nx, ny) not in visited and (nx, ny) not in walls:
                    visited.add((nx, ny))
                    q.append((nx, ny))
    return False


def pick_wall_position(terrain_walls, breakable_positions, gen):
    """Pick a random wall position around map center that maintains base connectivity."""
    base0 = (2, 2)
    base1 = (MAP_W - 3, MAP_H - 3)
    all_walls = terrain_walls | breakable_positions
    no_place = all_walls | RESERVED
    for _ in range(300):
        cx, cy = MAP_W // 2, MAP_H // 2
        spread = 10 + gen // 2
        x = int(random.gauss(cx, spread))
        y = int(random.gauss(cy, spread))
        x = max(0, min(MAP_W - 1, x))
        y = max(0, min(MAP_H - 1, y))
        if (x, y) in no_place:
            continue
        test_walls = all_walls | {(x, y)}
        if bfs_reachable(test_walls, base0, base1):
            return (x, y)
    return None


def add_walls_for_generation(gen, terrain_walls, breakable_positions):
    """Progressively add blockades for generation gen."""
    if gen <= 7:
        n_new = WALLS_PER_GEN_EARLY
    elif gen <= 14:
        n_new = WALLS_PER_GEN_MID
    else:
        n_new = WALLS_PER_GEN_LATE

    new_t, new_b = set(), set()
    for _ in range(n_new):
        pos = pick_wall_position(terrain_walls | new_t, breakable_positions | new_b, gen)
        if pos is None:
            break
        if random.random() < BREAKABLE_RATIO:
            new_b.add(pos)
        else:
            new_t.add(pos)

    terrain_walls = terrain_walls | new_t
    breakable_positions = breakable_positions | new_b
    return terrain_walls, breakable_positions, len(new_t), len(new_b)


def write_map_xml(terrain_walls, breakable_positions, path):
    """Write 48×48 physical game state XML with blockades encoded in terrain string."""
    all_blockades = terrain_walls | breakable_positions
    terrain = ['0'] * (MAP_W * MAP_H)
    for wx, wy in all_blockades:
        terrain[wy * MAP_W + wx] = '1'
    terrain_str = ''.join(terrain)

    units_xml = [
        '    <rts.units.Unit type="Resource" ID="1" player="-1" x="0" y="0" resources="25" hitpoints="1"></rts.units.Unit>',
        '    <rts.units.Unit type="Resource" ID="2" player="-1" x="0" y="1" resources="25" hitpoints="1"></rts.units.Unit>',
        f'    <rts.units.Unit type="Resource" ID="5" player="-1" x="{MAP_W-1}" y="{MAP_H-1}" resources="25" hitpoints="1"></rts.units.Unit>',
        f'    <rts.units.Unit type="Resource" ID="6" player="-1" x="{MAP_W-1}" y="{MAP_H-2}" resources="25" hitpoints="1"></rts.units.Unit>',
        '    <rts.units.Unit type="Base" ID="3" player="0" x="2" y="2" resources="0" hitpoints="10"></rts.units.Unit>',
        f'    <rts.units.Unit type="Base" ID="7" player="1" x="{MAP_W-3}" y="{MAP_H-3}" resources="0" hitpoints="10"></rts.units.Unit>',
        '    <rts.units.Unit type="Worker" ID="4" player="0" x="1" y="1" resources="0" hitpoints="1"></rts.units.Unit>',
        f'    <rts.units.Unit type="Worker" ID="8" player="1" x="{MAP_W-2}" y="{MAP_H-2}" resources="0" hitpoints="1"></rts.units.Unit>',
    ]

    xml = f'''<rts.PhysicalGameState width="{MAP_W}" height="{MAP_H}">
  <terrain>{terrain_str}</terrain>
  <players>
    <rts.Player ID="0" resources="5"></rts.Player>
    <rts.Player ID="1" resources="5"></rts.Player>
  </players>
  <units>
{chr(10).join(units_xml)}
  </units>
</rts.PhysicalGameState>'''

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding="utf-8") as f:
        f.write(xml)


# ═══════════════════════════════════════════════════════════════
#  73-D FINGERPRINT & SIMILARITY (For EC-SA-TDCCEA)
# ═══════════════════════════════════════════════════════════════

POP_SL  = slice(0, 2 * D)
OPP_SL  = slice(2 * D, 2 * D + N_OPP)
GAME_SL = slice(OPP_SL.stop, OPP_SL.stop + 2)
FIT_SL  = slice(GAME_SL.stop, GAME_SL.stop + 3)
BLOCK_W = (0.4, 0.3, 0.1, 0.2)

def build_fingerprint_73d(gen, opp, pop, archive, fit_stats):
    cen = centroid([a.params for a in pop])
    if archive:
        top = sorted(archive, key=lambda a: a.e_i, reverse=True)[:3]
        elite = centroid([a.params for a in top])
    else:
        elite = cen
    onehot = [1.0 if o == opp else 0.0 for o in OPPONENT_ROTATION]
    div = mean(euclidean_dist(a.params, cen) for a in pop) / SQRT_D
    game = [gen / TOTAL_GENERATIONS, min(1.0, 2.0 * div)]
    return cen + elite + onehot + game + list(fit_stats)


def _esim(a, b):
    if not a or not b or len(a) == 0:
        return 1.0
    return max(0.0, 1.0 - euclidean_dist(a, b) / math.sqrt(len(a)))


def env_sim_73d(e1, e2):
    if len(e1) <= D or len(e2) <= D:
        return max(0.0, 1.0 - euclidean_dist(e1[:D], e2[:D]) / SQRT_D)
    s_pop  = _esim(e1[POP_SL], e2[POP_SL])
    s_opp  = max(0.0, 1.0 - 0.5 * sum(abs(p - q) for p, q in zip(e1[OPP_SL], e2[OPP_SL])))
    s_game = _esim(e1[GAME_SL], e2[GAME_SL])
    s_fit  = _esim(e1[FIT_SL], e2[FIT_SL])
    return sum(w * s for w, s in zip(BLOCK_W, (s_pop, s_opp, s_game, s_fit))) / sum(BLOCK_W)


# ═══════════════════════════════════════════════════════════════
#  MULTI-MODAL STRATEGIC MODES & OPTIMAL TRANSPORT
# ═══════════════════════════════════════════════════════════════

_fp_counter = 0

def _next_uid():
    global _fp_counter
    _fp_counter += 1
    return _fp_counter

def _sse(X, idx):
    p = X[idx]
    return float(((p - p.mean(axis=0)) ** 2).sum())

def _best_split(X, idx):
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


def build_fingerprint_modes(params_list, fits, tau0=TAU0, lam=LAM):
    """
    Construct Multi-Modal Strategic Fingerprint F_t = {(theta_m, w_m, g_m)}.
    Clustering operates in normalized space (x / sqrt(D)).
    """
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
        "uid": _next_uid()
    }


def build_subpop_fingerprint(params_list, fits, dim, tau0=TAU0, lam=LAM):
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


def _hungarian(C):
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


def d_env_modes(FA, FB):
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


# ═══════════════════════════════════════════════════════════════
#  DYNAMIC SENSITIVITY & COVARIANCE INTERACTION (EC-SA Parts)
# ═══════════════════════════════════════════════════════════════

def ablation_update(pool, memory, baseline, opp, E, is_modes=False):
    """Local ablation of each parameter j around baseline."""
    vecs = []
    for j in range(D):
        for v in (0.0, 1.0):
            t = baseline[:]
            t[j] = v
            vecs.append(t)
    fits = pool.eval_batch_vs_bot(vecs, opp, 1)

    for j in range(D):
        S = abs(fits[2 * j + 1] - fits[2 * j])
        clusters = memory[j]
        best_k, best_s = None, -1.0
        for k, c in enumerate(clusters):
            if is_modes:
                s = max(0.0, 1.0 - d_env_modes(E, c["mu"]) / D_MAX)
            else:
                s = env_sim_73d(E, c["mu"])
            if s > best_s:
                best_k, best_s = k, s
        if best_k is not None and best_s >= TAU_I:
            c = clusters[best_k]
            n = c["n"]
            if not is_modes:
                c["mu"] = [(n * m + e) / (n + 1) for m, e in zip(c["mu"], E)]
            c["n"]  = n + 1
            c["S"]  = (1.0 - ETA) * c["S"] + ETA * S
        else:
            clusters.append({"mu": E, "S": S, "n": 1})


def retrieve_sensitivity(memory, E, is_modes=False):
    """Retrieve sensitivity conditioned on current environmental similarity."""
    out = [None] * D
    for j in range(D):
        num = den = 0.0
        for c in memory[j]:
            if is_modes:
                s = max(0.0, 1.0 - d_env_modes(E, c["mu"]) / D_MAX)
            else:
                s = env_sim_73d(E, c["mu"])
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
    return np.array([r / m for r in rate])


def interaction_observation(P, f):
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


def covariance_factor(I):
    """Compute Cholesky factor L of correlation matrix C = Id + g*I."""
    C = np.eye(D) + I_COUPLING * (I - np.diag(np.diag(I)))
    w, V = np.linalg.eigh((C + C.T) / 2)
    C = (V * np.clip(w, 1e-3, None)) @ V.T
    dg = np.sqrt(np.diag(C))
    return np.linalg.cholesky(C / np.outer(dg, dg) + 1e-9 * np.eye(D))


# ═══════════════════════════════════════════════════════════════
#  AGENT & SUB-AGENT CLASSES
# ═══════════════════════════════════════════════════════════════

class Agent:
    """Agent for monolithic 20-D algorithms (Pure TD-CCEA and SA-TDCCEA)."""
    def __init__(self, params=None, pwm=None):
        self.params = ([random.random() for _ in range(D)] if params is None
                       else [clamp01(x) for x in params])
        self.H_i = []
        self.e_i = 0.0
        self.mu_i = 0.0
        self.pwm = pwm

    def update_history(self, f, E):
        self.H_i.append((f, E))
        if len(self.H_i) > TRAJ_WINDOW:
            self.H_i.pop(0)

    def compute_trajectory(self, E, alpha_fn):
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
    """Sub-agent for Cooperative Co-Evolution (Macro 8-D and Micro 12-D)."""
    def __init__(self, dim, params=None, env_offset=0):
        self.dim = dim
        self.env_offset = env_offset
        self.params = ([random.random() for _ in range(dim)] if params is None
                       else [clamp01(x) for x in params])
        self.H_i = []
        self.e_i = 0.0
        self.mu_i = 0.0

    def update_history(self, f, E):
        self.H_i.append((f, E))
        if len(self.H_i) > TRAJ_WINDOW:
            self.H_i.pop(0)

    def compute_trajectory(self, E, alpha_fn):
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


def select_and_reproduce(pop, archive, mutate_fn, target_size=None):
    """Redundancy removal, failure pruning, archive update, and tournament selection."""
    if target_size is None:
        target_size = len(pop)
    dim = len(pop[0].params)
    sqrt_dim = math.sqrt(dim)
    min_keep = min(MIN_KEEP, target_size)

    ranked = sorted(pop, key=lambda a: a.e_i, reverse=True)
    survivors = []
    for a in ranked:
        if not any(1.0 - euclidean_dist(a.params, u.params) / sqrt_dim > DELTA_S for u in survivors):
            survivors.append(a)

    final = [a for a in survivors if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
    while len(final) < min_keep and len(survivors) > len(final):
        for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
            if a not in final:
                final.append(a)
                break

    for top in ranked[:3]:
        archive.append(top.snapshot())
    archive.sort(key=lambda a: a.e_i, reverse=True)
    del archive[A_MAX:]

    pool_ = final + archive
    nxt = final[:]
    while len(nxt) < target_size:
        cands = random.sample(pool_, min(3, len(pool_)))
        nxt.append(mutate_fn(max(cands, key=lambda a: a.e_i)))
    return nxt, archive


def subpop_survival_step(pop, archive, target_size=None):
    """Survival & reproduction specifically for CCE sub-populations."""
    if target_size is None:
        target_size = len(pop)
    return select_and_reproduce(pop, archive, lambda parent: parent.mutate(), target_size=target_size)


def breed_std_ccea(pop, fits, pop_size=N_POP, mut=0.15):
    """Tournament selection and Gaussian mutation for Standard CCEA sub-populations (COMPARE_ALL)."""
    bi = max(range(len(pop)), key=lambda i: fits[i])
    new_pop = [pop[bi][:]]  # Elitism
    while len(new_pop) < pop_size:
        t = random.sample(range(len(pop)), min(3, len(pop)))
        pidx = max(t, key=lambda i: fits[i])
        new_pop.append([clamp01(x + random.gauss(0, mut)) for x in pop[pidx]])
    return new_pop


# ═══════════════════════════════════════════════════════════════
#  ALGORITHM SPECIFICATIONS & SUITE REGISTRY (ALL 14 ALGORITHMS)
# ═══════════════════════════════════════════════════════════════

ALL_VARIANTS = {
    # ── Family 1: Pure Monolithic TD-CCEA (3 variants) ──
    "td": {
        "name": "TD-CCEA",
        "family": "TD-CCEA",
        "is_cce": False,
        "is_baseline": False,
        "env_type": "centroid",
        "sens_type": "none",
        "cov": False,
        "dc_max": None,
        "d_max": None,
    },
    "td_sharp": {
        "name": "TD-CCEA+SharpKernel",
        "family": "TD-CCEA",
        "is_cce": False,
        "is_baseline": False,
        "env_type": "centroid",
        "sens_type": "none",
        "cov": False,
        "dc_max": DC_MAX_SHARP,
        "d_max": None,
    },
    "td_up": {
        "name": "TD-CCEA-Upgraded",
        "family": "TD-CCEA",
        "is_cce": False,
        "is_baseline": False,
        "env_type": "modes_ot",
        "sens_type": "none",
        "cov": False,
        "dc_max": None,
        "d_max": D_MAX,
    },

    # ── Family 2: Sensitivity-Adaptive SA-TDCCEA (4 variants) ──
    "sa": {
        "name": "SA-TDCCEA",
        "family": "SA-TDCCEA",
        "is_cce": False,
        "is_baseline": False,
        "env_type": "centroid",
        "sens_type": "static",
        "cov": False,
        "dc_max": None,
        "d_max": None,
    },
    "sa_sharp": {
        "name": "SA-TDCCEA+SharpKernel",
        "family": "SA-TDCCEA",
        "is_cce": False,
        "is_baseline": False,
        "env_type": "centroid",
        "sens_type": "static",
        "cov": False,
        "dc_max": DC_MAX_SHARP,
        "d_max": None,
    },
    "ec_sa": {
        "name": "EC-SA-TDCCEA",
        "family": "SA-TDCCEA",
        "is_cce": False,
        "is_baseline": False,
        "env_type": "fingerprint_73d",
        "sens_type": "dynamic",
        "cov": True,
        "dc_max": None,
        "d_max": None,
    },
    "ec_sa_up": {
        "name": "EC-SA-TDCCEA-Upgraded",
        "family": "SA-TDCCEA",
        "is_cce": False,
        "is_baseline": False,
        "env_type": "modes_ot",
        "sens_type": "dynamic",
        "cov": True,
        "dc_max": None,
        "d_max": D_MAX,
    },

    # ── Family 3: Cooperative Co-Evolution CCE-TDCCEA (3 variants) ──
    "cce": {
        "name": "CCE-TDCCEA",
        "family": "CCE-TDCCEA",
        "is_cce": True,
        "is_baseline": False,
        "env_type": "centroid",
        "sens_type": "none",
        "cov": False,
        "dc_max": None,
        "d_max": None,
    },
    "cce_sharp": {
        "name": "CCE-TDCCEA+SharpKernel",
        "family": "CCE-TDCCEA",
        "is_cce": True,
        "is_baseline": False,
        "env_type": "centroid",
        "sens_type": "none",
        "cov": False,
        "dc_max": DC_MAX_SHARP,
        "d_max": None,
    },
    "cce_up": {
        "name": "CCE-TDCCEA-Upgraded",
        "family": "CCE-TDCCEA",
        "is_cce": True,
        "is_baseline": False,
        "env_type": "modes_ot",
        "sens_type": "none",
        "cov": False,
        "dc_max": None,
        "d_max": D_MAX,
    },

    # ── Family 4: Classical Optimization & Evolutionary Baselines (COMPARE_ALL) ──
    "std_ccea": {
        "name": "Standard CCEA",
        "family": "Baseline",
        "is_cce": False,
        "is_baseline": True,
        "base_type": "std_ccea",
        "env_type": "none",
        "sens_type": "none",
        "cov": False,
        "dc_max": None,
        "d_max": None,
    },
    "cmaes": {
        "name": "CMA-ES",
        "family": "Baseline",
        "is_cce": False,
        "is_baseline": True,
        "base_type": "cmaes",
        "env_type": "none",
        "sens_type": "none",
        "cov": True,
        "dc_max": None,
        "d_max": None,
    },
    "pso": {
        "name": "PSO",
        "family": "Baseline",
        "is_cce": False,
        "is_baseline": True,
        "base_type": "pso",
        "env_type": "none",
        "sens_type": "none",
        "cov": False,
        "dc_max": None,
        "d_max": None,
    },
    "hill": {
        "name": "Hill Climbing",
        "family": "Baseline",
        "is_cce": False,
        "is_baseline": True,
        "base_type": "hill",
        "env_type": "none",
        "sens_type": "none",
        "cov": False,
        "dc_max": None,
        "d_max": None,
    },
}

SUITE_CHOICES = {
    "all": [
        "td", "td_sharp", "td_up",
        "sa", "sa_sharp", "ec_sa", "ec_sa_up",
        "cce", "cce_sharp", "cce_up",
        "std_ccea", "cmaes", "pso", "hill"
    ],
    "cce":      ["cce", "cce_sharp", "cce_up"],
    "td":       ["td", "td_sharp", "td_up"],
    "sa":       ["sa", "sa_sharp", "ec_sa", "ec_sa_up"],
    "baseline": ["std_ccea", "cmaes", "pso", "hill"],
}

def make_alpha_fn(cfg):
    env_type = cfg["env_type"]
    if env_type == "centroid":
        dc = cfg["dc_max"]
        if dc is None:
            return lambda E, Fk: max(0.0, 1.0 - euclidean_dist(E, Fk) / math.sqrt(len(E)))
        return lambda E, Fk: max(0.0, 1.0 - (euclidean_dist(E, Fk) / math.sqrt(len(E))) / dc)
    elif env_type == "fingerprint_73d":
        return lambda E, Fk: env_sim_73d(E, Fk)
    elif env_type == "modes_ot":
        d_max = cfg["d_max"]
        cache = {}
        def alpha(Ft, Fk):
            key = (Ft["uid"], Fk["uid"])
            if key not in cache:
                cache[key] = d_env_modes(Ft, Fk)
            return max(0.0, 1.0 - cache[key] / d_max)
        return alpha


def make_cce_subpop_alpha_fn(cfg, dim):
    env_type = cfg["env_type"]
    sqrt_dim = math.sqrt(dim)
    if env_type == "centroid":
        dc = cfg["dc_max"]
        if dc is None:
            return lambda E, Fk: max(0.0, 1.0 - euclidean_dist(E, Fk) / sqrt_dim)
        return lambda E, Fk: max(0.0, 1.0 - (euclidean_dist(E, Fk) / sqrt_dim) / dc)
    elif env_type == "modes_ot":
        d_max = cfg["d_max"]
        cache = {}
        def alpha(Ft, Fk):
            key = (Ft["uid"], Fk["uid"])
            if key not in cache:
                cache[key] = d_env_modes(Ft, Fk)
            return max(0.0, 1.0 - cache[key] / d_max)
        return alpha


def load_or_run_static_sensitivity(pool):
    """Load cached static sensitivity (pwm) or compute across bots."""
    fp = os.path.join(SCRIPT_DIR, SENSITIVITY_FILE)
    if os.path.exists(fp):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            rates = data.get("normalized_mutation_rates")
            if rates and len(rates) == DIMENSIONS:
                print(f"  [SA] Loaded cached static sensitivity from {SENSITIVITY_FILE}")
                return rates
        except Exception:
            pass

    print("  [SA] Running parallel static sensitivity analysis across 7 bots...")
    baseline = [0.5] * D
    vecs = []
    for j in range(D):
        for v in (0.0, 1.0):
            t = baseline[:]
            t[j] = v
            vecs.append(t)

    influence = [0.0] * D
    for opp in OPPONENT_ROTATION:
        fits = pool.eval_batch_vs_bot(vecs, opp, 1)
        for j in range(D):
            influence[j] += abs(fits[2 * j + 1] - fits[2 * j])

    influence = [inf / len(OPPONENT_ROTATION) for inf in influence]
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


# ═══════════════════════════════════════════════════════════════
#  GAME SERVERS & PARALLEL DISPATCH POOL
# ═══════════════════════════════════════════════════════════════

class GameServer:
    """Java microRTS process wrapper supporting EVOLVED vs BOT and EVOLVED vs EVOLVED."""
    def __init__(self):
        self.process = None
        self._total_count = 0
        self._fail_count = 0
        self._forced_draws = 0

    def start(self):
        self.process = subprocess.Popen(
            [JAVA_CMD, "-cp", "lib/*;bin", "tests.EvolvedBatchGameRunner"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=MICRORTS_DIR,
            encoding="utf-8", errors="replace", bufsize=1
        )
        line = self.process.stdout.readline().strip()
        if line != "READY":
            err = ""
            try:
                err = self.process.stderr.read()
            except Exception:
                pass
            raise RuntimeError(f"Server failed to start, received: '{line}'. Java Stderr: {err}")

    def _check_alive(self):
        return self.process is not None and self.process.poll() is None

    def run_game(self, weights, opponent, map_path=TEMP_MAP_REL):
        self._total_count += 1
        if not self._check_alive():
            self._fail_count += 1
            return -1, MAX_CYCLES

        ws = ";".join(f"{float(w):.4f}" for w in weights)
        cmd = f"EVOLVED:{ws},{opponent},{map_path},{MAX_CYCLES}\n"
        try:
            self.process.stdin.write(cmd)
            self.process.stdin.flush()
            r = self.process.stdout.readline().strip()
        except Exception:
            self._fail_count += 1
            return -1, MAX_CYCLES

        if r.startswith("RESULT:"):
            p = r.replace("RESULT:", "").split(",")
            winner, cycles = int(p[0]), int(p[1])
            if cycles >= MAX_CYCLES and winner != -1:
                self._forced_draws += 1
                winner = -1
            return winner, cycles

        self._fail_count += 1
        return -1, MAX_CYCLES

    def run_game_evolved_vs_evolved(self, p0_weights, p1_weights, map_path=TEMP_MAP_REL):
        self._total_count += 1
        if not self._check_alive():
            self._fail_count += 1
            return -1, MAX_CYCLES

        w0 = ";".join(f"{float(w):.4f}" for w in p0_weights)
        w1 = ";".join(f"{float(w):.4f}" for w in p1_weights)
        req = f"EVOLVED:{w0},EVOLVED:{w1},{map_path},{MAX_CYCLES}\n"
        try:
            self.process.stdin.write(req)
            self.process.stdin.flush()
            resp = self.process.stdout.readline().strip()
        except Exception:
            self._fail_count += 1
            return -1, MAX_CYCLES

        if resp.startswith("RESULT:"):
            parts = resp.replace("RESULT:", "").split(",")
            winner, cycles = int(parts[0]), int(parts[1])
            if cycles >= MAX_CYCLES and winner != -1:
                self._forced_draws += 1
                winner = -1
            return winner, cycles

        self._fail_count += 1
        return -1, MAX_CYCLES

    def stop(self):
        if self.process:
            try:
                self.process.stdin.write("EXIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=3)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        self.process = None


class MockGameServer:
    """Mock runner for instant offline debugging and verification without microRTS."""
    def start(self): pass
    def stop(self): pass

    def run_game(self, weights, opp, map_path=None):
        p = 0.15 + 0.70 * (0.4 * weights[0] + 0.3 * weights[7] + 0.3 * weights[14])
        r = random.random()
        return (0, 150) if r < p else ((-1, MAX_CYCLES) if r < p + 0.05 else (1, 150))

    def run_game_evolved_vs_evolved(self, p0_weights, p1_weights, map_path=None):
        s0 = sum(p0_weights) / len(p0_weights)
        s1 = sum(p1_weights) / len(p1_weights)
        diff = s0 - s1
        prob_p0 = 0.50 + 0.40 * math.tanh(diff * 2.0)
        r = random.random()
        if r < 0.05:
            return -1, MAX_CYCLES
        return (0, 200) if r < prob_p0 else (1, 200)


class ParallelGamePool:
    """Parallel pool dispatching microRTS games across CPU worker threads."""
    def __init__(self, n_workers, is_mock=False):
        self.n_workers = n_workers
        self.is_mock = is_mock
        self.pool_queue = queue.Queue()
        self.servers = []

    def start(self):
        server_cls = MockGameServer if self.is_mock else GameServer
        for _ in range(self.n_workers):
            s = server_cls()
            s.start()
            self.servers.append(s)
            self.pool_queue.put(s)
        print(f"  [POOL] Started {self.n_workers} game server instances "
              f"({'MOCK' if self.is_mock else 'JAVA'}).")

    def eval_batch_vs_bot(self, weight_list, opponent, k_games):
        def _eval_one(w):
            s = self.pool_queue.get()
            try:
                score = 0.0
                for _ in range(k_games):
                    w_res, _ = s.run_game(w, opponent)
                    if w_res == 0:
                        score += 1.0
                    elif w_res == -1:
                        score += DRAW_SCORE
                return score / k_games
            finally:
                self.pool_queue.put(s)

        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            return list(ex.map(_eval_one, weight_list))

    def eval_batch_vs_agents(self, weight_list, opponent_weights_list):
        def _eval_one(w):
            s = self.pool_queue.get()
            try:
                score = 0.0
                n = len(opponent_weights_list)
                if n == 0:
                    return 0.0
                for i, opp_w in enumerate(opponent_weights_list):
                    if i % 2 == 0:
                        winner, _ = s.run_game_evolved_vs_evolved(w, opp_w)
                        if winner == 0: score += 1.0
                        elif winner == -1: score += DRAW_SCORE
                    else:
                        winner, _ = s.run_game_evolved_vs_evolved(opp_w, w)
                        if winner == 1: score += 1.0
                        elif winner == -1: score += DRAW_SCORE
                return score / n
            finally:
                self.pool_queue.put(s)

        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            return list(ex.map(_eval_one, weight_list))

    def validate_all_bots(self, weights, games_per_bot):
        def _eval_bot(opp):
            s = self.pool_queue.get()
            try:
                wins = 0
                for _ in range(games_per_bot):
                    w_res, _ = s.run_game(weights, opp)
                    if w_res == 0:
                        wins += 1
                return opp, (wins / games_per_bot) * 100.0
            finally:
                self.pool_queue.put(s)

        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            futures = [ex.submit(_eval_bot, opp) for opp in OPPONENT_ROTATION]
            per_opp = {}
            for fut in futures:
                opp, wr = fut.result()
                per_opp[opp] = wr
        overall = sum(per_opp.values()) / len(per_opp)
        return overall, per_opp

    def run_h2h_parallel(self, p0_weights, p1_weights, n_games):
        def _run_one(swap):
            s = self.pool_queue.get()
            try:
                if swap:
                    winner, _ = s.run_game_evolved_vs_evolved(p1_weights, p0_weights)
                else:
                    winner, _ = s.run_game_evolved_vs_evolved(p0_weights, p1_weights)
                return winner, swap
            finally:
                self.pool_queue.put(s)

        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            futs = [ex.submit(_run_one, g % 2 == 1) for g in range(n_games)]
            return [f.result() for f in futs]

    def run_test_game(self):
        s = self.pool_queue.get()
        try:
            if self.is_mock:
                return True
            w = [0.5] * DIMENSIONS
            res, _ = s.run_game(w, "WorkerRush")
            return res != -1 or s._fail_count == 0
        finally:
            self.pool_queue.put(s)

    def stop(self):
        while not self.pool_queue.empty():
            s = self.pool_queue.get()
            s.stop()


# ═══════════════════════════════════════════════════════════════
#  STATISTICAL ANALYSIS (WILCOXON & TOURNAMENT MATRIX)
# ═══════════════════════════════════════════════════════════════

def wilcoxon_test(x, y):
    """Wilcoxon signed-rank test with continuity correction."""
    diffs = [xi - yi for xi, yi in zip(x, y) if xi != yi]
    n = len(diffs)
    if n < 3:
        return None, n
    abs_d = sorted(enumerate(abs(d) for d in diffs), key=lambda t: t[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and abs_d[j][1] == abs_d[j+1][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[abs_d[k][0]] = avg_rank
        i = j + 1
    w_plus  = sum(ranks[idx] for idx, d in enumerate(diffs) if d > 0)
    w_minus = sum(ranks[idx] for idx, d in enumerate(diffs) if d < 0)
    w_stat  = min(w_plus, w_minus)
    mu_w    = n * (n + 1) / 4.0
    sig_w   = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sig_w == 0:
        return None, n
    z = (abs(w_stat - mu_w) - 0.5) / sig_w
    z = max(z, 0.0)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return p, n


def run_head_to_head(pool, algo_weights, algo_names, n_games=H2H_GAMES):
    n = len(algo_names)
    matrix = [[None] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            i_wins, j_wins, draws = 0, 0, 0
            game_results = pool.run_h2h_parallel(algo_weights[i], algo_weights[j], n_games)

            for winner, swap in game_results:
                if not swap:
                    if winner == 0: i_wins += 1
                    elif winner == 1: j_wins += 1
                    else: draws += 1
                else:
                    if winner == 0: j_wins += 1
                    elif winner == 1: i_wins += 1
                    else: draws += 1

            matrix[i][j] = (i_wins, j_wins, draws)
            matrix[j][i] = (j_wins, i_wins, draws)
            print(f"  {algo_names[i]:26s} vs {algo_names[j]:26s} -> {i_wins}W-{j_wins}L-{draws}D")

    return matrix


def format_h2h_matrix(matrix, algo_names):
    n = len(algo_names)
    col_w = 11
    total_wins = [0] * n
    total_games = [0] * n

    for i in range(n):
        for j in range(n):
            if i != j and matrix[i][j] is not None:
                w, l, d = matrix[i][j]
                total_wins[i] += w
                total_games[i] += (w + l + d)

    win_rates = [(total_wins[i] / total_games[i] * 100.0 if total_games[i] > 0 else 0.0)
                 for i in range(n)]

    lines = []
    header = f"  {'':22s}"
    for name in algo_names:
        header += f" {name[:col_w]:>{col_w}s}"
    header += f"  {'Wins':>6s}  {'WR%':>7s}"
    lines.append(header)

    sep = "  " + "─" * 22
    for _ in algo_names:
        sep += " " + "─" * col_w
    sep += "  " + "─" * 6 + "  " + "─" * 7
    lines.append(sep)

    for i in range(n):
        row = f"  {algo_names[i]:22s}"
        for j in range(n):
            if i == j:
                row += f" {'—':>{col_w}s}"
            else:
                w, l, d = matrix[i][j]
                cell = f"{w}W-{l}L-{d}D"
                row += f" {cell:>{col_w}s}"
        row += f"  {total_wins[i]:>6d}  {win_rates[i]:>6.1f}%"
        lines.append(row)
    lines.append(sep)

    return lines, total_wins, win_rates


def export_excel(results, matrix, algo_names, total_wins, h2h_win_rates, out_path):
    if not _OPENPYXL_AVAILABLE:
        return
    wb = openpyxl.Workbook()
    ws_bots = wb.active
    ws_bots.title = "Bot Validation"

    f_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    fill_header = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    fill_alt = PatternFill(start_color="EDF2F8", end_color="EDF2F8", fill_type="solid")
    center = Alignment(horizontal="center", vertical="center")

    headers_bots = ["Rank", "Algorithm", "Family", "Overall Win Rate (%)", "Time (s)"] + OPPONENT_ROTATION
    ws_bots.append(headers_bots)
    for col in range(1, len(headers_bots) + 1):
        cell = ws_bots.cell(row=1, column=col)
        cell.font = f_header
        cell.fill = fill_header
        cell.alignment = center

    ranked = sorted(results, key=lambda x: x["wr"], reverse=True)
    for rank_idx, r in enumerate(ranked, 1):
        family = ALL_VARIANTS[r["key"]]["family"]
        row = [f"#{rank_idx}", r["name"], family, f"{r['wr']:.1f}%", f"{r['time']:.1f}s"]
        for opp in OPPONENT_ROTATION:
            row.append(f"{r['per_opp'].get(opp, 0.0):.1f}%")
        ws_bots.append(row)
        if rank_idx % 2 == 0:
            for c in range(1, len(row) + 1):
                ws_bots.cell(row=rank_idx + 1, column=c).fill = fill_alt

    ws_h2h = wb.create_sheet(title="Head-to-Head Tournament")
    headers_h2h = ["Algorithm"] + algo_names + ["Total Wins", "H2H Win Rate (%)"]
    ws_h2h.append(headers_h2h)
    for col in range(1, len(headers_h2h) + 1):
        cell = ws_h2h.cell(row=1, column=col)
        cell.font = f_header
        cell.fill = fill_header
        cell.alignment = center

    for i in range(len(algo_names)):
        row = [algo_names[i]]
        for j in range(len(algo_names)):
            if i == j:
                row.append("—")
            else:
                w, l, d = matrix[i][j]
                row.append(f"{w}W-{l}L-{d}D")
        row.append(total_wins[i])
        row.append(f"{h2h_win_rates[i]:.1f}%")
        ws_h2h.append(row)
        if i % 2 == 1:
            for c in range(1, len(row) + 1):
                ws_h2h.cell(row=i + 2, column=c).fill = fill_alt

    for sheet in [ws_bots, ws_h2h]:
        for col in sheet.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = openpyxl.utils.get_column_letter(col[0].column)
            sheet.column_dimensions[col_letter].width = max(max_len + 3, 12)

    wb.save(out_path)
    print(f"  Saved Excel Report -> {out_path}")


# ═══════════════════════════════════════════════════════════════
#  MAIN EXECUTION PIPELINE
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SEM7_ER: Complete 14-Algorithm TD-CCEA, SA, CCE, and Classical Baselines Benchmark on 48×48 Dynamic Map.")
    parser.add_argument("--suite", choices=["all", "td", "sa", "cce", "baseline"], default="all",
                        help="Algorithm suite to run ('all': all 14 algorithms, 'td': 3 pure TD-CCEA, 'sa': 4 sensitivity-adaptive, 'cce': 3 cooperative co-evolution, 'baseline': 4 classical baselines [Std CCEA, CMA-ES, PSO, Hill Climbing])")
    parser.add_argument("--generations", type=int, default=TOTAL_GENERATIONS, help="Number of evolutionary generations")
    parser.add_argument("--pop-size", type=int, default=N_POP, help="Population size (or per sub-population for CCE)")
    parser.add_argument("--k-games", type=int, default=K_GAMES, help="Games per individual per evaluation")
    parser.add_argument("--h2h-games", type=int, default=H2H_GAMES, help="Games per H2H matchup pair")
    parser.add_argument("--val-games", type=int, default=VALIDATION_GAMES, help="Validation games per bot")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel game servers")
    parser.add_argument("--eval", choices=["bots", "rivals"], default="bots",
                        help="Evaluation mode during evolution ('bots': rotating opponent bots, 'rivals': rival algorithm champions)")
    parser.add_argument("--dc-max", type=float, default=DC_MAX_SHARP, help="DC_MAX threshold for sharpened kernel controls")
    parser.add_argument("--d-max", type=float, default=D_MAX, help="D_MAX threshold for upgraded modal OT variants")
    parser.add_argument("--tau0", type=float, default=TAU0, help="Base clustering threshold tau_0")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mock", action="store_true", help="Run fast dry-run without Java microRTS")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    active_keys = SUITE_CHOICES[args.suite]
    n_algos = len(active_keys)

    n_workers = args.workers or max(1, min(10, (os.cpu_count() or 4) - 1))

    print("\n" + "═" * 80)
    print(f"  SEM7_ER — Master Benchmark: {n_algos} Algorithms on 48×48 Dynamic Map")
    print("═" * 80)
    print("  Active Suite Algorithms:")
    for idx, k in enumerate(active_keys, 1):
        v = ALL_VARIANTS[k]
        print(f"    {idx:2d}. {v['name']:<26s} [{v['family']:<10s}] (env={v['env_type']}, sens={v['sens_type']})")
    print(f"  Map:         {MAP_W}×{MAP_H} with Progressive Blockades | MAX_CYCLES={MAX_CYCLES}")
    print(f"  Evolution:   Pop={args.pop_size} | Gens={args.generations} | K_Games={args.k_games} | Eval={args.eval.upper()}")
    print(f"  Tournament:  {n_algos}×{n_algos} Matrix ({n_algos*(n_algos-1)//2} Pairs, {args.h2h_games} Games/Pair) | Workers={n_workers}")
    if args.mock:
        print("  MODE:        *** MOCK TESTING MODE (No Java required) ***")
    print("═" * 80 + "\n")

    # Initialize dynamic environment map
    terrain_walls = set()
    breakable_positions = set()
    write_map_xml(terrain_walls, breakable_positions, TEMP_MAP_PATH)
    print(f"  Initialized clean 48×48 map (0 walls): {TEMP_MAP_PATH}")

    pool = ParallelGamePool(n_workers, is_mock=args.mock)
    pool.start()

    # Pre-flight diagnostic check
    if not pool.run_test_game():
        print("\n  [ERROR] Diagnostic test game failed on 48×48 map! Aborting.")
        pool.stop()
        return

    # Load or run static sensitivity scores (pwm) if SA variants are present
    needs_pwm = any(ALL_VARIANTS[k]["sens_type"] == "static" for k in active_keys)
    pwm = load_or_run_static_sensitivity(pool) if needs_pwm else None

    # Update hyperparameters
    for k in active_keys:
        if ALL_VARIANTS[k]["dc_max"] is not None:
            ALL_VARIANTS[k]["dc_max"] = args.dc_max
        if ALL_VARIANTS[k]["d_max"] is not None:
            ALL_VARIANTS[k]["d_max"] = args.d_max

    # Monolithic alpha similarity functions
    alpha_fns = {k: make_alpha_fn(ALL_VARIANTS[k]) for k in active_keys if not ALL_VARIANTS[k]["is_cce"] and not ALL_VARIANTS[k].get("is_baseline")}

    # CCE sub-population alpha functions
    dim_macro = len(MACRO_INDICES)
    dim_micro = len(MICRO_INDICES)
    cce_alpha_macro = {k: make_cce_subpop_alpha_fn(ALL_VARIANTS[k], dim_macro) for k in active_keys if ALL_VARIANTS[k]["is_cce"]}
    cce_alpha_micro = {k: make_cce_subpop_alpha_fn(ALL_VARIANTS[k], dim_micro) for k in active_keys if ALL_VARIANTS[k]["is_cce"]}

    # Shared initial population seeds across all monolithic algorithms
    init_params = [[random.random() for _ in range(DIMENSIONS)] for _ in range(args.pop_size)]
    init_params_macro = [[random.random() for _ in range(dim_macro)] for _ in range(args.pop_size)]
    init_params_micro = [[random.random() for _ in range(dim_micro)] for _ in range(args.pop_size)]

    populations = {}
    archives = {k: [] for k in active_keys if not ALL_VARIANTS[k].get("is_baseline")}
    convergence_hist = {k: [] for k in active_keys}

    # CCE-specific and Baseline-specific state structures
    cce_state = {}
    baseline_state = {}
    for k in active_keys:
        if ALL_VARIANTS[k]["is_cce"]:
            cce_state[k] = {
                "pop_a": [SubAgent(dim_macro, p, env_offset=0) for p in init_params_macro],
                "pop_b": [SubAgent(dim_micro, p, env_offset=dim_macro) for p in init_params_micro],
                "arch_a": [],
                "arch_b": [],
            }
        elif ALL_VARIANTS[k].get("is_baseline"):
            btype = ALL_VARIANTS[k]["base_type"]
            if btype == "std_ccea":
                baseline_state[k] = {
                    "pm": [p[:] for p in init_params_macro],
                    "pi": [p[:] for p in init_params_micro],
                    "fm": [0.0] * args.pop_size,
                    "fi": [0.0] * args.pop_size,
                }
            elif btype == "cmaes":
                has_cma = False
                es_inst = None
                try:
                    import cma
                    es_inst = cma.CMAEvolutionStrategy([0.5] * DIMENSIONS, 0.3, {
                        'maxiter': args.generations,
                        'popsize': args.pop_size,
                        'bounds': [0, 1],
                        'verbose': -9,
                        'seed': random.randint(0, 99999)
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
            p_pwm = pwm if ALL_VARIANTS[k]["sens_type"] == "static" else None
            populations[k] = [Agent(p, pwm=p_pwm) for p in init_params]

    # Dynamic EC-SA specific state trackers
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

    # ──────────────────────────────────────────────────────────────
    #  PHASE 1: DYNAMIC ENVIRONMENT EVOLUTION
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  PHASE 1: DYNAMIC EVOLUTION — {n_algos} Algorithms × {args.generations} Generations")
    print(f"{'='*80}")

    t_evol_start = time.time()

    current_best_weights = {}
    for k in active_keys:
        if ALL_VARIANTS[k]["is_cce"]:
            best_a = cce_state[k]["pop_a"][0]
            best_b = cce_state[k]["pop_b"][0]
            current_best_weights[k] = combine_subagents(best_a, best_b)
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
            current_best_weights[k] = max(populations[k], key=lambda a: a.e_i).params

    for gen in range(1, args.generations + 1):
        # 1. Update dynamic blockades
        terrain_walls, breakable_positions, n_terr, n_brk = add_walls_for_generation(
            gen, terrain_walls, breakable_positions
        )
        total_walls = len(terrain_walls) + len(breakable_positions)
        base0, base1 = (2, 2), (MAP_W - 3, MAP_H - 3)
        reachable = bfs_reachable(terrain_walls | breakable_positions, base0, base1)
        bfs_status = "REACHABLE" if reachable else "BLOCKED"
        write_map_xml(terrain_walls, breakable_positions, TEMP_MAP_PATH)

        opp = OPPONENT_ROTATION[(gen - 1) % N_OPP]
        gen_log = [f"Gen {gen:2d}/{args.generations:2d} | W:{total_walls:3d} [{bfs_status}] | {opp[:12]:12s}"]

        new_best_weights = {}

        # 2. Step each active algorithm synchronously
        for k in active_keys:
            cfg = ALL_VARIANTS[k]

            # ── A. Cooperative Co-Evolution Execution (CCE) ──
            if cfg["is_cce"]:
                st = cce_state[k]
                pop_a, pop_b = st["pop_a"], st["pop_b"]
                arch_a, arch_b = st["arch_a"], st["arch_b"]

                top_a = sorted(pop_a, key=lambda a: a.e_i, reverse=True)[:TOP_PARTNERS]
                top_b = sorted(pop_b, key=lambda b: b.e_i, reverse=True)[:TOP_PARTNERS]

                # Cooperative evaluation
                combos_a = [combine_subagents(a, pb) for a in pop_a for pb in top_b]
                if args.eval == "bots":
                    scores_a = pool.eval_batch_vs_bot(combos_a, opp, args.k_games)
                else:
                    rivals = [current_best_weights[o] for o in active_keys if o != k]
                    scores_a = pool.eval_batch_vs_agents(combos_a, rivals)
                k_b = len(top_b)
                fits_a = [sum(scores_a[i * k_b : (i + 1) * k_b]) / k_b for i in range(len(pop_a))]

                combos_b = [combine_subagents(pa, b) for b in pop_b for pa in top_a]
                if args.eval == "bots":
                    scores_b = pool.eval_batch_vs_bot(combos_b, opp, args.k_games)
                else:
                    rivals = [current_best_weights[o] for o in active_keys if o != k]
                    scores_b = pool.eval_batch_vs_agents(combos_b, rivals)
                k_a = len(top_a)
                fits_b = [sum(scores_b[j * k_a : (j + 1) * k_a]) / k_a for j in range(len(pop_b))]

                # Trajectory update
                if cfg["env_type"] == "centroid":
                    ca = [sum(a.params[j] for a in pop_a) / len(pop_a) for j in range(dim_macro)]
                    cb = [sum(b.params[j] for b in pop_b) / len(pop_b) for j in range(dim_micro)]
                    for a, f in zip(pop_a, fits_a):
                        a.update_history(f, ca)
                        a.compute_trajectory(ca, cce_alpha_macro[k])
                        a.compute_mutation()
                    for b, f in zip(pop_b, fits_b):
                        b.update_history(f, cb)
                        b.compute_trajectory(cb, cce_alpha_micro[k])
                        b.compute_mutation()
                    modes_tag = ""
                else: # modes_ot
                    Fa = build_subpop_fingerprint([a.params for a in pop_a], fits_a, dim_macro, args.tau0, LAM)
                    Fb = build_subpop_fingerprint([b.params for b in pop_b], fits_b, dim_micro, args.tau0, LAM)
                    for a, f in zip(pop_a, fits_a):
                        a.update_history(f, Fa)
                        a.compute_trajectory(Fa, cce_alpha_macro[k])
                        a.compute_mutation()
                    for b, f in zip(pop_b, fits_b):
                        b.update_history(f, Fb)
                        b.compute_trajectory(Fb, cce_alpha_micro[k])
                        b.compute_mutation()
                    modes_tag = f" Ma={len(Fa['count'])} Mb={len(Fb['count'])}"

                best_a = max(pop_a, key=lambda a: a.e_i)
                best_b = max(pop_b, key=lambda b: b.e_i)
                champ_20d = combine_subagents(best_a, best_b)
                new_best_weights[k] = champ_20d
                convergence_hist[k].append((best_a.e_i + best_b.e_i) / 2.0)

                gen_log.append(f"[{cfg['name'][:7]:7s} Ae={best_a.e_i:.2f} Be={best_b.e_i:.2f}{modes_tag}]")

                st["pop_a"], st["arch_a"] = subpop_survival_step(pop_a, arch_a, target_size=args.pop_size)
                st["pop_b"], st["arch_b"] = subpop_survival_step(pop_b, arch_b, target_size=args.pop_size)

            # ── B. Classical Optimization & Evolutionary Baselines (COMPARE_ALL) ──
            elif cfg.get("is_baseline"):
                bst = baseline_state[k]
                btype = cfg["base_type"]

                if btype == "std_ccea":
                    pm, pi = bst["pm"], bst["pi"]
                    tmi = sorted(range(len(pi)), key=lambda i: bst["fi"][i], reverse=True)[:TOP_PARTNERS]
                    tma = sorted(range(len(pm)), key=lambda i: bst["fm"][i], reverse=True)[:TOP_PARTNERS]

                    combos_m = [combine_subagents(pm[i], pi[j]) for i in range(len(pm)) for j in tmi]
                    if args.eval == "bots":
                        scores_m = pool.eval_batch_vs_bot(combos_m, opp, args.k_games)
                    else:
                        rivals = [current_best_weights[o] for o in active_keys if o != k]
                        scores_m = pool.eval_batch_vs_agents(combos_m, rivals)
                    k_mi = len(tmi)
                    for i in range(len(pm)):
                        bst["fm"][i] = sum(scores_m[i * k_mi : (i + 1) * k_mi]) / k_mi

                    combos_i = [combine_subagents(pm[j], pi[i]) for i in range(len(pi)) for j in tma]
                    if args.eval == "bots":
                        scores_i = pool.eval_batch_vs_bot(combos_i, opp, args.k_games)
                    else:
                        rivals = [current_best_weights[o] for o in active_keys if o != k]
                        scores_i = pool.eval_batch_vs_agents(combos_i, rivals)
                    k_ma = len(tma)
                    for i in range(len(pi)):
                        bst["fi"][i] = sum(scores_i[i * k_ma : (i + 1) * k_ma]) / k_ma

                    bm = max(range(len(pm)), key=lambda i: bst["fm"][i])
                    bi = max(range(len(pi)), key=lambda i: bst["fi"][i])
                    best_w = combine_subagents(pm[bm], pi[bi])
                    new_best_weights[k] = best_w
                    best_score = (bst["fm"][bm] + bst["fi"][bi]) / 2.0
                    convergence_hist[k].append(best_score)
                    gen_log.append(f"[{cfg['name'][:7]:7s} M={bst['fm'][bm]:.2f} m={bst['fi'][bi]:.2f}]")

                    # Breed next generation
                    best_fm = bst["fm"][bm]
                    best_fi = bst["fi"][bi]
                    bst["pm"] = breed_std_ccea(pm, bst["fm"], pop_size=args.pop_size)
                    bst["pi"] = breed_std_ccea(pi, bst["fi"], pop_size=args.pop_size)
                    bst["fm"] = [0.0] * args.pop_size
                    bst["fi"] = [0.0] * args.pop_size
                    bst["fm"][0] = best_fm
                    bst["fi"][0] = best_fi

                elif btype == "cmaes":
                    if bst["has_cma"] and not bst["es"].stop():
                        sols = bst["es"].ask()
                        cands = [[clamp01(x) for x in s] for s in sols]
                        if args.eval == "bots":
                            fits = pool.eval_batch_vs_bot(cands, opp, args.k_games)
                        else:
                            rivals = [current_best_weights[o] for o in active_keys if o != k]
                            fits = pool.eval_batch_vs_agents(cands, rivals)
                        for c, f in zip(cands, fits):
                            if f > bst["best_f"]:
                                bst["best_f"] = f
                                bst["best_p"] = c[:]
                        bst["es"].tell(sols, [-f for f in fits])
                        new_best_weights[k] = bst["best_p"][:]
                        convergence_hist[k].append(bst["best_f"])
                        gen_log.append(f"[{cfg['name'][:7]:7s} Best={bst['best_f']:.2f}]")
                    else:
                        # Fallback ES
                        cands = bst["pop"]
                        if args.eval == "bots":
                            fits = pool.eval_batch_vs_bot(cands, opp, args.k_games)
                        else:
                            rivals = [current_best_weights[o] for o in active_keys if o != k]
                            fits = pool.eval_batch_vs_agents(cands, rivals)
                        for c, f in zip(cands, fits):
                            if f > bst["best_f"]:
                                bst["best_f"] = f
                                bst["best_p"] = c[:]
                        ranked = sorted(range(len(cands)), key=lambda i: fits[i], reverse=True)
                        n_parents = max(1, len(cands) // 2)
                        parents = [cands[i] for i in ranked[:n_parents]]
                        new_pop = [parents[0][:]]
                        while len(new_pop) < args.pop_size:
                            p = random.choice(parents)
                            new_pop.append([clamp01(x + random.gauss(0, bst["sigma"])) for x in p])
                        bst["pop"] = new_pop
                        bst["sigma"] = max(0.05, bst["sigma"] * 0.98)
                        new_best_weights[k] = bst["best_p"][:]
                        convergence_hist[k].append(bst["best_f"])
                        gen_log.append(f"[{cfg['name'][:7]:7s} Best={bst['best_f']:.2f}]")

                elif btype == "pso":
                    cands = bst["pos"]
                    if args.eval == "bots":
                        fits = pool.eval_batch_vs_bot(cands, opp, args.k_games)
                    else:
                        rivals = [current_best_weights[o] for o in active_keys if o != k]
                        fits = pool.eval_batch_vs_agents(cands, rivals)
                    for i in range(len(cands)):
                        if fits[i] > bst["pf"][i]:
                            bst["pf"][i] = fits[i]
                            bst["pb"][i] = cands[i][:]
                        if fits[i] > bst["gf"]:
                            bst["gf"] = fits[i]
                            bst["gb"] = cands[i][:]
                    W, C1, C2 = bst["W"], bst["C1"], bst["C2"]
                    for i in range(len(cands)):
                        for j in range(DIMENSIONS):
                            bst["vel"][i][j] = (W * bst["vel"][i][j]
                                                 + C1 * random.random() * (bst["pb"][i][j] - bst["pos"][i][j])
                                                 + C2 * random.random() * (bst["gb"][j]    - bst["pos"][i][j]))
                            bst["pos"][i][j] = clamp01(bst["pos"][i][j] + bst["vel"][i][j])
                    new_best_weights[k] = bst["gb"][:]
                    convergence_hist[k].append(bst["gf"])
                    gen_log.append(f"[{cfg['name'][:7]:7s} Best={bst['gf']:.2f}]")

                elif btype == "hill":
                    curr = bst["current"]
                    cands = [curr[:]]
                    for _ in range(args.pop_size - 1):
                        cands.append([clamp01(x + random.gauss(0, 0.15)) for x in curr])
                    if args.eval == "bots":
                        fits = pool.eval_batch_vs_bot(cands, opp, args.k_games)
                    else:
                        rivals = [current_best_weights[o] for o in active_keys if o != k]
                        fits = pool.eval_batch_vs_agents(cands, rivals)
                    best_idx = max(range(len(cands)), key=lambda i: fits[i])
                    if fits[best_idx] >= bst["cfit"]:
                        bst["current"] = cands[best_idx][:]
                        bst["cfit"] = fits[best_idx]
                    new_best_weights[k] = bst["current"][:]
                    convergence_hist[k].append(bst["cfit"])
                    gen_log.append(f"[{cfg['name'][:7]:7s} Best={bst['cfit']:.2f}]")

            # ── C. Monolithic Algorithms (Pure TD-CCEA & SA-TDCCEA) ──
            else:
                pop = populations[k]
                arch = archives[k]
                params = [a.params for a in pop]

                # Evaluate
                if args.eval == "bots":
                    fits = pool.eval_batch_vs_bot(params, opp, args.k_games)
                else:
                    rivals = [current_best_weights[o] for o in active_keys if o != k]
                    fits = pool.eval_batch_vs_agents(params, rivals)

                # Build environment representation
                env_type = cfg["env_type"]
                modes_count = None
                if env_type == "centroid":
                    E = centroid(params)
                elif env_type == "fingerprint_73d":
                    E = build_fingerprint_73d(gen, opp, pop, arch, ec_state[k]["stats"])
                elif env_type == "modes_ot":
                    E = build_fingerprint_modes(params, fits, args.tau0, LAM)
                    modes_count = len(E["count"])

                # Update histories & trajectory scores
                for a, f in zip(pop, fits):
                    a.update_history(f, E)

                for a in pop:
                    a.compute_trajectory(E, alpha_fns[k])
                    a.compute_mutation()

                best_agent = max(pop, key=lambda a: a.e_i)
                convergence_hist[k].append(best_agent.e_i)

                # Handle sensitivity & mutation depending on algorithm type
                scale = np.ones(D)
                L = None
                ablate_flag = "-"

                if cfg["sens_type"] == "dynamic":
                    st = ec_state[k]
                    st["stats"] = (mean(fits), max(fits), std(fits))
                    is_modes = (env_type == "modes_ot")

                    if is_modes:
                        d_prev = 0.0 if st["E_prev"] is None else d_env_modes(E, st["E_prev"])
                        d_last = float("inf") if st["E_last"] is None else d_env_modes(E, st["E_last"])
                    else:
                        d_prev = 0.0 if st["E_prev"] is None else (1.0 - env_sim_73d(E, st["E_prev"]))
                        d_last = float("inf") if st["E_last"] is None else (1.0 - env_sim_73d(E, st["E_last"]))

                    st["R"] = RHO * st["R"] + d_prev
                    ablate = (st["E_last"] is None or d_last > TAU_E or st["R"] > TAU_R or (gen - st["t_last"]) >= B_MAX)

                    if ablate:
                        ablate_flag = "A"
                        ablation_update(pool, st["memory"], best_agent.params, opp, E, is_modes=is_modes)
                        st["E_last"] = E
                        st["t_last"] = gen
                        st["R"] = 0.0
                    st["E_prev"] = E

                    S = retrieve_sensitivity(st["memory"], E, is_modes=is_modes)
                    if S:
                        scale = sensitivity_to_scale(S)

                    # Covariance factor from strategy-interaction matrix
                    st["I"] = (1.0 - I_EMA) * st["I"] + I_EMA * interaction_observation(params, fits)
                    np.fill_diagonal(st["I"], 0.0)
                    L = covariance_factor(st["I"])

                # Formulate mutation closure
                if cfg["sens_type"] == "none":
                    def mutate_fn(parent):
                        sig = MU_BASE * (1.0 - parent.e_i) + EPS_DYNAMIC
                        z = rng.standard_normal(D)
                        return Agent(np.asarray(parent.params) + sig * z)
                elif cfg["sens_type"] == "static":
                    def mutate_fn(parent):
                        sig = np.array([pwm[j] * (1.0 - parent.e_i) + EPS_STATIC for j in range(D)])
                        z = rng.standard_normal(D)
                        return Agent(np.asarray(parent.params) + sig * z, pwm=pwm)
                else:
                    def mutate_fn(parent, L=L, scale=scale):
                        sig = (MU_BASE * (1.0 - parent.e_i) + EPS_DYNAMIC) * scale
                        z = rng.standard_normal(D)
                        step = sig * ((L @ z) if L is not None else z)
                        return Agent(np.asarray(parent.params) + step)

                extra = f" M={modes_count}" if modes_count is not None else (f" {ablate_flag}" if cfg["sens_type"] == "dynamic" else "")
                gen_log.append(f"[{cfg['name'][:7]:7s} e={best_agent.e_i:.2f}{extra}]")

                populations[k], archives[k] = select_and_reproduce(
                    pop, arch, mutate_fn, target_size=args.pop_size
                )
                champ_now = max(archives[k], key=lambda a: a.e_i) if archives[k] else best_agent
                new_best_weights[k] = champ_now.params

        current_best_weights = new_best_weights
        print("  " + " | ".join(gen_log))

    elapsed_evol = time.time() - t_evol_start
    final_walls = len(terrain_walls) + len(breakable_positions)
    print(f"\n  Dynamic evolution complete in {elapsed_evol:.1f}s.")
    print(f"  Final blockaded map: {final_walls} total walls ({len(terrain_walls)} terrain + {len(breakable_positions)} breakable).")

    # ──────────────────────────────────────────────────────────────
    #  PHASE 2: BOT VALIDATION & RANKING
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  PHASE 2: BOT VALIDATION — Validation vs All 7 microRTS Bots")
    print(f"  ({args.val_games} games per bot on the final blockaded 48×48 map)")
    print(f"{'='*80}")

    results = []
    for k in active_keys:
        name = ALL_VARIANTS[k]["name"]
        if ALL_VARIANTS[k]["is_cce"]:
            st = cce_state[k]
            champ_a = max(st["arch_a"], key=lambda a: a.e_i) if st["arch_a"] else max(st["pop_a"], key=lambda a: a.e_i)
            champ_b = max(st["arch_b"], key=lambda a: a.e_i) if st["arch_b"] else max(st["pop_b"], key=lambda a: a.e_i)
            champ_weights = combine_subagents(champ_a, champ_b)
        elif ALL_VARIANTS[k].get("is_baseline"):
            champ_weights = list(current_best_weights[k])
        else:
            champ = (max(archives[k], key=lambda a: a.e_i) if archives[k]
                     else max(populations[k], key=lambda a: a.e_i))
            champ_weights = list(champ.params)

        print(f"  Validating {name}...")
        wr, per_opp = pool.validate_all_bots(champ_weights, args.val_games)
        print(f"  -> {name}: {wr:.1f}% overall win rate")
        results.append({
            "name": name,
            "key": k,
            "wr": wr,
            "per_opp": per_opp,
            "weights": champ_weights,
            "convergence": convergence_hist[k],
            "time": elapsed_evol,
        })

    # Ranking Table
    ranked = sorted(results, key=lambda x: x["wr"], reverse=True)
    medals = [f"#{i}" for i in range(1, n_algos + 1)]

    print(f"\n{'='*72}")
    print(f"  RANKING — Overall Win Rate vs 7 Bots")
    print(f"{'='*72}")
    print(f"  {'Rank':<6s} {'Algorithm':<26s} {'Family':<12s} {'Win Rate':>10s} {'Time':>8s}")
    print(f"  {'_'*6} {'_'*26} {'_'*12} {'_'*10} {'_'*8}")
    for i, r in enumerate(ranked):
        fam = ALL_VARIANTS[r["key"]]["family"]
        print(f"  {medals[i]:<6s} {r['name']:<26s} {fam:<12s} {r['wr']:>9.1f}%  {r['time']:>7.1f}s")

    # Per-Bot Table
    print(f"\n{'='*72}")
    print(f"  PER-BOT WIN RATE MATRIX (%)")
    print(f"{'='*72}")
    hdr = f"  {'Bot':<22s}"
    for r in ranked:
        hdr += f" {r['name'][:10]:>11s}"
    print(hdr)
    print("  " + "─" * (22 + 12 * len(ranked)))
    for opp in OPPONENT_ROTATION:
        row = f"  {opp:<22s}"
        rates = [r["per_opp"].get(opp, 0.0) for r in ranked]
        mx = max(rates)
        for r in ranked:
            w = r["per_opp"].get(opp, 0.0)
            star = "*" if w == mx and rates.count(mx) == 1 else " "
            row += f" {w:>9.1f}%{star}"
        print(row)
    print("  " + "─" * (22 + 12 * len(ranked)))

    # ──────────────────────────────────────────────────────────────
    #  PHASE 3: HEAD-TO-HEAD TOURNAMENT MATRIX
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  PHASE 3: {n_algos}×{n_algos} HEAD-TO-HEAD TOURNAMENT MATRIX")
    print(f"  Each pair plays {args.h2h_games} games (alternating P0/P1) on final blockaded map")
    print(f"{'='*80}\n")

    algo_names = [r["name"] for r in results]
    algo_weights = [r["weights"] for r in results]

    matrix = run_head_to_head(pool, algo_weights, algo_names, args.h2h_games)

    print(f"\n{'='*80}")
    print(f"  HEAD-TO-HEAD RESULTS MATRIX ({n_algos}×{n_algos})")
    print(f"{'='*80}")
    h2h_lines, total_wins, h2h_win_rates = format_h2h_matrix(matrix, algo_names)
    for line in h2h_lines:
        print(line)

    # ──────────────────────────────────────────────────────────────
    #  PHASE 4: STATISTICAL ARTIFACTS & EXPORTS
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  GENERATING ARTIFACTS (Prefix: 'SEM7_ER_')")
    print(f"{'='*80}")

    # Top reference algorithm for Wilcoxon (EC-SA-Upgraded if present, else CCE-Upgraded, else highest ranked)
    ref_key = "ec_sa_up" if "ec_sa_up" in active_keys else ("cce_up" if "cce_up" in active_keys else ranked[0]["key"])
    ref_res = next(r for r in results if r["key"] == ref_key)

    # 1. Wilcoxon Signed-Rank Test Artifact: SEM7_ER_wilcoxon.txt
    wilcoxon_lines = [
        "Wilcoxon Signed-Rank Test Results",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Map: 48x48 Dynamic Blockades ({final_walls} total walls, MAX_CYCLES={MAX_CYCLES})",
        f"Generations: {args.generations}, Pop: {args.pop_size}, K_Games: {args.k_games}, H2H_Games: {args.h2h_games}",
        f"Reference Algorithm: {ref_res['name']}",
        "",
        f"{'Baseline':<26s}  {'p-value':>9s}  {'Significant?':>14s}  Direction",
        f"{'_'*26}  {'_'*9}  {'_'*14}  {'_'*24}",
    ]

    ref_rates = [ref_res["per_opp"].get(o, 0.0) for o in OPPONENT_ROTATION]
    for r in results:
        if r["key"] == ref_key:
            continue
        other_rates = [r["per_opp"].get(o, 0.0) for o in OPPONENT_ROTATION]
        p, n_s = wilcoxon_test(ref_rates, other_rates)
        if p is None:
            w_line = f"{r['name']:<26s}  {'N/A':>9s}  {'N/A':>14s}  — (all pairs tie)"
        else:
            sig = "YES (p<0.05)" if p < 0.05 else "NO  (p>=0.05)"
            dirn = (f"{ref_res['name'][:8]} BETTER" if sum(ref_rates) > sum(other_rates) else "BASELINE BETTER")
            warn = f" (n={n_s}, low power)" if n_s < 10 else f" (n={n_s})"
            w_line = f"{r['name']:<26s}  {p:>9.4f}  {sig:>14s}  {dirn}{warn}"
        wilcoxon_lines.append(w_line)
        print(f"  {w_line}")

    wilcoxon_lines.append("")
    wilcoxon_lines.append("=" * 64)
    wilcoxon_lines.append("Head-to-Head Tournament Summary")
    wilcoxon_lines.append("=" * 64)
    for i, name in enumerate(algo_names):
        wilcoxon_lines.append(f"{name:<26s}  Total Wins: {total_wins[i]:>3d}  H2H Win Rate: {h2h_win_rates[i]:>5.1f}%")

    wilcoxon_path = os.path.join(SCRIPT_DIR, "SEM7_ER_wilcoxon.txt")
    with open(wilcoxon_path, "w", encoding="utf-8") as f:
        f.write("\n".join(wilcoxon_lines) + "\n")
    print(f"  Saved -> {wilcoxon_path}")

    # 2. Ablation Analysis Artifact: SEM7_ER_ablation.txt
    res_by_key = {r["key"]: r for r in results}
    ablation_lines = [
        "Comprehensive Multi-Tier Ablation Analysis Results",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Map: 48x48 Dynamic Blockades ({final_walls} total walls, MAX_CYCLES={MAX_CYCLES})",
        f"Generations: {args.generations}, Pop: {args.pop_size}, K_Games: {args.k_games}, Suite: {args.suite.upper()}",
        "",
    ]

    def _add_comp(title, key_a, key_b):
        if key_a in res_by_key and key_b in res_by_key:
            ra, rb = res_by_key[key_a], res_by_key[key_b]
            delta = ra["wr"] - rb["wr"]
            verdict = "HELPS" if delta >= 0 else "HURTS"
            ablation_lines.extend([
                title,
                f"   {ra['name']}: {ra['wr']:.1f}% vs {rb['name']}: {rb['wr']:.1f}%",
                f"   Delta: {delta:+.1f}% | Verdict: {verdict} ({abs(delta):.1f}%)",
                "",
            ])

    # Multi-tier decomposed ablation across families
    _add_comp("1. Multi-Modal Trajectory Contribution (Pure TD-CCEA):", "td_up", "td")
    _add_comp("2. Kernel Sharpening Cutoff Control (Pure TD-CCEA):", "td_sharp", "td")
    _add_comp("3. Multi-Modal Trajectory vs Kernel Cutoff (Pure TD-CCEA):", "td_up", "td_sharp")
    _add_comp("4. Multi-Modal Trajectory Contribution in CCE (CCE-Upgraded vs CCE-Baseline):", "cce_up", "cce")
    _add_comp("5. Kernel Sharpening Cutoff Control in CCE (CCE+Sharp vs CCE-Baseline):", "cce_sharp", "cce")
    _add_comp("6. Multi-Modal Trajectory vs Kernel Cutoff in CCE (CCE-Upgraded vs CCE+Sharp):", "cce_up", "cce_sharp")
    _add_comp("7. Cooperative Decomposition Contribution (CCE-Baseline vs Monolithic TD):", "cce", "td")
    _add_comp("8. Cooperative Decomposition + Upgraded Trajectory Synergy (CCE-Upgraded vs TD-Upgraded):", "cce_up", "td_up")
    _add_comp("9. Static Sensitivity Adaptation Contribution (SA vs TD):", "sa", "td")
    _add_comp("10. Kernel Sharpening Cutoff Control on SA (SA+Sharp vs SA):", "sa_sharp", "sa")
    _add_comp("11. Dynamic Environment-Conditioned Sensitivity & Covariance (EC-SA vs SA):", "ec_sa", "sa")
    _add_comp("12. Multi-Modal Trajectory Synergy on Dynamic SA (EC-SA-Upgraded vs EC-SA):", "ec_sa_up", "ec_sa")
    _add_comp("13. Total Full System Architectural Leap (EC-SA-Upgraded vs Pure TD-CCEA Baseline):", "ec_sa_up", "td")
    _add_comp("14. Cross-Paradigm Peak Comparison (EC-SA-Upgraded vs CCE-TDCCEA-Upgraded):", "ec_sa_up", "cce_up")
    _add_comp("15. Trajectory Tracking in Cooperative Co-Evolution (CCE-Baseline vs Standard CCEA):", "cce", "std_ccea")
    _add_comp("16. Trajectory-Driven Search vs Covariance Matrix Adaptation (TD-CCEA vs CMA-ES):", "td", "cmaes")
    _add_comp("17. Trajectory-Driven Search vs Particle Swarm Optimization (TD-CCEA vs PSO):", "td", "pso")
    _add_comp("18. Evolutionary Search vs Stochastic Hill Climbing (TD-CCEA vs Hill Climbing):", "td", "hill")

    ablation_lines.extend([
        "=" * 64,
        f"Head-to-Head Tournament Results Matrix ({n_algos}x{n_algos})",
        "=" * 64,
    ])
    ablation_lines.extend(h2h_lines)

    ablation_path = os.path.join(SCRIPT_DIR, "SEM7_ER_ablation.txt")
    with open(ablation_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ablation_lines) + "\n")
    print(f"  Saved -> {ablation_path}")

    # 3. JSON Summary Artifact: SEM7_ER_results.json
    json_data = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "map": f"48x48 dynamic blockades ({final_walls} walls)",
            "generations": args.generations,
            "pop_size": args.pop_size,
            "k_games": args.k_games,
            "h2h_games": args.h2h_games,
            "validation_games": args.val_games,
            "max_cycles": MAX_CYCLES,
            "suite": args.suite,
            "evaluation_mode": args.eval,
            "tau0": args.tau0,
            "d_max": args.d_max,
            "dc_max": args.dc_max,
        },
        "vs_bots": [
            {
                "name": r["name"],
                "key": r["key"],
                "family": ALL_VARIANTS[r["key"]]["family"],
                "overall_winrate": r["wr"],
                "per_opponent": r["per_opp"],
                "time": r["time"],
                "weights": r["weights"],
            }
            for r in results
        ],
        "head_to_head": {
            "algorithms": algo_names,
            "matrix": [
                [
                    {"wins": matrix[i][j][0], "losses": matrix[i][j][1], "draws": matrix[i][j][2]}
                    if matrix[i][j] is not None else None
                    for j in range(len(algo_names))
                ]
                for i in range(len(algo_names))
            ],
            "total_wins": total_wins,
            "win_rates": h2h_win_rates,
        },
    }

    json_path = os.path.join(SCRIPT_DIR, "SEM7_ER_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved -> {json_path}")

    # 4. Excel Summary Artifact: SEM7_ER_comparison.xlsx
    excel_path = os.path.join(SCRIPT_DIR, "SEM7_ER_comparison.xlsx")
    export_excel(results, matrix, algo_names, total_wins, h2h_win_rates, excel_path)

    pool.stop()
    print("\n" + "═" * 80)
    print("  SEM7_ER BENCHMARK COMPLETED SUCCESSFULLY!")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    main()
