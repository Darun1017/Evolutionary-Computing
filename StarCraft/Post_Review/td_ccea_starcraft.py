#!/usr/bin/env python3
"""
TD-CCEA for StarCraft: Brood War  (Trajectory-Driven Competitive Co-Evolutionary Algorithm)
=========================================================================================

Your slides' TD-CCEA loop is kept intact:

    meta-state c_t -> fitness f -> trajectory memory H_i -> dynamic similarity weights alpha_k
    -> trajectory score e -> adaptive mutation mu_i = mu_base (1-e)+eps -> weak/redundant pruning
    -> Hall-of-Fame archive A (top-m, A_max) -> breed from survivors U A with Gaussian N(0, mu_i)

What changed to make it fit StarCraft (sources: Justesen & Risi, GECCO'17 "COEP";
Garcia-Sanchez et al., CIG'15 "StarCraftGP"):

 1. AGENT O_i = (build order, theta in [0,1]^20)                         [COEP + GP paper]
      - build order : fixed-length list of BuildTypes (COEP genome)
      - theta[0:8]  : MACRO / economy policy  (workers/base, supply buffer, gas workers,
                      expansion trigger, upgrade priority, bank reserve, scout time, production scale)
      - theta[8:20] : MICRO / combat parameters (6 unit-type weights, aggression, retreat,
                      harass, defend, focus-fire, kiting)  -> exactly the 8D / 12D split of CCE-TDCCEA
 2. FITNESS f_i^t = COEP's "discounted accumulated fitness" (Alg. 3) computed with a build-order
    FORWARD MODEL (Alg. 2) and the unit-matchup HEURISTIC (Tables 1-2), averaged over K simulated
    games against opponent strategies sampled from a belief over scripted archetypes (the tiers
    of the GP paper + COEP's rushes). Victory-tier win-rate is blended in (GP "victory-based fitness").
 3. ENVIRONMENT META-STATE c_t = mean_i theta_i  (your formula)  ++ opponent-composition fingerprint.
    In StarCraft the "Red Queen" drift is mostly the opponent's changing army (marines -> firebats),
    so the scouting fingerprint is appended (weight cfg.fingerprint_weight; set 0 for the pure slide formula).
 4. MUTATION: Gaussian N(0, mu_i) on theta; build-order genes use COEP's four operators
    (clone / swap / add-with-requirements / remove) whose activation probability is scaled by mu_i/mu_base
    (= COEP's tuned 50% when e=0).  Crossover: two-point on build orders (COEP), uniform on theta.
 5. ONLINE / CONTINUAL operation (COEP Alg. 1): game-state updates, champion transfer on restart,
    "first instance of a started build is removed from every genome", background thread runner.
 6. Four variants (your slide 3): td_ccea | sa_tdccea | cce_tdccea | standard_ccea.

HONEST CAVEATS
 * Build costs/times are standard BW values but approximate -> verify vs BWAPI UnitType.
 * Zerg/Protoss rows of the matchup table and Reaver/Carrier columns are ESTIMATES (COEP only
   publishes Terran-vs-Protoss, 4 columns). Firebat-vs-Dragoon uses 0.4 (the paper's text/worked
   example); the printed table shows 0.1.
 * Micro-theta effects are surrogate multipliers inside the heuristic. In a live bot, export them with
   Strategy.to_bot_config() and let the real combat module consume them; real results then feed
   ExternalGameEvaluator (GP-paper style) instead of the surrogate.

Usage:
    python td_ccea_starcraft.py --demo                    # online game demo (marines -> firebats drift)
    python td_ccea_starcraft.py --demo --variant sa_tdccea
    python td_ccea_starcraft.py --compare                 # benchmark all 4 variants
Requires: Python 3.9+, numpy.
"""
from __future__ import annotations

import argparse
import copy
import math
import random
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------------------------------
# 0. Configuration
# ----------------------------------------------------------------------------------------------
FPS = 24
MINE_SPEED = 0.05      # minerals / worker / frame   (COEP Sec 3.1)
GAS_SPEED = 0.07       # gas / worker / frame
SCOUT_FRAMES = 1200    # a worker is diverted for this long (macro theta[6])
DIM_MACRO, DIM_MICRO = 8, 12
DIM_THETA = DIM_MACRO + DIM_MICRO


@dataclass
class Config:
    variant: str = "td_ccea"
    # evolutionary loop
    pop_size: int = 32            # COEP used 64
    generations: int = 30         # generations per restart (COEP used 100)
    build_len: int = 57           # COEP genome length
    k_games: int = 4              # K simulated games per evaluation
    seed: int = 0
    # fitness (COEP Alg. 3): 8-min horizon, 2-min step, discount 0.9 (24 fps)
    horizon: int = 8 * 60 * FPS
    step_size: int = 2 * 60 * FPS
    discount: float = 0.9
    victory_blend: float = 0.5    # weight of tier-weighted win-rate (GP victory-based fitness)
    pen_expand: float = 1.0       # COEP expansion penalty weight
    pen_supply: float = 0.5       # surplus-pylon penalty weight
    # trajectory-driven part
    use_memory: bool = True
    window: int = 5               # W
    adaptive_mutation: bool = True
    mu_base: float = 0.15
    mu_fixed: float = 0.15        # Standard CCEA baseline
    eps: float = 0.01
    sensitivity_adaptive: bool = False   # SA-TDCCEA
    mu_min: float = 0.08
    mu_max: float = 0.20
    cooperative: bool = False            # CCE-TDCCEA (8D macro / 12D micro species)
    # pruning / archive
    delta_f: float = 0.15         # weak threshold on mean trajectory score
    delta_s: float = 0.95         # redundancy threshold on similarity S(i,j)
    min_age: int = 3              # |H_i| >= 3 before an agent may be pruned as weak
    archive_max: int = 15
    top_m: int = 3
    survival_min: float = 0.25    # COEP survival rate
    survival_max: float = 0.50
    p_op_ref: float = 0.5         # COEP operator activation prob at e = 0
    # environment
    opp_race: str = "Terran"
    fingerprint_weight: float = 1.0

    @classmethod
    def preset(cls, name: str, **kw) -> "Config":
        c = cls(variant=name, **kw)
        if name == "td_ccea":
            pass
        elif name == "sa_tdccea":
            c.sensitivity_adaptive = True
        elif name == "cce_tdccea":
            c.cooperative = True
        elif name == "standard_ccea":      # no trajectory memory, fixed mu = 0.15
            c.use_memory = False
            c.adaptive_mutation = False
        else:
            raise ValueError(f"unknown variant {name}")
        return c


VARIANTS = ["td_ccea", "sa_tdccea", "cce_tdccea", "standard_ccea"]

# ----------------------------------------------------------------------------------------------
# 1. StarCraft data (Protoss; approximate BW values)
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BuildType:
    name: str
    minerals: int
    gas: int
    supply: int            # supply cost (units)
    provides: int          # supply provided (pylon/nexus)
    frames: int            # build time in frames
    reqs: Tuple[str, ...]  # tech-tree requirements (incl. producer)
    producer: Optional[str]
    kind: str              # unit | building | tech
    max_count: int = 999


def _bt(*a, **k) -> BuildType:
    return BuildType(*a, **k)


_BUILDS = [
    _bt("Probe", 50, 0, 1, 0, 300, ("Nexus",), "Nexus", "unit"),
    _bt("Zealot", 100, 0, 2, 0, 600, ("Gateway",), "Gateway", "unit"),
    _bt("Dragoon", 125, 50, 2, 0, 750, ("Gateway", "Cybernetics_Core"), "Gateway", "unit"),
    _bt("Dark_Templar", 125, 100, 2, 0, 900, ("Gateway", "Templar_Archives"), "Gateway", "unit"),
    _bt("Reaver", 200, 100, 4, 0, 1050, ("Robotics_Facility", "Robotics_Support_Bay"), "Robotics_Facility", "unit"),
    _bt("Scout", 275, 125, 3, 0, 1200, ("Stargate",), "Stargate", "unit"),
    _bt("Carrier", 350, 250, 6, 0, 2100, ("Stargate", "Fleet_Beacon"), "Stargate", "unit"),
    _bt("Pylon", 100, 0, 0, 8, 450, (), None, "building", 12),
    _bt("Assimilator", 100, 0, 0, 0, 450, (), None, "building", 4),
    _bt("Gateway", 150, 0, 0, 0, 900, ("Pylon",), None, "building", 6),
    _bt("Nexus", 400, 0, 0, 9, 1800, (), None, "building", 3),
    _bt("Cybernetics_Core", 200, 0, 0, 0, 900, ("Gateway",), None, "building", 1),
    _bt("Citadel_of_Adun", 150, 100, 0, 0, 900, ("Cybernetics_Core",), None, "building", 1),
    _bt("Templar_Archives", 150, 200, 0, 0, 900, ("Citadel_of_Adun",), None, "building", 1),
    _bt("Robotics_Facility", 200, 200, 0, 0, 1200, ("Cybernetics_Core",), None, "building", 2),
    _bt("Robotics_Support_Bay", 150, 100, 0, 0, 450, ("Robotics_Facility",), None, "building", 1),
    _bt("Stargate", 150, 150, 0, 0, 1050, ("Cybernetics_Core",), None, "building", 3),
    _bt("Fleet_Beacon", 300, 200, 0, 0, 900, ("Stargate",), None, "building", 1),
    _bt("Singularity_Charge", 150, 150, 0, 0, 2500, ("Cybernetics_Core",), "Cybernetics_Core", "tech", 1),
]
B: Dict[str, BuildType] = {b.name: b for b in _BUILDS}
ALL_BUILDS = [b.name for b in _BUILDS]
OUR_UNITS = [b.name for b in _BUILDS if b.kind == "unit"]
COMBAT_UNITS = [u for u in OUR_UNITS if u != "Probe"]
PRODUCERS = {b.producer for b in _BUILDS if b.producer}
WORKERS = {"Probe", "SCV", "Drone"}
MICRO_UNITS = ["Zealot", "Dragoon", "Dark_Templar", "Reaver", "Scout", "Carrier"]   # theta[8:14]

# ---- Matchup table: enemy-unit value against OUR unit (COEP Table 1, range [0,2]) ----------------
_COLS = ["Zealot", "Dragoon", "Dark_Templar", "Scout", "Reaver", "Carrier"]
ENEMY_VS_PROTOSS: Dict[str, List[float]] = {
    # Terran rows 1-5: first 4 columns from the paper (Firebat/Dragoon = 0.4, see header). Rest ESTIMATED.
    "Marine": [0.7, 0.5, 0.6, 1.6, 0.5, 0.8],
    "Firebat": [1.3, 0.4, 0.8, 0.7, 0.6, 0.3],
    "Vulture": [1.6, 0.7, 1.6, 0.7, 0.8, 0.4],
    "Goliath": [0.9, 0.7, 0.9, 1.5, 0.8, 1.5],
    "Siege_Tank": [0.8, 1.4, 0.5, 0.9, 1.0, 0.7],
    "Wraith": [1.9, 0.6, 1.9, 1.0, 1.9, 0.8],            # estimated (air: ground melee cannot hit)
    "Battlecruiser": [1.9, 0.8, 1.9, 1.3, 1.9, 1.0],     # estimated
    # Zerg (ESTIMATED)
    "Zergling": [0.7, 1.2, 0.4, 0.1, 1.0, 0.1],
    "Hydralisk": [1.0, 0.8, 0.4, 1.4, 0.7, 1.1],
    "Mutalisk": [1.9, 0.5, 1.9, 1.0, 1.9, 0.7],
    "Lurker": [1.6, 1.2, 0.6, 0.1, 1.2, 0.1],
    "Ultralisk": [1.3, 1.4, 0.9, 0.1, 1.0, 0.1],
    "Scourge": [0.1, 0.1, 0.1, 1.6, 0.1, 1.2],
    # Protoss mirror (ESTIMATED); Zealot/Dragoon/Dark_Templar/Reaver/Scout/Carrier as enemy units
    "E_Zealot": [1.0, 0.7, 0.5, 0.1, 0.9, 0.1],
    "E_Dragoon": [1.3, 1.0, 0.9, 1.6, 1.1, 1.4],
    "E_Dark_Templar": [1.5, 1.3, 1.0, 0.1, 1.6, 0.1],
    "E_Reaver": [1.6, 1.3, 1.2, 0.1, 1.0, 0.1],
    "E_Scout": [0.1, 0.1, 0.1, 1.0, 0.1, 0.9],
    "E_Carrier": [1.6, 1.2, 1.5, 1.3, 1.3, 1.0],
}
# Upgrade multipliers (COEP Table 2) -- only Singularity Charge exists in our build list.
UPGRADE_MULT = {("Singularity_Charge", "Dragoon"): 1.25}


class MatchupTable:
    """matchup(x, y): value of attacker-type x against y; enemy-vs-ours from the table, ours-vs-enemy = 2 - table."""

    def __init__(self):
        self.m: Dict[Tuple[str, str], float] = {}
        for e, row in ENEMY_VS_PROTOSS.items():
            for col, v in zip(_COLS, row):
                self.m[(e, col)] = v
                self.m[(col, e)] = 2.0 - v
        self.m[("Probe", "Probe")] = 0.3

    def __call__(self, x: str, y: str) -> float:
        v = self.m.get((x, y))
        if v is not None:
            return v
        if x in WORKERS:
            return 0.3
        if y in WORKERS:
            return 1.5
        return 1.0


ENEMY_VOCAB = sorted({e.replace("E_", "") for e in ENEMY_VS_PROTOSS} | {"SCV", "Drone"})


def _enemy_key(race: str, unit: str) -> str:
    """Observed unit names for a Protoss enemy are stored with an E_ prefix inside the table."""
    return "E_" + unit if race == "Protoss" else unit


# ----------------------------------------------------------------------------------------------
# 2. Game state and build-order forward model (COEP Algorithm 2)
# ----------------------------------------------------------------------------------------------
_EFF = [0.0]
for _k in range(1, 121):                 # workers 1-10 full, 11-20 half, 21-30 a third ... (per base)
    _EFF.append(_EFF[-1] + 1.0 / math.ceil(_k / 10))


@dataclass
class GameState:
    frame: int = 0
    minerals: float = 50.0
    gas: float = 0.0
    supply_used: int = 4
    done: Dict[str, int] = field(default_factory=lambda: {"Nexus": 1, "Probe": 4, "Pylon": 1})
    under: List[List] = field(default_factory=list)            # [name, done_at]
    busy: Dict[str, List[int]] = field(default_factory=lambda: {"Nexus": [0]})
    gas_per_ass: int = 3
    scout: Tuple[int, int] = (-1, -1)

    def clone(self) -> "GameState":
        return GameState(self.frame, self.minerals, self.gas, self.supply_used, dict(self.done),
                         [list(u) for u in self.under], {k: list(v) for k, v in self.busy.items()},
                         self.gas_per_ass, self.scout)

    def supply_cap(self) -> int:
        return min(200, 9 * self.done.get("Nexus", 0) + 8 * self.done.get("Pylon", 0))

    def pending(self, name: str) -> int:
        return sum(1 for n, _ in self.under if n == name)

    def total(self, name: str) -> int:
        return self.done.get(name, 0) + self.pending(name)

    def gas_workers(self) -> int:
        return min(self.gas_per_ass * self.done.get("Assimilator", 0), max(0, self.done.get("Probe", 0) - 4))

    def mineral_workers(self) -> int:
        w = self.done.get("Probe", 0) - self.gas_workers()
        if self.scout[0] <= self.frame < self.scout[1]:
            w -= 1
        return max(0, w)

    def mineral_rate(self) -> float:
        bases = max(1, self.done.get("Nexus", 0))
        per = min(self.mineral_workers() / bases, 119.0)
        lo = int(per)
        eff = _EFF[lo] + (per - lo) * (_EFF[lo + 1] - _EFF[lo])
        return MINE_SPEED * eff * bases

    def gas_rate(self) -> float:
        return GAS_SPEED * self.gas_workers()

    def unit_counts(self) -> Dict[str, float]:
        return {u: float(self.done.get(u, 0)) for u in OUR_UNITS if self.done.get(u, 0) > 0}


def progress(s: GameState, to: int) -> GameState:
    """Advance state in place to frame `to`, integrating income between completion events."""
    while s.frame < to:
        t_end = to
        for _, d in s.under:
            if s.frame < d < t_end:
                t_end = d
        for b in s.scout:
            if s.frame < b < t_end:
                t_end = b
        dt = t_end - s.frame
        s.minerals += dt * s.mineral_rate()
        s.gas += dt * s.gas_rate()
        s.frame = t_end
        if s.under:
            keep = []
            for n, d in s.under:
                if d <= t_end:
                    s.done[n] = s.done.get(n, 0) + 1
                    if n in PRODUCERS:
                        s.busy.setdefault(n, []).append(d)
                else:
                    keep.append([n, d])
            s.under = keep
    return s


def start_build(s: GameState, bt: BuildType) -> None:
    s.under.append([bt.name, s.frame + bt.frames])
    s.minerals -= bt.minerals
    s.gas -= bt.gas
    s.supply_used += bt.supply
    if bt.producer:
        lst = s.busy[bt.producer]
        lst[lst.index(min(lst))] = s.frame + bt.frames


def advance_for(s: GameState, bt: BuildType, reserve: float = 0.0) -> Optional[GameState]:
    """PRODUCE_FRAME: returns a clone advanced to the earliest frame where `bt` can start
    (requirements, producer, supply, resources), or None if it can never be produced (illegal)."""
    t = s.frame
    for r in bt.reqs:
        if s.done.get(r, 0) > 0:
            continue
        ds = [d for n, d in s.under if n == r]
        if not ds:
            return None
        t = max(t, min(ds))
    if bt.producer:
        cand = [max(f, s.frame) for f in s.busy.get(bt.producer, ())]
        cand += [d for n, d in s.under if n == bt.producer]
        if not cand:
            return None
        t = max(t, min(cand))
    if bt.supply:
        need, cap = s.supply_used + bt.supply, s.supply_cap()
        if need > cap:
            for d, p in sorted((d, B[n].provides) for n, d in s.under if B[n].provides):
                cap += p
                if cap >= need:
                    t = max(t, d)
                    break
            else:
                return None
    ns = progress(s.clone(), t)
    for _ in range(4):
        dm, dg = bt.minerals + reserve - ns.minerals, bt.gas - ns.gas
        if dm <= 1e-9 and dg <= 1e-9:
            return ns
        rm, rg = ns.mineral_rate(), ns.gas_rate()
        if (dm > 1e-9 and rm <= 0) or (dg > 1e-9 and rg <= 0):
            return None
        dt = max(dm / rm if dm > 0 else 0.0, dg / rg if dg > 0 else 0.0)
        progress(ns, ns.frame + int(math.ceil(dt)))
    return None


# ----------------------------------------------------------------------------------------------
# 3. Macro policy (theta[0:8]) and micro multipliers (theta[8:20])
# ----------------------------------------------------------------------------------------------
@dataclass
class MacroPolicy:
    worker_target: int = 16      # workers per base
    supply_buffer: int = 6
    gas_per_ass: int = 3
    expand_trigger: float = 16.0  # workers/base that triggers a new Nexus
    upgrade: bool = False
    reserve: float = 0.0
    scout_frame: int = 3000
    gateways_per_base: int = 2

    @classmethod
    def from_theta(cls, t: np.ndarray) -> "MacroPolicy":
        return cls(worker_target=int(round(12 + 12 * t[0])), supply_buffer=int(round(2 + 12 * t[1])),
                   gas_per_ass=int(round(1 + 2 * t[2])), expand_trigger=10 + 10 * t[3], upgrade=bool(t[4] > 0.5),
                   reserve=300.0 * t[5], scout_frame=int(1000 + 5000 * t[6]),
                   gateways_per_base=int(round(1 + 3 * t[7])))

    def auto_build(self, s: GameState, blocked: set) -> Optional[str]:
        """Bot-level automatic production (as UAlbertaBot's production manager does); first match wins."""
        nex, nex_p = s.done.get("Nexus", 0), s.pending("Nexus")
        probes = s.done.get("Probe", 0)
        if ("Pylon" not in blocked and s.supply_cap() - s.supply_used < self.supply_buffer
                and s.pending("Pylon") == 0 and s.supply_cap() < 200):
            return "Pylon"
        if ("Probe" not in blocked and s.pending("Probe") < max(1, nex)
                and probes + s.pending("Probe") < min(self.worker_target * (nex + nex_p), 75)):
            return "Probe"
        if ("Assimilator" not in blocked and s.total("Gateway") > 0 and probes >= 8 and s.total("Assimilator") < 1):
            return "Assimilator"
        if ("Assimilator" not in blocked and probes >= 14 and s.total("Assimilator") < 2 and
                any(s.total(x) for x in ("Robotics_Facility", "Stargate", "Templar_Archives", "Citadel_of_Adun"))):
            return "Assimilator"
        if ("Nexus" not in blocked and nex_p == 0 and nex < 3 and probes / max(1, nex) >= self.expand_trigger):
            return "Nexus"
        if ("Singularity_Charge" not in blocked and self.upgrade and s.done.get("Cybernetics_Core", 0) > 0
                and s.total("Singularity_Charge") == 0):
            return "Singularity_Charge"
        if ("Gateway" not in blocked and s.done.get("Cybernetics_Core", 0) > 0
                and s.total("Gateway") < self.gateways_per_base * nex):
            return "Gateway"
        return None


def micro_multipliers(tm: np.ndarray) -> Dict[str, float]:
    """12D combat parameters -> per-unit value multipliers used by the heuristic (surrogate)."""
    m = {u: 0.75 + 0.5 * tm[i] for i, u in enumerate(MICRO_UNITS)}
    aggression, retreat, harass, defend, focus, kiting = tm[6:12]
    for u in ("Zealot", "Dark_Templar"):
        m[u] *= 0.9 + 0.2 * aggression
    for u in ("Dark_Templar", "Scout", "Carrier"):
        m[u] *= 0.9 + 0.2 * harass
    for u in ("Dragoon", "Reaver"):
        m[u] *= 0.9 + 0.2 * kiting
    g = (1 + 0.10 * retreat) * (0.95 + 0.10 * defend) * (0.95 + 0.10 * focus)
    return {u: v * g for u, v in m.items()}


@dataclass
class Strategy:
    """A complete, assembled solution (what is actually evaluated / sent to the bot)."""
    build: List[str]
    theta: np.ndarray

    @property
    def macro(self) -> MacroPolicy:
        return MacroPolicy.from_theta(self.theta[:DIM_MACRO])

    @property
    def micro(self) -> Dict[str, float]:
        return micro_multipliers(self.theta[DIM_MACRO:])

    def to_bot_config(self) -> dict:
        """JSON-ready dict, in the spirit of UAlbertaBot's scripted OpeningBuildOrder config."""
        m = self.macro
        return {"Race": "Protoss", "OpeningBuildOrder": list(self.build),
                "Macro": {k: getattr(m, k) for k in m.__dataclass_fields__},
                "MicroUnitWeights": {k: round(v, 3) for k, v in self.micro.items()},
                "CombatParams": dict(zip(["aggression", "retreat", "harass", "defend", "focus_fire", "kiting"],
                                         [round(float(x), 3) for x in self.theta[DIM_MACRO + 6:]]))}


def predict(s: GameState, order: Sequence[str], end: int, pol: Optional[MacroPolicy], idx: int = 0,
            log: Optional[list] = None) -> Tuple[GameState, int]:
    """PREDICT (COEP Alg. 2): produce the builds of `order` (from index idx) in `s` until frame `end`.
    Illegal builds are ignored; policy auto-builds (probes, pylons, expansions ...) are interleaved."""
    if pol:
        s.gas_per_ass = pol.gas_per_ass
        if s.scout[0] < 0 and pol.scout_frame >= s.frame:
            s.scout = (pol.scout_frame, pol.scout_frame + SCOUT_FRAMES)
    blocked: set = set()
    for _ in range(600):
        if s.frame >= end:
            break
        auto = pol.auto_build(s, blocked) if pol else None
        if auto:
            name = auto
        elif idx < len(order):
            name = order[idx]
        else:
            break
        bt = B[name]
        free = bt.name in ("Probe", "Pylon", "Nexus", "Assimilator")
        ns = advance_for(s, bt, 0.0 if (free or not pol) else pol.reserve)
        if ns is None:
            if auto:
                blocked.add(name)
            else:
                idx += 1
            continue
        if ns.frame > end:
            break
        start_build(ns, bt)
        if log is not None:
            log.append((name, ns.frame, bool(auto)))
        s = ns
        if not auto:
            idx += 1
    progress(s, end)
    return s, idx


# ----------------------------------------------------------------------------------------------
# 4. Opponent model: archetypes (GP-paper tiers + COEP rushes), belief, K sampled futures
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Archetype:
    race: str
    name: str
    tier: int                    # 1 easy, 2 medium, 3 hard (GP paper)
    comp: Tuple[Tuple[str, float], ...]
    rate: float                  # combat units per minute once active
    start: int                   # frame at which army production ramps up


def _a(race, name, tier, comp, rate, start):
    return Archetype(race, name, tier, tuple(comp.items()), rate, start)


ARCHETYPES = [
    _a("Terran", "Dummy", 1, {"Marine": 1.0}, 1.0, 5000),
    _a("Terran", "Defensive", 2, {"Marine": .5, "Siege_Tank": .3, "Firebat": .2}, 3.0, 6000),
    _a("Terran", "MarineRush", 3, {"Marine": 1.0}, 8.0, 2400),
    _a("Terran", "BioPush", 3, {"Marine": .5, "Firebat": .5}, 6.0, 4000),
    _a("Terran", "MechPush", 2, {"Vulture": .3, "Goliath": .3, "Siege_Tank": .4}, 4.0, 5000),
    _a("Terran", "WraithHarass", 3, {"Wraith": .7, "Battlecruiser": .3}, 2.5, 4500),
    _a("Zerg", "HydraMuta", 1, {"Hydralisk": .5, "Mutalisk": .5}, 4.0, 5500),
    _a("Zerg", "LurkerRush", 3, {"Zergling": .3, "Hydralisk": .3, "Lurker": .4}, 5.0, 5000),
    _a("Zerg", "ZerglingRush", 3, {"Zergling": 1.0}, 12.0, 2200),
    _a("Zerg", "UltraSwarm", 2, {"Ultralisk": .4, "Zergling": .6}, 5.0, 7000),
    _a("Protoss", "ZealotRush", 3, {"E_Zealot": 1.0}, 6.0, 2000),
    _a("Protoss", "DragoonRush", 3, {"E_Dragoon": 1.0}, 4.0, 3500),
    _a("Protoss", "DTRush", 3, {"E_Dark_Templar": 1.0}, 3.0, 4000),
    _a("Protoss", "ReaverDrop", 2, {"E_Zealot": .5, "E_Reaver": .5}, 3.0, 5500),
    _a("Protoss", "Defensive", 1, {"E_Dragoon": 1.0}, 3.0, 7000),
]
TIER_WEIGHT = {1: 1.0, 2: 2.0, 3: 4.0}
ENEMY_WORKER = {"Terran": "SCV", "Zerg": "Drone", "Protoss": "Probe"}


@dataclass
class OppSample:
    base: Dict[str, float]
    arche: Archetype
    tempo: float
    start: int
    t_obs: int
    race: str

    def counts_at(self, frame: int) -> Dict[str, float]:
        out = dict(self.base)
        dt = max(0, frame - max(self.t_obs, self.start))
        n = self.arche.rate * self.tempo * dt / (60 * FPS)
        for u, w in self.arche.comp:
            out[u] = out.get(u, 0.0) + n * w
        wk = ENEMY_WORKER[self.race]
        out[wk] = max(out.get(wk, 0.0), min(40.0, 4 + frame / 450.0))
        return out


class OpponentModel:
    """Belief over archetypes from scouting (a light stand-in for Synnaeve & Bessiere's Bayesian model)."""

    def __init__(self, race: str):
        self.race = race
        self.arch = [a for a in ARCHETYPES if a.race == race]
        self.belief = np.ones(len(self.arch)) / len(self.arch)
        self.observed: Dict[str, float] = {}
        self.frame = 0

    def update(self, observed: Dict[str, float], frame: int) -> None:
        self.observed, self.frame = dict(observed), frame
        army = {u: n for u, n in observed.items() if u not in WORKERS and n > 0}
        if not army:
            return
        keys = sorted(set(army) | {u for a in self.arch for u, _ in a.comp})
        v = np.array([army.get(k, 0.0) for k in keys])
        like = []
        for a in self.arch:
            c = dict(a.comp)
            w = np.array([c.get(k, 0.0) for k in keys])
            cos = float(v @ w / (np.linalg.norm(v) * np.linalg.norm(w) + 1e-9))
            expected = a.rate * max(0, frame - a.start) / (60 * FPS)       # tempo consistency
            seen = float(sum(army.values()))
            tempo = math.exp(-abs(math.log((seen + 1.0) / (expected + 1.0))))
            like.append((cos + 0.05) ** 3 * (0.2 + tempo))
        post = self.belief * np.array(like)
        post = post / post.sum()
        self.belief = 0.9 * post + 0.1 / len(self.arch)

    def sample(self, k: int, rng: random.Random) -> List[OppSample]:
        out = []
        for _ in range(k):
            a = rng.choices(self.arch, weights=list(self.belief))[0]
            out.append(OppSample(self.observed, a, math.exp(rng.gauss(0, 0.25)),
                                 int(a.start * rng.uniform(0.9, 1.1)), self.frame, self.race))
        return out

    def fingerprint(self) -> np.ndarray:
        vec = np.zeros(len(ENEMY_VOCAB))
        src = {u.replace("E_", ""): n for u, n in self.observed.items() if u not in WORKERS and n > 0}
        if not src:                                  # nothing scouted yet: belief-weighted expectation
            for p, a in zip(self.belief, self.arch):
                for u, w in a.comp:
                    src[u.replace("E_", "")] = src.get(u.replace("E_", ""), 0.0) + p * w
        for u, n in src.items():
            if u in ENEMY_VOCAB:
                vec[ENEMY_VOCAB.index(u)] = n
        s = vec.sum()
        return vec / s if s > 0 else vec


# ----------------------------------------------------------------------------------------------
# 5. Heuristic (COEP Sec 3.2) and discounted accumulated fitness (Alg. 3)
# ----------------------------------------------------------------------------------------------
def side_value(me: Dict[str, float], opp: Dict[str, float], mt: MatchupTable,
               mods: Dict[str, float]) -> float:
    """sum_x sum_y matchup(x,y)*n(x)*n(y)*(1 - n(x)/N(p)); n(x) halved in the last factor for workers."""
    N = sum(me.values())
    if N <= 0:
        return 0.0
    tot = 0.0
    for x, nx in me.items():
        if nx <= 0:
            continue
        share = (nx / 2 if x in WORKERS else nx) / N
        mod = mods.get(x, 1.0)
        for y, ny in opp.items():
            if ny > 0:
                tot += mt(x, y) * mod * nx * ny * (1.0 - share)
    return tot


def heuristic(s: GameState, opp: Dict[str, float], mt: MatchupTable, mods: Dict[str, float],
              cfg: Config, race: str) -> float:
    mine = s.unit_counts()
    mods = dict(mods)
    if s.done.get("Singularity_Charge", 0):
        mods["Dragoon"] = mods.get("Dragoon", 1.0) * UPGRADE_MULT[("Singularity_Charge", "Dragoon")]
    theirs = {_enemy_key(race, u) if u in ENEMY_VOCAB else u: n for u, n in opp.items()}
    h = side_value(mine, theirs, mt, mods) - side_value(theirs, mine, mt, {})
    # penalties from COEP: expansion vs. worker count, surplus supply buildings
    h -= cfg.pen_expand * abs(s.done.get("Nexus", 0) * 14 - s.mineral_workers())
    h -= cfg.pen_supply * max(0, s.supply_cap() - s.supply_used - 16)
    return h


def discounted_fitness(st: Strategy, s0: GameState, opp: OppSample, cfg: Config,
                       mt: MatchupTable) -> Tuple[float, bool]:
    """COEP Alg. 3: accumulate discount^step * heuristic(state) every `step_size` frames over the horizon."""
    s = s0.clone()
    pol, mods = st.macro, st.micro
    t_end, idx, step, total, last = s0.frame + cfg.horizon, 0, 0, 0.0, 0.0
    while s.frame < t_end:
        nxt = min(t_end, s.frame + cfg.step_size)
        s, idx = predict(s, st.build, nxt, pol, idx)
        last = heuristic(s, opp.counts_at(s.frame), mt, mods, cfg, opp.race)
        total += last * (cfg.discount ** step)
        step += 1
    return total, last > 0


# ----------------------------------------------------------------------------------------------
# 6. Individuals, species, genetic operators
# ----------------------------------------------------------------------------------------------
@dataclass
class Individual:
    uid: int
    build: List[str]
    theta: np.ndarray                      # sub-vector owned by the species
    hist: List[Tuple[float, np.ndarray]] = field(default_factory=list)   # H_i = [(f_k, c_k)]
    e_hist: List[float] = field(default_factory=list)
    f_raw: float = 0.0
    f: float = 0.0
    e: float = 0.0
    mu: float = 0.15
    mu_vec: Optional[np.ndarray] = None
    parent_theta: Optional[np.ndarray] = None
    parent_f: Optional[float] = None


@dataclass
class Species:
    idx: int
    lo: int
    hi: int
    has_build: bool
    pop: List[Individual] = field(default_factory=list)
    archive: List[Individual] = field(default_factory=list)
    champion: Optional[Individual] = None
    sens: Optional[np.ndarray] = None      # per-gene sensitivity (SA-TDCCEA)

    @property
    def dims(self) -> int:
        return self.hi - self.lo


def similarity(a: Individual, b: Individual) -> float:
    """S(i,j) in [0,1]: theta distance (normalised) and build-order unit-histogram overlap."""
    d_theta = float(np.linalg.norm(a.theta - b.theta)) / math.sqrt(max(1, len(a.theta)))
    s = 1.0 - d_theta
    if a.build and b.build:
        ca, cb = Counter(a.build), Counter(b.build)
        tot = sum(ca.values()) + sum(cb.values())
        l1 = sum(abs(ca[k] - cb[k]) for k in set(ca) | set(cb))
        s = 0.5 * s + 0.5 * (1.0 - l1 / max(1, tot))
    return s


def insert_with_requirements(prefix: List[str], name: str, have: Counter) -> List[str]:
    """COEP 'Add' mutation: insert `name`; every unmet requirement is recursively inserted in front."""
    out = list(prefix)
    for r in B[name].reqs:
        if have[r] == 0:
            out = insert_with_requirements(out, r, have)
    out.append(name)
    have[name] += 1
    return out


class TDCCEA:
    """Online TD-CCEA planner (COEP-style): call update_state(), then step_generation() repeatedly;
    request_build() returns the champion's next build and removes it from every genome."""

    def __init__(self, cfg: Config, state: GameState, opp: Optional[OpponentModel] = None):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.np_rng = np.random.RandomState(cfg.seed)
        self.mt = MatchupTable()
        self.state = state.clone()
        self.opp = opp or OpponentModel(cfg.opp_race)
        self.lock = threading.RLock()
        self._uid = 0
        self.generation = 0
        self.samples: List[OppSample] = []
        self.history: List[dict] = []
        if cfg.cooperative:
            self.species = [Species(0, 0, DIM_MACRO, True), Species(1, DIM_MACRO, DIM_THETA, False)]
        else:
            self.species = [Species(0, 0, DIM_THETA, True)]
        for sp in self.species:
            sp.sens = np.full(sp.dims, 0.5)
        self.ref_theta = np.full(DIM_THETA, 0.5)
        self.d_max = math.sqrt(DIM_THETA + 2.0 * cfg.fingerprint_weight ** 2)
        self.restart()

    # ---------------------------------------------------------------- state / observation
    def update_state(self, state: GameState, observed_enemy: Dict[str, float]) -> None:
        with self.lock:
            self.state = state.clone()
            self.opp.update(observed_enemy, state.frame)

    # ---------------------------------------------------------------- population creation
    def _unit_prefs(self) -> List[float]:
        return [0.3 + float(self.ref_theta[DIM_MACRO + i]) for i in range(len(MICRO_UNITS))]

    def _random_build_order(self, length: int, prefix: Sequence[str] = ()) -> List[str]:
        have = Counter({k: v for k, v in self.state.done.items()})
        for n, _ in self.state.under:
            have[n] += 1
        order = list(prefix)
        for n in order:
            have[n] += 1
        prefs = dict(zip(MICRO_UNITS, self._unit_prefs()))
        while len(order) < length:
            cands, w = [], []
            for n in ALL_BUILDS:
                b = B[n]
                if have[n] >= b.max_count and b.kind != "unit":
                    continue
                if any(have[r] == 0 for r in b.reqs):
                    continue
                cands.append(n)
                w.append(4.0 * prefs.get(n, 1.0) if b.kind == "unit" and n != "Probe" else (2.0 if n == "Probe" else 1.0))
            n = self.rng.choices(cands, weights=w)[0]
            order.append(n)
            have[n] += 1
        return order

    def _new_individual(self, sp: Species, theta: Optional[np.ndarray] = None) -> Individual:
        self._uid += 1
        th = self.np_rng.uniform(0, 1, sp.dims) if theta is None else np.clip(theta, 0, 1)
        build = self._random_build_order(self.cfg.build_len) if sp.has_build else []
        return Individual(self._uid, build, th, mu=self.cfg.mu_base)

    def restart(self) -> None:
        """New population each loop (COEP). The champion is carried over unchanged; the archive persists.
        theta is warm-started around the champion (macro/micro parameters are not game-state specific)."""
        with self.lock:
            for sp in self.species:
                champ = sp.champion
                pop = []
                if champ is not None:
                    c = copy.deepcopy(champ)
                    if sp.has_build:
                        c.build = (c.build + self._random_build_order(self.cfg.build_len, c.build)[len(c.build):])
                    pop.append(c)
                while len(pop) < self.cfg.pop_size:
                    th = None if champ is None else champ.theta + self.np_rng.normal(0, 0.2, sp.dims)
                    pop.append(self._new_individual(sp, th))
                sp.pop = pop
                for a in sp.archive:     # keep archive genomes full-length after consumed builds
                    if sp.has_build and len(a.build) < self.cfg.build_len:
                        a.build = self._random_build_order(self.cfg.build_len, a.build)
            self.generation = 0

    # ---------------------------------------------------------------- assembling / evaluating
    def assemble(self, parts: Dict[int, Individual]) -> Strategy:
        theta = np.full(DIM_THETA, 0.5)
        build: List[str] = []
        for si, ind in parts.items():
            sp = self.species[si]
            theta[sp.lo:sp.hi] = ind.theta
            if sp.has_build:
                build = ind.build
        return Strategy(build, theta)

    def _collaborators(self, si: int) -> List[Dict[int, Individual]]:
        """Cooperative evaluation: best representative + one random member of every other species."""
        if not self.cfg.cooperative:
            return [{}]
        out_best: Dict[int, Individual] = {}
        out_rand: Dict[int, Individual] = {}
        for j, sp in enumerate(self.species):
            if j == si:
                continue
            out_best[j] = sp.champion or sp.pop[0]
            out_rand[j] = self.rng.choice(sp.pop)
        return [out_best, out_rand]

    def _score(self, st: Strategy) -> float:
        tot, wr, ws = 0.0, 0.0, 0.0
        for smp in self.samples:
            sc, win = discounted_fitness(st, self.state, smp, self.cfg, self.mt)
            w = TIER_WEIGHT[smp.arche.tier]
            tot, wr, ws = tot + sc, wr + w * win, ws + w
        mean = tot / len(self.samples)
        return mean + self.cfg.victory_blend * max(1.0, abs(mean)) * (wr / ws - 0.5)

    def meta_state(self) -> np.ndarray:
        """c_t = (1/N) sum_i theta_i^t   (assembled over species)   ++ w * opponent fingerprint."""
        c = np.full(DIM_THETA, 0.5)
        for sp in self.species:
            c[sp.lo:sp.hi] = np.mean([i.theta for i in sp.pop], axis=0)
        if self.cfg.fingerprint_weight > 0:
            c = np.concatenate([c, self.cfg.fingerprint_weight * self.opp.fingerprint()])
        return c

    # ---------------------------------------------------------------- trajectory-driven scoring
    def _trajectory_score(self, ind: Individual, c_t: np.ndarray) -> float:
        """e_i^t = sum_k alpha_hat_k f^(k),  alpha_k = 1 - d(c_t,c_k)/d_max,  alpha_hat = alpha / sum alpha."""
        if not self.cfg.use_memory:
            return ind.f
        al = np.array([max(0.0, 1.0 - float(np.linalg.norm(c_t - ck)) / self.d_max) for _, ck in ind.hist])
        fs = np.array([f for f, _ in ind.hist])
        if al.sum() <= 1e-9:
            return float(fs[-1])
        return float((al / al.sum()) @ fs)

    def _mutation_rates(self, ind: Individual, sp: Species) -> None:
        cfg = self.cfg
        mu = cfg.mu_base * (1.0 - ind.e) + cfg.eps if cfg.adaptive_mutation else cfg.mu_fixed
        ind.mu = mu
        if cfg.sensitivity_adaptive:
            s = sp.sens / (sp.sens.max() + 1e-9)
            ind.mu_vec = np.clip(mu * 2.0 * (1.0 - s), cfg.mu_min, cfg.mu_max)
        else:
            ind.mu_vec = np.full(sp.dims, mu)

    def _update_sensitivity(self, sp: Species) -> None:
        for ind in sp.pop:
            if ind.parent_theta is None or ind.parent_f is None:
                continue
            dth = np.abs(ind.theta - ind.parent_theta)
            if dth.sum() < 1e-9:
                continue
            sp.sens = 0.9 * sp.sens + 0.1 * abs(ind.f - ind.parent_f) * dth / dth.sum() * sp.dims

    # ---------------------------------------------------------------- one generation
    def step_generation(self) -> dict:
        with self.lock:
            cfg = self.cfg
            self.samples = self.opp.sample(cfg.k_games, self.rng)           # K simulated opponents
            c_t = self.meta_state()                                         # environment meta-state
            for si, sp in enumerate(self.species):
                pool = sp.pop + sp.archive
                for ind in pool:                                            # evaluate fitness via K games
                    collabs = self._collaborators(si)
                    ind.f_raw = max(self._score(self.assemble({**c, si: ind})) for c in collabs)
                fr = np.array([i.f_raw for i in pool])
                lo, hi = fr.min(), fr.max()
                for ind in pool:
                    ind.f = float((ind.f_raw - lo) / (hi - lo)) if hi - lo > 1e-9 else 0.5
                    ind.hist.append((ind.f, c_t.copy()))                    # update trajectory memory
                    del ind.hist[:-cfg.window]
                    ind.e = self._trajectory_score(ind, c_t)
                    ind.e_hist.append(ind.e)
                    del ind.e_hist[:-cfg.window]
                    self._mutation_rates(ind, sp)                           # adaptive mutation rate
                sp.champion = copy.deepcopy(max(pool, key=lambda i: i.e))
                if cfg.sensitivity_adaptive:
                    self._update_sensitivity(sp)
            self.ref_theta = self.assemble({i: sp.champion for i, sp in enumerate(self.species)}).theta
            for sp in self.species:
                self._select_and_breed(sp)
            self.generation += 1
            rec = {"gen": self.generation, "best_f_raw": max(i.f_raw for i in self.species[0].pop + self.species[0].archive),
                   "mean_mu": float(np.mean([i.mu for i in self.species[0].pop])) if self.species[0].pop else 0.0,
                   "archive": len(self.species[0].archive)}
            self.history.append(rec)
            return rec

    # ---------------------------------------------------------------- pruning, archive, reproduction
    def _select_and_breed(self, sp: Species) -> None:
        cfg, n = self.cfg, self.cfg.pop_size
        evaluated = [i for i in sp.pop if i.hist]
        ranked = sorted(evaluated, key=lambda i: i.e, reverse=True)
        bar = lambda i: float(np.mean(i.e_hist))
        # (a) remove weak agents: e_bar < delta_f and |H| >= 3
        kept = [i for i in ranked if not (bar(i) < cfg.delta_f and len(i.hist) >= cfg.min_age)]
        # (b) remove redundant agents: S(i,j) > delta_s (the better-ranked one survives)
        uniq: List[Individual] = []
        for i in kept:
            if all(similarity(i, j) <= cfg.delta_s for j in uniq):
                uniq.append(i)
        # (c) COEP survival band
        lo_n, hi_n = max(2, math.ceil(cfg.survival_min * n)), max(2, math.ceil(cfg.survival_max * n))
        if len(uniq) < lo_n:
            rest = [i for i in ranked if i not in uniq]
            uniq += rest[:lo_n - len(uniq)]
        survivors = uniq[:hi_n]
        # (d) Hall-of-Fame: insert top-m survivors, dedupe, trim to A_max
        for top in survivors[:cfg.top_m]:
            twin = next((a for a in sp.archive if similarity(a, top) > cfg.delta_s), None)
            if twin is not None:
                sp.archive.remove(twin)
            sp.archive.append(copy.deepcopy(top))
        sp.archive.sort(key=lambda i: i.e, reverse=True)
        del sp.archive[cfg.archive_max:]
        # (e) breed from survivors U A with Gaussian mutation N(0, mu_i)
        parents = survivors + sp.archive
        children: List[Individual] = []
        while len(survivors) + len(children) < n:
            pa, pb = self._tournament(parents), self._tournament(parents)
            children.append(self._offspring(sp, pa, pb))
        sp.pop = survivors + children

    def _tournament(self, pool: List[Individual], k: int = 2) -> Individual:
        return max(self.rng.sample(pool, min(k, len(pool))), key=lambda i: i.e)

    def _offspring(self, sp: Species, pa: Individual, pb: Individual) -> Individual:
        cfg = self.cfg
        self._uid += 1
        mask = self.np_rng.rand(sp.dims) < 0.5                              # uniform crossover on theta
        theta = np.where(mask, pa.theta, pb.theta)
        sigma = pa.mu_vec if pa.mu_vec is not None else np.full(sp.dims, cfg.mu_base)
        theta = np.clip(theta + self.np_rng.normal(0, 1, sp.dims) * sigma, 0.0, 1.0)   # N(0, mu_i)
        build: List[str] = []
        if sp.has_build:
            build = self._crossover(pa.build, pb.build)
            p_op = max(0.05, cfg.p_op_ref * float(np.mean(sigma)) / cfg.mu_base)
            p_op = min(p_op, 0.9)
            if self.rng.random() < p_op:
                build = self._m_clone(build)
            if self.rng.random() < p_op:
                build = self._m_swap(build)
            if self.rng.random() < p_op:
                build = self._m_add(build)
            if self.rng.random() < p_op:
                build = self._m_remove(build)
        return Individual(self._uid, build, theta, parent_theta=pa.theta.copy(), parent_f=pa.f, mu=pa.mu)

    # --- COEP genetic operators on build orders
    def _crossover(self, a: List[str], b: List[str]) -> List[str]:
        L = min(len(a), len(b))
        if L < 3:
            return list(a)
        p, q = sorted(self.rng.sample(range(1, L), 2))
        return a[:p] + b[p:q] + a[q:]

    def _m_clone(self, o: List[str]) -> List[str]:
        o = list(o)
        if len(o) > 1:
            i, j = self.rng.sample(range(len(o)), 2)
            o[i] = o[j]
        return o

    def _m_swap(self, o: List[str]) -> List[str]:
        o = list(o)
        if len(o) > 1:
            i, j = self.rng.sample(range(len(o)), 2)
            o[i], o[j] = o[j], o[i]
        return o

    def _m_add(self, o: List[str]) -> List[str]:
        prefs = dict(zip(MICRO_UNITS, self._unit_prefs()))
        if self.rng.random() < 0.7:
            name = self.rng.choices(MICRO_UNITS, weights=[prefs[u] for u in MICRO_UNITS])[0]
        else:
            name = self.rng.choice(ALL_BUILDS)
        idx = self.rng.randrange(len(o) + 1)
        have = Counter(self.state.done)
        for n, _ in self.state.under:
            have[n] += 1
        for n in o[:idx]:
            have[n] += 1
        new = insert_with_requirements(o[:idx], name, have) + o[idx:]
        return new[:max(len(o), 1)]                 # fixed length

    def _m_remove(self, o: List[str]) -> List[str]:
        o = list(o)
        if o:
            o.append(o.pop(self.rng.randrange(len(o))))     # moved to the end, rest slide forward
        return o

    # ---------------------------------------------------------------- bot interface (COEP Sec 3.3)
    def champion_strategy(self) -> Strategy:
        with self.lock:
            return self.assemble({i: (sp.champion or sp.pop[0]) for i, sp in enumerate(self.species)})

    def peek_build(self) -> Optional[str]:
        b = self.champion_strategy().build
        return b[0] if b else None

    def consume_build(self, name: str) -> None:
        """A build went into production: remove its first instance from every genome."""
        with self.lock:
            for sp in self.species:
                if not sp.has_build:
                    continue
                for ind in sp.pop + sp.archive + ([sp.champion] if sp.champion else []):
                    if name in ind.build:
                        ind.build.remove(name)

    def request_build(self) -> Optional[str]:
        with self.lock:
            name = self.peek_build()
            if name:
                self.consume_build(name)
            return name

    def run(self, generations: Optional[int] = None) -> None:
        for _ in range(generations or self.cfg.generations):
            self.step_generation()


class ContinualRunner:
    """Runs restart + G generations forever in a background thread (COEP's parallel loop)."""

    def __init__(self, planner: TDCCEA):
        self.p, self._stop = planner, threading.Event()
        self.t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.p.restart()
            for _ in range(self.p.cfg.generations):
                if self._stop.is_set():
                    return
                self.p.step_generation()

    def start(self): self.t.start(); return self
    def stop(self): self._stop.set(); self.t.join(timeout=5)


# ----------------------------------------------------------------------------------------------
# 7. Hooks for real-game evaluation (GP-paper style) -- not executable without BWAPI
# ----------------------------------------------------------------------------------------------
@dataclass
class GameReport:
    """GP paper's report-based fitness components (lexicographic there; scalarised here for TD memory)."""
    military_victories: float = 0
    economic_victories: float = 0
    relative_destruction: float = 1.0
    time_to_loss: float = 0.0
    relative_economy: float = 1.0

    def scalar(self) -> float:
        return (4.0 * self.military_victories + 2.0 * self.economic_victories + self.relative_destruction
                + 0.0005 * self.time_to_loss + 0.5 * self.relative_economy)


class ExternalGameEvaluator:
    """Replace the surrogate: compile/export Strategy.to_bot_config(), play K games in BWAPI
    (e.g. TournamentManager on VMs, as in StarCraftGP) and return a GameReport."""

    def evaluate(self, strategy: Strategy, opponents: Sequence[str], k: int) -> GameReport:
        raise NotImplementedError("Wire this to BWAPI / TournamentManager.")


# ----------------------------------------------------------------------------------------------
# 8. Demo: online adaptation against a drifting opponent (COEP Fig. 5 style)
# ----------------------------------------------------------------------------------------------
class ScriptedTerran:
    """'Real' opponent for the demo: marines first, then it switches to firebats at `shift`."""

    def __init__(self, shift: int = 7500):
        self.shift = shift

    def observed(self, frame: int) -> Dict[str, float]:
        f = frame - 600                                  # scouting lag
        if f < 3000:
            return {}
        marines = 6.0 / (60 * FPS) * max(0, min(f, self.shift + 3000) - 2400)
        firebats = 8.0 / (60 * FPS) * max(0, f - self.shift)
        out = {"Marine": float(int(marines))}
        if firebats >= 1:
            out["Firebat"] = float(int(firebats))
        return out


def run_demo(cfg: Config, max_frame: int = 12000, interval: int = 1200, verbose: bool = True) -> Dict[str, int]:
    real = GameState()
    opp = ScriptedTerran()
    planner = TDCCEA(cfg, real)
    produced: Counter = Counter()
    t0 = time.time()
    if verbose:
        print(f"[{cfg.variant}] online demo  pop={cfg.pop_size} gens={cfg.generations} K={cfg.k_games}")
    while real.frame < max_frame:
        planner.update_state(real, opp.observed(real.frame))
        planner.restart()
        planner.run()
        st = planner.champion_strategy()
        log: list = []
        real, _ = predict(real, st.build, real.frame + interval, st.macro, 0, log)
        for name, _, auto in log:
            if not auto:
                planner.consume_build(name)
        combat = Counter(n for n, _, _ in log if n in COMBAT_UNITS)
        produced.update(combat)
        if verbose:
            top = planner.opp.arch[int(np.argmax(planner.opp.belief))].name
            print(f"  frame {real.frame:6d} | seen {str(opp.observed(real.frame) or '-'):<40} | belief->{top:<11} | built {dict(combat)}")
    if verbose:
        print(f"  total combat units produced: {dict(produced)}   ({time.time() - t0:.1f}s)")
        print("  champion bot config:", planner.champion_strategy().to_bot_config()["CombatParams"])
    return dict(produced)


def benchmark(base: Config, trials: int = 3) -> None:
    """Held-out score of each variant's final champion against 20 fresh opponent samples at frame 0."""
    print(f"{'variant':<15}{'mean held-out fitness':>24}")
    for v in VARIANTS:
        scores = []
        for t in range(trials):
            cfg = Config.preset(v, **{k: getattr(base, k) for k in
                                      ("pop_size", "generations", "k_games", "build_len", "seed")})
            cfg.seed = base.seed + t
            p = TDCCEA(cfg, GameState())
            p.opp.update({"Marine": 6.0}, 3000)
            p.run()
            held = p.opp.sample(20, random.Random(999))
            st = p.champion_strategy()
            scores.append(np.mean([discounted_fitness(st, p.state, h, cfg, p.mt)[0] for h in held]))
        print(f"{v:<15}{np.mean(scores):>24.1f}  (+/- {np.std(scores):.1f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="td_ccea", choices=VARIANTS)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = Config.preset(a.variant, pop_size=a.pop, generations=a.gens, k_games=a.k, seed=a.seed)
    if a.compare:
        benchmark(cfg)
    else:
        run_demo(cfg)


if __name__ == "__main__":
    main()
