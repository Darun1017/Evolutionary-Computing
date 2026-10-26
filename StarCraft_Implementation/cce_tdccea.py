"""
cce_tdccea.py — Evolutionary Algorithm Core for CCE-TDCCEA + SA-TDCCEA.

This module is the **biological engine** — it has zero knowledge of StarCraft II.
It receives fitness scalars from the evaluator and produces next-generation
genome vectors.

Architecture
────────────
  SubAgent       — one individual in a sub-population.
                   Stores params, trajectory history H_i, trajectory score e_i,
                   and per-gene mutation rates mu_j (SA mode).

  CCEACore       — manages the three sub-populations (Pop A / B / C),
                   cooperative handshake evaluation, centroid computation,
                   trajectory updates, diversity pruning, archive, and breeding.

Mathematics (per plan spec)
───────────────────────────
  Centroid:         c_t = mean(params) over each sub-population independently

  Trajectory score: e_i^t = Σ_k [ w_k · f_i^(k) ]
                    where w_k = (1 - d(c_t, c_k)/√dim) / Σ_j(...)
                    d = Euclidean distance between historical and current centroid

  Adaptive mutation (standard TD-CCEA):
                    μ_i = 0.15 · (1 − e_i^t) + 0.03

  SA mutation (per gene j):
                    σ_j(t) = μ_j · (1 − e_i^t) + MU_FLOOR
                    μ_j = MU_MIN + (1 − I_j/I_max) · (MU_MAX − MU_MIN)

  Diversity pruning: remove agent if similarity to any survivor > DELTA_S
                    similarity(a, b) = 1 − d(a, b)/√dim

  Fitness pruning:  remove agent if e_i < DELTA_F AND |H_i| >= 3
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple, Optional

import config as cfg

# ── Convenience aliases ───────────────────────────────────────────────────────
N_POP       = cfg.N_POP
HISTORY_LEN = cfg.HISTORY_LEN
A_MAX       = cfg.A_MAX
DELTA_S     = cfg.DELTA_S
DELTA_F     = cfg.DELTA_F
MIN_SURV    = cfg.MIN_SURVIVORS
MU_MIN      = cfg.MU_MIN
MU_MAX      = cfg.MU_MAX
MU_FLOOR    = cfg.MU_FLOOR
MU_DEFAULT  = cfg.MU_DEFAULT


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def _euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


# ─────────────────────────────────────────────────────────────────────────────
#  SUB-AGENT
# ─────────────────────────────────────────────────────────────────────────────

class SubAgent:
    """
    One individual in a sub-population.

    Attributes:
        dim         — length of this agent's parameter slice
        params      — current gene values (continuous, in [0, 1])
        H_i         — trajectory history: list of (fitness, centroid) tuples
        e_i         — current trajectory score (dynamic similarity weighted)
        mu_i        — scalar mutation std (standard TD-CCEA mode)
        mu_j        — per-gene mutation stds (SA-TDCCEA mode, optional)
    """

    def __init__(
        self,
        dim:  int,
        params: Optional[List[float]] = None,
        mu_j:   Optional[List[float]] = None,
    ) -> None:
        self.dim    = dim
        self.params = (
            [random.random() for _ in range(dim)]
            if params is None
            else [_clamp(float(x)) for x in params]
        )
        self.H_i: List[Tuple[float, List[float]]] = []
        self.e_i:  float = 0.0
        self.mu_i: float = MU_DEFAULT
        # SA per-gene mutation rates (None = standard scalar mutation)
        self.mu_j: Optional[List[float]] = mu_j

    # ── Trajectory ────────────────────────────────────────────────────────────

    def update_history(self, fitness: float, centroid: List[float]) -> None:
        """Append (fitness, centroid) to the sliding window H_i."""
        self.H_i.append((fitness, centroid[:]))
        if len(self.H_i) > HISTORY_LEN:
            self.H_i.pop(0)

    def compute_trajectory(self, current_centroid: List[float]) -> float:
        """
        Compute the dynamic similarity-weighted trajectory score e_i^t.

        Weights each past fitness by its environmental similarity to the
        current centroid.  Returns e_i and stores it in self.e_i.
        """
        if not self.H_i:
            self.e_i = 0.0
            return 0.0

        max_dist = math.sqrt(self.dim)
        raw_weights = []
        for _f, past_centroid in self.H_i:
            d   = _euclidean(current_centroid, past_centroid)
            sim = 1.0 - d / max_dist   # ∈ [−1, 1]; clamp below
            raw_weights.append(max(0.0, sim))

        total = sum(raw_weights)
        if total == 0.0:
            weights = [1.0 / len(self.H_i)] * len(self.H_i)
        else:
            weights = [w / total for w in raw_weights]

        self.e_i = sum(w * self.H_i[i][0] for i, w in enumerate(weights))
        return self.e_i

    def compute_mutation(self) -> float:
        """Compute scalar mutation std (standard TD-CCEA): μ = 0.15(1−e_i)+0.03."""
        self.mu_i = 0.15 * (1.0 - self.e_i) + 0.03
        return self.mu_i

    # ── Reproduction ──────────────────────────────────────────────────────────

    def mutate(self) -> "SubAgent":
        """
        Produce one offspring via Gaussian mutation.

        If mu_j is set (SA mode), each gene uses its own sensitivity-scaled
        mutation std:  σ_j = μ_j · (1 − e_i) + MU_FLOOR
        Otherwise, all genes use the scalar self.mu_i.
        """
        if self.mu_j is not None:
            offspring_params = [
                _clamp(p + random.gauss(0.0, self.mu_j[j] * (1.0 - self.e_i) + MU_FLOOR))
                for j, p in enumerate(self.params)
            ]
        else:
            offspring_params = [
                _clamp(p + random.gauss(0.0, self.mu_i))
                for p in self.params
            ]
        child = SubAgent(self.dim, offspring_params, self.mu_j)
        return child

    def __repr__(self) -> str:
        return (f"SubAgent(dim={self.dim}, e_i={self.e_i:.3f}, "
                f"mu_i={self.mu_i:.3f}, |H|={len(self.H_i)})")


# ─────────────────────────────────────────────────────────────────────────────
#  PRUNE + BREED HELPER  (shared by all three sub-populations)
# ─────────────────────────────────────────────────────────────────────────────

def _prune_and_breed(
    population: List[SubAgent],
    archive:    List[SubAgent],
) -> Tuple[List[SubAgent], List[SubAgent]]:
    """
    Apply TD-CCEA survival selection, archive update, and offspring breeding.

    Steps (mirror of existing CCE_TDCCEA.py logic, extended to match plan):
      1. Sort by e_i desc; remove agents too similar to survivors (> DELTA_S).
      2. Remove stagnant agents: e_i < DELTA_F and |H_i| >= 3.
      3. Ensure at least MIN_SURVIVORS remain.
      4. Promote top-3 to archive; trim archive to A_MAX by fitness.
      5. Breed from survivors ∪ archive until population reaches N_POP.
    """
    dim    = population[0].dim
    max_d  = math.sqrt(dim)

    # ── Step 1: Diversity pruning ─────────────────────────────────────────────
    survivors: List[SubAgent] = []
    for agent in sorted(population, key=lambda a: a.e_i, reverse=True):
        too_similar = any(
            1.0 - _euclidean(agent.params, u.params) / max_d > DELTA_S
            for u in survivors
        )
        if not too_similar:
            survivors.append(agent)

    # ── Step 2: Fitness floor pruning ─────────────────────────────────────────
    final: List[SubAgent] = [
        a for a in survivors
        if not (a.e_i < DELTA_F and len(a.H_i) >= 3)
    ]

    # ── Step 3: Minimum survivors guarantee ───────────────────────────────────
    for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
        if len(final) >= MIN_SURV:
            break
        if a not in final:
            final.append(a)

    # ── Step 4: Archive update ────────────────────────────────────────────────
    for top in sorted(population, key=lambda a: a.e_i, reverse=True)[:3]:
        copy = SubAgent(top.dim, top.params[:], top.mu_j)
        copy.e_i, copy.H_i, copy.mu_i = top.e_i, top.H_i[:], top.mu_i
        archive.append(copy)
    archive.sort(key=lambda a: a.e_i, reverse=True)
    del archive[A_MAX:]

    # ── Step 5: Breed offspring ───────────────────────────────────────────────
    pool = final + archive
    new_pop = final[:]
    while len(new_pop) < N_POP:
        candidates = random.sample(pool, min(3, len(pool)))
        parent = max(candidates, key=lambda a: a.e_i)
        new_pop.append(parent.mutate())

    return new_pop, archive


# ─────────────────────────────────────────────────────────────────────────────
#  CCEA CORE
# ─────────────────────────────────────────────────────────────────────────────

class CCEACore:
    """
    Manages three sub-populations (Pop A / Pop B / Pop C) with full TD-CCEA
    logic and cooperative evaluation.

    CCE-TDCCEA Cooperative Handshake
    ─────────────────────────────────
    To evaluate an agent in Pop A, concatenate its 10D slice with the current
    best agents from Pop B (15D) and Pop C (10D) → full 35D genome → SC2 game.
    Similarly for Pop B and Pop C.

    SA-TDCCEA Integration
    ─────────────────────
    If `mu_j_array` is provided (from sa_profiler output), each gene gets its
    own mutation boundary derived from offline influence profiling.
    """

    DIM_A = 10
    DIM_B = 15
    DIM_C = 10

    def __init__(self, mu_j_array: Optional[List[float]] = None) -> None:
        """
        Args:
            mu_j_array: 35D per-gene mutation rates from SA profiler.
                        If None, standard scalar mutation is used.
        """
        self.mu_j = mu_j_array  # full 35D array or None

        # Extract per-sub-population slices of mu_j
        mu_a = mu_j_array[0:10]  if mu_j_array else None
        mu_b = mu_j_array[10:25] if mu_j_array else None
        mu_c = mu_j_array[25:35] if mu_j_array else None

        # Initialise populations
        self.pop_a: List[SubAgent] = [SubAgent(self.DIM_A, mu_j=mu_a) for _ in range(N_POP)]
        self.pop_b: List[SubAgent] = [SubAgent(self.DIM_B, mu_j=mu_b) for _ in range(N_POP)]
        self.pop_c: List[SubAgent] = [SubAgent(self.DIM_C, mu_j=mu_c) for _ in range(N_POP)]

        self.archive_a: List[SubAgent] = []
        self.archive_b: List[SubAgent] = []
        self.archive_c: List[SubAgent] = []

        # Best representatives (used for cooperative evaluation)
        self.best_a: SubAgent = self.pop_a[0]
        self.best_b: SubAgent = self.pop_b[0]
        self.best_c: SubAgent = self.pop_c[0]

        self.generation: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def centroid_a(self) -> List[float]:
        return [sum(a.params[j] for a in self.pop_a) / N_POP
                for j in range(self.DIM_A)]

    def centroid_b(self) -> List[float]:
        return [sum(b.params[j] for b in self.pop_b) / N_POP
                for j in range(self.DIM_B)]

    def centroid_c(self) -> List[float]:
        return [sum(c.params[j] for c in self.pop_c) / N_POP
                for j in range(self.DIM_C)]

    def assemble_genome(
        self,
        agent_a: SubAgent,
        agent_b: SubAgent,
        agent_c: SubAgent,
    ) -> List[float]:
        """
        Concatenate three sub-agent slices into the full 35D genome.
        Indices: Pop A (0–9) | Pop B (10–24) | Pop C (25–34).
        """
        return agent_a.params + agent_b.params + agent_c.params

    def best_genome(self) -> List[float]:
        """Return the full 35D genome of the current best agents."""
        return self.assemble_genome(self.best_a, self.best_b, self.best_c)

    def update_fitness_a(
        self,
        agent: SubAgent,
        fitness: float,
        centroid: List[float],
    ) -> None:
        """Update history + trajectory + mutation for a Pop A agent."""
        agent.update_history(fitness, centroid)
        agent.compute_trajectory(centroid)
        agent.compute_mutation()

    def update_fitness_b(self, agent: SubAgent, fitness: float, centroid: List[float]) -> None:
        agent.update_history(fitness, centroid)
        agent.compute_trajectory(centroid)
        agent.compute_mutation()

    def update_fitness_c(self, agent: SubAgent, fitness: float, centroid: List[float]) -> None:
        agent.update_history(fitness, centroid)
        agent.compute_trajectory(centroid)
        agent.compute_mutation()

    def evolve_generation(self) -> None:
        """
        Apply TD-CCEA survival + archive + breeding to all three sub-populations.
        Call this AFTER all agents in the generation have been evaluated.
        Updates best_a / best_b / best_c.
        """
        self.best_a = max(self.pop_a, key=lambda a: a.e_i)
        self.best_b = max(self.pop_b, key=lambda a: a.e_i)
        self.best_c = max(self.pop_c, key=lambda a: a.e_i)

        self.pop_a, self.archive_a = _prune_and_breed(self.pop_a, self.archive_a)
        self.pop_b, self.archive_b = _prune_and_breed(self.pop_b, self.archive_b)
        self.pop_c, self.archive_c = _prune_and_breed(self.pop_c, self.archive_c)

        self.generation += 1

    def state_summary(self) -> dict:
        """Return a loggable dict of current generation statistics."""
        return {
            "generation": self.generation,
            "best_e_a":   round(self.best_a.e_i, 4),
            "best_e_b":   round(self.best_b.e_i, 4),
            "best_e_c":   round(self.best_c.e_i, 4),
            "avg_e_a":    round(sum(a.e_i for a in self.pop_a) / N_POP, 4),
            "avg_e_b":    round(sum(b.e_i for b in self.pop_b) / N_POP, 4),
            "avg_e_c":    round(sum(c.e_i for c in self.pop_c) / N_POP, 4),
            "archive_a":  len(self.archive_a),
            "archive_b":  len(self.archive_b),
            "archive_c":  len(self.archive_c),
        }

    def get_elite_genome(self) -> List[float]:
        """
        Return the best 35D genome from archives (highest e_i across all pops).
        Falls back to current best if archives are empty.
        """
        best_from_archive_a = (max(self.archive_a, key=lambda a: a.e_i)
                               if self.archive_a else self.best_a)
        best_from_archive_b = (max(self.archive_b, key=lambda b: b.e_i)
                               if self.archive_b else self.best_b)
        best_from_archive_c = (max(self.archive_c, key=lambda c: c.e_i)
                               if self.archive_c else self.best_c)
        return self.assemble_genome(best_from_archive_a, best_from_archive_b, best_from_archive_c)
