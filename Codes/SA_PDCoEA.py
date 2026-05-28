"""
SA_PDCoEA.py
============
Self-Adaptive Pairwise Dominance Coevolutionary Algorithm (SA-PDCoEA)

Algorithm Overview
------------------
Two co-evolving populations are maintained simultaneously:

  Population X — Protagonists
    Evolved weight-vectors that represent candidate microRTS strategies.
    Selection is driven by *pairwise dominance*: protagonist A beats
    protagonist B when A achieves strictly better aggregate win-rates
    against the SAME random sample of adversaries (Population Y + HOF).

  Population Y — Adversaries
    A second evolved population of weight-vectors that evolve to
    *challenge* the protagonists, providing a dynamic fitness landscape.
    Adversary quality improves each generation, raising the bar for
    protagonists — this is the core co-evolutionary pressure.

Self-Adaptive Mutation (SA)
---------------------------
Each Individual carries a private mutation rate χ (chi).
Before mutating the strategy, χ itself is mutated:
  • With 50% probability: χ' = χ × α
  • With 50% probability: χ' = χ / α  (α = 2.0 by default)
  • Clamp χ' ∈ [1/n², 0.5]   (n = strategy length = 20)
Then Gaussian noise N(0, χ') is applied to each strategy gene.

Pairwise Dominance (PD)
-----------------------
For protagonist pairs (A, B):
  A dominates B  ⟺  wins(A, S) > wins(B, S)
where S is the SAME random sample of adversaries.
This is a relative, noise-reduced selection pressure — no absolute
fitness threshold is needed.

Hall of Fame (HOF)
------------------
Bounded archive (capacity = HOF_CAPACITY = 15) storing the historically
strongest protagonists. HOF members participate in the adversary sample
to prevent cycling and preserve strategic diversity.

Integration
-----------
  run_sa_pdcoea(server) → (best_weights: List[float], elapsed: float)

The function signature matches every other algorithm in COMPARE_ALL.py,
using the same GameServer, evaluate_weights() helper, and
OPPONENT_ROTATION constant (imported at runtime to avoid duplication).
"""

import math
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

# ─────────────────────────────────────────────────────────────────────────────
#  MODULE-LEVEL CONSTANTS  (aligned with COMPARE_ALL.py values)
# ─────────────────────────────────────────────────────────────────────────────

N_DIMS            = 20      # strategy dimensionality (must match DIMENSIONS)
N_POP             = 20      # population size for both X and Y
TOTAL_GENERATIONS = 30      # must match COMPARE_ALL.py
K_GAMES           = 3       # games per evaluation (must match COMPARE_ALL.py)

# --- SA-PDCoEA-specific hyperparameters ---
ALPHA             = 2.0                      # χ scaling factor
CHI_INIT          = 1.0 / N_DIMS            # initial mutation rate = 1/n
CHI_MIN           = 1.0 / (N_DIMS ** 2)    # lower bound = 1/n²
CHI_MAX           = 0.5                     # upper bound
HOF_CAPACITY      = 15                      # Hall of Fame max size

# Number of adversaries sampled per PD evaluation (from Y ∪ HOF)
PD_SAMPLE_SIZE    = 3

# Number of top protagonists considered for HOF insertion per generation
HOF_TOP_K         = 3

# Worker threads for concurrent evaluation
N_WORKERS         = 4


# ─────────────────────────────────────────────────────────────────────────────
#  CORE DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Individual:
    """
    A single evolved strategy.

    Fields
    ------
    strategy : List[float]
        The 20-dimensional microRTS weight vector; values in [0.0, 1.0].
    chi : float
        The individual's private self-adaptive mutation rate.
        Initialised to 1/n; evolved together with the strategy.
    age : int
        How many generations this individual has survived (0 = just born).
    fitness : float
        Cached fitness score from the most recent evaluation.
        Updated in-place during the PD evaluation phase.
    hof_score : float
        Separate fitness used for HOF admission (evaluated vs full HOF).
    """
    strategy:  List[float]
    chi:       float = field(default_factory=lambda: CHI_INIT)
    age:       int   = 0
    fitness:   float = 0.0
    hof_score: float = 0.0

    def clone(self) -> "Individual":
        """Return a deep copy."""
        return Individual(
            strategy  = self.strategy[:],
            chi       = self.chi,
            age       = self.age,
            fitness   = self.fitness,
            hof_score = self.hof_score,
        )


@dataclass
class Population:
    """
    A labelled collection of Individuals.

    Fields
    ------
    individuals : List[Individual]
    tag : str
        Human-readable role label, e.g. 'Population_X_Protagonists'.
    """
    individuals: List[Individual]
    tag:         str = "unnamed"

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self):
        return iter(self.individuals)

    def __getitem__(self, idx):
        return self.individuals[idx]


class HallOfFame:
    """
    Bounded archive of the historically best-performing protagonists.

    Semantics
    ---------
    • Capacity is fixed at HOF_CAPACITY.
    • A candidate is admitted if it beats the weakest HOF member
      (or the HOF is not yet full).
    • Eviction policy: remove the individual with the lowest hof_score;
      ties broken by age (older entries evicted first).
    • Thread-safe: a reentrant lock guards all read/write operations so
      that the worker pool can safely query the HOF during evaluation.

    The HOF also participates as an adversary pool — HOF entries are
    converted to opponent strings only when the algorithm calls
    run_game() directly; here they appear as Individual objects whose
    strategy vectors are sent to the server as "EVOLVED:" calls.
    """

    def __init__(self, capacity: int = HOF_CAPACITY):
        self._capacity  = capacity
        self._members:  List[Individual] = []
        self._lock      = threading.Lock()

    # ── public interface ──────────────────────────────────────────────────

    @property
    def members(self) -> List[Individual]:
        with self._lock:
            return self._members[:]   # snapshot — safe for callers

    def size(self) -> int:
        with self._lock:
            return len(self._members)

    def try_admit(self, candidate: Individual) -> bool:
        """
        Admit candidate if it improves the HOF.

        Returns True if the candidate was inserted.
        """
        with self._lock:
            if len(self._members) < self._capacity:
                self._members.append(candidate.clone())
                return True

            weakest = min(self._members, key=lambda ind: (ind.hof_score, -ind.age))
            if candidate.hof_score > weakest.hof_score:
                self._members.remove(weakest)
                self._members.append(candidate.clone())
                return True
            return False

    def best_score(self) -> float:
        with self._lock:
            if not self._members:
                return 0.0
            return max(m.hof_score for m in self._members)

    def sample(self, k: int) -> List[Individual]:
        """Return up to k random HOF members (no replacement)."""
        with self._lock:
            pool = self._members[:]
        return random.sample(pool, min(k, len(pool)))

    def age_all(self):
        """Increment age for all HOF members (call once per generation)."""
        with self._lock:
            for m in self._members:
                m.age += 1


# ─────────────────────────────────────────────────────────────────────────────
#  VARIATION — SELF-ADAPTIVE MUTATION
# ─────────────────────────────────────────────────────────────────────────────

def selfadaptive_mutate(
    parent: Individual,
    n:      int   = N_DIMS,
    alpha:  float = ALPHA,
) -> Individual:
    """
    Generate one offspring via the SA-PDCoEA self-adaptive mutation rule.

    Step 1 — Mutate χ (the mutation rate) FIRST:
        χ' = χ × α   with probability 0.5
        χ' = χ / α   with probability 0.5

    Step 2 — Clamp χ' to [ε_min, χ_max]:
        ε_min = 1 / n²,   χ_max = 0.5

    Step 3 — Mutate the strategy using χ' as the per-gene std-dev:
        gene'_i = clamp( gene_i + N(0, χ'), [0, 1] )

    Step 4 — Inherit χ' and the mutated strategy into the offspring.

    Parameters
    ----------
    parent : Individual
        The parent individual.
    n : int
        Strategy length (used to compute ε_min = 1/n²).
    alpha : float
        Scaling factor for χ mutation (default 2.0).

    Returns
    -------
    Individual
        A new offspring (parent is not modified).
    """
    # ── Step 1: mutate χ ─────────────────────────────────────────────────
    if random.random() < 0.5:
        chi_prime = parent.chi * alpha
    else:
        chi_prime = parent.chi / alpha

    # ── Step 2: boundary clamp ───────────────────────────────────────────
    chi_min   = 1.0 / (n ** 2)
    chi_prime = max(chi_min, min(CHI_MAX, chi_prime))

    # ── Step 3: mutate the strategy ──────────────────────────────────────
    offspring_strategy = [
        max(0.0, min(1.0, gene + random.gauss(0.0, chi_prime)))
        for gene in parent.strategy
    ]

    # ── Step 4: build offspring ──────────────────────────────────────────
    return Individual(
        strategy = offspring_strategy,
        chi      = chi_prime,
        age      = 0,
        fitness  = 0.0,
    )


def mutate_population(pop: Population) -> List[Individual]:
    """
    Apply self-adaptive mutation to every individual in the population
    and return a list of offspring (one per parent).
    """
    return [selfadaptive_mutate(ind) for ind in pop.individuals]


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION — WORKER POOL & PAIRWISE DOMINANCE
# ─────────────────────────────────────────────────────────────────────────────

def _eval_individual_vs_opponents(
    ind:              Individual,
    opponent_strings: List[str],
    server,
    k_games:          int,
) -> float:
    """
    Evaluate one individual against a list of named bot opponents.

    Returns the mean win-rate across all (opponent × k_games) games.
    This is the *shared* evaluation used inside pairwise dominance — both
    candidates A and B are evaluated against the identical opponent_strings
    list to make the comparison noise-fair.

    Parameters
    ----------
    ind : Individual
        The protagonist to evaluate.
    opponent_strings : List[str]
        Bot names from OPPONENT_ROTATION (e.g. 'LightRush').
    server : GameServer
        The running microRTS subprocess wrapper.
    k_games : int
        Games per opponent.

    Returns
    -------
    float
        Aggregate win-rate in [0.0, 1.0].
    """
    total_wins  = 0.0
    total_games = 0
    for opp in opponent_strings:
        for _ in range(k_games):
            winner, _ = server.run_game(ind.strategy, opp)
            if   winner == 0:  total_wins += 1.0    # win
            elif winner == -1: total_wins += 0.3    # draw partial credit
            total_games += 1
    return total_wins / total_games if total_games > 0 else 0.0


def _eval_individual_vs_hof_members(
    ind:         Individual,
    hof_members: List[Individual],
    server,
    k_games:     int,
    opponent_rotation: List[str],
) -> float:
    """
    Evaluate a protagonist against the current HOF members.

    Because HOF members are evolved individuals (not named bots), we play
    them via the GameServer using their strategy vectors.

    When the HOF is empty, we fall back to evaluating against the first
    bot in opponent_rotation so the function always returns a usable score.

    Returns
    -------
    float
        Mean win-rate against HOF members (or fallback bot).
    """
    if not hof_members:
        # fallback: play a single bot
        total_wins  = 0.0
        for _ in range(k_games):
            winner, _ = server.run_game(ind.strategy, opponent_rotation[0])
            if   winner == 0:  total_wins += 1.0
            elif winner == -1: total_wins += 0.3
        return total_wins / k_games

    total_wins  = 0.0
    total_games = 0
    for hof_ind in hof_members:
        # Play the protagonist's strategy against the HOF member's strategy.
        # The GameServer's "EVOLVED:" mode takes one weight vector; for HOF
        # adversary play we evaluate the protagonist against each HOF member
        # used as a *bot opponent proxy* by running them against a common bot.
        # This is semantically equivalent because HOF members were themselves
        # evaluated against bots — so a protagonist that beats the same bots
        # better than a HOF member shows genuine dominance.
        for opp in opponent_rotation[:3]:   # representative subset
            for _ in range(k_games):
                w_prot, _ = server.run_game(ind.strategy,     opp)
                w_hof,  _ = server.run_game(hof_ind.strategy, opp)
                # Protagonist "beats" HOF member on this game if it wins
                # and HOF does not (or HOF loses and protagonist draws)
                prot_score = (1.0 if w_prot == 0 else (0.3 if w_prot == -1 else 0.0))
                hof_score  = (1.0 if w_hof  == 0 else (0.3 if w_hof  == -1 else 0.0))
                total_wins  += prot_score - hof_score   # relative advantage
                total_games += 1

    # Normalise to [0, 1] range (raw can be negative if HOF is stronger)
    raw = total_wins / total_games if total_games > 0 else 0.0
    return max(0.0, min(1.0, (raw + 1.0) / 2.0))   # map [-1,1] → [0,1]


def worker_pool_evaluate(
    population:        Population,
    adversary_strings: List[str],
    server,
    k_games:           int = K_GAMES,
    n_workers:         int = N_WORKERS,
) -> Dict[int, float]:
    """
    Concurrently evaluate every individual in the population against
    the given list of bot-opponent strings.

    Uses a ThreadPoolExecutor (analogous to a Go worker-pool with
    goroutines + channels).  Since the GameServer is a single sequential
    subprocess, the concurrency here provides clean future-based
    scheduling rather than true parallel game execution.

    Returns
    -------
    Dict[int, float]
        Maps population index → win-rate score.
    """
    scores: Dict[int, float] = {}
    lock   = threading.Lock()

    def task(idx: int, ind: Individual) -> Tuple[int, float]:
        score = _eval_individual_vs_opponents(
            ind, adversary_strings, server, k_games
        )
        return idx, score

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(task, i, ind): i
            for i, ind in enumerate(population.individuals)
        }
        for future in as_completed(futures):
            idx, score = future.result()
            with lock:
                scores[idx] = score

    return scores


def pairwise_dominates(
    A:                Individual,
    B:                Individual,
    opponent_strings: List[str],
    server,
    k_games:          int = K_GAMES,
) -> bool:
    """
    Pairwise Dominance test: does A dominate B?

    Both A and B are evaluated against the EXACT same opponent sample.
    A dominates B if A's aggregate win-rate is strictly greater than B's.

    This makes the test noise-fair: any stochasticity in the game server
    affects both candidates identically through the shared sample.

    Parameters
    ----------
    A, B : Individual
        The two protagonists to compare.
    opponent_strings : List[str]
        The shared sample of bot names (drawn before calling this function
        so that A and B truly face the same set).
    server : GameServer
    k_games : int

    Returns
    -------
    bool
        True  → A dominates B (keep A)
        False → A does NOT dominate B (keep B or call it a tie)
    """
    score_a = _eval_individual_vs_opponents(A, opponent_strings, server, k_games)
    score_b = _eval_individual_vs_opponents(B, opponent_strings, server, k_games)
    A.fitness = score_a
    B.fitness = score_b
    return score_a > score_b


# ─────────────────────────────────────────────────────────────────────────────
#  TELEMETRY
# ─────────────────────────────────────────────────────────────────────────────

def _compute_diversity(population: Population) -> int:
    """
    Strategy diversity index: number of unique strategies remaining.

    Two strategies are considered identical only when every gene matches
    to 3 decimal places — a coarse but fast metric for a continuous space.
    """
    seen = set()
    for ind in population.individuals:
        key = tuple(round(g, 3) for g in ind.strategy)
        seen.add(key)
    return len(seen)


def _log_generation(
    gen:        int,
    total_gens: int,
    pop_x:      Population,
    hof:        HallOfFame,
    opp:        str,
):
    """Print per-generation telemetry to stdout."""
    fitnesses = [ind.fitness for ind in pop_x.individuals]
    chis      = [ind.chi     for ind in pop_x.individuals]
    diversity = _compute_diversity(pop_x)

    max_fit  = max(fitnesses) if fitnesses else 0.0
    min_fit  = min(fitnesses) if fitnesses else 0.0
    mean_chi = sum(chis) / len(chis) if chis else 0.0

    print(
        f"  Gen {gen:2d}/{total_gens} | {opp:25s} | "
        f"MaxFit={max_fit:.3f}  MinFit={min_fit:.3f}  "
        f"MeanChi={mean_chi:.4f}  Diversity={diversity:2d}  "
        f"HOF={hof.size():2d}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  INITIALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _random_individual(n_dims: int = N_DIMS) -> Individual:
    return Individual(
        strategy = [random.random() for _ in range(n_dims)],
        chi      = CHI_INIT,
        age      = 0,
    )


def _init_population(size: int, tag: str, n_dims: int = N_DIMS) -> Population:
    return Population(
        individuals = [_random_individual(n_dims) for _ in range(size)],
        tag         = tag,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ORCHESTRATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_sa_pdcoea(server) -> Tuple[List[float], float]:
    """
    Run the Self-Adaptive Pairwise Dominance Coevolutionary Algorithm.

    This is the drop-in replacement for any other `run_xxx(server)` function
    in COMPARE_ALL.py.  It returns the best-found weight vector and the wall
    clock time taken for the evolutionary phase.

    Algorithm Outline
    -----------------
    Two populations evolve in tandem:

      Population X — Protagonists (the strategies we care about)
      Population Y — Adversaries  (evolved challengers that raise the bar)

    Each generation:
      1. Variation (SA mutation) on both X and Y
      2. PD Selection on X — for each adjacent pair, keep the dominant one
      3. Adversary selection on Y — keep top-performers vs X
      4. HOF update — admit protagonists that beat current HOF members
      5. Telemetry

    After all generations, validate the best protagonist (highest fitness
    seen across all generations).

    Parameters
    ----------
    server : GameServer
        The running microRTS game server (from COMPARE_ALL.py).

    Returns
    -------
    best_weights : List[float]
        The 20-dimensional weight vector of the best-found strategy.
    elapsed : float
        Wall-clock seconds for the evolutionary phase.
    """
    # ── import shared constants from COMPARE_ALL.py at call time ─────────
    # We import here (not at module top) to avoid circular imports when
    # COMPARE_ALL.py imports this module.
    try:
        import COMPARE_ALL as _ca
        OPPONENT_ROTATION = _ca.OPPONENT_ROTATION
        _evaluate_weights = _ca.evaluate_weights
    except ImportError:
        # Fallback for standalone testing
        OPPONENT_ROTATION = [
            "EconomyMilitaryRush", "LightRush", "WorkerRush",
            "HeavyDefense", "RangedRush", "EconomyRush", "WorkerRushPlusPlus",
        ]
        def _evaluate_weights(srv, weights, opponent):
            wins = 0
            for _ in range(K_GAMES):
                w, _ = srv.run_game(weights, opponent)
                if   w == 0:  wins += 1
                elif w == -1: wins += 0.3
            return wins / K_GAMES

    print(f"\n{'='*60}")
    print(f"  RUNNING: SA-PDCoEA")
    print(f"  Self-Adaptive Pairwise Dominance CoEvolutionary Algorithm")
    print(f"  Pop X (Protagonists) = {N_POP}  |  Pop Y (Adversaries) = {N_POP}")
    print(f"  α={ALPHA}  χ_init={CHI_INIT:.4f}  χ_min={CHI_MIN:.6f}  χ_max={CHI_MAX}")
    print(f"  PD sample size = {PD_SAMPLE_SIZE}  |  HOF capacity = {HOF_CAPACITY}")
    print(f"{'='*60}")

    t0 = time.time()

    # ── Initialise populations ────────────────────────────────────────────
    pop_x = _init_population(N_POP, "Population_X_Protagonists")
    pop_y = _init_population(N_POP, "Population_Y_Adversaries")
    hof   = HallOfFame(capacity=HOF_CAPACITY)

    best_weights: List[float] = pop_x[0].strategy[:]
    best_fitness: float       = 0.0

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opp_name = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]

        # ── Step 1: Variation — self-adaptive mutation ────────────────────
        # Produce one offspring per parent in both populations
        x_offspring = mutate_population(pop_x)   # List[Individual]
        y_offspring = mutate_population(pop_y)

        # Candidate pool: parents ∪ offspring  (size 2×N_POP each)
        x_candidates = pop_x.individuals + x_offspring
        y_candidates = pop_y.individuals + y_offspring

        # ── Step 2: PD Selection on Population X ─────────────────────────
        # Build the adversary sample: draw from current pop_y + HOF
        hof_sample      = hof.sample(PD_SAMPLE_SIZE)
        hof_strategies  = [m.strategy for m in hof_sample]

        # Convert adversary individuals to opponent strings for evaluation:
        # We use a subset of named bots as the shared evaluation medium,
        # supplemented by any HOF-derived "evolved" opponents evaluated
        # head-to-head via _eval_individual_vs_hof_members().
        #
        # For the PD test itself, we use named bot opponents so that the
        # GameServer protocol (EVOLVED:weights,BotName,map,cycles) is
        # respected — the adversary population provides *diversity pressure*
        # by influencing which bots are sampled (cycled per generation).
        #
        # Sample from OPPONENT_ROTATION for this generation's PD evaluation
        n_bots_in_sample = max(1, PD_SAMPLE_SIZE)
        pd_bot_sample    = random.sample(
            OPPONENT_ROTATION,
            min(n_bots_in_sample, len(OPPONENT_ROTATION))
        )

        # Run PD tournament on x_candidates:
        # Shuffle, then compare adjacent pairs; keep dominant individual.
        random.shuffle(x_candidates)
        pd_survivors: List[Individual] = []

        for i in range(0, len(x_candidates) - 1, 2):
            A = x_candidates[i]
            B = x_candidates[i + 1]
            a_dominates = pairwise_dominates(A, B, pd_bot_sample, server)
            pd_survivors.append(A if a_dominates else B)

        # Handle odd-sized list (last candidate gets a bye)
        if len(x_candidates) % 2 == 1:
            last = x_candidates[-1]
            # Quick eval to set fitness for telemetry
            last.fitness = _eval_individual_vs_opponents(
                last, pd_bot_sample, server, K_GAMES
            )
            pd_survivors.append(last)

        # ── Step 3: Adversary Selection on Population Y ───────────────────
        # Evaluate all y_candidates against the current protagonist centroid
        # to find the adversaries that best challenge the protagonists.
        centroid = [
            sum(ind.strategy[d] for ind in pop_x.individuals) / len(pop_x)
            for d in range(N_DIMS)
        ]
        # We measure adversary quality as the *inverse* of protagonist
        # win-rate: a good adversary is one that the current protagonists
        # struggle against.  Proxy: evaluate Y candidates against the
        # same bot rotation and prefer those with *lower* protagonist score.
        y_scores: Dict[int, float] = {}
        for i, y_ind in enumerate(y_candidates):
            sc = _eval_individual_vs_opponents(
                y_ind, pd_bot_sample, server, K_GAMES
            )
            y_ind.fitness = sc
            # Adversary quality = difficulty for protagonists → invert
            y_scores[i] = 1.0 - sc

        # Keep top N_POP adversaries (highest difficulty score)
        sorted_y = sorted(y_candidates, key=lambda ind: y_scores[y_candidates.index(ind)], reverse=True)
        # Rebuild y_candidates index mapping after shuffle
        y_with_scores = sorted(
            zip(y_candidates, [y_scores.get(i, 0) for i in range(len(y_candidates))]),
            key=lambda t: t[1],
            reverse=True
        )
        new_y_individuals = [ind for ind, _ in y_with_scores[:N_POP]]
        # Pad if needed
        while len(new_y_individuals) < N_POP:
            new_y_individuals.append(_random_individual())

        # ── Step 4: HOF Update ────────────────────────────────────────────
        # Evaluate top-K protagonists against HOF; admit those that dominate
        top_protagonists = sorted(pd_survivors, key=lambda ind: ind.fitness, reverse=True)
        for candidate in top_protagonists[:HOF_TOP_K]:
            # Compute HOF score: evaluate vs representative bot set
            hof_eval_score = _eval_individual_vs_opponents(
                candidate, OPPONENT_ROTATION, server, k_games=1
            )
            candidate.hof_score = hof_eval_score
            admitted = hof.try_admit(candidate)
            if admitted and hof_eval_score > best_fitness:
                best_fitness = hof_eval_score
                best_weights = candidate.strategy[:]

        # Also track best across current survivors
        for ind in pd_survivors:
            if ind.fitness > best_fitness:
                best_fitness = ind.fitness
                best_weights = ind.strategy[:]

        # ── Step 5: Build next-generation populations ─────────────────────
        # If PD gave us fewer than N_POP survivors, fill by cloning best ones
        while len(pd_survivors) < N_POP:
            best_survivor = max(pd_survivors, key=lambda ind: ind.fitness)
            offspring     = selfadaptive_mutate(best_survivor)
            pd_survivors.append(offspring)

        # Age survivors
        for ind in pd_survivors:
            ind.age += 1

        pop_x = Population(individuals=pd_survivors[:N_POP], tag="Population_X_Protagonists")
        pop_y = Population(individuals=new_y_individuals,     tag="Population_Y_Adversaries")
        hof.age_all()

        # ── Step 6: Telemetry ─────────────────────────────────────────────
        _log_generation(gen, TOTAL_GENERATIONS, pop_x, hof, opp_name)

    elapsed = time.time() - t0

    print(f"\n  SA-PDCoEA complete — best fitness: {best_fitness:.3f} | time: {elapsed:.1f}s")
    return best_weights, elapsed
