"""
DI_CCE_TDCCEA_COMPARE.py
========================
Dynamic Interaction-Driven CCE-TDCCEA vs Original CCE-TDCCEA
on a 32×32 map with full multi-core parallelism.

ALGORITHMS:
  1. CCE-TDCCEA       — Original cooperative co-evolutionary TD-CCEA
                        (fixed Macro/Micro decomposition)
  2. DI-CCE-TDCCEA    — Dynamic Interaction-Driven CCE-TDCCEA (NEW)
                        Replaces the fixed Macro/Micro decomposition with
                        a dynamically learned decomposition based on a
                        strategy-interaction matrix and Maximum Spanning Tree.

NEW VARIANT (DI-CCE-TDCCEA) — From algo_modification_fyp.pdf Algorithm 2:
  ─────────────────────────────────────────────────────────────────────────
  The principal transformation is:
      Fixed CCE Decomposition  →  Interaction-Driven Dynamic Decomposition

  Adaptive cycle:
      Et → Evaluate → Interaction Matrix → Maximum Spanning Tree
         → Dynamic Groups → Co-evolve → Evaluate

  Key mechanisms:
    • Environmental fingerprint  Et = [Rpop, Ropp, Rgame, Rfit]
    • Strategy-interaction matrix  I  updated from observed joint-fitness
    • Maximum Spanning Tree  Tt  constructed from I
    • Temporary strategy clusters  G1..Gm  extracted from strong MST paths
    • Co-evolution windows with min/max length and improvement thresholds
    • Trajectory-driven evaluation identical to TD-CCEA

PARALLELISM:
  Uses a pool of Java game servers (one per CPU core - 1) to run
  evaluations concurrently via ThreadPoolExecutor.

INSTALL: pip install openpyxl   (optional, for Excel export)
"""

import math, random, time, subprocess, os, json, copy, queue, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import openpyxl
    _OPENPYXL_AVAILABLE = True
except ImportError:
    _OPENPYXL_AVAILABLE = False
    print("[WARNING] openpyxl not installed — Excel export disabled.")

os.environ["PYTHONIOENCODING"] = "utf-8"


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
cand_dirs    = [
    os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "Simulation", "microrts"),
    os.path.join(os.path.dirname(SCRIPT_DIR), "Simulation", "microrts"),
]
MICRORTS_DIR = next((d for d in cand_dirs if os.path.isdir(d)), cand_dirs[0])

WEIGHT_NAMES = [
    "worker_ratio", "max_workers", "expand_threshold", "light_preference",
    "heavy_preference", "ranged_preference", "barracks_timing", "aggression",
    "attack_threshold", "defend_radius", "target_priority", "defense_trigger",
    "counter_rush", "tech_switch", "economy_pressure", "scout_priority",
    "group_attack", "second_barracks", "worker_fight_ratio", "focus_fire",
]
DIMENSIONS    = len(WEIGHT_NAMES)          # 20
MACRO_INDICES = [0, 1, 2, 6, 14, 15, 17, 18]        # 8 economy weights
MICRO_INDICES = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 16, 19]  # 12 combat weights
MAP_PATH      = "maps/chambers32x32.xml"   # 32×32 map
MAX_CYCLES    = 10000

OPPONENT_ROTATION = [
    "EconomyMilitaryRush", "LightRush", "WorkerRush",
    "HeavyDefense", "RangedRush", "EconomyRush", "WorkerRushPlusPlus",
]

# ── Shared hyperparameters ────────────────────────────────────────
N_POP             = 20
K_GAMES           = 3
TOTAL_GENERATIONS = 30
VALIDATION_GAMES  = 5
A_MAX             = 15
DELTA_S           = 0.85
DELTA_F           = 0.20
TOP_PARTNERS      = 3

# ── DI-CCE-TDCCEA specific hyperparameters ────────────────────────
TAU_I             = 0.15     # interaction threshold for MST edge inclusion
L_MIN             = 3        # minimum co-evolution window length
L_MAX             = 8        # maximum co-evolution window length
TAU_IMP           = 0.02     # joint-improvement threshold
INTERACTION_DECAY = 0.9      # EMA decay for interaction matrix smoothing
TRAJECTORY_WINDOW = 7        # trajectory history window W

# ── CPU parallelism ───────────────────────────────────────────────
N_WORKERS = max(2, min((os.cpu_count() or 4) - 1, N_POP))


# ═══════════════════════════════════════════════════════════════════
#  GAME SERVER
# ═══════════════════════════════════════════════════════════════════
class GameServer:
    def __init__(self):
        self.process = None
        self._fail_count  = 0
        self._total_count = 0

    def start(self):
        self.process = subprocess.Popen(
            ["java", "-cp", "lib/*;bin", "tests.EvolvedBatchGameRunner"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=MICRORTS_DIR
        )
        line = self.process.stdout.readline().strip()
        if line != "READY":
            err = ""
            try:
                err = self.process.stderr.read(2000)
            except Exception:
                pass
            raise RuntimeError(f"Server failed to start: stdout='{line}' stderr='{err}'")

    def run_game(self, weights, opponent, map_path=None):
        """Run EVOLVED vs BOT. Returns (winner, cycles)."""
        if map_path is None:
            map_path = MAP_PATH
        self._total_count += 1

        if self.process and self.process.poll() is not None:
            self._fail_count += 1
            return -1, MAX_CYCLES

        ws  = ";".join(f"{w:.4f}" for w in weights)
        cmd = f"EVOLVED:{ws},{opponent},{map_path},{MAX_CYCLES}\n"
        try:
            self.process.stdin.write(cmd)
            self.process.stdin.flush()
            r = self.process.stdout.readline().strip()
        except (BrokenPipeError, OSError):
            self._fail_count += 1
            return -1, MAX_CYCLES

        if r.startswith("RESULT:"):
            p = r.replace("RESULT:", "").split(",")
            return int(p[0]), int(p[1])

        self._fail_count += 1
        return -1, MAX_CYCLES

    def stop(self):
        if self.process:
            try:
                self.process.stdin.write("EXIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()


# ═══════════════════════════════════════════════════════════════════
#  PARALLEL GAME SERVER POOL
#  One live Java game server per worker thread; evaluations run
#  concurrently across all available CPU cores.
# ═══════════════════════════════════════════════════════════════════
class ParallelGamePool:
    def __init__(self, n_workers):
        self.n_workers  = n_workers
        self.pool_queue = queue.Queue()

    def start(self):
        print(f"  [CPU] Starting {self.n_workers} parallel game servers...")
        for _ in range(self.n_workers):
            s = GameServer()
            s.start()
            self.pool_queue.put(s)
        print(f"  [OK] {self.n_workers} servers ready (of {os.cpu_count()} CPU cores)")

    # ── evaluate ONE weight vector vs ONE bot ────────────────────
    def _eval_one(self, weights, opponent, k_games):
        """Borrow a server from the pool, evaluate, then return it."""
        server = self.pool_queue.get()
        try:
            wins = 0.0
            for _ in range(k_games):
                w, _ = server.run_game(weights, opponent)
                if w == 0:    wins += 1.0
                elif w == -1: wins += 0.3
            return wins / k_games
        finally:
            self.pool_queue.put(server)

    # ── evaluate a BATCH of weight vectors (all vs same bot) ─────
    def eval_batch(self, weight_list, opponent, k_games=K_GAMES):
        """
        Evaluate every weight-vector in `weight_list` against `opponent`
        in parallel.  Returns a list of fitness scores in the same order.
        """
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(self._eval_one, w, opponent, k_games)
                for w in weight_list
            ]
            return [f.result() for f in futures]

    # ── validate ONE weight vector against ALL 7 bots ────────────
    def validate_weights(self, weights):
        """
        Validate a single weight vector against all seven bots.
        Returns (overall_wr%, per_opponent_dict%).
        """
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                opp: executor.submit(self._eval_one, weights, opp, VALIDATION_GAMES)
                for opp in OPPONENT_ROTATION
            }
            per_opp = {opp: futures[opp].result() * 100.0 for opp in OPPONENT_ROTATION}
        overall = sum(per_opp.values()) / len(per_opp)
        return overall, per_opp

    def stop(self):
        while not self.pool_queue.empty():
            self.pool_queue.get().stop()
        print("  [OK] All parallel game servers stopped")


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════
def euclidean_dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ═══════════════════════════════════════════════════════════════════
#  WILCOXON SIGNED-RANK TEST  (pure Python, no scipy)
# ═══════════════════════════════════════════════════════════════════
def wilcoxon_test(x, y):
    diffs = [xi - yi for xi, yi in zip(x, y) if xi != yi]
    if len(diffs) < 3:
        return None
    n     = len(diffs)
    abs_d = sorted(enumerate(abs(d) for d in diffs), key=lambda t: t[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and abs_d[j][1] == abs_d[j+1][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j+1):
            ranks[abs_d[k][0]] = avg_rank
        i = j + 1
    w_plus  = sum(ranks[i] for i, d in enumerate(diffs) if d > 0)
    w_minus = sum(ranks[i] for i, d in enumerate(diffs) if d < 0)
    w_stat  = min(w_plus, w_minus)
    mu_w    = n * (n + 1) / 4.0
    sig_w   = math.sqrt(n * (n + 1) * (2*n + 1) / 24.0)
    if sig_w == 0:
        return None
    z = (w_stat - mu_w) / sig_w
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2))))
    return p


# ═══════════════════════════════════════════════════════════════════
#  CCE-TDCCEA CLASSES  (Original — UNCHANGED from COMPARE_ALL.py)
# ═══════════════════════════════════════════════════════════════════
class SubAgent:
    """Sub-agent for one population partition (Macro or Micro)."""
    def __init__(self, dim, params=None):
        self.dim    = dim
        self.params = ([random.random() for _ in range(dim)] if params is None
                       else [max(0.0, min(1.0, float(x))) for x in params])
        self.H_i  = []
        self.e_i  = 0.0
        self.mu_i = 0.0

    def update_history(self, f, env):
        self.H_i.append((f, env))
        if len(self.H_i) > TRAJECTORY_WINDOW:
            self.H_i.pop(0)

    def compute_trajectory(self, env):
        if not self.H_i:
            self.e_i = 0.0
            return
        md = math.sqrt(self.dim)
        ws = [max(0.0, 1.0 - euclidean_dist(env[:self.dim], pe[:self.dim]) / md)
              if len(pe) >= self.dim else 1.0 for _, pe in self.H_i]
        t  = sum(ws)
        ws = [w / t for w in ws] if t > 0 else [1.0 / len(ws)] * len(ws)
        self.e_i = sum(w * self.H_i[i][0] for i, w in enumerate(ws))

    def compute_mutation(self):
        self.mu_i = 0.15 * (1.0 - self.e_i) + 0.03

    def mutate(self):
        return SubAgent(self.dim,
            [max(0.0, min(1.0, p + random.gauss(0, self.mu_i))) for p in self.params])


def combine_subagents(macro, micro):
    """Combine Macro and Micro sub-agents into a full 20-dim weight vector."""
    full = [0.5] * DIMENSIONS
    for i, idx in enumerate(MACRO_INDICES):
        full[idx] = macro.params[i]
    for i, idx in enumerate(MICRO_INDICES):
        full[idx] = micro.params[i]
    return full


def subpop_survival(pop, archive):
    """
    Apply TD-CCEA survival logic to a sub-population:
    1. Redundancy removal (δs)
    2. Fitness pruning (δf)
    3. Archive update
    4. Breed new individuals via tournament
    """
    dim  = pop[0].dim
    md   = math.sqrt(dim)
    # Redundancy removal
    survivors = []
    for a in sorted(pop, key=lambda a: a.e_i, reverse=True):
        if not any(1.0 - euclidean_dist(a.params, u.params) / md > DELTA_S
                   for u in survivors):
            survivors.append(a)
    # Fitness pruning
    final = [a for a in survivors if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
    while len(final) < 8 and len(survivors) > len(final):
        for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
            if a not in final:
                final.append(a)
                break
    # Archive update
    for top in sorted(pop, key=lambda a: a.e_i, reverse=True)[:3]:
        ar = SubAgent(dim, top.params)
        ar.e_i, ar.H_i, ar.mu_i = top.e_i, top.H_i[:], top.mu_i
        archive.append(ar)
    archive.sort(key=lambda a: a.e_i, reverse=True)
    del archive[A_MAX:]
    # Breed
    pool    = final + archive
    new_pop = final[:]
    while len(new_pop) < N_POP:
        cands = random.sample(pool, min(3, len(pool)))
        new_pop.append(max(cands, key=lambda a: a.e_i).mutate())
    return new_pop, archive


# ═══════════════════════════════════════════════════════════════════
#  ALGORITHM 1 — CCE-TDCCEA (Original)
#  Fixed Macro/Micro decomposition with TD-CCEA on each partition.
#  Parallel evaluation: all individuals in a sub-population are
#  evaluated concurrently across N_WORKERS game servers.
# ═══════════════════════════════════════════════════════════════════
def run_ccea(pool):
    """
    Original CCE-TDCCEA with parallel evaluation.
    Each generation:
      1. Compute centroids for each sub-population
      2. Evaluate Pop A: pair each macro with top 3 micro partners (parallel)
      3. Evaluate Pop B: pair each micro with top 3 macro partners (parallel)
      4. Update trajectory scores and mutation rates
      5. Apply survival selection on each sub-population
    """
    print(f"\n{'='*60}")
    print(f"  RUNNING: CCE-TDCCEA (Original Algorithm)")
    print(f"  Pop A (Macro): {len(MACRO_INDICES)} weights")
    print(f"  Pop B (Micro): {len(MICRO_INDICES)} weights")
    print(f"  Parallel workers: {pool.n_workers}")
    print(f"{'='*60}")

    pop_a = [SubAgent(len(MACRO_INDICES)) for _ in range(N_POP)]
    pop_b = [SubAgent(len(MICRO_INDICES)) for _ in range(N_POP)]
    arc_a, arc_b   = [], []
    top_a, top_b   = pop_a[:TOP_PARTNERS], pop_b[:TOP_PARTNERS]
    best_a, best_b = pop_a[0], pop_b[0]
    convergence    = []
    t0             = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opp = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]

        # Centroids
        ca = [sum(a.params[j] for a in pop_a) / len(pop_a)
              for j in range(len(MACRO_INDICES))]
        cb = [sum(b.params[j] for b in pop_b) / len(pop_b)
              for j in range(len(MICRO_INDICES))]
        env = ca + cb

        # ── Evaluate Pop A in parallel ───────────────────────────
        # For each macro agent, we evaluate against each of the top micro partners
        eval_tasks_a = []
        for a in pop_a:
            for pb in top_b:
                eval_tasks_a.append(combine_subagents(a, pb))
        fit_a_flat = pool.eval_batch(eval_tasks_a, opp)
        idx = 0
        for a in pop_a:
            fs = fit_a_flat[idx:idx + len(top_b)]
            idx += len(top_b)
            avg_f = sum(fs) / len(fs)
            a.update_history(avg_f, env)
            a.compute_trajectory(env)
            a.compute_mutation()

        # ── Evaluate Pop B in parallel ───────────────────────────
        eval_tasks_b = []
        for b in pop_b:
            for pa in top_a:
                eval_tasks_b.append(combine_subagents(pa, b))
        fit_b_flat = pool.eval_batch(eval_tasks_b, opp)
        idx = 0
        for b in pop_b:
            fs = fit_b_flat[idx:idx + len(top_a)]
            idx += len(top_a)
            avg_f = sum(fs) / len(fs)
            b.update_history(avg_f, env)
            b.compute_trajectory(env)
            b.compute_mutation()

        best_a = max(pop_a, key=lambda a: a.e_i)
        best_b = max(pop_b, key=lambda a: a.e_i)
        top_a  = sorted(pop_a, key=lambda a: a.e_i, reverse=True)[:TOP_PARTNERS]
        top_b  = sorted(pop_b, key=lambda a: a.e_i, reverse=True)[:TOP_PARTNERS]

        # Evaluate best combination
        cmb_score = pool.eval_batch(
            [combine_subagents(best_a, best_b)], opp
        )[0]
        convergence.append(cmb_score)

        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | "
              f"A={best_a.e_i:.3f} B={best_b.e_i:.3f} Cmb={cmb_score:.3f}")

        pop_a, arc_a = subpop_survival(pop_a, arc_a)
        pop_b, arc_b = subpop_survival(pop_b, arc_b)

    best_weights = combine_subagents(best_a, best_b)
    elapsed      = time.time() - t0
    return best_weights, elapsed, convergence


# ═══════════════════════════════════════════════════════════════════
#  MONOLITHIC AGENT  (for DI-CCE-TDCCEA dynamic groups)
# ═══════════════════════════════════════════════════════════════════
class Agent:
    """Full 20-dimensional agent with trajectory-driven evaluation."""
    def __init__(self, params=None):
        self.params = ([random.random() for _ in range(DIMENSIONS)] if params is None
                       else [max(0.0, min(1.0, float(x))) for x in params])
        self.H_i  = []
        self.e_i  = 0.0
        self.mu_i = 0.0

    def update_history(self, f, env):
        self.H_i.append((f, env))
        if len(self.H_i) > TRAJECTORY_WINDOW:
            self.H_i.pop(0)

    def compute_trajectory(self, env):
        if not self.H_i:
            self.e_i = 0.0
            return
        md = math.sqrt(DIMENSIONS)
        ws = [max(0.0, 1.0 - euclidean_dist(env, pe) / md) for _, pe in self.H_i]
        t  = sum(ws)
        ws = [w / t for w in ws] if t > 0 else [1.0 / len(ws)] * len(ws)
        self.e_i = sum(w * self.H_i[i][0] for i, w in enumerate(ws))

    def compute_mutation(self):
        self.mu_i = 0.15 * (1.0 - self.e_i) + 0.03

    def mutate(self):
        cp = [max(0.0, min(1.0, p + random.gauss(0, self.mu_i)))
              for p in self.params]
        return Agent(cp)

    def mutate_indices(self, indices, sigma):
        """Mutate only the specified parameter indices with given sigma."""
        cp = self.params[:]
        for j in indices:
            cp[j] = max(0.0, min(1.0, cp[j] + random.gauss(0, sigma)))
        child = Agent(cp)
        child.H_i  = [h for h in self.H_i]
        child.e_i  = self.e_i
        child.mu_i = self.mu_i
        return child

    def clone(self):
        c = Agent(self.params[:])
        c.H_i  = [h for h in self.H_i]
        c.e_i  = self.e_i
        c.mu_i = self.mu_i
        return c


# ═══════════════════════════════════════════════════════════════════
#  MAXIMUM SPANNING TREE  (Prim's algorithm)
# ═══════════════════════════════════════════════════════════════════
def build_mst(interaction_matrix, n):
    """
    Build a Maximum Spanning Tree from the interaction matrix using
    Prim's algorithm. Returns a list of edges [(i, j, weight), ...].
    """
    if n <= 1:
        return []
    in_mst  = [False] * n
    key     = [-float('inf')] * n
    parent  = [-1] * n
    key[0]  = 0.0
    edges   = []

    for _ in range(n):
        # Pick vertex with maximum key value not yet in MST
        u = -1
        for v in range(n):
            if not in_mst[v] and (u == -1 or key[v] > key[u]):
                u = v
        if u == -1:
            break
        in_mst[u] = True
        if parent[u] != -1:
            w = interaction_matrix[parent[u]][u]
            edges.append((parent[u], u, w))
        # Update keys of adjacent vertices
        for v in range(n):
            if not in_mst[v] and interaction_matrix[u][v] > key[v]:
                key[v]    = interaction_matrix[u][v]
                parent[v] = u
    return edges


def extract_dynamic_groups(mst_edges, n, tau_i):
    """
    Extract temporary strategy clusters from strong MST pathways.
    Edges with weight ≥ τI are kept; connected components form the groups.
    Parameters not in any strong group form singleton groups.
    """
    # Filter strong edges
    strong_edges = [(i, j) for i, j, w in mst_edges if w >= tau_i]

    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in strong_edges:
        union(i, j)

    # Collect groups
    groups_map = defaultdict(list)
    for i in range(n):
        groups_map[find(i)].append(i)

    groups = list(groups_map.values())
    return groups


# ═══════════════════════════════════════════════════════════════════
#  ENVIRONMENTAL FINGERPRINT
# ═══════════════════════════════════════════════════════════════════
def compute_env_fingerprint(population, opp_idx, gen, best_fitness, avg_fitness):
    """
    Construct environmental fingerprint Et = [Rpop, Ropp, Rgame, Rfit]
    as described in Algorithm 2 of the PDF.

    Components:
      Rpop  — population centroid + diversity (std dev of params)
      Ropp  — opponent index (normalised) + opponent rotation position
      Rgame — generation progress (normalised)
      Rfit  — best fitness, avg fitness, fitness variance
    """
    n_agents = len(population)

    # Rpop: centroid
    centroid = [sum(a.params[j] for a in population) / n_agents
                for j in range(DIMENSIONS)]
    # Rpop: diversity (avg std dev across dimensions)
    diversity = []
    for j in range(DIMENSIONS):
        mean_j = centroid[j]
        var_j  = sum((a.params[j] - mean_j) ** 2 for a in population) / n_agents
        diversity.append(math.sqrt(var_j))
    avg_div = sum(diversity) / len(diversity) if diversity else 0.0

    # Ropp
    opp_norm = opp_idx / max(1, len(OPPONENT_ROTATION) - 1)

    # Rgame
    gen_norm = gen / TOTAL_GENERATIONS

    # Rfit
    fit_values = [a.e_i for a in population]
    fit_var = sum((f - avg_fitness) ** 2 for f in fit_values) / max(1, n_agents)

    # Combine into fingerprint vector
    env = centroid + [avg_div, opp_norm, gen_norm, best_fitness, avg_fitness, fit_var]
    return env


# ═══════════════════════════════════════════════════════════════════
#  ALGORITHM 2 — DI-CCE-TDCCEA (NEW — from algo_modification_fyp.pdf)
#
#  Dynamic Interaction-Driven Trajectory-Driven Competitive
#  Co-Evolutionary Algorithm
#
#  Key differences from original CCE-TDCCEA:
#  1. No fixed Macro/Micro split — decomposition is LEARNED
#  2. Strategy-interaction matrix I tracks pairwise parameter synergy
#  3. Maximum Spanning Tree identifies strongest relationships
#  4. Temporary co-evolutionary groups change every few generations
#  5. Co-evolution windows have min/max duration + improvement check
# ═══════════════════════════════════════════════════════════════════

def run_di_cce_tdccea(pool):
    """
    DI-CCE-TDCCEA implementation following Algorithm 2 from the PDF.

    Adaptive cycle each generation:
      1. Construct environmental fingerprint Et
      2. Evaluate all agents using K simulated games (parallel)
      3. Update trajectory memories Hi
      4. Measure pairwise strategy interactions → update matrix I
      5. Smooth I with EMA
      6. Build Maximum Spanning Tree Tt from I
      7. Extract temporary strategy clusters G1..Gm
      8. For each cluster Gk:
         a. Select candidate sub-agents
         b. Select complementary partners from other clusters
         c. Evaluate candidates across partner contexts
         d. Breed offspring by mutating only Gk parameters
      9. Evaluate full population after all co-evolution rounds
     10. Check joint-improvement for window termination
     11. Apply survival selection (redundancy + fitness pruning)
     12. Update archive
    """
    print(f"\n{'='*60}")
    print(f"  RUNNING: DI-CCE-TDCCEA (Dynamic Interaction-Driven)")
    print(f"  Decomposition: DYNAMIC (learned from interaction matrix)")
    print(f"  MST threshold τI = {TAU_I}")
    print(f"  Window: [{L_MIN}, {L_MAX}] gens, τimp = {TAU_IMP}")
    print(f"  Parallel workers: {pool.n_workers}")
    print(f"{'='*60}")

    # ── Step 2: Initialize population ─────────────────────────────
    population = [Agent() for _ in range(N_POP)]
    archive    = []
    convergence = []
    t0         = time.time()

    # ── Step 4: Initialize strategy-interaction matrix I ──────────
    # I[i][j] = estimated interaction strength between parameter i and j
    interaction_matrix = [[0.0] * DIMENSIONS for _ in range(DIMENSIONS)]
    # Initialize with a small prior based on known domain knowledge
    # (macro-macro and micro-micro pairs get small positive values)
    for i in MACRO_INDICES:
        for j in MACRO_INDICES:
            if i != j:
                interaction_matrix[i][j] = 0.05
    for i in MICRO_INDICES:
        for j in MICRO_INDICES:
            if i != j:
                interaction_matrix[i][j] = 0.05

    # ── Step 5: Initialize environment memory ─────────────────────
    env_memory = []

    # Active group windows: {group_key: {"indices": [...], "start_gen": int,
    #                                     "prev_fitness": float}}
    active_windows = {}
    current_groups = None   # will be set after first MST extraction

    # Previous generation's per-agent fitness (for joint improvement tracking)
    prev_fitnesses = [0.0] * N_POP

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opp_idx  = (gen - 1) % len(OPPONENT_ROTATION)
        opp      = OPPONENT_ROTATION[opp_idx]

        # ── Step 7: Environmental fingerprint ─────────────────────
        avg_fit  = sum(a.e_i for a in population) / len(population)
        best_fit = max(a.e_i for a in population) if gen > 1 else 0.0
        env_fp   = compute_env_fingerprint(
            population, opp_idx, gen, best_fit, avg_fit
        )

        # ── Step 8-9: Evaluate all agents (parallel) ──────────────
        all_weights = [a.params for a in population]
        fitnesses   = pool.eval_batch(all_weights, opp)

        # ── Step 9: Update trajectory memories ────────────────────
        centroid = [sum(a.params[j] for a in population) / len(population)
                    for j in range(DIMENSIONS)]
        for i, agent in enumerate(population):
            agent.update_history(fitnesses[i], centroid)
            agent.compute_trajectory(centroid)
            agent.compute_mutation()

        # Store environment memory
        env_memory.append(env_fp)
        if len(env_memory) > TRAJECTORY_WINDOW * 2:
            env_memory.pop(0)

        # ── Step 10: Measure pairwise strategy interactions ───────
        # For each pair (p, q) of parameters, estimate their interaction
        # by measuring correlation of per-agent joint perturbation effects.
        #
        # Method: for a sample of high-performing agents, measure how
        # correlated their parameter values are with fitness improvements
        # over the previous generation.
        fitness_deltas = [fitnesses[i] - prev_fitnesses[i]
                          for i in range(len(population))]

        # Update interaction matrix using observed joint-fitness changes
        # Agents sorted by fitness improvement
        sorted_agents = sorted(range(len(population)),
                               key=lambda i: fitness_deltas[i], reverse=True)
        top_improvers = sorted_agents[:max(3, N_POP // 4)]

        if gen > 1:
            for i_param in range(DIMENSIONS):
                for j_param in range(i_param + 1, DIMENSIONS):
                    # Measure co-movement: when both params are similar in
                    # top improvers vs bottom performers
                    top_vals_i = [population[k].params[i_param] for k in top_improvers]
                    top_vals_j = [population[k].params[j_param] for k in top_improvers]

                    bot_improvers = sorted_agents[-(max(3, N_POP // 4)):]
                    bot_vals_i = [population[k].params[i_param] for k in bot_improvers]
                    bot_vals_j = [population[k].params[j_param] for k in bot_improvers]

                    # Compute correlation-like interaction signal
                    mean_ti = sum(top_vals_i) / len(top_vals_i)
                    mean_tj = sum(top_vals_j) / len(top_vals_j)
                    mean_bi = sum(bot_vals_i) / len(bot_vals_i)
                    mean_bj = sum(bot_vals_j) / len(bot_vals_j)

                    # Joint deviation: how much do top improvers' params
                    # move together compared to bottom?
                    top_cov = sum(
                        (top_vals_i[k] - mean_ti) * (top_vals_j[k] - mean_tj)
                        for k in range(len(top_improvers))
                    ) / len(top_improvers)

                    # Fitness-weighted interaction strength
                    delta_signal = abs(top_cov) * (
                        sum(fitness_deltas[k] for k in top_improvers) /
                        max(0.01, sum(abs(fitness_deltas[k]) for k in range(len(population))))
                    )

                    # ── Step 12: Smooth I with EMA ────────────────
                    interaction_matrix[i_param][j_param] = (
                        INTERACTION_DECAY * interaction_matrix[i_param][j_param]
                        + (1 - INTERACTION_DECAY) * delta_signal
                    )
                    interaction_matrix[j_param][i_param] = (
                        interaction_matrix[i_param][j_param]
                    )

        # ── Step 13: Build Maximum Spanning Tree ──────────────────
        mst_edges = build_mst(interaction_matrix, DIMENSIONS)

        # ── Step 14: Extract temporary strategy clusters ──────────
        new_groups = extract_dynamic_groups(mst_edges, DIMENSIONS, TAU_I)

        # Merge very small groups (singletons) into nearest strong group
        merged_groups = []
        small_params  = []
        for g in new_groups:
            if len(g) >= 2:
                merged_groups.append(g)
            else:
                small_params.extend(g)

        # Assign singleton params to their strongest neighbour's group
        if small_params and merged_groups:
            for sp in small_params:
                best_group_idx = 0
                best_strength  = -1.0
                for gi, group in enumerate(merged_groups):
                    strength = max(interaction_matrix[sp][gp] for gp in group)
                    if strength > best_strength:
                        best_strength  = strength
                        best_group_idx = gi
                merged_groups[best_group_idx].append(sp)
        elif small_params and not merged_groups:
            # No strong groups at all — fall back to 2-way split
            merged_groups = [
                small_params[:len(small_params)//2],
                small_params[len(small_params)//2:]
            ]
            if not merged_groups[0]:
                merged_groups = [small_params]

        # Ensure no empty groups
        merged_groups = [g for g in merged_groups if g]
        if not merged_groups:
            merged_groups = [list(range(DIMENSIONS))]

        current_groups = merged_groups

        # ── Step 15-23: Co-evolve within each dynamic group ───────
        group_fitnesses = {}  # track per-group fitness for window mgmt

        for gk_idx, gk_indices in enumerate(current_groups):
            gk_key = tuple(sorted(gk_indices))

            # Select candidate sub-agents (those performing best on
            # the dimensions belonging to this group)
            # Score each agent by how well their Gk parameters contribute
            agent_scores = []
            for ai, agent in enumerate(population):
                # Use trajectory score as primary, with a bonus for diversity
                # in the Gk parameter space
                gk_params = [agent.params[j] for j in gk_indices]
                agent_scores.append((ai, agent.e_i))

            agent_scores.sort(key=lambda x: x[1], reverse=True)

            # Select top partners from other groups (complementary context)
            # — use the best agent's non-Gk parameters as context
            best_agent = population[agent_scores[0][0]]

            # ── Breed offspring by mutating only Gk parameters ────
            # Adaptive mutation scale: σ = μ_base * (1 - e_i) + ε
            n_offspring = max(2, N_POP // len(current_groups))
            new_agents  = []

            for _ in range(n_offspring):
                # Tournament selection (k=3) from current population
                tournament = random.sample(range(len(population)),
                                           min(3, len(population)))
                parent_idx = max(tournament, key=lambda i: population[i].e_i)
                parent     = population[parent_idx]

                # Compute group-specific mutation rate
                sigma = parent.mu_i  # base adaptive rate from trajectory

                # Create offspring: mutate ONLY the Gk indices
                child = parent.mutate_indices(gk_indices, sigma)
                new_agents.append(child)

            # Evaluate offspring (parallel)
            offspring_weights = [a.params for a in new_agents]
            offspring_fits    = pool.eval_batch(offspring_weights, opp)

            # Update offspring trajectory data
            for oi, child in enumerate(new_agents):
                child.update_history(offspring_fits[oi], centroid)
                child.compute_trajectory(centroid)
                child.compute_mutation()

            # ── Step 22: Insert improved offspring ────────────────
            # Replace weakest current agents if offspring are better
            pop_sorted = sorted(range(len(population)),
                                key=lambda i: population[i].e_i)
            replaceable = pop_sorted[:len(new_agents)]

            for oi, child in enumerate(new_agents):
                if oi < len(replaceable):
                    target_idx = replaceable[oi]
                    if child.e_i > population[target_idx].e_i:
                        population[target_idx] = child

            # Track group fitness for window management
            gk_fit = sum(population[i].e_i for i in range(len(population))) / len(population)
            group_fitnesses[gk_key] = gk_fit

        # ── Step 25-26: Re-evaluate full population ───────────────
        # (Already evaluated via offspring insertion above; compute overall best)
        all_weights_post = [a.params for a in population]
        post_fitnesses   = pool.eval_batch(all_weights_post, opp)
        for i, agent in enumerate(population):
            agent.update_history(post_fitnesses[i], centroid)
            agent.compute_trajectory(centroid)
            agent.compute_mutation()

        # ── Step 27-34: Window management (joint improvement check) ─
        for gk_idx, gk_indices in enumerate(current_groups):
            gk_key = tuple(sorted(gk_indices))
            current_fit = group_fitnesses.get(gk_key, 0.0)

            if gk_key in active_windows:
                window = active_windows[gk_key]
                window_len = gen - window["start_gen"]
                improvement = current_fit - window["prev_fitness"]

                if improvement < TAU_IMP and window_len >= L_MIN:
                    # Terminate window: interaction not productive enough
                    del active_windows[gk_key]
                elif window_len >= L_MAX:
                    # Maximum window reached: force termination
                    del active_windows[gk_key]
                else:
                    # Continue window
                    window["prev_fitness"] = current_fit
            else:
                # Start new window for this group
                active_windows[gk_key] = {
                    "indices":      gk_indices,
                    "start_gen":    gen,
                    "prev_fitness": current_fit,
                }

        # ── Step 35-37: Update interaction matrix from new outcomes ─
        for gk_indices in current_groups:
            for i_p in gk_indices:
                for j_p in gk_indices:
                    if i_p != j_p:
                        # Reinforce interactions within successful groups
                        gk_key = tuple(sorted(gk_indices))
                        gk_fit = group_fitnesses.get(gk_key, 0.0)
                        reinforce = 0.01 * max(0, gk_fit - avg_fit)
                        interaction_matrix[i_p][j_p] = min(
                            1.0,
                            interaction_matrix[i_p][j_p] + reinforce
                        )

        # Track best for convergence
        best = max(population, key=lambda a: a.e_i)
        best_eval = pool.eval_batch([best.params], opp)[0]
        convergence.append(best_eval)

        # Display group decomposition
        group_str = " | ".join(
            f"G{gi+1}({len(g)}d)" for gi, g in enumerate(current_groups)
        )
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | "
              f"e_i={best.e_i:.3f} | Groups: [{group_str}]")

        prev_fitnesses = post_fitnesses[:]

        # ── Step 39-43: Survival selection ────────────────────────
        md = math.sqrt(DIMENSIONS)
        # Redundancy removal
        survivors = []
        for a in sorted(population, key=lambda a: a.e_i, reverse=True):
            if not any(1.0 - euclidean_dist(a.params, u.params) / md > DELTA_S
                       for u in survivors):
                survivors.append(a)
        # Fitness pruning
        final = [a for a in survivors
                 if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
        while len(final) < 10 and len(survivors) > len(final):
            for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
                if a not in final:
                    final.append(a)
                    break

        # ── Step 44-47: Archive update ────────────────────────────
        for top in sorted(population, key=lambda a: a.e_i, reverse=True)[:3]:
            ar = top.clone()
            archive.append(ar)
        archive.sort(key=lambda a: a.e_i, reverse=True)
        del archive[A_MAX:]

        # Breed to refill population
        breed_pool = final + archive
        population = final[:]
        while len(population) < N_POP:
            cands = random.sample(breed_pool, min(3, len(breed_pool)))
            population.append(max(cands, key=lambda a: a.e_i).mutate())

    # Return best from archive
    best_all = max(archive, key=lambda a: a.e_i) if archive else max(population, key=lambda a: a.e_i)
    elapsed  = time.time() - t0

    # Print final decomposition info
    if current_groups:
        print(f"\n  Final dynamic decomposition:")
        for gi, g in enumerate(current_groups):
            names = [WEIGHT_NAMES[j] for j in g]
            print(f"    Group {gi+1} ({len(g)} params): {', '.join(names)}")

    return best_all.params, elapsed, convergence


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("\n" + "#" * 70)
    print("  DI-CCE-TDCCEA vs CCE-TDCCEA — Dynamic vs Fixed Decomposition")
    print("#" * 70)
    print(f"  Pop={N_POP} | Gens={TOTAL_GENERATIONS} | K={K_GAMES} | "
          f"Validation={VALIDATION_GAMES}/bot")
    print(f"  Map: {MAP_PATH} (32×32)")
    print(f"  CPU cores: {os.cpu_count()} | Workers: {N_WORKERS}")
    print()

    algos = [
        ("CCE-TDCCEA",    "ccea"),
        ("DI-CCE-TDCCEA", "di_ccea"),
    ]

    results = []
    for name, key in algos:
        # Start a fresh parallel pool for each algorithm
        pool = ParallelGamePool(N_WORKERS)
        pool.start()

        if key == "ccea":
            best, elapsed, convergence = run_ccea(pool)
        else:
            best, elapsed, convergence = run_di_cce_tdccea(pool)

        print(f"\n  Validating {name} against all 7 bots...")
        wr, per_opp = pool.validate_weights(best)
        pool.stop()

        print(f"  {name}: {wr:.1f}% overall in {elapsed:.1f}s")
        results.append({
            "name": name, "key": key, "time": elapsed,
            "wr": wr, "per_opp": per_opp, "weights": best,
            "convergence": convergence,
        })

    # ═══════════════════════════════════════════════════════════════
    #  RANKING TABLE
    # ═══════════════════════════════════════════════════════════════
    ranked = sorted(results, key=lambda x: x["wr"], reverse=True)
    medals = ["1st", "2nd"]

    print(f"\n{'='*70}")
    print(f"  FINAL RANKING — Win Rate vs 7 Bots")
    print(f"{'='*70}")
    print(f"  {'Rank':<6s} {'Algorithm':<18s} {'Win Rate':>10s} {'Time':>8s}")
    print(f"  {'_'*6} {'_'*18} {'_'*10} {'_'*8}")
    for i, r in enumerate(ranked):
        tag = ""
        if r["key"] == "ccea":    tag = "  ◄ Original"
        if r["key"] == "di_ccea": tag = "  ◄ NEW (Dynamic Interaction)"
        print(f"  {medals[i]:<6s} {r['name']:<18s} {r['wr']:>9.1f}%  "
              f"{r['time']:>7.1f}s{tag}")

    # ═══════════════════════════════════════════════════════════════
    #  PER-BOT TABLE with best-marker (*)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  WIN RATE PER BOT (%)")
    print(f"{'='*70}")
    header = f"  {'Bot':<26s}"
    for r in ranked:
        header += f" {r['name'][:14]:>15s}"
    print(header)
    print("  " + "-" * (26 + 16 * len(ranked)))
    for opp in OPPONENT_ROTATION:
        row   = f"  {opp:<26s}"
        rates = [r["per_opp"].get(opp, 0) for r in ranked]
        mx    = max(rates)
        for r in ranked:
            w  = r["per_opp"].get(opp, 0)
            mk = "*" if w == mx and rates.count(mx) == 1 else " "
            row += f" {w:>13.0f}%{mk}"
        print(row)
    print("  " + "-" * (26 + 16 * len(ranked)))
    row = f"  {'OVERALL':<26s}"
    for r in ranked:
        row += f" {r['wr']:>14.1f}% "
    print(row)

    # ═══════════════════════════════════════════════════════════════
    #  WILCOXON SIGNED-RANK TEST
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STATISTICAL SIGNIFICANCE — Wilcoxon Signed-Rank Test")
    print(f"  (DI-CCE-TDCCEA vs CCE-TDCCEA, per-bot win rates as samples)")
    print(f"{'='*70}")

    cce = next((r for r in results if r["key"] == "ccea"), None)
    di  = next((r for r in results if r["key"] == "di_ccea"), None)

    w_out = [f"\nRun Date: {time.strftime('%Y-%m-%d %H:%M:%S')}"]

    if cce and di:
        cce_rates = [cce["per_opp"].get(o, 0) for o in OPPONENT_ROTATION]
        di_rates  = [di["per_opp"].get(o, 0)  for o in OPPONENT_ROTATION]
        p = wilcoxon_test(di_rates, cce_rates)
        if p is None:
            line = f"  p-value: N/A  (insufficient non-tied pairs)"
        else:
            sig  = "YES (p<0.05)" if p < 0.05 else "NO  (p>=0.05)"
            dirn = ("DI-CCE-TDCCEA BETTER" if sum(di_rates) > sum(cce_rates)
                    else "CCE-TDCCEA BETTER")
            line = (f"  p-value: {p:.4f}  |  Significant: {sig}  |  "
                    f"Direction: {dirn}")
        print(f"\n  {line}")
        w_out.append(line)

        # Detailed per-bot comparison
        print(f"\n  {'Bot':<26s} {'CCE-TDCCEA':>12s} {'DI-CCE-TDCCEA':>14s} {'Δ':>8s}")
        print(f"  {'_'*26} {'_'*12} {'_'*14} {'_'*8}")
        for opp in OPPONENT_ROTATION:
            c_wr = cce["per_opp"].get(opp, 0)
            d_wr = di["per_opp"].get(opp, 0)
            delta = d_wr - c_wr
            sign  = "+" if delta >= 0 else ""
            print(f"  {opp:<26s} {c_wr:>11.0f}% {d_wr:>13.0f}% {sign}{delta:>6.0f}%")

    w_path = os.path.join(SCRIPT_DIR, "di_cce_tdccea_wilcoxon.txt")
    with open(w_path, "a", encoding="utf-8") as f:
        f.write("\n".join(w_out) + "\n\n")

    # ═══════════════════════════════════════════════════════════════
    #  ABLATION ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  ABLATION ANALYSIS")
    print(f"{'='*70}")

    abl_out = [f"\nRun Date: {time.strftime('%Y-%m-%d %H:%M:%S')}"]

    if cce and di:
        d = di["wr"] - cce["wr"]
        lines = [
            f"\n  Dynamic decomposition contribution:",
            f"    DI-CCE-TDCCEA ({di['wr']:.1f}%) vs CCE-TDCCEA ({cce['wr']:.1f}%)",
            f"    Result: {'Dynamic Decomp HELPS (+' if d >= 0 else 'Dynamic Decomp HURTS ('}{abs(d):.1f}%)",
            f"",
            f"  Time comparison:",
            f"    DI-CCE-TDCCEA: {di['time']:.1f}s",
            f"    CCE-TDCCEA:    {cce['time']:.1f}s",
            f"    Overhead: {di['time'] - cce['time']:+.1f}s "
            f"({(di['time']/max(0.1, cce['time']) - 1)*100:+.1f}%)",
        ]
        for line in lines:
            print(line)
        abl_out.extend(lines)

    abl_path = os.path.join(SCRIPT_DIR, "di_cce_tdccea_ablation.txt")
    with open(abl_path, "a", encoding="utf-8") as f:
        f.write("\n".join(abl_out) + "\n\n")

    # ═══════════════════════════════════════════════════════════════
    #  CONVERGENCE COMPARISON
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  CONVERGENCE COMPARISON (Best fitness per generation)")
    print(f"{'='*70}")
    print(f"  {'Gen':>4s}  {'CCE-TDCCEA':>12s}  {'DI-CCE-TDCCEA':>14s}")
    print(f"  {'_'*4}  {'_'*12}  {'_'*14}")
    max_gens = min(len(results[0]["convergence"]), len(results[1]["convergence"]))
    for g in range(max_gens):
        c_val = results[0]["convergence"][g] if g < len(results[0]["convergence"]) else 0
        d_val = results[1]["convergence"][g] if g < len(results[1]["convergence"]) else 0
        print(f"  {g+1:4d}  {c_val:>12.3f}  {d_val:>14.3f}")

    # ═══════════════════════════════════════════════════════════════
    #  SAVE JSON
    # ═══════════════════════════════════════════════════════════════
    out_path = os.path.join(SCRIPT_DIR, "di_cce_tdccea_results.json")
    with open(out_path, "w") as f:
        json.dump([{
            "name":             r["name"],
            "key":              r["key"],
            "overall_winrate":  r["wr"],
            "per_opponent":     r["per_opp"],
            "time":             r["time"],
            "convergence":      r["convergence"],
        } for r in results], f, indent=2)
    print(f"\n  Results saved → {out_path}")

    # ═══════════════════════════════════════════════════════════════
    #  EXCEL EXPORT
    # ═══════════════════════════════════════════════════════════════
    if _OPENPYXL_AVAILABLE:
        xlsx_path = os.path.join(SCRIPT_DIR, "di_cce_tdccea_comparison.xlsx")

        algo_names = [r["name"] for r in results]
        algo_wr    = {r["name"]: round(r["wr"],   2) for r in results}
        algo_time  = {r["name"]: round(r["time"], 1) for r in results}

        if not os.path.exists(xlsx_path):
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Win Rates"
            ws.append(["Algorithm", "Run1", "Time1"])
            for algo in algo_names:
                ws.append([algo, algo_wr[algo], algo_time[algo]])
            wb.save(xlsx_path)
            run_col_name  = "Run1"
            time_col_name = "Time1"
        else:
            wb = openpyxl.load_workbook(xlsx_path)
            ws = wb.active
            header_row  = [cell.value for cell in ws[1]]
            run_numbers = [
                int(h[3:]) for h in header_row
                if isinstance(h, str) and h.lower().startswith("run")
                and h[3:].isdigit()
            ]
            next_run      = (max(run_numbers) + 1) if run_numbers else 1
            run_col_name  = f"Run{next_run}"
            time_col_name = f"Time{next_run}"

            run_col_idx  = len(header_row) + 1
            time_col_idx = len(header_row) + 2
            ws.cell(row=1, column=run_col_idx,  value=run_col_name)
            ws.cell(row=1, column=time_col_idx, value=time_col_name)

            algo_row_map = {}
            for row_idx in range(2, ws.max_row + 1):
                cell_val = ws.cell(row=row_idx, column=1).value
                if cell_val:
                    algo_row_map[cell_val] = row_idx

            for algo in algo_names:
                if algo in algo_row_map:
                    ri = algo_row_map[algo]
                else:
                    ri = ws.max_row + 1
                    ws.cell(row=ri, column=1, value=algo)
                ws.cell(row=ri, column=run_col_idx,  value=algo_wr[algo])
                ws.cell(row=ri, column=time_col_idx, value=algo_time[algo])

            wb.save(xlsx_path)

        print(f"  Excel updated  → {xlsx_path}  "
              f"(columns: {run_col_name}, {time_col_name})")
    else:
        print("  [SKIP] Excel export skipped — install openpyxl to enable.")

    print(f"\n{'='*70}")
    print(f"  BENCHMARK COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
