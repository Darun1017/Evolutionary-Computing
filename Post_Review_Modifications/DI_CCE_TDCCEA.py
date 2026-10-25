"""
DI_CCE_TDCCEA.py — Dynamic Interaction-Driven CCE-TDCCEA
                    Trajectory-Driven Competitive Co-Evolutionary Algorithm

Implements Algorithm 2 (DI-CCE-TDCCEA) from the post-review modifications.

Key changes from the original CCE-TDCCEA:
  1. Fixed Macro/Micro decomposition → dynamic interaction-driven decomposition
  2. Static 2-group split → MST-based temporary strategy clusters G_1, …, G_m
  3. Independent sub-pop evolution → per-cluster co-evolution with complementary partners
  4. Permanent groups → adaptive co-evolution window lifecycle management

Adaptive cycle:
  E_t → Evaluate → Interaction Matrix → Maximum Spanning Tree
      → Dynamic Groups → Co-evolve → Evaluate

Compares:
  1. DI-CCE-TDCCEA   (NEW: dynamic interaction-driven co-evolution)
  2. TD-CCEA Centroid (standard monolithic baseline)
  3. Hill Climbing    (simple baseline)
"""

import math, random, time, subprocess, os, json

os.environ["PYTHONIOENCODING"] = "utf-8"

# ===============================================================
#  CONFIGURATION
# ===============================================================
MICRORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "Simulation", "microrts")

WEIGHT_NAMES = [
    "worker_ratio", "max_workers", "expand_threshold",
    "light_preference", "heavy_preference", "ranged_preference",
    "barracks_timing", "aggression", "attack_threshold",
    "defend_radius", "target_priority", "defense_trigger",
    "counter_rush", "tech_switch", "economy_pressure",
    "scout_priority", "group_attack", "second_barracks",
    "worker_fight_ratio", "focus_fire",
]
DIMENSIONS = len(WEIGHT_NAMES)

MAP_PATH   = "maps/chambers32x32.xml"
MAX_CYCLES = 10000

OPPONENT_ROTATION = [
    "EconomyMilitaryRush", "LightRush", "WorkerRush",
    "HeavyDefense", "RangedRush", "EconomyRush", "WorkerRushPlusPlus",
]

# Population and evolution parameters (Algorithm 2, line 1)
N_POP       = 20      # Population size N
W           = 7       # Trajectory window
K_GAMES     = 3       # Simulated games per evaluation
TOTAL_GENS  = 30      # T_max
VALID_GAMES = 5       # Games per opponent for final validation

# TD-CCEA parameters (Algorithm 2, line 1)
MU_BASE  = 0.15       # Base mutation rate
EPSILON  = 0.03       # Minimum mutation offset
DELTA_F  = 0.20       # Fitness threshold δ_f
DELTA_S  = 0.85       # Redundancy threshold δ_s
A_MAX    = 15         # Archive size

# DI-CCE-TDCCEA specific parameters (Algorithm 2)
TAU_I    = 0.15       # Interaction threshold τ_I (MST group extraction; calibrated to the
                      # pooled partial-correlation scale of I — retune on real runs)
L_MIN    = 3          # Minimum co-evolution window L_min
L_MAX    = 10         # Maximum co-evolution window L_max
TAU_IMP  = 0.01       # Joint-improvement threshold τ_imp
MAX_GROUP    = 5      # Cap on a temporary strategy cluster's size
SUBPOP_SIZE  = 6      # Sub-agents kept per temporary cluster sub-population
N_OFFSPRING  = 3      # Offspring bred per cluster per generation

# Environment fingerprint dimensionality
ENV_DIM  = 10


# ===============================================================
#  GAME SERVER
# ===============================================================
class GameServer:
    """Manages the microRTS Java process for game evaluation."""

    def __init__(self):
        self.process = None

    def start(self):
        cmd = ["java", "-cp", "lib/*;bin", "tests.EvolvedBatchGameRunner"]
        self.process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=MICRORTS_DIR,
        )
        line = self.process.stdout.readline().strip()
        if line != "READY":
            raise RuntimeError(f"Failed to start: {line}")

    def run_game(self, weights, opponent):
        weights_str = ";".join(f"{w:.4f}" for w in weights)
        request = f"EVOLVED:{weights_str},{opponent},{MAP_PATH},{MAX_CYCLES}\n"
        self.process.stdin.write(request)
        self.process.stdin.flush()
        response = self.process.stdout.readline().strip()
        if response.startswith("RESULT:"):
            parts = response.replace("RESULT:", "").split(",")
            return int(parts[0]), int(parts[1])
        return -1, MAX_CYCLES

    def stop(self):
        if self.process:
            try:
                self.process.stdin.write("EXIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()


# ===============================================================
#  HELPERS
# ===============================================================
def euclidean_dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine_similarity(a, b):
    """Cosine similarity between two vectors, clamped to [0, 1]."""
    dot   = sum(x * y for x, y in zip(a, b))
    nrm_a = math.sqrt(sum(x * x for x in a))
    nrm_b = math.sqrt(sum(x * x for x in b))
    if nrm_a == 0 or nrm_b == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (nrm_a * nrm_b)))


def evaluate_weights(server, weights, opponent):
    """Play K_GAMES against an opponent, return win rate 0.0–1.0."""
    wins = 0
    for _ in range(K_GAMES):
        winner, _ = server.run_game(weights, opponent)
        if winner == 0:
            wins += 1
        elif winner == -1:
            wins += 0.3
    return wins / K_GAMES


def evaluate_once(server, weights, opponent):
    """Play ONE game; 1.0 = win, 0.3 = timeout, 0.0 = loss."""
    winner, _ = server.run_game(weights, opponent)
    return 1.0 if winner == 0 else (0.3 if winner == -1 else 0.0)


def validate_weights(server, weights):
    """Test weights against ALL opponents, return overall win rate."""
    total_wins, total_games, per_opp = 0, 0, {}
    for opp in OPPONENT_ROTATION:
        opp_wins = 0
        for _ in range(VALID_GAMES):
            winner, _ = server.run_game(weights, opp)
            if winner == 0:
                opp_wins += 1
            total_games += 1
        total_wins += opp_wins
        per_opp[opp] = opp_wins / VALID_GAMES * 100
    return total_wins / total_games * 100, per_opp


# ===============================================================
#  ENVIRONMENT FINGERPRINT  (Algorithm 2, line 7)
# ===============================================================
def construct_environment_fingerprint(population, opponent, gen,
                                      recent_fitnesses, env_memory):
    """
    Construct environmental fingerprint
        E_t = [R_pop, R_opp, R_game, R_fit]

    using population representatives, elite and Hall-of-Fame strategies,
    population diversity, opponent information, game context, and recent
    fitness characteristics.

    Returns a 10-dimensional vector.
    """
    # ── R_pop ──
    centroid = [0.0] * DIMENSIONS
    for agent in population:
        for j in range(DIMENSIONS):
            centroid[j] += agent.params[j]
    centroid = [x / len(population) for x in centroid]
    centroid_mean = sum(centroid) / DIMENSIONS

    param_stds = []
    for j in range(DIMENSIONS):
        vals = [a.params[j] for a in population]
        mu   = sum(vals) / len(vals)
        var  = sum((v - mu) ** 2 for v in vals) / len(vals)
        param_stds.append(math.sqrt(var))
    diversity = sum(param_stds) / len(param_stds)

    best_ei  = max(a.e_i for a in population) if population else 0.0
    elite_qi = (sum(e[1] for e in env_memory[-5:]) / min(5, max(len(env_memory), 1))
                if env_memory else 0.0)

    # ── R_opp ──
    opp_idx   = OPPONENT_ROTATION.index(opponent) / max(len(OPPONENT_ROTATION) - 1, 1)
    recent_wr = (sum(recent_fitnesses[-3:]) / len(recent_fitnesses[-3:])
                 if recent_fitnesses else 0.5)

    # ── R_game ──
    gen_progress = gen / TOTAL_GENS

    # ── R_fit ──
    if recent_fitnesses:
        mean_fit = sum(recent_fitnesses) / len(recent_fitnesses)
        best_fit = max(recent_fitnesses)
        fit_std  = math.sqrt(
            sum((f - mean_fit) ** 2 for f in recent_fitnesses) / len(recent_fitnesses)
        )
    else:
        mean_fit, best_fit, fit_std = 0.0, 0.0, 0.0

    return [centroid_mean, diversity, best_ei, elite_qi,
            opp_idx, recent_wr, gen_progress,
            mean_fit, best_fit, fit_std]


# ===============================================================
#  STRATEGY-INTERACTION MATRIX  (Algorithm 2, lines 4, 14-16)
# ===============================================================
class InteractionMatrix:
    """
    Strategy-interaction matrix I.  I[i][j] in [0, 1] is the smoothed strength of
    the pairwise (epistatic) interaction between parameters i and j.

    Observation  O_ij = |corr( f , z_i * z_j )|  across the current population,
    where f = observed fitness and z = standardised parameter values.  A high
    value means the joint setting of (i, j) explains fitness beyond what the
    individual settings do.  This plays the role of dF_ij.
    """

    def __init__(self, dim, alpha=0.3, window=3, pool_gens=6):
        self.dim       = dim
        self.alpha     = alpha
        self.window    = window
        self.pool_gens = pool_gens            # generations of samples pooled per observation
        self.I         = [[0.0 if i == j else 0.01 for j in range(dim)] for i in range(dim)]
        self.recent    = []                   # last `window` observation matrices
        self.samples   = []                   # per-generation [(params, centred fitness)]

    @staticmethod
    def _residual(y, a, b):
        """Residual of y after least-squares removal of a and b (and its mean)."""
        n  = len(y)
        ym = sum(y) / n
        yc = [v - ym for v in y]
        saa = sum(v * v for v in a); sbb = sum(v * v for v in b)
        sab = sum(x * w for x, w in zip(a, b))
        say = sum(x * w for x, w in zip(a, yc)); sby = sum(x * w for x, w in zip(b, yc))
        det = saa * sbb - sab * sab
        if det < 1e-9:
            return yc
        ba = (say * sbb - sby * sab) / det
        bb = (sby * saa - say * sab) / det
        return [yc[k] - ba * a[k] - bb * b[k] for k in range(n)]

    def observe(self, population, fitnesses):
        """
        Measure pairwise interactions from observed fitness outcomes.

        Samples (params, fitness) from the last `pool_gens` generations are pooled
        (fitness centred within each generation, since the opponent changes).
        For each pair the observation is the PARTIAL correlation between fitness
        and the product z_i*z_j after removing the main effects of i and j.
        """
        fm = sum(fitnesses) / len(fitnesses)
        self.samples.append([(a.params[:], f - fm) for a, f in zip(population, fitnesses)])
        if len(self.samples) > self.pool_gens:
            self.samples.pop(0)
        data = [s for gen in self.samples for s in gen]
        n = len(data)
        if n < 8:
            return None
        f_c = [d[1] for d in data]
        z = []
        for j in range(self.dim):
            col = [d[0][j] for d in data]
            mu  = sum(col) / n
            sd  = math.sqrt(sum((v - mu) ** 2 for v in col) / n)
            z.append([(v - mu) / sd if sd > 1e-9 else 0.0 for v in col])
        O = [[0.0] * self.dim for _ in range(self.dim)]
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                prod  = [z[i][k] * z[j][k] for k in range(n)]
                f_res = self._residual(f_c,  z[i], z[j])
                p_res = self._residual(prod, z[i], z[j])
                fn = math.sqrt(sum(v * v for v in f_res))
                pn = math.sqrt(sum(v * v for v in p_res))
                if fn < 1e-9 or pn < 1e-9:
                    continue
                O[i][j] = O[j][i] = min(1.0, abs(sum(x * y for x, y in zip(f_res, p_res))) / (fn * pn))
        return O

    def update(self, O):
        """I_ij <- Update(I_ij, dF_ij): exponential moving average of observations."""
        if O is None:
            return
        self.recent.append(O)
        if len(self.recent) > self.window:
            self.recent.pop(0)
        a = self.alpha
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                self.I[i][j] = (1 - a) * self.I[i][j] + a * O[i][j]
                self.I[j][i] = self.I[i][j]

    def smooth(self):
        """Smooth I using the recent interaction observations."""
        if not self.recent:
            return
        m = len(self.recent)
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                avg = sum(O[i][j] for O in self.recent) / m
                self.I[i][j] = 0.5 * self.I[i][j] + 0.5 * avg
                self.I[j][i] = self.I[i][j]


# ===============================================================
#  MAXIMUM SPANNING TREE  &  DYNAMIC GROUPING
#  (Algorithm 2, lines 17-18)
# ===============================================================
class UnionFind:
    """Union-Find (Disjoint Set Union) with component sizes, for Kruskal's MST."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank   = [0] * n
        self.size   = [1] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path-halving
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx]  += self.size[ry]
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def build_maximum_spanning_tree(interaction_matrix, dim):
    """
    Algorithm 2, line 17: Construct Maximum Spanning Tree T_t from
    the updated interaction matrix I.
    """
    edges = []
    for i in range(dim):
        for j in range(i + 1, dim):
            edges.append((interaction_matrix[i][j], i, j))
    edges.sort(reverse=True)

    uf  = UnionFind(dim)
    mst = []
    for weight, i, j in edges:
        if uf.union(i, j):
            mst.append((i, j, weight))
        if len(mst) == dim - 1:
            break
    return mst


def extract_strategy_clusters(mst_edges, dim, threshold, max_size=MAX_GROUP):
    """
    Keep MST edges with weight >= tau_I (strongest first); connected components
    are the temporary groups.  Components are capped at `max_size` so noise
    cannot chain everything into one giant group.
    """
    uf = UnionFind(dim)
    for i, j, w in sorted(mst_edges, key=lambda e: e[2], reverse=True):
        if w < threshold:
            break
        ri, rj = uf.find(i), uf.find(j)
        if ri != rj and uf.size[ri] + uf.size[rj] <= max_size:
            uf.union(i, j)

    groups = {}
    for node in range(dim):
        groups.setdefault(uf.find(node), []).append(node)
    return list(groups.values())


# ===============================================================
#  DI-CCE-TDCCEA AGENT
# ===============================================================
class DICCEAgent:
    """
    Full-strategy agent for DI-CCE-TDCCEA.
    Each agent holds all DIMENSIONS parameters but is evaluated and
    bred through dynamic cluster-based co-evolution.
    """

    def __init__(self, params=None):
        if params is None:
            self.params = [random.random() for _ in range(DIMENSIONS)]
        else:
            self.params = [max(0.0, min(1.0, float(x))) for x in params]
        self.H_i  = []      # trajectory memory
        self.e_i  = 0.0     # trajectory score
        self.mu_i = 0.0     # adaptive mutation rate

    # Algorithm 2, line 9
    def update_history(self, fitness, env_fingerprint, merge_current=False):
        """
        H_i ← H_i ∪ {(f_i, E_t)};  trim to window W.
        merge_current: if this generation's entry already exists (same E_t
        object), average into it instead of adding a second entry, so the
        window still holds one entry per generation.
        """
        if (merge_current and self.H_i
                and self.H_i[-1][1] is env_fingerprint):
            self.H_i[-1] = ((self.H_i[-1][0] + fitness) / 2.0, env_fingerprint)
            return
        self.H_i.append((fitness, env_fingerprint))
        if len(self.H_i) > W:
            self.H_i.pop(0)

    def compute_trajectory_score(self, current_env):
        """Trajectory score using similarity between E_t and historical states."""
        if not self.H_i:
            self.e_i = 0.0
            return self.e_i
        weights = [max(0.0, cosine_similarity(current_env, past_env))
                   for _, past_env in self.H_i]
        total_w = sum(weights)
        if total_w == 0:
            weights = [1.0 / len(weights)] * len(weights)
        else:
            weights = [w / total_w for w in weights]
        self.e_i = sum(weights[i] * self.H_i[i][0]
                       for i in range(len(self.H_i)))
        return self.e_i

    def compute_mutation(self):
        self.mu_i = MU_BASE * (1.0 - self.e_i) + EPSILON

    # Algorithm 2, line 26 — breed by modifying only group components
    def mutate_within_cluster(self, cluster_indices):
        """
        Breed offspring by modifying only the strategy components
        belonging to G_k.  Non-cluster parameters are inherited unchanged.
        """
        child_params = self.params[:]
        for j in cluster_indices:
            noise = random.gauss(0, self.mu_i)
            child_params[j] = max(0.0, min(1.0, child_params[j] + noise))
        child = DICCEAgent(child_params)
        return child

    def mutate_full(self):
        """Full mutation across all parameters (fallback)."""
        child_params = [max(0.0, min(1.0, p + random.gauss(0, self.mu_i)))
                        for p in self.params]
        return DICCEAgent(child_params)


# ===============================================================
#  STRATEGY CLUSTER LIFECYCLE
# ===============================================================
class ClusterLifecycle:
    """
    One temporary co-evolution window for a strategy cluster G_k.  Owns the
    cluster's temporary sub-population of sub-agents (vectors over G_k's
    parameter indices).

    Algorithm 2, lines 30-36: terminate if joint improvement < tau_imp or the
    interaction strength decreases; otherwise retain until L_max.  The window
    is never closed before L_min rounds.
    """

    def __init__(self, indices, seed_agents):
        self.indices        = list(indices)
        self.subpop         = [[a.params[j] for j in self.indices] for a in seed_agents]
        self.window         = 0
        self.round_fitness  = 0.0     # best cluster fitness in the latest round
        self.prev_fitness   = None
        self.prev_strength  = None

    def update(self, cluster_fitness, strength):
        """Return True if the window stays open."""
        self.window += 1
        improvement = 0.0 if self.prev_fitness  is None else cluster_fitness - self.prev_fitness
        trend       = 0.0 if self.prev_strength is None else strength - self.prev_strength
        self.prev_fitness, self.prev_strength = cluster_fitness, strength

        if self.window >= L_MAX:
            return False
        if self.window >= L_MIN and (improvement < TAU_IMP or trend < 0):
            return False
        return True


# ===============================================================
#  DI-CCE-TDCCEA  MAIN LOOP  (Algorithm 2)
# ===============================================================
def run_di_cce_tdccea(server):
    """
    Run DI-CCE-TDCCEA: Dynamic Interaction-Driven Trajectory-Driven
    Competitive Co-Evolutionary Algorithm.

    Full implementation of Algorithm 2 from the post-review modifications.
    """
    print(f"\n{'='*70}")
    print(f"  RUNNING: DI-CCE-TDCCEA")
    print(f"  Dynamic Interaction-Driven Co-Evolutionary TD-CCEA")
    print(f"  N={N_POP}  W={W}  K={K_GAMES}  δ_f={DELTA_F}  δ_s={DELTA_S}  A_max={A_MAX}")
    print(f"  τ_I={TAU_I}  L_min={L_MIN}  L_max={L_MAX}  τ_imp={TAU_IMP}")
    print(f"{'='*70}")

    # ── Step 1: Set defaults ──

    # ── Step 2: Initialize strategy population P_0 of N agents ──
    population = [DICCEAgent() for _ in range(N_POP)]

    # ── Step 3: Initialize archive A ← ∅ ──
    archive = []

    # ── Step 4: Initialize strategy-interaction matrix I ← I_0 ──
    interaction_mx = InteractionMatrix(DIMENSIONS)

    # ── Step 5: Initialize environment memory ε ← ∅ ──
    env_memory = []       # list of (E_t, best_fitness) tuples

    gen_best_scores = []
    recent_fits     = []
    lifecycles = {}           # cluster key -> open ClusterLifecycle window
    cooldown   = {}           # cluster key -> generations before it may reopen
    t_start = time.time()

    # ── Step 6: for t = 1, 2, …, T_max do ──
    for gen in range(1, TOTAL_GENS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]

        # ── Step 7: Construct environmental fingerprint E_t ──
        E_t = construct_environment_fingerprint(
            population, opponent, gen, recent_fits, env_memory
        )

        # ── Step 8: Evaluate current agents using K simulated games ──
        fitnesses_now = []
        for agent in population:
            f_i = evaluate_weights(server, agent.params, opponent)
            fitnesses_now.append(f_i)

            # ── Step 9: Update trajectory memories ──
            agent.update_history(f_i, E_t)

        # ── Steps 10-12: Window trimming (handled inside update_history) ──

        # ── Step 13: trajectory score e_i ──
        for agent in population:
            agent.compute_trajectory_score(E_t)
            agent.compute_mutation()

        # Update environment memory
        gen_best_fit = max(fitnesses_now)
        recent_fits.append(gen_best_fit)
        env_memory.append((E_t[:], gen_best_fit))

        # ── Steps 14-16: interactions -> update -> smooth ──
        interaction_mx.update(interaction_mx.observe(population, fitnesses_now))
        interaction_mx.smooth()

        # ── Step 17: Maximum Spanning Tree T_t ──
        mst = build_maximum_spanning_tree(interaction_mx.I, DIMENSIONS)

        # ── Step 18: temporary strategy clusters G_1, …, G_m ──
        strategy_clusters = extract_strategy_clusters(mst, DIMENSIONS, TAU_I, MAX_GROUP)

        # Cooldown tick: a cluster whose window just closed stays idle for L_min gens
        for k in list(cooldown):
            cooldown[k] -= 1
            if cooldown[k] <= 0:
                del cooldown[k]

        # ── Steps 19-28: for each strategy cluster G_k do ──
        ranked   = sorted(population, key=lambda a: a.e_i, reverse=True)
        live     = {}          # clusters co-evolved this generation
        injected = []          # complete phenotypes produced by co-evolution
        for cluster_indices in strategy_clusters:
            if len(cluster_indices) <= 1:
                continue                      # singletons need no co-evolution
            key = tuple(sorted(cluster_indices))
            if key in cooldown:
                continue                      # window recently terminated
            idx = list(key)

            # Step 20: candidate sub-agents representing G_k — the cluster's own
            # temporary sub-population (seeded from the current elite on first use)
            lc = lifecycles.get(key)
            if lc is None:
                lc = ClusterLifecycle(idx, ranked[:SUBPOP_SIZE])
            live[key] = lc

            # Step 21: K complementary partner contexts — elite agents that supply
            # the components of all the OTHER clusters
            elite_pool = ranked[:max(K_GAMES, N_POP // 4)]
            partners   = random.sample(elite_pool, min(K_GAMES, len(elite_pool)))

            # Step 22: complete phenotype = G_k sub-agent + partner's other components
            def build(sub, partner, idx=idx):
                p = partner.params[:]
                for pos, j in enumerate(idx):
                    p[j] = sub[pos]
                return p

            # Steps 23-24: cluster fitness = mean over the K partner contexts
            # (one game per context, so K games per candidate)
            def cluster_fit(sub, partners=partners, build=build):
                return sum(evaluate_once(server, build(sub, pt), opponent)
                           for pt in partners) / len(partners)

            scored = sorted(((cluster_fit(sub), sub) for sub in lc.subpop),
                            key=lambda x: x[0], reverse=True)

            # Step 25: strongest members of G_k
            strong = scored[:max(2, len(scored) // 2)]

            # Step 26: breed offspring modifying ONLY the G_k components
            offspring = []
            for _ in range(N_OFFSPRING):
                pf, parent = random.choice(strong)
                mu    = MU_BASE * (1.0 - pf) + EPSILON
                child = [max(0.0, min(1.0, v + random.gauss(0, mu))) for v in parent]
                offspring.append((cluster_fit(child), child))

            # Step 27: insert offspring into the temporary sub-population
            # (keep the best SUBPOP_SIZE of survivors + offspring)
            merged = sorted(scored + offspring, key=lambda x: x[0], reverse=True)
            merged = merged[:SUBPOP_SIZE]
            lc.subpop        = [sub for _, sub in merged]
            lc.round_fitness = merged[0][0]

            # Hand the best sub-agent back to the main population as a full phenotype
            injected.append(DICCEAgent(build(merged[0][1], ranked[0])))

        # ── Step 28: end for ──

        # ── Step 29: Evaluate the complete population after co-evolution ──
        if injected:
            population = population + injected
            for agent in population:
                f_i = evaluate_weights(server, agent.params, opponent)
                agent.update_history(f_i, E_t, merge_current=True)
                agent.compute_trajectory_score(E_t)
                agent.compute_mutation()

        # ── Steps 30-36: adaptive window per active cluster ──
        for key, lc in live.items():
            pairs = [(a, b) for ai, a in enumerate(key) for b in key[ai + 1:]]
            strength = sum(interaction_mx.I[a][b] for a, b in pairs) / len(pairs)
            if lc.update(lc.round_fitness, strength):
                lifecycles[key] = lc                    # retain G_k
            else:
                lifecycles.pop(key, None)               # terminate window
                cooldown[key] = L_MIN + 1
        for key in list(lifecycles):                    # cluster dissolved in new MST
            if key not in live:
                del lifecycles[key]

        # ── TD-CCEA SURVIVAL ──
        best_agent = max(population, key=lambda a: a.e_i)
        gen_best_scores.append(best_agent.e_i)

        max_dist  = math.sqrt(DIMENSIONS)
        survivors = []
        for a in sorted(population, key=lambda a: a.e_i, reverse=True):
            if not any(1.0 - euclidean_dist(a.params, u.params) / max_dist > DELTA_S
                       for u in survivors):
                survivors.append(a)

        final = [a for a in survivors
                 if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
        while len(final) < 10 and len(survivors) > len(final):
            for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
                if a not in final:
                    final.append(a)
                    break

        final = sorted(final, key=lambda a: a.e_i, reverse=True)[:N_POP]

        # Archive update
        for top_ag in sorted(population, key=lambda a: a.e_i, reverse=True)[:3]:
            ar = DICCEAgent(top_ag.params)
            ar.e_i, ar.H_i, ar.mu_i = top_ag.e_i, top_ag.H_i[:], top_ag.mu_i
            archive.append(ar)
        archive.sort(key=lambda a: a.e_i, reverse=True)
        archive = archive[:A_MAX]

        # Breed offspring (full mutation for agents not covered by co-evolution)
        pool = final + archive
        population = final[:]
        while len(population) < N_POP:
            cands  = random.sample(pool, min(3, len(pool)))
            parent = max(cands, key=lambda a: a.e_i)
            population.append(parent.mutate_full())

        # ── Generation summary ──
        n_clusters = len([c for c in strategy_clusters if len(c) > 1])
        active_lc  = len(lifecycles)
        g_sizes = sorted([len(g) for g in strategy_clusters], reverse=True)
        off = [interaction_mx.I[i][j] for i in range(DIMENSIONS) for j in range(i + 1, DIMENSIONS)]
        print(f"  Gen {gen:2d}/{TOTAL_GENS} | Opp: {opponent:25s} | "
              f"Best e_i: {best_agent.e_i:.3f} | "
              f"Clusters: {n_clusters} {g_sizes[:4]} | Active: {active_lc} | "
              f"I max/mean: {max(off):.2f}/{sum(off)/len(off):.2f}")

    elapsed = time.time() - t_start
    best = max(archive, key=lambda a: a.e_i) if archive else max(population, key=lambda a: a.e_i)
    return best.params, elapsed, gen_best_scores


# ===============================================================
#  STANDARD TD-CCEA CENTROID  (baseline for comparison)
# ===============================================================
class TDCCEAAgent:
    """Standard TD-CCEA agent with centroid-based environment."""

    def __init__(self, params=None):
        if params is None:
            self.params = [random.random() for _ in range(DIMENSIONS)]
        else:
            self.params = [max(0.0, min(1.0, float(x))) for x in params]
        self.H_i, self.e_i, self.mu_i = [], 0.0, 0.0

    def update_history(self, f, env):
        self.H_i.append((f, env))
        if len(self.H_i) > W:
            self.H_i.pop(0)

    def compute_trajectory(self, env):
        if not self.H_i:
            self.e_i = 0.0
            return
        md = math.sqrt(DIMENSIONS)
        ws = [1.0 - euclidean_dist(env, pe) / md for _, pe in self.H_i]
        t  = sum(ws)
        ws = [w / t for w in ws] if t > 0 else [1.0 / len(ws)] * len(ws)
        self.e_i = sum(w * self.H_i[i][0] for i, w in enumerate(ws))

    def compute_mutation(self):
        self.mu_i = MU_BASE * (1.0 - self.e_i) + EPSILON

    def mutate(self):
        cp = [max(0.0, min(1.0, p + random.gauss(0, self.mu_i)))
              for p in self.params]
        return TDCCEAAgent(cp)


def run_tdccea_centroid(server):
    """Standard TD-CCEA with centroid environment (monolithic baseline)."""
    print(f"\n{'='*60}")
    print(f"  RUNNING: TD-CCEA Centroid (Baseline)")
    print(f"{'='*60}")

    population = [TDCCEAAgent() for _ in range(N_POP)]
    archive    = []
    gen_scores = []
    t0 = time.time()

    for gen in range(1, TOTAL_GENS + 1):
        opp = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
        centroid = [sum(a.params[j] for a in population) / len(population)
                    for j in range(DIMENSIONS)]

        for agent in population:
            f_i = evaluate_weights(server, agent.params, opp)
            agent.update_history(f_i, centroid)
            agent.compute_trajectory(centroid)
            agent.compute_mutation()

        best = max(population, key=lambda a: a.e_i)
        gen_scores.append(best.e_i)
        print(f"  Gen {gen:2d}/{TOTAL_GENS} | Opp: {opp:25s} | "
              f"Best e_i: {best.e_i:.3f}")

        md = math.sqrt(DIMENSIONS)
        survivors = []
        for a in sorted(population, key=lambda a: a.e_i, reverse=True):
            if not any(1.0 - euclidean_dist(a.params, u.params) / md > DELTA_S
                       for u in survivors):
                survivors.append(a)
        final = [a for a in survivors
                 if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
        while len(final) < 10 and len(survivors) > len(final):
            for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
                if a not in final:
                    final.append(a)
                    break

        for top_ag in sorted(population, key=lambda a: a.e_i, reverse=True)[:3]:
            archive.append(TDCCEAAgent(top_ag.params))
        archive.sort(key=lambda a: a.e_i, reverse=True)
        archive = archive[:A_MAX]

        pool = final + archive
        population = final[:]
        while len(population) < N_POP:
            cands = random.sample(pool, min(3, len(pool)))
            population.append(max(cands, key=lambda a: a.e_i).mutate())

    best_all = max(archive, key=lambda a: a.e_i) if archive else max(population, key=lambda a: a.e_i)
    return best_all.params, time.time() - t0, gen_scores


# ===============================================================
#  HILL CLIMBING  (simple baseline)
# ===============================================================
def run_hill_climbing(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: Hill Climbing")
    print(f"{'='*60}")

    cur = [random.random() for _ in range(DIMENSIONS)]
    cf, gs = 0.0, []
    t0 = time.time()

    for gen in range(1, TOTAL_GENS + 1):
        opp = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
        cf  = evaluate_weights(server, cur, opp)
        bc, bf = cur[:], cf
        for _ in range(N_POP - 1):
            c = [max(0.0, min(1.0, p + random.gauss(0, 0.15))) for p in cur]
            f = evaluate_weights(server, c, opp)
            if f > bf:
                bc, bf = c[:], f
        if bf >= cf:
            cur, cf = bc[:], bf
        gs.append(cf)
        print(f"  Gen {gen:2d}/{TOTAL_GENS} | Opp: {opp:25s} | Best={cf:.3f}")

    return cur, time.time() - t0, gs


# ===============================================================
#  MAIN
# ===============================================================
def main():
    print()
    print("#" * 70)
    print("  DI-CCE-TDCCEA: DYNAMIC INTERACTION-DRIVEN CO-EVOLUTIONARY TD-CCEA")
    print("  vs TD-CCEA CENTROID vs HILL CLIMBING")
    print("#" * 70)
    print(f"  Generations: {TOTAL_GENS} | Opponents: {len(OPPONENT_ROTATION)} | Pop: {N_POP}")
    print()

    algorithms = [
        ("DI-CCE-TDCCEA",    "di_cce"),
        ("TD-CCEA Centroid", "centroid"),
        ("Hill Climbing",    "hc"),
    ]

    all_results = []

    for algo_name, key in algorithms:
        server = GameServer()
        server.start()
        print(f"  [OK] Server started for {algo_name}")

        if key == "di_cce":
            best, elapsed, gen_scores = run_di_cce_tdccea(server)
        elif key == "centroid":
            best, elapsed, gen_scores = run_tdccea_centroid(server)
        else:
            best, elapsed, gen_scores = run_hill_climbing(server)

        print(f"\n  Validating {algo_name} against all opponents...")
        overall_wr, per_opp = validate_weights(server, best)
        server.stop()

        print(f"\n  +-- {algo_name} RESULTS ----------------------------")
        print(f"  |  Training Time:    {elapsed:.1f}s")
        print(f"  |  Overall Win Rate: {overall_wr:.1f}%")
        for opp, wr in per_opp.items():
            status = "[W]" if wr >= 50 else "[L]"
            print(f"  |  {status} vs {opp:25s}: {wr:.0f}%")
        print(f"  +--------------------------------------------------")

        all_results.append({
            "name": algo_name,
            "time": elapsed,
            "overall_winrate": overall_wr,
            "per_opponent": per_opp,
            "gen_scores": gen_scores,
            "best_params": best,
        })

    # ── FINAL COMPARISON TABLE ──
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"  {'Algorithm':<25s} {'Win Rate':>10s} {'Time':>10s}")
    print(f"  {'_'*25} {'_'*10} {'_'*10}")
    for r in sorted(all_results, key=lambda x: x["overall_winrate"], reverse=True):
        print(f"  {r['name']:<25s} {r['overall_winrate']:>9.1f}% {r['time']:>9.1f}s")

    # ── PER-OPPONENT TABLE ──
    print(f"\n{'='*60}")
    print(f"  WIN RATE PER OPPONENT")
    print(f"{'='*60}\n")

    header = f"  {'Opponent':<25s}"
    for r in all_results:
        header += f"  {r['name']:>14s}"
    print(header)
    print("  " + "-" * (25 + 16 * len(all_results)))

    for opp in OPPONENT_ROTATION:
        row   = f"  {opp:<25s}"
        rates = [r["per_opponent"].get(opp, 0) for r in all_results]
        best_rate = max(rates)
        for r in all_results:
            wr = r["per_opponent"].get(opp, 0)
            marker = " *" if wr == best_rate and rates.count(best_rate) == 1 else "  "
            row += f"  {wr:>11.0f}%{marker}"
        print(row)

    print("  " + "-" * (25 + 16 * len(all_results)))
    overall_row = f"  {'OVERALL':<25s}"
    for r in all_results:
        overall_row += f"  {r['overall_winrate']:>11.1f}%  "
    print(overall_row)

    # ── RANKING ──
    ranked = sorted(all_results, key=lambda x: x["overall_winrate"], reverse=True)
    medals = ["1st", "2nd", "3rd"]
    print(f"\n  +{'_'*58}+")
    print(f"  |  FINAL RANKING                                          |")
    print(f"  +{'_'*58}+")
    for i, r in enumerate(ranked):
        print(f"  |  {medals[i]}  {r['name']:<25s}  "
              f"{r['overall_winrate']:5.1f}% win rate  ({r['time']:5.0f}s)  |")
    print(f"  +{'_'*58}+")

    if len(ranked) >= 2:
        d = ranked[0]["overall_winrate"] - ranked[1]["overall_winrate"]
        print(f"\n  {ranked[0]['name']} beats {ranked[1]['name']} by {d:.1f}%")
    if len(ranked) >= 3:
        d = ranked[0]["overall_winrate"] - ranked[2]["overall_winrate"]
        print(f"  {ranked[0]['name']} beats {ranked[2]['name']} by {d:.1f}%")

    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
