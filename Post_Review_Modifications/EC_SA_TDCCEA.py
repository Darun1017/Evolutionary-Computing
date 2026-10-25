"""
EC_SA_TDCCEA.py — Environment-Conditioned Sensitivity-Adaptive
                   Trajectory-Driven Competitive Co-Evolutionary Algorithm

Implements Algorithm 1 (EC-SA-TDCCEA) from the post-review modifications.

Key changes from the original SA-TDCCEA:
  1. Static sensitivity → environment-conditioned sensitivity via memory clusters
  2. Simple centroid → rich environment fingerprint E_t = [R_pop, R_opp, R_game, R_fit]
  3. Independent mutation → MST-based parameter grouping for joint mutation
  4. No interaction tracking → strategy-interaction matrix I for parameter relationships

Adaptive cycle:  E_t → Environment Memory → Sensitivity → Interaction Structure → Mutation

Compares:
  1. EC-SA-TDCCEA  (NEW: environment-conditioned sensitivity + interaction grouping)
  2. TD-CCEA Centroid  (standard monolithic baseline)
  3. Hill Climbing  (simple baseline)
"""

import math, random, time, subprocess, os, sys, copy, json

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

MAP_PATH = "maps/chambers32x32.xml"
MAX_CYCLES = 10000

OPPONENT_ROTATION = [
    "EconomyMilitaryRush", "LightRush", "WorkerRush",
    "HeavyDefense", "RangedRush", "EconomyRush", "WorkerRushPlusPlus",
]

# Population and evolution parameters (Algorithm 1, line 1)
N_POP       = 20      # Population size N
W           = 7       # Trajectory window
K_GAMES     = 3       # Simulated games per evaluation
TOTAL_GENS  = 30      # T_max
VALID_GAMES = 5       # Games per opponent for final validation

# TD-CCEA parameters (Algorithm 1, line 1)
MU_BASE  = 0.15       # Base mutation rate μ_base
EPSILON  = 0.03       # Minimum mutation offset ε
DELTA_F  = 0.20       # Fitness threshold δ_f
DELTA_S  = 0.85       # Similarity threshold δ_s
A_MAX    = 15         # Archive size

# EC-SA-TDCCEA specific parameters (Algorithm 1, line 1)
TAU_E    = 0.70       # Environment similarity threshold τ_E
TAU_I    = 0.15       # Interaction threshold τ_I (MST group extraction; calibrated to the
                      # pooled partial-correlation scale of I — retune on real runs)
B        = 5          # Sensitivity update interval
MAX_GROUP = 5         # Cap on a temporary parameter group's size

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
#  ENVIRONMENT FINGERPRINT  (Algorithm 1, line 7)
# ===============================================================
def construct_environment_fingerprint(population, opponent, gen,
                                      recent_fitnesses, archive):
    """
    E_t = [R_pop, R_opp, R_game, R_fit]   (10-dimensional)

      R_pop  [0..2]: centroid mean, diversity, best e_i
      R_opp  [3..5]: opponent index (normalised), recent win rate,
                     Hall-of-Fame (archive) quality
      R_game [6]:    generation progress
      R_fit  [7..9]: mean fitness, best fitness, fitness std
    """
    centroid = [sum(a.params[j] for a in population) / len(population)
                for j in range(DIMENSIONS)]
    centroid_mean = sum(centroid) / DIMENSIONS

    param_stds = []
    for j in range(DIMENSIONS):
        vals = [a.params[j] for a in population]
        mu   = sum(vals) / len(vals)
        param_stds.append(math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)))
    diversity = sum(param_stds) / len(param_stds)
    best_ei   = max(a.e_i for a in population) if population else 0.0

    opp_idx   = OPPONENT_ROTATION.index(opponent) / max(len(OPPONENT_ROTATION) - 1, 1)
    recent_wr = (sum(recent_fitnesses[-3:]) / len(recent_fitnesses[-3:])
                 if recent_fitnesses else 0.5)
    hof_quality = (sum(a.e_i for a in archive) / len(archive)) if archive else 0.0

    gen_progress = gen / TOTAL_GENS

    if recent_fitnesses:
        mean_fit = sum(recent_fitnesses) / len(recent_fitnesses)
        best_fit = max(recent_fitnesses)
        fit_std  = math.sqrt(sum((f - mean_fit) ** 2 for f in recent_fitnesses)
                             / len(recent_fitnesses))
    else:
        mean_fit, best_fit, fit_std = 0.0, 0.0, 0.0

    return [centroid_mean, diversity, best_ei,
            opp_idx, recent_wr, hof_quality,
            gen_progress,
            mean_fit, best_fit, fit_std]


# ===============================================================
#  ENVIRONMENT-MEMORY CLUSTER  (Algorithm 1, lines 4, 18-25)
# ===============================================================
class EnvironmentMemoryCluster:
    """
    One cluster C_{j,k} in the memory set M_j for parameter j.

    Stores:
        mu_E        – running mean of environment fingerprints
        sensitivity – running mean of sensitivity  S_{C_{j,k}}
        n           – observation count
    """

    def __init__(self, env_fingerprint, sensitivity, n=1):
        self.mu_E        = env_fingerprint[:]
        self.sensitivity = sensitivity
        self.n           = n

    # Algorithm 1, lines 20-22
    def update(self, env_fingerprint, new_sensitivity):
        """Update the matched cluster:  μ_E ← (n·μ_E + E_t) / (n+1)  etc."""
        n = self.n
        self.mu_E = [(n * mu + e) / (n + 1)
                     for mu, e in zip(self.mu_E, env_fingerprint)]
        self.sensitivity = (n * self.sensitivity + new_sensitivity) / (n + 1)
        self.n = n + 1

    # Algorithm 1, line 19
    def sim(self, env_fingerprint):
        """
        sim(E_t, C_{j,k}) = max(0, 1 - ||E_t - mu_E|| / 2).
        (Cosine similarity scored every fingerprint > 0.8 against every other,
        so no cluster was ever created at tau_E = 0.7.)
        """
        return max(0.0, 1.0 - euclidean_dist(self.mu_E, env_fingerprint) / 2.0)


def conditioned_sensitivity(memory_clusters, E_t):
    """
    For every parameter j return S_{C_{j,k}} of the cluster in M_j most similar
    to the CURRENT environment E_t.  Called every generation, so mutation is
    always conditioned on the present environment (not the last update's).
    """
    return [max(clusters, key=lambda c: c.sim(E_t)).sensitivity
            for clusters in memory_clusters]


# ===============================================================
#  STRATEGY-INTERACTION MATRIX  (Algorithm 1, line 5)
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
#  MAXIMUM SPANNING TREE  &  PARAMETER GROUPING
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
    """Kruskal's algorithm on I to obtain the Maximum Spanning Tree T_t."""
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


def extract_parameter_groups(mst_edges, dim, threshold, max_size=MAX_GROUP):
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
#  EC-SA-TDCCEA AGENT
# ===============================================================
class ECSATDCCEAAgent:
    """
    Agent for EC-SA-TDCCEA.
    Maintains its own trajectory memory H_i and trajectory score e_i.
    """

    def __init__(self, params=None):
        if params is None:
            self.params = [random.random() for _ in range(DIMENSIONS)]
        else:
            self.params = [max(0.0, min(1.0, float(x))) for x in params]
        self.H_i  = []      # trajectory memory: [(f_i, E_t), ...]
        self.e_i  = 0.0     # trajectory score
        self.mu_i = 0.0     # base adaptive mutation rate

    # Algorithm 1, line 9
    def update_history(self, fitness, env_fingerprint):
        """H_i ← H_i ∪ {(f_i, E_t)};  trim to window W."""
        self.H_i.append((fitness, env_fingerprint))
        if len(self.H_i) > W:
            self.H_i.pop(0)

    # Algorithm 1, line 13
    def compute_trajectory_score(self, current_env):
        """
        Compute trajectory score e_i using similarity between the current
        environmental fingerprint E_t and historical environmental states
        stored in H_i.
        """
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
        self.e_i = sum(weights[i] * self.H_i[i][0] for i in range(len(self.H_i)))
        return self.e_i

    def compute_adaptive_mutation(self):
        self.mu_i = MU_BASE * (1.0 - self.e_i) + EPSILON

    # Algorithm 1 — mutation using cluster-conditioned sensitivity
    def mutate_with_sensitivity(self, cluster_sens, param_groups):
        """
        Mutate using environment-conditioned sensitivity S_{C_{j,k}} and
        MST-based parameter groups.

        Within each group the parameters share a correlated noise component
        (joint mutation); each parameter also gets its own individual noise
        scaled by its cluster sensitivity.

            σ_j = S_{C_{j,k}}(E_t) · (1 − e_i) + ε
        """
        child_params = self.params[:]
        for group in param_groups:
            # Group-level shared noise (correlated mutation)
            group_sens = [cluster_sens[j] for j in group]
            mean_sens  = sum(group_sens) / len(group_sens)
            shared_noise = random.gauss(0, mean_sens * (1.0 - self.e_i) + EPSILON)

            for j in group:
                sigma_j = cluster_sens[j] * (1.0 - self.e_i) + EPSILON
                individual_noise = random.gauss(0, sigma_j)
                combined = 0.5 * individual_noise + 0.5 * shared_noise
                child_params[j] = max(0.0, min(1.0, self.params[j] + combined))

        return ECSATDCCEAAgent(child_params)

    def mutate_standard(self):
        """Fallback: standard uniform-rate mutation."""
        child_params = [max(0.0, min(1.0, p + random.gauss(0, self.mu_i)))
                        for p in self.params]
        return ECSATDCCEAAgent(child_params)


# ===============================================================
#  LOCAL ABLATION  (Algorithm 1, lines 16-17)
# ===============================================================
def compute_parameter_sensitivity(server, population, j, opponent):
    """
    S_j(E_t) = |WinRate(x_j = 1, E_t) − WinRate(x_j = 0, E_t)|

    Perform local ablation of parameter j under the current environmental
    state E_t by using the population's best agent as the base solution.
    """
    best = max(population, key=lambda a: a.e_i)
    base = best.params[:]

    test_high = base[:]; test_high[j] = 1.0
    wr_high = evaluate_weights(server, test_high, opponent)

    test_low = base[:]; test_low[j] = 0.0
    wr_low = evaluate_weights(server, test_low, opponent)

    return abs(wr_high - wr_low)


# ===============================================================
#  EC-SA-TDCCEA  MAIN LOOP  (Algorithm 1)
# ===============================================================
def run_ec_sa_tdccea(server):
    """
    Run EC-SA-TDCCEA: Environment-Conditioned Sensitivity-Adaptive TD-CCEA.
    Full implementation of Algorithm 1 from the post-review modifications.
    """
    print(f"\n{'='*70}")
    print(f"  RUNNING: EC-SA-TDCCEA")
    print(f"  Environment-Conditioned Sensitivity-Adaptive TD-CCEA")
    print(f"  N={N_POP}  W={W}  K={K_GAMES}  μ_base={MU_BASE}  ε={EPSILON}")
    print(f"  δ_f={DELTA_F}  δ_s={DELTA_S}  A_max={A_MAX}")
    print(f"  τ_E={TAU_E}  τ_I={TAU_I}  B={B}")
    print(f"{'='*70}")

    # ── Step 1: Set defaults (line 1) ──
    # (done via module-level constants)

    # ── Step 2: Initialize population P_0 of N agents ──
    population = [ECSATDCCEAAgent() for _ in range(N_POP)]

    # ── Step 3: Initialize archive A ← ∅ ──
    archive = []

    # ── Step 4: Initialize environment-memory cluster M_j = {C_{j,0}}
    #            for each parameter j ∈ {1, …, 20} ──
    neutral_env = [0.5] * ENV_DIM
    memory_clusters = [[EnvironmentMemoryCluster(neutral_env, 0.10)]
                       for _ in range(DIMENSIONS)]

    # ── Step 5: Initialize strategy-interaction matrix I ← I_0 ──
    interaction_mx = InteractionMatrix(DIMENSIONS)

    # Working state
    cluster_sens   = [0.10] * DIMENSIONS          # current S_{C_{j,k}} per param
    param_groups   = [[j] for j in range(DIMENSIONS)]  # starts fully independent
    gen_best_scores = []
    recent_fits     = []
    t_start = time.time()

    # ── Step 6: for t = 1, 2, …, T_max do ──
    for gen in range(1, TOTAL_GENS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]

        # ── Step 7: Construct environmental fingerprint E_t ──
        E_t = construct_environment_fingerprint(
            population, opponent, gen, recent_fits, archive
        )

        # ── Step 8: Evaluate fitness f_i for every agent O_i ∈ P_t ──
        agent_fitness = {}
        for agent in population:
            f_i = evaluate_weights(server, agent.params, opponent)
            agent_fitness[id(agent)] = f_i

            # ── Step 9: Update trajectory memory ──
            agent.update_history(f_i, E_t)

        # ── Steps 10–12: Window trimming (handled inside update_history) ──

        # ── Step 13: Compute trajectory score e_i ──
        for agent in population:
            agent.compute_trajectory_score(E_t)
            agent.compute_adaptive_mutation()

        # Record best fitness this generation
        gen_fit = max(agent_fitness.values())
        recent_fits.append(gen_fit)

        # ── Steps 14–26: Sensitivity update  (if t mod B = 0) ──
        if gen % B == 0:
            print(f"    ┌─ Sensitivity update at generation {gen} ──")
            for j in range(DIMENSIONS):
                # Step 16–17: local ablation -> S_j(E_t)
                s_j = compute_parameter_sensitivity(
                    server, population, j, opponent
                )

                # Step 18: cluster C_{j,k} in M_j most similar to E_t
                best_cluster = max(memory_clusters[j], key=lambda c: c.sim(E_t))

                # Steps 19–25: update matched cluster, or create a new one
                if best_cluster.sim(E_t) >= TAU_E:
                    best_cluster.update(E_t, s_j)
                else:
                    memory_clusters[j].append(
                        EnvironmentMemoryCluster(E_t, s_j, n=1))
            print(f"    │  Cluster counts:  {[len(m) for m in memory_clusters]}")

        # ── Environment-conditioned sensitivity for THIS generation ──
        cluster_sens = conditioned_sensitivity(memory_clusters, E_t)
        if gen % B == 0:
            ranked = sorted(range(DIMENSIONS),
                            key=lambda i: cluster_sens[i], reverse=True)
            print(f"    │  Top-5 sensitive: "
                  f"{', '.join(WEIGHT_NAMES[i] for i in ranked[:5])}")
            print(f"    └──")

        # ── Interaction structure: observe -> update -> smooth -> MST -> groups ──
        interaction_mx.update(interaction_mx.observe(
            population, [agent_fitness[id(a)] for a in population]))
        interaction_mx.smooth()
        mst = build_maximum_spanning_tree(interaction_mx.I, DIMENSIONS)
        param_groups = extract_parameter_groups(mst, DIMENSIONS, TAU_I, MAX_GROUP)

        # ── Generation summary ──
        best_agent = max(population, key=lambda a: a.e_i)
        gen_best_scores.append(best_agent.e_i)
        g_sizes = sorted([len(g) for g in param_groups], reverse=True)
        off = [interaction_mx.I[i][j] for i in range(DIMENSIONS) for j in range(i + 1, DIMENSIONS)]
        print(f"  Gen {gen:2d}/{TOTAL_GENS} | Opp: {opponent:25s} | "
              f"Best e_i: {best_agent.e_i:.3f} | "
              f"Groups: {len(param_groups)} {g_sizes[:4]} | "
              f"I max/mean: {max(off):.2f}/{sum(off)/len(off):.2f}")

        # ── TD-CCEA SURVIVAL ──
        # Redundancy removal (similarity > δ_s)
        max_dist  = math.sqrt(DIMENSIONS)
        survivors = []
        for agent in sorted(population, key=lambda a: a.e_i, reverse=True):
            if not any(1.0 - euclidean_dist(agent.params, u.params) / max_dist > DELTA_S
                       for u in survivors):
                survivors.append(agent)

        # Fitness pruning (e_i < δ_f AND |H_i| ≥ 3 → remove)
        final = [a for a in survivors
                 if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
        while len(final) < 10 and len(survivors) > len(final):
            for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
                if a not in final:
                    final.append(a)
                    break

        # Archive update  (top-m → archive)
        for top_ag in sorted(population, key=lambda a: a.e_i, reverse=True)[:3]:
            ar = ECSATDCCEAAgent(top_ag.params)
            ar.e_i, ar.H_i = top_ag.e_i, top_ag.H_i[:]
            archive.append(ar)
        if len(archive) > A_MAX:
            archive = archive[-A_MAX:]

        # Breed offspring using cluster-conditioned sensitivity + groups
        breed_pool = final + archive
        population = final[:]
        while len(population) < N_POP:
            cands  = random.sample(breed_pool, min(3, len(breed_pool)))
            parent = max(cands, key=lambda a: a.e_i)
            population.append(
                parent.mutate_with_sensitivity(cluster_sens, param_groups)
            )

    elapsed = time.time() - t_start
    best = max(population, key=lambda a: a.e_i)
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
        self.H_i  = []
        self.e_i  = 0.0
        self.mu_i = 0.0

    def update_history(self, fitness, current_environment):
        self.H_i.append((fitness, current_environment))
        if len(self.H_i) > W:
            self.H_i.pop(0)

    def compute_dynamic_trajectory(self, current_environment):
        if not self.H_i:
            self.e_i = 0.0
            return self.e_i
        max_d = math.sqrt(DIMENSIONS)
        dyn_w = [1.0 - euclidean_dist(current_environment, pe) / max_d
                 for _, pe in self.H_i]
        total = sum(dyn_w)
        dyn_w = ([w / total for w in dyn_w] if total > 0
                 else [1.0 / len(dyn_w)] * len(dyn_w))
        self.e_i = sum(dyn_w[i] * self.H_i[i][0] for i in range(len(self.H_i)))
        return self.e_i

    def compute_adaptive_mutation(self):
        self.mu_i = MU_BASE * (1.0 - self.e_i) + EPSILON

    def mutate(self):
        child_params = [max(0.0, min(1.0, p + random.gauss(0, self.mu_i)))
                        for p in self.params]
        return TDCCEAAgent(child_params)


def run_tdccea_centroid(server):
    """Standard TD-CCEA with centroid environment (monolithic baseline)."""
    print(f"\n{'='*60}")
    print(f"  RUNNING: TD-CCEA Centroid (Baseline)")
    print(f"{'='*60}")

    population = [TDCCEAAgent() for _ in range(N_POP)]
    archive = []
    gen_best_scores = []
    t_start = time.time()

    for gen in range(1, TOTAL_GENS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
        centroid = [sum(a.params[j] for a in population) / len(population)
                    for j in range(DIMENSIONS)]

        for agent in population:
            f_i = evaluate_weights(server, agent.params, opponent)
            agent.update_history(f_i, centroid)
            agent.compute_dynamic_trajectory(centroid)
            agent.compute_adaptive_mutation()

        best = max(population, key=lambda a: a.e_i)
        gen_best_scores.append(best.e_i)
        print(f"  Gen {gen:2d}/{TOTAL_GENS} | Opp: {opponent:25s} | "
              f"Best e_i: {best.e_i:.3f}")

        # Survival
        max_d = math.sqrt(DIMENSIONS)
        survivors = []
        for a in sorted(population, key=lambda a: a.e_i, reverse=True):
            if not any(1.0 - euclidean_dist(a.params, u.params) / max_d > DELTA_S
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
            ar = TDCCEAAgent(top_ag.params)
            archive.append(ar)
        if len(archive) > A_MAX:
            archive = archive[-A_MAX:]

        pool = final + archive
        population = final[:]
        while len(population) < N_POP:
            cands = random.sample(pool, min(3, len(pool)))
            population.append(max(cands, key=lambda a: a.e_i).mutate())

    elapsed = time.time() - t_start
    best = max(population, key=lambda a: a.e_i)
    return best.params, elapsed, gen_best_scores


# ===============================================================
#  HILL CLIMBING  (simple baseline)
# ===============================================================
def run_hill_climbing(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: Hill Climbing")
    print(f"{'='*60}")

    current     = [random.random() for _ in range(DIMENSIONS)]
    current_fit = 0.0
    gen_scores  = []
    t_start = time.time()

    for gen in range(1, TOTAL_GENS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
        current_fit = evaluate_weights(server, current, opponent)
        best_cand, best_fit = current[:], current_fit

        for _ in range(N_POP - 1):
            cand = [max(0.0, min(1.0, p + random.gauss(0, 0.15)))
                    for p in current]
            fit = evaluate_weights(server, cand, opponent)
            if fit > best_fit:
                best_cand, best_fit = cand[:], fit

        if best_fit >= current_fit:
            current, current_fit = best_cand[:], best_fit

        gen_scores.append(current_fit)
        print(f"  Gen {gen:2d}/{TOTAL_GENS} | Opp: {opponent:25s} | "
              f"Best: {current_fit:.3f}")

    return current, time.time() - t_start, gen_scores


# ===============================================================
#  MAIN
# ===============================================================
def main():
    print()
    print("#" * 70)
    print("  EC-SA-TDCCEA: ENVIRONMENT-CONDITIONED SENSITIVITY-ADAPTIVE TD-CCEA")
    print("  vs TD-CCEA CENTROID vs HILL CLIMBING")
    print("#" * 70)
    print(f"  Generations: {TOTAL_GENS} | Opponents: {len(OPPONENT_ROTATION)} | Pop: {N_POP}")
    print()

    algorithms = [
        ("EC-SA-TDCCEA",     "ec_sa"),
        ("TD-CCEA Centroid", "centroid"),
        ("Hill Climbing",    "hc"),
    ]

    all_results = []

    for algo_name, key in algorithms:
        server = GameServer()
        server.start()
        print(f"  [OK] Server started for {algo_name}")

        if key == "ec_sa":
            best, elapsed, gen_scores = run_ec_sa_tdccea(server)
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
