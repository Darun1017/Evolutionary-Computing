"""
benchmark_all.py — Unified Comparative Benchmark
Runs TD-CCEA, Standard GA, Hill Climbing, and Random Search
against the same MicroRTS opponents, records time & win rates,
and generates comparison graphs.

Usage: python benchmark_all.py
"""

import math, random, time, subprocess, os, json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# ═══════════════════════════════════════════════════════════════
#  SHARED CONFIGURATION (identical for all algorithms)
# ═══════════════════════════════════════════════════════════════
MICRORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "Simulation", "microrts")
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

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

N_POP = 20
K_GAMES = 3
TOTAL_GENERATIONS = 21
VALIDATION_GAMES = 10  # games per opponent during final validation

# ═══════════════════════════════════════════════════════════════
#  GAME SERVER
# ═══════════════════════════════════════════════════════════════
class GameServer:
    def __init__(self):
        self.process = None

    def start(self):
        cmd = ["java", "-cp", "lib/*;bin", "tests.EvolvedBatchGameRunner"]
        self.process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=MICRORTS_DIR
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

# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def euclidean_dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_centroid(population_params):
    n = len(population_params)
    centroid = [0.0] * DIMENSIONS
    for params in population_params:
        for j in range(DIMENSIONS):
            centroid[j] += params[j]
    return [x / n for x in centroid]

def evaluate_weights(server, weights, opponent):
    """Play K_GAMES and return win rate [0.0 - 1.0]."""
    wins = 0
    for _ in range(K_GAMES):
        winner, _ = server.run_game(weights, opponent)
        if winner == 0:
            wins += 1
        elif winner == -1:
            wins += 0.3
    return wins / K_GAMES

def validate_weights(server, weights):
    """Test weights against ALL opponents, return per-opponent and overall win rates."""
    per_opp = {}
    total_wins = 0
    total_games = 0
    for opp in OPPONENT_ROTATION:
        opp_wins = 0
        for _ in range(VALIDATION_GAMES):
            winner, _ = server.run_game(weights, opp)
            if winner == 0:
                opp_wins += 1
            total_games += 1
        total_wins += opp_wins
        per_opp[opp] = opp_wins / VALIDATION_GAMES * 100
    overall = total_wins / total_games * 100
    return overall, per_opp

# ═══════════════════════════════════════════════════════════════
#  ALGORITHM 1: TD-CCEA (Our Algorithm)
# ═══════════════════════════════════════════════════════════════
class EvolvedAgent:
    def __init__(self, params=None):
        if params is None:
            self.params = [random.random() for _ in range(DIMENSIONS)]
        else:
            self.params = [max(0.0, min(1.0, float(x))) for x in params]
        self.H_i = []
        self.e_i = 0.0
        self.mu_i = 0.0

    def update_history(self, fitness, current_environment):
        self.H_i.append((fitness, current_environment))
        if len(self.H_i) > 7:
            self.H_i.pop(0)

    def compute_dynamic_trajectory(self, current_environment):
        if len(self.H_i) == 0:
            self.e_i = 0.0
            return self.e_i

        dynamic_weights = []
        # Environment vector length varies, so use its actual length for max_dist
        env_len = len(current_environment)
        max_dist = math.sqrt(env_len) * 1.0

        for past_fitness, past_env in self.H_i:
            dist = euclidean_dist(current_environment, past_env)
            similarity = 1.0 - (dist / max_dist)
            dynamic_weights.append(similarity)

        total_w = sum(dynamic_weights)
        if total_w == 0:
            dynamic_weights = [1.0 / len(dynamic_weights)] * len(dynamic_weights)
        else:
            dynamic_weights = [w / total_w for w in dynamic_weights]

        self.e_i = 0.0
        for i in range(len(self.H_i)):
            self.e_i += dynamic_weights[i] * self.H_i[i][0]

        return self.e_i

    def compute_adaptive_mutation(self):
        self.mu_i = 0.15 * (1.0 - self.e_i) + 0.03
        return self.mu_i

    def mutate(self):
        offspring_params = []
        for p in self.params:
            noise = random.gauss(0, self.mu_i)
            offspring_params.append(p + noise)
        return EvolvedAgent(offspring_params)


def get_population_centroid(population):
    centroid = [0.0] * DIMENSIONS
    for agent in population:
        for j in range(DIMENSIONS):
            centroid[j] += agent.params[j]
    return [x / len(population) for x in centroid]


def run_tdccea(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: TD-CCEA (Our Algorithm)")
    print(f"{'='*60}")

    population = [EvolvedAgent() for _ in range(N_POP)]
    archive = []
    gen_best_scores = []

    DELTA_S = 0.85
    DELTA_F = 0.20
    A_MAX = 15

    t_start = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
        # ── Build Hybrid Environment Vector ──
        # Instead of just the population centroid (which barely changes),
        # we encode: [opponent_one_hot (7)] + [avg_fitness, fitness_variance, diversity_ratio (3)]
        # This lets the similarity calculation truly distinguish between
        # "facing a rush" vs "facing a turtle" environments.
        opp_idx = OPPONENT_ROTATION.index(opponent)
        opp_one_hot = [0.0] * len(OPPONENT_ROTATION)
        opp_one_hot[opp_idx] = 1.0

        # Population fitness stats (from previous gen, use 0.5 for gen 1)
        if gen > 1:
            all_ei = [a.e_i for a in population]
            avg_fit = sum(all_ei) / len(all_ei)
            fit_var = sum((x - avg_fit) ** 2 for x in all_ei) / len(all_ei)
            unique = len(set(round(a.e_i, 2) for a in population)) / len(population)
        else:
            avg_fit, fit_var, unique = 0.5, 0.25, 1.0

        current_env = opp_one_hot + [avg_fit, fit_var, unique]

        # ── Evaluate each agent ──
        for agent in population:
            wins = 0
            for _ in range(K_GAMES):
                winner, _ = server.run_game(agent.params, opponent)
                if winner == 0:
                    wins += 1
                elif winner == -1:
                    wins += 0.3

            f_i = wins / K_GAMES
            agent.update_history(f_i, current_env)
            agent.compute_dynamic_trajectory(current_env)
            agent.compute_adaptive_mutation()

        best_agent = max(population, key=lambda a: a.e_i)
        gen_best_scores.append(best_agent.e_i)
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | Opp: {opponent:25s} | Best e_i: {best_agent.e_i:.3f}")

        # ── Diversity-based survival ──
        max_dist = math.sqrt(DIMENSIONS) * 1.0
        survivors = []
        for agent in sorted(population, key=lambda a: a.e_i, reverse=True):
            is_redundant = False
            for u in survivors:
                dist = euclidean_dist(agent.params, u.params)
                if 1.0 - (dist / max_dist) > DELTA_S:
                    is_redundant = True
                    break
            if not is_redundant:
                survivors.append(agent)

        # ── Fitness pruning ──
        final_survivors = []
        for agent in survivors:
            if agent.e_i < DELTA_F and len(agent.H_i) >= 3:
                pass
            else:
                final_survivors.append(agent)

        while len(final_survivors) < 10 and len(survivors) > len(final_survivors):
            for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
                if a not in final_survivors:
                    final_survivors.append(a)
                    break

        # ── Archive ──
        sorted_gen = sorted(population, key=lambda a: a.e_i, reverse=True)
        for top_ag in sorted_gen[:3]:
            archive.append(EvolvedAgent(top_ag.params))
        if len(archive) > A_MAX:
            archive = archive[-A_MAX:]

        # ── Breed ──
        breeding_pool = final_survivors
        population = final_survivors.copy()
        while len(population) < N_POP:
            candidates = random.sample(
                breeding_pool + archive,
                min(3, len(breeding_pool + archive))
            )
            parent = max(candidates, key=lambda a: a.e_i)
            population.append(parent.mutate())

    elapsed = time.time() - t_start
    best = max(archive, key=lambda a: a.e_i)
    return best.params, elapsed, gen_best_scores


# ═══════════════════════════════════════════════════════════════
#  ALGORITHM 2: TD-CCEA with Centroid (Original)
# ═══════════════════════════════════════════════════════════════
class CentroidAgent:
    def __init__(self, params=None):
        if params is None:
            self.params = [random.random() for _ in range(DIMENSIONS)]
        else:
            self.params = [max(0.0, min(1.0, float(x))) for x in params]
        self.H_i = []
        self.e_i = 0.0
        self.mu_i = 0.0

    def update_history(self, fitness, current_environment):
        self.H_i.append((fitness, current_environment))
        if len(self.H_i) > 7:
            self.H_i.pop(0)

    def compute_dynamic_trajectory(self, current_environment):
        if len(self.H_i) == 0:
            self.e_i = 0.0
            return self.e_i
        dynamic_weights = []
        max_dist = math.sqrt(DIMENSIONS) * 1.0
        for past_fitness, past_env in self.H_i:
            dist = euclidean_dist(current_environment, past_env)
            similarity = 1.0 - (dist / max_dist)
            dynamic_weights.append(similarity)
        total_w = sum(dynamic_weights)
        if total_w == 0:
            dynamic_weights = [1.0 / len(dynamic_weights)] * len(dynamic_weights)
        else:
            dynamic_weights = [w / total_w for w in dynamic_weights]
        self.e_i = 0.0
        for i in range(len(self.H_i)):
            self.e_i += dynamic_weights[i] * self.H_i[i][0]
        return self.e_i

    def compute_adaptive_mutation(self):
        self.mu_i = 0.15 * (1.0 - self.e_i) + 0.03
        return self.mu_i

    def mutate(self):
        offspring_params = []
        for p in self.params:
            noise = random.gauss(0, self.mu_i)
            offspring_params.append(p + noise)
        return CentroidAgent(offspring_params)


def run_tdccea_centroid(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: TD-CCEA with Centroid (Original)")
    print(f"{'='*60}")

    population = [CentroidAgent() for _ in range(N_POP)]
    archive = []
    gen_best_scores = []
    DELTA_S = 0.85
    DELTA_F = 0.20
    A_MAX = 15

    t_start = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]

        # ── Original: Population centroid as environment ──
        centroid = [0.0] * DIMENSIONS
        for agent in population:
            for j in range(DIMENSIONS):
                centroid[j] += agent.params[j]
        current_env = [x / len(population) for x in centroid]

        for agent in population:
            wins = 0
            for _ in range(K_GAMES):
                winner, _ = server.run_game(agent.params, opponent)
                if winner == 0:
                    wins += 1
                elif winner == -1:
                    wins += 0.3
            f_i = wins / K_GAMES
            agent.update_history(f_i, current_env)
            agent.compute_dynamic_trajectory(current_env)
            agent.compute_adaptive_mutation()

        best_agent = max(population, key=lambda a: a.e_i)
        gen_best_scores.append(best_agent.e_i)
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | Opp: {opponent:25s} | Best e_i: {best_agent.e_i:.3f}")

        max_dist = math.sqrt(DIMENSIONS) * 1.0
        survivors = []
        for agent in sorted(population, key=lambda a: a.e_i, reverse=True):
            is_redundant = False
            for u in survivors:
                dist = euclidean_dist(agent.params, u.params)
                if 1.0 - (dist / max_dist) > DELTA_S:
                    is_redundant = True
                    break
            if not is_redundant:
                survivors.append(agent)

        final_survivors = []
        for agent in survivors:
            if agent.e_i < DELTA_F and len(agent.H_i) >= 3:
                pass
            else:
                final_survivors.append(agent)
        while len(final_survivors) < 10 and len(survivors) > len(final_survivors):
            for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
                if a not in final_survivors:
                    final_survivors.append(a)
                    break

        sorted_gen = sorted(population, key=lambda a: a.e_i, reverse=True)
        for top_ag in sorted_gen[:3]:
            archive.append(CentroidAgent(top_ag.params))
        if len(archive) > A_MAX:
            archive = archive[-A_MAX:]

        breeding_pool = final_survivors
        population = final_survivors.copy()
        while len(population) < N_POP:
            candidates = random.sample(
                breeding_pool + archive,
                min(3, len(breeding_pool + archive))
            )
            parent = max(candidates, key=lambda a: a.e_i)
            population.append(parent.mutate())

    elapsed = time.time() - t_start
    best = max(archive, key=lambda a: a.e_i)
    return best.params, elapsed, gen_best_scores


# ═══════════════════════════════════════════════════════════════
#  ALGORITHM 3: Standard GA (No Trajectory)
# ═══════════════════════════════════════════════════════════════
def run_standard_ga(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: Standard Genetic Algorithm")
    print(f"{'='*60}")

    population = [[random.random() for _ in range(DIMENSIONS)] for _ in range(N_POP)]
    archive = []
    gen_best_scores = []
    FIXED_MUT = 0.15
    DELTA_S = 0.85
    A_MAX = 15

    t_start = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
        fitnesses = [evaluate_weights(server, p, opponent) for p in population]

        best_idx = max(range(N_POP), key=lambda i: fitnesses[i])
        gen_best_scores.append(fitnesses[best_idx])
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | Opp: {opponent:25s} | Best: {fitnesses[best_idx]:.3f}")

        # Diversity survival
        max_dist = math.sqrt(DIMENSIONS)
        indexed = sorted(range(N_POP), key=lambda i: fitnesses[i], reverse=True)
        survivors = []
        for idx in indexed:
            redundant = False
            for s_idx in survivors:
                dist = euclidean_dist(population[idx], population[s_idx])
                if 1.0 - (dist / max_dist) > DELTA_S:
                    redundant = True
                    break
            if not redundant:
                survivors.append(idx)
        while len(survivors) < 10:
            survivors.append(random.randint(0, N_POP - 1))

        # Archive
        for i in indexed[:3]:
            archive.append(population[i][:])
        if len(archive) > A_MAX:
            archive = archive[-A_MAX:]

        # Breed
        surv_params = [population[i][:] for i in survivors]
        new_pop = surv_params[:]
        pool = surv_params + archive
        while len(new_pop) < N_POP:
            parent = random.choice(pool)
            child = [max(0.0, min(1.0, p + random.gauss(0, FIXED_MUT))) for p in parent]
            new_pop.append(child)
        population = new_pop

    elapsed = time.time() - t_start
    best_params = population[0]
    return best_params, elapsed, gen_best_scores

# ═══════════════════════════════════════════════════════════════
#  ALGORITHM 3: Hill Climbing
# ═══════════════════════════════════════════════════════════════
def run_hill_climbing(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: Hill Climbing")
    print(f"{'='*60}")

    current = [random.random() for _ in range(DIMENSIONS)]
    current_fit = 0.0
    gen_best_scores = []
    MUT = 0.15

    t_start = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
        current_fit = evaluate_weights(server, current, opponent)
        best_candidate = current[:]
        best_fit = current_fit

        for _ in range(N_POP - 1):
            candidate = [max(0.0, min(1.0, p + random.gauss(0, MUT))) for p in current]
            fit = evaluate_weights(server, candidate, opponent)
            if fit > best_fit:
                best_candidate = candidate[:]
                best_fit = fit

        if best_fit >= current_fit:
            current = best_candidate[:]
            current_fit = best_fit

        gen_best_scores.append(current_fit)
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | Opp: {opponent:25s} | Best: {current_fit:.3f}")

    elapsed = time.time() - t_start
    return current, elapsed, gen_best_scores

# ═══════════════════════════════════════════════════════════════
#  ALGORITHM 4: Random Search
# ═══════════════════════════════════════════════════════════════
def run_random_search(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: Random Search")
    print(f"{'='*60}")

    best_params = None
    best_fit = -1.0
    gen_best_scores = []

    t_start = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
        gen_best = -1.0
        gen_best_p = None

        for _ in range(N_POP):
            params = [random.random() for _ in range(DIMENSIONS)]
            fit = evaluate_weights(server, params, opponent)
            if fit > gen_best:
                gen_best = fit
                gen_best_p = params[:]

        if gen_best > best_fit:
            best_fit = gen_best
            best_params = gen_best_p[:]

        gen_best_scores.append(best_fit)
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | Opp: {opponent:25s} | Best Ever: {best_fit:.3f}")

    elapsed = time.time() - t_start
    return best_params, elapsed, gen_best_scores

# ═══════════════════════════════════════════════════════════════
#  GRAPH GENERATION
# ═══════════════════════════════════════════════════════════════
def generate_graphs(all_results):
    """Generate comparison graphs and save them."""
    algo_names = [r['name'] for r in all_results]
    colors = ['#2ecc71', '#27ae60', '#3498db', '#e67e22', '#e74c3c']

    # ── Graph 1: Fitness Convergence Over Generations ──
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, r in enumerate(all_results):
        ax.plot(range(1, len(r['gen_scores']) + 1), r['gen_scores'],
                marker='o', markersize=4, linewidth=2, color=colors[i], label=r['name'])
    ax.set_xlabel('Generation', fontsize=13)
    ax.set_ylabel('Best Fitness (Win Rate)', fontsize=13)
    ax.set_title('Fitness Convergence: TD-CCEA vs Baselines', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)
    plt.tight_layout()
    path1 = os.path.join(RESULTS_DIR, "benchmark_convergence.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"  [✓] Saved: {path1}")

    # ── Graph 2: Overall Win Rate Bar Chart ──
    fig, ax = plt.subplots(figsize=(10, 6))
    win_rates = [r['overall_winrate'] for r in all_results]
    bars = ax.bar(algo_names, win_rates, color=colors, edgecolor='black', linewidth=0.8)
    for bar, wr in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{wr:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax.set_ylabel('Overall Win Rate (%)', fontsize=13)
    ax.set_title('Overall Win Rate Comparison', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(RESULTS_DIR, "benchmark_winrate.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"  [✓] Saved: {path2}")

    # ── Graph 3: Time Taken Bar Chart ──
    fig, ax = plt.subplots(figsize=(10, 6))
    times = [r['time'] for r in all_results]
    bars = ax.bar(algo_names, times, color=colors, edgecolor='black', linewidth=0.8)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{t:.1f}s', ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Training Time (seconds)', fontsize=13)
    ax.set_title('Training Time Comparison', fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path3 = os.path.join(RESULTS_DIR, "benchmark_time.png")
    plt.savefig(path3, dpi=150)
    plt.close()
    print(f"  [✓] Saved: {path3}")

    # ── Graph 4: Per-Opponent Heatmap ──
    fig, ax = plt.subplots(figsize=(14, 6))
    opp_names = OPPONENT_ROTATION
    data = []
    for r in all_results:
        row = [r['per_opponent'].get(opp, 0) for opp in opp_names]
        data.append(row)

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(opp_names)))
    ax.set_xticklabels([o.replace('Rush', '\nRush').replace('Defense', '\nDefense') for o in opp_names],
                       fontsize=10, ha='center')
    ax.set_yticks(range(len(algo_names)))
    ax.set_yticklabels(algo_names, fontsize=12)
    for i in range(len(algo_names)):
        for j in range(len(opp_names)):
            val = data[i][j]
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white' if val < 40 else 'black')
    ax.set_title('Win Rate vs Each Opponent (%)', fontsize=15, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Win Rate %')
    plt.tight_layout()
    path4 = os.path.join(RESULTS_DIR, "benchmark_heatmap.png")
    plt.savefig(path4, dpi=150)
    plt.close()
    print(f"  [✓] Saved: {path4}")

# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print(f"\n{'='*60}")
    print(f"  UNIFIED BENCHMARK: TD-CCEA (Hybrid vs Centroid) + 3 Baselines")
    print(f"{'='*60}")
    print(f"  Map: {MAP_PATH}")
    print(f"  Opponents: {len(OPPONENT_ROTATION)} rotating")
    print(f"  Generations: {TOTAL_GENERATIONS} | Population: {N_POP}")
    print(f"  Validation: {VALIDATION_GAMES} games per opponent")
    print(f"  Total games per algorithm: ~{TOTAL_GENERATIONS * N_POP * K_GAMES + len(OPPONENT_ROTATION) * VALIDATION_GAMES}")
    print()

    algorithms = [
        ("TD-CCEA (Hybrid)", run_tdccea),
        ("TD-CCEA (Centroid)", run_tdccea_centroid),
        ("Standard GA", run_standard_ga),
        ("Hill Climbing", run_hill_climbing),
        ("Random Search", run_random_search),
    ]

    all_results = []

    for algo_name, algo_func in algorithms:
        server = GameServer()
        server.start()
        print(f"  [✓] Server started for {algo_name}")

        best_params, elapsed, gen_scores = algo_func(server)

        # Validation
        print(f"\n  Validating {algo_name} against all opponents...")
        overall_wr, per_opp = validate_weights(server, best_params)

        server.stop()
        print(f"  [✓] Server stopped for {algo_name}")

        print(f"\n  ┌─ {algo_name} RESULTS ─────────────────────────")
        print(f"  │  Training Time:   {elapsed:.1f}s")
        print(f"  │  Overall Win Rate: {overall_wr:.1f}%")
        for opp, wr in per_opp.items():
            status = "✓" if wr >= 50 else "✗"
            print(f"  │  {status} vs {opp:25s}: {wr:.0f}%")
        print(f"  └──────────────────────────────────────────────")

        all_results.append({
            'name': algo_name,
            'time': elapsed,
            'overall_winrate': overall_wr,
            'per_opponent': per_opp,
            'gen_scores': gen_scores,
            'best_params': best_params,
        })

    # ── Summary Table ──
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"  {'Algorithm':<20s} {'Win Rate':>10s} {'Time':>10s}")
    print(f"  {'─'*20} {'─'*10} {'─'*10}")
    for r in all_results:
        print(f"  {r['name']:<20s} {r['overall_winrate']:>9.1f}% {r['time']:>9.1f}s")

    # ── Generate Graphs ──
    print(f"\n  Generating comparison graphs...")
    generate_graphs(all_results)

    # ── Save raw results ──
    json_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    save_data = []
    for r in all_results:
        save_data.append({
            'name': r['name'],
            'time': r['time'],
            'overall_winrate': r['overall_winrate'],
            'per_opponent': r['per_opponent'],
            'gen_scores': r['gen_scores'],
        })
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"  [✓] Raw results saved: {json_path}")

    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
