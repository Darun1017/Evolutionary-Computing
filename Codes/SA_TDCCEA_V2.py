"""
SA_TDCCEA_V2.py - Sensitivity-Adaptive TD-CCEA (Fully Corrected)

Fixes applied over original:
  FIX 1 - Meaningful environment vector (opponent + fitness stats + diversity + progress)
  FIX 2 - Sensitivity mutation direction corrected (high influence = small careful step)
  FIX 3 - Watchdog thresholds fixed + mode lock to prevent rapid switching
  FIX 4 - Archive keeps BEST agents, not most recent
  FIX 5 - Temporal decay added to trajectory (recent history weighted more)
  FIX 6 - Elite injection every N generations (prevents losing best solution)
"""

import math
import random
import time
import subprocess
import os
import json

os.environ["PYTHONIOENCODING"] = "utf-8"

# ===============================================================
#  CONFIGURATION
# ===============================================================
MICRORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Simulation", "microrts"
)

WEIGHT_NAMES = [
    "worker_ratio",       # 0
    "max_workers",        # 1
    "expand_threshold",   # 2
    "light_preference",   # 3
    "heavy_preference",   # 4
    "ranged_preference",  # 5
    "barracks_timing",    # 6
    "aggression",         # 7
    "attack_threshold",   # 8
    "defend_radius",      # 9
    "target_priority",    # 10
    "defense_trigger",    # 11
    "counter_rush",       # 12
    "tech_switch",        # 13
    "economy_pressure",   # 14
    "scout_priority",     # 15
    "group_attack",       # 16
    "second_barracks",    # 17
    "worker_fight_ratio", # 18
    "focus_fire",         # 19
]
DIMENSIONS = len(WEIGHT_NAMES)

MAP_PATH    = "maps/chambers32x32.xml"
MAX_CYCLES  = 10000

OPPONENT_ROTATION = [
    "EconomyMilitaryRush",
    "LightRush",
    "WorkerRush",
    "HeavyDefense",
    "RangedRush",
    "EconomyRush",
    "WorkerRushPlusPlus",
]

N_POP             = 20
K_GAMES           = 3
TOTAL_GENERATIONS = 30
VALIDATION_GAMES  = 5

SENSITIVITY_FILE = "sensitivity_scores.json"

# Watchdog settings (FIX 3)
STAGNATION_LIMIT    = 5      # was 3 — more patience before fallback
RECOVERY_THRESHOLD  = 0.55   # was 0.8 — actually reachable
MODE_LOCK_GENS      = 5      # stay in any mode at least this many gens

# Elite injection (FIX 6)
ELITE_INJECTION_FREQ = 5     # inject best archive agent every N gens

# Temporal decay (FIX 5)
HISTORY_DECAY = 0.75         # recent entries weighted more

# Archive
A_MAX  = 15
DELTA_S = 0.85
DELTA_F = 0.20


# ===============================================================
#  GAME SERVER
# ===============================================================
class GameServer:
    def __init__(self):
        self.process = None

    def start(self):
        cmd = ["java", "-cp", "lib/*;bin", "tests.EvolvedBatchGameRunner"]
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=MICRORTS_DIR,
        )
        line = self.process.stdout.readline().strip()
        if line != "READY":
            raise RuntimeError(f"Server failed to start — got: {line!r}")

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


def evaluate_weights(server, weights, opponent):
    """Play K_GAMES and return win rate [0.0–1.0]."""
    wins = 0
    for _ in range(K_GAMES):
        winner, _ = server.run_game(weights, opponent)
        if winner == 0:
            wins += 1
        elif winner == -1:
            wins += 0.3          # draw credit
    return wins / K_GAMES


def validate_weights(server, weights):
    """Test weights against ALL opponents; return overall + per-opponent rates."""
    total_wins  = 0
    total_games = 0
    per_opp     = {}
    for opp in OPPONENT_ROTATION:
        opp_wins = 0
        for _ in range(VALIDATION_GAMES):
            winner, _ = server.run_game(weights, opp)
            if winner == 0:
                opp_wins += 1
            total_games += 1
        total_wins      += opp_wins
        per_opp[opp]     = opp_wins / VALIDATION_GAMES * 100
    overall = total_wins / total_games * 100
    return overall, per_opp


# ===============================================================
#  FIX 1 — MEANINGFUL ENVIRONMENT VECTOR
# ===============================================================
def build_environment_vector(population, opponent, gen, total_gens):
    """
    Build an environment vector that actually changes each generation:
      - Opponent one-hot  (len = number of opponents)
      - avg_fitness, fitness_variance, best_fitness, worst_fitness
      - population diversity (avg distance from centroid)
      - training progress [0..1]
    """
    # Opponent one-hot
    opp_idx     = OPPONENT_ROTATION.index(opponent)
    opp_one_hot = [0.0] * len(OPPONENT_ROTATION)
    opp_one_hot[opp_idx] = 1.0

    # Fitness statistics
    all_ei   = [a.e_i for a in population]
    avg_fit  = sum(all_ei) / len(all_ei)
    fit_var  = sum((x - avg_fit) ** 2 for x in all_ei) / len(all_ei)
    best_fit = max(all_ei)
    worst_fit= min(all_ei)

    # Population diversity — avg Euclidean distance from centroid
    centroid = [
        sum(a.params[j] for a in population) / len(population)
        for j in range(DIMENSIONS)
    ]
    avg_dist  = sum(euclidean_dist(a.params, centroid) for a in population) / len(population)
    diversity = avg_dist / math.sqrt(DIMENSIONS)   # normalise to [0,1]

    # Training progress
    progress = gen / total_gens

    return opp_one_hot + [avg_fit, fit_var, best_fit, worst_fit, diversity, progress]


# ===============================================================
#  SENSITIVITY ANALYSIS
# ===============================================================
def load_or_run_sensitivity():
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), SENSITIVITY_FILE)
    if os.path.exists(filepath):
        print(f"  Found existing {SENSITIVITY_FILE}, loading...")
        with open(filepath, "r") as f:
            data = json.load(f)
        per_weight_mut = data["normalized_mutation_rates"]
        print(f"  Loaded {len(per_weight_mut)} mutation rates")
        return per_weight_mut
    else:
        print(f"  No {SENSITIVITY_FILE} found — running sensitivity analysis...")
        server = GameServer()
        server.start()
        per_weight_mut = run_sensitivity_analysis(server)
        server.stop()
        return per_weight_mut


def run_sensitivity_analysis(server):
    print()
    print("=" * 70)
    print("  PHASE 1: SENSITIVITY ANALYSIS")
    print("=" * 70)

    baseline = [0.5] * DIMENSIONS

    # Baseline win rate
    print("  Testing baseline [all 0.5]...")
    baseline_wins  = 0
    baseline_games = 0
    for opp in OPPONENT_ROTATION:
        for _ in range(K_GAMES):
            winner, _ = server.run_game(baseline, opp)
            if winner == 0:
                baseline_wins += 1
            elif winner == -1:
                baseline_wins += 0.3
            baseline_games += 1
    baseline_wr = baseline_wins / baseline_games
    print(f"  Baseline win rate: {baseline_wr:.1%}\n")

    influence_scores = []
    print(f"  {'#':>3s}  {'Weight':<22s}  {'WR@0.0':>8s}  {'WR@1.0':>8s}  {'Influence':>10s}")
    print("  " + "-" * 60)

    for w_idx in range(DIMENSIONS):
        # Test at 0.0
        test_low        = baseline[:]
        test_low[w_idx] = 0.0
        low_wins = low_games = 0
        for opp in OPPONENT_ROTATION:
            for _ in range(K_GAMES):
                winner, _ = server.run_game(test_low, opp)
                if winner == 0:
                    low_wins += 1
                elif winner == -1:
                    low_wins += 0.3
                low_games += 1
        wr_low = low_wins / low_games

        # Test at 1.0
        test_high         = baseline[:]
        test_high[w_idx]  = 1.0
        high_wins = high_games = 0
        for opp in OPPONENT_ROTATION:
            for _ in range(K_GAMES):
                winner, _ = server.run_game(test_high, opp)
                if winner == 0:
                    high_wins += 1
                elif winner == -1:
                    high_wins += 0.3
                high_games += 1
        wr_high = high_wins / high_games

        influence = abs(wr_high - wr_low)
        influence_scores.append(influence)
        print(f"  w{w_idx:2d}  {WEIGHT_NAMES[w_idx]:<22s}  {wr_low:8.1%}  {wr_high:8.1%}  {influence:10.1%}")

    # FIX 2 — CORRECTED direction:
    #   high influence gene → SMALL mutation (explore carefully)
    #   low  influence gene → LARGE mutation (explore freely, doesn't matter much)
    max_inf    = max(influence_scores) if max(influence_scores) > 0 else 1.0
    normalized = []
    for inf in influence_scores:
        scaled = 0.08 + (1.0 - inf / max_inf) * 0.12
        # Result range: 0.08 (most important) → 0.20 (least important)
        normalized.append(scaled)

    result = {
        "influence_scores":        influence_scores,
        "normalized_mutation_rates": normalized,
        "baseline_winrate":        baseline_wr,
        "weight_names":            WEIGHT_NAMES,
    }
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), SENSITIVITY_FILE)
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved sensitivity data to {SENSITIVITY_FILE}")

    print(f"\n  {'Weight':<22s}  {'Influence':>10s}  {'Mutation Rate':>14s}")
    print("  " + "-" * 52)
    for i, name in enumerate(WEIGHT_NAMES):
        print(f"  {name:<22s}  {influence_scores[i]:10.1%}  {normalized[i]:14.4f}")

    return normalized


# ===============================================================
#  TD-CCEA AGENT (with all fixes)
# ===============================================================
class TDCCEAAgentV2:
    def __init__(self, params=None, per_weight_mutation=None):
        if params is None:
            self.params = [random.random() for _ in range(DIMENSIONS)]
        else:
            self.params = [max(0.0, min(1.0, float(x))) for x in params]

        self.H_i          = []      # history list: [(fitness, env_vector), ...]
        self.e_i          = 0.0     # trajectory score
        self.mu_i         = 0.0     # base adaptive mutation rate
        self.per_weight_mut = per_weight_mutation
        self.use_sa_mode  = True    # True = sensitivity-adaptive, False = standard

    # ------------------------------------------------------------------
    def update_history(self, fitness, current_environment):
        self.H_i.append((fitness, current_environment))
        if len(self.H_i) > 7:
            self.H_i.pop(0)

    # ------------------------------------------------------------------
    # FIX 5 — TEMPORAL DECAY IN TRAJECTORY
    def compute_dynamic_trajectory(self, current_environment):
        if not self.H_i:
            self.e_i = 0.0
            return self.e_i

        env_len  = len(current_environment)
        max_dist = math.sqrt(env_len)

        weighted_sum  = 0.0
        weight_total  = 0.0

        for t, (past_fitness, past_env) in enumerate(self.H_i):
            # Recency: most recent entry (largest t index) gets highest weight
            recency = HISTORY_DECAY ** (len(self.H_i) - 1 - t)

            # Environmental similarity (use min length to handle vector size change)
            cmp_len = min(len(past_env), env_len)
            dist    = euclidean_dist(current_environment[:cmp_len], past_env[:cmp_len])
            max_d   = math.sqrt(cmp_len)
            similarity = max(0.0, 1.0 - (dist / max_d)) if max_d > 0 else 1.0

            combined      = recency * similarity
            weighted_sum += combined * past_fitness
            weight_total += combined

        self.e_i = weighted_sum / weight_total if weight_total > 0 else 0.0
        return self.e_i

    # ------------------------------------------------------------------
    def compute_adaptive_mutation(self):
        self.mu_i = 0.15 * (1.0 - self.e_i) + 0.03
        return self.mu_i

    # ------------------------------------------------------------------
    # FIX 2 — CORRECTED SENSITIVITY MUTATION
    def mutate(self):
        offspring_params = []

        if self.per_weight_mut is not None and self.use_sa_mode:
            # SA mode: per-gene mutation rate from sensitivity analysis
            # High-influence genes get smaller sigma → careful, precise steps
            # Low-influence genes get larger sigma  → explore freely
            for j, p in enumerate(self.params):
                sigma_j = self.per_weight_mut[j] * (1.0 - self.e_i) + 0.02
                noise   = random.gauss(0, sigma_j)
                offspring_params.append(p + noise)
        else:
            # Standard mode: uniform mutation for all genes
            for p in self.params:
                noise = random.gauss(0, self.mu_i)
                offspring_params.append(p + noise)

        child = TDCCEAAgentV2(
            [max(0.0, min(1.0, x)) for x in offspring_params],
            self.per_weight_mut
        )
        child.use_sa_mode = self.use_sa_mode
        return child


# ===============================================================
#  SA-TDCCEA V2 MAIN EVOLUTION LOOP (all fixes applied)
# ===============================================================
def run_sa_v2(server, per_weight_mutation):
    print(f"\n{'='*60}")
    print(f"  RUNNING: SA-TDCCEA V2 (Fully Corrected)")
    print(f"{'='*60}")

    population = [TDCCEAAgentV2(per_weight_mutation=per_weight_mutation)
                  for _ in range(N_POP)]
    archive    = []
    gen_best_scores = []

    # Watchdog state
    consecutive_drops = 0
    prev_best         = 0.0
    sa_mode_active    = True
    mode_locked_until = 0
    sa_gens           = 0
    std_gens          = 0

    t_start = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]

        # FIX 6 — ELITE INJECTION: replace worst agent with mutated archive best
        if gen % ELITE_INJECTION_FREQ == 0 and archive:
            best_archived = max(archive, key=lambda a: a.e_i)
            worst_idx     = min(range(len(population)), key=lambda i: population[i].e_i)
            elite_clone   = TDCCEAAgentV2(best_archived.params, per_weight_mutation)
            elite_clone.e_i         = best_archived.e_i
            elite_clone.use_sa_mode = sa_mode_active
            population[worst_idx]   = elite_clone

        # FIX 1 — MEANINGFUL ENVIRONMENT VECTOR
        current_env = build_environment_vector(population, opponent, gen, TOTAL_GENERATIONS)

        # Evaluate every agent
        for agent in population:
            f_i = evaluate_weights(server, agent.params, opponent)
            agent.update_history(f_i, current_env)
            agent.compute_dynamic_trajectory(current_env)   # FIX 5 inside here
            agent.compute_adaptive_mutation()

        best_agent   = max(population, key=lambda a: a.e_i)
        current_best = best_agent.e_i
        gen_best_scores.append(current_best)

        # FIX 3 — FIXED WATCHDOG (with mode lock + reachable threshold)
        if gen > 1:
            # Small tolerance (0.01) avoids switching on noise
            if current_best < prev_best - 0.01:
                consecutive_drops += 1
            else:
                consecutive_drops = 0

            if (consecutive_drops >= STAGNATION_LIMIT
                    and sa_mode_active
                    and gen > mode_locked_until):
                sa_mode_active    = False
                mode_locked_until = gen + MODE_LOCK_GENS
                consecutive_drops = 0
                for agent in population:
                    agent.use_sa_mode = False
                print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opponent:25s} "
                      f"| e_i={current_best:.3f} | >> FALLBACK to standard mutation")

            elif (current_best >= RECOVERY_THRESHOLD
                    and not sa_mode_active
                    and gen > mode_locked_until):
                sa_mode_active    = True
                consecutive_drops = 0
                mode_locked_until = gen + MODE_LOCK_GENS
                for agent in population:
                    agent.use_sa_mode = True
                print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opponent:25s} "
                      f"| e_i={current_best:.3f} | >> RECOVERED → SA mode")
            else:
                mode = "SA" if sa_mode_active else "STD"
                print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opponent:25s} "
                      f"| e_i={current_best:.3f} | [{mode}]")
        else:
            print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opponent:25s} "
                  f"| e_i={current_best:.3f} | [SA]")

        if sa_mode_active:
            sa_gens += 1
        else:
            std_gens += 1
        prev_best = current_best

        # Diversity-based survival
        max_dist  = math.sqrt(DIMENSIONS)
        survivors = []
        for agent in sorted(population, key=lambda a: a.e_i, reverse=True):
            redundant = any(
                1.0 - (euclidean_dist(agent.params, u.params) / max_dist) > DELTA_S
                for u in survivors
            )
            if not redundant:
                survivors.append(agent)

        # Fitness pruning — drop persistently poor agents
        final_survivors = [
            a for a in survivors
            if not (a.e_i < DELTA_F and len(a.H_i) >= 3)
        ]
        # Ensure minimum survivors
        while len(final_survivors) < 10 and len(survivors) > len(final_survivors):
            for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
                if a not in final_survivors:
                    final_survivors.append(a)
                    break

        # FIX 4 — ARCHIVE KEEPS BEST (sorted, not latest)
        for top_ag in sorted(population, key=lambda a: a.e_i, reverse=True)[:3]:
            archived            = TDCCEAAgentV2(top_ag.params, per_weight_mutation)
            archived.e_i        = top_ag.e_i
            archived.H_i        = top_ag.H_i[:]
            archived.mu_i       = top_ag.mu_i
            archived.use_sa_mode= sa_mode_active
            archive.append(archived)

        # Sort archive by fitness — keep best A_MAX, discard worst
        archive.sort(key=lambda a: a.e_i, reverse=True)
        archive = archive[:A_MAX]

        # Breed new population
        pool       = final_survivors + archive
        population = final_survivors[:]
        while len(population) < N_POP:
            candidates = random.sample(pool, min(3, len(pool)))
            parent     = max(candidates, key=lambda a: a.e_i)
            population.append(parent.mutate())

    elapsed = time.time() - t_start

    # Return best from archive (guaranteed best seen across all generations)
    best_overall = max(archive, key=lambda a: a.e_i) if archive else max(population, key=lambda a: a.e_i)

    print(f"\n  Watchdog Summary: SA mode = {sa_gens} gens | STD mode = {std_gens} gens")
    return best_overall.params, elapsed, gen_best_scores


# ===============================================================
#  STANDARD TD-CCEA CENTROID (comparison baseline)
# ===============================================================
def run_tdccea_centroid(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: TD-CCEA Centroid (Standard Baseline)")
    print(f"{'='*60}")

    population = [TDCCEAAgentV2() for _ in range(N_POP)]
    archive    = []
    gen_best_scores = []

    t_start = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]

        # Original centroid environment
        centroid = [0.0] * DIMENSIONS
        for agent in population:
            for j in range(DIMENSIONS):
                centroid[j] += agent.params[j]
        current_env = [x / len(population) for x in centroid]

        for agent in population:
            f_i = evaluate_weights(server, agent.params, opponent)
            agent.update_history(f_i, current_env)
            agent.compute_dynamic_trajectory(current_env)
            agent.compute_adaptive_mutation()

        best_agent = max(population, key=lambda a: a.e_i)
        gen_best_scores.append(best_agent.e_i)
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opponent:25s} | e_i={best_agent.e_i:.3f}")

        max_dist  = math.sqrt(DIMENSIONS)
        survivors = []
        for agent in sorted(population, key=lambda a: a.e_i, reverse=True):
            redundant = any(
                1.0 - (euclidean_dist(agent.params, u.params) / max_dist) > DELTA_S
                for u in survivors
            )
            if not redundant:
                survivors.append(agent)

        final_survivors = [
            a for a in survivors
            if not (a.e_i < DELTA_F and len(a.H_i) >= 3)
        ]
        while len(final_survivors) < 10 and len(survivors) > len(final_survivors):
            for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
                if a not in final_survivors:
                    final_survivors.append(a)
                    break

        for top_ag in sorted(population, key=lambda a: a.e_i, reverse=True)[:3]:
            archive.append(TDCCEAAgentV2(top_ag.params))
        archive.sort(key=lambda a: a.e_i, reverse=True)
        archive = archive[:A_MAX]

        pool       = final_survivors + archive
        population = final_survivors[:]
        while len(population) < N_POP:
            candidates = random.sample(pool, min(3, len(pool)))
            parent     = max(candidates, key=lambda a: a.e_i)
            population.append(parent.mutate())

    elapsed      = time.time() - t_start
    best_overall = max(archive, key=lambda a: a.e_i) if archive else max(population, key=lambda a: a.e_i)
    return best_overall.params, elapsed, gen_best_scores


# ===============================================================
#  HILL CLIMBING (comparison baseline)
# ===============================================================
def run_hill_climbing(server):
    print(f"\n{'='*60}")
    print(f"  RUNNING: Hill Climbing (Comparison)")
    print(f"{'='*60}")

    current     = [random.random() for _ in range(DIMENSIONS)]
    current_fit = 0.0
    gen_best_scores = []
    MUT = 0.15

    t_start = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opponent    = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
        current_fit = evaluate_weights(server, current, opponent)
        best_candidate = current[:]
        best_fit       = current_fit

        for _ in range(N_POP - 1):
            candidate = [max(0.0, min(1.0, p + random.gauss(0, MUT))) for p in current]
            fit       = evaluate_weights(server, candidate, opponent)
            if fit > best_fit:
                best_candidate = candidate[:]
                best_fit       = fit

        if best_fit >= current_fit:
            current     = best_candidate[:]
            current_fit = best_fit

        gen_best_scores.append(current_fit)
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opponent:25s} | Best={current_fit:.3f}")

    elapsed = time.time() - t_start
    return current, elapsed, gen_best_scores


# ===============================================================
#  MAIN
# ===============================================================
def main():
    print()
    print("#" * 70)
    print("  SA-TDCCEA V2 (Fully Corrected) vs TD-CCEA Centroid vs Hill Climbing")
    print("#" * 70)
    print(f"  Weights     : {DIMENSIONS}")
    print(f"  Population  : {N_POP}")
    print(f"  Generations : {TOTAL_GENERATIONS}")
    print(f"  Opponents   : {len(OPPONENT_ROTATION)}")
    print(f"  K_GAMES     : {K_GAMES}")
    print(f"  Validation  : {VALIDATION_GAMES} games/opponent")
    total_games = TOTAL_GENERATIONS * N_POP * K_GAMES + len(OPPONENT_ROTATION) * VALIDATION_GAMES
    print(f"  Est. games  : ~{total_games} per algorithm")
    print()

    # Load (or compute) sensitivity scores
    per_weight_mut = load_or_run_sensitivity()
    print()

    algorithms = [
        ("SA-TDCCEA V2",     "sa_v2"),
        ("TD-CCEA Centroid", "centroid"),
        ("Hill Climbing",    "hc"),
    ]

    all_results = []

    for algo_name, algo_key in algorithms:
        server = GameServer()
        server.start()
        print(f"  [OK] Game server started for: {algo_name}")

        if algo_key == "sa_v2":
            best_params, elapsed, gen_scores = run_sa_v2(server, per_weight_mut)
        elif algo_key == "centroid":
            best_params, elapsed, gen_scores = run_tdccea_centroid(server)
        else:
            best_params, elapsed, gen_scores = run_hill_climbing(server)

        print(f"\n  Validating {algo_name} against all opponents...")
        overall_wr, per_opp = validate_weights(server, best_params)
        server.stop()
        print(f"  [OK] Server stopped for: {algo_name}")

        print(f"\n  +-- {algo_name} RESULTS --------------------------------")
        print(f"  |  Training Time   : {elapsed:.1f}s")
        print(f"  |  Overall Win Rate: {overall_wr:.1f}%")
        for opp, wr in per_opp.items():
            status = "[W]" if wr >= 50 else "[L]"
            print(f"  |  {status} vs {opp:<26s}: {wr:5.0f}%")
        print(f"  +-------------------------------------------------------")

        all_results.append({
            "name":          algo_name,
            "time":          elapsed,
            "overall_winrate": overall_wr,
            "per_opponent":  per_opp,
            "gen_scores":    gen_scores,
            "best_params":   best_params,
        })

    # ── Final comparison table ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"  {'Algorithm':<22s}  {'Win Rate':>10s}  {'Time':>8s}")
    print(f"  {'─'*22}  {'─'*10}  {'─'*8}")
    for r in sorted(all_results, key=lambda x: x["overall_winrate"], reverse=True):
        print(f"  {r['name']:<22s}  {r['overall_winrate']:>9.1f}%  {r['time']:>7.1f}s")

    # ── Per-opponent breakdown ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  WIN RATE PER OPPONENT")
    print(f"{'='*60}")
    header = f"  {'Opponent':<26s}"
    for r in all_results:
        header += f"  {r['name']:>16s}"
    print(header)
    print("  " + "-" * (26 + 18 * len(all_results)))

    for opp in OPPONENT_ROTATION:
        row   = f"  {opp:<26s}"
        rates = [r["per_opponent"].get(opp, 0) for r in all_results]
        best  = max(rates)
        for i, r in enumerate(all_results):
            wr     = r["per_opponent"].get(opp, 0)
            marker = " *" if wr == best and rates.count(best) == 1 else "  "
            row   += f"  {wr:>13.0f}%{marker}"
        print(row)

    print("  " + "-" * (26 + 18 * len(all_results)))
    overall_row = f"  {'OVERALL':<26s}"
    for r in all_results:
        overall_row += f"  {r['overall_winrate']:>14.1f}%  "
    print(overall_row)

    # ── Ranking ────────────────────────────────────────────────
    ranked = sorted(all_results, key=lambda x: x["overall_winrate"], reverse=True)
    medals = ["1st 🥇", "2nd 🥈", "3rd 🥉"]
    print(f"\n{'='*60}")
    print(f"  RANKING")
    print(f"{'='*60}")
    for i, r in enumerate(ranked):
        print(f"  {medals[i]}  {r['name']:<22s}  {r['overall_winrate']:5.1f}%   ({r['time']:.0f}s)")

    if len(ranked) >= 2:
        diff = ranked[0]["overall_winrate"] - ranked[1]["overall_winrate"]
        sign = "beats" if diff > 0 else "loses to"
        print(f"\n  {ranked[0]['name']} {sign} {ranked[1]['name']} by {abs(diff):.1f}%")
    if len(ranked) >= 3:
        diff = ranked[0]["overall_winrate"] - ranked[2]["overall_winrate"]
        sign = "beats" if diff > 0 else "loses to"
        print(f"  {ranked[0]['name']} {sign} {ranked[2]['name']} by {abs(diff):.1f}%")

    # ── Save results ───────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results_v2.json")
    save_data = [{
        "name":            r["name"],
        "time":            r["time"],
        "overall_winrate": r["overall_winrate"],
        "per_opponent":    r["per_opponent"],
        "gen_scores":      r["gen_scores"],
    } for r in all_results]
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Raw results saved → {out_path}")

    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()