"""
SA_TDCCEA.py - Sensitivity-Adaptive TD-CCEA

Phase 1: Sensitivity Analysis — test each weight at 0.0 and 1.0 to find influence
Phase 2: Evolution — TD-CCEA with per-weight mutation rates based on influence
Phase 3: Comparison — side-by-side vs standard TD-CCEA (centroid)
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

N_POP = 20
K_GAMES = 3
TOTAL_GENERATIONS = 30
VALIDATION_GAMES = 5    # games per opponent for final validation

SENSITIVITY_FILE = "sensitivity_scores.json"


# ===============================================================
#  GAME SERVER
# ===============================================================
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


# ===============================================================
#  HELPERS
# ===============================================================
def euclidean_dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


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
    total_wins = 0
    total_games = 0
    per_opp = {}
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


# ===============================================================
#  PHASE 1: SENSITIVITY ANALYSIS
# ===============================================================
def run_sensitivity_analysis(server):
    """Test each weight at 0.0 and 1.0 (others at 0.5) to find influence."""
    print()
    print("=" * 70)
    print("  PHASE 1: SENSITIVITY ANALYSIS")
    print("=" * 70)
    print("  Testing each weight at 0.0 and 1.0 (others fixed at 0.5)")
    print()

    baseline = [0.5] * DIMENSIONS

    # Get baseline win rate
    print("  Testing baseline [all 0.5]...")
    baseline_wins = 0
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
    print(f"  Baseline win rate: {baseline_wr:.1%}")
    print()

    # Test each weight
    influence_scores = []

    print(f"  {'#':>3s}  {'Weight':<22s}  {'WR at 0.0':>10s}  {'WR at 1.0':>10s}  {'Influence':>10s}  Bar")
    print("  " + "-" * 75)

    for w_idx in range(DIMENSIONS):
        # Test at 0.0
        test_low = baseline[:]
        test_low[w_idx] = 0.0
        low_wins = 0
        low_games = 0
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
        test_high = baseline[:]
        test_high[w_idx] = 1.0
        high_wins = 0
        high_games = 0
        for opp in OPPONENT_ROTATION:
            for _ in range(K_GAMES):
                winner, _ = server.run_game(test_high, opp)
                if winner == 0:
                    high_wins += 1
                elif winner == -1:
                    high_wins += 0.3
                high_games += 1
        wr_high = high_wins / high_games

        # Influence = how much this weight changes the outcome
        influence = abs(wr_high - wr_low)
        influence_scores.append(influence)

        bar_len = int(influence * 50)
        bar = "#" * bar_len + "." * (50 - bar_len)
        print(f"  w{w_idx:2d}  {WEIGHT_NAMES[w_idx]:<22s}  {wr_low:10.1%}  {wr_high:10.1%}  {influence:10.1%}  [{bar[:20]}]")

    # Normalize influence scores to mutation rates
    # INVERTED: high influence = SMALL mutation (fine-tune carefully)
    #           low influence  = LARGE mutation (explore freely)
    max_inf = max(influence_scores) if max(influence_scores) > 0 else 1.0
    normalized = []
    for inf in influence_scores:
        # Scale to [0.08, 0.20]: high influence → 0.08, low influence → 0.20
        scaled = 0.20 - (inf / max_inf) * 0.12
        normalized.append(scaled)

    # Print ranking
    print()
    print("  INFLUENCE RANKING (most to least important):")
    print("  " + "-" * 50)
    ranked = sorted(range(DIMENSIONS), key=lambda i: influence_scores[i], reverse=True)
    for rank, idx in enumerate(ranked):
        marker = " <<<" if rank < 5 else ""
        print(f"  {rank+1:2d}. w{idx:2d} {WEIGHT_NAMES[idx]:<22s} influence={influence_scores[idx]:.1%}  mut_rate={normalized[idx]:.3f}{marker}")

    # Save to file
    result = {
        "influence_scores": influence_scores,
        "normalized_mutation_rates": normalized,
        "baseline_winrate": baseline_wr,
        "weight_names": WEIGHT_NAMES,
    }
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), SENSITIVITY_FILE)
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {SENSITIVITY_FILE}")

    return normalized


# ===============================================================
#  TD-CCEA AGENT (BASE CLASS)
# ===============================================================
class TDCCEAAgent:
    """Standard TD-CCEA agent with centroid-based environment."""

    def __init__(self, params=None, per_weight_mutation=None):
        if params is None:
            self.params = [random.random() for _ in range(DIMENSIONS)]
        else:
            self.params = [max(0.0, min(1.0, float(x))) for x in params]
        self.H_i = []
        self.e_i = 0.0
        self.mu_i = 0.0
        self.per_weight_mut = per_weight_mutation  # None = standard, list = SA version

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
        if self.per_weight_mut is not None:
            # SA-TDCCEA: per-weight mutation rates
            for j, p in enumerate(self.params):
                sigma_j = self.per_weight_mut[j] * (1.0 - self.e_i) + 0.03
                noise = random.gauss(0, sigma_j)
                offspring_params.append(p + noise)
        else:
            # Standard: same mutation for all weights
            for p in self.params:
                noise = random.gauss(0, self.mu_i)
                offspring_params.append(p + noise)
        child = TDCCEAAgent(offspring_params, self.per_weight_mut)
        return child


# ===============================================================
#  EVOLUTION LOOP (shared by both versions)
# ===============================================================
def run_evolution(server, label, per_weight_mutation=None):
    """Run TD-CCEA evolution. If per_weight_mutation is provided, use SA version."""

    print(f"\n{'='*60}")
    print(f"  RUNNING: {label}")
    print(f"{'='*60}")

    population = [TDCCEAAgent(per_weight_mutation=per_weight_mutation) for _ in range(N_POP)]
    archive = []
    gen_best_scores = []
    DELTA_S = 0.85
    DELTA_F = 0.20
    A_MAX = 15

    t_start = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]

        # Step 1: Compute environment meta-state (centroid)
        centroid = [0.0] * DIMENSIONS
        for agent in population:
            for j in range(DIMENSIONS):
                centroid[j] += agent.params[j]
        current_env = [x / len(population) for x in centroid]

        # Step 2 & 3: Evaluate fitness, update history
        for agent in population:
            f_i = evaluate_weights(server, agent.params, opponent)
            agent.update_history(f_i, current_env)
            agent.compute_dynamic_trajectory(current_env)
            agent.compute_adaptive_mutation()

        # Best agent this generation
        best_agent = max(population, key=lambda a: a.e_i)
        gen_best_scores.append(best_agent.e_i)
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | Opp: {opponent:25s} | Best e_i: {best_agent.e_i:.3f}")

        # Step 9: Remove redundant (similarity > delta_s)
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

        # Step 8: Remove if e_i < delta_f AND |H_i| >= 3
        final_survivors = []
        for agent in survivors:
            if agent.e_i < DELTA_F and len(agent.H_i) >= 3:
                pass  # removed
            else:
                final_survivors.append(agent)

        # Ensure minimum population
        while len(final_survivors) < 10 and len(survivors) > len(final_survivors):
            for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
                if a not in final_survivors:
                    final_survivors.append(a)
                    break

        # Step 10: Insert top-m into archive
        sorted_gen = sorted(population, key=lambda a: a.e_i, reverse=True)
        for top_ag in sorted_gen[:3]:
            archived = TDCCEAAgent(top_ag.params, per_weight_mutation)
            archive.append(archived)
        if len(archive) > A_MAX:
            archive = archive[-A_MAX:]

        # Step 11: Breed offspring from survivors union archive
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

    # Return best from final population (not archive, to avoid e_i=0 issue)
    best = max(population, key=lambda a: a.e_i)
    return best.params, elapsed, gen_best_scores


# ===============================================================
#  HILL CLIMBING
# ===============================================================
def run_hill_climbing(server):
    """Standard Hill Climbing: single solution + neighbors."""
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


# ===============================================================
#  MAIN
# ===============================================================
def main():
    print()
    print("#" * 70)
    print("  SA-TDCCEA: SENSITIVITY-ADAPTIVE TD-CCEA")
    print("  vs TD-CCEA CENTROID vs HILL CLIMBING")
    print("#" * 70)
    print()

    # ---- PHASE 1: Sensitivity Analysis ----
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), SENSITIVITY_FILE)

    if os.path.exists(filepath):
        print(f"  Found existing {SENSITIVITY_FILE}, loading...")
        with open(filepath, "r") as f:
            data = json.load(f)
        per_weight_mut = data["normalized_mutation_rates"]
        print(f"  Loaded {len(per_weight_mut)} mutation rates")
        print()
        for i in range(DIMENSIONS):
            bar_len = int(per_weight_mut[i] * 100)
            bar = "#" * bar_len + "." * (25 - bar_len)
            print(f"    w{i:2d} {WEIGHT_NAMES[i]:22s} mut={per_weight_mut[i]:.3f} [{bar[:25]}]")
        print()
    else:
        print(f"  No {SENSITIVITY_FILE} found. Running sensitivity analysis...")
        server = GameServer()
        server.start()
        per_weight_mut = run_sensitivity_analysis(server)
        server.stop()
        print()

    # ---- PHASE 2: Evolution (3 algorithms) ----
    algorithms = [
        ("SA-TDCCEA", per_weight_mut),
        ("TD-CCEA Centroid", None),
    ]

    all_results = []

    for algo_name, pwm in algorithms:
        server = GameServer()
        server.start()
        print(f"  [OK] Server started for {algo_name}")
        best_params, elapsed, gen_scores = run_evolution(server, algo_name, per_weight_mutation=pwm)

        # Validate against all opponents
        print(f"\n  Validating {algo_name} against all opponents...")
        overall_wr, per_opp = validate_weights(server, best_params)
        server.stop()

        # Show results like benchmark_all.py
        print(f"\n  +-- {algo_name} RESULTS ----------------------------")
        print(f"  |  Training Time:   {elapsed:.1f}s")
        print(f"  |  Overall Win Rate: {overall_wr:.1f}%")
        for opp, wr in per_opp.items():
            status = "[W]" if wr >= 50 else "[L]"
            print(f"  |  {status} vs {opp:25s}: {wr:.0f}%")
        print(f"  +--------------------------------------------------")

        all_results.append({
            'name': algo_name,
            'time': elapsed,
            'overall_winrate': overall_wr,
            'per_opponent': per_opp,
            'gen_scores': gen_scores,
            'best_params': best_params,
        })

    # Hill Climbing
    server = GameServer()
    server.start()
    print(f"  [OK] Server started for Hill Climbing")
    hc_params, hc_time, hc_scores = run_hill_climbing(server)

    print(f"\n  Validating Hill Climbing against all opponents...")
    hc_overall, hc_per_opp = validate_weights(server, hc_params)
    server.stop()

    print(f"\n  +-- Hill Climbing RESULTS ----------------------------")
    print(f"  |  Training Time:   {hc_time:.1f}s")
    print(f"  |  Overall Win Rate: {hc_overall:.1f}%")
    for opp, wr in hc_per_opp.items():
        status = "[W]" if wr >= 50 else "[L]"
        print(f"  |  {status} vs {opp:25s}: {wr:.0f}%")
    print(f"  +--------------------------------------------------")

    all_results.append({
        'name': 'Hill Climbing',
        'time': hc_time,
        'overall_winrate': hc_overall,
        'per_opponent': hc_per_opp,
        'gen_scores': hc_scores,
        'best_params': hc_params,
    })

    # ---- FINAL COMPARISON TABLE (like benchmark_all.py) ----
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"  {'Algorithm':<20s} {'Win Rate':>10s} {'Time':>10s}")
    print(f"  {'_'*20} {'_'*10} {'_'*10}")
    for r in sorted(all_results, key=lambda x: x['overall_winrate'], reverse=True):
        print(f"  {r['name']:<20s} {r['overall_winrate']:>9.1f}% {r['time']:>9.1f}s")

    # ---- PER-OPPONENT HEATMAP TABLE ----
    print(f"\n{'='*60}")
    print(f"  WIN RATE PER OPPONENT")
    print(f"{'='*60}")
    print()

    header = f"  {'Opponent':<25s}"
    for r in all_results:
        header += f"  {r['name']:>12s}"
    print(header)
    print("  " + "-" * (25 + 14 * len(all_results)))

    for opp in OPPONENT_ROTATION:
        row = f"  {opp:<25s}"
        rates = [r['per_opponent'].get(opp, 0) for r in all_results]
        best_rate = max(rates)
        for i, r in enumerate(all_results):
            wr = r['per_opponent'].get(opp, 0)
            marker = " *" if wr == best_rate and rates.count(best_rate) == 1 else "  "
            row += f"  {wr:>9.0f}%{marker}"
        print(row)

    print("  " + "-" * (25 + 14 * len(all_results)))
    overall_row = f"  {'OVERALL':<25s}"
    for r in all_results:
        overall_row += f"  {r['overall_winrate']:>9.1f}%  "
    print(overall_row)

    # ---- RANKING ----
    ranked = sorted(all_results, key=lambda x: x['overall_winrate'], reverse=True)
    print(f"\n  +{'_'*55}+")
    print(f"  |  FINAL RANKING                                      |")
    print(f"  +{'_'*55}+")
    medals = ["1st", "2nd", "3rd"]
    for i, r in enumerate(ranked):
        print(f"  |  {medals[i]}  {r['name']:<22s}  {r['overall_winrate']:5.1f}% win rate  ({r['time']:5.0f}s)  |")
    print(f"  +{'_'*55}+")

    if len(ranked) >= 2:
        d = ranked[0]['overall_winrate'] - ranked[1]['overall_winrate']
        print(f"\n  {ranked[0]['name']} beats {ranked[1]['name']} by {d:.1f}%")
    if len(ranked) >= 3:
        d = ranked[0]['overall_winrate'] - ranked[2]['overall_winrate']
        print(f"  {ranked[0]['name']} beats {ranked[2]['name']} by {d:.1f}%")

    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
