import math, random, time, subprocess, os, json

# ===============================================================
# CONFIGURATION
# ===============================================================
MICRORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Simulation", "microrts"
)
WEIGHT_NAMES = [
    "worker_ratio","max_workers","expand_threshold","light_preference",
    "heavy_preference","ranged_preference","barracks_timing","aggression",
    "attack_threshold","defend_radius","target_priority","defense_trigger",
    "counter_rush","tech_switch","economy_pressure","scout_priority",
    "group_attack","second_barracks","worker_fight_ratio","focus_fire",
]
DIMENSIONS    = len(WEIGHT_NAMES)
MAP_PATH      = "maps/chambers32x32.xml"
MAX_CYCLES    = 10000

OPPONENT_ROTATION = [
    "EconomyMilitaryRush","LightRush","WorkerRush",
    "HeavyDefense","RangedRush","EconomyRush","WorkerRushPlusPlus",
]

N_POP             = 20
K_GAMES           = 3   # For evolutionary generations
SA_GAMES          = 15  # For bias checking (sensitivity analysis)
TOTAL_GENERATIONS = 30  # Kept short for quick testing
VALIDATION_GAMES  = 5
A_MAX             = 15
DELTA_S           = 0.85
DELTA_F           = 0.20

# ===============================================================
# GAME SERVER
# ===============================================================
class GameServer:
    def __init__(self): self.process = None

    def start(self):
        self.process = subprocess.Popen(
            ["java", "-cp", "lib/*;bin", "tests.EvolvedBatchGameRunner"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=MICRORTS_DIR
        )
        if self.process.stdout.readline().strip() != "READY":
            raise RuntimeError("Server failed to start")

    def run_game(self, weights, opponent):
        ws = ";".join(f"{w:.4f}" for w in weights)
        self.process.stdin.write(f"EVOLVED:{ws},{opponent},{MAP_PATH},{MAX_CYCLES}\n")
        self.process.stdin.flush()
        r = self.process.stdout.readline().strip()
        if r.startswith("RESULT:"):
            p = r.replace("RESULT:", "").split(",")
            return int(p[0]), int(p[1])
        return -1, MAX_CYCLES

    def stop(self):
        if self.process:
            try:
                self.process.stdin.write("EXIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()


def euclidean_dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def evaluate_weights(server, weights, opponent):
    wins = 0
    for _ in range(K_GAMES):
        w, _ = server.run_game(weights, opponent)
        if w == 0:    wins += 1
        elif w == -1: wins += 0.3
    return wins / K_GAMES


# ===============================================================
# DIRECTIONAL SENSITIVITY ANALYSIS (NEW)
# ===============================================================
def run_directional_sensitivity(server):
    print(f"\n{'='*60}")
    print("  RUNNING NEW DIRECTIONAL SENSITIVITY ANALYSIS")
    print(f"{'='*60}")
    
    baseline = [0.5] * DIMENSIONS
    bw = bg = 0
    for opp in OPPONENT_ROTATION:
        for _ in range(SA_GAMES):
            w, _ = server.run_game(baseline, opp)
            if w == 0:    bw += 1
            elif w == -1: bw += 0.3
            bg += 1
    baseline_wr = bw / bg
    print(f"  Baseline win rate (0.5 all): {baseline_wr:.1%}\n")

    directions = [0.0] * DIMENSIONS
    influence  = [0.0] * DIMENSIONS

    for w_idx in range(DIMENSIONS):
        # 1. Test Extremes (0.0 and 1.0)
        win_0 = win_1 = 0.0
        
        test_0 = baseline[:]; test_0[w_idx] = 0.0
        test_1 = baseline[:]; test_1[w_idx] = 1.0
        
        w0_wins = w0_games = 0
        w1_wins = w1_games = 0
        
        for opp in OPPONENT_ROTATION:
            for _ in range(SA_GAMES):
                w, _ = server.run_game(test_0, opp)
                if w == 0: w0_wins += 1
                elif w == -1: w0_wins += 0.3
                w0_games += 1
                
                w, _ = server.run_game(test_1, opp)
                if w == 0: w1_wins += 1
                elif w == -1: w1_wins += 0.3
                w1_games += 1
                
        win_0 = w0_wins / w0_games
        win_1 = w1_wins / w1_games
        
        delta = win_1 - win_0
        
        # 2. If tie, check 0.25 and 0.75
        if abs(delta) < 0.001:
            test_025 = baseline[:]; test_025[w_idx] = 0.25
            test_075 = baseline[:]; test_075[w_idx] = 0.75
            
            w025_w = w025_g = 0
            w075_w = w075_g = 0
            
            for opp in OPPONENT_ROTATION:
                for _ in range(SA_GAMES):
                    w, _ = server.run_game(test_025, opp)
                    if w == 0: w025_w += 1
                    elif w == -1: w025_w += 0.3
                    w025_g += 1
                    
                    w, _ = server.run_game(test_075, opp)
                    if w == 0: w075_w += 1
                    elif w == -1: w075_w += 0.3
                    w075_g += 1
            
            win_025 = w025_w / w025_g
            win_075 = w075_w / w075_g
            delta = win_075 - win_025
            influence[w_idx] = abs(delta)
        else:
            influence[w_idx] = abs(delta)

        # 3. Assign Direction Bias
        if delta > 0.01:
            directions[w_idx] = 1.0   # Going up is better
            dir_str = "UP (+)"
        elif delta < -0.01:
            directions[w_idx] = -1.0  # Going down is better
            dir_str = "DOWN (-)"
        else:
            # Still tied? Assign random direction to break symmetry
            directions[w_idx] = random.choice([-1.0, 1.0])
            dir_str = f"RANDOM ({'+' if directions[w_idx]>0 else '-'})"
            
        print(f"  w{w_idx:2d} {WEIGHT_NAMES[w_idx]:<22s} | win_0={win_0:.2f} win_1={win_1:.2f} | diff={delta:+.2f} | Bias: {dir_str}")

    # 4. Calculate adaptive mutation spreads (proportional to influence)
    max_inf = max(influence) if max(influence) > 0 else 1.0
    spreads = [0.08 + (inf / max_inf) * 0.12 for inf in influence]
    
    return directions, spreads


# ===============================================================
# AGENT WITH DIRECTIONAL MUTATION
# ===============================================================
class DirectionalAgent:
    def __init__(self, params=None, directions=None, spreads=None):
        self.params = ([random.random() for _ in range(DIMENSIONS)] if params is None
                       else [max(0.0, min(1.0, float(x))) for x in params])
        self.H_i = []
        self.e_i = 0.0
        self.directions = directions
        self.spreads = spreads

    def update_history(self, f, env):
        self.H_i.append((f, env))
        if len(self.H_i) > 7: self.H_i.pop(0)

    def compute_trajectory(self, env):
        if not self.H_i: self.e_i = 0.0; return
        md = math.sqrt(DIMENSIONS)
        ws = [max(0.0, 1.0 - euclidean_dist(env, pe) / md) for _, pe in self.H_i]
        t  = sum(ws)
        ws = [w/t for w in ws] if t > 0 else [1.0/len(ws)] * len(ws)
        self.e_i = sum(w * self.H_i[i][0] for i, w in enumerate(ws))

    def mutate(self):
        cp = []
        for j, p in enumerate(self.params):
            # Base spread dynamically scales down as e_i (fitness) improves
            sigma_j = self.spreads[j] * (1.0 - self.e_i) + 0.02
            
            # Directional Bias Shift:
            # If direction is +1, mean of mutation is slightly positive.
            # If direction is -1, mean of mutation is slightly negative.
            # The shift magnitude is proportional to sigma_j.
            shift = self.directions[j] * (sigma_j * 0.5) 
            
            mutated_val = p + random.gauss(shift, sigma_j)
            cp.append(max(0.0, min(1.0, mutated_val)))
            
        return DirectionalAgent(cp, self.directions, self.spreads)


# ===============================================================
# EVOLUTIONARY LOOP
# ===============================================================
def run_directional_evolution(server, directions, spreads):
    print(f"\n{'='*60}")
    print(f"  RUNNING DIRECTIONAL SA-TDCCEA")
    print(f"{'='*60}")
    
    population = [DirectionalAgent(directions=directions, spreads=spreads) for _ in range(N_POP)]
    archive    = []
    t0         = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        opp      = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
        centroid = [sum(a.params[j] for a in population) / len(population)
                    for j in range(DIMENSIONS)]
                    
        for agent in population:
            f_i = evaluate_weights(server, agent.params, opp)
            agent.update_history(f_i, centroid)
            agent.compute_trajectory(centroid)
            
        best = max(population, key=lambda a: a.e_i)
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | {opp:25s} | Best e_i={best.e_i:.3f}")
        
        md        = math.sqrt(DIMENSIONS)
        survivors = []
        for a in sorted(population, key=lambda a: a.e_i, reverse=True):
            if not any(1.0 - euclidean_dist(a.params, u.params) / md > DELTA_S for u in survivors):
                survivors.append(a)
                
        final = [a for a in survivors if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
        while len(final) < 10 and len(survivors) > len(final):
            for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
                if a not in final: final.append(a); break
                
        for top in sorted(population, key=lambda a: a.e_i, reverse=True)[:3]:
            ar = DirectionalAgent(top.params, directions, spreads)
            ar.e_i, ar.H_i = top.e_i, top.H_i[:]
            archive.append(ar)
            
        archive.sort(key=lambda a: a.e_i, reverse=True)
        archive    = archive[:A_MAX]
        pool       = final + archive
        population = final[:]
        
        while len(population) < N_POP:
            cands = random.sample(pool, min(3, len(pool)))
            population.append(max(cands, key=lambda a: a.e_i).mutate())

    best_all = max(archive, key=lambda a: a.e_i) if archive else max(population, key=lambda a: a.e_i)
    return best_all.params, time.time() - t0


def validate_weights(server, weights):
    tw = tg = 0
    for opp in OPPONENT_ROTATION:
        for _ in range(VALIDATION_GAMES):
            w, _ = server.run_game(weights, opp)
            if w == 0: tw += 1
            tg += 1
    return tw / tg * 100


# ===============================================================
# MAIN
# ===============================================================
if __name__ == "__main__":
    print("Initializing Game Server...")
    server = GameServer()
    server.start()
    
    try:
        # Step 1: Directional Analysis
        directions, spreads = run_directional_sensitivity(server)
        
        # Step 2: Evolve with directionality
        best_weights, elapsed = run_directional_evolution(server, directions, spreads)
        
        # Step 3: Validate
        print(f"\n  Validating best weights vs all bots...")
        final_wr = validate_weights(server, best_weights)
        
        print(f"\n{'='*60}")
        print(f"  FINISHED in {elapsed:.1f}s")
        print(f"  Final Win Rate: {final_wr:.1f}%")
        weights_str = ", ".join(f"{w:.4f}" for w in best_weights)
        print(f"  Best Weights:   [{weights_str}]")
        print(f"{'='*60}")
        
        # Save to final_strategies.txt
        strat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_strategies.txt")
        with open(strat_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"Run Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n")
            f.write(f"Algorithm: SA_NEW Directional Mutation (Win Rate: {final_wr:.1f}%)\n")
            f.write(f"Weights: [{weights_str}]\n")
            f.write("-" * 40 + "\n")
        print(f"  Saved strategy to: {strat_path}")
        
    finally:
        server.stop()
