"""
===========================================================================
TD-CCEA: 14-STRATEGY EVOLUTION ON CUSTOM 32x32 BLOCKADE MAP
===========================================================================

This script evolves an agent across ALL 14 available microRTS strategies
on a custom 32x32 map with random wall blockades. The blockades force
pathfinding-dependent strategies and make the optimal choice non-obvious.

Strategies (14 total):
  AGGRESSIVE:  WorkerRush, WorkerRushPlusPlus, LightRush, HeavyRush, RangedRush
  DEFENSIVE:   LightDefense, HeavyDefense, RangedDefense, WorkerDefense
  ECONOMY:     EconomyRush, EconomyMilitaryRush
  BASELINE:    RandomBiased, Random, Passive

Key: The 14-dimensional search space makes this MUCH harder for simple 
baselines (GA, HC, ES) while TD-CCEA's trajectory memory and adaptive
mutation should discover the optimal strategy efficiently.
===========================================================================
"""

import math
import random
import time
import subprocess
import os

# ─── microRTS Server Configuration ───
MICRORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "Simulation", "microrts")

# ALL 14 strategies available in microRTS
BOT_TYPES = [
    # Aggressive Rush Strategies (5)
    "WorkerRush",           # 0: Swarm with workers — fast, cheap, no barracks needed
    "WorkerRushPlusPlus",   # 1: Improved worker targeting — smarter worker attacks
    "LightRush",            # 2: Build barracks → train Light units → attack
    "HeavyRush",            # 3: Build barracks → train Heavy units (slow but powerful)
    "RangedRush",           # 4: Build barracks → train Ranged units → kite attack
    # Defensive Strategies (4)
    "LightDefense",         # 5: Build Light units → defend base perimeter
    "HeavyDefense",         # 6: Build Heavy units → hold defensive positions
    "RangedDefense",        # 7: Build Ranged units → defend from distance
    "WorkerDefense",        # 8: Use workers for base defense
    # Economy-First Strategies (2)
    "EconomyRush",          # 9: Balance economy expansion + mixed military
    "EconomyMilitaryRush",  # 10: Heavy economy → then massive military buildup
    # Baseline/Random Strategies (3)
    "RandomBiased",         # 11: Semi-random with bias toward useful actions
    "Random",               # 12: Completely random actions
    "Passive",              # 13: Do absolutely nothing
]

# Custom 32x32 blockade map
MAP_PATH = "maps/chambers32x32.xml"
OPPONENT_BOT = "WorkerRush"  
MAX_CYCLES = 10000           # 32x32 maps need more cycles

# ─── TD-CCEA Hyperparameters ───
N_POP = 20               # Population size
K_GAMES = 2              # Games per fitness evaluation
H_WINDOW = 5             # Trajectory history window
MU_BASE = 0.20           # Base mutation rate
DELTA_F = 0.25           # Fitness pruning threshold
DELTA_S = 0.85           # Similarity pruning threshold
A_MAX = 15               # Maximum archive size
TOTAL_GENERATIONS = 30   # Number of evolutionary generations
DIMENSIONS = len(BOT_TYPES)  # 14 dimensions!

print(f"[CONFIG] Strategy Space: {DIMENSIONS} dimensions")
print(f"[CONFIG] Strategies: {', '.join(BOT_TYPES)}")
print(f"[CONFIG] Map: {MAP_PATH}")
print(f"[CONFIG] Opponent: {OPPONENT_BOT}")
print(f"[CONFIG] Search Space: {DIMENSIONS}D continuous → argmax → discrete strategy")


# ─── Java Game Server ───
class MicroRTSServer:
    """Manages a persistent Java process for running games."""

    def __init__(self):
        self.process = None

    def start(self):
        cmd = ["java", "-cp", "lib/*;bin", "tests.BatchGameRunner"]
        self.process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=MICRORTS_DIR
        )
        line = self.process.stdout.readline().strip()
        if line != "READY":
            raise RuntimeError(f"BatchGameRunner failed to start: {line}")
        print("  [✓] microRTS game server started (14 strategies loaded)")

    def run_game(self, bot1, bot2, map_path=MAP_PATH, max_cycles=MAX_CYCLES):
        request = f"{bot1},{bot2},{map_path},{max_cycles}\n"
        self.process.stdin.write(request)
        self.process.stdin.flush()
        response = self.process.stdout.readline().strip()
        if response.startswith("RESULT:"):
            parts = response.replace("RESULT:", "").split(",")
            return int(parts[0]), int(parts[1])
        else:
            print(f"    [!] Unexpected response: {response}")
            return -1, max_cycles

    def stop(self):
        if self.process:
            try:
                self.process.stdin.write("EXIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            print("  [✓] microRTS game server stopped")


# ─── TD-CCEA Math Functions ───
def euclidean_dist(param_i, param_j):
    return math.sqrt(sum((a-b)**2 for a, b in zip(param_i, param_j)))

def get_population_centroid(population):
    """Calculates the exact average DNA of the current environment/population."""
    centroid = [0.0] * DIMENSIONS
    for agent in population:
        for j in range(DIMENSIONS):
            centroid[j] += agent.params[j]
    return [x / len(population) for x in centroid]


# ─── TD-CCEA Agent ───
class TDCCEAAgent:
    def __init__(self, params=None):
        if params is None:
            self.params = [random.random() for _ in range(DIMENSIONS)]
        else:
            self.params = [max(0.0, min(1.0, float(x))) for x in params]
        self.H_i = []  # Stores Tuples: (Fitness, Environment_Centroid)
        self.e_i = 0.0
        self.mu_i = 0.0
        
    def get_bot_type(self):
        """Select strategy via argmax of the 14-dimensional weight vector."""
        best_idx = self.params.index(max(self.params))
        return BOT_TYPES[best_idx]
    
    def get_top_strategies(self, n=3):
        """Return top N strategies by weight (for logging diversity)."""
        indexed = sorted(enumerate(self.params), key=lambda x: x[1], reverse=True)
        return [(BOT_TYPES[i], w) for i, w in indexed[:n]]
        
    def update_history(self, fitness, current_environment):
        self.H_i.append((fitness, current_environment))
        if len(self.H_i) > H_WINDOW:
            self.H_i.pop(0)
            
    def compute_dynamic_trajectory(self, current_environment):
        """Dynamically calculate similarity weights instead of hardcoded ALPHA_W."""
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
        self.mu_i = MU_BASE * (1.0 - self.e_i) + 0.05
        return self.mu_i
        
    def mutate(self):
        offspring_params = []
        for p in self.params:
            noise = random.gauss(0, self.mu_i)
            offspring_params.append(p + noise)
        return TDCCEAAgent(offspring_params)


# ─── Main Evolution Loop ───
def run_tdccea_14strat():
    print("\n" + "=" * 70)
    print("  TD-CCEA: 14-STRATEGY EVOLUTION | 32x32 BLOCKADE MAP")
    print("=" * 70)
    
    server = MicroRTSServer()
    server.start()
    
    population = [TDCCEAAgent() for _ in range(N_POP)]
    archive = []
    
    # Track strategy distribution across generations
    strategy_history = []
    
    t_start = time.time()
    
    for gen in range(1, TOTAL_GENERATIONS + 1):
        # Calculate the mathematical meta-state of the environment
        current_env = get_population_centroid(population)
        
        # Count strategy distribution in current population
        strat_count = {}
        for agent in population:
            s = agent.get_bot_type()
            strat_count[s] = strat_count.get(s, 0) + 1
        
        print(f"\n{'─'*70}")
        print(f"  Generation {gen:02d}/{TOTAL_GENERATIONS:02d} | Population Strategy Distribution:")
        for s, c in sorted(strat_count.items(), key=lambda x: -x[1]):
            bar = "█" * c + "░" * (N_POP - c)
            print(f"    {s:22s}: {c:2d}/{N_POP} {bar}")
        
        strategy_history.append(strat_count.copy())
        
        # Evaluate each agent
        for i, agent in enumerate(population):
            wins = 0
            for _ in range(K_GAMES):
                winner, cycles = server.run_game(agent.get_bot_type(), OPPONENT_BOT, 
                                                  map_path=MAP_PATH, max_cycles=MAX_CYCLES)
                if winner == 0: wins += 1
                elif winner == -1: wins += 0.3
            f_i = wins / K_GAMES
            
            agent.update_history(f_i, current_env)
            agent.compute_dynamic_trajectory(current_env)
            agent.compute_adaptive_mutation()
            
        # Find and display best agent
        best_agent = max(population, key=lambda a: a.e_i)
        top3 = best_agent.get_top_strategies(3)
        top3_str = " | ".join([f"{name}:{w:.2f}" for name, w in top3])
        
        avg_ei = sum(a.e_i for a in population) / len(population)
        print(f"\n  Best: {best_agent.get_bot_type()} (e_i={best_agent.e_i:.3f}) | Avg e_i={avg_ei:.3f}")
        print(f"  Top-3 weights: {top3_str}")
        print(f"  Mutation rate: {best_agent.mu_i:.3f}")

        # Diversity-based survival selection
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

        # Fitness-based pruning
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
                    
        # Archive top performers
        sorted_gen = sorted(population, key=lambda a: a.e_i, reverse=True)
        for top_ag in sorted_gen[:3]:
            archive.append(TDCCEAAgent(top_ag.params))
        if len(archive) > A_MAX:
             archive = archive[-A_MAX:]
             
        # Breed next generation
        breeding_pool = final_survivors
        population = final_survivors.copy()
        while len(population) < N_POP:
            candidates = random.sample(breeding_pool + archive, min(3, len(breeding_pool + archive)))
            parent = max(candidates, key=lambda a: a.e_i)
            population.append(parent.mutate())
    
    elapsed = time.time() - t_start
    server.stop()
    
    # ─── Final Results ───
    best = archive[-1]
    print("\n" + "=" * 70)
    print("  TD-CCEA 14-STRATEGY EVOLUTION COMPLETE")
    print("=" * 70)
    print(f"  Dominant Strategy: {best.get_bot_type()}")
    print(f"  Total Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Total Games Played: {TOTAL_GENERATIONS * N_POP * K_GAMES}")
    print()
    
    # Show final weight distribution across all 14 strategies
    print("  Final Weight Distribution (14D DNA):")
    indexed_weights = sorted(enumerate(best.params), key=lambda x: x[1], reverse=True)
    for idx, weight in indexed_weights:
        bar = "█" * int(weight * 40)
        marker = " ← SELECTED" if idx == best.params.index(max(best.params)) else ""
        print(f"    {BOT_TYPES[idx]:22s}: {weight:.3f} {bar}{marker}")
    
    # Show strategy evolution over generations
    print(f"\n  Strategy Evolution Over {TOTAL_GENERATIONS} Generations:")
    all_strats_seen = set()
    for sh in strategy_history:
        all_strats_seen.update(sh.keys())
    
    for strat in sorted(all_strats_seen):
        timeline = ""
        for sh in strategy_history:
            count = sh.get(strat, 0)
            if count == 0:
                timeline += "·"
            elif count <= 3:
                timeline += "░"
            elif count <= 7:
                timeline += "▒"
            elif count <= 12:
                timeline += "▓"
            else:
                timeline += "█"
        print(f"    {strat:22s}: [{timeline}]")
    
    # Save best model
    with open("best_model_32x32_14strat.txt", "w") as f:
        f.write(best.get_bot_type())
    print(f"\n  Saved to: best_model_32x32_14strat.txt")
        
    # Launch visual replay
    print("\n  [Visualizer] Launching Post-Training Replay...")
    time.sleep(1)
    
    cmd = [
        "java", "-cp", "lib/*;bin",
        "tests.VisualGameRunner",
        best.get_bot_type(), OPPONENT_BOT, MAP_PATH
    ]
    print(f"  [Visualizer] {best.get_bot_type()} vs {OPPONENT_BOT}")
    subprocess.run(cmd, cwd=MICRORTS_DIR)
    print("  [Visualizer] GUI Terminated.")

    
if __name__ == "__main__":
    run_tdccea_14strat()
