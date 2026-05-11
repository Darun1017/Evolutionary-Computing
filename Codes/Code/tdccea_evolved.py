"""
tdccea_evolved.py — TD-CCEA with Evolved Behavioral Weights

Instead of selecting from 14 pre-built strategies, this version evolves
15 continuous behavioral weights that control a custom Java AI ("EvolvedBrainAI")
which makes per-tick decisions during gameplay.

The 15 weights control:
  w0  worker_ratio      – economy vs military production priority
  w1  max_workers        – maximum worker count (scaled 1–12)
  w2  expand_threshold   – when to build second base
  w3  light_preference   – preference for Light units
  w4  heavy_preference   – preference for Heavy units  
  w5  ranged_preference  – preference for Ranged units
  w6  barracks_timing    – build barracks early vs late
  w7  aggression         – eagerness to attack
  w8  attack_threshold   – minimum army before attacking
  w9  defend_radius      – patrol distance from base
  w10 target_priority    – target workers (0) vs military (1)
  w11 defense_trigger    – when to switch to defense mode
  w12 counter_rush       – rush defense reaction strength
  w13 tech_switch        – adapt unit mix to counter enemy
  w14 economy_pressure   – prioritize destroying enemy economy

NOTE: This file does NOT modify any existing code. All original files
      (tdccea_32x32_14strat.py, BatchGameRunner.java, etc.) remain untouched.
"""

import math
import random
import time
import subprocess
import os

# ─── Paths ───
MICRORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "Simulation", "microrts")

# ─── Behavioral Weight Definitions ───
WEIGHT_NAMES = [
    "worker_ratio",        # w0:  economy vs military
    "max_workers",         # w1:  worker cap
    "expand_threshold",    # w2:  when to expand
    "light_preference",    # w3:  Light unit priority
    "heavy_preference",    # w4:  Heavy unit priority
    "ranged_preference",   # w5:  Ranged unit priority
    "barracks_timing",     # w6:  when to build barracks
    "aggression",          # w7:  attack eagerness
    "attack_threshold",    # w8:  min army to attack
    "defend_radius",       # w9:  defense perimeter
    "target_priority",     # w10: target workers vs military
    "defense_trigger",     # w11: when to shift to defense
    "counter_rush",        # w12: rush reaction strength
    "tech_switch",         # w13: counter enemy composition
    "economy_pressure",    # w14: destroy enemy economy
]

DIMENSIONS = len(WEIGHT_NAMES)  # 15

# ─── Game Configuration ───
MAP_PATH = "maps/chambers32x32.xml"
OPPONENT_BOT = "LightRush"
MAX_CYCLES = 10000

# ─── TD-CCEA Hyperparameters ───
N_POP = 20              # Population size
K_GAMES = 2             # Games per fitness evaluation
H_WINDOW = 5            # Trajectory history window
MU_BASE = 0.15          # Base mutation rate
DELTA_F = 0.25          # Fitness pruning threshold
DELTA_S = 0.85          # Similarity pruning threshold
A_MAX = 15              # Maximum archive size
TOTAL_GENERATIONS = 15  # Number of evolutionary generations

print(f"\n{'='*70}")
print(f"  TD-CCEA EVOLVED BRAIN | {DIMENSIONS} Behavioral Weights")
print(f"{'='*70}")
print(f"  [CONFIG] Weights: {DIMENSIONS} continuous behavioral parameters")
print(f"  [CONFIG] Map: {MAP_PATH}")
print(f"  [CONFIG] Opponent: {OPPONENT_BOT}")
print(f"  [CONFIG] Population: {N_POP} | Generations: {TOTAL_GENERATIONS}")
print(f"  [CONFIG] Games per evaluation: {K_GAMES}")
print(f"  [CONFIG] Search Space: CONTINUOUS {DIMENSIONS}D (impossible to brute-force)")
print()
for i, name in enumerate(WEIGHT_NAMES):
    print(f"    w{i:2d} = {name}")
print()


# ─── Java Game Server (uses EvolvedBatchGameRunner) ───
class EvolvedGameServer:
    """Manages a persistent Java process using EvolvedBatchGameRunner."""

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
            raise RuntimeError(f"EvolvedBatchGameRunner failed to start: {line}")
        print("  [✓] Evolved game server started")

    def run_game(self, weights, opponent, map_path=MAP_PATH, max_cycles=MAX_CYCLES):
        """
        Run a game with an evolved weight vector vs a standard opponent.
        
        Args:
            weights: list of 15 float values (the evolved brain DNA)
            opponent: string name of opponent bot (e.g. "EconomyMilitaryRush")
        
        Returns:
            (winner, cycles) — winner: 0=our AI, 1=opponent, -1=draw
        """
        # Format weights as semicolon-separated string
        weights_str = ";".join(f"{w:.4f}" for w in weights)
        request = f"EVOLVED:{weights_str},{opponent},{map_path},{max_cycles}\n"
        
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
            print("  [✓] Evolved game server stopped")


# ─── TD-CCEA Math Functions ───
def euclidean_dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def get_population_centroid(population):
    """Calculates the average DNA of the current population (environment state)."""
    centroid = [0.0] * DIMENSIONS
    for agent in population:
        for j in range(DIMENSIONS):
            centroid[j] += agent.params[j]
    return [x / len(population) for x in centroid]


# ─── TD-CCEA Agent ───
class EvolvedAgent:
    """
    An agent whose DNA is a 15-dimensional weight vector controlling
    every aspect of gameplay behavior.
    """
    
    def __init__(self, params=None):
        if params is None:
            self.params = [random.random() for _ in range(DIMENSIONS)]
        else:
            self.params = [max(0.0, min(1.0, float(x))) for x in params]
        self.H_i = []       # Trajectory history: list of (fitness, environment_centroid)
        self.e_i = 0.0      # Dynamic trajectory fitness
        self.mu_i = 0.0     # Adaptive mutation rate
        
    def get_weights_str(self):
        """Return a compact string of the weight vector for display."""
        return " ".join(f"{w:.2f}" for w in self.params)
    
    def get_strategy_profile(self):
        """Return a human-readable summary of what this weight vector means."""
        profile = []
        
        # Economy style
        if self.params[0] > 0.6:
            profile.append("heavy-economy")
        elif self.params[0] < 0.3:
            profile.append("minimal-workers")
        
        # Military composition
        prefs = [self.params[3], self.params[4], self.params[5]]
        total = sum(prefs) or 1
        names = ["Light", "Heavy", "Ranged"]
        dominant = names[prefs.index(max(prefs))]
        profile.append(f"{dominant}-focused")
        
        # Aggression
        if self.params[7] > 0.7:
            profile.append("aggressive")
        elif self.params[7] < 0.3:
            profile.append("defensive")
        else:
            profile.append("balanced")
            
        # Timing
        if self.params[6] < 0.3:
            profile.append("early-barracks")
        elif self.params[6] > 0.7:
            profile.append("late-barracks")
            
        return " | ".join(profile)
        
    def update_history(self, fitness, current_environment):
        """Store fitness result with the environment snapshot."""
        self.H_i.append((fitness, current_environment))
        if len(self.H_i) > H_WINDOW:
            self.H_i.pop(0)
            
    def compute_dynamic_trajectory(self, current_environment):
        """
        TD-CCEA core: dynamically weight past fitness by environment similarity.
        Environments that were similar to the current one get higher influence.
        """
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
        """Mutation rate adapts: low fitness → explore more, high fitness → fine-tune."""
        self.mu_i = MU_BASE * (1.0 - self.e_i) + 0.03
        return self.mu_i
        
    def mutate(self):
        """Create offspring by adding Gaussian noise scaled by adaptive mutation rate."""
        offspring_params = []
        for p in self.params:
            noise = random.gauss(0, self.mu_i)
            offspring_params.append(p + noise)
        return EvolvedAgent(offspring_params)


# ─── Main Evolution Loop ───
def run_evolved_tdccea():
    print(f"\n{'='*70}")
    print(f"  STARTING EVOLUTION: {DIMENSIONS} Behavioral Weights")
    print(f"  Opponent: {OPPONENT_BOT} | Map: {MAP_PATH}")
    print(f"{'='*70}\n")
    
    server = EvolvedGameServer()
    server.start()
    
    population = [EvolvedAgent() for _ in range(N_POP)]
    archive = []
    
    # Track evolution progress
    generation_results = []
    
    t_start = time.time()
    
    for gen in range(1, TOTAL_GENERATIONS + 1):
        gen_start = time.time()
        
        # Calculate the environment state (population centroid)
        current_env = get_population_centroid(population)
        
        print(f"  ┌─ Generation {gen}/{TOTAL_GENERATIONS} {'─'*45}")
        
        # ── Evaluate each agent by playing actual games ──
        for i, agent in enumerate(population):
            wins = 0
            total_cycles = 0
            for _ in range(K_GAMES):
                winner, cycles = server.run_game(
                    agent.params, OPPONENT_BOT,
                    map_path=MAP_PATH, max_cycles=MAX_CYCLES
                )
                if winner == 0:
                    wins += 1
                elif winner == -1:
                    wins += 0.3  # Partial credit for draws
                total_cycles += cycles
            
            f_i = wins / K_GAMES
            avg_cycles = total_cycles / K_GAMES
            
            agent.update_history(f_i, current_env)
            agent.compute_dynamic_trajectory(current_env)
            agent.compute_adaptive_mutation()
        
        # ── Find and display results ──
        best_agent = max(population, key=lambda a: a.e_i)
        avg_ei = sum(a.e_i for a in population) / len(population)
        
        # Count unique strategy profiles
        profiles = set(a.get_strategy_profile() for a in population)
        
        print(f"  │  Best e_i: {best_agent.e_i:.3f} | Avg e_i: {avg_ei:.3f} | Diversity: {len(profiles)} profiles")
        print(f"  │  Best Strategy: {best_agent.get_strategy_profile()}")
        print(f"  │  Best Weights: [{best_agent.get_weights_str()}]")
        
        generation_results.append({
            'gen': gen,
            'best_ei': best_agent.e_i,
            'avg_ei': avg_ei,
            'best_profile': best_agent.get_strategy_profile(),
            'best_params': best_agent.params.copy(),
            'diversity': len(profiles)
        })
        
        # ── Diversity-based survival selection ──
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

        # ── Fitness-based pruning ──
        final_survivors = []
        for agent in survivors:
            if agent.e_i < DELTA_F and len(agent.H_i) >= 3:
                pass  # prune low-fitness agents with enough history
            else:
                final_survivors.append(agent)
                
        # Ensure minimum population
        while len(final_survivors) < 10 and len(survivors) > len(final_survivors):
            for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
                if a not in final_survivors:
                    final_survivors.append(a)
                    break
                    
        # ── Archive top performers ──
        sorted_gen = sorted(population, key=lambda a: a.e_i, reverse=True)
        for top_ag in sorted_gen[:3]:
            archive.append(EvolvedAgent(top_ag.params))
        if len(archive) > A_MAX:
            archive = archive[-A_MAX:]
             
        # ── Breed next generation ──
        breeding_pool = final_survivors
        population = final_survivors.copy()
        while len(population) < N_POP:
            candidates = random.sample(
                breeding_pool + archive, 
                min(3, len(breeding_pool + archive))
            )
            parent = max(candidates, key=lambda a: a.e_i)
            population.append(parent.mutate())
        
        gen_time = time.time() - gen_start
        print(f"  │  Survivors: {len(final_survivors)} | Archive: {len(archive)} | Time: {gen_time:.1f}s")
        print(f"  └{'─'*60}")
    
    elapsed = time.time() - t_start
    server.stop()
    
    # ═══════════════════════════════════════════════════════════════
    #  FINAL RESULTS
    # ═══════════════════════════════════════════════════════════════
    
    best = max(archive, key=lambda a: a.e_i)
    
    print(f"\n{'='*70}")
    print(f"  EVOLUTION COMPLETE")
    print(f"{'='*70}")
    print(f"  Total Time:  {elapsed:.1f}s")
    print(f"  Total Games: {TOTAL_GENERATIONS * N_POP * K_GAMES}")
    print(f"  Opponent:    {OPPONENT_BOT}")
    print(f"")
    print(f"  ┌─ BEST EVOLVED STRATEGY ─────────────────────────")
    print(f"  │  Profile: {best.get_strategy_profile()}")
    print(f"  │  Fitness: {best.e_i:.3f}")
    print(f"  │")
    print(f"  │  Evolved Behavioral Weights:")
    
    for i, name in enumerate(WEIGHT_NAMES):
        bar_len = int(best.params[i] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        pct = best.params[i] * 100
        print(f"  │   w{i:2d} {name:20s} {bar} {pct:5.1f}%")
    
    print(f"  │")
    print(f"  │  What this strategy does:")
    
    # Interpret the weights in plain English
    wk = best.params[0]
    print(f"  │   • Economy:  {'Heavy worker production' if wk > 0.6 else 'Minimal workers, fast military' if wk < 0.3 else 'Balanced economy'}")
    
    prefs = [best.params[3], best.params[4], best.params[5]]
    total = sum(prefs) or 1
    print(f"  │   • Army Mix: {prefs[0]/total*100:.0f}% Light, {prefs[1]/total*100:.0f}% Heavy, {prefs[2]/total*100:.0f}% Ranged")
    
    ag = best.params[7]
    print(f"  │   • Playstyle: {'Very aggressive (rushes early)' if ag > 0.7 else 'Very defensive (turtles)' if ag < 0.3 else 'Balanced attack timing'}")
    
    tp = best.params[10]
    print(f"  │   • Targeting: {'Destroy enemy economy first' if tp < 0.4 else 'Eliminate enemy army first' if tp > 0.6 else 'Balanced targeting'}")
    
    cr = best.params[12]
    print(f"  │   • Rush Defense: {'Strongly counters rushes' if cr > 0.6 else 'Weakly counters rushes' if cr < 0.3 else 'Moderate rush defense'}")
    
    print(f"  └────────────────────────────────────────────────────")
    
    # ── Save results ──
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evolved_best_weights.txt")
    with open(results_path, "w") as f:
        f.write("# TD-CCEA Evolved Brain — Best Weight Vector\n")
        f.write(f"# Opponent: {OPPONENT_BOT}\n")
        f.write(f"# Map: {MAP_PATH}\n")
        f.write(f"# Fitness: {best.e_i:.3f}\n")
        f.write(f"# Profile: {best.get_strategy_profile()}\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#\n")
        for i, name in enumerate(WEIGHT_NAMES):
            f.write(f"{name}={best.params[i]:.6f}\n")
    print(f"\n  Saved weights to: {results_path}")
    
    # ── Evolution history ──
    print(f"\n  Evolution History:")
    print(f"  {'Gen':>4s}  {'Best e_i':>8s}  {'Avg e_i':>8s}  {'Diversity':>9s}  Strategy Profile")
    print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*30}")
    for r in generation_results:
        print(f"  {r['gen']:4d}  {r['best_ei']:8.3f}  {r['avg_ei']:8.3f}  {r['diversity']:9d}  {r['best_profile']}")
    
    # ── Headless validation games ──
    print(f"\n  [Test] Running 5 validation games...")
    test_server = EvolvedGameServer()
    test_server.start()
    
    test_wins = 0
    test_games = 5
    for g in range(test_games):
        winner, cycles = test_server.run_game(best.params, OPPONENT_BOT)
        result_str = "WIN" if winner == 0 else "LOSS" if winner == 1 else "DRAW"
        print(f"    Game {g+1}/{test_games}: {result_str} (cycles: {cycles})")
        if winner == 0:
            test_wins += 1
    
    test_server.stop()
    
    win_rate = test_wins / test_games * 100
    print(f"\n  ┌─ FINAL VALIDATION ─────────────────────────────")
    print(f"  │  Win Rate: {test_wins}/{test_games} ({win_rate:.0f}%)")
    print(f"  │  vs {OPPONENT_BOT}")
    print(f"  └────────────────────────────────────────────────────")
    
    # ── Launch VISUAL GUI replay of the best evolved brain ──
    print(f"\n  [Visualizer] Launching VISUAL replay: EvolvedBrain vs {OPPONENT_BOT}")
    print(f"  [Visualizer] Watch the evolved AI play in real-time!")
    print(f"  [Visualizer] Window will close automatically after the game ends.\n")
    time.sleep(1)
    
    weights_str = ";".join(f"{w:.4f}" for w in best.params)
    
    cmd = [
        "java", "-cp", "lib/*;bin",
        "tests.EvolvedVisualGameRunner",
        f"EVOLVED:{weights_str}", OPPONENT_BOT, MAP_PATH
    ]
    subprocess.run(cmd, cwd=MICRORTS_DIR)
    print("  [Visualizer] GUI closed.")
    print()


if __name__ == "__main__":
    run_evolved_tdccea()
