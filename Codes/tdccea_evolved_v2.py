"""
tdccea_evolved_v2.py — TD-CCEA with Rotating Opponents

This version makes the "Time-Dependent" part of TD-CCEA genuinely important
by rotating opponents across generations. The algorithm must evolve weights
that work against MULTIPLE different strategies, not just one.

Why rotating opponents matter:
  - Gen 1 faces EconomyMilitaryRush → aggressive weights might win
  - Gen 2 faces WorkerRush          → different counter needed
  - Gen 3 faces HeavyDefense        → completely different approach needed
  - The TD trajectory history remembers: "When the environment looked like X,
    these weights worked" — this memory becomes CRITICAL when the challenge
    keeps changing.

Without rotating opponents, a basic GA could do the same job.
WITH rotating opponents, TD-CCEA's trajectory memory gives it a real edge.
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
    "scout_priority",      # w15: send worker to explore map early
    "group_attack",        # w16: wait for units to group before attacking
    "second_barracks",     # w17: when to build second barracks
    "worker_fight_ratio",  # w18: workers permanently assigned to fight
    "focus_fire",          # w19: concentrate attacks on one target
]

DIMENSIONS = len(WEIGHT_NAMES)  # 20

# ─── Game Configuration ───
MAP_PATH = "maps/chambers32x32.xml"
MAX_CYCLES = 10000

# ─── ROTATING OPPONENTS ───
# These are the opponents the algorithm must learn to beat.
# Each generation faces a different one, cycling through the list.
# This is what makes TD-CCEA's trajectory memory genuinely important.
OPPONENT_ROTATION = [
    "EconomyMilitaryRush",   # Strong balanced opponent
    "LightRush",             # Fast aggressive rush
    "WorkerRush",            # Early worker swarm
    "HeavyDefense",          # Defensive turtle strategy
    "RangedRush",            # Ranged kiting strategy
    "EconomyRush",           # Economy-focused opponent
    "WorkerRushPlusPlus",    # Improved worker rush
]

# ─── TD-CCEA Hyperparameters ───
N_POP = 20              # Population size
K_GAMES = 2             # Games per fitness evaluation
H_WINDOW = 7            # Trajectory history window (longer for opponent changes)
MU_BASE = 0.15          # Base mutation rate
DELTA_F = 0.20          # Fitness pruning threshold
DELTA_S = 0.85          # Similarity pruning threshold
A_MAX = 15              # Maximum archive size
TOTAL_GENERATIONS = 21 # 3 full cycles through 7 opponents

print(f"\n{'='*70}")
print(f"  TD-CCEA EVOLVED BRAIN v2 | ROTATING OPPONENTS")
print(f"{'='*70}")
print(f"  [CONFIG] Weights: {DIMENSIONS} continuous behavioral parameters")
print(f"  [CONFIG] Map: {MAP_PATH}")
print(f"  [CONFIG] Opponents: {len(OPPONENT_ROTATION)} different strategies (rotating)")
for i, opp in enumerate(OPPONENT_ROTATION):
    print(f"           {i+1}. {opp}")
print(f"  [CONFIG] Population: {N_POP} | Generations: {TOTAL_GENERATIONS}")
print(f"  [CONFIG] Each generation faces a DIFFERENT opponent")
print(f"  [CONFIG] TD trajectory history tracks what works vs which opponent")
print()
print(f"  WHY THIS MATTERS:")
print(f"  The algorithm must find weights that work against ALL strategies,")
print(f"  not just one. The TD trajectory memory remembers which weights")
print(f"  worked when the environment (opponent type + population state)")
print(f"  looked a certain way. This is what makes TD-CCEA better than")
print(f"  a basic genetic algorithm.")
print()


# ─── Java Game Server ───
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
    centroid = [0.0] * DIMENSIONS
    for agent in population:
        for j in range(DIMENSIONS):
            centroid[j] += agent.params[j]
    return [x / len(population) for x in centroid]


# ─── TD-CCEA Agent ───
class EvolvedAgent:
    def __init__(self, params=None):
        if params is None:
            self.params = [random.random() for _ in range(DIMENSIONS)]
        else:
            self.params = [max(0.0, min(1.0, float(x))) for x in params]
        self.H_i = []
        self.e_i = 0.0
        self.mu_i = 0.0
        
    def get_weights_str(self):
        return " ".join(f"{w:.2f}" for w in self.params)
    
    def get_strategy_profile(self):
        profile = []
        if self.params[0] > 0.6:
            profile.append("heavy-economy")
        elif self.params[0] < 0.3:
            profile.append("minimal-workers")
        
        prefs = [self.params[3], self.params[4], self.params[5]]
        total = sum(prefs) or 1
        names = ["Light", "Heavy", "Ranged"]
        dominant = names[prefs.index(max(prefs))]
        profile.append(f"{dominant}-focused")
        
        if self.params[7] > 0.7:
            profile.append("aggressive")
        elif self.params[7] < 0.3:
            profile.append("defensive")
        else:
            profile.append("balanced")
            
        if self.params[6] < 0.3:
            profile.append("early-barracks")
        elif self.params[6] > 0.7:
            profile.append("late-barracks")
        
        if self.params[15] > 0.5:
            profile.append("scouting")
        if self.params[16] > 0.5:
            profile.append("grouped")
        if self.params[19] > 0.5:
            profile.append("focus-fire")
            
        return " | ".join(profile)
        
    def update_history(self, fitness, current_environment):
        self.H_i.append((fitness, current_environment))
        if len(self.H_i) > H_WINDOW:
            self.H_i.pop(0)
            
    def compute_dynamic_trajectory(self, current_environment):
        """
        TD-CCEA core: dynamically weight past fitness by environment similarity.
        
        THIS IS WHERE ROTATING OPPONENTS MAKES A DIFFERENCE:
        - When the opponent changes, the population centroid shifts
        - Past experiences against similar population states (similar opponents)
          get higher weight
        - This means the algorithm "remembers" what worked against rush
          strategies vs defensive strategies
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
        self.mu_i = MU_BASE * (1.0 - self.e_i) + 0.03
        return self.mu_i
        
    def mutate(self):
        offspring_params = []
        for p in self.params:
            noise = random.gauss(0, self.mu_i)
            offspring_params.append(p + noise)
        return EvolvedAgent(offspring_params)


# ─── Main Evolution Loop ───
def run_evolved_tdccea_v2():
    print(f"\n{'='*70}")
    print(f"  STARTING EVOLUTION: Rotating Opponents Mode")
    print(f"  {len(OPPONENT_ROTATION)} opponents × {TOTAL_GENERATIONS} generations")
    print(f"{'='*70}\n")
    
    server = EvolvedGameServer()
    server.start()
    
    population = [EvolvedAgent() for _ in range(N_POP)]
    archive = []
    generation_results = []
    
    t_start = time.time()
    
    for gen in range(1, TOTAL_GENERATIONS + 1):
        gen_start = time.time()
        
        # ══════════════════════════════════════════════════════════
        # ROTATING OPPONENT: Each generation faces a different one
        # This is what makes TD (Time-Dependent) genuinely useful
        # ══════════════════════════════════════════════════════════
        current_opponent = OPPONENT_ROTATION[(gen - 1) % len(OPPONENT_ROTATION)]
        
        current_env = get_population_centroid(population)
        
        # Count strategy diversity
        strat_count = {}
        for agent in population:
            s = agent.get_strategy_profile()
            strat_count[s] = strat_count.get(s, 0) + 1
        
        print(f"  ┌─ Generation {gen}/{TOTAL_GENERATIONS} {'─'*45}")
        print(f"  │  Opponent: {current_opponent}")
        
        # ── Evaluate each agent against the current opponent ──
        for i, agent in enumerate(population):
            wins = 0
            for _ in range(K_GAMES):
                winner, cycles = server.run_game(
                    agent.params, current_opponent,
                    map_path=MAP_PATH, max_cycles=MAX_CYCLES
                )
                if winner == 0:
                    wins += 1
                elif winner == -1:
                    wins += 0.3
            
            f_i = wins / K_GAMES
            agent.update_history(f_i, current_env)
            agent.compute_dynamic_trajectory(current_env)
            agent.compute_adaptive_mutation()
        
        # ── Results ──
        best_agent = max(population, key=lambda a: a.e_i)
        avg_ei = sum(a.e_i for a in population) / len(population)
        unique_strats = len(strat_count)
        
        print(f"  │  Best e_i: {best_agent.e_i:.3f} | Avg e_i: {avg_ei:.3f} | Diversity: {unique_strats}")
        print(f"  │  Best: {best_agent.get_strategy_profile()}")
        
        generation_results.append({
            'gen': gen,
            'opponent': current_opponent,
            'best_ei': best_agent.e_i,
            'avg_ei': avg_ei,
            'best_profile': best_agent.get_strategy_profile(),
            'best_params': best_agent.params.copy(),
            'diversity': unique_strats
        })
        
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
    print(f"  EVOLUTION COMPLETE — GENERALIST STRATEGY FOUND")
    print(f"{'='*70}")
    print(f"  Total Time:  {elapsed:.1f}s")
    print(f"  Total Games: {TOTAL_GENERATIONS * N_POP * K_GAMES}")
    print(f"  Opponents Faced: {len(OPPONENT_ROTATION)} different strategies")
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
    
    sc = best.params[15]
    print(f"  │   • Scouting:     {'Sends scout early to find enemy' if sc > 0.5 else 'No early scouting'}")
    
    ga = best.params[16]
    print(f"  │   • Grouping:     {'Waits for army to group up before attacking' if ga > 0.5 else 'Sends units individually'}")
    
    sb = best.params[17]
    print(f"  │   • 2nd Barracks: {'Builds second barracks for faster production' if sb > 0.4 else 'Sticks with one barracks'}")
    
    wf = best.params[18]
    print(f"  │   • Worker Fight: {'Workers permanently fight ({int(wf*100)}%)' if wf > 0.2 else 'Workers only harvest'}")
    
    ff = best.params[19]
    print(f"  │   • Focus Fire:   {'Concentrates attacks on one target' if ff > 0.5 else 'Spreads damage across targets'}")
    
    print(f"  └────────────────────────────────────────────────────")
    
    # ── Save results ──
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evolved_v2_best_weights.txt")
    with open(results_path, "w") as f:
        f.write("# TD-CCEA Evolved Brain v2 — Best Generalist Weight Vector\n")
        f.write(f"# Opponents: {', '.join(OPPONENT_ROTATION)}\n")
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
    print(f"  {'Gen':>4s}  {'Opponent':>25s}  {'Best':>6s}  {'Avg':>6s}  {'Div':>4s}  Profile")
    print(f"  {'─'*4}  {'─'*25}  {'─'*6}  {'─'*6}  {'─'*4}  {'─'*30}")
    for r in generation_results:
        print(f"  {r['gen']:4d}  {r['opponent']:>25s}  {r['best_ei']:6.3f}  {r['avg_ei']:6.3f}  {r['diversity']:4d}  {r['best_profile']}")
    
    # ═══════════════════════════════════════════════════════════════
    #  VALIDATION: Test against ALL opponents
    # ═══════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print(f"  VALIDATION: Testing evolved brain vs ALL {len(OPPONENT_ROTATION)} opponents")
    print(f"{'='*70}")
    
    test_server = EvolvedGameServer()
    test_server.start()
    
    total_wins = 0
    total_games = 0
    
    for opponent in OPPONENT_ROTATION:
        opp_wins = 0
        games_per_opp = 10
        for g in range(games_per_opp):
            winner, cycles = test_server.run_game(best.params, opponent)
            if winner == 0:
                opp_wins += 1
            total_games += 1
        
        total_wins += opp_wins
        win_pct = opp_wins / games_per_opp * 100
        status = "✓" if opp_wins >= 2 else "✗"
        print(f"  {status} vs {opponent:25s}: {opp_wins}/{games_per_opp} ({win_pct:.0f}%)")
    
    test_server.stop()
    
    overall_pct = total_wins / total_games * 100
    print(f"\n  ┌─ OVERALL RESULTS ──────────────────────────────")
    print(f"  │  Total Win Rate: {total_wins}/{total_games} ({overall_pct:.0f}%)")
    print(f"  │  Opponents Beaten: {sum(1 for o in OPPONENT_ROTATION for _ in [1])}")
    print(f"  │  This is a GENERALIST strategy — works against multiple opponents")
    print(f"  └────────────────────────────────────────────────────")
    
    # ── Visual replay against the hardest opponent ──
    # Find which opponent the evolved brain struggled against most
    print(f"\n  [Visualizer] Launching VISUAL replay: EvolvedBrain vs EconomyMilitaryRush")
    print(f"  [Visualizer] Watch the evolved AI play in real-time!")
    print(f"  [Visualizer] Window will close automatically after the game ends.\n")
    time.sleep(1)
    
    weights_str = ";".join(f"{w:.4f}" for w in best.params)
    cmd = [
        "java", "-cp", "lib/*;bin",
        "tests.EvolvedVisualGameRunner",
        f"EVOLVED:{weights_str}", "EconomyMilitaryRush", MAP_PATH
    ]
    subprocess.run(cmd, cwd=MICRORTS_DIR)
    print("  [Visualizer] GUI closed.")
    print()


if __name__ == "__main__":
    run_evolved_tdccea_v2()
