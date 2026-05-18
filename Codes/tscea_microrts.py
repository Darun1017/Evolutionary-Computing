"""
TSCEA - Trajectory-guided Self-adaptive Co-Evolutionary Algorithm
Implementation for MicroRTS Grid Battles.

Matches the exact mathematical specification from the project presentation.
"""

import math
import random
import time
from ea_microrts import MicroRTSServer, BOT_TYPES, MAP_PATH, MAX_CYCLES, OPPONENT_BOT

# --- Hyperparameters from TSCEA Specification ---
N_AGENTS = 20
MAX_ARCHIVE = 10
MAX_GENERATIONS = 20
GAMES_PER_EVAL = 3
K_TRAJECTORY = 8   
DELTA_S = 0.95     # Distance threshold for redundancy 

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class TSCEAAgent:
    def __init__(self, weights=None, lambda_i=0.15):
        # The Strategy Vector (DNA)
        if weights is None:
            self.weights = [random.random() for _ in range(len(BOT_TYPES))]
        else:
            self.weights = list(weights)
            
        self.fitness = 0.0
        self.tau = []            # Trajectory: Last K fitness values
        self.lambda_i = lambda_i # Self-adaptive mutation rate
        
    def get_bot_type(self):
        best_idx = self.weights.index(max(self.weights))
        return BOT_TYPES[best_idx]
        
    def update_trajectory(self, cur_fitness):
        self.tau.append(cur_fitness)
        if len(self.tau) > K_TRAJECTORY:
            self.tau.pop(0)
            
    def compute_expressibility(self):
        """ e_i^t = sigmoid( sum_k (0.9^{k-1} * f_{t-k}) ) """
        if not self.tau: return 0.5
        
        weighted_sum = 0.0
        # Iterate backwards through trajectory
        for k, f in enumerate(reversed(self.tau)):
            alpha_k = math.pow(0.9, k)
            weighted_sum += alpha_k * f
            
        return sigmoid(weighted_sum - 2.0) # shifted sigmoid for normalized fitness 0-1
        
    def mutate(self):
        """Mutate using current self-adapted lambda_i"""
        new_weights = []
        for w in self.weights:
            # Add Gaussian noise scaled by lambda_i
            nw = w + random.gauss(0, self.lambda_i)
            new_weights.append(max(0.0, min(1.0, nw)))
        return TSCEAAgent(new_weights, self.lambda_i)

def euclidean_dist(w1, w2):
    return math.sqrt(sum((a-b)**2 for a, b in zip(w1, w2)))

def run_tscea():
    print("=========================================================")
    print("  TSCEA (Trajectory-guided Self-adaptive Co-Evolutionary) ")
    print("  MicroRTS Implementation")
    print("=========================================================")
    
    server = MicroRTSServer()
    server.start()
    
    population = [TSCEAAgent() for _ in range(N_AGENTS)]
    archive = []
    
    for gen in range(MAX_GENERATIONS):
        print(f"\n[Generation {gen+1}/{MAX_GENERATIONS}]")
        
        # 1. Evaluate Fitness & Update Trajectory
        pop_fitnesses = []
        for i, agent in enumerate(population):
            wins = 0
            cycles_taken = 0
            for _ in range(GAMES_PER_EVAL):
                winner, cycles = server.run_game(agent.get_bot_type(), OPPONENT_BOT)
                if winner == 0: wins += 1
                elif winner == -1: wins += 0.3
                cycles_taken += cycles
                
            # Normalize fitness 0.0 to 1.0
            agent.fitness = wins / GAMES_PER_EVAL
            agent.update_trajectory(agent.fitness)
            pop_fitnesses.append(agent.fitness)
            
            bar = "█" * int(agent.fitness * 20) + "░" * (20 - int(agent.fitness * 20))
            print(f"  Agent {i+1:2d} | Strat: {agent.get_bot_type():12s} | Fit: {agent.fitness:.2f} {bar}")
            
        # 2. Encode Environment (z_t)
        # 5-Dimensional Hand-crafted vector:
        avg_fitness = sum(pop_fitnesses) / len(pop_fitnesses)
        diversity = sum(euclidean_dist(a.weights, population[0].weights) for a in population) / len(population)
        turn_phase = gen / MAX_GENERATIONS
        recent_trend = avg_fitness - (population[0].tau[-2] if len(population[0].tau)>1 else 0)
        speed_ratio = 1.0 # placeholder for resource ratio
        
        z_t = [avg_fitness, recent_trend, diversity, turn_phase, speed_ratio]
        
        # 3. Update Hyperparameters (lambda_i)
        for agent in population:
            e_i = agent.compute_expressibility()
            mean_tau = sum(agent.tau) / len(agent.tau) if agent.tau else 0.0
            
            # g(z_t, \tau) = 0.3 * z_t[0] + 0.1 * mean(tau)
            g_val = 0.3 * z_t[0] + 0.1 * mean_tau
            epsilon = random.gauss(0, 0.03)
            
            agent.lambda_i = agent.lambda_i + (e_i * g_val) + epsilon
            agent.lambda_i = max(0.01, min(0.5, agent.lambda_i)) # bound mutation
            
        # 4. Remove weak agents (Bottom 50th percentile)
        pop_fitnesses.sort()
        delta_f = pop_fitnesses[int(len(pop_fitnesses) * 0.5)]
        survivors = [a for a in population if a.fitness >= delta_f]
        
        # 5. Remove redundant agents (Similarity > delta_s)
        unique_survivors = []
        for agent in sorted(survivors, key=lambda a: a.fitness, reverse=True):
            is_redundant = False
            for u in unique_survivors:
                # Our weights are 0-1, so max distance is ~1.73. We scale DELTA_S (0.85) appropriately.
                if euclidean_dist(agent.weights, u.weights) < (1.0 - DELTA_S):
                    is_redundant = True
                    break
            if not is_redundant:
                unique_survivors.append(agent)
                
        # 6. Insert survivors into Archive
        for agent in unique_survivors:
            archive.append(TSCEAAgent(agent.weights, agent.lambda_i))
        archive.sort(key=lambda a: a.fitness, reverse=True)
        archive = archive[:MAX_ARCHIVE]
        
        # 7. Generate offspring
        population = unique_survivors.copy()
        while len(population) < N_AGENTS:
            # Select parent from survivors or archive
            parent = random.choice(unique_survivors + archive)
            population.append(parent.mutate())
            
        print(f"  > Env [AvgFit:{z_t[0]:.2f}, Div:{z_t[2]:.2f}] | Archive Size: {len(archive)}")
        
    server.stop()
    
    print("\n=========================================================")
    print("  TSCEA OPTIMIZATION COMPLETE")
    print("=========================================================")
    best = archive[0]
    print(f"  Best Strategy Found: {best.get_bot_type()}")
    print(f"  Final Adapted Mutation Rate (\lambda_i): {best.lambda_i:.3f}")
    
if __name__ == "__main__":
    run_tscea()
