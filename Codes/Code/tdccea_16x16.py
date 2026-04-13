

import math
import random
from ea_microrts import MicroRTSServer, BOT_TYPES, MAP_PATH_16X16, OPPONENT_BOT_16X16
from tdccea_demo_visual import run_visual_game

N_POP = 20
K_GAMES = 2
H_WINDOW = 5
MU_BASE = 0.20
DELTA_F = 0.25
DELTA_S = 0.85
A_MAX = 15
TOTAL_GENERATIONS = 100
DIMENSIONS = len(BOT_TYPES)

def euclidean_dist(param_i, param_j):
    return math.sqrt(sum((a-b)**2 for a, b in zip(param_i, param_j)))

def get_population_centroid(population):
    """Calculates the exact average DNA of the current environment/population."""
    centroid = [0.0] * DIMENSIONS
    for agent in population:
        for j in range(DIMENSIONS):
            centroid[j] += agent.params[j]
    return [x / len(population) for x in centroid]

class TDCCEAAgent:
    def __init__(self, params=None):
        if params is None:
            self.params = [random.random() for _ in range(DIMENSIONS)]
        else:
            self.params = [max(0.0, min(1.0, float(x))) for x in params]
        self.H_i = [] # Stores Tuples: (Fitness, Environment_Centroid)
        self.e_i = 0.0
        self.mu_i = 0.0
        
    def get_bot_type(self):
        best_idx = self.params.index(max(self.params))
        return BOT_TYPES[best_idx]
        
    def update_history(self, fitness, current_environment):
        # [CRITICAL UPDATE] Store both performance AND the environment state at t*
        self.H_i.append( (fitness, current_environment) )
        if len(self.H_i) > H_WINDOW:
            self.H_i.pop(0)
            
    def compute_dynamic_trajectory(self, current_environment):
        # [CRITICAL UPDATE] Instead of hardcoded ALPHA_W, we dynamically calculate similarity weights
        if len(self.H_i) == 0:
            self.e_i = 0.0
            return self.e_i
            
        dynamic_weights = []
        max_dist = math.sqrt(DIMENSIONS) * 1.0 # Maximum mathematical deviation possible
        
        # Calculate exactly how similar the past environments were to the current one
        for past_fitness, past_env in self.H_i:
            dist = euclidean_dist(current_environment, past_env)
            # 1.0 means environments are perfectly identical. 0.0 means completely opposite.
            similarity = 1.0 - (dist / max_dist) 
            dynamic_weights.append(similarity)
            
        # Normalize the dynamic weights so they add up to 100% (1.0)
        total_w = sum(dynamic_weights)
        if total_w == 0:
            dynamic_weights = [1.0 / len(dynamic_weights)] * len(dynamic_weights)
        else:
            dynamic_weights = [w / total_w for w in dynamic_weights]
            
        # Apply the dynamically generated similarity weights to calculate final e_i
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

def run_dynamic_similarity_tdccea():
    print("\n[TD-CCEA Dynamic] Booting Dynamic Environment-Similarity Matrix...")
    server = MicroRTSServer()
    server.start()
    
    population = [TDCCEAAgent() for _ in range(N_POP)]
    archive = []
    
    for gen in range(1, TOTAL_GENERATIONS + 1):
        # Calculate the mathematical meta-state of the environment for this generation
        current_env = get_population_centroid(population)
        print(f"[TD-CCEA] Generation {gen:02d}/{TOTAL_GENERATIONS:02d} | Computing Dynamic Environment Similarities...".ljust(100), end="\r")
        
        for i, agent in enumerate(population):
            wins = 0
            for _ in range(K_GAMES):
                winner, _ = server.run_game(agent.get_bot_type(), OPPONENT_BOT_16X16, map_path=MAP_PATH_16X16, max_cycles=6000)
                if winner == 0: wins += 1
                elif winner == -1: wins += 0.3
            f_i = wins / K_GAMES
            
            # Agent securely logs its fitness alongside the exact environment state at this moment
            agent.update_history(f_i, current_env)
            # Agent evaluates its trajectory by comparing past environments to current_env
            agent.compute_dynamic_trajectory(current_env)
            agent.compute_adaptive_mutation()
            
        best_agent = max(population, key=lambda a: a.e_i)
        dna_str = "[" + ", ".join([f"{p:.2f}" for p in best_agent.params]) + "]"
        print(f"[TD-CCEA] Gen {gen:02d}/{TOTAL_GENERATIONS:02d} | e_i: {best_agent.e_i:.2f} | DNA: {dna_str} | Strat: {best_agent.get_bot_type()}          ")

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
            archive.append(TDCCEAAgent(top_ag.params))
        if len(archive) > A_MAX:
             archive = archive[-A_MAX:]
             
        breeding_pool = final_survivors
        population = final_survivors.copy()
        while len(population) < N_POP:
            candidates = random.sample(breeding_pool + archive, min(3, len(breeding_pool + archive)))
            parent = max(candidates, key=lambda a: a.e_i)
            population.append(parent.mutate())
            
    server.stop()
    best = archive[-1]
    print(f"\n[TD-CCEA 16x16] Full Sequence Complete. Dominant Strategy: {best.get_bot_type()}")
    
    with open("best_model_16x16.txt", "w") as f:
        f.write(best.get_bot_type())
        
    print("\n[Visualizer] Launching Post-Training Replay...")
    import time
    time.sleep(1)
    run_visual_game(best.get_bot_type(), OPPONENT_BOT_16X16, MAP_PATH_16X16)
    
if __name__ == "__main__":
    run_dynamic_similarity_tdccea()
