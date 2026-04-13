"""
baseline_ga.py
GENETIC ALGORITHM (Standard Baseline Algorithm)
"""
import random
from baseline_config import BaselineMicroRTSServer, BOT_TYPES, MAP_PATH_16X16, OPPONENT_BOT

POP_SIZE = 20
TOTAL_GENERATIONS = 100
GAMES_PER_EVAL = 2
MUTATION_RATE = 0.20
DIMENSIONS = len(BOT_TYPES)

# CHANGE THIS VARIABLE TO TEST DIFFERENT MAPS (MAP_PATH_8X8, MAP_PATH_16X16, MAP_PATH_32X32)
MAP = MAP_PATH_16X16 

class GAAgent:
    def __init__(self, params=None):
        if params is None:
            self.params = [random.random() for _ in range(DIMENSIONS)]
        else:
            self.params = [max(0.0, min(1.0, float(x))) for x in params]
        self.fitness = 0.0

    def get_bot_type(self):
        best_idx = self.params.index(max(self.params))
        return BOT_TYPES[best_idx]

def crossover(parent1, parent2):
    # Uniform crossover: randomly swap DNA blocks between Mother and Father
    child1_params = []
    child2_params = []
    for p1, p2 in zip(parent1.params, parent2.params):
        if random.random() < 0.5:
            child1_params.append(p1)
            child2_params.append(p2)
        else:
            child1_params.append(p2)
            child2_params.append(p1)
    return GAAgent(child1_params), GAAgent(child2_params)

def mutate(agent):
    new_params = []
    for p in agent.params:
        # Standard rigid mutation
        if random.random() < 0.3:
            new_params.append(p + random.gauss(0, MUTATION_RATE))
        else:
            new_params.append(p)
    return GAAgent(new_params)

def evaluate_agent(server, agent):
    wins = 0
    for _ in range(GAMES_PER_EVAL):
        winner, _ = server.run_game(agent.get_bot_type(), OPPONENT_BOT, map_path=MAP, max_cycles=6000)
        if winner == 0: wins += 1
        elif winner == -1: wins += 0.3
    agent.fitness = wins / GAMES_PER_EVAL

def run_genetic_algorithm():
    print(f"\n[Genetic Algorithm Baseline] Testing on Map: {MAP}")
    server = BaselineMicroRTSServer()
    server.start()
    
    population = [GAAgent() for _ in range(POP_SIZE)]
    
    for gen in range(1, TOTAL_GENERATIONS + 1):
        for agent in population:
            evaluate_agent(server, agent)
            
        population.sort(key=lambda a: a.fitness, reverse=True)
        best_agent = population[0]
        
        dna_str = "[" + ", ".join([f"{p:.2f}" for p in best_agent.params]) + "]"
        print(f"[GA] Generation {gen:03d}/{TOTAL_GENERATIONS:03d} | Max Fit: {best_agent.fitness:.2f} | Strategy: {best_agent.get_bot_type():12} | DNA: {dna_str}")
        
        # Elitism - Securely keep top 2 without mutation
        next_gen = population[:2]
        
        # Tournament Selection & Crossover
        while len(next_gen) < POP_SIZE:
            # Tournament of size 3 (Grab 3 random agents, best one wins the right to breed)
            p1 = max(random.sample(population, 3), key=lambda a: a.fitness)
            p2 = max(random.sample(population, 3), key=lambda a: a.fitness)
            
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            
            next_gen.append(c1)
            if len(next_gen) < POP_SIZE:
                next_gen.append(c2)
                
        population = next_gen
        
    server.stop()
    best = max(population, key=lambda a: a.fitness)
    best_strat = best.get_bot_type()
    print("\n[Genetic Algorithm] Finished. Final Elite Strategy:", best_strat)

    print("\n[Visualizer] Launching Post-Training Replay...")
    from tdccea_demo_visual import run_visual_game
    import time
    time.sleep(1)
    run_visual_game(best_strat, OPPONENT_BOT, MAP)

if __name__ == "__main__":
    run_genetic_algorithm()
