"""
baseline_es.py
(μ, λ) EVOLUTION STRATEGY (Baseline Algorithm)
"""
import random
from baseline_config import BaselineMicroRTSServer, BOT_TYPES, MAP_PATH_16X16, OPPONENT_BOT

MU = 5        # Number of parents (The Elite)
LAMBDA = 20   # Number of offspring generated
TOTAL_GENERATIONS = 100
GAMES_PER_EVAL = 2
MUTATION_RATE = 0.20
DIMENSIONS = len(BOT_TYPES)

# CHANGE THIS VARIABLE TO TEST DIFFERENT MAPS (MAP_PATH_8X8, MAP_PATH_16X16, MAP_PATH_32X32)
MAP = MAP_PATH_16X16 

class ESAgent:
    def __init__(self, params=None):
        if params is None:
            self.params = [random.random() for _ in range(DIMENSIONS)]
        else:
            self.params = [max(0.0, min(1.0, float(x))) for x in params]
        self.fitness = 0.0

    def get_bot_type(self):
        best_idx = self.params.index(max(self.params))
        return BOT_TYPES[best_idx]

    def mutate(self):
        new_params = []
        for p in self.params:
            # ES is highly mutation-focused (no crossover needed)
            new_params.append(p + random.gauss(0, MUTATION_RATE))
        return ESAgent(new_params)

def evaluate_agent(server, agent):
    wins = 0
    for _ in range(GAMES_PER_EVAL):
        winner, _ = server.run_game(agent.get_bot_type(), OPPONENT_BOT, map_path=MAP, max_cycles=6000)
        if winner == 0: wins += 1
        elif winner == -1: wins += 0.3
    agent.fitness = wins / GAMES_PER_EVAL

def run_evolution_strategies():
    print(f"\n[(μ, λ)-ES Baseline] Testing on Map: {MAP}")
    server = BaselineMicroRTSServer()
    server.start()
    
    # Initialize basic parents
    parents = [ESAgent() for _ in range(MU)]
    for p in parents:
        evaluate_agent(server, p)
    parents.sort(key=lambda x: x.fitness, reverse=True)
    
    for gen in range(1, TOTAL_GENERATIONS + 1):
        # 1. Produce Lambda Offspring
        offspring = []
        for _ in range(LAMBDA):
            # Select a random parent from the Mu pool
            parent = random.choice(parents)
            child = parent.mutate()
            offspring.append(child)
            
        # 2. Evaluate all Offspring
        for child in offspring:
            evaluate_agent(server, child)
            
        # 3. Sort Offspring and select top Mu to become the completely new Parents
        offspring.sort(key=lambda x: x.fitness, reverse=True)
        parents = offspring[:MU]
        
        # Output Best
        best_agent = parents[0]
        dna_str = "[" + ", ".join([f"{p:.2f}" for p in best_agent.params]) + "]"
        print(f"[ES] Generation {gen:03d}/{TOTAL_GENERATIONS:03d} | Max Fit: {best_agent.fitness:.2f} | Strategy: {best_agent.get_bot_type():12} | DNA: {dna_str}")
        
    server.stop()
    best_strat = parents[0].get_bot_type()
    print("\n[(μ, λ)-ES] Finished. Final Elite Strategy:", best_strat)

    print("\n[Visualizer] Launching Post-Training Replay...")
    from tdccea_demo_visual import run_visual_game
    import time
    time.sleep(1)
    run_visual_game(best_strat, OPPONENT_BOT, MAP)

if __name__ == "__main__":
    run_evolution_strategies()
