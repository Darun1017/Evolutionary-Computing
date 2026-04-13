"""
baseline_hc.py
STOCHASTIC HILL CLIMBING (Baseline Algorithm)
"""
import random
from baseline_config import BaselineMicroRTSServer, BOT_TYPES, MAP_PATH_16X16, OPPONENT_BOT

TOTAL_GENERATIONS = 100
GAMES_PER_EVAL = 2
MUTATION_NOISE = 0.20
DIMENSIONS = len(BOT_TYPES)

# CHANGE THIS VARIABLE TO TEST DIFFERENT MAPS (MAP_PATH_8X8, MAP_PATH_16X16, MAP_PATH_32X32)
MAP = MAP_PATH_16X16 

class HCAgent:
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
            if random.random() < 0.5:
                # Stochastic local exploration
                new_params.append(p + random.gauss(0, MUTATION_NOISE))
            else:
                new_params.append(p)
        return HCAgent(new_params)

def evaluate_agent(server, agent):
    wins = 0
    for _ in range(GAMES_PER_EVAL):
        winner, _ = server.run_game(agent.get_bot_type(), OPPONENT_BOT, map_path=MAP, max_cycles=6000)
        if winner == 0: wins += 1
        elif winner == -1: wins += 0.3
    agent.fitness = wins / GAMES_PER_EVAL
    return agent.fitness

def run_hill_climbing():
    print(f"\n[Hill Climbing Baseline] Testing on Map: {MAP}")
    server = BaselineMicroRTSServer()
    server.start()
    
    current_agent = HCAgent()
    evaluate_agent(server, current_agent)
    
    for gen in range(1, TOTAL_GENERATIONS + 1):
        # Generate exactly 1 neighbor mathematically
        candidate_agent = current_agent.mutate()
        evaluate_agent(server, candidate_agent)
        
        # If the candidate strictly improves upon or equals the current alpha, it overtakes
        if candidate_agent.fitness >= current_agent.fitness:
            current_agent = candidate_agent
            
        dna_str = "[" + ", ".join([f"{p:.2f}" for p in current_agent.params]) + "]"
        print(f"[HC] Generation {gen:03d}/{TOTAL_GENERATIONS:03d} | Max Fit: {current_agent.fitness:.2f} | Strategy: {current_agent.get_bot_type():12} | DNA: {dna_str}")
        
    server.stop()
    best_strat = current_agent.get_bot_type()
    print("\n[Hill Climbing] Finished. Final Peak Strategy:", best_strat)
    
    print("\n[Visualizer] Launching Post-Training Replay...")
    from tdccea_demo_visual import run_visual_game
    import time
    time.sleep(1)
    run_visual_game(best_strat, OPPONENT_BOT, MAP)

if __name__ == "__main__":
    run_hill_climbing()
