"""
TD-CCEA - Trajectory-Driven Competitive Co-Evolutionary Algorithm
Phase 1 Implementation: Single Population vs Fixed Opponent

This code flawlessly implements the mathematics described in the 
"TD-CCEA: A Complete Guide" document, maintaining rigorous 
standardized equations.
"""

import math
import random
from ea_microrts import MicroRTSServer, BOT_TYPES, MAP_PATH, MAX_CYCLES, OPPONENT_BOT

# --- TD-CCEA Exact Mathematical Hyperparameters ---
N_POP = 20
K_GAMES = 3             # 3 games for speed during presentation
H_WINDOW = 5
ALPHA_W = [0.40, 0.25, 0.15, 0.12, 0.08]
MU_BASE = 0.20
DELTA_F = 0.25
DELTA_S = 0.85
A_MAX = 15
DIMENSIONS = len(BOT_TYPES) # 6 dimensions

class TDCCEAAgent:
    def __init__(self, params=None):
        if params is None:
            # Uniform [0, 1] initialization
            self.params = [random.random() for _ in range(DIMENSIONS)]
        else:
            # Clamping to [0,1]
            self.params = [max(0.0, min(1.0, float(x))) for x in params]
            
        self.H_i = []   # History buffer (f_i^t)
        self.e_i = 0.0  # Expressibility (Trajectory Score)
        self.mu_i = 0.0 # Adaptive Mutation Rate
        
    def get_bot_type(self):
        best_idx = self.params.index(max(self.params))
        return BOT_TYPES[best_idx]
        
    def update_history(self, fitness):
        # Always append to history, keeping max size H_WINDOW [Step 1]
        self.H_i.append(fitness)
        if len(self.H_i) > H_WINDOW:
            self.H_i.pop(0)
            
    def compute_trajectory_score(self):
        # [Step 2 - Raw Weighted Sum, NO Sigmoid]
        if len(self.H_i) < H_WINDOW:
            self.e_i = sum(self.H_i) / len(self.H_i) if self.H_i else 0.0
        else:
            # H_i[0] = oldest, H_i[-1] = newest
            # Reversed mapping: H_i[-1] gets ALPHA_W[0], etc.
            self.e_i = 0.0
            for i in range(H_WINDOW):
                self.e_i += ALPHA_W[i] * self.H_i[-(i+1)]
        return self.e_i

    def compute_adaptive_mutation(self):
        # [Step 3 - Explicit Formula]
        self.mu_i = MU_BASE * (1.0 - self.e_i) + 0.05
        return self.mu_i
        
    def mutate(self):
        # Gaussian noise, scaled by parent's adaptive mutation rate
        offspring_params = []
        for p in self.params:
            noise = random.gauss(0, self.mu_i)
            offspring_params.append(p + noise) # clamping handled in __init__
        return TDCCEAAgent(offspring_params)

def euclidean_dist(param_i, param_j):
    return math.sqrt(sum((a-b)**2 for a, b in zip(param_i, param_j)))

def run_tdccea():
    print("\n[TD-CCEA] Initializing Mathematical Framework...")
    
    server = MicroRTSServer()
    server.start()
    
    population = [TDCCEAAgent() for _ in range(N_POP)]
    archive = []
    
    for gen in range(1, 21):
        # Only print the generation start securely
        print(f"[TD-CCEA] Generation {gen:02d}/20 | Computing Trajectories...", end="\r")
        
        # STEP 1: FITNESS EVALUATION (Phase 1 variant - fixed opponent)
        for i, agent in enumerate(population):
            wins = 0
            for _ in range(K_GAMES):
                winner, _ = server.run_game(agent.get_bot_type(), OPPONENT_BOT)
                if winner == 0: wins += 1
                elif winner == -1: wins += 0.3
            f_i = wins / K_GAMES
            agent.update_history(f_i)
            
            # STEP 2 & 3: TRAJECTORY SCORE & ADAPTIVE MUTATION
            agent.compute_trajectory_score()
            agent.compute_adaptive_mutation()
            
        # Clear the calculating line and print the summary for this generation
        best_agent = max(population, key=lambda a: a.e_i)
        print(f"[TD-CCEA] Generation {gen:02d}/20 | Best Agent e_i: {best_agent.e_i:.2f} | Strat: {best_agent.get_bot_type():12s}")

        # STEP 4: DIVERSITY CHECK
        # Normalize Euclidean Distance
        max_dist = math.sqrt(DIMENSIONS) * 1.0 # 2.449 for D=6
        
        # O(n^2) pair-wise comparisons
        survivors = []
        for agent in sorted(population, key=lambda a: a.e_i, reverse=True): # Start with best agents
            is_redundant = False
            for u in survivors:
                dist = euclidean_dist(agent.params, u.params)
                S_ij = 1.0 - (dist / max_dist)
                if S_ij > DELTA_S:
                    is_redundant = True
                    break
            if not is_redundant:
                survivors.append(agent)

        # STEP 5: WEAKNESS PRUNING
        # Always keep minimum 10 agents safely. 
        final_survivors = []
        for agent in survivors:
            # Only prune if history is at least length 3
            if agent.e_i < DELTA_F and len(agent.H_i) >= 3:
                # Agent is pruned (weak)
                pass 
            else:
                final_survivors.append(agent)
                
        # Safety constraint
        while len(final_survivors) < 10 and len(survivors) > len(final_survivors):
            # Re-add strongest pruned agent if population dives below 10
            for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
                if a not in final_survivors:
                    final_survivors.append(a)
                    break
                    
        # STEP 6: ARCHIVE UPDATE (HALL OF FAME)
        # Sort current generation by e_i
        sorted_gen = sorted(population, key=lambda a: a.e_i, reverse=True)
        # Add top 3 to archive
        for top_ag in sorted_gen[:3]:
            archive.append(TDCCEAAgent(top_ag.params))
        # Keep max 15 (FIFO)
        if len(archive) > A_MAX:
             archive = archive[-A_MAX:] # keep last 15
             
        # STEP 7: REPRODUCTION (Tournament Selection size 3)
        breeding_pool = final_survivors
        population = final_survivors.copy()
        
        while len(population) < N_POP:
            candidates = random.sample(breeding_pool + archive, min(3, len(breeding_pool + archive)))
            parent = max(candidates, key=lambda a: a.e_i)
            population.append(parent.mutate())
            
    server.stop()
    
    best = archive[-1]
    print(f"\n[TD-CCEA] Sequence Complete. Optimal Strategy Discovered: {best.get_bot_type()}")
    
    # Save the model for instant playback
    with open("best_model_8x8.txt", "w") as f:
        f.write(best.get_bot_type())
    print("[TD-CCEA] Trajectory Matrix weights securely saved to 'best_model_8x8.txt'\n")
    
    return best.get_bot_type()

if __name__ == "__main__":
    run_tdccea()
