"""
=============================================================
SIMPLE EA DEMO — Agents Evolve to Play a Mini Game
=============================================================

This demo shows how Evolutionary Algorithms work step by step.
No external libraries needed — just Python.

THE GAME: "Number Guessing Battle"
- There's a SECRET target strategy [t1, t2, t3, t4, t5]
- Each agent has a strategy [s1, s2, s3, s4, s5]
- Fitness = how CLOSE the agent's strategy is to the target
- Agents that are closer SURVIVE, others DIE
- Survivors create CHILDREN with small random changes
- Over generations, agents get CLOSER to the target

This is the SIMPLEST possible demonstration of:
  1. Population of agents
  2. Fitness evaluation
  3. Selection (survival of fittest)
  4. Mutation (random changes)
  5. Evolution over generations
=============================================================
"""

import random
import math


# ─────────────────────────────────────────────
# THE "GAME" — A simple optimization problem
# ─────────────────────────────────────────────

# The SECRET optimal strategy (agents don't know this)
TARGET = [0.8, 0.2, 0.6, 0.9, 0.4]

def evaluate_fitness(agent_strategy):
    """
    Fitness = how close agent is to the TARGET.
    Higher = better. Max possible = 1.0 (perfect match).
    
    This simulates: "How well does this agent play the game?"
    In a real game, this would be win_rate or score.
    """
    distance = sum((a - t) ** 2 for a, t in zip(agent_strategy, TARGET))
    max_distance = len(TARGET)  # max possible distance
    fitness = 1.0 - (distance / max_distance)
    return max(0.0, fitness)


# ─────────────────────────────────────────────
# AGENT — A simple strategy vector
# ─────────────────────────────────────────────

def create_random_agent():
    """Create an agent with random strategy values [0.0 to 1.0]."""
    return [random.random() for _ in range(5)]

def mutate(agent, mutation_rate=0.1):
    """Create a child with small random changes."""
    child = []
    for val in agent:
        # Add small Gaussian noise
        new_val = val + random.gauss(0, mutation_rate)
        new_val = max(0.0, min(1.0, new_val))  # keep in [0, 1]
        child.append(new_val)
    return child

def crossover(parent1, parent2):
    """Mix two parents to create a child."""
    child = []
    for v1, v2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child.append(v1)
        else:
            child.append(v2)
    return child


# ─────────────────────────────────────────────
# EVOLUTIONARY ALGORITHM — The main loop
# ─────────────────────────────────────────────

def run_ea(pop_size=20, generations=30, mutation_rate=0.1):
    """
    Run a basic Evolutionary Algorithm.
    
    Step by step:
    1. Create random population
    2. Evaluate fitness of each agent
    3. Select best agents (top 50%)
    4. Create children through crossover + mutation
    5. Repeat
    """
    
    print("=" * 65)
    print("  EVOLUTIONARY ALGORITHM DEMO")
    print("  Target strategy:", [f"{t:.1f}" for t in TARGET])
    print("=" * 65)
    print()
    
    # STEP 1: Initialize random population
    population = [create_random_agent() for _ in range(pop_size)]
    
    print("Generation 0 (Random Start):")
    print(f"  Agent 1: {format_strategy(population[0])} → fitness = {evaluate_fitness(population[0]):.4f}")
    print(f"  Agent 2: {format_strategy(population[1])} → fitness = {evaluate_fitness(population[1]):.4f}")
    print(f"  Agent 3: {format_strategy(population[2])} → fitness = {evaluate_fitness(population[2]):.4f}")
    print()
    
    for gen in range(generations):
        # STEP 2: Evaluate fitness
        fitness_scores = [(agent, evaluate_fitness(agent)) for agent in population]
        
        # Sort by fitness (best first)
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Stats
        all_fits = [f for _, f in fitness_scores]
        avg_fit = sum(all_fits) / len(all_fits)
        best_fit = all_fits[0]
        best_agent = fitness_scores[0][0]
        
        # Print progress
        bar = "█" * int(best_fit * 40) + "░" * (40 - int(best_fit * 40))
        print(f"  Gen {gen+1:3d} │ Best: {best_fit:.4f} │ Avg: {avg_fit:.4f} │ {bar} │", end="")
        
        if best_fit > 0.99:
            print(" ★ NEAR PERFECT!")
        elif best_fit > 0.95:
            print(" ✓ Very good!")
        elif best_fit > 0.80:
            print(" → Improving...")
        else:
            print(" · Learning...")
        
        # STEP 3: Selection — keep top 50%
        survivors = [agent for agent, fit in fitness_scores[:pop_size // 2]]
        
        # STEP 4: Create children
        children = []
        while len(children) < pop_size // 2:
            # Pick two random parents from survivors
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)
            
            # Crossover (mix parents)
            child = crossover(p1, p2)
            
            # Mutation (random changes)
            child = mutate(child, mutation_rate)
            
            children.append(child)
        
        # STEP 5: New population = survivors + children
        population = survivors + children
    
    # Final results
    final_best = max(population, key=evaluate_fitness)
    final_fitness = evaluate_fitness(final_best)
    
    print()
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  Target:        {format_strategy(TARGET)}")
    print(f"  Best found:    {format_strategy(final_best)}")
    print(f"  Final fitness: {final_fitness:.4f}")
    print(f"  Match: {final_fitness * 100:.1f}%")
    print()
    
    # Show how close each value is
    print("  Value-by-value comparison:")
    for i, (found, target) in enumerate(zip(final_best, TARGET)):
        diff = abs(found - target)
        match = "✓" if diff < 0.05 else "~" if diff < 0.15 else "✗"
        print(f"    Strategy[{i}]: found={found:.3f}, target={target:.3f}, diff={diff:.3f} {match}")
    
    print()
    return final_best


def format_strategy(s):
    return "[" + ", ".join(f"{v:.2f}" for v in s) + "]"


# ─────────────────────────────────────────────
# RUN IT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("  This demo shows how an Evolutionary Algorithm works.")
    print("  20 agents start with RANDOM strategies.")
    print("  Over 30 generations, they evolve to find the TARGET.")
    print()
    
    best = run_ea(pop_size=20, generations=30, mutation_rate=0.1)
    
    print("  HOW IT WORKED:")
    print("  1. 20 random agents were created")
    print("  2. Each was tested (fitness measured)")
    print("  3. Worst 50% were removed")
    print("  4. Best 50% created children (crossover + mutation)")
    print("  5. Repeated 30 times")
    print("  6. Result: agents evolved to match the target!")
    print()
    print("  This SAME logic applies to your microRTS project:")
    print("  - Instead of matching numbers → agents play games")
    print("  - Fitness = win rate instead of distance to target")
    print("  - Same selection, mutation, crossover process")
    print()
