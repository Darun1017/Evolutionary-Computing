"""
=============================================================
EA + microRTS Integration (Batch Mode)
=============================================================

How it works:
  1. Start ONE Java process (BatchGameRunner) — keeps JVM alive
  2. Send game requests via stdin: "BotType1,BotType2,map,maxCycles"
  3. Receive results via stdout: "RESULT:winner,cycles"
  4. EA evolves agent strategies based on win/loss results

Each EA agent = a weight vector selecting which bot to use:
  WorkerRush  : Rush with workers (aggressive, fast)
  LightRush   : Build light units and attack (balanced)
  RandomBiased: Semi-random play (opponent baseline)
=============================================================
"""

import subprocess
import random
import os
import sys
import time

# ─── Configuration ───
MICRORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "Simulation", "microrts")
MAP_PATH_8X8 = "maps/8x8/basesWorkers8x8Obstacle.xml"
MAP_PATH_16X16 = "maps/16x16/basesWorkers16x16A.xml"
MAP_PATH_32X32 = "maps/basesWorkers32x32A.xml"
OPPONENT_BOT_8X8 = "LightRush"
OPPONENT_BOT_16X16 = "WorkerRush"
OPPONENT_BOT_32X32 = "LightRush"

MAX_CYCLES = 4000
#maps/16x16/custom_16x16.xml
#maps/8x8/custom_unfair_2bases.xml


BOT_TYPES = ["WorkerRush", "LightRush", "RandomBiased", "HeavyRush", "Random", "Passive"]

POP_SIZE = 8
GENERATIONS = 20
GAMES_PER_EVAL = 3
MUTATION_RATE = 0.2
OPPONENT_BOT = "WorkerRush"


# ─── Java Game Server ───
class MicroRTSServer:
    """Manages a persistent Java process for running games."""

    def __init__(self):
        self.process = None

    def start(self):
        """Start the Java BatchGameRunner process."""
        cmd = ["java", "-cp", "lib/*;bin", "tests.BatchGameRunner"]
        self.process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=MICRORTS_DIR
        )
        # Wait for READY signal
        line = self.process.stdout.readline().strip()
        if line != "READY":
            raise RuntimeError(f"BatchGameRunner failed to start: {line}")
        print("  [✓] microRTS game server started")

    def run_game(self, bot1, bot2, map_path=MAP_PATH_8X8, max_cycles=MAX_CYCLES):
        """Run a single game and return (winner, cycles)."""
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
        """Shutdown the Java process."""
        if self.process:
            try:
                self.process.stdin.write("EXIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            print("  [✓] microRTS game server stopped")


# ─── EA Agent ───
class Agent:
    def __init__(self, weights=None):
        if weights is None:
            self.weights = [random.random() for _ in range(len(BOT_TYPES))]
        else:
            self.weights = list(weights)
        self.fitness = 0.0

    def get_bot_type(self):
        best_idx = self.weights.index(max(self.weights))
        return BOT_TYPES[best_idx]

    def mutate(self):
        new_weights = []
        for w in self.weights:
            if random.random() < 0.5:
                new_w = w + random.gauss(0, MUTATION_RATE)
                new_w = max(0.0, min(1.0, new_w))
            else:
                new_w = w
            new_weights.append(new_w)
        return Agent(new_weights)

    def __repr__(self):
        return f"Agent(bot={self.get_bot_type()}, fit={self.fitness:.3f})"


# ─── Main EA Loop ───
def run_ea():
    print("=" * 60)
    print("  EA + microRTS — Evolving Game Strategies")
    print("=" * 60)
    print(f"  Population: {POP_SIZE} agents")
    print(f"  Generations: {GENERATIONS}")
    print(f"  Opponent: {OPPONENT_BOT_8X8}")
    print(f"  Bots: {', '.join(BOT_TYPES)}")
    print(f"  Map: {MAP_PATH_8X8}")
    print("=" * 60)
    print()

    # Start game server
    server = MicroRTSServer()
    server.start()
    print()

    # Create population
    population = [Agent() for _ in range(POP_SIZE)]
    best_ever = None
    best_ever_fitness = 0
    total_games = 0

    t_start = time.time()

    for gen in range(GENERATIONS):
        print(f"  ┌─ Generation {gen + 1}/{GENERATIONS} ─────────────────────")

        # Evaluate each agent
        for i, agent in enumerate(population):
            wins = 0
            bot = agent.get_bot_type()

            for g in range(GAMES_PER_EVAL):
                winner, cycles = server.run_game(bot, OPPONENT_BOT)
                if winner == 0:
                    wins += 1
                elif winner == -1:
                    wins += 0.3
                total_games += 1

            agent.fitness = wins / GAMES_PER_EVAL
            bar = "█" * int(agent.fitness * 20) + "░" * (20 - int(agent.fitness * 20))
            print(f"  │ Agent {i+1:2d} [{bot:14s}] "
                  f"fit={agent.fitness:.2f} {bar}")

        # Sort + stats
        population.sort(key=lambda a: a.fitness, reverse=True)
        best = population[0]
        avg_fit = sum(a.fitness for a in population) / len(population)

        if best.fitness > best_ever_fitness:
            best_ever = Agent(best.weights)
            best_ever.fitness = best.fitness
            best_ever_fitness = best.fitness

        print(f"  │")
        print(f"  │ Best: {best.get_bot_type()} ({best.fitness:.2f}) "
              f"│ Avg: {avg_fit:.2f} │ Games: {total_games}")
        print(f"  └──────────────────────────────────────────")
        print()

        # Selection + Mutation
        survivors = population[:POP_SIZE // 2]
        children = []
        while len(children) < POP_SIZE // 2:
            parent = random.choice(survivors)
            children.append(parent.mutate())
        population = survivors + children

    elapsed = time.time() - t_start
    server.stop()

    # Results
    print()
    print("=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"  Best strategy: {best_ever.get_bot_type()}")
    print(f"  Best fitness:  {best_ever_fitness:.3f}")
    print(f"  Total games:   {total_games}")
    print(f"  Total time:    {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print()
    print("  Bot weights evolved to:")
    for bot, w in zip(BOT_TYPES, best_ever.weights):
        bar = "█" * int(w * 30)
        print(f"    {bot:14s}: {w:.3f} {bar}")
    print()
    print("  EXPLANATION:")
    print("  The EA tested different bot strategies against RandomBiased.")
    print(f"  Over {GENERATIONS} generations, it discovered that")
    print(f"  '{best_ever.get_bot_type()}' is the most effective strategy.")
    print()


if __name__ == "__main__":
    run_ea()
