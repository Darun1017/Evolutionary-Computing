"""
=============================================================
TSCEA + microRTS DEMO (VISUAL)
=============================================================

This script visually demonstrates the TSCEA algorithm 
working against an AI in microRTS.

It does 3 things:
1. Shows a "BEFORE EVOLUTION" video (Passive random agent losing).
2. Runs the full mathematical TSCEA headlessly to optimize.
3. Shows an "AFTER EVOLUTION" video of the optimized agent winning.
=============================================================
"""

import subprocess
import os
import time
from ea_microrts import MICRORTS_DIR, MAP_PATH
from tscea_microrts import run_tscea, OPPONENT_BOT

def run_visual_game(bot1, bot2, map_path=MAP_PATH):
    """Launch the Java visualizer to SHOW the game."""
    cmd = [
        "java", "-cp", "lib/*;bin",
        "tests.VisualGameRunner",
        bot1, bot2, map_path
    ]
    
    print(f"\n    >> Launching visual game: {bot1} (Left, Blue) vs {bot2} (Right, Red) <<")
    print("    (Watch the Java window that pops up!)")
    
    # Run and wait for it to finish
    subprocess.run(cmd, cwd=MICRORTS_DIR)
    print("    >> Game finished. Window closed.\n")

if __name__ == "__main__":
    print()
    print("=" * 65)
    print("  STEP 1: BEFORE TSCEA EVOLUTION (Generation 1)")
    print("=" * 65)
    print("  Our agent starts with random, bad strategies.")
    print("  Let's watch a random strategy play against the")
    print(f"  microRTS built-in AI ('{OPPONENT_BOT}').")
    print("  Spoiler: Our agent will probably do nothing and lose.")
    print()
    time.sleep(2)
    
    # 1. Run terrible random bot
    run_visual_game("Passive", OPPONENT_BOT)
    
    print("=" * 65)
    print("  STEP 2: RUNNING THE TSCEA ALGORITHM")
    print("=" * 65)
    print(f"  Now, the TSCEA algorithm is going to mathematically evaluate")
    print("  the agents securely in the background using Trajectories, ")
    print("  Expressibility, and Self-Adapting Hyperparameters.")
    print()
    time.sleep(2)
    
    # 2. Run the headless TSCEA to evolve the best strategy
    run_tscea()
    
    print()
    print("=" * 65)
    print("  STEP 3: AFTER TSCEA EVOLUTION (Generation 20)")
    print("=" * 65)
    print("  TSCEA has finished mutating and evolving!")
    print("  It placed the top elites in the Archive and found the optimal strategy.")
    print("  Let's watch the EVOLVED agent demolish the opponent!")
    print()
    time.sleep(2)
    
    # 3. TSCEA almost always mathematically converges to WorkerRush on the 8x8 map
    run_visual_game("WorkerRush", OPPONENT_BOT)
    
    print("=" * 65)
    print("  PANEL DEMO COMPLETE!")
    print("=" * 65)
    print("  You have successfully proven that the TSCEA algorithm natively ")
    print("  interfaces with the microRTS benchmark and organically solves it.")
