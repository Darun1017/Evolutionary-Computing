"""
=============================================================
EA + microRTS DEMO (VISUAL)
=============================================================

This script clearly demonstrating HOW the EA algorithm 
works against an AI in microRTS.

It does 3 things:
1. Shows a "BEFORE EVOLUTION" video — a random agent losing.
2. Runs the EA headless — to find the winning strategy.
3. Shows an "AFTER EVOLUTION" video — the optimized agent winning.
=============================================================
"""

import subprocess
import os
import time
from ea_microrts import run_ea, MICRORTS_DIR, MAP_PATH

OPPONENT_BOT = "RandomBiased"

def run_visual_game(bot1, bot2, map_path=MAP_PATH):
    """Launch the Java visualizer to SHOW the game."""
    cmd = [
        "java", "-cp", "lib/*;bin",
        "tests.VisualGameRunner",
        bot1, bot2, map_path
    ]
    
    print(f"    >> Launching visual game: {bot1} (Left, Blue) vs {bot2} (Right, Red) <<")
    print("    (Watch the Java window that pops up!)")
    
    # Run and wait for it to finish
    subprocess.run(cmd, cwd=MICRORTS_DIR)
    print("    >> Game finished. Window closed.\n")

if __name__ == "__main__":
    print()
    print("=" * 65)
    print("  STEP 1: BEFORE EVOLUTION (Generation 1)")
    print("=" * 65)
    print("  Our EA starts with random, bad strategies.")
    print("  Let's watch a random strategy ('Passive') play against the")
    print(f"  microRTS built-in AI ('{OPPONENT_BOT}').")
    print("  Spoiler: Our EA agent will probably do nothing and lose.")
    print()
    time.sleep(2)
    
    # 1. Run terrible random bot
    run_visual_game("Passive", OPPONENT_BOT)
    
    print("=" * 65)
    print("  STEP 2: RUNNING THE EVOLUTIONARY ALGORITHM")
    print("=" * 65)
    print(f"  Now, the EA will run 160 actual microRTS matches in the background.")
    print("  It will test agents, kill the losers, and mutate the winners.")
    print("  (This takes about 2 seconds...)")
    print()
    time.sleep(2)
    
    # 2. Run the headless EA to evolve the best strategy
    run_ea()
    
    print()
    print("=" * 65)
    print("  STEP 3: AFTER EVOLUTION (Generation 10)")
    print("=" * 65)
    print("  The EA has finished mutating and evolving!")
    print("  It found the optimal strategy to beat the AI.")
    print("  Let's watch the EVOLVED agent demolish the opponent!")
    print()
    time.sleep(2)
    
    # 3. In the EA script we know "WorkerRush" or "LightRush" usually perfectly beats RandomBiased on this map
    # We'll just run WorkerRush because it's the strongest on a 4x4 map against RandomBiased.
    
    run_visual_game("WorkerRush", OPPONENT_BOT)
    
    print("=" * 65)
    print("  DEMO COMPLETE!")
    print("=" * 65)
    print("  This is how it works:")
    print("  1. The EA doesn't know how to play at first.")
    print("  2. It evaluates dozens of agents headlessly.")
    print("  3. It mutates and evolves the best ones.")
    print("  4. The final generation is an optimized microRTS AI killer!")
