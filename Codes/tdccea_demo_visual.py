"""
=============================================================
TD-CCEA + microRTS DEMO (VISUAL PRESENTAION)
=============================================================

This script visually demonstrates the rigorous TD-CCEA algorithm 
(Trajectory-Driven Competitive Co-Evolutionary Algorithm).

It does 3 things:
1. Shows a "BEFORE EVOLUTION" video (Passive random agent losing).
2. Runs the full mathematical TD-CCEA headlessly to optimize.
3. Shows an "AFTER EVOLUTION" video of the optimized agent winning.
=============================================================
"""

import subprocess
import os
import time
from ea_microrts import MICRORTS_DIR, MAP_PATH_8X8

def run_visual_game(bot1, bot2, map_path=MAP_PATH_8X8):
    cmd = [
        "java", "-cp", "lib/*;bin",
        "tests.VisualGameRunner",
        bot1, bot2, map_path
    ]
    print(f"[Visualizer] Invoking Native Java GUI -> {bot1} vs {bot2}")
    subprocess.run(cmd, cwd=MICRORTS_DIR)
    print("[Visualizer] GUI Terminated.\n")
