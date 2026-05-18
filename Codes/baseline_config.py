"""
baseline_config.py
Independent infrastructure for comparative baseline testing against TD-CCEA.
"""
import subprocess
import os

MICRORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Simulation", "microrts")

BOT_TYPES = ["WorkerRush", "LightRush", "RandomBiased", "HeavyRush", "Random", "Passive"]

# --- MAP TOPOLOGIES (Switch these out in your baseline scripts if needed) ---
MAP_PATH_8X8 = "maps/8x8/basesWorkers8x8Obstacle.xml"
MAP_PATH_16X16 = "maps/16x16/basesWorkers16x16A.xml" # or custom_unfair_16x16.xml
MAP_PATH_32X32 = "maps/basesWorkers32x32A.xml"

# The standard opponent bot for the baseline tests
OPPONENT_BOT = "WorkerRush"

class BaselineMicroRTSServer:
    """Manages a persistent Java process for running headless benchmark games."""
    def __init__(self):
        self.process = None

    def start(self):
        cmd = ["java", "-cp", "lib/*;bin", "tests.BatchGameRunner"]
        self.process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=MICRORTS_DIR
        )
        line = self.process.stdout.readline().strip()
        if line != "READY":
            raise RuntimeError(f"BatchGameRunner failed to start: {line}")

    def run_game(self, bot1, bot2, map_path, max_cycles=4000):
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
        if self.process:
            try:
                self.process.stdin.write("EXIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
