import os
import subprocess


MY_STRATEGY = [0.2436, 0.5265, 0.5230, 0.9714, 0.1496, 
0.2987, 0.6205, 0.1790, 0.7140, 0.6314, 0.1318, 0.0662,
 0.9955, 0.1236, 0.0000, 0.6163, 0.9946, 0.4662, 0.5703, 1]

# =====================================================================
# GAME SETTINGS
# =====================================================================

# Choose your opponent. Options include:
# WorkerRush, LightRush, HeavyRush, RangedRush, EconomyRush, 
# EconomyMilitaryRush, WorkerDefense, LightDefense, HeavyDefense...
OPPONENT = "EconomyRush"

# Choose your map. Options include:
# maps/8x8/basesWorkers8x8.xml
# maps/16x16/basesWorkers16x16.xml
# maps/chambers32x32.xml
# maps/fortress_corridors_32x32.xml
# maps/cross_barricade_32x32.xml
MAP_PATH = "maps/strategic_depth_32x32.xml"

# =====================================================================
# RUN SCRIPT
# =====================================================================
def main():
    if len(MY_STRATEGY) != 20:
        print(f"ERROR: Strategy list must have exactly 20 weights. You provided {len(MY_STRATEGY)}.")
        return

    print("==================================================")
    print(" LAUNCHING VISUAL MATCH")
    print("==================================================")
    print(f" Your Strategy vs {OPPONENT}")
    print(f" Map: {MAP_PATH}")
    print("==================================================\n")

    # Format the weights for the Java backend (EVOLVED:w0;w1;...)
    weight_str = ";".join(f"{w:.4f}" for w in MY_STRATEGY)
    ai_arg = f"EVOLVED:{weight_str}"

    # Path to the microrts Simulation folder
    microrts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        "Simulation", "microrts"
    )

    # Command to run the visual GUI
    cmd = [
        "java", "-cp", "lib/*;bin", 
        "tests.EvolvedVisualGameRunner", 
        ai_arg, 
        OPPONENT, 
        MAP_PATH
    ]

    print("Starting GUI... (Please wait a moment)")
    print("To close the game early, just close the popup window.\n")
    
    try:
        # Run the java process
        process = subprocess.Popen(
            cmd,
            cwd=microrts_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Print output in real time
        for line in process.stdout:
            print(line.strip())
            
        process.wait()

    except FileNotFoundError:
        print("ERROR: Java not found! Make sure Java is installed and in your PATH.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
