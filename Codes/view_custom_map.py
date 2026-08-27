import time
from tdccea_demo_visual import run_visual_game
from ea_microrts import OPPONENT_BOT

CUSTOM_UNFAIR_MAP = "maps/8x8/custom_unfair_2bases.xml"

if __name__ == "__main__":
    print("\n=========================================================")
    print("  CUSTOM MAP TOPOLOGY DEMONSTRATION")
    print("=========================================================")
    print("  We have injected a custom XML file into the Java engine.")
    print("  Player 1 (Red) now spawns with TWO separate bases and ")
    print("  extra resources, giving them a massive unfair advantage.")
    print("\n  Launching Visualizer to verify topography...\n")
    
    time.sleep(2)
    # We run "Passive" vs "Passive" so nothing happens and you can stare at the map layout
    run_visual_game("Passive", "Passive", map_path=CUSTOM_UNFAIR_MAP)
