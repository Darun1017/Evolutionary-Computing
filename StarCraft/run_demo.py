"""
StarCraft II Demo Match Runner
================================
Runs a demo match: Terran Bot vs Zerg Bot (or vs Built-in AI)

Usage:
    python run_demo.py                    # Bot vs Bot (Terran vs Zerg)
    python run_demo.py --mode bot_vs_ai   # Terran Bot vs Built-in Zerg AI
    python run_demo.py --mode ai_vs_ai    # Built-in Terran AI vs Zerg AI
    python run_demo.py --realtime         # Watch in real-time speed
    python run_demo.py --difficulty Hard   # Set AI difficulty

Available difficulties:
    VeryEasy, Easy, Medium, MediumHard, Hard, Harder, VeryHard, CheatVision, 
    CheatMoney, CheatInsane

Maps available (from SC2 AI Arena 2024 S4):
    AbyssalReefAIE, AcropolisAIE, AutomatonAIE, EphemeronAIE, 
    InterloperAIE, ThunderbirdAIE
"""

import sys
import argparse

from sc2 import maps
from sc2.player import Bot, Computer
from sc2.main import run_game
from sc2.data import Race, Difficulty

from terran_bot import TerranMarineBot
from zerg_bot import ZergRoachHydraBot


# Map difficulty levels
DIFFICULTY_MAP = {
    "VeryEasy": Difficulty.VeryEasy,
    "Easy": Difficulty.Easy,
    "Medium": Difficulty.Medium,
    "MediumHard": Difficulty.MediumHard,
    "Hard": Difficulty.Hard,
    "Harder": Difficulty.Harder,
    "VeryHard": Difficulty.VeryHard,
    "CheatVision": Difficulty.CheatVision,
    "CheatMoney": Difficulty.CheatMoney,
    "CheatInsane": Difficulty.CheatInsane,
}


def main():
    parser = argparse.ArgumentParser(description="StarCraft II Demo Match Runner")
    parser.add_argument(
        "--mode",
        choices=["bot_vs_bot", "bot_vs_ai", "ai_vs_ai"],
        default="bot_vs_ai",
        help="Match mode (default: bot_vs_ai)",
    )
    parser.add_argument(
        "--map",
        default="AcropolisAIE",
        help="Map name (default: AcropolisAIE)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Run in real-time speed (slower but watchable)",
    )
    parser.add_argument(
        "--difficulty",
        default="Hard",
        choices=list(DIFFICULTY_MAP.keys()),
        help="Built-in AI difficulty (default: Hard)",
    )
    args = parser.parse_args()

    difficulty = DIFFICULTY_MAP[args.difficulty]
    map_name = args.map

    print("=" * 60)
    print("  STARCRAFT II - DEMO MATCH")
    print("=" * 60)
    print(f"  Map:        {map_name}")
    print(f"  Mode:       {args.mode}")
    print(f"  Real-time:  {args.realtime}")
    print(f"  Difficulty: {args.difficulty}")
    print("=" * 60)

    if args.mode == "bot_vs_bot":
        # Custom Terran Bot vs Custom Zerg Bot
        print("\n  >> Terran Bot (Marine-Medivac) vs Zerg Bot (Roach-Hydra)")
        print("  >> Both bots are custom scripted AI\n")
        players = [
            Bot(Race.Terran, TerranMarineBot()),
            Bot(Race.Zerg, ZergRoachHydraBot()),
        ]

    elif args.mode == "bot_vs_ai":
        # Custom Terran Bot vs Built-in Zerg AI
        print(f"\n  >> Terran Bot (Marine-Medivac) vs Built-in Zerg AI ({args.difficulty})")
        print("  >> Your bot controls Terran, SC2 controls Zerg\n")
        players = [
            Bot(Race.Terran, TerranMarineBot()),
            Computer(Race.Zerg, difficulty),
        ]

    elif args.mode == "ai_vs_ai":
        # Built-in Terran AI vs Built-in Zerg AI (observer mode)
        print(f"\n  >> Built-in Terran AI vs Built-in Zerg AI ({args.difficulty})")
        print("  >> Both sides are SC2's built-in AI - observe only\n")
        players = [
            Computer(Race.Terran, difficulty),
            Computer(Race.Zerg, difficulty),
        ]

    print("  Launching StarCraft II...")
    print("  (The game window will open shortly)\n")

    # Run the game
    result = run_game(
        maps.get(map_name),
        players,
        realtime=args.realtime,
    )

    print("\n" + "=" * 60)
    print(f"  MATCH RESULT: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
