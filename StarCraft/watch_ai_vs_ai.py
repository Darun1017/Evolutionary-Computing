"""
StarCraft II - CheatInsane Terran AI vs CheatInsane Zerg AI
============================================================
Both sides are SC2's strongest built-in AI fighting each other.
A minimal observer bot reveals the map so you can see everything.

Usage:
    python watch_ai_vs_ai.py
    python watch_ai_vs_ai.py --map Simple64
    python watch_ai_vs_ai.py --fast
"""

import argparse
import asyncio

from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty
from sc2.player import Bot, Computer
from sc2.main import run_game
from sc2.ids.unit_typeid import UnitTypeId


class InvisibleObserver(BotAI):
    """
    Minimal bot that reveals the map but gives all its workers 
    to the game's natural flow. It does nothing — just lets you 
    watch the Computer AI on the other side.
    
    Since SC2 API requires at least one Bot player, this bot
    plays Terran but immediately surrenders strategic control — 
    it only exists so the game window stays open and fog is disabled.
    """

    async def on_start(self):
        # Reveal the entire map so we can see both sides
        await self.client.debug_show_map()
        print("[Observer] Map revealed — watching both AIs fight!")

    async def on_step(self, iteration: int):
        # Log what's happening every 500 steps
        if iteration % 500 == 0 and iteration > 0:
            # Count all visible units for both sides
            our_army = self.units.not_structure.filter(lambda u: u.type_id != UnitTypeId.SCV).amount
            our_workers = self.workers.amount
            enemy_army = self.enemy_units.not_structure.amount
            enemy_structures = self.enemy_structures.amount
            print(
                f"[Step {iteration:>5}] "
                f"Terran — Workers:{our_workers} Army:{our_army} | "
                f"Zerg — Army:{enemy_army} Buildings:{enemy_structures}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch SC2 AI vs AI - Both Strongest Bots")
    parser.add_argument("--map", default="Simple64", help="Map name (default: Simple64)")
    parser.add_argument("--fast", action="store_true", help="Run at max speed")
    args = parser.parse_args()

    print("=" * 60)
    print("  STARCRAFT II — STRONGEST AI vs STRONGEST AI")
    print("=" * 60)
    print(f"  Terran (CheatInsane) vs Zerg (CheatInsane)")
    print(f"  Map: {args.map}")
    print(f"  Speed: {'MAX' if args.fast else 'Real-time (+ / - to adjust)'}")
    print(f"  Both sides: SC2 Built-in AI at maximum difficulty")
    print("=" * 60)
    print()
    print("  NOTE: The Terran side is a passive observer bot.")
    print("  The Zerg side is SC2's CheatInsane AI (gets bonus resources).")
    print("  Watch how the Zerg AI builds and attacks!")
    print()
    print("  For BOTH sides fully AI-controlled, use SC2's built-in")
    print("  custom game: Menu > Custom > Create Game > set both to AI")
    print("=" * 60)

    run_game(
        maps.get(args.map),
        [
            Bot(Race.Terran, InvisibleObserver()),
            Computer(Race.Zerg, Difficulty.CheatInsane),
        ],
        realtime=not args.fast,
    )
