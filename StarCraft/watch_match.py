"""
StarCraft II - Observer Match: Watch Terran vs Zerg (Both Sides Visible)
=========================================================================
Runs a match where you can see EVERYTHING on the map — both players' bases,
armies, and strategies. Fog of war is disabled via debug commands.

Your Terran bot plays against the built-in Zerg AI.
You observe from a god-view with full map visibility.

Usage:
    python watch_match.py
    python watch_match.py --difficulty CheatInsane
    python watch_match.py --map Flat64
"""

import argparse
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.data import Race, Difficulty
from sc2.player import Bot, Computer
from sc2.main import run_game
from sc2 import maps


class ObservableTerranBot(BotAI):
    """
    Terran bot with full map vision enabled so you can watch both sides.
    Plays a Marine-Medivac push while revealing the entire map.
    """

    def __init__(self):
        super().__init__()
        self.attack_triggered = False

    async def on_start(self):
        """Reveal entire map at game start so we can see both players."""
        await self.client.debug_show_map()

    async def on_step(self, iteration: int):
        await self.distribute_workers()
        await self.build_workers()
        await self.build_supply()
        await self.build_gas()
        await self.build_barracks()
        await self.build_factory_starport()
        await self.train_army()
        await self.expand()
        await self.manage_army()

        if iteration % 300 == 0:
            m = self.units(UnitTypeId.MARINE).amount
            mv = self.units(UnitTypeId.MEDIVAC).amount
            print(f"[Terran] Step {iteration} | Workers:{self.workers.amount} Marines:{m} Medivacs:{mv} | {self.supply_used}/{self.supply_cap}")

    async def build_workers(self):
        for cc in self.townhalls.idle:
            if self.workers.amount < 44 and self.can_afford(UnitTypeId.SCV):
                cc.train(UnitTypeId.SCV)

    async def build_supply(self):
        if self.supply_left < 5 and self.supply_cap < 200 and not self.already_pending(UnitTypeId.SUPPLYDEPOT) and self.can_afford(UnitTypeId.SUPPLYDEPOT):
            if self.townhalls:
                await self.build(UnitTypeId.SUPPLYDEPOT, near=self.townhalls.first.position.towards(self.game_info.map_center, 5))
        for depot in self.structures(UnitTypeId.SUPPLYDEPOT).ready:
            depot(AbilityId.MORPH_SUPPLYDEPOT_LOWER)

    async def build_gas(self):
        for cc in self.townhalls.ready:
            for geyser in self.vespene_geyser.closer_than(10, cc):
                if not self.gas_buildings.closer_than(1, geyser) and self.can_afford(UnitTypeId.REFINERY) and self.already_pending(UnitTypeId.REFINERY) < 1:
                    await self.build(UnitTypeId.REFINERY, geyser)

    async def build_barracks(self):
        count = self.structures(UnitTypeId.BARRACKS).amount + self.already_pending(UnitTypeId.BARRACKS)
        if count < 3 and self.can_afford(UnitTypeId.BARRACKS) and self.supply_used >= 14:
            if self.townhalls:
                await self.build(UnitTypeId.BARRACKS, near=self.townhalls.first.position.towards(self.game_info.map_center, 8))

    async def build_factory_starport(self):
        if self.structures(UnitTypeId.BARRACKS).ready and not self.structures(UnitTypeId.FACTORY) and not self.already_pending(UnitTypeId.FACTORY) and self.can_afford(UnitTypeId.FACTORY):
            if self.townhalls:
                await self.build(UnitTypeId.FACTORY, near=self.townhalls.first.position.towards(self.game_info.map_center, 12))
        if self.structures(UnitTypeId.FACTORY).ready and not self.structures(UnitTypeId.STARPORT) and not self.already_pending(UnitTypeId.STARPORT) and self.can_afford(UnitTypeId.STARPORT):
            if self.townhalls:
                await self.build(UnitTypeId.STARPORT, near=self.townhalls.first.position.towards(self.game_info.map_center, 14))

    async def train_army(self):
        for rax in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if self.can_afford(UnitTypeId.MARINE) and self.supply_left > 0:
                rax.train(UnitTypeId.MARINE)
        for sp in self.structures(UnitTypeId.STARPORT).ready.idle:
            if self.can_afford(UnitTypeId.MEDIVAC) and self.supply_left > 1:
                sp.train(UnitTypeId.MEDIVAC)

    async def expand(self):
        if self.townhalls.amount < 2 and self.structures(UnitTypeId.BARRACKS).ready and self.can_afford(UnitTypeId.COMMANDCENTER) and not self.already_pending(UnitTypeId.COMMANDCENTER):
            await self.expand_now()

    async def manage_army(self):
        marines = self.units(UnitTypeId.MARINE)
        medivacs = self.units(UnitTypeId.MEDIVAC)
        if marines.amount >= 20 or (marines.amount >= 12 and medivacs.amount >= 1):
            self.attack_triggered = True
        if self.attack_triggered:
            for u in marines.idle:
                u.attack(self.enemy_start_locations[0])
            for u in medivacs.idle:
                u.move(self.enemy_start_locations[0])
        elif self.townhalls:
            rally = self.townhalls.first.position.towards(self.game_info.map_center, 8)
            for u in marines.idle:
                u.move(rally)


DIFFICULTIES = {
    "VeryEasy": Difficulty.VeryEasy, "Easy": Difficulty.Easy,
    "Medium": Difficulty.Medium, "MediumHard": Difficulty.MediumHard,
    "Hard": Difficulty.Hard, "Harder": Difficulty.Harder,
    "VeryHard": Difficulty.VeryHard, "CheatVision": Difficulty.CheatVision,
    "CheatMoney": Difficulty.CheatMoney, "CheatInsane": Difficulty.CheatInsane,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch SC2 Match - Full Map Vision")
    parser.add_argument("--map", default="AcropolisAIE", help="Map name (default: AcropolisAIE)")
    parser.add_argument("--difficulty", default="Hard", choices=list(DIFFICULTIES.keys()), help="AI difficulty")
    parser.add_argument("--fast", action="store_true", help="Run at max speed (not real-time)")
    args = parser.parse_args()

    speed = "MAX SPEED" if args.fast else "Real-time (press + / - in-game to adjust)"

    print("=" * 55)
    print("  STARCRAFT II - OBSERVER MATCH (Full Map Vision)")
    print("=" * 55)
    print(f"  Terran Bot vs Zerg AI ({args.difficulty})")
    print(f"  Map: {args.map}")
    print(f"  Speed: {speed}")
    print(f"  Fog of War: DISABLED (you see everything)")
    print("=" * 55)
    print("  TIP: Press + / - in-game to change speed")
    print("=" * 55)

    run_game(
        maps.get(args.map),
        [
            Bot(Race.Terran, ObservableTerranBot()),
            Computer(Race.Zerg, DIFFICULTIES[args.difficulty]),
        ],
        realtime=not args.fast,
    )
