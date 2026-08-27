"""
Zerg Bot - Zergling Rush into Roach-Hydra Composition
======================================================
This bot demonstrates a standard Zerg build order:
  1. Build Drones (workers) on Hatchery larvae
  2. Expand early (Zerg economy depends on multiple bases)
  3. Build Spawning Pool → mass Zerglings
  4. Build Roach Warren + Hydralisk Den for mid-game transition
  5. Attack with a Zergling flood, then follow up with Roach-Hydra

This is a scripted bot (not ML-based) to showcase how the SC2 API works.
"""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2


class ZergRoachHydraBot(BotAI):
    """
    A Zerg bot that executes a Zergling Rush into Roach-Hydra mid-game.
    
    Build Order:
    - 13 supply: Overlord
    - 17 supply: Hatchery (Natural expansion)
    - 17 supply: Spawning Pool
    - 19 supply: Extractor (Gas)
    - Queen production
    - Mass Zerglings → Roach Warren → Hydralisk Den
    - Attack at 20+ army units
    """

    def __init__(self):
        super().__init__()
        self.attack_triggered = False
        self.max_workers = 44
        self.zerglings_made = 0

    async def on_step(self, iteration: int):
        """Main bot loop - called every game step."""
        
        # Inject larvae into Hatcheries using Queens
        await self.inject_larvae()
        
        # Distribute workers across bases
        await self.distribute_workers()
        
        # Execute build order
        await self.build_workers()
        await self.build_overlords()
        await self.build_gas()
        await self.build_spawning_pool()
        await self.build_queens()
        await self.build_roach_warren()
        await self.build_hydralisk_den()
        await self.train_army()
        await self.expand()
        await self.research_upgrades()
        await self.manage_army()

        # Log status every 200 steps
        if iteration % 200 == 0:
            zerglings = self.units(UnitTypeId.ZERGLING).amount
            roaches = self.units(UnitTypeId.ROACH).amount
            hydras = self.units(UnitTypeId.HYDRALISK).amount
            queens = self.units(UnitTypeId.QUEEN).amount
            workers = self.workers.amount
            print(
                f"[Zerg]   Step {iteration} | "
                f"Workers: {workers} | Lings: {zerglings} | "
                f"Roaches: {roaches} | Hydras: {hydras} | "
                f"Queens: {queens} | Supply: {self.supply_used}/{self.supply_cap}"
            )

    async def inject_larvae(self):
        """Use Queens to inject larvae into Hatcheries for faster unit production."""
        for queen in self.units(UnitTypeId.QUEEN).idle:
            # Find nearest hatchery that isn't already injected
            if self.townhalls:
                hatch = self.townhalls.closest_to(queen)
                if queen.energy >= 25:
                    queen(AbilityId.EFFECT_INJECTLARVA, hatch)

    async def build_workers(self):
        """Train Drones from larvae at idle Hatcheries."""
        if self.workers.amount < self.max_workers and self.larva:
            if self.can_afford(UnitTypeId.DRONE):
                self.larva.first.train(UnitTypeId.DRONE)

    async def build_overlords(self):
        """Build Overlords for supply (Zerg's supply buildings)."""
        if (
            self.supply_left < 4
            and self.supply_cap < 200
            and self.already_pending(UnitTypeId.OVERLORD) < 2
            and self.larva
            and self.can_afford(UnitTypeId.OVERLORD)
        ):
            self.larva.first.train(UnitTypeId.OVERLORD)

    async def build_gas(self):
        """Build Extractors on Vespene Geysers near Hatcheries."""
        for hatch in self.townhalls.ready:
            geysers = self.vespene_geyser.closer_than(10, hatch)
            for geyser in geysers:
                if (
                    not self.gas_buildings.closer_than(1, geyser)
                    and self.can_afford(UnitTypeId.EXTRACTOR)
                    and self.already_pending(UnitTypeId.EXTRACTOR) < 1
                    and self.structures(UnitTypeId.SPAWNINGPOOL).ready.amount >= 1
                ):
                    await self.build(UnitTypeId.EXTRACTOR, geyser)

    async def build_spawning_pool(self):
        """Build Spawning Pool - unlocks Zerglings and Queens."""
        if (
            self.structures(UnitTypeId.SPAWNINGPOOL).amount
            + self.already_pending(UnitTypeId.SPAWNINGPOOL) < 1
            and self.can_afford(UnitTypeId.SPAWNINGPOOL)
            and self.supply_used >= 15
        ):
            if self.townhalls:
                pos = self.townhalls.first.position.towards(
                    self.game_info.map_center, 5
                )
                await self.build(UnitTypeId.SPAWNINGPOOL, near=pos)

    async def build_queens(self):
        """Train Queens from Hatcheries (up to 1 per base)."""
        queens_count = self.units(UnitTypeId.QUEEN).amount + self.already_pending(UnitTypeId.QUEEN)
        if (
            self.structures(UnitTypeId.SPAWNINGPOOL).ready.amount >= 1
            and queens_count < self.townhalls.amount
            and self.can_afford(UnitTypeId.QUEEN)
        ):
            for hatch in self.townhalls.idle:
                if queens_count < self.townhalls.amount:
                    hatch.train(UnitTypeId.QUEEN)
                    queens_count += 1

    async def build_roach_warren(self):
        """Build Roach Warren - unlocks Roaches (tanky ground unit)."""
        if (
            self.structures(UnitTypeId.SPAWNINGPOOL).ready.amount >= 1
            and self.structures(UnitTypeId.ROACHWARREN).amount
            + self.already_pending(UnitTypeId.ROACHWARREN) < 1
            and self.can_afford(UnitTypeId.ROACHWARREN)
            and self.vespene >= 100
        ):
            if self.townhalls:
                pos = self.townhalls.first.position.towards(
                    self.game_info.map_center, 7
                )
                await self.build(UnitTypeId.ROACHWARREN, near=pos)

    async def build_hydralisk_den(self):
        """Build Hydralisk Den (requires Lair) - unlocks Hydralisks."""
        # First, morph a Hatchery into a Lair
        if (
            self.structures(UnitTypeId.SPAWNINGPOOL).ready.amount >= 1
            and not self.structures(UnitTypeId.LAIR)
            and not self.structures(UnitTypeId.HIVE)
            and self.already_pending(UnitTypeId.LAIR) < 1
        ):
            for hatch in self.townhalls(UnitTypeId.HATCHERY).idle:
                if self.can_afford(UnitTypeId.LAIR):
                    hatch(AbilityId.UPGRADETOLAIR_LAIR)
                    break

        # Then build Hydralisk Den
        lair_or_hive = self.structures(UnitTypeId.LAIR).ready.amount + self.structures(UnitTypeId.HIVE).ready.amount
        if (
            lair_or_hive >= 1
            and self.structures(UnitTypeId.HYDRALISKDEN).amount
            + self.already_pending(UnitTypeId.HYDRALISKDEN) < 1
            and self.can_afford(UnitTypeId.HYDRALISKDEN)
        ):
            if self.townhalls:
                pos = self.townhalls.first.position.towards(
                    self.game_info.map_center, 9
                )
                await self.build(UnitTypeId.HYDRALISKDEN, near=pos)

    async def train_army(self):
        """Train army units from larvae based on tech availability."""
        if not self.larva:
            return

        # If Hydralisk Den is ready, prioritize Hydralisks
        if self.structures(UnitTypeId.HYDRALISKDEN).ready:
            if self.can_afford(UnitTypeId.HYDRALISK) and self.supply_left >= 2:
                self.larva.first.train(UnitTypeId.HYDRALISK)
                return

        # If Roach Warren is ready, train Roaches
        if self.structures(UnitTypeId.ROACHWARREN).ready:
            if self.can_afford(UnitTypeId.ROACH) and self.supply_left >= 2:
                self.larva.first.train(UnitTypeId.ROACH)
                return

        # Otherwise, mass Zerglings
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready:
            if self.can_afford(UnitTypeId.ZERGLING) and self.supply_left >= 1:
                self.larva.first.train(UnitTypeId.ZERGLING)

    async def expand(self):
        """Take additional bases (Zerg economy thrives on expansions)."""
        if (
            self.townhalls.amount < 3
            and self.can_afford(UnitTypeId.HATCHERY)
            and not self.already_pending(UnitTypeId.HATCHERY)
        ):
            await self.expand_now()

    async def research_upgrades(self):
        """Research Metabolic Boost (Zergling speed) and Glial Reconstitution (Roach speed)."""
        # Zergling Speed (Metabolic Boost)
        if (
            self.structures(UnitTypeId.SPAWNINGPOOL).ready
            and self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED) == 0
            and self.can_afford(UpgradeId.ZERGLINGMOVEMENTSPEED)
        ):
            pool = self.structures(UnitTypeId.SPAWNINGPOOL).ready.first
            pool.research(UpgradeId.ZERGLINGMOVEMENTSPEED)

    async def manage_army(self):
        """
        Attack logic:
        - Gather army at staging area
        - Attack when army reaches critical mass
        """
        zerglings = self.units(UnitTypeId.ZERGLING)
        roaches = self.units(UnitTypeId.ROACH)
        hydras = self.units(UnitTypeId.HYDRALISK)

        total_army = zerglings.amount + roaches.amount + hydras.amount

        # Trigger attack at 20+ army supply
        if total_army >= 20:
            self.attack_triggered = True

        if self.attack_triggered:
            target = self.enemy_start_locations[0]
            
            for unit in zerglings.idle:
                unit.attack(target)
            for unit in roaches.idle:
                unit.attack(target)
            for unit in hydras.idle:
                unit.attack(target)
        else:
            # Rally at natural expansion or main base ramp
            if self.townhalls:
                rally_point = self.townhalls.first.position.towards(
                    self.game_info.map_center, 8
                )
                for unit in zerglings.idle:
                    unit.move(rally_point)
                for unit in roaches.idle:
                    unit.move(rally_point)
                for unit in hydras.idle:
                    unit.move(rally_point)
