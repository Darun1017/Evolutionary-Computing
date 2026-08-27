"""
Terran Bot - Marine & Medivac Push Strategy
=============================================
This bot demonstrates a standard Terran build order:
  1. Build SCVs (workers) for economy
  2. Construct Supply Depots to avoid supply blocks
  3. Build Barracks → train Marines
  4. Build Factory → Starport → train Medivacs
  5. Expand to a second base when affordable
  6. Attack when army reaches critical mass

This is a scripted bot (not ML-based) to showcase how the SC2 API works.
"""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
import random


class TerranMarineBot(BotAI):
    """
    A Terran bot that executes a Marine-Medivac timing push.
    
    Build Order:
    - 14 supply: Supply Depot
    - 16 supply: Barracks x2
    - ~20 supply: Expand (Command Center)
    - Continuous Marine production
    - Factory → Starport → Medivac production
    - Attack at 25+ Marines with Medivac support
    """

    def __init__(self):
        super().__init__()
        self.attack_triggered = False
        self.max_workers = 44  # ~22 per base

    async def on_step(self, iteration: int):
        """Main bot loop - called every game step."""
        
        # Distribute workers optimally across mineral patches and gas
        await self.distribute_workers()
        
        # Execute build order phases
        await self.build_workers()
        await self.build_supply()
        await self.build_gas()
        await self.build_barracks()
        await self.build_addon()
        await self.build_factory_and_starport()
        await self.train_army()
        await self.expand()
        await self.manage_army()

        # Log status every 200 steps
        if iteration % 200 == 0:
            marines = self.units(UnitTypeId.MARINE).amount
            medivacs = self.units(UnitTypeId.MEDIVAC).amount
            workers = self.workers.amount
            print(
                f"[Terran] Step {iteration} | "
                f"Workers: {workers} | Marines: {marines} | "
                f"Medivacs: {medivacs} | Supply: {self.supply_used}/{self.supply_cap}"
            )

    async def build_workers(self):
        """Train SCVs from idle Command Centers."""
        for cc in self.townhalls.idle:
            if self.workers.amount < self.max_workers and self.can_afford(UnitTypeId.SCV):
                cc.train(UnitTypeId.SCV)

    async def build_supply(self):
        """Build Supply Depots when supply is getting tight."""
        # Build supply depot if supply is almost maxed and not already building one
        if (
            self.supply_left < 5
            and self.supply_cap < 200
            and not self.already_pending(UnitTypeId.SUPPLYDEPOT)
            and self.can_afford(UnitTypeId.SUPPLYDEPOT)
        ):
            # Build near the command center
            if self.townhalls:
                pos = self.townhalls.first.position.towards(
                    self.game_info.map_center, 5
                )
                await self.build(UnitTypeId.SUPPLYDEPOT, near=pos)

        # Lower completed supply depots (aesthetic + smaller hitbox)
        for depot in self.structures(UnitTypeId.SUPPLYDEPOT).ready:
            depot(AbilityId.MORPH_SUPPLYDEPOT_LOWER)

    async def build_gas(self):
        """Build Refineries on available Vespene Geysers near bases."""
        for cc in self.townhalls.ready:
            geysers = self.vespene_geyser.closer_than(10, cc)
            for geyser in geysers:
                if (
                    not self.gas_buildings.closer_than(1, geyser)
                    and self.can_afford(UnitTypeId.REFINERY)
                    and self.already_pending(UnitTypeId.REFINERY) < 1
                ):
                    await self.build(UnitTypeId.REFINERY, geyser)

    async def build_barracks(self):
        """Build Barracks for Marine production (up to 3)."""
        barracks_count = self.structures(UnitTypeId.BARRACKS).amount + self.already_pending(UnitTypeId.BARRACKS)
        
        # First Barracks at 16 supply
        if barracks_count < 1 and self.supply_used >= 14 and self.can_afford(UnitTypeId.BARRACKS):
            if self.townhalls:
                pos = self.townhalls.first.position.towards(self.game_info.map_center, 8)
                await self.build(UnitTypeId.BARRACKS, near=pos)
        
        # Second and Third Barracks after first is done
        elif barracks_count < 3 and self.structures(UnitTypeId.BARRACKS).ready.amount >= 1:
            if self.can_afford(UnitTypeId.BARRACKS):
                if self.townhalls:
                    pos = self.townhalls.first.position.towards(self.game_info.map_center, 10)
                    await self.build(UnitTypeId.BARRACKS, near=pos)

    async def build_addon(self):
        """Add Tech Labs to Barracks for Stimpack research."""
        for rax in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if not rax.has_add_on:
                if self.can_afford(UnitTypeId.BARRACKSTECHLAB):
                    rax.build(UnitTypeId.BARRACKSTECHLAB)

    async def build_factory_and_starport(self):
        """Build Factory and Starport for Medivac production."""
        # Factory - prerequisite for Starport
        if (
            self.structures(UnitTypeId.BARRACKS).ready.amount >= 1
            and self.structures(UnitTypeId.FACTORY).amount + self.already_pending(UnitTypeId.FACTORY) < 1
            and self.can_afford(UnitTypeId.FACTORY)
        ):
            if self.townhalls:
                pos = self.townhalls.first.position.towards(self.game_info.map_center, 12)
                await self.build(UnitTypeId.FACTORY, near=pos)

        # Starport - for Medivac production
        if (
            self.structures(UnitTypeId.FACTORY).ready.amount >= 1
            and self.structures(UnitTypeId.STARPORT).amount + self.already_pending(UnitTypeId.STARPORT) < 1
            and self.can_afford(UnitTypeId.STARPORT)
        ):
            if self.townhalls:
                pos = self.townhalls.first.position.towards(self.game_info.map_center, 14)
                await self.build(UnitTypeId.STARPORT, near=pos)

    async def train_army(self):
        """Train Marines from Barracks and Medivacs from Starport."""
        # Train Marines
        for rax in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if self.can_afford(UnitTypeId.MARINE) and self.supply_left > 0:
                rax.train(UnitTypeId.MARINE)

        # Train Medivacs
        for starport in self.structures(UnitTypeId.STARPORT).ready.idle:
            if self.can_afford(UnitTypeId.MEDIVAC) and self.supply_left > 1:
                starport.train(UnitTypeId.MEDIVAC)

    async def expand(self):
        """Take a second base when affordable and barracks are built."""
        if (
            self.townhalls.amount < 2
            and self.structures(UnitTypeId.BARRACKS).ready.amount >= 1
            and self.can_afford(UnitTypeId.COMMANDCENTER)
            and not self.already_pending(UnitTypeId.COMMANDCENTER)
        ):
            await self.expand_now()

    async def manage_army(self):
        """
        Attack logic:
        - Rally units at a staging point until critical mass
        - Attack-move toward the enemy base when ready
        - Send reinforcements to the fight
        """
        marines = self.units(UnitTypeId.MARINE)
        medivacs = self.units(UnitTypeId.MEDIVAC)

        # Determine attack threshold
        if marines.amount >= 25 or (marines.amount >= 15 and medivacs.amount >= 2):
            self.attack_triggered = True

        if self.attack_triggered:
            # Attack-move all army units toward enemy start location
            target = self.enemy_start_locations[0]
            
            for marine in marines.idle:
                marine.attack(target)
            for medivac in medivacs.idle:
                medivac.move(target)
        else:
            # Rally idle units near the main ramp
            if self.townhalls:
                rally_point = self.townhalls.first.position.towards(
                    self.game_info.map_center, 8
                )
                for marine in marines.idle:
                    marine.move(rally_point)
