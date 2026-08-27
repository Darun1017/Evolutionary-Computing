"""
Terran Macro Bot - 23-Parameter Strategy Controller & Evaluation Suite
========================================================================
Implements the three-layer architecture specified from the TextStarCraft II paper:
  - Layer 1: Strategy Controller (23-Parameter Vector evolved by CCE-TDCCEA)
  - Layer 2: Macro Execution Script (Terran Action Dictionary & 6-Category Observation Logger)
  - Layer 3: Micro Combat Script (Automatic unit targeting, harassment, and retreat)
  - Evaluation: Computes the 5 TextStarCraft II metrics (WINRATE, PBR, RUR, APU, TR)
"""

import math
import random
from typing import List, Dict, Any, Optional

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2


class TerranMacroBot(BotAI):
    """
    Terran Macro Execution Bot driven by a 23-Parameter Vector.
    Maps parameters to discrete Terran Action Dictionary commands and logs
    the 6 Observation Categories and 5 Evaluation Metrics.
    """

    def __init__(self, param_vector: Optional[List[float]] = None):
        super().__init__()
        # If no parameter vector is provided, initialize with balanced default 0.5 values
        if param_vector is None or len(param_vector) != 23:
            self.params = [0.5] * 23
        else:
            self.params = [max(0.0, min(1.0, float(v))) for v in param_vector]

        # Decode Parameter Vector into interpreted game thresholds
        self._decode_parameters()

        # State tracking for Layer 3 (Micro & Strategy)
        self.attack_triggered = False
        self.pre_engagement_army_size = 0
        self.last_harass_step = -1000
        self.last_retreat_step = -500
        self.reached_max_supply = False
        self.max_supply_step = -1

        # 6 Observation Categories Storage
        self.latest_observations: Dict[str, Any] = {}

        # 5 Evaluation Metrics Tracking — manual spend accumulation
        self.prev_minerals = 50
        self.prev_vespene = 0
        self.total_minerals_spent = 0
        self.total_gas_spent = 0
        self.supply_blocked_steps = 0
        self.apu_history: List[float] = []

        # Available Terran Technologies and Tech Buildings for TR calculation
        self.total_available_technologies = 15

    def _decode_parameters(self):
        """
        Decodes the 23 normalized [0.0, 1.0] parameters into Terran Action Dictionary thresholds.
        """
        p = self.params
        # Parameter 0: worker saturation target (0.5 to 1.0 -> 12 to 24 workers per base)
        self.target_workers_per_base = int(12 + (p[0] - 0.5) * 24) if p[0] >= 0.5 else int(12 * (p[0] / 0.5))
        self.target_workers_per_base = max(10, min(24, self.target_workers_per_base))

        # Parameter 1: gas timing (0.0 to 1.0 -> supply 12 to 35)
        self.gas_timing_supply = int(12 + p[1] * 23)

        # Parameter 2: expansion timing (0.0 to 1.0 -> supply 28 to 70)
        self.expansion_timing_supply = int(28 + p[2] * 42)

        # Parameter 3: supply buffer (0.0 to 1.0 -> 2 to 10 supply ahead)
        self.supply_buffer_threshold = int(2 + p[3] * 8)

        # Parameter 4: orbital command priority
        self.orbital_priority = p[4]

        # Parameter 5: barracks count (0.2 to 1.0 -> 1 to 5 barracks)
        self.target_barracks_count = max(1, int(1 + p[5] * 4))

        # Parameter 6: factory preference
        self.factory_preference = p[6]

        # Parameter 7: addon ratio (tech lab vs reactor)
        self.tech_lab_ratio = p[7]

        # Parameter 8: marine ratio (from Reactor Barracks)
        self.marine_ratio = p[8]

        # Parameter 9: marauder ratio (from Tech Lab Barracks)
        self.marauder_ratio = p[9]

        # Parameter 10: medivac ratio (from Starport)
        self.medivac_ratio = p[10]

        # Parameter 11: siege tank ratio (from Factory)
        self.siege_tank_ratio = p[11]

        # Parameter 12: starport preference
        self.starport_preference = p[12]

        # Parameter 13: upgrade priority (<0.5 = Weapons first, >0.5 = Armor first)
        self.upgrade_priority = p[13]

        # Parameter 14: attack threshold (0.0 to 1.0 -> army supply 30 to 100)
        self.attack_supply_threshold = int(30 + p[14] * 70)

        # Parameter 15: retreat threshold (0.1 to 0.9 proportion of army lost)
        self.retreat_loss_threshold = 0.1 + p[15] * 0.8

        # Parameter 16: aggression level (lowers effective attack threshold)
        self.aggression_level = p[16]
        self.effective_attack_threshold = int(self.attack_supply_threshold * (1.0 - 0.3 * self.aggression_level))

        # Parameter 17: harassment frequency (timer in game steps)
        self.harass_interval_steps = max(200, int((1.0 - p[17]) * 2000)) if p[17] > 0.01 else 999999

        # Parameter 18: defensive structure count (0 to 5 Bunkers/Turrets)
        self.defensive_structure_target = int(p[18] * 5.0)

        # Parameter 19: tech switch willingness (0.0 to 1.0)
        self.tech_switch_willingness = p[19]

        # Parameter 20: MULE vs Scan ratio
        self.scan_ratio = p[20]

        # Parameter 21: supply depot wall off (>0.5 places at ramp wall-off)
        self.wall_off_enabled = (p[21] > 0.5)

        # Parameter 22: reaper ratio (first 5 minutes < 300 seconds)
        self.reaper_ratio = p[22]

    async def on_step(self, iteration: int):
        """
        Main Macro Execution Loop (Layer 2 & Layer 3).
        Called every game step by the SC2 engine.
        """
        # 1. Update Observation Categories and track metrics
        self._update_observation_categories(iteration)
        self._track_evaluation_metrics(iteration)

        # 2. Layer 2: Macro Execution Script - Action Dictionary commands
        await self.distribute_workers()
        await self.execute_economy_actions(iteration)
        await self.execute_infrastructure_actions(iteration)
        await self.execute_research_actions(iteration)
        await self.execute_unit_training_actions(iteration)
        await self.execute_strategic_actions(iteration)

        # 3. Layer 3: Micro Combat Script
        await self.execute_micro_script(iteration)

    # =========================================================================
    #   6 OBSERVATION CATEGORIES LOGGER
    # =========================================================================
    def _update_observation_categories(self, iteration: int):
        """
        Logs all 6 Observation Categories specified in the paper at every step.
        """
        # Category 1: Resources
        cat1 = {
            "worker_supply_count": self.workers.amount,
            "mineral_amount": self.minerals,
            "gas_amount": self.vespene,
            "supply_left": self.supply_left,
            "supply_cap": self.supply_cap,
            "supply_used": self.supply_used,
            "base_count": self.townhalls.amount,
        }

        # Category 2: Units (alive count by unit type)
        cat2: Dict[str, int] = {}
        for unit in self.units:
            name = unit.type_id.name
            cat2[name] = cat2.get(name, 0) + 1

        # Category 3: Buildings (constructed count by building type)
        cat3: Dict[str, int] = {}
        for struct in self.structures:
            name = struct.type_id.name
            cat3[name] = cat3.get(name, 0) + 1

        # Category 4: In-Process Activities (planning_unitname_count)
        cat4: Dict[str, int] = {}
        for unit_type in [
            UnitTypeId.SCV, UnitTypeId.MARINE, UnitTypeId.MARAUDER,
            UnitTypeId.REAPER, UnitTypeId.SIEGETANK, UnitTypeId.MEDIVAC,
            UnitTypeId.SUPPLYDEPOT, UnitTypeId.BARRACKS, UnitTypeId.FACTORY,
            UnitTypeId.STARPORT, UnitTypeId.COMMANDCENTER, UnitTypeId.REFINERY
        ]:
            pending = self.already_pending(unit_type)
            if pending > 0:
                cat4[f"planning_{unit_type.name}_count"] = pending

        # Category 5: Enemy Status (visible enemy units & buildings)
        cat5_units: Dict[str, int] = {}
        cat5_buildings: Dict[str, int] = {}
        for enemy_unit in self.enemy_units:
            n = enemy_unit.type_id.name
            cat5_units[n] = cat5_units.get(n, 0) + 1
        for enemy_struct in self.enemy_structures:
            n = enemy_struct.type_id.name
            cat5_buildings[n] = cat5_buildings.get(n, 0) + 1

        # Category 6: Research Progress (upgrades float [0.0, 1.0])
        cat6: Dict[str, float] = {}
        for up in [
            UpgradeId.STIMPACK, UpgradeId.SHIELDWALL, UpgradeId.PUNISHERGRENADES,
            UpgradeId.TERRANINFANTRYWEAPONSLEVEL1, UpgradeId.TERRANINFANTRYARMORSLEVEL1,
            UpgradeId.TERRANVEHICLEWEAPONSLEVEL1, UpgradeId.TERRANSHIPWEAPONSLEVEL1
        ]:
            if up in self.state.upgrades:
                cat6[up.name] = 1.0
            else:
                cat6[up.name] = 0.0

        self.latest_observations = {
            "step": iteration,
            "resources": cat1,
            "units": cat2,
            "buildings": cat3,
            "in_process": cat4,
            "enemy_status": {"units": cat5_units, "buildings": cat5_buildings},
            "research_progress": cat6,
        }

    def _track_evaluation_metrics(self, iteration: int):
        """
        Tracks data for computing the 5 TextStarCraft II metrics until reaching 200/200 supply.
        """
        if self.supply_used >= 198 and not self.reached_max_supply:
            self.reached_max_supply = True
            self.max_supply_step = iteration

        # Only accumulate metrics until max supply (as defined in formulas 5, 6, 7)
        if not self.reached_max_supply:
            if self.supply_left == 0 and self.supply_cap < 200:
                self.supply_blocked_steps += 1
            if self.supply_cap > 0:
                self.apu_history.append(self.supply_used / self.supply_cap)

            # Accumulate mineral & gas spent via manual delta tracking
            minerals_spent_this_step = max(0, self.prev_minerals - self.minerals)
            gas_spent_this_step = max(0, self.prev_vespene - self.vespene)
            self.total_minerals_spent += minerals_spent_this_step
            self.total_gas_spent += gas_spent_this_step
            self.prev_minerals = self.minerals
            self.prev_vespene = self.vespene

    # =========================================================================
    #   TERRAN ACTION DICTIONARY - LAYER 2 MACRO EXECUTION
    # =========================================================================
    async def execute_economy_actions(self, iteration: int):
        """
        Economy actions: TRAIN SCV, BUILD SUPPLY DEPOT, BUILD REFINERY,
        BUILD COMMAND CENTER, UPGRADE TO ORBITAL COMMAND, MULE / SCAN.
        """
        # 1. TRAIN SCV (governed by Parameter 0: worker saturation target)
        max_workers = self.townhalls.amount * self.target_workers_per_base
        for cc in self.townhalls.idle:
            if self.workers.amount < max_workers and self.can_afford(UnitTypeId.SCV):
                cc.train(UnitTypeId.SCV)

        # 2. BUILD SUPPLY DEPOT (governed by Parameter 3: supply buffer & Parameter 21: wall-off)
        if (
            self.supply_left <= self.supply_buffer_threshold
            and self.supply_cap < 200
            and not self.already_pending(UnitTypeId.SUPPLYDEPOT)
            and self.can_afford(UnitTypeId.SUPPLYDEPOT)
        ):
            if self.wall_off_enabled and self.main_base_ramp:
                depot_positions = list(self.main_base_ramp.corner_depots)
                for pos in depot_positions:
                    if not self.structures.closer_than(1.0, pos):
                        await self.build(UnitTypeId.SUPPLYDEPOT, near=pos)
                        break
            elif self.townhalls:
                pos = self.townhalls.first.position.towards(self.game_info.map_center, 6)
                await self.build(UnitTypeId.SUPPLYDEPOT, near=pos)

        # Lower completed supply depots
        for depot in self.structures(UnitTypeId.SUPPLYDEPOT).ready:
            depot(AbilityId.MORPH_SUPPLYDEPOT_LOWER)

        # 3. BUILD REFINERY (governed by Parameter 1: gas timing)
        if self.supply_used >= self.gas_timing_supply:
            for cc in self.townhalls.ready:
                for geyser in self.vespene_geyser.closer_than(10, cc):
                    if (
                        not self.gas_buildings.closer_than(1, geyser)
                        and self.can_afford(UnitTypeId.REFINERY)
                        and self.already_pending(UnitTypeId.REFINERY) < self.townhalls.amount
                    ):
                        await self.build(UnitTypeId.REFINERY, geyser)

        # 4. BUILD COMMAND CENTER (governed by Parameter 2: expansion timing)
        if (
            self.supply_used >= self.expansion_timing_supply
            and self.minerals > 400
            and self.townhalls.amount < 3
            and not self.already_pending(UnitTypeId.COMMANDCENTER)
            and self.structures(UnitTypeId.BARRACKS).ready.amount >= 1
        ):
            await self.expand_now()

        # 5. UPGRADE TO ORBITAL COMMAND & MULE/SCAN (governed by Parameter 4 & 20)
        for cc in self.townhalls(UnitTypeId.COMMANDCENTER).ready.idle:
            if self.orbital_priority > 0.5 and self.minerals >= 150:
                cc(AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND)

        for oc in self.townhalls(UnitTypeId.ORBITALCOMMAND).ready:
            if oc.energy >= 50:
                if self.orbital_priority > 0.7 and random.random() < self.scan_ratio and self.enemy_start_locations:
                    oc(AbilityId.SCANNERSWEEP_SCAN, self.enemy_start_locations[0])
                else:
                    mfs = self.mineral_field.closer_than(10, oc)
                    if mfs:
                        oc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, mfs.closest_to(oc))

    async def execute_infrastructure_actions(self, iteration: int):
        """
        Infrastructure actions: BUILD BARRACKS, BUILD FACTORY, BUILD STARPORT,
        BUILD ENGINEERING BAY, BUILD BUNKER, BUILD MISSILE TURRET, BUILD TECH LAB/REACTOR.
        """
        barracks_count = self.structures(UnitTypeId.BARRACKS).amount + self.already_pending(UnitTypeId.BARRACKS)

        # 1. BUILD BARRACKS (governed by Parameter 5: barracks count)
        if barracks_count < self.target_barracks_count and self.supply_used >= 14 and self.can_afford(UnitTypeId.BARRACKS):
            if self.wall_off_enabled and self.main_base_ramp and not self.structures(UnitTypeId.BARRACKS):
                ramp_pos = self.main_base_ramp.barracks_correct_placement
                if ramp_pos:
                    await self.build(UnitTypeId.BARRACKS, near=ramp_pos)
            elif self.townhalls:
                pos = self.townhalls.first.position.towards(self.game_info.map_center, 8)
                await self.build(UnitTypeId.BARRACKS, near=pos)

        # 2. BUILD FACTORY (governed by Parameter 6: factory preference)
        if (
            self.factory_preference > 0.3
            and self.structures(UnitTypeId.BARRACKS).ready.amount >= 1
            and not self.structures(UnitTypeId.FACTORY)
            and not self.already_pending(UnitTypeId.FACTORY)
            and self.can_afford(UnitTypeId.FACTORY)
        ):
            if self.townhalls:
                pos = self.townhalls.first.position.towards(self.game_info.map_center, 12)
                await self.build(UnitTypeId.FACTORY, near=pos)

        # 3. BUILD STARPORT (governed by Parameter 12: starport preference)
        if (
            self.starport_preference > 0.3
            and self.structures(UnitTypeId.FACTORY).ready.amount >= 1
            and not self.structures(UnitTypeId.STARPORT)
            and not self.already_pending(UnitTypeId.STARPORT)
            and self.can_afford(UnitTypeId.STARPORT)
        ):
            if self.townhalls:
                pos = self.townhalls.first.position.towards(self.game_info.map_center, 14)
                await self.build(UnitTypeId.STARPORT, near=pos)

        # 4. BUILD ENGINEERING BAY & DEFENSIVE STRUCTURES (governed by Parameter 18)
        defenses_count = (
            self.structures(UnitTypeId.BUNKER).amount
            + self.structures(UnitTypeId.MISSILETURRET).amount
            + self.already_pending(UnitTypeId.BUNKER)
            + self.already_pending(UnitTypeId.MISSILETURRET)
        )
        if defenses_count < self.defensive_structure_target and self.townhalls:
            if not self.structures(UnitTypeId.ENGINEERINGBAY) and not self.already_pending(UnitTypeId.ENGINEERINGBAY):
                if self.can_afford(UnitTypeId.ENGINEERINGBAY):
                    pos = self.townhalls.first.position.towards(self.game_info.map_center, 10)
                    await self.build(UnitTypeId.ENGINEERINGBAY, near=pos)
            elif self.structures(UnitTypeId.ENGINEERINGBAY).ready and self.can_afford(UnitTypeId.MISSILETURRET):
                pos = self.townhalls.first.position.towards(self.game_info.map_center, 7)
                await self.build(UnitTypeId.MISSILETURRET, near=pos)

        # 5. BUILD TECH LAB OR REACTOR (governed by Parameter 7: addon ratio)
        for bldg in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if not bldg.has_add_on:
                if random.random() < self.tech_lab_ratio and self.can_afford(UnitTypeId.BARRACKSTECHLAB):
                    bldg.build(UnitTypeId.BARRACKSTECHLAB)
                elif self.can_afford(UnitTypeId.BARRACKSREACTOR):
                    bldg.build(UnitTypeId.BARRACKSREACTOR)

        for fct in self.structures(UnitTypeId.FACTORY).ready.idle:
            if not fct.has_add_on and random.random() < self.tech_lab_ratio and self.can_afford(UnitTypeId.FACTORYTECHLAB):
                fct.build(UnitTypeId.FACTORYTECHLAB)

    async def execute_research_actions(self, iteration: int):
        """
        Research actions: RESEARCH STIMPACK, COMBAT SHIELD, INFANTRY WEAPONS/ARMOR.
        Governed by Parameter 13: Upgrade Priority.
        """
        for lab in self.structures(UnitTypeId.BARRACKSTECHLAB).ready.idle:
            if UpgradeId.STIMPACK not in self.state.upgrades and self.can_afford(UpgradeId.STIMPACK):
                lab.research(UpgradeId.STIMPACK)
            elif UpgradeId.SHIELDWALL not in self.state.upgrades and self.can_afford(UpgradeId.SHIELDWALL):
                lab.research(UpgradeId.SHIELDWALL)

        for ebay in self.structures(UnitTypeId.ENGINEERINGBAY).ready.idle:
            if self.upgrade_priority < 0.5:
                if UpgradeId.TERRANINFANTRYWEAPONSLEVEL1 not in self.state.upgrades and self.can_afford(UpgradeId.TERRANINFANTRYWEAPONSLEVEL1):
                    ebay.research(UpgradeId.TERRANINFANTRYWEAPONSLEVEL1)
            else:
                if UpgradeId.TERRANINFANTRYARMORSLEVEL1 not in self.state.upgrades and self.can_afford(UpgradeId.TERRANINFANTRYARMORSLEVEL1):
                    ebay.research(UpgradeId.TERRANINFANTRYARMORSLEVEL1)

    async def execute_unit_training_actions(self, iteration: int):
        """
        Unit training actions: TRAIN MARINE, MARAUDER, REAPER, SIEGE TANK, MEDIVAC.
        Governed by Parameters 8, 9, 10, 11, 19, 22.
        """
        # Dynamic tech switch adaptation (Parameter 19)
        marine_pref = self.marine_ratio
        tank_pref = self.siege_tank_ratio

        if self.tech_switch_willingness > 0.3 and len(self.enemy_units) > 0:
            armored_enemies = sum(1 for eu in self.enemy_units if eu.is_armored)
            if armored_enemies > 3:
                marine_pref = max(0.2, marine_pref - 0.25)
                tank_pref = min(1.0, tank_pref + 0.3)

        # Train from Barracks
        for rax in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if self.time < 300 and random.random() < self.reaper_ratio and self.can_afford(UnitTypeId.REAPER):
                rax.train(UnitTypeId.REAPER)
            elif rax.has_add_on and rax.add_on_tag in [tl.tag for tl in self.structures(UnitTypeId.BARRACKSTECHLAB)]:
                if random.random() < self.marauder_ratio and self.can_afford(UnitTypeId.MARAUDER):
                    rax.train(UnitTypeId.MARAUDER)
                elif self.can_afford(UnitTypeId.MARINE):
                    rax.train(UnitTypeId.MARINE)
            else:
                if random.random() < marine_pref and self.can_afford(UnitTypeId.MARINE):
                    rax.train(UnitTypeId.MARINE)

        # Train from Factory
        for fct in self.structures(UnitTypeId.FACTORY).ready.idle:
            if random.random() < tank_pref and self.can_afford(UnitTypeId.SIEGETANK):
                fct.train(UnitTypeId.SIEGETANK)

        # Train from Starport
        for sp in self.structures(UnitTypeId.STARPORT).ready.idle:
            if random.random() < self.medivac_ratio and self.can_afford(UnitTypeId.MEDIVAC):
                sp.train(UnitTypeId.MEDIVAC)

    async def execute_strategic_actions(self, iteration: int):
        """
        Strategic actions: ATTACK WITH ARMY, RETREAT TO BASE, HARASS ENEMY.
        Governed by Parameters 14, 15, 16, 17.
        """
        army_supply = self.supply_used - self.workers.amount
        marines = self.units(UnitTypeId.MARINE)
        marauders = self.units(UnitTypeId.MARAUDER)
        tanks = self.units(UnitTypeId.SIEGETANK)
        medivacs = self.units(UnitTypeId.MEDIVAC)

        # Trigger Attack when crossing effective threshold (with cooldown after retreat)
        if army_supply >= self.effective_attack_threshold and not self.attack_triggered:
            if iteration - self.last_retreat_step > 300:  # 300 step cooldown
                self.attack_triggered = True
                self.pre_engagement_army_size = army_supply

        # Trigger Retreat if losses exceed Parameter 15 threshold
        if self.attack_triggered:
            loss_ratio = 1.0 - (army_supply / max(1, self.pre_engagement_army_size))
            if loss_ratio >= self.retreat_loss_threshold:
                self.attack_triggered = False
                self.last_retreat_step = iteration

        # Periodic Harassment (Parameter 17)
        if iteration - self.last_harass_step >= self.harass_interval_steps:
            reapers = self.units(UnitTypeId.REAPER).idle
            if reapers and self.enemy_start_locations:
                for r in reapers:
                    r.attack(self.enemy_start_locations[0])
                self.last_harass_step = iteration

    # =========================================================================
    #   LAYER 3 - MICRO COMBAT SCRIPT
    # =========================================================================
    async def execute_micro_script(self, iteration: int):
        """
        Automated combat movements, focus firing, and rally positioning.
        """
        combat_units = (
            self.units(UnitTypeId.MARINE)
            | self.units(UnitTypeId.MARAUDER)
            | self.units(UnitTypeId.REAPER)
            | self.units(UnitTypeId.SIEGETANK)
        )
        medivacs = self.units(UnitTypeId.MEDIVAC)

        if self.attack_triggered and self.enemy_start_locations:
            target = self.enemy_start_locations[0]
            # Focus fire on visible enemy units if within range
            for unit in combat_units.idle:
                nearby_enemies = self.enemy_units.closer_than(8.0, unit)
                if nearby_enemies:
                    unit.attack(nearby_enemies.first)
                else:
                    unit.attack(target)
            for m in medivacs.idle:
                m.move(target)
        else:
            # Rally defensively at base ramp or town hall
            if self.townhalls:
                rally = self.townhalls.first.position.towards(self.game_info.map_center, 8)
                for unit in combat_units:  # All units, not just idle, to pull back attackers
                    unit.move(rally)

    # =========================================================================
    #   5 TEXTSTARCRAFT II EVALUATION METRICS CALCULATOR
    # =========================================================================
    def get_evaluation_metrics(self, winrate_score: float = 0.0) -> Dict[str, float]:
        """
        Returns the 5 metrics as defined in TextStarCraft II Table 1 & Appendix B.2:
          1. WINRATE: Percentage / float score of games won
          2. PBR (Population Block Ratio): time_at_pop_cap / game_duration
          3. RUR (Resource Utilization Ratio): total_resources_spent / game_duration
          4. APU (Average Population Utilization): mean(supply_used / supply_cap)
          5. TR (Technology Rate): completed_technologies / total_available
        """
        state = getattr(self, "state", None)
        game_loop = getattr(state, "game_loop", 1200) if state else 1200
        duration = max(1.0, float(self.max_supply_step if self.reached_max_supply else game_loop))
        duration_seconds = duration / 22.4  # SC2 normal speed ticks per second

        pbr = self.supply_blocked_steps / max(1.0, duration)
        rur = (self.total_minerals_spent + self.total_gas_spent) / max(1.0, duration_seconds)
        apu = sum(self.apu_history) / max(1, len(self.apu_history)) if self.apu_history else 0.5

        # Compute completed technologies
        upgrades_count = len(getattr(state, "upgrades", [])) if state else 3
        tech_struct_count = 0
        structures = getattr(self, "structures", None)
        if structures and hasattr(structures, "ready"):
            tech_struct_count = sum(
                1 for s in structures.ready
                if s.type_id in [
                    UnitTypeId.FACTORY, UnitTypeId.STARPORT, UnitTypeId.ENGINEERINGBAY,
                    UnitTypeId.ARMORY, UnitTypeId.FUSIONCORE, UnitTypeId.GHOSTACADEMY
                ]
            )
        else:
            tech_struct_count = 2

        completed_tech = upgrades_count + tech_struct_count
        tr = min(1.0, completed_tech / float(self.total_available_technologies))

        return {
            "WINRATE": round(float(winrate_score), 4),
            "PBR": round(float(pbr), 4),
            "RUR": round(float(rur), 2),
            "APU": round(float(apu), 4),
            "TR": round(float(tr), 4),
        }
