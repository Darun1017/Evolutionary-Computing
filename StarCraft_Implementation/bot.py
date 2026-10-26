"""
bot.py — API Binding Layer for CCE-TDCCEA StarCraft II implementation.

EvolvedSC2Bot(BotAI) translates a continuous 35D weight vector into hard
StarCraft II API actions.  The class is race-configurable (Terran / Protoss /
Zerg) and dispatches on_step() to race-specific logic after decoding the
genome into concrete thresholds once in on_start().

Gene → Threshold mapping overview
──────────────────────────────────
  Pop A (Macro/Economy, w[0–9])
    w[0] worker_ratio          → worker training eagerness
    w[1] max_workers           → 8 + round(w*14) ∈ [8, 22]
    w[2] expand_threshold      → 400 + round(w*500) minerals ∈ [400, 900]
    w[3] supply_buffer         → 2 + round(w*6) ∈ [2, 8]
    w[4] prod_building_timing  → 10 + round(w*10) supply ∈ [10, 20]
    w[5] second_prod_building  → 18 + round(w*16) supply ∈ [18, 34]
    w[6] refinery_timing       → 8  + round(w*8)  workers ∈ [8, 16]
    w[7] economy_pressure      → weight (higher = prefer eco over army)
    w[8] vespene_ratio         → gas priority (0=minerals, 1=gas)
    w[9] expansion_aggression  → 0=passive, 1=greedy expander

  Pop B (Micro/Combat, w[10–24])
    w[10] aggression           → global aggression weight
    w[11] attack_threshold     → 5 + round(w*25) army supply to push ∈ [5, 30]
    w[12] defend_radius        → 5 + w*20 game-units from base ∈ [5, 25]
    w[13] target_priority      → <0.5=workers, >=0.5=military targets
    w[14] defense_trigger      → 5 + w*20 enemy distance ∈ [5, 25]
    w[15] counter_rush         → >0.5 → workers fight early rushes
    w[16] group_attack         → >0.5 → wait to consolidate army
    w[17] focus_fire           → >0.5 → concentrate on one target
    w[18] kite_distance        → 0..4 game-unit retreat per cycle
    w[19] retreat_threshold    → 0.05+w*0.35 HP% to retreat ∈ [5%, 40%]
    w[20] ability_threshold    → HP%/energy% to trigger special abilities
    w[21] assault_timing       → game-time fraction for first timing push
    w[22] harassment_weight    → >0.5 → enable worker harassment / drops
    w[23] unit_composition_a   → preference for primary unit type
    w[24] unit_composition_b   → preference for secondary unit type

  Pop C (Tech/Build-Order, w[25–34])
    w[25] tech_switch          → >0.5 → willing to pivot tech
    w[26] advanced_build       → 20 + round(w*30) supply ∈ [20, 50]
    w[27] upgrade_priority     → 15 + round(w*25) supply ∈ [15, 40]
    w[28] tier3_timing         → 30 + round(w*30) supply ∈ [30, 60]
    w[29] drop_timing          → 25 + round(w*35) supply ∈ [25, 60]
    w[30] detection_timing     → 20 + round(w*30) supply ∈ [20, 50]
    w[31] air_transition       → 0=ground, 1=air unit preference
    w[32] support_unit_timing  → 20 + round(w*30) supply ∈ [20, 50]
    w[33] ability_usage        → frequency / aggression of special abilities
    w[34] defensive_structure  → 8  + round(w*12) supply ∈ [8, 20]
"""

from __future__ import annotations

import random
from typing import List

from sc2.bot_ai import BotAI
from sc2.data import Race, Result
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.ids.ability_id import AbilityId

from evaluator import MetricsTracker
from config import FITNESS_WEIGHTS, DIMENSIONS

# ── Race-specific unit / building type maps ──────────────────────────────────
_RACE_WORKER = {
    Race.Terran:  UnitTypeId.SCV,
    Race.Protoss: UnitTypeId.PROBE,
    Race.Zerg:    UnitTypeId.DRONE,
}

_RACE_SUPPLY = {
    Race.Terran:  UnitTypeId.SUPPLYDEPOT,
    Race.Protoss: UnitTypeId.PYLON,
    Race.Zerg:    UnitTypeId.OVERLORD,
}

_RACE_GAS = {
    Race.Terran:  UnitTypeId.REFINERY,
    Race.Protoss: UnitTypeId.ASSIMILATOR,
    Race.Zerg:    UnitTypeId.EXTRACTOR,
}

# Primary production building per race
_RACE_PROD_A = {
    Race.Terran:  UnitTypeId.BARRACKS,
    Race.Protoss: UnitTypeId.GATEWAY,
    Race.Zerg:    UnitTypeId.SPAWNINGPOOL,
}

# Primary unit A / B per race
_RACE_UNIT_A = {
    Race.Terran:  UnitTypeId.MARINE,
    Race.Protoss: UnitTypeId.ZEALOT,
    Race.Zerg:    UnitTypeId.ZERGLING,
}

_RACE_UNIT_B = {
    Race.Terran:  UnitTypeId.MARAUDER,
    Race.Protoss: UnitTypeId.STALKER,
    Race.Zerg:    UnitTypeId.ROACH,
}

# Tech building (Factory / Robotics / Roach Warren)
_RACE_TECH = {
    Race.Terran:  UnitTypeId.FACTORY,
    Race.Protoss: UnitTypeId.ROBOTICSFACILITY,
    Race.Zerg:    UnitTypeId.ROACHWARREN,
}

# Support / heal unit (Medivac / Warp Prism / Overlord)
_RACE_SUPPORT = {
    Race.Terran:  UnitTypeId.MEDIVAC,
    Race.Protoss: UnitTypeId.WARPPRISM,
    Race.Zerg:    UnitTypeId.OVERLORD,
}

# Defensive structure
_RACE_DEFENSE = {
    Race.Terran:  UnitTypeId.BUNKER,
    Race.Protoss: UnitTypeId.PHOTONCANNON,
    Race.Zerg:    UnitTypeId.SPINECRAWLER,
}


class EvolvedSC2Bot(BotAI):
    """
    StarCraft II bot driven by a 35D continuous weight vector.

    The genome is decoded into hard thresholds once in `on_start()`.
    Race-specific action methods share common abstract logic; only the
    unit/building type IDs differ.
    """

    def __init__(self, weights: List[float], race: Race) -> None:
        super().__init__()
        assert len(weights) == DIMENSIONS, (
            f"Expected {DIMENSIONS}D genome; got {len(weights)}D"
        )
        self.weights     = [max(0.0, min(1.0, float(w))) for w in weights]
        self.bot_race    = race
        self.tracker     = MetricsTracker()
        self.fitness_result: dict | None = None
        self._t: dict = {}   # decoded thresholds

    # ─────────────────────────────────────────────────────────────────────────
    #  LIFECYCLE HOOKS
    # ─────────────────────────────────────────────────────────────────────────

    async def on_start(self) -> None:
        self._decode_thresholds()

    async def on_step(self, iteration: int) -> None:
        # Sample metrics for APU / PBR / RUR
        self.tracker.update(self)

        # Race-specific step logic
        if self.bot_race == Race.Terran:
            await self._step_terran(iteration)
        elif self.bot_race == Race.Protoss:
            await self._step_protoss(iteration)
        elif self.bot_race == Race.Zerg:
            await self._step_zerg(iteration)

    async def on_end(self, game_result: Result) -> None:
        won = game_result == Result.Victory
        self.fitness_result = self.tracker.compute_fitness(won, FITNESS_WEIGHTS)

    # ─────────────────────────────────────────────────────────────────────────
    #  GENOME DECODER
    # ─────────────────────────────────────────────────────────────────────────

    def _decode_thresholds(self) -> None:
        """Map continuous weights w_j ∈ [0,1] to discrete/real thresholds."""
        w = self.weights
        t = self._t

        # ── Pop A: Macro / Economy ────────────────────────────────────────────
        t["max_workers"]          = int(8 + w[1] * 14)           # [8, 22]
        t["expand_minerals"]      = int(400 + w[2] * 500)        # [400, 900]
        t["supply_buffer"]        = max(2, int(2 + w[3] * 6))    # [2, 8]
        t["prod_at_supply"]       = int(10 + w[4] * 10)          # [10, 20]
        t["second_prod_supply"]   = int(18 + w[5] * 16)          # [18, 34]
        t["refinery_at_workers"]  = int(8 + w[6] * 8)            # [8, 16]
        t["eco_weight"]           = w[7]
        t["gas_priority"]         = w[8]
        t["expand_score"]         = w[9]

        # ── Pop B: Micro / Combat ─────────────────────────────────────────────
        t["aggression"]           = w[10]
        t["attack_supply"]        = int(5 + w[11] * 25)          # [5, 30]
        t["defend_radius"]        = 5.0 + w[12] * 20.0           # [5, 25]
        t["target_workers"]       = w[13] < 0.5                  # True=target workers
        t["defense_dist"]         = 5.0 + w[14] * 20.0           # [5, 25]
        t["worker_fight"]         = w[15] >= 0.5
        t["group_attack"]         = w[16] >= 0.5
        t["focus_fire"]           = w[17] >= 0.5
        t["kite_dist"]            = w[18] * 4.0                   # [0, 4]
        t["retreat_hp"]           = 0.05 + w[19] * 0.35          # [5%, 40%]
        t["ability_hp"]           = 0.3 + w[20] * 0.5            # [30%, 80%]
        t["assault_time_frac"]    = w[21]                         # [0, 1]
        t["harassment"]           = w[22] >= 0.5
        t["comp_a"]               = w[23]
        t["comp_b"]               = w[24]
        t["prefer_a"]             = w[23] >= w[24]               # True → prefer unit A

        # ── Pop C: Tech / Build-Order ─────────────────────────────────────────
        t["tech_switch"]          = w[25] >= 0.5
        t["adv_build_supply"]     = int(20 + w[26] * 30)         # [20, 50]
        t["upgrade_supply"]       = int(15 + w[27] * 25)         # [15, 40]
        t["tier3_supply"]         = int(30 + w[28] * 30)         # [30, 60]
        t["drop_supply"]          = int(25 + w[29] * 35)         # [25, 60]
        t["detection_supply"]     = int(20 + w[30] * 30)         # [20, 50]
        t["air_weight"]           = w[31]
        t["support_supply"]       = int(20 + w[32] * 30)         # [20, 50]
        t["ability_freq"]         = w[33]
        t["defense_supply"]       = int(8 + w[34] * 12)          # [8, 20]

    # ─────────────────────────────────────────────────────────────────────────
    #  TERRAN STEP LOGIC
    # ─────────────────────────────────────────────────────────────────────────

    async def _step_terran(self, iteration: int) -> None:
        t = self._t
        await self.distribute_workers()

        # 1. Supply Depot
        if (self.supply_left < t["supply_buffer"] and
                self.supply_cap < 200 and
                not self.already_pending(UnitTypeId.SUPPLYDEPOT) and
                self.can_afford(UnitTypeId.SUPPLYDEPOT)):
            await self.build(
                UnitTypeId.SUPPLYDEPOT,
                near=self.start_location.towards(self.game_info.map_center, 6),
            )

        # 2. First Barracks
        if (self.supply_used >= t["prod_at_supply"] and
                not self.structures(UnitTypeId.BARRACKS) and
                not self.already_pending(UnitTypeId.BARRACKS) and
                self.can_afford(UnitTypeId.BARRACKS)):
            await self.build(
                UnitTypeId.BARRACKS,
                near=self.start_location.towards(self.game_info.map_center, 9),
            )

        # 3. Second Barracks
        if (self.supply_used >= t["second_prod_supply"] and
                self.structures(UnitTypeId.BARRACKS).ready and
                self.structures(UnitTypeId.BARRACKS).amount < 2 and
                not self.already_pending(UnitTypeId.BARRACKS) and
                self.can_afford(UnitTypeId.BARRACKS)):
            await self.build(
                UnitTypeId.BARRACKS,
                near=self.start_location.towards(self.game_info.map_center, 12),
            )

        # 4. Refinery
        await self._build_gas(UnitTypeId.REFINERY)

        # 5. Factory (tech gene)
        if (self.supply_used >= t["adv_build_supply"] and
                self.structures(UnitTypeId.BARRACKS).ready and
                not self.structures(UnitTypeId.FACTORY) and
                not self.already_pending(UnitTypeId.FACTORY) and
                self.can_afford(UnitTypeId.FACTORY)):
            await self.build(
                UnitTypeId.FACTORY,
                near=self.start_location.towards(self.game_info.map_center, 15),
            )

        # 6. Bunker (defensive gene)
        if (self.supply_used >= t["defense_supply"] and
                self.structures(UnitTypeId.BARRACKS).ready and
                not self.structures(UnitTypeId.BUNKER) and
                not self.already_pending(UnitTypeId.BUNKER) and
                self.can_afford(UnitTypeId.BUNKER)):
            await self.build(
                UnitTypeId.BUNKER,
                near=self.start_location.towards(self.game_info.map_center, 4),
            )

        # 7. Orbital Command upgrade
        for cc in self.structures(UnitTypeId.COMMANDCENTER).ready.idle:
            if self.can_afford(UnitTypeId.ORBITALCOMMAND):
                cc(AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND)

        # 8. MULE (call-down)
        for oc in self.structures(UnitTypeId.ORBITALCOMMAND).filter(lambda o: o.energy >= 50):
            mf = self.mineral_field.closest_to(oc.position) if self.mineral_field else None
            if mf:
                oc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, mf)

        # 9. Train workers
        if self.workers.amount < t["max_workers"]:
            for cc in self.townhalls.ready.idle:
                if self.can_afford(UnitTypeId.SCV) and self.workers.amount < t["max_workers"]:
                    cc.train(UnitTypeId.SCV)

        # 10. Train army from barracks
        for rax in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if t["prefer_a"]:
                if self.can_afford(UnitTypeId.MARINE):
                    rax.train(UnitTypeId.MARINE)
            else:
                # Marauder needs Tech Lab; fall back to Marine if not ready
                if (self.structures(UnitTypeId.BARRACKSTECHLAB).ready and
                        self.can_afford(UnitTypeId.MARAUDER)):
                    rax.train(UnitTypeId.MARAUDER)
                elif self.can_afford(UnitTypeId.MARINE):
                    rax.train(UnitTypeId.MARINE)

        # 11. Stim research (ability gene)
        if (self.structures(UnitTypeId.BARRACKSTECHLAB).ready and
                not self.already_pending_upgrade(UpgradeId.STIMPACK) and
                self.supply_used >= t["upgrade_supply"] and
                self.can_afford(UpgradeId.STIMPACK)):
            self.structures(UnitTypeId.BARRACKSTECHLAB).ready.first.research(
                UpgradeId.STIMPACK
            )

        # 12. Stim active units
        if t["ability_freq"] > 0.5:
            for marine in self.units(UnitTypeId.MARINE).filter(
                lambda u: u.health_percentage >= t["ability_hp"] and not u.has_buff(
                    # STIMPACK buff id
                    # Using raw ability instead
                    AbilityId.EFFECT_STIM_MARINE
                )
            ):
                if self.enemy_units.closer_than(t["defend_radius"], marine.position):
                    marine(AbilityId.EFFECT_STIM_MARINE)

        # 13. Combat logic
        await self._combat_logic()

    # ─────────────────────────────────────────────────────────────────────────
    #  PROTOSS STEP LOGIC
    # ─────────────────────────────────────────────────────────────────────────

    async def _step_protoss(self, iteration: int) -> None:
        t = self._t
        await self.distribute_workers()

        # 1. Pylon
        if (self.supply_left < t["supply_buffer"] and
                self.supply_cap < 200 and
                not self.already_pending(UnitTypeId.PYLON) and
                self.can_afford(UnitTypeId.PYLON)):
            await self.build(
                UnitTypeId.PYLON,
                near=self.start_location.towards(self.game_info.map_center, 6),
            )

        pylons_ready = self.structures(UnitTypeId.PYLON).ready

        # 2. Gateway
        if (self.supply_used >= t["prod_at_supply"] and
                pylons_ready and
                not self.structures(UnitTypeId.GATEWAY) and
                not self.already_pending(UnitTypeId.GATEWAY) and
                self.can_afford(UnitTypeId.GATEWAY)):
            pylon = pylons_ready.random
            await self.build(UnitTypeId.GATEWAY, near=pylon)

        # 3. Second Gateway
        if (self.supply_used >= t["second_prod_supply"] and
                pylons_ready and
                self.structures(UnitTypeId.GATEWAY).ready and
                self.structures(UnitTypeId.GATEWAY).amount < 2 and
                not self.already_pending(UnitTypeId.GATEWAY) and
                self.can_afford(UnitTypeId.GATEWAY)):
            pylon = pylons_ready.random
            await self.build(UnitTypeId.GATEWAY, near=pylon)

        # 4. Cybernetics Core
        if (self.structures(UnitTypeId.GATEWAY).ready and
                not self.structures(UnitTypeId.CYBERNETICSCORE) and
                not self.already_pending(UnitTypeId.CYBERNETICSCORE) and
                pylons_ready and
                self.can_afford(UnitTypeId.CYBERNETICSCORE)):
            await self.build(UnitTypeId.CYBERNETICSCORE, near=pylons_ready.random)

        # 5. Assimilator
        await self._build_gas(UnitTypeId.ASSIMILATOR)

        # 6. Robotics Facility (tech gene)
        if (self.supply_used >= t["adv_build_supply"] and
                self.structures(UnitTypeId.CYBERNETICSCORE).ready and
                not self.structures(UnitTypeId.ROBOTICSFACILITY) and
                not self.already_pending(UnitTypeId.ROBOTICSFACILITY) and
                pylons_ready and
                self.can_afford(UnitTypeId.ROBOTICSFACILITY)):
            await self.build(UnitTypeId.ROBOTICSFACILITY, near=pylons_ready.random)

        # 7. Train workers
        if self.workers.amount < t["max_workers"]:
            for nexus in self.townhalls.ready.idle:
                if self.can_afford(UnitTypeId.PROBE) and self.workers.amount < t["max_workers"]:
                    nexus.train(UnitTypeId.PROBE)

        # 8. Train army from gateways
        for gw in self.structures(UnitTypeId.GATEWAY).ready.idle:
            if t["prefer_a"]:
                if self.can_afford(UnitTypeId.ZEALOT):
                    gw.train(UnitTypeId.ZEALOT)
            else:
                if (self.structures(UnitTypeId.CYBERNETICSCORE).ready and
                        self.can_afford(UnitTypeId.STALKER)):
                    gw.train(UnitTypeId.STALKER)
                elif self.can_afford(UnitTypeId.ZEALOT):
                    gw.train(UnitTypeId.ZEALOT)

        # 9. Train Immortals from Robotics
        if t["tech_switch"] and t["adv_build_supply"] <= self.supply_used:
            for robo in self.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle:
                if self.can_afford(UnitTypeId.IMMORTAL):
                    robo.train(UnitTypeId.IMMORTAL)

        await self._combat_logic()

    # ─────────────────────────────────────────────────────────────────────────
    #  ZERG STEP LOGIC
    # ─────────────────────────────────────────────────────────────────────────

    async def _step_zerg(self, iteration: int) -> None:
        t = self._t
        await self.distribute_workers()

        # 1. Overlord if supply needed
        if (self.supply_left < t["supply_buffer"] and
                self.supply_cap < 200 and
                not self.already_pending(UnitTypeId.OVERLORD) and
                self.larva and
                self.can_afford(UnitTypeId.OVERLORD)):
            self.larva.random.train(UnitTypeId.OVERLORD)

        # 2. Train drones
        if self.workers.amount < t["max_workers"] and self.larva:
            if self.can_afford(UnitTypeId.DRONE):
                self.larva.random.train(UnitTypeId.DRONE)

        # 3. Spawning Pool
        if (self.supply_used >= t["prod_at_supply"] and
                not self.structures(UnitTypeId.SPAWNINGPOOL) and
                not self.already_pending(UnitTypeId.SPAWNINGPOOL) and
                self.can_afford(UnitTypeId.SPAWNINGPOOL)):
            await self.build(
                UnitTypeId.SPAWNINGPOOL,
                near=self.start_location.towards(self.game_info.map_center, 5),
            )

        # 4. Roach Warren (tech gene)
        if (self.supply_used >= t["adv_build_supply"] and
                self.structures(UnitTypeId.SPAWNINGPOOL).ready and
                not self.structures(UnitTypeId.ROACHWARREN) and
                not self.already_pending(UnitTypeId.ROACHWARREN) and
                self.can_afford(UnitTypeId.ROACHWARREN)):
            await self.build(
                UnitTypeId.ROACHWARREN,
                near=self.start_location.towards(self.game_info.map_center, 7),
            )

        # 5. Extractor
        await self._build_gas(UnitTypeId.EXTRACTOR)

        # 6. Train army from larvae
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready and self.larva:
            if not t["prefer_a"] and self.structures(UnitTypeId.ROACHWARREN).ready:
                if self.can_afford(UnitTypeId.ROACH):
                    self.larva.random.train(UnitTypeId.ROACH)
                elif self.can_afford(UnitTypeId.ZERGLING):
                    self.larva.random.train(UnitTypeId.ZERGLING)
            else:
                if self.can_afford(UnitTypeId.ZERGLING):
                    self.larva.random.train(UnitTypeId.ZERGLING)

        # 7. Queen per hatchery
        for hatch in self.townhalls.ready:
            if (not self.units(UnitTypeId.QUEEN).closer_than(8, hatch) and
                    self.can_afford(UnitTypeId.QUEEN) and
                    self.structures(UnitTypeId.SPAWNINGPOOL).ready):
                hatch.train(UnitTypeId.QUEEN)

        # 8. Queen inject
        for queen in self.units(UnitTypeId.QUEEN).idle.filter(lambda q: q.energy >= 25):
            if self.townhalls:
                queen(AbilityId.EFFECT_INJECTLARVA, self.townhalls.closest_to(queen))

        # 9. Spine Crawler (defensive gene)
        if (self.supply_used >= t["defense_supply"] and
                self.structures(UnitTypeId.SPAWNINGPOOL).ready and
                not self.structures(UnitTypeId.SPINECRAWLER) and
                not self.already_pending(UnitTypeId.SPINECRAWLER) and
                self.can_afford(UnitTypeId.SPINECRAWLER)):
            await self.build(
                UnitTypeId.SPINECRAWLER,
                near=self.start_location.towards(self.game_info.map_center, 4),
            )

        await self._combat_logic()

    # ─────────────────────────────────────────────────────────────────────────
    #  SHARED HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    async def _build_gas(self, gas_type: UnitTypeId) -> None:
        """Build first gas building once worker count threshold is reached."""
        t = self._t
        if (self.workers.amount < t["refinery_at_workers"] or
                self.structures(gas_type) or
                self.already_pending(gas_type) or
                not self.townhalls):
            return
        if not self.can_afford(gas_type):
            return
        geysers = self.vespene_geyser.closer_than(12, self.townhalls.first)
        if geysers:
            await self.build(gas_type, geysers.first)

    async def _combat_logic(self) -> None:
        """
        Shared attack / defend logic driven by Pop A (economy_weight) and
        Pop B (aggression, attack_threshold, defend_radius, focus_fire …).
        """
        t = self._t

        # Army = all non-worker combat-capable units
        army = self.units.filter(
            lambda u: not u.is_worker and u.can_attack and u.is_ready and not u.is_structure
        )
        if not army:
            return

        army_supply = sum(u.supply_cost for u in army)

        # --- RETREAT: low-HP units pull back to base ---
        for unit in army:
            if unit.health_percentage < t["retreat_hp"]:
                unit.move(self.start_location)

        # --- DEFEND: enemies within defend_radius of base ---
        threats = self.enemy_units.closer_than(t["defense_dist"], self.start_location)
        if threats:
            # Workers join fight if gene says so
            fighters = army
            if t["worker_fight"]:
                workers = self.workers.closer_than(t["defense_dist"] + 3, self.start_location)
                fighters = fighters | workers

            for unit in fighters:
                if unit.health_percentage >= t["retreat_hp"]:
                    if t["focus_fire"] and threats:
                        target = threats.closest_to(unit)
                        if t["target_workers"]:
                            worker_threats = threats.filter(lambda u: u.is_worker)
                            if worker_threats:
                                target = worker_threats.closest_to(unit)
                        unit.attack(target)
                    else:
                        unit.attack(threats.center)
            return

        # --- ATTACK: army large enough to push ---
        if army_supply >= t["attack_supply"]:
            if self.enemy_structures:
                target = self.enemy_structures.closest_to(self.start_location).position
            else:
                target = self.enemy_start_locations[0]

            for unit in army:
                if unit.health_percentage >= t["retreat_hp"]:
                    unit.attack(target)
        else:
            # --- RALLY: consolidate near base before pushing ---
            if not t["group_attack"]:
                return
            rally_point = self.start_location.towards(self.game_info.map_center, 5)
            for unit in army:
                if unit.distance_to(rally_point) > 7:
                    unit.move(rally_point)
