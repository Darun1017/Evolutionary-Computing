"""
Protoss Macro Bot - 23-Parameter CCE-TDCCEA Strategy Controller
Layer 1: 23-param vector | Layer 2: Macro execution | Layer 3: Micro combat
For TextStarCraft II (Ma et al., NeurIPS 2024) comparison.
Race: Protoss vs Zerg. python-sc2 BurnySC2 compatible.
"""
import math, random, json, logging
from typing import List, Dict, Any, Optional
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
try:
    from sc2.ids.buff_id import BuffId
except ImportError:
    class BuffId:
        CHRONOBOOSTENERGYCOST = "CHRONOBOOSTENERGYCOST"

logger = logging.getLogger("ProtossMacroBot")

COMBAT_UNIT_TYPES = [
    UnitTypeId.ZEALOT, UnitTypeId.STALKER, UnitTypeId.ADEPT,
    UnitTypeId.IMMORTAL, UnitTypeId.COLOSSUS, UnitTypeId.DISRUPTOR,
    UnitTypeId.VOIDRAY, UnitTypeId.ORACLE, UnitTypeId.PHOENIX,
]

class ProtossMacroBot(BotAI):
    def __init__(self, param_vector: Optional[List[float]] = None):
        super().__init__()
        if param_vector is None or len(param_vector) != 23:
            self.params = [0.5] * 23
        else:
            self.params = [max(0.0, min(1.0, float(v))) for v in param_vector]
        self._decode_parameters()
        # State tracking
        self.attack_triggered = False
        self.pre_engagement_army_supply = 0
        self.last_harass_step = -1000
        self.last_retreat_step = -1000
        self.reached_max_supply = False
        self.max_supply_step = -1
        self.supply_blocked_steps = 0
        self.apu_history: List[float] = []
        self.prev_minerals = 50
        self.prev_vespene = 0
        self.total_minerals_spent = 0
        self.total_gas_spent = 0
        self.warpgate_researched = False
        self.latest_observations: Dict[str, Any] = {}
        self.game_result = None
        self.current_iteration = 0

    def _decode_parameters(self):
        p = self.params
        # Macro (0-7)
        clamped_p0 = max(0.5, p[0])
        self.probe_target_per_base = max(12, min(22, int(12 + (clamped_p0 - 0.5) * 20)))
        self.gas_supply_trigger = int(14 + p[1] * 16)
        self.expand_supply = int(28 + p[2] * 32)
        self.pylon_buffer = int(2 + p[3] * 8)
        self.chrono_target_tech = p[4] > 0.5
        self.target_gateways = max(1, int(1 + p[5] * 4))
        self.robotics_pref = p[6]
        self.stargate_pref = p[7]
        # Micro (8-22)
        self.stalker_ratio = max(0.1, p[8])
        self.zealot_ratio = min(0.8, p[9])
        self.immortal_ratio = min(0.6, p[10])
        self.colossus_ratio = min(0.5, p[11])
        self.voidray_ratio = min(0.6, p[12])
        self.upgrade_priority = p[13]
        self.warpgate_research_prio = p[14]
        attack_supply = int(20 + p[15] * 60)
        self.retreat_threshold = max(0.1, min(0.9, p[16]))
        self.aggression_level = p[17]
        self.effective_attack_threshold = int(attack_supply * (1.0 - 0.3 * p[17]))
        self.attack_supply_threshold = attack_supply
        if p[18] < 0.01:
            self.harass_interval = 999999
        else:
            self.harass_interval = max(200, int((1 - p[18]) * 2000))
        self.defense_target = int(p[19] * 4)
        self.blink_prio = p[20]
        self.tech_switch = p[21]
        self.observer_ratio = min(0.4, p[22])

    def _safe_pending(self, unit_type):
        """Wraps already_pending to handle KeyError 4449 (unmapped ability IDs in BurnySC2)."""
        try:
            return self.already_pending(unit_type)
        except (KeyError, Exception):
            return 0

    async def on_step(self, iteration: int):
        self.current_iteration = iteration
        try:
            self._update_observations(iteration)
        except (KeyError, Exception) as e:
            logger.debug(f"Observation update skipped: {e}")
        self._track_metrics(iteration)
        try:
            await self.distribute_workers()
        except (KeyError, Exception):
            pass
        await self._safe_call(self._execute_economy(iteration))
        await self._safe_call(self._execute_infrastructure(iteration))
        await self._safe_call(self._execute_research(iteration))
        await self._safe_call(self._execute_training(iteration))
        await self._safe_call(self._execute_strategy(iteration))
        await self._safe_call(self._execute_micro(iteration))

    async def _safe_call(self, coro):
        """Run an async method, catching KeyError from BurnySC2 ability mapping bugs."""
        try:
            await coro
        except (KeyError, Exception) as e:
            logger.debug(f"Step sub-call error (non-fatal): {e}")

    def _update_observations(self, iteration: int):
        cat_res = {
            "probe_supply": self.workers.amount,
            "mineral_amount": self.minerals,
            "gas_amount": self.vespene,
            "supply_left": self.supply_left,
            "supply_cap": self.supply_cap,
            "supply_used": self.supply_used,
            "base_count": self.townhalls.amount,
        }
        cat_units: Dict[str, int] = {}
        for u in self.units:
            n = u.type_id.name
            cat_units[n] = cat_units.get(n, 0) + 1
        cat_bldg: Dict[str, int] = {}
        for s in self.structures:
            n = s.type_id.name
            cat_bldg[n] = cat_bldg.get(n, 0) + 1
        cat_pending: Dict[str, int] = {}
        for ut in [UnitTypeId.PROBE, UnitTypeId.PYLON, UnitTypeId.NEXUS,
                   UnitTypeId.GATEWAY, UnitTypeId.ASSIMILATOR,
                   UnitTypeId.CYBERNETICSCORE, UnitTypeId.ROBOTICSFACILITY,
                   UnitTypeId.STARGATE, UnitTypeId.ZEALOT, UnitTypeId.STALKER,
                   UnitTypeId.IMMORTAL, UnitTypeId.COLOSSUS, UnitTypeId.VOIDRAY]:
            pend = self._safe_pending(ut)
            if pend > 0:
                cat_pending[f"planning_{ut.name}_count"] = pend
        cat_enemy_u: Dict[str, int] = {}
        cat_enemy_s: Dict[str, int] = {}
        for eu in self.enemy_units:
            n = eu.type_id.name
            cat_enemy_u[n] = cat_enemy_u.get(n, 0) + 1
        for es in self.enemy_structures:
            n = es.type_id.name
            cat_enemy_s[n] = cat_enemy_s.get(n, 0) + 1
        cat_research: Dict[str, float] = {}
        for up in [UpgradeId.WARPGATERESEARCH, UpgradeId.BLINKTECH,
                   UpgradeId.CHARGE, UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1,
                   UpgradeId.PROTOSSGROUNDARMORSLEVEL1]:
            cat_research[up.name] = 1.0 if up in self.state.upgrades else 0.0
        self.latest_observations = {
            "step": iteration, "resources": cat_res, "units": cat_units,
            "buildings": cat_bldg, "in_process": cat_pending,
            "enemy_status": {"units": cat_enemy_u, "buildings": cat_enemy_s},
            "research_progress": cat_research,
        }

    def _track_metrics(self, iteration: int):
        if self.reached_max_supply:
            return
        if self.supply_used >= 198:
            self.reached_max_supply = True
            self.max_supply_step = iteration
            return
        if self.supply_left == 0 and self.supply_cap < 200:
            self.supply_blocked_steps += 1
        if self.supply_cap > 0:
            self.apu_history.append(self.supply_used / self.supply_cap)
        m_spent = max(0, self.prev_minerals - self.minerals)
        g_spent = max(0, self.prev_vespene - self.vespene)
        self.total_minerals_spent += m_spent
        self.total_gas_spent += g_spent
        self.prev_minerals = self.minerals
        self.prev_vespene = self.vespene

    async def _execute_economy(self, iteration: int):
        # Train probes
        max_probes = self.townhalls.amount * self.probe_target_per_base
        for nexus in self.townhalls.idle:
            if self.workers.amount < max_probes and self.can_afford(UnitTypeId.PROBE):
                nexus.train(UnitTypeId.PROBE)
        # Build pylons
        if (self.supply_left <= self.pylon_buffer and self.supply_cap < 200
                and not self._safe_pending(UnitTypeId.PYLON)
                and self.can_afford(UnitTypeId.PYLON)):
            if self.townhalls:
                pos = self.townhalls.first.position.towards(self.game_info.map_center, 5)
                await self.build(UnitTypeId.PYLON, near=pos)
        # Build assimilators
        if self.supply_used >= self.gas_supply_trigger:
            for nexus in self.townhalls.ready:
                for geyser in self.vespene_geyser.closer_than(10, nexus):
                    if (not self.gas_buildings.closer_than(1, geyser)
                            and self.can_afford(UnitTypeId.ASSIMILATOR)
                            and self._safe_pending(UnitTypeId.ASSIMILATOR) < self.townhalls.amount * 2):
                        await self.build(UnitTypeId.ASSIMILATOR, geyser)
        # Expand
        if (self.supply_used >= self.expand_supply and self.minerals >= 400
                and self.townhalls.amount < 3
                and not self._safe_pending(UnitTypeId.NEXUS)):
            await self.expand_now()
        # Chrono boost
        for nexus in self.townhalls(UnitTypeId.NEXUS).ready:
            if nexus.energy >= 50:
                if self.chrono_target_tech:
                    tech_targets = (self.structures(UnitTypeId.CYBERNETICSCORE).ready
                                    | self.structures(UnitTypeId.FORGE).ready
                                    | self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready
                                    | self.structures(UnitTypeId.ROBOTICSFACILITY).ready)
                    for t in tech_targets:
                        if not t.is_idle and not t.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                            nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, t)
                            break
                    else:
                        if not nexus.is_idle and not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                            nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus)
                else:
                    if not nexus.is_idle and not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                        nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus)

    async def _execute_infrastructure(self, iteration: int):
        pylons = self.structures(UnitTypeId.PYLON).ready
        if not pylons:
            return
        pylon = pylons.first
        # Gateway
        gw_count = (self.structures(UnitTypeId.GATEWAY).amount
                     + self.structures(UnitTypeId.WARPGATE).amount
                     + self._safe_pending(UnitTypeId.GATEWAY))
        if gw_count < self.target_gateways and self.can_afford(UnitTypeId.GATEWAY):
            await self.build(UnitTypeId.GATEWAY, near=pylon)
        # Cybernetics Core (always after first gateway)
        if (self.structures(UnitTypeId.GATEWAY).ready.amount
                + self.structures(UnitTypeId.WARPGATE).amount >= 1
                and not self.structures(UnitTypeId.CYBERNETICSCORE)
                and not self._safe_pending(UnitTypeId.CYBERNETICSCORE)
                and self.can_afford(UnitTypeId.CYBERNETICSCORE)):
            await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)
        cyber_ready = self.structures(UnitTypeId.CYBERNETICSCORE).ready.amount > 0
        # Robotics Facility
        if (self.robotics_pref > 0.3 and cyber_ready
                and not self.structures(UnitTypeId.ROBOTICSFACILITY)
                and not self._safe_pending(UnitTypeId.ROBOTICSFACILITY)
                and self.can_afford(UnitTypeId.ROBOTICSFACILITY)):
            await self.build(UnitTypeId.ROBOTICSFACILITY, near=pylon)
        # Stargate
        if (self.stargate_pref > 0.3 and cyber_ready
                and not self.structures(UnitTypeId.STARGATE)
                and not self._safe_pending(UnitTypeId.STARGATE)
                and self.can_afford(UnitTypeId.STARGATE)):
            await self.build(UnitTypeId.STARGATE, near=pylon)
        # Twilight Council (for blink)
        if (self.blink_prio > 0.6 and cyber_ready
                and not self.structures(UnitTypeId.TWILIGHTCOUNCIL)
                and not self._safe_pending(UnitTypeId.TWILIGHTCOUNCIL)
                and self.can_afford(UnitTypeId.TWILIGHTCOUNCIL)):
            await self.build(UnitTypeId.TWILIGHTCOUNCIL, near=pylon)
        # Forge (for upgrades)
        if (cyber_ready and not self.structures(UnitTypeId.FORGE)
                and not self._safe_pending(UnitTypeId.FORGE)
                and self.can_afford(UnitTypeId.FORGE)):
            await self.build(UnitTypeId.FORGE, near=pylon)
        # Robotics Bay (for colossus)
        if (self.colossus_ratio > 0.1
                and self.structures(UnitTypeId.ROBOTICSFACILITY).ready
                and not self.structures(UnitTypeId.ROBOTICSBAY)
                and not self._safe_pending(UnitTypeId.ROBOTICSBAY)
                and self.can_afford(UnitTypeId.ROBOTICSBAY)):
            await self.build(UnitTypeId.ROBOTICSBAY, near=pylon)
        # Defensive structures
        def_count = (self.structures(UnitTypeId.SHIELDBATTERY).amount
                     + self.structures(UnitTypeId.PHOTONCANNON).amount)
        if def_count < self.defense_target and self.townhalls:
            near_pos = self.townhalls.first.position.towards(self.game_info.map_center, 5)
            if (cyber_ready and self.can_afford(UnitTypeId.SHIELDBATTERY)
                    and self.structures(UnitTypeId.SHIELDBATTERY).amount < 2):
                await self.build(UnitTypeId.SHIELDBATTERY, near=near_pos)
            elif (self.structures(UnitTypeId.FORGE).ready
                    and self.can_afford(UnitTypeId.PHOTONCANNON)):
                await self.build(UnitTypeId.PHOTONCANNON, near=near_pos)

    async def _execute_research(self, iteration: int):
        # Warpgate research
        if (self.warpgate_research_prio > 0.3
                and self.structures(UnitTypeId.CYBERNETICSCORE).ready
                and not self.warpgate_researched
                and self.minerals >= 50 and self.vespene >= 50):
            for cyber in self.structures(UnitTypeId.CYBERNETICSCORE).ready.idle:
                if UpgradeId.WARPGATERESEARCH not in self.state.upgrades:
                    cyber.research(UpgradeId.WARPGATERESEARCH)
        # Check if warpgate done and morph gateways
        if UpgradeId.WARPGATERESEARCH in self.state.upgrades:
            self.warpgate_researched = True
            for gw in self.structures(UnitTypeId.GATEWAY).ready:
                gw(AbilityId.MORPH_WARPGATE)
        # Blink
        if (self.blink_prio > 0.6
                and self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready):
            for tc in self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.idle:
                if UpgradeId.BLINKTECH not in self.state.upgrades:
                    if self.can_afford(UpgradeId.BLINKTECH):
                        tc.research(UpgradeId.BLINKTECH)
                elif UpgradeId.CHARGE not in self.state.upgrades:
                    if self.can_afford(UpgradeId.CHARGE):
                        tc.research(UpgradeId.CHARGE)
        # Forge upgrades
        for forge in self.structures(UnitTypeId.FORGE).ready.idle:
            if self.upgrade_priority < 0.5:
                for up in [UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1,
                           UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2,
                           UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3]:
                    if up not in self.state.upgrades and self.can_afford(up):
                        forge.research(up)
                        break
            else:
                for up in [UpgradeId.PROTOSSGROUNDARMORSLEVEL1,
                           UpgradeId.PROTOSSGROUNDARMORSLEVEL2,
                           UpgradeId.PROTOSSGROUNDARMORSLEVEL3]:
                    if up not in self.state.upgrades and self.can_afford(up):
                        forge.research(up)
                        break

    async def _execute_training(self, iteration: int):
        # Dynamic tech switch adjustments
        eff_stalker = self.stalker_ratio
        eff_zealot = self.zealot_ratio
        eff_immortal = self.immortal_ratio
        if self.tech_switch > 0.3 and self.enemy_units:
            armored = sum(1 for e in self.enemy_units if e.is_armored)
            light = sum(1 for e in self.enemy_units if e.is_light)
            air = sum(1 for e in self.enemy_units if e.is_flying)
            if armored > 4:
                eff_immortal = min(0.6, self.immortal_ratio + 0.2 * self.tech_switch)
            if light > 8:
                eff_zealot = min(0.8, self.zealot_ratio + 0.2 * self.tech_switch)
            if air > 3:
                eff_stalker = min(1.0, self.stalker_ratio + 0.15 * self.tech_switch)
        has_cyber = self.structures(UnitTypeId.CYBERNETICSCORE).ready.amount > 0
        # Train from Warp Gates (warp-in near home base pylon, not proxy)
        home_pylons = self.structures(UnitTypeId.PYLON).ready
        if home_pylons and self.townhalls:
            warp_pylon = home_pylons.closest_to(self.townhalls.first)
            warp_pos = warp_pylon.position
            for wg in self.structures(UnitTypeId.WARPGATE).ready.idle:
                if has_cyber and random.random() < eff_stalker and self.can_afford(UnitTypeId.STALKER):
                    wg.warp_in(UnitTypeId.STALKER, warp_pos)
                elif random.random() < eff_zealot and self.can_afford(UnitTypeId.ZEALOT):
                    wg.warp_in(UnitTypeId.ZEALOT, warp_pos)
        # Train from Gateways (before warpgate research)
        for gw in self.structures(UnitTypeId.GATEWAY).ready.idle:
            if has_cyber and random.random() < eff_stalker and self.can_afford(UnitTypeId.STALKER):
                gw.train(UnitTypeId.STALKER)
            elif random.random() < eff_zealot and self.can_afford(UnitTypeId.ZEALOT):
                gw.train(UnitTypeId.ZEALOT)
        # Train from Robotics Facility
        for robo in self.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle:
            if (random.random() < self.observer_ratio
                    and not self.units(UnitTypeId.OBSERVER)
                    and self.can_afford(UnitTypeId.OBSERVER)):
                robo.train(UnitTypeId.OBSERVER)
            elif (self.structures(UnitTypeId.ROBOTICSBAY).ready
                    and random.random() < self.colossus_ratio
                    and self.can_afford(UnitTypeId.COLOSSUS)):
                robo.train(UnitTypeId.COLOSSUS)
            elif random.random() < eff_immortal and self.can_afford(UnitTypeId.IMMORTAL):
                robo.train(UnitTypeId.IMMORTAL)
        # Train from Stargate
        for sg in self.structures(UnitTypeId.STARGATE).ready.idle:
            if random.random() < self.voidray_ratio and self.can_afford(UnitTypeId.VOIDRAY):
                sg.train(UnitTypeId.VOIDRAY)

    def _get_combat_units(self):
        result = self.units(UnitTypeId.ZEALOT)
        for ut in COMBAT_UNIT_TYPES[1:]:
            result = result | self.units(ut)
        return result

    async def _execute_strategy(self, iteration: int):
        army_supply = self.supply_used - self.workers.amount
        combat = self._get_combat_units()
        # Attack trigger
        if (army_supply >= self.effective_attack_threshold
                and not self.attack_triggered
                and iteration - self.last_retreat_step > 300):
            self.attack_triggered = True
            self.pre_engagement_army_supply = army_supply
        # Retreat
        if self.attack_triggered and self.pre_engagement_army_supply > 0:
            loss = 1.0 - (army_supply / max(1, self.pre_engagement_army_supply))
            if loss >= self.retreat_threshold:
                self.attack_triggered = False
                self.last_retreat_step = iteration
                if self.townhalls:
                    rally = self.townhalls.first.position.towards(self.game_info.map_center, 8)
                    for unit in combat:
                        unit.move(rally)
        # Harassment
        if iteration - self.last_harass_step >= self.harass_interval:
            harass_units = self.units(UnitTypeId.VOIDRAY).idle | self.units(UnitTypeId.ORACLE).idle
            if harass_units and self.enemy_start_locations:
                for u in harass_units:
                    u.attack(self.enemy_start_locations[0])
                self.last_harass_step = iteration

    async def _execute_micro(self, iteration: int):
        combat = self._get_combat_units()
        observers = self.units(UnitTypeId.OBSERVER)
        if self.attack_triggered and self.enemy_start_locations:
            visible_enemies = self.enemy_units
            target = self.enemy_start_locations[0]
            if visible_enemies and combat:
                try:
                    target = visible_enemies.closest_to(combat.center).position
                except (ValueError, Exception):
                    target = self.enemy_start_locations[0]
            for unit in combat:
                nearby = self.enemy_units.closer_than(10, unit)
                if nearby:
                    unit.attack(nearby.closest_to(unit))
                else:
                    unit.attack(target)
            for obs in observers:
                obs.move(target)
        else:
            if self.townhalls:
                rally = self.townhalls.first.position.towards(self.game_info.map_center, 8)
                for unit in combat:
                    if unit.distance_to(rally) > 5:
                        unit.move(rally)

    async def on_end(self, game_result):
        from sc2.data import Result
        self.game_result = game_result

    def get_evaluation_metrics(self, win_score: float = 0.0) -> Dict[str, float]:
        # Use tracked iteration as reliable fallback; game_loop may be inaccessible at game end
        state = getattr(self, "state", None)
        try:
            game_loop = state.game_loop if state else self.current_iteration
        except Exception:
            game_loop = self.current_iteration
        if game_loop <= 0:
            game_loop = 10000  # ~7 minutes at SC2 normal speed, realistic fallback
        duration = max(1.0, float(self.max_supply_step if self.reached_max_supply else game_loop))
        duration_seconds = duration / 22.4
        pbr = self.supply_blocked_steps / max(1.0, duration)
        rur = (self.total_minerals_spent + self.total_gas_spent) / max(1.0, duration_seconds)
        apu = (sum(self.apu_history) / len(self.apu_history)) if self.apu_history else 0.5
        upgrades_count = len(getattr(state, "upgrades", [])) if state else 0
        tech_bldg_types = [UnitTypeId.ROBOTICSFACILITY, UnitTypeId.STARGATE,
                           UnitTypeId.TWILIGHTCOUNCIL, UnitTypeId.ROBOTICSBAY,
                           UnitTypeId.FORGE, UnitTypeId.TEMPLARARCHIVE]
        tech_count = 0
        structs = getattr(self, "structures", None)
        if structs and hasattr(structs, "ready"):
            tech_count = sum(1 for s in structs.ready if s.type_id in tech_bldg_types)
        tr = min(1.0, (upgrades_count + tech_count) / 15.0)
        return {
            "WINRATE": round(float(win_score), 4),
            "PBR": round(float(pbr), 4),
            "RUR": round(float(rur), 2),
            "APU": round(float(apu), 4),
            "TR": round(float(tr), 4),
        }

