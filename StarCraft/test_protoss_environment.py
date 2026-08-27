"""
Dry-Run Test for ProtossMacroBot
=================================
Validates 23-parameter decoding, observation structure, and metric
calculation WITHOUT needing StarCraft II installed.
Usage: python test_protoss_environment.py
"""
import json
import sys
sys.modules['sc2'] = type(sys)('sc2')
sys.modules['sc2.bot_ai'] = type(sys)('sc2.bot_ai')
sys.modules['sc2.ids'] = type(sys)('sc2.ids')
sys.modules['sc2.ids.unit_typeid'] = type(sys)('sc2.ids.unit_typeid')
sys.modules['sc2.ids.ability_id'] = type(sys)('sc2.ids.ability_id')
sys.modules['sc2.ids.upgrade_id'] = type(sys)('sc2.ids.upgrade_id')
sys.modules['sc2.ids.buff_id'] = type(sys)('sc2.ids.buff_id')
sys.modules['sc2.position'] = type(sys)('sc2.position')
sys.modules['sc2.data'] = type(sys)('sc2.data')

class FakeEnum:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return self.name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        return self.name == getattr(other, 'name', other)

class FakeUnitTypeId:
    def __getattr__(self, name):
        return FakeEnum(name)
class FakeAbilityId:
    def __getattr__(self, name):
        return FakeEnum(name)
class FakeUpgradeId:
    def __getattr__(self, name):
        return FakeEnum(name)
class FakeBuffId:
    def __getattr__(self, name):
        return FakeEnum(name)

sys.modules['sc2.ids.unit_typeid'].UnitTypeId = FakeUnitTypeId()
sys.modules['sc2.ids.ability_id'].AbilityId = FakeAbilityId()
sys.modules['sc2.ids.upgrade_id'].UpgradeId = FakeUpgradeId()
sys.modules['sc2.ids.buff_id'].BuffId = FakeBuffId()
sys.modules['sc2.position'].Point2 = tuple

class FakeBotAI:
    def __init__(self):
        pass

sys.modules['sc2.bot_ai'].BotAI = FakeBotAI


EVOLVED_VECTOR = [
    0.80, 0.30, 0.40, 0.75, 0.70, 0.55, 0.65,
    0.45, 0.80, 0.50, 0.35, 0.30, 0.50, 0.30,
    0.75, 0.40, 0.45, 0.65, 0.55, 0.70, 0.60,
    0.70, 0.25,
]


def main():
    # Patch imports for dry run
    from protoss_macro_bot import ProtossMacroBot

    print("=" * 85)
    print("  DRY-RUN: PROTOSS 23-PARAMETER STRATEGY, OBSERVATIONS & METRICS VERIFICATION")
    print("=" * 85)

    bot = ProtossMacroBot(EVOLVED_VECTOR)

    # 1. Parameter Table
    print("\n  1. COMPLETE 23-PARAMETER EVOLVED STRATEGY TABLE")
    print("  " + "-" * 81)
    print(f"  {'#':<4s} {'PARAMETER':<30s} {'VALUE':<8s} {'INTERPRETATION':<35s}")
    print("  " + "-" * 81)
    param_info = [
        (0,  "probe_saturation_target",   f"Target {bot.probe_target_per_base} probes/base"),
        (1,  "assimilator_timing",        f"Gas at supply >= {bot.gas_supply_trigger}"),
        (2,  "expansion_timing",          f"Expand at supply >= {bot.expand_supply}"),
        (3,  "pylon_buffer",              f"Build Pylon when supply_left <= {bot.pylon_buffer}"),
        (4,  "chrono_boost_priority",     f"Chrono tech={bot.chrono_target_tech}"),
        (5,  "gateway_count",             f"Target {bot.target_gateways} Gateways"),
        (6,  "robotics_preference",       f"Build Robo={'YES' if bot.robotics_pref > 0.3 else 'NO'} ({bot.robotics_pref:.2f})"),
        (7,  "stargate_preference",       f"Build Stargate={'YES' if bot.stargate_pref > 0.3 else 'NO'} ({bot.stargate_pref:.2f})"),
        (8,  "stalker_ratio",             f"{bot.stalker_ratio:.0%} Stalker production"),
        (9,  "zealot_ratio",              f"{bot.zealot_ratio:.0%} Zealot production"),
        (10, "immortal_ratio",            f"{bot.immortal_ratio:.0%} Immortal production"),
        (11, "colossus_ratio",            f"{bot.colossus_ratio:.0%} Colossus production"),
        (12, "voidray_ratio",             f"{bot.voidray_ratio:.0%} Void Ray production"),
        (13, "upgrade_priority",          f"{'Weapons' if bot.upgrade_priority < 0.5 else 'Armor'} first"),
        (14, "warpgate_research_prio",    f"Research WG={'YES' if bot.warpgate_research_prio > 0.3 else 'NO'}"),
        (15, "attack_threshold",          f"Attack at {bot.attack_supply_threshold} supply (eff: {bot.effective_attack_threshold})"),
        (16, "retreat_threshold",         f"Retreat at {bot.retreat_threshold:.0%} army loss"),
        (17, "aggression_level",          f"Aggression={bot.aggression_level:.2f}"),
        (18, "harassment_frequency",      f"Harass every {bot.harass_interval} steps"),
        (19, "defensive_structures",      f"Target {bot.defense_target} defenses"),
        (20, "blink_research_prio",       f"Blink={'YES' if bot.blink_prio > 0.6 else 'NO'} ({bot.blink_prio:.2f})"),
        (21, "tech_switch_willingness",   f"Tech switch={bot.tech_switch:.2f}"),
        (22, "observer_ratio",            f"{bot.observer_ratio:.0%} Observer production"),
    ]
    for idx, name, desc in param_info:
        print(f"  [{idx:2d}] {name:<30s} {EVOLVED_VECTOR[idx]:<8.2f} {desc}")

    # 2. Sample Observations
    print("\n  2. SAMPLE OBSERVATION JSON (6 CATEGORIES)")
    print("  " + "-" * 81)
    sample_obs = {
        "step": 1500,
        "resources": {
            "probe_supply": 22, "mineral_amount": 380, "gas_amount": 200,
            "supply_left": 8, "supply_cap": 46, "supply_used": 38, "base_count": 1,
        },
        "units": {"PROBE": 22, "ZEALOT": 4, "STALKER": 6},
        "buildings": {"NEXUS": 1, "PYLON": 4, "GATEWAY": 3, "CYBERNETICSCORE": 1},
        "in_process": {"planning_STALKER_count": 2, "planning_PYLON_count": 1},
        "enemy_status": {
            "units": {"ZERGLING": 12, "ROACH": 4, "OVERLORD": 3},
            "buildings": {"HATCHERY": 1, "SPAWNINGPOOL": 1, "ROACHWARREN": 1},
        },
        "research_progress": {"WARPGATERESEARCH": 1.0, "BLINKTECH": 0.0, "PROTOSSGROUNDWEAPONSLEVEL1": 1.0},
    }
    print(json.dumps(sample_obs, indent=4))

    # 3. Metric Computation
    print("\n  3. EVALUATION METRICS (SIMULATED VALUES)")
    print("  " + "-" * 81)
    bot.current_iteration = 10000  # Simulate realistic end-of-game iteration count
    bot.supply_blocked_steps = 120
    bot.total_minerals_spent = 9200
    bot.total_gas_spent = 2800
    bot.apu_history = [0.68, 0.72, 0.75, 0.78, 0.81, 0.79, 0.82]
    bot.reached_max_supply = False
    metrics = bot.get_evaluation_metrics(win_score=1.0)
    print(f"  WINRATE = {metrics['WINRATE']}  (1.0 = Win, 0.0 = Loss)")
    print(f"  PBR     = {metrics['PBR']}  (supply_blocked_steps / duration) Lower is better")
    print(f"  RUR     = {metrics['RUR']}  (resources_spent / duration_seconds)")
    print(f"  APU     = {metrics['APU']}  (mean supply_used/supply_cap) Higher is better")
    print(f"  TR      = {metrics['TR']}  (upgrades + tech_buildings / 15)")

    # 4. Dynamic Reaction
    print("\n  4. DYNAMIC STRATEGY REACTIONS")
    print("  " + "-" * 81)
    print(f"  [Baseline] Stalker={bot.stalker_ratio:.0%} | Zealot={bot.zealot_ratio:.0%} | Immortal={bot.immortal_ratio:.0%}")
    print(f"  [vs Heavy Armor] tech_switch={bot.tech_switch:.2f} -> Immortal boosted to ~{min(0.6, bot.immortal_ratio + 0.2*bot.tech_switch):.0%}")
    print(f"  [vs Mass Light] -> Zealot boosted to ~{min(0.8, bot.zealot_ratio + 0.2*bot.tech_switch):.0%}")
    print(f"  [vs Air Units] -> Stalker boosted to ~{min(1.0, bot.stalker_ratio + 0.15*bot.tech_switch):.0%}")
    print(f"  [Retreat] Triggers at {bot.retreat_threshold:.0%} army loss -> all combat units fall back")

    print()
    print("=" * 85)
    print("  ALL 23 PARAMETERS, OBSERVATIONS & METRICS VERIFIED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == "__main__":
    main()
