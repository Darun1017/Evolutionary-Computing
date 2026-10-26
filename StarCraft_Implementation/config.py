"""
config.py — Central configuration for CCE-TDCCEA StarCraft II implementation.

All hyperparameters, gene definitions, race selection, and difficulty
schedule are defined here. Edit this file to change behaviour without
touching any algorithm or bot logic.
"""

from sc2.data import Difficulty, Race

# ────────────────────────────────────────────────────────────────────────────
#  RACE CONFIGURATION
#  Set BOT_RACE to Race.Terran, Race.Protoss, or Race.Zerg.
#  OPPONENT_RACE = Race.Random keeps training general.
# ────────────────────────────────────────────────────────────────────────────
BOT_RACE       = Race.Terran
OPPONENT_RACE  = Race.Random

# ────────────────────────────────────────────────────────────────────────────
#  MAP & GAME SETTINGS
# ────────────────────────────────────────────────────────────────────────────
MAP_NAME           = "CatalystLE"      # Standard 1v1 ladder map
GAME_TIME_LIMIT    = 480              # 8-minute cap per game (in-game seconds)

# ────────────────────────────────────────────────────────────────────────────
#  DIFFICULTY SCHEDULE  (ramps up from VeryEasy → Harder over all generations)
#  Index i is used for generation i (0-based).
#  If gen index >= len(schedule), the last entry is used.
# ────────────────────────────────────────────────────────────────────────────
DIFFICULTY_SCHEDULE = [
    Difficulty.VeryEasy,   # Generation 1
    Difficulty.Easy,       # Generation 2
    Difficulty.Medium,     # Generation 3
    Difficulty.Hard,       # Generation 4
    Difficulty.Harder,     # Generation 5  ← and all subsequent gens
]
BENCHMARK_DIFFICULTY = Difficulty.Harder    # Level 5 for final validation

def get_difficulty(gen_index: int) -> Difficulty:
    """Return the difficulty for a given 0-based generation index."""
    idx = min(gen_index, len(DIFFICULTY_SCHEDULE) - 1)
    return DIFFICULTY_SCHEDULE[idx]

# ────────────────────────────────────────────────────────────────────────────
#  EVOLUTIONARY HYPERPARAMETERS
# ────────────────────────────────────────────────────────────────────────────
TOTAL_GENERATIONS = 5        # Reduced for feasibility; ramp-up covered by schedule
N_POP             = 20       # Population size per sub-population
K_GAMES           = 3        # Games per fitness evaluation (averaged)
HISTORY_LEN       = 7        # Maximum trajectory window (H_i entries)
A_MAX             = 15       # Archive capacity per sub-population
DELTA_S           = 0.85     # Similarity ceiling for diversity pruning
DELTA_F           = 0.20     # Fitness floor for stagnant-agent removal
MIN_SURVIVORS     = 8        # Minimum agents to keep after pruning

# ────────────────────────────────────────────────────────────────────────────
#  SA (SENSITIVITY-ADAPTIVE) MUTATION BOUNDARIES
#  mu_j = MU_MIN + (1 - I_j/I_max) * (MU_MAX - MU_MIN)
#  High-influence gene → mu_j ≈ MU_MIN (fine-tune carefully)
#  Low-influence gene  → mu_j ≈ MU_MAX (explore freely)
# ────────────────────────────────────────────────────────────────────────────
MU_MIN   = 0.08    # Lower bound on per-gene mutation std
MU_MAX   = 0.20    # Upper bound on per-gene mutation std
MU_FLOOR = 0.02    # Additive floor on sigma (prevents collapse at e_i → 1)

# Default flat mutation (used before sensitivity profiling completes)
MU_DEFAULT = 0.14

# ────────────────────────────────────────────────────────────────────────────
#  COMPOSITE FITNESS WEIGHTS
#  f_i = w_wr·WR + w_apu·APU + w_pbr·PBR + w_rur·RUR
# ────────────────────────────────────────────────────────────────────────────
FITNESS_WEIGHTS = {
    "win_rate": 0.40,
    "apu":      0.25,
    "pbr":      0.20,
    "rur":      0.15,
}

# ────────────────────────────────────────────────────────────────────────────
#  GENOME: 35D GENE VECTOR
#
#  All weights w_j ∈ [0, 1] (continuous).
#  Each sub-population evolves its slice independently; the cooperative
#  handshake concatenates all three into a full 35D bot for evaluation.
#
#  Pop A  — Macro / Economy     (indices  0– 9,  dim=10)
#  Pop B  — Micro / Combat      (indices 10–24,  dim=15)
#  Pop C  — Tech / Build-Order  (indices 25–34,  dim=10)
# ────────────────────────────────────────────────────────────────────────────

# Pop A — Macro / Economy
MACRO_GENES = [
    "worker_ratio",            #  0  aggressiveness of worker production [0=slow, 1=fast]
    "max_workers",             #  1  worker ceiling (maps to absolute count [8, 22])
    "expand_threshold",        #  2  mineral threshold to expand [400, 900]
    "supply_buffer",           #  3  supply headroom before building supply building [2, 8]
    "prod_building_timing",    #  4  supply count to build 1st production building [10, 20]
    "second_prod_building",    #  5  supply count to build 2nd production building [18, 34]
    "refinery_timing",         #  6  worker count to build gas building [8, 16]
    "economy_pressure",        #  7  weight economy vs military (0=military, 1=eco)
    "vespene_ratio",           #  8  target gas fraction of income [0=minerals, 1=gas]
    "expansion_aggression",    #  9  eagerness to take additional bases [0=passive, 1=greedy]
]

# Pop B — Micro / Combat
MICRO_GENES = [
    "aggression",              # 10  overall aggression level
    "attack_threshold",        # 11  minimum army supply before attacking [5, 30]
    "defend_radius",           # 12  distance from base to intercept threats [5, 25]
    "target_priority",         # 13  0=prefer worker targets, 1=prefer military targets
    "defense_trigger",         # 14  enemy proximity (game units) to trigger base defense
    "counter_rush",            # 15  pull workers into combat during early rush (0=no, 1=yes)
    "group_attack",            # 16  wait for army to consolidate before attacking
    "focus_fire",              # 17  concentrate fire on single enemy unit
    "kite_distance",           # 18  retreat distance per attack cycle [0, 4 game-units]
    "retreat_threshold",       # 19  unit HP fraction to trigger individual retreat [5%, 40%]
    "ability_threshold",       # 20  HP% / energy% trigger for special abilities (Stim/Charge)
    "assault_timing",          # 21  game-time fraction for first timing attack
    "harassment_weight",       # 22  priority for economic harassment / drops
    "unit_composition_a",      # 23  preference for primary unit type (Marine/Zealot/Zergling)
    "unit_composition_b",      # 24  preference for secondary unit type (Marauder/Stalker/Roach)
]

# Pop C — Tech / Build-Order
TECH_GENES = [
    "tech_switch",             # 25  willingness to pivot tech path mid-game
    "advanced_building_timing",# 26  supply count to build tech building (Factory/Robo/Hydra Den)
    "upgrade_priority",        # 27  how early to research upgrades [0=late, 1=early]
    "tier3_timing",            # 28  supply count to transition to tier-3 units
    "drop_timing",             # 29  supply count to build transport for drops
    "detection_timing",        # 30  supply count to build detection building
    "air_transition",          # 31  weight towards air tech [0=ground, 1=air]
    "support_unit_timing",     # 32  supply count to build heal/transport units (Medivac etc.)
    "ability_usage",           # 33  frequency / aggression of special ability usage
    "defensive_structure",     # 34  supply count to build defensive structures (bunker/spine)
]

GENE_NAMES   = MACRO_GENES + MICRO_GENES + TECH_GENES
DIMENSIONS   = len(GENE_NAMES)   # Must equal 35

MACRO_INDICES = list(range(0,  10))   # Pop A slice indices into full 35D vector
MICRO_INDICES = list(range(10, 25))   # Pop B slice indices
TECH_INDICES  = list(range(25, 35))   # Pop C slice indices

assert DIMENSIONS == 35, f"Genome must be 35D; got {DIMENSIONS}"
assert len(MACRO_INDICES) == 10
assert len(MICRO_INDICES) == 15
assert len(TECH_INDICES)  == 10

# ────────────────────────────────────────────────────────────────────────────
#  SENSITIVITY PROFILING
# ────────────────────────────────────────────────────────────────────────────
SENSITIVITY_FILE        = "results/sensitivity_scores_sc2.json"
SENSITIVITY_GAMES       = 2     # games per boundary test (offline phase)
SENSITIVITY_TIME_LIMIT  = 480   # 8-minute truncated games for profiling

# Influence score uses combined RUR+APU delta (not win-rate alone)
SENSITIVITY_RUR_WEIGHT = 0.55   # weight of RUR in influence composite
SENSITIVITY_APU_WEIGHT = 0.45   # weight of APU in influence composite

# ────────────────────────────────────────────────────────────────────────────
#  I/O
# ────────────────────────────────────────────────────────────────────────────
RESULTS_DIR      = "results"
ELITE_SAVE_FREQ  = 1          # save elite genome every N generations
LOG_FILE         = "results/evolution_log.json"
