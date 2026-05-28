"""
GESY Framework — Module 3: Game Simulator
==========================================
A deterministic, tick-based RTS mini-simulator.  Two strategy scripts play
head-to-head; the simulator advances the game state each tick by applying
both scripts' chosen commands simultaneously, then returns a match result.

Design notes
------------
* Pure Python, no external dependencies — runs inside worker threads safely.
* All randomness is seeded per-match so replays are reproducible.
* Normalised state values keep thresholds in evolved scripts portable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from gesy.core.ast_nodes import GameState, StrategyScript


# ---------------------------------------------------------------------------
# Match result
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    winner:       Optional[str]   # script name, or None for draw
    loser:        Optional[str]
    draw:         bool
    ticks:        int
    p1_final_hp:  float
    p2_final_hp:  float
    p1_name:      str
    p2_name:      str


# ---------------------------------------------------------------------------
# Simulator constants
# ---------------------------------------------------------------------------

MAX_TICKS        = 600
UNIT_GOLD_COST   = 0.05     # normalised
BASE_HP_INIT     = 1.0
UNIT_HP          = 0.10     # each unit contributes this much to aggregate HP
GOLD_PER_TICK    = 0.008    # passive income
RESOURCE_BONUS   = 0.004    # extra income when controlling the centre

# Combat constants (per tick per unit advantage)
MELEE_DPS        = 0.015
RANGED_DPS       = 0.010
SIEGE_DPS        = 0.020    # vs base only


# ---------------------------------------------------------------------------
# Player state (mutable, internal to the simulator)
# ---------------------------------------------------------------------------

@dataclass
class _Player:
    name:         str
    base_hp:      float = BASE_HP_INIT
    gold:         float = 0.20           # start with some gold
    melee_units:  int   = 2
    ranged_units: int   = 1
    siege_units:  int   = 0
    controls_res: bool  = False          # holds resource node?

    @property
    def total_units(self) -> int:
        return self.melee_units + self.ranged_units + self.siege_units

    @property
    def aggregate_hp(self) -> float:
        """Normalised [0,1] HP proxy based on unit count."""
        raw = self.melee_units * UNIT_HP + self.ranged_units * UNIT_HP
        return min(raw, 1.0)

    def is_defeated(self) -> bool:
        return self.base_hp <= 0.0


def _build_state(
    me:    _Player,
    them:  _Player,
    tick:  int,
    can_attack:   bool,
    under_attack: bool,
) -> GameState:
    return GameState(
        own_hp            = me.aggregate_hp,
        own_gold          = min(me.gold, 1.0),
        own_unit_count    = me.total_units,
        own_base_hp       = me.base_hp,
        enemy_hp          = them.aggregate_hp,
        enemy_gold        = min(them.gold, 1.0),
        enemy_unit_count  = them.total_units,
        enemy_base_hp     = them.base_hp,
        can_attack        = can_attack,
        can_build         = me.gold >= UNIT_GOLD_COST,
        under_attack      = under_attack,
        ranged_advantage  = me.ranged_units > them.ranged_units,
        melee_advantage   = me.melee_units > them.melee_units,
        resource_node_controlled = me.controls_res,
        elapsed_ticks     = tick,
    )


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def simulate_match(
    script_a: StrategyScript,
    script_b: StrategyScript,
    seed:     Optional[int] = None,
    verbose:  bool = False,
) -> MatchResult:
    """
    Run a full head-to-head match between two strategy scripts.
    Returns a MatchResult.  Thread-safe (all state is local).
    """
    rng = random.Random(seed)

    p1 = _Player(name=script_a.name)
    p2 = _Player(name=script_b.name)

    # Resource node randomly starts neutral or with one side
    contested = rng.random() < 0.50
    if not contested:
        if rng.random() < 0.5:
            p1.controls_res = True
        else:
            p2.controls_res = True

    log = []

    for tick in range(MAX_TICKS):
        # -----------------------------------------------------------------
        # Build game-state views (each player sees from their perspective)
        # -----------------------------------------------------------------
        p1_can_atk = p1.total_units > 0 and p2.total_units > 0
        p2_can_atk = p2.total_units > 0 and p1.total_units > 0
        state1 = _build_state(p1, p2, tick, p1_can_atk, p2.total_units > 0)
        state2 = _build_state(p2, p1, tick, p2_can_atk, p1.total_units > 0)

        # -----------------------------------------------------------------
        # Execute scripts → get commands
        # -----------------------------------------------------------------
        cmd1 = script_a.execute(state1) or "no_op"
        cmd2 = script_b.execute(state2) or "no_op"

        # -----------------------------------------------------------------
        # Apply commands (simultaneously)
        # -----------------------------------------------------------------
        _apply_command(cmd1, p1, p2, rng)
        _apply_command(cmd2, p2, p1, rng)

        # -----------------------------------------------------------------
        # Passive income
        # -----------------------------------------------------------------
        p1.gold += GOLD_PER_TICK + (RESOURCE_BONUS if p1.controls_res else 0)
        p2.gold += GOLD_PER_TICK + (RESOURCE_BONUS if p2.controls_res else 0)

        # -----------------------------------------------------------------
        # Combat resolution: melee
        # -----------------------------------------------------------------
        if p1.total_units > 0 and p2.total_units > 0:
            # Net damage proportional to unit advantage
            p1_advantage = max(0, p1.melee_units - p2.melee_units)
            p2_advantage = max(0, p2.melee_units - p1.melee_units)
            p2.base_hp -= p1_advantage * MELEE_DPS * rng.uniform(0.7, 1.3)
            p1.base_hp -= p2_advantage * MELEE_DPS * rng.uniform(0.7, 1.3)
            # Ranged
            p1r = p1.ranged_units * RANGED_DPS
            p2r = p2.ranged_units * RANGED_DPS
            p2.base_hp -= p1r * rng.uniform(0.6, 1.4)
            p1.base_hp -= p2r * rng.uniform(0.6, 1.4)

        # -----------------------------------------------------------------
        # Siege vs base (only affects base, not units)
        # -----------------------------------------------------------------
        p2.base_hp -= p1.siege_units * SIEGE_DPS * rng.uniform(0.5, 1.5)
        p1.base_hp -= p2.siege_units * SIEGE_DPS * rng.uniform(0.5, 1.5)

        # Clamp
        p1.base_hp = max(0.0, min(1.0, p1.base_hp))
        p2.base_hp = max(0.0, min(1.0, p2.base_hp))

        if verbose:
            log.append(
                f"T{tick:04d}  {p1.name}: cmd={cmd1:>15s} base={p1.base_hp:.3f} "
                f"units={p1.total_units}  |  "
                f"{p2.name}: cmd={cmd2:>15s} base={p2.base_hp:.3f} units={p2.total_units}"
            )

        # -----------------------------------------------------------------
        # Termination check
        # -----------------------------------------------------------------
        if p1.is_defeated() and p2.is_defeated():
            return _make_result(None, p1, p2, tick, draw=True)
        if p1.is_defeated():
            return _make_result(p2.name, p1, p2, tick)
        if p2.is_defeated():
            return _make_result(p1.name, p1, p2, tick)

    # Time limit → decide by base HP
    if p1.base_hp > p2.base_hp:
        return _make_result(p1.name, p1, p2, MAX_TICKS)
    elif p2.base_hp > p1.base_hp:
        return _make_result(p2.name, p1, p2, MAX_TICKS)
    else:
        return _make_result(None, p1, p2, MAX_TICKS, draw=True)


def _apply_command(
    cmd:  str,
    me:   _Player,
    them: _Player,
    rng:  random.Random,
) -> None:
    """Mutate player state based on chosen command."""
    if cmd == "attack":
        if me.total_units > 0:
            dmg = (me.melee_units * 0.012 + me.ranged_units * 0.008) * rng.uniform(0.8, 1.2)
            them.base_hp -= dmg

    elif cmd == "retreat":
        # Sacrifice some units to gain base HP (regrouping)
        if me.melee_units > 0:
            me.base_hp = min(1.0, me.base_hp + 0.005)

    elif cmd == "build_melee":
        if me.gold >= UNIT_GOLD_COST:
            me.gold -= UNIT_GOLD_COST
            me.melee_units += 1

    elif cmd == "build_ranged":
        if me.gold >= UNIT_GOLD_COST:
            me.gold -= UNIT_GOLD_COST
            me.ranged_units += 1

    elif cmd == "build_siege":
        cost = UNIT_GOLD_COST * 2
        if me.gold >= cost:
            me.gold -= cost
            me.siege_units += 1

    elif cmd == "fortify":
        if me.gold >= 0.08:
            me.gold -= 0.08
            me.base_hp = min(1.0, me.base_hp + 0.04)

    elif cmd == "rush":
        # Big attack push — more damage, but lose a unit
        if me.total_units > 0:
            dmg = (me.total_units * 0.025) * rng.uniform(0.7, 1.3)
            them.base_hp -= dmg
            # Attacker takes losses
            if me.melee_units > 0 and rng.random() < 0.35:
                me.melee_units -= 1

    elif cmd == "hold_position":
        # Defensive bonus: reduces incoming damage next tick (modelled as HP regen)
        me.base_hp = min(1.0, me.base_hp + 0.003)

    elif cmd == "harass":
        # Reduces enemy economy
        them.gold = max(0.0, them.gold - 0.03 * rng.uniform(0.5, 1.5))

    elif cmd == "gather_resource":
        # Attempt to take the resource node
        if not me.controls_res:
            if rng.random() < 0.25:
                me.controls_res = True
                them.controls_res = False

    # "no_op" → nothing


def _make_result(
    winner_name: Optional[str],
    p1:    _Player,
    p2:    _Player,
    ticks: int,
    draw:  bool = False,
) -> MatchResult:
    loser = None
    if winner_name == p1.name:
        loser = p2.name
    elif winner_name == p2.name:
        loser = p1.name

    return MatchResult(
        winner       = winner_name,
        loser        = loser,
        draw         = draw,
        ticks        = ticks,
        p1_final_hp  = p1.base_hp,
        p2_final_hp  = p2.base_hp,
        p1_name      = p1.name,
        p2_name      = p2.name,
    )
