"""
GESY + MicroRTS Bridge — command_mapping.py
============================================
Translates GESY DSL action commands into MicroRTS Java bot selections,
and provides a GameStateProxy for evaluating a StrategyScript without
a live game tick.

Design
------
GESY scripts output one of these abstract commands each tick:
    attack, retreat, build_melee, build_ranged, build_siege,
    fortify, rush, hold_position, harass, gather_resource, no_op

MicroRTS BatchGameRunner accepts Java bot names:
    WorkerRush, LightRush, HeavyRush, RandomBiased, Random, Passive

The mapping converts a script's dominant command tendency into the
closest Java bot archetype.  A GameStateProxy creates a synthetic
GameState snapshot so the script can be "polled" for its preferred
command without running a real game.

Usage
-----
    from gesy_microrts_bridge.command_mapping import script_to_bot

    bot_name = script_to_bot(my_strategy_script)   # e.g. "WorkerRush"
"""

from __future__ import annotations

import sys
import os
from typing import Optional

# ── Ensure GESY package is importable ──────────────────────────────────────
# Bridge lives in Code/, GESY package lives in GESY/gesy/
_CODES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GESY_DIR  = os.path.join(_CODES_DIR, "GESY")
for _p in [_CODES_DIR, _GESY_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gesy.core.ast_nodes import GameState, StrategyScript  # noqa: E402


# ---------------------------------------------------------------------------
# 1.  Command → Bot mapping table
# ---------------------------------------------------------------------------

# Maps GESY DSL command names to MicroRTS Java bot archetypes.
# When a script issues a particular command, we interpret it as choosing
# the corresponding Java bot for that game.

COMMAND_TO_BOT: dict[str, str] = {
    "rush":             "WorkerRush",    # all-out early push with workers
    "attack":           "LightRush",     # standard offensive with light units
    "build_melee":      "LightRush",     # balanced melee-focused build
    "build_ranged":     "LightRush",     # ranged = light rush archetype
    "build_siege":      "HeavyRush",     # siege → heavy/slow strategy
    "fortify":          "Passive",       # turtle/defend → passive bot
    "retreat":          "Passive",       # pull back → passive
    "hold_position":    "Passive",       # hold line → passive
    "harass":           "RandomBiased",  # disruptive/unpredictable play
    "gather_resource":  "WorkerRush",    # economy focus → worker-heavy
    "no_op":            "Random",        # no command → random fallback
}

# All available Java bots (must match BatchGameRunner expectations)
ALL_BOT_TYPES: list[str] = [
    "WorkerRush",
    "LightRush",
    "HeavyRush",
    "RandomBiased",
    "Random",
    "Passive",
]

# Default fallback bot if script returns None or unknown command
DEFAULT_BOT = "LightRush"


# ---------------------------------------------------------------------------
# 2.  GameStateProxy — a representative game state for script polling
# ---------------------------------------------------------------------------

def make_early_game_state() -> GameState:
    """
    Synthetic GameState representing a typical early-game situation
    (both sides have moderate units, some gold, no one is losing yet).
    Used to poll a StrategyScript for its preferred command without
    running a full simulation.
    """
    return GameState(
        own_hp              = 0.80,
        own_gold            = 0.30,
        own_unit_count      = 3,
        own_base_hp         = 1.0,
        enemy_hp            = 0.80,
        enemy_gold          = 0.30,
        enemy_unit_count    = 3,
        enemy_base_hp       = 1.0,
        can_attack          = True,
        can_build           = True,
        under_attack        = False,
        ranged_advantage    = False,
        melee_advantage     = False,
        resource_node_controlled = False,
        elapsed_ticks       = 50,
    )


def make_mid_game_state() -> GameState:
    """
    Synthetic GameState representing a mid-game situation
    (armies engaged, economy running, some pressure).
    """
    return GameState(
        own_hp              = 0.60,
        own_gold            = 0.50,
        own_unit_count      = 5,
        own_base_hp         = 0.85,
        enemy_hp            = 0.55,
        enemy_gold          = 0.45,
        enemy_unit_count    = 4,
        enemy_base_hp       = 0.80,
        can_attack          = True,
        can_build           = True,
        under_attack        = False,
        ranged_advantage    = False,
        melee_advantage     = True,
        resource_node_controlled = True,
        elapsed_ticks       = 200,
    )


def make_late_game_state() -> GameState:
    """
    Synthetic GameState representing a late-game situation
    (both sides depleted, bases damaged, high urgency).
    """
    return GameState(
        own_hp              = 0.30,
        own_gold            = 0.70,
        own_unit_count      = 2,
        own_base_hp         = 0.50,
        enemy_hp            = 0.25,
        enemy_gold          = 0.60,
        enemy_unit_count    = 2,
        enemy_base_hp       = 0.40,
        can_attack          = True,
        can_build           = True,
        under_attack        = True,
        ranged_advantage    = False,
        melee_advantage     = False,
        resource_node_controlled = False,
        elapsed_ticks       = 450,
    )


# ---------------------------------------------------------------------------
# 3.  Dominant command extraction (voting over multiple states)
# ---------------------------------------------------------------------------

# Use two early-game probes so rush/aggressive scripts get more votes
_PROBE_STATES = [
    make_early_game_state(),
    make_early_game_state(),   # doubled: early-game command is most representative
    make_mid_game_state(),
    make_late_game_state(),
]


def get_dominant_command(script: StrategyScript) -> str:
    """
    Poll the script against multiple representative game states and
    return the most frequently issued command (majority vote).

    This gives a more stable bot mapping than a single evaluation.

    Returns
    -------
    str
        The most common command name, or DEFAULT_BOT on failure.
    """
    from collections import Counter
    votes: list[str] = []

    for state in _PROBE_STATES:
        try:
            cmd = script.execute(state)
            if cmd and cmd in COMMAND_TO_BOT:
                votes.append(cmd)
        except Exception:
            pass  # malformed AST — skip this probe

    if not votes:
        return "no_op"

    # Return the command that appeared most often
    most_common, _ = Counter(votes).most_common(1)[0]
    return most_common


def script_to_bot(script: StrategyScript) -> str:
    """
    Determine the MicroRTS Java bot type that best represents
    the behaviour encoded in a GESY StrategyScript.

    Parameters
    ----------
    script : StrategyScript
        A fully initialised GESY strategy script.

    Returns
    -------
    str
        A Java bot name accepted by BatchGameRunner (e.g. "WorkerRush").
    """
    cmd = get_dominant_command(script)
    return COMMAND_TO_BOT.get(cmd, DEFAULT_BOT)


# ---------------------------------------------------------------------------
# 4.  Standalone test / mapping display
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  GESY → MicroRTS Command Mapping Table")
    print("=" * 55)
    print(f"  {'GESY Command':<22s}  →  {'MicroRTS Bot'}")
    print("  " + "─" * 50)
    for cmd, bot in COMMAND_TO_BOT.items():
        print(f"  {cmd:<22s}  →  {bot}")
    print("=" * 55)

    # Quick smoke test: build a simple script and map it
    try:
        from gesy.core.ast_nodes import LeafAction, StrategyScript
        simple = StrategyScript(name="TestRush", root=LeafAction("rush"))
        bot = script_to_bot(simple)
        print(f"\n  [Test] Script 'rush' → bot: {bot}")
        assert bot == "WorkerRush", f"Unexpected: {bot}"
        print("  [Test] PASSED ✓")
    except Exception as e:
        print(f"  [Test] FAILED: {e}")
    print()
