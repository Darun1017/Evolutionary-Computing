"""
GESY Framework — Module 2: DSL Primitive Library
=================================================
Defines the complete vocabulary of actions and conditions available to
evolved scripts.  Random generators here are used by:
  - Seeded initialisation (to build ASTs from text specs)
  - Random newcomer injection
  - Mutation (swapping individual nodes)
"""

from __future__ import annotations

import random
from typing import Optional

from gesy.core.ast_nodes import (
    ActionNode,
    ConditionNode,
    GameState,
    IfThenElseNode,
    LeafAction,
    AndCondition,
    OrCondition,
    NotCondition,
    SequenceNode,
    SimpleCondition,
)


# ---------------------------------------------------------------------------
# 1.  Action vocabulary
# ---------------------------------------------------------------------------
# Each string maps to a command the game simulator interprets.

ALL_ACTIONS: list[str] = [
    "attack",           # Move nearest unit toward nearest enemy and strike
    "retreat",          # Pull all units back toward own base
    "build_melee",      # Spend gold: train a melee unit
    "build_ranged",     # Spend gold: train a ranged unit
    "build_siege",      # Spend gold: train a siege / artillery unit
    "fortify",          # Upgrade base defences (costs gold)
    "rush",             # Aggressive all-out push toward enemy base
    "hold_position",    # Defensive — do not advance, hold current line
    "harass",           # Hit-and-run raid on enemy resource nodes
    "gather_resource",  # Send workers to contested resource node
    "no_op",            # Explicitly skip this tick
]


def random_action() -> LeafAction:
    return LeafAction(random.choice(ALL_ACTIONS))


# ---------------------------------------------------------------------------
# 2.  Condition vocabulary
# ---------------------------------------------------------------------------
# Each entry: (name, lambda GameState → bool)

_CONDITION_SPECS: list[tuple[str, object]] = [
    ("hp_low",              lambda s: s.own_hp < 0.35),
    ("hp_critical",         lambda s: s.own_hp < 0.15),
    ("hp_high",             lambda s: s.own_hp > 0.70),
    ("enemy_hp_low",        lambda s: s.enemy_hp < 0.35),
    ("enemy_hp_critical",   lambda s: s.enemy_hp < 0.15),
    ("gold_rich",           lambda s: s.own_gold > 0.60),
    ("gold_scarce",         lambda s: s.own_gold < 0.20),
    ("outnumbered",         lambda s: s.own_unit_count < s.enemy_unit_count),
    ("overpowered",         lambda s: s.own_unit_count > s.enemy_unit_count * 1.5),
    ("can_attack",          lambda s: s.can_attack),
    ("can_build",           lambda s: s.can_build),
    ("under_attack",        lambda s: s.under_attack),
    ("ranged_advantage",    lambda s: s.ranged_advantage),
    ("melee_advantage",     lambda s: s.melee_advantage),
    ("controls_resource",   lambda s: s.resource_node_controlled),
    ("early_game",          lambda s: s.elapsed_ticks < 100),
    ("late_game",           lambda s: s.elapsed_ticks > 400),
    ("base_hp_low",         lambda s: s.own_base_hp < 0.40),
    ("enemy_base_weak",     lambda s: s.enemy_base_hp < 0.30),
    ("numerically_equal",   lambda s: abs(s.own_unit_count - s.enemy_unit_count) <= 1),
]


def _make_condition(name: str, predicate) -> SimpleCondition:  # noqa: ANN001
    return SimpleCondition(name, predicate)


ALL_CONDITIONS: list[SimpleCondition] = [
    _make_condition(n, p) for n, p in _CONDITION_SPECS
]


def random_condition() -> SimpleCondition:
    spec = random.choice(_CONDITION_SPECS)
    return _make_condition(*spec)


# ---------------------------------------------------------------------------
# 3.  Random tree generators (used by newcomer injection and seeding helpers)
# ---------------------------------------------------------------------------

def random_condition_tree(max_depth: int = 2) -> ConditionNode:
    """
    Recursively build a random boolean condition subtree.
    Leaves are SimpleConditions; internal nodes are And/Or/Not.
    """
    if max_depth <= 0 or random.random() < 0.45:
        return random_condition()

    choice = random.random()
    if choice < 0.40:
        left  = random_condition_tree(max_depth - 1)
        right = random_condition_tree(max_depth - 1)
        return AndCondition(left, right)
    elif choice < 0.75:
        left  = random_condition_tree(max_depth - 1)
        right = random_condition_tree(max_depth - 1)
        return OrCondition(left, right)
    else:
        child = random_condition_tree(max_depth - 1)
        return NotCondition(child)


def random_action_tree(max_depth: int = 3) -> ActionNode:
    """
    Recursively build a random action subtree.
    Leaves are LeafActions; internal nodes are IfThenElse or Sequence.
    """
    if max_depth <= 0 or random.random() < 0.35:
        return random_action()

    if random.random() < 0.70:
        cond      = random_condition_tree(max_depth - 1)
        then_br   = random_action_tree(max_depth - 1)
        else_br   = random_action_tree(max_depth - 1)
        return IfThenElseNode(cond, then_br, else_br)
    else:
        n_children = random.randint(2, 3)
        children   = [random_action_tree(max_depth - 1) for _ in range(n_children)]
        return SequenceNode(children)


# ---------------------------------------------------------------------------
# 4.  Lookup helpers (used by mutation engine)
# ---------------------------------------------------------------------------

_CONDITION_DICT: dict[str, SimpleCondition] = {
    n: _make_condition(n, p) for n, p in _CONDITION_SPECS
}


def condition_by_name(name: str) -> Optional[SimpleCondition]:
    return _CONDITION_DICT.get(name)


def action_by_name(name: str) -> Optional[LeafAction]:
    if name in ALL_ACTIONS:
        return LeafAction(name)
    return None
