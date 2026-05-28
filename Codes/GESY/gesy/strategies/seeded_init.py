"""
GESY Framework — Module 4: Seeded Initialisation
=================================================
Step 1 of the GESY pipeline.  Instead of a fully random initial population,
we seed the first generation with three hand-crafted expert strategies:
  • Rush       — aggressive, early-pressure, melee-heavy
  • Defensive  — turtle, fortify, survive to late game
  • Ranged     — economy build, ranged units, harass and kite

These are each described as a human-readable specification dictionary and
are converted deterministically into AST trees by `build_*` functions.

The remainder of the population (N - 3) is filled with random newcomers.
"""

from __future__ import annotations

import random

from gesy.core.ast_nodes import (
    ActionNode,
    ConditionNode,
    IfThenElseNode,
    LeafAction,
    SequenceNode,
    SimpleCondition,
    StrategyScript,
    AndCondition,
    OrCondition,
    NotCondition,
)
from gesy.core.dsl_primitives import (
    condition_by_name,
    random_action_tree,
    random_condition_tree,
)


# ---------------------------------------------------------------------------
# Helper: look up a condition by name (assertion-safe)
# ---------------------------------------------------------------------------

def _cond(name: str) -> SimpleCondition:
    c = condition_by_name(name)
    if c is None:
        raise ValueError(f"Unknown condition name: '{name}'")
    return c


# ---------------------------------------------------------------------------
# Strategy 1 — RUSH
# ---------------------------------------------------------------------------
# Logic:
#   IF (early_game AND can_attack)  THEN rush
#   ELSE IF can_build               THEN build_melee
#   ELSE IF can_attack              THEN attack
#   ELSE                                 hold_position

def build_rush_strategy() -> StrategyScript:
    tree: ActionNode = IfThenElseNode(
        condition   = AndCondition(_cond("early_game"), _cond("can_attack")),
        then_branch = LeafAction("rush"),
        else_branch = IfThenElseNode(
            condition   = _cond("can_build"),
            then_branch = LeafAction("build_melee"),
            else_branch = IfThenElseNode(
                condition   = _cond("can_attack"),
                then_branch = LeafAction("attack"),
                else_branch = LeafAction("hold_position"),
            ),
        ),
    )
    return StrategyScript(name="Rush_Seed", root=tree, generation=0)


# ---------------------------------------------------------------------------
# Strategy 2 — DEFENSIVE
# ---------------------------------------------------------------------------
# Logic:
#   IF under_attack AND base_hp_low THEN fortify
#   ELSE IF under_attack            THEN hold_position
#   ELSE IF hp_low                  THEN retreat
#   ELSE IF can_build               THEN build_melee
#   ELSE IF enemy_base_weak         THEN attack
#   ELSE                                 hold_position

def build_defensive_strategy() -> StrategyScript:
    tree: ActionNode = IfThenElseNode(
        condition   = AndCondition(_cond("under_attack"), _cond("base_hp_low")),
        then_branch = LeafAction("fortify"),
        else_branch = IfThenElseNode(
            condition   = _cond("under_attack"),
            then_branch = LeafAction("hold_position"),
            else_branch = IfThenElseNode(
                condition   = _cond("hp_low"),
                then_branch = LeafAction("retreat"),
                else_branch = IfThenElseNode(
                    condition   = _cond("can_build"),
                    then_branch = LeafAction("build_melee"),
                    else_branch = IfThenElseNode(
                        condition   = _cond("enemy_base_weak"),
                        then_branch = LeafAction("attack"),
                        else_branch = LeafAction("hold_position"),
                    ),
                ),
            ),
        ),
    )
    return StrategyScript(name="Defensive_Seed", root=tree, generation=0)


# ---------------------------------------------------------------------------
# Strategy 3 — RANGED UNIT
# ---------------------------------------------------------------------------
# Logic:
#   IF (NOT controls_resource)      THEN gather_resource
#   ELSE IF can_build AND gold_rich THEN build_ranged
#   ELSE IF ranged_advantage        THEN harass
#   ELSE IF enemy_hp_low            THEN attack
#   ELSE IF hp_low                  THEN retreat
#   ELSE                                 build_ranged

def build_ranged_strategy() -> StrategyScript:
    tree: ActionNode = IfThenElseNode(
        condition   = NotCondition(_cond("controls_resource")),
        then_branch = LeafAction("gather_resource"),
        else_branch = IfThenElseNode(
            condition   = AndCondition(_cond("can_build"), _cond("gold_rich")),
            then_branch = LeafAction("build_ranged"),
            else_branch = IfThenElseNode(
                condition   = _cond("ranged_advantage"),
                then_branch = LeafAction("harass"),
                else_branch = IfThenElseNode(
                    condition   = _cond("enemy_hp_low"),
                    then_branch = LeafAction("attack"),
                    else_branch = IfThenElseNode(
                        condition   = _cond("hp_low"),
                        then_branch = LeafAction("retreat"),
                        else_branch = LeafAction("build_ranged"),
                    ),
                ),
            ),
        ),
    )
    return StrategyScript(name="Ranged_Seed", root=tree, generation=0)


# ---------------------------------------------------------------------------
# Random newcomer factory
# ---------------------------------------------------------------------------

def build_random_script(name: str, max_depth: int = 4) -> StrategyScript:
    """
    Generate a completely random (but syntactically valid) strategy script.
    Used for Step 7 (Random Newcomers / Invaders).
    """
    root = random_action_tree(max_depth=max_depth)
    return StrategyScript(name=name, root=root, generation=0)


# ---------------------------------------------------------------------------
# Population initialiser
# ---------------------------------------------------------------------------

def initialise_population(
    population_size: int,
    random_seed:     int = 42,
) -> list[StrategyScript]:
    """
    Step 1 — Seeded Initialisation.

    Returns a list of `population_size` StrategyScripts.
    The first three are the hand-crafted seeds; the rest are random newcomers.

    Args:
        population_size: Total N individuals in generation 0.
        random_seed:     For reproducibility of the random portion.

    Returns:
        List[StrategyScript] — the initial population Φ₀.
    """
    random.seed(random_seed)

    seeds: list[StrategyScript] = [
        build_rush_strategy(),
        build_defensive_strategy(),
        build_ranged_strategy(),
    ]

    n_random = max(0, population_size - len(seeds))
    randoms: list[StrategyScript] = [
        build_random_script(name=f"Random_Init_{i:03d}")
        for i in range(n_random)
    ]

    population = seeds + randoms
    print(
        f"[Init] Population of {len(population)} created: "
        f"{len(seeds)} seeded + {len(randoms)} random newcomers."
    )
    return population
