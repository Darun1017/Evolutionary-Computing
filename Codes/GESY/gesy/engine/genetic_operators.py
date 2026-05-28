"""
GESY Framework — Module 5: Genetic Operators
=============================================
Implements Steps 2–5 of the GESY pipeline:

  Step 2 — Co-evolutionary fitness evaluation (with bloat penalty)
  Step 3 — Tournament selection
  Step 4 — Type-safe subtree crossover
  Step 5 — Type-safe node / subtree mutation

All genetic operators preserve the STGP type invariant: ActionNode subtrees
are only ever swapped with ActionNode subtrees, ConditionNode subtrees only
with ConditionNode subtrees.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from gesy.core.ast_nodes import (
    ActionNode,
    ASTNode,
    ConditionNode,
    GameState,
    NodeType,
    StrategyScript,
)
from gesy.core.dsl_primitives import random_action_tree, random_condition_tree
from gesy.core.simulator import simulate_match


# ---------------------------------------------------------------------------
# Step 2: Co-evolutionary Fitness Evaluation
# ---------------------------------------------------------------------------

# Tunable constants
BLOAT_PENALTY_FACTOR: float = 0.002    # per extra node above threshold
BLOAT_THRESHOLD:       int  = 25       # nodes below this → no penalty
MATCHES_PER_SCRIPT:    int  = 6        # how many opponents each script faces
WIN_SCORE:   float = 1.0
DRAW_SCORE:  float = 0.4
LOSS_SCORE:  float = 0.0


def compute_fitness(script: StrategyScript) -> float:
    """
    Raw win-rate score minus a bloat penalty.
    Called after all matches for this script have been tallied.
    """
    total_matches = script.wins + script.losses + script.draws
    if total_matches == 0:
        return 0.0

    raw_score = (
        script.wins  * WIN_SCORE  +
        script.draws * DRAW_SCORE +
        script.losses * LOSS_SCORE
    ) / total_matches

    # Bloat control penalty
    n_nodes = script.node_count()
    excess  = max(0, n_nodes - BLOAT_THRESHOLD)
    penalty = excess * BLOAT_PENALTY_FACTOR

    return max(0.0, raw_score - penalty)


def _run_match_task(
    script_a: StrategyScript,
    script_b: StrategyScript,
    match_id: int,
) -> tuple[str, str, str]:
    """
    Worker task: simulate one match, return (winner_name, loser_name, 'draw'|'decisive').
    Runs in a thread from the pool.
    """
    result = simulate_match(script_a, script_b, seed=match_id)
    return (result.winner, result.loser, "draw" if result.draw else "decisive")


def evaluate_population(
    population:  list[StrategyScript],
    n_workers:   int = 4,
    seed_offset: int = 0,
) -> list[StrategyScript]:
    """
    Step 2: Evaluate every script by running it against random opponents.

    Args:
        population:  Current generation of scripts.
        n_workers:   Thread-pool size for parallel match execution.
        seed_offset: Added to match IDs for seed diversity across generations.

    Returns:
        Same population list with fitness / W/L/D updated in-place.
    """
    # Reset match statistics
    for s in population:
        s.wins = s.losses = s.draws = 0
        s.fitness = 0.0

    n = len(population)
    tasks: list[tuple[StrategyScript, StrategyScript, int]] = []

    # For each script, sample MATCHES_PER_SCRIPT random opponents (no self-play)
    for idx, script in enumerate(population):
        opponents = random.sample(
            [p for j, p in enumerate(population) if j != idx],
            k=min(MATCHES_PER_SCRIPT, n - 1),
        )
        for opp_idx, opp in enumerate(opponents):
            match_id = seed_offset * 10000 + idx * 1000 + opp_idx
            tasks.append((script, opp, match_id))

    # Execute matches in parallel
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_run_match_task, a, b, mid): (a, b)
            for a, b, mid in tasks
        }
        for future in as_completed(futures):
            script_a, script_b = futures[future]
            winner_name, loser_name, match_type = future.result()

            if winner_name == script_a.name:
                script_a.wins   += 1
                script_b.losses += 1
            elif winner_name == script_b.name:
                script_b.wins   += 1
                script_a.losses += 1
            else:
                script_a.draws += 1
                script_b.draws += 1

    # Compute final fitness
    for s in population:
        s.fitness = compute_fitness(s)

    return population


# ---------------------------------------------------------------------------
# Step 3: Tournament Selection
# ---------------------------------------------------------------------------

def tournament_select(
    population:       list[StrategyScript],
    tournament_size:  int = 3,
) -> StrategyScript:
    """
    Pick `tournament_size` scripts at random; return the fittest.
    Classic k-tournament selection — no fitness-proportionate bias.
    """
    contestants = random.sample(population, k=min(tournament_size, len(population)))
    return max(contestants, key=lambda s: s.fitness)


# ---------------------------------------------------------------------------
# Step 4: Strongly-Typed Subtree Crossover
# ---------------------------------------------------------------------------

def _collect_typed_nodes(root: ASTNode, target_type: NodeType) -> list[ASTNode]:
    """Flatten all nodes of the given type from the subtree."""
    return root.collect_nodes(target_type)


def _replace_node_in_tree(
    root:     ActionNode,
    old_node: ASTNode,
    new_node: ASTNode,
) -> ActionNode:
    """
    Return a new tree where the first occurrence of `old_node`
    (by identity) is replaced by `new_node`.  Works recursively.
    """
    if root is old_node:
        return new_node  # type: ignore[return-value]

    cloned = root.clone()
    _replace_in_subtree(cloned, old_node, new_node)
    return cloned  # type: ignore[return-value]


def _replace_in_subtree(node: ASTNode, old: ASTNode, new: ASTNode) -> bool:
    """
    Mutate `node._children` in-place: replace the first child matching
    `old` by identity, or recurse.  Returns True if replacement was made.
    """
    for i, child in enumerate(node._children):
        if child is old:
            node._children[i] = new
            return True
        if _replace_in_subtree(child, old, new):
            return True
    return False


def crossover(
    parent_a: StrategyScript,
    parent_b: StrategyScript,
    child_name: str,
    generation: int,
) -> StrategyScript:
    """
    Step 4: Produce one offspring via subtree crossover.

    1. Choose a random node type (ACTION or CONDITION) uniformly.
    2. Sample one crossover point from each parent of that type.
    3. Graft the subtree from parent_b into a clone of parent_a.

    If no valid crossover point exists, return a clone of parent_a unchanged.
    """
    child_root = parent_a.root.clone()

    # Decide which type of subtree to swap
    target_type = random.choice([NodeType.ACTION, NodeType.CONDITION])

    nodes_a = _collect_typed_nodes(child_root, target_type)
    nodes_b = _collect_typed_nodes(parent_b.root, target_type)

    if not nodes_a or not nodes_b:
        # Fallback: straight clone
        return StrategyScript(
            name       = child_name,
            root       = child_root,
            generation = generation,
        )

    # Pick crossover points
    point_a = random.choice(nodes_a)
    point_b = random.choice(nodes_b).clone()

    # Replace point_a in child with point_b
    if child_root is point_a:
        # Crossover at root — only valid if types match at root level
        if isinstance(point_b, ActionNode):
            child_root = point_b  # type: ignore[assignment]
    else:
        _replace_in_subtree(child_root, point_a, point_b)

    return StrategyScript(
        name       = child_name,
        root       = child_root,
        generation = generation,
    )


# ---------------------------------------------------------------------------
# Step 5: Mutation
# ---------------------------------------------------------------------------

def mutate(
    script:          StrategyScript,
    mutation_rate:   float = 0.25,
    max_new_depth:   int   = 2,
) -> StrategyScript:
    """
    Step 5: Mutate one or more nodes in the script's AST.

    Two mutation operators are applied probabilistically:
      a) Node replacement  — swap a leaf action or condition for another of the same type
      b) Subtree regrowth  — replace an internal subtree with a new random one of the same type

    The STGP invariant is maintained: only same-type nodes are ever swapped.
    """
    mutated = script.clone(new_name=script.name + "_mut")
    root    = mutated.root

    # ---- a) Action leaf replacement
    if random.random() < mutation_rate:
        from gesy.core.ast_nodes import LeafAction
        from gesy.core.dsl_primitives import random_action
        action_leaves = [n for n in root.collect_nodes(NodeType.ACTION)
                         if isinstance(n, LeafAction)]
        if action_leaves:
            target = random.choice(action_leaves)
            replacement = random_action()
            _replace_in_subtree(root, target, replacement)

    # ---- b) Condition subtree regrowth
    if random.random() < mutation_rate:
        cond_nodes = root.collect_nodes(NodeType.CONDITION)
        if cond_nodes:
            target      = random.choice(cond_nodes)
            replacement = random_condition_tree(max_depth=max_new_depth)
            if root is target:
                pass  # Can't replace root condition directly here (it's buried in ITE)
            else:
                _replace_in_subtree(root, target, replacement)

    # ---- c) Action subtree regrowth (less frequent)
    if random.random() < mutation_rate * 0.4:
        from gesy.core.ast_nodes import IfThenElseNode
        ite_nodes = [n for n in root.collect_nodes(NodeType.ACTION)
                     if isinstance(n, IfThenElseNode)]
        if ite_nodes:
            target      = random.choice(ite_nodes)
            replacement = random_action_tree(max_depth=max_new_depth)
            if root is target:
                mutated.root = replacement
            else:
                _replace_in_subtree(root, target, replacement)

    return mutated
