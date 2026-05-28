"""
GESY Framework — Module 1: AST Node Structures & DSL Execution Engine
======================================================================
Implements Strongly Typed Genetic Programming (STGP) with a two-tier type
system: ActionNode (imperative effects) and ConditionNode (boolean predicates).
Cross-over and mutation are type-safe — they only ever swap nodes of matching types.
"""

from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# 1. Domain type system
# ---------------------------------------------------------------------------

class NodeType(Enum):
    ACTION    = auto()   # Imperative — produces a side-effect (game command)
    CONDITION = auto()   # Boolean    — evaluates a predicate on game state


# ---------------------------------------------------------------------------
# 2. Game state object (read-only snapshot passed to every DSL evaluation)
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """
    Snapshot of the battlefield visible to one player's strategy script.

    All numeric values are normalised to [0.0, 1.0] where appropriate so
    that thresholds in evolved scripts generalise across map sizes.
    """
    # Own resources
    own_hp:          float   # 0.0 (dead) – 1.0 (full health)
    own_gold:        float   # normalised gold count
    own_unit_count:  int     # number of alive own units
    own_base_hp:     float   # base / nexus health

    # Enemy resources
    enemy_hp:        float
    enemy_gold:      float
    enemy_unit_count: int
    enemy_base_hp:   float

    # Tactical flags (set by the simulator)
    can_attack:      bool    # any own unit in range of an enemy
    can_build:       bool    # enough gold for at least one unit type
    under_attack:    bool    # enemy unit within our base perimeter
    ranged_advantage: bool   # own ranged > enemy ranged units
    melee_advantage:  bool   # own melee  > enemy melee  units

    # Economy / map
    resource_node_controlled: bool   # we hold the central resource node
    elapsed_ticks:            int    # age of the match (for timing strategies)

    # Simulation bookkeeping (not used by scripts directly)
    winner: Optional[str] = None     # set at end of match


# ---------------------------------------------------------------------------
# 3. Base AST node
# ---------------------------------------------------------------------------

class ASTNode(ABC):
    """Abstract base for every node in a strategy AST."""

    node_type: NodeType          # CLASS-level attribute; subclasses set it

    def __init__(self) -> None:
        self._children: list[ASTNode] = []

    # ------------------------------------------------------------------
    # Core protocol
    # ------------------------------------------------------------------

    @abstractmethod
    def evaluate(self, state: GameState) -> Any:
        """
        Execute / evaluate this node.
        ActionNodes    → return a string command (or None for no-op)
        ConditionNodes → return bool
        """

    @abstractmethod
    def to_string(self, indent: int = 0) -> str:
        """Human-readable pretty-print of the subtree."""

    # ------------------------------------------------------------------
    # Tree utilities
    # ------------------------------------------------------------------

    @property
    def children(self) -> list[ASTNode]:
        return self._children

    def node_count(self) -> int:
        """Total number of nodes in this subtree (for bloat penalty)."""
        return 1 + sum(c.node_count() for c in self._children)

    def depth(self) -> int:
        if not self._children:
            return 0
        return 1 + max(c.depth() for c in self._children)

    def collect_nodes(self, target_type: NodeType) -> list[ASTNode]:
        """BFS collection of all nodes of a given type within the subtree."""
        result = []
        queue = [self]
        while queue:
            node = queue.pop(0)
            if node.node_type == target_type:
                result.append(node)
            queue.extend(node._children)
        return result

    def clone(self) -> ASTNode:
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# 4. Condition (boolean predicate) nodes
# ---------------------------------------------------------------------------

class ConditionNode(ASTNode, ABC):
    node_type = NodeType.CONDITION

    @abstractmethod
    def evaluate(self, state: GameState) -> bool: ...


class SimpleCondition(ConditionNode):
    """
    Leaf condition: wraps a named predicate function.
    e.g.  SimpleCondition("hp_low", lambda s: s.own_hp < 0.3)
    """
    def __init__(self, name: str, predicate: Callable[[GameState], bool],
                 threshold: float = 0.5) -> None:
        super().__init__()
        self.name      = name
        self.predicate = predicate
        self.threshold = threshold   # mutable — a mutation target

    def evaluate(self, state: GameState) -> bool:
        return self.predicate(state)

    def to_string(self, indent: int = 0) -> str:
        return "  " * indent + f"COND({self.name})"


class AndCondition(ConditionNode):
    """Logical AND of exactly two child ConditionNodes."""
    def __init__(self, left: ConditionNode, right: ConditionNode) -> None:
        super().__init__()
        self._children = [left, right]

    def evaluate(self, state: GameState) -> bool:
        return self._children[0].evaluate(state) and self._children[1].evaluate(state)

    def to_string(self, indent: int = 0) -> str:
        pad = "  " * indent
        lines = [pad + "AND("]
        for c in self._children:
            lines.append(c.to_string(indent + 1))
        lines.append(pad + ")")
        return "\n".join(lines)


class OrCondition(ConditionNode):
    """Logical OR of exactly two child ConditionNodes."""
    def __init__(self, left: ConditionNode, right: ConditionNode) -> None:
        super().__init__()
        self._children = [left, right]

    def evaluate(self, state: GameState) -> bool:
        return self._children[0].evaluate(state) or self._children[1].evaluate(state)

    def to_string(self, indent: int = 0) -> str:
        pad = "  " * indent
        lines = [pad + "OR("]
        for c in self._children:
            lines.append(c.to_string(indent + 1))
        lines.append(pad + ")")
        return "\n".join(lines)


class NotCondition(ConditionNode):
    """Logical NOT of a single child ConditionNode."""
    def __init__(self, child: ConditionNode) -> None:
        super().__init__()
        self._children = [child]

    def evaluate(self, state: GameState) -> bool:
        return not self._children[0].evaluate(state)

    def to_string(self, indent: int = 0) -> str:
        pad = "  " * indent
        return pad + "NOT(\n" + self._children[0].to_string(indent + 1) + "\n" + pad + ")"


# ---------------------------------------------------------------------------
# 5. Action nodes
# ---------------------------------------------------------------------------

class ActionNode(ASTNode, ABC):
    node_type = NodeType.ACTION

    @abstractmethod
    def evaluate(self, state: GameState) -> Optional[str]:
        """Return a command string or None (= no-op)."""


class LeafAction(ActionNode):
    """
    Atomic action (leaf): e.g. 'attack', 'build_melee', 'retreat'.
    The simulator interprets the returned command string.
    """
    def __init__(self, command: str) -> None:
        super().__init__()
        self.command = command

    def evaluate(self, state: GameState) -> str:
        return self.command

    def to_string(self, indent: int = 0) -> str:
        return "  " * indent + f"ACTION({self.command})"


class IfThenElseNode(ActionNode):
    """
    Branching node: IF <condition> THEN <action> ELSE <action>.
    This is the main structural node of a strategy script.
    """
    def __init__(
        self,
        condition: ConditionNode,
        then_branch: ActionNode,
        else_branch: ActionNode,
    ) -> None:
        super().__init__()
        # Children ordered: [condition, then_branch, else_branch]
        self._children = [condition, then_branch, else_branch]

    @property
    def condition(self)    -> ConditionNode: return self._children[0]  # type: ignore[return-value]
    @property
    def then_branch(self)  -> ActionNode:    return self._children[1]  # type: ignore[return-value]
    @property
    def else_branch(self)  -> ActionNode:    return self._children[2]  # type: ignore[return-value]

    def evaluate(self, state: GameState) -> Optional[str]:
        if self.condition.evaluate(state):
            return self.then_branch.evaluate(state)
        return self.else_branch.evaluate(state)

    def to_string(self, indent: int = 0) -> str:
        pad = "  " * indent
        lines = [
            pad + "IF",
            self.condition.to_string(indent + 1),
            pad + "THEN",
            self.then_branch.to_string(indent + 1),
            pad + "ELSE",
            self.else_branch.to_string(indent + 1),
        ]
        return "\n".join(lines)


class SequenceNode(ActionNode):
    """
    Execute child actions in order; return the result of the first
    non-None command (priority list / fallback chain).
    """
    def __init__(self, actions: list[ActionNode]) -> None:
        super().__init__()
        self._children = list(actions)

    def evaluate(self, state: GameState) -> Optional[str]:
        for child in self._children:
            result = child.evaluate(state)
            if result is not None:
                return result
        return None

    def to_string(self, indent: int = 0) -> str:
        pad = "  " * indent
        lines = [pad + "SEQ("]
        for c in self._children:
            lines.append(c.to_string(indent + 1))
        lines.append(pad + ")")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Strategy Script wrapper (the individual in the GP population)
# ---------------------------------------------------------------------------

@dataclass
class StrategyScript:
    """
    One individual in the population: a named root ActionNode (the AST)
    plus its fitness score and provenance metadata.
    """
    name:       str
    root:       ActionNode
    fitness:    float = 0.0
    generation: int   = 0
    wins:       int   = 0
    losses:     int   = 0
    draws:      int   = 0

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, state: GameState) -> Optional[str]:
        """Run the script against a game state; return a command string."""
        try:
            return self.root.evaluate(state)
        except Exception:
            return None   # malformed AST → no-op, not a crash

    # ------------------------------------------------------------------
    # Tree metrics
    # ------------------------------------------------------------------

    def node_count(self) -> int:
        return self.root.node_count()

    def depth(self) -> int:
        return self.root.depth()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def to_string(self) -> str:
        header = (
            f"=== Strategy: {self.name} ===\n"
            f"  Fitness: {self.fitness:.4f}  |  Nodes: {self.node_count()}"
            f"  |  Depth: {self.depth()}\n"
            f"  W/L/D: {self.wins}/{self.losses}/{self.draws}\n"
            f"  Gen: {self.generation}\n"
            f"--- AST ---\n"
        )
        return header + self.root.to_string()

    def clone(self, new_name: str = "") -> StrategyScript:
        cloned = StrategyScript(
            name       = new_name or self.name + "_clone",
            root       = self.root.clone(),
            fitness    = 0.0,
            generation = self.generation,
        )
        return cloned
