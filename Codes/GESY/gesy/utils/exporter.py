"""
GESY Framework — Module 7: Script Exporter
===========================================
Utility to serialise the best-evolved strategy into human-readable formats:
  • Pretty-printed AST tree (plain text)
  • Pseudo-code representation
  • JSON export for external tooling / replay

Also provides a simple loader to reconstruct a StrategyScript from JSON.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

from gesy.core.ast_nodes import (
    ASTNode,
    ActionNode,
    ConditionNode,
    GameState,
    IfThenElseNode,
    LeafAction,
    NodeType,
    SequenceNode,
    StrategyScript,
    AndCondition,
    OrCondition,
    NotCondition,
    SimpleCondition,
)


# ---------------------------------------------------------------------------
# 1. Human-readable AST display
# ---------------------------------------------------------------------------

def export_pretty(script: StrategyScript) -> str:
    """
    Return a nicely formatted, colour-annotated string representation
    of the script — suitable for terminal output or a .txt file.
    """
    lines = [
        "╔" + "═" * 58 + "╗",
        f"║  Strategy Script: {script.name:<38s}║",
        f"║  Fitness : {script.fitness:.4f}  │  "
        f"Nodes: {script.node_count():<4d} │  "
        f"Depth: {script.depth():<4d}    ║",
        f"║  W/L/D  : {script.wins}/{script.losses}/{script.draws:<8d}  │  "
        f"Generation: {script.generation:<14d}║",
        "╠" + "═" * 58 + "╣",
        "║  ABSTRACT SYNTAX TREE                                    ║",
        "╠" + "═" * 58 + "╣",
    ]
    ast_text = script.root.to_string(indent=0)
    for line in ast_text.splitlines():
        padded = f"║  {line:<56s}║"
        lines.append(padded)
    lines.append("╚" + "═" * 58 + "╝")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Pseudo-code exporter
# ---------------------------------------------------------------------------

def _node_to_pseudocode(node: ASTNode, indent: int = 0) -> str:
    pad = "    " * indent

    if isinstance(node, LeafAction):
        return pad + node.command.upper().replace("_", " ")

    if isinstance(node, SimpleCondition):
        return pad + node.name.upper().replace("_", " ")

    if isinstance(node, AndCondition):
        left  = _node_to_pseudocode(node.children[0], 0).strip()
        right = _node_to_pseudocode(node.children[1], 0).strip()
        return pad + f"({left} AND {right})"

    if isinstance(node, OrCondition):
        left  = _node_to_pseudocode(node.children[0], 0).strip()
        right = _node_to_pseudocode(node.children[1], 0).strip()
        return pad + f"({left} OR {right})"

    if isinstance(node, NotCondition):
        inner = _node_to_pseudocode(node.children[0], 0).strip()
        return pad + f"NOT ({inner})"

    if isinstance(node, IfThenElseNode):
        cond      = _node_to_pseudocode(node.condition,   0).strip()
        then_part = _node_to_pseudocode(node.then_branch, indent + 1)
        else_part = _node_to_pseudocode(node.else_branch, indent + 1)
        lines = [
            pad + f"IF {cond}:",
            then_part,
            pad + "ELSE:",
            else_part,
        ]
        return "\n".join(lines)

    if isinstance(node, SequenceNode):
        lines = [pad + "DO IN SEQUENCE:"]
        for child in node.children:
            lines.append(_node_to_pseudocode(child, indent + 1))
        return "\n".join(lines)

    return pad + f"[UNKNOWN NODE: {type(node).__name__}]"


def export_pseudocode(script: StrategyScript) -> str:
    """
    Produce readable pseudo-code resembling a high-level strategy description.
    """
    header = (
        f"# Evolved Strategy: {script.name}\n"
        f"# Fitness: {script.fitness:.4f}  |  Nodes: {script.node_count()}"
        f"  |  Depth: {script.depth()}\n"
        f"# W/L/D: {script.wins}/{script.losses}/{script.draws}"
        f"  |  Generation: {script.generation}\n"
        "# " + "─" * 50 + "\n"
        "STRATEGY BEGIN\n"
    )
    body   = _node_to_pseudocode(script.root, indent=1)
    footer = "\nSTRATEGY END\n"
    return header + body + footer


# ---------------------------------------------------------------------------
# 3. JSON serialiser / deserialiser
# ---------------------------------------------------------------------------

def _node_to_dict(node: ASTNode) -> dict[str, Any]:
    d: dict[str, Any] = {"type": type(node).__name__}

    if isinstance(node, LeafAction):
        d["command"] = node.command

    elif isinstance(node, SimpleCondition):
        d["name"]      = node.name
        d["threshold"] = node.threshold

    elif isinstance(node, (AndCondition, OrCondition, NotCondition, SequenceNode)):
        d["children"] = [_node_to_dict(c) for c in node.children]

    elif isinstance(node, IfThenElseNode):
        d["condition"]   = _node_to_dict(node.condition)
        d["then_branch"] = _node_to_dict(node.then_branch)
        d["else_branch"] = _node_to_dict(node.else_branch)

    else:
        d["children"] = [_node_to_dict(c) for c in node.children]

    return d


def _dict_to_node(d: dict[str, Any]) -> ASTNode:
    t = d["type"]

    if t == "LeafAction":
        return LeafAction(d["command"])

    if t == "SimpleCondition":
        from gesy.core.dsl_primitives import condition_by_name
        node = condition_by_name(d["name"])
        if node is None:
            from gesy.core.ast_nodes import SimpleCondition
            node = SimpleCondition(d["name"], lambda s: False)
        return node

    if t == "AndCondition":
        children = [_dict_to_node(c) for c in d["children"]]
        return AndCondition(children[0], children[1])  # type: ignore[arg-type]

    if t == "OrCondition":
        children = [_dict_to_node(c) for c in d["children"]]
        return OrCondition(children[0], children[1])   # type: ignore[arg-type]

    if t == "NotCondition":
        children = [_dict_to_node(c) for c in d["children"]]
        return NotCondition(children[0])                # type: ignore[arg-type]

    if t == "IfThenElseNode":
        cond     = _dict_to_node(d["condition"])
        then_br  = _dict_to_node(d["then_branch"])
        else_br  = _dict_to_node(d["else_branch"])
        return IfThenElseNode(cond, then_br, else_br)   # type: ignore[arg-type]

    if t == "SequenceNode":
        children = [_dict_to_node(c) for c in d["children"]]
        return SequenceNode(children)                    # type: ignore[arg-type]

    raise ValueError(f"Unknown node type in JSON: {t}")


def export_json(script: StrategyScript) -> str:
    """Serialise a StrategyScript to a JSON string."""
    doc = {
        "name":       script.name,
        "fitness":    script.fitness,
        "generation": script.generation,
        "wins":       script.wins,
        "losses":     script.losses,
        "draws":      script.draws,
        "node_count": script.node_count(),
        "depth":      script.depth(),
        "ast":        _node_to_dict(script.root),
    }
    return json.dumps(doc, indent=2)


def import_json(json_str: str) -> StrategyScript:
    """Reconstruct a StrategyScript from a JSON string."""
    doc  = json.loads(json_str)
    root = _dict_to_node(doc["ast"])
    return StrategyScript(
        name       = doc["name"],
        root       = root,        # type: ignore[arg-type]
        fitness    = doc.get("fitness", 0.0),
        generation = doc.get("generation", 0),
        wins       = doc.get("wins", 0),
        losses     = doc.get("losses", 0),
        draws      = doc.get("draws", 0),
    )


# ---------------------------------------------------------------------------
# 4. File export convenience wrappers
# ---------------------------------------------------------------------------

def save_to_file(script: StrategyScript, output_dir: str = "/mnt/user-data/outputs") -> dict[str, str]:
    """
    Save the best script in all three formats.
    Returns a dict of {format: filepath}.
    """
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    safe_name = script.name.replace(" ", "_").replace("/", "-")

    paths: dict[str, str] = {}

    # Pretty AST
    p1 = base / f"{safe_name}_ast.txt"
    p1.write_text(export_pretty(script), encoding="utf-8")
    paths["pretty"] = str(p1)

    # Pseudo-code
    p2 = base / f"{safe_name}_pseudocode.txt"
    p2.write_text(export_pseudocode(script), encoding="utf-8")
    paths["pseudocode"] = str(p2)

    # JSON
    p3 = base / f"{safe_name}.json"
    p3.write_text(export_json(script), encoding="utf-8")
    paths["json"] = str(p3)

    return paths
