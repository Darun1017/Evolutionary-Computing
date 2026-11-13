"""
Official Example Bot Implementations in Python
==============================================
Exact deterministic reimplementations of the official Java example bots:
- BullyBot
- RageBot
- ProspectorBot
- DualBot
- RandomBot (with seed support)

These run directly on `GameState` inside `sim.py` for blazing-fast evaluation,
while also being binary-equivalent to the official `.jar` bots.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple, Callable
from planetwars.sim import GameState, Order, Planet


def bully_bot(state: GameState, player: int) -> List[Tuple[int, int, int]]:
    """
    BullyBot logic (BullyBot.java):
    1. If we have any fleet in flight, do nothing.
    2. Source: my planet with maximum ship count (first on ties).
    3. Dest: not-my planet with maximum 1.0 / (1 + ships) (first on ties).
    4. Send source.num_ships // 2.
    """
    if len(state.my_fleets(player)) >= 1:
        return []

    my_planets = state.my_planets(player)
    if not my_planets:
        return []

    # Java: score > sourceScore retains first encountered on tie
    source = max(my_planets, key=lambda p: p.num_ships)

    not_my = state.not_my_planets(player)
    if not_my:
        dest = max(not_my, key=lambda p: 1.0 / (1.0 + p.num_ships))
    else:
        dest = None

    if source is not None and dest is not None:
        num_ships = source.num_ships // 2
        if num_ships > 0:
            return [(source.planet_id, dest.planet_id, num_ships)]
    return []


def rage_bot(state: GameState, player: int) -> List[Tuple[int, int, int]]:
    """
    RageBot logic (RageBot.java):
    For each planet I own:
      if ships >= 10 * growth_rate:
        send all ships to nearest enemy planet (first on ties).
    """
    orders = []
    enemy_planets = state.enemy_planets(player)
    if not enemy_planets:
        return []

    for source in state.my_planets(player):
        if source.num_ships < 10 * source.growth_rate:
            continue

        # Java: dist < bestDistance retains first encountered in iteration order (smallest planet_id)
        best_dest = min(
            enemy_planets,
            key=lambda p: (state.distance(source.planet_id, p.planet_id), p.planet_id),
        )
        if best_dest is not None and source.num_ships > 0:
            orders.append((source.planet_id, best_dest.planet_id, source.num_ships))

    return orders


def prospector_bot(state: GameState, player: int) -> List[Tuple[int, int, int]]:
    """
    ProspectorBot logic (ProspectorBot.java):
    1. If we have any fleet in flight, do nothing.
    2. Source: max (ships / (1 + growth)) (first on ties).
    3. Dest: not-my planet with max ((1 + growth) / max(1, ships)) (first on ties).
    4. Send source.num_ships // 2.
    """
    if len(state.my_fleets(player)) >= 1:
        return []

    my_planets = state.my_planets(player)
    if not my_planets:
        return []

    source = max(
        my_planets,
        key=lambda p: float(p.num_ships) / (1.0 + p.growth_rate),
    )

    not_my = state.not_my_planets(player)
    if not_my:
        dest = max(
            not_my,
            key=lambda p: float(1 + p.growth_rate) / float(max(1, p.num_ships)),
        )
    else:
        dest = None

    if source is not None and dest is not None:
        num_ships = source.num_ships // 2
        if num_ships > 0:
            return [(source.planet_id, dest.planet_id, num_ships)]
    return []


def dual_bot(state: GameState, player: int) -> List[Tuple[int, int, int]]:
    """
    DualBot logic (DualBot.java):
    Adapts fleet cap and target selection based on relative ship count and production.
    """
    rival = 2 if player == 1 else 1
    my_ships = state.num_ships(player)
    enemy_ships = state.num_ships(rival)
    my_prod = state.production(player)
    enemy_prod = state.production(rival)

    attack_mode = False
    if my_ships > enemy_ships:
        if my_prod > enemy_prod:
            num_fleets = 1
            attack_mode = True
        else:
            num_fleets = 3
    else:
        if my_prod > enemy_prod:
            num_fleets = 1
        else:
            num_fleets = 5

    if len(state.my_fleets(player)) >= num_fleets:
        return []

    my_planets = state.my_planets(player)
    if not my_planets:
        return []

    source = max(
        my_planets,
        key=lambda p: float(p.num_ships) / (1.0 + p.growth_rate),
    )

    candidates = state.enemy_planets(player) if attack_mode else state.not_my_planets(player)
    if candidates:
        dest = max(
            candidates,
            key=lambda p: float(1 + p.growth_rate) / float(max(1, p.num_ships)),
        )
    else:
        dest = None

    if source is not None and dest is not None:
        num_ships = source.num_ships // 2
        if num_ships > 0:
            return [(source.planet_id, dest.planet_id, num_ships)]
    return []


def make_random_bot(seed: Optional[int] = None) -> Callable[[GameState, int], List[Tuple[int, int, int]]]:
    """RandomBot logic (RandomBot.java): picks random source and dest."""
    rng = random.Random(seed)

    def _random_bot(state: GameState, player: int) -> List[Tuple[int, int, int]]:
        if len(state.my_fleets(player)) >= 1:
            return []

        my_planets = state.my_planets(player)
        if not my_planets:
            return []

        source = rng.choice(my_planets)
        all_planets = state.planets
        dest = rng.choice(all_planets)

        if source.planet_id != dest.planet_id and source.num_ships > 1:
            return [(source.planet_id, dest.planet_id, source.num_ships // 2)]
        return []

    return _random_bot


# Registry of opponents
OPPONENT_BOTS: Dict[str, Callable[[GameState, int], List[Tuple[int, int, int]]]] = {
    "BullyBot": bully_bot,
    "RageBot": rage_bot,
    "ProspectorBot": prospector_bot,
    "DualBot": dual_bot,
    "RandomBot": make_random_bot(seed=42),
}
