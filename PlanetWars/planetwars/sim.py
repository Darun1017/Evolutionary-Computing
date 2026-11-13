"""
Planet Wars Pure-Python Simulator
=================================
Fast, deterministic simulation of the official 2010 Google AI Challenge
Planet Wars game engine matching SPECIFICATION.md exactly.

No subprocesses, no JVM overhead. Designed for high-throughput
evolutionary algorithms (TD-CCEA).
"""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union


# =====================================================================
#  DATA STRUCTURES
# =====================================================================

@dataclass(slots=True)
class Planet:
    """Represents a planet on the map."""
    planet_id: int
    x: float
    y: float
    owner: int          # 0: Neutral, 1: Player 1, 2: Player 2
    num_ships: int
    growth_rate: int

    def copy(self) -> Planet:
        return Planet(
            planet_id=self.planet_id,
            x=self.x,
            y=self.y,
            owner=self.owner,
            num_ships=self.num_ships,
            growth_rate=self.growth_rate,
        )


@dataclass(slots=True)
class Fleet:
    """Represents a fleet in transit between planets."""
    owner: int          # 1: Player 1, 2: Player 2
    num_ships: int
    source_planet: int
    destination_planet: int
    total_trip_length: int
    turns_remaining: int

    def copy(self) -> Fleet:
        return Fleet(
            owner=self.owner,
            num_ships=self.num_ships,
            source_planet=self.source_planet,
            destination_planet=self.destination_planet,
            total_trip_length=self.total_trip_length,
            turns_remaining=self.turns_remaining,
        )


@dataclass(slots=True)
class Order:
    """An order issued by a player to dispatch ships."""
    source_planet: int
    destination_planet: int
    num_ships: int


# =====================================================================
#  DISTANCE CACHE & HELPERS
# =====================================================================

def compute_travel_turns(p1: Planet, p2: Planet) -> int:
    """Official distance metric: Euclidean distance rounded up (ceil)."""
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    return int(math.ceil(math.sqrt(dx * dx + dy * dy)))


# =====================================================================
#  GAME STATE & SIMULATOR
# =====================================================================

class GameState:
    """
    Complete state of a Planet Wars match at a specific turn.
    Designed for fast cloning / branching.
    """
    __slots__ = (
        "planets",
        "fleets",
        "turn",
        "max_turns",
        "winner",
        "game_over",
        "_dist_matrix",
    )

    def __init__(
        self,
        planets: List[Planet],
        fleets: Optional[List[Fleet]] = None,
        turn: int = 0,
        max_turns: int = 200,
        dist_matrix: Optional[List[List[int]]] = None,
    ):
        self.planets: List[Planet] = planets
        self.fleets: List[Fleet] = fleets if fleets is not None else []
        self.turn: int = turn
        self.max_turns: int = max_turns
        self.winner: Optional[int] = None   # None: ongoing, 0: draw, 1: P1, 2: P2
        self.game_over: bool = False

        if dist_matrix is not None:
            self._dist_matrix = dist_matrix
        else:
            n = len(planets)
            self._dist_matrix = [
                [compute_travel_turns(planets[i], planets[j]) for j in range(n)]
                for i in range(n)
            ]

    def distance(self, source_id: int, dest_id: int) -> int:
        """Cached distance lookup between two planet IDs."""
        return self._dist_matrix[source_id][dest_id]

    def clone(self) -> GameState:
        """Fast state copying for EA branching and lookaheads."""
        new_planets = [p.copy() for p in self.planets]
        new_fleets = [f.copy() for f in self.fleets]
        st = GameState(
            planets=new_planets,
            fleets=new_fleets,
            turn=self.turn,
            max_turns=self.max_turns,
            dist_matrix=self._dist_matrix,  # Shared read-only distance matrix
        )
        st.winner = self.winner
        st.game_over = self.game_over
        return st

    # -----------------------------------------------------------------
    #  State Inspection Queries
    # -----------------------------------------------------------------

    def num_planets(self) -> int:
        return len(self.planets)

    def num_fleets(self) -> int:
        return len(self.fleets)

    def my_planets(self, player: int) -> List[Planet]:
        return [p for p in self.planets if p.owner == player]

    def enemy_planets(self, player: int) -> List[Planet]:
        rival = 2 if player == 1 else 1
        return [p for p in self.planets if p.owner == rival]

    def neutral_planets(self) -> List[Planet]:
        return [p for p in self.planets if p.owner == 0]

    def not_my_planets(self, player: int) -> List[Planet]:
        return [p for p in self.planets if p.owner != player]

    def my_fleets(self, player: int) -> List[Fleet]:
        return [f for f in self.fleets if f.owner == player]

    def enemy_fleets(self, player: int) -> List[Fleet]:
        rival = 2 if player == 1 else 1
        return [f for f in self.fleets if f.owner == rival]

    def num_ships(self, player: int) -> int:
        """Total ships for player (on planets + in transit)."""
        planet_ships = sum(p.num_ships for p in self.planets if p.owner == player)
        fleet_ships = sum(f.num_ships for f in self.fleets if f.owner == player)
        return planet_ships + fleet_ships

    def production(self, player: int) -> int:
        """Total ship production per turn for player."""
        return sum(p.growth_rate for p in self.planets if p.owner == player)

    def is_alive(self, player: int) -> bool:
        """Player is alive if they own any planet or have any fleet in flight."""
        return any(p.owner == player for p in self.planets) or any(
            f.owner == player for f in self.fleets
        )

    # -----------------------------------------------------------------
    #  Simulation Step
    # -----------------------------------------------------------------

    def step(
        self,
        orders_p1: Sequence[Union[Order, Tuple[int, int, int]]],
        orders_p2: Sequence[Union[Order, Tuple[int, int, int]]],
    ) -> Tuple[bool, Optional[int]]:
        """
        Execute one full game turn according to SPECIFICATION.md:
          Phase 1: Departure
          Phase 2: Advancement (fleets move, player planets grow)
          Phase 3: Arrival & Combat resolution
          Phase 4: Endgame check
        Returns (game_over, winner).
        """
        if self.game_over:
            return True, self.winner

        # Helper to normalize orders
        def normalize_orders(orders, player_id):
            norm = []
            for o in orders:
                if isinstance(o, Order):
                    norm.append((o.source_planet, o.destination_planet, o.num_ships))
                elif isinstance(o, (tuple, list)) and len(o) == 3:
                    norm.append((int(o[0]), int(o[1]), int(o[2])))
                else:
                    raise ValueError(f"Invalid order format: {o}")
            return norm

        p1_orders = normalize_orders(orders_p1, 1)
        p2_orders = normalize_orders(orders_p2, 2)

        # -------------------------------------------------------------
        # Phase 1: Departure
        # -------------------------------------------------------------
        # Process P1 departures
        self._process_departures(1, p1_orders)
        self._process_departures(2, p2_orders)

        # -------------------------------------------------------------
        # Phase 2: Advancement
        # -------------------------------------------------------------
        # Advance fleets
        for f in self.fleets:
            f.turns_remaining -= 1

        # Grow player-owned planets (neutral planets do NOT grow)
        for p in self.planets:
            if p.owner > 0:
                p.num_ships += p.growth_rate

        # -------------------------------------------------------------
        # Phase 3: Arrival & Battles
        # -------------------------------------------------------------
        arrived_fleets: List[Fleet] = []
        remaining_fleets: List[Fleet] = []

        for f in self.fleets:
            if f.turns_remaining <= 0:
                arrived_fleets.append(f)
            else:
                remaining_fleets.append(f)

        self.fleets = remaining_fleets

        if arrived_fleets:
            # Group arriving fleets by destination planet
            arrivals_by_planet: Dict[int, List[Fleet]] = {}
            for f in arrived_fleets:
                arrivals_by_planet.setdefault(f.destination_planet, []).append(f)

            for pid, fleets_here in arrivals_by_planet.items():
                self._resolve_battle(self.planets[pid], fleets_here)

        # -------------------------------------------------------------
        # Phase 4: Endgame Condition Check
        # -------------------------------------------------------------
        self.turn += 1

        p1_alive = self.is_alive(1)
        p2_alive = self.is_alive(2)

        if not p1_alive and not p2_alive:
            self.game_over = True
            self.winner = 0  # Draw
        elif not p1_alive:
            self.game_over = True
            self.winner = 2  # P2 wins
        elif not p2_alive:
            self.game_over = True
            self.winner = 1  # P1 wins
        elif self.turn >= self.max_turns:
            self.game_over = True
            s1 = self.num_ships(1)
            s2 = self.num_ships(2)
            if s1 > s2:
                self.winner = 1
            elif s2 > s1:
                self.winner = 2
            else:
                self.winner = 0  # Draw

        return self.game_over, self.winner

    def _process_departures(self, player: int, orders: List[Tuple[int, int, int]]) -> None:
        """
        Validate and execute departure orders for a player.
        Invalid orders are discarded or cause forfeit if strictly enforced.
        Here we safely execute valid orders without crashing.
        """
        # Sum requested departures per source planet to prevent over-allocation
        requested_by_source: Dict[int, int] = {}
        for src, dst, ships in orders:
            if ships <= 0 or src == dst:
                continue
            if src < 0 or src >= len(self.planets) or dst < 0 or dst >= len(self.planets):
                continue
            if self.planets[src].owner != player:
                continue
            requested_by_source[src] = requested_by_source.get(src, 0) + ships

        # Clamp if sum exceeds available ships (official engine drops bot on illegal move,
        # but bots in simulation are safe)
        for src, dst, ships in orders:
            if ships <= 0 or src == dst:
                continue
            if src < 0 or src >= len(self.planets) or dst < 0 or dst >= len(self.planets):
                continue
            planet = self.planets[src]
            if planet.owner != player:
                continue
            if planet.num_ships <= 0:
                continue

            actual_ships = min(ships, planet.num_ships)
            planet.num_ships -= actual_ships

            trip_length = self.distance(src, dst)
            fleet = Fleet(
                owner=player,
                num_ships=actual_ships,
                source_planet=src,
                destination_planet=dst,
                total_trip_length=trip_length,
                turns_remaining=trip_length,
            )
            self.fleets.append(fleet)

    def _resolve_battle(self, planet: Planet, arriving_fleets: List[Fleet]) -> None:
        """
        Official SPECIFICATION.md combat resolution:
          Combine forces by owner (0: neutral, 1: P1, 2: P2).
          Largest force wins; ship count = largest - second largest.
          In a tie between top forces, original owner retains planet with 0 ships.
        """
        # Aggregate forces
        forces: Dict[int, int] = {0: 0, 1: 0, 2: 0}
        # Planet's own garrison
        forces[planet.owner] += planet.num_ships

        # Arriving fleets
        for f in arriving_fleets:
            forces[f.owner] += f.num_ships

        # Filter out 0-ship forces
        active_forces = [(owner, count) for owner, count in forces.items() if count > 0]

        if not active_forces:
            # Completely emptied
            planet.num_ships = 0
            return

        if len(active_forces) == 1:
            owner, count = active_forces[0]
            planet.owner = owner
            planet.num_ships = count
            return

        # Sort descending by ship count
        active_forces.sort(key=lambda x: x[1], reverse=True)

        top_owner, top_count = active_forces[0]
        second_owner, second_count = active_forces[1]

        if top_count > second_count:
            # Clear winner
            planet.owner = top_owner
            planet.num_ships = top_count - second_count
        else:
            # Tie for 1st place!
            # SPECIFICATION: "if both forces are the same size, then the winner
            # is the original owner of the planet, and the planet's new ship count is zero."
            # "If the top two forces are the same size, the original owner retains
            # ownership of the planet but the forces are set to zero."
            planet.num_ships = 0
            # planet.owner remains unchanged!

    # -----------------------------------------------------------------
    #  Serialization / Map Format
    # -----------------------------------------------------------------

    def to_map_string(self, pov_player: int = 1) -> str:
        """
        Serialize game state into official map format for bots.
        If pov_player == 2, swap 1 and 2 so bot always sees itself as player 1.
        """
        def swap_owner(o: int) -> int:
            if o == 0:
                return 0
            if pov_player == 1:
                return o
            return 1 if o == 2 else 2

        lines = []
        for p in self.planets:
            lines.append(
                f"P {p.x:.6f} {p.y:.6f} {swap_owner(p.owner)} {p.num_ships} {p.growth_rate}"
            )
        for f in self.fleets:
            lines.append(
                f"F {swap_owner(f.owner)} {f.num_ships} {f.source_planet} {f.destination_planet} "
                f"{f.total_trip_length} {f.turns_remaining}"
            )
        return "\n".join(lines)


# =====================================================================
#  MAP PARSER
# =====================================================================

def parse_map_file(file_path: Union[str, Path]) -> GameState:
    """Parse an official Planet Wars map file into a GameState instance."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Map file not found: {path}")

    planets: List[Planet] = []
    fleets: List[Fleet] = []
    planet_id = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Strip comments and whitespace
            line = line.split("#")[0].strip()
            if not line:
                continue

            tokens = line.split()
            tag = tokens[0].upper()

            if tag == "P":
                # P <x:float> <y:float> <owner:int> <ships:int> <growth:int>
                x = float(tokens[1])
                y = float(tokens[2])
                owner = int(tokens[3])
                num_ships = int(tokens[4])
                growth = int(tokens[5])
                planets.append(
                    Planet(
                        planet_id=planet_id,
                        x=x,
                        y=y,
                        owner=owner,
                        num_ships=num_ships,
                        growth_rate=growth,
                    )
                )
                planet_id += 1

            elif tag == "F":
                # F <owner:int> <ships:int> <source:int> <destination:int> <total_turns:int> <remaining_turns:int>
                owner = int(tokens[1])
                num_ships = int(tokens[2])
                source = int(tokens[3])
                dest = int(tokens[4])
                total_turns = int(tokens[5])
                remaining_turns = int(tokens[6])
                fleets.append(
                    Fleet(
                        owner=owner,
                        num_ships=num_ships,
                        source_planet=source,
                        destination_planet=dest,
                        total_trip_length=total_turns,
                        turns_remaining=remaining_turns,
                    )
                )

    return GameState(planets=planets, fleets=fleets, turn=0)

