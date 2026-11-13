"""
Unit test suite for pure-Python Planet Wars simulator.
Verifies rules, map parsing, combat resolution, and state copying.
"""

import unittest
from pathlib import Path
from planetwars.sim import GameState, Planet, Fleet, parse_map_file, compute_travel_turns
from planetwars.opponents import bully_bot, rage_bot, prospector_bot, dual_bot

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPlanetWarsSim(unittest.TestCase):
    def test_map_parser(self):
        map_path = PROJECT_ROOT / "engine" / "maps" / "map1.txt"
        self.assertTrue(map_path.exists())
        state = parse_map_file(map_path)
        self.assertGreater(state.num_planets(), 0)
        self.assertEqual(len(state.my_planets(1)), 1)
        self.assertEqual(len(state.enemy_planets(1)), 1)

    def test_state_cloning(self):
        map_path = PROJECT_ROOT / "engine" / "maps" / "map1.txt"
        state = parse_map_file(map_path)
        clone = state.clone()
        self.assertEqual(state.num_planets(), clone.num_planets())
        # Modifying clone shouldn't affect original
        clone.planets[0].num_ships = 9999
        self.assertNotEqual(state.planets[0].num_ships, clone.planets[0].num_ships)

    def test_combat_resolution_tie(self):
        # 1 neutral planet with 10 ships. P1 sends 20, P2 sends 20.
        p = Planet(planet_id=0, x=0.0, y=0.0, owner=0, num_ships=10, growth_rate=2)
        st = GameState(planets=[p])
        f1 = Fleet(owner=1, num_ships=20, source_planet=0, destination_planet=0, total_trip_length=1, turns_remaining=0)
        f2 = Fleet(owner=2, num_ships=20, source_planet=0, destination_planet=0, total_trip_length=1, turns_remaining=0)
        st.fleets = [f1, f2]
        st._resolve_battle(p, [f1, f2])
        # P1 and P2 tied (20 vs 20) > 10 neutral. Ties mean original owner retains with 0 ships.
        self.assertEqual(p.owner, 0)
        self.assertEqual(p.num_ships, 0)

    def test_combat_resolution_win(self):
        p = Planet(planet_id=0, x=0.0, y=0.0, owner=0, num_ships=10, growth_rate=2)
        f1 = Fleet(owner=1, num_ships=25, source_planet=0, destination_planet=0, total_trip_length=1, turns_remaining=0)
        f2 = Fleet(owner=2, num_ships=15, source_planet=0, destination_planet=0, total_trip_length=1, turns_remaining=0)
        st = GameState(planets=[p])
        st._resolve_battle(p, [f1, f2])
        # Forces: Neutral=10, P1=25, P2=15.
        # Top = 25 (P1), second = 15 (P2). Winner P1 with 25 - 15 = 10 ships.
        self.assertEqual(p.owner, 1)
        self.assertEqual(p.num_ships, 10)


if __name__ == "__main__":
    unittest.main()

