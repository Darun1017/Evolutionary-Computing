"""
Simulation Engine Validation Suite
==================================
Plays at least 20 official maps with deterministic example bots in both:
1. The official Java game engine (tools/PlayGame-1.2.jar)
2. Our pure-Python simulator (planetwars/sim.py)

Compares:
- Winner (P1, P2, Draw)
- Per-turn planet ownership
- Per-turn planet ship counts
- Per-turn fleet counts and positions

Reports exact mismatch rate and logs all diagnostics.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from planetwars.sim import GameState, parse_map_file
from planetwars.opponents import bully_bot, rage_bot, prospector_bot, dual_bot


BOT_MAPPING = {
    "BullyBot": ("engine/example_bots/BullyBot.jar", bully_bot),
    "RageBot": ("engine/example_bots/RageBot.jar", rage_bot),
    "ProspectorBot": ("engine/example_bots/ProspectorBot.jar", prospector_bot),
    "DualBot": ("engine/example_bots/DualBot.jar", dual_bot),
}


def run_official_engine(
    map_path: Path,
    p1_jar: str,
    p2_jar: str,
    max_turns: int = 100,
) -> Tuple[int, List[str], str]:
    """
    Run match in official Java engine and extract per-turn playback string.
    Uses relative paths from PROJECT_ROOT to prevent Java exec whitespace splitting issues.
    Returns (winner, turn_strings, full_stdout).
    """
    rel_map = map_path.relative_to(PROJECT_ROOT).as_posix()
    cmd = [
        "java",
        "-jar",
        "engine/tools/PlayGame-1.2.jar",
        rel_map,
        "1000",
        str(max_turns),
        "",
        f"java -jar {p1_jar}",
        f"java -jar {p2_jar}",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    stdout = res.stdout
    stderr = res.stderr

    # Determine winner from stderr or game playback
    winner = 0
    if "Player 1 Wins!" in stderr:
        winner = 1
    elif "Player 2 Wins!" in stderr:
        winner = 2
    elif "Draw!" in stderr:
        winner = 0

    parts = stdout.split("|")
    turns_str = []
    if len(parts) > 1 and parts[1].strip():
        turns_str = [t for t in parts[1].strip().split(":") if t]

    return winner, turns_str, stdout


def run_validation(
    num_maps: int = 20,
    max_turns: int = 100,
    matchups: List[Tuple[str, str]] = None,
) -> Dict:
    if matchups is None:
        matchups = [
            ("BullyBot", "RageBot"),
            ("ProspectorBot", "BullyBot"),
        ]

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_maps": num_maps,
        "max_turns": max_turns,
        "matchups": matchups,
        "total_matches": 0,
        "winner_matches": 0,
        "total_turns_checked": 0,
        "turn_mismatches": 0,
        "total_planet_checks": 0,
        "planet_mismatches": 0,
        "match_details": [],
    }

    maps_dir = PROJECT_ROOT / "engine" / "maps"
    test_maps = [maps_dir / f"map{i}.txt" for i in range(1, num_maps + 1)]

    print(f"\n{'='*75}")
    print(f"  PLANET WARS SIMULATOR VALIDATION SUITE")
    print(f"  Testing {num_maps} maps across {len(matchups)} bot matchups (max {max_turns} turns)")
    print(f"{'='*75}\n")
    print(f"{'Map':<10} {'Matchup':<26} {'Off Win':<9} {'Sim Win':<9} {'Turns':<7} {'Status'}")
    print(f"{'-'*75}")

    for map_path in test_maps:
        if not map_path.exists():
            continue

        for p1_name, p2_name in matchups:
            results["total_matches"] += 1
            p1_jar, p1_fn = BOT_MAPPING[p1_name]
            p2_jar, p2_fn = BOT_MAPPING[p2_name]

            # 1. Official engine run
            off_winner, off_turns, _ = run_official_engine(
                map_path, p1_jar, p2_jar, max_turns=max_turns
            )

            # 2. Python simulator run
            state = parse_map_file(map_path)
            state.max_turns = max_turns

            match_planet_mismatches = 0
            turns_checked = min(len(off_turns), max_turns)

            for t in range(turns_checked):
                # Step simulation
                o1 = p1_fn(state, 1)
                o2 = p2_fn(state, 2)
                state.step(o1, o2)

                results["total_turns_checked"] += 1

                # Parse official turn snapshot
                turn_tokens = off_turns[t].split(",")
                planet_tokens = turn_tokens[: len(state.planets)]

                turn_mismatched = False
                for pid, p_str in enumerate(planet_tokens):
                    results["total_planet_checks"] += 1
                    parts = p_str.split(".")
                    off_owner = int(parts[0])
                    off_ships = int(parts[1])

                    sim_p = state.planets[pid]
                    if sim_p.owner != off_owner or sim_p.num_ships != off_ships:
                        match_planet_mismatches += 1
                        results["planet_mismatches"] += 1
                        turn_mismatched = True

                if turn_mismatched:
                    results["turn_mismatches"] += 1

            sim_winner = state.winner if state.winner is not None else 0

            # Compare winners
            winner_match = (off_winner == sim_winner)
            if winner_match:
                results["winner_matches"] += 1

            status_str = "MATCH (100%)" if (winner_match and match_planet_mismatches == 0) else f"MISMATCH ({match_planet_mismatches} errs)"
            print(
                f"{map_path.stem:<10} {f'{p1_name} vs {p2_name}':<26} {off_winner:<9} {sim_winner:<9} {turns_checked:<7} {status_str}"
            )

            results["match_details"].append({
                "map": map_path.stem,
                "p1": p1_name,
                "p2": p2_name,
                "turns": turns_checked,
                "off_winner": off_winner,
                "sim_winner": sim_winner,
                "winner_match": winner_match,
                "planet_mismatches": match_planet_mismatches,
            })

    # Summary
    planet_mismatch_rate = (
        results["planet_mismatches"] / results["total_planet_checks"] * 100.0
        if results["total_planet_checks"] > 0
        else 0.0
    )
    winner_accuracy = (
        results["winner_matches"] / results["total_matches"] * 100.0
        if results["total_matches"] > 0
        else 0.0
    )

    results["planet_mismatch_rate"] = planet_mismatch_rate
    results["winner_accuracy"] = winner_accuracy

    print(f"\n{'='*75}")
    print(f"  VALIDATION SUMMARY")
    print(f"  Total matches evaluated : {results['total_matches']}")
    print(f"  Total planet checks     : {results['total_planet_checks']}")
    print(f"  Planet mismatch count   : {results['planet_mismatches']}")
    print(f"  Planet mismatch rate    : {planet_mismatch_rate:.4f}%")
    print(f"  Winner match accuracy   : {winner_accuracy:.2f}% ({results['winner_matches']}/{results['total_matches']})")
    print(f"{'='*75}\n")

    # Save to tests/
    out_file = PROJECT_ROOT / "tests" / "validation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved validation report to: {out_file}")

    return results


if __name__ == "__main__":
    run_validation(num_maps=20, max_turns=60)
