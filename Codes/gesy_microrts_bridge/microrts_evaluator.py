"""
GESY + MicroRTS Bridge — microrts_evaluator.py
===============================================
Evaluates a single GESY StrategyScript by translating it to a Java bot
and running actual MicroRTS games via the BatchGameRunner Java process.

This replaces gesy.core.simulator.simulate_match for real-game evaluation.

Usage
-----
    from gesy_microrts_bridge.microrts_evaluator import MicroRTSEvaluator

    evaluator = MicroRTSEvaluator(
        map_path     = "maps/8x8/basesWorkers8x8Obstacle.xml",
        opponent_bot = "WorkerRush",
        n_games      = 3,
    )
    evaluator.start()

    result = evaluator.evaluate_script(my_script)
    print(result.wins, result.losses, result.draws, result.fitness)

    evaluator.stop()
"""

from __future__ import annotations

import subprocess
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

# ── Path setup ─────────────────────────────────────────────────────────────
_CODES_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GESY_DIR    = os.path.join(_CODES_DIR, "GESY")
_MICRORTS_DIR = os.path.join(_CODES_DIR, "Simulation", "microrts")

for _p in [_CODES_DIR, _GESY_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gesy.core.ast_nodes import StrategyScript  # noqa: E402
from gesy_microrts_bridge.command_mapping import script_to_bot  # noqa: E402


# ---------------------------------------------------------------------------
# Fitness scoring constants (mirrors gesy.engine.genetic_operators)
# ---------------------------------------------------------------------------

WIN_SCORE:  float = 1.0
DRAW_SCORE: float = 0.4
LOSS_SCORE: float = 0.0

# Bloat penalty (applied after win-rate scoring)
BLOAT_PENALTY_FACTOR: float = 0.002
BLOAT_THRESHOLD:       int   = 25


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Outcome of evaluating a StrategyScript on real MicroRTS games."""
    script_name: str
    bot_used:    str          # Java bot that was selected
    wins:        int
    losses:      int
    draws:       int
    total_games: int
    avg_cycles:  float        # average game length in ticks
    fitness:     float        # final score with bloat penalty applied


# ---------------------------------------------------------------------------
# MicroRTS server wrapper (single persistent Java process)
# ---------------------------------------------------------------------------

class MicroRTSEvaluator:
    """
    Manages a long-lived Java BatchGameRunner process and evaluates
    GESY StrategyScript objects against a fixed opponent bot.

    Parameters
    ----------
    map_path : str
        MicroRTS map XML path relative to the microrts/ directory.
    opponent_bot : str
        Java bot name that all scripts will compete against.
    n_games : int
        Number of games played per script evaluation.
    max_cycles : int
        Hard game-cycle cap passed to MicroRTS.
    microrts_dir : str, optional
        Override path to the microrts/ directory.
    """

    def __init__(
        self,
        map_path:     str  = "maps/8x8/basesWorkers8x8Obstacle.xml",
        opponent_bot: str  = "WorkerRush",
        n_games:      int  = 3,
        max_cycles:   int  = 4000,
        microrts_dir: Optional[str] = None,
    ) -> None:
        self.map_path     = map_path
        self.opponent_bot = opponent_bot
        self.n_games      = n_games
        self.max_cycles   = max_cycles
        self.microrts_dir = microrts_dir or _MICRORTS_DIR
        self._process: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the persistent Java BatchGameRunner process."""
        cmd = ["java", "-cp", "lib/*;bin", "tests.BatchGameRunner"]
        self._process = subprocess.Popen(
            cmd,
            stdin  = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text   = True,
            cwd    = self.microrts_dir,
        )
        # Wait for the server ready signal
        line = self._process.stdout.readline().strip()
        if line != "READY":
            raise RuntimeError(
                f"BatchGameRunner failed to start. Got: '{line}'\n"
                f"Check that Java is installed and the MicroRTS binaries "
                f"are compiled (run `ant` in {self.microrts_dir})."
            )
        print(f"  [✓] MicroRTS game server started (map={self.map_path}, "
              f"opponent={self.opponent_bot})")

    def stop(self) -> None:
        """Gracefully shut down the Java process."""
        if self._process:
            try:
                self._process.stdin.write("EXIT\n")
                self._process.stdin.flush()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            print("  [✓] MicroRTS game server stopped")
            self._process = None

    def __enter__(self) -> "MicroRTSEvaluator":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Game execution
    # ------------------------------------------------------------------

    def _run_single_game(self, bot1: str, bot2: str) -> tuple[int, int]:
        """
        Run one game bot1 vs bot2.

        Returns
        -------
        (winner, cycles)
            winner: 0 = bot1 won, 1 = bot2 won, -1 = draw/error
        """
        if self._process is None:
            raise RuntimeError("MicroRTSEvaluator not started. Call .start() first.")

        request = f"{bot1},{bot2},{self.map_path},{self.max_cycles}\n"
        self._process.stdin.write(request)
        self._process.stdin.flush()

        response = self._process.stdout.readline().strip()
        if response.startswith("RESULT:"):
            parts = response.replace("RESULT:", "").split(",")
            try:
                return int(parts[0]), int(parts[1])
            except (IndexError, ValueError):
                pass
        # Unexpected response
        print(f"    [!] Unexpected MicroRTS response: {response}")
        return -1, self.max_cycles

    # ------------------------------------------------------------------
    # Script evaluation
    # ------------------------------------------------------------------

    def evaluate_script(self, script: StrategyScript) -> EvalResult:
        """
        Evaluate a GESY StrategyScript by:
          1. Converting it to a Java bot via command_mapping.script_to_bot()
          2. Playing self.n_games against self.opponent_bot
          3. Computing fitness = win-rate score minus bloat penalty

        Parameters
        ----------
        script : StrategyScript

        Returns
        -------
        EvalResult
        """
        bot_name = script_to_bot(script)

        wins = losses = draws = 0
        total_cycles = 0

        for _ in range(self.n_games):
            winner, cycles = self._run_single_game(bot_name, self.opponent_bot)
            total_cycles += cycles
            if winner == 0:
                wins += 1
            elif winner == 1:
                losses += 1
            else:
                draws += 1

        # Raw win-rate score
        total = wins + losses + draws
        raw_score = (
            wins   * WIN_SCORE  +
            draws  * DRAW_SCORE +
            losses * LOSS_SCORE
        ) / max(total, 1)

        # Bloat penalty
        n_nodes = script.node_count()
        excess  = max(0, n_nodes - BLOAT_THRESHOLD)
        penalty = excess * BLOAT_PENALTY_FACTOR

        fitness = max(0.0, raw_score - penalty)

        return EvalResult(
            script_name = script.name,
            bot_used    = bot_name,
            wins        = wins,
            losses      = losses,
            draws       = draws,
            total_games = total,
            avg_cycles  = total_cycles / max(total, 1),
            fitness     = fitness,
        )

    def update_script_fitness(self, script: StrategyScript) -> EvalResult:
        """
        Evaluate script and update its fitness / W/L/D fields in-place.

        Convenience wrapper for use in population loops.
        """
        result = self.evaluate_script(script)
        script.fitness = result.fitness
        script.wins    = result.wins
        script.losses  = result.losses
        script.draws   = result.draws
        return result


# ---------------------------------------------------------------------------
# Standalone test (requires Java + compiled MicroRTS)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MicroRTSEvaluator — Connection Test")
    print("=" * 60)

    try:
        from gesy.core.ast_nodes import LeafAction, StrategyScript
        from gesy.strategies.seeded_init import build_rush_strategy

        script = build_rush_strategy()
        print(f"  Script: {script.name}")

        with MicroRTSEvaluator(n_games=3) as ev:
            result = ev.evaluate_script(script)

        print(f"\n  Results for '{result.script_name}':")
        print(f"    Bot selected : {result.bot_used}")
        print(f"    W/L/D        : {result.wins}/{result.losses}/{result.draws}")
        print(f"    Avg cycles   : {result.avg_cycles:.0f}")
        print(f"    Fitness      : {result.fitness:.4f}")
        print("\n  [Test] PASSED ✓")

    except Exception as e:
        print(f"\n  [Test] FAILED: {e}")
        print("  (If Java is not installed or MicroRTS not compiled, this is expected.)")
    print()
