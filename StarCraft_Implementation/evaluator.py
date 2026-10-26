"""
evaluator.py — Composite fitness metric tracking for EvolvedSC2Bot.

Metrics (all normalised to [0, 1]):
  APU  — Average Population Utilization
         Fraction of army supply actively engaged near enemies each step,
         averaged over the full game.  Penalises idle armies.

  PBR  — Production Build Rate
         Fraction of time production buildings are busy (not idle).
         Penalises bot that builds barracks but never trains units.

  RUR  — Resource Utilization Ratio
         total_resources_spent / (total_spent + excess_banked).
         Penalises resource hoarding (high mineral/gas float).

Composite fitness:
  f_i = 0.40·WR + 0.25·APU + 0.20·PBR + 0.15·RUR

Usage inside EvolvedSC2Bot:
  self.tracker = MetricsTracker()
  # Each on_step():
  self.tracker.update(self)
  # In on_end():
  self.fitness_result = self.tracker.compute_fitness(won, fitness_weights)
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sc2.bot_ai import BotAI

# Race-specific IDs for production-building detection
_PRODUCTION_IDS: set[int] = set()

try:
    from sc2.ids.unit_typeid import UnitTypeId
    _PRODUCTION_IDS = {
        # Terran
        UnitTypeId.BARRACKS.value,
        UnitTypeId.FACTORY.value,
        UnitTypeId.STARPORT.value,
        # Protoss
        UnitTypeId.GATEWAY.value,
        UnitTypeId.WARPGATE.value,
        UnitTypeId.ROBOTICSFACILITY.value,
        UnitTypeId.STARGATE.value,
        # Zerg
        UnitTypeId.HATCHERY.value,
        UnitTypeId.LAIR.value,
        UnitTypeId.HIVE.value,
    }
except ImportError:
    pass  # Will be populated at runtime


class MetricsTracker:
    """
    Tracks per-step SC2 metrics and computes composite fitness after a game.

    Call `update(bot)` every `on_step()` and `compute_fitness(won, weights)`
    once in `on_end()`.
    """

    # Minimum army supply needed before APU sampling starts (avoid noise at
    # game-start when army is 0).
    _APU_MIN_ARMY_SUPPLY = 2

    # Resources above this threshold are counted as "excess" (banked float).
    _MINERAL_FLOAT_THRESHOLD = 50
    _GAS_FLOAT_THRESHOLD     = 25

    # Scale factor when converting excess to the same units as spent resources.
    # A lower value makes RUR more forgiving about short-term banking.
    _EXCESS_PENALTY_SCALE = 0.08

    def __init__(self) -> None:
        # ── APU ─────────────────────────────────────────────────────────────
        self._apu_samples:   list[float] = []

        # ── PBR ─────────────────────────────────────────────────────────────
        self._pbr_busy_steps:  int = 0
        self._pbr_total_steps: int = 0

        # ── RUR ─────────────────────────────────────────────────────────────
        self._spent_minerals:  float = 0.0
        self._spent_vespene:   float = 0.0
        self._excess_accum:    float = 0.0   # cumulative excess resources

        # ── House-keeping ────────────────────────────────────────────────────
        self._step_count: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def update(self, bot: "BotAI") -> None:
        """
        Sample all metrics from the current bot state.
        Call once per `on_step()` invocation.
        """
        self._step_count += 1
        self._update_apu(bot)
        self._update_pbr(bot)
        self._update_rur(bot)

    def compute_fitness(self, won: bool, fitness_weights: dict) -> dict:
        """
        Compute final metrics and composite fitness.

        Args:
            won:             True if the game was a Victory.
            fitness_weights: Dict with keys win_rate / apu / pbr / rur.

        Returns:
            Dict containing individual metrics, composite f_i, and diagnostics.
        """
        win_rate = 1.0 if won else 0.0
        apu      = self._compute_apu()
        pbr      = self._compute_pbr()
        rur      = self._compute_rur()

        f_i = (fitness_weights["win_rate"] * win_rate +
               fitness_weights["apu"]      * apu      +
               fitness_weights["pbr"]      * pbr      +
               fitness_weights["rur"]      * rur)

        return {
            "f_i":       round(f_i, 5),
            "win_rate":  win_rate,
            "apu":       round(apu, 4),
            "pbr":       round(pbr, 4),
            "rur":       round(rur, 4),
            "apu_n":     len(self._apu_samples),
            "pbr_busy":  self._pbr_busy_steps,
            "pbr_total": self._pbr_total_steps,
            "steps":     self._step_count,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  PRIVATE METRIC UPDATES
    # ─────────────────────────────────────────────────────────────────────────

    def _update_apu(self, bot: "BotAI") -> None:
        """
        APU sample: fraction of army supply that is near a visible enemy unit
        (within weapon range + 4 units).  Only sampled when army is present.
        """
        army = bot.units.filter(lambda u: not u.is_worker and u.can_attack and u.is_ready)
        if not army:
            return

        total_supply = sum(u.supply_cost for u in army)
        if total_supply < self._APU_MIN_ARMY_SUPPLY:
            return

        engaged_supply = 0
        for unit in army:
            # "Engaged" = any visible enemy within (weapon range + 4) units
            weapon_range = unit.air_range if not bot.enemy_units.filter(lambda e: not e.is_flying) else unit.ground_range
            threat_radius = (weapon_range if weapon_range > 0 else 4.0) + 4.0
            if bot.enemy_units.closer_than(threat_radius, unit.position):
                engaged_supply += unit.supply_cost

        self._apu_samples.append(engaged_supply / total_supply)

    def _update_pbr(self, bot: "BotAI") -> None:
        """
        PBR sample: each ready production building is sampled as busy (1) or
        idle (0).  Averaged over the full game.
        """
        for struct in bot.structures:
            if not struct.is_ready:
                continue
            if struct.type_id.value not in _PRODUCTION_IDS:
                continue
            self._pbr_total_steps += 1
            if not struct.is_idle:
                self._pbr_busy_steps += 1

    def _update_rur(self, bot: "BotAI") -> None:
        """
        RUR: accumulate spent resources (from score) and excess float.
        """
        # Use game score for cumulative spent resources (most accurate)
        try:
            self._spent_minerals = bot.state.score.spent_minerals
            self._spent_vespene  = bot.state.score.spent_vespene
        except AttributeError:
            pass   # score not available this step

        # Accumulate excess float above safe thresholds
        excess = (max(0.0, bot.minerals - self._MINERAL_FLOAT_THRESHOLD) +
                  max(0.0, bot.vespene  - self._GAS_FLOAT_THRESHOLD))
        self._excess_accum += excess

    # ─────────────────────────────────────────────────────────────────────────
    #  PRIVATE METRIC COMPUTATIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_apu(self) -> float:
        if not self._apu_samples:
            return 0.0
        return sum(self._apu_samples) / len(self._apu_samples)

    def _compute_pbr(self) -> float:
        if self._pbr_total_steps == 0:
            return 0.0
        return self._pbr_busy_steps / self._pbr_total_steps

    def _compute_rur(self) -> float:
        total_spent  = self._spent_minerals + self._spent_vespene
        penalty      = self._excess_accum * self._EXCESS_PENALTY_SCALE
        denominator  = total_spent + penalty
        if denominator == 0:
            return 0.0
        return min(1.0, total_spent / denominator)


def compute_composite(metrics: dict, fitness_weights: dict) -> float:
    """Standalone helper: compute f_i from a metrics dict."""
    return (fitness_weights["win_rate"] * metrics["win_rate"] +
            fitness_weights["apu"]      * metrics["apu"]      +
            fitness_weights["pbr"]      * metrics["pbr"]      +
            fitness_weights["rur"]      * metrics["rur"])
