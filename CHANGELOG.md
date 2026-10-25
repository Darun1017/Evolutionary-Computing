# Changelog

A milestone-oriented history of the TD-CCEA / Evolutionary-Computing project.
This is a human-readable narrative grouped by development phase rather than a
strict per-release log. It is derived from the repository's commit history.

## Phase 4 — StarCraft II extension & post-review work (latest)

- Added environment-conditioned, sensitivity-adaptive TD-CCEA.
- Added a dynamic, interaction-driven CCE-TDCCEA variant.
- Adapted the TD-CCEA framework to StarCraft II build-order planning, with
  Protoss / Terran / Zerg macro bots, evaluation harnesses, and
  match-watching utilities (see `StarCraft/`).
- Added post-review modification scripts (`Post_Review_Modifications/`).

## Phase 3 — Cooperative co-evolution & expanded benchmarking

- Introduced CCE-TDCCEA (cooperative co-evolution splitting the 20-D strategy
  space into Macro/Economy and Micro/Combat sub-populations).
- Added 2-way and 3-way co-evolution benchmarks and baseline comparisons.
- Added GESY and SA-PDCoEA components and sensitivity scoring.
- Added CCEA benchmarking and the NEW_TDCCEA variant.
- Added new symmetric 32x32 maps (standard and heavy-blockade), with BFS
  reachability verification ensuring resources stay accessible from both bases.
- Organized map generators into `MapGenerators/` and added visualization tools.

## Phase 2 — Benchmarks, baselines & statistical validation

- Added a benchmark suite comparing TD-CCEA against Standard GA, Hill Climbing,
  and Random Search baselines.
- Added evolved-brain variants, head-to-head tournaments, and evolution against
  custom opponents.
- Added Wilcoxon signed-rank and ablation tests, plus additional benchmark runs
  with time tracking and Excel-formatted result exports.
- Added SEM7 PR1 experiment variants (single-core and multi-core).

## Phase 1 — Core microRTS pipeline & documentation

- Added the microRTS simulation engine integration and the `EvolvedBrainAI`
  Java bot, parameterized by a 20-weight behavioral strategy vector.
- Established the Python <-> Java communication bridge for headless, batched
  game evaluation.
- Added the initial TD-CCEA implementation and a README with setup,
  compilation, and run instructions documenting the algorithm.

## Phase 0 — Project bootstrap

- Uploaded initial presentations and practice code.
