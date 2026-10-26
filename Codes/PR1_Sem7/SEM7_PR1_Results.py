"""
SEM7_PR1_Results.py
====================
Head-to-Head Comparison of 4 Evolutionary Algorithms on a Dynamic 48x48 Map

ALGORITHMS:
  1. CCE-TDCCEA      — Cooperative Co-Evolutionary TD-CCEA (macro/micro split)
  2. TD-CCEA         — Monolithic Trajectory-guided CCEA
  3. SA-TDCCEA       — Sensitivity-Adaptive TD-CCEA (per-weight mutation rates)
  4. Standard CCEA   — Same split as CCE-TDCCEA, no trajectory memory

WORKFLOW:
  Phase 1 — EVOLUTION
    Each algorithm co-evolves against the current best weights of the other
    three algorithms on a 48×48 map that gains progressive blockades each
    generation (BFS-validated). Final weights are later validated against bots.

  Phase 2 — HEAD-TO-HEAD TOURNAMENT
    The best evolved weights from each algorithm play EVOLVED vs EVOLVED
    on the final (blockaded) map. 10 games per pair (5 as P0, 5 as P1).

  Phase 3 — RESULTS
    ┌ Console matrix (image format) ┐
    │ Wilcoxon signed-rank test      │  → wilcoxon_results.txt
    │ Ablation analysis              │  → ablation_results.txt
    │ JSON summary                   │  → pr1_results.json
    └────────────────────────────────┘

SOURCE REFERENCES:
  - Algorithm code:    COMPARE_ALL.py  (lines 269–604)
  - Dynamic map/BFS:   tdccea_dynamic_env.py (lines 74–201)
  - Base map:          maps/open48x48.xml (48×48, no blockades)
"""

import math, random, time, subprocess, os, json, collections
import queue
from concurrent.futures import ThreadPoolExecutor

os.environ["PYTHONIOENCODING"] = "utf-8"

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CODES_DIR    = os.path.dirname(SCRIPT_DIR)
MICRORTS_DIR = os.path.join(CODES_DIR, "Simulation", "microrts")

# 48×48 map
MAP_W, MAP_H  = 48, 48
BASE_MAP_PATH  = os.path.join(MICRORTS_DIR, "maps", "open48x48.xml")
TEMP_MAP_PATH  = os.path.join(MICRORTS_DIR, "maps", "temp_pr1_48x48.xml")
TEMP_MAP_REL   = "maps/temp_pr1_48x48.xml"
MAX_CYCLES     = 12000   # cycles for 48x48 map

WEIGHT_NAMES = [
    "worker_ratio","max_workers","expand_threshold","light_preference",
    "heavy_preference","ranged_preference","barracks_timing","aggression",
    "attack_threshold","defend_radius","target_priority","defense_trigger",
    "counter_rush","tech_switch","economy_pressure","scout_priority",
    "group_attack","second_barracks","worker_fight_ratio","focus_fire",
]
DIMENSIONS    = len(WEIGHT_NAMES)
MACRO_INDICES = [0,1,2,6,14,15,17,18]
MICRO_INDICES = [3,4,5,7,8,9,10,11,12,13,16,19]

OPPONENT_ROTATION = [
    "EconomyMilitaryRush","LightRush","WorkerRush",
    "HeavyDefense","RangedRush","EconomyRush","WorkerRushPlusPlus",
]

# Evolution hyperparameters
N_POP             = 20
K_GAMES           = 3
# NOTE: Evolution now evaluates each individual against the CURRENT BEST of the
# other 3 algorithms every generation (3 games/eval, one per rival algorithm)
# instead of against 1 fixed bot. That is ~3x the games per generation of the
# old bot-based evolution, so TOTAL_GENERATIONS has been reduced 30 -> 15 to
# keep total runtime roughly comparable. Adjust freely.
TOTAL_GENERATIONS = 22
VALIDATION_GAMES  = 5
A_MAX             = 15
DELTA_S           = 0.85
DELTA_F           = 0.20
TOP_PARTNERS      = 3
SENSITIVITY_FILE  = "sensitivity_scores.json"

# Head-to-head tournament
H2H_GAMES = 10   # per matchup pair (5 as P0, 5 as P1)

# CPU parallelism (10 workers — using 10 of 12 logical processors)
N_WORKERS = 10

# Reserved tiles (bases, workers, resources + buffers) for 48×48 map
RESERVED = set()
# Player 0 corner (top-left)
for bx, by in [(0,0),(0,1),(1,1),(2,2),(1,2),(2,1),(3,2),(2,3),(3,3)]:
    RESERVED.add((bx, by))
# Player 1 corner (bottom-right)
for bx, by in [(MAP_W-1,MAP_H-1),(MAP_W-1,MAP_H-2),(MAP_W-2,MAP_H-2),(MAP_W-3,MAP_H-3),(MAP_W-2,MAP_H-3),(MAP_W-3,MAP_H-2),(MAP_W-4,MAP_H-3),(MAP_W-3,MAP_H-4),(MAP_W-4,MAP_H-4)]:
    RESERVED.add((bx, by))

# Dynamic environment — scaled for 48×48
WALLS_PER_GEN_EARLY = 6
WALLS_PER_GEN_MID   = 10
WALLS_PER_GEN_LATE  = 14
BREAKABLE_RATIO     = 0.4


# ═══════════════════════════════════════════════════════════════
#  MAP GENERATION + BFS VALIDATION
#  (Adapted from tdccea_dynamic_env.py for 48×48)
# ═══════════════════════════════════════════════════════════════

def bfs_reachable(walls, start, end):
    """BFS to check if `start` can reach `end` avoiding `walls`."""
    if start in walls or end in walls:
        return False
    visited = {start}
    q = collections.deque([start])
    while q:
        x, y = q.popleft()
        if (x, y) == end:
            return True
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < MAP_W and 0 <= ny < MAP_H:
                if (nx, ny) not in visited and (nx, ny) not in walls:
                    visited.add((nx, ny))
                    q.append((nx, ny))
    return False


def pick_wall_position(terrain_walls, breakable_positions, gen):
    """Pick a random wall position that doesn't break base-to-base path."""
    base0, base1 = (2, 2), (MAP_W - 3, MAP_H - 3)   # 48×48 bases
    all_walls = terrain_walls | breakable_positions
    no_place  = all_walls | RESERVED
    for _ in range(300):
        cx, cy = MAP_W // 2, MAP_H // 2
        spread = 10 + gen // 2  # wider spread for larger map
        x = int(random.gauss(cx, spread))
        y = int(random.gauss(cy, spread))
        x = max(0, min(MAP_W - 1, x))
        y = max(0, min(MAP_H - 1, y))
        if (x, y) in no_place:
            continue
        test_walls = all_walls | {(x, y)}
        if bfs_reachable(test_walls, base0, base1):
            return (x, y)
    return None


def add_walls_for_generation(gen, terrain_walls, breakable_positions):
    """Add progressive blockades for a generation."""
    if gen <= 7:
        n_new = WALLS_PER_GEN_EARLY
    elif gen <= 14:
        n_new = WALLS_PER_GEN_MID
    else:
        n_new = WALLS_PER_GEN_LATE
    new_t, new_b = set(), set()
    for _ in range(n_new):
        pos = pick_wall_position(terrain_walls | new_t,
                                 breakable_positions | new_b, gen)
        if pos is None:
            break
        if random.random() < BREAKABLE_RATIO:
            new_b.add(pos)
        else:
            new_t.add(pos)
    terrain_walls      = terrain_walls | new_t
    breakable_positions = breakable_positions | new_b
    return terrain_walls, breakable_positions, len(new_t), len(new_b)


def write_map_xml(terrain_walls, breakable_positions, path):
    """Write a 48×48 map XML with blockades encoded in the terrain string."""
    all_blockades = terrain_walls | breakable_positions
    terrain = ['0'] * (MAP_W * MAP_H)
    for wx, wy in all_blockades:
        terrain[wy * MAP_W + wx] = '1'
    terrain_str = ''.join(terrain)

    units_xml = [
        '    <rts.units.Unit type="Resource" ID="1" player="-1" x="0" y="0" resources="25" hitpoints="1"></rts.units.Unit>',
        '    <rts.units.Unit type="Resource" ID="2" player="-1" x="0" y="1" resources="25" hitpoints="1"></rts.units.Unit>',
        f'    <rts.units.Unit type="Resource" ID="5" player="-1" x="{MAP_W-1}" y="{MAP_H-1}" resources="25" hitpoints="1"></rts.units.Unit>',
        f'    <rts.units.Unit type="Resource" ID="6" player="-1" x="{MAP_W-1}" y="{MAP_H-2}" resources="25" hitpoints="1"></rts.units.Unit>',
        '    <rts.units.Unit type="Base" ID="3" player="0" x="2" y="2" resources="0" hitpoints="10"></rts.units.Unit>',
        f'    <rts.units.Unit type="Base" ID="7" player="1" x="{MAP_W-3}" y="{MAP_H-3}" resources="0" hitpoints="10"></rts.units.Unit>',
        '    <rts.units.Unit type="Worker" ID="4" player="0" x="1" y="1" resources="0" hitpoints="1"></rts.units.Unit>',
        f'    <rts.units.Unit type="Worker" ID="8" player="1" x="{MAP_W-2}" y="{MAP_H-2}" resources="0" hitpoints="1"></rts.units.Unit>',
    ]

    xml = f'''<rts.PhysicalGameState width="{MAP_W}" height="{MAP_H}">
  <terrain>{terrain_str}</terrain>
  <players>
    <rts.Player ID="0" resources="5"></rts.Player>
    <rts.Player ID="1" resources="5"></rts.Player>
  </players>
  <units>
{chr(10).join(units_xml)}
  </units>
</rts.PhysicalGameState>'''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(xml)


# ═══════════════════════════════════════════════════════════════
#  GAME SERVER
#  Supports both EVOLVED vs BOT and EVOLVED vs EVOLVED modes
# ═══════════════════════════════════════════════════════════════

class GameServer:
    def __init__(self):
        self.process = None
        self._fail_count = 0
        self._total_count = 0
        self._raw_winner_counts = {}   # diagnostic: raw winner values seen
        self._forced_draw_count = 0    # count of timeouts forced to draw

    def start(self):
        print(f"  [SERVER] cwd = {MICRORTS_DIR}")
        print(f"  [SERVER] Map file exists: {os.path.isfile(TEMP_MAP_PATH)}")
        self.process = subprocess.Popen(
            ["java", "-cp", "lib/*;bin", "tests.EvolvedBatchGameRunner"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=MICRORTS_DIR
        )
        line = self.process.stdout.readline().strip()
        if line != "READY":
            # Try to read stderr for more info
            err = ""
            try:
                import select
                err = self.process.stderr.read(2000)
            except Exception:
                pass
            raise RuntimeError(f"Server failed to start: stdout='{line}' stderr='{err}'")
        print(f"  [SERVER] Started successfully (READY)")

    def _check_alive(self):
        """Check if the Java process is still running."""
        if self.process and self.process.poll() is not None:
            rc = self.process.returncode
            err = ""
            try:
                err = self.process.stderr.read(2000)
            except Exception:
                pass
            print(f"  [SERVER ERROR] Java process DIED! exit code={rc}")
            if err:
                print(f"  [SERVER STDERR] {err[:500]}")
            return False
        return True

    def run_game(self, weights, opponent, map_path=None):
        """Run EVOLVED vs BOT. Returns (winner, cycles)."""
        if map_path is None:
            map_path = TEMP_MAP_REL
        self._total_count += 1

        if not self._check_alive():
            self._fail_count += 1
            return -1, MAX_CYCLES

        ws = ";".join(f"{w:.4f}" for w in weights)
        cmd = f"EVOLVED:{ws},{opponent},{map_path},{MAX_CYCLES}\n"
        try:
            self.process.stdin.write(cmd)
            self.process.stdin.flush()
            r = self.process.stdout.readline().strip()
        except (BrokenPipeError, OSError) as e:
            self._fail_count += 1
            if self._fail_count <= 3:
                print(f"  [SERVER ERROR] Pipe broken: {e}")
            return -1, MAX_CYCLES

        if r.startswith("RESULT:"):
            p = r.replace("RESULT:", "").split(",")
            winner, cycles = int(p[0]), int(p[1])

            # Same fix as run_game_evolved_vs_evolved: a timeout must not
            # be silently credited as a win for the evolved agent (which
            # always sits in player 0's slot against bots).
            if cycles >= MAX_CYCLES and winner != -1:
                self._forced_draw_count += 1
                if self._forced_draw_count <= 5:
                    print(f"  [SERVER FIX] Game #{self._total_count} hit "
                          f"MAX_CYCLES ({cycles}) but engine reported "
                          f"winner={winner} — forcing to draw (-1).")
                winner = -1

            return winner, cycles

        # Not a valid result — log it
        self._fail_count += 1
        if self._fail_count <= 5:
            print(f"  [SERVER WARNING] Game #{self._total_count}: "
                  f"Expected 'RESULT:...' got '{r[:80]}'")
            if not r:
                print(f"  [SERVER WARNING] Empty response — server likely crashed!")
                self._check_alive()
        return -1, MAX_CYCLES

    def run_game_evolved_vs_evolved(self, p0_weights, p1_weights, map_path=None):
        """Run EVOLVED vs EVOLVED. Returns (winner, cycles)."""
        if map_path is None:
            map_path = TEMP_MAP_REL
        self._total_count += 1

        if not self._check_alive():
            self._fail_count += 1
            return -1, MAX_CYCLES

        w0 = ";".join(f"{w:.4f}" for w in p0_weights)
        w1 = ";".join(f"{w:.4f}" for w in p1_weights)
        req = f"EVOLVED:{w0},EVOLVED:{w1},{map_path},{MAX_CYCLES}\n"
        try:
            self.process.stdin.write(req)
            self.process.stdin.flush()
            resp = self.process.stdout.readline().strip()
        except (BrokenPipeError, OSError) as e:
            self._fail_count += 1
            if self._fail_count <= 3:
                print(f"  [SERVER ERROR] Pipe broken: {e}")
            return -1, MAX_CYCLES

        if resp.startswith("RESULT:"):
            parts = resp.replace("RESULT:", "").split(",")
            winner, cycles = int(parts[0]), int(parts[1])

            # ── DIAGNOSTIC LOG (raw, unfiltered) ──────────────────────
            # Tally raw winner values so we can tell whether the engine
            # is genuinely deciding games or defaulting to a fixed slot.
            self._raw_winner_counts[winner] = self._raw_winner_counts.get(winner, 0) + 1

            # ── FIX: timeouts must not be silently credited as a win ──
            # If the game ran the full MAX_CYCLES without the engine
            # itself reporting a draw (-1), it was very likely undecided.
            # Some microRTS wrappers default the winner field to 0 in
            # that case instead of -1, which turns every timeout into a
            # "Player 0 win" and can make head-to-head results look
            # deterministic by board slot rather than by actual play
            # (exactly the 5W-5L-0D pattern split along swap/no-swap).
            # Force these to a draw so W/L/D bookkeeping isn't corrupted.
            if cycles >= MAX_CYCLES and winner != -1:
                self._forced_draw_count += 1
                if self._forced_draw_count <= 5:
                    print(f"  [SERVER FIX] Game #{self._total_count} hit "
                          f"MAX_CYCLES ({cycles}) but engine reported "
                          f"winner={winner} — forcing to draw (-1).")
                winner = -1

            return winner, cycles

        self._fail_count += 1
        if self._fail_count <= 5:
            print(f"  [SERVER WARNING] H2H Game #{self._total_count}: "
                  f"Expected 'RESULT:...' got '{resp[:80]}'")
        return -1, MAX_CYCLES

    def run_test_game(self):
        """Run a diagnostic test game to verify the server works."""
        print(f"  [DIAG] Running test game...")
        test_weights = [0.5] * DIMENSIONS
        w, c = self.run_game(test_weights, "WorkerRush")
        if w == -1 and self._fail_count > 0:
            print(f"  [DIAG] *** TEST GAME FAILED! Server may not support this map. ***")
            print(f"  [DIAG] Trying with basesWorkers32x32A.xml...")
            w2, c2 = self.run_game(test_weights, "WorkerRush",
                                   map_path="maps/basesWorkers32x32A.xml")
            if w2 != -1:
                print(f"  [DIAG] 32x32 map works (winner={w2}, cycles={c2})")
                print(f"  [DIAG] Problem is with the 48x48 temp map!")
            else:
                print(f"  [DIAG] 32x32 map also fails — server has deeper issues")
            return False
        else:
            print(f"  [DIAG] Test game OK: winner={w}, cycles={c}")
            return True

    def stop(self):
        if self._fail_count > 0:
            print(f"  [SERVER] Total games: {self._total_count}, "
                  f"Failures: {self._fail_count} "
                  f"({self._fail_count/max(1,self._total_count)*100:.1f}%)")
        if self._raw_winner_counts:
            print(f"  [SERVER DIAG] Raw winner distribution: "
                  f"{self._raw_winner_counts}")
        if self._forced_draw_count > 0:
            print(f"  [SERVER DIAG] Timeouts forced to draw: "
                  f"{self._forced_draw_count}")
        if self.process:
            try:
                self.process.stdin.write("EXIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()


# ═══════════════════════════════════════════════════════════════
#  PARALLEL GAME SERVER POOL
#  One live Java game server per worker; evaluations run concurrently.
# ═══════════════════════════════════════════════════════════════

class ParallelGamePool:
    def __init__(self, n_workers):
        self.n_workers = n_workers
        self.pool_queue = queue.Queue()

    def start(self):
        print(f"  [CPU] Starting {self.n_workers} parallel game servers...")
        for _ in range(self.n_workers):
            s = GameServer()
            s.start()
            self.pool_queue.put(s)
        print(f"  [OK] {self.n_workers} servers ready (of {os.cpu_count()} CPU cores)")

    def _eval_vs_agents_one(self, weights, opponents):
        """
        Evaluate one weight vector against all rival-algorithm weight vectors.
        Alternates which player slot `weights` occupies for fairness.
        Returns a win rate in [0, 1].
        """
        server = self.pool_queue.get()
        try:
            wins = 0.0
            n = len(opponents)
            if n == 0:
                return 0.0

            for i, opp_weights in enumerate(opponents):
                if i % 2 == 0:
                    winner, _ = server.run_game_evolved_vs_evolved(
                        weights, opp_weights
                    )
                    if winner == 0:
                        wins += 1.0
                    elif winner == -1:
                        wins += 0.3
                else:
                    winner, _ = server.run_game_evolved_vs_evolved(
                        opp_weights, weights
                    )
                    if winner == 1:
                        wins += 1.0
                    elif winner == -1:
                        wins += 0.3

            return wins / n
        finally:
            self.pool_queue.put(server)

    def eval_batch_vs_agents(self, weight_list, opponents):
        """
        Evaluate a batch of evolved agents in parallel against the same
        rival-algorithm weight vectors.
        """
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(self._eval_vs_agents_one, weights, opponents)
                for weights in weight_list
            ]
            return [future.result() for future in futures]

    def _eval_vs_bot_one(self, weights, opponent, k_games):
        """Evaluate one weight vector against one named bot."""
        server = self.pool_queue.get()
        try:
            wins = 0.0
            for _ in range(k_games):
                winner, _ = server.run_game(weights, opponent)
                if winner == 0:
                    wins += 1.0
                elif winner == -1:
                    wins += 0.3
            return wins / k_games
        finally:
            self.pool_queue.put(server)

    def eval_all_bots_parallel(self, weights, k_games):
        """
        Evaluate one weight vector against all seven built-in bots in parallel.
        Returns {bot_name: win_rate}.
        """
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                opponent: executor.submit(
                    self._eval_vs_bot_one, weights, opponent, k_games
                )
                for opponent in OPPONENT_ROTATION
            }
            return {
                opponent: futures[opponent].result()
                for opponent in OPPONENT_ROTATION
            }

    def validate_all_bots(self, weights):
        """
        Validate weights against all seven bots.
        Returns (overall_percentage, per_opponent_percentage_dict).
        """
        per_opp = self.eval_all_bots_parallel(weights, VALIDATION_GAMES)
        overall = sum(per_opp.values()) / len(per_opp)
        return overall * 100.0, {
            opponent: win_rate * 100.0
            for opponent, win_rate in per_opp.items()
        }

    def _run_h2h_game_one(self, p0_weights, p1_weights, swap):
        """
        Run one H2H game. `swap=True` means p1 is placed in player 0's slot.
        Returns (winner, swap).
        """
        server = self.pool_queue.get()
        try:
            if swap:
                winner, _ = server.run_game_evolved_vs_evolved(
                    p1_weights, p0_weights
                )
            else:
                winner, _ = server.run_game_evolved_vs_evolved(
                    p0_weights, p1_weights
                )
            return winner, swap
        finally:
            self.pool_queue.put(server)

    def run_h2h_parallel(self, p0_weights, p1_weights, n_games):
        """
        Run all H2H games for one algorithm pair concurrently.
        Alternates the player slots exactly as the original sequential version.
        """
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(
                    self._run_h2h_game_one,
                    p0_weights,
                    p1_weights,
                    g % 2 == 1
                )
                for g in range(n_games)
            ]
            return [future.result() for future in futures]

    def run_test_game(self):
        """Borrow one pooled server for the diagnostic test."""
        server = self.pool_queue.get()
        try:
            return server.run_test_game()
        finally:
            self.pool_queue.put(server)

    def run_mirror_match_check(self, weights, n_games=10):
        """
        DIAGNOSTIC: pit an identical weight vector against itself, swapping
        slots exactly like run_h2h_parallel does. Because both sides are
        the *same* strategy, a legitimate engine should produce results
        close to 50/50 (or all draws) — any lopsided split (e.g. every
        non-swapped game "won" and every swapped game "lost") proves the
        result is being decided by board position/slot, not by strategy,
        which is the bug this fix targets.
        Returns (wins_as_slot0, wins_as_slot1, draws).
        """
        results = self.run_h2h_parallel(weights, weights, n_games)
        slot0_wins, slot1_wins, draws = 0, 0, 0
        for winner, swap in results:
            if winner == -1:
                draws += 1
            elif (winner == 0 and not swap) or (winner == 1 and swap):
                slot0_wins += 1
            else:
                slot1_wins += 1
        print(f"\n  [MIRROR-MATCH CHECK] identical weights vs themselves, "
              f"{n_games} games:")
        print(f"    Slot-0 side won: {slot0_wins}  |  "
              f"Slot-1 side won: {slot1_wins}  |  Draws: {draws}")
        if slot0_wins >= n_games - 1 or slot1_wins >= n_games - 1:
            print(f"    ⚠ WARNING: near-total lopsidedness with IDENTICAL "
                  f"weights on both sides. This confirms the win is being "
                  f"decided by board/player slot, not by strategy.")
        else:
            print(f"    OK: roughly balanced — no strong slot bias detected.")
        return slot0_wins, slot1_wins, draws

    def stop(self):
        while not self.pool_queue.empty():
            self.pool_queue.get().stop()
        print("  [OK] All parallel game servers stopped")


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def euclidean_dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))





# ═══════════════════════════════════════════════════════════════
#  WILCOXON SIGNED-RANK TEST
# ═══════════════════════════════════════════════════════════════

def wilcoxon_test(x, y):
    """Pure Python Wilcoxon signed-rank test with continuity correction.
    Returns (p_value, n_samples) tuple, or (None, 0) if insufficient data.
    Note: Normal approximation is less reliable for n < 10."""
    diffs = [xi - yi for xi, yi in zip(x, y) if xi != yi]
    if len(diffs) < 3:
        return None, len(diffs)
    n     = len(diffs)
    abs_d = sorted(enumerate(abs(d) for d in diffs), key=lambda t: t[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and abs_d[j][1] == abs_d[j+1][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j+1):
            ranks[abs_d[k][0]] = avg_rank
        i = j + 1
    w_plus  = sum(ranks[i] for i, d in enumerate(diffs) if d > 0)
    w_minus = sum(ranks[i] for i, d in enumerate(diffs) if d < 0)
    w_stat  = min(w_plus, w_minus)
    mu_w    = n * (n + 1) / 4.0
    sig_w   = math.sqrt(n * (n + 1) * (2*n + 1) / 24.0)
    if sig_w == 0:
        return None, n
    # Continuity correction (improves normal approximation for small samples)
    z = (abs(w_stat - mu_w) - 0.5) / sig_w
    z = max(z, 0.0)  # ensure non-negative after correction
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2))))
    return p, n


# ═══════════════════════════════════════════════════════════════
#  SENSITIVITY ANALYSIS LOADER
# ═══════════════════════════════════════════════════════════════

def run_sensitivity_analysis(pool):
    """Run sensitivity analysis to get per-weight mutation rates."""
    print("  Running sensitivity analysis...")
    baseline = [0.5] * DIMENSIONS

    baseline_per_opp = pool.eval_all_bots_parallel(baseline, K_GAMES)
    baseline_wr = sum(baseline_per_opp.values()) / len(baseline_per_opp)
    print(f"  Baseline win rate: {baseline_wr:.1%}")

    influence = []
    for w_idx in range(DIMENSIONS):
        results = []
        for val in [0.0, 1.0]:
            test = baseline[:]
            test[w_idx] = val
            per_opp = pool.eval_all_bots_parallel(test, K_GAMES)
            results.append(
                sum(per_opp.values()) / len(per_opp)
            )

        influence.append(abs(results[1] - results[0]))
        print(
            f"  w{w_idx:2d} {WEIGHT_NAMES[w_idx]:<22s} "
            f"influence={influence[-1]:.3f}"
        )

    max_inf    = max(influence) if max(influence) > 0 else 1.0
    normalized = [0.08 + (1.0 - inf / max_inf) * 0.12 for inf in influence]

    data = {
        "influence_scores":          influence,
        "normalized_mutation_rates": normalized,
        "baseline_winrate":          baseline_wr,
        "weight_names":              WEIGHT_NAMES,
    }
    fp = os.path.join(SCRIPT_DIR, SENSITIVITY_FILE)
    with open(fp, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved to {SENSITIVITY_FILE}")
    return normalized


def load_or_run_sensitivity(pool):
    """Load sensitivity scores or run analysis if missing."""
    # Check in PR1_Sem7 directory first, then parent Codes directory
    for search_dir in [SCRIPT_DIR, CODES_DIR]:
        fp = os.path.join(search_dir, SENSITIVITY_FILE)
        if os.path.exists(fp):
            print(f"  Loading {fp}...")
            with open(fp) as f:
                data = json.load(f)
            rates = data["normalized_mutation_rates"]
            if min(rates) < 0.07 or max(rates) > 0.21:
                print("  WARNING: sensitivity_scores.json has out-of-range mutation rates — re-running analysis.")
                break
            return rates
    print(f"  {SENSITIVITY_FILE} not found — running analysis...")
    rates = run_sensitivity_analysis(pool)
    return rates


# ═══════════════════════════════════════════════════════════════
#  CCE-TDCCEA CLASSES
#  (From COMPARE_ALL.py — Sub-population cooperative coevolution
#   with trajectory-guided mutation)
# ═══════════════════════════════════════════════════════════════

class SubAgent:
    """Sub-agent for CCE-TDCCEA (handles either macro or micro weights).
    env_offset: starting index into the concatenated env vector for this
    sub-agent's portion (0 for macro, len(MACRO_INDICES) for micro)."""
    def __init__(self, dim, params=None, env_offset=0):
        self.dim    = dim
        self.env_offset = env_offset
        self.params = ([random.random() for _ in range(dim)] if params is None
                       else [max(0.0, min(1.0, float(x))) for x in params])
        self.H_i  = []
        self.e_i  = 0.0
        self.mu_i = 0.0

    def update_history(self, f, env):
        self.H_i.append((f, env))
        if len(self.H_i) > 7: self.H_i.pop(0)

    def compute_trajectory(self, env):
        if not self.H_i: self.e_i = 0.0; return
        md = math.sqrt(self.dim)
        off = self.env_offset
        ws = [max(0.0, 1.0 - euclidean_dist(env[off:off+self.dim], pe[off:off+self.dim]) / md)
              if len(pe) >= off + self.dim else 1.0 for _, pe in self.H_i]
        t  = sum(ws)
        ws = [w/t for w in ws] if t > 0 else [1.0/len(ws)] * len(ws)
        self.e_i = sum(w * self.H_i[i][0] for i, w in enumerate(ws))

    def compute_mutation(self): self.mu_i = 0.15 * (1.0 - self.e_i) + 0.03

    def mutate(self):
        return SubAgent(self.dim,
            [max(0.0, min(1.0, p + random.gauss(0, self.mu_i))) for p in self.params],
            env_offset=self.env_offset)


def combine_subagents(macro, micro):
    """Combine macro and micro sub-agents into a full weight vector."""
    full = [0.5] * DIMENSIONS
    for i, idx in enumerate(MACRO_INDICES): full[idx] = macro.params[i]
    for i, idx in enumerate(MICRO_INDICES): full[idx] = micro.params[i]
    return full


def subpop_survival(pop, archive):
    """Survival selection with diversity maintenance and archive."""
    dim  = pop[0].dim
    md   = math.sqrt(dim)
    survivors = []
    for a in sorted(pop, key=lambda a: a.e_i, reverse=True):
        if not any(1.0 - euclidean_dist(a.params, u.params) / md > DELTA_S
                   for u in survivors):
            survivors.append(a)
    final = [a for a in survivors if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
    while len(final) < 8 and len(survivors) > len(final):
        for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
            if a not in final: final.append(a); break
    for top in sorted(pop, key=lambda a: a.e_i, reverse=True)[:3]:
        ar = SubAgent(dim, top.params, env_offset=top.env_offset)
        ar.e_i, ar.H_i, ar.mu_i = top.e_i, top.H_i[:], top.mu_i
        archive.append(ar)
    archive.sort(key=lambda a: a.e_i, reverse=True)
    del archive[A_MAX:]
    pool    = final + archive
    new_pop = final[:]
    while len(new_pop) < N_POP:
        cands = random.sample(pool, min(3, len(pool)))
        new_pop.append(max(cands, key=lambda a: a.e_i).mutate())
    return new_pop, archive


# ═══════════════════════════════════════════════════════════════
#  MONOLITHIC AGENT
#  (From COMPARE_ALL.py — used by TD-CCEA and SA-TDCCEA)
# ═══════════════════════════════════════════════════════════════

class Agent:
    """Monolithic agent for TD-CCEA and SA-TDCCEA."""
    def __init__(self, params=None, pwm=None):
        self.params = ([random.random() for _ in range(DIMENSIONS)] if params is None
                       else [max(0.0, min(1.0, float(x))) for x in params])
        self.H_i  = []
        self.e_i  = 0.0
        self.mu_i = 0.0
        self.pwm  = pwm

    def update_history(self, f, env):
        self.H_i.append((f, env))
        if len(self.H_i) > 7: self.H_i.pop(0)

    def compute_trajectory(self, env):
        if not self.H_i: self.e_i = 0.0; return
        md = math.sqrt(DIMENSIONS)
        ws = [max(0.0, 1.0 - euclidean_dist(env, pe) / md) for _, pe in self.H_i]
        t  = sum(ws)
        ws = [w/t for w in ws] if t > 0 else [1.0/len(ws)] * len(ws)
        self.e_i = sum(w * self.H_i[i][0] for i, w in enumerate(ws))

    def compute_mutation(self): self.mu_i = 0.15 * (1.0 - self.e_i) + 0.03

    def mutate(self):
        # Offspring start with fresh trajectory history (e_i=0, H_i=[])
        # so they earn their own fitness estimate through evaluation.
        cp = []
        if self.pwm:
            for j, p in enumerate(self.params):
                sigma_j = self.pwm[j] * (1.0 - self.e_i) + 0.02
                cp.append(max(0.0, min(1.0, p + random.gauss(0, sigma_j))))
        else:
            for p in self.params:
                cp.append(max(0.0, min(1.0, p + random.gauss(0, self.mu_i))))
        return Agent(cp, self.pwm)


# ═══════════════════════════════════════════════════════════════
#  ALGORITHM 1 — CCE-TDCCEA
#  (Now evolves against the CURRENT BEST of the other 3 algorithms,
#   one generation at a time, instead of against a fixed bot.)
# ═══════════════════════════════════════════════════════════════

class CCEAState:
    """Holds CCE-TDCCEA's evolving state across synchronized generations."""
    def __init__(self):
        self.pop_a   = [SubAgent(len(MACRO_INDICES), env_offset=0) for _ in range(N_POP)]
        self.pop_b   = [SubAgent(len(MICRO_INDICES), env_offset=len(MACRO_INDICES)) for _ in range(N_POP)]
        self.arc_a   = []
        self.arc_b   = []
        self.top_a   = self.pop_a[:TOP_PARTNERS]
        self.top_b   = self.pop_b[:TOP_PARTNERS]
        self.best_a  = self.pop_a[0]
        self.best_b  = self.pop_b[0]
        self.t0      = time.time()
        self.convergence = []

    def current_best_weights(self):
        return combine_subagents(self.best_a, self.best_b)


def step_ccea(pool, state, opponents, gen):
    """Run ONE generation of CCE-TDCCEA against rival-algorithm `opponents`."""
    pop_a, pop_b = state.pop_a, state.pop_b
    top_a, top_b = state.top_a, state.top_b
    ca = [
        sum(a.params[j] for a in pop_a) / len(pop_a)
        for j in range(len(MACRO_INDICES))
    ]
    cb = [
        sum(b.params[j] for b in pop_b) / len(pop_b)
        for j in range(len(MICRO_INDICES))
    ]
    env = ca + cb

    # Evaluate every macro agent × top micro partner combination in parallel.
    macro_weight_list = [
        combine_subagents(a, pb)
        for a in pop_a
        for pb in top_b
    ]
    macro_fitness = pool.eval_batch_vs_agents(
        macro_weight_list, opponents
    )

    n_micro_partners = len(top_b)
    for idx, a in enumerate(pop_a):
        fs = macro_fitness[
            idx * n_micro_partners:(idx + 1) * n_micro_partners
        ]
        a.update_history(sum(fs) / len(fs), env)
        a.compute_trajectory(env)
        a.compute_mutation()

    # Evaluate every micro agent × top macro partner combination in parallel.
    micro_weight_list = [
        combine_subagents(pa, b)
        for b in pop_b
        for pa in top_a
    ]
    micro_fitness = pool.eval_batch_vs_agents(
        micro_weight_list, opponents
    )

    n_macro_partners = len(top_a)
    for idx, b in enumerate(pop_b):
        fs = micro_fitness[
            idx * n_macro_partners:(idx + 1) * n_macro_partners
        ]
        b.update_history(sum(fs) / len(fs), env)
        b.compute_trajectory(env)
        b.compute_mutation()

    state.best_a = max(pop_a, key=lambda a: a.e_i)
    state.best_b = max(pop_b, key=lambda a: a.e_i)
    state.top_a = sorted(
        pop_a, key=lambda a: a.e_i, reverse=True
    )[:TOP_PARTNERS]
    state.top_b = sorted(
        pop_b, key=lambda a: a.e_i, reverse=True
    )[:TOP_PARTNERS]

    # Evaluate the current best combined solution through the same pool.
    best_combined = combine_subagents(state.best_a, state.best_b)
    cmb = pool.eval_batch_vs_agents([best_combined], opponents)[0]
    state.convergence.append(cmb)

    print(
        f"  [CCE-TDCCEA] Gen {gen:2d}/{TOTAL_GENERATIONS} | "
        f"A={state.best_a.e_i:.3f} B={state.best_b.e_i:.3f} Cmb={cmb:.3f}"
    )

    state.pop_a, state.arc_a = subpop_survival(pop_a, state.arc_a)
    state.pop_b, state.arc_b = subpop_survival(pop_b, state.arc_b)

    return combine_subagents(state.best_a, state.best_b)


# ═══════════════════════════════════════════════════════════════
#  ALGORITHM 2 — STANDARD CCEA
# ═══════════════════════════════════════════════════════════════

STD_MUT = 0.15


class StdCCEAState:
    """Holds Standard CCEA's evolving state across synchronized generations."""
    def __init__(self):
        self.pm = [[random.random() for _ in range(len(MACRO_INDICES))] for _ in range(N_POP)]
        self.pi = [[random.random() for _ in range(len(MICRO_INDICES))] for _ in range(N_POP)]
        self.fm = [0.0] * N_POP
        self.fi = [0.0] * N_POP
        self.t0 = time.time()
        self.convergence = []

    def current_best_weights(self):
        bm  = max(range(N_POP), key=lambda i: self.fm[i])
        bi2 = max(range(N_POP), key=lambda i: self.fi[i])
        return std_ccea_mk(self.pm[bm], self.pi[bi2])


def std_ccea_mk(mac, mic):
    f = [0.5] * DIMENSIONS
    for k, idx in enumerate(MACRO_INDICES): f[idx] = mac[k]
    for k, idx in enumerate(MICRO_INDICES): f[idx] = mic[k]
    return f


def _std_ccea_breed(pop, fits):
    bi      = max(range(len(pop)), key=lambda i: fits[i])
    new_pop = [pop[bi][:]]
    while len(new_pop) < N_POP:
        t    = random.sample(range(len(pop)), min(3, len(pop)))
        pidx = max(t, key=lambda i: fits[i])
        new_pop.append([max(0.0, min(1.0, x + random.gauss(0, STD_MUT))) for x in pop[pidx]])
    return new_pop


def step_std_ccea(pool, state, opponents, gen):
    """Run ONE generation of Standard CCEA against rival-algorithm `opponents`."""
    pm, pi = state.pm, state.pi
    tmi = sorted(
        range(N_POP),
        key=lambda i: state.fi[i],
        reverse=True
    )[:TOP_PARTNERS]
    tma = sorted(
        range(N_POP),
        key=lambda i: state.fm[i],
        reverse=True
    )[:TOP_PARTNERS]

    macro_weight_list = [
        std_ccea_mk(pm[i], pi[j])
        for i in range(N_POP)
        for j in tmi
    ]
    macro_fitness = pool.eval_batch_vs_agents(
        macro_weight_list, opponents
    )
    for i in range(N_POP):
        fs = macro_fitness[
            i * len(tmi):(i + 1) * len(tmi)
        ]
        state.fm[i] = sum(fs) / len(fs)

    micro_weight_list = [
        std_ccea_mk(pm[j], pi[i])
        for i in range(N_POP)
        for j in tma
    ]
    micro_fitness = pool.eval_batch_vs_agents(
        micro_weight_list, opponents
    )
    for i in range(N_POP):
        fs = micro_fitness[
            i * len(tma):(i + 1) * len(tma)
        ]
        state.fi[i] = sum(fs) / len(fs)

    bm = max(range(N_POP), key=lambda i: state.fm[i])
    bi2 = max(range(N_POP), key=lambda i: state.fi[i])

    best_weights = std_ccea_mk(pm[bm], pi[bi2])
    bf = pool.eval_batch_vs_agents([best_weights], opponents)[0]

    state.convergence.append(bf)
    print(
        f"  [Std CCEA]   Gen {gen:2d}/{TOTAL_GENERATIONS} | "
        f"Best={bf:.3f}"
    )

    # Save best fitness before breeding (elite is placed at index 0 by _std_ccea_breed)
    best_fm = state.fm[bm]
    best_fi = state.fi[bi2]
    state.pm = _std_ccea_breed(pm, state.fm)
    state.pi = _std_ccea_breed(pi, state.fi)
    # Carry elite fitness forward so top-partner selection is correct next generation
    state.fm = [0.0] * N_POP
    state.fi = [0.0] * N_POP
    state.fm[0] = best_fm
    state.fi[0] = best_fi

    return best_weights


# ═══════════════════════════════════════════════════════════════
#  ALGORITHMS 3/4 — TD-CCEA / SA-TDCCEA
# ═══════════════════════════════════════════════════════════════

class EvolveCoreState:
    """Holds TD-CCEA / SA-TDCCEA's evolving state across synchronized generations."""
    def __init__(self, label, pwm=None):
        self.label      = label
        self.pwm        = pwm
        self.population = [Agent(pwm=pwm) for _ in range(N_POP)]
        self.archive    = []
        self.t0         = time.time()
        self.convergence = []

    def current_best_weights(self):
        pool = self.archive if self.archive else self.population
        return max(pool, key=lambda a: a.e_i).params


def step_evolve_core(pool, state, opponents, gen):
    """Run ONE generation of TD-CCEA / SA-TDCCEA against rival-algorithm `opponents`."""
    population = state.population
    pwm = state.pwm
    centroid = [
        sum(a.params[j] for a in population) / len(population)
        for j in range(DIMENSIONS)
    ]

    weight_list = [a.params for a in population]
    fitness_values = pool.eval_batch_vs_agents(
        weight_list, opponents
    )

    for agent, f_i in zip(population, fitness_values):
        agent.update_history(f_i, centroid)
        agent.compute_trajectory(centroid)
        agent.compute_mutation()

    best = max(population, key=lambda a: a.e_i)
    state.convergence.append(best.e_i)
    print(
        f"  [{state.label:11s}] Gen {gen:2d}/{TOTAL_GENERATIONS} | "
        f"e_i={best.e_i:.3f}"
    )

    md = math.sqrt(DIMENSIONS)
    survivors = []
    for a in sorted(
        population, key=lambda a: a.e_i, reverse=True
    ):
        if not any(
            1.0 - euclidean_dist(a.params, u.params) / md > DELTA_S
            for u in survivors
        ):
            survivors.append(a)

    final = [
        a for a in survivors
        if not (a.e_i < DELTA_F and len(a.H_i) >= 3)
    ]

    while len(final) < 10 and len(survivors) > len(final):
        for a in sorted(
            survivors, key=lambda a: a.e_i, reverse=True
        ):
            if a not in final:
                final.append(a)
                break

    for top in sorted(
        population, key=lambda a: a.e_i, reverse=True
    )[:3]:
        ar = Agent(top.params, pwm)
        ar.e_i, ar.H_i, ar.mu_i = top.e_i, top.H_i[:], top.mu_i
        state.archive.append(ar)

    state.archive.sort(key=lambda a: a.e_i, reverse=True)
    state.archive = state.archive[:A_MAX]

    breed_pool = final + state.archive
    new_population = final[:]

    while len(new_population) < N_POP:
        cands = random.sample(
            breed_pool, min(3, len(breed_pool))
        )
        new_population.append(
            max(cands, key=lambda a: a.e_i).mutate()
        )

    state.population = new_population

    best_all = (
        max(state.archive, key=lambda a: a.e_i)
        if state.archive else best
    )
    return best_all.params


# ═══════════════════════════════════════════════════════════════
#  HEAD-TO-HEAD TOURNAMENT
# ═══════════════════════════════════════════════════════════════

def run_head_to_head(pool, algo_weights, algo_names):
    """
    Run head-to-head tournament between all algorithm pairs.
    Returns a matrix of results: matrix[i][j] = (wins_i, losses_i, draws)
    """
    n = len(algo_names)
    matrix = [[None] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            i_wins, j_wins, draws = 0, 0, 0

            game_results = pool.run_h2h_parallel(
                algo_weights[i],
                algo_weights[j],
                H2H_GAMES
            )

            for winner, swap in game_results:
                if not swap:
                    if winner == 0:
                        i_wins += 1
                    elif winner == 1:
                        j_wins += 1
                    else:
                        draws += 1
                else:
                    if winner == 0:
                        j_wins += 1
                    elif winner == 1:
                        i_wins += 1
                    else:
                        draws += 1

            matrix[i][j] = (i_wins, j_wins, draws)
            matrix[j][i] = (j_wins, i_wins, draws)

            print(
                f"  {algo_names[i]:15s} vs {algo_names[j]:15s} → "
                f"{i_wins}W-{j_wins}L-{draws}D"
            )

    return matrix


def print_h2h_matrix(matrix, algo_names):
    """Print head-to-head results in the image format."""
    n = len(algo_names)
    col_w = 14  # column width

    # Compute totals
    total_wins = [0] * n
    total_games = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i][j] is not None:
                w, l, d = matrix[i][j]
                total_wins[i]  += w
                total_games[i] += (w + l + d)
    win_rates = [total_wins[i] / total_games[i] * 100 if total_games[i] > 0 else 0
                 for i in range(n)]

    # Header
    header = f"  {'':17s}"
    for name in algo_names:
        header += f" {name:>{col_w}s}"
    header += f"  {'Wins':>6s}  {'WR%':>6s}"
    print(header)

    # Separator
    sep = "  " + "─" * 17
    for _ in algo_names:
        sep += " " + "─" * col_w
    sep += "  " + "─" * 6 + "  " + "─" * 6
    print(sep)

    # Rows
    for i in range(n):
        row = f"  {algo_names[i]:17s}"
        for j in range(n):
            if i == j:
                row += f" {'—':>{col_w}s}"
            else:
                w, l, d = matrix[i][j]
                cell = f"{w}W-{l}L-{d}D"
                row += f" {cell:>{col_w}s}"
        row += f"  {total_wins[i]:>6d}  {win_rates[i]:>5.1f}%"
        print(row)

    print(sep)
    return total_wins, win_rates


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "#" * 70)
    print("  SEM7 PR1 — Head-to-Head Algorithm Comparison")
    print("  48×48 Dynamic Map with Progressive Blockades")
    print("#" * 70)
    print(f"  Pop={N_POP} | Gens={TOTAL_GENERATIONS} | K={K_GAMES} | "
          f"H2H Games={H2H_GAMES}/pair | Map=48×48")
    print(f"  Algorithms: CCE-TDCCEA, TD-CCEA, SA-TDCCEA, Standard CCEA\n")

    # ──────────────────────────────────────────────────────────────
    #  PHASE 0: Sensitivity analysis (needed for SA-TDCCEA)
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 0: Loading/Running Sensitivity Analysis")
    print(f"{'='*60}")

    # Write initial clean map for sensitivity analysis
    write_map_xml(set(), set(), TEMP_MAP_PATH)
    print(f"  Wrote temp map: {TEMP_MAP_PATH}")
    print(f"  Temp map exists: {os.path.isfile(TEMP_MAP_PATH)}")

    # One shared parallel pool is used for every evaluation phase.
    pool = ParallelGamePool(N_WORKERS)
    pool.start()

    # Run diagnostic test game to verify one pooled server + map work.
    if not pool.run_test_game():
        print("\n  *** DIAGNOSTIC FAILED ***")
        print("  The Java game server cannot run games on the 48x48 map.")
        print("  Possible causes:")
        print("    1. The map XML format is invalid for this server version")
        print("    2. The Java EvolvedBatchGameRunner doesn't support 48x48")
        print("    3. The temp map path is wrong")
        print(f"  Map path (abs): {TEMP_MAP_PATH}")
        print(f"  Map path (rel): {TEMP_MAP_REL}")
        pool.stop()
        return

    sa_mut = load_or_run_sensitivity(pool)
    print()

    # ──────────────────────────────────────────────────────────────
    #  PHASE 1: EVOLUTION (with dynamic blockades)
    #  All 4 algorithms now evolve TOGETHER, generation by generation.
    #  Each individual's fitness is its win rate against the CURRENT BEST
    #  weights of the other 3 algorithms (as of the previous generation),
    #  instead of against a fixed built-in bot.
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 1: EVOLUTION — 4 Algorithms × {TOTAL_GENERATIONS} Generations")
    print(f"  (co-evolving against each other, not built-in bots)")
    print(f"  Dynamic blockades added progressively (BFS-validated)")
    print(f"{'='*60}")

    algos = [
        ("CCE-TDCCEA",    "ccea"),
        ("TD-CCEA",       "tdccea"),
        ("SA-TDCCEA",     "sa_tdccea"),
        ("Std CCEA",      "std_ccea"),
    ]

    # Progressive blockades: walls are added EACH generation during evolution.
    # Start with a clean map; each generation adds new walls, verifies BFS
    # reachability between both bases, rewrites the map XML, then evolves.
    # The final blockaded map is kept for the H2H tournament evaluation.
    terrain_walls     = set()
    breakable_positions = set()

    # Write the initial clean map (no walls)
    write_map_xml(terrain_walls, breakable_positions, TEMP_MAP_PATH)
    print(f"  Starting with clean 48×48 map (0 walls)")

    # Initialize per-algorithm state
    ccea_state      = CCEAState()
    std_state       = StdCCEAState()
    tdccea_state    = EvolveCoreState("TD-CCEA")
    sa_tdccea_state = EvolveCoreState("SA-TDCCEA", pwm=sa_mut)

    states = {
        "ccea":      ccea_state,
        "tdccea":    tdccea_state,
        "sa_tdccea": sa_tdccea_state,
        "std_ccea":  std_state,
    }
    step_fns = {
        "ccea":      lambda opp, gen: step_ccea(pool, ccea_state, opp, gen),
        "tdccea":    lambda opp, gen: step_evolve_core(pool, tdccea_state, opp, gen),
        "sa_tdccea": lambda opp, gen: step_evolve_core(pool, sa_tdccea_state, opp, gen),
        "std_ccea":  lambda opp, gen: step_std_ccea(pool, std_state, opp, gen),
    }
    keys = ["ccea", "tdccea", "sa_tdccea", "std_ccea"]

    # Initial "current best" weights (before any evolution) used as opponents
    # for generation 1 — taken from each algorithm's randomly-initialized state.
    current_best = {k: states[k].current_best_weights() for k in keys}

    t_evolution_start = time.time()
    for gen in range(1, TOTAL_GENERATIONS + 1):
        # ── DYNAMIC ENVIRONMENT: Add walls for this generation ──
        terrain_walls, breakable_positions, n_terrain, n_breakable = \
            add_walls_for_generation(gen, terrain_walls, breakable_positions)
        total_walls = len(terrain_walls) + len(breakable_positions)

        # Verify BFS reachability between both bases
        # (must match the actual base positions used in pick_wall_position
        # and write_map_xml — (61, 61) was off-grid for a 48×48 map and made
        # this check unconditionally report BLOCKED)
        base0, base1 = (2, 2), (MAP_W - 3, MAP_H - 3)
        reachable = bfs_reachable(terrain_walls | breakable_positions, base0, base1)
        status = "✓ REACHABLE" if reachable else "✗ BLOCKED"

        # Rewrite map XML with the updated walls for this generation
        write_map_xml(terrain_walls, breakable_positions, TEMP_MAP_PATH)

        print(f"\n  ┌─ Generation {gen}/{TOTAL_GENERATIONS} {'─'*40}")
        print(f"  │  Map: +{n_terrain} terrain +{n_breakable} breakable → "
              f"{total_walls} total walls | BFS: {status}")

        # All 4 algorithms use the SAME snapshot of last generation's bests
        # (fully synchronized schedule) — computed here before anyone updates.
        opponents_for = {
            k: [current_best[o] for o in keys if o != k] for k in keys
        }
        new_best = {}
        for k in keys:
            new_best[k] = step_fns[k](opponents_for[k], gen)
        current_best = new_best  # simultaneous update after all 4 have stepped
        print(f"  └{'─'*55}")

    # Final map stats
    final_total = len(terrain_walls) + len(breakable_positions)
    print(f"\n  Final map: {final_total} total walls "
          f"({len(terrain_walls)} terrain + {len(breakable_positions)} breakable)")
    print(f"  This final blockaded map will be used for the H2H tournament.")

    elapsed_shared = time.time() - t_evolution_start

    # Validate each algorithm's final evolved weights against all 7 bots.
    results = []
    for algo_name, algo_key in algos:
        best = current_best[algo_key]
        print(f"\n  Validating {algo_name} against all 7 bots...")
        wr, per_opp = pool.validate_all_bots(best)
        print(f"  {algo_name}: {wr:.1f}% overall")
        results.append({
            "name":     algo_name,
            "key":      algo_key,
            "time":     elapsed_shared,
            "wr":       wr,
            "per_opp":  per_opp,
            "weights":  best,
            "convergence": states[algo_key].convergence,
        })

    # ──────────────────────────────────────────────────────────────
    #  RANKING TABLE (vs bots)
    # ──────────────────────────────────────────────────────────────
    ranked = sorted(results, key=lambda x: x["wr"], reverse=True)
    medals = ["1st","2nd","3rd","4th"]

    print(f"\n{'='*60}")
    print(f"  RANKING — Win Rate vs 7 Bots")
    print(f"{'='*60}")
    print(f"  {'Rank':<6s} {'Algorithm':<18s} {'Win Rate':>10s} {'Time':>8s}")
    print(f"  {'_'*6} {'_'*18} {'_'*10} {'_'*8}")
    for i, r in enumerate(ranked):
        print(f"  {medals[i]:<6s} {r['name']:<18s} {r['wr']:>9.1f}%  {r['time']:>7.1f}s")

    # ── PER-BOT TABLE ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  WIN RATE PER BOT (%)")
    print(f"{'='*60}")
    header = f"  {'Bot':<26s}"
    for r in ranked: header += f" {r['name'][:12]:>13s}"
    print(header)
    print("  " + "-" * (26 + 14 * len(ranked)))
    for opp in OPPONENT_ROTATION:
        row   = f"  {opp:<26s}"
        rates = [r["per_opp"].get(opp, 0) for r in ranked]
        mx    = max(rates)
        for r in ranked:
            w  = r["per_opp"].get(opp, 0)
            mk = "*" if w == mx and rates.count(mx) == 1 else " "
            row += f" {w:>11.0f}%{mk}"
        print(row)
    print("  " + "-" * (26 + 14 * len(ranked)))

    # ──────────────────────────────────────────────────────────────
    #  PHASE 2: HEAD-TO-HEAD TOURNAMENT
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PHASE 2: HEAD-TO-HEAD TOURNAMENT")
    print(f"  Each pair plays {H2H_GAMES} games (alternating P0/P1)")
    print(f"  Map: 48×48 with {len(terrain_walls) + len(breakable_positions)} blockades")
    print(f"{'='*70}\n")

    # Write tournament map
    write_map_xml(terrain_walls, breakable_positions, TEMP_MAP_PATH)

    algo_names   = [r["name"] for r in results]
    algo_weights = [r["weights"] for r in results]

    # ── DIAGNOSTIC: mirror-match sanity check ─────────────────────
    # Run before the real tournament so a bad result here is caught
    # before you spend time interpreting a corrupted matrix.
    pool.run_mirror_match_check(algo_weights[0], n_games=H2H_GAMES)

    matrix = run_head_to_head(pool, algo_weights, algo_names)

    # Print matrix
    print(f"\n{'='*70}")
    print(f"  HEAD-TO-HEAD RESULTS MATRIX")
    print(f"{'='*70}\n")
    total_wins, win_rates = print_h2h_matrix(matrix, algo_names)

    # ──────────────────────────────────────────────────────────────
    #  PHASE 3: STATISTICAL ANALYSIS
    # ──────────────────────────────────────────────────────────────

    # ── WILCOXON SIGNED-RANK TEST ────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  STATISTICAL SIGNIFICANCE — Wilcoxon Signed-Rank Test")
    print(f"  (CCE-TDCCEA vs each baseline, per-bot win rates as samples)")
    print(f"{'='*70}")

    cce = next((r for r in results if r["key"] == "ccea"), None)
    wilcoxon_lines = []
    wilcoxon_lines.append(f"Wilcoxon Signed-Rank Test Results")
    wilcoxon_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    wilcoxon_lines.append(f"Map: 48×48 dynamic blockades")
    wilcoxon_lines.append(f"Generations: {TOTAL_GENERATIONS}, Pop: {N_POP}, K_Games: {K_GAMES}")
    wilcoxon_lines.append(f"Reference algorithm: CCE-TDCCEA")
    wilcoxon_lines.append("")
    wilcoxon_lines.append(f"{'Baseline':<20s}  {'p-value':>9s}  {'Significant?':>13s}  Direction")
    wilcoxon_lines.append(f"{'_'*20}  {'_'*9}  {'_'*13}  {'_'*22}")

    if cce:
        cce_rates = [cce["per_opp"].get(o, 0) for o in OPPONENT_ROTATION]
        print(f"\n  {'Baseline':<20s}  {'p-value':>9s}  {'Significant?':>13s}  Direction")
        print(f"  {'_'*20}  {'_'*9}  {'_'*13}  {'_'*22}")
        for r in results:
            if r["key"] == "ccea": continue
            other = [r["per_opp"].get(o, 0) for o in OPPONENT_ROTATION]
            p, n_samp = wilcoxon_test(cce_rates, other)
            if p is None:
                line = f"{r['name']:<20s}  {'N/A':>9s}  {'N/A':>13s}  —"
            else:
                sig  = "YES (p<0.05)" if p < 0.05 else "NO  (p>=0.05)"
                dirn = ("CCE-TDCCEA BETTER" if sum(cce_rates) > sum(other)
                        else "BASELINE BETTER")
                warn = f" (n={n_samp}, low power)" if n_samp < 10 else f" (n={n_samp})"
                line = f"{r['name']:<20s}  {p:>9.4f}  {sig:>13s}  {dirn}{warn}"
            print(f"  {line}")
            wilcoxon_lines.append(line)

    # Add head-to-head Wilcoxon section
    wilcoxon_lines.append("")
    wilcoxon_lines.append("=" * 60)
    wilcoxon_lines.append("Head-to-Head Tournament Summary")
    wilcoxon_lines.append("=" * 60)
    for i, name in enumerate(algo_names):
        games = sum(1 for j in range(len(algo_names)) if i != j)
        wilcoxon_lines.append(
            f"{name:<17s}  Total Wins: {total_wins[i]:>3d}  "
            f"Win Rate: {win_rates[i]:>5.1f}%")

    # Save wilcoxon_results.txt
    wilcoxon_path = os.path.join(SCRIPT_DIR, "wilcoxon_results.txt")
    with open(wilcoxon_path, "a", encoding="utf-8") as f:
        f.write("\n" + "="*70 + "\n")
        f.write("\n".join(wilcoxon_lines) + "\n")
    print(f"\n  Saved → {wilcoxon_path}")

    # ── ABLATION ANALYSIS ────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ABLATION ANALYSIS")
    print(f"{'='*70}")

    ablation_lines = []
    ablation_lines.append(f"Ablation Analysis Results")
    ablation_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    ablation_lines.append(f"Map: 48×48 dynamic blockades")
    ablation_lines.append(f"Generations: {TOTAL_GENERATIONS}, Pop: {N_POP}, K_Games: {K_GAMES}")
    ablation_lines.append("")

    cce   = next((r for r in results if r["key"] == "ccea"),     None)
    sccea = next((r for r in results if r["key"] == "std_ccea"), None)
    cent  = next((r for r in results if r["key"] == "tdccea"),   None)
    sa    = next((r for r in results if r["key"] == "sa_tdccea"),None)

    # 1. Trajectory memory contribution: CCE-TDCCEA vs Std CCEA
    if cce and sccea:
        d = cce["wr"] - sccea["wr"]
        lines_out = [
            f"1. Trajectory Memory Contribution",
            f"   CCE-TDCCEA ({cce['wr']:.1f}%) vs Std CCEA ({sccea['wr']:.1f}%)",
            f"   Difference: {d:+.1f}%",
            f"   Result: {'Trajectory HELPS' if d >= 0 else 'Trajectory HURTS'} ({abs(d):.1f}%)",
            "",
        ]
        for line in lines_out: print(f"  {line}")
        ablation_lines.extend(lines_out)

    # 2. Cooperative decomposition: CCE-TDCCEA vs TD-CCEA
    if cce and cent:
        d = cce["wr"] - cent["wr"]
        lines_out = [
            f"2. Cooperative Decomposition Contribution",
            f"   CCE-TDCCEA ({cce['wr']:.1f}%) vs TD-CCEA ({cent['wr']:.1f}%)",
            f"   Difference: {d:+.1f}%",
            f"   Result: {'Decomposition HELPS' if d >= 0 else 'Decomposition HURTS'} ({abs(d):.1f}%)",
            "",
        ]
        for line in lines_out: print(f"  {line}")
        ablation_lines.extend(lines_out)

    # 3. Sensitivity-adaptive mutation: SA-TDCCEA vs TD-CCEA
    if sa and cent:
        d = sa["wr"] - cent["wr"]
        lines_out = [
            f"3. Sensitivity-Adaptive Mutation Contribution",
            f"   SA-TDCCEA ({sa['wr']:.1f}%) vs TD-CCEA ({cent['wr']:.1f}%)",
            f"   Difference: {d:+.1f}%",
            f"   Result: {'SA Mutation HELPS' if d >= 0 else 'SA Mutation HURTS'} ({abs(d):.1f}%)",
            "",
        ]
        for line in lines_out: print(f"  {line}")
        ablation_lines.extend(lines_out)

    # 4. Full novel contribution: CCE-TDCCEA vs SA-TDCCEA
    if cce and sa:
        d = cce["wr"] - sa["wr"]
        lines_out = [
            f"4. Full Novel Contribution (CCE-TDCCEA vs SA-TDCCEA)",
            f"   CCE-TDCCEA ({cce['wr']:.1f}%) vs SA-TDCCEA ({sa['wr']:.1f}%)",
            f"   Difference: {d:+.1f}%",
            f"   Result: {'CCE-TDCCEA BETTER' if d >= 0 else 'SA-TDCCEA BETTER'} ({abs(d):.1f}%)",
            "",
        ]
        for line in lines_out: print(f"  {line}")
        ablation_lines.extend(lines_out)

    # Add head-to-head summary to ablation
    ablation_lines.append("=" * 60)
    ablation_lines.append("Head-to-Head Tournament Results")
    ablation_lines.append("=" * 60)
    ablation_lines.append("")
    # Reproduce matrix in text form
    col_w = 14
    header_line = f"{'':17s}"
    for name in algo_names:
        header_line += f" {name:>{col_w}s}"
    header_line += f"  {'Wins':>6s}  {'WR%':>6s}"
    ablation_lines.append(header_line)
    sep_line = "─" * 17
    for _ in algo_names:
        sep_line += " " + "─" * col_w
    sep_line += "  " + "─" * 6 + "  " + "─" * 6
    ablation_lines.append(sep_line)
    for i in range(len(algo_names)):
        row = f"{algo_names[i]:17s}"
        for j in range(len(algo_names)):
            if i == j:
                row += f" {'—':>{col_w}s}"
            else:
                w, l, d = matrix[i][j]
                cell = f"{w}W-{l}L-{d}D"
                row += f" {cell:>{col_w}s}"
        row += f"  {total_wins[i]:>6d}  {win_rates[i]:>5.1f}%"
        ablation_lines.append(row)
    ablation_lines.append(sep_line)

    # Save ablation_results.txt
    ablation_path = os.path.join(SCRIPT_DIR, "ablation_results.txt")
    with open(ablation_path, "a", encoding="utf-8") as f:
        f.write("\n" + "="*70 + "\n")
        f.write("\n".join(ablation_lines) + "\n")
    print(f"\n  Saved → {ablation_path}")

    # ── SAVE JSON SUMMARY ────────────────────────────────────────
    json_data = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "map": "48×48 dynamic blockades",
            "generations": TOTAL_GENERATIONS,
            "pop_size": N_POP,
            "k_games": K_GAMES,
            "h2h_games": H2H_GAMES,
        },
        "vs_bots": [{
            "name": r["name"],
            "key": r["key"],
            "overall_winrate": r["wr"],
            "per_opponent": r["per_opp"],
            "time": r["time"],
        } for r in results],
        "head_to_head": {
            "algorithms": algo_names,
            "matrix": [
                [
                    {"wins": matrix[i][j][0],
                     "losses": matrix[i][j][1],
                     "draws": matrix[i][j][2]} if matrix[i][j] else None
                    for j in range(len(algo_names))
                ]
                for i in range(len(algo_names))
            ],
            "total_wins": total_wins,
            "win_rates": win_rates,
        },
    }
    json_path = os.path.join(SCRIPT_DIR, "pr1_results.json")
    json_list = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    json_list = data
                else:
                    json_list = [data]
        except (json.JSONDecodeError, ValueError):
            pass
    json_list.append(json_data)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_list, f, indent=2)
    print(f"\n  Saved → {json_path}")

    # All evolutionary, validation, sensitivity, and H2H work is complete.
    pool.stop()

    # ── FINAL SUMMARY ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SEM7 PR1 BENCHMARK COMPLETE!")
    print(f"{'='*70}")
    print(f"  Output files (in PR1_Sem7/):")
    print(f"    • wilcoxon_results.txt")
    print(f"    • ablation_results.txt")
    print(f"    • pr1_results.json")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
