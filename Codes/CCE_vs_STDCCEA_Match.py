"""
CCE_vs_STDCCEA_Match.py
========================
1-vs-1 Fight: CCE-TDCCEA vs Standard CCEA on a Dynamic 48×48 Map

This script pits our novel CCE-TDCCEA algorithm against the Standard CCEA
baseline in a head-to-head match. Both algorithms co-evolve against each other
over 30 generations on a 48×48 map with progressive blockades (BFS-validated).

After evolution, the best evolved weights from each algorithm play a tournament
(10 games, alternating player slots). The final blockaded 48×48 map is rendered
as a publication-quality PNG with match results annotated.

WORKFLOW:
  Phase 0 — Setup (servers, initial map)
  Phase 1 — Co-Evolution (30 generations, dynamic blockades)
  Phase 2 — Head-to-Head Tournament (10 games)
  Phase 3 — Visual Map Render + Results Display

Adapted from: SEM7_PR1_Results_fixed_v3.py
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
TEMP_MAP_PATH  = os.path.join(MICRORTS_DIR, "maps", "temp_match_48x48.xml")
TEMP_MAP_REL   = "maps/temp_match_48x48.xml"
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
TOTAL_GENERATIONS = 30
VALIDATION_GAMES  = 5
A_MAX             = 15
DELTA_S           = 0.85
DELTA_F           = 0.20
TOP_PARTNERS      = 3

# Head-to-head tournament
H2H_GAMES = 10   # per matchup pair (5 as P0, 5 as P1)

# CPU parallelism
N_WORKERS = 10

# Reserved tiles (bases, workers, resources + buffers) for 48×48 map
RESERVED = set()
for bx, by in [(0,0),(0,1),(1,1),(2,2),(1,2),(2,1),(3,2),(2,3),(3,3)]:
    RESERVED.add((bx, by))
for bx, by in [(MAP_W-1,MAP_H-1),(MAP_W-1,MAP_H-2),(MAP_W-2,MAP_H-2),(MAP_W-3,MAP_H-3),(MAP_W-2,MAP_H-3),(MAP_W-3,MAP_H-2),(MAP_W-4,MAP_H-3),(MAP_W-3,MAP_H-4),(MAP_W-4,MAP_H-4)]:
    RESERVED.add((bx, by))

# Dynamic environment — scaled for 48×48
WALLS_PER_GEN_EARLY = 6
WALLS_PER_GEN_MID   = 10
WALLS_PER_GEN_LATE  = 14
BREAKABLE_RATIO     = 0.4


# ═══════════════════════════════════════════════════════════════
#  MAP GENERATION + BFS VALIDATION
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
    base0, base1 = (2, 2), (MAP_W - 3, MAP_H - 3)
    all_walls = terrain_walls | breakable_positions
    no_place  = all_walls | RESERVED
    for _ in range(300):
        cx, cy = MAP_W // 2, MAP_H // 2
        spread = 10 + gen // 2
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
    """Write a 48x48 map XML with blockades encoded in the terrain string."""
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
# ═══════════════════════════════════════════════════════════════

class GameServer:
    def __init__(self):
        self.process = None
        self._fail_count = 0
        self._total_count = 0
        self._raw_winner_counts = {}
        self._forced_draw_count = 0

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
            err = ""
            try:
                err = self.process.stderr.read(2000)
            except Exception:
                pass
            raise RuntimeError(f"Server failed to start: stdout='{line}' stderr='{err}'")
        print(f"  [SERVER] Started successfully (READY)")

    def _check_alive(self):
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
            self._raw_winner_counts[winner] = self._raw_winner_counts.get(winner, 0) + 1
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
        """Evaluate one weight vector against rival-algorithm weight vectors."""
        server = self.pool_queue.get()
        try:
            wins = 0.0
            n = len(opponents)
            if n == 0:
                return 0.0
            for i, opp_weights in enumerate(opponents):
                if i % 2 == 0:
                    winner, _ = server.run_game_evolved_vs_evolved(weights, opp_weights)
                    if winner == 0:
                        wins += 1.0
                    elif winner == -1:
                        wins += 0.3
                else:
                    winner, _ = server.run_game_evolved_vs_evolved(opp_weights, weights)
                    if winner == 1:
                        wins += 1.0
                    elif winner == -1:
                        wins += 0.3
            return wins / n
        finally:
            self.pool_queue.put(server)

    def eval_batch_vs_agents(self, weight_list, opponents):
        """Evaluate a batch of evolved agents in parallel against rivals."""
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
        """Evaluate one weight vector against all seven built-in bots."""
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                opponent: executor.submit(self._eval_vs_bot_one, weights, opponent, k_games)
                for opponent in OPPONENT_ROTATION
            }
            return {opponent: futures[opponent].result() for opponent in OPPONENT_ROTATION}

    def validate_all_bots(self, weights):
        """Validate weights against all seven bots."""
        per_opp = self.eval_all_bots_parallel(weights, VALIDATION_GAMES)
        overall = sum(per_opp.values()) / len(per_opp)
        return overall * 100.0, {
            opponent: win_rate * 100.0 for opponent, win_rate in per_opp.items()
        }

    def _run_h2h_game_one(self, p0_weights, p1_weights, swap):
        """Run one H2H game."""
        server = self.pool_queue.get()
        try:
            if swap:
                winner, _ = server.run_game_evolved_vs_evolved(p1_weights, p0_weights)
            else:
                winner, _ = server.run_game_evolved_vs_evolved(p0_weights, p1_weights)
            return winner, swap
        finally:
            self.pool_queue.put(server)

    def run_h2h_parallel(self, p0_weights, p1_weights, n_games):
        """Run all H2H games for one algorithm pair concurrently."""
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(self._run_h2h_game_one, p0_weights, p1_weights, g % 2 == 1)
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
#  CCE-TDCCEA CLASSES
#  (Sub-population cooperative coevolution with trajectory memory)
# ═══════════════════════════════════════════════════════════════

class SubAgent:
    """Sub-agent for CCE-TDCCEA (handles either macro or micro weights)."""
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
        ws = [max(0.0, 1.0 - euclidean_dist(env[off:off+self.dim], pe[off:off+self.dim])/ md)
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
#  ALGORITHM 1 — CCE-TDCCEA (Our Novel Algorithm)
# ═══════════════════════════════════════════════════════════════

class CCEAState:
    """Holds CCE-TDCCEA's evolving state."""
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
    """Run ONE generation of CCE-TDCCEA against rival `opponents`."""
    pop_a, pop_b = state.pop_a, state.pop_b
    top_a, top_b = state.top_a, state.top_b
    ca = [sum(a.params[j] for a in pop_a) / len(pop_a) for j in range(len(MACRO_INDICES))]
    cb = [sum(b.params[j] for b in pop_b) / len(pop_b) for j in range(len(MICRO_INDICES))]
    env = ca + cb

    # Evaluate every macro agent x top micro partner
    macro_weight_list = [combine_subagents(a, pb) for a in pop_a for pb in top_b]
    macro_fitness = pool.eval_batch_vs_agents(macro_weight_list, opponents)

    n_micro_partners = len(top_b)
    for idx, a in enumerate(pop_a):
        fs = macro_fitness[idx * n_micro_partners:(idx + 1) * n_micro_partners]
        a.update_history(sum(fs) / len(fs), env)
        a.compute_trajectory(env)
        a.compute_mutation()

    # Evaluate every micro agent x top macro partner
    micro_weight_list = [combine_subagents(pa, b) for b in pop_b for pa in top_a]
    micro_fitness = pool.eval_batch_vs_agents(micro_weight_list, opponents)

    n_macro_partners = len(top_a)
    for idx, b in enumerate(pop_b):
        fs = micro_fitness[idx * n_macro_partners:(idx + 1) * n_macro_partners]
        b.update_history(sum(fs) / len(fs), env)
        b.compute_trajectory(env)
        b.compute_mutation()

    state.best_a = max(pop_a, key=lambda a: a.e_i)
    state.best_b = max(pop_b, key=lambda a: a.e_i)
    state.top_a = sorted(pop_a, key=lambda a: a.e_i, reverse=True)[:TOP_PARTNERS]
    state.top_b = sorted(pop_b, key=lambda a: a.e_i, reverse=True)[:TOP_PARTNERS]

    best_combined = combine_subagents(state.best_a, state.best_b)
    cmb = pool.eval_batch_vs_agents([best_combined], opponents)[0]
    state.convergence.append(cmb)

    print(f"  [CCE-TDCCEA] Gen {gen:2d}/{TOTAL_GENERATIONS} | "
          f"A={state.best_a.e_i:.3f} B={state.best_b.e_i:.3f} Cmb={cmb:.3f}")

    state.pop_a, state.arc_a = subpop_survival(pop_a, state.arc_a)
    state.pop_b, state.arc_b = subpop_survival(pop_b, state.arc_b)

    return combine_subagents(state.best_a, state.best_b)


# ═══════════════════════════════════════════════════════════════
#  ALGORITHM 2 — STANDARD CCEA (Baseline)
# ═══════════════════════════════════════════════════════════════

STD_MUT = 0.15


class StdCCEAState:
    """Holds Standard CCEA's evolving state."""
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
    """Run ONE generation of Standard CCEA against rival `opponents`."""
    pm, pi = state.pm, state.pi
    tmi = sorted(range(N_POP), key=lambda i: state.fi[i], reverse=True)[:TOP_PARTNERS]
    tma = sorted(range(N_POP), key=lambda i: state.fm[i], reverse=True)[:TOP_PARTNERS]

    macro_weight_list = [std_ccea_mk(pm[i], pi[j]) for i in range(N_POP) for j in tmi]
    macro_fitness = pool.eval_batch_vs_agents(macro_weight_list, opponents)
    for i in range(N_POP):
        fs = macro_fitness[i * len(tmi):(i + 1) * len(tmi)]
        state.fm[i] = sum(fs) / len(fs)

    micro_weight_list = [std_ccea_mk(pm[j], pi[i]) for i in range(N_POP) for j in tma]
    micro_fitness = pool.eval_batch_vs_agents(micro_weight_list, opponents)
    for i in range(N_POP):
        fs = micro_fitness[i * len(tma):(i + 1) * len(tma)]
        state.fi[i] = sum(fs) / len(fs)

    bm = max(range(N_POP), key=lambda i: state.fm[i])
    bi2 = max(range(N_POP), key=lambda i: state.fi[i])

    best_weights = std_ccea_mk(pm[bm], pi[bi2])
    bf = pool.eval_batch_vs_agents([best_weights], opponents)[0]

    state.convergence.append(bf)
    print(f"  [Std CCEA]   Gen {gen:2d}/{TOTAL_GENERATIONS} | Best={bf:.3f}")

    best_fm = state.fm[bm]
    best_fi = state.fi[bi2]
    state.pm = _std_ccea_breed(pm, state.fm)
    state.pi = _std_ccea_breed(pi, state.fi)
    state.fm = [0.0] * N_POP
    state.fi = [0.0] * N_POP
    state.fm[0] = best_fm
    state.fi[0] = best_fi

    return best_weights


# ═══════════════════════════════════════════════════════════════
#  MAP VISUALIZATION
#  Renders the final 48x48 map as a publication-quality PNG
# ═══════════════════════════════════════════════════════════════

def render_match_map(terrain_walls, breakable_positions, match_result, output_path):
    """
    Render the final 48x48 blockaded map with match results.

    Parameters
    ----------
    terrain_walls : set of (x, y) - permanent wall positions
    breakable_positions : set of (x, y) - breakable wall positions
    match_result : dict with keys: cce_wins, std_wins, draws, winner_name
    output_path : str - path to save the PNG
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib import rcParams
    except ImportError:
        print("  [WARN] matplotlib not installed - skipping map visualization.")
        print("  Install with: pip install matplotlib")
        return

    # -- Color Palette --
    COLORS = {
        "ground":        "#C8D8A0",
        "ground_edge":   "#B0C080",
        "wall":          "#3E3A39",
        "wall_edge":     "#2C2928",
        "breakable":     "#7B5B3A",
        "breakable_edge":"#5C3D1E",
        "p0_base":       "#1565C0",
        "p0_worker":     "#42A5F5",
        "p1_base":       "#C62828",
        "p1_worker":     "#EF5350",
        "resource":      "#FFC107",
        "resource_edge": "#F57F17",
        "resource_text": "#4E342E",
    }

    # -- Publication font settings --
    rcParams["font.family"] = "serif"
    rcParams["font.size"] = 9

    fig, ax = plt.subplots(1, 1, figsize=(12, 13), dpi=200)

    # -- Draw terrain tiles --
    for r in range(MAP_H):
        for c in range(MAP_W):
            if (c, r) in terrain_walls:
                fc, ec, lw = COLORS["wall"], COLORS["wall_edge"], 0.3
            elif (c, r) in breakable_positions:
                fc, ec, lw = COLORS["breakable"], COLORS["breakable_edge"], 0.3
            else:
                fc, ec, lw = COLORS["ground"], COLORS["ground_edge"], 0.15
            rect = mpatches.FancyBboxPatch(
                (c, r), 1, 1, boxstyle="square,pad=0",
                facecolor=fc, edgecolor=ec, linewidth=lw
            )
            ax.add_patch(rect)

    # -- Draw units (bases, workers, resources) --
    units = [
        ("Resource", 0, 0, -1, 25),
        ("Resource", 0, 1, -1, 25),
        ("Resource", MAP_W-1, MAP_H-1, -1, 25),
        ("Resource", MAP_W-1, MAP_H-2, -1, 25),
        ("Base", 2, 2, 0, 0),
        ("Base", MAP_W-3, MAP_H-3, 1, 0),
        ("Worker", 1, 1, 0, 0),
        ("Worker", MAP_W-2, MAP_H-2, 1, 0),
    ]

    for utype, ux, uy, player, res in units:
        cx, cy = ux + 0.5, uy + 0.5

        if utype == "Base":
            color = COLORS["p0_base"] if player == 0 else COLORS["p1_base"]
            rect = mpatches.FancyBboxPatch(
                (ux + 0.08, uy + 0.08), 0.84, 0.84,
                boxstyle="round,pad=0.04", facecolor=color,
                edgecolor="white", linewidth=1.2, zorder=5
            )
            ax.add_patch(rect)
            label = "B"
            plabel = "CCE-TDCCEA" if player == 0 else "Std CCEA"
            ax.text(cx, cy, label, ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white", zorder=6)
            # Player label near base
            offset_x = 2.5 if player == 0 else -2.5
            ax.text(cx + offset_x, cy + 1.5, plabel,
                    ha="center", va="center", fontsize=6,
                    fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=color, alpha=0.9),
                    zorder=7)

        elif utype == "Worker":
            color = COLORS["p0_worker"] if player == 0 else COLORS["p1_worker"]
            circle = mpatches.Circle(
                (cx, cy), 0.35, facecolor=color,
                edgecolor="white", linewidth=0.8, zorder=5
            )
            ax.add_patch(circle)
            ax.text(cx, cy, "W", ha="center", va="center",
                    fontsize=5, fontweight="bold", color="white", zorder=6)

        elif utype == "Resource":
            rect = mpatches.FancyBboxPatch(
                (ux + 0.1, uy + 0.1), 0.8, 0.8,
                boxstyle="round,pad=0.04", facecolor=COLORS["resource"],
                edgecolor=COLORS["resource_edge"], linewidth=0.8, zorder=5
            )
            ax.add_patch(rect)
            ax.text(cx, cy, f"{res}", ha="center", va="center",
                    fontsize=5, fontweight="bold",
                    color=COLORS["resource_text"], zorder=6)

    # -- Axes formatting --
    ax.set_xlim(0, MAP_W)
    ax.set_ylim(MAP_H, 0)   # flip y-axis
    ax.set_aspect("equal")
    ax.set_xticks(range(0, MAP_W + 1, 4))
    ax.set_yticks(range(0, MAP_H + 1, 4))
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.15, linewidth=0.3)

    # -- Title with match result --
    cce_w = match_result["cce_wins"]
    std_w = match_result["std_wins"]
    draws = match_result["draws"]
    winner = match_result["winner_name"]

    title_line1 = "CCE-TDCCEA  vs  Standard CCEA  -  1v1 FIGHT"
    title_line2 = f"Final 48x48 Blockaded Map  |  {len(terrain_walls)} terrain + {len(breakable_positions)} breakable walls"
    ax.set_title(f"{title_line1}\n{title_line2}", fontsize=12, fontweight="bold", pad=15)

    # -- Match result banner at bottom --
    result_text = (f"MATCH RESULT:  CCE-TDCCEA {cce_w}W  -  "
                   f"Std CCEA {std_w}W  -  {draws} Draws")
    if winner == "CCE-TDCCEA":
        winner_color = COLORS["p0_base"]
        winner_text = "WINNER: CCE-TDCCEA"
    elif winner == "Std CCEA":
        winner_color = COLORS["p1_base"]
        winner_text = "WINNER: Standard CCEA"
    else:
        winner_color = "#666666"
        winner_text = "DRAW - No Clear Winner"

    fig.text(0.5, 0.03, result_text, ha="center", va="center",
             fontsize=11, fontweight="bold", color="#333333")
    fig.text(0.5, 0.005, winner_text, ha="center", va="center",
             fontsize=14, fontweight="bold", color=winner_color)

    # -- Legend --
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["ground"], edgecolor=COLORS["ground_edge"],
                       label="Open Ground"),
        mpatches.Patch(facecolor=COLORS["wall"], edgecolor=COLORS["wall_edge"],
                       label="Terrain Wall (permanent)"),
        mpatches.Patch(facecolor=COLORS["breakable"], edgecolor=COLORS["breakable_edge"],
                       label="Breakable Wall"),
        mpatches.Patch(facecolor=COLORS["p0_base"], edgecolor="white",
                       label="CCE-TDCCEA Base (P0)"),
        mpatches.Patch(facecolor=COLORS["p1_base"], edgecolor="white",
                       label="Std CCEA Base (P1)"),
        mpatches.Patch(facecolor=COLORS["p0_worker"], edgecolor="white",
                       label="CCE-TDCCEA Worker"),
        mpatches.Patch(facecolor=COLORS["p1_worker"], edgecolor="white",
                       label="Std CCEA Worker"),
        mpatches.Patch(facecolor=COLORS["resource"], edgecolor=COLORS["resource_edge"],
                       label="Resource"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              fontsize=6, framealpha=0.9, edgecolor="#999999",
              bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"  [MAP] Saved final map image -> {output_path}")

    # Also try to display interactively
    try:
        plt.switch_backend("TkAgg")
        plt.show()
    except Exception:
        pass  # non-interactive environment


def render_convergence_plot(cce_convergence, std_convergence, output_path):
    """
    Render convergence curves for both algorithms side by side.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
    except ImportError:
        print("  [WARN] matplotlib not installed - skipping convergence plot.")
        return

    rcParams["font.family"] = "serif"
    rcParams["font.size"] = 10

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)

    gens = list(range(1, len(cce_convergence) + 1))
    ax.plot(gens, cce_convergence, '-o', label="CCE-TDCCEA (Ours)",
            linewidth=2, markersize=4, color="#1565C0")
    ax.plot(gens, std_convergence, '-s', label="Standard CCEA (Baseline)",
            linewidth=2, markersize=4, color="#C62828")

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Fitness (Win Rate vs Opponent)", fontsize=12)
    ax.set_title("Convergence: CCE-TDCCEA vs Standard CCEA\n(48x48 Dynamic Map, Co-Evolution)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(cce_convergence))
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"  [PLOT] Saved convergence plot -> {output_path}")


# ═══════════════════════════════════════════════════════════════
#  MAIN — 1v1 FIGHT
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "#" * 70)
    print("  CCE-TDCCEA  vs  STANDARD CCEA  -  1v1 FIGHT")
    print("  48x48 Dynamic Map with Progressive Blockades")
    print("#" * 70)
    print(f"  Pop={N_POP} | Gens={TOTAL_GENERATIONS} | K={K_GAMES} | "
          f"H2H={H2H_GAMES} games | Map=48x48\n")

    # ------------------------------------------------------------------
    #  PHASE 0: SETUP
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  PHASE 0: SETUP")
    print(f"{'='*60}")

    write_map_xml(set(), set(), TEMP_MAP_PATH)
    print(f"  Wrote initial clean map: {TEMP_MAP_PATH}")
    print(f"  Map file exists: {os.path.isfile(TEMP_MAP_PATH)}")

    pool = ParallelGamePool(N_WORKERS)
    pool.start()

    if not pool.run_test_game():
        print("\n  *** DIAGNOSTIC FAILED - Cannot run games on the 48x48 map ***")
        pool.stop()
        return

    # ------------------------------------------------------------------
    #  PHASE 1: CO-EVOLUTION (CCE-TDCCEA vs Standard CCEA)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  PHASE 1: CO-EVOLUTION - CCE-TDCCEA vs Std CCEA")
    print(f"  {TOTAL_GENERATIONS} generations, dynamic blockades (BFS-validated)")
    print(f"  Each algorithm evolves against the other's current best")
    print(f"{'='*60}")

    terrain_walls      = set()
    breakable_positions = set()

    ccea_state = CCEAState()
    std_state  = StdCCEAState()

    current_best = {
        "ccea":     ccea_state.current_best_weights(),
        "std_ccea": std_state.current_best_weights(),
    }

    t_evolution_start = time.time()

    print(f"\n  -- Building 30 generations of blockades --")
    for map_gen in range(1, 31):
        terrain_walls, breakable_positions, n_terrain, n_breakable = \
            add_walls_for_generation(map_gen, terrain_walls, breakable_positions)
    
    total_walls = len(terrain_walls) + len(breakable_positions)
    base0, base1 = (2, 2), (MAP_W - 3, MAP_H - 3)
    reachable = bfs_reachable(terrain_walls | breakable_positions, base0, base1)
    status = "REACHABLE" if reachable else "BLOCKED"
    
    write_map_xml(terrain_walls, breakable_positions, TEMP_MAP_PATH)
    print(f"  |  Map: {len(terrain_walls)} terrain + {len(breakable_positions)} breakable -> "
          f"{total_walls} total walls | BFS: {status}")

    print(f"\n  -- Running 1 generation of evolution --")
    
    # Force evolution to 1 generation
    global TOTAL_GENERATIONS
    TOTAL_GENERATIONS = 1

    for gen in range(1, TOTAL_GENERATIONS + 1):
        # Each algorithm fights against the OTHER's current best
        ccea_opponents = [current_best["std_ccea"]]
        std_opponents  = [current_best["ccea"]]

        new_ccea_best = step_ccea(pool, ccea_state, ccea_opponents, gen)
        new_std_best  = step_std_ccea(pool, std_state, std_opponents, gen)

        # Simultaneous update
        current_best = {
            "ccea":     new_ccea_best,
            "std_ccea": new_std_best,
        }
        print(f"  {'--'*28}")

    elapsed = time.time() - t_evolution_start
    final_total = len(terrain_walls) + len(breakable_positions)
    print(f"\n  Evolution complete in {elapsed:.1f}s")
    print(f"  Final map: {final_total} walls "
          f"({len(terrain_walls)} terrain + {len(breakable_positions)} breakable)")

    # ------------------------------------------------------------------
    #  VALIDATION: Test both algorithms against built-in bots
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  VALIDATION - Both Algorithms vs 7 Built-in Bots")
    print(f"{'='*60}")

    ccea_best_weights = current_best["ccea"]
    std_best_weights  = current_best["std_ccea"]

    print(f"\n  Validating CCE-TDCCEA...")
    cce_wr, cce_per_opp = pool.validate_all_bots(ccea_best_weights)
    print(f"  CCE-TDCCEA:  {cce_wr:.1f}% overall win rate")

    print(f"\n  Validating Standard CCEA...")
    std_wr, std_per_opp = pool.validate_all_bots(std_best_weights)
    print(f"  Std CCEA:    {std_wr:.1f}% overall win rate")

    # Per-bot table
    print(f"\n  {'Bot':<26s} {'CCE-TDCCEA':>12s} {'Std CCEA':>12s}  {'Better':>12s}")
    print(f"  {'-'*66}")
    for opp in OPPONENT_ROTATION:
        cce_val = cce_per_opp.get(opp, 0)
        std_val = std_per_opp.get(opp, 0)
        better = "CCE-TDCCEA" if cce_val > std_val else ("Std CCEA" if std_val > cce_val else "Tied")
        marker = "<" if better == "CCE-TDCCEA" else (">" if better == "Std CCEA" else "=")
        print(f"  {opp:<26s} {cce_val:>10.0f}%  {std_val:>10.0f}%   {marker} {better}")
    print(f"  {'-'*66}")
    print(f"  {'OVERALL':<26s} {cce_wr:>10.1f}%  {std_wr:>10.1f}%")

    # ------------------------------------------------------------------
    #  PHASE 2: HEAD-TO-HEAD TOURNAMENT
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  PHASE 2: HEAD-TO-HEAD TOURNAMENT - THE FIGHT!")
    print(f"  CCE-TDCCEA vs Standard CCEA")
    print(f"  {H2H_GAMES} games (alternating P0/P1 slots)")
    print(f"  Map: 48x48 with {final_total} blockades")
    print(f"{'='*60}\n")

    # Write final map for tournament
    write_map_xml(terrain_walls, breakable_positions, TEMP_MAP_PATH)

    # Run tournament
    game_results = pool.run_h2h_parallel(ccea_best_weights, std_best_weights, H2H_GAMES)

    cce_wins, std_wins, draws = 0, 0, 0
    print(f"  {'Game':>6s}  {'Slot':>10s}  {'Winner':>15s}")
    print(f"  {'-'*38}")

    for g_idx, (winner, swap) in enumerate(game_results):
        slot = "SWAPPED" if swap else "NORMAL"
        if not swap:
            if winner == 0:
                cce_wins += 1
                who = "CCE-TDCCEA"
            elif winner == 1:
                std_wins += 1
                who = "Std CCEA"
            else:
                draws += 1
                who = "DRAW"
        else:
            if winner == 0:
                std_wins += 1
                who = "Std CCEA"
            elif winner == 1:
                cce_wins += 1
                who = "CCE-TDCCEA"
            else:
                draws += 1
                who = "DRAW"
        print(f"  {g_idx+1:>6d}  {slot:>10s}  {who:>15s}")

    print(f"\n  {'='*50}")
    print(f"  |  FINAL SCORE:                                  |")
    print(f"  |    CCE-TDCCEA :  {cce_wins:2d} wins                       |")
    print(f"  |    Std CCEA   :  {std_wins:2d} wins                       |")
    print(f"  |    Draws      :  {draws:2d}                              |")

    if cce_wins > std_wins:
        winner_name = "CCE-TDCCEA"
        print(f"  |                                                |")
        print(f"  |  WINNER: CCE-TDCCEA (Our Algorithm!)            |")
    elif std_wins > cce_wins:
        winner_name = "Std CCEA"
        print(f"  |                                                |")
        print(f"  |  WINNER: Standard CCEA (Baseline)               |")
    else:
        winner_name = "DRAW"
        print(f"  |                                                |")
        print(f"  |  DRAW - No Clear Winner                         |")
    print(f"  {'='*50}")

    # Print evolved weights
    print(f"\n  CCE-TDCCEA Best Weights:")
    for i, (name, val) in enumerate(zip(WEIGHT_NAMES, ccea_best_weights)):
        print(f"    {name:<22s} = {val:.4f}")

    print(f"\n  Std CCEA Best Weights:")
    for i, (name, val) in enumerate(zip(WEIGHT_NAMES, std_best_weights)):
        print(f"    {name:<22s} = {val:.4f}")

    # ------------------------------------------------------------------
    #  PHASE 3: VISUAL MAP RENDER + CONVERGENCE PLOT
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  PHASE 3: RENDERING FINAL MAP + CONVERGENCE")
    print(f"{'='*60}")

    match_result = {
        "cce_wins":    cce_wins,
        "std_wins":    std_wins,
        "draws":       draws,
        "winner_name": winner_name,
    }

    map_output = os.path.join(SCRIPT_DIR, "match_map_48x48.png")
    render_match_map(terrain_walls, breakable_positions, match_result, map_output)

    conv_output = os.path.join(SCRIPT_DIR, "match_convergence.png")
    render_convergence_plot(
        ccea_state.convergence,
        std_state.convergence,
        conv_output,
    )

    # ------------------------------------------------------------------
    #  SAVE JSON RESULTS
    # ------------------------------------------------------------------
    json_data = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "match": "CCE-TDCCEA vs Standard CCEA",
            "map": "48x48 dynamic blockades",
            "generations": TOTAL_GENERATIONS,
            "pop_size": N_POP,
            "k_games": K_GAMES,
            "h2h_games": H2H_GAMES,
            "evolution_time_seconds": elapsed,
        },
        "validation_vs_bots": {
            "CCE-TDCCEA": {"overall_wr": cce_wr, "per_opponent": cce_per_opp},
            "Std_CCEA":   {"overall_wr": std_wr, "per_opponent": std_per_opp},
        },
        "head_to_head": {
            "cce_wins": cce_wins,
            "std_wins": std_wins,
            "draws": draws,
            "winner": winner_name,
        },
        "convergence": {
            "cce_tdccea": ccea_state.convergence,
            "std_ccea":   std_state.convergence,
        },
        "best_weights": {
            "cce_tdccea": ccea_best_weights,
            "std_ccea":   std_best_weights,
        },
    }

    json_path = os.path.join(SCRIPT_DIR, "match_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"\n  Saved results -> {json_path}")

    # Stop all servers
    pool.stop()

    # -- FINAL SUMMARY --
    print(f"\n{'#'*70}")
    print(f"  MATCH COMPLETE!")
    print(f"{'#'*70}")
    print(f"  Output files:")
    print(f"    - match_map_48x48.png     - Final blockaded map with result")
    print(f"    - match_convergence.png   - Convergence curves")
    print(f"    - match_results.json      - Full results data")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
