# cce_vs_stdccea_match.py
# ======================
# CCE-TDCCEA vs Standard CCEA on a 48x48 dynamic map.
# Based on SEM7_PR1_Results.py (2-algorithm version).
# Phases:
#   1. Co-evolution  (TOTAL_GENERATIONS gens, progressive BFS-validated blockades)
#   2. H2H tournament (H2H_GAMES games, alternating P0/P1 slots)
#   3. Final 48x48 map displayed in BLACK-AND-WHITE:
#        solid black = unbreakable (terrain) wall
#        light grey  = breakable wall
#        white       = open cell
#      + coloured accents: blue=P0 base, red=P1 base, gold=resources

import math, random, time, subprocess, os, collections, queue, shutil
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

os.environ['PYTHONIOENCODING'] = 'utf-8'

# ── Locate ffmpeg for matplotlib's animation writer ──────────────
# WinGet installs to a versioned path; fall back to whatever is on PATH.
_FFMPEG_WINGET = (
    r'C:\Users\rgdar\AppData\Local\Microsoft\WinGet\Packages'
    r'\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe'
    r'\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe'
)
_ffmpeg_exe = (_FFMPEG_WINGET if os.path.isfile(_FFMPEG_WINGET)
               else shutil.which('ffmpeg') or 'ffmpeg')
matplotlib.rcParams['animation.ffmpeg_path'] = _ffmpeg_exe


# ================================================================
#  CONFIGURATION
# ================================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CODES_DIR    = os.path.dirname(SCRIPT_DIR)
MICRORTS_DIR = os.path.join(CODES_DIR, 'Simulation', 'microrts')
MAP_W, MAP_H  = 48, 48
TEMP_MAP_PATH = os.path.join(MICRORTS_DIR, 'maps', 'temp_cce_vs_std_48x48.xml')
TEMP_MAP_REL  = 'maps/temp_cce_vs_std_48x48.xml'
MAX_CYCLES    = 12000

WEIGHT_NAMES = [
    'worker_ratio','max_workers','expand_threshold','light_preference',
    'heavy_preference','ranged_preference','barracks_timing','aggression',
    'attack_threshold','defend_radius','target_priority','defense_trigger',
    'counter_rush','tech_switch','economy_pressure','scout_priority',
    'group_attack','second_barracks','worker_fight_ratio','focus_fire',
]
DIMENSIONS    = len(WEIGHT_NAMES)
MACRO_INDICES = [0,1,2,6,14,15,17,18]
MICRO_INDICES = [3,4,5,7,8,9,10,11,12,13,16,19]

N_POP             = 20
TOTAL_GENERATIONS = 30
A_MAX             = 15
DELTA_S           = 0.85
DELTA_F           = 0.20
TOP_PARTNERS      = 3
H2H_GAMES         = 10
N_WORKERS         = 10
STD_MUT           = 0.15
WALLS_PER_GEN_EARLY = 6
WALLS_PER_GEN_MID   = 10
WALLS_PER_GEN_LATE  = 14
BREAKABLE_RATIO     = 0.4

RESERVED = set()
for _bx, _by in [(0,0),(0,1),(1,1),(2,2),(1,2),(2,1),(3,2),(2,3),(3,3)]:
    RESERVED.add((_bx, _by))
for _bx, _by in [(MAP_W-1,MAP_H-1),(MAP_W-1,MAP_H-2),(MAP_W-2,MAP_H-2),
                 (MAP_W-3,MAP_H-3),(MAP_W-2,MAP_H-3),(MAP_W-3,MAP_H-2),
                 (MAP_W-4,MAP_H-3),(MAP_W-3,MAP_H-4),(MAP_W-4,MAP_H-4)]:
    RESERVED.add((_bx, _by))

BASE_P0   = (2, 2)
BASE_P1   = (MAP_W - 3, MAP_H - 3)
WORKER_P0 = (1, 1)
WORKER_P1 = (MAP_W - 2, MAP_H - 2)
RES_P0    = [(0, 0), (0, 1)]
RES_P1    = [(MAP_W - 1, MAP_H - 1), (MAP_W - 1, MAP_H - 2)]

# Black-and-white colour palette + coloured accents
CLR_OPEN      = (1.00, 1.00, 1.00, 1.0)   # white
CLR_TERRAIN   = (0.05, 0.05, 0.05, 1.0)   # near-black
CLR_BREAKABLE = (0.72, 0.72, 0.72, 1.0)   # light grey
CLR_RESOURCE  = (0.95, 0.80, 0.10, 1.0)   # gold
CLR_BASE_P0   = (0.18, 0.45, 0.90, 1.0)   # blue
CLR_BASE_P1   = (0.88, 0.22, 0.18, 1.0)   # red
CLR_WORKER_P0 = (0.50, 0.75, 1.00, 1.0)   # light-blue
CLR_WORKER_P1 = (1.00, 0.55, 0.50, 1.0)   # light-red

# ================================================================
#  BFS + WALL PLACEMENT  (identical to SEM7_PR1_Results.py)
# ================================================================
def bfs_reachable(walls, start, end):
    if start in walls or end in walls:
        return False
    visited, q = {start}, collections.deque([start])
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

def bfs_path(walls, start, end):
    if start in walls or end in walls:
        return []
    visited, q = {start: None}, collections.deque([start])
    while q:
        x, y = q.popleft()
        if (x, y) == end:
            path, node = [], end
            while node is not None:
                path.append(node)
                node = visited[node]
            return path[::-1]
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < MAP_W and 0 <= ny < MAP_H:
                if (nx, ny) not in visited and (nx, ny) not in walls:
                    visited[(nx, ny)] = (x, y)
                    q.append((nx, ny))
    return []

def pick_wall_position(terrain_walls, breakable_positions, gen):
    all_walls = terrain_walls | breakable_positions
    no_place  = all_walls | RESERVED
    for _ in range(300):
        cx, cy = MAP_W // 2, MAP_H // 2
        spread = 10 + gen // 2
        x = max(0, min(MAP_W-1, int(random.gauss(cx, spread))))
        y = max(0, min(MAP_H-1, int(random.gauss(cy, spread))))
        if (x, y) in no_place:
            continue
        if bfs_reachable(all_walls | {(x, y)}, BASE_P0, BASE_P1):
            return (x, y)
    return None

def add_walls_for_generation(gen, terrain_walls, breakable_positions):
    n_new = (WALLS_PER_GEN_EARLY if gen <= 7
             else (WALLS_PER_GEN_MID if gen <= 14 else WALLS_PER_GEN_LATE))
    new_t, new_b = set(), set()
    for _ in range(n_new):
        pos = pick_wall_position(terrain_walls | new_t, breakable_positions | new_b, gen)
        if pos is None:
            break
        if random.random() < BREAKABLE_RATIO:
            new_b.add(pos)
        else:
            new_t.add(pos)
    return terrain_walls | new_t, breakable_positions | new_b, len(new_t), len(new_b)

def write_map_xml(terrain_walls, breakable_positions, path):
    terrain = ['0'] * (MAP_W * MAP_H)
    for wx, wy in terrain_walls | breakable_positions:
        terrain[wy * MAP_W + wx] = '1'
    terrain_str = ''.join(terrain)
    units_lines = [
        '    <rts.units.Unit type="Resource" ID="1" player="-1" x="0" y="0" resources="25" hitpoints="1"></rts.units.Unit>',
        '    <rts.units.Unit type="Resource" ID="2" player="-1" x="0" y="1" resources="25" hitpoints="1"></rts.units.Unit>',
        f'    <rts.units.Unit type="Resource" ID="5" player="-1" x="{MAP_W-1}" y="{MAP_H-1}" resources="25" hitpoints="1"></rts.units.Unit>',
        f'    <rts.units.Unit type="Resource" ID="6" player="-1" x="{MAP_W-1}" y="{MAP_H-2}" resources="25" hitpoints="1"></rts.units.Unit>',
        '    <rts.units.Unit type="Base" ID="3" player="0" x="2" y="2" resources="0" hitpoints="10"></rts.units.Unit>',
        f'    <rts.units.Unit type="Base" ID="7" player="1" x="{MAP_W-3}" y="{MAP_H-3}" resources="0" hitpoints="10"></rts.units.Unit>',
        '    <rts.units.Unit type="Worker" ID="4" player="0" x="1" y="1" resources="0" hitpoints="1"></rts.units.Unit>',
        f'    <rts.units.Unit type="Worker" ID="8" player="1" x="{MAP_W-2}" y="{MAP_H-2}" resources="0" hitpoints="1"></rts.units.Unit>',
    ]
    xml = (
        f'<rts.PhysicalGameState width="{MAP_W}" height="{MAP_H}">' + chr(10) +
        f'  <terrain>{terrain_str}</terrain>' + chr(10) +
        '  <players>' + chr(10) +
        '    <rts.Player ID="0" resources="5"></rts.Player>' + chr(10) +
        '    <rts.Player ID="1" resources="5"></rts.Player>' + chr(10) +
        '  </players>' + chr(10) +
        '  <units>' + chr(10) +
        chr(10).join(units_lines) + chr(10) +
        '  </units>' + chr(10) +
        '</rts.PhysicalGameState>'
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(xml)

# ================================================================
#  GAME SERVER
# ================================================================
class GameServer:
    def __init__(self):
        self.process = None
        self._fail_count = 0
        self._total_count = 0
        self._forced_draw_count = 0
        self._raw_winner_counts = {}

    def start(self):
        print(f'  [SERVER] cwd = {MICRORTS_DIR}')
        self.process = subprocess.Popen(
            ['java', '-cp', 'lib/*;bin', 'tests.EvolvedBatchGameRunner'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=MICRORTS_DIR)
        line = self.process.stdout.readline().strip()
        if line != 'READY':
            raise RuntimeError(f"Server failed to start: got '{line}'")
        print('  [SERVER] Started (READY)')

    def _check_alive(self):
        if self.process and self.process.poll() is not None:
            print(f'  [SERVER ERROR] Java process DIED (exit={self.process.returncode})')
            return False
        return True

    def run_game_evolved_vs_evolved(self, p0_w, p1_w, map_path=None):
        if map_path is None:
            map_path = TEMP_MAP_REL
        self._total_count += 1
        if not self._check_alive():
            self._fail_count += 1
            return -1, MAX_CYCLES
        w0  = ';'.join(f'{w:.4f}' for w in p0_w)
        w1  = ';'.join(f'{w:.4f}' for w in p1_w)
        req = f'EVOLVED:{w0},EVOLVED:{w1},{map_path},{MAX_CYCLES}' + chr(10)
        try:
            self.process.stdin.write(req)
            self.process.stdin.flush()
            resp = self.process.stdout.readline().strip()
        except (BrokenPipeError, OSError) as e:
            self._fail_count += 1
            print(f'  [SERVER ERROR] {e}')
            return -1, MAX_CYCLES
        if resp.startswith('RESULT:'):
            parts  = resp.replace('RESULT:', '').split(',')
            winner, cycles = int(parts[0]), int(parts[1])
            self._raw_winner_counts[winner] = self._raw_winner_counts.get(winner, 0) + 1
            if cycles >= MAX_CYCLES and winner != -1:
                self._forced_draw_count += 1
                winner = -1
            return winner, cycles
        self._fail_count += 1
        return -1, MAX_CYCLES

    def stop(self):
        if self._fail_count > 0:
            print(f'  [SERVER] Failures: {self._fail_count}/{self._total_count}')
        if self._forced_draw_count > 0:
            print(f'  [SERVER] Timeouts forced to draw: {self._forced_draw_count}')
        if self.process:
            try:
                self.process.stdin.write('EXIT' + chr(10))
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()

# ================================================================
#  PARALLEL GAME SERVER POOL
# ================================================================
class ParallelGamePool:
    def __init__(self, n_workers):
        self.n_workers  = n_workers
        self.pool_queue = queue.Queue()

    def start(self):
        print(f'  [CPU] Starting {self.n_workers} parallel servers...')
        for _ in range(self.n_workers):
            s = GameServer()
            s.start()
            self.pool_queue.put(s)
        print(f'  [OK] {self.n_workers} servers ready')

    def _eval_one(self, weights, opponents):
        server = self.pool_queue.get()
        try:
            wins, n = 0.0, len(opponents)
            if n == 0:
                return 0.0
            for i, opp_w in enumerate(opponents):
                if i % 2 == 0:
                    winner, _ = server.run_game_evolved_vs_evolved(weights, opp_w)
                    if winner == 0:
                        wins += 1.0
                    elif winner == -1:
                        wins += 0.3
                else:
                    winner, _ = server.run_game_evolved_vs_evolved(opp_w, weights)
                    if winner == 1:
                        wins += 1.0
                    elif winner == -1:
                        wins += 0.3
            return wins / n
        finally:
            self.pool_queue.put(server)

    def eval_batch_vs_agents(self, weight_list, opponents):
        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            futs = [ex.submit(self._eval_one, w, opponents) for w in weight_list]
            return [f.result() for f in futs]

    def _h2h_one(self, p0_w, p1_w, swap):
        server = self.pool_queue.get()
        try:
            if swap:
                winner, _ = server.run_game_evolved_vs_evolved(p1_w, p0_w)
            else:
                winner, _ = server.run_game_evolved_vs_evolved(p0_w, p1_w)
            return winner, swap
        finally:
            self.pool_queue.put(server)

    def run_h2h_parallel(self, p0_w, p1_w, n_games):
        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            futs = [ex.submit(self._h2h_one, p0_w, p1_w, g % 2 == 1) for g in range(n_games)]
            return [f.result() for f in futs]

    def stop(self):
        while not self.pool_queue.empty():
            self.pool_queue.get().stop()
        print('  [OK] All servers stopped')

# ================================================================
#  HELPERS
# ================================================================
def euclidean_dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# ================================================================
#  CCE-TDCCEA  (Algorithm 1, from SEM7_PR1_Results.py)
# ================================================================
class SubAgent:
    def __init__(self, dim, params=None, env_offset=0):
        self.dim, self.env_offset = dim, env_offset
        self.params = ([random.random() for _ in range(dim)]
                       if params is None
                       else [max(0.0, min(1.0, float(x))) for x in params])
        self.H_i, self.e_i, self.mu_i = [], 0.0, 0.0

    def update_history(self, f, env):
        self.H_i.append((f, env))
        if len(self.H_i) > 7:
            self.H_i.pop(0)

    def compute_trajectory(self, env):
        if not self.H_i:
            self.e_i = 0.0
            return
        md, off = math.sqrt(self.dim), self.env_offset
        ws = [max(0.0, 1.0 - euclidean_dist(env[off:off+self.dim], pe[off:off+self.dim]) / md)
              if len(pe) >= off + self.dim else 1.0 for _, pe in self.H_i]
        t  = sum(ws)
        ws = [w/t for w in ws] if t > 0 else [1.0/len(ws)] * len(ws)
        self.e_i = sum(w * self.H_i[i][0] for i, w in enumerate(ws))

    def compute_mutation(self):
        self.mu_i = 0.15 * (1.0 - self.e_i) + 0.03

    def mutate(self):
        return SubAgent(
            self.dim,
            [max(0.0, min(1.0, p + random.gauss(0, self.mu_i))) for p in self.params],
            env_offset=self.env_offset)

def combine_subagents(macro, micro):
    full = [0.5] * DIMENSIONS
    for i, idx in enumerate(MACRO_INDICES): full[idx] = macro.params[i]
    for i, idx in enumerate(MICRO_INDICES): full[idx] = micro.params[i]
    return full

def subpop_survival(pop, archive):
    dim = pop[0].dim
    md  = math.sqrt(dim)
    survivors = []
    for a in sorted(pop, key=lambda a: a.e_i, reverse=True):
        if not any(1.0 - euclidean_dist(a.params, u.params)/md > DELTA_S for u in survivors):
            survivors.append(a)
    final = [a for a in survivors if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
    while len(final) < 8 and len(survivors) > len(final):
        for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
            if a not in final:
                final.append(a)
                break
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

class CCEAState:
    def __init__(self):
        self.pop_a   = [SubAgent(len(MACRO_INDICES), env_offset=0) for _ in range(N_POP)]
        self.pop_b   = [SubAgent(len(MICRO_INDICES),  env_offset=len(MACRO_INDICES)) for _ in range(N_POP)]
        self.arc_a, self.arc_b = [], []
        self.top_a   = self.pop_a[:TOP_PARTNERS]
        self.top_b   = self.pop_b[:TOP_PARTNERS]
        self.best_a  = self.pop_a[0]
        self.best_b  = self.pop_b[0]
        self.convergence = []

    def current_best_weights(self):
        return combine_subagents(self.best_a, self.best_b)

def step_ccea(gpool, state, opponents, gen):
    pop_a, pop_b = state.pop_a, state.pop_b
    top_a, top_b = state.top_a, state.top_b
    ca  = [sum(a.params[j] for a in pop_a)/len(pop_a) for j in range(len(MACRO_INDICES))]
    cb  = [sum(b.params[j] for b in pop_b)/len(pop_b) for j in range(len(MICRO_INDICES))]
    env = ca + cb
    mwl = [combine_subagents(a, pb) for a in pop_a for pb in top_b]
    mf  = gpool.eval_batch_vs_agents(mwl, opponents)
    nm  = len(top_b)
    for idx, a in enumerate(pop_a):
        fs = mf[idx*nm:(idx+1)*nm]
        a.update_history(sum(fs)/len(fs), env)
        a.compute_trajectory(env)
        a.compute_mutation()
    iwl = [combine_subagents(pa, b) for b in pop_b for pa in top_a]
    if_ = gpool.eval_batch_vs_agents(iwl, opponents)
    na  = len(top_a)
    for idx, b in enumerate(pop_b):
        fs = if_[idx*na:(idx+1)*na]
        b.update_history(sum(fs)/len(fs), env)
        b.compute_trajectory(env)
        b.compute_mutation()
    state.best_a = max(pop_a, key=lambda a: a.e_i)
    state.best_b = max(pop_b, key=lambda a: a.e_i)
    state.top_a  = sorted(pop_a, key=lambda a: a.e_i, reverse=True)[:TOP_PARTNERS]
    state.top_b  = sorted(pop_b, key=lambda a: a.e_i, reverse=True)[:TOP_PARTNERS]
    bc  = combine_subagents(state.best_a, state.best_b)
    cmb = gpool.eval_batch_vs_agents([bc], opponents)[0]
    state.convergence.append(cmb)
    print(f'  [CCE-TDCCEA] Gen {gen:2d}/{TOTAL_GENERATIONS} | A={state.best_a.e_i:.3f} B={state.best_b.e_i:.3f} Cmb={cmb:.3f}')
    state.pop_a, state.arc_a = subpop_survival(pop_a, state.arc_a)
    state.pop_b, state.arc_b = subpop_survival(pop_b, state.arc_b)
    return combine_subagents(state.best_a, state.best_b)

# ================================================================
#  STANDARD CCEA  (Algorithm 2, from SEM7_PR1_Results.py)
# ================================================================
class StdCCEAState:
    def __init__(self):
        self.pm = [[random.random() for _ in range(len(MACRO_INDICES))] for _ in range(N_POP)]
        self.pi = [[random.random() for _ in range(len(MICRO_INDICES))]  for _ in range(N_POP)]
        self.fm, self.fi = [0.0]*N_POP, [0.0]*N_POP
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

def _std_breed(pop, fits):
    bi = max(range(len(pop)), key=lambda i: fits[i])
    new_pop = [pop[bi][:]]
    while len(new_pop) < N_POP:
        t    = random.sample(range(len(pop)), min(3, len(pop)))
        pidx = max(t, key=lambda i: fits[i])
        new_pop.append([max(0.0, min(1.0, x + random.gauss(0, STD_MUT))) for x in pop[pidx]])
    return new_pop

def step_std_ccea(gpool, state, opponents, gen):
    pm, pi = state.pm, state.pi
    tmi = sorted(range(N_POP), key=lambda i: state.fi[i], reverse=True)[:TOP_PARTNERS]
    tma = sorted(range(N_POP), key=lambda i: state.fm[i], reverse=True)[:TOP_PARTNERS]
    mwl = [std_ccea_mk(pm[i], pi[j]) for i in range(N_POP) for j in tmi]
    mf  = gpool.eval_batch_vs_agents(mwl, opponents)
    for i in range(N_POP):
        fs = mf[i*len(tmi):(i+1)*len(tmi)]
        state.fm[i] = sum(fs)/len(fs)
    iwl = [std_ccea_mk(pm[j], pi[i]) for i in range(N_POP) for j in tma]
    if_ = gpool.eval_batch_vs_agents(iwl, opponents)
    for i in range(N_POP):
        fs = if_[i*len(tma):(i+1)*len(tma)]
        state.fi[i] = sum(fs)/len(fs)
    bm  = max(range(N_POP), key=lambda i: state.fm[i])
    bi2 = max(range(N_POP), key=lambda i: state.fi[i])
    bw  = std_ccea_mk(pm[bm], pi[bi2])
    bf  = gpool.eval_batch_vs_agents([bw], opponents)[0]
    state.convergence.append(bf)
    print(f'  [Std CCEA  ] Gen {gen:2d}/{TOTAL_GENERATIONS} | Best={bf:.3f}')
    bfm, bfi = state.fm[bm], state.fi[bi2]
    state.pm = _std_breed(pm, state.fm)
    state.pi = _std_breed(pi, state.fi)
    state.fm = [0.0]*N_POP
    state.fi = [0.0]*N_POP
    state.fm[0] = bfm
    state.fi[0] = bfi
    return bw

# ================================================================
#  HEAD-TO-HEAD
# ================================================================
def run_h2h_2(gpool, ccea_w, std_w):
    ccea_wins = std_wins = draws = 0
    for winner, swap in gpool.run_h2h_parallel(ccea_w, std_w, H2H_GAMES):
        if not swap:
            if winner == 0:    ccea_wins += 1
            elif winner == 1:  std_wins  += 1
            else:              draws     += 1
        else:
            if winner == 0:    std_wins  += 1
            elif winner == 1:  ccea_wins += 1
            else:              draws     += 1
    return ccea_wins, std_wins, draws

# ================================================================
#  MAP VISUALISATION  (black-and-white theme)
# ================================================================
def build_bw_map(terrain_walls, breakable_positions):
    img = np.tile(CLR_OPEN, (MAP_H, MAP_W, 1)).astype(float)
    for wx, wy in terrain_walls:
        img[wy, wx] = CLR_TERRAIN
    for wx, wy in breakable_positions:
        img[wy, wx] = CLR_BREAKABLE
    for rx, ry in RES_P0 + RES_P1:
        img[ry, rx] = CLR_RESOURCE
    img[BASE_P0[1], BASE_P0[0]]     = CLR_BASE_P0
    img[BASE_P1[1], BASE_P1[0]]     = CLR_BASE_P1
    img[WORKER_P0[1], WORKER_P0[0]] = CLR_WORKER_P0
    img[WORKER_P1[1], WORKER_P1[0]] = CLR_WORKER_P1
    return img

# How many seconds the static frame is held in the exported video
VIDEO_DURATION_SEC = 8
VIDEO_FPS          = 24


def _build_figure(terrain_walls, breakable_positions, ccea_wins, std_wins, draws):
    """Build and return (fig, ax) for the final B+W map."""
    n_t   = len(terrain_walls)
    n_b   = len(breakable_positions)
    total = n_t + n_b
    aw    = terrain_walls | breakable_positions
    bfs_ok  = bfs_reachable(aw, BASE_P0, BASE_P1)
    path    = bfs_path(aw, BASE_P0, BASE_P1)
    bfs_str = 'REACHABLE' if bfs_ok else 'BLOCKED'
    total_g = ccea_wins + std_wins + draws

    if ccea_wins > std_wins:   verdict = 'CCE-TDCCEA WINS'
    elif std_wins > ccea_wins: verdict = 'Standard CCEA WINS'
    else:                      verdict = 'DRAW'

    fig, ax = plt.subplots(1, 1, figsize=(11, 11), facecolor='white')
    ax.set_facecolor('white')
    ax.imshow(build_bw_map(terrain_walls, breakable_positions),
              interpolation='nearest', origin='upper')

    # grid lines -- thin mid-grey, matching reference image style
    ax.set_xticks(np.arange(-0.5, MAP_W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, MAP_H, 1), minor=True)
    ax.grid(which='minor', color='#888888', linestyle='-', linewidth=0.55, alpha=0.7)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    title = (
        f'{total} walls ({n_t} unbreakable \u25a0 + {n_b} breakable \u25a1)'
        f'  |  BFS: {bfs_str}  |  Path: {len(path)}' + chr(10) +
        f'CCE-TDCCEA  {ccea_wins}W - {draws}D - {std_wins}L  Standard CCEA'
        f'  ({total_g} games)  --  {verdict}'
    )
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12,
                 color='black', linespacing=1.6)

    for sp in ax.spines.values():
        sp.set_linewidth(1.0)
        sp.set_color('#444444')

    legend_items = [
        mpatches.Patch(facecolor=CLR_TERRAIN[:3],   edgecolor='#333333',
                       label='Unbreakable wall (terrain)  \u25a0'),
        mpatches.Patch(facecolor=CLR_BREAKABLE[:3], edgecolor='#888888',
                       label='Breakable wall  \u25a1'),
        mpatches.Patch(facecolor=CLR_OPEN[:3],      edgecolor='#888888',
                       label='Open cell'),
        mpatches.Patch(facecolor=CLR_BASE_P0[:3],   edgecolor='none',
                       label='P0 base  (CCE-TDCCEA)'),
        mpatches.Patch(facecolor=CLR_BASE_P1[:3],   edgecolor='none',
                       label='P1 base  (Std CCEA)'),
        mpatches.Patch(facecolor=CLR_RESOURCE[:3],  edgecolor='none',
                       label='Resource'),
    ]
    leg = ax.legend(handles=legend_items, loc='lower left', fontsize=9.5,
                    frameon=True, fancybox=False, edgecolor='#555555')
    leg.get_frame().set_facecolor('#F5F5F5')
    for t in leg.get_texts():
        t.set_color('black')

    plt.tight_layout()
    return fig, ax


def save_map_mp4(terrain_walls, breakable_positions, ccea_wins, std_wins, draws,
                 mp4_path, duration_sec=VIDEO_DURATION_SEC, fps=VIDEO_FPS):
    """
    Save the final map as a static-frame .mp4 video.
    Uses matplotlib FFMpegWriter (requires ffmpeg on PATH).
    Falls back to PillowWriter (.gif) if ffmpeg is not available.
    """
    from matplotlib import animation

    fig, _ = _build_figure(terrain_walls, breakable_positions, ccea_wins, std_wins, draws)
    total_frames = max(1, int(duration_sec * fps))

    # Dummy animation: every frame is identical (static image held for duration_sec)
    def _noop(frame_idx):
        pass  # figure already drawn; nothing to update

    anim = animation.FuncAnimation(fig, _noop, frames=total_frames,
                                   interval=1000 / fps, blit=False)

    # Try FFMpegWriter first (produces .mp4)
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000,
                                        codec='libx264',
                                        extra_args=['-pix_fmt', 'yuv420p'])
        anim.save(mp4_path, writer=writer, dpi=120)
        print(f'  [MP4] Saved -> {mp4_path}')
    except Exception as e_ffmpeg:
        # Fall back to PillowWriter -> .gif
        gif_path = mp4_path.replace('.mp4', '.gif')
        print(f'  [MP4] FFMpeg unavailable ({e_ffmpeg}); falling back to GIF.')
        try:
            writer_gif = animation.PillowWriter(fps=fps)
            anim.save(gif_path, writer=writer_gif, dpi=100)
            print(f'  [GIF] Saved -> {gif_path}')
        except Exception as e_gif:
            print(f'  [ERROR] Could not save animation: {e_gif}')
    finally:
        plt.close(fig)


def show_final_map(terrain_walls, breakable_positions, ccea_wins, std_wins, draws, gen,
                   save_mp4=True, mp4_path=None):
    """
    Display the final 48x48 map in B+W with match result in the title.
    Optionally saves a VIDEO_DURATION_SEC static-frame video to `mp4_path`.
    """
    if save_mp4:
        if mp4_path is None:
            mp4_path = os.path.join(SCRIPT_DIR, 'match_final_map.mp4')
        print(f'  [MP4] Rendering video ({VIDEO_DURATION_SEC}s @ {VIDEO_FPS}fps) ...')
        save_map_mp4(terrain_walls, breakable_positions, ccea_wins, std_wins, draws,
                     mp4_path)

    # Also open the interactive window
    fig, _ = _build_figure(terrain_walls, breakable_positions, ccea_wins, std_wins, draws)
    print('  [MAP] Interactive window open -- close it to exit.')
    plt.show()

# ================================================================
#  MAIN
# ================================================================
def main():
    print(chr(10) + '#'*70)
    print('  CCE-TDCCEA  vs  Standard CCEA  --  48x48 Dynamic Map Match')
    print('#'*70)
    print(f'  Pop={N_POP} | Gens={TOTAL_GENERATIONS} | H2H={H2H_GAMES} games | Workers={N_WORKERS}' + chr(10))

    write_map_xml(set(), set(), TEMP_MAP_PATH)
    print(f'  Temp map: {TEMP_MAP_PATH}' + chr(10))

    gpool = ParallelGamePool(N_WORKERS)
    gpool.start()

    # ---------- PHASE 1: CO-EVOLUTION ----------
    print(chr(10) + '='*60)
    print(f'  PHASE 1: CO-EVOLUTION  ({TOTAL_GENERATIONS} generations)')
    print("  Each algo uses the other's current best as opponent")
    print('='*60)

    terrain_walls       = set()
    breakable_positions = set()
    ccea_state = CCEAState()
    std_state  = StdCCEAState()
    cur = {'ccea': ccea_state.current_best_weights(),
           'std':  std_state.current_best_weights()}

    t0 = time.time()
    for gen in range(1, TOTAL_GENERATIONS + 1):
        terrain_walls, breakable_positions, n_t, n_b =                 add_walls_for_generation(gen, terrain_walls, breakable_positions)
        tw = len(terrain_walls) + len(breakable_positions)
        ok = bfs_reachable(terrain_walls | breakable_positions, BASE_P0, BASE_P1)
        write_map_xml(terrain_walls, breakable_positions, TEMP_MAP_PATH)
        print(f'  --- Gen {gen}/{TOTAL_GENERATIONS}  Map: +{n_t}T +{n_b}B -> {tw} walls  BFS: {"OK" if ok else "BLOCKED"}')
        nb = {}
        nb['ccea'] = step_ccea(gpool, ccea_state, [cur['std']],  gen)
        nb['std']  = step_std_ccea(gpool, std_state, [cur['ccea']], gen)
        cur = nb
    print(f'  Evolution done in {time.time()-t0:.1f}s')

    # ---------- PHASE 2: H2H TOURNAMENT ----------
    ft = len(terrain_walls) + len(breakable_positions)
    print(chr(10) + '='*70)
    print(f'  PHASE 2: HEAD-TO-HEAD  ({H2H_GAMES} games)  Map: {ft} blockades')
    print('='*70 + chr(10))
    write_map_xml(terrain_walls, breakable_positions, TEMP_MAP_PATH)
    ccea_wins, std_wins, draws = run_h2h_2(gpool, cur['ccea'], cur['std'])
    tg = ccea_wins + std_wins + draws
    if ccea_wins > std_wins:   verdict = 'CCE-TDCCEA WINS the match'
    elif std_wins > ccea_wins: verdict = 'Standard CCEA WINS the match'
    else:                      verdict = 'DRAW'
    print(f'  +-------------------------------------------------------+')
    print(f'  |  MATCH RESULT  ({tg} games)                               |')
    print(f'  |  CCE-TDCCEA {ccea_wins:>2d}W  |  {draws:>2d}D  |  {std_wins:>2d}L  Standard CCEA     |')
    print(f'  |  Verdict: {verdict:<43s}|')
    print(f'  +-------------------------------------------------------+')
    gpool.stop()

    # ---------- PHASE 3: DISPLAY FINAL MAP + SAVE MP4 ----------
    mp4_out = os.path.join(SCRIPT_DIR, 'match_final_map.mp4')
    print(chr(10) + '='*70)
    print('  PHASE 3: FINAL 48x48 MAP  (black-and-white)')
    print(f'  Video will be saved to: {mp4_out}')
    print('='*70 + chr(10))
    show_final_map(terrain_walls, breakable_positions, ccea_wins, std_wins, draws,
                   TOTAL_GENERATIONS, save_mp4=True, mp4_path=mp4_out)
    print('  Done.')

if __name__ == '__main__':
    main()
