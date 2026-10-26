"""
tdccea_dynamic_env.py — TD-CCEA vs Hill Climbing: Competitive Coevolution

COMPETITIVE COEVOLUTION with dynamic environment:
  - TD-CCEA evolves bots that play AGAINST Hill Climbing's best bot
  - Hill Climbing evolves bots that play AGAINST TD-CCEA's best bot
  - Each generation adds unbreakable AND breakable walls to the 32x32 map
  - Arms race: as one algorithm improves, the other must adapt
  - Final showdown: TD-CCEA's champion vs HC's champion on the evolved terrain

The two algorithms are opponents — they evolve strategies to beat EACH OTHER,
not a fixed bot. The dynamic environment (progressive walls) adds extra pressure.
"""

import math, random, time, subprocess, os, collections, queue
from concurrent.futures import ThreadPoolExecutor

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MICRORTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Simulation", "microrts")
TEMP_MAP = os.path.join(MICRORTS_DIR, "maps", "32x32", "temp_dynenv.xml")
TEMP_MAP_REL = "maps/32x32/temp_dynenv.xml"

MAP_W, MAP_H = 32, 32
MAX_CYCLES = 8000

# Reserved tiles (bases, workers, resources + buffers)
RESERVED = set()
for bx, by in [(0,0),(1,0),(0,1),(2,2),(3,2),(2,3),(1,2),(2,1),(1,1),
               (31,31),(30,31),(31,30),(29,29),(28,29),(29,28),(30,29),(29,30),(30,30)]:
    RESERVED.add((bx, by))

# TD-CCEA parameters
N_STRATEGY = 20
N_POP = 15
K_GAMES = 2
H_WINDOW = 5
MU_BASE = 0.15
DELTA_F = 0.20
DELTA_S = 0.85
A_MAX = 10
TOTAL_GENERATIONS =30


# Hill Climbing
HC_CANDIDATES = N_POP
HC_GAMES = K_GAMES

# Dynamic environment
WALLS_PER_GEN_EARLY = 3
WALLS_PER_GEN_MID = 5
WALLS_PER_GEN_LATE = 7
BREAKABLE_RATIO = 0.4

# CPU parallelism
N_WORKERS = max(2, min((os.cpu_count() or 4) - 1, N_POP))

STRATEGY_NAMES = [
    "worker_ratio","max_workers","expand_threshold","light_preference",
    "heavy_preference","ranged_preference","barracks_timing","aggression",
    "attack_threshold","defend_radius","target_priority","defense_trigger",
    "counter_rush","tech_switch","economy_pressure","scout_priority",
    "group_attack","second_barracks","worker_fight_ratio","focus_fire",
]


# ═══════════════════════════════════════════════════════════════
#  MAP GENERATION + BFS VALIDATION
# ═══════════════════════════════════════════════════════════════

def bfs_reachable(walls, start, end):
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
    base0, base1 = (2, 2), (29, 29)
    all_walls = terrain_walls | breakable_positions
    no_place = all_walls | RESERVED
    for _ in range(200):
        cx, cy = MAP_W // 2, MAP_H // 2
        spread = 6 + gen // 3
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
    if gen <= 7:
        n_new = WALLS_PER_GEN_EARLY
    elif gen <= 14:
        n_new = WALLS_PER_GEN_MID
    else:
        n_new = WALLS_PER_GEN_LATE
    new_t, new_b = set(), set()
    for _ in range(n_new):
        pos = pick_wall_position(terrain_walls | new_t, breakable_positions | new_b, gen)
        if pos is None:
            break
        if random.random() < BREAKABLE_RATIO:
            new_b.add(pos)
        else:
            new_t.add(pos)
    terrain_walls = terrain_walls | new_t
    breakable_positions = breakable_positions | new_b
    return terrain_walls, breakable_positions, len(new_t), len(new_b)


N_EXTRA_RESOURCES = 6       # number of random resource patches per map
RESOURCE_AMOUNT = 10        # resources per patch (smaller than base resources)

def generate_random_resources(terrain_walls, breakable_positions):
    """Generate random resource positions avoiding walls, bases, workers."""
    occupied = terrain_walls | breakable_positions | RESERVED
    resources = []
    for _ in range(N_EXTRA_RESOURCES):
        for _ in range(100):  # attempts
            x = random.randint(3, MAP_W - 4)
            y = random.randint(3, MAP_H - 4)
            if (x, y) not in occupied and (x, y) not in [(r[0], r[1]) for r in resources]:
                resources.append((x, y))
                break
    return resources


def write_map_xml(terrain_walls, breakable_positions, extra_resources, path):
    terrain = ['0'] * (MAP_W * MAP_H)
    for wx, wy in terrain_walls:
        terrain[wy * MAP_W + wx] = '1'
    terrain_str = ''.join(terrain)
    units_xml = []
    # Base resources (always present)
    units_xml.append('    <rts.units.Unit type="Resource" ID="1" player="-1" x="0" y="0" resources="25" hitpoints="1"></rts.units.Unit>')
    units_xml.append('    <rts.units.Unit type="Resource" ID="2" player="-1" x="31" y="31" resources="25" hitpoints="1"></rts.units.Unit>')
    # Bases and workers
    units_xml.append('    <rts.units.Unit type="Base" ID="3" player="0" x="2" y="2" resources="0" hitpoints="10"></rts.units.Unit>')
    units_xml.append('    <rts.units.Unit type="Base" ID="4" player="1" x="29" y="29" resources="0" hitpoints="10"></rts.units.Unit>')
    units_xml.append('    <rts.units.Unit type="Worker" ID="5" player="0" x="3" y="2" resources="0" hitpoints="1"></rts.units.Unit>')
    units_xml.append('    <rts.units.Unit type="Worker" ID="6" player="1" x="28" y="29" resources="0" hitpoints="1"></rts.units.Unit>')
    uid = 7
    # Random extra resources
    for rx, ry in extra_resources:
        units_xml.append(f'    <rts.units.Unit type="Resource" ID="{uid}" player="-1" x="{rx}" y="{ry}" resources="{RESOURCE_AMOUNT}" hitpoints="1"></rts.units.Unit>')
        uid += 1
    # Breakable walls
    for wx, wy in sorted(breakable_positions):
        units_xml.append(f'    <rts.units.Unit type="Wall" ID="{uid}" player="-1" x="{wx}" y="{wy}" resources="0" hitpoints="5"></rts.units.Unit>')
        uid += 1
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
    with open(path, 'w') as f:
        f.write(xml)


def render_map_ascii(terrain_walls, breakable_positions, extra_resources=None):
    grid = [['.' for _ in range(MAP_W)] for _ in range(MAP_H)]
    for wx, wy in terrain_walls:
        grid[wy][wx] = 'X'
    for wx, wy in breakable_positions:
        grid[wy][wx] = '#'
    if extra_resources:
        for rx, ry in extra_resources:
            grid[ry][rx] = '$'
    grid[0][0] = 'R'; grid[31][31] = 'R'
    grid[2][2] = 'B'; grid[29][29] = 'B'
    grid[2][3] = 'W'; grid[29][28] = 'W'
    lines = []
    for row in grid:
        lines.append('  | ' + ''.join(row))
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  PARALLEL GAME SERVER POOL (supports EVOLVED vs EVOLVED)
# ═══════════════════════════════════════════════════════════════

class GameServer:
    def __init__(self):
        self.process = None

    def start(self):
        cmd = ["java", "-cp", "lib/*;bin", "tests.EvolvedBatchGameRunner"]
        self.process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=MICRORTS_DIR)
        line = self.process.stdout.readline().strip()
        if line != "READY":
            raise RuntimeError(f"Server failed: {line}")

    def run_game(self, p0_weights, p1_weights, map_path=TEMP_MAP_REL):
        """Run EVOLVED vs EVOLVED: both players are evolved bots."""
        w0 = ";".join(f"{w:.4f}" for w in p0_weights)
        w1 = ";".join(f"{w:.4f}" for w in p1_weights)
        req = f"EVOLVED:{w0},EVOLVED:{w1},{map_path},{MAX_CYCLES}\n"
        self.process.stdin.write(req)
        self.process.stdin.flush()
        resp = self.process.stdout.readline().strip()
        if resp.startswith("RESULT:"):
            parts = resp.replace("RESULT:", "").split(",")
            return int(parts[0]), int(parts[1])
        return -1, MAX_CYCLES

    def stop(self):
        if self.process:
            try:
                self.process.stdin.write("EXIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()


class ParallelGamePool:
    def __init__(self, n_workers):
        self.n_workers = n_workers
        self.server_queue = queue.Queue()

    def start(self):
        print(f"  [CPU] Starting {self.n_workers} parallel game servers...")
        for _ in range(self.n_workers):
            s = GameServer()
            s.start()
            self.server_queue.put(s)
        print(f"  [OK] {self.n_workers} servers ready (of {os.cpu_count()} CPU cores)")

    def _eval_one(self, p0_weights, p1_weights, k_games):
        """Thread-safe: evaluate P0 vs P1 for k_games."""
        server = self.server_queue.get()
        try:
            wins, total_cyc = 0, 0
            for _ in range(k_games):
                w, c = server.run_game(p0_weights, p1_weights)
                total_cyc += c
                if w == 0:
                    wins += 1
                elif w == -1:
                    wins += 0.3  # draw
            return wins / k_games, total_cyc / k_games
        finally:
            self.server_queue.put(server)

    def evaluate_batch(self, p0_list, p1_weights, k_games):
        """Evaluate a batch of P0 agents all vs the same P1 opponent."""
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(self._eval_one, p0, p1_weights, k_games)
                       for p0 in p0_list]
            return [f.result() for f in futures]

    def stop(self):
        while not self.server_queue.empty():
            self.server_queue.get().stop()
        print(f"  [OK] All servers stopped")


# ═══════════════════════════════════════════════════════════════
#  TD-CCEA ALGORITHM
# ═══════════════════════════════════════════════════════════════

def euclidean_dist(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a, b)))

class TDCCEAAgent:
    def __init__(self, params=None):
        self.params = [max(0.0, min(1.0, p)) for p in params] if params else [random.random() for _ in range(N_STRATEGY)]
        self.H_i = []
        self.e_i = 0.0
        self.mu_i = MU_BASE

    def update(self, fitness, env):
        self.H_i.append((fitness, env))
        if len(self.H_i) > H_WINDOW:
            self.H_i.pop(0)
        if self.H_i:
            max_d = math.sqrt(N_STRATEGY)
            dw = [max(0.01, 1.0 - euclidean_dist(env, pe)/max_d) for _, pe in self.H_i]
            t = sum(dw)
            self.e_i = sum(dw[i]/t * self.H_i[i][0] for i in range(len(self.H_i)))
        self.mu_i = MU_BASE * (1.0 - self.e_i) + 0.03

    def mutate(self):
        return TDCCEAAgent([p + random.gauss(0, self.mu_i) for p in self.params])


class TDCCEA:
    def __init__(self):
        self.population = [TDCCEAAgent() for _ in range(N_POP)]
        self.archive = []
        self.best_ever = None

    def get_best_weights(self):
        if self.best_ever:
            return self.best_ever.params
        best = max(self.population, key=lambda a: a.e_i)
        return best.params

    def get_centroid(self):
        c = [0.0] * N_STRATEGY
        for a in self.population:
            for j in range(N_STRATEGY):
                c[j] += a.params[j]
        return [x / N_POP for x in c]

    def evaluate(self, pool, opponent_weights):
        """Evaluate all agents vs the opponent (HC's best bot)."""
        env = self.get_centroid()
        weight_vectors = [a.params for a in self.population]
        results = pool.evaluate_batch(weight_vectors, opponent_weights, K_GAMES)

        for agent, (wr, cyc) in zip(self.population, results):
            speed = max(0, (MAX_CYCLES - cyc) / MAX_CYCLES) * 0.15
            fit = wr + (speed if wr > 0.5 else 0)
            agent.update(fit, env)

        best = max(self.population, key=lambda a: a.e_i)
        avg = sum(a.e_i for a in self.population) / N_POP

        if self.best_ever is None or best.e_i > self.best_ever.e_i:
            self.best_ever = TDCCEAAgent(best.params)
            self.best_ever.e_i = best.e_i

        return best.e_i, avg

    def evolve(self):
        max_d = math.sqrt(N_STRATEGY)
        survivors = []
        for a in sorted(self.population, key=lambda a: a.e_i, reverse=True):
            if not any(1.0 - euclidean_dist(a.params, s.params)/max_d > DELTA_S for s in survivors):
                survivors.append(a)
        final = [a for a in survivors if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
        while len(final) < 6 and survivors:
            for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
                if a not in final:
                    final.append(a)
                    break
        for top in sorted(self.population, key=lambda a: a.e_i, reverse=True)[:3]:
            self.archive.append(TDCCEAAgent(top.params))
        if len(self.archive) > A_MAX:
            self.archive = self.archive[-A_MAX:]
        self.population = final.copy()
        while len(self.population) < N_POP:
            p = final + self.archive
            cands = random.sample(p, min(3, len(p)))
            parent = max(cands, key=lambda a: a.e_i)
            self.population.append(parent.mutate())


# ═══════════════════════════════════════════════════════════════
#  HILL CLIMBING ALGORITHM
# ═══════════════════════════════════════════════════════════════

class HillClimber:
    def __init__(self):
        self.current = [random.random() for _ in range(N_STRATEGY)]
        self.current_fitness = 0.0
        self.best_ever = None
        self.best_ever_fitness = 0.0
        self.mutation_sigma = 0.15

    def get_best_weights(self):
        return self.best_ever if self.best_ever else self.current

    def evaluate(self, pool, opponent_weights):
        """Evaluate candidates vs the opponent (TD-CCEA's best bot)."""
        candidates = [self.current]
        for _ in range(HC_CANDIDATES - 1):
            mutant = [max(0.0, min(1.0, p + random.gauss(0, self.mutation_sigma)))
                      for p in self.current]
            candidates.append(mutant)

        results = pool.evaluate_batch(candidates, opponent_weights, HC_GAMES)

        best_idx, best_fit = 0, -1
        for i, (wr, cyc) in enumerate(results):
            speed = max(0, (MAX_CYCLES - cyc) / MAX_CYCLES) * 0.15
            fit = wr + (speed if wr > 0.5 else 0)
            if fit > best_fit:
                best_fit = fit
                best_idx = i

        if best_fit >= self.current_fitness:
            self.current = candidates[best_idx]
            self.current_fitness = best_fit

        if best_fit > self.current_fitness:
            self.mutation_sigma = max(0.05, self.mutation_sigma * 0.95)
        else:
            self.mutation_sigma = min(0.30, self.mutation_sigma * 1.05)

        if best_fit > self.best_ever_fitness:
            self.best_ever = list(self.current)
            self.best_ever_fitness = best_fit

        return self.current_fitness, best_fit


# ═══════════════════════════════════════════════════════════════
#  MAIN: COMPETITIVE COEVOLUTION
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*70}")
    print(f"  COMPETITIVE COEVOLUTION: TD-CCEA vs Hill Climbing")
    print(f"  with Dynamic Environment (Progressive Blockades)")
    print(f"{'='*70}")
    print(f"  Map: {MAP_W}x{MAP_H} open -> progressive blockades")
    print(f"  Walls: terrain (unbreakable) + Wall units (breakable, HP=5)")
    print(f"  Matchup: TD-CCEA's bots  <-->  HC's bots (FACE EACH OTHER)")
    print(f"  TD-CCEA: pop={N_POP}, {K_GAMES} games/eval")
    print(f"  Hill-HC: {HC_CANDIDATES} candidates, {HC_GAMES} games/eval")
    print(f"  CPU: {N_WORKERS} parallel workers (of {os.cpu_count()} cores)")
    print(f"  Generations: {TOTAL_GENERATIONS}")
    print(f"{'='*70}\n")

    pool = ParallelGamePool(N_WORKERS)
    pool.start()

    tdccea = TDCCEA()
    hc = HillClimber()

    terrain_walls = set()
    breakable_positions = set()
    history = []
    t_start = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        gen_start = time.time()

        # -- Add walls & generate random resources --
        terrain_walls, breakable_positions, n_t, n_b = \
            add_walls_for_generation(gen, terrain_walls, breakable_positions)
        extra_resources = generate_random_resources(terrain_walls, breakable_positions)
        total_walls = len(terrain_walls) + len(breakable_positions)
        write_map_xml(terrain_walls, breakable_positions, extra_resources, TEMP_MAP)

        print(f"  +-- Generation {gen}/{TOTAL_GENERATIONS} " + "-"*45)
        print(f"  |  Environment: +{n_t} terrain +{n_b} breakable -> Total: {total_walls} walls")

        # -- TD-CCEA evaluates its agents vs HC's current best --
        hc_opponent = hc.get_best_weights()
        t1 = time.time()
        tdccea_best, tdccea_avg = tdccea.evaluate(pool, hc_opponent)
        t_td = time.time() - t1
        tdccea.evolve()

        # -- HC evaluates its candidates vs TD-CCEA's current best --
        tdccea_opponent = tdccea.get_best_weights()
        t2 = time.time()
        hc_curr, hc_best = hc.evaluate(pool, tdccea_opponent)
        t_hc = time.time() - t2

        print(f"  |")
        print(f"  |  TD-CCEA vs HC's best:  Best={tdccea_best:.3f}  Avg={tdccea_avg:.3f}  ({t_td:.1f}s)")
        print(f"  |  HC vs TD-CCEA's best:  Best={hc_curr:.3f}  ({t_hc:.1f}s)")
        print(f"  |")

        gen_time = time.time() - gen_start
        print(f"  |  Gen time: {gen_time:.1f}s")
        print(f"  +" + "-"*60)

        history.append({
            'gen': gen, 'walls': total_walls,
            'tdccea_best': tdccea_best, 'tdccea_avg': tdccea_avg,
            'hc_best': hc_curr,
        })

    elapsed = time.time() - t_start
    pool.stop()

    # ═══════════════════════════════════════════════════════════
    #  FINAL HEAD-TO-HEAD SHOWDOWN
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  FINAL SHOWDOWN: TD-CCEA Champion vs HC Champion")
    print(f"  10 matches on the evolved terrain ({total_walls} walls)")
    print(f"{'='*70}")

    write_map_xml(terrain_walls, breakable_positions, extra_resources, TEMP_MAP)
    showdown_server = GameServer()
    showdown_server.start()

    td_weights = tdccea.get_best_weights()
    hc_weights = hc.get_best_weights()

    td_wins, hc_wins, draws = 0, 0, 0
    for g in range(10):
        # Alternate who plays P0/P1 for fairness
        if g % 2 == 0:
            winner, cyc = showdown_server.run_game(td_weights, hc_weights)
            if winner == 0:
                td_wins += 1; tag = "TD-CCEA WINS"
            elif winner == 1:
                hc_wins += 1; tag = "HC WINS"
            else:
                draws += 1; tag = "DRAW"
        else:
            winner, cyc = showdown_server.run_game(hc_weights, td_weights)
            if winner == 0:
                hc_wins += 1; tag = "HC WINS"
            elif winner == 1:
                td_wins += 1; tag = "TD-CCEA WINS"
            else:
                draws += 1; tag = "DRAW"
        print(f"  Game {g+1:2d}: {tag}  (cycles: {cyc})")

    showdown_server.stop()

    print(f"\n  +-- FINAL RESULTS " + "-"*40)
    print(f"  |  TD-CCEA:  {td_wins}/10 wins")
    print(f"  |  Hill-HC:  {hc_wins}/10 wins")
    print(f"  |  Draws:    {draws}/10")
    champion = "TD-CCEA" if td_wins > hc_wins else "Hill Climbing" if hc_wins > td_wins else "TIE"
    print(f"  |")
    print(f"  |  CHAMPION: {champion}!")
    print(f"  +" + "-"*60)

    # -- Evolution history --
    print(f"\n  {'Gen':>4}  {'Walls':>5}  {'TD-CCEA':>8}  {'HC':>8}")
    print(f"  {'---':>4}  {'---':>5}  {'---':>8}  {'---':>8}")
    for r in history:
        print(f"  {r['gen']:4d}  {r['walls']:5d}  {r['tdccea_best']:8.3f}  {r['hc_best']:8.3f}")

    # -- Final terrain --
    print(f"\n  Final Terrain ({len(terrain_walls)} terrain + {len(breakable_positions)} breakable + {len(extra_resources)} resource patches):")
    print(render_map_ascii(terrain_walls, breakable_positions, extra_resources))
    print(f"  Legend: X=unbreakable  #=breakable  $=resource patch  B=base  W=worker  R=base resource")

    # -- Save results --
    rpath = os.path.join(SCRIPT_DIR, "dynamic_env_results.txt")
    with open(rpath, "w") as f:
        f.write("# Competitive Coevolution: TD-CCEA vs Hill Climbing\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total time: {elapsed:.1f}s\n")
        f.write(f"# Final: TD-CCEA {td_wins} - HC {hc_wins} - Draws {draws}\n\n")
        f.write("Gen,Walls,TDCCEA_Best,TDCCEA_Avg,HC_Best\n")
        for r in history:
            f.write(f"{r['gen']},{r['walls']},{r['tdccea_best']:.4f},"
                    f"{r['tdccea_avg']:.4f},{r['hc_best']:.4f}\n")
        f.write(f"\n# TD-CCEA best weights:\n")
        for i, n in enumerate(STRATEGY_NAMES):
            f.write(f"{n}={td_weights[i]:.6f}\n")
        f.write(f"\n# HC best weights:\n")
        for i, n in enumerate(STRATEGY_NAMES):
            f.write(f"{n}={hc_weights[i]:.6f}\n")
        f.write(f"\n# Terrain walls:\n")
        for wx, wy in sorted(terrain_walls):
            f.write(f"terrain_wall={wx},{wy}\n")
        f.write(f"\n# Breakable walls:\n")
        for wx, wy in sorted(breakable_positions):
            f.write(f"breakable_wall={wx},{wy}\n")
        f.write(f"\n# Extra resource patches:\n")
        for rx, ry in sorted(extra_resources):
            f.write(f"extra_resource={rx},{ry}\n")
    print(f"\n  Saved: {rpath}")

    # ═══════════════════════════════════════════════════════════
    #  VISUAL REPLAY: Champion vs Champion
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  VISUAL REPLAY: TD-CCEA vs Hill Climbing on evolved terrain!")
    print(f"{'='*70}")
    time.sleep(1)

    w0 = ";".join(f"{w:.4f}" for w in td_weights)
    w1 = ";".join(f"{w:.4f}" for w in hc_weights)
    cmd = ["java", "-cp", "lib/*;bin", "tests.EvolvedVisualGameRunner",
           f"EVOLVED:{w0}", f"EVOLVED:{w1}", TEMP_MAP_REL]
    subprocess.run(cmd, cwd=MICRORTS_DIR)
    print("  [OK] Visual replay closed.\n")


if __name__ == "__main__":
    main()
