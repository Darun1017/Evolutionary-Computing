"""
tdccea_blockade.py — TD-CCEA: Co-Evolution of Environment + Strategy

Co-evolves BOTH the bot's 20 behavioral weights AND the map terrain layout.
Each agent encodes: 20 strategy weights + 12 environment genes (3 wall segments).

Before each game, the Python script:
  1. Decodes environment genes into wall positions on the 16x16 map
  2. Validates reachability with BFS (ensures bases are still connected)
  3. Writes a temporary map XML with unbreakable terrain walls
  4. Runs the game on that dynamically generated map

This creates a genuine "Dynamic Environment Evolution" where the terrain
itself evolves alongside the bot's strategy.
"""

import math, random, time, subprocess, os, collections

# ─── Paths ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MICRORTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Simulation", "microrts")
TEMP_MAP_PATH = os.path.join(MICRORTS_DIR, "maps", "16x16", "temp_evolved.xml")
TEMP_MAP_REL = "maps/16x16/temp_evolved.xml"

# ─── Map Constants ───
MAP_W, MAP_H = 16, 16
# Reserved tiles (units + 1-tile buffer around bases/resources for accessibility)
RESERVED = set()
for bx, by in [(0,0),(1,0),(0,1), (1,1),(2,1),(1,2),(3,1),(1,3),
               (15,15),(14,15),(15,14), (14,14),(13,14),(14,13),(12,14),(14,12)]:
    RESERVED.add((bx, by))

# ─── Strategy Weight Names ───
STRATEGY_NAMES = [
    "worker_ratio","max_workers","expand_threshold","light_preference",
    "heavy_preference","ranged_preference","barracks_timing","aggression",
    "attack_threshold","defend_radius","target_priority","defense_trigger",
    "counter_rush","tech_switch","economy_pressure","scout_priority",
    "group_attack","second_barracks","worker_fight_ratio","focus_fire",
]
N_STRATEGY = len(STRATEGY_NAMES)  # 20

# ─── Environment Gene Names (3 wall segments × 4 params) ───
ENV_NAMES = []
for i in range(3):
    ENV_NAMES += [f"seg{i}_cx", f"seg{i}_cy", f"seg{i}_len", f"seg{i}_dir"]
N_ENV = len(ENV_NAMES)  # 12
DIMENSIONS = N_STRATEGY + N_ENV  # 32

# ─── Game Config ───
MAX_CYCLES = 5000
OPPONENT = "WorkerRush"

# ─── TD-CCEA Hyperparameters ───
N_POP = 15
K_GAMES = 2
H_WINDOW = 5
MU_BASE = 0.15
DELTA_F = 0.20
DELTA_S = 0.85
A_MAX = 10
TOTAL_GENERATIONS = 15


# ═══════════════════════════════════════════════════════════════
#  ENVIRONMENT FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def decode_walls(env_genes):
    """Decode 12 environment genes into a set of (x,y) wall positions."""
    walls = set()
    for seg in range(3):
        b = seg * 4
        cx = int(env_genes[b] * (MAP_W - 6)) + 3      # 3..12
        cy = int(env_genes[b+1] * (MAP_H - 6)) + 3    # 3..12
        length = int(env_genes[b+2] * 4) + 2           # 2..6
        vertical = env_genes[b+3] >= 0.5
        for i in range(length):
            if vertical:
                wx, wy = cx, cy - length // 2 + i
            else:
                wx, wy = cx - length // 2 + i, cy
            if 0 <= wx < MAP_W and 0 <= wy < MAP_H and (wx, wy) not in RESERVED:
                walls.add((wx, wy))
    return walls


def bfs_reachable(walls, start, end):
    """BFS flood fill: returns True if start can reach end avoiding walls."""
    if start == end:
        return True
    visited = {start}
    queue = collections.deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < MAP_W and 0 <= ny < MAP_H:
                if (nx, ny) not in visited and (nx, ny) not in walls:
                    if (nx, ny) == end:
                        return True
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    return False


def validate_and_fix_walls(walls):
    """Remove walls until bases are reachable. Returns valid wall set."""
    base0, base1 = (1, 1), (14, 14)
    if bfs_reachable(walls, base0, base1):
        return walls
    # Remove walls one by one (farthest from center first) until valid
    wall_list = sorted(walls, key=lambda w: -abs(w[0]-8)-abs(w[1]-8))
    for w in wall_list:
        walls.discard(w)
        if bfs_reachable(walls, base0, base1):
            return walls
    return set()  # fallback: no walls


def write_map_xml(walls, path):
    """Write a 16x16 map XML with terrain walls at the given positions."""
    terrain = ['0'] * (MAP_W * MAP_H)
    for wx, wy in walls:
        terrain[wy * MAP_W + wx] = '1'
    terrain_str = ''.join(terrain)

    xml = f'''<rts.PhysicalGameState width="{MAP_W}" height="{MAP_H}">
  <terrain>{terrain_str}</terrain>
  <players>
    <rts.Player ID="0" resources="5"></rts.Player>
    <rts.Player ID="1" resources="5"></rts.Player>
  </players>
  <units>
    <rts.units.Unit type="Resource" ID="1" player="-1" x="0" y="0" resources="25" hitpoints="1"></rts.units.Unit>
    <rts.units.Unit type="Resource" ID="2" player="-1" x="15" y="15" resources="25" hitpoints="1"></rts.units.Unit>
    <rts.units.Unit type="Base" ID="3" player="0" x="1" y="1" resources="0" hitpoints="10"></rts.units.Unit>
    <rts.units.Unit type="Base" ID="4" player="1" x="14" y="14" resources="0" hitpoints="10"></rts.units.Unit>
    <rts.units.Unit type="Worker" ID="5" player="0" x="2" y="1" resources="0" hitpoints="1"></rts.units.Unit>
    <rts.units.Unit type="Worker" ID="6" player="1" x="13" y="14" resources="0" hitpoints="1"></rts.units.Unit>
  </units>
</rts.PhysicalGameState>'''
    with open(path, 'w') as f:
        f.write(xml)


def render_terrain_ascii(walls):
    """Render a small ASCII view of the terrain for console output."""
    grid = [['.' for _ in range(MAP_W)] for _ in range(MAP_H)]
    for wx, wy in walls:
        grid[wy][wx] = '█'
    grid[0][0] = 'R'; grid[15][15] = 'R'   # Resources
    grid[1][1] = 'B'; grid[14][14] = 'B'   # Bases
    grid[1][2] = 'W'; grid[14][13] = 'W'   # Workers
    lines = []
    for row in grid:
        lines.append('  │  ' + ''.join(row))
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  GAME SERVER
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
            err = self.process.stderr.read()
            raise RuntimeError(f"Server failed: {line}\n{err}")
        print("  [✓] Game server started")

    def run_game(self, strategy_weights, map_rel_path=TEMP_MAP_REL):
        w_str = ";".join(f"{w:.4f}" for w in strategy_weights)
        req = f"EVOLVED:{w_str},{OPPONENT},{map_rel_path},{MAX_CYCLES}\n"
        self.process.stdin.write(req)
        self.process.stdin.flush()
        resp = self.process.stdout.readline().strip()
        if resp.startswith("RESULT:"):
            parts = resp.replace("RESULT:", "").split(",")
            return int(parts[0]), int(parts[1])
        print(f"    [!] Unexpected: {resp}")
        return -1, MAX_CYCLES

    def stop(self):
        if self.process:
            try:
                self.process.stdin.write("EXIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            print("  [✓] Game server stopped")


# ═══════════════════════════════════════════════════════════════
#  TD-CCEA AGENT (Strategy + Environment Co-Evolution)
# ═══════════════════════════════════════════════════════════════

def euclidean_dist(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a, b)))

def get_centroid(pop):
    c = [0.0]*DIMENSIONS
    for a in pop:
        for j in range(DIMENSIONS):
            c[j] += a.params[j]
    return [x/len(pop) for x in c]


class CoEvoAgent:
    def __init__(self, params=None):
        if params is None:
            self.params = [random.random() for _ in range(DIMENSIONS)]
        else:
            self.params = [max(0.0, min(1.0, float(x))) for x in params]
        self.H_i = []
        self.e_i = 0.0
        self.mu_i = 0.0

    @property
    def strategy(self):
        return self.params[:N_STRATEGY]

    @property
    def env_genes(self):
        return self.params[N_STRATEGY:]

    def get_walls(self):
        raw = decode_walls(self.env_genes)
        return validate_and_fix_walls(raw)

    def get_profile(self):
        p = self.params
        parts = []
        walls = self.get_walls()
        parts.append(f"{len(walls)}-walls")
        if p[7] > 0.6: parts.append("aggressive")
        elif p[7] < 0.3: parts.append("defensive")
        else: parts.append("balanced")
        if p[12] > 0.5: parts.append("anti-rush")
        prefs = [p[3], p[4], p[5]]
        names = ["Light","Heavy","Ranged"]
        parts.append(f"{names[prefs.index(max(prefs))]}-focused")
        return " | ".join(parts)

    def update_history(self, fitness, env):
        self.H_i.append((fitness, env))
        if len(self.H_i) > H_WINDOW:
            self.H_i.pop(0)

    def compute_trajectory(self, env):
        if not self.H_i:
            self.e_i = 0.0
            return self.e_i
        max_d = math.sqrt(DIMENSIONS)
        dw = [1.0 - euclidean_dist(env, pe)/max_d for _, pe in self.H_i]
        t = sum(dw) or 1.0
        dw = [w/t for w in dw]
        self.e_i = sum(dw[i]*self.H_i[i][0] for i in range(len(self.H_i)))
        return self.e_i

    def compute_mutation(self):
        self.mu_i = MU_BASE * (1.0 - self.e_i) + 0.03
        return self.mu_i

    def mutate(self):
        child = [p + random.gauss(0, self.mu_i) for p in self.params]
        return CoEvoAgent(child)


# ═══════════════════════════════════════════════════════════════
#  MAIN EVOLUTION
# ═══════════════════════════════════════════════════════════════

def run():
    print(f"\n{'='*70}")
    print(f"  TD-CCEA: CO-EVOLUTION OF ENVIRONMENT + STRATEGY")
    print(f"{'='*70}")
    print(f"  Genes: {N_STRATEGY} strategy + {N_ENV} environment = {DIMENSIONS} total")
    print(f"  Map: 16x16 open | Opponent: {OPPONENT}")
    print(f"  Pop: {N_POP} | Gens: {TOTAL_GENERATIONS} | Games/eval: {K_GAMES}")
    print(f"  Environment: Unbreakable terrain walls (BFS-validated)")
    print(f"{'='*70}\n")

    server = GameServer()
    server.start()

    population = [CoEvoAgent() for _ in range(N_POP)]
    archive = []
    history = []
    t_start = time.time()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        gen_start = time.time()
        cur_env = get_centroid(population)

        print(f"  ┌─ Generation {gen}/{TOTAL_GENERATIONS} {'─'*45}")

        for agent in population:
            wins = 0
            total_cyc = 0
            walls = agent.get_walls()

            for _ in range(K_GAMES):
                write_map_xml(walls, TEMP_MAP_PATH)
                winner, cycles = server.run_game(agent.strategy)
                total_cyc += cycles
                if winner == 0:
                    wins += 1
                elif winner == -1:
                    wins += 0.3

            wr = wins / K_GAMES
            speed = max(0, (MAX_CYCLES - total_cyc/K_GAMES) / MAX_CYCLES) * 0.2
            f_i = wr + (speed if wr > 0.5 else 0)

            agent.update_history(f_i, cur_env)
            agent.compute_trajectory(cur_env)
            agent.compute_mutation()

        best = max(population, key=lambda a: a.e_i)
        avg = sum(a.e_i for a in population) / N_POP
        best_walls = best.get_walls()

        print(f"  │  Best e_i: {best.e_i:.3f} | Avg: {avg:.3f} | Walls: {len(best_walls)}")
        print(f"  │  Profile: {best.get_profile()}")

        history.append({
            'gen': gen, 'best_ei': best.e_i, 'avg_ei': avg,
            'n_walls': len(best_walls), 'profile': best.get_profile(),
            'params': best.params.copy(),
        })

        # ── Diversity survival ──
        max_d = math.sqrt(DIMENSIONS)
        survivors = []
        for a in sorted(population, key=lambda a: a.e_i, reverse=True):
            if not any(1.0 - euclidean_dist(a.params, s.params)/max_d > DELTA_S for s in survivors):
                survivors.append(a)

        final = [a for a in survivors if not (a.e_i < DELTA_F and len(a.H_i) >= 3)]
        while len(final) < 8 and survivors:
            for a in sorted(survivors, key=lambda a: a.e_i, reverse=True):
                if a not in final:
                    final.append(a)
                    break

        for top in sorted(population, key=lambda a: a.e_i, reverse=True)[:3]:
            archive.append(CoEvoAgent(top.params))
        if len(archive) > A_MAX:
            archive = archive[-A_MAX:]

        population = final.copy()
        while len(population) < N_POP:
            pool = final + archive
            cands = random.sample(pool, min(3, len(pool)))
            parent = max(cands, key=lambda a: a.e_i)
            population.append(parent.mutate())

        gt = time.time() - gen_start
        print(f"  │  Survivors: {len(final)} | Archive: {len(archive)} | Time: {gt:.1f}s")
        print(f"  └{'─'*60}")

    elapsed = time.time() - t_start
    server.stop()

    # ═══════════════════════════════════════════════════════════
    #  RESULTS
    # ═══════════════════════════════════════════════════════════

    best = max(archive, key=lambda a: a.e_i)
    best_walls = best.get_walls()

    print(f"\n{'='*70}")
    print(f"  CO-EVOLUTION COMPLETE")
    print(f"{'='*70}")
    print(f"  Time: {elapsed:.1f}s | Games: {TOTAL_GENERATIONS*N_POP*K_GAMES}")
    print(f"\n  ┌─ BEST EVOLVED STRATEGY + TERRAIN ─────────────────")
    print(f"  │  Fitness: {best.e_i:.3f}")
    print(f"  │  Profile: {best.get_profile()}")
    print(f"  │  Walls: {len(best_walls)} terrain blocks")
    print(f"  │")
    print(f"  │  Strategy Weights:")
    for i, name in enumerate(STRATEGY_NAMES):
        bar = "█" * int(best.params[i]*25) + "░" * (25 - int(best.params[i]*25))
        print(f"  │   w{i:2d} {name:22s} {bar} {best.params[i]*100:5.1f}%")
    print(f"  │")
    print(f"  │  Environment Genes:")
    for i, name in enumerate(ENV_NAMES):
        val = best.env_genes[i]
        print(f"  │   e{i:2d} {name:12s} = {val:.3f}")
    print(f"  │")
    print(f"  │  Evolved Terrain Map (█=wall, B=base, W=worker, R=resource):")
    print(render_terrain_ascii(best_walls))
    print(f"  └────────────────────────────────────────────────────")

    # ── Save ──
    rpath = os.path.join(SCRIPT_DIR, "blockade_best_weights.txt")
    with open(rpath, "w") as f:
        f.write("# TD-CCEA Co-Evolution — Best Strategy + Environment\n")
        f.write(f"# Fitness: {best.e_i:.3f} | Walls: {len(best_walls)}\n")
        f.write(f"# Profile: {best.get_profile()}\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n#\n")
        for i, n in enumerate(STRATEGY_NAMES):
            f.write(f"{n}={best.params[i]:.6f}\n")
        for i, n in enumerate(ENV_NAMES):
            f.write(f"{n}={best.env_genes[i]:.6f}\n")
        f.write(f"\n# Wall positions:\n")
        for wx, wy in sorted(best_walls):
            f.write(f"wall={wx},{wy}\n")
    print(f"\n  Saved to: {rpath}")

    # ── History ──
    print(f"\n  {'Gen':>4}  {'Best':>6}  {'Avg':>6}  {'Walls':>5}  Profile")
    print(f"  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*35}")
    for r in history:
        print(f"  {r['gen']:4d}  {r['best_ei']:6.3f}  {r['avg_ei']:6.3f}  {r['n_walls']:5d}  {r['profile']}")

    # ═══════════════════════════════════════════════════════════
    #  VALIDATION
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  VALIDATION: Best strategy on evolved terrain vs {OPPONENT}")
    print(f"{'='*70}")

    ts = GameServer()
    ts.start()
    write_map_xml(best_walls, TEMP_MAP_PATH)

    tw, td = 0, 0
    for g in range(10):
        w, c = ts.run_game(best.strategy)
        s = "WIN" if w == 0 else "LOSS" if w == 1 else "DRAW"
        if w == 0: tw += 1
        elif w == -1: td += 1
        print(f"  Game {g+1:2d}: {s} (cycles: {c})")
    ts.stop()

    print(f"\n  Result: {tw}/10 wins, {td} draws, {10-tw-td} losses")

    # ═══════════════════════════════════════════════════════════
    #  VISUAL REPLAY
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  LAUNCHING VISUAL REPLAY")
    print(f"  Watch the evolved bot play on its co-evolved terrain!")
    print(f"{'='*70}")
    time.sleep(1)

    write_map_xml(best_walls, TEMP_MAP_PATH)
    w_str = ";".join(f"{w:.4f}" for w in best.strategy)
    cmd = ["java", "-cp", "lib/*;bin", "tests.EvolvedVisualGameRunner",
           f"EVOLVED:{w_str}", OPPONENT, TEMP_MAP_REL]
    subprocess.run(cmd, cwd=MICRORTS_DIR)
    print("  [✓] Visual replay closed.\n")


if __name__ == "__main__":
    run()
