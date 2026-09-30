"""
CCE_vs_STDCCEA_Visual_Match.py
===============================
Visual 1v1 Match: CCE-TDCCEA vs Standard CCEA on a 48×48 Blockaded Map

NO EVOLUTION — just blockade buildup + fight + visualization.

HOW IT WORKS:
  1. Define two strategy weight sets (CCE-TDCCEA and Standard CCEA)
  2. Build up blockades over 30 generations (walls only, no algorithm runs)
  3. Fight the two strategies on the final blockaded map (10 games)
  4. Render the final 48×48 map as a publication-quality PNG with results

Adapted from: SEM7_PR1_Results_fixed_v3.py
"""

import math, random, time, subprocess, os, json, collections

os.environ["PYTHONIOENCODING"] = "utf-8"

# ═══════════════════════════════════════════════════════════════
#  STRATEGY WEIGHTS — EDIT THESE!
#  Each strategy is a list of 20 floats (0.0 to 1.0).
# ═══════════════════════════════════════════════════════════════

# Our Novel Algorithm: CCE-TDCCEA
CCE_TDCCEA_WEIGHTS = [
      0.1631562971466478,
      0.9142085683469012,
      0.7267034005713626,
      0.6415502273412712,
      0.132109011071469,
      0.3162912794782645,
      0.20036595299756788,
      0.8201500599399819,
      0.8567073909929236,
      0.336368075842556,
      0.9820622963602281,
      0.8940311498397342,
      0.4671529141228479,
      0.9966075691398314,
      0.050033336363248804,
      0.5353216540486343,
      0.2731181110864497,
      0.07065683338325757,
      0.7937253582768544,
      0.5804077830504009]



# Baseline: Standard CCEA
STD_CCEA_WEIGHTS = [
      0.7962506898320032,
      0.6236563444930644,
      0.3627926159103164,
      0.3669233007663065,
      0.485015310943819,
      0.4388022654345669,
      0.7182545374428759,
      0.735132042004145,
      0.5003527848539554,
      0.8001589230329272,
      0.9136671361878417,
      0.537927473768097,
      0.5098103845269748,
      0.14404201068461553,
      0.08462561154703507,
      0.19072685809979895,
      0.882462186785449,
      0.2296896762608519,
      0.9971079068775449,
      0.3597235025886867
    ]

WEIGHT_NAMES = [
    "worker_ratio", "max_workers", "expand_threshold", "light_preference",
    "heavy_preference", "ranged_preference", "barracks_timing", "aggression",
    "attack_threshold", "defend_radius", "target_priority", "defense_trigger",
    "counter_rush", "tech_switch", "economy_pressure", "scout_priority",
    "group_attack", "second_barracks", "worker_fight_ratio", "focus_fire",
]

# ═══════════════════════════════════════════════════════════════
#  MAP CONFIGURATION
# ═══════════════════════════════════════════════════════════════

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CODES_DIR    = os.path.dirname(SCRIPT_DIR)
MICRORTS_DIR = os.path.join(CODES_DIR, "Simulation", "microrts")

MAP_W, MAP_H  = 48, 48
TEMP_MAP_PATH  = os.path.join(MICRORTS_DIR, "maps", "temp_visual_match_48x48.xml")
TEMP_MAP_REL   = "maps/temp_visual_match_48x48.xml"
MAX_CYCLES     = 12000

TOTAL_GENERATIONS = 30
H2H_GAMES         = 10   # games in the tournament (5 as P0, 5 as P1)

# Reserved tiles (bases, workers, resources)
RESERVED = set()
for bx, by in [(0,0),(0,1),(1,1),(2,2),(1,2),(2,1),(3,2),(2,3),(3,3)]:
    RESERVED.add((bx, by))
for bx, by in [(MAP_W-1,MAP_H-1),(MAP_W-1,MAP_H-2),(MAP_W-2,MAP_H-2),
               (MAP_W-3,MAP_H-3),(MAP_W-2,MAP_H-3),(MAP_W-3,MAP_H-2),
               (MAP_W-4,MAP_H-3),(MAP_W-3,MAP_H-4),(MAP_W-4,MAP_H-4)]:
    RESERVED.add((bx, by))

# Blockade settings
WALLS_PER_GEN_EARLY = 6    # Gen 1-7
WALLS_PER_GEN_MID   = 10   # Gen 8-14
WALLS_PER_GEN_LATE  = 14   # Gen 15-30
BREAKABLE_RATIO     = 0.4


# ═══════════════════════════════════════════════════════════════
#  MAP GENERATION + BFS
# ═══════════════════════════════════════════════════════════════

def bfs_reachable(walls, start, end):
    """BFS: can start reach end without crossing walls?"""
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
    """Pick a random wall that doesn't break base-to-base path."""
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
    """Add blockades for one generation."""
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
    return terrain_walls | new_t, breakable_positions | new_b, len(new_t), len(new_b)


def write_map_xml(terrain_walls, breakable_positions, path):
    """Write 48x48 map XML."""
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
#  GAME SERVER (single server, simple)
# ═══════════════════════════════════════════════════════════════

class GameServer:
    def __init__(self):
        self.process = None

    def start(self):
        print(f"  [SERVER] Starting Java game server...")
        print(f"  [SERVER] cwd = {MICRORTS_DIR}")
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
            raise RuntimeError(f"Server failed: stdout='{line}' stderr='{err}'")
        print(f"  [SERVER] Ready!")

    def run_game(self, p0_weights, p1_weights):
        """Run EVOLVED vs EVOLVED. Returns (winner, cycles)."""
        w0 = ";".join(f"{w:.4f}" for w in p0_weights)
        w1 = ";".join(f"{w:.4f}" for w in p1_weights)
        req = f"EVOLVED:{w0},EVOLVED:{w1},{TEMP_MAP_REL},{MAX_CYCLES}\n"
        try:
            self.process.stdin.write(req)
            self.process.stdin.flush()
            resp = self.process.stdout.readline().strip()
        except (BrokenPipeError, OSError) as e:
            print(f"  [ERROR] Pipe broken: {e}")
            return -1, MAX_CYCLES

        if resp.startswith("RESULT:"):
            parts = resp.replace("RESULT:", "").split(",")
            winner, cycles = int(parts[0]), int(parts[1])
            # Timeout fix: force draw if game hit max cycles
            if cycles >= MAX_CYCLES and winner != -1:
                winner = -1
            return winner, cycles
        return -1, MAX_CYCLES

    def stop(self):
        if self.process:
            try:
                self.process.stdin.write("EXIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
        print(f"  [SERVER] Stopped.")


# ═══════════════════════════════════════════════════════════════
#  MAP VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def render_match_map(terrain_walls, breakable_positions, match_result, output_path):
    """Render the final 48x48 blockaded map with match results as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib import rcParams
    except ImportError:
        print("  [WARN] matplotlib not installed! Run: pip install matplotlib")
        return

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

    rcParams["font.family"] = "serif"
    rcParams["font.size"] = 9

    fig, ax = plt.subplots(1, 1, figsize=(12, 13), dpi=200)

    # Draw terrain
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

    # Draw units
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
            plabel = "CCE-TDCCEA" if player == 0 else "Std CCEA"
            ax.text(cx, cy, "B", ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white", zorder=6)
            offset_x = 2.5 if player == 0 else -2.5
            ax.text(cx + offset_x, cy + 1.5, plabel,
                    ha="center", va="center", fontsize=6,
                    fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=color, alpha=0.9), zorder=7)

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

    # Axes
    ax.set_xlim(0, MAP_W)
    ax.set_ylim(MAP_H, 0)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, MAP_W + 1, 4))
    ax.set_yticks(range(0, MAP_H + 1, 4))
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.15, linewidth=0.3)

    # Title
    cce_w = match_result["cce_wins"]
    std_w = match_result["std_wins"]
    draws = match_result["draws"]
    winner = match_result["winner_name"]

    title1 = "CCE-TDCCEA  vs  Standard CCEA  -  1v1 FIGHT"
    title2 = f"Final 48x48 Map after {TOTAL_GENERATIONS} generations  |  {len(terrain_walls)} terrain + {len(breakable_positions)} breakable walls"
    ax.set_title(f"{title1}\n{title2}", fontsize=12, fontweight="bold", pad=15)

    # Result banner
    result_text = f"MATCH RESULT:  CCE-TDCCEA {cce_w}W  -  Std CCEA {std_w}W  -  {draws} Draws"
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

    # Legend
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
    print(f"\n  [MAP] Saved -> {output_path}")

    try:
        plt.switch_backend("TkAgg")
        plt.show()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "#" * 70)
    print("  CCE-TDCCEA  vs  STANDARD CCEA  -  VISUAL MATCH")
    print("  48x48 Map with 30-Generation Blockade Buildup")
    print("#" * 70)

    # ── Print the strategy weights ────────────────────────────
    print(f"\n  CCE-TDCCEA Strategy Weights:")
    for name, val in zip(WEIGHT_NAMES, CCE_TDCCEA_WEIGHTS):
        print(f"    {name:<22s} = {val:.4f}")

    print(f"\n  Standard CCEA Strategy Weights:")
    for name, val in zip(WEIGHT_NAMES, STD_CCEA_WEIGHTS):
        print(f"    {name:<22s} = {val:.4f}")

    # ══════════════════════════════════════════════════════════
    #  STEP 1: BUILD BLOCKADED MAP (30 generations of walls)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STEP 1: Building 48x48 map with {TOTAL_GENERATIONS} generations of blockades")
    print(f"{'='*60}\n")

    terrain_walls       = set()
    breakable_positions = set()

    for gen in range(1, TOTAL_GENERATIONS + 1):
        terrain_walls, breakable_positions, n_t, n_b = \
            add_walls_for_generation(gen, terrain_walls, breakable_positions)
        total = len(terrain_walls) + len(breakable_positions)

        # BFS check
        base0, base1 = (2, 2), (MAP_W - 3, MAP_H - 3)
        reachable = bfs_reachable(terrain_walls | breakable_positions, base0, base1)
        status = "OK" if reachable else "BLOCKED!"

        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS}  |  +{n_t} terrain +{n_b} breakable  |  "
              f"Total: {total:3d} walls  |  BFS: {status}")

    final_terrain = len(terrain_walls)
    final_breakable = len(breakable_positions)
    final_total = final_terrain + final_breakable
    print(f"\n  Final map: {final_total} walls ({final_terrain} terrain + {final_breakable} breakable)")

    # Write the final map
    write_map_xml(terrain_walls, breakable_positions, TEMP_MAP_PATH)
    print(f"  Map written to: {TEMP_MAP_PATH}")

    # ══════════════════════════════════════════════════════════
    #  STEP 2: FIGHT — 10 games on the final blockaded map
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STEP 2: FIGHTING — {H2H_GAMES} games on final blockaded map")
    print(f"  CCE-TDCCEA (P0/Blue) vs Standard CCEA (P1/Red)")
    print(f"{'='*60}\n")

    server = GameServer()
    server.start()

    cce_wins, std_wins, draws = 0, 0, 0

    print(f"  {'Game':>6s}  {'Slot':>10s}  {'Winner':>15s}  {'Cycles':>8s}")
    print(f"  {'-'*45}")

    for g in range(H2H_GAMES):
        swap = (g % 2 == 1)  # alternate player slots

        if swap:
            # Std CCEA as P0, CCE-TDCCEA as P1
            winner, cycles = server.run_game(STD_CCEA_WEIGHTS, CCE_TDCCEA_WEIGHTS)
        else:
            # CCE-TDCCEA as P0, Std CCEA as P1
            winner, cycles = server.run_game(CCE_TDCCEA_WEIGHTS, STD_CCEA_WEIGHTS)

        slot = "SWAPPED" if swap else "NORMAL"

        if not swap:
            if winner == 0:
                cce_wins += 1; who = "CCE-TDCCEA"
            elif winner == 1:
                std_wins += 1; who = "Std CCEA"
            else:
                draws += 1; who = "DRAW"
        else:
            if winner == 0:
                std_wins += 1; who = "Std CCEA"
            elif winner == 1:
                cce_wins += 1; who = "CCE-TDCCEA"
            else:
                draws += 1; who = "DRAW"

        print(f"  {g+1:>6d}  {slot:>10s}  {who:>15s}  {cycles:>8d}")

    server.stop()

    # Determine winner
    if cce_wins > std_wins:
        winner_name = "CCE-TDCCEA"
    elif std_wins > cce_wins:
        winner_name = "Std CCEA"
    else:
        winner_name = "DRAW"

    # Print final score
    print(f"\n  {'='*50}")
    print(f"  |{'FINAL SCORE':^48s}|")
    print(f"  {'='*50}")
    print(f"  |  CCE-TDCCEA :  {cce_wins:2d} wins{' '*27}|")
    print(f"  |  Std CCEA   :  {std_wins:2d} wins{' '*27}|")
    print(f"  |  Draws      :  {draws:2d}{' '*32}|")
    print(f"  |{' '*48}|")
    if winner_name == "CCE-TDCCEA":
        print(f"  |  >>> WINNER: CCE-TDCCEA (Our Algorithm!) <<<  |")
    elif winner_name == "Std CCEA":
        print(f"  |  >>> WINNER: Standard CCEA (Baseline) <<<     |")
    else:
        print(f"  |  >>> DRAW - No Clear Winner <<<                |")
    print(f"  {'='*50}")

    # ══════════════════════════════════════════════════════════
    #  STEP 3: VISUALIZE the final blockaded map
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STEP 3: Rendering final 48x48 blockaded map")
    print(f"{'='*60}")

    match_result = {
        "cce_wins":    cce_wins,
        "std_wins":    std_wins,
        "draws":       draws,
        "winner_name": winner_name,
    }

    output_path = os.path.join(SCRIPT_DIR, "visual_match_map_48x48.png")
    render_match_map(terrain_walls, breakable_positions, match_result, output_path)

    # ══════════════════════════════════════════════════════════
    #  STEP 4: LAUNCH VISUAL SIMULATION (Java GUI)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STEP 4: Launching VISUAL SIMULATION (Java GUI)")
    print(f"  Watch CCE-TDCCEA vs Standard CCEA fight LIVE!")
    print(f"  Close the game window when done.")
    print(f"{'='*60}\n")

    cce_weight_str = ";".join(f"{w:.4f}" for w in CCE_TDCCEA_WEIGHTS)
    std_weight_str = ";".join(f"{w:.4f}" for w in STD_CCEA_WEIGHTS)

    visual_cmd = [
        "java", "-cp", "lib/*;bin",
        "tests.EvolvedVisualGameRunner",
        f"EVOLVED:{cce_weight_str}",
        f"EVOLVED:{std_weight_str}",
        TEMP_MAP_REL,
    ]

    print(f"  Starting visual game GUI...")
    print(f"  CCE-TDCCEA (Player 0 / Blue) vs Std CCEA (Player 1 / Red)")
    print(f"  Map: 48x48 with {final_total} blockades\n")

    try:
        visual_process = subprocess.Popen(
            visual_cmd,
            cwd=MICRORTS_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Print output in real time
        for line in visual_process.stdout:
            stripped = line.strip()
            if stripped:
                print(f"  [GAME] {stripped}")

        visual_process.wait()
        print(f"\n  Visual game finished!")

    except FileNotFoundError:
        print("  [ERROR] Java not found! Make sure Java is installed and in PATH.")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # ══════════════════════════════════════════════════════════
    #  DONE
    # ══════════════════════════════════════════════════════════
    print(f"\n{'#'*70}")
    print(f"  MATCH COMPLETE!")
    print(f"{'#'*70}")
    print(f"  Output: visual_match_map_48x48.png")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
