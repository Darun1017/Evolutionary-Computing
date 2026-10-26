"""
show_walls_10_20_30.py
======================
Simulate progressive wall addition on a 48x48 map up to 30 generations,
then display the map state at Gen 10, Gen 20, and Gen 30 one at a time,
each shown for 10 seconds so you can take a screenshot.
"""

import math, random, collections, time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION (mirrors SEM7_PR1_Results.py)
# ═══════════════════════════════════════════════════════════════

MAP_W, MAP_H = 48, 48
TOTAL_GENERATIONS = 30

WALLS_PER_GEN_EARLY = 6    # gen 1-7
WALLS_PER_GEN_MID   = 10   # gen 8-14
WALLS_PER_GEN_LATE  = 14   # gen 15+
BREAKABLE_RATIO     = 0.4

RESERVED = set()
for bx, by in [(0,0),(0,1),(1,1),(2,2),(1,2),(2,1),(3,2),(2,3),(3,3)]:
    RESERVED.add((bx, by))
for bx, by in [(MAP_W-1,MAP_H-1),(MAP_W-1,MAP_H-2),(MAP_W-2,MAP_H-2),
               (MAP_W-3,MAP_H-3),(MAP_W-2,MAP_H-3),(MAP_W-3,MAP_H-2),
               (MAP_W-4,MAP_H-3),(MAP_W-3,MAP_H-4),(MAP_W-4,MAP_H-4)]:
    RESERVED.add((bx, by))

BASE_P0    = (2, 2)
BASE_P1    = (MAP_W - 3, MAP_H - 3)
WORKER_P0  = (1, 1)
WORKER_P1  = (MAP_W - 2, MAP_H - 2)
RES_P0     = [(0, 0), (0, 1)]
RES_P1     = [(MAP_W - 1, MAP_H - 1), (MAP_W - 1, MAP_H - 2)]

SEED = 42

# ═══════════════════════════════════════════════════════════════
#  BFS + WALL PLACEMENT (identical to SEM7_PR1_Results.py)
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


def bfs_path(walls, start, end):
    if start in walls or end in walls:
        return []
    visited = {start: None}
    q = collections.deque([start])
    while q:
        x, y = q.popleft()
        if (x, y) == end:
            path = []
            node = end
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
    terrain_walls       = terrain_walls | new_t
    breakable_positions = breakable_positions | new_b
    return terrain_walls, breakable_positions, len(new_t), len(new_b)


# ═══════════════════════════════════════════════════════════════
#  RENDERING
# ═══════════════════════════════════════════════════════════════

def build_map_image(terrain_walls, breakable_positions):
    """Build a (H, W, 4) RGBA image array for the map."""
    img = np.ones((MAP_H, MAP_W, 4), dtype=float)
    # Dark blue-grey background (matches image)
    bg = np.array([0.14, 0.15, 0.18, 1.0])
    img[:] = bg

    # Terrain walls (permanent) - Dark Green
    wall_color = np.array([0.0, 0.35, 0.0, 1.0])
    for (wx, wy) in terrain_walls:
        img[wy, wx] = wall_color

    # Breakable walls - Slightly lighter/olive green
    brk_color = np.array([0.2, 0.45, 0.1, 1.0])
    for (wx, wy) in breakable_positions:
        img[wy, wx] = brk_color

    # Resources - Yellow/Gold
    res_color = np.array([0.85, 0.75, 0.15, 1.0])
    for rx, ry in RES_P0 + RES_P1:
        img[ry, rx] = res_color

    # Bases
    img[BASE_P0[1], BASE_P0[0]] = np.array([0.15, 0.45, 0.85, 1.0]) # Blue
    img[BASE_P1[1], BASE_P1[0]] = np.array([0.85, 0.25, 0.15, 1.0]) # Red

    # Workers
    img[WORKER_P0[1], WORKER_P0[0]] = np.array([0.35, 0.65, 0.95, 1.0]) # Light Blue
    img[WORKER_P1[1], WORKER_P1[0]] = np.array([0.95, 0.45, 0.35, 1.0]) # Light Red

    return img


def show_map_for_seconds(terrain_walls, breakable_positions, gen, seconds=10):
    """Display a single map full-screen for `seconds` seconds."""

    n_terrain  = len(terrain_walls)
    n_breakable = len(breakable_positions)
    total = n_terrain + n_breakable

    phase = "Early" if gen <= 7 else ("Mid" if gen <= 14 else "Late")

    all_walls = terrain_walls | breakable_positions
    reachable = bfs_reachable(all_walls, BASE_P0, BASE_P1)
    path = bfs_path(all_walls, BASE_P0, BASE_P1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='#1E2128')

    img = build_map_image(terrain_walls, breakable_positions)
    ax.imshow(img, interpolation='nearest', origin='upper')

    # Add grid lines
    ax.set_xticks(np.arange(-.5, MAP_W, 1), minor=True)
    ax.set_yticks(np.arange(-.5, MAP_H, 1), minor=True)
    ax.grid(which="minor", color="#5A6370", linestyle='-', linewidth=0.8)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Draw BFS path
    # if path:
    #     px = [p[0] for p in path]
    #     py = [p[1] for p in path]
    #     ax.plot(px, py, color='#00e5ff', linewidth=1.5, alpha=0.8, zorder=5)

    # Title
    bfs_status = "REACHABLE" if reachable else "BLOCKED"
    ax.set_title(
        f"Generation {gen} [{phase} Phase]\n"
        f"{total} walls ({n_terrain} terrain + {n_breakable} breakable) | "
        f"BFS: {bfs_status} | Path length: {len(path)}",
        fontsize=14, fontweight='bold', pad=12, fontfamily='serif', color='white'
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#5A6370')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#005900', edgecolor='none',
                       label='Terrain wall (permanent)'),
        mpatches.Patch(facecolor='#33731A', edgecolor='none',
                       label='Breakable wall'),
        mpatches.Patch(facecolor='#2673D9', edgecolor='none',
                       label='Player 0 base'),
        mpatches.Patch(facecolor='#D94026', edgecolor='none',
                       label='Player 1 base'),
        mpatches.Patch(facecolor='#D9BF26', edgecolor='none',
                       label='Resource'),
    ]
    legend = ax.legend(handles=legend_elements, loc='lower left', fontsize=9,
                       frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('#2A2D35')
    legend.get_frame().set_edgecolor('#5A6370')
    for text in legend.get_texts():
        text.set_color('white')

    # Countdown text at the bottom
    fig.text(0.5, 0.01,
             f"Window will close in {seconds} seconds -- take your screenshot now!",
             ha='center', fontsize=11, color='#ff6666', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    # Show non-blocking, then wait
    plt.ion()
    plt.show()
    plt.pause(seconds)
    plt.close(fig)
    plt.ioff()

    print(f"  [CLOSED] Generation {gen} window closed after {seconds}s")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    random.seed(SEED)

    target_gens = [10, 20, 30]
    snapshots = {}

    terrain_walls     = set()
    breakable_positions = set()

    print(f"\n  Simulating {TOTAL_GENERATIONS} generations of wall placement...")
    print(f"  Will display maps at generations: {target_gens}")
    print(f"  Each map shown for 10 seconds for screenshots.\n")

    for gen in range(1, TOTAL_GENERATIONS + 1):
        terrain_walls, breakable_positions, n_t, n_b = \
            add_walls_for_generation(gen, terrain_walls, breakable_positions)
        total = len(terrain_walls) + len(breakable_positions)
        reachable = bfs_reachable(terrain_walls | breakable_positions,
                                  BASE_P0, BASE_P1)
        status = "OK" if reachable else "BLOCKED"
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | "
              f"+{n_t} terrain +{n_b} breakable -> {total:3d} total | "
              f"BFS {status}")

        if gen in target_gens:
            snapshots[gen] = (terrain_walls.copy(), breakable_positions.copy())

    # Now display each snapshot for 10 seconds
    print(f"\n{'='*60}")
    print(f"  DISPLAYING MAPS - Take screenshots!")
    print(f"{'='*60}\n")

    for gen in target_gens:
        tw, bp = snapshots[gen]
        total = len(tw) + len(bp)
        print(f"  >> Showing Generation {gen} ({total} walls) for 10 seconds...")
        show_map_for_seconds(tw, bp, gen, seconds=10)

    print(f"\n  All done! You should have 3 screenshots now.")


if __name__ == "__main__":
    main()
