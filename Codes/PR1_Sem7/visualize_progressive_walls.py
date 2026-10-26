"""
visualize_progressive_walls.py
==============================
Generate a publication-quality figure showing the progressive addition of
blockades on a 48×48 microRTS map across 22 evolutionary generations.

This reuses the SAME wall-placement and BFS-validation logic from
SEM7_PR1_Results.py but does NOT run any evolutionary algorithm — it only
adds walls each generation and renders the resulting map states.

OUTPUT
  • progressive_walls.png   — tiled snapshot of the map at key generations
  • progressive_walls.pdf   — same figure in vector format (for papers)
"""

import math, random, collections, os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import numpy as np

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION (mirrors SEM7_PR1_Results.py)
# ═══════════════════════════════════════════════════════════════

MAP_W, MAP_H = 48, 48
TOTAL_GENERATIONS = 22

WALLS_PER_GEN_EARLY = 6    # gen 1–7
WALLS_PER_GEN_MID   = 10   # gen 8–14
WALLS_PER_GEN_LATE  = 14   # gen 15–22
BREAKABLE_RATIO     = 0.4

# Reserved tiles (bases, workers, resources + buffers)
RESERVED = set()
for bx, by in [(0,0),(0,1),(1,1),(2,2),(1,2),(2,1),(3,2),(2,3),(3,3)]:
    RESERVED.add((bx, by))
for bx, by in [(MAP_W-1,MAP_H-1),(MAP_W-1,MAP_H-2),(MAP_W-2,MAP_H-2),
               (MAP_W-3,MAP_H-3),(MAP_W-2,MAP_H-3),(MAP_W-3,MAP_H-2),
               (MAP_W-4,MAP_H-3),(MAP_W-3,MAP_H-4),(MAP_W-4,MAP_H-4)]:
    RESERVED.add((bx, by))

# Unit positions (for rendering)
BASE_P0    = (2, 2)
BASE_P1    = (MAP_W - 3, MAP_H - 3)
WORKER_P0  = (1, 1)
WORKER_P1  = (MAP_W - 2, MAP_H - 2)
RES_P0     = [(0, 0), (0, 1)]
RES_P1     = [(MAP_W - 1, MAP_H - 1), (MAP_W - 1, MAP_H - 2)]

SEED = 42  # Reproducible layout

# ═══════════════════════════════════════════════════════════════
#  WALL PLACEMENT + BFS (identical to SEM7_PR1_Results.py)
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
    """Return the shortest path between start and end, or [] if unreachable."""
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

def render_map(ax, terrain_walls, breakable_positions, gen, total_gen,
               show_path=True, new_terrain=None, new_breakable=None):
    """Render one generation's map state on a matplotlib Axes."""

    # Build the grid image  (H × W × 4 RGBA)
    img = np.ones((MAP_H, MAP_W, 4), dtype=float)
    # Dark blue-grey background
    bg = np.array([0.14, 0.15, 0.18, 1.0])
    img[:] = bg

    # ── terrain walls (solid) ─────────────────────────────────
    wall_color = np.array([0.0, 0.35, 0.0, 1.0])        # dark green
    for (wx, wy) in terrain_walls:
        if new_terrain and (wx, wy) in new_terrain:
            continue  # drawn separately
        img[wy, wx] = wall_color

    # ── breakable walls ───────────────────────────────────────
    brk_color = np.array([0.2, 0.45, 0.1, 1.0])         # olive green
    for (wx, wy) in breakable_positions:
        if new_breakable and (wx, wy) in new_breakable:
            continue
        img[wy, wx] = brk_color

    # ── newly added walls this gen (highlighted) ──────────────
    new_t_color = np.array([0.90, 0.20, 0.20, 1.0])       # red highlight
    new_b_color = np.array([1.00, 0.55, 0.10, 1.0])       # orange highlight
    if new_terrain:
        for (wx, wy) in new_terrain:
            img[wy, wx] = new_t_color
    if new_breakable:
        for (wx, wy) in new_breakable:
            img[wy, wx] = new_b_color

    # ── reserved / unit tiles ─────────────────────────────────
    # Resources
    res_color = np.array([0.85, 0.75, 0.15, 1.0])         # yellow
    for rx, ry in RES_P0 + RES_P1:
        img[ry, rx] = res_color
    # Bases
    base_p0_color = np.array([0.15, 0.45, 0.85, 1.0])     # blue
    base_p1_color = np.array([0.85, 0.25, 0.15, 1.0])     # red
    img[BASE_P0[1], BASE_P0[0]] = base_p0_color
    img[BASE_P1[1], BASE_P1[0]] = base_p1_color
    # Workers
    worker_p0_color = np.array([0.35, 0.65, 0.95, 1.0])   # light blue
    worker_p1_color = np.array([0.95, 0.45, 0.35, 1.0])   # light red
    img[WORKER_P0[1], WORKER_P0[0]] = worker_p0_color
    img[WORKER_P1[1], WORKER_P1[0]] = worker_p1_color

    ax.imshow(img, interpolation='nearest', origin='upper')

    # Add grid lines
    ax.set_xticks(np.arange(-.5, MAP_W, 1), minor=True)
    ax.set_yticks(np.arange(-.5, MAP_H, 1), minor=True)
    ax.grid(which="minor", color="#5A6370", linestyle='-', linewidth=0.5)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # ── BFS shortest path overlay ─────────────────────────────
    if show_path:
        all_walls = terrain_walls | breakable_positions
        path = bfs_path(all_walls, BASE_P0, BASE_P1)
        if path:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            ax.plot(px, py, color='#00e5ff', linewidth=1.5, alpha=0.8, zorder=5)

    # ── title ─────────────────────────────────────────────────
    total_walls = len(terrain_walls) + len(breakable_positions)
    if gen == 0:
        title = f"Gen 0 (initial)\n0 walls"
    else:
        phase = "Early" if gen <= 7 else ("Mid" if gen <= 14 else "Late")
        title = f"Gen {gen}/{total_gen} [{phase}]\n{total_walls} walls"
    ax.set_title(title, fontsize=9, fontweight='bold', pad=4, color='white')
    
    # Thin border
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('#5A6370')


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    random.seed(SEED)

    # ── Simulate all generations ──────────────────────────────
    terrain_walls     = set()
    breakable_positions = set()

    # Store snapshots for selected generations
    # Show: 0, 1, 4, 7, 10, 14, 18, 22  (8 panels → 2×4 grid)
    snapshot_gens = [0, 1, 4, 7, 10, 14, 18, 22]
    snapshots = {}

    # Gen 0 = initial clean map
    snapshots[0] = {
        'terrain':   set(),
        'breakable': set(),
        'new_t':     set(),
        'new_b':     set(),
    }

    for gen in range(1, TOTAL_GENERATIONS + 1):
        old_terrain   = terrain_walls.copy()
        old_breakable = breakable_positions.copy()

        terrain_walls, breakable_positions, n_t, n_b = \
            add_walls_for_generation(gen, terrain_walls, breakable_positions)

        new_t = terrain_walls - old_terrain
        new_b = breakable_positions - old_breakable

        total = len(terrain_walls) + len(breakable_positions)
        reachable = bfs_reachable(terrain_walls | breakable_positions,
                                  BASE_P0, BASE_P1)
        status = "OK" if reachable else "BLOCKED"
        print(f"  Gen {gen:2d}/{TOTAL_GENERATIONS} | "
              f"+{n_t} terrain +{n_b} breakable -> {total:3d} total | "
              f"BFS {status}")

        if gen in snapshot_gens:
            snapshots[gen] = {
                'terrain':   terrain_walls.copy(),
                'breakable': breakable_positions.copy(),
                'new_t':     new_t.copy(),
                'new_b':     new_b.copy(),
            }

    # ── Plot ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 9.5), facecolor='#1E2128')
    fig.suptitle("Progressive Blockade Addition on 48×48 Dynamic Map",
                 fontsize=16, fontweight='bold', y=0.97,
                 fontfamily='serif', color='white')
    fig.text(0.5, 0.935,
             f"Early (Gen 1–7): {WALLS_PER_GEN_EARLY}/gen  |  "
             f"Mid (Gen 8–14): {WALLS_PER_GEN_MID}/gen  |  "
             f"Late (Gen 15–22): {WALLS_PER_GEN_LATE}/gen  |  "
             f"Breakable ratio: {BREAKABLE_RATIO:.0%}",
             ha='center', fontsize=10, color='#A0AAB5',
             fontfamily='serif')

    gs = GridSpec(2, 4, figure=fig, hspace=0.30, wspace=0.15,
                  left=0.04, right=0.92, top=0.88, bottom=0.06)

    for idx, gen in enumerate(snapshot_gens):
        row, col = divmod(idx, 4)
        ax = fig.add_subplot(gs[row, col])
        # Darken subplot background just in case
        ax.set_facecolor('#1E2128')
        snap = snapshots[gen]
        render_map(ax, snap['terrain'], snap['breakable'], gen,
                   TOTAL_GENERATIONS, show_path=True,
                   new_terrain=snap['new_t'] if gen > 0 else None,
                   new_breakable=snap['new_b'] if gen > 0 else None)

    # ── Legend ────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor='#005900', edgecolor='none',
                       label='Terrain wall (permanent)'),
        mpatches.Patch(facecolor='#33731A', edgecolor='none',
                       label='Breakable wall'),
        mpatches.Patch(facecolor='#E63333', edgecolor='none',
                       label='New terrain (this gen)'),
        mpatches.Patch(facecolor='#FF8C1A', edgecolor='none',
                       label='New breakable (this gen)'),
        mpatches.Patch(facecolor='#2673D9', edgecolor='none',
                       label='Player 0 base'),
        mpatches.Patch(facecolor='#D94026', edgecolor='none',
                       label='Player 1 base'),
        mpatches.Patch(facecolor='#D9BF26', edgecolor='none',
                       label='Resource'),
        mpatches.Patch(facecolor='#00e5ff', edgecolor='none',
                       label='BFS shortest path'),
    ]
    legend = fig.legend(handles=legend_elements, loc='center right',
               bbox_to_anchor=(0.995, 0.50), fontsize=9,
               frameon=True, fancybox=True, shadow=True,
               title='Legend', title_fontsize=10)
    legend.get_frame().set_facecolor('#2A2D35')
    legend.get_frame().set_edgecolor('#5A6370')
    plt.setp(legend.get_title(), color='white')
    for text in legend.get_texts():
        text.set_color('white')

    # ── Wall count summary bar ────────────────────────────────
    # Small inset axes showing cumulative wall count over gens
    ax_bar = fig.add_axes([0.935, 0.08, 0.055, 0.32])  # [left, bottom, w, h]

    # Re-simulate for the bar chart data
    random.seed(SEED)
    tw, bp = set(), set()
    gen_nums = list(range(TOTAL_GENERATIONS + 1))
    wall_counts_t = [0]
    wall_counts_b = [0]
    for g in range(1, TOTAL_GENERATIONS + 1):
        tw, bp, _, _ = add_walls_for_generation(g, tw, bp)
        wall_counts_t.append(len(tw))
        wall_counts_b.append(len(bp))

    ax_bar.fill_between(gen_nums, wall_counts_t, color='#383842', alpha=0.8,
                        label='Terrain')
    ax_bar.fill_between(gen_nums,
                        [t+b for t, b in zip(wall_counts_t, wall_counts_b)],
                        wall_counts_t, color='#8C6B47', alpha=0.8,
                        label='Breakable')
    ax_bar.set_xlabel('Gen', fontsize=7)
    ax_bar.set_ylabel('Walls', fontsize=7)
    ax_bar.tick_params(labelsize=6)
    ax_bar.set_title('Cumulative\nWalls', fontsize=7, fontweight='bold')
    # Phase boundaries
    ax_bar.axvline(x=7, color='#aaa', linewidth=0.5, linestyle='--')
    ax_bar.axvline(x=14, color='#aaa', linewidth=0.5, linestyle='--')

    # ── Save ──────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(out_dir, "progressive_walls.png")
    pdf_path = os.path.join(out_dir, "progressive_walls.pdf")

    fig.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='#1E2128')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='#1E2128')
    print(f"\n  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")
    plt.close(fig)
    print("  Done!")


if __name__ == "__main__":
    main()
