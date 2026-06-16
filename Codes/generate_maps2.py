"""
Generate 4 MORE symmetric 32x32 microRTS maps with HEAVY blockades.
These maps have significantly more walls than the first batch.

FIXED: Resources are now placed in accessible open areas outside
wall enclosures, ensuring workers can always reach them.
"""

import os
import numpy as np

def grid_to_terrain(grid):
    return ''.join(''.join(str(c) for c in row) for row in grid)

def make_symmetric(grid):
    n = len(grid)
    for y in range(n):
        for x in range(n):
            my, mx = n - 1 - y, n - 1 - x
            if y < my or (y == my and x < mx):
                grid[my][mx] = grid[y][x]
    return grid

def create_map_xml(grid, units):
    terrain = grid_to_terrain(grid)
    xml = '<rts.PhysicalGameState width="32" height="32">\n'
    xml += f'  <terrain>{terrain}</terrain>\n'
    xml += '  <players>\n'
    xml += '    <rts.Player ID="0" resources="5">\n    </rts.Player>\n'
    xml += '    <rts.Player ID="1" resources="5">\n    </rts.Player>\n'
    xml += '  </players>\n  <units>\n'
    uid = 100
    for u in units:
        xml += f'    <rts.units.Unit type="{u["type"]}" ID="{uid}" player="{u["player"]}" x="{u["x"]}" y="{u["y"]}" resources="{u.get("resources", 0)}" hitpoints="{u.get("hp", 1)}" >\n    </rts.units.Unit>\n'
        uid += 1
    xml += '  </units>\n</rts.PhysicalGameState>\n'
    return xml

def empty():
    return [[0]*32 for _ in range(32)]

def std_units(extra=[]):
    return [
        {"type": "Base", "player": 0, "x": 2, "y": 2, "hp": 10},
        {"type": "Worker", "player": 0, "x": 1, "y": 2, "hp": 1},
        {"type": "Worker", "player": 0, "x": 2, "y": 1, "hp": 1},
        {"type": "Resource", "player": -1, "x": 0, "y": 0, "resources": 25},
        {"type": "Resource", "player": -1, "x": 1, "y": 0, "resources": 25},
        {"type": "Resource", "player": -1, "x": 0, "y": 1, "resources": 25},
        {"type": "Base", "player": 1, "x": 29, "y": 29, "hp": 10},
        {"type": "Worker", "player": 1, "x": 30, "y": 29, "hp": 1},
        {"type": "Worker", "player": 1, "x": 29, "y": 30, "hp": 1},
        {"type": "Resource", "player": -1, "x": 31, "y": 31, "resources": 25},
        {"type": "Resource", "player": -1, "x": 30, "y": 31, "resources": 25},
        {"type": "Resource", "player": -1, "x": 31, "y": 30, "resources": 25},
    ] + extra


def is_accessible_from_base(grid, start_x, start_y, target_x, target_y):
    """BFS flood fill to verify a target tile is reachable from a start tile."""
    n = len(grid)
    visited = set()
    queue = [(start_x, start_y)]
    visited.add((start_x, start_y))
    while queue:
        cx, cy = queue.pop(0)
        if cx == target_x and cy == target_y:
            return True
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in visited and grid[ny][nx] == 0:
                visited.add((nx, ny))
                queue.append((nx, ny))
    return False


# =====================================================================
# MAP 5: "Labyrinth"
# Dense maze with narrow corridors. Resources placed in the CORRIDORS
# between the rings, not inside the innermost ring.
# =====================================================================
def make_labyrinth():
    g = empty()
    
    # Outer ring wall
    for i in range(4, 28):
        g[4][i] = 1; g[27][i] = 1
        g[i][4] = 1; g[i][27] = 1
    # Gaps in outer ring (wide enough for units)
    g[4][8] = 0; g[4][9] = 0
    g[8][4] = 0; g[9][4] = 0
    
    # Inner ring wall
    for i in range(8, 24):
        g[8][i] = 1; g[23][i] = 1
        g[i][8] = 1; g[i][23] = 1
    # Gaps in inner ring
    g[8][14] = 0; g[8][15] = 0
    g[14][8] = 0; g[15][8] = 0
    
    # Core ring
    for i in range(12, 20):
        g[12][i] = 1; g[19][i] = 1
        g[i][12] = 1; g[i][19] = 1
    # Gaps in core ring
    g[12][15] = 0; g[12][16] = 0
    g[15][12] = 0; g[16][12] = 0
    
    # Connecting corridor walls between rings
    for y in range(5, 8):
        g[y][14] = 1; g[y][17] = 1
    g[6][14] = 0; g[6][17] = 0
    
    for x in range(5, 8):
        g[14][x] = 1; g[17][x] = 1
    g[14][6] = 0; g[17][6] = 0
    
    g = make_symmetric(g)
    
    # Resources placed in the OPEN CORRIDORS between rings (accessible!)
    # Between outer wall and inner wall (corridor area)
    units = std_units([
        {"type": "Resource", "player": -1, "x": 6, "y": 6, "resources": 15},
        {"type": "Resource", "player": -1, "x": 25, "y": 25, "resources": 15},
        # Between inner ring and core ring (reachable via gaps)
        {"type": "Resource", "player": -1, "x": 10, "y": 10, "resources": 10},
        {"type": "Resource", "player": -1, "x": 21, "y": 21, "resources": 10},
    ])
    return g, units

# =====================================================================
# MAP 6: "Chokepoint Gauntlet"
# Multiple parallel horizontal walls with staggered gaps force
# a zig-zag path. Resources placed in open areas between walls.
# =====================================================================
def make_gauntlet():
    g = empty()
    
    # Row 1: wall across with gap on left
    for x in range(4, 28):
        g[5][x] = 1
    g[5][6] = 0; g[5][7] = 0
    
    # Row 2: wall across with gap on right
    for x in range(4, 28):
        g[9][x] = 1
    g[9][24] = 0; g[9][25] = 0
    
    # Row 3: wall across with gap in center
    for x in range(4, 28):
        g[13][x] = 1
    g[13][15] = 0; g[13][16] = 0
    
    # Vertical walls on sides creating chambers
    for y in range(5, 14):
        g[y][4] = 1; g[y][27] = 1
    g[7][4] = 0; g[11][27] = 0
    
    # Extra vertical walls mid-field
    for y in range(6, 9):
        g[y][12] = 1; g[y][19] = 1
    for y in range(10, 13):
        g[y][10] = 1; g[y][21] = 1
    
    g = make_symmetric(g)
    
    # Resources placed in OPEN lanes between horizontal walls
    units = std_units([
        {"type": "Resource", "player": -1, "x": 8, "y": 7, "resources": 15},
        {"type": "Resource", "player": -1, "x": 23, "y": 24, "resources": 15},
        {"type": "Resource", "player": -1, "x": 15, "y": 11, "resources": 10},
        {"type": "Resource", "player": -1, "x": 16, "y": 20, "resources": 10},
    ])
    return g, units

# =====================================================================
# MAP 7: "Fortress Islands"
# Heavy wall enclosures around each base with only 2 exits,
# plus a walled center. Resources placed OUTSIDE the fortress
# walls in the open battlefield.
# =====================================================================
def make_fortress_islands():
    g = empty()
    
    # Top-left fortress (thick walls)
    for x in range(0, 12):
        g[6][x] = 1; g[7][x] = 1
    for y in range(0, 8):
        g[y][11] = 1; g[y][12] = 1
    # Exit south
    g[6][3] = 0; g[7][3] = 0
    g[6][4] = 0; g[7][4] = 0
    # Exit east
    g[3][11] = 0; g[3][12] = 0
    g[4][11] = 0; g[4][12] = 0
    
    # Center arena walls
    for x in range(11, 21):
        g[11][x] = 1; g[20][x] = 1
    for y in range(11, 21):
        g[y][11] = 1; g[y][20] = 1
    # Arena entrances (4 gaps)
    g[11][15] = 0; g[11][16] = 0
    g[15][11] = 0; g[16][11] = 0
    g[20][15] = 0; g[20][16] = 0
    g[15][20] = 0; g[16][20] = 0
    
    # Connecting walls from fortress to arena
    for x in range(6, 11):
        g[9][x] = 1
    g[9][8] = 0; g[9][9] = 0
    for y in range(6, 11):
        g[y][9] = 1
    g[8][9] = 0; g[9][9] = 0
    
    g = make_symmetric(g)
    
    # Resources placed in OPEN BATTLEFIELD outside fortress walls
    units = std_units([
        {"type": "Resource", "player": -1, "x": 8, "y": 9, "resources": 15},
        {"type": "Resource", "player": -1, "x": 23, "y": 22, "resources": 15},
        {"type": "Resource", "player": -1, "x": 5, "y": 14, "resources": 10},
        {"type": "Resource", "player": -1, "x": 26, "y": 17, "resources": 10},
    ])
    return g, units

# =====================================================================
# MAP 8: "Siege Walls"
# Three concentric defensive wall layers around each base,
# each with a single narrow opening. Resources placed in the
# open space between the outermost layer and mid-field.
# =====================================================================
def make_siege_walls():
    g = empty()
    
    # Layer 1 (outermost) - L-shaped wall near top-left
    for x in range(0, 16):
        g[14][x] = 1
    for y in range(0, 15):
        g[y][15] = 1
    # Single gap
    g[14][5] = 0; g[14][6] = 0
    g[5][15] = 0; g[6][15] = 0
    
    # Layer 2 (middle)
    for x in range(0, 12):
        g[10][x] = 1
    for y in range(0, 11):
        g[y][11] = 1
    # Single gap
    g[10][8] = 0; g[10][9] = 0
    g[8][11] = 0; g[9][11] = 0
    
    # Layer 3 (innermost)
    for x in range(0, 8):
        g[6][x] = 1
    for y in range(0, 7):
        g[y][7] = 1
    # Single gap
    g[6][4] = 0; g[6][5] = 0
    g[4][7] = 0; g[5][7] = 0
    
    g = make_symmetric(g)
    
    # Resources placed BETWEEN wall layers (accessible via gaps)
    units = std_units([
        {"type": "Resource", "player": -1, "x": 4, "y": 8, "resources": 20},
        {"type": "Resource", "player": -1, "x": 27, "y": 23, "resources": 20},
        {"type": "Resource", "player": -1, "x": 9, "y": 12, "resources": 10},
        {"type": "Resource", "player": -1, "x": 22, "y": 19, "resources": 10},
    ])
    return g, units


# =====================================================================
# Generate all maps + verify accessibility
# =====================================================================
maps_dir = r"C:\Users\Pradeep S\Desktop\Semester VI\Project Phase - 1\Simulation\microrts\maps"

configs = [
    ("labyrinth_32x32.xml", make_labyrinth, "Labyrinth"),
    ("chokepoint_gauntlet_32x32.xml", make_gauntlet, "Chokepoint Gauntlet"),
    ("fortress_islands_32x32.xml", make_fortress_islands, "Fortress Islands"),
    ("siege_walls_32x32.xml", make_siege_walls, "Siege Walls"),
]

for filename, make_fn, desc in configs:
    grid, units = make_fn()
    xml = create_map_xml(grid, units)
    filepath = os.path.join(maps_dir, filename)
    with open(filepath, 'w') as f:
        f.write(xml)
    
    wall_count = sum(c for row in grid for c in row)
    total = 32*32
    print(f"\n{'='*50}")
    print(f"MAP: {desc} ({filename})")
    print(f"Wall density: {wall_count}/{total} ({wall_count/total*100:.1f}%)")
    print(f"{'='*50}")
    for row in grid:
        print(''.join(['#' if c == 1 else '.' for c in row]))
    
    # Verify accessibility of all resources from Player 0 base (2,2)
    print(f"\n  Accessibility Check (from P0 base at 2,2):")
    all_ok = True
    for u in units:
        if u["type"] == "Resource":
            reachable = is_accessible_from_base(grid, 2, 2, u["x"], u["y"])
            status = "OK" if reachable else "BLOCKED!"
            if not reachable:
                all_ok = False
            print(f"    Resource at ({u['x']},{u['y']}): {status}")
    if all_ok:
        print(f"  All resources accessible!")
    else:
        print(f"  WARNING: Some resources are blocked!")
    
    print(f"Saved to: {filepath}")

print("\n\nAll 4 heavy-blockade maps created successfully!")
