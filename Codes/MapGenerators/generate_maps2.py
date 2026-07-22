"""
Generate 4 SPARSE 32x32 microRTS maps.
These maps are designed to have around 10-20% blockades using simple shapes 
(like pillars or short lines) so that the A* pathfinding runs extremely fast (like the chambers map).
"""

import os

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

# =====================================================================
# MAP 1: "Sparse Pillars"
# Just scattered 2x2 blocks acting as simple obstacles. Very fast A*.
# =====================================================================
def make_sparse_pillars():
    g = empty()
    centers = [(6, 15), (15, 6), (10, 10), (10, 21)]
    for cx, cy in centers:
        for dx in (0, 1):
            for dy in (0, 1):
                g[cy+dy][cx+dx] = 1
    g = make_symmetric(g)
    
    units = std_units([
        {"type": "Resource", "player": -1, "x": 15, "y": 15, "resources": 15},
        {"type": "Resource", "player": -1, "x": 16, "y": 16, "resources": 15},
    ])
    return g, units

# =====================================================================
# MAP 2: "Simple Cross"
# A simple '+' shape in the very center, but lots of open space around it.
# =====================================================================
def make_simple_cross():
    g = empty()
    for i in range(12, 20):
        g[15][i] = 1
        g[16][i] = 1
        g[i][15] = 1
        g[i][16] = 1
    g = make_symmetric(g)
    
    units = std_units([
        {"type": "Resource", "player": -1, "x": 8, "y": 8, "resources": 15},
        {"type": "Resource", "player": -1, "x": 23, "y": 23, "resources": 15},
    ])
    return g, units

# =====================================================================
# MAP 3: "Wide Lanes"
# Two horizontal lines dividing the map, but with massive gaps.
# =====================================================================
def make_wide_lanes():
    g = empty()
    for x in range(0, 10):
        g[10][x] = 1
        g[11][x] = 1
    for x in range(22, 32):
        g[10][x] = 1
        g[11][x] = 1
    g = make_symmetric(g)
    
    units = std_units([
        {"type": "Resource", "player": -1, "x": 15, "y": 5, "resources": 15},
        {"type": "Resource", "player": -1, "x": 16, "y": 26, "resources": 15},
    ])
    return g, units

# =====================================================================
# MAP 4: "Corner Shields"
# Walls only exist near the bases to act as defensive shields, center is open.
# =====================================================================
def make_corner_shields():
    g = empty()
    for i in range(4, 9):
        g[4][i] = 1
        g[i][4] = 1
    g = make_symmetric(g)
    
    units = std_units([
        {"type": "Resource", "player": -1, "x": 4, "y": 27, "resources": 15},
        {"type": "Resource", "player": -1, "x": 27, "y": 4, "resources": 15},
    ])
    return g, units

# =====================================================================
# MAIN GENERATOR
# =====================================================================
if __name__ == "__main__":
    maps = {
        "sparse_pillars_32x32.xml": make_sparse_pillars(),
        "simple_cross_32x32.xml": make_simple_cross(),
        "wide_lanes_32x32.xml": make_wide_lanes(),
        "corner_shields_32x32.xml": make_corner_shields(),
    }
    
    # Save to the microRTS maps folder directly
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Simulation", "microrts", "maps")
    os.makedirs(out_dir, exist_ok=True)
    
    for filename, (grid, units) in maps.items():
        xml_str = create_map_xml(grid, units)
        path = os.path.join(out_dir, filename)
        with open(path, "w") as f:
            f.write(xml_str)
        print(f"Generated {filename} ({sum(sum(row) for row in grid)} walls / 1024 tiles = {sum(sum(row) for row in grid)/1024*100:.1f}%)")
