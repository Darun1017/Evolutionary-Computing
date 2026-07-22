"""
Generate FAST 32x32 microRTS maps with HIGH blockade percentage.
The key to fast A* pathfinding isn't fewer walls, but *convex* walls. 
Labyrinths and U-shapes cause A* to explode. Thick, solid blocks have a high wall count
but are mathematically simple for the pathfinder to navigate around.
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
# MAP 1: "Thick Blocks" (~25% Blockade)
# 4 massive 8x8 solid blocks. Huge wall count, but A* routes instantly.
# =====================================================================
def make_thick_blocks():
    g = empty()
    centers = [(9, 9), (9, 22), (22, 9), (22, 22)]
    for cx, cy in centers:
        for dx in range(-4, 4):
            for dy in range(-4, 4):
                g[cy+dy][cx+dx] = 1
    g = make_symmetric(g)
    
    units = std_units([
        {"type": "Resource", "player": -1, "x": 15, "y": 15, "resources": 15},
        {"type": "Resource", "player": -1, "x": 16, "y": 16, "resources": 15},
    ])
    return g, units

# =====================================================================
# MAP 2: "Checkered Fortress" (~25% Blockade)
# Sixteen 4x4 solid blocks evenly spaced. Lots of blockades, but all paths
# are straight 4-tile wide avenues. Units never get stuck.
# =====================================================================
def make_checkered():
    g = empty()
    for cx in [6, 12, 19, 25]:
        for cy in [6, 12, 19, 25]:
            for dx in range(-1, 3):
                for dy in range(-1, 3):
                    g[cy+dy][cx+dx] = 1
    g = make_symmetric(g)
    
    units = std_units([
        {"type": "Resource", "player": -1, "x": 15, "y": 15, "resources": 15},
        {"type": "Resource", "player": -1, "x": 16, "y": 16, "resources": 15},
    ])
    return g, units

# =====================================================================
# MAP 3: "Massive Center Wall" (~20% Blockade)
# One giant 10x20 wall in the middle, forcing units to go top or bottom.
# Very high wall count, practically zero pathfinding complexity.
# =====================================================================
def make_center_wall():
    g = empty()
    for x in range(11, 21):
        for y in range(6, 26):
            g[y][x] = 1
    g = make_symmetric(g)
    
    units = std_units([
        {"type": "Resource", "player": -1, "x": 15, "y": 3, "resources": 15},
        {"type": "Resource", "player": -1, "x": 16, "y": 28, "resources": 15},
    ])
    return g, units

# =====================================================================
# MAIN GENERATOR
# =====================================================================
if __name__ == "__main__":
    maps = {
        "thick_blocks_32x32.xml": make_thick_blocks(),
        "checkered_fortress_32x32.xml": make_checkered(),
        "massive_center_32x32.xml": make_center_wall(),
    }
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Simulation", "microrts", "maps")
    os.makedirs(out_dir, exist_ok=True)
    
    for filename, (grid, units) in maps.items():
        xml_str = create_map_xml(grid, units)
        path = os.path.join(out_dir, filename)
        with open(path, "w") as f:
            f.write(xml_str)
        
        wall_count = sum(sum(row) for row in grid)
        pct = (wall_count / 1024) * 100
        print(f"Generated {filename} ({wall_count} walls = {pct:.1f}% blockade)")
