"""
Generate 4 complex 32x32 microRTS maps with 2 BASES PER PLAYER.
These maps mimic the "Labyrinth" and "Gauntlet" aesthetic but with 
WIDE corridors (3-4 tiles) and NO enclosed spaces, so bots can always 
move freely and A* pathfinding remains fast.
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

def two_base_units(extra=[]):
    """
    Player 0: Bases at (2,2) and (2, 29)
    Player 1: Bases at (29,29) and (29, 2)
    Each base gets 2 workers and 3 nearby resources.
    """
    return [
        # --- PLAYER 0 ---
        # Base 1 (Top Left)
        {"type": "Base", "player": 0, "x": 2, "y": 2, "hp": 10},
        {"type": "Worker", "player": 0, "x": 1, "y": 2, "hp": 1},
        {"type": "Worker", "player": 0, "x": 2, "y": 1, "hp": 1},
        {"type": "Resource", "player": -1, "x": 0, "y": 0, "resources": 25},
        {"type": "Resource", "player": -1, "x": 1, "y": 0, "resources": 25},
        {"type": "Resource", "player": -1, "x": 0, "y": 1, "resources": 25},
        
        # Base 2 (Bottom Left)
        {"type": "Base", "player": 0, "x": 2, "y": 29, "hp": 10},
        {"type": "Worker", "player": 0, "x": 1, "y": 29, "hp": 1},
        {"type": "Worker", "player": 0, "x": 2, "y": 30, "hp": 1},
        {"type": "Resource", "player": -1, "x": 0, "y": 31, "resources": 25},
        {"type": "Resource", "player": -1, "x": 1, "y": 31, "resources": 25},
        {"type": "Resource", "player": -1, "x": 0, "y": 30, "resources": 25},

        # --- PLAYER 1 ---
        # Base 1 (Bottom Right)
        {"type": "Base", "player": 1, "x": 29, "y": 29, "hp": 10},
        {"type": "Worker", "player": 1, "x": 30, "y": 29, "hp": 1},
        {"type": "Worker", "player": 1, "x": 29, "y": 30, "hp": 1},
        {"type": "Resource", "player": -1, "x": 31, "y": 31, "resources": 25},
        {"type": "Resource", "player": -1, "x": 30, "y": 31, "resources": 25},
        {"type": "Resource", "player": -1, "x": 31, "y": 30, "resources": 25},
        
        # Base 2 (Top Right)
        {"type": "Base", "player": 1, "x": 29, "y": 2, "hp": 10},
        {"type": "Worker", "player": 1, "x": 30, "y": 2, "hp": 1},
        {"type": "Worker", "player": 1, "x": 29, "y": 1, "hp": 1},
        {"type": "Resource", "player": -1, "x": 31, "y": 0, "resources": 25},
        {"type": "Resource", "player": -1, "x": 30, "y": 0, "resources": 25},
        {"type": "Resource", "player": -1, "x": 31, "y": 1, "resources": 25},
    ] + extra

# =====================================================================
# MAP 1: "Two Bases Labyrinth" (~17% walls)
# Thick concentric rings with massive 4-tile wide openings. No traps.
# =====================================================================
def make_two_bases_labyrinth():
    g = empty()
    
    # Outer ring
    for i in range(4, 28):
        g[4][i] = 1; g[5][i] = 1
        g[26][i] = 1; g[27][i] = 1
        g[i][4] = 1; g[i][5] = 1
        g[i][26] = 1; g[i][27] = 1
        
    # Massive gaps in outer ring
    for i in range(12, 20):
        g[4][i] = 0; g[5][i] = 0; g[26][i] = 0; g[27][i] = 0
        g[i][4] = 0; g[i][5] = 0; g[i][26] = 0; g[i][27] = 0
        
    # Inner ring
    for i in range(10, 22):
        g[10][i] = 1; g[11][i] = 1
        g[20][i] = 1; g[21][i] = 1
        g[i][10] = 1; g[i][11] = 1
        g[i][20] = 1; g[i][21] = 1
        
    # Gaps in inner ring at the corners
    for i in range(10, 14):
        for j in range(10, 14):
            g[i][j] = 0; g[31-i][31-j] = 0
            g[i][31-j] = 0; g[31-i][j] = 0
    
    g = make_symmetric(g)
    extra_res = [
        {"type": "Resource", "player": -1, "x": 15, "y": 15, "resources": 15},
        {"type": "Resource", "player": -1, "x": 16, "y": 16, "resources": 15},
    ]
    return g, two_base_units(extra_res)

# =====================================================================
# MAP 2: "Two Bases Gauntlet" (~17% walls)
# Horizontal zig-zag lines. Corridors are a massive 4-5 tiles wide!
# =====================================================================
def make_two_bases_gauntlet():
    g = empty()
    
    # Line 1
    for x in range(0, 24):
        g[6][x] = 1; g[7][x] = 1
        g[8][x] = 1
        
    # Line 2
    for x in range(8, 32):
        g[14][x] = 1; g[15][x] = 1
        g[16][x] = 1
        
    # Line 3
    for x in range(0, 24):
        g[23][x] = 1; g[24][x] = 1
        g[25][x] = 1

    g = make_symmetric(g)
    extra_res = [
        {"type": "Resource", "player": -1, "x": 15, "y": 11, "resources": 15},
        {"type": "Resource", "player": -1, "x": 16, "y": 20, "resources": 15},
    ]
    return g, two_base_units(extra_res)

# =====================================================================
# MAP 3: "Two Bases Fortress" (~12% walls)
# Each of the 4 bases has an L-shaped shield wall.
# Wide 5-tile entries, impossible to get trapped.
# =====================================================================
def make_two_bases_fortress():
    g = empty()
    
    # Top-left shield
    for i in range(0, 12):
        g[10][i] = 1; g[11][i] = 1
        g[i][10] = 1; g[i][11] = 1
    # Gap in the corner
    for i in range(8, 12):
        for j in range(8, 12):
            g[i][j] = 0
    for i in range(0, 3):
        for j in range(10, 12):
            g[i][j] = 0; g[j][i] = 0

    g = make_symmetric(g)
    extra_res = [
        {"type": "Resource", "player": -1, "x": 15, "y": 15, "resources": 15},
        {"type": "Resource", "player": -1, "x": 16, "y": 16, "resources": 15},
    ]
    return g, two_base_units(extra_res)

# =====================================================================
# MAP 4: "Two Bases Siege Walls" (~14% walls)
# Multiple detached vertical/horizontal line segments.
# =====================================================================
def make_two_bases_siege():
    g = empty()
    
    # Vertical pillars
    for y in range(4, 12):
        g[y][10] = 1; g[y][11] = 1
        
    for y in range(20, 28):
        g[y][10] = 1; g[y][11] = 1
        
    # Center block
    for x in range(10, 22):
        for y in range(14, 18):
            g[y][x] = 1
            
    g = make_symmetric(g)
    extra_res = [
        {"type": "Resource", "player": -1, "x": 6, "y": 16, "resources": 15},
        {"type": "Resource", "player": -1, "x": 25, "y": 15, "resources": 15},
    ]
    return g, two_base_units(extra_res)


if __name__ == "__main__":
    maps = {
        "two_bases_labyrinth_32x32.xml": make_two_bases_labyrinth(),
        "two_bases_gauntlet_32x32.xml": make_two_bases_gauntlet(),
        "two_bases_fortress_32x32.xml": make_two_bases_fortress(),
        "two_bases_siege_32x32.xml": make_two_bases_siege(),
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
