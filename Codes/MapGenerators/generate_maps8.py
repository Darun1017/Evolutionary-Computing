"""
microRTS Map Generator - L & H Chambers
- 2-layer outer walls (corners open)
- Long L-shaped obstacles
- Central H structure with resources
- Fully symmetric
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
        xml += f'    <rts.units.Unit type="{u["type"]}" ID="{uid}" player="{u["player"]}" '
        xml += f'x="{u["x"]}" y="{u["y"]}" resources="{u.get("resources", 0)}" '
        xml += f'hitpoints="{u.get("hp", 1)}" >\n    </rts.units.Unit>\n'
        uid += 1
    xml += '  </units>\n</rts.PhysicalGameState>\n'
    return xml


def empty_grid():
    return [[0] * 32 for _ in range(32)]


def standard_units(extra=None):
    units = [
        # Player 0 (bottom-left)
        {"type": "Base", "player": 0, "x": 3, "y": 3, "hp": 10},
        {"type": "Worker", "player": 0, "x": 2, "y": 4, "hp": 1},
        {"type": "Worker", "player": 0, "x": 4, "y": 2, "hp": 1},
        
        # Player 1 (top-right)
        {"type": "Base", "player": 1, "x": 28, "y": 28, "hp": 10},
        {"type": "Worker", "player": 1, "x": 27, "y": 27, "hp": 1},
        {"type": "Worker", "player": 1, "x": 29, "y": 29, "hp": 1},
        
        # Corner starting resources
        {"type": "Resource", "player": -1, "x": 0, "y": 0, "resources": 30},
        {"type": "Resource", "player": -1, "x": 1, "y": 1, "resources": 30},
        {"type": "Resource", "player": -1, "x": 31, "y": 31, "resources": 30},
        {"type": "Resource", "player": -1, "x": 30, "y": 30, "resources": 30},
    ]
    if extra:
        units.extend(extra)
    return units


# =====================================================================
# NEW MAP: L & H Structure
# =====================================================================
def make_l_h_map():
    g = empty_grid()
    
    # ====================== OUTER BORDER (2 tiles thick) ======================
    # Top and Bottom
    for x in range(32):
        g[0][x] = 1
        g[1][x] = 1
        g[30][x] = 1
        g[31][x] = 1
    
    # Left and Right
    for y in range(32):
        g[y][0] = 1
        g[y][1] = 1
        g[y][30] = 1
        g[y][31] = 1
    
    # Open the corners (4x4 passable areas)
    for y in range(4):
        for x in range(4):
            g[y][x] = 0
            g[y][31-x] = 0
            g[31-y][x] = 0
            g[31-y][31-x] = 0
    
    # ====================== LONG L-SHAPED OBSTACLES ======================
    # Top-Left L (Attached to Left Wall)
    for x in range(2, 10):
        g[8][x] = 1
        g[9][x] = 1
    for y in range(8, 16):
        g[y][8] = 1
        g[y][9] = 1
        
    # Top-Right L (Attached to Right Wall)
    for x in range(22, 30):
        g[8][x] = 1
        g[9][x] = 1
    for y in range(8, 16):
        g[y][22] = 1
        g[y][23] = 1
    # ====================== CENTRAL H-SHAPED STRUCTURE ======================
    # Left vertical bar of H
    for y in range(12, 20):
        g[y][10] = 1
        g[y][11] = 1
    
    # Right vertical bar of H
    for y in range(12, 20):
        g[y][20] = 1
        g[y][21] = 1
    
    # Horizontal bar of H (middle)
    for x in range(10, 22):
        g[15][x] = 1
        g[16][x] = 1
    
    g = make_symmetric(g)
    
    # ====================== RESOURCES ======================
    extra_units = [
        # Resources behind L obstacles (need to go around)
        {"type": "Resource", "player": -1, "x": 20, "y": 6, "resources": 25},
        {"type": "Resource", "player": -1, "x": 6, "y": 20, "resources": 25},
        
        # Resources inside / near the central H
        {"type": "Resource", "player": -1, "x": 15, "y": 13, "resources": 30},
        {"type": "Resource", "player": -1, "x": 16, "y": 18, "resources": 30},
        {"type": "Resource", "player": -1, "x": 12, "y": 15, "resources": 20},
        {"type": "Resource", "player": -1, "x": 19, "y": 15, "resources": 20},
    ]
    
    return g, standard_units(extra_units)


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    maps = {
        "l_h_chambers_32x32.xml": make_l_h_map(),
    }
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "maps")
    os.makedirs(out_dir, exist_ok=True)
    
    for filename, (grid, units) in maps.items():
        xml_str = create_map_xml(grid, units)
        path = os.path.join(out_dir, filename)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(xml_str)
        
        wall_count = sum(sum(row) for row in grid)
        print(f"* Generated {filename}")
        print(f"   Blockades: {wall_count}/1024 tiles ({wall_count/1024*100:.1f}%)")
    
    print("\nL & H map generated successfully!")