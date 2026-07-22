"""
Generate 4 symmetric 32x32 microRTS maps with blockades.
All maps are diagonally symmetric (rotate 180 degrees) so both players face identical terrain.
"""

def grid_to_terrain(grid):
    """Convert a 2D grid list to a flat terrain string."""
    return ''.join(''.join(str(c) for c in row) for row in grid)

def make_symmetric(grid):
    """Make the grid 180-degree rotationally symmetric.
    We design the top-left half, then mirror it to bottom-right."""
    n = len(grid)
    for y in range(n):
        for x in range(n):
            # Mirror: (y, x) -> (n-1-y, n-1-x)
            my, mx = n - 1 - y, n - 1 - x
            if y < my or (y == my and x < mx):
                grid[my][mx] = grid[y][x]
    return grid

def create_map_xml(name, grid, resources_p0, resources_p1, units):
    """Create a complete microRTS XML map file."""
    terrain = grid_to_terrain(grid)
    
    xml = f'<rts.PhysicalGameState width="32" height="32">\n'
    xml += f'  <terrain>{terrain}</terrain>\n'
    xml += f'  <players>\n'
    xml += f'    <rts.Player ID="0" resources="5">\n'
    xml += f'    </rts.Player>\n'
    xml += f'    <rts.Player ID="1" resources="5">\n'
    xml += f'    </rts.Player>\n'
    xml += f'  </players>\n'
    xml += f'  <units>\n'
    
    uid = 100
    for u in units:
        xml += f'    <rts.units.Unit type="{u["type"]}" ID="{uid}" player="{u["player"]}" x="{u["x"]}" y="{u["y"]}" resources="{u.get("resources", 0)}" hitpoints="{u.get("hp", 1)}" >\n'
        xml += f'    </rts.units.Unit>\n'
        uid += 1
    
    xml += f'  </units>\n'
    xml += f'</rts.PhysicalGameState>\n'
    return xml

def empty_grid():
    return [[0]*32 for _ in range(32)]

def print_grid(grid):
    for row in grid:
        print(''.join(['#' if c == 1 else '.' for c in row]))

# =====================================================================
# MAP 1: "Fortress Corridors"
# Two fortress-like enclosures connected by narrow corridors.
# Walls create natural chokepoints and defensive positions.
# =====================================================================
def make_map1():
    g = empty_grid()
    
    # Horizontal walls creating corridors on the top half
    # Top fortress walls
    for x in range(6, 26):
        g[5][x] = 1
        g[10][x] = 1
    
    # Gaps in walls for corridors
    g[5][10] = 0; g[5][11] = 0
    g[5][20] = 0; g[5][21] = 0
    g[10][14] = 0; g[10][15] = 0; g[10][16] = 0; g[10][17] = 0
    
    # Vertical side walls
    for y in range(5, 11):
        g[y][6] = 1
        g[y][25] = 1
    
    # Center vertical blockade
    for y in range(12, 20):
        g[y][15] = 1
        g[y][16] = 1
    # Gap in center blockade
    g[15][15] = 0; g[15][16] = 0
    g[16][15] = 0; g[16][16] = 0
    
    g = make_symmetric(g)
    
    units = [
        # Player 0
        {"type": "Base", "player": 0, "x": 2, "y": 2, "hp": 10},
        {"type": "Worker", "player": 0, "x": 1, "y": 2, "hp": 1},
        {"type": "Worker", "player": 0, "x": 2, "y": 1, "hp": 1},
        {"type": "Resource", "player": -1, "x": 0, "y": 0, "resources": 25},
        {"type": "Resource", "player": -1, "x": 1, "y": 0, "resources": 25},
        {"type": "Resource", "player": -1, "x": 0, "y": 1, "resources": 25},
        # Player 1
        {"type": "Base", "player": 1, "x": 29, "y": 29, "hp": 10},
        {"type": "Worker", "player": 1, "x": 30, "y": 29, "hp": 1},
        {"type": "Worker", "player": 1, "x": 29, "y": 30, "hp": 1},
        {"type": "Resource", "player": -1, "x": 31, "y": 31, "resources": 25},
        {"type": "Resource", "player": -1, "x": 30, "y": 31, "resources": 25},
        {"type": "Resource", "player": -1, "x": 31, "y": 30, "resources": 25},
        # Contested center resources
        {"type": "Resource", "player": -1, "x": 13, "y": 15, "resources": 15},
        {"type": "Resource", "player": -1, "x": 18, "y": 16, "resources": 15},
    ]
    
    return g, units

# =====================================================================
# MAP 2: "Diamond Maze"
# Diamond-shaped walls radiating from center, creating a maze-like
# structure with multiple approach angles.
# =====================================================================
def make_map2():
    g = empty_grid()
    
    # Diamond walls from center
    center = 15
    # Inner diamond (size 6)
    for i in range(7):
        g[center - 6 + i][center - i] = 1
        g[center - 6 + i][center + 1 + i] = 1
    for i in range(7):
        g[center + 1 + i][center - 6 + i] = 1
        g[center + 1 + i][center + 7 - i] = 1
    
    # Openings in diamond (N, S, E, W)
    g[center - 6][center] = 0  # N
    g[center - 6][center + 1] = 0
    g[center + 7][center] = 0  # S (will be set by symmetry)
    g[center][center - 6] = 0  # W
    g[center + 1][center - 6] = 0
    g[center][center + 7] = 0  # E
    g[center + 1][center + 7] = 0
    
    # Outer blockade walls near corners
    for i in range(8):
        g[3][4 + i] = 1
        g[4 + i][3] = 1
    g[3][7] = 0; g[3][8] = 0  # gaps
    g[7][3] = 0; g[8][3] = 0
    
    g = make_symmetric(g)
    
    units = [
        {"type": "Base", "player": 0, "x": 1, "y": 1, "hp": 10},
        {"type": "Worker", "player": 0, "x": 2, "y": 1, "hp": 1},
        {"type": "Worker", "player": 0, "x": 1, "y": 2, "hp": 1},
        {"type": "Resource", "player": -1, "x": 0, "y": 0, "resources": 25},
        {"type": "Resource", "player": -1, "x": 1, "y": 0, "resources": 25},
        {"type": "Resource", "player": -1, "x": 0, "y": 1, "resources": 25},
        
        {"type": "Base", "player": 1, "x": 30, "y": 30, "hp": 10},
        {"type": "Worker", "player": 1, "x": 29, "y": 30, "hp": 1},
        {"type": "Worker", "player": 1, "x": 30, "y": 29, "hp": 1},
        {"type": "Resource", "player": -1, "x": 31, "y": 31, "resources": 25},
        {"type": "Resource", "player": -1, "x": 30, "y": 31, "resources": 25},
        {"type": "Resource", "player": -1, "x": 31, "y": 30, "resources": 25},
        
        # Center contested resources
        {"type": "Resource", "player": -1, "x": 15, "y": 15, "resources": 10},
        {"type": "Resource", "player": -1, "x": 16, "y": 16, "resources": 10},
    ]
    
    return g, units

# =====================================================================
# MAP 3: "Cross Barricade"  
# A large cross-shaped wall divides the map into 4 quadrants.
# Players must navigate through gaps to attack.
# =====================================================================
def make_map3():
    g = empty_grid()
    
    # Horizontal bar of the cross
    for x in range(8, 24):
        g[15][x] = 1
        g[16][x] = 1
    
    # Vertical bar of the cross
    for y in range(8, 24):
        g[y][15] = 1
        g[y][16] = 1
    
    # Openings in horizontal bar
    g[15][10] = 0; g[16][10] = 0
    g[15][21] = 0; g[16][21] = 0
    
    # Openings in vertical bar
    g[10][15] = 0; g[10][16] = 0
    g[21][15] = 0; g[21][16] = 0
    
    # Small corner fortifications
    for x in range(4, 8):
        g[4][x] = 1
    for y in range(4, 8):
        g[y][4] = 1
    g[4][6] = 0  # gap
    g[6][4] = 0  # gap
    
    g = make_symmetric(g)
    
    units = [
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
        
        # Resources near the center cross intersections
        {"type": "Resource", "player": -1, "x": 12, "y": 12, "resources": 15},
        {"type": "Resource", "player": -1, "x": 19, "y": 19, "resources": 15},
        {"type": "Resource", "player": -1, "x": 12, "y": 19, "resources": 10},
        {"type": "Resource", "player": -1, "x": 19, "y": 12, "resources": 10},
    ]
    
    return g, units

# =====================================================================
# MAP 4: "Twin Trenches"
# Two parallel diagonal trenches (walls) create a no-man's-land
# in the center. Flanking routes exist on both sides.
# =====================================================================
def make_map4():
    g = empty_grid()
    
    # Diagonal trench 1 (top-left to center-right)
    for i in range(18):
        y = 6 + i
        x = 6 + i
        if 0 <= y < 32 and 0 <= x < 32:
            g[y][x] = 1
        if 0 <= y < 32 and x + 1 < 32:
            g[y][x + 1] = 1
    
    # Gaps in diagonal trench
    for i in [4, 5, 10, 11]:
        y = 6 + i
        x = 6 + i
        if 0 <= y < 32 and 0 <= x < 32:
            g[y][x] = 0
        if 0 <= y < 32 and x + 1 < 32:
            g[y][x + 1] = 0
    
    # Horizontal blockade near top-left base
    for x in range(3, 10):
        g[4][x] = 1
    g[4][6] = 0; g[4][7] = 0  # gap
    
    # Vertical blockade near top-left base
    for y in range(3, 10):
        g[y][4] = 1
    g[6][4] = 0; g[7][4] = 0  # gap
    
    g = make_symmetric(g)
    
    units = [
        {"type": "Base", "player": 0, "x": 1, "y": 1, "hp": 10},
        {"type": "Worker", "player": 0, "x": 2, "y": 1, "hp": 1},
        {"type": "Worker", "player": 0, "x": 1, "y": 2, "hp": 1},
        {"type": "Resource", "player": -1, "x": 0, "y": 0, "resources": 25},
        {"type": "Resource", "player": -1, "x": 1, "y": 0, "resources": 25},
        {"type": "Resource", "player": -1, "x": 0, "y": 1, "resources": 25},
        
        {"type": "Base", "player": 1, "x": 30, "y": 30, "hp": 10},
        {"type": "Worker", "player": 1, "x": 29, "y": 30, "hp": 1},
        {"type": "Worker", "player": 1, "x": 30, "y": 29, "hp": 1},
        {"type": "Resource", "player": -1, "x": 31, "y": 31, "resources": 25},
        {"type": "Resource", "player": -1, "x": 30, "y": 31, "resources": 25},
        {"type": "Resource", "player": -1, "x": 31, "y": 30, "resources": 25},
        
        # Flanking resources
        {"type": "Resource", "player": -1, "x": 0, "y": 15, "resources": 15},
        {"type": "Resource", "player": -1, "x": 31, "y": 16, "resources": 15},
        {"type": "Resource", "player": -1, "x": 15, "y": 0, "resources": 15},
        {"type": "Resource", "player": -1, "x": 16, "y": 31, "resources": 15},
    ]
    
    return g, units


# =====================================================================
# Generate all maps
# =====================================================================
import os

maps_dir = r"C:\Users\Pradeep S\Desktop\Semester VI\Project Phase - 1\Simulation\microrts\maps"

map_configs = [
    ("fortress_corridors_32x32.xml", make_map1, "Fortress Corridors"),
    ("diamond_maze_32x32.xml", make_map2, "Diamond Maze"),
    ("cross_barricade_32x32.xml", make_map3, "Cross Barricade"),
    ("twin_trenches_32x32.xml", make_map4, "Twin Trenches"),
]

for filename, make_fn, desc in map_configs:
    grid, units = make_fn()
    xml = create_map_xml(desc, grid, 5, 5, units)
    filepath = os.path.join(maps_dir, filename)
    with open(filepath, 'w') as f:
        f.write(xml)
    
    print(f"\n{'='*50}")
    print(f"MAP: {desc} ({filename})")
    print(f"{'='*50}")
    print_grid(grid)
    print(f"\nSaved to: {filepath}")

print("\n\nAll 4 maps created successfully!")
