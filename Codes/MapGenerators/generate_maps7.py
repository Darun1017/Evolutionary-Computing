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

def base_units():
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
        
        # Central contested resources INSIDE the chamber
        {"type": "Resource", "player": -1, "x": 15, "y": 15, "resources": 20},
        {"type": "Resource", "player": -1, "x": 16, "y": 16, "resources": 20},
        
        # Flank resources OUTSIDE the chamber
        {"type": "Resource", "player": -1, "x": 4, "y": 27, "resources": 15},
        {"type": "Resource", "player": -1, "x": 27, "y": 4, "resources": 15},
    ]

# =====================================================================
# MAP: "Arena Chambers" (~20% walls)
# Inverts the original Chambers map. Instead of 4 corner rooms, there
# is ONE massive central walled arena with 4 entrances.
# =====================================================================
def make_arena_chambers():
    g = empty()
    
    # Draw the central arena box (massive, 3 tiles thick)
    for t in range(0, 3):
        # Top and bottom
        for i in range(8, 24):
            g[8+t][i] = 1; g[23-t][i] = 1
        # Left and right
        for i in range(8, 24):
            g[i][8+t] = 1; g[i][23-t] = 1
            
    # Create 4-tile wide entrances in the middle of each wall (cut all the way through)
    for i in range(14, 18):
        for t in range(0, 3):
            g[8+t][i] = 0; g[23-t][i] = 0
            g[i][8+t] = 0; g[i][23-t] = 0
        
    # Add four 3-thick L-shaped "corner shields" protecting the bases
    for t in range(0, 2):
        for i in range(2, 6): # Starts at 2 to leave a massive 2-tile gap at the map edges!
            g[5+t][i] = 1; g[i][5+t] = 1
        
    g = make_symmetric(g)
    return g, base_units()


if __name__ == "__main__":
    grid, units = make_arena_chambers()
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Simulation", "microrts", "maps")
    os.makedirs(out_dir, exist_ok=True)
    
    filename = "arena_chambers_32x32.xml"
    xml_str = create_map_xml(grid, units)
    path = os.path.join(out_dir, filename)
    
    with open(path, "w") as f:
        f.write(xml_str)
        
    wall_count = sum(sum(row) for row in grid)
    pct = (wall_count / 1024) * 100
    print(f"Generated {filename} ({wall_count} walls = {pct:.1f}% blockade)")
