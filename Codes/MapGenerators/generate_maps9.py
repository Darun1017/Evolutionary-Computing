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

# Helper to enforce 4-way symmetry
def make_4way_symmetric(grid):
    n = len(grid)
    # Mirror Left to Right
    for y in range(n):
        for x in range(n // 2):
            grid[y][n - 1 - x] = grid[y][x]
    # Mirror Top to Bottom
    for x in range(n):
        for y in range(n // 2):
            grid[n - 1 - y][x] = grid[y][x]
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

def verify_connectivity(grid, start_x, start_y, target_x, target_y):
    n = len(grid)
    if grid[start_y][start_x] == 1 or grid[target_y][target_x] == 1:
        return False
    visited = set()
    queue = [(start_x, start_y)]
    visited.add((start_x, start_y))
    while queue:
        x, y = queue.pop(0)
        if x == target_x and y == target_y:
            return True
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < n and 0 <= ny < n and (nx,ny) not in visited and grid[ny][nx] == 0:
                visited.add((nx,ny))
                queue.append((nx,ny))
    return False

def verify_resource_accessible(grid, base_x, base_y, res_x, res_y):
    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0),(0,0)]:
        nx, ny = res_x+dx, res_y+dy
        if 0 <= nx < 32 and 0 <= ny < 32 and grid[ny][nx] == 0:
            if verify_connectivity(grid, base_x, base_y, nx, ny):
                return True
    return False

# =====================================================================
# MAP 1: Fixed L & H Chambers (~15% Walls)
# Fixed the L shape so it doesn't trap the bot!
# =====================================================================
def make_lh_fixed():
    g = empty_grid()

    # Two-tile outer border
    for t in range(2):
        for x in range(32):
            g[t][x] = 1
            g[31-t][x] = 1
        for y in range(32):
            g[y][t] = 1
            g[y][31-t] = 1

    # Open corners for base placement (4x4)
    for y in range(1, 5):
        for x in range(1, 5):
            g[y][x] = 0

    # Simple 4x4 Square Obstacles in each quadrant
    for y in range(5, 9):
        for x in range(5, 9):
            g[y][x] = 1

    # Central H - Full H shape with crossbar
    # Left vertical bar
    for y in range(11, 21):
        g[y][11] = 1
        g[y][12] = 1
    # Right vertical bar
    for y in range(11, 21):
        g[y][19] = 1
        g[y][20] = 1
    # Horizontal crossbar
    for x in range(13, 19):
        g[15][x] = 1
        g[16][x] = 1

    # Mirror to all 4 quadrants!
    g = make_4way_symmetric(g)

    connected = verify_connectivity(g, 2, 2, 29, 29)
    print(f"  [LH_Fixed] P0->P1 connectivity: {'OK' if connected else 'BLOCKED'}")

    wall_count = sum(sum(row) for row in g)
    print(f"  [LH_Fixed] Walls: {wall_count}/1024 ({wall_count/1024*100:.1f}%)")

    extra_units = [
        # Near-base
        {"type": "Resource", "player": -1, "x": 4,  "y": 2,  "resources": 20},
        {"type": "Resource", "player": -1, "x": 27, "y": 29, "resources": 20},
        # Mid-field (require flanking the L)
        {"type": "Resource", "player": -1, "x": 10, "y": 14, "resources": 25},
        {"type": "Resource", "player": -1, "x": 21, "y": 17, "resources": 25},
        # Central (between H bars)
        {"type": "Resource", "player": -1, "x": 15, "y": 11, "resources": 30},
        {"type": "Resource", "player": -1, "x": 16, "y": 20, "resources": 30},
    ]

    for r in extra_units:
        ok = verify_resource_accessible(g, 2, 2, r["x"], r["y"])
        print(f"  [LH_Fixed] Resource ({r['x']},{r['y']}) accessible: {'OK' if ok else 'BLOCKED'}")

    units = [
        {"type": "Base",   "player": 0, "x": 2,  "y": 2,  "hp": 10},
        {"type": "Worker", "player": 0, "x": 2,  "y": 3,  "hp": 1},
        {"type": "Worker", "player": 0, "x": 3,  "y": 2,  "hp": 1},
        {"type": "Base",   "player": 1, "x": 29, "y": 29, "hp": 10},
        {"type": "Worker", "player": 1, "x": 29, "y": 28, "hp": 1},
        {"type": "Worker", "player": 1, "x": 28, "y": 29, "hp": 1},
    ]
    units.extend(extra_units)
    return g, units

# =====================================================================
# MAP 2: Split Arena
# Diagonal corridor with 2 territories. ~12% walls.
# =====================================================================
def make_split_arena():
    g = empty_grid()

    for x in range(32):
        g[0][x] = 1
        g[31][x] = 1
    for y in range(32):
        g[y][0] = 1
        g[y][31] = 1

    for y in range(1, 5):
        for x in range(1, 5):
            g[y][x] = 0

    # Diagonal dividing wall
    for y in range(1, 13):
        wx = 31 - y
        if 1 <= wx <= 30:
            g[y][wx] = 1
            if wx-1 >= 1: g[y][wx-1] = 1

    for y in range(19, 31):
        wx = 31 - y
        if 1 <= wx <= 30:
            g[y][wx] = 1
            if wx-1 >= 1: g[y][wx-1] = 1

    # Small flanking obstacles
    for x in range(22, 28):
        g[6][x] = 1
    for x in range(4, 10):
        g[25][x] = 1

    g = make_symmetric(g)

    connected = verify_connectivity(g, 2, 2, 29, 29)
    print(f"  [SplitArena] P0->P1 connectivity: {'OK' if connected else 'BLOCKED'}")
    
    wall_count = sum(sum(row) for row in g)
    print(f"  [SplitArena] Walls: {wall_count}/1024 ({wall_count/1024*100:.1f}%)")

    extra_units = [
        {"type": "Resource", "player": -1, "x": 5,  "y": 5,  "resources": 20},
        {"type": "Resource", "player": -1, "x": 26, "y": 26, "resources": 20},
        {"type": "Resource", "player": -1, "x": 14, "y": 12, "resources": 30},
        {"type": "Resource", "player": -1, "x": 17, "y": 19, "resources": 30},
        {"type": "Resource", "player": -1, "x": 26, "y": 6,  "resources": 25},
        {"type": "Resource", "player": -1, "x": 5,  "y": 25, "resources": 25},
    ]

    for r in extra_units:
        ok = verify_resource_accessible(g, 2, 2, r["x"], r["y"])
        print(f"  [SplitArena] Resource ({r['x']},{r['y']}) accessible: {'OK' if ok else 'BLOCKED'}")

    units = [
        {"type": "Base",   "player": 0, "x": 2,  "y": 2,  "hp": 10},
        {"type": "Worker", "player": 0, "x": 2,  "y": 3,  "hp": 1},
        {"type": "Worker", "player": 0, "x": 3,  "y": 2,  "hp": 1},
        {"type": "Base",   "player": 1, "x": 29, "y": 29, "hp": 10},
        {"type": "Worker", "player": 1, "x": 29, "y": 28, "hp": 1},
        {"type": "Worker", "player": 1, "x": 28, "y": 29, "hp": 1},
    ]
    units.extend(extra_units)
    return g, units

if __name__ == "__main__":
    # Save to Simulation/microrts/maps/ where the Java server expects them
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Simulation", "microrts", "maps")
    os.makedirs(out_dir, exist_ok=True)

    maps = {
        "map2.xml": make_lh_fixed(),
        "split_arena_32x32.xml": make_split_arena(),
    }

    for filename, (grid, units) in maps.items():
        xml_str = create_map_xml(grid, units)
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(xml_str)
        print(f"* Generated: {filename}\n")
