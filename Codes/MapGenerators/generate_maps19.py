import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def grid_to_terrain(grid):
    return ''.join(''.join(str(c) for c in row) for row in grid)

def make_4way_symmetric(grid):
    n = len(grid)
    for y in range(n):
        for x in range(n // 2):
            grid[y][n - 1 - x] = grid[y][x]
    for x in range(n):
        for y in range(n // 2):
            grid[n - 1 - y][x] = grid[y][x]
    return grid

def empty_grid():
    return [[0] * 32 for _ in range(32)]

def verify_connectivity(grid, start_x, start_y, target_x, target_y):
    n = len(grid)
    if grid[start_y][start_x] == 1 or grid[target_y][target_x] == 1:
        return False
    visited = set([(start_x, start_y)])
    queue = [(start_x, start_y)]
    while queue:
        x, y = queue.pop(0)
        if x == target_x and y == target_y: return True
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < n and 0 <= ny < n and (nx,ny) not in visited and grid[ny][nx] == 0:
                visited.add((nx,ny))
                queue.append((nx,ny))
    return False

def make_map():
    g = empty_grid()

    # ---- CORNER L-SHAPES (same aesthetic as chambers32x32) ----
    # Top-left L
    for y in range(7, 9):
        for x in range(2, 9):
            g[y][x] = 1
    for y in range(2, 9):
        for x in range(7, 9):
            g[y][x] = 1

    # ---- CENTRAL CROSS ----
    for y in range(13, 19):
        for x in range(15, 17):
            g[y][x] = 1
    for y in range(15, 17):
        for x in range(13, 19):
            g[y][x] = 1

    # Mirror to all 4 quadrants
    g = make_4way_symmetric(g)

    # ---- UNITS ----
    # KEY INSIGHT FROM chambers32x32.xml:
    # Resources must be FAR from the base (12+ tiles away)
    # so the economy is crippled, armies stay small, and games end fast.
    units = [
        # Player 0: Top-Left
        {"type": "Base",   "player": 0, "x": 2,  "y": 2,  "hp": 10},
        {"type": "Worker", "player": 0, "x": 1,  "y": 1,  "hp": 1},
        
        # Player 1: Bottom-Right
        {"type": "Base",   "player": 1, "x": 29, "y": 29, "hp": 10},
        {"type": "Worker", "player": 1, "x": 30, "y": 30, "hp": 1},
        
        # ---- RESOURCES: ALL FAR FROM BASES ----
        # Tiny starter (right at the corner, just enough for 1 worker or barracks)
        {"type": "Resource", "player": -1, "x": 0,  "y": 0,  "resources": 10},
        {"type": "Resource", "player": -1, "x": 31, "y": 31, "resources": 10},
        
        # Edge resources (FAR from both bases, along the perimeter)
        {"type": "Resource", "player": -1, "x": 0,  "y": 15, "resources": 20},
        {"type": "Resource", "player": -1, "x": 0,  "y": 16, "resources": 20},
        {"type": "Resource", "player": -1, "x": 31, "y": 15, "resources": 20},
        {"type": "Resource", "player": -1, "x": 31, "y": 16, "resources": 20},
        
        {"type": "Resource", "player": -1, "x": 15, "y": 0,  "resources": 20},
        {"type": "Resource", "player": -1, "x": 16, "y": 0,  "resources": 20},
        {"type": "Resource", "player": -1, "x": 15, "y": 31, "resources": 20},
        {"type": "Resource", "player": -1, "x": 16, "y": 31, "resources": 20},
        
        # Opposite corners (maximum distance from own base)
        {"type": "Resource", "player": -1, "x": 31, "y": 0,  "resources": 20},
        {"type": "Resource", "player": -1, "x": 0,  "y": 31, "resources": 20},
    ]
    
    return g, units

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

def visualize_grid(grid, units, title, out_path):
    height = len(grid)
    width = len(grid[0])
    color_map = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            if grid[y][x] == 1: color_map[y][x] = [0.20, 0.20, 0.25]
            else: color_map[y][x] = [0.85, 0.88, 0.78]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(color_map, interpolation='nearest')
    for u in units:
        utype = u['type']; player = u['player']
        ux = u['x']; uy = u['y']
        if utype == 'Base': color = '#2196F3' if player == 0 else '#F44336'; marker = 's'; size = 300
        elif utype == 'Worker': color = '#64B5F6' if player == 0 else '#E57373'; marker = 'o'; size = 150
        elif utype == 'Resource': color = '#FFD700'; marker = 'D'; size = 200
        else: continue
        ax.scatter(ux, uy, c=color, marker=marker, s=size, edgecolors='black', linewidths=1.5, zorder=5)
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which='minor', color='#cccccc', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    wall_count = sum(sum(row) for row in grid)
    pct = wall_count / (32*32) * 100
    ax.set_title(f"{title}\n({wall_count} walls = {pct:.1f}% blockade)", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    fname = "arena_32x32.xml"
    sim_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Simulation", "microrts", "maps")
    code_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "maps")
    os.makedirs(sim_dir, exist_ok=True)
    os.makedirs(code_dir, exist_ok=True)
    
    g, u = make_map()
    connected = verify_connectivity(g, 2, 2, 29, 29)
    print(f"Connectivity check: {'PASSED' if connected else 'FAILED'}")
    
    total_res = sum(unit.get('resources', 0) for unit in u)
    print(f"Total resources: {total_res}")
    
    xml_str = create_map_xml(g, u)
    with open(os.path.join(sim_dir, fname), "w", encoding="utf-8") as f: f.write(xml_str)
    with open(os.path.join(code_dir, fname), "w", encoding="utf-8") as f: f.write(xml_str)
    print(f"Saved XML to {fname}")
    
    vis_dir = r"C:\Users\Pradeep S\.gemini\antigravity\brain\fc6d16c5-4899-4f4b-9fa6-c2edaa8b519d"
    vis_path = os.path.join(vis_dir, "arena_32x32_visual.png")
    visualize_grid(g, u, "Arena 32x32 (Far Resources)", vis_path)
    print(f"Saved Visualization to {vis_path}")
