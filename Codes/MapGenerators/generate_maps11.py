import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import xml.etree.ElementTree as ET

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

def make_arena_chambers_3thick():
    g = empty_grid()

    # 1) THREE-LAYER THICK BORDER
    for t in range(3):
        for x in range(32):
            g[t][x] = 1
            g[31-t][x] = 1
        for y in range(32):
            g[y][t] = 1
            g[y][31-t] = 1

    # 2) CHAMBER WALLS
    # We want a chamber from x=3..9, y=3..9
    # Wall at x=10 and y=10
    for i in range(3, 11):
        g[10][i] = 1
        g[i][10] = 1
        
    # 3) DOORS
    # Create 2-tile wide doors to exit the chamber into the arena
    g[10][6] = 0; g[10][7] = 0
    g[6][10] = 0; g[7][10] = 0

    # 4) CENTRAL ARENA OBSTACLE (Just a small 2x2 in the exact center to block direct line of sight)
    g[15][15] = 1
    g[15][16] = 1
    g[16][15] = 1
    g[16][16] = 1

    # Mirror to all 4 quadrants
    g = make_4way_symmetric(g)

    # 5) UNITS
    units = [
        {"type": "Base",   "player": 0, "x": 3,  "y": 3,  "hp": 10},
        {"type": "Worker", "player": 0, "x": 3,  "y": 4,  "hp": 1},
        {"type": "Worker", "player": 0, "x": 4,  "y": 3,  "hp": 1},
        {"type": "Base",   "player": 1, "x": 28, "y": 28, "hp": 10},
        {"type": "Worker", "player": 1, "x": 28, "y": 27, "hp": 1},
        {"type": "Worker", "player": 1, "x": 27, "y": 28, "hp": 1},
        
        # Resources safely inside the chamber
        {"type": "Resource", "player": -1, "x": 8,  "y": 4,  "resources": 20},
        {"type": "Resource", "player": -1, "x": 4,  "y": 8,  "resources": 20},
        {"type": "Resource", "player": -1, "x": 27, "y": 23, "resources": 20},
        {"type": "Resource", "player": -1, "x": 23, "y": 27, "resources": 20},
        
        # Highly contested arena resources
        {"type": "Resource", "player": -1, "x": 12, "y": 12, "resources": 30},
        {"type": "Resource", "player": -1, "x": 19, "y": 19, "resources": 30},
        {"type": "Resource", "player": -1, "x": 12, "y": 19, "resources": 30},
        {"type": "Resource", "player": -1, "x": 19, "y": 12, "resources": 30},
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
    fname = "arena_chambers_32x32.xml"
    
    # Save directly to Simulation/microrts/maps
    sim_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Simulation", "microrts", "maps")
    code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps")
    
    os.makedirs(sim_dir, exist_ok=True)
    os.makedirs(code_dir, exist_ok=True)
    
    g, u = make_arena_chambers_3thick()
    
    connected = verify_connectivity(g, 3, 3, 28, 28)
    print(f"Connectivity check: {'PASSED' if connected else 'FAILED'}")
    
    xml_str = create_map_xml(g, u)
    
    with open(os.path.join(sim_dir, fname), "w", encoding="utf-8") as f: f.write(xml_str)
    with open(os.path.join(code_dir, fname), "w", encoding="utf-8") as f: f.write(xml_str)
    
    print(f"Saved XML to {fname}")
    
    # Visualize
    vis_dir = r"C:\Users\Pradeep S\.gemini\antigravity\brain\fc6d16c5-4899-4f4b-9fa6-c2edaa8b519d"
    vis_path = os.path.join(vis_dir, "arena_chambers_3thick_visual.png")
    visualize_grid(g, u, "Arena Chambers (3-Thick)", vis_path)
    print(f"Saved Visualization to {vis_path}")
