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

def make_exposed_bases():
    g = empty_grid()

    # 1) FLOATING BARRICADES (Pushed into the middle, leaving corners totally open)
    # Top barricade
    for y in range(6, 8):
        for x in range(10, 22):
            g[y][x] = 1
    # Left barricade
    for y in range(10, 22):
        for x in range(6, 8):
            g[y][x] = 1
            
    # 2) CENTRAL CROSS
    # Vertical bar
    for y in range(10, 22):
        for x in range(14, 18):
            g[y][x] = 1
    # Horizontal bar
    for y in range(14, 18):
        for x in range(10, 22):
            g[y][x] = 1

    # Mirror to all 4 quadrants
    g = make_4way_symmetric(g)

    # 3) UNITS
    units = [
        # Player 0: Top-Left Corner
        {"type": "Base",   "player": 0, "x": 2,  "y": 2,  "hp": 10},
        {"type": "Worker", "player": 0, "x": 3,  "y": 3,  "hp": 1},
        
        # Player 1: Bottom-Right Corner
        {"type": "Base",   "player": 1, "x": 29, "y": 29, "hp": 10},
        {"type": "Worker", "player": 1, "x": 28, "y": 28, "hp": 1},
        
        # Immediate Base Resources (Low amount to force early expansion/rushing)
        {"type": "Resource", "player": -1, "x": 4,  "y": 2,  "resources": 10},
        {"type": "Resource", "player": -1, "x": 2,  "y": 4,  "resources": 10},
        {"type": "Resource", "player": -1, "x": 27, "y": 29, "resources": 10},
        {"type": "Resource", "player": -1, "x": 29, "y": 27, "resources": 10},
        
        # Mid-range expansion patches (Encourages leaving the base)
        {"type": "Resource", "player": -1, "x": 6,  "y": 6,  "resources": 20},
        {"type": "Resource", "player": -1, "x": 25, "y": 25, "resources": 20},
        
        # Empty corners
        {"type": "Resource", "player": -1, "x": 25, "y": 6,  "resources": 10}, 
        {"type": "Resource", "player": -1, "x": 6,  "y": 25, "resources": 10}, 
        
        # Contested Center Resources (Nestled in the corners of the cross)
        {"type": "Resource", "player": -1, "x": 12, "y": 12, "resources": 10},
        {"type": "Resource", "player": -1, "x": 19, "y": 12, "resources": 10},
        {"type": "Resource", "player": -1, "x": 12, "y": 19, "resources": 10},
        {"type": "Resource", "player": -1, "x": 19, "y": 19, "resources": 10},
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
    fname = "exposed_bases_32x32.xml"
    
    # Save directly to Simulation/microrts/maps
    sim_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Simulation", "microrts", "maps")
    code_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "maps")
    
    os.makedirs(sim_dir, exist_ok=True)
    os.makedirs(code_dir, exist_ok=True)
    
    g, u = make_exposed_bases()
    
    connected = verify_connectivity(g, 2, 2, 29, 29)
    print(f"Connectivity check: {'PASSED' if connected else 'FAILED'}")
    
    xml_str = create_map_xml(g, u)
    
    with open(os.path.join(sim_dir, fname), "w", encoding="utf-8") as f: f.write(xml_str)
    with open(os.path.join(code_dir, fname), "w", encoding="utf-8") as f: f.write(xml_str)
    
    print(f"Saved XML to {fname}")
    
    # Visualize
    vis_dir = r"C:\Users\Pradeep S\.gemini\antigravity\brain\fc6d16c5-4899-4f4b-9fa6-c2edaa8b519d"
    vis_path = os.path.join(vis_dir, "exposed_bases_visual.png")
    visualize_grid(g, u, "Exposed Bases (Fast Rush Map)", vis_path)
    print(f"Saved Visualization to {vis_path}")
