import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET

def visualize_map(xml_path, out_path, title):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    width = int(root.attrib['width'])
    height = int(root.attrib['height'])
    
    terrain_str = root.find('terrain').text.strip()
    grid = [[int(terrain_str[y * width + x]) for x in range(width)] for y in range(height)]
    
    units = []
    for unit in root.find('units'):
        u_type = unit.attrib['type']
        u_player = int(unit.attrib.get('player', -1))
        u_x = int(unit.attrib['x'])
        u_y = int(unit.attrib['y'])
        units.append({'type': u_type, 'player': u_player, 'x': u_x, 'y': u_y})

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
    pct = wall_count / (width * height) * 100
    ax.set_title(f"{title}\n({wall_count} walls = {pct:.1f}% blockade)", fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    maps_dir = r"../Simulation/microrts/maps"
    out_dir = r"C:\Users\Pradeep S\.gemini\antigravity\brain\fc6d16c5-4899-4f4b-9fa6-c2edaa8b519d"
    
    map1_path = os.path.join(maps_dir, "corridor_ts_32x32.xml")
    out1_path = os.path.join(out_dir, "corridor_ts_visual.png")
    if os.path.exists(map1_path):
        visualize_map(map1_path, out1_path, "Corridor Ts 32x32")
        print(f"Saved {out1_path}")
    else:
        print(f"Map not found: {map1_path}")
        
    map2_path = os.path.join(maps_dir, "open_flanks_32x32.xml")
    out2_path = os.path.join(out_dir, "open_flanks_visual.png")
    if os.path.exists(map2_path):
        visualize_map(map2_path, out2_path, "Open Flanks 32x32")
        print(f"Saved {out2_path}")
    else:
        print(f"Map not found: {map2_path}")
