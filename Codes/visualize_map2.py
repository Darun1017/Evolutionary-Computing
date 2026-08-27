import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import xml.etree.ElementTree as ET
import os

maps_dir = r"C:\Users\Pradeep S\Desktop\Semester VI\Project Phase - 1\Code\maps"
out_dir = r"C:\Users\Pradeep S\.gemini\antigravity\brain\fc6d16c5-4899-4f4b-9fa6-c2edaa8b519d"

fname = "map2.xml"
fpath = os.path.join(maps_dir, fname)

fig, ax = plt.subplots(figsize=(10, 10))

if not os.path.exists(fpath):
    print(f"Error: {fpath} not found.")
else:
    tree = ET.parse(fpath)
    root = tree.getroot()
    
    width = int(root.attrib['width'])
    height = int(root.attrib['height'])
    terrain_str = root.find('terrain').text
    
    grid = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            grid[y][x] = int(terrain_str[y * width + x])
    
    color_map = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            if grid[y][x] == 1:
                color_map[y][x] = [0.20, 0.20, 0.25]
            else:
                color_map[y][x] = [0.85, 0.88, 0.78]
    
    ax.imshow(color_map, interpolation='nearest')
    
    units_elem = root.find('units')
    for unit in units_elem:
        utype = unit.attrib['type']
        player = int(unit.attrib['player'])
        ux = int(unit.attrib['x'])
        uy = int(unit.attrib['y'])
        
        if utype == 'Base':
            color = '#2196F3' if player == 0 else '#F44336'
            marker = 's'; size = 300
        elif utype == 'Worker':
            color = '#64B5F6' if player == 0 else '#E57373'
            marker = 'o'; size = 150
        elif utype == 'Resource':
            color = '#FFD700'
            marker = 'D'; size = 200
        else:
            continue
        ax.scatter(ux, uy, c=color, marker=marker, s=size, edgecolors='black', linewidths=1.5, zorder=5)
    
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which='minor', color='#cccccc', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    
    wall_count = int(np.sum(grid))
    pct = wall_count / (32*32) * 100
    ax.set_title(f"map2.xml\n({wall_count} walls = {pct:.1f}% blockade)", fontsize=18)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, "map2_visual.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved to: {out_path}")
