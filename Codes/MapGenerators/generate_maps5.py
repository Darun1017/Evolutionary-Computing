import xml.etree.ElementTree as ET
import os

maps_dir = r"C:\Users\Pradeep S\Desktop\Semester VI\Project Phase - 1\Simulation\microrts\maps"

source_maps = [
    "labyrinth_32x32.xml",
    "chokepoint_gauntlet_32x32.xml",
    "fortress_islands_32x32.xml",
    "siege_walls_32x32.xml"
]

def two_base_units():
    """
    Player 0: Bases at (2,2) and (2, 29)
    Player 1: Bases at (29,29) and (29, 2)
    Each base gets 2 workers and 3 nearby resources.
    """
    units = [
        # --- PLAYER 0 ---
        {"type": "Base", "player": 0, "x": 2, "y": 2, "hp": 10},
        {"type": "Worker", "player": 0, "x": 1, "y": 2, "hp": 1},
        {"type": "Worker", "player": 0, "x": 2, "y": 1, "hp": 1},
        {"type": "Resource", "player": -1, "x": 0, "y": 0, "resources": 25},
        {"type": "Resource", "player": -1, "x": 1, "y": 0, "resources": 25},
        {"type": "Resource", "player": -1, "x": 0, "y": 1, "resources": 25},
        
        {"type": "Base", "player": 0, "x": 2, "y": 29, "hp": 10},
        {"type": "Worker", "player": 0, "x": 1, "y": 29, "hp": 1},
        {"type": "Worker", "player": 0, "x": 2, "y": 30, "hp": 1},
        {"type": "Resource", "player": -1, "x": 0, "y": 31, "resources": 25},
        {"type": "Resource", "player": -1, "x": 1, "y": 31, "resources": 25},
        {"type": "Resource", "player": -1, "x": 0, "y": 30, "resources": 25},

        # --- PLAYER 1 ---
        {"type": "Base", "player": 1, "x": 29, "y": 29, "hp": 10},
        {"type": "Worker", "player": 1, "x": 30, "y": 29, "hp": 1},
        {"type": "Worker", "player": 1, "x": 29, "y": 30, "hp": 1},
        {"type": "Resource", "player": -1, "x": 31, "y": 31, "resources": 25},
        {"type": "Resource", "player": -1, "x": 30, "y": 31, "resources": 25},
        {"type": "Resource", "player": -1, "x": 31, "y": 30, "resources": 25},
        
        {"type": "Base", "player": 1, "x": 29, "y": 2, "hp": 10},
        {"type": "Worker", "player": 1, "x": 30, "y": 2, "hp": 1},
        {"type": "Worker", "player": 1, "x": 29, "y": 1, "hp": 1},
        {"type": "Resource", "player": -1, "x": 31, "y": 0, "resources": 25},
        {"type": "Resource", "player": -1, "x": 30, "y": 0, "resources": 25},
        {"type": "Resource", "player": -1, "x": 31, "y": 1, "resources": 25},
    ]
    return units

def create_map_xml(terrain_str, units):
    xml = '<rts.PhysicalGameState width="32" height="32">\n'
    xml += f'  <terrain>{terrain_str}</terrain>\n'
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

for src_fname in source_maps:
    src_path = os.path.join(maps_dir, src_fname)
    
    if not os.path.exists(src_path):
        print(f"Error: Could not find original map {src_fname}")
        continue
        
    tree = ET.parse(src_path)
    root = tree.getroot()
    terrain_str = root.find('terrain').text
    
    # We will grab any resources from the original map that are not in the corners
    extra_units = []
    units_elem = root.find('units')
    for unit in units_elem:
        utype = unit.attrib['type']
        if utype == 'Resource':
            ux = int(unit.attrib['x'])
            uy = int(unit.attrib['y'])
            # Only keep resources that are not in the 4 corners (our 2-base setup handles corner resources)
            if not ((ux <= 3 and uy <= 3) or (ux >= 28 and uy >= 28) or 
                    (ux <= 3 and uy >= 28) or (ux >= 28 and uy <= 3)):
                extra_units.append({
                    "type": "Resource", "player": -1, "x": ux, "y": uy, 
                    "resources": int(unit.attrib.get('resources', 15))
                })
                
    all_units = two_base_units() + extra_units
    xml_str = create_map_xml(terrain_str, all_units)
    
    # Save as 2_bases_...
    new_fname = "2_bases_" + src_fname
    new_path = os.path.join(maps_dir, new_fname)
    with open(new_path, "w") as f:
        f.write(xml_str)
        
    print(f"Successfully cloned {src_fname} into {new_fname} with 2 bases!")
