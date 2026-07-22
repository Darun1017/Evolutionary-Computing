import xml.etree.ElementTree as ET
import os

maps_dir = r"C:\Users\Pradeep S\Desktop\Semester VI\Project Phase - 1\Simulation\microrts\maps"
src_fname = "chokepoint_gauntlet_32x32.xml"

def four_base_units():
    """
    Player 0: Bases at (2,2), (2,11), (2,20), (2, 29)
    Player 1: Bases at (29,29), (29,20), (29,11), (29, 2)
    Each base gets 2 workers and 3 nearby resources.
    """
    units = []
    
    # --- PLAYER 0 BASES ---
    p0_positions = [(2,2), (2,11), (2,20), (2,29)]
    for x, y in p0_positions:
        units.extend([
            {"type": "Base", "player": 0, "x": x, "y": y, "hp": 10},
            {"type": "Worker", "player": 0, "x": x-1, "y": y, "hp": 1},
            {"type": "Worker", "player": 0, "x": x, "y": y-1 if y > 2 else y+1, "hp": 1},
            {"type": "Resource", "player": -1, "x": x-2, "y": y-2, "resources": 25},
            {"type": "Resource", "player": -1, "x": x-1, "y": y-2, "resources": 25},
            {"type": "Resource", "player": -1, "x": x-2, "y": y-1, "resources": 25},
        ])

    # --- PLAYER 1 BASES ---
    p1_positions = [(29,29), (29,20), (29,11), (29,2)]
    for x, y in p1_positions:
        units.extend([
            {"type": "Base", "player": 1, "x": x, "y": y, "hp": 10},
            {"type": "Worker", "player": 1, "x": x+1, "y": y, "hp": 1},
            {"type": "Worker", "player": 1, "x": x, "y": y+1 if y < 29 else y-1, "hp": 1},
            {"type": "Resource", "player": -1, "x": x+2, "y": y+2, "resources": 25},
            {"type": "Resource", "player": -1, "x": x+1, "y": y+2, "resources": 25},
            {"type": "Resource", "player": -1, "x": x+2, "y": y+1, "resources": 25},
        ])
        
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

src_path = os.path.join(maps_dir, src_fname)

if not os.path.exists(src_path):
    print(f"Error: Could not find original map {src_fname}")
else:
    tree = ET.parse(src_path)
    root = tree.getroot()
    terrain_str = root.find('terrain').text
    
    # We will grab any central resources from the original map
    extra_units = []
    units_elem = root.find('units')
    for unit in units_elem:
        utype = unit.attrib['type']
        if utype == 'Resource':
            ux = int(unit.attrib['x'])
            uy = int(unit.attrib['y'])
            # Only keep resources that are securely in the middle (x between 5 and 26)
            if 5 <= ux <= 26:
                extra_units.append({
                    "type": "Resource", "player": -1, "x": ux, "y": uy, 
                    "resources": int(unit.attrib.get('resources', 15))
                })
                
    all_units = four_base_units() + extra_units
    xml_str = create_map_xml(terrain_str, all_units)
    
    new_fname = "4_bases_" + src_fname
    new_path = os.path.join(maps_dir, new_fname)
    with open(new_path, "w") as f:
        f.write(xml_str)
        
    print(f"Successfully generated {new_fname} with 4 BASES PER PLAYER!")
