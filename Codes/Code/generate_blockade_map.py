"""
Generate a SYMMETRIC 32x32 microRTS map with blockade formations.
The map is diagonally symmetric so neither player has an advantage.
Blockades are placed in the center and mid-field to force strategic pathing.
"""

W, H = 32, 32
terrain = [[0]*W for _ in range(H)]

# --- Central fortress wall (symmetric) ---
# Creates a diamond-shaped obstacle in the center that forces flanking
center_walls = [
    # Horizontal bars above and below center
    (13, 14), (14, 14), (15, 14), (16, 14), (17, 14), (18, 14),
    (13, 17), (14, 17), (15, 17), (16, 17), (17, 17), (18, 17),
    # Vertical bars left and right of center
    (12, 15), (12, 16),
    (19, 15), (19, 16),
]

for x, y in center_walls:
    terrain[y][x] = 1

# --- Mid-field blockade pillars (symmetric along diagonal) ---
# Upper-left quadrant pillars
pillars_q1 = [
    (8, 8), (9, 8), (8, 9),      # Pillar 1
    (6, 12), (7, 12), (6, 13),    # Pillar 2
    (12, 6), (12, 7), (13, 6),    # Pillar 3
]
for x, y in pillars_q1:
    terrain[y][x] = 1
    # Mirror to lower-right quadrant (diagonal symmetry)
    mx, my = W-1-x, H-1-y
    terrain[my][mx] = 1

# --- Chokepoint walls near each base (symmetric) ---
# Near player 0 base (top-left): vertical wall with gap
for y in range(4, 9):
    terrain[y][5] = 1
terrain[6][5] = 0  # Gap for passage

# Near player 1 base (bottom-right): mirrored vertical wall with gap
for y in range(23, 28):
    terrain[y][26] = 1
terrain[25][26] = 0  # Gap for passage

# --- Clear spawn areas (corners) ---
for y in range(4):
    for x in range(4):
        terrain[y][x] = 0
        terrain[H-1-y][W-1-x] = 0

# Flatten terrain
terrain_str = ""
for row in terrain:
    terrain_str += "".join(str(c) for c in row)

# --- Build XML ---
xml = f'''<rts.PhysicalGameState width="{W}" height="{H}">
  <terrain>{terrain_str}</terrain>
  <players>
    <rts.Player ID="0" resources="5">
    </rts.Player>
    <rts.Player ID="1" resources="5">
    </rts.Player>
  </players>
  <units>
    <rts.units.Unit type="Resource" ID="100" player="-1" x="0" y="0" resources="25" hitpoints="1" >
    </rts.units.Unit>
    <rts.units.Unit type="Resource" ID="101" player="-1" x="1" y="0" resources="25" hitpoints="1" >
    </rts.units.Unit>
    <rts.units.Unit type="Resource" ID="102" player="-1" x="0" y="1" resources="25" hitpoints="1" >
    </rts.units.Unit>
    <rts.units.Unit type="Resource" ID="103" player="-1" x="31" y="31" resources="25" hitpoints="1" >
    </rts.units.Unit>
    <rts.units.Unit type="Resource" ID="104" player="-1" x="30" y="31" resources="25" hitpoints="1" >
    </rts.units.Unit>
    <rts.units.Unit type="Resource" ID="105" player="-1" x="31" y="30" resources="25" hitpoints="1" >
    </rts.units.Unit>
    <rts.units.Unit type="Base" ID="200" player="0" x="2" y="2" resources="0" hitpoints="10" >
    </rts.units.Unit>
    <rts.units.Unit type="Worker" ID="201" player="0" x="1" y="2" resources="0" hitpoints="1" >
    </rts.units.Unit>
    <rts.units.Unit type="Worker" ID="202" player="0" x="2" y="1" resources="0" hitpoints="1" >
    </rts.units.Unit>
    <rts.units.Unit type="Base" ID="300" player="1" x="29" y="29" resources="0" hitpoints="10" >
    </rts.units.Unit>
    <rts.units.Unit type="Worker" ID="301" player="1" x="30" y="29" resources="0" hitpoints="1" >
    </rts.units.Unit>
    <rts.units.Unit type="Worker" ID="302" player="1" x="29" y="30" resources="0" hitpoints="1" >
    </rts.units.Unit>
  </units>
</rts.PhysicalGameState>
'''

output_path = r"c:\Users\Pradeep S\Desktop\Semester VI\Project Phase - 1\Simulation\microrts\maps\custom_blockade_32x32.xml"
with open(output_path, "w") as f:
    f.write(xml)

print("Custom 32x32 SYMMETRIC Blockade Map Generated!")
print(f"Saved to: {output_path}")
print(f"Wall count: {terrain_str.count('1')} / {W*H} cells")
print()
print("Visual Preview (. = open, # = wall, B = base area, R = resource):")
for y in range(H):
    row = ""
    for x in range(W):
        if terrain[y][x] == 1:
            row += "# "
        elif (x <= 2 and y <= 2):
            row += "B "
        elif (x >= 29 and y >= 29):
            row += "B "
        else:
            row += ". "
    print(row)

print()
print("Key features:")
print("  - Diagonally SYMMETRIC (fair for both players)")
print("  - Central fortress blocks direct rush → forces flanking")
print("  - Chokepoint walls near each base with gap → rewards defensive play")
print("  - Mid-field pillars → pathfinding obstacles")
