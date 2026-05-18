"""
Generate a 128x128 microRTS map with chambers, blockades, and resource patches.
Design inspired by chambers32x32.xml but scaled up with more features.

Layout:
  - Player 0 base: top-left corner
  - Player 1 base: bottom-right corner
  - Multiple chambers separated by walls with narrow passages
  - Resource patches spread across the map (contested middle resources)
  - Multiple chokepoints that create strategic decisions
"""

W, H = 128, 128

# Initialize empty terrain (0 = walkable)
terrain = [[0] * W for _ in range(H)]

def wall_h(y, x1, x2):
    """Draw horizontal wall from x1 to x2 at row y."""
    for x in range(max(0, x1), min(W, x2 + 1)):
        terrain[y][x] = 1

def wall_v(x, y1, y2):
    """Draw vertical wall from y1 to y2 at column x."""
    for y in range(max(0, y1), min(H, y2 + 1)):
        terrain[y][x] = 1

def clear(y, x):
    """Open a passage at (x, y)."""
    if 0 <= y < H and 0 <= x < W:
        terrain[y][x] = 0

# ═══════════════════════════════════════════════════════
#  OUTER BORDER — walls around the entire map edge
# ═══════════════════════════════════════════════════════
for x in range(W):
    terrain[0][x] = 1
    terrain[H-1][x] = 1
for y in range(H):
    terrain[y][0] = 1
    terrain[y][W-1] = 1

# Open spawn areas
for y in range(1, 8):
    for x in range(1, 8):
        terrain[y][x] = 0
for y in range(H-8, H-1):
    for x in range(W-8, W-1):
        terrain[y][x] = 0

# ═══════════════════════════════════════════════════════
#  QUADRANT WALLS — divide map into 4 large chambers
# ═══════════════════════════════════════════════════════

# Horizontal wall at y=32 (top half / bottom half)
wall_h(32, 1, W-2)
# Gaps (passages between top and bottom)
for x in range(28, 36):
    clear(32, x)
for x in range(92, 100):
    clear(32, x)
for x in range(60, 68):
    clear(32, x)

# Horizontal wall at y=96
wall_h(96, 1, W-2)
for x in range(28, 36):
    clear(96, x)
for x in range(92, 100):
    clear(96, x)
for x in range(60, 68):
    clear(96, x)

# Vertical wall at x=32
wall_v(32, 1, H-2)
for y in range(28, 36):
    clear(y, 32)
for y in range(92, 100):
    clear(y, 32)
for y in range(60, 68):
    clear(y, 32)

# Vertical wall at x=96
wall_v(96, 1, H-2)
for y in range(28, 36):
    clear(y, 96)
for y in range(92, 100):
    clear(y, 96)
for y in range(60, 68):
    clear(y, 96)

# ═══════════════════════════════════════════════════════
#  INNER WALLS — create sub-chambers within quadrants
# ═══════════════════════════════════════════════════════

# Horizontal wall at y=64 (exact middle)
wall_h(64, 1, W-2)
for x in range(14, 20):
    clear(64, x)
for x in range(44, 52):
    clear(64, x)
for x in range(76, 84):
    clear(64, x)
for x in range(108, 116):
    clear(64, x)

# Vertical wall at x=64 (exact middle)
wall_v(64, 1, H-2)
for y in range(14, 20):
    clear(y, 64)
for y in range(44, 52):
    clear(y, 64)
for y in range(76, 84):
    clear(y, 64)
for y in range(108, 116):
    clear(y, 64)

# ═══════════════════════════════════════════════════════
#  FORTRESS WALLS — small defensive structures near bases
# ═══════════════════════════════════════════════════════

# Player 0 fortress (top-left)
wall_h(16, 1, 24)
wall_v(24, 1, 16)
for x in range(10, 14):
    clear(16, x)  # entrance
clear(16, 24)  # corner passage

# Player 1 fortress (bottom-right)
wall_h(112, 104, W-2)
wall_v(104, 112, H-2)
for x in range(114, 118):
    clear(112, x)
clear(112, 104)

# ═══════════════════════════════════════════════════════
#  CHOKEPOINT CORRIDORS — narrow diagonal-ish paths
# ═══════════════════════════════════════════════════════

# Corridor walls near center creating narrow passages
for i in range(8):
    # Top-left to center corridor walls
    terrain[48 + i][48] = 1
    terrain[48][48 + i] = 1
    # Bottom-right to center corridor walls
    terrain[72 + i][80] = 1
    terrain[80][72 + i] = 1

# Open passages through corridors
clear(52, 48)
clear(48, 52)
clear(76, 80)
clear(80, 76)

# ═══════════════════════════════════════════════════════
#  ISLAND OBSTACLES — scattered wall blocks in open areas
# ═══════════════════════════════════════════════════════

# Small 3x3 wall blocks scattered around
islands = [(20, 50), (50, 20), (78, 108), (108, 78),
           (40, 80), (80, 40), (20, 100), (100, 20)]
for iy, ix in islands:
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            y, x = iy + dy, ix + dx
            if 1 <= y < H-1 and 1 <= x < W-1:
                terrain[y][x] = 1

# ═══════════════════════════════════════════════════════
#  BUILD TERRAIN STRING
# ═══════════════════════════════════════════════════════

terrain_str = ""
for y in range(H):
    for x in range(W):
        terrain_str += str(terrain[y][x])

# ═══════════════════════════════════════════════════════
#  UNITS — Bases, Workers, Resources
# ═══════════════════════════════════════════════════════

units = []
uid = 1

# Player 0 — Top-left
units.append(f'    <rts.units.Unit type="Base" ID="{uid}" player="0" x="4" y="4" resources="0" hitpoints="10" />')
uid += 1
units.append(f'    <rts.units.Unit type="Worker" ID="{uid}" player="0" x="3" y="3" resources="0" hitpoints="1" />')
uid += 1
units.append(f'    <rts.units.Unit type="Worker" ID="{uid}" player="0" x="5" y="3" resources="0" hitpoints="1" />')
uid += 1

# Player 1 — Bottom-right
units.append(f'    <rts.units.Unit type="Base" ID="{uid}" player="1" x="123" y="123" resources="0" hitpoints="10" />')
uid += 1
units.append(f'    <rts.units.Unit type="Worker" ID="{uid}" player="1" x="124" y="124" resources="0" hitpoints="1" />')
uid += 1
units.append(f'    <rts.units.Unit type="Worker" ID="{uid}" player="1" x="122" y="124" resources="0" hitpoints="1" />')
uid += 1

# Resources near Player 0 (top-left)
p0_res = [(2, 1), (1, 2), (1, 6), (6, 1), (1, 14), (14, 1), (8, 8)]
for rx, ry in p0_res:
    units.append(f'    <rts.units.Unit type="Resource" ID="{uid}" player="-1" x="{rx}" y="{ry}" resources="15" hitpoints="1" />')
    uid += 1

# Resources near Player 1 (bottom-right)
p1_res = [(125, 126), (126, 125), (126, 121), (121, 126), (126, 113), (113, 126), (119, 119)]
for rx, ry in p1_res:
    units.append(f'    <rts.units.Unit type="Resource" ID="{uid}" player="-1" x="{rx}" y="{ry}" resources="15" hitpoints="1" />')
    uid += 1

# Contested resources in the CENTER (worth fighting over)
center_res = [
    (60, 60), (61, 60), (60, 61), (61, 61),  # cluster 1
    (66, 66), (67, 66), (66, 67), (67, 67),  # cluster 2
    (62, 63), (65, 64),                        # scattered
]
for rx, ry in center_res:
    units.append(f'    <rts.units.Unit type="Resource" ID="{uid}" player="-1" x="{rx}" y="{ry}" resources="25" hitpoints="1" />')
    uid += 1

# Resources along corridors (strategic pickups)
corridor_res = [
    (30, 30), (30, 98), (98, 30), (98, 98),  # at wall intersections
    (30, 64), (98, 64), (64, 30), (64, 98),  # at mid-wall openings
    (16, 48), (48, 16), (80, 112), (112, 80),  # near chokepoints
]
for rx, ry in corridor_res:
    units.append(f'    <rts.units.Unit type="Resource" ID="{uid}" player="-1" x="{rx}" y="{ry}" resources="20" hitpoints="1" />')
    uid += 1

# ═══════════════════════════════════════════════════════
#  WRITE XML
# ═══════════════════════════════════════════════════════

import os
map_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "Simulation", "microrts", "maps")

xml = f'''<rts.PhysicalGameState width="{W}" height="{H}">
  <terrain>{terrain_str}</terrain>
  <players>
    <rts.Player ID="0" resources="5">
    </rts.Player>
    <rts.Player ID="1" resources="5">
    </rts.Player>
  </players>
  <units>
{chr(10).join(units)}
  </units>
</rts.PhysicalGameState>
'''

output_path = os.path.join(map_dir, "chambers128x128.xml")
with open(output_path, "w") as f:
    f.write(xml)

print(f"Map saved to: {output_path}")
print(f"Size: {W}x{H} = {W*H} tiles")
print(f"Units: {uid - 1} total (2 bases, 4 workers, {uid - 7} resources)")

# Print a visual minimap
print(f"\nMinimap (each char = 4x4 tiles):")
for y in range(0, H, 4):
    row = ""
    for x in range(0, W, 4):
        # Check if this 4x4 block has any walls
        walls = sum(terrain[y+dy][x+dx] for dy in range(min(4, H-y)) for dx in range(min(4, W-x)))
        if walls > 12:
            row += "█"
        elif walls > 4:
            row += "▓"
        elif walls > 0:
            row += "░"
        else:
            row += " "
    print(f"  |{row}|")
print(f"  Map generated successfully!")
