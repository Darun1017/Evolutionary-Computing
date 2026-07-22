"""
Generate a 64x64 microRTS map with chambers, blockades, and resource patches.
Design inspired by chambers32x32.xml scaled up with more features.
"""

W, H = 64, 64

terrain = [[0] * W for _ in range(H)]

def wall_h(y, x1, x2):
    for x in range(max(0, x1), min(W, x2 + 1)):
        terrain[y][x] = 1

def wall_v(x, y1, y2):
    for y in range(max(0, y1), min(H, y2 + 1)):
        terrain[y][x] = 1

def clear(y, x):
    if 0 <= y < H and 0 <= x < W:
        terrain[y][x] = 0

# ── Outer border ──
for x in range(W):
    terrain[0][x] = 1
    terrain[H-1][x] = 1
for y in range(H):
    terrain[y][0] = 1
    terrain[y][W-1] = 1

# Open spawn corners
for y in range(1, 6):
    for x in range(1, 6):
        terrain[y][x] = 0
for y in range(H-6, H-1):
    for x in range(W-6, W-1):
        terrain[y][x] = 0

# ── Main divider walls (cross shape) ──
# Horizontal wall at y=32 (middle)
wall_h(32, 1, W-2)
for x in range(14, 18): clear(32, x)   # left passage
for x in range(30, 34): clear(32, x)   # center-left passage
for x in range(46, 50): clear(32, x)   # center-right passage

# Vertical wall at x=32 (middle)
wall_v(32, 1, H-2)
for y in range(14, 18): clear(y, 32)
for y in range(30, 34): clear(y, 32)
for y in range(46, 50): clear(y, 32)

# ── Quadrant inner walls ──
# Top-left quadrant: horizontal wall at y=16
wall_h(16, 1, 31)
for x in range(8, 12): clear(16, x)
for x in range(22, 26): clear(16, x)

# Top-right quadrant: horizontal wall at y=16
wall_h(16, 33, W-2)
for x in range(40, 44): clear(16, x)
for x in range(52, 56): clear(16, x)

# Bottom-left quadrant: horizontal wall at y=48
wall_h(48, 1, 31)
for x in range(8, 12): clear(48, x)
for x in range(22, 26): clear(48, x)

# Bottom-right quadrant: horizontal wall at y=48
wall_h(48, 33, W-2)
for x in range(40, 44): clear(48, x)
for x in range(52, 56): clear(48, x)

# Vertical sub-walls
wall_v(16, 1, 31)
for y in range(8, 12): clear(y, 16)
for y in range(22, 26): clear(y, 16)

wall_v(48, 1, 31)
for y in range(8, 12): clear(y, 48)
for y in range(22, 26): clear(y, 48)

wall_v(16, 33, H-2)
for y in range(40, 44): clear(y, 16)
for y in range(52, 56): clear(y, 16)

wall_v(48, 33, H-2)
for y in range(40, 44): clear(y, 48)
for y in range(52, 56): clear(y, 48)

# ── Fortress walls near bases ──
# Player 0 fortress
wall_h(10, 1, 12)
wall_v(12, 1, 10)
for x in range(5, 8): clear(10, x)

# Player 1 fortress
wall_h(53, 52, W-2)
wall_v(52, 53, H-2)
for x in range(56, 59): clear(53, x)

# ── Small obstacle blocks ──
obstacles = [(24, 24), (24, 40), (40, 24), (40, 40), (10, 40), (40, 10), (54, 24), (24, 54)]
for oy, ox in obstacles:
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            y, x = oy + dy, ox + dx
            if 1 <= y < H-1 and 1 <= x < W-1:
                terrain[y][x] = 1

# Build terrain string
terrain_str = "".join(str(terrain[y][x]) for y in range(H) for x in range(W))

# ── Units ──
units = []
uid = 1

# Player 0 — Top-left
units.append(f'    <rts.units.Unit type="Base" ID="{uid}" player="0" x="3" y="3" resources="0" hitpoints="10" />')
uid += 1
units.append(f'    <rts.units.Unit type="Worker" ID="{uid}" player="0" x="2" y="2" resources="0" hitpoints="1" />')
uid += 1
units.append(f'    <rts.units.Unit type="Worker" ID="{uid}" player="0" x="4" y="2" resources="0" hitpoints="1" />')
uid += 1

# Player 1 — Bottom-right
units.append(f'    <rts.units.Unit type="Base" ID="{uid}" player="1" x="60" y="60" resources="0" hitpoints="10" />')
uid += 1
units.append(f'    <rts.units.Unit type="Worker" ID="{uid}" player="1" x="61" y="61" resources="0" hitpoints="1" />')
uid += 1
units.append(f'    <rts.units.Unit type="Worker" ID="{uid}" player="1" x="59" y="61" resources="0" hitpoints="1" />')
uid += 1

# Resources near Player 0
p0_res = [(1, 1), (1, 4), (4, 1), (1, 8), (8, 1), (5, 5)]
for rx, ry in p0_res:
    units.append(f'    <rts.units.Unit type="Resource" ID="{uid}" player="-1" x="{rx}" y="{ry}" resources="15" hitpoints="1" />')
    uid += 1

# Resources near Player 1
p1_res = [(62, 62), (62, 59), (59, 62), (62, 55), (55, 62), (58, 58)]
for rx, ry in p1_res:
    units.append(f'    <rts.units.Unit type="Resource" ID="{uid}" player="-1" x="{rx}" y="{ry}" resources="15" hitpoints="1" />')
    uid += 1

# Contested center resources (worth fighting over!)
center_res = [
    (30, 30), (31, 30), (30, 31), (31, 31),  # center cluster
    (33, 33), (34, 33), (33, 34), (34, 34),  # center cluster 2
]
for rx, ry in center_res:
    units.append(f'    <rts.units.Unit type="Resource" ID="{uid}" player="-1" x="{rx}" y="{ry}" resources="25" hitpoints="1" />')
    uid += 1

# Strategic resources at chokepoints
choke_res = [
    (15, 15), (15, 48), (48, 15), (48, 48),  # at wall gaps
    (32, 15), (32, 48), (15, 32), (48, 32),  # at mid-wall openings
]
for rx, ry in choke_res:
    units.append(f'    <rts.units.Unit type="Resource" ID="{uid}" player="-1" x="{rx}" y="{ry}" resources="20" hitpoints="1" />')
    uid += 1

# ── Write XML ──
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

output_path = os.path.join(map_dir, "chambers64x64.xml")
with open(output_path, "w") as f:
    f.write(xml)

print(f"Map saved to: {output_path}")
print(f"Size: {W}x{H} = {W*H} tiles")
print(f"Units: {uid - 1} total (2 bases, 4 workers, {uid - 7} resources)")
print("Done!")
