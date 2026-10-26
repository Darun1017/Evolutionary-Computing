"""Generate a 48x48 microRTS map with NO blockades (fully open terrain)."""

W, H = 48, 48
terrain = "0" * (W * H)  # 2304 zeros = no walls

xml = f'''<rts.PhysicalGameState width="{W}" height="{H}">
  <terrain>{terrain}</terrain>
  <players>
    <rts.Player ID="0" resources="5">
    </rts.Player>
    <rts.Player ID="1" resources="5">
    </rts.Player>
  </players>
  <units>
    <rts.units.Unit type="Resource" ID="1" player="-1" x="0" y="0" resources="25" hitpoints="1" >
    </rts.units.Unit>
    <rts.units.Unit type="Resource" ID="2" player="-1" x="0" y="1" resources="25" hitpoints="1" >
    </rts.units.Unit>
    <rts.units.Unit type="Base" ID="3" player="0" x="2" y="2" resources="0" hitpoints="10" >
    </rts.units.Unit>
    <rts.units.Unit type="Worker" ID="4" player="0" x="1" y="1" resources="0" hitpoints="1" >
    </rts.units.Unit>
    <rts.units.Unit type="Resource" ID="5" player="-1" x="47" y="47" resources="25" hitpoints="1" >
    </rts.units.Unit>
    <rts.units.Unit type="Resource" ID="6" player="-1" x="47" y="46" resources="25" hitpoints="1" >
    </rts.units.Unit>
    <rts.units.Unit type="Base" ID="7" player="1" x="45" y="45" resources="0" hitpoints="10" >
    </rts.units.Unit>
    <rts.units.Unit type="Worker" ID="8" player="1" x="46" y="46" resources="0" hitpoints="1" >
    </rts.units.Unit>
  </units>
</rts.PhysicalGameState>
'''

import os
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "Simulation", "microrts", "maps", "open48x48.xml")
with open(out, "w") as f:
    f.write(xml)
print(f"Wrote {out}")
print(f"Terrain length: {len(terrain)} (expected {W*H})")
