"""Generate a 64x64 microRTS map with NO blockades (fully open terrain)."""

W, H = 64, 64
terrain = "0" * (W * H)  # 4096 zeros = no walls

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
    <rts.units.Unit type="Resource" ID="5" player="-1" x="63" y="63" resources="25" hitpoints="1" >
    </rts.units.Unit>
    <rts.units.Unit type="Resource" ID="6" player="-1" x="63" y="62" resources="25" hitpoints="1" >
    </rts.units.Unit>
    <rts.units.Unit type="Base" ID="7" player="1" x="61" y="61" resources="0" hitpoints="10" >
    </rts.units.Unit>
    <rts.units.Unit type="Worker" ID="8" player="1" x="62" y="62" resources="0" hitpoints="1" >
    </rts.units.Unit>
  </units>
</rts.PhysicalGameState>
'''

import os
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "Simulation", "microrts", "maps", "open64x64.xml")
with open(out, "w") as f:
    f.write(xml)
print(f"Wrote {out}")
print(f"Terrain length: {len(terrain)} (expected {W*H})")
