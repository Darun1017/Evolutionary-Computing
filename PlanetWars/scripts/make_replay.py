"""
Planet Wars Cinematic HTML5 Replay Visualizer
=============================================
A polished, high-definition space battle visualizer:
- Starfield space background with nebula glow and twinkling stars
- 3D-shaded glowing planets with real-time garrison counts and +growth rates
- Arrowhead spacecraft fleets with glowing thruster engine trails pointing in direction of travel
- Shockwave explosion rings when battles occur and planets are conquered
- Live dynamic Ship Count Area Chart at the bottom (Google AI Challenge style)
- Real-time HUD showing Turn, Fleet Counts, Production Rates, and Live Score
- Autoplay on load, Play/Pause, Step, Speed buttons, and Timeline Scrubbing
"""

from __future__ import annotations

import json
import math
import sys
import webbrowser
from pathlib import Path
from typing import Callable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from planetwars.sim import GameState, parse_map_file
from planetwars.opponents import OPPONENT_BOTS


def run_match_and_record_events(
    map_path: Path,
    p1_fn: Callable[[GameState, int], list],
    p2_fn: Callable[[GameState, int], list],
    max_turns: int = 100,
) -> Tuple[GameState, dict]:
    """
    Run match in pure Python simulator, tracking:
    - per-turn planet states
    - per-turn fleet states (with source, dest, progress)
    - conquest events (when planet ownership changes)
    - arrival battles
    """
    state = parse_map_file(map_path)
    state.max_turns = max_turns

    planets_init = [
        {"id": p.planet_id, "x": p.x, "y": p.y, "growth": p.growth_rate}
        for p in state.planets
    ]

    turns_data = []
    prev_owners = {p.planet_id: p.owner for p in state.planets}

    def record_turn(st: GameState):
        nonlocal prev_owners
        p_states = []
        conquests = []

        for p in st.planets:
            p_states.append({
                "id": p.planet_id,
                "owner": p.owner,
                "ships": p.num_ships,
            })
            if p.owner != prev_owners.get(p.planet_id, p.owner) and prev_owners.get(p.planet_id) is not None:
                conquests.append({"planet_id": p.planet_id, "new_owner": p.owner})
            prev_owners[p.planet_id] = p.owner

        f_states = [
            {
                "owner": f.owner,
                "ships": f.num_ships,
                "src": f.source_planet,
                "dst": f.destination_planet,
                "total": f.total_trip_length,
                "remaining": f.turns_remaining,
            }
            for f in st.fleets
        ]

        turns_data.append({
            "turn": st.turn,
            "planets": p_states,
            "fleets": f_states,
            "conquests": conquests,
            "p1_ships": st.num_ships(1),
            "p2_ships": st.num_ships(2),
            "p1_prod": st.production(1),
            "p2_prod": st.production(2),
        })

    record_turn(state)

    while not state.game_over and state.turn < max_turns:
        o1 = p1_fn(state, 1)
        o2 = p2_fn(state, 2)
        state.step(o1, o2)
        record_turn(state)

    winner = state.winner if state.winner is not None else 0

    match_data = {
        "map_name": map_path.stem,
        "max_turns": max_turns,
        "total_turns": state.turn,
        "winner": winner,
        "planets": planets_init,
        "turns": turns_data,
    }

    return state, match_data


def generate_cinematic_html(
    match_data: dict,
    p1_name: str,
    p2_name: str,
    output_html_path: Path,
):
    json_data = json.dumps(match_data)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Planet Wars Epic Battle: {p1_name} vs {p2_name}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #090a10;
      color: #e5e9f0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      overflow-x: hidden;
      padding: 15px 0;
    }}
    .hud {{
      width: 900px;
      background: linear-gradient(180deg, #181b29 0%, #12141f 100%);
      border: 1px solid #2b304a;
      border-bottom: none;
      border-radius: 10px 10px 0 0;
      padding: 14px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }}
    .player-badge {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .p1-name {{
      font-size: 17px;
      font-weight: 700;
      color: #00d2ff;
      text-shadow: 0 0 10px rgba(0,210,255,0.4);
    }}
    .p2-name {{
      font-size: 17px;
      font-weight: 700;
      color: #ff3366;
      text-shadow: 0 0 10px rgba(255,51,102,0.4);
    }}
    .stats-pill {{
      font-size: 13px;
      background: #202438;
      padding: 4px 10px;
      border-radius: 15px;
      border: 1px solid #323854;
      color: #d8dee9;
    }}
    .turn-badge {{
      font-size: 16px;
      font-weight: 800;
      letter-spacing: 0.5px;
      color: #ffd166;
      background: #1c2033;
      padding: 6px 18px;
      border-radius: 6px;
      border: 1px solid #3d456b;
      box-shadow: inset 0 0 8px rgba(0,0,0,0.3);
    }}
    #canvas-container {{
      width: 900px;
      height: 600px;
      position: relative;
      background: #06070c;
      border-left: 1px solid #2b304a;
      border-right: 1px solid #2b304a;
      overflow: hidden;
    }}
    canvas#view {{ width: 100%; height: 100%; display: block; }}
    .chart-container {{
      width: 900px;
      height: 80px;
      background: #0e1018;
      border-left: 1px solid #2b304a;
      border-right: 1px solid #2b304a;
      border-top: 1px solid #1c2033;
      position: relative;
    }}
    canvas#chart {{ width: 100%; height: 100%; display: block; }}
    .controls {{
      width: 900px;
      background: linear-gradient(180deg, #12141f 0%, #0d0f17 100%);
      border: 1px solid #2b304a;
      border-top: none;
      border-radius: 0 0 10px 10px;
      padding: 14px 24px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    }}
    .timeline-row {{ display: flex; align-items: center; gap: 15px; }}
    .timeline {{
      flex: 1;
      accent-color: #00d2ff;
      cursor: pointer;
      height: 7px;
    }}
    .btn-row {{ display: flex; justify-content: space-between; align-items: center; }}
    .btn-group {{ display: flex; gap: 8px; align-items: center; }}
    button {{
      background: #202438;
      color: #eceff4;
      border: 1px solid #383f61;
      padding: 7px 15px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }}
    button:hover {{ background: #2c324e; border-color: #00d2ff; transform: translateY(-1px); }}
    button.active {{ background: #00d2ff; color: #0b0f19; border-color: #00d2ff; font-weight: 700; }}
    .speed-btn {{ padding: 5px 11px; font-size: 12px; }}
    .winner-banner {{
      font-size: 15px;
      font-weight: 800;
      color: #06d6a0;
      text-shadow: 0 0 10px rgba(6,214,160,0.5);
    }}
  </style>
</head>
<body>
  <div class="hud">
    <div class="player-badge">
      <span class="p1-name">{p1_name}</span>
      <span class="stats-pill" id="p1-stats">100 ships (+5/t)</span>
    </div>
    <div class="turn-badge" id="turn-display">Turn 00 / 00</div>
    <div class="player-badge">
      <span class="stats-pill" id="p2-stats">100 ships (+5/t)</span>
      <span class="p2-name">{p2_name}</span>
    </div>
  </div>

  <div id="canvas-container">
    <canvas id="view" width="900" height="600"></canvas>
  </div>

  <div class="chart-container">
    <canvas id="chart" width="900" height="80"></canvas>
  </div>

  <div class="controls">
    <div class="timeline-row">
      <span id="curr-turn-num" style="font-weight: bold; color: #00d2ff;">0</span>
      <input type="range" id="scrub" class="timeline" min="0" max="100" value="0">
      <span id="max-turn-num" style="color: #7b849b;">100</span>
    </div>
    <div class="btn-row">
      <div class="btn-group">
        <button id="btn-restart">&#124;&laquo; Reset</button>
        <button id="btn-playpause" class="active">&#9646;&#9646; Pause</button>
        <button id="btn-step">&#9654;&#124; Step</button>
        <button id="btn-loop">&#10227; Loop: OFF</button>
      </div>
      <div id="winner-tag" class="winner-banner"></div>
      <div class="btn-group">
        <span style="font-size: 13px; color: #8892b0;">Speed:</span>
        <button class="speed-btn" data-speed="0.5">0.5x</button>
        <button class="speed-btn active" data-speed="1">1x</button>
        <button class="speed-btn" data-speed="2">2x</button>
        <button class="speed-btn" data-speed="4">4x</button>
      </div>
    </div>
  </div>

  <script>
    const data = {json_data};
    const canvas = document.getElementById("view");
    const ctx = canvas.getContext("2d");

    const chartCanvas = document.getElementById("chart");
    const cctx = chartCanvas.getContext("2d");

    const scrub = document.getElementById("scrub");
    const btnPlayPause = document.getElementById("btn-playpause");
    const btnRestart = document.getElementById("btn-restart");
    const btnStep = document.getElementById("btn-step");
    const btnLoop = document.getElementById("btn-loop");
    const turnDisplay = document.getElementById("turn-display");
    const p1Stats = document.getElementById("p1-stats");
    const p2Stats = document.getElementById("p2-stats");
    const winnerTag = document.getElementById("winner-tag");
    const maxTurnNum = document.getElementById("max-turn-num");
    const currTurnNum = document.getElementById("curr-turn-num");

    const turnsCount = data.turns.length;
    scrub.max = turnsCount - 1;
    maxTurnNum.textContent = turnsCount - 1;

    // Coordinate scale
    let minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
    data.planets.forEach(p => {{
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }});
    const pad = 3.8;
    minX -= pad; maxX += pad; minY -= pad; maxY += pad;

    function toScreenX(x) {{ return ((x - minX) / (maxX - minX)) * canvas.width; }}
    function toScreenY(y) {{ return canvas.height - ((y - minY) / (maxY - minY)) * canvas.height; }}

    const planetMap = {{}};
    data.planets.forEach(p => {{ planetMap[p.id] = p; }});

    // Starfield generation
    const stars = [];
    for (let i = 0; i < 150; i++) {{
      stars.push({{
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        r: Math.random() * 1.5 + 0.3,
        alpha: Math.random() * 0.7 + 0.3,
      }});
    }}

    // Palettes
    const PALETTE = {{
      0: {{ main: "#4a5568", glow: "rgba(74,85,104,0.4)", text: "#cbd5e0", dark: "#2d3748" }},
      1: {{ main: "#00d2ff", glow: "rgba(0,210,255,0.55)", text: "#e0f7ff", dark: "#0984e3" }},
      2: {{ main: "#ff3366", glow: "rgba(255,51,102,0.55)", text: "#ffebee", dark: "#d63031" }},
    }};

    let currentTurnIndex = 0;
    let turnProgress = 0.0;    // 0.0 to 1.0 interpolation between turns
    let isPlaying = true;
    let isLooping = false;     // Default to pause on victory so user sees full finished match!
    let speed = 2.0;           // Default to 2x speed for smooth pacing
    let lastTime = 0;
    let endgameWaitCount = 0;
    const msPerTurn = 400;

    // Shockwave animations
    const shockwaves = [];

    function addShockwave(x, y, color) {{
      shockwaves.push({{ x, y, r: 15, maxR: 45, alpha: 1.0, color }});
    }}

    function drawChart() {{
      cctx.clearRect(0, 0, chartCanvas.width, chartCanvas.height);
      const w = chartCanvas.width;
      const h = chartCanvas.height;

      // Draw faint background grid
      cctx.strokeStyle = "#161824";
      cctx.lineWidth = 1;
      for (let y = 0; y < h; y += 20) {{
        cctx.beginPath(); cctx.moveTo(0, y); cctx.lineTo(w, y); cctx.stroke();
      }}

      // Find max ships
      let maxShips = 1;
      data.turns.forEach(t => {{
        if (t.p1_ships > maxShips) maxShips = t.p1_ships;
        if (t.p2_ships > maxShips) maxShips = t.p2_ships;
      }});
      maxShips = Math.max(maxShips, 50);

      // Draw P1 ship count area
      cctx.beginPath();
      cctx.moveTo(0, h);
      for (let i = 0; i < turnsCount; i++) {{
        const x = (i / (turnsCount - 1)) * w;
        const y = h - (data.turns[i].p1_ships / maxShips) * (h - 8);
        cctx.lineTo(x, y);
      }}
      cctx.lineTo(w, h);
      cctx.fillStyle = "rgba(0,210,255,0.25)";
      cctx.fill();
      cctx.strokeStyle = "#00d2ff";
      cctx.lineWidth = 1.5;
      cctx.stroke();

      // Draw P2 ship count area
      cctx.beginPath();
      cctx.moveTo(0, h);
      for (let i = 0; i < turnsCount; i++) {{
        const x = (i / (turnsCount - 1)) * w;
        const y = h - (data.turns[i].p2_ships / maxShips) * (h - 8);
        cctx.lineTo(x, y);
      }}
      cctx.lineTo(w, h);
      cctx.fillStyle = "rgba(255,51,102,0.25)";
      cctx.fill();
      cctx.strokeStyle = "#ff3366";
      cctx.lineWidth = 1.5;
      cctx.stroke();

      // Draw current turn scrubber line
      const currX = (currentTurnIndex / (turnsCount - 1)) * w;
      cctx.strokeStyle = "#ffd166";
      cctx.lineWidth = 2;
      cctx.beginPath();
      cctx.moveTo(currX, 0);
      cctx.lineTo(currX, h);
      cctx.stroke();
    }}

    function render(turnIdx, progress) {{
      const curr = data.turns[turnIdx];
      const next = data.turns[Math.min(turnsCount - 1, turnIdx + 1)];

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 1. Draw Starfield
      stars.forEach(s => {{
        ctx.fillStyle = "rgba(255,255,255," + s.alpha + ")";
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fill();
      }});

      // 2. Faint orbit constellation paths
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(40, 48, 75, 0.35)";
      for (let i = 0; i < data.planets.length; i++) {{
        for (let j = i + 1; j < data.planets.length; j++) {{
          const p1 = data.planets[i];
          const p2 = data.planets[j];
          if (Math.hypot(p1.x - p2.x, p1.y - p2.y) < 7.5) {{
            ctx.beginPath();
            ctx.moveTo(toScreenX(p1.x), toScreenY(p1.y));
            ctx.lineTo(toScreenX(p2.x), toScreenY(p2.y));
            ctx.stroke();
          }}
        }}
      }}

      // 3. Draw Shockwaves
      for (let i = shockwaves.length - 1; i >= 0; i--) {{
        const sw = shockwaves[i];
        ctx.beginPath();
        ctx.arc(sw.x, sw.y, sw.r, 0, Math.PI * 2);
        ctx.strokeStyle = sw.color.replace("0.55", sw.alpha);
        ctx.lineWidth = 2.5;
        ctx.stroke();
        sw.r += 1.8;
        sw.alpha -= 0.04;
        if (sw.alpha <= 0) shockwaves.splice(i, 1);
      }}

      // 4. Draw Planets
      curr.planets.forEach(ps => {{
        const p = planetMap[ps.id];
        const sx = toScreenX(p.x);
        const sy = toScreenY(p.y);
        const r = 14 + p.growth * 4.2;
        const pal = PALETTE[ps.owner] || PALETTE[0];

        // Atmosphere Glow
        const grad = ctx.createRadialGradient(sx, sy, r * 0.7, sx, sy, r + 8);
        grad.addColorStop(0, pal.glow);
        grad.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(sx, sy, r + 8, 0, Math.PI * 2);
        ctx.fill();

        // 3D Planet Body
        const bodyGrad = ctx.createRadialGradient(sx - r*0.3, sy - r*0.3, r*0.1, sx, sy, r);
        bodyGrad.addColorStop(0, pal.text);
        bodyGrad.addColorStop(0.35, pal.main);
        bodyGrad.addColorStop(1, pal.dark);
        ctx.fillStyle = bodyGrad;
        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
        ctx.fill();

        // High-tech ring border
        ctx.strokeStyle = ps.owner === 0 ? "#718096" : "#ffffff";
        ctx.lineWidth = ps.owner === 0 ? 1.5 : 2.5;
        ctx.stroke();

        // Ships text
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 13px -apple-system, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(ps.ships, sx, sy);

        // Growth rate badge
        ctx.fillStyle = "#a0aec0";
        ctx.font = "bold 10px -apple-system, sans-serif";
        ctx.fillText("+" + p.growth, sx, sy + r + 13);
      }});

      // 5. Draw Fleets with Smooth Interpolation & Arrowhead Craft
      curr.fleets.forEach(fl => {{
        const src = planetMap[fl.src];
        const dst = planetMap[fl.dst];
        const total = Math.max(1, fl.total);

        // Smooth flight progress between turns
        const currentTurnProgress = (total - fl.remaining) + progress;
        const normProgress = Math.max(0.0, Math.min(1.0, currentTurnProgress / total));

        const fx = src.x + (dst.x - src.x) * normProgress;
        const fy = src.y + (dst.y - src.y) * normProgress;
        const fsx = toScreenX(fx);
        const fsy = toScreenY(fy);

        const angle = Math.atan2(toScreenY(dst.y) - toScreenY(src.y), toScreenX(dst.x) - toScreenX(src.x));
        const pal = PALETTE[fl.owner] || PALETTE[0];

        // Laser trajectory track
        ctx.save();
        ctx.setLineDash([4, 6]);
        ctx.strokeStyle = pal.glow;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(toScreenX(src.x), toScreenY(src.y));
        ctx.lineTo(toScreenX(dst.x), toScreenY(dst.y));
        ctx.stroke();
        ctx.restore();

        // Draw Arrowhead Spaceship
        ctx.save();
        ctx.translate(fsx, fsy);
        ctx.rotate(angle);

        // Thruster glow
        ctx.fillStyle = "#ffd166";
        ctx.beginPath();
        ctx.arc(-8, 0, 3, 0, Math.PI * 2);
        ctx.fill();

        // Ship hull
        ctx.fillStyle = pal.main;
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(9, 0);
        ctx.lineTo(-7, -6);
        ctx.lineTo(-4, 0);
        ctx.lineTo(-7, 6);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.restore();

        // Fleet count tag
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 11px -apple-system, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(fl.ships, fsx, fsy - 12);
      }});

      // Update HUD & Chart
      turnDisplay.textContent = "Turn " + curr.turn + " / " + (turnsCount - 1);
      currTurnNum.textContent = curr.turn;
      scrub.value = turnIdx;
      p1Stats.textContent = curr.p1_ships + " ships (+" + curr.p1_prod + "/t)";
      p2Stats.textContent = curr.p2_ships + " ships (+" + curr.p2_prod + "/t)";

      if (turnIdx === turnsCount - 1) {{
        let wmsg = "DRAW";
        let submsg = "Game reached max turns in a tie.";
        let bannerColor = "#ffd166";
        if (data.winner === 1) {{
          wmsg = "{p1_name} WINS!";
          submsg = curr.p2_ships === 0
            ? "TOTAL KNOCKOUT: Enemy completely eliminated to 0 ships!"
            : "Turn limit reached — won on ship count dominance ({p1_name}: " + curr.p1_ships + " vs {p2_name}: " + curr.p2_ships + ")";
          bannerColor = "#00d2ff";
        }} else if (data.winner === 2) {{
          wmsg = "{p2_name} WINS!";
          submsg = curr.p1_ships === 0
            ? "TOTAL KNOCKOUT: Enemy completely eliminated to 0 ships!"
            : "Turn limit reached — won on ship count dominance ({p2_name}: " + curr.p2_ships + " vs {p1_name}: " + curr.p1_ships + ")";
          bannerColor = "#ff3366";
        }}
        winnerTag.textContent = wmsg;

        // Big visual endgame banner on canvas
        ctx.save();
        ctx.fillStyle = "rgba(10, 12, 20, 0.88)";
        ctx.fillRect(canvas.width / 2 - 320, canvas.height / 2 - 65, 640, 130);
        ctx.strokeStyle = bannerColor;
        ctx.lineWidth = 3;
        ctx.strokeRect(canvas.width / 2 - 320, canvas.height / 2 - 65, 640, 130);

        ctx.fillStyle = bannerColor;
        ctx.font = "bold 24px -apple-system, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(wmsg, canvas.width / 2, canvas.height / 2 - 18);

        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 15px -apple-system, sans-serif";
        ctx.fillText(submsg, canvas.width / 2, canvas.height / 2 + 15);

        ctx.fillStyle = "#a0aec0";
        ctx.font = "12px -apple-system, sans-serif";
        ctx.fillText("Match completed at Turn " + curr.turn + " of " + (turnsCount - 1), canvas.width / 2, canvas.height / 2 + 42);
        ctx.restore();
      }} else {{
        winnerTag.textContent = "";
      }}

      drawChart();
    }}

    function animate(timestamp) {{
      if (isPlaying) {{
        if (!lastTime) lastTime = timestamp;
        const dt = timestamp - lastTime;
        lastTime = timestamp;

        const turnDuration = msPerTurn / speed;
        turnProgress += dt / turnDuration;

        if (turnProgress >= 1.0) {{
          turnProgress = 0.0;
          if (currentTurnIndex < turnsCount - 1) {{
            currentTurnIndex++;
            // Check for planet conquests to trigger shockwaves
            const cEvents = data.turns[currentTurnIndex].conquests;
            if (cEvents) {{
              cEvents.forEach(ce => {{
                const p = planetMap[ce.planet_id];
                const pal = PALETTE[ce.new_owner] || PALETTE[0];
                addShockwave(toScreenX(p.x), toScreenY(p.y), pal.glow);
              }});
            }}
          }} else {{
            // Game reached final turn! Stay on the Victory screen
            if (endgameWaitCount < 8) {{
              endgameWaitCount++;
            }} else if (isLooping) {{
              endgameWaitCount = 0;
              currentTurnIndex = 0;
            }} else {{
              isPlaying = false;
              updatePlayBtn();
            }}
          }}
        }}
      }}
      render(currentTurnIndex, turnProgress);
      requestAnimationFrame(animate);
    }}

    function updatePlayBtn() {{
      if (isPlaying) {{
        btnPlayPause.textContent = "⏸ Pause";
        btnPlayPause.classList.add("active");
      }} else {{
        btnPlayPause.textContent = "▶ Play";
        btnPlayPause.classList.remove("active");
      }}
    }}

    btnPlayPause.addEventListener("click", () => {{
      isPlaying = !isPlaying;
      updatePlayBtn();
      lastTime = 0;
    }});

    btnRestart.addEventListener("click", () => {{
      currentTurnIndex = 0;
      turnProgress = 0.0;
      lastTime = 0;
      render(0, 0);
    }});

    btnStep.addEventListener("click", () => {{
      isPlaying = false;
      updatePlayBtn();
      if (currentTurnIndex < turnsCount - 1) currentTurnIndex++;
      turnProgress = 0.0;
      render(currentTurnIndex, 0);
    }});

    btnLoop.addEventListener("click", () => {{
      isLooping = !isLooping;
      btnLoop.textContent = isLooping ? "🔁 Loop: ON" : "🔁 Loop: OFF";
      btnLoop.classList.toggle("active", isLooping);
    }});

    scrub.addEventListener("input", (e) => {{
      isPlaying = false;
      updatePlayBtn();
      currentTurnIndex = parseInt(e.target.value);
      turnProgress = 0.0;
      render(currentTurnIndex, 0);
    }});

    document.querySelectorAll(".speed-btn").forEach(btn => {{
      btn.addEventListener("click", (e) => {{
        document.querySelectorAll(".speed-btn").forEach(b => b.classList.remove("active"));
        e.target.classList.add("active");
        speed = parseFloat(e.target.dataset.speed);
      }});
    }});

    // Launch immediately on page load
    render(0, 0);
    requestAnimationFrame(animate);
  </script>
</body>
</html>
"""
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"Generated Cinematic Replay to: {output_html_path}")


def make_replay(
    map_name: str = "map1.txt",
    p1_name: str = "ProspectorBot",
    p2_name: str = "DualBot",
    max_turns: int = 200,
    open_browser: bool = True,
):
    map_path = PROJECT_ROOT / "engine" / "maps" / map_name
    p1_fn = OPPONENT_BOTS[p1_name]
    p2_fn = OPPONENT_BOTS[p2_name]

    print(f"\n[Cinematic Replay] Simulating full match: {p1_name} (P1) vs {p2_name} (P2) on {map_name} (Full {max_turns} turns)...")
    state, match_data = run_match_and_record_events(map_path, p1_fn, p2_fn, max_turns=max_turns)

    w_str = f"Player {state.winner} Wins" if state.winner > 0 else "Draw"
    print(f"[Cinematic Replay] Finished at Turn {state.turn} -> Result: {w_str}!")

    replays_dir = PROJECT_ROOT / "results" / "replays"
    replays_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{map_path.stem}_{p1_name}_vs_{p2_name}"
    html_path = replays_dir / f"{base_name}_replay.html"

    generate_cinematic_html(match_data, p1_name, p2_name, html_path)

    if open_browser:
        print(f"[Cinematic Replay] Opening video player in your default browser...")
        webbrowser.open(html_path.as_uri())

    return html_path


if __name__ == "__main__":
    make_replay(map_name="map1.txt", p1_name="ProspectorBot", p2_name="DualBot", max_turns=200, open_browser=True)
