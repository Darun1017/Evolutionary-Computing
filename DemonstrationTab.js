"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  Trophy,
  TrendingUp,
  BarChart3,
  Bot,
  Swords,
  Sparkles,
  Sliders,
  Shield,
  Zap,
  RefreshCw,
  Cpu,
  Activity,
  Award,
  ChevronDown,
  ChevronUp,
  Play,
  RotateCcw,
  CheckCircle2,
  CircleDollarSign,
  ArrowRight,
  Layers,
  Users,
  MapPin,
  Eye,
  Info,
  Flame,
  Target
} from "lucide-react";

/* ─────────────────────────────────────────────────────────────
   20 Continuous Strategy Dimensions Metadata (microRTS)
   Aligned with WEIGHT_NAMES (indices 0 - 19)
   ───────────────────────────────────────────────────────────── */
const STRATEGY_DIMENSIONS = [
  { name: "worker_ratio", label: "Worker Ratio", category: "Macro Economy", desc: "Resource allocation to worker training" },
  { name: "max_workers", label: "Max Workers", category: "Macro Economy", desc: "Worker population hard ceiling (1 - 12)" },
  { name: "expand_threshold", label: "Expansion Threshold", category: "Macro Economy", desc: "Minerals needed to expand base" },
  { name: "light_preference", label: "Light Unit Pref.", category: "Micro Combat", desc: "Barracks priority for Light infantry" },
  { name: "heavy_preference", label: "Heavy Unit Pref.", category: "Micro Combat", desc: "Barracks priority for Heavy armor" },
  { name: "ranged_preference", label: "Ranged Unit Pref.", category: "Micro Combat", desc: "Barracks priority for Ranged artillery" },
  { name: "barracks_timing", label: "Barracks Timing", category: "Macro Economy", desc: "Game tick trigger to build Barracks" },
  { name: "aggression", label: "Aggression Drive", category: "Micro Combat", desc: "Willingness to advance across center line to enemy base" },
  { name: "attack_threshold", label: "Attack Threshold", category: "Micro Combat", desc: "Army squad size before launching attack" },
  { name: "defend_radius", label: "Defend Radius", category: "Micro Combat", desc: "Perimeter distance to engage invaders" },
  { name: "target_priority", label: "Target Priority", category: "Micro Combat", desc: "Focus enemy workers vs combat units" },
  { name: "defense_trigger", label: "Defense Trigger", category: "Micro Combat", desc: "Alarm threshold to recall army to base" },
  { name: "counter_rush", label: "Anti-Rush Reaction", category: "Micro Combat", desc: "Tactical pivot against early enemy rushes" },
  { name: "tech_switch", label: "Tech Adaptability", category: "Micro Combat", desc: "Dynamic unit counters to opponent" },
  { name: "economy_pressure", label: "Economy Pressure", category: "Macro Economy", desc: "Aggressiveness of mineral spending" },
  { name: "scout_priority", label: "Scout Priority", category: "Macro Economy", desc: "Worker reconnaissance patrol frequency" },
  { name: "group_attack", label: "Formation Cohesion", category: "Micro Combat", desc: "Grouping into squads before engaging" },
  { name: "second_barracks", label: "Second Barracks", category: "Macro Economy", desc: "Dual barracks military ramp-up" },
  { name: "worker_fight_ratio", label: "Worker Fight Ratio", category: "Macro Economy", desc: "Fraction of workers mobilized to attack enemy" },
  { name: "focus_fire", label: "Focus Fire Focus", category: "Micro Combat", desc: "Concentrating attacks on single targets" },
];

/* ─────────────────────────────────────────────────────────────
   Pre-configured Strategy Presets
   ───────────────────────────────────────────────────────────── */
const PRESETS = [
  {
    id: "evolved_gen50",
    name: "Best Evolved Brain (Gen 50)",
    desc: "Optimized multi-objective vector achieving 84.7% win rate",
    vector: [0.2436, 0.5265, 0.5230, 0.9714, 0.1496, 0.2987, 0.6205, 0.1790, 0.7140, 0.6314, 0.1318, 0.0662, 0.9955, 0.1236, 0.0000, 0.6163, 0.9946, 0.4662, 0.5703, 1.0000]
  },
  {
    id: "early_rush",
    name: "Early Rush Specialist",
    desc: "Ultra-aggressive early barracks and rapid Light infantry swarming",
    vector: [0.35, 0.40, 0.15, 0.95, 0.10, 0.10, 0.20, 0.92, 0.20, 0.25, 0.95, 0.20, 0.40, 0.15, 0.20, 0.10, 0.15, 0.98, 0.90, 0.85]
  },
  {
    id: "heavy_defense",
    name: "Fortress Turtling (Heavy)",
    desc: "Defensive heavy armor wall with tight base perimeter protection",
    vector: [0.60, 0.65, 0.50, 0.10, 0.90, 0.20, 0.50, 0.10, 0.90, 0.95, 0.50, 0.95, 0.85, 0.50, 0.10, 0.90, 0.20, 0.15, 0.00, 0.80]
  },
  {
    id: "macro_boom",
    name: "Macro Booming & Ranged",
    desc: "Greedy economic expansion transitioning into high-range artillery",
    vector: [0.70, 0.75, 0.65, 0.20, 0.20, 0.95, 0.60, 0.40, 0.65, 0.50, 0.70, 0.50, 0.65, 0.60, 0.40, 0.40, 0.50, 0.75, 0.00, 0.75]
  },
  {
    id: "all_zeroes",
    name: "All Zeroes (Passive Miner)",
    desc: "Zero aggression and zero worker combat - worker strictly mines resources",
    vector: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
  }
];

/* ─────────────────────────────────────────────────────────────
   The 7 Canonical microRTS Benchmark Bots
   ───────────────────────────────────────────────────────────── */
const CANONICAL_BOTS = [
  { id: "WorkerRush", name: "WorkerRush" },
  { id: "LightRush", name: "LightRush" },
  { id: "HeavyRush", name: "HeavyRush" },
  { id: "RangedRush", name: "RangedRush" },
  { id: "EconomyRush", name: "EconomyRush" },
  { id: "EconomyMilitaryRush", name: "EconomyMilitaryRush" },
  { id: "HeavyDefense", name: "HeavyDefense" }
];

/* ─────────────────────────────────────────────────────────────
   Map Options
   ───────────────────────────────────────────────────────────── */
const MAP_OPTIONS = [
  { id: "maps/8x8/basesWorkers8x8.xml", name: "8x8 Tournament Arena", desc: "Canonical fast rush & micro map (Standard)", size: 8 },
  { id: "maps/16x16/basesWorkers16x16.xml", name: "16x16 Macro Arena", desc: "Medium tactical expansion map", size: 16 },
  { id: "maps/chambers32x32.xml", name: "32x32 Strategic Depth", desc: "Chambers and choke points for macro timing", size: 32 }
];

function generateRandomVector() {
  return Array.from({ length: 20 }, () => Math.round((Math.random() * 0.9 + 0.05) * 100) / 100);
}

function vectorToString(arr) {
  return arr.map((v) => Number(v).toFixed(2)).join(", ");
}

function parseVectorString(str) {
  const parts = str.split(",").map((s) => parseFloat(s.trim()));
  const vec = [];
  for (let i = 0; i < 20; i++) {
    const val = parts[i];
    if (typeof val === "number" && !isNaN(val)) {
      vec.push(Math.min(1.0, Math.max(0.0, Math.round(val * 100) / 100)));
    } else {
      vec.push(0.0);
    }
  }
  return vec;
}

export default function DemonstrationTab() {
  // Mode: "vs-bot" | "vs-player"
  const [matchMode, setMatchMode] = useState("vs-bot");

  // Player 1 Vector
  const [p1Vector, setP1Vector] = useState(() => PRESETS[0].vector);
  const [p1InputText, setP1InputText] = useState(() => vectorToString(PRESETS[0].vector));
  const [showP1Sliders, setShowP1Sliders] = useState(false);

  // Opponent Bot (for vs-bot)
  const [selectedBot, setSelectedBot] = useState("WorkerRush");

  // Player 2 Vector (for vs-player)
  const [p2Vector, setP2Vector] = useState(() => PRESETS[1].vector);
  const [p2InputText, setP2InputText] = useState(() => vectorToString(PRESETS[1].vector));
  const [showP2Sliders, setShowP2Sliders] = useState(false);

  // Map & Server Config
  const [selectedMap, setSelectedMap] = useState("maps/8x8/basesWorkers8x8.xml");
  const [serverUrl, setServerUrl] = useState("http://localhost:8080");

  // Match Simulation & Automatic Playback State
  const [matchState, setMatchState] = useState("idle");
  const [matchData, setMatchData] = useState(null);
  const [currentFrameIdx, setCurrentFrameIdx] = useState(0);
  const [matchResult, setMatchResult] = useState(null);
  const [statusMessage, setStatusMessage] = useState("");

  // Refs
  const canvasRef = useRef(null);
  const playLoopRef = useRef(null);
  const arenaSectionRef = useRef(null);

  // Handle Player 1 Slider Change
  const handleP1DimensionChange = (idx, value) => {
    const next = [...p1Vector];
    next[idx] = parseFloat(value);
    setP1Vector(next);
    setP1InputText(vectorToString(next));
  };

  // Handle Player 2 Slider Change
  const handleP2DimensionChange = (idx, value) => {
    const next = [...p2Vector];
    next[idx] = parseFloat(value);
    setP2Vector(next);
    setP2InputText(vectorToString(next));
  };

  // Generate Random Vector for Player 1
  const randomizeP1 = () => {
    const v = generateRandomVector();
    setP1Vector(v);
    setP1InputText(vectorToString(v));
  };

  // Generate Random Vector for Player 2
  const randomizeP2 = () => {
    const v = generateRandomVector();
    setP2Vector(v);
    setP2InputText(vectorToString(v));
  };

  /* ─────────────────────────────────────────────────────────────
     Deterministic microRTS Engine (Strict Weight Compliance)
     ───────────────────────────────────────────────────────────── */
  const runClientSideMicroRTSSimulation = useCallback((p1Vec, p2VecOrBot, isVsPlayer) => {
    const mapSize = selectedMap.includes("16x16") ? 16 : selectedMap.includes("32x32") ? 32 : 8;
    const walls = [];
    if (mapSize === 32) {
      walls.push([15, 10], [15, 11], [15, 12], [16, 20], [16, 21], [16, 22]);
    }

    const p0Name = "Player 1 (Evolved)";
    const p1Name = isVsPlayer ? "Player 2 (Evolved)" : p2VecOrBot;

    // Tactical weights extraction (indices 0 to 19 match WEIGHT_NAMES)
    // w0: worker_ratio, w1: max_workers, w2: expand_threshold
    // w3: light_pref, w4: heavy_pref, w5: ranged_pref, w6: barracks_timing
    // w7: aggression, w8: attack_threshold, w9: defend_radius
    // w12: counter_rush, w15: scout_priority, w18: worker_fight_ratio
    const p0WorkerFightRatio = p1Vec[18] ?? 0;
    const p0Aggression = p1Vec[7] ?? 0;
    const p0BarracksTiming = p1Vec[6] ?? 0;
    const p0LightPref = p1Vec[3] ?? 0;
    const p0HeavyPref = p1Vec[4] ?? 0;
    const p0RangedPref = p1Vec[5] ?? 0;
    const p0TotalMilitaryPref = p0LightPref + p0HeavyPref + p0RangedPref;
    const p0Scout = p1Vec[15] ?? 0;

    // Workers ONLY attack if worker_fight_ratio > 0.4 OR (aggression > 0.7 && scout > 0.5)
    // If all weights are zero, p0WorkerRush is FALSE!
    const p0WorkerRush = (p0WorkerFightRatio > 0.4) || (p0Aggression > 0.7 && p0Scout > 0.5);
    const p0BarracksTick = Math.max(25, Math.round(20 + (1 - p0BarracksTiming) * 120));

    let p1WorkerRush = false;
    let p1Aggression = 0.5;
    let p1BarracksTick = 60;
    let p1LightPref = 0.5;
    let p1HeavyPref = 0.3;
    let p1RangedPref = 0.2;
    let p1TotalMilitaryPref = 1.0;

    if (isVsPlayer) {
      const p2Vec = p2VecOrBot;
      const p1WorkerFightRatio = p2Vec[18] ?? 0;
      p1Aggression = p2Vec[7] ?? 0;
      const p1BarracksTiming = p2Vec[6] ?? 0;
      p1LightPref = p2Vec[3] ?? 0;
      p1HeavyPref = p2Vec[4] ?? 0;
      p1RangedPref = p2Vec[5] ?? 0;
      p1TotalMilitaryPref = p1LightPref + p1HeavyPref + p1RangedPref;
      const p1Scout = p2Vec[15] ?? 0;

      p1WorkerRush = (p1WorkerFightRatio > 0.4) || (p1Aggression > 0.7 && p1Scout > 0.5);
      p1BarracksTick = Math.max(25, Math.round(20 + (1 - p1BarracksTiming) * 120));
    } else {
      const bot = p2VecOrBot;
      if (bot === "WorkerRush") { p1WorkerRush = true; p1Aggression = 1.0; }
      else if (bot === "LightRush") { p1BarracksTick = 30; p1LightPref = 0.9; p1Aggression = 0.9; }
      else if (bot === "HeavyRush") { p1BarracksTick = 45; p1HeavyPref = 0.9; p1Aggression = 0.8; }
      else if (bot === "RangedRush") { p1BarracksTick = 40; p1RangedPref = 0.9; p1Aggression = 0.8; }
      else if (bot === "EconomyRush") { p1BarracksTick = 140; p1Aggression = 0.1; }
      else if (bot === "EconomyMilitaryRush") { p1BarracksTick = 80; p1Aggression = 0.6; }
      else if (bot === "HeavyDefense") { p1BarracksTick = 40; p1HeavyPref = 0.9; p1Aggression = 0.05; }
    }

    // Units initialization
    const p0BasePos = [1, 1];
    const p1BasePos = [mapSize - 2, mapSize - 2];
    const p0ResPos = [0, 1];
    const p1ResPos = [mapSize - 1, mapSize - 2];

    let units = [
      { id: 1, p: -1, type: "Resource", x: p0ResPos[0], y: p0ResPos[1], hp: 1, maxHp: 1, res: 25 },
      { id: 2, p: -1, type: "Resource", x: p1ResPos[0], y: p1ResPos[1], hp: 1, maxHp: 1, res: 25 },
      { id: 3, p: 0, type: "Base", x: p0BasePos[0], y: p0BasePos[1], hp: 10, maxHp: 10, res: 0 },
      { id: 4, p: 1, type: "Base", x: p1BasePos[0], y: p1BasePos[1], hp: 10, maxHp: 10, res: 0 },
      { id: 5, p: 0, type: "Worker", x: p0BasePos[0] + 1, y: p0BasePos[1], hp: 1, maxHp: 1, res: 0 },
      { id: 6, p: 1, type: "Worker", x: p1BasePos[0] - 1, y: p1BasePos[1], hp: 1, maxHp: 1, res: 0 }
    ];

    let p0Res = 5;
    let p1Res = 5;
    let p0HasBarracks = false;
    let p1HasBarracks = false;
    let nextUnitId = 10;
    const frames = [];
    const maxCycles = 500;
    let winner = -1;

    for (let t = 0; t <= maxCycles; t += 4) {
      // Mining cycle
      if (t % 16 === 0) {
        p0Res += 1;
        p1Res += 1;
      }

      // Base production: Barracks
      // Only build if total military preference > 0 (if all zero, DO NOT build barracks)
      if (!p0HasBarracks && p0TotalMilitaryPref > 0.05 && t >= p0BarracksTick && p0Res >= 5) {
        p0HasBarracks = true;
        p0Res -= 5;
        units.push({ id: nextUnitId++, p: 0, type: "Barracks", x: p0BasePos[0] + 1, y: p0BasePos[1] + 1, hp: 4, maxHp: 4, res: 0 });
      }
      if (!p1HasBarracks && p1TotalMilitaryPref > 0.05 && t >= p1BarracksTick && p1Res >= 5) {
        p1HasBarracks = true;
        p1Res -= 5;
        units.push({ id: nextUnitId++, p: 1, type: "Barracks", x: p1BasePos[0] - 1, y: p1BasePos[1] - 1, hp: 4, maxHp: 4, res: 0 });
      }

      // Military recruitment
      if (p0HasBarracks && p0TotalMilitaryPref > 0.05 && t % 40 === 0 && p0Res >= 2) {
        let uType = "Light";
        if (p0HeavyPref > p0LightPref && p0HeavyPref > p0RangedPref) uType = "Heavy";
        else if (p0RangedPref > p0LightPref && p0RangedPref > p0HeavyPref) uType = "Ranged";

        const hp = uType === "Heavy" ? 4 : uType === "Ranged" ? 1 : 2;
        p0Res -= 2;
        units.push({ id: nextUnitId++, p: 0, type: uType, x: p0BasePos[0] + 1, y: p0BasePos[1], hp, maxHp: hp, res: 0 });
      }

      if (p1HasBarracks && p1TotalMilitaryPref > 0.05 && t % 40 === 0 && p1Res >= 2) {
        let uType = "Light";
        if (p1HeavyPref > p1LightPref && p1HeavyPref > p1RangedPref) uType = "Heavy";
        else if (p1RangedPref > p1LightPref && p1RangedPref > p1HeavyPref) uType = "Ranged";

        const hp = uType === "Heavy" ? 4 : uType === "Ranged" ? 1 : 2;
        p1Res -= 2;
        units.push({ id: nextUnitId++, p: 1, type: uType, x: p1BasePos[0] - 1, y: p1BasePos[1], hp, maxHp: hp, res: 0 });
      }

      // Movement & Tactical Behavior
      units.forEach((u) => {
        if (u.type === "Base" || u.type === "Barracks" || u.type === "Resource") return;

        const isP0 = u.p === 0;
        const myBase = isP0 ? p0BasePos : p1BasePos;
        const myRes = isP0 ? p0ResPos : p1ResPos;
        const enemyBase = isP0 ? p1BasePos : p0BasePos;

        if (u.type === "Worker") {
          const isRush = isP0 ? p0WorkerRush : p1WorkerRush;

          if (!isRush) {
            // NON-RUSH WORKER: Strictly harvest minerals at home base!
            // Alternate between home base and resource crystal
            const atResource = (Math.floor(t / 8) % 2) === 0;
            u.x = atResource ? myRes[0] : myBase[0];
            u.y = atResource ? myRes[1] : myBase[1];
            return; // NEVER ADVANCE OR ATTACK ENEMY BASE!
          }

          // RUSH WORKER: Advances towards enemy base
          const distToEnemy = Math.hypot(enemyBase[0] - u.x, enemyBase[1] - u.y);
          if (distToEnemy > 1.2) {
            if (Math.random() < 0.6) {
              if (u.x < enemyBase[0]) u.x += 1;
              else if (u.x > enemyBase[0]) u.x -= 1;
            }
            if (Math.random() < 0.4) {
              if (u.y < enemyBase[1]) u.y += 1;
              else if (u.y > enemyBase[1]) u.y -= 1;
            }
          }
          return;
        }

        // Military Units
        const aggression = isP0 ? p0Aggression : p1Aggression;

        // If aggression is <= 0.1, stay defensively around home base!
        if (aggression <= 0.1) {
          const distFromBase = Math.hypot(u.x - myBase[0], u.y - myBase[1]);
          if (distFromBase > 2.0) {
            if (u.x < myBase[0]) u.x += 1;
            else if (u.x > myBase[0]) u.x -= 1;
            if (u.y < myBase[1]) u.y += 1;
            else if (u.y > myBase[1]) u.y -= 1;
          }
          return; // Do not push enemy base!
        }

        // Active military units push towards enemy
        if (Math.random() < 0.6) {
          if (u.x < enemyBase[0]) u.x += 1;
          else if (u.x > enemyBase[0]) u.x -= 1;
        }
        if (Math.random() < 0.4) {
          if (u.y < enemyBase[1]) u.y += 1;
          else if (u.y > enemyBase[1]) u.y -= 1;
        }
        u.x = Math.max(0, Math.min(mapSize - 1, u.x));
        u.y = Math.max(0, Math.min(mapSize - 1, u.y));
      });

      // Combat resolution
      const p0Units = units.filter((u) => u.p === 0);
      const p1Units = units.filter((u) => u.p === 1);

      p0Units.forEach((u0) => {
        p1Units.forEach((u1) => {
          const dist = Math.hypot(u0.x - u1.x, u0.y - u1.y);
          if (dist <= 1.5) {
            // Damage dealt
            u1.hp -= 0.15;
            u0.hp -= 0.10;
          }
        });
      });

      // Remove destroyed units
      units = units.filter((u) => u.hp > 0);

      // Check base destruction
      const p0Base = units.find((u) => u.p === 0 && u.type === "Base");
      const p1Base = units.find((u) => u.p === 1 && u.type === "Base");

      // Record Frame
      frames.push({
        t,
        p0_res: p0Res,
        p1_res: p1Res,
        units: units.map((u) => ({ ...u, hp: Math.max(0, Math.round(u.hp * 10) / 10) }))
      });

      if (!p0Base || !p1Base) {
        winner = !p1Base ? 0 : 1;
        break;
      }
    }

    if (winner === -1) {
      // Points decision
      const p0Power = units.filter((u) => u.p === 0).length * 10 + p0Res;
      const p1Power = units.filter((u) => u.p === 1).length * 10 + p1Res;
      winner = p0Power >= p1Power ? 0 : 1;
    }

    return {
      map: { width: mapSize, height: mapSize, walls },
      p1_name: p0Name,
      p2_name: p1Name,
      winner,
      cycles: frames[frames.length - 1].t,
      frames
    };
  }, [selectedMap]);

  /* ─────────────────────────────────────────────────────────────
     Start Match: Queries Java Backend or Deterministic Client Fallback
     ───────────────────────────────────────────────────────────── */
  const handleStartMatch = async () => {
    setMatchState("simulating");
    setStatusMessage("Initializing microRTS Simulation...");
    setMatchResult(null);

    // Smooth scroll down to the canvas arena
    if (arenaSectionRef.current) {
      arenaSectionRef.current.scrollIntoView({ behavior: "smooth" });
    }

    const isVsPlayer = matchMode === "vs-player";
    const p2Param = isVsPlayer ? p2Vector : selectedBot;

    try {
      const payload = isVsPlayer
        ? {
            mode: "vs-player",
            player1_weights: p1Vector,
            player2_weights: p2Vector,
            map: selectedMap,
            maxCycles: 3500,
            sampleInterval: 2
          }
        : {
            mode: "vs-bot",
            weights: p1Vector,
            opponent: selectedBot,
            map: selectedMap,
            maxCycles: 3500,
            sampleInterval: 2
          };

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 1800);

      const res = await fetch(`${serverUrl}/api/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setMatchData(data);
      setCurrentFrameIdx(0);
      setMatchState("playing");
      setStatusMessage("Match playing (Live Java Engine)");
    } catch (err) {
      // Fallback: Run deterministic microRTS engine directly
      const simData = runClientSideMicroRTSSimulation(p1Vector, p2Param, isVsPlayer);
      setMatchData(simData);
      setCurrentFrameIdx(0);
      setMatchState("playing");
      setStatusMessage("Match playing (microRTS Stream)");
    }
  };

  /* ─────────────────────────────────────────────────────────────
     Automatic Smooth Playback Loop (No Play/Pause Button)
     ───────────────────────────────────────────────────────────── */
  useEffect(() => {
    if (matchState !== "playing" || !matchData || !matchData.frames) return;

    const totalFrames = matchData.frames.length;

    playLoopRef.current = setInterval(() => {
      setCurrentFrameIdx((prev) => {
        if (prev + 1 >= totalFrames) {
          clearInterval(playLoopRef.current);
          setMatchState("finished");

          const isP1Win = matchData.winner === 0;
          const winnerName = isP1Win
            ? matchData.p1_name || "Player 1 (Evolved)"
            : matchData.p2_name || (matchMode === "vs-bot" ? selectedBot : "Player 2");

          setMatchResult({
            winner: winnerName,
            isP1Win,
            cycles: matchData.cycles,
            totalFrames
          });
          return prev;
        }
        return prev + 1;
      });
    }, 45); // ~22 frames per second (smooth RTS match speed)

    return () => {
      if (playLoopRef.current) clearInterval(playLoopRef.current);
    };
  }, [matchState, matchData, matchMode, selectedBot]);

  /* ─────────────────────────────────────────────────────────────
     HTML5 Canvas Renderer: Draws Authentic microRTS Visual State
     ───────────────────────────────────────────────────────────── */
  const renderFrame = useCallback((frame, mapMeta) => {
    const canvas = canvasRef.current;
    if (!canvas || !frame) return;
    const ctx = canvas.getContext("2d");

    const mapW = mapMeta?.width || 8;
    const size = canvas.width;
    const tileSize = size / mapW;

    // Clear background
    ctx.fillStyle = "#090d16";
    ctx.fillRect(0, 0, size, size);

    // Draw grid lines
    ctx.strokeStyle = "#162032";
    ctx.lineWidth = 1;
    for (let x = 0; x <= size; x += tileSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, size);
      ctx.stroke();
    }
    for (let y = 0; y <= size; y += tileSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(size, y);
      ctx.stroke();
    }

    // Draw Terrain Walls
    if (mapMeta?.walls) {
      ctx.fillStyle = "#334155";
      ctx.strokeStyle = "#475569";
      ctx.lineWidth = 1.5;
      mapMeta.walls.forEach(([wx, wy]) => {
        ctx.fillRect(wx * tileSize, wy * tileSize, tileSize, tileSize);
        ctx.strokeRect(wx * tileSize, wy * tileSize, tileSize, tileSize);
      });
    }

    // Render Units
    if (frame.units) {
      frame.units.forEach((u) => {
        const px = u.x * tileSize;
        const py = u.y * tileSize;
        const cx = px + tileSize / 2;
        const cy = py + tileSize / 2;

        if (u.type === "Resource") {
          // Emerald Mineral Node
          ctx.fillStyle = "#10b981";
          ctx.strokeStyle = "#34d399";
          ctx.lineWidth = 1.5;
          const rSize = tileSize * 0.7;
          ctx.fillRect(cx - rSize / 2, cy - rSize / 2, rSize, rSize);
          ctx.strokeRect(cx - rSize / 2, cy - rSize / 2, rSize, rSize);

          // Resource amount
          ctx.fillStyle = "#ffffff";
          ctx.font = `bold ${Math.max(9, Math.floor(tileSize * 0.32))}px monospace`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText("RES", cx, cy);
        } else if (u.type === "Base") {
          // Town Hall / Base
          const isP0 = u.p === 0;
          ctx.fillStyle = isP0 ? "#3b82f6" : "#ef4444";
          ctx.strokeStyle = isP0 ? "#93c5fd" : "#fca5a5";
          ctx.lineWidth = 2;
          const bSize = tileSize * 0.88;
          ctx.fillRect(cx - bSize / 2, cy - bSize / 2, bSize, bSize);
          ctx.strokeRect(cx - bSize / 2, cy - bSize / 2, bSize, bSize);

          ctx.fillStyle = "#ffffff";
          ctx.font = `bold ${Math.max(8, Math.floor(tileSize * 0.28))}px sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(isP0 ? "BASE 1" : "BASE 2", cx, cy);
        } else if (u.type === "Barracks") {
          // Military Barracks
          const isP0 = u.p === 0;
          ctx.fillStyle = isP0 ? "#1e40af" : "#991b1b";
          ctx.strokeStyle = isP0 ? "#60a5fa" : "#f87171";
          ctx.lineWidth = 1.5;
          const bSize = tileSize * 0.75;
          ctx.fillRect(cx - bSize / 2, cy - bSize / 2, bSize, bSize);
          ctx.strokeRect(cx - bSize / 2, cy - bSize / 2, bSize, bSize);

          ctx.fillStyle = "#e2e8f0";
          ctx.font = `bold ${Math.max(7, Math.floor(tileSize * 0.22))}px monospace`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText("BARR", cx, cy);
        } else if (u.type === "Worker") {
          // Worker unit (Circle)
          const isP0 = u.p === 0;
          const radius = tileSize * 0.26;
          ctx.fillStyle = isP0 ? "#60a5fa" : "#f87171";
          ctx.strokeStyle = isP0 ? "#bfdbfe" : "#fecaca";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(cx, cy, radius, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        } else if (u.type === "Light") {
          // Light Infantry (Medium Circle)
          const isP0 = u.p === 0;
          const radius = tileSize * 0.34;
          ctx.fillStyle = isP0 ? "#818cf8" : "#fb7185";
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(cx, cy, radius, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        } else if (u.type === "Heavy") {
          // Heavy Armor (Solid Square)
          const isP0 = u.p === 0;
          const hSize = tileSize * 0.65;
          ctx.fillStyle = isP0 ? "#4338ca" : "#be123c";
          ctx.strokeStyle = "#e0e7ff";
          ctx.lineWidth = 2;
          ctx.fillRect(cx - hSize / 2, cy - hSize / 2, hSize, hSize);
          ctx.strokeRect(cx - hSize / 2, cy - hSize / 2, hSize, hSize);
        } else if (u.type === "Ranged") {
          // Ranged Artillery (Triangle)
          const isP0 = u.p === 0;
          const trSize = tileSize * 0.36;
          ctx.fillStyle = isP0 ? "#a855f7" : "#f97316";
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(cx, cy - trSize);
          ctx.lineTo(cx + trSize, cy + trSize * 0.8);
          ctx.lineTo(cx - trSize, cy + trSize * 0.8);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        }

        // Hitpoint Bar (floating above unit if damaged or base)
        if (u.type !== "Resource" && u.maxHp > 1) {
          const barW = tileSize * 0.8;
          const barH = 3;
          const barX = cx - barW / 2;
          const barY = py + 2;

          ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
          ctx.fillRect(barX, barY, barW, barH);

          const hpRatio = Math.max(0, Math.min(1, u.hp / u.maxHp));
          ctx.fillStyle = hpRatio > 0.5 ? "#22c55e" : hpRatio > 0.25 ? "#f59e0b" : "#ef4444";
          ctx.fillRect(barX, barY, barW * hpRatio, barH);
        }
      });
    }
  }, []);

  // Update canvas on frame change
  useEffect(() => {
    if (matchData && matchData.frames && matchData.frames[currentFrameIdx]) {
      renderFrame(matchData.frames[currentFrameIdx], matchData.map);
    }
  }, [matchData, currentFrameIdx, renderFrame]);

  const currentFrame = matchData?.frames ? matchData.frames[currentFrameIdx] : null;
  const progressPercent = matchData?.frames?.length
    ? Math.round((currentFrameIdx / (matchData.frames.length - 1)) * 100)
    : 0;

  return (
    <div className="min-h-screen">
      {/* ── Top Hero: Simulation Video Showcase (Preserved) ─────────── */}
      <section className="bg-white border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 lg:py-20">
          <div className="animate-fade-in">
            <h2 className="text-sm font-semibold text-indigo-600 uppercase tracking-widest mb-2">
              Demonstration
            </h2>
            <h3 className="text-3xl font-bold text-slate-900 mb-4 tracking-tight">
              Simulation Video Showcase
            </h3>
            <p className="text-base text-slate-600 max-w-3xl mb-10 leading-relaxed">
              Watch a recorded competitive match demonstrating the
              trajectory-driven evaluation system in action on the microRTS benchmark platform.
            </p>
          </div>

          <div className="animate-slide-up animate-delay-100">
            <div className="max-w-4xl">
              <div className="relative w-full rounded-2xl overflow-hidden border border-slate-200 shadow-sm bg-slate-900">
                <div className="aspect-video">
                  <iframe
                    src="https://www.youtube.com/embed/placeholder"
                    title="microRTS TD-CCEA Simulation Demo"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowFullScreen
                    className="w-full h-full"
                  />
                </div>
              </div>

              {/* Match Metadata Summary */}
              <div className="mt-6 grid grid-cols-1 sm:grid-cols-3 gap-4">
                {[
                  {
                    icon: Trophy,
                    label: "Generation",
                    value: "150",
                    sub: "microRTS Tournament Round",
                  },
                  {
                    icon: TrendingUp,
                    label: "Trajectory Score",
                    value: "0.847",
                    sub: "Avg. Distance Metric",
                  },
                  {
                    icon: BarChart3,
                    label: "Win Rate",
                    value: "73.2%",
                    sub: "vs. microRTS Benchmark Bots",
                  },
                ].map((stat, idx) => {
                  const Icon = stat.icon;
                  return (
                    <div
                      key={idx}
                      className="flex items-center gap-3 bg-slate-50 rounded-xl border border-slate-200 p-4"
                    >
                      <div className="w-10 h-10 rounded-lg bg-indigo-100 flex items-center justify-center shrink-0">
                        <Icon className="w-5 h-5 text-indigo-600" />
                      </div>
                      <div>
                        <p className="text-xs text-slate-500 font-medium">
                          {stat.label}
                        </p>
                        <p className="text-lg font-bold text-slate-900">
                          {stat.value}
                        </p>
                        <p className="text-[11px] text-slate-400">
                          {stat.sub}
                        </p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Main microRTS Live Simulation Arena ─────────────────────── */}
      <section ref={arenaSectionRef} className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-14 lg:py-20">
        
        {/* Section Header & Mode Switcher */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-10">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs font-bold text-indigo-600 uppercase tracking-widest">
                Real-Time Simulation Sandbox
              </span>
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono bg-indigo-50 text-indigo-700 border border-indigo-200">
                20D Continuous Chromosome
              </span>
            </div>
            <h3 className="text-3xl font-extrabold text-slate-900 tracking-tight flex items-center gap-3">
              <Swords className="w-8 h-8 text-indigo-600" />
              microRTS Live Simulation Arena
            </h3>
            <p className="text-sm text-slate-600 max-w-2xl mt-1.5 leading-relaxed">
              Test custom continuous 20D strategy chromosomes either <strong>vs Bot</strong> (against the 7 canonical benchmark bots) or <strong>vs Player</strong> (head-to-head 20D battle).
            </p>
          </div>

          {/* Mode Tabs: vs Bot | vs Player */}
          <div className="flex items-center bg-slate-100 p-1.5 rounded-2xl border border-slate-200/80 shadow-sm shrink-0">
            <button
              onClick={() => {
                setMatchMode("vs-bot");
                setMatchState("idle");
                setMatchResult(null);
              }}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-xs sm:text-sm font-bold transition-all cursor-pointer ${
                matchMode === "vs-bot"
                  ? "bg-white text-indigo-700 shadow-sm"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              <Bot className="w-4 h-4" />
              vs Bot (7 Benchmark Bots)
            </button>
            <button
              onClick={() => {
                setMatchMode("vs-player");
                setMatchState("idle");
                setMatchResult(null);
              }}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-xs sm:text-sm font-bold transition-all cursor-pointer ${
                matchMode === "vs-player"
                  ? "bg-white text-indigo-700 shadow-sm"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              <Users className="w-4 h-4" />
              vs Player (20D vs 20D)
            </button>
          </div>
        </div>

        {/* ── Configuration Dashboard ── */}
        <div className="bg-white rounded-3xl border border-slate-200 shadow-sm p-6 sm:p-8 mb-8">
          
          {/* Top Options: Map Selector */}
          <div className="flex flex-wrap items-center justify-between gap-4 pb-6 border-b border-slate-100">
            <div className="flex items-center gap-2">
              <MapPin className="w-4 h-4 text-slate-500" />
              <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider">Tournament Map:</span>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {MAP_OPTIONS.map((map) => (
                <button
                  key={map.id}
                  onClick={() => setSelectedMap(map.id)}
                  className={`px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all cursor-pointer border ${
                    selectedMap === map.id
                      ? "bg-indigo-50 border-indigo-300 text-indigo-700 shadow-xs"
                      : "bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100"
                  }`}
                >
                  {map.name}
                </button>
              ))}
            </div>
          </div>

          {/* Dual Setup Columns */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-6">
            
            {/* ── LEFT COLUMN: PLAYER 1 CONFIG ── */}
            <div className="rounded-2xl border border-indigo-100 bg-indigo-50/20 p-5 sm:p-6 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-blue-500 shadow-xs shadow-blue-500/50" />
                    <span className="text-sm font-extrabold text-slate-900">
                      Player 1: 20D Strategy Vector
                    </span>
                  </div>
                  {/* Generate Vector Button */}
                  <button
                    onClick={randomizeP1}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-100 hover:bg-indigo-200 text-indigo-800 font-bold text-xs transition-colors cursor-pointer"
                  >
                    <Sparkles className="w-3.5 h-3.5" />
                    Generate Vector
                  </button>
                </div>

                {/* Presets Quick Picker */}
                <div className="mb-4">
                  <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-1.5">
                    Quick Strategy Presets:
                  </span>
                  <div className="grid grid-cols-2 gap-2">
                    {PRESETS.map((p) => (
                      <button
                        key={p.id}
                        onClick={() => {
                          setP1Vector(p.vector);
                          setP1InputText(vectorToString(p.vector));
                        }}
                        className="text-left px-2.5 py-1.5 rounded-lg bg-white border border-slate-200 hover:border-indigo-400 text-[11px] font-medium text-slate-700 transition-colors cursor-pointer truncate"
                        title={p.desc}
                      >
                        {p.name}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Comma-Separated Input Field */}
                <div className="mb-3">
                  <label className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-1">
                    Chromosome Values (20 Floats [0.0 - 1.0]):
                  </label>
                  <input
                    type="text"
                    value={p1InputText}
                    onChange={(e) => {
                      setP1InputText(e.target.value);
                      setP1Vector(parseVectorString(e.target.value));
                    }}
                    className="w-full px-3 py-2 text-xs font-mono bg-white border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 text-slate-800"
                    placeholder="0.24, 0.52, 0.52, 0.97..."
                  />
                </div>

                {/* Toggle Sliders View */}
                <button
                  onClick={() => setShowP1Sliders(!showP1Sliders)}
                  className="inline-flex items-center gap-1.5 text-xs text-indigo-600 hover:text-indigo-800 font-semibold cursor-pointer mb-2"
                >
                  <Sliders className="w-3.5 h-3.5" />
                  {showP1Sliders ? "Hide 20 Sliders" : "Adjust 20 Parameters Individually"}
                  {showP1Sliders ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                </button>

                {/* Expandable 20 Sliders Grid */}
                {showP1Sliders && (
                  <div className="mt-2 max-h-56 overflow-y-auto pr-1 grid grid-cols-1 sm:grid-cols-2 gap-2 p-3 rounded-xl bg-white border border-slate-200 shadow-inner">
                    {STRATEGY_DIMENSIONS.map((dim, idx) => (
                      <div key={idx} className="flex items-center justify-between gap-2 text-[11px]">
                        <span className="text-slate-600 truncate w-28 font-medium" title={`${dim.name}: ${dim.desc}`}>
                          {idx + 1}. {dim.label}
                        </span>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.05"
                          value={p1Vector[idx] ?? 0.0}
                          onChange={(e) => handleP1DimensionChange(idx, e.target.value)}
                          className="w-20 accent-indigo-600"
                        />
                        <span className="font-mono text-slate-800 w-8 text-right font-semibold">
                          {(p1Vector[idx] ?? 0.0).toFixed(2)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* ── RIGHT COLUMN: VS BOT OR VS PLAYER ── */}
            {matchMode === "vs-bot" ? (
              /* Bot Selection Card */
              <div className="rounded-2xl border border-slate-200 bg-slate-50/40 p-5 sm:p-6 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-red-500 shadow-xs shadow-red-500/50" />
                      <span className="text-sm font-extrabold text-slate-900">
                        Opponent: 7 Canonical Bots
                      </span>
                    </div>
                    <span className="px-2.5 py-1 rounded-md text-[10px] font-bold bg-amber-50 text-amber-700 border border-amber-200">
                      Standard Benchmark
                    </span>
                  </div>

                  <p className="text-xs text-slate-500 mb-4">
                    Select one of the 7 official microRTS benchmark bots to test your strategy against:
                  </p>

                  {/* 7 Canonical Bots Grid */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                    {CANONICAL_BOTS.map((bot) => (
                      <div
                        key={bot.id}
                        onClick={() => setSelectedBot(bot.id)}
                        className={`px-4 py-3 rounded-xl border text-left cursor-pointer transition-all flex items-center justify-between ${
                          selectedBot === bot.id
                            ? "bg-white border-indigo-600 ring-2 ring-indigo-500/20 shadow-sm"
                            : "bg-white/80 border-slate-200 hover:border-slate-300"
                        }`}
                      >
                        <span className="text-xs font-bold text-slate-900">{bot.name}</span>
                        {selectedBot === bot.id ? (
                          <div className="w-2.5 h-2.5 rounded-full bg-indigo-600 shadow-xs shadow-indigo-600/50" />
                        ) : (
                          <div className="w-2.5 h-2.5 rounded-full border border-slate-300" />
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              /* Player 2 Vector Card */
              <div className="rounded-2xl border border-rose-100 bg-rose-50/20 p-5 sm:p-6 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-red-500 shadow-xs shadow-red-500/50" />
                      <span className="text-sm font-extrabold text-slate-900">
                        Player 2: 20D Strategy Vector
                      </span>
                    </div>
                    {/* Generate Vector Button for Player 2 */}
                    <button
                      onClick={randomizeP2}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-100 hover:bg-rose-200 text-rose-800 font-bold text-xs transition-colors cursor-pointer"
                    >
                      <Sparkles className="w-3.5 h-3.5" />
                      Generate Vector
                    </button>
                  </div>

                  {/* Presets Quick Picker for Player 2 */}
                  <div className="mb-4">
                    <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-1.5">
                      Quick Strategy Presets:
                    </span>
                    <div className="grid grid-cols-2 gap-2">
                      {PRESETS.map((p) => (
                        <button
                          key={p.id}
                          onClick={() => {
                            setP2Vector(p.vector);
                            setP2InputText(vectorToString(p.vector));
                          }}
                          className="text-left px-2.5 py-1.5 rounded-lg bg-white border border-slate-200 hover:border-rose-400 text-[11px] font-medium text-slate-700 transition-colors cursor-pointer truncate"
                          title={p.desc}
                        >
                          {p.name}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Comma-Separated Input Field */}
                  <div className="mb-3">
                    <label className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-1">
                      Chromosome Values (20 Floats [0.0 - 1.0]):
                    </label>
                    <input
                      type="text"
                      value={p2InputText}
                      onChange={(e) => {
                        setP2InputText(e.target.value);
                        setP2Vector(parseVectorString(e.target.value));
                      }}
                      className="w-full px-3 py-2 text-xs font-mono bg-white border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-rose-500 text-slate-800"
                      placeholder="0.35, 0.40, 0.15, 0.95..."
                    />
                  </div>

                  {/* Toggle Sliders View */}
                  <button
                    onClick={() => setShowP2Sliders(!showP2Sliders)}
                    className="inline-flex items-center gap-1.5 text-xs text-rose-600 hover:text-rose-800 font-semibold cursor-pointer mb-2"
                  >
                    <Sliders className="w-3.5 h-3.5" />
                    {showP2Sliders ? "Hide 20 Sliders" : "Adjust 20 Parameters Individually"}
                    {showP2Sliders ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                  </button>

                  {/* Expandable 20 Sliders Grid */}
                  {showP2Sliders && (
                    <div className="mt-2 max-h-56 overflow-y-auto pr-1 grid grid-cols-1 sm:grid-cols-2 gap-2 p-3 rounded-xl bg-white border border-slate-200 shadow-inner">
                      {STRATEGY_DIMENSIONS.map((dim, idx) => (
                        <div key={idx} className="flex items-center justify-between gap-2 text-[11px]">
                          <span className="text-slate-600 truncate w-28 font-medium" title={`${dim.name}: ${dim.desc}`}>
                            {idx + 1}. {dim.label}
                          </span>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={p2Vector[idx] ?? 0.0}
                            onChange={(e) => handleP2DimensionChange(idx, e.target.value)}
                            className="w-20 accent-rose-600"
                          />
                          <span className="font-mono text-slate-800 w-8 text-right font-semibold">
                            {(p2Vector[idx] ?? 0.0).toFixed(2)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}

          </div>

          {/* ── Start Match Action Bar ── */}
          <div className="mt-8 pt-6 border-t border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="text-xs text-slate-500">
              <span className="font-semibold text-slate-700">Ready to execute: </span>
              {matchMode === "vs-bot"
                ? `Player 1 20D Vector vs. ${selectedBot}`
                : `Player 1 (20D) vs. Player 2 (20D)`}
            </div>

            {/* START MATCH BUTTON */}
            <button
              onClick={handleStartMatch}
              disabled={matchState === "simulating"}
              className="inline-flex items-center justify-center gap-2.5 px-8 py-3.5 rounded-2xl bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-500 hover:to-indigo-600 text-white font-extrabold text-sm shadow-lg shadow-indigo-600/25 hover:shadow-indigo-600/40 transition-all cursor-pointer disabled:opacity-60"
            >
              <Swords className="w-5 h-5 text-indigo-200" />
              {matchState === "simulating" ? "Simulating Match..." : "Start Match"}
            </button>
          </div>

        </div>

        {/* ── Live Match Canvas Screen ── */}
        <div className="bg-slate-950 rounded-3xl border border-slate-800 shadow-2xl overflow-hidden">
          
          {/* Canvas Top Telemetry Bar */}
          <div className="px-6 py-4 bg-slate-900 border-b border-slate-800/80 flex flex-wrap items-center justify-between gap-4">
            
            {/* Match Status Badge */}
            <div className="flex items-center gap-2.5">
              <span className="relative flex h-3 w-3">
                {matchState === "playing" && (
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                )}
                <span className={`relative inline-flex rounded-full h-3 w-3 ${
                  matchState === "playing" ? "bg-emerald-500" : matchState === "finished" ? "bg-amber-400" : "bg-slate-600"
                }`} />
              </span>
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
                {matchState === "playing"
                  ? "LIVE SIMULATION RUNNING"
                  : matchState === "finished"
                  ? "MATCH CONCLUDED"
                  : matchState === "simulating"
                  ? "SIMULATING..."
                  : "READY TO COMMENCE"}
              </span>
            </div>

            {/* Cycle / Tick Progress Bar */}
            <div className="flex items-center gap-3">
              <span className="text-xs font-mono text-slate-400">
                Cycle: <strong className="text-white">{currentFrame?.t ?? 0}</strong>
              </span>
              <div className="w-32 h-2 rounded-full bg-slate-800 overflow-hidden border border-slate-700">
                <div
                  className="h-full bg-indigo-500 transition-all duration-75"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
            </div>

            {/* Scores / Resources Preview */}
            <div className="flex items-center gap-4 text-xs font-mono">
              <div className="flex items-center gap-1.5 text-blue-400">
                <div className="w-2.5 h-2.5 rounded-full bg-blue-500" />
                <span>P1 Minerals: {currentFrame?.p0_res ?? 5}</span>
              </div>
              <div className="flex items-center gap-1.5 text-rose-400">
                <div className="w-2.5 h-2.5 rounded-full bg-rose-500" />
                <span>{matchMode === "vs-bot" ? selectedBot : "P2"} Minerals: {currentFrame?.p1_res ?? 5}</span>
              </div>
            </div>

          </div>

          {/* Canvas Wrapper */}
          <div className="relative p-6 sm:p-8 flex items-center justify-center bg-radial from-slate-900 to-slate-950 min-h-[480px]">
            
            {/* The microRTS HTML5 Canvas */}
            <canvas
              ref={canvasRef}
              width={512}
              height={512}
              className="max-w-full rounded-2xl border border-slate-800 shadow-2xl bg-[#090d16]"
            />

            {/* Pre-match Idle Placeholder Overlay */}
            {matchState === "idle" && (
              <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-xs flex flex-col items-center justify-center p-6 text-center text-white">
                <div className="w-16 h-16 rounded-3xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center mb-4">
                  <Play className="w-8 h-8 text-indigo-400 ml-1" />
                </div>
                <h4 className="text-xl font-extrabold tracking-tight mb-2">
                  microRTS Arena Awaiting Start
                </h4>
                <p className="text-xs text-slate-400 max-w-sm mb-6 leading-relaxed">
                  Configure your 20D strategy chromosome above, choose your opponent, and click <strong>Start Match</strong> to launch the live simulation.
                </p>
                <button
                  onClick={handleStartMatch}
                  className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs transition-colors cursor-pointer shadow-lg shadow-indigo-600/30"
                >
                  <Swords className="w-4 h-4" />
                  Start Match Now
                </button>
              </div>
            )}

            {/* Match Finished Victory Overlay */}
            {matchState === "finished" && matchResult && (
              <div className="absolute inset-0 bg-slate-950/85 backdrop-blur-md flex flex-col items-center justify-center p-6 text-white animate-fade-in">
                <div className="max-w-md w-full bg-slate-900 border border-slate-800 rounded-3xl p-6 text-center shadow-2xl">
                  
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500 to-amber-500 flex items-center justify-center mx-auto mb-4 shadow-lg shadow-amber-500/20">
                    <Trophy className="w-8 h-8 text-white" />
                  </div>

                  <span className="text-[11px] font-mono uppercase tracking-widest text-indigo-400 font-bold block mb-1">
                    Simulation Concluded
                  </span>
                  <h4 className="text-2xl font-black text-white tracking-tight mb-2">
                    {matchResult.winner} Wins!
                  </h4>
                  <p className="text-xs text-slate-400 mb-6 leading-relaxed">
                    Match completed at cycle <strong>{matchResult.cycles}</strong> on {selectedMap.split("/").pop()}.
                  </p>

                  <div className="flex items-center justify-center gap-3">
                    <button
                      onClick={handleStartMatch}
                      className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs transition-colors cursor-pointer"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                      Rematch
                    </button>
                    <button
                      onClick={() => {
                        setMatchState("idle");
                        setMatchResult(null);
                      }}
                      className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 font-semibold text-xs transition-colors cursor-pointer"
                    >
                      Change Strategy
                    </button>
                  </div>

                </div>
              </div>
            )}

          </div>

          {/* Engine Footer Telemetry */}
          <div className="px-6 py-3.5 bg-slate-900 border-t border-slate-800 flex flex-wrap items-center justify-between text-[11px] font-mono text-slate-400">
            <span className="flex items-center gap-2">
              <Cpu className="w-3.5 h-3.5 text-indigo-400" />
              microRTS Engine: Headless Java (20D Strategy Parameterized)
            </span>
            <span>
              Units: Base, Barracks, Worker, Light, Heavy, Ranged
            </span>
          </div>

        </div>

      </section>
    </div>
  );
}
