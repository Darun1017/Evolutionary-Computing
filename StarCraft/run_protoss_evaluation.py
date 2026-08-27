"""
CCE-TDCCEA Protoss Evaluation Runner
=====================================
Runs a single SC2 game (Protoss vs Zerg) using ProtossMacroBot,
logs all metrics to a JSON file, and prints comparison table.
Usage:
    python run_protoss_evaluation.py
    python run_protoss_evaluation.py --difficulty 5 --map AltitudeLE
"""
import argparse
import json
import time
import os
import sys
from datetime import datetime

from sc2 import maps
from sc2.player import Bot, Computer
from sc2.main import run_game
from sc2.data import Race, Difficulty, Result

from protoss_macro_bot import ProtossMacroBot

PAPER_MAPS = ["AltitudeLE", "AncientCisternLE"]

# Corrected BLZ difficulty mapping matching TextStarCraft II paper Table 5.
# Paper Level 5 = BLZ "Harder" = Difficulty.Harder (primary comparison level).
# BurnySC2 has no Difficulty.Elite; Level 7 uses VeryHard as closest available.
DIFFICULTY_MAP = {
    1: Difficulty.VeryEasy,
    2: Difficulty.Easy,
    3: Difficulty.Medium,
    4: Difficulty.Hard,           # Paper Level 4 = BLZ "Hard"
    5: Difficulty.Harder,         # Paper Level 5 = BLZ "Harder" (primary comparison)
    6: Difficulty.VeryHard,       # Paper Level 6 = BLZ "Very Hard"
    7: Difficulty.VeryHard,       # Paper Level 7 = BLZ "Elite" (not in BurnySC2, VeryHard closest)
}

EVOLVED_VECTOR = [
    0.80, 0.30, 0.40, 0.75, 0.70, 0.55, 0.65,
    0.45, 0.80, 0.50, 0.35, 0.30, 0.50, 0.30,
    0.75, 0.40, 0.45, 0.65, 0.55, 0.70, 0.60,
    0.70, 0.25,
]

TABLE_1_BASELINES = [
    {"model": "GPT3.5-Turbo",     "wins": "11/20", "pct": 55.0, "pbr": 0.0781, "rur": 7875,  "apu": 0.7608, "tr": 0.4476},
    {"model": "GPT4-Turbo",       "wins": "12/20", "pct": 60.0, "pbr": 0.0337, "rur": 8306,  "apu": 0.7194, "tr": 0.3452},
    {"model": "Gemini-Pro",       "wins": "5/20",  "pct": 25.0, "pbr": 0.0318, "rur": 9284,  "apu": 0.6611, "tr": 0.3571},
    {"model": "GLM4",             "wins": "4/20",  "pct": 20.0, "pbr": 0.0327, "rur": 3131,  "apu": 0.6644, "tr": 0.2904},
    {"model": "Claude 2.1",       "wins": "3/20",  "pct": 15.0, "pbr": 0.0219, "rur": 10867, "apu": 0.6599, "tr": 0.4312},
    {"model": "Qwen-7B (FT)",     "wins": "9/20",  "pct": 45.0, "pbr": 0.0421, "rur": 12276, "apu": 0.7234, "tr": 0.3214},
    {"model": "Qwen-1.8B (FT)",   "wins": "8/20",  "pct": 40.0, "pbr": 0.0384, "rur": 12826, "apu": 0.7506, "tr": 0.2095},
]

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "protoss_logs")


def run_single_game(param_vector, map_name, difficulty_enum):
    bot = ProtossMacroBot(param_vector)
    sc2_map = None
    for candidate in [map_name] + PAPER_MAPS + ["AcropolisAIE", "Simple64"]:
        try:
            m = maps.get(candidate)
            if m is not None:
                sc2_map = m
                break
        except Exception:
            continue
    if sc2_map is None:
        print(f"ERROR: Could not find map '{map_name}'. Install SC2 maps.")
        sys.exit(1)

    print(f"  Launching: Protoss vs Zerg on {sc2_map.name} (Difficulty: {difficulty_enum.name})")
    print(f"  Mode: realtime=False (headless)")
    result = run_game(
        sc2_map,
        [Bot(Race.Protoss, bot), Computer(Race.Zerg, difficulty_enum)],
        realtime=False,
    )
    win_val = 1.0 if result == Result.Victory else 0.0
    metrics = bot.get_evaluation_metrics(win_score=win_val)
    metrics["result"] = "WIN" if win_val == 1.0 else "LOSS"
    metrics["map"] = sc2_map.name
    metrics["observations_sample"] = bot.latest_observations
    return metrics


def print_comparison_table(metrics):
    win_str = "1/1" if metrics["WINRATE"] == 1.0 else "0/1"
    win_pct = metrics["WINRATE"] * 100.0

    print()
    print("=" * 75)
    print("TEXTSTARCRAFT II TABLE 1 COMPARISON (Level 5/Harder)")
    print("Both systems play as Protoss vs Zerg")
    print("Evaluation framework: Ma et al. NeurIPS 2024")
    print("=" * 75)
    hdr = f"{'MODEL':<22s} {'WINS':<8s} {'WIN%':>6s} {'PBR':>8s} {'RUR':>7s} {'APU':>8s} {'TR':>8s}"
    print(hdr)
    print("-" * 75)
    print(f"{'CCE-TDCCEA [Ours]':<22s} {win_str:<8s} {win_pct:5.1f}% {metrics['PBR']:8.4f} {metrics['RUR']:7.0f} {metrics['APU']:8.4f} {metrics['TR']:8.4f}")
    print("-" * 75)
    for b in TABLE_1_BASELINES:
        print(f"{b['model']:<22s} {b['wins']:<8s} {b['pct']:5.1f}% {b['pbr']:8.4f} {b['rur']:7.0f} {b['apu']:8.4f} {b['tr']:8.4f}")
    print("=" * 75)
    print("Note: TextStarCraft II baselines use 20 games.")
    print("This is a single-run result. Run 25 times for statistical power.")
    print("TextStarCraft II results from Ma et al. (2024) Table 1.")
    print("=" * 75)
    print()
    print("Metric Interpretations:")
    print("  PBR  - Lower is better (less supply blocking)")
    print("  RUR  - Resource throughput rate (minerals+gas spent per second)")
    print("  APU  - Higher is better (fuller army maintained)")
    print("  TR   - Technology exploration depth (upgrades + tech buildings / 15)")
    print()


def main():
    parser = argparse.ArgumentParser(description="CCE-TDCCEA Protoss Single Game Runner")
    parser.add_argument("--difficulty", type=int, default=5, choices=range(1, 8),
                        help="AI difficulty 1-7 (default: 5=Harder)")
    parser.add_argument("--map", type=str, default="AltitudeLE",
                        help="Map name (default: AltitudeLE)")
    args = parser.parse_args()

    diff_enum = DIFFICULTY_MAP.get(args.difficulty, Difficulty.Hard)

    print("=" * 75)
    print("  CCE-TDCCEA PROTOSS EVALUATION — SINGLE GAME RUN")
    print("=" * 75)
    print(f"  Map         : {args.map}")
    print(f"  Difficulty  : {args.difficulty} ({diff_enum.name})")
    print(f"  Race        : Protoss vs Zerg")
    print(f"  Parameters  : 23-dim evolved vector")
    print(f"  Vector      : {EVOLVED_VECTOR}")
    print("=" * 75)

    t0 = time.time()
    metrics = run_single_game(EVOLVED_VECTOR, args.map, diff_enum)
    elapsed = time.time() - t0

    print()
    print(f"  Result: {metrics['result']} in {elapsed:.1f} seconds")
    print(f"  WINRATE = {metrics['WINRATE']}")
    print(f"  PBR     = {metrics['PBR']}")
    print(f"  RUR     = {metrics['RUR']}")
    print(f"  APU     = {metrics['APU']}")
    print(f"  TR      = {metrics['TR']}")

    # Save log
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"game_{timestamp}.json")
    log_data = {
        "timestamp": timestamp,
        "map": metrics.get("map", args.map),
        "difficulty": args.difficulty,
        "difficulty_name": diff_enum.name,
        "param_vector": EVOLVED_VECTOR,
        "result": metrics["result"],
        "metrics": {
            "WINRATE": metrics["WINRATE"],
            "PBR": metrics["PBR"],
            "RUR": metrics["RUR"],
            "APU": metrics["APU"],
            "TR": metrics["TR"],
        },
        "observations_sample": metrics.get("observations_sample", {}),
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    print(f"\n  Log saved to: {log_file}")

    print_comparison_table(metrics)


if __name__ == "__main__":
    main()
