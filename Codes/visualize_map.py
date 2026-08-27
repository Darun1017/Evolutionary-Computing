"""
visualize_map.py
================
Renders any microRTS map XML as a publication-quality PNG image.

Usage:
    python visualize_map.py                          # renders current MAP_PATH from COMPARE_ALL_NEW
    python visualize_map.py maps/chambers32x32.xml   # renders a specific map
    python visualize_map.py --all                    # renders ALL maps in the maps/ directory
"""

import os
import sys
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# ── Config ──────────────────────────────────────────────────────────────
MICRORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Simulation", "microrts"
)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_visuals")

# ── Colour Palette ──────────────────────────────────────────────────────
COLORS = {
    "ground":       "#C8D8A0",
    "ground_edge":  "#B0C080",
    "wall":         "#3E3A39",
    "wall_edge":    "#2C2928",
    "p0_base":      "#1565C0",
    "p0_worker":    "#42A5F5",
    "p1_base":      "#C62828",
    "p1_worker":    "#EF5350",
    "resource":     "#FFC107",
    "resource_edge":"#F57F17",
    "resource_text":"#4E342E",
}


def render_map(xml_path: str, output_path: str = None, show: bool = False):
    """
    Parse a microRTS XML map and render it as a high-quality PNG.

    Parameters
    ----------
    xml_path : str
        Path to the .xml map file.
    output_path : str, optional
        Where to save the PNG. If None, auto-generates from the map name.
    show : bool
        If True, display the plot interactively.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    W = int(root.get("width"))
    H = int(root.get("height"))
    terrain = root.find("terrain").text.strip()

    # ── Publication font settings ────────────────────────────────────────
    rcParams["font.family"] = "serif"
    rcParams["font.size"] = 9

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=200)

    # ── Draw terrain ─────────────────────────────────────────────────────
    for r in range(H):
        for c in range(W):
            tile = terrain[r * W + c]
            if tile == "1":
                fc, ec, lw = COLORS["wall"], COLORS["wall_edge"], 0.3
            else:
                fc, ec, lw = COLORS["ground"], COLORS["ground_edge"], 0.2
            rect = mpatches.FancyBboxPatch(
                (c, r), 1, 1, boxstyle="square,pad=0",
                facecolor=fc, edgecolor=ec, linewidth=lw
            )
            ax.add_patch(rect)

    # ── Draw units ───────────────────────────────────────────────────────
    resource_total = 0
    resource_count = 0

    for u in root.findall(".//rts.units.Unit"):
        t = u.get("type")
        x, y = int(u.get("x")), int(u.get("y"))
        p = u.get("player")
        res = int(u.get("resources", "0"))
        cx, cy = x + 0.5, y + 0.5

        if t == "Base":
            color = COLORS["p0_base"] if p == "0" else COLORS["p1_base"]
            rect = mpatches.FancyBboxPatch(
                (x + 0.08, y + 0.08), 0.84, 0.84,
                boxstyle="round,pad=0.04", facecolor=color,
                edgecolor="white", linewidth=1.2, zorder=5
            )
            ax.add_patch(rect)
            ax.text(cx, cy, "B", ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white", zorder=6)

        elif t == "Worker":
            color = COLORS["p0_worker"] if p == "0" else COLORS["p1_worker"]
            circle = plt.Circle(
                (cx, cy), 0.35, facecolor=color,
                edgecolor="white", linewidth=1, zorder=5
            )
            ax.add_patch(circle)
            ax.text(cx, cy, "w", ha="center", va="center",
                    fontsize=6, fontweight="bold", color="white", zorder=6)

        elif t == "Resource":
            resource_total += res
            resource_count += 1
            rect = mpatches.FancyBboxPatch(
                (x + 0.08, y + 0.08), 0.84, 0.84,
                boxstyle="round,pad=0.04", facecolor=COLORS["resource"],
                edgecolor=COLORS["resource_edge"], linewidth=1, zorder=5
            )
            ax.add_patch(rect)
            ax.text(cx, cy, str(res), ha="center", va="center",
                    fontsize=5, fontweight="bold",
                    color=COLORS["resource_text"], zorder=6)

        elif t == "Barracks":
            color = COLORS["p0_base"] if p == "0" else COLORS["p1_base"]
            rect = mpatches.FancyBboxPatch(
                (x + 0.08, y + 0.08), 0.84, 0.84,
                boxstyle="square,pad=0.04", facecolor=color,
                edgecolor="white", linewidth=1, zorder=5, alpha=0.7
            )
            ax.add_patch(rect)
            ax.text(cx, cy, "Bk", ha="center", va="center",
                    fontsize=5, fontweight="bold", color="white", zorder=6)

    # ── Axes ─────────────────────────────────────────────────────────────
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # ── Title ────────────────────────────────────────────────────────────
    map_name = os.path.splitext(os.path.basename(xml_path))[0]
    title = f"{map_name.replace('_', ' ').title()} ({W}\u00d7{H})"
    subtitle = f"{resource_count} Resource Patches | {resource_total} Total Resources"
    ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold",
                 pad=8, family="serif")

    # ── Legend ────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["p0_base"], edgecolor="white", label="Player 0 Base"),
        mpatches.Patch(facecolor=COLORS["p1_base"], edgecolor="white", label="Player 1 Base"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=COLORS["p0_worker"], markersize=8, label="P0 Worker"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=COLORS["p1_worker"], markersize=8, label="P1 Worker"),
        mpatches.Patch(facecolor=COLORS["resource"],
                       edgecolor=COLORS["resource_edge"], label="Resource"),
        mpatches.Patch(facecolor=COLORS["wall"],
                       edgecolor=COLORS["wall_edge"], label="Wall"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", ncol=3,
              fontsize=7, framealpha=0.95, bbox_to_anchor=(0.5, -0.06),
              edgecolor="#888")

    # ── Save ─────────────────────────────────────────────────────────────
    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, f"{map_name}.png")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  [OK] Saved: {output_path}")
    print(f"       {resource_count} patches, {resource_total} total resources")

    if show:
        plt.show()
    plt.close(fig)

    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    maps_dir = os.path.join(MICRORTS_DIR, "maps")

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        # Render every XML in the maps directory
        xml_files = sorted([
            f for f in os.listdir(maps_dir)
            if f.endswith(".xml") and os.path.isfile(os.path.join(maps_dir, f))
        ])
        print(f"Rendering {len(xml_files)} maps from {maps_dir}...\n")
        for f in xml_files:
            try:
                render_map(os.path.join(maps_dir, f))
            except Exception as e:
                print(f"  [FAIL] {f}: {e}")
        print(f"\nDone! All images saved to: {OUTPUT_DIR}")

    elif len(sys.argv) > 1:
        # Render a specific map
        xml_path = sys.argv[1]
        if not os.path.isabs(xml_path):
            xml_path = os.path.join(MICRORTS_DIR, xml_path)
        render_map(xml_path)

    else:
        # Default: render corridor_ts_32x32.xml
        default_map = os.path.join(maps_dir, "corridor_ts_32x32.xml")
        print(f"No arguments given. Rendering default: {default_map}\n")
        render_map(default_map)
