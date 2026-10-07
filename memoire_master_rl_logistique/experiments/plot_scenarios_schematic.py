"""Génère le schéma conceptuel des 3 scénarios expérimentaux.

Affiche pour chaque scénario : flotte de camions, shovels, dumps,
et le taux de pannes — sous forme de diagramme minimaliste.

Sortie : reports/figures/fig3_scenarios.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np


TRUCK_COLOR   = "#546E7A"
SHOVEL_COLOR  = "#1565C0"
DUMP_COLOR    = "#4CAF50"
BROKEN_COLOR  = "#E53935"
ARROW_COLOR   = "#78909C"

SCENARIOS = [
    {
        "name": "Nominal",
        "label": "N",
        "trucks": 12,
        "broken_trucks": 0,
        "shovels": 3,
        "dumps": 2,
        "breakdown_pct": 2,
        "sub": "(a)",
    },
    {
        "name": "High-Load",
        "label": "HL",
        "trucks": 18,
        "broken_trucks": 0,
        "shovels": 3,
        "dumps": 2,
        "breakdown_pct": 2,
        "sub": "(b)",
    },
    {
        "name": "High-Breakdown",
        "label": "HB",
        "trucks": 12,
        "broken_trucks": 2,
        "shovels": 3,
        "dumps": 2,
        "breakdown_pct": 10,
        "sub": "(c)",
    },
]


def _draw_icon(ax, x, y, color, shape, size=14):
    if shape == "truck":
        ax.plot(x, y, marker="s", markersize=size, color=color,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        ax.plot(x + 0.08, y - 0.07, marker="o", markersize=4,
                color="#B0BEC5", zorder=4)
        ax.plot(x - 0.04, y - 0.07, marker="o", markersize=4,
                color="#B0BEC5", zorder=4)
    elif shape == "shovel":
        ax.plot(x, y, marker="^", markersize=size, color=color,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    elif shape == "dump":
        ax.plot(x, y, marker="D", markersize=size, color=color,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)


def draw_scenario(ax, scenario: dict) -> None:
    n_trucks  = scenario["trucks"]
    n_broken  = scenario["broken_trucks"]
    n_shovels = scenario["shovels"]
    n_dumps   = scenario["dumps"]

    # --- Trucks (left column) ---
    cols = 3
    rows = int(np.ceil(n_trucks / cols))
    t_xs = np.linspace(0.08, 0.32, cols)
    t_ys = np.linspace(0.85, 0.15, rows)

    for i in range(n_trucks):
        col_i = i % cols
        row_i = i // cols
        x = t_xs[col_i]
        y = t_ys[row_i] if row_i < len(t_ys) else t_ys[-1] - 0.05
        broken = i >= (n_trucks - n_broken)
        color = BROKEN_COLOR if broken else TRUCK_COLOR
        size  = 10
        ax.plot(x, y, marker="s", markersize=size, color=color,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        if broken:
            ax.text(x, y, "✕", ha="center", va="center",
                    fontsize=6, color="white", fontweight="bold", zorder=4)

    # --- Shovels (centre) ---
    s_ys = np.linspace(0.8, 0.2, n_shovels)
    for sy in s_ys:
        ax.plot(0.55, sy, marker="^", markersize=14, color=SHOVEL_COLOR,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        # Arrow from trucks to shovel
        ax.annotate("", xy=(0.50, sy), xytext=(0.37, 0.50),
                    arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR,
                                   lw=0.8), zorder=2)

    # --- Dumps (right) ---
    d_ys = np.linspace(0.7, 0.3, n_dumps)
    for dy in d_ys:
        ax.plot(0.85, dy, marker="D", markersize=12, color=DUMP_COLOR,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        # Arrow from shovel zone to dump
        ax.annotate("", xy=(0.78, dy), xytext=(0.62, 0.50),
                    arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR,
                                   lw=0.8), zorder=2)

    # --- Labels ---
    ax.text(0.20, 0.95, f"{n_trucks} trucks", ha="center", fontsize=8,
            color=TRUCK_COLOR, fontweight="bold")
    if n_broken:
        ax.text(0.20, 0.905,
                f"({n_broken} broken = {scenario['breakdown_pct']}%)",
                ha="center", fontsize=7, color=BROKEN_COLOR)
    else:
        ax.text(0.20, 0.905,
                f"(breakdown: {scenario['breakdown_pct']}%)",
                ha="center", fontsize=7, color="#9E9E9E")

    ax.text(0.55, 0.96, f"{n_shovels} shovels", ha="center", fontsize=8,
            color=SHOVEL_COLOR, fontweight="bold")
    ax.text(0.85, 0.96, f"{n_dumps} dumps", ha="center", fontsize=8,
            color=DUMP_COLOR, fontweight="bold")

    # Title box
    sub = scenario["sub"]
    name = scenario["name"]
    ax.text(0.50, 0.04,
            f"{sub} {name}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECEFF1",
                      edgecolor="#90A4AE", linewidth=0.8))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax, sc in zip(axes, SCENARIOS):
        draw_scenario(ax, sc)

    # Legend
    legend_handles = [
        mpatches.Patch(color=TRUCK_COLOR,  label="Active truck"),
        mpatches.Patch(color=BROKEN_COLOR, label="Broken-down truck"),
        mpatches.Patch(color=SHOVEL_COLOR, label="Shovel (loading face)"),
        mpatches.Patch(color=DUMP_COLOR,   label="Dump point"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=8.5, frameon=True, framealpha=0.8,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "Experimental scenarios: fleet size, loading faces, and disruption regimes",
        fontsize=11, fontweight="bold", y=1.01)

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig3_scenarios.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()