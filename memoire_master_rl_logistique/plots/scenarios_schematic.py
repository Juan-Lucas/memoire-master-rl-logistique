"""Schéma des trois régimes de fonctionnement (figure ``fig:scenarios`` de l'article).

Pour chaque régime : la flotte, les 3 pelles, les 2 verses, la probabilité de panne
d'un camion par cycle et le facteur d'appariement (MF, section 6.1). Les noms des
régimes sont ceux de l'article anglais.

Depuis la v3, la figure ne dessine plus de camions « en panne » : 10 % est une
probabilité par cycle, pas une part de la flotte immobilisée (l'ancienne légende
« 2 broken = 10% » le laissait croire). Le dossier de sortie est obligatoire : la
figure v1 de ``reports/figures`` n'est jamais réécrite.

Lancement :
    python -m memoire_master_rl_logistique.plots.scenarios_schematic DOSSIER [--overwrite]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from memoire_master_rl_logistique.plots.common import prepare_outputs

TRUCK_COLOR   = "#546E7A"
SHOVEL_COLOR  = "#1565C0"
DUMP_COLOR    = "#4CAF50"
BREAKDOWN_COLOR = "#E53935"
ARROW_COLOR   = "#78909C"

SCENARIOS = [
    {
        "name": "Nominal",
        "trucks": 12,
        "shovels": 3,
        "dumps": 2,
        "breakdown_pct": 2,
        "match_factor": 0.35,
        "sub": "(a)",
    },
    {
        "name": "High load",
        "trucks": 18,
        "shovels": 3,
        "dumps": 2,
        "breakdown_pct": 2,
        "match_factor": 0.52,
        "sub": "(b)",
    },
    {
        "name": "Frequent breakdowns",
        "trucks": 12,
        "shovels": 3,
        "dumps": 2,
        "breakdown_pct": 10,
        "match_factor": 0.35,
        "sub": "(c)",
    },
]
OUTPUT_FILE = "fig3_scenarios.png"


def draw_scenario(ax, scenario: dict) -> None:
    n_trucks  = scenario["trucks"]
    n_shovels = scenario["shovels"]
    n_dumps   = scenario["dumps"]

    # --- Camions (colonne de gauche) ---
    cols = 3
    rows = int(np.ceil(n_trucks / cols))
    t_xs = np.linspace(0.08, 0.32, cols)
    t_ys = np.linspace(0.85, 0.15, rows)
    for i in range(n_trucks):
        ax.plot(t_xs[i % cols], t_ys[i // cols], marker="s", markersize=10,
                color=TRUCK_COLOR, markeredgecolor="white", markeredgewidth=0.5, zorder=3)

    # --- Pelles (centre) ---
    for sy in np.linspace(0.8, 0.2, n_shovels):
        ax.plot(0.55, sy, marker="^", markersize=14, color=SHOVEL_COLOR,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        ax.annotate("", xy=(0.50, sy), xytext=(0.37, 0.50),
                    arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=0.8), zorder=2)

    # --- Verses (droite) ---
    for dy in np.linspace(0.7, 0.3, n_dumps):
        ax.plot(0.85, dy, marker="D", markersize=12, color=DUMP_COLOR,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        ax.annotate("", xy=(0.78, dy), xytext=(0.62, 0.50),
                    arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=0.8), zorder=2)

    # --- Libellés ---
    ax.text(0.20, 0.95, f"{n_trucks} trucks", ha="center", fontsize=8,
            color=TRUCK_COLOR, fontweight="bold")
    frequent = scenario["breakdown_pct"] > 2
    ax.text(0.20, 0.905, f"breakdown: {scenario['breakdown_pct']}% per cycle",
            ha="center", fontsize=7, color=BREAKDOWN_COLOR if frequent else "#757575",
            fontweight="bold" if frequent else "normal")
    ax.text(0.55, 0.96, f"{n_shovels} shovels", ha="center", fontsize=8,
            color=SHOVEL_COLOR, fontweight="bold")
    ax.text(0.85, 0.96, f"{n_dumps} dumps", ha="center", fontsize=8,
            color=DUMP_COLOR, fontweight="bold")
    ax.text(0.55, 0.10, f"match factor {scenario['match_factor']:.2f}",
            ha="center", fontsize=7.5, color="#455A64")
    ax.text(0.50, 0.01, f"{scenario['sub']} {scenario['name']}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECEFF1",
                      edgecolor="#90A4AE", linewidth=0.8))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")


def main(out_dir: str | Path, overwrite: bool = False) -> Path:
    """Écrit ``fig3_scenarios.png`` dans ``out_dir`` ; refuse d'écraser sans ``overwrite``."""
    (out_path,) = prepare_outputs(out_dir, [OUTPUT_FILE], overwrite)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, scenario in zip(axes, SCENARIOS):
        draw_scenario(ax, scenario)

    legend_handles = [
        mpatches.Patch(color=TRUCK_COLOR,  label="Truck"),
        mpatches.Patch(color=SHOVEL_COLOR, label="Shovel"),
        mpatches.Patch(color=DUMP_COLOR,   label="Dump"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=8.5, frameon=True, framealpha=0.8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Écrit : {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path, help="dossier de sortie (créé si besoin)")
    parser.add_argument("--overwrite", action="store_true", help="autoriser l'écrasement")
    args = parser.parse_args()
    main(args.out_dir, args.overwrite)
