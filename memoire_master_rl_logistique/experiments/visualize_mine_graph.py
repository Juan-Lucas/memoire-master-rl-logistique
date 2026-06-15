"""Visualisation du réseau routier minier (Section 3.2 du mémoire).

Génère une figure du graphe G = (V, E) représentant le réseau routier
de la mine : yard, jonctions, pelles, dumps et les routes avec distances.

Usage :
    python -m memoire_master_rl_logistique.experiments.visualize_mine_graph
    python -m memoire_master_rl_logistique.experiments.visualize_mine_graph --shovels 5 --dumps 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from memoire_master_rl_logistique.simulation.graph_model import build_mine_graph

# Couleurs par type de nœud
_NODE_COLORS = {
    "yard": "#4CAF50",
    "junction": "#9E9E9E",
    "shovel": "#FF9800",
    "dump": "#2196F3",
}

_NODE_LABELS = {
    "yard": "Yard",
    "junction": "Jonction",
    "shovel": "Pelle",
    "dump": "Dump",
}


def generate_mine_graph_figure(
    shovel_count: int = 3,
    dump_count: int = 2,
    output_path: str = "reports/figures/graphe_reseau_minier.png",
    show: bool = False,
) -> Path:
    """Génère la figure du réseau routier minier.

    Parameters
    ----------
    shovel_count : nombre de pelles
    dump_count : nombre de dumps
    output_path : chemin de sortie pour la figure
    show : afficher la figure (``plt.show()``)

    Returns
    -------
    Chemin absolu du fichier généré.
    """
    graph = build_mine_graph(shovel_count=shovel_count, dump_count=dump_count)
    g = graph.nx_graph

    # Position manuelle pour un rendu clair
    pos: dict[str, tuple[float, float]] = {}
    pos["yard"] = (0.0, 0.0)
    pos["junction_1"] = (2.0, 0.0)

    # Pelles en arc à gauche-haut
    for i in range(1, shovel_count + 1):
        angle_frac = (i - 1) / max(shovel_count - 1, 1)
        y = 2.0 - angle_frac * 4.0 if shovel_count > 1 else 1.5
        pos[f"shovel_{i}"] = (4.5, y)

    # Dumps en arc à droite-bas
    for i in range(1, dump_count + 1):
        angle_frac = (i - 1) / max(dump_count - 1, 1)
        y = 2.0 - angle_frac * 4.0 if dump_count > 1 else -1.5
        pos[f"dump_{i}"] = (7.0, y)

    # Couleurs et tailles
    node_colors = []
    node_sizes = []
    labels = {}
    for node_id in g.nodes():
        ntype = g.nodes[node_id].get("node_type", "junction")
        node_colors.append(_NODE_COLORS.get(ntype, "#9E9E9E"))
        node_sizes.append(1200 if ntype == "yard" else 900)

        if ntype == "yard":
            labels[node_id] = "Yard"
        elif ntype == "shovel":
            idx = node_id.split("_")[1]
            labels[node_id] = f"P{idx}"
        elif ntype == "dump":
            idx = node_id.split("_")[1]
            labels[node_id] = f"D{idx}"
        else:
            labels[node_id] = "J1"

    # Labels d'arêtes (distances en km)
    edge_labels = {}
    for u, v, data in g.edges(data=True):
        dist = data.get("distance_km", 0)
        edge_labels[(u, v)] = f"{dist:.1f}"

    # Figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    fig.patch.set_facecolor("white")

    nx.draw_networkx_nodes(
        g, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="black",
        linewidths=1.5,
    )

    nx.draw_networkx_labels(
        g, pos, ax=ax,
        labels=labels,
        font_size=11,
        font_weight="bold",
        font_color="white",
    )

    nx.draw_networkx_edges(
        g, pos, ax=ax,
        edge_color="#555555",
        arrows=True,
        arrowsize=15,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08",
        width=1.2,
        alpha=0.7,
    )

    nx.draw_networkx_edge_labels(
        g, pos, ax=ax,
        edge_labels=edge_labels,
        font_size=7,
        font_color="#333333",
        label_pos=0.35,
    )

    # Légende
    legend_elements = []
    for ntype, color in _NODE_COLORS.items():
        label = _NODE_LABELS.get(ntype, ntype)
        legend_elements.append(
            plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=color, markersize=12,
                markeredgecolor="black", markeredgewidth=1,
                label=label,
            ),
        )
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=10,
        framealpha=0.9,
        title="Type de nœud",
        title_fontsize=11,
    )

    ax.set_title(
        f"Réseau routier minier — G = (V, E)\n"
        f"{shovel_count} pelles, {dump_count} dumps, "
        f"{g.number_of_nodes()} nœuds, {g.number_of_edges()} arêtes",
        fontsize=14,
        fontweight="bold",
    )
    ax.axis("off")
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    print(f"Figure sauvegardée : {out}")

    if show:
        plt.show()
    plt.close(fig)

    return out.resolve()


def main() -> None:
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Visualisation du réseau routier minier",
    )
    parser.add_argument(
        "--shovels", type=int, default=3,
        help="Nombre de pelles (défaut: 3)",
    )
    parser.add_argument(
        "--dumps", type=int, default=2,
        help="Nombre de dumps (défaut: 2)",
    )
    parser.add_argument(
        "--output", type=str,
        default="reports/figures/graphe_reseau_minier.png",
        help="Chemin de sortie (défaut: reports/figures/graphe_reseau_minier.png)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Afficher la figure",
    )
    args = parser.parse_args()

    generate_mine_graph_figure(
        shovel_count=args.shovels,
        dump_count=args.dumps,
        output_path=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
