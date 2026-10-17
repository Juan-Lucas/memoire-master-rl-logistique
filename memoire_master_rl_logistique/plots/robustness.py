"""Heatmap de robustesse et tableau 13 de l'article.

Lancement :
    python -m memoire_master_rl_logistique.plots.robustness CSV DOSSIER [--overwrite]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from memoire_master_rl_logistique.analysis.robustness import (
    robustness_ratios,
    robustness_table,
    wait_change_pct,
    wait_column,
)
from memoire_master_rl_logistique.analysis.statistical_tests import load_results
from memoire_master_rl_logistique.experiments.scenarios import PERTURBED_SCENARIOS
from memoire_master_rl_logistique.plots.common import (
    SCENARIO_LABELS,
    WAIT_LABELS,
    order_policies,
    policy_label,
    prepare_outputs,
)

OUTPUT_FILES = ("robustness_ratios.csv", "robustness_heatmap.png")


def plot_robustness_heatmap(
    ratios: pd.DataFrame,
    wait_pct: pd.DataFrame,
    out_path: Path,
    wait_label: str = WAIT_LABELS["queue_wait_min_per_truck"],
) -> None:
    """Deux heatmaps côte à côte, en % de variation par rapport au nominal.

    Gauche : productivité, (ratio − 1) × 100. Droite : attente. Les nombres
    affichés sont les vraies variations ; seules les couleurs sont orientées
    pour que le bleu signifie toujours « mieux » (d'où ``RdBu_r`` pour l'attente).
    """
    order = order_policies(ratios.index)
    labels = [policy_label(p) for p in order]
    panels = [
        ((ratios.loc[order] - 1) * 100, "Productivité (par camion en forte charge)", "RdBu"),
        (wait_pct.loc[order], wait_label, "RdBu_r"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (data, title, cmap) in zip(axes, panels):
        data = data[PERTURBED_SCENARIOS].rename(columns=SCENARIO_LABELS)
        data.index = labels
        limit = max(float(abs(data.to_numpy()).max()), 1.0)  # échelle symétrique autour de 0
        sns.heatmap(data, annot=True, fmt=".1f", cmap=cmap, center=0, vmin=-limit, vmax=limit,
                    linewidths=0.5, ax=ax, cbar_kws={"label": "%"})
        ax.set_title(title)
        ax.set_xlabel("")
    axes[0].set_ylabel("Politique")

    fig.suptitle("Variation par rapport aux conditions nominales (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(csv_path: str | Path, out_dir: str | Path, overwrite: bool = False) -> pd.DataFrame:
    """Calcule, affiche et enregistre le tableau de robustesse et la heatmap.

    Refuse d'écraser des fichiers existants sauf si ``overwrite`` est vrai.
    Renvoie le tableau, pour pouvoir le réutiliser ou le tester.
    """
    csv_out, png_out = prepare_outputs(out_dir, OUTPUT_FILES, overwrite)
    df = load_results(csv_path)
    ratios = robustness_ratios(df)
    wait_col = wait_column(df)
    wait_pct = wait_change_pct(df, wait_col)
    table = robustness_table(ratios, wait_pct)
    table = table.loc[order_policies(table.index)]

    print(table.round(3).to_string())
    table.to_csv(csv_out)
    plot_robustness_heatmap(ratios, wait_pct, png_out, WAIT_LABELS[wait_col])
    print(f"Écrits : {csv_out}, {png_out}")
    return table


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv_path", type=Path, help="CSV de résultats de benchmark")
    parser.add_argument("out_dir", type=Path, help="dossier de sortie (créé si besoin)")
    parser.add_argument("--overwrite", action="store_true", help="autoriser l'écrasement")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.csv_path, args.out_dir, args.overwrite)
