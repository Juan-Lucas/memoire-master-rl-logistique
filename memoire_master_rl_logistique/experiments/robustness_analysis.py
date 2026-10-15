"""Robustesse : chaque régime perturbé comparé au régime nominal (tableau 13 et heatmap).

- Ratio de robustesse = productivité **par camion** dans le régime / en nominal.
  En forte charge (18 camions au lieu de 12), un ratio sur la productivité totale
  mesurerait surtout l'ajout de 50 % de camions : une politique inchangée
  afficherait déjà 1,5. En fortes pannes, le nombre de camions est le même et les
  deux calculs coïncident.
- Variation de l'attente moyenne par camion (en %) par rapport au nominal.
  Il s'agit de l'attente totale ; l'attente en file seule viendra avec T10.

Lancement :
    python -m memoire_master_rl_logistique.experiments.robustness_analysis CSV DOSSIER [--overwrite]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from memoire_master_rl_logistique.experiments.reporting import (
    PERTURBED_SCENARIOS,
    SCENARIO_LABELS,
    order_policies,
    policy_label,
    prepare_outputs,
)
from memoire_master_rl_logistique.experiments.statistical_tests import load_results

PRODUCTIVITY = "productivity_tph"
WAIT = "avg_wait_min_per_truck"
OUTPUT_FILES = ("robustness_ratios.csv", "robustness_heatmap.png")


def _scenario_means(df: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    """Moyenne de ``values`` par politique (lignes) et par régime (colonnes).

    Lève ``ValueError`` si le régime nominal ou un régime perturbé manque.
    """
    means = values.groupby([df["policy"], df["scenario"]]).mean().unstack("scenario")
    missing = [s for s in ["nominal", *PERTURBED_SCENARIOS] if s not in means.columns]
    if missing:
        raise ValueError(f"Régimes absents des données : {missing}")
    return means


def robustness_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio de robustesse par politique : productivité par camion du régime / nominal.

    Renvoie un tableau politiques × régimes perturbés ; 1 = aucune perte.
    """
    per_truck = df[PRODUCTIVITY] / df["truck_count"]
    means = _scenario_means(df, per_truck)
    return means[PERTURBED_SCENARIOS].div(means["nominal"], axis=0)


def wait_change_pct(df: pd.DataFrame, wait_col: str = WAIT) -> pd.DataFrame:
    """Variation (%) de l'attente moyenne par camion par rapport au nominal.

    Positive = l'attente augmente (dégradation). Tableau politiques × régimes perturbés.
    """
    means = _scenario_means(df, df[wait_col])
    nominal = means["nominal"]
    return means[PERTURBED_SCENARIOS].sub(nominal, axis=0).div(nominal, axis=0) * 100


def robustness_table(ratios: pd.DataFrame, wait_pct: pd.DataFrame) -> pd.DataFrame:
    """Assemble ratios et variations d'attente, dans l'ordre d'affichage des politiques."""
    table = ratios.add_prefix("ratio_").join(wait_pct.add_prefix("wait_pct_"))
    return table.loc[order_policies(table.index)]


def plot_robustness_heatmap(ratios: pd.DataFrame, wait_pct: pd.DataFrame, out_path: Path) -> None:
    """Deux heatmaps côte à côte, en % de variation par rapport au nominal.

    Gauche : productivité, (ratio − 1) × 100. Droite : attente. Les nombres
    affichés sont les vraies variations ; seules les couleurs sont orientées
    pour que le bleu signifie toujours « mieux » (d'où ``RdBu_r`` pour l'attente).
    """
    order = order_policies(ratios.index)
    labels = [policy_label(p) for p in order]
    panels = [
        ((ratios.loc[order] - 1) * 100, "Productivité (par camion en forte charge)", "RdBu"),
        (wait_pct.loc[order], "Attente moyenne par camion", "RdBu_r"),
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
    wait_pct = wait_change_pct(df)
    table = robustness_table(ratios, wait_pct)

    print(table.round(3).to_string())
    table.to_csv(csv_out)
    plot_robustness_heatmap(ratios, wait_pct, png_out)
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
