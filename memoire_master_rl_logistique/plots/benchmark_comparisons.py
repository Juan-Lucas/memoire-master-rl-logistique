"""Figures de comparaison des politiques (productivité, attente, front de Pareto).

- ``barplot_productivite_scenarios.png`` : productivité moyenne par politique et par
  régime, avec l'IC bootstrap à 95 % sur les 30 réplications uniques.
- ``wait_time_comparison.png`` : attente en file par camion (depuis T10 ; immobilisation
  totale pour un CSV antérieur), même présentation.
  Pas de « seuil cible » : aucune source ne justifiait la ligne des 30 min de la v1.
- ``pareto_productivity_waiting.png`` : productivité en fonction de l'attente,
  **un panneau par régime** (les régimes n'ont pas la même échelle) ; les politiques
  non dominées sont en gras et reliées par le front de Pareto.

L'attente est choisie par ``analysis.robustness.wait_column`` : l'attente en file si le CSV
la contient (depuis T10), sinon l'immobilisation totale (CSV v1).

Lancement :
    python -m memoire_master_rl_logistique.plots.benchmark_comparisons CSV DOSSIER [--overwrite]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from memoire_master_rl_logistique.analysis.robustness import wait_column
from memoire_master_rl_logistique.analysis.statistical_tests import (
    average_over_training_seeds,
    load_results,
    paired_bootstrap_ci,
)
from memoire_master_rl_logistique.plots.common import (
    SCENARIO_COLORS,
    SCENARIO_LABELS,
    SCENARIO_ORDER,
    WAIT_LABELS,
    LANGUAGES,
    set_language,
    tr,
    order_policies,
    policy_label,
    prepare_outputs,
)

PRODUCTIVITY = "productivity_tph"
OUTPUT_FILES = (
    "barplot_productivite_scenarios.png",
    "wait_time_comparison.png",
    "pareto_productivity_waiting.png",
)


def summarise(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Moyenne et IC bootstrap à 95 % de ``metric`` pour chaque (politique, régime).

    L'IC est celui de la moyenne d'un groupe : même calcul que ``paired_bootstrap_ci``
    appliqué aux valeurs du groupe (10 000 tirages, graine 0, donc reproductible).
    Renvoie un tableau long : policy, scenario, mean, ci_low, ci_high.
    """
    rows = []
    for (policy, scenario), values in df.groupby(["policy", "scenario"])[metric]:
        low, high = paired_bootstrap_ci(values.to_numpy())
        rows.append(
            {"policy": policy, "scenario": scenario, "mean": values.mean(),
             "ci_low": low, "ci_high": high}
        )
    return pd.DataFrame(rows)


def pareto_front(cost: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Masque des points non dominés quand on minimise ``cost`` et maximise ``gain``.

    Un point est dominé s'il existe un autre point au moins aussi bon sur les deux
    critères et strictement meilleur sur l'un d'eux.
    """
    cost = np.asarray(cost, dtype=float)
    gain = np.asarray(gain, dtype=float)
    at_least_as_good = (cost[None, :] <= cost[:, None]) & (gain[None, :] >= gain[:, None])
    strictly_better = (cost[None, :] < cost[:, None]) | (gain[None, :] > gain[:, None])
    dominated = (at_least_as_good & strictly_better).any(axis=1)
    return ~dominated


def plot_grouped_bars(summary: pd.DataFrame, ylabel: str, title: str, out_path: Path) -> None:
    """Barres groupées : une famille par politique, une barre par régime, avec IC 95 %."""
    policies = order_policies(summary["policy"].unique())
    scenarios = [s for s in SCENARIO_ORDER if s in set(summary["scenario"])]
    table = summary.set_index(["policy", "scenario"])
    x = np.arange(len(policies))
    width = 0.8 / len(scenarios)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, scenario in enumerate(scenarios):
        rows = table.xs(scenario, level="scenario").loc[policies]
        offsets = x + (i - (len(scenarios) - 1) / 2) * width
        yerr = np.vstack([rows["mean"] - rows["ci_low"], rows["ci_high"] - rows["mean"]])
        ax.bar(offsets, rows["mean"], width, yerr=yerr, capsize=3,
               color=SCENARIO_COLORS.get(scenario), label=SCENARIO_LABELS.get(scenario, scenario),
               error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels([policy_label(p) for p in policies], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title=tr("Régime", "Regime"))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pareto(
    prod: pd.DataFrame,
    wait: pd.DataFrame,
    out_path: Path,
    wait_label: str = WAIT_LABELS["queue_wait_min_per_truck"],
) -> None:
    """Productivité en fonction de l'attente, un panneau par régime, front de Pareto en pointillés."""
    points = prod.merge(wait, on=["policy", "scenario"], suffixes=("_prod", "_wait"))
    scenarios = [s for s in SCENARIO_ORDER if s in set(points["scenario"])]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4.5), squeeze=False)
    for ax, scenario in zip(axes[0], scenarios):
        sub = points[points["scenario"] == scenario].reset_index(drop=True)
        front = pareto_front(sub["mean_wait"], sub["mean_prod"])
        ax.scatter(sub["mean_wait"], sub["mean_prod"], color=SCENARIO_COLORS.get(scenario))
        for row, on_front in zip(sub.itertuples(), front):
            ax.annotate(policy_label(row.policy), (row.mean_wait, row.mean_prod),
                        xytext=(4, 4), textcoords="offset points", fontsize=8,
                        fontweight="bold" if on_front else "normal")
        frontier = sub[front].sort_values("mean_wait")
        ax.plot(frontier["mean_wait"], frontier["mean_prod"], linestyle="--", color="grey")
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario))
        ax.set_xlabel(wait_label)
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel(tr("Productivité (t/h)", "Productivity (t/h)"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(csv_path: str | Path, out_dir: str | Path, overwrite: bool = False,
         lang: str = "fr") -> list[Path]:
    """Produit les trois figures dans ``out_dir`` ; refuse d'écraser sans ``overwrite``."""
    set_language(lang)
    bars_prod, bars_wait, pareto = prepare_outputs(out_dir, OUTPUT_FILES, overwrite)
    # Un agent à plusieurs graines d'entraînement (T30) : une valeur par réplication, la
    # moyenne de ses modèles, pour que l'IC porte sur les 30 réplications, comme pour une
    # heuristique. Régimes de l'article seulement (la campagne évalue aussi l'ablation).
    df = average_over_training_seeds(load_results(csv_path))
    df = df[df["scenario"].isin(SCENARIO_ORDER)]
    prod = summarise(df, PRODUCTIVITY)
    wait_col = wait_column(df)
    wait_label = WAIT_LABELS[wait_col]
    wait = summarise(df, wait_col)

    plot_grouped_bars(prod, tr("Productivité (t/h)", "Productivity (t/h)"),
                      tr("Productivité moyenne par politique et par régime (IC 95 %)",
                         "Mean productivity by policy and regime (95% CI)"), bars_prod)
    plot_grouped_bars(wait, wait_label,
                      tr("Attente par politique et par régime (IC 95 %)",
                         "Queueing time by policy and regime (95% CI)"), bars_wait)
    plot_pareto(prod, wait, pareto, wait_label)

    outputs = [bars_prod, bars_wait, pareto]
    print("Écrits : " + ", ".join(str(p) for p in outputs))
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv_path", type=Path, help="CSV de résultats de benchmark")
    parser.add_argument("out_dir", type=Path, help="dossier de sortie (créé si besoin)")
    parser.add_argument("--overwrite", action="store_true", help="autoriser l'écrasement")
    parser.add_argument("--lang", choices=LANGUAGES, default="fr",
                        help="langue des textes de la figure (en : article anglais)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.csv_path, args.out_dir, args.overwrite, args.lang)
