"""Figure du comportement des politiques sur 30 réplications (T12 bis).

Un panneau par régime ; pour chaque politique, une barre empilée donne la part des
cycles réalisés à chaque pelle, et le nombre à droite la part des décisions ATTENDRE.
La figure montre des proportions mesurées, sans légende interprétative.

Entrée : ``behaviour_summary.csv`` produit par ``analysis.behavior``.

Lancement :
    python -m memoire_master_rl_logistique.plots.behaviour SUMMARY_CSV DOSSIER [--overwrite]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from memoire_master_rl_logistique.plots.common import (
    SCENARIO_LABELS,
    SCENARIO_ORDER,
    order_policies,
    policy_label,
    prepare_outputs,
)

SHOVEL_COLORS = ["#1565C0", "#AD1457", "#2E7D32", "#F9A825", "#6A1B9A"]


def plot_behaviour(summary: pd.DataFrame, out_path: Path) -> None:
    """Parts des cycles par pelle (barres empilées) et part ATTENDRE (nombre), par régime."""
    shares = sorted(c for c in summary.columns if c.startswith("share_"))
    scenarios = [s for s in SCENARIO_ORDER if s in set(summary["scenario"])]
    policies = order_policies(summary["policy"].unique())

    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4.5),
                             sharey=True, squeeze=False)
    for ax, scenario in zip(axes[0], scenarios):
        data = summary[summary["scenario"] == scenario].set_index("policy").loc[policies]
        left = pd.Series(0.0, index=data.index)
        for color, col in zip(SHOVEL_COLORS, shares):
            ax.barh(range(len(policies)), data[col], left=left, color=color,
                    label=col.removeprefix("share_").removesuffix("_pct"))
            left += data[col]
        for i, wait in enumerate(data["wait_share_pct"]):
            ax.text(103, i, f"ATT {wait:.0f} %", va="center", fontsize=8)
        ax.set_xlim(0, 125)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Part des cycles par pelle (%)")
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario))
        ax.invert_yaxis()
    axes[0][0].set_yticks(range(len(policies)))
    axes[0][0].set_yticklabels([policy_label(p) for p in policies])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Pelle", loc="lower center", ncol=len(shares),
               fontsize=8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Répartition des cycles entre les pelles et part des décisions ATTENDRE "
                 "(moyennes sur 30 réplications)")
    fig.tight_layout(rect=(0, 0.08, 1, 1))  # place pour la légende sous les panneaux
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(summary_csv: str | Path, out_dir: str | Path, overwrite: bool = False) -> Path:
    """Produit ``behaviour_30_replications.png`` dans ``out_dir``."""
    (out_path,) = prepare_outputs(out_dir, ["behaviour_30_replications.png"], overwrite)
    plot_behaviour(pd.read_csv(summary_csv), out_path)
    print(f"Figure enregistrée : {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("summary_csv", type=Path, help="behaviour_summary.csv")
    parser.add_argument("out_dir", type=Path, help="dossier de sortie (créé si besoin)")
    parser.add_argument("--overwrite", action="store_true", help="autoriser l'écrasement")
    args = parser.parse_args()
    main(args.summary_csv, args.out_dir, args.overwrite)
