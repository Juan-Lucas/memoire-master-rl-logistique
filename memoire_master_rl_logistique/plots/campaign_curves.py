"""Courbes d'apprentissage de la campagne (T31) : productivité des épisodes d'entraînement,
moyenne ± IC 95 % sur les graines d'entraînement, un panneau par régime, les quatre agents.

Entrée : ``analyses/courbes_productivity_tph.csv`` écrit par ``experiments.campaign analyze``
(``analysis.learning_curves``) : une ligne par (agent, régime, intervalle d'avancement).
L'axe horizontal est l'avancement de l'entraînement (% du budget), commun aux agents
profonds (pas) et tabulaires (épisodes). Les épisodes d'entraînement sont joués avec
l'exploration : la courbe d'un agent à valeurs n'est pas sa politique gloutonne.

Lancement :
    python -m memoire_master_rl_logistique.plots.campaign_curves COURBES_CSV DOSSIER [--overwrite]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from memoire_master_rl_logistique.plots.common import (
    AGENT_COLORS,
    SCENARIO_LABELS,
    SCENARIO_ORDER,
    plot_band,
    prepare_outputs,
    save_figure,
)

OUTPUT_FILE = "learning_curves_campagne.png"
AGENT_ORDER = ["PPO", "DQN", "Q-Learning", "SARSA"]


def plot_campaign_curves(curves: pd.DataFrame, out_path: Path) -> None:
    """Un panneau par régime ; une courbe (moyenne et bande IC 95 %) par agent."""
    scenarios = [s for s in SCENARIO_ORDER if s in set(curves["scenario"])]
    agents = [a for a in AGENT_ORDER if a in set(curves["algorithm"])]
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.2 * len(scenarios), 4.2),
                             squeeze=False)
    for ax, scenario in zip(axes[0], scenarios):
        for agent in agents:
            curve = curves[(curves["scenario"] == scenario) & (curves["algorithm"] == agent)]
            if not curve.empty:
                plot_band(ax, curve.sort_values("progress_pct"), AGENT_COLORS[agent], agent,
                          x="progress_pct")
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario))
        ax.set_xlabel("Avancement de l'entraînement (% du budget)")
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel("Productivité des épisodes d'entraînement (t/h)")
    axes[0][0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    save_figure(fig, out_path)


def main(curves_csv: str | Path, out_dir: str | Path, overwrite: bool = False) -> Path:
    """Produit ``learning_curves_campagne.png`` dans ``out_dir`` (sans écraser par défaut)."""
    (out_path,) = prepare_outputs(out_dir, [OUTPUT_FILE], overwrite)
    plot_campaign_curves(pd.read_csv(curves_csv), out_path)
    print(f"Écrit : {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("curves_csv", type=Path, help="courbes_productivity_tph.csv")
    parser.add_argument("out_dir", type=Path, help="dossier de sortie (créé si besoin)")
    parser.add_argument("--overwrite", action="store_true", help="autoriser l'écrasement")
    args = parser.parse_args()
    main(args.curves_csv, args.out_dir, args.overwrite)
