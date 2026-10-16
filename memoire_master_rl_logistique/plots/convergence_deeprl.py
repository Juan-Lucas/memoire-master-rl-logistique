"""Courbes de convergence DQN/PPO : productivité au cours de l'entraînement, moyenne ± IC
95 % sur les graines d'entraînement (T30).

Source : ``training_kpis.csv`` de chaque modèle évalué (``analysis.learning_curves``).
Sortie : ``<out_dir>/learning_curves_deeprl_{nominal,highload,highbreakdown}.png``

Lancement :
    python -m memoire_master_rl_logistique.plots.convergence_deeprl DOSSIER_MODELES DOSSIER_FIGURES
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from memoire_master_rl_logistique.analysis.learning_curves import learning_curve
from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS
from memoire_master_rl_logistique.plots.common import (
    AGENT_COLORS,
    SCENARIO_LABELS,
    plot_band,
    prepare_outputs,
    save_figure,
    scenario_suffix,
    training_figures_parser,
)
from memoire_master_rl_logistique.seeds import TRAIN_SEEDS

ALGORITHMS = ("DQN", "PPO")


def main(
    results_dir: str | Path, out_dir: str | Path, overwrite: bool = False,
    scenarios: Sequence[str] = MAIN_SCENARIOS, train_seeds: Sequence[int] = TRAIN_SEEDS,
) -> list[Path]:
    paths = prepare_outputs(
        out_dir, [f"learning_curves_deeprl_{scenario_suffix(s)}.png" for s in scenarios],
        overwrite)
    for scenario, out_path in zip(scenarios, paths, strict=True):
        fig, ax = plt.subplots(figsize=(10, 5))
        for algorithm in ALGORITHMS:
            curve = learning_curve(results_dir, algorithm, scenario, "productivity_tph",
                                   train_seeds)
            plot_band(ax, curve, AGENT_COLORS[algorithm], algorithm, scale=1e6)
        ax.set_xlabel("Pas d'entraînement (millions)")
        ax.set_ylabel("Productivité (t/h)")
        ax.set_title(f"Convergence de DQN et PPO — {SCENARIO_LABELS.get(scenario, scenario)}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_figure(fig, out_path)
    return paths


if __name__ == "__main__":
    args = training_figures_parser(__doc__.splitlines()[0]).parse_args()
    main(args.results_dir, args.out_dir, args.overwrite, args.scenarios, args.train_seeds)
