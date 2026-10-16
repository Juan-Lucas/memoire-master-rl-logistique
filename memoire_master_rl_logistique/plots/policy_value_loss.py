"""Courbes de perte de DQN (réseau Q) et de PPO (politique, valeur), moyenne ± IC 95 % sur
les graines d'entraînement (T30).

Source : ``sb3_logs/progress.csv`` des modèles **évalués** (N7 : plus de réentraînement
séparé ``*_losscurve``).
Sortie : ``<out_dir>/loss_dqn_<régime>.png`` et ``<out_dir>/loss_ppo_<régime>.png``

Lancement :
    python -m memoire_master_rl_logistique.plots.policy_value_loss DOSSIER_MODELES DOSSIER_FIGURES
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


def plot_dqn(results_dir, scenario: str, train_seeds: Sequence[int], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    curve = learning_curve(results_dir, "DQN", scenario, "loss", train_seeds)
    plot_band(ax, curve, AGENT_COLORS["DQN"], "DQN", scale=1e6)
    ax.set_xlabel("Pas d'entraînement (millions)")
    ax.set_ylabel("Perte du réseau Q (train/loss)")
    ax.set_title(f"DQN — perte du réseau Q ({SCENARIO_LABELS.get(scenario, scenario)})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_ppo(results_dir, scenario: str, train_seeds: Sequence[int], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax, curve_name, ylabel in [(axes[0], "policy_loss", "Perte de la politique"),
                                   (axes[1], "value_loss", "Perte de la valeur")]:
        curve = learning_curve(results_dir, "PPO", scenario, curve_name, train_seeds)
        plot_band(ax, curve, AGENT_COLORS["PPO"], "PPO", scale=1e6)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, which="both")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Perte de la valeur (échelle log)")
    axes[1].set_xlabel("Pas d'entraînement (millions)")
    axes[0].legend()
    axes[0].set_title("PPO — pertes de la politique et de la valeur "
                      f"({SCENARIO_LABELS.get(scenario, scenario)})")
    fig.tight_layout()
    save_figure(fig, out_path)


def main(
    results_dir: str | Path, out_dir: str | Path, overwrite: bool = False,
    scenarios: Sequence[str] = MAIN_SCENARIOS, train_seeds: Sequence[int] = TRAIN_SEEDS,
) -> list[Path]:
    names = [f"loss_{algo}_{scenario_suffix(s)}.png" for s in scenarios for algo in ("dqn", "ppo")]
    paths = prepare_outputs(out_dir, names, overwrite)
    for i, scenario in enumerate(scenarios):
        plot_dqn(results_dir, scenario, train_seeds, paths[2 * i])
        plot_ppo(results_dir, scenario, train_seeds, paths[2 * i + 1])
    return paths


if __name__ == "__main__":
    args = training_figures_parser(__doc__.splitlines()[0]).parse_args()
    main(args.results_dir, args.out_dir, args.overwrite, args.scenarios, args.train_seeds)
