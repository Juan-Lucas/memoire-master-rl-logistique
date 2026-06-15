"""Figure « Erreur TD » (|delta| moyen par episode) pour Q-Learning et SARSA
-- les 3 scenarios (nominal, high_load, high_breakdown).

Source :
- data/results/q_learning_{scenario}_diag/training_diagnostics.csv
- data/results/sarsa_{scenario}_diag/training_diagnostics.csv

Sortie : reports/figures/td_error_{nominal,highload,highbreakdown}.png
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from memoire_master_rl_logistique.experiments.plot_utils import save_figure
from memoire_master_rl_logistique.experiments.scenarios import SCENARIO_DISPLAYS

WINDOW = 300

SCENARIOS = [(s.key, s.suffix, f"scénario {s.key}") for s in SCENARIO_DISPLAYS]


def main() -> None:
    out_dir = Path("reports/figures")

    for scenario_dir, suffix, scenario_label in SCENARIOS:
        ql = pd.read_csv(f"data/results/q_learning_{scenario_dir}_diag/training_diagnostics.csv")
        sarsa = pd.read_csv(f"data/results/sarsa_{scenario_dir}_diag/training_diagnostics.csv")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

        axes[0].plot(ql["episode"], ql["mean_abs_td_error"], color="#A5D6A7", linewidth=0.5, label="Brut")
        axes[0].plot(ql["episode"], ql["mean_abs_td_error"].rolling(WINDOW, min_periods=1).mean(),
                      color="#4CAF50", linewidth=2, label=f"Moyenne mobile ({WINDOW} ep.)")
        axes[0].set_title("Q-Learning")
        axes[0].set_xlabel("Épisode")
        axes[0].set_ylabel(r"Erreur TD moyenne $|\delta|$")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(sarsa["episode"], sarsa["mean_abs_td_error"], color="#CE93D8", linewidth=0.5, label="Brut")
        axes[1].plot(sarsa["episode"], sarsa["mean_abs_td_error"].rolling(WINDOW, min_periods=1).mean(),
                      color="#9C27B0", linewidth=2, label=f"Moyenne mobile ({WINDOW} ep.)")
        axes[1].set_title("SARSA")
        axes[1].set_xlabel("Épisode")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f"Évolution de l'erreur TD au cours de l'entraînement ({scenario_label})")
        plt.tight_layout()

        out_path = out_dir / f"td_error_{suffix}.png"
        save_figure(fig, out_path)


if __name__ == "__main__":
    main()
