"""Courbes de convergence DQN/PPO (productivité lissée par épisode).

Source : data/results/convergence_deeprl_powerbi.csv
Sortie : reports/figures/learning_curves_deeprl_{nominal,highload,highbreakdown}.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCENARIOS = [
    ("Nominal", "nominal", "Scénario nominal"),
    ("High Load", "highload", "Scénario high_load"),
    ("High Breakdown", "highbreakdown", "Scénario high_breakdown"),
]

ALGO_STYLE = {
    "DQN": "#2196F3",
    "PPO": "#FF5722",
}


def main() -> None:
    df = pd.read_csv("data/results/convergence_deeprl_powerbi.csv")
    out_dir = Path("reports/figures")

    for scenario_value, suffix, scenario_label in SCENARIOS:
        fig, ax = plt.subplots(figsize=(10, 5))

        for algo, color in ALGO_STYLE.items():
            sub = df[(df["algorithm"] == algo) & (df["scenario"] == scenario_value)]
            ax.plot(sub["episode"], sub["productivity_smooth"], color=color, label=algo, linewidth=1.5)

        ax.set_xlabel("Épisode")
        ax.set_ylabel("Productivité (t/h) — lissée")
        ax.set_title(f"Convergence de DQN et PPO — {scenario_label}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = out_dir / f"learning_curves_deeprl_{suffix}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Figure sauvegardée : {out_path}")


if __name__ == "__main__":
    main()
