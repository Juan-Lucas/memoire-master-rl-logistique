"""Sample Efficiency : pourcentage du budget d'entraînement nécessaire pour
atteindre 95% de la performance finale (récompense ou productivité lissée).

Source :
- data/results/convergence_qlearning_sarsa_powerbi.csv (Q-Learning, SARSA)
- data/results/convergence_deeprl_powerbi.csv (DQN, PPO)

Sortie : data/results/sample_efficiency.csv
Colonnes : algorithm, scenario, pct_budget_95
"""

from __future__ import annotations

import pandas as pd

from memoire_master_rl_logistique.experiments.scenarios import SCENARIO_DISPLAYS

SCENARIOS = [s.label for s in SCENARIO_DISPLAYS]


def convergence_point(df: pd.DataFrame, value_col: str) -> float:
    """Plus petit episode (en % du budget) à partir duquel la valeur lissée
    reste à moins de 5% de la valeur finale jusqu'à la fin de l'entraînement."""
    values = df[value_col].to_numpy()
    episodes = df["episode"].to_numpy()
    final = values[-1]
    threshold = 0.05 * abs(final)

    idx = len(values) - 1
    for i in range(len(values) - 1, -1, -1):
        if abs(values[i] - final) <= threshold:
            idx = i
        else:
            break

    return 100 * episodes[idx] / episodes[-1]


def main() -> None:
    ql = pd.read_csv("data/results/convergence_qlearning_sarsa_powerbi.csv")
    deep = pd.read_csv("data/results/convergence_deeprl_powerbi.csv")

    rows = []
    for algo in ["Q-Learning", "SARSA"]:
        for scenario in SCENARIOS:
            sub = ql[(ql["algorithm"] == algo) & (ql["scenario"] == scenario)]
            rows.append({
                "algorithm": algo,
                "scenario": scenario,
                "pct_budget_95": round(convergence_point(sub, "reward_smooth"), 1),
            })

    for algo in ["DQN", "PPO"]:
        for scenario in SCENARIOS:
            sub = deep[(deep["algorithm"] == algo) & (deep["scenario"] == scenario)]
            rows.append({
                "algorithm": algo,
                "scenario": scenario,
                "pct_budget_95": round(convergence_point(sub, "productivity_smooth"), 1),
            })

    out = pd.DataFrame(rows)
    out_path = "data/results/sample_efficiency.csv"
    out.to_csv(out_path, index=False)
    print(f"Sauvegardé : {out_path}")
    print(out.pivot(index="algorithm", columns="scenario", values="pct_budget_95"))


if __name__ == "__main__":
    main()
