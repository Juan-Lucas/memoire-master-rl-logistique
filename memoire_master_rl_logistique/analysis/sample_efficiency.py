"""Sample Efficiency : pourcentage du budget d'entraînement nécessaire pour
atteindre 95 % de la performance finale, pour chaque graine d'entraînement (T30).

Pour chaque run, la courbe de productivité est résumée sur 100 intervalles du budget
(``analysis.learning_curves``) ; le point de convergence est le premier intervalle à
partir duquel la courbe reste à moins de 5 % de sa valeur finale (dernier intervalle).
On rapporte ensuite la moyenne des runs, son IC bootstrap, le plus rapide et le plus lent.

Avant T30, le calcul portait sur une seule courbe par agent, lue dans des CSV Power BI
issus de réentraînements séparés (récompense pour le tabulaire, productivité pour DQN/PPO) :
toutes les courbes sont maintenant la productivité des modèles évalués.

Lancement :
    python -m memoire_master_rl_logistique.analysis.sample_efficiency DOSSIER_MODELES SORTIE.csv
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from memoire_master_rl_logistique.analysis.aggregate_seeds import bootstrap_ci
from memoire_master_rl_logistique.analysis.learning_curves import curve_matrix
from memoire_master_rl_logistique.experiments.policies import AGENTS
from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS
from memoire_master_rl_logistique.seeds import TRAIN_SEEDS

TOLERANCE = 0.05


def convergence_point(x: np.ndarray, values: np.ndarray, budget: float,
                      tolerance: float = TOLERANCE) -> float:
    """Plus petit x (en % du budget) à partir duquel la courbe reste à moins de
    ``tolerance`` de sa valeur finale jusqu'à la fin de l'entraînement.

    Les intervalles vides (nan) sont ignorés.
    """
    keep = ~np.isnan(values)
    x, values = np.asarray(x)[keep], np.asarray(values)[keep]
    final = values[-1]
    outside = np.abs(values - final) > tolerance * abs(final)
    # Premier point après le dernier écart supérieur à la tolérance.
    idx = int(np.flatnonzero(outside)[-1]) + 1 if outside.any() else 0
    return float(100 * x[idx] / budget)


def efficiency_table(
    results_dir: str | Path, scenarios: Sequence[str] = MAIN_SCENARIOS,
    train_seeds: Sequence[int] = TRAIN_SEEDS, tolerance: float = TOLERANCE,
) -> pd.DataFrame:
    """Une ligne par (agent, régime) : point de convergence moyen des runs, IC, extrêmes."""
    rows = []
    for algorithm in AGENTS:
        for scenario in scenarios:
            x, matrix, budget = curve_matrix(results_dir, algorithm, scenario,
                                             "productivity_tph", train_seeds)
            points = np.array([convergence_point(x, run, budget, tolerance) for run in matrix])
            low, high = bootstrap_ci(points, "mean")
            rows.append({"algorithm": algorithm, "scenario": scenario,
                         "n_train_seeds": len(points), "pct_budget_95": points.mean(),
                         "ci_low": low, "ci_high": high,
                         "fastest": points.min(), "slowest": points.max()})
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=list(TRAIN_SEEDS))
    args = parser.parse_args(argv)

    if args.output_csv.exists():  # vérifié avant le calcul, pas après
        raise FileExistsError(f"Fichier déjà présent : {args.output_csv}")
    table = efficiency_table(args.results_dir, train_seeds=args.train_seeds)
    with open(args.output_csv, "x", newline="", encoding="utf-8") as f:  # jamais d'écrasement
        table.to_csv(f, index=False)
    print(f"Sauvegardé : {args.output_csv}")
    print(table.round(1).to_string(index=False))
    return table


if __name__ == "__main__":
    main()
