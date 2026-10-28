"""Étude d'ablation sur w4, le poids de la pénalité d'attente.

Valeurs testées : w4 = 0,0 / 0,1 / 0,3 (valeur retenue) / 0,5, régime nominal sinon.
Agents : PPO et DQN seulement (``algorithms`` des scénarios ``ablation_w4_*`` dans
``configs/experiment_config.yaml``, décision du 22 octobre 2026) ; les heuristiques,
qui ne dépendent pas de w4, sont évaluées aussi, comme repère.

L'ablation se lance dans le **dossier de la campagne** : un modèle par (algorithme,
régime, graine d'entraînement), 5 graines, comme la campagne. Le point w4 = 0,3 est le
régime nominal de la campagne : ses modèles et ses résultats ne sont ni réentraînés ni
réévalués. Chaque lancement écrit un nouveau ``benchmark_<run_id>.csv`` (``run_benchmark``).

    python -m memoire_master_rl_logistique.experiments.ablation_study DOSSIER_CAMPAGNE
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from memoire_master_rl_logistique.experiments.run_benchmark import run_all_benchmarks
from memoire_master_rl_logistique.experiments.scenarios import (
    SCENARIOS,
    SPECIALISED,
    TRANSFER_SOURCE,
)

ABLATION_SCENARIOS = [name for name in SCENARIOS if name.startswith("ablation_w4")]
W4_INDEX = 3  # w4 est le 4e poids de reward_weights (simulation/reward.py)


def run_ablation(output_dir: str | Path, n_workers: int = 1) -> Path:
    """Entraîne (si besoin) et évalue les régimes d'ablation ; renvoie le CSV écrit."""
    return run_all_benchmarks(str(output_dir), scenario_names=ABLATION_SCENARIOS,
                              n_workers=n_workers)


def ablation_table(csv_paths: Iterable[str | Path]) -> pd.DataFrame:
    """Productivité et part d'ATTENDRE par (w4, politique), à partir des CSV de la
    campagne (régime nominal, w4 = 0,3) et de l'ablation.

    Seuls les agents de l'ablation sont gardés au point nominal (pas Q-Learning ni
    SARSA), et seulement en mode spécialisé. Moyenne et écart-type sur toutes les
    lignes (graines d'entraînement × graines d'évaluation).
    """
    df = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
    names = [TRANSFER_SOURCE, *ABLATION_SCENARIOS]
    agents = set().union(*(SCENARIOS[name].algorithms for name in ABLATION_SCENARIOS))
    heuristic = df["mode"].isna()  # les heuristiques n'ont pas de mode (T27)
    agent = df["policy"].isin(agents) & (df["mode"] == SPECIALISED)
    df = df[df["scenario"].isin(names) & (heuristic | agent)].copy()
    df["w4"] = df["scenario"].map(lambda s: SCENARIOS[s].reward_weights[W4_INDEX])
    return (df.groupby(["w4", "policy"])[["productivity_tph", "wait_share_pct"]]
            .agg(["mean", "std"]).round(2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation sur w4 (PPO et DQN).")
    parser.add_argument("output_dir", type=Path, help="dossier de la campagne")
    parser.add_argument("--workers", type=int, default=1, help="processus d'évaluation")
    args = parser.parse_args()
    print(f"Résultats : {run_ablation(args.output_dir, args.workers)}")
