"""Temps de calcul par décision selon la taille de la flotte (T29, M28).

Mesure, pour chaque politique et pour 12, 18 et 24 camions, le temps moyen d'une
décision : appel de la politique (``decision_time_ms``) plus construction de
l'observation (``obs_time_ms``), ainsi que la durée d'un épisode. Les agents sont les
modèles **nominaux** évalués sans réentraînement : l'observation a la même taille quel
que soit le nombre de camions (T18), donc le même réseau sert partout.

Toujours en séquentiel : des processus concurrents fausseraient les durées.

Lancement :
    python -m memoire_master_rl_logistique.experiments.timing SORTIE.csv
        [--models-dir DOSSIER --train-seed 1] [--n-seeds 5]
Sans ``--models-dir``, seules les heuristiques sont mesurées.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import dataclasses
from pathlib import Path

import pandas as pd

from memoire_master_rl_logistique.experiments.policies import AGENTS, HEURISTICS
from memoire_master_rl_logistique.experiments.run_benchmark import PolicySpec, _run_tasks
from memoire_master_rl_logistique.experiments.scenarios import (
    MAIN_SCENARIOS,
    SCENARIOS,
    TRANSFERRED,
    Scenario,
)
from memoire_master_rl_logistique.rl.model_loading import model_file
from memoire_master_rl_logistique.rl.model_metadata import EnvConfig
from memoire_master_rl_logistique.rl.model_paths import model_dir
from memoire_master_rl_logistique.seeds import EVAL_SEEDS

TRUCK_COUNTS = (12, 18, 24)


def fleet_scenario(base: Scenario, truck_count: int, seeds: Sequence[int]) -> Scenario:
    """Le scénario ``base`` avec ``truck_count`` camions (le reste inchangé)."""
    return dataclasses.replace(base, name=f"{base.name}_{truck_count}_camions",
                               truck_count=truck_count, seeds=list(seeds))


def agent_specs(models_dir: str | Path, train_seed: int | None, base: Scenario) -> list[PolicySpec]:
    """Les agents entraînés sur ``base`` présents dans ``models_dir`` (les absents sont ignorés)."""
    trained_on = EnvConfig.from_scenario(base)
    specs = []
    for algorithm in AGENTS:
        mdir = model_dir(models_dir, algorithm, base.name, train_seed)
        if model_file(mdir, algorithm).exists():
            specs.append(PolicySpec(algorithm, TRANSFERRED, train_seed, str(mdir),
                                    model_file(mdir, algorithm).stat().st_mtime_ns, trained_on))
    return specs


def latency_table(
    specs: Sequence[PolicySpec],
    base: Scenario,
    truck_counts: Sequence[int] = TRUCK_COUNTS,
    seeds: Sequence[int] = EVAL_SEEDS[:5],
) -> pd.DataFrame:
    """Une ligne par (nombre de camions, politique) : décisions et durées moyennes par épisode."""
    tasks = [(fleet_scenario(base, n, seeds), spec, seed)
             for n in truck_counts for spec in specs for seed in seeds]
    rows = pd.DataFrame(_run_tasks(tasks, n_workers=1))  # séquentiel : durées non perturbées
    rows["truck_count"] = [task[0].truck_count for task in tasks]
    table = (rows.groupby(["truck_count", "policy"], sort=False)
             .agg(n_episodes=("seed", "size"), decisions=("decisions", "mean"),
                  decision_ms=("decision_time_ms_mean", "mean"),
                  obs_ms=("obs_time_ms_mean", "mean"),
                  episode_s=("episode_time_s", "mean"))
             .reset_index())
    table["total_ms"] = table["decision_ms"] + table["obs_ms"]
    return table


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Temps par décision selon la taille de la flotte.")
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args(argv)

    base = SCENARIOS[MAIN_SCENARIOS[0]]
    specs = [PolicySpec(name) for name in HEURISTICS]
    if args.models_dir is not None:
        specs += agent_specs(args.models_dir, args.train_seed, base)
    table = latency_table(specs, base, seeds=EVAL_SEEDS[:args.n_seeds])
    with open(args.output_csv, "x", newline="", encoding="utf-8") as f:  # jamais d'écrasement
        table.to_csv(f, index=False)
    print(table.round(4).to_string(index=False))
    return table


if __name__ == "__main__":
    main()
