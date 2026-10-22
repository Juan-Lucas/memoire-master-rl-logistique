"""Agrégation des résultats sur plusieurs graines d'entraînement (T30, M7).

Un agent est entraîné avec 5 graines (T26) : on obtient 5 modèles, donc 5 « runs ».
Le score d'un run est la moyenne d'une métrique sur les graines d'évaluation (42 à 71).
On résume ces runs comme le recommandent Agarwal et al. (2021, « Deep Reinforcement
Learning at the Edge of the Statistical Precipice ») :

- **IQM** (moyenne interquartile) : moyenne des 50 % de scores centraux, plus robuste
  qu'une moyenne à un run aberrant, moins variable qu'une médiane ;
- **IC bootstrap stratifié** à 95 % : on rééchantillonne les runs **à l'intérieur de
  chaque régime** (une « tâche »), jamais d'un régime à l'autre ;
- **probabilité d'amélioration** P(X > Y) : chance qu'un run de X fasse mieux qu'un run
  de Y dans un même régime, moyennée sur les régimes.

Ce sont exactement les définitions de la bibliothèque ``rliable`` 1.2.0
(``aggregate_iqm``, ``StratifiedBootstrap``, ``probability_of_improvement``). Elles sont
recodées ici parce que ``rliable`` dépend de ``arch<8``, qui ne s'importe pas avec
pandas 3 ; ``tests/test_aggregate_seeds.py`` vérifie qu'on retrouve ses valeurs.

Les heuristiques n'ont pas de graine d'entraînement : elles ne figurent pas dans ces
tableaux (elles servent seulement de référence pour normaliser entre régimes).

Lancement : ``python -m memoire_master_rl_logistique.analysis.aggregate_seeds RESULTATS.csv``
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

from memoire_master_rl_logistique.analysis.statistical_tests import METRICS, load_results
from memoire_master_rl_logistique.experiments.policies import AGENTS
from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS, SPECIALISED

N_BOOT = 50_000  # nombre de rééchantillonnages, celui de rliable
BOOT_SEED = 0
CONFIDENCE = 0.95
REFERENCE_POLICY = "Fixed"  # normalisation des agrégats (référence de la v1 ; voir aussi statistical_tests.PAIRS)
IMPROVEMENT_PAIRS = [("PPO", "DQN"), ("PPO", "Q-Learning"), ("DQN", "Q-Learning"),
                     ("Q-Learning", "SARSA")]
# Métriques pour lesquelles une valeur plus faible est meilleure.
LOWER_IS_BETTER = {"specific_fuel_l_per_ton", "queue_wait_min_per_truck",
                   "avg_wait_min_per_truck"}
RUN_KEY = ["scenario", "policy", "mode", "train_seed"]


# --- Statistiques sur une matrice runs × tâches -------------------------------------

def _iqm(samples: np.ndarray) -> np.ndarray:
    """IQM de chaque ligne : moyenne tronquée à 25 % de chaque côté (rliable.aggregate_iqm)."""
    return stats.trim_mean(samples, proportiontocut=0.25, axis=1)


STATISTICS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "iqm": _iqm,
    "mean": lambda samples: samples.mean(axis=1),
    "median": lambda samples: np.median(samples, axis=1),
}


def _as_matrix(scores: np.ndarray | Sequence[float]) -> np.ndarray:
    """Matrice (runs × tâches) ; un vecteur est un seul régime."""
    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.size == 0 or np.isnan(matrix).any():
        raise ValueError(f"Scores attendus en matrice runs × tâches sans valeur manquante : "
                         f"{matrix.shape}")
    return matrix


def aggregate(scores: np.ndarray | Sequence[float], statistic: str = "iqm") -> float:
    """Statistique (``iqm``, ``mean`` ou ``median``) de tous les scores runs × tâches."""
    matrix = _as_matrix(scores)
    return float(STATISTICS[statistic](matrix.reshape(1, -1))[0])


def _bootstrap_rows(shape: tuple[int, int], n_boot: int, seed: int) -> np.ndarray:
    """Indices de runs tirés avec remise, indépendamment dans chaque case (runs × tâches)."""
    n_runs, n_tasks = shape
    return np.random.default_rng(seed).integers(n_runs, size=(n_boot, n_runs, n_tasks))


def stratified_bootstrap(
    scores: np.ndarray | Sequence[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED,
) -> np.ndarray:
    """Échantillons bootstrap stratifiés, de forme (n_boot, runs, tâches).

    Dans chaque tâche (colonne), les runs sont tirés avec remise parmi **ses** runs :
    on mesure l'incertitude due aux graines d'entraînement à régimes fixés
    (rliable.StratifiedBootstrap, ``task_bootstrap=False``).
    """
    matrix = _as_matrix(scores)
    return matrix[_bootstrap_rows(matrix.shape, n_boot, seed), np.arange(matrix.shape[1])]


def bootstrap_ci(
    scores: np.ndarray | Sequence[float], statistic: str = "iqm",
    n_boot: int = N_BOOT, seed: int = BOOT_SEED, confidence: float = CONFIDENCE,
) -> tuple[float, float]:
    """IC percentile de la statistique par bootstrap stratifié.

    Avec un seul run, il n'y a aucune variabilité d'entraînement à estimer : (nan, nan).
    """
    matrix = _as_matrix(scores)
    if matrix.shape[0] < 2:
        return float("nan"), float("nan")
    samples = stratified_bootstrap(matrix, n_boot, seed)
    values = STATISTICS[statistic](samples.reshape(n_boot, -1))
    alpha = 100 * (1 - confidence) / 2
    low, high = np.percentile(values, [alpha, 100 - alpha])
    return float(low), float(high)


def _improvement(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """P(X > Y) + ½ P(X = Y) par tâche, moyennée sur les tâches ; x, y : (..., runs, tâches)."""
    greater = (x[..., :, None, :] > y[..., None, :, :]).mean(axis=(-3, -2))
    ties = (x[..., :, None, :] == y[..., None, :, :]).mean(axis=(-3, -2))
    return (greater + 0.5 * ties).mean(axis=-1)


def probability_of_improvement(
    scores_x: np.ndarray | Sequence[float], scores_y: np.ndarray | Sequence[float],
    n_boot: int = 10_000, seed: int = BOOT_SEED, confidence: float = CONFIDENCE,
) -> tuple[float, float, float]:
    """P(X > Y) et son IC bootstrap stratifié.

    C'est la statistique U de Mann-Whitney divisée par n_x × n_y, dans chaque tâche
    (rliable.probability_of_improvement). 0,5 : aucun des deux ne domine.
    Comme rliable (``arch`` applique un même tirage à tous ses tableaux), les mêmes indices
    de runs servent à X et à Y : il faut donc autant de runs de chaque côté.
    """
    x, y = _as_matrix(scores_x), _as_matrix(scores_y)
    if x.shape != y.shape:
        raise ValueError(f"Matrices runs × régimes de formes différentes : {x.shape}, {y.shape}")
    point = float(_improvement(x, y))
    if x.shape[0] < 2:
        return point, float("nan"), float("nan")
    rows = _bootstrap_rows(x.shape, n_boot, seed)
    tasks = np.arange(x.shape[1])
    values = _improvement(x[rows, tasks], y[rows, tasks])
    alpha = 100 * (1 - confidence) / 2
    low, high = np.percentile(values, [alpha, 100 - alpha])
    return point, float(low), float(high)


# --- Des résultats de benchmark aux scores par run ----------------------------------

def run_scores(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Score de chaque run (agent, régime, mode, graine d'entraînement) : la moyenne de
    ``metric`` sur ses graines d'évaluation. Les heuristiques sont exclues.

    Tous les runs d'un même agent doivent avoir été évalués sur les mêmes graines.
    """
    if "train_seed" not in data.columns:
        raise ValueError("Pas de colonne train_seed : résultats d'avant T26, un seul run.")
    agents = data[data["train_seed"].notna()].copy()
    if "mode" not in agents.columns:
        agents["mode"] = SPECIALISED
    for key, group in agents.groupby(RUN_KEY[:3]):
        seed_sets = {frozenset(g["seed"]) for _, g in group.groupby("train_seed")}
        if len(seed_sets) > 1:
            raise ValueError(f"{key} : les graines d'entraînement n'ont pas été évaluées sur "
                             "les mêmes graines d'évaluation.")
    scores = (agents.groupby(RUN_KEY)
              .agg(score=(metric, "mean"), n_eval_seeds=("seed", "size"))
              .reset_index())
    scores["train_seed"] = scores["train_seed"].astype(int)
    return scores


def _order(table: pd.DataFrame, by_scenario: bool = True) -> pd.DataFrame:
    """Régimes de l'article puis les autres ; agents dans l'ordre de AGENTS."""
    scenario_rank = {s: i for i, s in enumerate(MAIN_SCENARIOS)}
    agent_rank = {a: i for i, a in enumerate(AGENTS)}
    keys = (["scenario"] if by_scenario else []) + ["policy"]
    ranks = {"scenario": lambda s: s.map(lambda v: (scenario_rank.get(v, len(scenario_rank)), v)),
             "policy": lambda s: s.map(lambda v: (agent_rank.get(v, len(agent_rank)), v))}
    return table.sort_values(keys, key=lambda col: ranks[col.name](col)).reset_index(drop=True)


def aggregate_table(
    data: pd.DataFrame, metric: str, n_boot: int = N_BOOT, seed: int = BOOT_SEED,
) -> pd.DataFrame:
    """Une ligne par (régime, agent, mode) : IQM des runs et son IC, moyenne, médiane,
    plus mauvais et meilleur run."""
    rows = []
    for (scenario, policy, mode), group in run_scores(data, metric).groupby(RUN_KEY[:3]):
        scores = group["score"].to_numpy()
        low, high = bootstrap_ci(scores, "iqm", n_boot, seed)
        rows.append({"scenario": scenario, "policy": policy, "mode": mode,
                     "n_train_seeds": len(scores), "iqm": aggregate(scores, "iqm"),
                     "ci_low": low, "ci_high": high, "mean": scores.mean(),
                     "median": float(np.median(scores)), "min": scores.min(),
                     "max": scores.max()})
    return _order(pd.DataFrame(rows))


def normalized_scores(
    data: pd.DataFrame, metric: str, reference: str = REFERENCE_POLICY,
    scenarios: Sequence[str] = MAIN_SCENARIOS,
) -> pd.DataFrame:
    """Scores des runs divisés par la moyenne de l'heuristique ``reference`` dans le même
    régime : 1 = aussi bien que la référence. Rend les régimes comparables entre eux,
    comme le score normalisé par l'humain sur Atari (Agarwal et al., 2021)."""
    scores = run_scores(data, metric)
    scores = scores[scores["scenario"].isin(scenarios)].copy()
    ref = data[(data["policy"] == reference) & data["scenario"].isin(scenarios)]
    ref_means = ref.groupby("scenario")[metric].mean()
    missing = sorted(set(scenarios) - set(ref_means.index))
    if missing:
        raise ValueError(f"Heuristique de référence {reference} absente de {missing}")
    scores["score"] = scores["score"] / scores["scenario"].map(ref_means)
    return scores


def _score_matrix(group: pd.DataFrame, scenarios: Sequence[str]) -> np.ndarray:
    """Matrice runs × régimes ; chaque régime doit avoir le même nombre de runs."""
    columns = [group.loc[group["scenario"] == s, "score"].to_numpy() for s in scenarios]
    sizes = {len(c) for c in columns}
    if len(sizes) != 1 or 0 in sizes:
        counts = dict(zip(scenarios, [len(c) for c in columns], strict=True))
        raise ValueError(f"Nombre de runs différent selon le régime : {counts}")
    return np.column_stack(columns)


def aggregate_across_scenarios(
    data: pd.DataFrame, metric: str, reference: str = REFERENCE_POLICY,
    scenarios: Sequence[str] = MAIN_SCENARIOS, n_boot: int = N_BOOT, seed: int = BOOT_SEED,
) -> pd.DataFrame:
    """IQM des scores normalisés sur tous les régimes, avec IC bootstrap stratifié par régime."""
    scores = normalized_scores(data, metric, reference, scenarios)
    rows = []
    for (policy, mode), group in scores.groupby(["policy", "mode"]):
        matrix = _score_matrix(group, scenarios)
        low, high = bootstrap_ci(matrix, "iqm", n_boot, seed)
        rows.append({"policy": policy, "mode": mode, "n_train_seeds": matrix.shape[0],
                     "n_scenarios": matrix.shape[1], "iqm": aggregate(matrix, "iqm"),
                     "ci_low": low, "ci_high": high, "mean": matrix.mean()})
    return _order(pd.DataFrame(rows), by_scenario=False)


def improvement_table(
    data: pd.DataFrame, metric: str, pairs: Sequence[tuple[str, str]] = IMPROVEMENT_PAIRS,
    scenarios: Sequence[str] = MAIN_SCENARIOS, mode: str = SPECIALISED,
    n_boot: int = 10_000, seed: int = BOOT_SEED,
) -> pd.DataFrame:
    """P(X meilleur que Y) dans chaque régime, puis sur l'ensemble des régimes.

    « Meilleur » tient compte du sens de la métrique (moins de carburant est mieux). Pas
    besoin de normaliser : on ne compare X et Y qu'à l'intérieur d'un même régime.
    """
    scores = run_scores(data, metric)
    scores = scores[(scores["mode"] == mode) & scores["scenario"].isin(scenarios)]
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
    rows = []
    for x, y in pairs:
        sx, sy = (scores[scores["policy"] == p] for p in (x, y))
        if sx.empty or sy.empty:
            continue
        for label, subset in [(s, [s]) for s in scenarios] + [("ensemble", list(scenarios))]:
            mx = sign * _score_matrix(sx, subset)
            my = sign * _score_matrix(sy, subset)
            point, low, high = probability_of_improvement(mx, my, n_boot, seed)
            rows.append({"scenario": label, "pair": f"{x} > {y}", "p_improvement": point,
                         "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def main(csv_path: str | Path, n_expected: int = 30) -> dict[str, pd.DataFrame]:
    """Tableaux d'agrégation pour la productivité et le carburant (agents spécialisés)."""
    data = load_results(csv_path, n_expected)
    tables = {}
    for metric in METRICS:
        tables[f"{metric}_by_scenario"] = aggregate_table(data, metric)
        tables[f"{metric}_overall"] = aggregate_across_scenarios(data, metric)
        tables[f"{metric}_improvement"] = improvement_table(data, metric)
        print(f"\n===== {metric} : IQM sur les graines d'entraînement (IC 95 % stratifié) =====")
        print(tables[f"{metric}_by_scenario"].round(4).to_string(index=False))
        print(f"\n----- {metric} : tous régimes, normalisé par {REFERENCE_POLICY} -----")
        print(tables[f"{metric}_overall"].round(4).to_string(index=False))
        print(f"\n----- {metric} : probabilité d'amélioration -----")
        print(tables[f"{metric}_improvement"].round(3).to_string(index=False))
    return tables


if __name__ == "__main__":
    main(sys.argv[1])
