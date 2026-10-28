"""Courbes d'apprentissage et de diagnostic agrégées sur les graines d'entraînement (T30).

Source : les journaux écrits par les entraînements **eux-mêmes**, dans le dossier de chaque
modèle évalué (N7) ; plus aucun réentraînement séparé pour les figures :

- Q-Learning, SARSA : ``training_diagnostics.csv`` (un épisode par ligne) ;
- DQN, PPO : ``training_kpis.csv`` (un épisode par ligne, avec le pas atteint) et
  ``sb3_logs/progress.csv`` (pertes, exploration, entropie de Stable-Baselines3).

Les graines n'ont pas des épisodes de même longueur : chaque run est d'abord résumé sur une
grille commune (``n_bins`` intervalles égaux du budget d'entraînement, moyenne des valeurs
qui y tombent), puis on calcule, point par point, la moyenne des runs et son IC bootstrap.

Lancement (export CSV, par exemple pour Power BI) :
    python -m memoire_master_rl_logistique.analysis.learning_curves DOSSIER_MODELES SORTIE.csv
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from memoire_master_rl_logistique.analysis.aggregate_seeds import bootstrap_ci
from memoire_master_rl_logistique.experiments.policies import AGENTS
from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS
from memoire_master_rl_logistique.rl.model_metadata import ModelMetadata, read_metadata
from memoire_master_rl_logistique.rl.model_paths import model_dir
from memoire_master_rl_logistique.seeds import TRAIN_SEEDS

N_BINS = 100  # un point par pour-cent du budget d'entraînement
N_BOOT = 10_000
TABULAR = ("Q-Learning", "SARSA")
TABULAR_LOG = "training_diagnostics.csv"
EPISODE_LOG = "training_kpis.csv"
SB3_LOG = Path("sb3_logs") / "progress.csv"
SB3_STEP = "time/total_timesteps"


@dataclass(frozen=True)
class Source:
    """Où lire une courbe pour un algorithme : fichier, axe x, colonne, transformation."""

    file: Path | str
    x: str
    column: str
    transform: Callable[[pd.Series, ModelMetadata], pd.Series] = lambda s, _meta: s


def _per_hour(tonnage: pd.Series, meta: ModelMetadata) -> pd.Series:
    """Tonnage d'un poste → productivité (t/h), comme ``kpi.productivity_tph``."""
    return tonnage / (meta.env.episode_minutes / 60.0)


_TABULAR_PRODUCTIVITY = Source(TABULAR_LOG, "episode", "tonnage_t", _per_hour)
_DEEP_PRODUCTIVITY = Source(EPISODE_LOG, "timesteps", "tonnage_t", _per_hour)

# Courbe → algorithme → source. Un algorithme absent n'a pas cette courbe.
CURVES: dict[str, dict[str, Source]] = {
    "productivity_tph": {"Q-Learning": _TABULAR_PRODUCTIVITY, "SARSA": _TABULAR_PRODUCTIVITY,
                         "DQN": _DEEP_PRODUCTIVITY, "PPO": _DEEP_PRODUCTIVITY},
    "reward": {"Q-Learning": Source(TABULAR_LOG, "episode", "total_reward"),
               "SARSA": Source(TABULAR_LOG, "episode", "total_reward"),
               "DQN": Source(EPISODE_LOG, "timesteps", "reward"),
               "PPO": Source(EPISODE_LOG, "timesteps", "reward")},
    "td_error": {"Q-Learning": Source(TABULAR_LOG, "episode", "mean_abs_td_error"),
                 "SARSA": Source(TABULAR_LOG, "episode", "mean_abs_td_error")},
    "epsilon": {"Q-Learning": Source(TABULAR_LOG, "episode", "epsilon"),
                "SARSA": Source(TABULAR_LOG, "episode", "epsilon"),
                "DQN": Source(SB3_LOG, SB3_STEP, "rollout/exploration_rate")},
    "loss": {"DQN": Source(SB3_LOG, SB3_STEP, "train/loss")},
    "policy_loss": {"PPO": Source(SB3_LOG, SB3_STEP, "train/policy_gradient_loss")},
    "value_loss": {"PPO": Source(SB3_LOG, SB3_STEP, "train/value_loss")},
    # SB3 enregistre l'opposé de l'entropie (perte) ; on trace l'entropie elle-même.
    "entropy": {"PPO": Source(SB3_LOG, SB3_STEP, "train/entropy_loss", lambda s, _m: -s)},
}


def training_budget(meta: ModelMetadata) -> int:
    """Budget d'entraînement : épisodes (tabulaire) ou pas (DQN, PPO)."""
    key = "n_episodes" if meta.algorithm in TABULAR else "total_timesteps"
    return int(meta.training[key])


def run_log(run_dir: str | Path, algorithm: str, curve: str) -> tuple[pd.DataFrame, int]:
    """Valeurs brutes d'une courbe pour un run : (DataFrame x, value) et budget."""
    if algorithm not in CURVES[curve]:
        raise ValueError(f"Pas de courbe {curve!r} pour {algorithm}")
    source = CURVES[curve][algorithm]
    meta = read_metadata(run_dir)
    log = pd.read_csv(Path(run_dir) / source.file)
    missing = [c for c in (source.x, source.column) if c not in log.columns]
    if missing:
        raise ValueError(f"{Path(run_dir) / source.file} : colonnes {missing} absentes "
                         "(journal d'avant T30 : réentraîner ce modèle)")
    values = pd.DataFrame({"x": log[source.x],
                           "value": source.transform(log[source.column], meta)})
    return values.dropna().reset_index(drop=True), training_budget(meta)


def bin_edges(budget: int, n_bins: int = N_BINS) -> np.ndarray:
    return np.linspace(0, budget, n_bins + 1)


def binned(log: pd.DataFrame, budget: int, n_bins: int = N_BINS) -> np.ndarray:
    """Moyenne des valeurs de chaque intervalle [k, k + 1) × budget / n_bins ; nan si vide.

    Le dernier pas (x = budget) compte dans le dernier intervalle.
    """
    bins = np.clip(np.searchsorted(bin_edges(budget, n_bins), log["x"], side="right") - 1,
                   0, n_bins - 1)
    sums = np.bincount(bins, weights=log["value"], minlength=n_bins)
    counts = np.bincount(bins, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def run_dirs(
    results_dir: str | Path, algorithm: str, regime: str, train_seeds: Sequence[int],
) -> dict[int, Path]:
    """Dossier de chaque run (le même que celui évalué par run_benchmark)."""
    dirs = {s: model_dir(results_dir, algorithm, regime, s) for s in train_seeds}
    missing = [str(d) for d in dirs.values() if not (d / "metadata.json").exists()]
    if missing:
        raise FileNotFoundError(f"Runs introuvables : {missing}")
    return dirs


def curve_matrix(
    results_dir: str | Path, algorithm: str, regime: str, curve: str,
    train_seeds: Sequence[int] = TRAIN_SEEDS, n_bins: int = N_BINS,
) -> tuple[np.ndarray, np.ndarray, int]:
    """(centres des intervalles, matrice runs × intervalles, budget)."""
    rows, budgets = [], set()
    for run_dir in run_dirs(results_dir, algorithm, regime, train_seeds).values():
        log, budget = run_log(run_dir, algorithm, curve)
        budgets.add(budget)
        rows.append(binned(log, budget, n_bins))
    if len(budgets) != 1:
        raise ValueError(f"{algorithm} {regime} : budgets d'entraînement différents {budgets}")
    budget = budgets.pop()
    edges = bin_edges(budget, n_bins)
    return (edges[:-1] + edges[1:]) / 2, np.vstack(rows), budget


def summarize(
    x: np.ndarray, matrix: np.ndarray, n_boot: int = N_BOOT, seed: int = 0,
) -> pd.DataFrame:
    """Par point : nombre de runs, moyenne et IC bootstrap à 95 % de la moyenne des runs."""
    rows = []
    for xi, column in zip(x, matrix.T, strict=True):
        values = column[~np.isnan(column)]
        low, high = (bootstrap_ci(values, "mean", n_boot, seed) if len(values)
                     else (np.nan, np.nan))
        rows.append({"x": xi, "n_runs": len(values),
                     "mean": values.mean() if len(values) else np.nan,
                     "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def learning_curve(
    results_dir: str | Path, algorithm: str, regime: str, curve: str,
    train_seeds: Sequence[int] = TRAIN_SEEDS, n_bins: int = N_BINS, n_boot: int = N_BOOT,
) -> pd.DataFrame:
    """Courbe moyenne ± IC sur les graines ; colonnes x (épisode ou pas), progress_pct,
    n_runs, mean, ci_low, ci_high."""
    x, matrix, budget = curve_matrix(results_dir, algorithm, regime, curve, train_seeds, n_bins)
    summary = summarize(x, matrix, n_boot)
    summary.insert(1, "progress_pct", 100 * summary["x"] / budget)
    return summary


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Courbes d'apprentissage multi-graines (CSV).")
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--curve", default="productivity_tph", choices=sorted(CURVES))
    parser.add_argument("--scenarios", nargs="+", default=list(MAIN_SCENARIOS))
    parser.add_argument("--train-seeds", nargs="+", type=int, default=list(TRAIN_SEEDS))
    args = parser.parse_args(argv)

    if args.output_csv.exists():  # vérifié avant le calcul, pas après
        raise FileExistsError(f"Fichier déjà présent : {args.output_csv}")
    frames = []
    for scenario in args.scenarios:
        for algorithm in AGENTS:
            if algorithm in CURVES[args.curve]:
                curve = learning_curve(args.results_dir, algorithm, scenario, args.curve,
                                       args.train_seeds)
                frames.append(curve.assign(algorithm=algorithm, scenario=scenario))
    table = pd.concat(frames, ignore_index=True)
    with open(args.output_csv, "x", newline="", encoding="utf-8") as f:  # jamais d'écrasement
        table.to_csv(f, index=False)
    print(f"Sauvegardé : {args.output_csv} ({len(table)} lignes)")
    return table


if __name__ == "__main__":
    main()
