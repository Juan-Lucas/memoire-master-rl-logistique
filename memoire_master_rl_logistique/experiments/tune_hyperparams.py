"""Réglage documenté des hyperparamètres de PPO et DQN (T25 bis, M12).

Principes, pour que la comparaison PPO / DQN reste équitable :
- **même budget** : autant de configurations pour PPO que pour DQN (2 × 2 chacun par
  défaut), même nombre de pas d'entraînement, même graine d'entraînement (``TRAIN_SEEDS[0]``) ;
- **graines de sélection distinctes de l'évaluation finale** : le critère est la
  productivité moyenne sur les graines 20 000 à 20 009, jamais sur les graines 42 à 71
  de l'évaluation. Régler sur les graines d'évaluation reviendrait à choisir la
  configuration qui a eu de la chance sur ces 30 épisodes-là ;
- **budget réduit, annoncé** : chaque configuration est entraînée ``timesteps`` pas
  (200 000 par défaut, soit 10 % de la campagne finale). Le tableau produit le dit.

Sorties dans ``out_dir`` : un dossier par configuration (modèle + metadata.json) et
``tuning_results.csv`` (une ligne par configuration : valeurs testées, critère moyen et
écart-type, meilleure configuration par algorithme).

Lancement (plusieurs heures avec les valeurs par défaut) :
    python -m memoire_master_rl_logistique.experiments.tune_hyperparams DOSSIER
        [--timesteps 200000] [--algorithms PPO DQN]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from itertools import product
from pathlib import Path

import pandas as pd

from memoire_master_rl_logistique.config import experiment_config, snapshot_configs
from memoire_master_rl_logistique.experiments.policies import agent_policy_fn, make_env
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.rl.model_loading import load_model
from memoire_master_rl_logistique.rl.train_dqn import train_dqn
from memoire_master_rl_logistique.rl.train_ppo import train_ppo
from memoire_master_rl_logistique.seeds import TRAIN_SEEDS, TUNING_SEEDS

TRAINING_SEED = TRAIN_SEEDS[0]

# Budget et grilles : rubrique ``tuning`` de configs/experiment_config.yaml (T28).
# Même nombre de configurations par algorithme ; les valeurs de la v1
# (PPO : ent_coef 0,02, lr 3e-4 ; DQN : lr 3e-4, exploration 0,3) en font partie.
_TUNING = experiment_config()["tuning"]
DEFAULT_TIMESTEPS: int = _TUNING["timesteps"]
GRIDS: dict[str, dict[str, tuple[float, ...]]] = {
    algorithm: {name: tuple(values) for name, values in grid.items()}
    for algorithm, grid in _TUNING["grids"].items()
}
TRAINERS = {"PPO": train_ppo, "DQN": train_dqn}


def configurations(algorithm: str, grid: dict[str, Sequence[float]] | None = None) -> list[dict]:
    """Toutes les combinaisons de la grille d'un algorithme."""
    grid = grid or GRIDS[algorithm]
    names = list(grid)
    return [dict(zip(names, values)) for values in product(*(grid[n] for n in names))]


def config_id(params: dict) -> str:
    """Nom de dossier lisible et stable : ``ent_coef=0.02_learning_rate=0.0003``."""
    return "_".join(f"{name}={value:g}" for name, value in sorted(params.items()))


def selection_criterion(model_dir: Path, algorithm: str, seeds: Sequence[int] = TUNING_SEEDS,
                        scenario: str = "nominal") -> tuple[float, float]:
    """Productivité moyenne et écart-type (t/h) du modèle sur les graines de sélection."""
    s = SCENARIOS[scenario]
    policy_fn = agent_policy_fn(load_model(model_dir, algorithm, make_env(s)), algorithm)
    values = pd.Series([run_single_episode(make_env(s), policy_fn, seed)["productivity_tph"]
                        for seed in seeds])
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def tune(
    out_dir: str | Path,
    algorithms: Sequence[str] = ("PPO", "DQN"),
    timesteps: int = DEFAULT_TIMESTEPS,
    grids: dict[str, dict[str, Sequence[float]]] | None = None,
    seeds: Sequence[int] = TUNING_SEEDS,
) -> pd.DataFrame:
    """Entraîne et évalue chaque configuration ; reprend sans réentraîner celles déjà faites.

    La configuration est archivée dans ``out_dir/configs`` ; une reprise avec une
    configuration modifiée est refusée (T28).
    """
    out_dir = Path(out_dir)
    snapshot_configs(out_dir / "configs")
    rows = []
    for algorithm in algorithms:
        for params in configurations(algorithm, (grids or {}).get(algorithm)):
            run_dir = out_dir / algorithm / config_id(params)
            if not (run_dir / "metadata.json").exists():
                TRAINERS[algorithm](total_timesteps=timesteps, seed=TRAINING_SEED,
                                    output_dir=str(run_dir), overwrite=True,
                                    progress_bar=False, **params)
            mean, std = selection_criterion(run_dir, algorithm, seeds)
            rows.append({"algorithm": algorithm, **params, "timesteps": timesteps,
                         "selection_seeds": f"{seeds[0]}-{seeds[-1]}", "n_seeds": len(seeds),
                         "productivity_mean": mean, "productivity_std": std})

    results = pd.DataFrame(rows)
    best = results.groupby("algorithm")["productivity_mean"].transform("max")
    results["best"] = results["productivity_mean"] == best
    return results


def main(out_dir: str | Path, algorithms: Sequence[str], timesteps: int) -> pd.DataFrame:
    """Lance le balayage et écrit ``tuning_results.csv``."""
    results = tune(out_dir, algorithms, timesteps)
    results.to_csv(Path(out_dir) / "tuning_results.csv", index=False)
    print(results.round(1).to_string(index=False))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Réglage des hyperparamètres (T25 bis).")
    parser.add_argument("out_dir", type=Path, help="dossier des modèles et du tableau")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--algorithms", nargs="+", default=["PPO", "DQN"], choices=list(GRIDS))
    args = parser.parse_args()
    main(args.out_dir, args.algorithms, args.timesteps)
