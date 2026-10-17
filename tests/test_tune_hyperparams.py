"""Tests de T25 bis : script de réglage des hyperparamètres (sur un budget minuscule)."""

import json

import pytest

from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.experiments.tune_hyperparams import (
    GRIDS,
    TUNING_SEEDS,
    config_id,
    configurations,
    tune,
)

TINY = 1_024  # un seul déroulement de PPO : on teste le script, pas la qualité du réglage
ONE_CONFIG = {"PPO": {"ent_coef": (0.02,), "learning_rate": (3e-4,)},
              "DQN": {"exploration_fraction": (0.3,), "learning_rate": (3e-4,)}}


def test_meme_budget_pour_ppo_et_dqn():
    """Même nombre de configurations pour les deux algorithmes ; les valeurs de la v1 en font partie."""
    assert len(configurations("PPO")) == len(configurations("DQN")) == 4
    assert {"ent_coef": 0.02, "learning_rate": 3e-4} in configurations("PPO")
    assert {"exploration_fraction": 0.3, "learning_rate": 3e-4} in configurations("DQN")
    assert set(GRIDS) == {"PPO", "DQN"}


def test_graines_de_selection_distinctes_de_l_evaluation():
    """Le critère n'utilise jamais une graine de l'évaluation finale."""
    assert len(TUNING_SEEDS) == 10
    for scenario in SCENARIOS.values():
        assert not set(TUNING_SEEDS) & set(scenario.seeds)


def test_identifiant_de_configuration():
    """Nom de dossier stable, indépendant de l'ordre des paramètres."""
    assert config_id({"learning_rate": 3e-4, "ent_coef": 0.02}) == "ent_coef=0.02_learning_rate=0.0003"


def test_criterion_reproductible(tmp_path):
    """Test du plan : relancer une configuration redonne exactement le même critère
    (graine d'entraînement et graines de sélection fixées)."""
    seeds = TUNING_SEEDS[:2]
    first = tune(tmp_path / "a", ["PPO", "DQN"], TINY, ONE_CONFIG, seeds)
    second = tune(tmp_path / "b", ["PPO", "DQN"], TINY, ONE_CONFIG, seeds)

    assert list(first["productivity_mean"]) == list(second["productivity_mean"])
    assert first["best"].all()  # une seule configuration par algorithme
    assert set(first.columns) >= {"algorithm", "timesteps", "selection_seeds",
                                  "productivity_mean", "productivity_std", "best"}


def test_hyperparametres_enregistres_et_reprise(tmp_path):
    """Les valeurs testées sont écrites dans metadata.json ; une configuration déjà
    entraînée est réévaluée sans être réentraînée."""
    seeds = TUNING_SEEDS[:1]
    tune(tmp_path, ["PPO"], TINY, ONE_CONFIG, seeds)
    run_dir = tmp_path / "PPO" / config_id({"ent_coef": 0.02, "learning_rate": 3e-4})
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    trained_at = (run_dir / "ppo_mine_agent.zip").stat().st_mtime

    tune(tmp_path, ["PPO"], TINY, ONE_CONFIG, seeds)

    # Depuis T28, tous les hyperparamètres utilisés sont écrits, pas seulement ceux réglés.
    assert metadata["training"]["total_timesteps"] == TINY
    assert metadata["training"]["learning_rate"] == pytest.approx(3e-4)
    assert metadata["training"]["ent_coef"] == pytest.approx(0.02)
    assert metadata["training"]["net_arch"] == [128, 128]
    assert (tmp_path / "configs" / "ppo_hyperparameters.yaml").exists()  # configuration archivée
    assert (run_dir / "ppo_mine_agent.zip").stat().st_mtime == trained_at
