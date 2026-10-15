"""Tests de T26 (M7) : graines d'entraînement et d'évaluation séparées."""

import dataclasses

import pandas as pd
import pytest

from memoire_master_rl_logistique.analysis.statistical_tests import load_results, paired_table
from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments import run_benchmark as rb
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.rl.model_paths import model_dir
from memoire_master_rl_logistique.rl.train_dqn import train_dqn
from memoire_master_rl_logistique.rl.train_ppo import train_ppo
from memoire_master_rl_logistique.rl.train_q_learning import train_q_learning
from memoire_master_rl_logistique.rl.train_sarsa import train_sarsa
from memoire_master_rl_logistique.seeds import (
    EVAL_SEEDS,
    TRAIN_SEEDS,
    TUNING_SEEDS,
    training_episode_range,
    training_episode_seed,
)

# Épisodes courts : plusieurs remises à zéro, donc le compteur interne de MineEnv
# (reset sans graine, utilisé par Stable-Baselines3) est bien exercé.
SHORT = dict(episode_minutes=60.0, overwrite=True)
TRAINERS = {
    "Q-Learning": lambda out: train_q_learning(n_episodes=3, output_dir=out, **SHORT),
    "SARSA": lambda out: train_sarsa(n_episodes=3, output_dir=out, **SHORT),
    "DQN": lambda out: train_dqn(total_timesteps=1_100, output_dir=out,
                                 progress_bar=False, **SHORT),
    "PPO": lambda out: train_ppo(total_timesteps=1_024, output_dir=out,
                                 progress_bar=False, **SHORT),
}


@pytest.fixture
def episode_seeds(monkeypatch):
    """Enregistre la graine réellement utilisée par chaque ``MineEnv.reset``."""
    seen: list[int] = []
    original = MineEnv.reset

    def recording_reset(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        seen.append(self.last_episode_seed)
        return result

    monkeypatch.setattr(MineEnv, "reset", recording_reset)
    return seen


def test_ensembles_de_graines_disjoints():
    """Évaluation, sélection et épisodes d'entraînement ne partagent aucune graine."""
    training = set()
    for s in TRAIN_SEEDS:
        training |= set(training_episode_range(s))
    assert not set(EVAL_SEEDS) & set(TUNING_SEEDS)
    assert not set(EVAL_SEEDS) & training
    assert not set(TUNING_SEEDS) & training
    for scenario in SCENARIOS.values():
        assert set(scenario.seeds) <= set(EVAL_SEEDS)


def test_graines_d_episode_distinctes_entre_graines_d_entrainement():
    """Deux graines d'entraînement ne voient jamais les mêmes épisodes."""
    assert training_episode_seed(1, 0) == 1_000_000
    assert training_episode_seed(1, 29_999) < training_episode_seed(2, 0)
    with pytest.raises(ValueError):
        training_episode_seed(1, 1_000_000)


@pytest.mark.parametrize("algorithm", list(TRAINERS))
def test_aucune_graine_d_evaluation_pendant_l_entrainement(algorithm, episode_seeds, tmp_path):
    """Test du plan : les graines passées à reset pendant l'entraînement (compteur
    interne compris) ne sont jamais des graines d'évaluation ni de sélection."""
    TRAINERS[algorithm](str(tmp_path / algorithm))

    assert len(episode_seeds) >= 2  # au moins une remise à zéro après le premier épisode
    assert not set(episode_seeds) & set(EVAL_SEEDS)
    assert not set(episode_seeds) & set(TUNING_SEEDS)
    assert set(episode_seeds) <= set(training_episode_range(TRAIN_SEEDS[0]))


def test_dossier_par_graine_d_entrainement():
    """Un dossier par (algorithme, régime, graine d'entraînement) ; sans graine : chemin v1."""
    assert model_dir("r", "PPO", "nominal", 3).as_posix() == "r/ppo_nominal_seed3"
    assert model_dir("r", "PPO", "nominal").as_posix() == "r/ppo_nominal"
    assert len({model_dir("r", "DQN", "nominal", s) for s in TRAIN_SEEDS}) == len(TRAIN_SEEDS)


def test_benchmark_une_ligne_par_graine_d_entrainement(monkeypatch, tmp_path):
    """Chaque agent est entraîné une fois par graine d'entraînement, dans son dossier,
    et chaque ligne du CSV porte sa ``train_seed`` (vide pour les heuristiques)."""
    calls = []

    def tiny_train(algorithm, scenario, train_seed, output_dir):
        calls.append((algorithm, train_seed, output_dir.name))
        trainer = train_q_learning if algorithm == "Q-Learning" else train_sarsa
        trainer(n_episodes=1, seed=train_seed, output_dir=str(output_dir),
                truck_count=scenario.truck_count, shovel_count=scenario.shovel_count,
                dump_count=scenario.dump_count, episode_minutes=scenario.episode_minutes,
                breakdown_probability=scenario.breakdown_probability,
                reward_weights=scenario.reward_weights)

    monkeypatch.setattr(rb, "_train", tiny_train)
    scenario = dataclasses.replace(SCENARIOS["nominal"], seeds=[42, 43])
    rows = rb.run_benchmark(scenario, tmp_path, train_ppo_flag=False, train_seeds=(1, 2))

    assert sorted(calls) == [("Q-Learning", 1, "q_learning_nominal_seed1"),
                             ("Q-Learning", 2, "q_learning_nominal_seed2"),
                             ("SARSA", 1, "sarsa_nominal_seed1"),
                             ("SARSA", 2, "sarsa_nominal_seed2")]
    df = pd.DataFrame(rows)
    agents = df[df["policy"].isin(["Q-Learning", "SARSA"])]
    assert agents.groupby(["policy", "train_seed"]).size().to_dict() == {
        ("Q-Learning", 1): 2, ("Q-Learning", 2): 2, ("SARSA", 1): 2, ("SARSA", 2): 2}
    assert df.loc[~df.index.isin(agents.index), "train_seed"].isna().all()

    # Le CSV accepte deux modèles d'une même politique sur une même graine d'évaluation.
    csv_path = rb.write_results(rows, tmp_path, run_id="t26")
    assert "train_seed" in pd.read_csv(csv_path).columns


def _results(train_seeds_by_policy, n_eval=3):
    rows = []
    for policy, train_seeds in train_seeds_by_policy.items():
        for ts in train_seeds:
            for seed in EVAL_SEEDS[:n_eval]:
                rows.append({"scenario": "nominal", "policy": policy, "train_seed": ts,
                             "seed": seed, "productivity_tph": 100.0 + seed + (ts or 0)})
    return pd.DataFrame(rows)


def test_load_results_cle_avec_graine_d_entrainement(tmp_path):
    """Deux graines d'entraînement ne sont pas des doublons ; chaque modèle est un groupe
    de n_expected lignes ; les heuristiques (train_seed vide) forment un groupe."""
    path = tmp_path / "r.csv"
    _results({"Fixed": [None], "PPO": [1, 2]}).to_csv(path, index=False)

    data = load_results(path, n_expected=3)

    assert len(data) == 9


def test_load_results_groupe_incomplet_par_graine_d_entrainement(tmp_path):
    """Un modèle auquel il manque une graine d'évaluation est signalé."""
    path = tmp_path / "r.csv"
    _results({"Fixed": [None], "PPO": [1, 2]}).iloc[:-1].to_csv(path, index=False)

    with pytest.raises(ValueError, match="incomplets"):
        load_results(path, n_expected=3)


def test_paired_table_refuse_plusieurs_graines_d_entrainement():
    """Le tableau apparié ne sait pas encore agréger plusieurs modèles (T30)."""
    one = _results({"Fixed": [None], "PPO": [1]})
    assert paired_table(one, "nominal", "productivity_tph").shape == (3, 2)

    with pytest.raises(ValueError, match="T30"):
        paired_table(_results({"Fixed": [None], "PPO": [1, 2]}), "nominal", "productivity_tph")
