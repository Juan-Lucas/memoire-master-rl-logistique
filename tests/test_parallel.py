"""Tests de T29 (M28) : évaluation parallèle, environnements PPO parallèles, durées mesurées."""

import dataclasses
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from memoire_master_rl_logistique.analysis.statistical_tests import DYNAMIC_COLUMNS
from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments import run_benchmark as rb
from memoire_master_rl_logistique.experiments import timing
from memoire_master_rl_logistique.experiments.run_benchmark import PolicySpec
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.rl.env_factory import make_training_vec_env
from memoire_master_rl_logistique.rl.model_metadata import ModelMetadata
from memoire_master_rl_logistique.rl.train_dqn import train_dqn
from memoire_master_rl_logistique.rl.train_ppo import train_ppo
from memoire_master_rl_logistique.rl.train_q_learning import train_q_learning
from memoire_master_rl_logistique.rl.train_sarsa import train_sarsa
from memoire_master_rl_logistique.seeds import training_episode_range, training_episode_seed

TIMING_COLUMNS = {"decision_time_ms_mean", "decision_time_ms_max", "obs_time_ms_mean",
                  "obs_time_ms_max", "episode_time_s"}


def _tiny_train(algorithm, scenario, train_seed, output_dir):
    env = dict(seed=train_seed, output_dir=str(output_dir), truck_count=scenario.truck_count,
               shovel_count=scenario.shovel_count, dump_count=scenario.dump_count,
               episode_minutes=scenario.episode_minutes,
               breakdown_probability=scenario.breakdown_probability,
               reward_weights=scenario.reward_weights)
    if algorithm in ("Q-Learning", "SARSA"):
        (train_q_learning if algorithm == "Q-Learning" else train_sarsa)(n_episodes=2, **env)
    else:
        (train_dqn if algorithm == "DQN" else train_ppo)(total_timesteps=1_100,
                                                          progress_bar=False, **env)


# --- Évaluation parallèle -----------------------------------------------------------

def test_resultats_paralleles_identiques_aux_sequentiels(monkeypatch, tmp_path):
    """Test du plan : mêmes résultats, graine par graine, en séquentiel et sur 3 processus.

    Les 10 politiques (DQN et PPO compris), spécialisées et transférées, en fortes pannes.
    Seules les durées mesurées peuvent différer.
    """
    monkeypatch.setattr(rb, "_train", _tiny_train)
    scenario = dataclasses.replace(SCENARIOS["high_breakdown"], seeds=[42, 43, 44])

    sequential = pd.DataFrame(rb.run_benchmark(scenario, tmp_path, train_seeds=(1,), n_workers=1))
    parallel = pd.DataFrame(rb.run_benchmark(scenario, tmp_path, train_seeds=(1,), n_workers=3))

    assert len(sequential) == (6 + 2 * 4) * 3  # 6 heuristiques + 4 agents × 2 modes
    compared = [c for c in sequential.columns if c not in TIMING_COLUMNS]
    pd.testing.assert_frame_equal(sequential[compared], parallel[compared])
    assert (parallel["episode_time_s"] > 0).all()


def test_nombre_de_processus_invalide():
    with pytest.raises(ValueError, match="n_workers"):
        rb._run_tasks([], n_workers=0)


def test_episode_independant_de_ce_qui_precede():
    """Un épisode ne dépend que de (politique, graine) : le rejouer après un autre
    redonne la même ligne. C'est ce qui rend la parallélisation exacte."""
    scenario = dataclasses.replace(SCENARIOS["nominal"], episode_minutes=60.0)
    task = (scenario, PolicySpec("ShortestPath"), 42)
    first = rb._evaluate(task)
    rb._evaluate((scenario, PolicySpec("ShortestPath"), 43))
    second = rb._evaluate(task)
    assert {k: v for k, v in first.items() if k not in TIMING_COLUMNS} == \
        {k: v for k, v in second.items() if k not in TIMING_COLUMNS}


def test_tache_transmissible_a_un_autre_processus():
    task = (SCENARIOS["nominal"], PolicySpec("Fixed"), 42)
    assert pickle.loads(pickle.dumps(task)) == task


def test_durees_ignorees_pour_les_doublons():
    """Deux lancements identiques ne diffèrent que par les durées : pas des doublons."""
    assert "episode_time_s" in DYNAMIC_COLUMNS


# --- Graines des environnements d'entraînement parallèles ---------------------------

def test_compteur_de_graines_avec_pas():
    env = MineEnv(seed=10, episode_minutes=30.0, seed_step=3)
    seeds = [env.reset()[1] and env.last_episode_seed for _ in range(2)]
    assert seeds == [13, 16]
    env.reset(seed=100)
    env.reset()
    assert env.last_episode_seed == 103  # une graine explicite repositionne le compteur
    with pytest.raises(ValueError, match="seed_step"):
        MineEnv(seed_step=0)


def test_compteur_inchange_avec_un_seul_environnement():
    """Pas de 1 : graine, + 1, + 2… comme avant T29 (première graine explicite comprise)."""
    env = MineEnv(seed=5, episode_minutes=30.0)
    env.reset(seed=5)
    env.reset()
    env.reset()
    assert env.last_episode_seed == 7


def _episode_seeds(vec_env, n_steps):
    """Graines jouées par chaque environnement pendant ``n_steps`` pas aléatoires."""
    vec_env.reset()
    seen = [[s] for s in vec_env.get_attr("last_episode_seed")]
    rng = np.random.default_rng(0)
    for _ in range(n_steps):
        vec_env.step(rng.integers(vec_env.action_space.n, size=vec_env.num_envs))
        for i, s in enumerate(vec_env.get_attr("last_episode_seed")):
            if s != seen[i][-1]:
                seen[i].append(s)
    return seen


def test_environnements_paralleles_sans_episode_commun():
    """Avec 3 environnements (SubprocVecEnv), l'environnement i joue base + i, base + i + 3…
    aucun épisode n'est joué deux fois, et tous restent dans la plage de la graine."""
    vec_env = make_training_vec_env(2, n_envs=3, episode_minutes=20.0)
    try:
        seen = _episode_seeds(vec_env, 120)
    finally:
        vec_env.close()
    base = training_episode_seed(2)
    for i, seeds in enumerate(seen):
        assert len(seeds) >= 2
        assert seeds == [base + i + 3 * k for k in range(len(seeds))]
    everything = [s for seeds in seen for s in seeds]
    assert len(everything) == len(set(everything))
    assert set(everything) <= set(training_episode_range(2))


def test_un_seul_environnement_comme_avant_t29():
    vec_env = make_training_vec_env(1, n_envs=1, episode_minutes=20.0)
    seen = _episode_seeds(vec_env, 60)[0]
    assert seen == [training_episode_seed(1) + k for k in range(len(seen))]


def test_nombre_d_environnements_change_le_modele(tmp_path):
    """Savoir expliquer : même graine, même budget, mais 1 ou 2 environnements donnent
    deux modèles différents (autres épisodes collectés, autres mises à jour). Actualisation
    par décision : l'actualisation par minute suppose un seul environnement."""
    common = dict(total_timesteps=2_048, episode_minutes=60.0, progress_bar=False,
                  time_discount=False)
    one = train_ppo(output_dir=str(tmp_path / "n1"), n_envs=1, **common)
    two = train_ppo(output_dir=str(tmp_path / "n2"), n_envs=2, **common)
    again = train_ppo(output_dir=str(tmp_path / "n1bis"), n_envs=1, **common)

    def weights(model):
        return [v.cpu().numpy() for v in model.policy.state_dict().values()]

    assert all(np.array_equal(a, b) for a, b in zip(weights(one), weights(again)))
    assert not all(np.array_equal(a, b) for a, b in zip(weights(one), weights(two)))
    metadata = json.loads((tmp_path / "n2" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["training"]["n_envs"] == 2


# --- Durées journalisées ------------------------------------------------------------

def test_duree_d_entrainement_dans_les_metadonnees(tmp_path):
    train_q_learning(n_episodes=1, episode_minutes=30.0, output_dir=str(tmp_path))
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["training_time_s"] > 0


def test_metadonnees_anciennes_sans_duree():
    """Un metadata.json d'avant T29 (sans training_time_s) se relit toujours."""
    old = {"algorithm": "PPO", "observation_size": 18, "seed": 1, "training": {},
           "git_commit": None, "git_dirty": None, "created_at": "", "schema_version": 1,
           "env": {"truck_count": 12, "shovel_count": 3, "dump_count": 2,
                   "episode_minutes": 480.0, "breakdown_probability": 0.02,
                   "reward_weights": [1.0, 0.1, 0.05, 0.3]}}
    assert ModelMetadata.from_json(json.dumps(old)).training_time_s is None


def test_latence_selon_la_taille_de_la_flotte(tmp_path):
    """Une ligne par (camions, politique) ; plus de camions, plus de décisions par poste ;
    l'agent nominal est évalué tel quel à 18 et 24 camions."""
    base = dataclasses.replace(SCENARIOS["nominal"], episode_minutes=60.0)
    train_q_learning(n_episodes=1, episode_minutes=60.0,
                     output_dir=str(tmp_path / "q_learning_nominal_seed1"))
    specs = [PolicySpec("Fixed")] + timing.agent_specs(tmp_path, 1, base)
    assert [s.name for s in specs] == ["Fixed", "Q-Learning"]  # agents absents ignorés

    table = timing.latency_table(specs, base, truck_counts=(12, 18, 24), seeds=[42])

    assert list(zip(table["truck_count"], table["policy"])) == [
        (n, p) for n in (12, 18, 24) for p in ("Fixed", "Q-Learning")]
    assert np.allclose(table["total_ms"], table["decision_ms"] + table["obs_ms"])
    fixed = table[table["policy"] == "Fixed"].set_index("truck_count")["decisions"]
    assert fixed[12] < fixed[18] < fixed[24]


def test_commande_de_latence_n_ecrase_pas(monkeypatch, tmp_path):
    short = dataclasses.replace(SCENARIOS["nominal"], episode_minutes=30.0)
    monkeypatch.setattr(timing, "SCENARIOS", {**SCENARIOS, "nominal": short})
    out = tmp_path / "latence.csv"

    table = timing.main([str(out), "--n-seeds", "1"])

    assert len(table) == 3 * len(timing.HEURISTICS) and out.exists()
    with pytest.raises(FileExistsError):
        timing.main([str(out), "--n-seeds", "1"])
