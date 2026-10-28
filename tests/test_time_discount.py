"""Actualisation par minute (rl/time_discount.py, décision du 21 octobre 2026).

Test central : quand toutes les décisions durent exactement la durée de référence
(1,89 min), l'actualisation par minute doit redonner **le même apprentissage** que
l'actualisation par décision, pour les quatre algorithmes. Seule la façon de compter le
temps change, rien d'autre.
"""

import json
import pickle

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.buffers import RolloutBuffer
import torch as th

from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments.policies import heuristic_policies, make_env
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.rl.time_discount import (
    REFERENCE_DECISION_INTERVAL_MIN,
    TimeDiscountReplayBuffer,
    TimeDiscountRolloutBuffer,
    per_minute_gamma,
    step_discount,
    transition_discount,
)
from memoire_master_rl_logistique.rl.train_dqn import train_dqn
from memoire_master_rl_logistique.rl.train_ppo import train_ppo
from memoire_master_rl_logistique.rl.train_q_learning import train_q_learning
from memoire_master_rl_logistique.rl.train_sarsa import train_sarsa

REF = REFERENCE_DECISION_INTERVAL_MIN


def constant_elapsed(monkeypatch, minutes=REF):
    """Toutes les décisions durent ``minutes`` (aux yeux de l'actualisation seulement)."""
    original = MineEnv._get_info

    def patched(self, free_at):
        return {**original(self, free_at), "elapsed_min": minutes}

    monkeypatch.setattr(MineEnv, "_get_info", patched)


# --- Durée des décisions dans l'environnement --------------------------------------------

def test_duree_des_decisions():
    """``info["elapsed_min"]`` : durée de chaque décision ; sur un poste, elles couvrent
    exactement les 480 minutes. 0 après reset."""
    scenario = SCENARIOS["nominal"]
    env = make_env(scenario)
    _, info = env.reset(seed=42)
    assert info["elapsed_min"] == 0.0
    policy = heuristic_policies(scenario)["MTWT"]
    total, previous_time, done = 0.0, 0.0, False
    while not done:
        _, _, terminated, truncated, info = env.step(policy(None, info, env))
        assert info["elapsed_min"] >= 0
        assert info["elapsed_min"] == pytest.approx(
            min(info["current_time_min"], 480.0) - previous_time)
        previous_time = info["current_time_min"]
        total += info["elapsed_min"]
        done = terminated or truncated
    assert total == pytest.approx(scenario.episode_minutes)


def test_la_duree_ne_change_pas_la_simulation():
    """Ajouter ``elapsed_min`` dans ``info`` ne change aucun KPI (référence de T13 à T16 bis,
    vérifiée aussi par test_regression)."""
    s = SCENARIOS["nominal"]
    kpis = run_single_episode(make_env(s), heuristic_policies(s)["Fixed"], 42)
    assert kpis["productivity_tph"] == pytest.approx(4252.5)


def test_intervalle_de_reference():
    """1,89 min = 480 min / nombre moyen de décisions de l'Affectation fixe en nominal
    (30 graines d'évaluation) : c'est la valeur testée par test_intervalle_entre_decisions."""
    s = SCENARIOS["nominal"]
    policy = heuristic_policies(s)["Fixed"]
    decisions = np.mean([run_single_episode(make_env(s), policy, seed)["decisions"]
                         for seed in s.seeds])
    assert s.episode_minutes / decisions == pytest.approx(REF, abs=0.01)


# --- Facteurs ----------------------------------------------------------------------------

def test_facteurs():
    g = per_minute_gamma(0.995)
    assert step_discount(g, REF) == pytest.approx(0.995)
    assert step_discount(g, 0.0) == 1.0
    assert step_discount(g, 0.2) > step_discount(g, 30.0)  # attendre 0,2 min ≈ sans coût
    assert transition_discount(0.99, None, {}) == 0.99      # par décision
    assert transition_discount(0.99, per_minute_gamma(0.99), {"elapsed_min": REF}) == \
        pytest.approx(0.99)
    with pytest.raises(ValueError):
        per_minute_gamma(1.5)
    with pytest.raises(ValueError):
        step_discount(g, -1.0)
    with pytest.raises(KeyError, match="elapsed_min"):
        transition_discount(0.99, g, {})


# --- Tampons de Stable-Baselines3 ----------------------------------------------------------

def test_gae_avec_la_duree_de_reference_identique_au_gae_usuel():
    """Avec Δt = 1,89 min à chaque pas, les avantages et retours du tampon de PPO sont ceux
    du GAE usuel avec γ ; avec des durées différentes, ils changent."""
    rng = np.random.default_rng(0)
    obs_space, act_space = gym.spaces.Box(0, 1, (3,)), gym.spaces.Discrete(2)
    gamma, size = 0.995, 64

    def filled(buffer, elapsed):
        buffer.reset()
        r = np.random.default_rng(1)
        for step in range(size):
            if isinstance(buffer, TimeDiscountRolloutBuffer):
                buffer.set_step_discounts([{"elapsed_min": elapsed[step]}])
            buffer.add(r.random((1, 3)), np.array([[r.integers(2)]]), r.normal(size=1),
                       np.array([step % 20 == 0], dtype=float), th.tensor(r.normal(size=1)),
                       th.tensor(r.normal(size=1)))
        buffer.compute_returns_and_advantage(th.tensor([0.3]), np.array([False]))
        return buffer

    usual = filled(RolloutBuffer(size, obs_space, act_space, gamma=gamma, gae_lambda=0.95), None)
    timed = filled(TimeDiscountRolloutBuffer(size, obs_space, act_space, gamma=gamma,
                                             gae_lambda=0.95,
                                             gamma_per_minute=per_minute_gamma(gamma)),
                   [REF] * size)
    np.testing.assert_allclose(timed.advantages, usual.advantages, rtol=1e-5)
    np.testing.assert_allclose(timed.returns, usual.returns, rtol=1e-5)

    varied = filled(TimeDiscountRolloutBuffer(size, obs_space, act_space, gamma=gamma,
                                              gae_lambda=0.95,
                                              gamma_per_minute=per_minute_gamma(gamma)),
                    list(rng.uniform(0.2, 30.0, size)))
    assert not np.allclose(varied.advantages, usual.advantages)


def test_memoire_de_rejeu_garde_le_facteur_de_chaque_transition():
    obs_space, act_space = gym.spaces.Box(0, 1, (3,)), gym.spaces.Discrete(2)
    g = per_minute_gamma(0.995)
    buffer = TimeDiscountReplayBuffer(10, obs_space, act_space, gamma_per_minute=g)
    elapsed = [0.2, 1.89, 30.0]
    for minutes in elapsed:
        buffer.add(np.zeros((1, 3)), np.zeros((1, 3)), np.array([[0]]), np.array([1.0]),
                   np.array([False]), [{"elapsed_min": minutes}])
    samples = buffer._get_samples(np.array([0, 1, 2]))
    np.testing.assert_allclose(samples.discounts.numpy().ravel(),
                               [g ** m for m in elapsed], rtol=1e-6)


# --- Entraînements : même résultat qu'avant quand les durées valent la référence ----------

RUN = dict(episode_minutes=60.0, checkpoint_freq=None)
TABULAR_RUN = dict(episode_minutes=60.0, checkpoint_every=None)


def weights(model):
    return [p.detach().cpu().numpy() for p in model.policy.parameters()]


@pytest.mark.parametrize("train, steps", [(train_ppo, 2_048), (train_dqn, 1_500)])
def test_deep_rl_identique_avec_la_duree_de_reference(tmp_path, monkeypatch, train, steps):
    constant_elapsed(monkeypatch)
    timed = train(output_dir=str(tmp_path / "t"), total_timesteps=steps, progress_bar=False,
                  time_discount=True, **RUN)
    plain = train(output_dir=str(tmp_path / "p"), total_timesteps=steps, progress_bar=False,
                  time_discount=False, **RUN)
    for a, b in zip(weights(timed), weights(plain), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("train, table", [(train_q_learning, "q_table.pkl"),
                                          (train_sarsa, "sarsa_table.pkl")])
def test_tabulaire_identique_avec_la_duree_de_reference(tmp_path, monkeypatch, train, table):
    constant_elapsed(monkeypatch)
    for name, flag in (("t", True), ("p", False)):
        train(output_dir=str(tmp_path / name), n_episodes=5, time_discount=flag,
              **TABULAR_RUN)
    timed, plain = (pickle.loads((tmp_path / n / table).read_bytes())["q_table"] for n in "tp")
    assert timed.keys() == plain.keys()
    for state in timed:
        np.testing.assert_allclose(timed[state], plain[state], rtol=1e-9, atol=1e-12)


def test_les_vraies_durees_changent_l_apprentissage(tmp_path):
    """Sans durée forcée, l'actualisation par minute apprend autre chose (sinon elle ne
    servirait à rien)."""
    for name, flag in (("t", True), ("p", False)):
        train_q_learning(output_dir=str(tmp_path / name), n_episodes=5, time_discount=flag,
                         **TABULAR_RUN)
    timed, plain = (pickle.loads((tmp_path / n / "q_table.pkl").read_bytes())["q_table"]
                    for n in "tp")
    assert any(not np.allclose(timed[s], plain[s]) for s in timed.keys() & plain.keys())


def test_metadonnees_et_gamma_du_modele(tmp_path):
    """Le modèle enregistré garde le γ de la configuration (pas le dernier γ_min^Δt) ;
    metadata.json déclare l'actualisation et γ_min."""
    model = train_ppo(output_dir=str(tmp_path), total_timesteps=1_024, progress_bar=False,
                      time_discount=True, **RUN)
    meta = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))["training"]
    assert model.gamma == meta["gamma"] == 0.995
    assert meta["time_discount"] is True
    assert meta["gamma_per_minute"] == pytest.approx(per_minute_gamma(0.995))
    assert isinstance(model.rollout_buffer, TimeDiscountRolloutBuffer)
    with pytest.raises(ValueError, match="n_envs"):
        train_ppo(output_dir=str(tmp_path / "x"), total_timesteps=1_024, progress_bar=False,
                  time_discount=True, n_envs=2, **RUN)
