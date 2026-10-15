"""Tests de T19 : comptage des décisions ATTENDRE et coût d'ATTENDRE dans la récompense."""

import numpy as np
import pytest

from memoire_master_rl_logistique.analysis.behavior import episode_behaviour
from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments.policies import heuristic_policies, make_env
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS

NOMINAL = SCENARIOS["nominal"]


def random_fn(seed):
    rng = np.random.default_rng(seed)
    return lambda obs, info, env: int(rng.integers(env.action_space.n))


def test_affectation_fixe_n_attend_jamais():
    """Une heuristique n'utilise jamais ATTENDRE : 0 %, et une décision par cycle."""
    env = make_env(NOMINAL, log_cycles=True)
    kpis = run_single_episode(env, heuristic_policies(NOMINAL)["Fixed"], seed=42)

    assert kpis["wait_actions"] == 0 and kpis["wait_share_pct"] == 0
    assert kpis["decisions"] == len(env.cycle_log)


def test_comptage_coherent_avec_le_journal():
    """Politique aléatoire : le compteur d'``info`` égale le nombre d'entrées « wait » du journal,
    et la part d'ATTENDRE égale celle de l'analyse du comportement (T12 bis)."""
    env = make_env(NOMINAL, log_cycles=True)
    kpis = run_single_episode(env, random_fn(0), seed=42)
    waits = sum(r.kind == "wait" for r in env.cycle_log)
    behaviour = episode_behaviour(env.cycle_log, [s.shovel_id for s in env.shovels])

    assert kpis["wait_actions"] == waits > 0
    assert kpis["decisions"] == len(env.cycle_log)
    assert kpis["wait_share_pct"] == pytest.approx(behaviour["wait_share_pct"])


def test_compteurs_remis_a_zero_au_reset():
    """Chaque épisode compte ses propres décisions."""
    env = MineEnv()
    env.reset(seed=42)
    env.step(env.action_space.n - 1)
    _, info = env.reset(seed=43)

    assert info["decisions"] == info["wait_actions"] == 0


def test_attendre_ne_paie_aucun_cout_de_distance():
    """« Savoir expliquer » de T19 : une décision ATTENDRE est récompensée sans coût de
    distance. Au début du poste (pelles libres, variance nulle), sa récompense se réduit à
    −w4 × min(durée / 30, 1) : ici 5 min, soit −0,05, et seulement −0,002 pour 0,2 min."""
    env = MineEnv()
    env.reset(seed=42)
    _, reward, *_ = env.step(env.action_space.n - 1)
    wait_min = env.trucks[0].voluntary_wait_min

    assert wait_min == pytest.approx(5.0)
    assert reward == pytest.approx(-env.w4 * min(wait_min / 30.0, 1.0))
    assert -env.w4 * 0.2 / 30.0 == pytest.approx(-0.002)


def test_un_cycle_paie_la_distance():
    """À l'inverse, un cycle paie sa distance : à tonnage égal, la récompense baisse de
    w3 × distance / 5 (cycle vers la pelle la plus éloignée)."""
    rewards = []
    for action in (0, 4):  # pelle 1 (la plus proche) puis pelle 3 (la plus éloignée), verse 1
        env = MineEnv()
        env.reset(seed=42)
        _, reward, *_ = env.step(action)
        rewards.append(reward)

    assert rewards[1] < rewards[0]
