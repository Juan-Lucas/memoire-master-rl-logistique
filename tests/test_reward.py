"""Tests de T25 : module de récompense séparé, termes exposés, valeurs inchangées."""

import numpy as np
import pandas as pd
import pytest

from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments.policies import heuristic_policies, make_env
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.simulation.reward import (
    RewardWeights,
    compute_reward,
    reward_terms,
)

WEIGHTS = RewardWeights.from_tuple((1.0, 0.1, 0.05, 0.3))


def v1_formula(tonnage, capacity, distance_km, waits, wait_min, w=(1.0, 0.1, 0.05, 0.3)):
    """La formule de ``MineEnv._compute_reward`` avant T25, recopiée telle quelle."""
    r1 = tonnage / max(capacity, 1e-9)
    if len(waits) >= 2:
        m = sum(waits) / len(waits)
        r2 = -(sum((x - m) ** 2 for x in waits) / len(waits)) / 100.0
    else:
        r2 = 0.0
    return w[0] * r1 + w[1] * r2 + w[2] * (-distance_km / 5.0) + w[3] * -min(wait_min / 30.0, 1.0)


def test_memes_valeurs_que_la_v1():
    """Sur 500 décisions aléatoires, le module donne exactement la récompense de la v1."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        args = (rng.choice([0.0, 140.0]), 140.0, rng.uniform(0, 30),
                list(rng.uniform(0, 40, 3)), rng.uniform(0, 60))
        assert compute_reward(reward_terms(*args), WEIGHTS) == pytest.approx(v1_formula(*args))


def test_termes_d_un_cycle_et_d_une_attente():
    """Cycle livré de 15 km sans file, pelles équilibrées : 1 − 0,05 × 3 = 0,85.
    ATTENDRE de 0,2 min : production et distance nulles, attente −0,2/30."""
    cycle = reward_terms(140.0, 140.0, 15.0, [2.0, 2.0, 2.0], 0.0)
    wait = reward_terms(0.0, 140.0, 0.0, [0.0, 0.0, 0.0], 0.2)

    assert (cycle.production, cycle.balance, cycle.distance, cycle.wait) == (1.0, 0.0, -3.0, 0.0)
    assert compute_reward(cycle, WEIGHTS) == pytest.approx(0.85)
    assert (wait.production, wait.distance) == (0.0, 0.0)
    assert wait.wait == pytest.approx(-0.2 / 30)


def test_plafonds_et_variance():
    """L'attente est plafonnée à −1 (30 min et plus) ; l'équilibre = −variance / 100."""
    terms = reward_terms(0.0, 140.0, 0.0, [0.0, 10.0, 20.0], 90.0)

    assert terms.wait == -1.0
    assert terms.balance == pytest.approx(-np.var([0.0, 10.0, 20.0]) / 100)


def test_poids_par_defaut():
    """Trois poids donnés : w4 = 0,3, comme dans la v1."""
    assert RewardWeights.from_tuple((1.0, 0.1, 0.05)).wait == 0.3


def test_somme_ponderee_egale_la_recompense_de_step():
    """Test du plan : à chaque décision (cycles et ATTENDRE), la somme des contributions
    pondérées exposées dans ``info`` égale la récompense renvoyée par ``step``."""
    env = MineEnv()
    rng = np.random.default_rng(0)
    env.reset(seed=42)
    total = 0.0
    while True:
        _, reward, terminated, truncated, info = env.step(int(rng.integers(env.action_space.n)))
        total += reward
        assert sum(info["reward_terms"].values()) == pytest.approx(reward)
        if terminated or truncated:
            break
    assert sum(info["episode_reward_terms"].values()) == pytest.approx(total)


def test_decomposition_d_un_poste_affectation_fixe():
    """Test du plan, Affectation fixe en nominal sur 30 graines : terme de distance ≈ −19 ;
    le terme de production égale le nombre de cycles, d'où « récompense ≈ nombre de cycles »."""
    scenario = SCENARIOS["nominal"]
    policy = heuristic_policies(scenario)["Fixed"]
    runs = pd.DataFrame([run_single_episode(make_env(scenario), policy, s) for s in scenario.seeds])
    terms = runs[["reward_production", "reward_balance", "reward_distance", "reward_wait"]]

    assert np.allclose(terms.sum(axis=1), runs["total_reward"])
    assert (runs["reward_production"] == runs["total_cycles"]).all()
    assert runs["reward_distance"].mean() == pytest.approx(-19.1, abs=0.05)
    assert runs["reward_wait"].mean() == pytest.approx(-2.71, abs=0.05)


@pytest.mark.parametrize(("scenario", "minutes"), [("nominal", 1.97), ("high_load", 1.36)])
def test_intervalle_entre_decisions(scenario, minutes):
    """« Savoir expliquer » : γ = 0,99 par décision, horizon 1/(1 − γ) = 100 décisions.
    Avec l'Affectation fixe : ≈ 1,97 min par décision avec 12 camions (≈ 197 min d'horizon),
    ≈ 1,36 min avec 18 camions (≈ 136 min)."""
    s = SCENARIOS[scenario]
    policy = heuristic_policies(s)["Fixed"]
    decisions = np.mean([run_single_episode(make_env(s), policy, seed)["decisions"]
                         for seed in s.seeds])

    assert s.episode_minutes / decisions == pytest.approx(minutes, abs=0.01)
