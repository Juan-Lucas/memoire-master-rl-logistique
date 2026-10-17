"""Tests de T19 : comptage des décisions ATTENDRE et coût d'ATTENDRE dans la récompense."""

import numpy as np
import pytest

from memoire_master_rl_logistique.analysis.behavior import episode_behaviour
from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments.policies import heuristic_policies, make_env
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.simulation.reward import OPPORTUNITY_CYCLE_MIN

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


def test_cout_d_attendre():
    """« Savoir expliquer » de T19, et sa correction en v2. ATTENDRE ne paie toujours aucun
    coût de distance. Dans la v1, au début du poste (pelles libres, variance nulle), sa
    récompense se réduisait à −w4 × min(durée / 30, 1) : −0,05 pour 5 min, et seulement
    −0,002 pour 0,2 min. La v2 ajoute la production perdue, −w1 × durée / 23,2 min :
    −0,216 de plus pour 5 min, soit environ le cinquième d'un cycle.

    ATTENDRE est la seule récompense versée au moment de la décision : l'immobilité est
    connue tout de suite (celle d'un cycle n'est connue qu'à sa fin, T15)."""
    env = MineEnv()
    env.reset(seed=42)
    _, reward, *_ = env.step(env.action_space.n - 1)
    wait_min = env.trucks[0].voluntary_wait_min

    assert wait_min == pytest.approx(5.0)
    v1 = -env.w4 * min(wait_min / 30.0, 1.0)
    assert v1 == pytest.approx(-0.05)
    assert reward == pytest.approx(v1 - env.w1 * wait_min / OPPORTUNITY_CYCLE_MIN)
    assert reward == pytest.approx(-0.2655, abs=1e-4)
    assert -env.w4 * 0.2 / 30.0 == pytest.approx(-0.002)


def test_un_cycle_paie_la_distance():
    """À l'inverse, un cycle paie sa distance : w3 × distance / 5, la distance allant du
    lieu de la décision à la pelle puis à la verse (T16 bis). Avec un seul camion, son
    cycle se termine pendant son propre ``step`` : la récompense y est versée."""
    terms = []
    for action in (0, 4):  # pelle 1 (la plus proche) puis pelle 3 (la plus éloignée), verse 1
        env = MineEnv(truck_count=1, breakdown_probability=0.0)
        env.reset(seed=42)
        _, _, _, _, info = env.step(action)
        terms.append(info["reward_terms"]["distance"])

    g = env.graph
    assert terms[0] == pytest.approx(-0.05 * (g.shortest_distance("yard", "shovel_1")
                                              + g.shortest_distance("shovel_1", "dump_1")) / 5)
    assert terms[1] < terms[0]
