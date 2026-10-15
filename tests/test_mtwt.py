"""Tests de T23 : règle du temps d'attente minimal des camions (MTWT)."""

import pandas as pd
import pytest

from memoire_master_rl_logistique.baselines.base import encode_action
from memoire_master_rl_logistique.baselines.fifo_policy import FIFOPolicy
from memoire_master_rl_logistique.baselines.mtwt_policy import MTWTPolicy
from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments.policies import heuristic_policies, make_env
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS


@pytest.fixture(scope="module")
def graph():
    env = MineEnv()
    env.reset(seed=42)
    return env.graph


@pytest.fixture(scope="module")
def mtwt(graph):
    return MTWTPolicy(graph, ["shovel_1", "shovel_2", "shovel_3"], ["dump_1", "dump_2"])


def travel(graph, src, dst, loaded=False):
    return graph.mean_route_time_minutes(src, dst, loaded)


def test_temps_moyen_d_un_trajet(graph):
    """Le temps d'un trajet est la somme des temps moyens des arêtes du plus court chemin."""
    path = graph.shortest_path("yard", "shovel_2")
    expected = sum(graph.mean_travel_time_minutes(a, b, False) for a, b in zip(path, path[1:]))

    assert travel(graph, "yard", "shovel_2") == pytest.approx(expected)
    assert travel(graph, "shovel_2", "shovel_2") == 0.0


def test_cas_du_plan(graph, mtwt):
    """Test du plan : pelle 1 libre dans 10 min, pelle 2 libre maintenant mais plus loin
    (≈ 0,8 min de trajet de plus depuis le parc) : MTWT choisit la pelle 2."""
    now = 100.0
    shovel_1_free = now + travel(graph, "yard", "shovel_1") + 10.0  # 10 min d'attente à l'arrivée
    shovels = [shovel_1_free, now, now + 60.0]

    shovel, _ = mtwt.choose("yard", now, shovels, [now, now])

    assert travel(graph, "yard", "shovel_2") > travel(graph, "yard", "shovel_1")
    assert shovel == 1


def test_mtwt_differe_de_fifo(graph, mtwt):
    """Camion à la pelle 1, pelle 1 libre dans 5 min, pelle 2 libre dans 6 min mais à 7 min :
    FIFO choisit la pelle 1 (libre le plus tôt) ; MTWT la pelle 2, où il n'attendra pas.
    C'est la différence entre les deux règles : MTWT tient compte du trajet."""
    now = 0.0
    shovels, dumps = [5.0, 6.0, 60.0], [0.0, 0.0]
    info = {"current_time_min": now, "shovel_available_at": shovels, "dump_available_at": dumps}

    fifo = FIFOPolicy(num_shovels=3, num_dumps=2).predict(None, info, truck_location="shovel_1")
    mtwt_action = mtwt.predict(None, info, truck_location="shovel_1")

    assert travel(graph, "shovel_1", "shovel_2") > 6.0
    assert fifo // 2 == 0  # FIFO : pelle 1, attente 5 min
    assert mtwt_action // 2 == 1  # MTWT : pelle 2, attente 0 à l'arrivée


def test_egalite_departagee_par_l_arrivee(graph, mtwt):
    """Aucune attente nulle part : la pelle où l'on arrive le plus tôt (la pelle 1 depuis le parc)."""
    shovel, _ = mtwt.choose("yard", 0.0, [0.0, 0.0, 0.0], [0.0, 0.0])

    assert shovel == 0


def test_verse_de_plus_petite_attente(graph, mtwt):
    """Depuis la pelle 1 : la verse 1 est occupée longtemps après l'arrivée, la verse 2 libre."""
    arrival_at_dump_1 = 2.0 + travel(graph, "shovel_1", "dump_1", loaded=True)
    _, dump = mtwt.choose("shovel_1", 0.0, [0.0, 50.0, 50.0], [arrival_at_dump_1 + 20.0, 0.0])

    assert dump == 1


def test_sans_information():
    """Sans ``info`` ni position : action 0."""
    policy = MTWTPolicy(None, ["shovel_1"], ["dump_1"])

    assert policy.predict(None, None) == encode_action(0, 0, 1)


def test_moins_de_file_que_fifo_et_fixe():
    """Sur 5 graines par régime, MTWT n'utilise jamais ATTENDRE et a la plus petite file par
    camion des trois règles ; elle produit plus que FIFO, mais moins que l'Affectation fixe."""
    rows = []
    for name in ("nominal", "high_load", "high_breakdown"):
        scenario = SCENARIOS[name]
        policies = heuristic_policies(scenario)
        for policy in ("FIFO", "Fixed", "MTWT"):
            for seed in scenario.seeds[:5]:
                kpis = run_single_episode(make_env(scenario), policies[policy], seed)
                rows.append({"scenario": name, "policy": policy, **kpis})
    means = pd.DataFrame(rows).groupby(["scenario", "policy"]).mean(numeric_only=True)

    for name in ("nominal", "high_load", "high_breakdown"):
        m = means.loc[name]
        assert m.loc["MTWT", "wait_actions"] == 0
        assert m.loc["MTWT", "queue_wait_min_per_truck"] < min(
            m.loc["FIFO", "queue_wait_min_per_truck"], m.loc["Fixed", "queue_wait_min_per_truck"])
        assert m.loc["FIFO", "productivity_tph"] < m.loc["MTWT", "productivity_tph"] < (
            m.loc["Fixed", "productivity_tph"])
