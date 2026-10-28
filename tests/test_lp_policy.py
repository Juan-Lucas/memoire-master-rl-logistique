"""Tests de T24 : politique fondée sur un programme linéaire (principe de DISPATCH)."""

import numpy as np
import pandas as pd
import pytest

from memoire_master_rl_logistique.baselines.lp_policy import (
    LPDispatchPolicy,
    LPPlan,
    select_max_utilization,
)
from memoire_master_rl_logistique.config import experiment_config
from memoire_master_rl_logistique.experiments.policies import (
    heuristic_policies,
    lp_plan,
    make_env,
)
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS, SCENARIOS
from memoire_master_rl_logistique.seeds import TUNING_SEEDS
from memoire_master_rl_logistique.simulation.constants import (
    LOAD_TIME_MEAN_MIN as LOAD_MIN,
)
from memoire_master_rl_logistique.simulation.constants import (
    UNLOAD_TIME_MEAN_MIN as UNLOAD_MIN,
)


def nominal_env():
    env = make_env(SCENARIOS["nominal"])
    env.reset(seed=42)
    return env


# --- Étage de planification ----------------------------------------------------------

@pytest.mark.parametrize("u", [0.7, 1.0])
@pytest.mark.parametrize("scenario", MAIN_SCENARIOS)
def test_le_plan_respecte_ses_contraintes(scenario, u):
    """Capacités des pelles et des verses (× u), heures-camions, conservation des camions
    à chaque pelle et à chaque verse."""
    s = SCENARIOS[scenario]
    plan = lp_plan(s, u)
    x, y = plan.rates_per_h, plan.empty_rates_per_h

    assert (x >= -1e-9).all() and (y >= -1e-9).all()
    assert (x.sum(axis=1) <= u * 60 / LOAD_MIN + 1e-6).all()      # pelles
    assert (x.sum(axis=0) <= u * 60 / UNLOAD_MIN + 1e-6).all()    # verses
    assert plan.truck_minutes_per_h <= 60 * s.truck_count + 1e-6   # heures-camions
    np.testing.assert_allclose(x.sum(axis=0), y.sum(axis=1), atol=1e-6)  # verses : entrée = sortie
    np.testing.assert_allclose(y.sum(axis=0), x.sum(axis=1), atol=1e-6)  # pelles : entrée = sortie
    assert x.sum() == pytest.approx(plan.max_production_per_h, rel=2e-3)


def test_sans_plafond_la_pelle_la_plus_proche_est_saturee():
    """u = 1 : le plan sature la pelle 1 (30 chargements par heure), la plus proche des
    verses ; c'est l'optimum sans aléa, qui accumule une file avec des durées aléatoires."""
    plan = lp_plan(SCENARIOS["nominal"], 1.0)
    assert plan.rates_per_h[0].sum() == pytest.approx(60 / LOAD_MIN)
    assert plan.max_production_per_h == pytest.approx(35.88, abs=0.01)


def test_plus_de_camions_plus_de_production_planifiee():
    nominal, high_load = (lp_plan(SCENARIOS[s], 0.7).max_production_per_h
                          for s in ("nominal", "high_load"))
    assert high_load > nominal


def test_plafond_invalide():
    env = nominal_env()
    for u in (0.0, 1.2):
        with pytest.raises(ValueError, match="max_utilization"):
            LPPlan(env.graph, ["shovel_1"], ["dump_1"], 12, [LOAD_MIN], [UNLOAD_MIN], 0.02, u)


# --- Étage d'affectation en temps réel -------------------------------------------------

def test_route_la_plus_en_retard():
    """Deux routes au même taux : après deux camions envoyés sur la première, le suivant va
    sur la seconde ; la remise à zéro au début d'un épisode efface ce compteur."""
    env = nominal_env()
    plan = lp_plan(SCENARIOS["nominal"], 0.7)
    plan.rates_per_h = np.array([[10.0, 0.0], [10.0, 0.0], [0.0, 0.0]])
    policy = LPDispatchPolicy(env.graph, [s.node_id for s in env.shovels],
                              [d.node_id for d in env.dumps], plan)
    policy.dispatched[0, 0] = 2

    assert policy.choose("dump_1", now=60.0) == (1, 0)
    policy.predict(None, {"decisions": 0, "current_time_min": 0.0}, "yard")
    assert policy.dispatched.sum() == 1  # nouvel épisode : compteur remis à zéro


def test_episode_independant_du_precedent():
    """Le même objet politique rejoue un épisode à l'identique, quel que soit l'épisode
    joué avant (évaluation parallèle, T29)."""
    s = SCENARIOS["nominal"]
    policy = heuristic_policies(s)["LP"]
    first = run_single_episode(make_env(s), policy, 42)
    run_single_episode(make_env(s), policy, 43)
    again = run_single_episode(make_env(s), policy, 42)
    assert first["productivity_tph"] == again["productivity_tph"]
    assert first["total_reward"] == again["total_reward"]


def test_les_cycles_suivent_le_plan():
    """Sur un poste, la part des cycles de chaque route suit la part planifiée (± 5 points)."""
    s = SCENARIOS["nominal"]
    env = make_env(s, log_cycles=True)
    run_single_episode(env, heuristic_policies(s)["LP"], 42)
    plan = lp_plan(s)
    planned = plan.rates_per_h / plan.rates_per_h.sum()
    cycles = [c for c in env.cycle_log if c.kind == "cycle"]
    realised = np.zeros_like(planned)
    for c in cycles:
        realised[int(c.shovel_id[1:]) - 1, int(c.dump_id[1:]) - 1] += 1
    np.testing.assert_allclose(realised / len(cycles), planned, atol=0.05)


# --- Choix du plafond et performance --------------------------------------------------

def test_choix_du_plafond():
    best, scores = select_max_utilization([0.6, 0.7, 1.0], {0.6: 1.0, 0.7: 3.0, 1.0: 2.0}.get)
    assert best == 0.7 and scores == {0.6: 1.0, 0.7: 3.0, 1.0: 2.0}
    best, _ = select_max_utilization([1.0, 0.7], {0.7: 2.0, 1.0: 2.0}.get)
    assert best == 0.7  # à égalité, le plan le plus prudent


def test_plafond_de_la_configuration_choisi_sur_les_graines_de_selection():
    """La valeur de ``configs/experiment_config.yaml`` (0,7) est celle que redonne la
    sélection : productivité moyenne des trois régimes sur les graines 20 000 à 20 009,
    jamais sur les graines d'évaluation."""
    heuristics = experiment_config()["heuristics"]

    def evaluate(u):
        return np.mean([
            run_single_episode(make_env(SCENARIOS[s]), heuristic_policies(SCENARIOS[s], u)["LP"],
                               seed)["productivity_tph"]
            for s in MAIN_SCENARIOS for seed in TUNING_SEEDS
        ])

    best, _ = select_max_utilization(heuristics["lp_utilization_candidates"], evaluate)
    assert best == heuristics["lp_max_utilization"] == 0.7
    assert set(TUNING_SEEDS).isdisjoint(SCENARIOS["nominal"].seeds)


def test_meilleure_que_l_affectation_fixe():
    """Sur 5 graines par régime, la règle LP n'attend jamais et produit plus que
    l'Affectation fixe (30 graines : 4 404 contre 4 246 t/h en nominal ; MTWT 4 437)."""
    rows = []
    for name in MAIN_SCENARIOS:
        s = SCENARIOS[name]
        policies = heuristic_policies(s)
        for policy in ("Fixed", "LP"):
            for seed in s.seeds[:5]:
                rows.append({"scenario": name, "policy": policy,
                             **run_single_episode(make_env(s), policies[policy], seed)})
    means = pd.DataFrame(rows).groupby(["scenario", "policy"]).mean(numeric_only=True)
    for name in MAIN_SCENARIOS:
        assert means.loc[(name, "LP"), "wait_actions"] == 0
        assert (means.loc[(name, "LP"), "productivity_tph"]
                > means.loc[(name, "Fixed"), "productivity_tph"])
