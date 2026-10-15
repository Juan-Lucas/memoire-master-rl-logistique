"""Tests de T10 : décomposition de l'immobilisation en file, réparation et action ATTENDRE."""

import pandas as pd
import pytest

from memoire_master_rl_logistique.analysis.robustness import (
    QUEUE_WAIT,
    TOTAL_WAIT,
    wait_change_pct,
    wait_column,
)
from memoire_master_rl_logistique.baselines.fixed_policy import FixedAssignmentPolicy
from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.simulation.entities import Truck
from memoire_master_rl_logistique.simulation.kpi import compute_kpis


def fixed_episode(seed, scenario="nominal"):
    """KPI d'un épisode de l'Affectation fixe."""
    s = SCENARIOS[scenario]
    policy = FixedAssignmentPolicy(num_shovels=3, num_dumps=2)
    env = MineEnv(truck_count=s.truck_count, breakdown_probability=s.breakdown_probability,
                  reward_weights=s.reward_weights)
    return run_single_episode(
        env, lambda obs, info, e: policy.predict(
            obs, info, truck_location=e.truck_locations[e.current_truck_idx]), seed)


def test_compute_kpis_unites():
    """Totaux de flotte → par camion (sur le poste) et file par cycle."""
    kpis = compute_kpis(episode_minutes=480, truck_count=4, total_tonnage_t=0,
                        queue_wait_min=120, repair_min=60, voluntary_wait_min=20,
                        total_active_min=0, total_fuel_l=0, total_cycles=40)

    assert kpis["total_wait_min"] == 200
    assert kpis["avg_wait_min_per_truck"] == 50
    assert (kpis["queue_wait_min_per_truck"], kpis["repair_min_per_truck"],
            kpis["voluntary_wait_min_per_truck"]) == (30, 15, 5)
    assert kpis["queue_wait_min_per_cycle"] == 3


def test_total_wait_est_la_somme_des_trois_compteurs():
    """``Truck.total_wait_min`` n'est plus un compteur : c'est la somme des trois."""
    truck = Truck("t1", queue_wait_min=10.0, repair_min=4.0, voluntary_wait_min=1.5)

    assert truck.total_wait_min == pytest.approx(15.5)


def test_meme_total_que_la_v1():
    """Test du plan : la somme des trois compteurs égale l'ancien total_wait_min.

    Fixed, nominal, graine 42 : 303,611025 min dans le CSV de l'article v1.
    """
    kpis = fixed_episode(42)

    assert kpis["queue_wait_min"] + kpis["repair_min"] + kpis["voluntary_wait_min"] == (
        pytest.approx(kpis["total_wait_min"]))
    assert kpis["total_wait_min"] == pytest.approx(303.611025, abs=1e-6)


def test_action_attendre_compte_comme_attente_volontaire():
    """L'action ATTENDRE n'alimente que ``voluntary_wait_min``."""
    env = MineEnv()
    env.reset(seed=42)
    truck = env.trucks[env.current_truck_idx]

    env.step(env.action_space.n - 1)  # dernière action = ATTENDRE

    assert truck.voluntary_wait_min >= 0.2  # durée minimale de l'action
    assert truck.queue_wait_min == 0.0 and truck.repair_min == 0.0


def test_valeurs_de_reference_du_plan():
    """Test du plan : Affectation fixe, nominal, 30 graines.

    Environ 22,2 min de file et 7,2 min de réparation par camion et par poste,
    soit 1,1 min de file par cycle ; leur somme (29,4 min) est le « 29 min » de la v1.
    """
    episodes = pd.DataFrame([fixed_episode(seed) for seed in SCENARIOS["nominal"].seeds])
    means = episodes.mean(numeric_only=True)

    assert means["queue_wait_min_per_truck"] == pytest.approx(22.17, abs=0.01)
    assert means["repair_min_per_truck"] == pytest.approx(7.18, abs=0.01)
    assert means["voluntary_wait_min_per_truck"] == 0.0
    assert means["queue_wait_min_per_cycle"] == pytest.approx(1.14, abs=0.01)
    assert means["avg_wait_min_per_truck"] == pytest.approx(29.36, abs=0.01)


def test_robustesse_utilise_l_attente_en_file():
    """Avec la colonne de file (depuis T10), on retrouve le Δ attente du tableau 13 :
    Affectation fixe, forte charge : +60,44 % (l'article écrit +60,5, l'arrondi correct
    est +60,4) ; fortes pannes : −12,2 %."""
    rows = [dict(fixed_episode(seed, scenario), policy="Fixed", scenario=scenario)
            for scenario in ("nominal", "high_load", "high_breakdown")
            for seed in SCENARIOS[scenario].seeds]
    df = pd.DataFrame(rows)

    assert wait_column(df) == QUEUE_WAIT
    assert wait_column(df.drop(columns=[QUEUE_WAIT])) == TOTAL_WAIT  # CSV v1
    delta = wait_change_pct(df)
    assert delta.loc["Fixed", "high_load"] == pytest.approx(60.44, abs=0.01)
    assert delta.loc["Fixed", "high_breakdown"] == pytest.approx(-12.2, abs=0.05)
