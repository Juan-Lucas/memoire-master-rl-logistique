"""Tests de T10 : décomposition de l'immobilisation en file, réparation et action ATTENDRE.

Les valeurs de référence du plan (22,2 / 7,2 / 1,1 min) ont évolué avec deux tâches :
- T11 borne les compteurs à la fin du poste : 21,80 / 6,72 / 1,13 min ;
- T13 change l'ordre des tirages aléatoires (même loi) : 22,26 / 7,37 / 1,15 min.
"""

import math

import pandas as pd
import pytest
from reference_values import reference

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


def test_carburant_par_tonne_non_defini_sans_tonne():
    """Un poste sans aucune tonne (agent qui attend tout le poste) n'a pas de carburant
    par tonne : NaN, et non le carburant divisé par 1e-9 (≈ 1e12 L/t avant le 28/10)."""
    common = dict(episode_minutes=480, truck_count=12, queue_wait_min=0, repair_min=0,
                  voluntary_wait_min=5000, total_active_min=0, total_fuel_l=1085.0)
    assert math.isnan(compute_kpis(total_tonnage_t=0, **common)["specific_fuel_l_per_ton"])
    assert compute_kpis(total_tonnage_t=2170.0, **common)["specific_fuel_l_per_ton"] == 0.5


def test_total_wait_est_la_somme_des_trois_compteurs():
    """``Truck.total_wait_min`` n'est plus un compteur : c'est la somme des trois."""
    truck = Truck("t1", queue_wait_min=10.0, repair_min=4.0, voluntary_wait_min=1.5)

    assert truck.total_wait_min == pytest.approx(15.5)


def test_total_egal_a_la_somme_des_trois():
    """Test du plan : la somme des trois compteurs est l'immobilisation totale.

    Historique, Fixed, nominal, graine 42 : 303,611 min en v1 (= ancien compteur unique,
    vérifié en T10), 299,886 min borné au poste (T11), 290,913 min (T13), 223,473 min
    depuis le moteur à événements (T15, T16 bis).
    """
    kpis = fixed_episode(42)

    assert kpis["queue_wait_min"] + kpis["repair_min"] + kpis["voluntary_wait_min"] == (
        pytest.approx(kpis["total_wait_min"]))
    assert kpis["total_wait_min"] == pytest.approx(
        reference("Fixed", 42)["total_wait_min"], abs=1e-6)


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

    Depuis le moteur à événements (T15, T16 bis) : 11,25 min de file et 7,69 min de
    réparation par camion et par poste, soit 0,56 min de file par cycle. La file a diminué
    de moitié : les verses ne servent plus les camions dans le désordre des arrivées.

    Historique : 22,26 / 7,37 / 1,15 (T13) ; 22,17 / 7,18 / 1,14 (T10, ancien tirage, non
    borné), dont la somme 29,36 min était le « 29 min » de la v1.
    """
    episodes = pd.DataFrame([fixed_episode(seed) for seed in SCENARIOS["nominal"].seeds])
    means = episodes.mean(numeric_only=True)

    assert means["queue_wait_min_per_truck"] == pytest.approx(11.25, abs=0.01)
    assert means["repair_min_per_truck"] == pytest.approx(7.69, abs=0.01)
    assert means["voluntary_wait_min_per_truck"] == 0.0
    assert means["queue_wait_min_per_cycle"] == pytest.approx(0.56, abs=0.01)
    assert means["avg_wait_min_per_truck"] == pytest.approx(18.94, abs=0.01)


def test_robustesse_utilise_l_attente_en_file():
    """Avec la colonne de file (depuis T10), le Δ attente de l'Affectation fixe :
    depuis T15 et T16 bis, +68,81 % en forte charge et −6,19 % en fortes pannes.
    (T13 : +56,46 / −13,78 ; T11 : +59,79 / −12,02 ; T10 : +60,44 / −12,21 ;
    article v2 : +60,5 / −12,2.)"""
    rows = [dict(fixed_episode(seed, scenario), policy="Fixed", scenario=scenario)
            for scenario in ("nominal", "high_load", "high_breakdown")
            for seed in SCENARIOS[scenario].seeds]
    df = pd.DataFrame(rows)

    assert wait_column(df) == QUEUE_WAIT
    assert wait_column(df.drop(columns=[QUEUE_WAIT])) == TOTAL_WAIT  # CSV v1
    delta = wait_change_pct(df)
    assert delta.loc["Fixed", "high_load"] == pytest.approx(68.81, abs=0.01)
    assert delta.loc["Fixed", "high_breakdown"] == pytest.approx(-6.19, abs=0.01)
