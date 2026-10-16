"""Tests de T14 : carburant paramétrable et calibré sur une référence publiée."""

import pandas as pd
import pytest

from memoire_master_rl_logistique.baselines import shortest_path_policy as sp
from memoire_master_rl_logistique.experiments.policies import (
    _baseline_fn,
    heuristic_policies,
    make_env,
)
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS, SCENARIOS
from memoire_master_rl_logistique.simulation.constants import (
    FUEL_CALIBRATION_FACTOR as F,
)
from memoire_master_rl_logistique.simulation.fuel_model import (
    DEFAULT_FUEL,
    FuelParams,
    estimate_idle_fuel_l,
    estimate_travel_fuel_l,
)

UNCALIBRATED = FuelParams(l_per_km_loaded=0.65, l_per_km_empty=0.45, idle_l_per_h=10.0)
CAT_785D_MEDIUM_L_PER_H = (81.4, 108.5)  # Caterpillar Performance Handbook, usage moyen


def test_un_seul_facteur_pour_toutes_les_consommations():
    """Calibré = non calibré × facteur, pour les trajets (chargé, à vide) et le ralenti."""
    assert F == pytest.approx(6.49, abs=0.01)
    for loaded in (True, False):
        calibrated = estimate_travel_fuel_l(2.0, 5.0, 0.8, loaded)
        assert calibrated == pytest.approx(F * estimate_travel_fuel_l(2.0, 5.0, 0.8, loaded, UNCALIBRATED))
    assert estimate_idle_fuel_l(30.0) == pytest.approx(F * estimate_idle_fuel_l(30.0, UNCALIBRATED))


def test_parametres_modifiables():
    """Le modèle est paramétrable sans toucher au code (étude de sensibilité)."""
    assert estimate_idle_fuel_l(60.0, FuelParams(idle_l_per_h=20.0)) == pytest.approx(20.0)
    assert DEFAULT_FUEL.idle_l_per_h == pytest.approx(10.0 * F)


def test_calibration_sur_le_cat_785d():
    """Affectation fixe, nominal, 30 graines : ≈ 95 L/h par camion, dans la plage
    « usage moyen » publiée pour le Cat 785D (81,4 à 108,5 L/h).

    Le facteur a été calibré sur le moteur de T13 (95,0 L/h). Avec le moteur à événements
    (T15, T16 bis), la même flotte consomme 95,7 L/h : le facteur, propriété du camion
    (L/km, L/h au ralenti), n'est pas recalé à chaque changement du simulateur."""
    scenario = SCENARIOS["nominal"]
    policy = heuristic_policies(scenario)["Fixed"]
    runs = pd.DataFrame([run_single_episode(make_env(scenario), policy, s) for s in scenario.seeds])
    l_per_h = (runs["total_fuel_l"] / runs["truck_count"] / (scenario.episode_minutes / 60)).mean()

    assert l_per_h == pytest.approx(95.72, abs=0.05)
    assert CAT_785D_MEDIUM_L_PER_H[0] <= l_per_h <= CAT_785D_MEDIUM_L_PER_H[1]
    assert runs["cost_per_cycle"].mean() == pytest.approx(37.9, abs=0.1)  # L par cycle


# Le test « l'agent voit les mêmes observations qu'avant T14 » a été retiré en T18 :
# l'observation a changé (variables agrégées), donc tous les agents profonds seront
# réentraînés de toute façon. Le carburant de l'observation est testé dans test_observation.py.


@pytest.mark.parametrize("scenario", MAIN_SCENARIOS)
def test_plus_court_chemin_decide_comme_avant(scenario, monkeypatch):
    """γ divisé par le facteur : « Plus court chemin » prend les mêmes décisions qu'avant
    T14 (γ = 0,2 avec le carburant non calibré)."""
    s = SCENARIOS[scenario]
    calibrated = [run_single_episode(make_env(s), heuristic_policies(s)["ShortestPath"], seed)
                  for seed in s.seeds[:5]]

    original = sp.estimate_travel_fuel_l
    monkeypatch.setattr(sp, "estimate_travel_fuel_l",
                        lambda **kw: original(**kw, params=UNCALIBRATED))
    env = make_env(s)
    env.reset(seed=s.seeds[0])
    shovels, dumps = [x.node_id for x in env.shovels], [x.node_id for x in env.dumps]
    before = [run_single_episode(
        make_env(s, fuel=UNCALIBRATED),
        _baseline_fn(sp.ShortestPathPolicy(env.graph, shovels, dumps, gamma=0.2), True), seed)
        for seed in s.seeds[:5]]

    for new, old in zip(calibrated, before):
        assert new["productivity_tph"] == old["productivity_tph"]
        assert new["total_fuel_l"] == pytest.approx(F * old["total_fuel_l"])
