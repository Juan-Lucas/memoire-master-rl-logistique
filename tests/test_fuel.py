"""Tests du modèle de carburant : ralenti et pente mesurés par Kubler (2015) sur un Cat 785D,
niveau moyen ramené au Caterpillar Performance Handbook (modèle du 22 octobre 2026)."""

import pandas as pd
import pytest

from memoire_master_rl_logistique.baselines import shortest_path_policy as sp
from memoire_master_rl_logistique.experiments.policies import heuristic_policies, make_env
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.seeds import TUNING_SEEDS
from memoire_master_rl_logistique.simulation.constants import (
    FUEL_REFERENCE_L_PER_H,
    FUEL_TRAVEL_SCALE,
)
from memoire_master_rl_logistique.simulation.fuel_model import (
    DEFAULT_FUEL,
    FuelParams,
    estimate_idle_fuel_l,
    estimate_travel_fuel_l,
    l_per_km,
)

CAT_785D_MEDIUM_L_PER_H = (81.4, 108.5)  # Handbook éd. 49, section 25, usage moyen
CAT_785D_MAX_L_PER_H = 135.6  # usage intensif, borne haute


def test_courbes_de_kubler():
    """Valeurs lues sur les figures 29 et 30 de Kubler (L/km) : environ 4,7 à vide et
    8,5 en charge à 1 % de pente ; environ 21 en charge et 11 à vide à 10 %."""
    assert l_per_km(1.0, loaded=False) == pytest.approx(4.70, abs=0.01)
    assert l_per_km(1.0, loaded=True) == pytest.approx(8.50, abs=0.01)
    assert l_per_km(10.0, loaded=True) == pytest.approx(20.96, abs=0.01)
    assert l_per_km(10.0, loaded=False) == pytest.approx(10.86, abs=0.01)
    # Chargé consomme plus qu'à vide, et la consommation croît avec la pente.
    for slope in (0.0, 2.0, 5.0, 8.0):
        assert l_per_km(slope, True) > l_per_km(slope, False)
        assert l_per_km(slope + 1.0, True) > l_per_km(slope, True)


def test_pente_bornee_au_domaine_mesure():
    """Une descente compte comme du plat ; au-delà de 18 %, la valeur à 18 %."""
    assert l_per_km(-5.0, True) == l_per_km(0.0, True)
    assert l_per_km(30.0, False) == l_per_km(18.0, False)


def test_trajet_et_ralenti():
    """Trajet = distance × échelle × L/km × majoration de piste ; ralenti mesuré 11,3 L/h."""
    assert estimate_idle_fuel_l(60.0) == pytest.approx(11.3)
    assert estimate_travel_fuel_l(2.0, 5.0, 1.0, True) == pytest.approx(
        2.0 * FUEL_TRAVEL_SCALE * l_per_km(5.0, True))
    assert estimate_travel_fuel_l(2.0, 5.0, 0.6, False) == pytest.approx(
        2.0 * FUEL_TRAVEL_SCALE * l_per_km(5.0, False) * 1.2)


def test_parametres_modifiables():
    """Le modèle est paramétrable sans toucher au code (étude de sensibilité)."""
    assert estimate_idle_fuel_l(60.0, FuelParams(idle_l_per_h=20.0)) == pytest.approx(20.0)
    flat = FuelParams(loaded_poly=(5.0,), empty_poly=(3.0,), travel_scale=1.0)
    assert estimate_travel_fuel_l(1.0, 7.0, 1.0, True, flat) == pytest.approx(5.0)
    assert DEFAULT_FUEL.idle_l_per_h == pytest.approx(11.3)


def _fixed_l_per_h(seeds):
    scenario = SCENARIOS["nominal"]
    policy = heuristic_policies(scenario)["Fixed"]
    runs = pd.DataFrame([run_single_episode(make_env(scenario), policy, s) for s in seeds])
    return (runs["total_fuel_l"] / runs["truck_count"] / (scenario.episode_minutes / 60)).mean()


def test_echelle_calibree_sur_les_graines_de_selection():
    """Le facteur d'échelle du roulage donne 95 L/h (milieu de la plage « usage moyen »)
    pour l'Affectation fixe en nominal, sur les graines de SÉLECTION : aucun choix n'est
    fait sur les graines d'évaluation."""
    assert _fixed_l_per_h(TUNING_SEEDS) == pytest.approx(FUEL_REFERENCE_L_PER_H, abs=0.01)


def test_graines_d_evaluation_dans_la_plage_publiee():
    """Sur les graines d'évaluation, la consommation reste dans la plage « usage moyen »
    du Cat 785D (95,4 L/h mesurés le 22/10/2026)."""
    l_per_h = _fixed_l_per_h(SCENARIOS["nominal"].seeds)
    assert l_per_h == pytest.approx(95.45, abs=0.05)
    assert CAT_785D_MEDIUM_L_PER_H[0] <= l_per_h <= CAT_785D_MEDIUM_L_PER_H[1]


def test_sans_echelle_le_simulateur_depasse_le_maximum_publie():
    """Les courbes de Kubler appliquées telles quelles donnent plus que le maximum publié :
    c'est ce qui justifie le facteur d'échelle (toutes les pistes du simulateur montent)."""
    scenario = SCENARIOS["nominal"]
    policy = heuristic_policies(scenario)["Fixed"]
    raw = FuelParams(travel_scale=1.0)
    runs = pd.DataFrame([run_single_episode(make_env(scenario, fuel=raw), policy, s)
                         for s in TUNING_SEEDS[:3]])
    assert (runs["total_fuel_l"] / runs["truck_count"] / 8).mean() > CAT_785D_MAX_L_PER_H


def test_poids_du_carburant_de_plus_court_chemin_inchange():
    """γ garde la valeur de T14 (0,2 divisé par l'ancien facteur 95 / 14,635)."""
    assert sp.DEFAULT_GAMMA == pytest.approx(0.2 / (95.0 / 14.635))
