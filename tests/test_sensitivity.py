"""Variantes du simulateur et expérience de sensibilité (M2, T1, T14 du rapport du 17/11/2026)."""

import pytest

from memoire_master_rl_logistique.experiments import policies
from memoire_master_rl_logistique.experiments import sensitivity as sens
from memoire_master_rl_logistique.experiments.policies import heuristic_policies, make_env
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.simulation.variants import NOMINAL_VARIANT, SimulationVariant

SEED = 20_000  # graine de sélection


def _fixed_episode(scenario):
    env = make_env(scenario)
    return env, run_single_episode(env, heuristic_policies(scenario)["Fixed"], SEED)


def test_variante_nominale_inchangee():
    """La variante de référence redonne exactement le simulateur de la campagne."""
    nominal = SCENARIOS["nominal"]
    _, kpis = _fixed_episode(nominal)
    _, kpis_ref = _fixed_episode(sens.variant_scenario(sens.REFERENCE, [SEED]))
    assert sens.variant_scenario(sens.REFERENCE).variant == NOMINAL_VARIANT
    assert kpis_ref["productivity_tph"] == kpis["productivity_tph"]
    assert kpis_ref["total_fuel_l"] == kpis["total_fuel_l"]


def test_durees_de_service_et_mtwt(monkeypatch):
    """Durées de chargement × 1,2 : moyenne et écart type des pelles, et MTWT les connaît."""
    seen = []
    original = policies.MTWTPolicy

    def recording(*args, **kwargs):
        seen.append(kwargs.get("load_time_min"))
        return original(*args, **kwargs)

    monkeypatch.setattr(policies, "MTWTPolicy", recording)
    scenario = sens.variant_scenario("load_x1.2", [SEED])
    env = make_env(scenario)
    env.reset(seed=SEED)
    assert all(s.load_time_mean_min == pytest.approx(2.4) for s in env.shovels)
    assert all(s.load_time_std_min == pytest.approx(0.36) for s in env.shovels)
    heuristic_policies(scenario)
    assert seen == [pytest.approx(2.4)]


def test_vitesses():
    """Vitesses × 1,1 : chaque trajet moyen est divisé par 1,1."""
    nominal, fast = make_env(SCENARIOS["nominal"]), make_env(sens.variant_scenario("speed_x1.1"))
    nominal.reset(seed=SEED)
    fast.reset(seed=SEED)
    for src, dst in (("dump_1", "shovel_2"), ("shovel_3", "dump_2")):
        assert fast.graph.mean_route_time_minutes(src, dst, loaded=False) == pytest.approx(
            nominal.graph.mean_route_time_minutes(src, dst, loaded=False) / 1.1)


def test_retours_sans_pente():
    """Retours à vide verse → pelle sans pente : moins de carburant, rien d'autre ne change
    dans le graphe."""
    env, kpis = _fixed_episode(sens.variant_scenario("flat_returns", [SEED]))
    _, kpis_ref = _fixed_episode(SCENARIOS["nominal"])
    for d in env.dumps:
        for s in env.shovels:
            assert env.graph.get_edge(d.node_id, s.node_id).slope_pct == 0.0
    assert env.graph.get_edge("shovel_1", "dump_1").slope_pct > 0
    assert kpis["specific_fuel_l_per_ton"] < kpis_ref["specific_fuel_l_per_ton"]


def test_usure_des_pistes():
    """Sans usure, aucune piste ne se dégrade pendant le poste."""
    scenario = sens.variant_scenario("reference", [SEED])
    scenario = sens.replace(scenario, variant=SimulationVariant(road_wear_probability=0.0))
    env, _ = _fixed_episode(scenario)
    assert all(env.graph.get_edge(a, b).road_state == 1.0 for a, b in env.graph.edges)


def test_variante_invalide():
    with pytest.raises(ValueError):
        SimulationVariant(speed_scale=0.0)
    with pytest.raises(ValueError):
        SimulationVariant(road_wear_probability=1.5)


def test_experience_courte(tmp_path):
    """Deux variantes, heuristiques seules, deux graines : une ligne par épisode, et la
    référence a un classement identique à lui-même (tau = 1)."""
    out = sens.run(tmp_path, models_dir=None, workers=1,
                   variants=(sens.REFERENCE, "idle_15"), seeds=(SEED, SEED + 1))
    with pytest.raises(FileExistsError):
        sens.run(tmp_path, models_dir=None, variants=(sens.REFERENCE,), seeds=(SEED,))
    import pandas as pd
    rows = pd.read_csv(out)
    assert len(rows) == 2 * len(policies.HEURISTICS) * 2
    assert set(rows["variant"]) == {sens.REFERENCE, "idle_15"}
    summary = sens.analyze(tmp_path)
    ref = summary.set_index("variant").loc[sens.REFERENCE]
    assert ref["tau_productivity"] == pytest.approx(1.0)
    # Le ralenti ne change que le carburant : même productivité, carburant plus élevé.
    by = rows.groupby(["variant", "policy"])[["productivity_tph", "specific_fuel_l_per_ton"]].mean()
    assert by.loc["idle_15", "productivity_tph"].equals(by.loc[sens.REFERENCE, "productivity_tph"])
    assert (by.loc["idle_15", "specific_fuel_l_per_ton"]
            > by.loc[sens.REFERENCE, "specific_fuel_l_per_ton"]).all()
