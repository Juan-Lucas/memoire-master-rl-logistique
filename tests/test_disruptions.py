"""Panne de pelle pendant le poste et régimes proches de la saturation (M3, option A)."""

import pandas as pd
import pytest

from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments import disruptions as dis
from memoire_master_rl_logistique.experiments.policies import HEURISTICS, heuristic_policies, make_env
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.simulation.variants import SimulationVariant

SEED = 20_000  # graine de sélection
OUTAGE = (0, 120.0, 240.0)


def test_disponibilite_de_la_pelle():
    """Pendant la panne, une pelle n'est prête qu'à sa fin ; une panne pas encore commencée
    est inconnue d'une prévision faite avant."""
    v = SimulationVariant(shovel_outages=(OUTAGE,))
    assert v.shovel_ready_at(0, 100.0) == 100.0
    assert v.shovel_ready_at(0, 150.0) == 240.0
    assert v.shovel_ready_at(1, 150.0) == 150.0           # autre pelle
    assert v.shovel_ready_at(0, 150.0, now=110.0) == 150.0  # pas encore commencée
    assert v.shovel_ready_at(0, 150.0, now=130.0) == 240.0
    with pytest.raises(ValueError):
        SimulationVariant(shovel_outages=((0, 50.0, 40.0),))


def _episode(variant, policy="Fixed"):
    scenario = dis.regime_scenario(dis.REFERENCE, [SEED])
    scenario = dis.replace(scenario, variant=variant)
    env = make_env(scenario, log_cycles=True)
    kpis = run_single_episode(env, heuristic_policies(scenario)[policy], SEED)
    return env, kpis


def test_aucun_chargement_pendant_la_panne():
    """Aucun chargement de la pelle 1 ne commence pendant la panne ; la production baisse."""
    env, kpis = _episode(SimulationVariant(shovel_outages=(OUTAGE,)))
    _, kpis_ref = _episode(SimulationVariant())
    for c in env.cycle_log:
        if c.kind == "cycle" and c.shovel_id == "S1":
            start_load = c.start_min + c.travel_to_shovel_min + c.shovel_wait_min
            assert not 120.0 <= start_load < 240.0 - 1e-9
    assert kpis["productivity_tph"] < kpis_ref["productivity_tph"]


def test_la_prevision_voit_la_panne_une_fois_commencee():
    """Avant la panne, la prévision l'ignore ; pendant, la pelle 1 est libre à sa fin."""
    env = MineEnv(variant=SimulationVariant(shovel_outages=(OUTAGE,)), breakdown_probability=0.0)
    env.reset(seed=SEED)
    assert env.forecast_free_at()[0][0] < 120.0
    env.current_time_min = 130.0
    for s in env.shovels:
        s.available_at_min = 0.0
    assert env.forecast_free_at()[0][0] == 240.0
    assert env.forecast_free_at()[0][1] == 130.0  # libre dès maintenant


def test_sans_panne_rien_ne_change():
    """La variante sans panne redonne exactement le simulateur nominal."""
    _, kpis = _episode(SimulationVariant())
    _, kpis_mtwt = _episode(SimulationVariant(), "MTWT")
    nominal = make_env(dis.SCENARIOS["nominal"])
    assert run_single_episode(nominal, heuristic_policies(dis.SCENARIOS["nominal"])["Fixed"],
                              SEED)["productivity_tph"] == kpis["productivity_tph"]
    assert kpis_mtwt["productivity_tph"] > 0


def test_regimes_de_saturation():
    """29 et 34 camions : facteur d'appariement 0,85 et 1,0 (2 min de chargement, cycle de
    22,7 min, 3 pelles)."""
    for key, mf in (("saturation_29", 0.85), ("saturation_34", 1.0)):
        trucks = dis.regime_scenario(key).truck_count
        assert trucks * 2.0 / (3 * 22.7) == pytest.approx(mf, abs=0.01)


def test_experience_courte(tmp_path):
    """Deux régimes, heuristiques seules, deux graines : tonnage par tranche cohérent avec la
    productivité, perte mesurée par rapport à la référence et récupération calculée."""
    out = dis.run(tmp_path, None, regimes=(dis.REFERENCE, dis.OUTAGE), seeds=(SEED, SEED + 1))
    rows = pd.read_csv(out)
    assert len(rows) == 2 * len(HEURISTICS) * 2
    bins = [c for c in rows.columns if c.startswith("t_")]
    assert len(bins) == 16
    assert (rows[bins].sum(axis=1) / 8.0).round(6).equals(rows["productivity_tph"].round(6))
    summary, recovery = dis.analyze(tmp_path)
    outage = summary[summary["regime"] == dis.OUTAGE].set_index("policy")
    assert (outage["loss_vs_reference_pct"] > 0).all()
    assert set(recovery["policy"]) == set(HEURISTICS)
    assert (recovery["tonnage_lost_t"] > 0).all()
