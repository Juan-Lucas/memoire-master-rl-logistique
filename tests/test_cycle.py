"""Tests de T12 : journal des cycles (``MineEnv(log_cycles=True)``) et cohérence des durées."""

import numpy as np
import pandas as pd
import pytest

from memoire_master_rl_logistique.baselines.fixed_policy import FixedAssignmentPolicy
from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS, SCENARIOS
from memoire_master_rl_logistique.simulation.entities import CycleRecord


def fixed_fn():
    policy = FixedAssignmentPolicy(num_shovels=3, num_dumps=2)
    return lambda obs, info, env: policy.predict(
        obs, info, truck_location=env.truck_locations[env.current_truck_idx])


def random_fn(seed):
    """Politique aléatoire, ATTENDRE compris, pour couvrir les deux types d'entrées."""
    rng = np.random.default_rng(seed)
    return lambda obs, info, env: int(rng.integers(env.action_space.n))


def logged_episode(scenario="nominal", policy_fn=None, seed=42):
    s = SCENARIOS[scenario]
    env = MineEnv(truck_count=s.truck_count, breakdown_probability=s.breakdown_probability,
                  reward_weights=s.reward_weights, log_cycles=True)
    kpis = run_single_episode(env, policy_fn or fixed_fn(), seed)
    return env, kpis, pd.DataFrame([r.as_dict() for r in env.cycle_log])


def test_journal_desactive_par_defaut():
    """Sans ``log_cycles``, rien n'est enregistré (aucun coût pendant l'entraînement)."""
    env = MineEnv()
    run_single_episode(env, fixed_fn(), seed=42)

    assert env.cycle_log == []


@pytest.mark.parametrize("scenario", MAIN_SCENARIOS)
@pytest.mark.parametrize("policy", ["Fixed", "aléatoire"])
def test_fin_moins_debut_egale_la_somme_des_composantes(scenario, policy):
    """Test du plan : pour chaque cycle, fin − début = somme des composantes, à 1e-6 près."""
    _, _, log = logged_episode(scenario, fixed_fn() if policy == "Fixed" else random_fn(0))
    components = log[list(CycleRecord.COMPONENTS)].sum(axis=1)

    assert len(log) > 0
    assert np.allclose(log["end_min"] - log["start_min"], components, atol=1e-6, rtol=0)
    assert (log[list(CycleRecord.COMPONENTS)] >= 0).all().all()


@pytest.mark.parametrize("policy", ["Fixed", "aléatoire"])
def test_chronologie_continue_de_chaque_camion(policy):
    """Chaque camion commence à 0, et chaque décision commence exactement à la fin de la
    précédente : aucune minute n'échappe au journal."""
    _, _, log = logged_episode("high_breakdown", fixed_fn() if policy == "Fixed" else random_fn(1))

    for _, truck in log.sort_values("start_min").groupby("truck_id"):
        assert truck["start_min"].iloc[0] == 0.0
        assert np.allclose(truck["start_min"].iloc[1:].to_numpy(),
                           truck["end_min"].iloc[:-1].to_numpy(), atol=1e-9, rtol=0)


def test_journal_coherent_avec_les_compteurs():
    """Tonnage et nombre de cycles livrés d'après le journal = compteurs de l'épisode."""
    env, kpis, log = logged_episode()

    assert log["delivered_t"].sum() == pytest.approx(kpis["total_tonnage_t"])
    assert (log["delivered_t"] > 0).sum() == kpis["total_cycles"]
    assert set(log["kind"]) == {"cycle"}  # l'Affectation fixe n'utilise jamais ATTENDRE


def test_attendre_est_journalise():
    """Une action ATTENDRE donne une entrée « wait » sans pelle ni verse."""
    _, _, log = logged_episode(policy_fn=random_fn(0))
    waits = log[log["kind"] == "wait"]

    assert len(waits) > 0
    assert waits["shovel_id"].isna().all()
    assert (waits["voluntary_wait_min"] >= 0.2).all()


def test_le_journal_ne_change_pas_la_simulation():
    """Avec ou sans journal, mêmes résultats (références v1 de T4)."""
    _, with_log, _ = logged_episode()
    without_log = run_single_episode(MineEnv(), fixed_fn(), seed=42)

    assert with_log["productivity_tph"] == without_log["productivity_tph"] == pytest.approx(4077.5)
    assert with_log["total_reward"] == pytest.approx(without_log["total_reward"])


def test_reset_vide_le_journal():
    """Chaque épisode a son propre journal."""
    env, _, log = logged_episode()
    env.reset(seed=43)

    assert len(log) > 0 and env.cycle_log == []
