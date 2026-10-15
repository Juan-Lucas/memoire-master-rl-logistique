"""Tests de T11 : compteurs bornés à la fin du poste, utilisation des camions et des pelles."""

import numpy as np
import pandas as pd
import pytest

from memoire_master_rl_logistique.baselines.fixed_policy import FixedAssignmentPolicy
from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS, SCENARIOS


def make_env(scenario):
    s = SCENARIOS[scenario]
    return MineEnv(truck_count=s.truck_count, breakdown_probability=s.breakdown_probability,
                   reward_weights=s.reward_weights)


def fixed_fn():
    policy = FixedAssignmentPolicy(num_shovels=3, num_dumps=2)
    return lambda obs, info, env: policy.predict(
        obs, info, truck_location=env.truck_locations[env.current_truck_idx])


def random_fn(seed):
    """Politique aléatoire, ATTENDRE compris : couvre aussi l'attente volontaire."""
    rng = np.random.default_rng(seed)
    return lambda obs, info, env: int(rng.integers(env.action_space.n))


def test_in_shift():
    """Seule la part d'une activité située avant 480 min est comptée."""
    env = MineEnv()

    assert env._in_shift(100.0, 5.0) == 5.0  # entièrement dans le poste
    assert env._in_shift(470.0, 20.0) == pytest.approx(10.0)  # déborde de 10 min
    assert env._in_shift(500.0, 5.0) == 0.0  # entièrement après


@pytest.mark.parametrize("scenario", MAIN_SCENARIOS)
@pytest.mark.parametrize("policy", ["Fixed", "aléatoire"])
def test_temps_de_chaque_camion_borne_au_poste(scenario, policy):
    """Test du plan : pour chaque camion, temps actif + attentes ≤ 480 min ;
    pour chaque pelle, temps de chargement ≤ 480 min."""
    for seed in (42, 43, 44):
        env = make_env(scenario)
        run_single_episode(env, fixed_fn() if policy == "Fixed" else random_fn(seed), seed)

        for truck in env.trucks:
            assert truck.total_active_min + truck.total_wait_min <= env.episode_minutes + 1e-9
        for shovel in env.shovels:
            assert 0 < shovel.busy_min <= env.episode_minutes + 1e-9


def test_la_borne_ne_change_pas_la_simulation():
    """Borner les compteurs ne change ni la production ni la récompense (références v1 de T4)."""
    kpis = run_single_episode(MineEnv(), fixed_fn(), seed=42)

    assert kpis["productivity_tph"] == pytest.approx(4077.5)
    assert kpis["total_reward"] == pytest.approx(210.9715212200733, abs=1e-6)


def test_utilisation_affectation_fixe():
    """Affectation fixe, 30 graines : camions ≈ 94,1 % en nominal (96,4 % en v1, compteurs
    non bornés) ; pelles ≈ 34 % en nominal et ≈ 49 % en forte charge, toujours < 50 %."""
    means = pd.DataFrame([
        dict(run_single_episode(make_env(scenario), fixed_fn(), seed), scenario=scenario)
        for scenario in ("nominal", "high_load")
        for seed in SCENARIOS[scenario].seeds
    ]).groupby("scenario")[["utilization_pct", "shovel_utilization_pct"]].mean()

    assert means.loc["nominal", "utilization_pct"] == pytest.approx(94.06, abs=0.01)
    assert means.loc["nominal", "shovel_utilization_pct"] == pytest.approx(33.92, abs=0.01)
    assert means.loc["high_load", "shovel_utilization_pct"] == pytest.approx(49.00, abs=0.01)
    assert (means["shovel_utilization_pct"] < 50).all()
