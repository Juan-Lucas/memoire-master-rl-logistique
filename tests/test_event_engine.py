"""Tests de T15 (moteur à événements) et de T16 bis (décision à la verse), avec la
récompense versée à la fin du cycle (option 1a)."""

import numpy as np
import pytest

from memoire_master_rl_logistique.env.mine_env import STATUS_IDLE, MineEnv
from memoire_master_rl_logistique.experiments.policies import heuristic_policies, make_env
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS, SCENARIOS

DECISION_PLACES = {"yard", "dump_1", "dump_2"}


def random_episode(scenario: str, seed: int, wait_share: float = 0.1):
    """Épisode d'une politique aléatoire (avec des ATTENDRE) ; renvoie l'environnement,
    les lieux et heures de chaque décision et les récompenses de chaque ``step``."""
    env = make_env(SCENARIOS[scenario], log_cycles=True)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    places, times, rewards = [], [], []
    while True:
        places.append(env.truck_locations[env.current_truck_idx])
        times.append(env.current_time_min)
        wait = rng.random() < wait_share
        action = env.action_space.n - 1 if wait else int(rng.integers(env.action_space.n - 1))
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            return env, places, times, rewards, info


def service_intervals(cycles, resource: str):
    """Par ressource : (arrivée, début de service, fin de service) de chaque cycle."""
    by = {}
    for c in cycles:
        if resource == "shovel":
            arrival = c.start_min + c.travel_to_shovel_min
            start = arrival + c.shovel_wait_min
            key, end = c.shovel_id, start + c.load_min
        else:
            arrival = (c.start_min + c.travel_to_shovel_min + c.shovel_wait_min + c.load_min
                       + c.travel_to_dump_min)
            start = arrival + c.dump_wait_min
            key, end = c.dump_id, start + c.unload_min
        by.setdefault(key, []).append((arrival, start, end))
    return by


@pytest.mark.parametrize("scenario", MAIN_SCENARIOS)
@pytest.mark.parametrize("resource", ["shovel", "dump"])
def test_service_dans_l_ordre_des_arrivees(scenario, resource):
    """T15 : chaque pelle et chaque verse sert les camions dans leur ordre d'arrivée, un
    seul à la fois, et n'attend jamais quand un camion est là (pas de temps mort)."""
    for seed in (42, 43):
        env, *_ = random_episode(scenario, seed)
        cycles = [c for c in env.cycle_log if c.kind == "cycle"]
        for rows in service_intervals(cycles, resource).values():
            served = sorted(rows, key=lambda r: r[1])
            for (a0, _s0, e0), (a1, s1, _e1) in zip(served, served[1:], strict=False):
                assert a1 >= a0 - 1e-9                       # premier arrivé, premier servi
                assert s1 >= e0 - 1e-9                       # un seul camion à la fois
                assert s1 == pytest.approx(max(a1, e0))      # servi dès que possible


def test_decisions_au_parc_ou_a_une_verse():
    """T16 bis : une décision se prend au parc (premier cycle) ou à une verse, jamais à une
    pelle ; l'horloge des décisions ne recule jamais."""
    env, places, times, _, _ = random_episode("nominal", 42)
    assert set(places) <= DECISION_PLACES
    assert places.count("yard") >= env.truck_count  # chaque camion part du parc
    assert all(t1 >= t0 for t0, t1 in zip(times, times[1:], strict=False))


def test_un_cycle_part_de_la_verse_precedente():
    """T16 bis : la distance d'un cycle va de la verse précédente à la pelle choisie, puis à
    la verse choisie ; plus de retour à l'ancienne pelle. Un seul camion : son cycle se
    termine pendant son propre ``step``."""
    env = MineEnv(truck_count=1, breakdown_probability=0.0)
    env.reset(seed=42)
    g = env.graph
    env.step(1)  # pelle 1, verse 2
    _, _, _, _, info = env.step(4)  # pelle 3, verse 1, depuis la verse 2

    expected_km = g.shortest_distance("dump_2", "shovel_3") + g.shortest_distance("shovel_3",
                                                                                   "dump_1")
    assert info["reward_terms"]["distance"] == pytest.approx(-0.05 * expected_km / 5)


@pytest.mark.parametrize("scenario", MAIN_SCENARIOS)
def test_toute_la_recompense_est_versee(scenario):
    """Option 1a : la somme des récompenses des ``step`` égale celle de tous les cycles.
    À la fin du poste, les cycles en cours sont menés à terme : chaque cycle décidé est
    terminé et payé une fois."""
    env, _, _, rewards, info = random_episode(scenario, 7)

    assert sum(rewards) == pytest.approx(sum(info["episode_reward_terms"].values()))
    decided_cycles = env.decision_count - env.wait_action_count
    assert len([c for c in env.cycle_log if c.kind == "cycle"]) == decided_cycles
    assert all(s == STATUS_IDLE for s in env.truck_statuses)
    assert env.current_time_min >= env.episode_minutes


def test_recompense_d_un_cycle_versee_a_sa_fin():
    """La récompense d'un cycle n'est versée qu'à sa fin : 0 tant qu'aucun cycle ne se
    termine (au début du poste, tous les camions décident avant le premier déversement)."""
    env = MineEnv(breakdown_probability=0.0)
    env.reset(seed=42)
    first_rewards = [env.step(0)[1] for _ in range(env.truck_count)]
    assert first_rewards[:-1] == [0.0] * (env.truck_count - 1)


def test_prevision_compte_les_camions_en_route():
    """La libération prévue d'une pelle ajoute, pour chaque camion en route, un chargement
    moyen à partir de son arrivée ; sans camion en route, c'est ``available_at_min``."""
    env = MineEnv(breakdown_probability=0.0)
    env.reset(seed=42)
    shovels, dumps = env.forecast_free_at()
    assert shovels == [s.available_at_min for s in env.shovels]
    assert dumps == [d.available_at_min for d in env.dumps]

    env.step(0)  # camion 1 → pelle 1
    env.step(0)  # camion 2 → pelle 1, arrive en même temps ou après
    arrivals = sorted(env.trucks[i].available_at_min for i in (0, 1))
    shovels, _ = env.forecast_free_at()
    expected = max(arrivals[0] + 2.0, arrivals[1]) + 2.0
    assert shovels[0] == pytest.approx(expected)
    assert shovels[1] == shovels[2] == 0.0


def test_plus_de_detour_pour_changer_de_pelle():
    """T16 bis : changer de pelle ne coûte plus de détour. FIFO, qui change de pelle dans
    environ 70 % de ses cycles, a des cycles aussi courts que l'Affectation fixe (≈ 23,2 min
    en nominal) ; avant, 29,8 min contre 24,1 min."""
    scenario = SCENARIOS["nominal"]
    policies = heuristic_policies(scenario)
    durations = {}
    for name in ("FIFO", "Fixed"):
        env = make_env(scenario, log_cycles=True)
        cycles = []
        for seed in scenario.seeds[:5]:
            run_single_episode(env, policies[name], seed)
            cycles += [c.end_min - c.start_min for c in env.cycle_log if c.kind == "cycle"]
        durations[name] = np.mean(cycles)
    assert durations["FIFO"] == pytest.approx(durations["Fixed"], rel=0.03)
