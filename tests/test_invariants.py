"""Invariants du simulateur (vérification, M2 du rapport du 17/11/2026).

Complètent ``test_event_engine.py`` (service dans l'ordre d'arrivée, un camion à la fois
par pelle et par verse, récompense versée une seule fois) et ``test_shift_bounds.py``
(compteurs bornés au poste) : un seul état par camion, ordre des événements simultanés,
bilan du tonnage, cycles terminés après le poste, carburant au moins égal au ralenti.
Politique aléatoire avec des ATTENDRE, sur les trois régimes et des graines de sélection.
"""

import numpy as np
import pytest

from memoire_master_rl_logistique.env.mine_env import (
    STATUS_IDLE,
    STATUS_TO_DUMP,
    STATUS_TO_SHOVEL,
    MineEnv,
)
from memoire_master_rl_logistique.experiments.policies import make_env
from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS, SCENARIOS
from memoire_master_rl_logistique.simulation.fuel_model import DEFAULT_FUEL

SEEDS = (20_000, 20_001)  # graines de sélection : jamais celles de l'évaluation
STATUSES = {STATUS_IDLE, STATUS_TO_SHOVEL, STATUS_TO_DUMP}


def play(scenario: str, seed: int, wait_share: float = 0.1, check_each_step=None) -> MineEnv:
    """Un poste joué par une politique aléatoire ; ``check_each_step(env)`` après chaque pas."""
    env = make_env(SCENARIOS[scenario], log_cycles=True)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    while True:
        wait = rng.random() < wait_share
        action = env.action_space.n - 1 if wait else int(rng.integers(env.action_space.n - 1))
        _, _, terminated, truncated, _ = env.step(action)
        if check_each_step is not None:
            check_each_step(env, done=terminated or truncated)
        if terminated or truncated:
            return env


@pytest.mark.parametrize("scenario", MAIN_SCENARIOS)
def test_un_seul_etat_par_camion(scenario):
    """À chaque décision, la flotte est complète, chaque camion a un seul état, un camion a
    un cycle en cours si et seulement s'il roule, et le camion qui décide est libre."""
    def check(env, done):
        assert len(env.trucks) == len(env.truck_statuses) == env.truck_count
        assert set(env.truck_statuses) <= STATUSES
        for status, cycle in zip(env.truck_statuses, env._cycles, strict=True):
            assert (cycle is None) == (status == STATUS_IDLE)
        if not done:
            assert env.truck_statuses[env.current_truck_idx] == STATUS_IDLE

    for seed in SEEDS:
        play(scenario, seed, check_each_step=check)


def test_evenements_simultanes():
    """Deux événements au même instant : l'arrivée passe avant la décision ; entre deux
    décisions simultanées, le camion d'indice le plus petit décide le premier."""
    env = MineEnv(truck_count=3, breakdown_probability=0.0)
    env.reset(seed=SEEDS[0])
    for truck in env.trucks:
        truck.available_at_min = 10.0
    env.truck_statuses = [STATUS_IDLE, STATUS_TO_SHOVEL, STATUS_IDLE]
    assert env._next_event() == 1                      # l'arrivée d'abord
    env.truck_statuses = [STATUS_IDLE, STATUS_IDLE, STATUS_IDLE]
    assert env._next_event() == 0                      # puis le plus petit indice
    env.trucks[0].available_at_min = 10.5
    assert env._next_event() == 1                      # le plus tôt l'emporte toujours


@pytest.mark.parametrize("scenario", MAIN_SCENARIOS)
def test_bilan_du_tonnage(scenario):
    """Le tonnage et le nombre de cycles des camions égalent ceux du journal des cycles."""
    for seed in SEEDS:
        env = play(scenario, seed)
        cycles = [c for c in env.cycle_log if c.kind == "cycle"]
        assert sum(t.total_tonnage_t for t in env.trucks) == pytest.approx(
            sum(c.delivered_t for c in cycles))
        assert sum(t.cycles_completed for t in env.trucks) == sum(
            c.delivered_t > 0 for c in cycles)


@pytest.mark.parametrize("scenario", MAIN_SCENARIOS)
def test_cycle_termine_apres_le_poste(scenario):
    """Un cycle n'apporte de tonnage que si le déversement finit avant 480 min ; un cycle
    déversé après la fin du poste est simulé jusqu'au bout, sans tonnage."""
    late = 0
    for seed in SEEDS:
        env = play(scenario, seed)
        for c in (c for c in env.cycle_log if c.kind == "cycle"):
            unload_end = c.end_min - c.repair_min
            assert (c.delivered_t > 0) == (unload_end < env.episode_minutes)
            late += unload_end >= env.episode_minutes
    assert late > 0  # le cas est bien rencontré : la fin du poste coupe des cycles


@pytest.mark.parametrize("scenario", MAIN_SCENARIOS)
def test_carburant_au_moins_le_ralenti(scenario):
    """Chaque camion consomme au moins le ralenti de tous ses temps à l'arrêt (files,
    chargement, déversement, réparation, ATTENDRE), et davantage puisqu'il roule."""
    idle_rate_l_per_min = DEFAULT_FUEL.idle_l_per_h / 60.0
    for seed in SEEDS:
        env = play(scenario, seed)
        for truck in env.trucks:
            records = [c for c in env.cycle_log if c.truck_id == truck.truck_id]
            idle_min = sum(c.shovel_wait_min + c.load_min + c.dump_wait_min + c.unload_min
                           + c.repair_min + c.voluntary_wait_min for c in records)
            assert truck.total_fuel_l > idle_rate_l_per_min * idle_min
