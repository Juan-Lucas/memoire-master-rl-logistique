"""Tests de T22 : FIFO et l'Affectation fixe font exactement ce que disent leurs docstrings."""

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from memoire_master_rl_logistique.baselines.base import encode_action
from memoire_master_rl_logistique.baselines.fifo_policy import FIFOPolicy
from memoire_master_rl_logistique.baselines.fixed_policy import FixedAssignmentPolicy
from memoire_master_rl_logistique.experiments.policies import heuristic_policies, make_env
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS

FIFO = FIFOPolicy(num_shovels=3, num_dumps=2)
FIXED = FixedAssignmentPolicy(num_shovels=3, num_dumps=2)


def info(shovels, dumps, truck_idx=0):
    return {"shovel_available_at": shovels, "dump_available_at": dumps,
            "current_truck_idx": truck_idx}


# --- FIFO : pelle libre le plus tôt, verse libre le plus tôt -------------------------

def test_fifo_pelle_et_verse_libres_le_plus_tot():
    """Pelle 2 libre à 2 min, verse 2 libre à 1 min : action (pelle 2, verse 2)."""
    assert FIFO.predict(None, info([10.0, 2.0, 5.0], [3.0, 1.0])) == encode_action(1, 1, 2)


def test_fifo_ignore_les_trajets():
    """La position du camion ne change rien : seule compte l'heure de disponibilité,
    même si la pelle choisie est loin et la pelle où se trouve le camion presque libre."""
    situation = info([10.0, 9.9, 30.0], [0.0, 0.0])

    actions = {FIFO.predict(None, situation, truck_location=loc)
               for loc in ("yard", "shovel_1", "shovel_2", "shovel_3")}

    print(actions)

    assert actions == {encode_action(1, 0, 2)}


def test_fifo_egalite_et_absence_d_information():
    """Égalité : le plus petit indice ; sans ``info`` : action 0."""
    assert FIFO.predict(None, info([4.0, 4.0, 4.0], [2.0, 2.0])) == 0
    assert FIFO.predict(None, None) == 0


# --- Affectation fixe : camion i → pelle i mod 3, verse i mod 2 -------------------------

@pytest.mark.parametrize("truck_idx", range(12))
def test_fixe_pelle_i_mod_3_verse_i_mod_2(truck_idx):
    """La règle ne lit que l'indice du camion."""
    expected = encode_action(truck_idx % 3, truck_idx % 2, 2)
    rng = np.random.default_rng(truck_idx)

    for _ in range(5):  # n'importe quelle observation, n'importe quelles files
        situation = info(list(rng.random(3) * 60), list(rng.random(2) * 60), truck_idx)
        assert FIXED.predict(rng.random(18), situation) == expected


@pytest.mark.parametrize("trucks", [12, 13, 18])
def test_fixe_sans_reaffectation_pendant_tout_le_poste(trucks):
    """Sur un épisode complet, chaque camion fait tous ses cycles à la même pelle et à la
    même verse ; la flotte est répartie i mod 3 (4 par pelle à 12 camions, 6 à 18, 5/4/4 à 13)."""
    scenario = replace(SCENARIOS["nominal"], truck_count=trucks)
    env = make_env(scenario, log_cycles=True)
    run_single_episode(env, heuristic_policies(scenario)["Fixed"], seed=42)

    pairs = {(r.truck_id, r.shovel_id, r.dump_id) for r in env.cycle_log}
    assert len(pairs) == trucks  # une seule paire (pelle, verse) par camion
    for truck_id, shovel_id, dump_id in pairs:
        i = int(truck_id[1:]) - 1
        assert (shovel_id, dump_id) == (f"S{i % 3 + 1}", f"D{i % 2 + 1}")
    per_shovel = Counter(shovel for _, shovel, _ in pairs)
    assert sorted(per_shovel.values(), reverse=True) == [
        len(range(k, trucks, 3)) for k in range(3)]


def test_fixe_toujours_le_meme_couple():
    """Un camion de l'Affectation fixe fait tous ses cycles sur le même couple pelle–verse.
    Depuis T16 bis, chaque cycle commence par le trajet à vide verse → pelle (le premier
    depuis le parc) : plus aucun cycle sans trajet vers la pelle."""
    env = make_env(SCENARIOS["nominal"], log_cycles=True)
    run_single_episode(env, heuristic_policies(SCENARIOS["nominal"])["Fixed"], seed=42)

    for truck in {r.truck_id for r in env.cycle_log}:
        cycles = [r for r in env.cycle_log if r.truck_id == truck]
        assert len({(r.shovel_id, r.dump_id) for r in cycles}) == 1
        assert all(r.travel_to_shovel_min > 0 for r in cycles)
