"""Tests de T13 : un flux aléatoire par processus (nombres aléatoires communs)."""

import pytest

from memoire_master_rl_logistique.experiments.policies import heuristic_policies, make_env
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS

PANNES = SCENARIOS["high_breakdown"]  # 10 % de pannes par cycle : beaucoup d'occasions de comparer


def breakdowns_by_cycle(policy: str, seed: int, scenario=PANNES) -> dict[str, list[tuple[int, float]]]:
    """Pour chaque camion : liste des (numéro de cycle, durée de réparation) de ses pannes,
    et nombre total de cycles commencés (clé ``"<camion>:n"``)."""
    env = make_env(scenario, log_cycles=True)
    run_single_episode(env, heuristic_policies(scenario)[policy], seed)
    result: dict[str, list] = {}
    for truck in {r.truck_id for r in env.cycle_log}:
        cycles = [r for r in env.cycle_log if r.truck_id == truck and r.kind == "cycle"]
        result[truck] = [(k, r.repair_min) for k, r in enumerate(cycles) if r.repair_min > 0]
        result[f"{truck}:n"] = len(cycles)
    return result


def common_part(breakdowns: list[tuple[int, float]], n_cycles: int) -> list[tuple[int, float]]:
    """Pannes survenues dans les ``n_cycles`` premiers cycles."""
    return [(k, d) for k, d in breakdowns if k < n_cycles]


def test_camion_3_meme_panne_au_meme_cycle():
    """Test du plan : avec la même graine, le camion 3 tombe en panne au même numéro de
    cycle sous FIFO et sous l'Affectation fixe, avec la même durée de réparation."""
    fifo, fixed = breakdowns_by_cycle("FIFO", 42), breakdowns_by_cycle("Fixed", 42)
    n = min(fifo["T3:n"], fixed["T3:n"])  # cycles réalisés par les deux politiques

    assert common_part(fifo["T3"], n)  # au moins une panne à comparer
    assert common_part(fifo["T3"], n) == common_part(fixed["T3"], n)


@pytest.mark.parametrize("seed", PANNES.seeds)
def test_pannes_communes_pour_tous_les_camions(seed):
    """Sur les 30 graines et pour chaque camion, les pannes des cycles communs sont
    identiques (numéro de cycle et durée). Avant T13, la première panne n'était identique
    que pour 5 graines sur 30."""
    fifo, fixed = breakdowns_by_cycle("FIFO", seed), breakdowns_by_cycle("Fixed", seed)

    for truck in (t for t in fifo if not t.endswith(":n")):
        n = min(fifo[f"{truck}:n"], fixed[f"{truck}:n"])
        assert common_part(fifo[truck], n) == common_part(fixed[truck], n), truck


def test_meme_graine_meme_episode():
    """Reproductibilité : rejouer la même graine redonne exactement le même épisode."""
    scenario = SCENARIOS["nominal"]
    policy = heuristic_policies(scenario)["Fixed"]

    first = run_single_episode(make_env(scenario), policy, seed=42)
    second = run_single_episode(make_env(scenario), policy, seed=42)
    other = run_single_episode(make_env(scenario), policy, seed=43)

    for key in ("productivity_tph", "total_reward", "total_wait_min", "total_fuel_l"):
        assert first[key] == second[key]
    assert first["total_wait_min"] != other["total_wait_min"]


def test_flux_independants():
    """Chaque processus a son propre générateur, et chaque camion son flux de pannes."""
    env = make_env(SCENARIOS["nominal"])
    env.reset(seed=42)
    streams = [env.rng_travel, env.rng_service, env.rng_road, *env.rng_breakdowns]

    assert len(env.rng_breakdowns) == env.truck_count
    assert len({id(s) for s in streams}) == len(streams)
    assert len({s.random() for s in streams}) == len(streams)  # suites différentes
