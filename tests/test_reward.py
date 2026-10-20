"""Tests de T25 : module de récompense séparé, termes exposés, valeurs inchangées.

Et de la correction du coût d'ATTENDRE (T19, v2) : terme de coût d'opportunité, étendu à
chaque cycle par l'option B (décision du 20 octobre 2026, docs/option_b_recompense.md)."""

import numpy as np
import pandas as pd
import pytest

from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments.policies import (
    HEURISTICS,
    heuristic_policies,
    make_env,
)
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.simulation.reward import (
    OPPORTUNITY_CYCLE_MIN,
    RewardWeights,
    compute_reward,
    reward_terms,
)

WEIGHTS = RewardWeights.from_tuple((1.0, 0.1, 0.05, 0.3))


def v1_formula(tonnage, capacity, distance_km, waits, wait_min, w=(1.0, 0.1, 0.05, 0.3)):
    """La formule de ``MineEnv._compute_reward`` avant T25, recopiée telle quelle."""
    r1 = tonnage / max(capacity, 1e-9)
    if len(waits) >= 2:
        m = sum(waits) / len(waits)
        r2 = -(sum((x - m) ** 2 for x in waits) / len(waits)) / 100.0
    else:
        r2 = 0.0
    return w[0] * r1 + w[1] * r2 + w[2] * (-distance_km / 5.0) + w[3] * -min(wait_min / 30.0, 1.0)


def test_memes_valeurs_que_la_v1():
    """Sur 500 décisions aléatoires, le module donne exactement la récompense de la v1."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        args = (rng.choice([0.0, 140.0]), 140.0, rng.uniform(0, 30),
                list(rng.uniform(0, 40, 3)), rng.uniform(0, 60))
        assert compute_reward(reward_terms(*args), WEIGHTS) == pytest.approx(v1_formula(*args))


def test_termes_d_un_cycle_et_d_une_attente():
    """Cycle livré de 15 km sans file, pelles équilibrées : 1 − 0,05 × 3 = 0,85.
    ATTENDRE de 0,2 min : production et distance nulles, attente −0,2/30."""
    cycle = reward_terms(140.0, 140.0, 15.0, [2.0, 2.0, 2.0], 0.0)
    wait = reward_terms(0.0, 140.0, 0.0, [0.0, 0.0, 0.0], 0.2)

    assert (cycle.production, cycle.balance, cycle.distance, cycle.wait) == (1.0, 0.0, -3.0, 0.0)
    assert compute_reward(cycle, WEIGHTS) == pytest.approx(0.85)
    assert (wait.production, wait.distance) == (0.0, 0.0)
    assert wait.wait == pytest.approx(-0.2 / 30)


def test_plafonds_et_variance():
    """L'attente est plafonnée à −1 (30 min et plus) ; l'équilibre = −variance / 100."""
    terms = reward_terms(0.0, 140.0, 0.0, [0.0, 10.0, 20.0], 90.0)

    assert terms.wait == -1.0
    assert terms.balance == pytest.approx(-np.var([0.0, 10.0, 20.0]) / 100)


def test_poids_par_defaut():
    """Trois poids donnés : w4 = 0,3, comme dans la v1."""
    assert RewardWeights.from_tuple((1.0, 0.1, 0.05)).wait == 0.3


def test_somme_ponderee_egale_la_recompense_de_step():
    """Test du plan : à chaque décision (cycles et ATTENDRE), la somme des contributions
    pondérées exposées dans ``info`` égale la récompense renvoyée par ``step``."""
    env = MineEnv()
    rng = np.random.default_rng(0)
    env.reset(seed=42)
    total = 0.0
    while True:
        _, reward, terminated, truncated, info = env.step(int(rng.integers(env.action_space.n)))
        total += reward
        assert sum(info["reward_terms"].values()) == pytest.approx(reward)
        if terminated or truncated:
            break
    assert sum(info["episode_reward_terms"].values()) == pytest.approx(total)


def test_decomposition_d_un_poste_affectation_fixe():
    """Test du plan, Affectation fixe en nominal sur 30 graines : terme de distance ≈ −19 ;
    le terme de production égale le nombre de cycles. Le coût d'opportunité (option B) vaut
    −12 × 480 / 23,2 sur chaque poste : tout le temps des camions, compté une fois.

    Valeurs depuis le moteur à événements (T15, T16 bis) : distance −19,4 (−19,1 avant :
    un cycle va maintenant du lieu de décision à la verse, sans retour), attente −1,38
    (−2,71 avant : les verses ne servent plus les camions dans le désordre des arrivées)."""
    scenario = SCENARIOS["nominal"]
    policy = heuristic_policies(scenario)["Fixed"]
    runs = pd.DataFrame([run_single_episode(make_env(scenario), policy, s) for s in scenario.seeds])
    terms = runs[["reward_production", "reward_balance", "reward_distance", "reward_wait",
                  "reward_opportunity"]]

    assert np.allclose(terms.sum(axis=1), runs["total_reward"])
    assert runs["reward_opportunity"].to_numpy() == pytest.approx(
        -scenario.truck_count * scenario.episode_minutes / OPPORTUNITY_CYCLE_MIN)
    assert (runs["reward_production"] == runs["total_cycles"]).all()
    assert runs["reward_distance"].mean() == pytest.approx(-19.42, abs=0.05)
    assert runs["reward_wait"].mean() == pytest.approx(-1.38, abs=0.05)


@pytest.mark.parametrize(("scenario", "minutes"), [("nominal", 1.89), ("high_load", 1.29)])
def test_intervalle_entre_decisions(scenario, minutes):
    """« Savoir expliquer » : γ = 0,99 par décision, horizon 1/(1 − γ) = 100 décisions.
    Avec l'Affectation fixe : ≈ 1,89 min par décision avec 12 camions (≈ 189 min d'horizon),
    ≈ 1,29 min avec 18 camions (≈ 129 min). Avant T15 et T16 bis : 1,97 et 1,36 min
    (cycles plus longs, à cause des files dans le désordre des arrivées)."""
    s = SCENARIOS[scenario]
    policy = heuristic_policies(s)["Fixed"]
    decisions = np.mean([run_single_episode(make_env(s), policy, seed)["decisions"]
                         for seed in s.seeds])

    assert s.episode_minutes / decisions == pytest.approx(minutes, abs=0.01)


# --- Coût d'opportunité (v2 : ATTENDRE ; option B : chaque cycle) ---------------------

def test_cout_d_opportunite_d_une_attente_et_d_un_cycle():
    """Attendre 0,2 min coûte 0,2/23,2 de cycle en plus (pondéré par w1). Un cycle de
    30 min coûte 30/23,2 (option B) : livré, il rapporte 1 − 30/23,2 avant les autres
    termes. Une décision qui dure un cycle de référence coûte exactement un cycle."""
    cycle = reward_terms(140.0, 140.0, 15.0, [2.0, 2.0, 2.0], 0.0, truck_min=30.0)
    wait = reward_terms(0.0, 140.0, 0.0, [0.0, 0.0, 0.0], 0.2, truck_min=0.2)
    full = reward_terms(0.0, 140.0, 0.0, [0.0, 0.0, 0.0], 0.0, truck_min=OPPORTUNITY_CYCLE_MIN)

    assert cycle.opportunity == pytest.approx(-30.0 / OPPORTUNITY_CYCLE_MIN)
    assert compute_reward(cycle, WEIGHTS) == pytest.approx(0.85 - 30.0 / OPPORTUNITY_CYCLE_MIN)
    assert reward_terms(140.0, 140.0, 15.0, [2.0, 2.0, 2.0], 0.0).opportunity == 0.0
    assert wait.opportunity == pytest.approx(-0.2 / OPPORTUNITY_CYCLE_MIN)
    assert full.opportunity == -1.0
    assert compute_reward(wait, WEIGHTS) == pytest.approx(
        v1_formula(0.0, 140.0, 0.0, [0.0, 0.0, 0.0], 0.2) - 0.2 / OPPORTUNITY_CYCLE_MIN)
    # Même unité que la production : son poids est w1, sans nouveau poids à régler.
    weights = RewardWeights.from_tuple((2.0, 0.1, 0.05, 0.3))
    assert wait.weighted(weights)["opportunity"] == pytest.approx(
        -2.0 * 0.2 / OPPORTUNITY_CYCLE_MIN)


def test_cycle_de_reference():
    """La durée de référence est celle d'un cycle de l'Affectation fixe en nominal."""
    scenario = SCENARIOS["nominal"]
    env = make_env(scenario, log_cycles=True)
    policy = heuristic_policies(scenario)["Fixed"]
    durations = []
    for seed in scenario.seeds[:10]:
        run_single_episode(env, policy, seed)
        durations += [c.end_min - c.start_min for c in env.cycle_log if c.kind == "cycle"]
    assert np.mean(durations) == pytest.approx(OPPORTUNITY_CYCLE_MIN, abs=0.3)


def test_attendre_dans_l_environnement():
    """Une ATTENDRE dans MineEnv : récompense v1 moins la production perdue pendant
    l'attente, comptée seulement dans le poste."""
    env = MineEnv(seed=42)
    env.reset(seed=42)
    _, reward, _, _, info = env.step(env.action_space.n - 1)  # ATTENDRE
    idle = env.trucks[0].voluntary_wait_min
    assert info["wait_actions"] == 1 and idle >= 0.2
    assert info["reward_terms"]["opportunity"] == pytest.approx(-idle / OPPORTUNITY_CYCLE_MIN)

    # En fin de poste, seule la part de l'attente avant 480 min est perdue.
    env.reset(seed=42)
    env.current_time_min = env.episode_minutes - 0.1
    for truck in env.trucks:
        truck.available_at_min = env.current_time_min
    _, _, _, _, info = env.step(env.action_space.n - 1)
    assert info["reward_terms"]["opportunity"] == pytest.approx(-0.1 / OPPORTUNITY_CYCLE_MIN)


def test_cout_d_un_cycle_dans_l_environnement():
    """Chaque cycle paie la durée qu'il occupe dans le poste, décision comprise : pour une
    politique qui n'attend jamais, le coût d'opportunité d'un poste est la somme des
    durées des cycles du journal, bornées à 480 min."""
    scenario = SCENARIOS["nominal"]
    env = make_env(scenario, log_cycles=True)
    run_single_episode(env, heuristic_policies(scenario)["MTWT"], 42)
    end = scenario.episode_minutes
    in_shift = sum(max(0.0, min(c.end_min, end) - min(c.start_min, end)) for c in env.cycle_log)
    assert all(c.kind == "cycle" for c in env.cycle_log)
    assert env.episode_reward_terms["opportunity"] == pytest.approx(-in_shift / OPPORTUNITY_CYCLE_MIN)
    assert in_shift == pytest.approx(scenario.truck_count * end)


@pytest.mark.parametrize("scenario", ["nominal", "high_load", "high_breakdown"])
def test_heuristiques_une_constante_par_poste(scenario):
    """Option B : les heuristiques n'attendent jamais, et le temps total des camions sur un
    poste est fixe. Leur récompense est donc celle d'avant l'option B, moins une constante
    (nombre de camions × durée du poste / 23,2), la même pour toutes : leur classement par
    la récompense d'un poste ne change pas."""
    s = SCENARIOS[scenario]
    constant = s.truck_count * s.episode_minutes / OPPORTUNITY_CYCLE_MIN
    policies = heuristic_policies(s)
    for name in HEURISTICS:
        kpis = run_single_episode(make_env(s), policies[name], s.seeds[0])
        assert kpis["wait_actions"] == 0, name
        assert kpis["reward_opportunity"] == pytest.approx(-constant), name
