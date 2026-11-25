"""MTWT sensible au carburant et comparaison énergétique (M5, T11 du rapport du 17/11/2026)."""

import pandas as pd
import pytest

from memoire_master_rl_logistique.baselines.mtwt_fuel_policy import FuelAwareMTWTPolicy
from memoire_master_rl_logistique.experiments import energy
from memoire_master_rl_logistique.experiments.policies import (
    _baseline_fn,
    heuristic_policies,
    make_env,
)
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS

SEED = 20_000  # graine de sélection


def _fuel_policy_fn(weight):
    env = make_env(SCENARIOS["nominal"])
    env.reset(seed=SEED)
    policy = FuelAwareMTWTPolicy(env.graph, [s.node_id for s in env.shovels],
                                 [d.node_id for d in env.dumps], weight)
    return policy, _baseline_fn(policy, needs_graph_sync=True)


def test_lambda_nul_redonne_mtwt():
    """λ = 0 : exactement les décisions et les résultats de MTWT."""
    _, fn = _fuel_policy_fn(0.0)
    mtwt = heuristic_policies(SCENARIOS["nominal"])["MTWT"]
    a = run_single_episode(make_env(SCENARIOS["nominal"]), fn, SEED)
    b = run_single_episode(make_env(SCENARIOS["nominal"]), mtwt, SEED)
    assert a["productivity_tph"] == b["productivity_tph"]
    assert a["total_fuel_l"] == b["total_fuel_l"]


def test_lambda_grand_choisit_le_cycle_le_plus_sobre():
    """Un λ très grand choisit la pelle dont le cycle consomme le moins, quelle que soit
    l'attente ; le carburant d'une route est positif et plus élevé chargé qu'à vide."""
    policy, _ = _fuel_policy_fn(1e6)
    loc, shovels, dumps = "dump_1", policy.shovel_node_ids, policy.dump_node_ids
    fuels = [policy.route_fuel_l(loc, s, False)
             + min(policy.route_fuel_l(s, d, True) for d in dumps) for s in shovels]
    shovel, _ = policy.choose(loc, 0.0, [0.0, 50.0, 50.0], [0.0, 0.0])
    assert shovel == min(range(len(shovels)), key=lambda j: fuels[j])
    assert policy.route_fuel_l("shovel_1", "dump_1", True) > policy.route_fuel_l(
        "dump_1", "shovel_1", False) > 0


def test_lambda_negatif_refuse():
    with pytest.raises(ValueError):
        _fuel_policy_fn(-1.0)


def test_experience_courte(tmp_path):
    """Un régime, heuristiques seules, deux graines : MTWT et MTWT sensible au carburant,
    LP et Pelle la plus proche ; front de Pareto avec sa part bootstrap entre 0 et 1."""
    out = energy.run(tmp_path, None, regimes=("nominal",), seeds=(SEED, SEED + 1),
                     lambdas=(0.0, 1.0))
    rows = pd.read_csv(out)
    assert set(rows["policy"]) == {"MTWT", "MTWT-fuel 1", "LP", "Nearest"}
    summary = energy.summarise(rows, n_boot=50)
    assert summary["pareto_share"].between(0, 1).all()
    assert summary["pareto_share"].max() == 1.0  # la plus productive est toujours sur le front
