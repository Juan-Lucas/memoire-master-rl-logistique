import csv
from pathlib import Path
import pytest
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.env import  MineEnv
from memoire_master_rl_logistique.baselines.fixed_policy import FixedAssignmentPolicy
from memoire_master_rl_logistique.baselines.fifo_policy import FIFOPolicy
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode

REFERENCES_CSV = Path(__file__).parent.parent / "data" / "results_v1_article" / "benchmark_results.csv"

POLICIES = {"Fixed": FixedAssignmentPolicy, "FIFO": FIFOPolicy}

def make_policy_fn(policy):
    """Adapte une politique heuristique à la signature de run_single_episode."""

    def policy_fn(obs, info, env_):
        """Retourne l'action recommandée par la politique."""

        position = env_.truck_locations[env_.current_truck_idx]
        return policy.predict(obs, info, truck_location=position)
    return policy_fn

def load_reference(scenario_name, policy_name, seed):
    """Charge les données de référence pour une politique et une graine données."""

    with open(REFERENCES_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["scenario"] == scenario_name and row["policy"] == policy_name and int(row["seed"]) == seed:
                return float(row["productivity_tph"]), float(row["total_reward"])
    raise ValueError(f"Aucune référence pour {scenario_name}, {policy_name}, {seed}")

@pytest.mark.skipif(not REFERENCES_CSV.exists(), reason="Résultats v1 absents")
@pytest.mark.parametrize("policy_name", ["Fixed", "FIFO"])
@pytest.mark.parametrize("seed", [42, 43, 44, 45, 46])
def test_reproduit_v1(policy_name, seed):
    """Teste la reproduction des résultats attendus pour les politiques Fixed et FIFO."""

    scenario = SCENARIOS["nominal"]

    env = MineEnv(
        truck_count=scenario.truck_count,
        shovel_count=scenario.shovel_count,
        dump_count=scenario.dump_count,
        episode_minutes=scenario.episode_minutes,
        breakdown_probability=scenario.breakdown_probability,
        reward_weights=scenario.reward_weights,
    )

    policy = POLICIES[policy_name](
        num_shovels=scenario.shovel_count,
        num_dumps=scenario.dump_count
    )

    kpis = run_single_episode(env, make_policy_fn(policy), seed)

    ref_prod, ref_reward = load_reference("nominal", policy_name, seed)

    assert kpis["productivity_tph"] == pytest.approx(ref_prod, abs=1e-6)
    assert kpis["total_reward"] == pytest.approx(ref_reward, abs=1e-6)



