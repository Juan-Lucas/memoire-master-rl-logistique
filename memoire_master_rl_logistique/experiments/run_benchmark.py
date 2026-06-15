"""Benchmark comparatif complet.

Exécute toutes les méthodes (heuristiques + RL classiques + Deep RL)
sur tous les scénarios avec plusieurs seeds pour produire les données
du Chapitre 6.

Protocole expérimental (Section 5.4 du mémoire) :
- Mêmes scénarios pour toutes les méthodes
- 10 réplications par scénario (seeds 42..51)
- KPIs calculés de la même manière
- Résultats sauvegardés en CSV reproductible

Méthodes comparées (Section 4.7.1 du mémoire) :
- Heuristiques : FIFO, Fixed, Nearest, ShortestPath
- RL classiques : Q-Learning, SARSA
- Deep RL : DQN, PPO
"""

from __future__ import annotations

import csv
from pathlib import Path

from stable_baselines3 import DQN, PPO

from memoire_master_rl_logistique.baselines.base import BaselinePolicy
from memoire_master_rl_logistique.baselines.fifo_policy import FIFOPolicy
from memoire_master_rl_logistique.baselines.fixed_policy import FixedAssignmentPolicy
from memoire_master_rl_logistique.baselines.nearest_policy import NearestShovelPolicy
from memoire_master_rl_logistique.baselines.shortest_path_policy import ShortestPathPolicy
from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS, Scenario
from memoire_master_rl_logistique.rl.train_dqn import train_dqn
from memoire_master_rl_logistique.rl.train_ppo import train_ppo
from memoire_master_rl_logistique.rl.train_q_learning import QLearningPolicy, train_q_learning
from memoire_master_rl_logistique.rl.train_sarsa import SarsaPolicy, train_sarsa
from memoire_master_rl_logistique.simulation.kpi import compute_kpis


def run_single_episode(
    env: MineEnv,
    policy_fn,
    seed: int,
    policy_name: str = "",
    debug_actions: bool = False,
) -> dict[str, float]:
    """Lance un épisode et retourne les KPIs."""
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    actions_taken = []

    while True:
        action = policy_fn(obs, info, env)
        if debug_actions and len(actions_taken) < 10:  # Log first 10 actions
            actions_taken.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    
    if debug_actions and actions_taken:
        print(f"    [{policy_name}] First 10 actions: {actions_taken}")

    total_tonnage = sum(t.total_tonnage_t for t in env.trucks)
    total_wait = sum(t.total_wait_min for t in env.trucks)
    total_active = sum(t.total_active_min for t in env.trucks)
    total_fuel = sum(t.total_fuel_l for t in env.trucks)

    total_cycles = sum(t.cycles_completed for t in env.trucks)

    kpis = compute_kpis(
        episode_minutes=env.episode_minutes,
        truck_count=env.truck_count,
        total_tonnage_t=total_tonnage,
        total_wait_min=total_wait,
        total_active_min=total_active,
        total_fuel_l=total_fuel,
        total_cycles=total_cycles,
    )
    kpis["total_reward"] = total_reward
    return kpis


def run_benchmark(
    scenario: Scenario,
    output_dir: Path,
    train_ppo_flag: bool = True,
) -> list[dict]:
    """Exécute le benchmark pour un scénario donné."""
    results = []

    # Créer un environnement temporaire pour initialiser les politiques
    temp_env = MineEnv(
        truck_count=scenario.truck_count,
        shovel_count=scenario.shovel_count,
        dump_count=scenario.dump_count,
        episode_minutes=scenario.episode_minutes,
        breakdown_probability=scenario.breakdown_probability,
        reward_weights=scenario.reward_weights,
    )
    temp_env.reset(seed=scenario.seeds[0])

    # Initialiser les politiques heuristiques (créées une seule fois)
    fixed_policy_obj = FixedAssignmentPolicy(
        num_shovels=scenario.shovel_count,
        num_dumps=scenario.dump_count,
    )

    fifo_policy_obj = FIFOPolicy(
        num_shovels=scenario.shovel_count,
        num_dumps=scenario.dump_count,
    )

    nearest_policy_obj = NearestShovelPolicy(
        graph=temp_env.graph,
        shovel_node_ids=[s.node_id for s in temp_env.shovels],
        dump_node_ids=[d.node_id for d in temp_env.dumps],
    )

    shortest_path_policy_obj = ShortestPathPolicy(
        graph=temp_env.graph,
        shovel_node_ids=[s.node_id for s in temp_env.shovels],
        dump_node_ids=[d.node_id for d in temp_env.dumps],
    )

    def _make_policy_fn(policy_obj: BaselinePolicy, needs_graph_sync: bool = False):
        """Adapte une BaselinePolicy à la signature policy_fn(obs, info, env)."""
        def policy_fn(obs, info, env):
            if needs_graph_sync:
                policy_obj.graph = env.graph
            return policy_obj.predict(
                obs,
                info,
                truck_location=env.truck_locations[env.current_truck_idx],
            )
        return policy_fn

    policies = {
        "FIFO": _make_policy_fn(fifo_policy_obj),
        "Fixed": _make_policy_fn(fixed_policy_obj),
        "Nearest": _make_policy_fn(nearest_policy_obj, needs_graph_sync=True),
        "ShortestPath": _make_policy_fn(shortest_path_policy_obj, needs_graph_sync=True),
    }

    # --- Q-Learning (Section 4.4.1) ---
    ql_dir = output_dir / f"q_learning_{scenario.name}"
    ql_path = ql_dir / "q_table.pkl"

    if ql_path.exists():
        print(f"  Chargement Q-Learning existant: {ql_path}")
        ql_agent = QLearningPolicy.load(ql_path)
    else:
        print(f"  Entraînement Q-Learning pour '{scenario.name}'...")
        train_q_learning(
            n_episodes=30_000,
            alpha=0.2,
            n_bins=8,
            seed=scenario.seeds[0],
            truck_count=scenario.truck_count,
            shovel_count=scenario.shovel_count,
            dump_count=scenario.dump_count,
            episode_minutes=scenario.episode_minutes,
            breakdown_probability=scenario.breakdown_probability,
            reward_weights=scenario.reward_weights,
            output_dir=str(ql_dir),
        )
        ql_agent = QLearningPolicy.load(ql_path)

    def q_learning_policy(obs, info, env, _agent=ql_agent):
        return _agent.predict(obs)

    policies["Q-Learning"] = q_learning_policy

    # --- SARSA (Section 4.4.3) ---
    sarsa_dir = output_dir / f"sarsa_{scenario.name}"
    sarsa_path = sarsa_dir / "sarsa_table.pkl"

    if sarsa_path.exists():
        print(f"  Chargement SARSA existant: {sarsa_path}")
        sarsa_agent = SarsaPolicy.load(sarsa_path)
    else:
        print(f"  Entraînement SARSA pour '{scenario.name}'...")
        train_sarsa(
            n_episodes=30_000,
            alpha=0.2,
            n_bins=8,
            seed=scenario.seeds[0],
            truck_count=scenario.truck_count,
            shovel_count=scenario.shovel_count,
            dump_count=scenario.dump_count,
            episode_minutes=scenario.episode_minutes,
            breakdown_probability=scenario.breakdown_probability,
            reward_weights=scenario.reward_weights,
            output_dir=str(sarsa_dir),
        )
        sarsa_agent = SarsaPolicy.load(sarsa_path)

    def sarsa_policy_fn(obs, info, env, _agent=sarsa_agent):
        return _agent.predict(obs)

    policies["SARSA"] = sarsa_policy_fn

    # --- DQN (Section 4.5.2) ---
    if train_ppo_flag:
        dqn_path_scenario = output_dir / f"dqn_{scenario.name}" / "dqn_mine_agent.zip"

        if dqn_path_scenario.exists():
            print(f"  Chargement DQN scénario '{scenario.name}': {dqn_path_scenario}")
            dqn_model = DQN.load(str(dqn_path_scenario))
        else:
            print(f"  Entraînement DQN pour scénario '{scenario.name}'...")
            dqn_model = train_dqn(
                total_timesteps=scenario.total_timesteps,
                seed=scenario.seeds[0],
                truck_count=scenario.truck_count,
                shovel_count=scenario.shovel_count,
                dump_count=scenario.dump_count,
                episode_minutes=scenario.episode_minutes,
                breakdown_probability=scenario.breakdown_probability,
                reward_weights=scenario.reward_weights,
                output_dir=str(output_dir / f"dqn_{scenario.name}"),
            )

        def dqn_policy(obs, info, env, _model=dqn_model):
            action, _ = _model.predict(obs, deterministic=True)
            return int(action)

        policies["DQN"] = dqn_policy

    # --- PPO (Section 4.5.4) ---
    ppo_model = None
    if train_ppo_flag:
        ppo_path_scenario = output_dir / f"ppo_{scenario.name}" / "ppo_mine_agent.zip"

        if ppo_path_scenario.exists():
            print(f"  Chargement PPO scénario '{scenario.name}': {ppo_path_scenario}")
            ppo_model = PPO.load(str(ppo_path_scenario))
        else:
            print(f"  Entraînement PPO pour scénario '{scenario.name}'...")
            ppo_model = train_ppo(
                total_timesteps=scenario.total_timesteps,
                seed=scenario.seeds[0],
                truck_count=scenario.truck_count,
                shovel_count=scenario.shovel_count,
                dump_count=scenario.dump_count,
                episode_minutes=scenario.episode_minutes,
                breakdown_probability=scenario.breakdown_probability,
                reward_weights=scenario.reward_weights,
                output_dir=str(output_dir / f"ppo_{scenario.name}"),
            )

        def ppo_policy(obs, info, env, _model=ppo_model):
            action, _ = _model.predict(obs, deterministic=True)
            return int(action)

        policies["PPO"] = ppo_policy

    for policy_name, policy_fn in policies.items():
        for seed in scenario.seeds:
            env = MineEnv(
                truck_count=scenario.truck_count,
                shovel_count=scenario.shovel_count,
                dump_count=scenario.dump_count,
                episode_minutes=scenario.episode_minutes,
                breakdown_probability=scenario.breakdown_probability,
                reward_weights=scenario.reward_weights,
            )
            # Debug actions pour DQN, Nearest, ShortestPath (première seed seulement)
            debug = (policy_name in ["DQN", "Nearest", "ShortestPath"] and seed == scenario.seeds[0])
            kpis = run_single_episode(env, policy_fn, seed, policy_name, debug)
            row = {
                "scenario": scenario.name,
                "policy": policy_name,
                "seed": seed,
                **kpis,
            }
            results.append(row)
            print(
                f"  [{scenario.name}] {policy_name} seed={seed}: "
                f"prod={kpis['productivity_tph']:.1f} t/h, "
                f"wait={kpis['avg_wait_min_per_truck']:.1f} min, "
                f"fuel={kpis['specific_fuel_l_per_ton']:.4f} L/t"
            )

    return results


def run_all_benchmarks(
    output_dir: str = "data/results",
    train_ppo_flag: bool = True,
    scenario_names: list[str] | None = None,
) -> None:
    """Lance les benchmarks sur tous les scénarios."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    scenarios_to_run = SCENARIOS
    if scenario_names is not None:
        scenarios_to_run = {
            k: v for k, v in SCENARIOS.items() if k in scenario_names
        }

    all_results: list[dict] = []

    for name, scenario in scenarios_to_run.items():
        print(f"\n{'='*60}")
        print(f"Scénario: {name} — {scenario.description}")
        print(f"{'='*60}")
        results = run_benchmark(scenario, out_path, train_ppo_flag)
        all_results.extend(results)

    csv_path = out_path / "benchmark_results.csv"
    if all_results:
        fieldnames = list(all_results[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nRésultats sauvegardés : {csv_path}")


def main() -> None:
    """Point d'entrée pour le benchmark complet."""
    run_all_benchmarks(
        output_dir="data/results",
        train_ppo_flag=True,
        scenario_names=["nominal", "high_load", "high_breakdown"], # Les 3 scénarios du Chapitre 6
    )


if __name__ == "__main__":
    main()
