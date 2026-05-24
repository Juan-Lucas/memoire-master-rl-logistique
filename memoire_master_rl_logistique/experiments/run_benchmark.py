"""Benchmark comparatif complet.

Exécute toutes les méthodes (baselines + PPO) sur tous les scénarios
avec plusieurs seeds pour produire les données du Chapitre 6.

Protocole expérimental (Section 5.4 du mémoire) :
- Mêmes scénarios pour toutes les méthodes
- 10 réplications par scénario (seeds 42..51)
- KPIs calculés de la même manière
- Résultats sauvegardés en CSV reproductible
"""

from __future__ import annotations

import csv
from pathlib import Path

from stable_baselines3 import PPO

from memoire_master_rl_logistique.baselines.fixed_policy import FixedAssignmentPolicy
from memoire_master_rl_logistique.baselines.nearest_policy import NearestShovelPolicy
from memoire_master_rl_logistique.baselines.queue_aware_policy import QueueAwarePolicy
from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS, Scenario
from memoire_master_rl_logistique.rl.train_ppo import train_ppo
from memoire_master_rl_logistique.simulation.kpi import compute_kpis


def run_single_episode(
    env: MineEnv,
    policy_fn,
    seed: int,
) -> dict[str, float]:
    """Lance un épisode et retourne les KPIs."""
    obs, info = env.reset(seed=seed)
    total_reward = 0.0

    while True:
        action = policy_fn(obs, info, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    total_tonnage = sum(t.total_tonnage_t for t in env.trucks)
    total_wait = sum(t.total_wait_min for t in env.trucks)
    total_active = sum(t.total_active_min for t in env.trucks)
    total_fuel = sum(t.total_fuel_l for t in env.trucks)

    kpis = compute_kpis(
        episode_minutes=env.episode_minutes,
        truck_count=env.truck_count,
        total_tonnage_t=total_tonnage,
        total_wait_min=total_wait,
        total_active_min=total_active,
        total_fuel_l=total_fuel,
    )
    kpis["total_reward"] = total_reward
    kpis["total_cycles"] = float(sum(t.cycles_completed for t in env.trucks))
    return kpis


def run_benchmark(
    scenario: Scenario,
    output_dir: Path,
    train_ppo_flag: bool = True,
) -> list[dict]:
    """Exécute le benchmark pour un scénario donné."""
    results = []

    def fixed_policy(obs, info, env):
        p = FixedAssignmentPolicy(
            num_shovels=env.shovel_count, num_dumps=env.dump_count,
        )
        return p.predict(obs, info)

    def nearest_policy(obs, info, env):
        p = NearestShovelPolicy(
            graph=env.graph,
            shovel_node_ids=[s.node_id for s in env.shovels],
            dump_node_ids=[d.node_id for d in env.dumps],
        )
        return p.predict(obs, info, env.truck_locations[env.current_truck_idx])

    def queue_aware_policy(obs, info, env):
        p = QueueAwarePolicy(
            graph=env.graph, shovels=env.shovels, dumps=env.dumps,
        )
        return p.predict(
            obs, info,
            env.truck_locations[env.current_truck_idx],
            env.current_time_min,
        )

    policies = {
        "Fixed": fixed_policy,
        "Nearest": nearest_policy,
        "QueueAware": queue_aware_policy,
    }

    ppo_model = None
    if train_ppo_flag:
        model_dir = output_dir / f"ppo_{scenario.name}"
        model_path = model_dir / "ppo_mine_agent.zip"

        if model_path.exists():
            print(f"  Chargement du modèle PPO existant: {model_path}")
            ppo_model = PPO.load(str(model_path))
        else:
            print(f"  Entraînement PPO pour '{scenario.name}'...")
            ppo_model = train_ppo(
                total_timesteps=scenario.total_timesteps,
                seed=scenario.seeds[0],
                truck_count=scenario.truck_count,
                shovel_count=scenario.shovel_count,
                dump_count=scenario.dump_count,
                episode_minutes=scenario.episode_minutes,
                reward_weights=scenario.reward_weights,
                output_dir=str(model_dir),
            )

        def ppo_policy(obs, info, env):
            action, _ = ppo_model.predict(obs, deterministic=True)
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
            kpis = run_single_episode(env, policy_fn, seed)
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
        scenario_names=["nominal"],
    )


if __name__ == "__main__":
    main()
