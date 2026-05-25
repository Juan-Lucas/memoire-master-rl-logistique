"""Évaluation de l'agent PPO entraîné et des baselines.

Produit les métriques de performance (Section 4.5 du mémoire) :
- Productivité horaire (t/h)
- Temps d'attente moyen par camion (min)
- Consommation spécifique (L/t)
- Taux d'utilisation (%)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from memoire_master_rl_logistique.baselines.fixed_policy import FixedAssignmentPolicy
from memoire_master_rl_logistique.baselines.nearest_policy import NearestShovelPolicy
from memoire_master_rl_logistique.baselines.queue_aware_policy import QueueAwarePolicy
from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.simulation.kpi import compute_kpis


def evaluate_policy(
    env: MineEnv,
    policy_fn,
    n_episodes: int = 10,
    policy_name: str = "policy",
) -> dict[str, float]:
    """Évalue une politique sur n épisodes et retourne les KPIs moyens."""
    all_kpis: list[dict[str, float]] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=42 + ep)
        total_reward = 0.0

        while True:
            action = policy_fn(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

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
        all_kpis.append(kpis)

    avg_kpis = {}
    for key in all_kpis[0]:
        values = [k[key] for k in all_kpis]
        avg_kpis[f"{key}_mean"] = float(np.mean(values))
        avg_kpis[f"{key}_std"] = float(np.std(values))

    avg_kpis["policy_name"] = policy_name
    return avg_kpis


def evaluate_ppo(
    model_path: str,
    n_episodes: int = 10,
    truck_count: int = 12,
    shovel_count: int = 3,
    dump_count: int = 2,
    episode_minutes: float = 480.0,
) -> dict[str, float]:
    """Évalue l'agent PPO entraîné."""
    env = MineEnv(
        truck_count=truck_count,
        shovel_count=shovel_count,
        dump_count=dump_count,
        episode_minutes=episode_minutes,
    )
    model = PPO.load(model_path)

    def ppo_policy(obs, info):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return evaluate_policy(env, ppo_policy, n_episodes, "PPO")


def evaluate_all_baselines(
    n_episodes: int = 10,
    truck_count: int = 12,
    shovel_count: int = 3,
    dump_count: int = 2,
    episode_minutes: float = 480.0,
) -> list[dict[str, float]]:
    """Évalue toutes les baselines et retourne les résultats."""
    results = []

    env = MineEnv(
        truck_count=truck_count, shovel_count=shovel_count,
        dump_count=dump_count, episode_minutes=episode_minutes,
    )
    fixed = FixedAssignmentPolicy(
        num_shovels=shovel_count, num_dumps=dump_count,
    )
    results.append(evaluate_policy(
        env, lambda obs, info: fixed.predict(obs, info), n_episodes, "Fixed",
    ))

    env = MineEnv(
        truck_count=truck_count, shovel_count=shovel_count,
        dump_count=dump_count, episode_minutes=episode_minutes,
    )
    obs, info = env.reset()
    nearest = NearestShovelPolicy(
        graph=env.graph,
        shovel_node_ids=[s.node_id for s in env.shovels],
        dump_node_ids=[d.node_id for d in env.dumps],
    )
    results.append(evaluate_policy(
        env,
        lambda obs, info: nearest.predict(
            obs, info, env.truck_locations[env.current_truck_idx],
        ),
        n_episodes,
        "Nearest",
    ))

    env = MineEnv(
        truck_count=truck_count, shovel_count=shovel_count,
        dump_count=dump_count, episode_minutes=episode_minutes,
    )
    obs, info = env.reset()
    queue_aware = QueueAwarePolicy(
        graph=env.graph, shovels=env.shovels, dumps=env.dumps,
    )
    results.append(evaluate_policy(
        env,
        lambda obs, info: queue_aware.predict(
            obs, info, env.truck_locations[env.current_truck_idx],
            env.current_time_min,
        ),
        n_episodes,
        "QueueAware",
    ))

    return results


def main() -> None:
    """Point d'entrée pour l'évaluation complète."""
    print("=== Évaluation des baselines ===")
    baseline_results = evaluate_all_baselines(n_episodes=10)

    for result in baseline_results:
        policy_name = result.get("policy_name", "Unknown")
        print(f"\n--- {policy_name} ---")
        print(
            f"  Productivité: {result['productivity_tph_mean']:.1f} "
            f"± {result['productivity_tph_std']:.1f} t/h"
        )
        print(
            f"  Attente moy/camion: "
            f"{result['avg_wait_min_per_truck_mean']:.1f} "
            f"± {result['avg_wait_min_per_truck_std']:.1f} min"
        )
        print(
            f"  Utilisation: {result['utilization_pct_mean']:.1f} "
            f"± {result['utilization_pct_std']:.1f} %"
        )
        print(
            f"  Conso spécifique: "
            f"{result['specific_fuel_l_per_ton_mean']:.3f} "
            f"± {result['specific_fuel_l_per_ton_std']:.3f} L/t"
        )

    model_path = Path("models/ppo_mine/ppo_mine_agent.zip")
    if model_path.exists():
        print("\n=== Évaluation PPO ===")
        ppo_result = evaluate_ppo(str(model_path), n_episodes=10)
        print(
            f"  Productivité: {ppo_result['productivity_tph_mean']:.1f} "
            f"± {ppo_result['productivity_tph_std']:.1f} t/h"
        )
        print(
            f"  Attente moy/camion: "
            f"{ppo_result['avg_wait_min_per_truck_mean']:.1f} "
            f"± {ppo_result['avg_wait_min_per_truck_std']:.1f} min"
        )
        print(
            f"  Utilisation: {ppo_result['utilization_pct_mean']:.1f} "
            f"± {ppo_result['utilization_pct_std']:.1f} %"
        )
        print(
            f"  Conso spécifique: "
            f"{ppo_result['specific_fuel_l_per_ton_mean']:.3f} "
            f"± {ppo_result['specific_fuel_l_per_ton_std']:.3f} L/t"
        )
    else:
        print(
            f"\nModèle PPO non trouvé ({model_path}). "
            "Lancez d'abord l'entraînement."
        )


if __name__ == "__main__":
    main()
