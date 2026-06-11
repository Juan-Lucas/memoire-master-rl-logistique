"""Distribution des actions choisies par chaque politique RL
sur les scénarios nominal, high_load et high_breakdown (10 épisodes, seeds 42-51).

Sortie : data/results/action_distribution.csv
Colonnes : scenario, algorithm, action_id, action_label, count, percentage
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
from stable_baselines3 import DQN, PPO

from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.rl.train_q_learning import QLearningPolicy
from memoire_master_rl_logistique.rl.train_sarsa import SarsaPolicy

ACTION_LABELS = {
    0: "Pelle1->Dump1",
    1: "Pelle1->Dump2",
    2: "Pelle2->Dump1",
    3: "Pelle2->Dump2",
    4: "Pelle3->Dump1",
    5: "Pelle3->Dump2",
    6: "ATTENDRE",
}


def count_actions(policy_fn, scenario, n_episodes: int = 10) -> Counter:
    counts: Counter = Counter()
    for seed in scenario.seeds[:n_episodes]:
        env = MineEnv(
            truck_count=scenario.truck_count,
            shovel_count=scenario.shovel_count,
            dump_count=scenario.dump_count,
            episode_minutes=scenario.episode_minutes,
            breakdown_probability=scenario.breakdown_probability,
            reward_weights=scenario.reward_weights,
        )
        obs, info = env.reset(seed=seed)
        while True:
            action = policy_fn(obs)
            counts[int(action)] += 1
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    return counts


def main() -> None:
    base = Path("data/results")
    rows = []

    for scenario_name in ["nominal", "high_load", "high_breakdown"]:
        scenario = SCENARIOS[scenario_name]

        ql = QLearningPolicy.load(base / f"q_learning_{scenario_name}" / "q_table.pkl")
        sarsa = SarsaPolicy.load(base / f"sarsa_{scenario_name}" / "sarsa_table.pkl")
        dqn = DQN.load(str(base / f"dqn_{scenario_name}" / "dqn_mine_agent.zip"))
        ppo = PPO.load(str(base / f"ppo_{scenario_name}" / "ppo_mine_agent.zip"))

        policies = {
            "Q-Learning": lambda obs, ql=ql: ql.predict(obs),
            "SARSA": lambda obs, sarsa=sarsa: sarsa.predict(obs),
            "DQN": lambda obs, dqn=dqn: int(dqn.predict(obs, deterministic=True)[0]),
            "PPO": lambda obs, ppo=ppo: int(ppo.predict(obs, deterministic=True)[0]),
        }

        for algo, fn in policies.items():
            counts = count_actions(fn, scenario)
            total = sum(counts.values())
            for action_id in range(7):
                rows.append({
                    "scenario": scenario_name,
                    "algorithm": algo,
                    "action_id": action_id,
                    "action_label": ACTION_LABELS[action_id],
                    "count": counts.get(action_id, 0),
                    "percentage": round(100 * counts.get(action_id, 0) / total, 2),
                })

    df = pd.DataFrame(rows)
    out = base / "action_distribution.csv"
    df.to_csv(out, index=False)
    print(f"Sauvegardé : {out}")
    for scenario_name in ["nominal", "high_load", "high_breakdown"]:
        print(f"\n=== {scenario_name} ===")
        print(df[df["scenario"] == scenario_name].pivot(index="algorithm", columns="action_label", values="percentage"))


if __name__ == "__main__":
    main()