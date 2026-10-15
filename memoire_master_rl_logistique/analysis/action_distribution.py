"""Distribution des actions choisies par chaque politique RL
sur les scénarios nominal, high_load et high_breakdown (10 épisodes, seeds 42-51).

Les modèles sont chargés avec contrôle de leurs métadonnées (``rl.model_loading``).

Sortie : <results_dir>/action_distribution.csv
Colonnes : scenario, algorithm, action_id, action_label, count, percentage

Lancement :
    python -m memoire_master_rl_logistique.analysis.action_distribution [--results-dir DIR]
        [--allow-legacy-models]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS, SCENARIOS
from memoire_master_rl_logistique.rl.model_loading import load_model

ACTION_LABELS = {
    0: "Pelle1->Dump1",
    1: "Pelle1->Dump2",
    2: "Pelle2->Dump1",
    3: "Pelle2->Dump2",
    4: "Pelle3->Dump1",
    5: "Pelle3->Dump2",
    6: "ATTENDRE",
}


def make_env(scenario) -> MineEnv:
    """Environnement d'évaluation d'un scénario."""
    return MineEnv(
        truck_count=scenario.truck_count,
        shovel_count=scenario.shovel_count,
        dump_count=scenario.dump_count,
        episode_minutes=scenario.episode_minutes,
        breakdown_probability=scenario.breakdown_probability,
        reward_weights=scenario.reward_weights,
    )


def count_actions(policy_fn, scenario, n_episodes: int = 10) -> Counter:
    """Nombre de fois où chaque action est choisie sur ``n_episodes`` épisodes."""
    counts: Counter = Counter()
    for seed in scenario.seeds[:n_episodes]:
        env = make_env(scenario)
        obs, info = env.reset(seed=seed)
        while True:
            action = policy_fn(obs)
            counts[int(action)] += 1
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    return counts


def main(results_dir: str | Path = "data/results", allow_legacy_models: bool = False) -> None:
    """Compte les actions des 4 agents RL dans chaque régime et écrit le CSV."""
    base = Path(results_dir)
    rows = []

    for scenario_name in MAIN_SCENARIOS:
        scenario = SCENARIOS[scenario_name]
        env = make_env(scenario)

        def load(prefix: str, algorithm: str, _env=env, _name=scenario_name):
            return load_model(base / f"{prefix}_{_name}", algorithm, _env,
                              allow_legacy=allow_legacy_models)

        ql = load("q_learning", "Q-Learning")
        sarsa = load("sarsa", "SARSA")
        dqn = load("dqn", "DQN")
        ppo = load("ppo", "PPO")

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
    for scenario_name in MAIN_SCENARIOS:
        print(f"\n=== {scenario_name} ===")
        print(df[df["scenario"] == scenario_name].pivot(index="algorithm", columns="action_label", values="percentage"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distribution des actions des agents RL.")
    parser.add_argument("--results-dir", default="data/results", help="dossier des modèles")
    parser.add_argument("--allow-legacy-models", action="store_true",
                        help="accepter des modèles sans metadata.json (antérieurs à T8)")
    args = parser.parse_args()
    main(args.results_dir, args.allow_legacy_models)