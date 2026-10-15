"""Benchmark : 8 politiques × scénarios × 30 graines d'évaluation.

- Heuristiques : FIFO, Fixed, Nearest, ShortestPath
- RL tabulaire : Q-Learning, SARSA
- RL profond : DQN, PPO

Les agents sont chargés depuis ``output_dir`` s'ils existent, sinon entraînés
(un agent par scénario, graine ``seeds[0]``). Chaque lancement écrit ses résultats
dans un nouveau fichier ``benchmark_<AAAAMMJJ_HHMMSS>.csv`` (colonne ``run_id``) ;
un doublon (scénario, politique, graine) lève ``ValueError`` (voir ``write_results``).
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
import time

from memoire_master_rl_logistique.baselines.base import BaselinePolicy
from memoire_master_rl_logistique.baselines.fifo_policy import FIFOPolicy
from memoire_master_rl_logistique.baselines.fixed_policy import FixedAssignmentPolicy
from memoire_master_rl_logistique.baselines.nearest_policy import NearestShovelPolicy
from memoire_master_rl_logistique.baselines.shortest_path_policy import ShortestPathPolicy
from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS, Scenario
from memoire_master_rl_logistique.rl.model_loading import load_model
from memoire_master_rl_logistique.rl.model_metadata import EnvConfig
from memoire_master_rl_logistique.rl.train_dqn import train_dqn
from memoire_master_rl_logistique.rl.train_ppo import train_ppo
from memoire_master_rl_logistique.rl.train_q_learning import train_q_learning
from memoire_master_rl_logistique.rl.train_sarsa import train_sarsa
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
    inference_times_ms: list[float] = []

    while True:
        t0 = time.perf_counter()
        action = policy_fn(obs, info, env)
        inference_times_ms.append((time.perf_counter() - t0) * 1_000)
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
    kpis["inference_time_ms_mean"] = (
        sum(inference_times_ms) / len(inference_times_ms) if inference_times_ms else 0.0
    )
    kpis["inference_time_ms_max"] = max(inference_times_ms) if inference_times_ms else 0.0
    return kpis


def run_benchmark(
    scenario: Scenario,
    output_dir: Path,
    train_ppo_flag: bool = True,
    model_scenario: str | None = None,
    allow_legacy_models: bool = False,
) -> list[dict]:
    """Exécute le benchmark pour un scénario donné.

    Avant de réutiliser un modèle déjà entraîné, ses métadonnées sont comparées à la
    configuration du scénario d'entraînement (``model_scenario``) : un modèle périmé
    lève ``ModelCompatibilityError``. ``allow_legacy_models`` permet de réévaluer, en
    connaissance de cause, des modèles antérieurs à T8 (sans ``metadata.json``).

    model_scenario : nom du scénario dont les modèles entraînés sont réutilisés.
        Si None, utilise scenario.name (comportement normal).
        Pour les scénarios de généralisation ("unseen_*"), passer "nominal" afin
        d'évaluer des modèles non spécialisés sur des conditions nouvelles.
    """
    results = []
    model_key = model_scenario if model_scenario is not None else scenario.name

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

    # Configuration avec laquelle les modèles réutilisés doivent avoir été entraînés.
    trained_on = EnvConfig.from_scenario(SCENARIOS.get(model_key, scenario))

    def _load(model_dir: Path, algorithm: str):
        return load_model(
            model_dir, algorithm, temp_env,
            trained_on=trained_on, allow_legacy=allow_legacy_models,
        )

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

    # --- Q-Learning ---
    ql_dir = output_dir / f"q_learning_{model_key}"
    ql_path = ql_dir / "q_table.pkl"

    if ql_path.exists():
        print(f"  Chargement Q-Learning existant: {ql_path}")
        ql_agent = _load(ql_dir, "Q-Learning")
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
        ql_agent = _load(ql_dir, "Q-Learning")

    def q_learning_policy(obs, info, env, _agent=ql_agent):
        return _agent.predict(obs)

    policies["Q-Learning"] = q_learning_policy

    # --- SARSA ---
    sarsa_dir = output_dir / f"sarsa_{model_key}"
    sarsa_path = sarsa_dir / "sarsa_table.pkl"

    if sarsa_path.exists():
        print(f"  Chargement SARSA existant: {sarsa_path}")
        sarsa_agent = _load(sarsa_dir, "SARSA")
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
        sarsa_agent = _load(sarsa_dir, "SARSA")

    def sarsa_policy_fn(obs, info, env, _agent=sarsa_agent):
        return _agent.predict(obs)

    policies["SARSA"] = sarsa_policy_fn

    # --- DQN ---
    if train_ppo_flag:
        dqn_path_scenario = output_dir / f"dqn_{model_key}" / "dqn_mine_agent.zip"

        if dqn_path_scenario.exists():
            print(f"  Chargement DQN ({model_key}): {dqn_path_scenario}")
            dqn_model = _load(dqn_path_scenario.parent, "DQN")
        else:
            print(f"  Entraînement DQN pour scénario '{model_key}'...")
            dqn_model = train_dqn(
                total_timesteps=scenario.total_timesteps,
                seed=scenario.seeds[0],
                truck_count=scenario.truck_count,
                shovel_count=scenario.shovel_count,
                dump_count=scenario.dump_count,
                episode_minutes=scenario.episode_minutes,
                breakdown_probability=scenario.breakdown_probability,
                reward_weights=scenario.reward_weights,
                output_dir=str(output_dir / f"dqn_{model_key}"),
            )

        def dqn_policy(obs, info, env, _model=dqn_model):
            action, _ = _model.predict(obs, deterministic=True)
            return int(action)

        policies["DQN"] = dqn_policy

    # --- PPO ---
    ppo_model = None
    if train_ppo_flag:
        ppo_path_scenario = output_dir / f"ppo_{model_key}" / "ppo_mine_agent.zip"

        if ppo_path_scenario.exists():
            print(f"  Chargement PPO ({model_key}): {ppo_path_scenario}")
            ppo_model = _load(ppo_path_scenario.parent, "PPO")
        else:
            print(f"  Entraînement PPO pour scénario '{model_key}'...")
            ppo_model = train_ppo(
                total_timesteps=scenario.total_timesteps,
                seed=scenario.seeds[0],
                truck_count=scenario.truck_count,
                shovel_count=scenario.shovel_count,
                dump_count=scenario.dump_count,
                episode_minutes=scenario.episode_minutes,
                breakdown_probability=scenario.breakdown_probability,
                reward_weights=scenario.reward_weights,
                output_dir=str(output_dir / f"ppo_{model_key}"),
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

def write_results(
        rows: list[dict],
        output_dir: Path,
        run_id: str | None = None
) -> Path:
    """Écrit les résultats d'une exécution dans un nouveau fichier daté. Renvoie son chemin."""

    if not rows:
        raise ValueError("La liste des résultats est vide.")

    seen = set()

    for row in rows:
        triplet = (row["scenario"], row["policy"], row["seed"])
        if triplet not in seen:
            seen.add(triplet)
        else:
            raise ValueError(f"Doublon détecté: {triplet}")

    if run_id is None:
        run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = output_dir / f"benchmark_{run_id}.csv"

    with open(csv_path, "x", newline="", encoding="utf-8") as f:
        fieldnames = ["run_id"] + list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"run_id": run_id, **row})

    return csv_path

def run_all_benchmarks(
    output_dir: str = "data/results",
    train_ppo_flag: bool = True,
    scenario_names: list[str] | None = None,
    allow_legacy_models: bool = False,
) -> None:
    """Lance les benchmarks sur tous les scénarios (voir ``run_benchmark`` pour les modèles)."""
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
        # Scénarios "non vus" : réutiliser les modèles nominal (test de généralisation)
        model_src = "nominal" if name.startswith("unseen_") else None
        results = run_benchmark(
            scenario, out_path, train_ppo_flag,
            model_scenario=model_src, allow_legacy_models=allow_legacy_models,
        )
        all_results.extend(results)

    if all_results:
        csv_path = write_results(all_results, out_path)
        print(f"\nRésultats sauvegardés : {csv_path}  ({len(all_results)} nouvelles lignes)")
    else:
        raise ValueError("Aucun résultat n'a été obtenu.")

def main() -> None:
    """Point d'entrée pour le benchmark complet."""
    run_all_benchmarks(
        output_dir="data/results",
        train_ppo_flag=True,
        scenario_names=["nominal", "high_load", "high_breakdown"],
    )


if __name__ == "__main__":
    main()
