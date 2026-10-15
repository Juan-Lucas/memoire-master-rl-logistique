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

from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments.policies import agent_policy_fn, heuristic_policies
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS, Scenario
from memoire_master_rl_logistique.rl.discretize import DEFAULT_N_BINS
from memoire_master_rl_logistique.rl.model_loading import load_model, model_file
from memoire_master_rl_logistique.rl.model_metadata import EnvConfig
from memoire_master_rl_logistique.rl.model_paths import model_dir
from memoire_master_rl_logistique.rl.train_dqn import train_dqn
from memoire_master_rl_logistique.rl.train_ppo import train_ppo
from memoire_master_rl_logistique.rl.train_q_learning import train_q_learning
from memoire_master_rl_logistique.rl.train_sarsa import train_sarsa
from memoire_master_rl_logistique.simulation.kpi import fleet_kpis


def run_single_episode(
    env: MineEnv,
    policy_fn,
    seed: int,
    policy_name: str = "",
    debug_actions: bool = False,
) -> dict[str, float]:
    """Lance un épisode et retourne les KPIs.

    Deux durées sont mesurées séparément (T9, m20) :
    - ``decision_time_ms`` : l'appel de la politique seul (observation déjà construite) ;
    - ``obs_time_ms`` : la construction de l'observation par l'environnement.
    La latence complète d'une décision est leur somme.
    """
    tabular_agent = getattr(policy_fn, "tabular_agent", None)
    if tabular_agent is not None:
        tabular_agent.reset_coverage()
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    actions_taken = []
    decision_times_ms: list[float] = []
    obs_times_ms: list[float] = [env.last_obs_time_ms]

    while True:
        t0 = time.perf_counter()
        action = policy_fn(obs, info, env)
        decision_times_ms.append((time.perf_counter() - t0) * 1_000)
        if debug_actions and len(actions_taken) < 10:  # Log first 10 actions
            actions_taken.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        obs_times_ms.append(env.last_obs_time_ms)
        total_reward += reward
        if terminated or truncated:
            break
    
    if debug_actions and actions_taken:
        print(f"    [{policy_name}] First 10 actions: {actions_taken}")

    kpis = fleet_kpis(env.trucks, env.shovels, env.episode_minutes)
    kpis["total_reward"] = total_reward
    # Décomposition de la récompense cumulée (T25) : reward_production + … = total_reward.
    for name, value in info["episode_reward_terms"].items():
        kpis[f"reward_{name}"] = value
    # Décisions et part d'ATTENDRE (T19), comptées par l'environnement.
    kpis["decisions"] = float(info["decisions"])
    kpis["wait_actions"] = float(info["wait_actions"])
    kpis["wait_share_pct"] = 100.0 * info["wait_actions"] / max(info["decisions"], 1)
    if tabular_agent is not None:  # Q-Learning, SARSA : états connus et actions par défaut (T20)
        kpis.update(tabular_agent.coverage())
    for name, times in (("decision_time_ms", decision_times_ms), ("obs_time_ms", obs_times_ms)):
        kpis[f"{name}_mean"] = sum(times) / len(times) if times else 0.0
        kpis[f"{name}_max"] = max(times) if times else 0.0
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

    # Heuristiques : même construction que l'analyse du comportement (experiments.policies).
    policies = heuristic_policies(scenario)

    # --- Q-Learning ---
    ql_dir = model_dir(output_dir, "Q-Learning", model_key)
    ql_path = ql_dir / "q_table.pkl"

    if ql_path.exists():
        print(f"  Chargement Q-Learning existant: {ql_path}")
        ql_agent = _load(ql_dir, "Q-Learning")
    else:
        print(f"  Entraînement Q-Learning pour '{scenario.name}'...")
        train_q_learning(
            n_episodes=30_000,
            alpha=0.2,
            n_bins=DEFAULT_N_BINS,
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

    policies["Q-Learning"] = agent_policy_fn(ql_agent, "Q-Learning")

    # --- SARSA ---
    sarsa_dir = model_dir(output_dir, "SARSA", model_key)
    sarsa_path = sarsa_dir / "sarsa_table.pkl"

    if sarsa_path.exists():
        print(f"  Chargement SARSA existant: {sarsa_path}")
        sarsa_agent = _load(sarsa_dir, "SARSA")
    else:
        print(f"  Entraînement SARSA pour '{scenario.name}'...")
        train_sarsa(
            n_episodes=30_000,
            alpha=0.2,
            n_bins=DEFAULT_N_BINS,
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

    policies["SARSA"] = agent_policy_fn(sarsa_agent, "SARSA")

    # --- DQN ---
    if train_ppo_flag:
        dqn_path_scenario = model_file(model_dir(output_dir, "DQN", model_key), "DQN")

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
                output_dir=str(dqn_path_scenario.parent),
            )

        policies["DQN"] = agent_policy_fn(dqn_model, "DQN")

    # --- PPO ---
    ppo_model = None
    if train_ppo_flag:
        ppo_path_scenario = model_file(model_dir(output_dir, "PPO", model_key), "PPO")

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
                output_dir=str(ppo_path_scenario.parent),
            )

        policies["PPO"] = agent_policy_fn(ppo_model, "PPO")

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
