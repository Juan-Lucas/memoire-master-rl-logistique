"""Benchmark : 8 politiques × scénarios × 30 graines d'évaluation.

- Heuristiques : FIFO, Fixed, Nearest, ShortestPath
- RL tabulaire : Q-Learning, SARSA
- RL profond : DQN, PPO

Les agents sont chargés depuis ``output_dir`` s'ils existent, sinon entraînés : un
agent par (algorithme, régime d'entraînement, graine d'entraînement de
``seeds.TRAIN_SEEDS``), jamais sur une graine d'évaluation (T26). En régime perturbé,
les agents spécialisés et les agents nominaux transférés sont évalués (T27, colonne
``mode``). Chaque lancement écrit ses résultats dans un nouveau fichier
``benchmark_<AAAAMMJJ_HHMMSS>.csv`` ; un doublon (scénario, politique, mode, graine
d'entraînement, graine d'évaluation) lève ``ValueError`` (voir ``write_results``).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
import datetime as dt
from pathlib import Path
import time

from memoire_master_rl_logistique.config import snapshot_configs
from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments.policies import (
    PolicyFn,
    agent_policy_fn,
    heuristic_policies,
    make_env,
)
from memoire_master_rl_logistique.experiments.scenarios import (
    MAIN_SCENARIOS,
    SCENARIOS,
    Scenario,
    evaluation_mode,
    training_regimes,
)
from memoire_master_rl_logistique.rl.model_loading import load_model, model_file
from memoire_master_rl_logistique.rl.model_metadata import EnvConfig
from memoire_master_rl_logistique.rl.model_paths import model_dir
from memoire_master_rl_logistique.rl.train_dqn import train_dqn
from memoire_master_rl_logistique.rl.train_ppo import train_ppo
from memoire_master_rl_logistique.rl.train_q_learning import train_q_learning
from memoire_master_rl_logistique.rl.train_sarsa import train_sarsa
from memoire_master_rl_logistique.seeds import TRAIN_SEEDS
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
    model_scenarios: Sequence[str] | None = None,
    allow_legacy_models: bool = False,
    train_seeds: Sequence[int | None] = TRAIN_SEEDS,
) -> list[dict]:
    """Exécute le benchmark pour un scénario donné.

    model_scenarios : régimes d'entraînement des agents évalués (T27). Par défaut
        ``training_regimes(scenario.name)`` : en régime perturbé, les agents spécialisés
        et les agents nominaux transférés ; pour un scénario non vu, les agents nominaux.
        Colonne ``mode`` du CSV : « spécialisé » ou « transféré » (vide pour les
        heuristiques, évaluées une seule fois).

    train_seeds : graines d'entraînement (T26) : un modèle par algorithme et par graine,
        évalué sur toutes les graines d'évaluation ; colonne ``train_seed`` du CSV.
        ``None`` désigne un modèle d'avant T26 (dossier sans suffixe, ex. modèles v1) :
        il est relu, jamais entraîné.

    Avant de réutiliser un modèle déjà entraîné, ses métadonnées sont comparées à la
    configuration de son régime d'entraînement : un modèle périmé lève
    ``ModelCompatibilityError``. ``allow_legacy_models`` permet de réévaluer, en
    connaissance de cause, des modèles antérieurs à T8 (sans ``metadata.json``).
    """
    if model_scenarios is None:
        model_scenarios = training_regimes(scenario.name)
    eval_env = make_env(scenario)  # sert à vérifier que chaque modèle accepte l'observation

    # Heuristiques : même construction que l'analyse du comportement (experiments.policies).
    runs: list[tuple[str, str | None, int | None, PolicyFn]] = [
        (name, None, None, fn) for name, fn in heuristic_policies(scenario).items()
    ]

    # Agents : un modèle par (régime d'entraînement, graine, algorithme), entraîné s'il manque.
    algorithms = ("Q-Learning", "SARSA") + (("DQN", "PPO") if train_ppo_flag else ())
    for regime in model_scenarios:
        training_scenario = SCENARIOS.get(regime, scenario)
        trained_on = EnvConfig.from_scenario(training_scenario)
        mode = evaluation_mode(scenario.name, regime)
        for train_seed in train_seeds:
            for algorithm in algorithms:
                mdir = model_dir(output_dir, algorithm, regime, train_seed)
                if not model_file(mdir, algorithm).exists():
                    if train_seed is None:
                        raise FileNotFoundError(f"Modèle d'avant T26 introuvable : {mdir}")
                    print(f"  Entraînement {algorithm} ({regime}, graine {train_seed})...")
                    _train(algorithm, training_scenario, train_seed, mdir)
                model = load_model(mdir, algorithm, eval_env, trained_on=trained_on,
                                   allow_legacy=allow_legacy_models)
                runs.append((algorithm, mode, train_seed, agent_policy_fn(model, algorithm)))

    results = []
    for policy_name, mode, train_seed, policy_fn in runs:
        for seed in scenario.seeds:
            kpis = run_single_episode(make_env(scenario), policy_fn, seed, policy_name)
            results.append({
                "scenario": scenario.name,
                "policy": policy_name,
                "mode": mode,  # None pour les heuristiques (rien à entraîner ni transférer)
                "train_seed": train_seed,
                "seed": seed,
                **kpis,
            })
            print(
                f"  [{scenario.name}] {policy_name} {mode or ''} entraînement={train_seed} "
                f"seed={seed}: prod={kpis['productivity_tph']:.1f} t/h, "
                f"file={kpis['queue_wait_min_per_truck']:.1f} min, "
                f"fuel={kpis['specific_fuel_l_per_ton']:.4f} L/t"
            )

    return results


def _train(algorithm: str, scenario: Scenario, train_seed: int, output_dir: Path) -> None:
    """Entraîne un agent : régime et budget du scénario, hyperparamètres de ``configs/``."""
    env_kwargs = dict(
        seed=train_seed,
        truck_count=scenario.truck_count,
        shovel_count=scenario.shovel_count,
        dump_count=scenario.dump_count,
        episode_minutes=scenario.episode_minutes,
        breakdown_probability=scenario.breakdown_probability,
        reward_weights=scenario.reward_weights,
        output_dir=str(output_dir),
    )
    # Hyperparamètres : configs/*_hyperparameters.yaml, lus par chaque entraîneur (T28).
    if algorithm in ("Q-Learning", "SARSA"):
        trainer = train_q_learning if algorithm == "Q-Learning" else train_sarsa
        trainer(**env_kwargs)
    else:
        trainer = train_dqn if algorithm == "DQN" else train_ppo
        trainer(total_timesteps=scenario.total_timesteps, **env_kwargs)


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
        # (régime, politique, mode, graine d'entraînement, graine d'évaluation) : T26 et
        # T27 ajoutent graine d'entraînement et mode, absents avant et pour les heuristiques.
        triplet = (row["scenario"], row["policy"], row.get("mode"), row.get("train_seed"),
                   row["seed"])
        if triplet not in seen:
            seen.add(triplet)
        else:
            raise ValueError(f"Doublon détecté: {triplet}")

    if run_id is None:
        run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = output_dir / f"benchmark_{run_id}.csv"

    with open(csv_path, "x", newline="", encoding="utf-8") as f:
        # Union des colonnes, dans l'ordre d'apparition : les agents tabulaires ont des
        # colonnes (known_state_pct…) que les heuristiques, écrites d'abord, n'ont pas.
        fieldnames = ["run_id"] + list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"run_id": run_id, **row})

    return csv_path

def run_all_benchmarks(
    output_dir: str = "data/results",
    train_ppo_flag: bool = True,
    scenario_names: Sequence[str] | None = None,
    allow_legacy_models: bool = False,
    train_seeds: Sequence[int] = TRAIN_SEEDS,
    run_id: str | None = None,
) -> Path:
    """Lance les benchmarks sur les scénarios demandés (tous si ``None``) et renvoie le CSV.

    La configuration utilisée est archivée **avant** le calcul dans
    ``<output_dir>/configs_<run_id>/`` (T28) : même un lancement interrompu garde la
    trace de ses réglages, et ``MINE_RL_CONFIG_DIR`` permet de le rejouer à l'identique.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Configuration archivée : {snapshot_configs(out_path / f'configs_{run_id}')}")

    unknown = set(scenario_names or ()) - set(SCENARIOS)
    if unknown:
        raise ValueError(f"Scénarios inconnus : {sorted(unknown)}")
    names = list(SCENARIOS) if scenario_names is None else list(scenario_names)

    all_results: list[dict] = []
    for name in names:
        scenario = SCENARIOS[name]
        print(f"\n{'='*60}")
        print(f"Scénario: {name} — {scenario.description}")
        print(f"{'='*60}")
        # Agents spécialisés et/ou nominaux transférés : voir scenarios.training_regimes (T27).
        all_results.extend(run_benchmark(
            scenario, out_path, train_ppo_flag,
            allow_legacy_models=allow_legacy_models, train_seeds=train_seeds,
        ))

    csv_path = write_results(all_results, out_path, run_id)
    print(f"\nRésultats sauvegardés : {csv_path}  ({len(all_results)} nouvelles lignes)")
    return csv_path


def main(argv: Sequence[str] | None = None) -> Path:
    """Campagne en une commande, avec la configuration de ``configs/`` (ou ``MINE_RL_CONFIG_DIR``).

    python -m memoire_master_rl_logistique.experiments.run_benchmark DOSSIER [--scenarios ...]
    """
    parser = argparse.ArgumentParser(description="Benchmark des politiques de dispatching.")
    parser.add_argument("output_dir", type=Path, help="dossier des modèles et des résultats")
    parser.add_argument("--scenarios", nargs="+", default=MAIN_SCENARIOS, choices=list(SCENARIOS),
                        help="scénarios à évaluer (défaut : main_scenarios de la configuration)")
    parser.add_argument("--allow-legacy-models", action="store_true")
    args = parser.parse_args(argv)
    return run_all_benchmarks(str(args.output_dir), scenario_names=args.scenarios,
                              allow_legacy_models=args.allow_legacy_models)


if __name__ == "__main__":
    main()
