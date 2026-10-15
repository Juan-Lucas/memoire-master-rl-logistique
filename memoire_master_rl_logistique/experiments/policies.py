"""Construction des huit politiques évaluées, avec une signature commune.

Toutes les politiques sont des fonctions ``policy_fn(obs, info, env) -> action``.
Ce module est partagé par le benchmark (``run_benchmark``) et l'analyse du
comportement (``analysis.behavior``) : les deux font jouer exactement les mêmes
politiques.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from memoire_master_rl_logistique.baselines.base import BaselinePolicy
from memoire_master_rl_logistique.baselines.fifo_policy import FIFOPolicy
from memoire_master_rl_logistique.baselines.fixed_policy import FixedAssignmentPolicy
from memoire_master_rl_logistique.baselines.mtwt_policy import MTWTPolicy
from memoire_master_rl_logistique.baselines.nearest_policy import NearestShovelPolicy
from memoire_master_rl_logistique.baselines.shortest_path_policy import ShortestPathPolicy
from memoire_master_rl_logistique.env.mine_env import MineEnv
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS, Scenario
from memoire_master_rl_logistique.rl.model_loading import load_model
from memoire_master_rl_logistique.rl.model_metadata import EnvConfig
from memoire_master_rl_logistique.rl.model_paths import model_dir

PolicyFn = Callable[[Any, dict, MineEnv], int]

HEURISTICS = ("FIFO", "Fixed", "Nearest", "ShortestPath", "MTWT")
AGENTS = ("Q-Learning", "SARSA", "DQN", "PPO")


def make_env(scenario: Scenario, **kwargs: Any) -> MineEnv:
    """Environnement d'un scénario ; ``kwargs`` est transmis à ``MineEnv`` (ex. ``log_cycles``)."""
    return MineEnv(
        truck_count=scenario.truck_count,
        shovel_count=scenario.shovel_count,
        dump_count=scenario.dump_count,
        episode_minutes=scenario.episode_minutes,
        breakdown_probability=scenario.breakdown_probability,
        reward_weights=scenario.reward_weights,
        **kwargs,
    )


def _baseline_fn(policy: BaselinePolicy, needs_graph_sync: bool = False) -> PolicyFn:
    """Adapte une ``BaselinePolicy`` à la signature ``policy_fn(obs, info, env)``."""
    def policy_fn(obs, info, env):
        if needs_graph_sync:
            policy.graph = env.graph
        return policy.predict(obs, info, truck_location=env.truck_locations[env.current_truck_idx])
    return policy_fn


def heuristic_policies(scenario: Scenario) -> dict[str, PolicyFn]:
    """Les heuristiques (4 de la v1, plus MTWT depuis T23), initialisées pour le scénario."""
    env = make_env(scenario)
    env.reset(seed=scenario.seeds[0])
    shovel_nodes = [s.node_id for s in env.shovels]
    dump_nodes = [d.node_id for d in env.dumps]
    counts = {"num_shovels": scenario.shovel_count, "num_dumps": scenario.dump_count}
    return {
        "FIFO": _baseline_fn(FIFOPolicy(**counts)),
        "Fixed": _baseline_fn(FixedAssignmentPolicy(**counts)),
        "Nearest": _baseline_fn(
            NearestShovelPolicy(env.graph, shovel_nodes, dump_nodes), needs_graph_sync=True),
        "ShortestPath": _baseline_fn(
            ShortestPathPolicy(env.graph, shovel_nodes, dump_nodes), needs_graph_sync=True),
        "MTWT": _baseline_fn(
            MTWTPolicy(env.graph, shovel_nodes, dump_nodes), needs_graph_sync=True),
    }


def agent_policy_fn(agent: Any, algorithm: str) -> PolicyFn:
    """Politique d'un agent chargé : gloutonne (tabulaire) ou déterministe (SB3)."""
    if algorithm in ("Q-Learning", "SARSA"):
        def tabular_fn(obs, info, env):
            return agent.predict(obs)
        # Permet à run_single_episode de mesurer la couverture de la Q-table (T20).
        tabular_fn.tabular_agent = agent
        return tabular_fn
    return lambda obs, info, env: int(agent.predict(obs, deterministic=True)[0])


def agent_policies(
    scenario: Scenario,
    results_dir: str | Path,
    model_scenario: str | None = None,
    allow_legacy: bool = False,
    algorithms: tuple[str, ...] = AGENTS,
) -> dict[str, PolicyFn]:
    """Charge (sans jamais entraîner) les agents du scénario, avec contrôle des métadonnées.

    ``model_scenario`` : régime d'entraînement des modèles, si différent du scénario évalué.
    """
    key = model_scenario or scenario.name
    env = make_env(scenario)
    trained_on = EnvConfig.from_scenario(SCENARIOS.get(key, scenario))
    return {
        algo: agent_policy_fn(
            load_model(model_dir(results_dir, algo, key), algo, env,
                       trained_on=trained_on, allow_legacy=allow_legacy),
            algo,
        )
        for algo in algorithms
    }
