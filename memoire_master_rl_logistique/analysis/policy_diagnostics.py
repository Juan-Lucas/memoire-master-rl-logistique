"""Diagnostics des politiques sans entraînement (T2, T3, T9, T17 du rapport du 17/11/2026).

Les modèles PPO de la campagne sont rejoués sur les graines de sélection (aucune graine
d'évaluation) ; à chaque décision on relève :

- **T17, accord avec MTWT** : la décision que MTWT prendrait dans le même état. PPO reçoit
  l'attente prévue par route, l'information de MTWT : apprend-il autre chose ? Accord sur la
  route et sur la pelle, et, quand la pelle diffère, si PPO choisit une pelle plus proche ;
- **T2, saturation de l'observation** : part des décisions où une composante atteint sa
  borne (1 après normalisation), par bloc ;
- **T3, attentes** : suites d'ATTENDRE d'un même camion et part des attentes de moins de
  0,5 min, à partir du journal des cycles ;
- **T9, latence** : distribution du temps d'une décision (politique + observation) pour
  MTWT, LP, PPO et DQN, en séquentiel, avec la description de la machine.

Régimes : nominal, forte charge, fortes pannes (modèles spécialisés) et 34 camions près de
la saturation (modèles nominaux, sans réentraînement).

    python -m memoire_master_rl_logistique.analysis.policy_diagnostics DOSSIER_SORTIE \\
        --models data/results_v2/campagne
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import psutil

from memoire_master_rl_logistique.env.observation_builder import observation_layout
from memoire_master_rl_logistique.experiments.disruptions import regime_scenario
from memoire_master_rl_logistique.experiments.policies import heuristic_policies, make_env
from memoire_master_rl_logistique.experiments.run_benchmark import policy_from_spec, policy_specs
from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS, SCENARIOS
from memoire_master_rl_logistique.seeds import TUNING_SEEDS

BLOCKS = ("shovel_wait", "dump_wait", "shovel_trucks", "route_wait", "tonnage", "fuel")
SHORT_WAIT_MIN = 0.5
SATURATION = "saturation_34"


def _regimes() -> list[tuple[str, str]]:
    """(régime évalué, régime d'entraînement des modèles)."""
    return [(r, r) for r in MAIN_SCENARIOS] + [(SATURATION, "nominal")]


def _scenario(regime: str, seeds: Sequence[int]):
    return regime_scenario(SATURATION, seeds) if regime == SATURATION else SCENARIOS[regime]


def wait_runs(cycle_log) -> dict[str, float]:
    """Suites d'ATTENDRE d'un même camion (la plus longue, en nombre de décisions) et part des
    attentes de moins de ``SHORT_WAIT_MIN`` minutes."""
    by_truck: dict[str, list] = {}
    for c in cycle_log:
        by_truck.setdefault(c.truck_id, []).append(c)
    longest, waits, short = 0, 0, 0
    for records in by_truck.values():
        run = 0
        for c in sorted(records, key=lambda r: r.start_min):
            if c.kind == "wait":
                run += 1
                waits += 1
                short += c.voluntary_wait_min < SHORT_WAIT_MIN
                longest = max(longest, run)
            else:
                run = 0
    return {"waits": waits, "longest_wait_run": longest,
            "short_wait_share": short / waits if waits else np.nan}


def replay(models_dir: Path, seeds: Sequence[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Une ligne par décision de PPO (accord, saturation) et une ligne par épisode (attentes)."""
    decisions, episodes = [], []
    for regime, trained in _regimes():
        scenario = _scenario(regime, seeds)
        mtwt = heuristic_policies(scenario)["MTWT"]
        specs = [s for s in policy_specs(SCENARIOS[trained], models_dir, model_scenarios=[trained],
                                         train_missing=False) if s.name == "PPO"]
        layout = None
        for spec in specs:
            ppo = policy_from_spec(spec, scenario)
            for seed in seeds:
                env = make_env(scenario, log_cycles=True)
                obs, info = env.reset(seed=seed)
                layout = layout or observation_layout(env.shovel_count, env.dump_count)
                wait_action = env.action_space.n - 1
                while True:
                    action = int(ppo(obs, info, env))
                    reference = int(mtwt(obs, info, env))
                    row = {"regime": regime, "train_seed": spec.train_seed, "seed": seed,
                           "action": action, "mtwt_action": reference, "wait": action == wait_action}
                    if action != wait_action:
                        ps, ms = divmod(action, env.dump_count)[0], divmod(reference, env.dump_count)[0]
                        dist = obs[layout["distance_to_shovel"]]
                        row |= {"same_route": action == reference, "same_shovel": ps == ms,
                                "ppo_shovel": ps, "mtwt_shovel": ms,
                                "ppo_closer": bool(dist[ps] < dist[ms])}
                    for block in BLOCKS:
                        row[f"sat_{block}"] = bool((obs[layout[block]] >= 0.9999).any())
                    decisions.append(row)
                    obs, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                episodes.append({"regime": regime, "train_seed": spec.train_seed, "seed": seed,
                                 "wait_share_pct": 100 * env.wait_action_count / env.decision_count,
                                 **wait_runs(env.cycle_log)})
    return pd.DataFrame(decisions), pd.DataFrame(episodes)


def latency(models_dir: Path, seeds: Sequence[int], regime: str = "nominal") -> pd.DataFrame:
    """Temps d'une décision (politique + observation), par décision, en séquentiel."""
    scenario = SCENARIOS[regime]
    heur = heuristic_policies(scenario)
    agents = {s.name: policy_from_spec(s, scenario)
              for s in policy_specs(scenario, models_dir, model_scenarios=[regime], train_missing=False)
              if s.name in ("PPO", "DQN") and s.train_seed == 1}
    rows = []
    for name, fn in {"MTWT": heur["MTWT"], "LP": heur["LP"], **agents}.items():
        times = []
        for seed in seeds:
            env = make_env(scenario)
            obs, info = env.reset(seed=seed)
            while True:
                t0 = time.perf_counter()
                action = fn(obs, info, env)
                decide = (time.perf_counter() - t0) * 1_000
                obs, _, terminated, truncated, info = env.step(action)
                times.append(decide + env.last_obs_time_ms)
                if terminated or truncated:
                    break
        t = np.array(times)
        rows.append({"policy": name, "n_decisions": len(t), "mean_ms": t.mean(),
                     "median_ms": np.median(t), "p95_ms": np.percentile(t, 95),
                     "p99_ms": np.percentile(t, 99), "max_ms": t.max()})
    return pd.DataFrame(rows)


def machine() -> dict[str, str]:
    return {"processor": platform.processor(), "physical_cores": str(psutil.cpu_count(logical=False)),
            "logical_cores": str(psutil.cpu_count()), "ram_gb": f"{psutil.virtual_memory().total / 2**30:.0f}",
            "os": platform.platform(), "python": platform.python_version(),
            "torch_threads": os.environ.get("OMP_NUM_THREADS", "default")}


def summarise(decisions: pd.DataFrame, episodes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    moves = decisions[~decisions["wait"]].copy()
    for column in ("same_route", "same_shovel", "ppo_closer"):
        moves[column] = moves[column].astype(bool)
    agreement = moves.groupby("regime").agg(
        decisions=("action", "size"), same_route=("same_route", "mean"),
        same_shovel=("same_shovel", "mean"))
    differ = moves[~moves["same_shovel"]]
    agreement["closer_when_different"] = differ.groupby("regime")["ppo_closer"].mean()
    agreement["wait_share"] = decisions.groupby("regime")["wait"].mean()
    agreement = agreement.reset_index()
    saturation = decisions.groupby("regime")[[f"sat_{b}" for b in BLOCKS]].mean().reset_index()
    waits = episodes.groupby("regime").agg(
        wait_share_pct=("wait_share_pct", "mean"), longest_wait_run=("longest_wait_run", "max"),
        short_wait_share=("short_wait_share", "mean")).reset_index()
    return {"agreement": agreement, "saturation": saturation, "waits": waits}


def run(out_dir: Path, models_dir: Path, seeds: Sequence[int] = TUNING_SEEDS[:5],
        latency_seeds: Sequence[int] = TUNING_SEEDS[:3]) -> dict[str, pd.DataFrame]:
    decisions, episodes = replay(models_dir, seeds)
    outputs = summarise(decisions, episodes)
    outputs["latency"] = latency(models_dir, latency_seeds)
    outputs["machine"] = pd.DataFrame([machine()])
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(out_dir / "episodes.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        for name, table in outputs.items():
            table.to_csv(out_dir / f"{name}.csv", index=False)
            print(f"\n===== {name} =====")
            print(table.round(4).to_string(index=False))
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.out_dir, args.models)


if __name__ == "__main__":
    main()
