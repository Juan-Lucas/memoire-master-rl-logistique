"""Comparaison énergétique équitable (M5 et T11 du rapport du 17/11/2026).

PPO consomme moins de carburant par tonne que MTWT, mais MTWT ne regarde que l'attente.
Cette expérience compare PPO à une référence orientée énergie, MTWT sensible au carburant
(``baselines.mtwt_fuel_policy``), pour plusieurs poids λ du carburant (λ = 0 : MTWT), dans
les trois régimes principaux, sur les graines 72 à 101. Aucun réentraînement : PPO est le
modèle spécialisé de chaque régime (5 graines d'entraînement).

Le résumé donne, par régime et par politique, la productivité et le carburant par tonne
avec leurs IC à 95 %, et la part des rééchantillonnages bootstrap où la politique est sur
le front de Pareto productivité / carburant par tonne (incertitude du front, T11).

    python -m memoire_master_rl_logistique.experiments.energy run DOSSIER \\
        --models data/results_v2/campagne --workers 4
    python -m memoire_master_rl_logistique.experiments.energy analyze DOSSIER
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from memoire_master_rl_logistique.analysis.statistical_tests import paired_bootstrap_ci
from memoire_master_rl_logistique.baselines.mtwt_fuel_policy import FuelAwareMTWTPolicy
from memoire_master_rl_logistique.experiments.policies import _baseline_fn, make_env
from memoire_master_rl_logistique.experiments.run_benchmark import (
    PolicySpec,
    _run_tasks,
    policy_from_spec,
    policy_specs,
    run_single_episode,
)
from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS, SCENARIOS, Scenario
from memoire_master_rl_logistique.plots.benchmark_comparisons import pareto_front
from memoire_master_rl_logistique.seeds import EXTENSION_EVAL_SEEDS

RESULTS_FILE = "energy_results.csv"
SUMMARY_FILE = "energy_summary.csv"
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)  # minutes d'attente par litre
FUEL_PREFIX = "MTWT-fuel "
OTHER_HEURISTICS = ("LP", "Nearest")
PRODUCTIVITY, FUEL = "productivity_tph", "specific_fuel_l_per_ton"
N_BOOT = 1_000


def fuel_policy_name(weight: float) -> str:
    return "MTWT" if weight == 0 else f"{FUEL_PREFIX}{weight:g}"


def evaluated_specs(regime: str, models_dir: Path | None,
                    lambdas: Sequence[float] = LAMBDAS) -> list[PolicySpec]:
    specs = [PolicySpec(fuel_policy_name(w)) for w in lambdas]
    specs += [PolicySpec(name) for name in OTHER_HEURISTICS]
    if models_dir is not None:
        found = policy_specs(SCENARIOS[regime], models_dir, model_scenarios=[regime],
                             train_missing=False)
        specs += [s for s in found if s.name == "PPO"]
    return specs


_FUEL_POLICIES: dict[tuple[str, float], object] = {}


def _policy(spec: PolicySpec, scenario: Scenario):  # noqa: ANN202
    if spec.name == "MTWT" or spec.name.startswith(FUEL_PREFIX):
        weight = 0.0 if spec.name == "MTWT" else float(spec.name.removeprefix(FUEL_PREFIX))
        key = (repr(scenario), weight)
        if key not in _FUEL_POLICIES:
            env = make_env(scenario)
            env.reset(seed=scenario.seeds[0])
            policy = FuelAwareMTWTPolicy(
                env.graph, [s.node_id for s in env.shovels], [d.node_id for d in env.dumps],
                weight, load_time_min=env.shovels[0].load_time_mean_min, fuel=scenario.fuel)
            _FUEL_POLICIES[key] = _baseline_fn(policy, needs_graph_sync=True)
        return _FUEL_POLICIES[key]
    return policy_from_spec(spec, scenario)


def _evaluate(task: tuple[Scenario, PolicySpec, int]) -> dict:
    scenario, spec, seed = task
    kpis = run_single_episode(make_env(scenario), _policy(spec, scenario), seed, spec.name)
    return {"regime": scenario.name, "policy": spec.name, "train_seed": spec.train_seed,
            "seed": seed, **kpis}


def run(out_dir: Path, models_dir: Path | None, workers: int = 1,
        regimes: Sequence[str] = MAIN_SCENARIOS, seeds: Sequence[int] = EXTENSION_EVAL_SEEDS,
        lambdas: Sequence[float] = LAMBDAS, overwrite: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / RESULTS_FILE
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} existe déjà (--overwrite pour le remplacer)")
    tasks = []
    for regime in regimes:
        scenario = replace(SCENARIOS[regime], seeds=list(seeds))
        tasks += [(scenario, spec, seed)
                  for spec in evaluated_specs(regime, models_dir, lambdas) for seed in seeds]
    print(f"{len(tasks)} épisodes")
    pd.DataFrame(_run_tasks(tasks, workers, worker=_evaluate)).to_csv(out, index=False)
    print(f"Écrit : {out}")
    return out


def summarise(results: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Par régime et politique : moyennes et IC 95 %, écart à PPO (%, IC apparié) et part
    des rééchantillonnages où la politique est sur le front de Pareto."""
    data = results.groupby(["regime", "policy", "seed"], sort=False)[[PRODUCTIVITY, FUEL]] \
        .mean().reset_index()  # PPO : moyenne de ses 5 modèles sur chaque réplication
    rng = np.random.default_rng(seed)
    rows = []
    for regime, d in data.groupby("regime", sort=False):
        prod = d.pivot(index="seed", columns="policy", values=PRODUCTIVITY)
        fuel = d.pivot(index="seed", columns="policy", values=FUEL)
        policies = list(prod.columns)
        on_front = np.zeros(len(policies))
        for _ in range(n_boot):
            idx = rng.integers(0, len(prod), len(prod))
            front = pareto_front(fuel.iloc[idx].mean().to_numpy(), prod.iloc[idx].mean().to_numpy())
            on_front += front
        for i, policy in enumerate(policies):
            row = {"regime": regime, "policy": policy,
                   PRODUCTIVITY: prod[policy].mean(), FUEL: fuel[policy].mean(),
                   "pareto_share": on_front[i] / n_boot}
            if "PPO" in prod and policy != "PPO":
                for metric, table in ((PRODUCTIVITY, prod), (FUEL, fuel)):
                    diff = 100 * (table["PPO"] - table[policy]) / table[policy]
                    low, high = paired_bootstrap_ci(diff.to_numpy())
                    row |= {f"ppo_vs_{metric}_pct": float(diff.mean()),
                            f"ppo_vs_{metric}_ci_low": low, f"ppo_vs_{metric}_ci_high": high}
            rows.append(row)
    return pd.DataFrame(rows)


def analyze(out_dir: Path) -> pd.DataFrame:
    summary = summarise(pd.read_csv(out_dir / RESULTS_FILE))
    summary.to_csv(out_dir / SUMMARY_FILE, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(summary.round(4).to_string(index=False))
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("out_dir", type=Path)
    p_run.add_argument("--models", type=Path, default=None)
    p_run.add_argument("--workers", type=int, default=1)
    p_run.add_argument("--overwrite", action="store_true")
    p_an = sub.add_parser("analyze")
    p_an.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.out_dir, args.models, args.workers, overwrite=args.overwrite)
    analyze(args.out_dir)


if __name__ == "__main__":
    main()
