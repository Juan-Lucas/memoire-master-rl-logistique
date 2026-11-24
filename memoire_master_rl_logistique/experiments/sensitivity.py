"""Sensibilité du simulateur (M2, T1 et T14 du rapport du 17/11/2026).

Les six heuristiques et les cinq modèles PPO nominaux de la campagne sont évalués, **sans
réentraînement**, dans des variantes de la mine nominale : durées de chargement et de
déversement, vitesses, retours à vide sans pente, usure des pistes, consommation au ralenti
et surconsommation sur piste dégradée. Chaque variante est évaluée sur les graines 72 à 101
(``seeds.EXTENSION_EVAL_SEEDS``), la référence nominale comprise : les graines 42 à 71 de
la campagne ne sont pas réutilisées, et toutes les comparaisons restent appariées.

On regarde si le classement des politiques change (tau de Kendall avec le classement de la
variante de référence), et ce que deviennent les écarts entre PPO et MTWT en productivité
et en carburant par tonne.

    python -m memoire_master_rl_logistique.experiments.sensitivity run DOSSIER \\
        --models data/results_v2/campagne --workers 4
    python -m memoire_master_rl_logistique.experiments.sensitivity analyze DOSSIER
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from memoire_master_rl_logistique.analysis.statistical_tests import (
    average_over_training_seeds,
    paired_bootstrap_ci,
)
from memoire_master_rl_logistique.experiments.policies import HEURISTICS
from memoire_master_rl_logistique.experiments.run_benchmark import (
    PolicySpec,
    _run_tasks,
    policy_specs,
)
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS, Scenario
from memoire_master_rl_logistique.seeds import EXTENSION_EVAL_SEEDS
from memoire_master_rl_logistique.simulation.fuel_model import DEFAULT_FUEL, FuelParams
from memoire_master_rl_logistique.simulation.variants import (
    NOMINAL_VARIANT,
    ROAD_WEAR_PROBABILITY,
    SimulationVariant,
)

RESULTS_FILE = "sensitivity_results.csv"
SUMMARY_FILE = "sensitivity_summary.csv"
REFERENCE = "reference"
PREFIX = "sensitivity_"

# Variante → (description, mine, carburant). Un seul facteur change à la fois.
VARIANTS: dict[str, tuple[str, SimulationVariant, FuelParams]] = {
    REFERENCE: ("nominal simulator", NOMINAL_VARIANT, DEFAULT_FUEL),
    "load_x0.8": ("loading time x0.8", SimulationVariant(load_time_scale=0.8), DEFAULT_FUEL),
    "load_x1.2": ("loading time x1.2", SimulationVariant(load_time_scale=1.2), DEFAULT_FUEL),
    "dump_x1.2": ("dumping time x1.2", SimulationVariant(unload_time_scale=1.2), DEFAULT_FUEL),
    "speed_x0.9": ("speeds x0.9", SimulationVariant(speed_scale=0.9), DEFAULT_FUEL),
    "speed_x1.1": ("speeds x1.1", SimulationVariant(speed_scale=1.1), DEFAULT_FUEL),
    "flat_returns": ("empty returns without grade", SimulationVariant(flat_returns=True), DEFAULT_FUEL),
    "road_wear_x2": ("road wear x2", SimulationVariant(road_wear_probability=2 * ROAD_WEAR_PROBABILITY),
                     DEFAULT_FUEL),
    "idle_8": ("idling 8 L/h", NOMINAL_VARIANT, replace(DEFAULT_FUEL, idle_l_per_h=8.0)),
    "idle_15": ("idling 15 L/h", NOMINAL_VARIANT, replace(DEFAULT_FUEL, idle_l_per_h=15.0)),
    "road_fuel_0": ("no overconsumption on degraded roads", NOMINAL_VARIANT,
                    replace(DEFAULT_FUEL, road_state_penalty=0.0)),
    "road_fuel_x2": ("overconsumption on degraded roads x2", NOMINAL_VARIANT,
                     replace(DEFAULT_FUEL, road_state_penalty=2 * DEFAULT_FUEL.road_state_penalty)),
}

PRODUCTIVITY, FUEL = "productivity_tph", "specific_fuel_l_per_ton"


def variant_scenario(key: str, seeds: Sequence[int] = EXTENSION_EVAL_SEEDS) -> Scenario:
    """Le régime nominal (12 camions, 2 % de pannes) dans la variante ``key``."""
    _, variant, fuel = VARIANTS[key]
    return replace(SCENARIOS["nominal"], name=PREFIX + key, seeds=list(seeds),
                   variant=variant, fuel=fuel)


def evaluated_specs(models_dir: Path | None) -> list[PolicySpec]:
    """Les six heuristiques, puis les cinq modèles PPO nominaux (aucun n'est entraîné)."""
    if models_dir is None:
        return [PolicySpec(name) for name in HEURISTICS]
    specs = policy_specs(SCENARIOS["nominal"], models_dir, model_scenarios=["nominal"],
                         train_missing=False)
    return [s for s in specs if s.model_dir is None or s.name == "PPO"]


def run(out_dir: Path, models_dir: Path | None, workers: int = 1,
        variants: Sequence[str] = tuple(VARIANTS), seeds: Sequence[int] = EXTENSION_EVAL_SEEDS,
        overwrite: bool = False) -> Path:
    """Évalue chaque politique dans chaque variante ; écrit ``sensitivity_results.csv``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / RESULTS_FILE
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} existe déjà (--overwrite pour le remplacer)")
    specs = evaluated_specs(models_dir)
    tasks = [(variant_scenario(key, seeds), spec, seed)
             for key in variants for spec in specs for seed in seeds]
    print(f"{len(tasks)} épisodes : {len(variants)} variantes × {len(specs)} politiques "
          f"× {len(seeds)} graines")
    rows = pd.DataFrame(_run_tasks(tasks, workers))
    rows.insert(1, "variant", rows["scenario"].str.removeprefix(PREFIX))
    rows.to_csv(out, index=False)
    print(f"Écrit : {out}")
    return out


def _ranking_tau(means: pd.DataFrame, reference: pd.Series) -> float:
    common = means.index.intersection(reference.index)
    tau, _ = kendalltau(means.loc[common], reference.loc[common])
    return float(tau)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par variante : tau de Kendall des classements (productivité, carburant par
    tonne) avec la référence, meilleure politique, écarts PPO − MTWT (en %) avec IC 95 %."""
    data = average_over_training_seeds(results.drop(columns=["variant"], errors="ignore"))
    data["variant"] = data["scenario"].str.removeprefix(PREFIX)
    means = data.groupby(["variant", "policy"])[[PRODUCTIVITY, FUEL]].mean()
    reference = means.loc[REFERENCE]
    rows = []
    for key in [k for k in VARIANTS if k in means.index.get_level_values(0)]:
        m = means.loc[key]
        row = {"variant": key, "description": VARIANTS[key][0],
               "best_policy": m[PRODUCTIVITY].idxmax(),
               "tau_productivity": _ranking_tau(m[PRODUCTIVITY], reference[PRODUCTIVITY]),
               "tau_fuel": _ranking_tau(m[FUEL], reference[FUEL])}
        if {"PPO", "MTWT"} <= set(m.index):
            sub = data[data["variant"] == key].pivot_table(index="seed", columns="policy",
                                                           values=[PRODUCTIVITY, FUEL])
            for metric, label in ((PRODUCTIVITY, "productivity"), (FUEL, "fuel")):
                diff = 100 * (sub[metric]["PPO"] - sub[metric]["MTWT"]) / sub[metric]["MTWT"]
                low, high = paired_bootstrap_ci(diff.to_numpy())
                row |= {f"ppo_vs_mtwt_{label}_pct": float(diff.mean()),
                        f"ppo_vs_mtwt_{label}_ci_low": low, f"ppo_vs_mtwt_{label}_ci_high": high}
        rows.append(row)
    return pd.DataFrame(rows)


def analyze(out_dir: Path) -> pd.DataFrame:
    """Lit ``sensitivity_results.csv``, écrit et affiche ``sensitivity_summary.csv``."""
    summary = summarise(pd.read_csv(out_dir / RESULTS_FILE))
    summary.to_csv(out_dir / SUMMARY_FILE, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.round(3).to_string(index=False))
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="évaluer les politiques dans chaque variante")
    p_run.add_argument("out_dir", type=Path)
    p_run.add_argument("--models", type=Path, default=None,
                       help="dossier des modèles de la campagne (sans : heuristiques seules)")
    p_run.add_argument("--workers", type=int, default=1)
    p_run.add_argument("--overwrite", action="store_true")
    p_an = sub.add_parser("analyze", help="résumer les résultats")
    p_an.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.out_dir, args.models, args.workers, overwrite=args.overwrite)
        analyze(args.out_dir)
    else:
        analyze(args.out_dir)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
