"""Perturbations plus sévères, sans réentraînement (M3 du rapport du 17/11/2026, option A).

Régimes évalués sur les graines 72 à 101 (``seeds.EXTENSION_EVAL_SEEDS``) :

- ``reference`` : régime nominal (12 camions), pour les comparaisons appariées ;
- ``shovel_outage`` : 12 camions, la pelle 1 (la plus proche, la plus chargée) en panne de
  120 à 240 min : perturbation **pendant** le poste ;
- ``saturation_29`` et ``saturation_34`` : 29 et 34 camions, facteur d'appariement 0,85
  et 1,0 (proche de la saturation des pelles).

Politiques : les six heuristiques, les cinq modèles PPO entraînés en conditions nominales
et les cinq entraînés en forte charge (18 camions), tous **sans réentraînement**. Chaque
épisode enregistre le tonnage livré par tranche de 30 min, pour mesurer la perte pendant la
panne et le temps de récupération.

    python -m memoire_master_rl_logistique.experiments.disruptions run DOSSIER \\
        --models data/results_v2/campagne --workers 4
    python -m memoire_master_rl_logistique.experiments.disruptions analyze DOSSIER
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from memoire_master_rl_logistique.analysis.statistical_tests import paired_bootstrap_ci
from memoire_master_rl_logistique.experiments.policies import HEURISTICS, make_env
from memoire_master_rl_logistique.experiments.run_benchmark import (
    PolicySpec,
    _run_tasks,
    policy_from_spec,
    policy_specs,
    run_single_episode,
)
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS, Scenario
from memoire_master_rl_logistique.seeds import EXTENSION_EVAL_SEEDS
from memoire_master_rl_logistique.simulation.variants import SimulationVariant

RESULTS_FILE = "disruptions_results.csv"
SUMMARY_FILE = "disruptions_summary.csv"
RECOVERY_FILE = "disruptions_recovery.csv"
REFERENCE, OUTAGE = "reference", "shovel_outage"
OUTAGE_WINDOW = (0, 120.0, 240.0)  # pelle 1, de 2 h à 4 h de poste
BIN_MIN = 30.0
PREFIX = "disruption_"
REGIMES: dict[str, dict] = {
    REFERENCE: {},
    OUTAGE: {"variant": SimulationVariant(shovel_outages=(OUTAGE_WINDOW,))},
    "saturation_29": {"truck_count": 29},
    "saturation_34": {"truck_count": 34},
}
# Modèles PPO : entraînés en nominal, ou en forte charge (le plus proche de la saturation).
PPO_TRAINING = {"nominal": "PPO (nominal)", "high_load": "PPO (high load)"}
PRODUCTIVITY = "productivity_tph"


def regime_scenario(key: str, seeds: Sequence[int] = EXTENSION_EVAL_SEEDS) -> Scenario:
    return replace(SCENARIOS["nominal"], name=PREFIX + key, seeds=list(seeds), **REGIMES[key])


def evaluated_specs(models_dir: Path | None) -> list[PolicySpec]:
    """Heuristiques, puis PPO entraîné en nominal et en forte charge (nom distinct)."""
    specs = [PolicySpec(name) for name in HEURISTICS]
    if models_dir is None:
        return specs
    for regime, label in PPO_TRAINING.items():
        found = policy_specs(SCENARIOS[regime], models_dir, model_scenarios=[regime],
                             train_missing=False)
        specs += [replace(s, mode=label) for s in found if s.name == "PPO"]
    return specs


def tonnage_by_bin(env, minutes: float) -> list[float]:
    """Tonnage livré par tranche de ``BIN_MIN`` minutes (fin du déversement)."""
    bins = np.zeros(int(np.ceil(minutes / BIN_MIN)))
    for c in env.cycle_log:
        if c.kind == "cycle" and c.delivered_t > 0:
            end = c.end_min - c.repair_min
            bins[min(int(end // BIN_MIN), len(bins) - 1)] += c.delivered_t
    return bins.tolist()


def _evaluate(task: tuple[Scenario, PolicySpec, int]) -> dict:
    scenario, spec, seed = task
    env = make_env(scenario, log_cycles=True)
    kpis = run_single_episode(env, policy_from_spec(spec, scenario), seed, spec.name)
    bins = {f"t_{int(i * BIN_MIN):03d}": v
            for i, v in enumerate(tonnage_by_bin(env, scenario.episode_minutes))}
    policy = spec.mode if spec.model_dir else spec.name
    return {"regime": scenario.name.removeprefix(PREFIX), "policy": policy,
            "train_seed": spec.train_seed, "seed": seed, **kpis, **bins}


def run(out_dir: Path, models_dir: Path | None, workers: int = 1,
        regimes: Sequence[str] = tuple(REGIMES), seeds: Sequence[int] = EXTENSION_EVAL_SEEDS,
        overwrite: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / RESULTS_FILE
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} existe déjà (--overwrite pour le remplacer)")
    specs = evaluated_specs(models_dir)
    tasks = [(regime_scenario(r, seeds), s, seed) for r in regimes for s in specs for seed in seeds]
    print(f"{len(tasks)} épisodes : {len(regimes)} régimes × {len(specs)} politiques × {len(seeds)} graines")
    pd.DataFrame(_run_tasks(tasks, workers, worker=_evaluate)).to_csv(out, index=False)
    print(f"Écrit : {out}")
    return out


def per_replication(results: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (régime, politique, graine) : moyenne des modèles d'un agent."""
    numeric = [c for c in results.select_dtypes("number").columns if c not in ("seed", "train_seed")]
    return results.groupby(["regime", "policy", "seed"], sort=False)[numeric].mean().reset_index()


def summarise(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Résumé par (régime, politique), et courbes de récupération de la panne de pelle.

    Résumé : productivité moyenne, rang, écart à MTWT (%, IC 95 % apparié), et pour la panne
    la perte de tonnage sur le poste par rapport à la référence (même politique, même graine).
    Récupération : tonnage moyen par tranche, panne et référence ; ``recovery_min`` est la
    durée, après la fin de la panne, avant que le débit retrouve 95 % de celui de la
    référence dans la même tranche.
    """
    data = per_replication(results)
    bins = sorted(c for c in data.columns if c.startswith("t_"))
    rows = []
    for regime, d in data.groupby("regime", sort=False):
        piv = d.pivot_table(index="seed", columns="policy", values=PRODUCTIVITY)
        means = piv.mean().sort_values(ascending=False)
        ref = data[data["regime"] == REFERENCE].pivot_table(index="seed", columns="policy",
                                                            values=PRODUCTIVITY)
        tau = float(kendalltau(means, ref.mean().loc[means.index])[0]) if regime != REFERENCE else 1.0
        for rank, (policy, mean) in enumerate(means.items(), 1):
            row = {"regime": regime, "policy": policy, "rank": rank, "productivity_tph": mean,
                   "tau_vs_reference": tau}
            if "MTWT" in piv and policy != "MTWT":
                diff = 100 * (piv[policy] - piv["MTWT"]) / piv["MTWT"]
                low, high = paired_bootstrap_ci(diff.to_numpy())
                row |= {"vs_mtwt_pct": float(diff.mean()), "vs_mtwt_ci_low": low, "vs_mtwt_ci_high": high}
            if regime != REFERENCE and policy in ref:
                loss = 100 * (ref[policy] - piv[policy]) / ref[policy]
                low, high = paired_bootstrap_ci(loss.to_numpy())
                row |= {"loss_vs_reference_pct": float(loss.mean()),
                        "loss_ci_low": low, "loss_ci_high": high}
            rows.append(row)
    summary = pd.DataFrame(rows)

    curves = []
    end_bin = int(OUTAGE_WINDOW[2] // BIN_MIN)
    for policy in data["policy"].unique():
        out = data[(data["regime"] == OUTAGE) & (data["policy"] == policy)][bins].mean()
        ref = data[(data["regime"] == REFERENCE) & (data["policy"] == policy)][bins].mean()
        if out.empty or ref.empty:
            continue
        recovered = [i for i in range(end_bin, len(bins)) if out.iloc[i] >= 0.95 * ref.iloc[i]]
        recovery = (recovered[0] - end_bin) * BIN_MIN if recovered else np.nan
        lost = float((ref - out).sum())
        curves.append({"policy": policy, "recovery_min": recovery, "tonnage_lost_t": lost,
                       **{f"outage_{b}": out[b] for b in bins}, **{f"reference_{b}": ref[b] for b in bins}})
    return summary, pd.DataFrame(curves)


def analyze(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, recovery = summarise(pd.read_csv(out_dir / RESULTS_FILE))
    summary.to_csv(out_dir / SUMMARY_FILE, index=False)
    recovery.to_csv(out_dir / RECOVERY_FILE, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(summary.round(3).to_string(index=False))
        print(recovery[["policy", "recovery_min", "tonnage_lost_t"]].round(1).to_string(index=False))
    return summary, recovery


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
