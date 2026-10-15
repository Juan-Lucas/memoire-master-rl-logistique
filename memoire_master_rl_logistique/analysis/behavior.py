"""Comportement des politiques sur 30 réplications (T12 bis, tableau ``tab:behaviour``).

Pour chaque politique, chaque régime et chaque graine, à partir du journal des cycles
(``MineEnv(log_cycles=True)``, T12) :
- la part des cycles réalisés à chaque pelle (%) ;
- la part des décisions ATTENDRE (%) ;
- la durée moyenne d'un cycle (min) ;
- la décomposition de l'immobilisation par camion (file, réparation, ATTENDRE ; T10).

Les résultats par graine sont écrits dans un CSV ; le tableau de l'article (moyennes
sur les graines) est produit à partir de ce fichier, en CSV et en LaTeX.

Lancement :
    python -m memoire_master_rl_logistique.analysis.behavior DOSSIER
        [--results-dir data/results] [--allow-legacy-models] [--overwrite]
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from memoire_master_rl_logistique.experiments.policies import (
    agent_policies,
    heuristic_policies,
    make_env,
)
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS, SCENARIOS
from memoire_master_rl_logistique.simulation.entities import CycleRecord

KPI_COLUMNS = (
    "productivity_tph", "queue_wait_min_per_truck", "repair_min_per_truck",
    "voluntary_wait_min_per_truck",
)
OUTPUT_FILES = ("behaviour_runs.csv", "behaviour_summary.csv", "tab_behaviour.tex")


def episode_behaviour(cycle_log: Iterable[CycleRecord], shovel_ids: Sequence[str]) -> dict[str, float]:
    """Indicateurs de comportement d'un épisode, à partir de son journal des cycles.

    - ``share_<pelle>_pct`` : part des cycles (affectations) réalisés à cette pelle ;
    - ``wait_share_pct`` : part des décisions qui sont des ATTENDRE ;
    - ``mean_cycle_min`` : durée moyenne d'un cycle, de l'affectation à la disponibilité.
    """
    log = list(cycle_log)
    cycles = [r for r in log if r.kind == "cycle"]
    n_cycles = max(len(cycles), 1)
    row: dict[str, float] = {
        f"share_{sid}_pct": 100.0 * sum(r.shovel_id == sid for r in cycles) / n_cycles
        for sid in shovel_ids
    }
    row["wait_share_pct"] = 100.0 * (len(log) - len(cycles)) / max(len(log), 1)
    row["mean_cycle_min"] = (
        sum(r.end_min - r.start_min for r in cycles) / len(cycles) if cycles else 0.0
    )
    row["n_decisions"] = float(len(log))
    return row


def behaviour_runs(
    policies_for,
    scenarios: Sequence[str] = MAIN_SCENARIOS,
    seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Une ligne par (régime, politique, graine).

    ``policies_for(scenario) -> {nom: policy_fn}`` fournit les politiques de chaque régime ;
    ``seeds`` : par défaut les 30 graines du scénario.
    """
    rows = []
    for name in scenarios:
        scenario = SCENARIOS[name]
        for policy, policy_fn in policies_for(scenario).items():
            for seed in seeds if seeds is not None else scenario.seeds:
                env = make_env(scenario, log_cycles=True)
                kpis = run_single_episode(env, policy_fn, seed)
                row = {"scenario": name, "policy": policy, "seed": seed}
                row.update(episode_behaviour(env.cycle_log, [s.shovel_id for s in env.shovels]))
                row.update({k: kpis[k] for k in KPI_COLUMNS})
                rows.append(row)
    return pd.DataFrame(rows)


def behaviour_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Moyennes sur les graines, par (régime, politique)."""
    values = [c for c in runs.columns if c not in ("scenario", "policy", "seed")]
    return runs.groupby(["scenario", "policy"], sort=False)[values].mean().reset_index()


def behaviour_latex(summary: pd.DataFrame, labels: dict[str, str] | None = None) -> str:
    """Lignes du tableau ``tab:behaviour`` : « P1 / P2 / P3 » et part ATTENDRE, par régime."""
    labels = labels or {}
    shares = sorted(c for c in summary.columns if c.startswith("share_"))
    scenarios = [s for s in MAIN_SCENARIOS if s in set(summary["scenario"])]
    table = summary.set_index(["policy", "scenario"])
    lines = []
    for policy in dict.fromkeys(summary["policy"]):
        cells = []
        for scenario in scenarios:
            row = table.loc[(policy, scenario)]
            cells.append(" / ".join(f"{row[c]:.0f}" for c in shares))
            cells.append(f"{row['wait_share_pct']:.0f}")
        lines.append(f"{labels.get(policy, policy):<20} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def main(
    out_dir: str | Path,
    results_dir: str | Path = "data/results",
    allow_legacy_models: bool = False,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Calcule le comportement des 8 politiques sur 30 graines et écrit CSV + LaTeX."""
    from memoire_master_rl_logistique.plots.common import POLICY_LABELS, prepare_outputs

    runs_csv, summary_csv, latex = prepare_outputs(out_dir, OUTPUT_FILES, overwrite)

    def policies_for(scenario):
        return heuristic_policies(scenario) | agent_policies(
            scenario, results_dir, allow_legacy=allow_legacy_models)

    runs = behaviour_runs(policies_for)
    summary = behaviour_summary(runs)
    runs.to_csv(runs_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    latex.write_text(behaviour_latex(summary, POLICY_LABELS), encoding="utf-8")
    print(summary.round(1).to_string(index=False))
    print(f"Écrits : {runs_csv}, {summary_csv}, {latex}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comportement des politiques (T12 bis).")
    parser.add_argument("out_dir", type=Path, help="dossier de sortie (créé si besoin)")
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--allow-legacy-models", action="store_true",
                        help="accepter des modèles sans metadata.json (antérieurs à T8)")
    parser.add_argument("--overwrite", action="store_true", help="autoriser l'écrasement")
    args = parser.parse_args()
    main(args.out_dir, args.results_dir, args.allow_legacy_models, args.overwrite)
