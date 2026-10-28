"""Comportement des politiques sur 30 réplications (T12 bis, tableau ``tab:behaviour``).

Pour chaque politique, chaque régime et chaque graine, à partir du journal des cycles
(``MineEnv(log_cycles=True)``, T12) :
- la part des cycles réalisés à chaque pelle (%) ;
- la part des décisions ATTENDRE (%) ;
- la durée moyenne d'un cycle (min) ;
- la décomposition de l'immobilisation par camion (file, réparation, ATTENDRE ; T10).

Les résultats par graine sont écrits dans un CSV ; le tableau de l'article (moyennes
sur les graines) est produit à partir de ce fichier, en CSV et en LaTeX.

Campagne (T31) : ``campaign_behaviour`` **rejoue** les épisodes de l'évaluation unique
(mêmes politiques, mêmes modèles, mêmes graines, construits par
``run_benchmark.policy_specs``) avec le journal des cycles, qui n'est pas dans le CSV.
Rien n'est choisi ni réglé sur ces épisodes : ils sont décrits. Comme un épisode ne
dépend que de sa politique et de sa graine, le rejeu est identique à l'évaluation ;
``check_replay`` le vérifie épisode par épisode (productivité, attente en file,
carburant) et rien n'est écrit s'il diffère.

Lancement :
    python -m memoire_master_rl_logistique.analysis.behavior DOSSIER
        [--results-dir data/results] [--allow-legacy-models] [--overwrite]
    python -m memoire_master_rl_logistique.experiments.campaign behaviour DOSSIER_CAMPAGNE
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
# Colonnes comparées entre le rejeu et le CSV de l'évaluation (égalité exacte).
REPLAY_CHECK_COLUMNS = ("productivity_tph", "queue_wait_min_per_truck", "total_fuel_l")
REPLAY_KEY = ["scenario", "policy", "mode", "train_seed", "seed"]
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
    """Moyennes sur les graines, par (régime, politique) ; aussi par mode (spécialisé ou
    transféré) quand la colonne existe. Pour un agent, la moyenne porte sur toutes ses
    graines d'entraînement et d'évaluation."""
    keys = ["scenario", "policy", *(["mode"] if "mode" in runs.columns else [])]
    values = [c for c in runs.select_dtypes("number").columns
              if c not in (*keys, "seed", "train_seed")]
    return runs.groupby(keys, sort=False, dropna=False)[values].mean().reset_index()


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


def _replay(task) -> dict:
    """Rejoue un épisode de l'évaluation avec le journal des cycles (dans un processus)."""
    from memoire_master_rl_logistique.experiments.run_benchmark import policy_from_spec

    scenario, spec, seed = task
    env = make_env(scenario, log_cycles=True)
    kpis = run_single_episode(env, policy_from_spec(spec, scenario), seed, spec.name)
    row = {"scenario": scenario.name, "policy": spec.name, "mode": spec.mode,
           "train_seed": spec.train_seed, "seed": seed}
    row.update(episode_behaviour(env.cycle_log, [s.shovel_id for s in env.shovels]))
    row.update({k: kpis[k] for k in dict.fromkeys((*KPI_COLUMNS, *REPLAY_CHECK_COLUMNS))})
    return row


def campaign_runs(
    root: str | Path, scenarios: Sequence[str] = MAIN_SCENARIOS, n_workers: int = 1,
) -> pd.DataFrame:
    """Comportement de toutes les politiques évaluées par la campagne dans ``scenarios`` :
    une ligne par épisode, avec ``mode`` et ``train_seed``. N'entraîne jamais."""
    from memoire_master_rl_logistique.experiments.run_benchmark import _run_tasks, policy_specs

    tasks = []
    for name in scenarios:
        scenario = SCENARIOS[name]
        specs = policy_specs(scenario, Path(root), train_missing=False)
        tasks += [(scenario, spec, seed) for spec in specs for seed in scenario.seeds]
    return pd.DataFrame(_run_tasks(tasks, n_workers, worker=_replay))


def _keyed(frame: pd.DataFrame) -> pd.DataFrame:
    """Clé d'un épisode comparable entre les deux sources (mode et graine d'entraînement
    vides pour une heuristique : None dans le rejeu, NaN dans le CSV)."""
    frame = frame.copy()
    frame["mode"] = frame["mode"].fillna("").astype(str)
    frame["train_seed"] = frame["train_seed"].fillna(-1).astype(int)
    return frame.set_index(REPLAY_KEY)


def check_replay(runs: pd.DataFrame, results: pd.DataFrame) -> None:
    """Vérifie que les épisodes rejoués sont ceux de l'évaluation, avec les mêmes valeurs
    (égalité exacte) ; ``ValueError`` sinon."""
    replay = _keyed(runs)
    reference = _keyed(results[results["scenario"].isin(set(runs["scenario"]))])
    if replay.index.duplicated().any():
        raise ValueError("Épisode rejoué deux fois.")
    missing = reference.index.difference(replay.index)
    extra = replay.index.difference(reference.index)
    if len(missing) or len(extra):
        raise ValueError(f"Épisodes différents de l'évaluation : {len(missing)} manquant(s), "
                         f"{len(extra)} en trop ; ex. {list(missing[:3]) + list(extra[:3])}")
    columns = list(REPLAY_CHECK_COLUMNS)
    differs = (replay[columns] != reference.loc[replay.index, columns]).any(axis=1)
    if differs.any():
        raise ValueError(f"{int(differs.sum())} épisode(s) rejoué(s) différent(s) de "
                         f"l'évaluation ; ex. {list(replay.index[differs][:3])}")


def campaign_behaviour(
    root: str | Path, results_csv: str | Path, out_dir: str | Path, n_workers: int = 1,
    scenarios: Sequence[str] = MAIN_SCENARIOS,
) -> pd.DataFrame:
    """Rejoue, vérifie, puis écrit le comportement des politiques de la campagne
    (``behaviour_runs.csv``, ``behaviour_summary.csv``) dans ``out_dir``, qui doit être
    neuf. Rien n'est écrit si le rejeu diffère de l'évaluation."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Dossier déjà présent : {out_dir}")
    runs = campaign_runs(root, scenarios, n_workers)
    # round_trip : le lecteur rapide de pandas peut s'écarter d'un bit du nombre écrit.
    check_replay(runs, pd.read_csv(results_csv, float_precision="round_trip"))
    summary = behaviour_summary(runs)
    out_dir.mkdir(parents=True)
    runs.to_csv(out_dir / OUTPUT_FILES[0], index=False)
    summary.to_csv(out_dir / OUTPUT_FILES[1], index=False)
    print(f"{len(runs)} épisodes rejoués, identiques à l'évaluation : {out_dir}", flush=True)
    return summary


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
