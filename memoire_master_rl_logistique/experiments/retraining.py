"""Réentraînements de diagnostic après la campagne (rapport du 17/11/2026 : M5, M6, T17).

Six variantes de PPO et de DQN (``rl/agent_variants.py``, ``retraining`` dans
``configs/experiment_config.yaml``), entraînées en régime nominal avec les 5 graines, les
hyperparamètres et le budget de la campagne. Analyse séparée de la campagne : les modèles
de la campagne ne sont ni réentraînés ni modifiés ; ils servent de référence.

1. ``train`` : entraîne les variantes, plusieurs à la fois (même lanceur que la campagne :
   modèle fini sauté, entraînement interrompu repris, un journal par modèle dans ``logs/``).
2. ``evaluate`` : une seule fois, sur les graines 72 à 101 (``seeds.evaluation_extension``),
   les variantes, PPO et DQN nominaux de la campagne, MTWT et l'Affectation fixe. Refuse de
   démarrer s'il manque un modèle.
3. ``analyze`` : par politique, moyennes sur les réplications (agents : moyenne des 5 modèles
   par réplication), écart à la référence de la campagne en % avec IC bootstrap apparié, et
   par modèle (graine d'entraînement).

    python -m memoire_master_rl_logistique.experiments.retraining train DOSSIER --campaign data\\results_v2\\campagne --workers 4
    python -m memoire_master_rl_logistique.experiments.retraining evaluate DOSSIER --campaign data\\results_v2\\campagne --workers 4
    python -m memoire_master_rl_logistique.experiments.retraining analyze DOSSIER
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
import contextlib
from dataclasses import dataclass, replace
import json
from pathlib import Path
import time

import pandas as pd

from memoire_master_rl_logistique.analysis.statistical_tests import paired_bootstrap_ci
from memoire_master_rl_logistique.experiments import campaign
from memoire_master_rl_logistique.experiments.policies import (
    agent_policy_fn,
    heuristic_policies,
    make_env,
)
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS, Scenario
from memoire_master_rl_logistique.rl.agent_variants import (
    AgentVariant,
    load_variants,
    retraining_config,
    variant_policy_fn,
    wrap_env,
)
from memoire_master_rl_logistique.rl.checkpoints import STATE_FILE, checkpoint_dir
from memoire_master_rl_logistique.rl.model_loading import load_model, model_file
from memoire_master_rl_logistique.rl.model_metadata import METADATA_FILENAME, EnvConfig
from memoire_master_rl_logistique.rl.model_paths import model_dir
from memoire_master_rl_logistique.seeds import EXTENSION_EVAL_SEEDS, TRAIN_SEEDS

RESULTS_FILE = "retraining_results.csv"
SUMMARY_FILE = "retraining_summary.csv"
PER_MODEL_FILE = "retraining_per_model.csv"
HEURISTICS = ("MTWT", "Fixed")
CAMPAIGN_AGENTS = ("PPO", "DQN")
PRODUCTIVITY, FUEL = "productivity_tph", "specific_fuel_l_per_ton"
QUEUE, WAIT = "queue_wait_min_per_truck", "wait_share_pct"
METRICS = (PRODUCTIVITY, FUEL, QUEUE, WAIT, "share_S1", "share_S2", "share_S3")
# Durée relative d'un entraînement : DQN d'abord (plus long par pas en parallèle), puis PPO.
ORDER = {"DQN": 0, "PPO": 1}


def regime() -> Scenario:
    return SCENARIOS[retraining_config()["regime"]]


@dataclass(frozen=True, order=True)
class VariantJob:
    """Un entraînement de variante ; même interface que ``campaign.TrainJob``."""

    variant: str
    train_seed: int

    def directory(self, root: str | Path) -> Path:
        return Path(root) / f"{self.variant}_seed{self.train_seed}"

    def algorithm(self) -> str:
        return load_variants()[self.variant].algorithm

    def done(self, root: str | Path) -> bool:
        return model_file(self.directory(root), self.algorithm()).exists()

    def label(self) -> str:
        return f"{self.variant}_seed{self.train_seed}"


def variant_jobs(variants: Sequence[str] | None = None,
                 train_seeds: Sequence[int] = TRAIN_SEEDS) -> list[VariantJob]:
    found = load_variants()
    names = list(found) if variants is None else list(variants)
    unknown = set(names) - set(found)
    if unknown:
        raise ValueError(f"Variantes inconnues : {sorted(unknown)} (connues : {list(found)})")
    jobs = [VariantJob(name, seed) for name in names for seed in train_seeds]
    return sorted(jobs, key=lambda j: (ORDER[found[j.variant].algorithm], names.index(j.variant),
                                       j.train_seed))


def train_variant(variant: AgentVariant, scenario: Scenario, train_seed: int,
                  output_dir: Path) -> None:
    """Entraîne une variante : régime et budget du scénario, hyperparamètres de la campagne."""
    from memoire_master_rl_logistique.rl.train_dqn import train_dqn
    from memoire_master_rl_logistique.rl.train_ppo import train_ppo

    resume = (checkpoint_dir(output_dir) / STATE_FILE).exists()
    if resume:
        print(f"Reprise de l'entraînement interrompu : {output_dir}")
    trainer = train_dqn if variant.algorithm == "DQN" else train_ppo
    trainer(total_timesteps=scenario.total_timesteps, resume=resume, seed=train_seed,
            truck_count=scenario.truck_count, shovel_count=scenario.shovel_count,
            dump_count=scenario.dump_count, episode_minutes=scenario.episode_minutes,
            breakdown_probability=scenario.breakdown_probability,
            reward_weights=variant.reward_weights, output_dir=str(output_dir),
            agent_variant=variant,
            fuel_cost_scale_l=(retraining_config()["fuel_cost_scale_l"]
                               if variant.fuel_in_reward else None))


def _run_job(job: VariantJob, root: str) -> tuple[VariantJob, float]:
    """Entraîne une variante dans le processus courant ; sortie dans ``logs/<job>.log``."""
    log = Path(root) / campaign.LOG_DIR / f"{job.label()}.log"
    start = time.perf_counter()
    with open(log, "a", encoding="utf-8", buffering=1) as f, contextlib.redirect_stdout(f), \
            contextlib.redirect_stderr(f):
        print(f"=== {job.label()} — début {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        train_variant(load_variants()[job.variant], regime(), job.train_seed, job.directory(root))
        print(f"=== fin {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    return job, (time.perf_counter() - start) / 60.0


def train(root: str | Path, workers: int = 4,
          variants: Sequence[str] | None = None) -> list[VariantJob]:
    return campaign.train(root, workers, jobs=variant_jobs(variants), runner=_run_job)


# --- Évaluation -------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    """Un épisode : politique (heuristique, agent de la campagne ou variante) et graine."""

    policy: str
    source: str  # "heuristique", "campagne" ou "variante"
    train_seed: int | None
    model_dir: str | None
    seed: int


def check_variant_metadata(directory: Path, variant: AgentVariant) -> None:
    """Refuse un modèle dont les métadonnées ne décrivent pas exactement la variante."""
    meta = json.loads((directory / METADATA_FILENAME).read_text(encoding="utf-8"))
    recorded = meta.get("training", {}).get("agent_variant")
    if recorded != variant.as_dict():
        raise ValueError(f"{directory} : variante enregistrée {recorded}, attendue {variant.as_dict()}")
    if variant.fuel_in_reward and meta["training"].get("fuel_cost_scale_l") != \
            retraining_config()["fuel_cost_scale_l"]:
        raise ValueError(f"{directory} : échelle du carburant différente de la configuration")


_CACHE: dict[tuple[str, str | None], object] = {}


def _policy(task: Task, scenario: Scenario):  # noqa: ANN202
    key = (task.policy, task.model_dir)
    if key not in _CACHE:
        if task.source == "heuristique":
            _CACHE[key] = heuristic_policies(scenario)[task.policy]
        elif task.source == "campagne":
            model = load_model(Path(task.model_dir), task.policy, make_env(scenario),
                               trained_on=EnvConfig.from_scenario(scenario))
            _CACHE[key] = agent_policy_fn(model, task.policy)
        else:
            variant = load_variants()[task.policy]
            directory = Path(task.model_dir)
            check_variant_metadata(directory, variant)
            trained_on = EnvConfig.from_scenario(replace(scenario,
                                                         reward_weights=variant.reward_weights))
            model = load_model(directory, variant.algorithm,
                               wrap_env(make_env(scenario), variant), trained_on=trained_on)
            _CACHE[key] = variant_policy_fn(model, variant)
    return _CACHE[key]


def _evaluate(task: Task) -> dict:
    """Un épisode ; répartition des cycles par pelle lue dans le journal des cycles."""
    scenario = regime()
    env = make_env(scenario, log_cycles=True)
    kpis = run_single_episode(env, _policy(task, scenario), task.seed, task.policy)
    counts = Counter(r.shovel_id for r in env.cycle_log if r.kind == "cycle")
    total = max(sum(counts.values()), 1)
    shares = {f"share_S{i + 1}": 100.0 * counts.get(s.shovel_id, 0) / total
              for i, s in enumerate(env.shovels)}
    return {"policy": task.policy, "source": task.source, "train_seed": task.train_seed,
            "seed": task.seed, **{k: kpis[k] for k in (PRODUCTIVITY, FUEL, QUEUE, WAIT)},
            **shares}


def evaluation_tasks(root: Path, campaign_dir: Path,
                     seeds: Sequence[int] = EXTENSION_EVAL_SEEDS) -> list[Task]:
    name = regime().name
    policies = [(h, "heuristique", None, None) for h in HEURISTICS]
    policies += [(a, "campagne", s, str(model_dir(campaign_dir, a, name, s)))
                 for a in CAMPAIGN_AGENTS for s in TRAIN_SEEDS]
    policies += [(job.variant, "variante", job.train_seed, str(job.directory(root)))
                 for job in variant_jobs()]
    missing = [d for p, src, _, d in policies if d is not None
               and not model_file(d, p if src == "campagne" else load_variants()[p].algorithm).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} modèle(s) manquant(s) : {missing[:5]}…")
    return [Task(p, src, s, d, seed) for p, src, s, d in policies for seed in seeds]


def evaluate(root: str | Path, campaign_dir: str | Path, workers: int = 4,
             seeds: Sequence[int] = EXTENSION_EVAL_SEEDS) -> Path:
    """Évalue une seule fois sur les graines 72 à 101 ; refuse d'écraser un résultat."""
    root, out = Path(root), Path(root) / RESULTS_FILE
    if out.exists():
        raise FileExistsError(f"{out} existe déjà : l'évaluation ne se fait qu'une fois")
    tasks = evaluation_tasks(root, Path(campaign_dir), seeds)
    for task in {(t.policy, t.model_dir): t for t in tasks}.values():  # contrôle avant calcul
        _policy(task, regime())
    print(f"{len(tasks)} épisodes ({workers} processus)", flush=True)
    if workers == 1:
        rows = [_evaluate(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_evaluate, tasks, chunksize=max(1, len(tasks) // (4 * workers))))
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Écrit : {out}")
    return out


# --- Analyse ----------------------------------------------------------------------------

def reference_of(policy: str) -> str:
    """Référence de la campagne d'une politique : l'agent de même algorithme."""
    variants = load_variants()
    return variants[policy].algorithm if policy in variants else policy


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Par politique : moyennes sur les réplications (agents : moyenne de leurs modèles), et
    écarts de productivité et de carburant par tonne à la référence de la campagne et à MTWT,
    en %, avec IC bootstrap apparié sur les réplications."""
    per_seed = results.groupby(["policy", "seed"], sort=False)[list(METRICS)].mean()
    rows = []
    for policy in results["policy"].unique():
        data = per_seed.loc[policy]
        row = {"policy": policy, **{m: data[m].mean() for m in METRICS}}
        for ref_name, ref in (("ref", reference_of(policy)), ("mtwt", "MTWT")):
            if ref == policy:
                continue
            ref_data = per_seed.loc[ref]
            for metric in (PRODUCTIVITY, FUEL):
                diff = 100 * (data[metric] - ref_data[metric]) / ref_data[metric]
                low, high = paired_bootstrap_ci(diff.to_numpy())
                row |= {f"{metric}_vs_{ref_name}_pct": float(diff.mean()),
                        f"{metric}_vs_{ref_name}_ci_low": low,
                        f"{metric}_vs_{ref_name}_ci_high": high}
        rows.append(row)
    return pd.DataFrame(rows)


def per_model(results: pd.DataFrame) -> pd.DataFrame:
    agents = results[results["train_seed"].notna()]
    return agents.groupby(["policy", "train_seed"])[list(METRICS)].mean().reset_index()


def analyze(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    results = pd.read_csv(root / RESULTS_FILE)
    summary = summarise(results)
    summary.to_csv(root / SUMMARY_FILE, index=False)
    models = per_model(results)
    models.to_csv(root / PER_MODEL_FILE, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(summary.round(3).to_string(index=False))
        print()
        print(models.round(2).to_string(index=False))
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("root", type=Path, help="dossier des réentraînements")
        p.add_argument("--campaign", type=Path, required=name == "evaluate",
                       help="dossier de la campagne (modèles de référence)")
        p.add_argument("--workers", type=int, default=4)
        if name == "train":
            p.add_argument("--variants", nargs="*", default=None)
    p_an = sub.add_parser("analyze")
    p_an.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "train":
        train(args.root, args.workers, args.variants)
    elif args.command == "evaluate":
        evaluate(args.root, args.campaign, args.workers)
    else:
        analyze(args.root)


if __name__ == "__main__":
    main()
