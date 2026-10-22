"""Campagne finale (T31) en trois étapes, et sa répétition à petite échelle.

1. ``train`` : entraîne tous les modèles de la campagne, plusieurs à la fois. Un modèle
   par (algorithme, régime d'entraînement, graine d'entraînement) : les quatre agents
   dans les trois régimes principaux, PPO et DQN dans les régimes d'ablation sur w4
   (``Scenario.algorithms``), avec les 5 graines d'entraînement. Un modèle déjà entraîné
   est sauté, un entraînement interrompu reprend à sa dernière sauvegarde
   (``rl/checkpoints.py``) : relancer la commande ne refait que ce qui manque. Chaque
   entraînement écrit son journal dans ``<dossier>/logs/``.
2. ``evaluate`` : évalue une seule fois toutes les politiques sur les graines
   d'évaluation, dans tous les régimes (principaux, non vus, ablation). Refuse de
   démarrer s'il manque un modèle : l'évaluation ne doit jamais entraîner.
3. ``analyze`` : agrégation sur les graines d'entraînement, tests appariés, coût du
   transfert, ablation, courbes d'apprentissage ; une sortie par analyse dans
   ``<dossier>/analyses/``.

Pourquoi un lanceur dédié : ``run_benchmark`` entraîne les modèles manquants l'un après
l'autre, dans un seul processus (environ 13 jours pour 90 entraînements). Lancé dans
plusieurs fenêtres, il pourrait entraîner deux fois le même modèle nominal, dont les
régimes perturbés ont besoin pour le transfert.

``rehearse`` enchaîne les trois étapes avec une configuration réduite (budgets
minuscules, graines d'évaluation DISJOINTES de 42 à 71), dans des processus séparés
qui lisent cette configuration par ``MINE_RL_CONFIG_DIR`` : toute la chaîne est
vérifiée sans toucher aux graines d'évaluation de la campagne.

    python -m memoire_master_rl_logistique.experiments.campaign train DOSSIER --workers 4
    python -m memoire_master_rl_logistique.experiments.campaign evaluate DOSSIER --workers 4
    python -m memoire_master_rl_logistique.experiments.campaign analyze DOSSIER
    python -m memoire_master_rl_logistique.experiments.campaign rehearse DOSSIER --workers 4
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
from dataclasses import dataclass
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

from memoire_master_rl_logistique.config import (
    CONFIG_DIR_ENV,
    CONFIG_FILES,
    EXPERIMENT_FILE,
    config_dir,
    snapshot_configs,
)
from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS, SCENARIOS
from memoire_master_rl_logistique.rl.model_loading import model_file
from memoire_master_rl_logistique.rl.model_paths import model_dir
from memoire_master_rl_logistique.seeds import EVAL_SEEDS, TRAIN_SEEDS

ABLATION_SCENARIOS = [name for name in SCENARIOS if name.startswith("ablation_w4")]
UNSEEN_SCENARIOS = [name for name in SCENARIOS if name.startswith("unseen_")]
# Régimes évalués par la campagne ; la configuration asymétrique n'en fait pas partie.
EVALUATED_SCENARIOS = [*MAIN_SCENARIOS, *UNSEEN_SCENARIOS, *ABLATION_SCENARIOS]
# Régimes dans lesquels des modèles sont entraînés (les régimes non vus utilisent les
# modèles nominaux).
TRAINING_REGIMES = [*MAIN_SCENARIOS, *ABLATION_SCENARIOS]
# Durée relative d'un entraînement (mesures du 21 octobre 2026) : les plus longs d'abord,
# pour que les derniers processus ne finissent pas seuls.
TRAINING_ORDER = {"Q-Learning": 0, "SARSA": 1, "PPO": 2, "DQN": 3}

MODULE = "memoire_master_rl_logistique.experiments.campaign"
LOG_DIR = "logs"
ANALYSES_DIR = "analyses"
REHEARSAL_CONFIG_DIR = "configs_repetition"
# Configuration de la répétition : budgets minuscules, et graines d'évaluation qui ne
# recoupent ni 42-71 (évaluation de la campagne), ni la sélection, ni l'entraînement.
REHEARSAL_TIMESTEPS = 4_096
REHEARSAL_EPISODES = 40
REHEARSAL_EVAL_START = 30_000


@dataclass(frozen=True, order=True)
class TrainJob:
    """Un entraînement de la campagne."""

    algorithm: str
    regime: str
    train_seed: int

    def directory(self, root: str | Path) -> Path:
        return model_dir(root, self.algorithm, self.regime, self.train_seed)

    def done(self, root: str | Path) -> bool:
        return model_file(self.directory(root), self.algorithm).exists()

    def label(self) -> str:
        return f"{self.algorithm}_{self.regime}_seed{self.train_seed}"


def campaign_jobs(regimes: Sequence[str] = TRAINING_REGIMES,
                  train_seeds: Sequence[int] = TRAIN_SEEDS) -> list[TrainJob]:
    """Tous les entraînements, les plus longs d'abord (algorithme), puis par régime et graine."""
    jobs = [TrainJob(algorithm, regime, seed) for regime in regimes
            for algorithm in SCENARIOS[regime].algorithms for seed in train_seeds]
    return sorted(jobs, key=lambda j: (TRAINING_ORDER[j.algorithm], regimes.index(j.regime),
                                       j.train_seed))


def _run_job(job: TrainJob, root: str) -> tuple[TrainJob, float]:
    """Entraîne un modèle dans le processus courant ; sortie dans ``logs/<job>.log``."""
    from memoire_master_rl_logistique.experiments.run_benchmark import _train

    log = Path(root) / LOG_DIR / f"{job.label()}.log"
    start = time.perf_counter()
    with open(log, "a", encoding="utf-8") as f, contextlib.redirect_stdout(f), \
            contextlib.redirect_stderr(f):
        print(f"=== {job.label()} — début {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        _train(job.algorithm, SCENARIOS[job.regime], job.train_seed, job.directory(root))
        print(f"=== fin {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    return job, (time.perf_counter() - start) / 60.0


def train(root: str | Path, workers: int = 4, jobs: Sequence[TrainJob] | None = None,
          runner: Callable[[TrainJob, str], tuple[TrainJob, float]] = _run_job) -> list[TrainJob]:
    """Entraîne les modèles manquants, ``workers`` à la fois ; renvoie ceux entraînés.

    La configuration est archivée avant le calcul (``snapshot_configs``) : une relance
    avec une configuration modifiée lève ``FileExistsError``.
    """
    root = Path(root)
    jobs = campaign_jobs() if jobs is None else list(jobs)
    (root / LOG_DIR).mkdir(parents=True, exist_ok=True)
    snapshot_configs(root / "configs_campagne")
    todo = [job for job in jobs if not job.done(root)]
    print(f"{len(jobs)} entraînements prévus, {len(jobs) - len(todo)} déjà faits, "
          f"{len(todo)} à faire ({workers} à la fois). Journaux : {root / LOG_DIR}", flush=True)
    if not todo:
        return []
    failures = []

    def report(n: int, job: TrainJob, outcome: Callable[[], tuple[TrainJob, float]]) -> None:
        try:
            _, minutes = outcome()
            print(f"[{n}/{len(todo)}] {job.label()} : {minutes:.1f} min", flush=True)
        except Exception as error:  # un échec n'arrête pas les autres entraînements
            failures.append(job)
            print(f"[{n}/{len(todo)}] ÉCHEC {job.label()} : {error!r}", flush=True)

    if workers == 1:  # dans le processus courant (tests, machine peu chargée)
        for n, job in enumerate(todo, start=1):
            report(n, job, lambda job=job: runner(job, str(root)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(runner, job, str(root)): job for job in todo}
            for n, future in enumerate(as_completed(futures), start=1):
                report(n, futures[future], future.result)
    if failures:
        raise RuntimeError(f"{len(failures)} entraînement(s) en échec : "
                           f"{[j.label() for j in failures]} (journaux dans {root / LOG_DIR})")
    return todo


def missing_models(root: str | Path, jobs: Sequence[TrainJob] | None = None) -> list[TrainJob]:
    return [job for job in (campaign_jobs() if jobs is None else jobs) if not job.done(root)]


def evaluate(root: str | Path, workers: int = 4,
             scenarios: Sequence[str] = EVALUATED_SCENARIOS) -> Path:
    """Évalue toutes les politiques une seule fois ; refuse s'il manque un modèle."""
    from memoire_master_rl_logistique.experiments.run_benchmark import run_all_benchmarks

    missing = missing_models(root)
    if missing:
        raise FileNotFoundError(f"{len(missing)} modèle(s) manquant(s), lancer d'abord "
                                f"« train » : {[j.label() for j in missing][:5]}…")
    return run_all_benchmarks(str(root), scenario_names=list(scenarios), n_workers=workers)


def results_csv(root: str | Path) -> Path:
    """Le fichier de résultats de l'évaluation ; il doit être unique."""
    csvs = sorted(Path(root).glob("benchmark_*.csv"))
    if len(csvs) != 1:
        raise FileNotFoundError(f"Un seul benchmark_*.csv attendu dans {root}, trouvé : {csvs}")
    return csvs[0]


def _capture(name: str, function: Callable[[], object], out_dir: Path) -> None:
    """Exécute une analyse et enregistre ce qu'elle affiche dans ``<out_dir>/<name>.txt``."""
    target = out_dir / f"{name}.txt"
    if target.exists():
        raise FileExistsError(f"Analyse déjà présente : {target}")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        function()
    target.write_text(buffer.getvalue(), encoding="utf-8")
    print(f"  {name} : {target}", flush=True)


def analyze(root: str | Path) -> Path:
    """Toutes les analyses de l'article, à partir du fichier de résultats unique."""
    from memoire_master_rl_logistique.analysis import (
        aggregate_seeds,
        learning_curves,
        statistical_tests,
        transfer,
    )
    from memoire_master_rl_logistique.experiments.ablation_study import ablation_table

    root = Path(root)
    csv_path = results_csv(root)
    out_dir = root / ANALYSES_DIR
    out_dir.mkdir(exist_ok=True)
    print(f"Analyses de {csv_path} :", flush=True)
    _capture("agregation_graines", lambda: aggregate_seeds.main(csv_path, len(EVAL_SEEDS)), out_dir)
    _capture("tests_apparies", lambda: statistical_tests.main(csv_path), out_dir)
    _capture("transfert", lambda: transfer.main(csv_path), out_dir)
    _capture("ablation_w4", lambda: print(ablation_table([csv_path]).to_string()), out_dir)
    for curve in ("productivity_tph",):
        learning_curves.main([str(root), str(out_dir / f"courbes_{curve}.csv"), "--curve", curve])
    return out_dir


# --- Répétition -------------------------------------------------------------------------

def rehearsal_config(destination: str | Path, source: str | Path | None = None) -> Path:
    """Copie de la configuration avec des budgets minuscules et des graines d'évaluation
    disjointes de celles de la campagne. Les hyperparamètres ne changent pas."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Configuration de répétition déjà présente : {destination}")
    destination.mkdir(parents=True)
    source = Path(source or config_dir())
    for name in CONFIG_FILES:
        shutil.copy(source / name, destination / name)

    def edit(name: str, change: Callable[[dict], None]) -> None:
        path = destination / name
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
        change(content)
        path.write_text(yaml.safe_dump(content, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")

    def experiment(c: dict) -> None:
        c["scenario_defaults"]["total_timesteps"] = REHEARSAL_TIMESTEPS
        c["seeds"]["evaluation"]["start"] = REHEARSAL_EVAL_START

    edit(EXPERIMENT_FILE, experiment)
    edit("tabular_hyperparameters.yaml", lambda c: c.update(n_episodes=REHEARSAL_EPISODES))
    return destination


def rehearse(root: str | Path, workers: int = 4) -> None:
    """Les trois étapes, avec la configuration réduite, chacune dans un processus séparé
    (les modules lisent la configuration à l'import)."""
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"La répétition demande un dossier vide : {root}")
    configuration = rehearsal_config(root / REHEARSAL_CONFIG_DIR)
    env = {**os.environ, CONFIG_DIR_ENV: str(configuration)}
    for step in ("train", "evaluate", "analyze"):
        command = [sys.executable, "-m", MODULE, step, str(root)]
        if step != "analyze":
            command += ["--workers", str(workers)]
        print(f"\n##### Répétition : {step}", flush=True)
        start = time.perf_counter()
        subprocess.run(command, env=env, check=True)
        print(f"##### {step} : {(time.perf_counter() - start) / 60:.1f} min", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Campagne finale (T31) et sa répétition.")
    parser.add_argument("step", choices=["train", "evaluate", "analyze", "rehearse"])
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.step == "train":
        train(args.output_dir, args.workers)
    elif args.step == "evaluate":
        print(f"Résultats : {evaluate(args.output_dir, args.workers)}")
    elif args.step == "analyze":
        print(f"Analyses : {analyze(args.output_dir)}")
    else:
        rehearse(args.output_dir, args.workers)


if __name__ == "__main__":
    main()
