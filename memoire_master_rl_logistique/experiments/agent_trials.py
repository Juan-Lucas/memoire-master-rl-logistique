"""Essais de la mise au point des agents (16 au 19 octobre 2026), pour pouvoir les refaire.

Après les corrections du simulateur (T15, T16 bis), les agents ont été mis au point avec
une règle écrite avant les résultats : voir « Mise au point des agents » dans
``docs/journal_corrections.md``. Cette règle exige de **déclarer tous les essais**.
Ce module fixe donc ce que chaque essai a entraîné et comment il a été évalué, pour
qu'on puisse le refaire.

Conditions communes : régime nominal, graine d'entraînement ``TRAIN_SEEDS[0]`` (1),
critère = productivité de la politique sans exploration sur les **graines de sélection**
(20 000 à 20 009), jamais sur celles de l'évaluation finale (42 à 71).

Les essais 5 à 9 et 14 à 17 se refont sur le code actuel (``TRIALS``). Seuls changent les
hyperparamètres que chaque essai fixe explicitement ; le reste vient de ``configs/``. Les
essais 5 à 9 précisent l'actualisation par décision (``PER_DECISION``), celle d'avant le
21 octobre 2026 ; les essais 14 à 17 contrôlent l'actualisation par minute.
Les essais 10 à 13 (contrôle de l'option B, 20 et 21 octobre 2026) ont utilisé une
récompense abandonnée depuis : ils se refont depuis le commit ``78ed5f5``
(``Trial.commit`` ; ``run_trial`` refuse de les lancer sur le code actuel, qui donnerait
un autre résultat). Les essais 0 à 4 utilisaient une version antérieure du simulateur (observation de
17 composantes) et, pour 1, 2 et 4, une récompense d'essai jamais commitée. Ils se refont
à partir du commit ``157f97b`` : voir ``archives/mise_au_point_agents/README.md`` et
``ARCHIVED_TRIALS``.

Pour chaque essai, ``run_trial`` écrit dans ``<dossier>/essai_<n>/`` le modèle et
``trial_result.json`` :

- durée d'entraînement ;
- productivité pendant l'entraînement, moyenne par dixième du budget ;
- productivité sur les graines de sélection (moyenne et écart-type) ;
- part des décisions ATTENDRE ;
- part des cycles par pelle.

Il affiche aussi les lignes ``DUREE``, ``COURBE`` et ``RESULT`` des scripts d'origine.
Un essai interrompu reprend à sa dernière sauvegarde (``rl/checkpoints.py``).

Lancement (environ 2 h par essai au budget complet) :
    python -m memoire_master_rl_logistique.experiments.agent_trials 6 DOSSIER
    python -m memoire_master_rl_logistique.experiments.agent_trials --list
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from memoire_master_rl_logistique.experiments.policies import (
    PolicyFn,
    agent_policy_fn,
    make_env,
)
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS, Scenario
from memoire_master_rl_logistique.rl.checkpoints import STATE_FILE, checkpoint_dir
from memoire_master_rl_logistique.rl.model_loading import load_model, model_file
from memoire_master_rl_logistique.rl.train_dqn import train_dqn
from memoire_master_rl_logistique.rl.train_ppo import train_ppo
from memoire_master_rl_logistique.rl.train_q_learning import train_q_learning
from memoire_master_rl_logistique.rl.train_sarsa import train_sarsa
from memoire_master_rl_logistique.seeds import TRAIN_SEEDS, TUNING_SEEDS

RESULT_FILE = "trial_result.json"
CURVE_BINS = 10  # la courbe d'entraînement est résumée par dixième du budget
SHOVEL_IDS = ("S1", "S2", "S3")

TRAINERS: dict[str, Callable[..., Any]] = {
    "PPO": train_ppo, "DQN": train_dqn, "Q-Learning": train_q_learning, "SARSA": train_sarsa,
}


@dataclass(frozen=True)
class Trial:
    """Un essai déclaré : ce qui est entraîné, et le résultat noté dans le journal."""

    number: int
    algorithm: str
    description: str
    reported: str  # résultat noté dans le journal (graines de sélection)
    overrides: dict[str, Any] = field(default_factory=dict)  # écarts à configs/
    commit: str | None = None  # code d'un essai qui ne se refait plus sur le code actuel


OPTION_B = "78ed5f5"  # commit de l'option B (coût d'opportunité de chaque cycle), abandonnée
PER_DECISION = {"time_discount": False}  # essais 0 à 13 : actualisation par décision
PENDING = "à venir"


TRIALS: dict[int, Trial] = {trial.number: trial for trial in (
    Trial(5, "PPO", "Piste 1 : attente prévue par route dans l'observation (23 composantes), "
          "sans la piste 2 (γ = 0,99, récompense brute)",
          "4 345,2 ± 66 t/h, ATTENDRE 1,7 %, pelles 64 / 24 / 12 %",
          {"ent_coef": 0.0, "learning_rate": 1e-4, "gamma": 0.99, "normalize_reward": False,
           **PER_DECISION}),
    Trial(6, "PPO", "Piste 2 : piste 1 + récompense normalisée + γ = 0,995 "
          "(= configuration de la campagne, actualisation par décision)",
          "4 404,8 ± 50,2 t/h, ATTENDRE 36 %, pelles 62 / 33 / 5 %", PER_DECISION),
    Trial(7, "DQN", "Contrôle de DQN au budget complet : piste 1 + γ = 0,995, sans "
          "normalisation (= configuration de la campagne, actualisation par décision)",
          "3 592,8 ± 59,5 t/h, ATTENDRE 0 %, pelles 0 / 1 / 99 %", PER_DECISION),
    Trial(8, "Q-Learning", "Chronométrage et point de départ du réglage (α = 0,2, "
          "8 intervalles, = configuration de la campagne, actualisation par décision)",
          "3 955,0 ± 36,9 t/h, ATTENDRE 1,9 %, pelles 1 / 36 / 62 %", PER_DECISION),
    Trial(9, "SARSA", "Chronométrage et point de départ du réglage (α = 0,2, "
          "8 intervalles, = configuration de la campagne, actualisation par décision)",
          "3 535,0 ± 42,9 t/h, ATTENDRE 0 %, pelles 1 / 1 / 98 %", PER_DECISION),
    # Contrôle avec la récompense de l'option B (règle, points 13 et 14) : configuration de
    # la campagne, rien d'autre ne change. Option B abandonnée ensuite (PPO apprend à ATTENDRE).
    Trial(10, "PPO", "Contrôle option B : PPO, configuration de la campagne",
          "210,0 ± 0,0 t/h, ATTENDRE 99,8 %, pelles 49 / 0 / 51 %", commit=OPTION_B),
    Trial(11, "DQN", "Contrôle option B : DQN, configuration de la campagne",
          "3 531,5 ± 42,7 t/h, ATTENDRE 94,8 %, pelles 33 / 36 / 31 %", commit=OPTION_B),
    Trial(12, "Q-Learning", "Contrôle option B : Q-Learning, configuration de la campagne "
          "(α = 0,2, 8 intervalles)",
          "4 098,5 ± 42,7 t/h, ATTENDRE 42,5 %, pelles 35 / 33 / 33 %", commit=OPTION_B),
    Trial(13, "SARSA", "Contrôle option B : SARSA, configuration de la campagne "
          "(α = 0,2, 8 intervalles)",
          "4 133,5 ± 55,2 t/h, ATTENDRE 37,2 %, pelles 35 / 38 / 27 %", commit=OPTION_B),
    # Contrôle de l'actualisation par minute (règle, points 19 à 23) : configuration de la
    # campagne (time_discount: true), récompense des essais 0 à 9.
    Trial(14, "PPO", "Actualisation par minute : PPO, configuration de la campagne", PENDING),
    Trial(15, "DQN", "Actualisation par minute : DQN, configuration de la campagne", PENDING),
    Trial(16, "Q-Learning", "Actualisation par minute : Q-Learning, configuration de la "
          "campagne (α = 0,2, 8 intervalles)", PENDING),
    Trial(17, "SARSA", "Actualisation par minute : SARSA, configuration de la campagne "
          "(α = 0,2, 8 intervalles)", PENDING),
)}

# Essais faits sur une version antérieure du code (voir archives/mise_au_point_agents/).
ARCHIVED_TRIALS: dict[int, str] = {
    0: "Réglage T25 bis (200 000 pas) : experiments/tune_hyperparams.py au commit 157f97b",
    1: "Récompense à la décision (1b), 200 000 pas : 157f97b + patch, reward_timing_test.py",
    2: "1b + coût des minutes en file, 200 000 pas : 157f97b + patch, reward_queue_test.py",
    3: "PPO au budget complet, récompense en fin de cycle (1a) : 157f97b + patch, "
       "diag_full.py cycle_end",
    4: "PPO au budget complet, récompense à la décision (1b) : 157f97b + patch, "
       "diag_full.py decision",
}


def training_curve(out: Path, algorithm: str, budget: int, bins: int = CURVE_BINS) -> list[int]:
    """Productivité pendant l'entraînement (t/h), moyenne par fraction du budget.

    Axe : pas écoulés à la fin de chaque épisode (PPO, DQN, entre 1 et ``budget``) ou
    numéro d'épisode (Q-Learning, SARSA, à partir de 0). Les scripts d'origine des essais
    7 à 9 rangeaient les épisodes de PPO et DQN avec ``pas × 10 // budget`` au lieu de
    ``(pas − 1) × 10 // budget`` : seul un épisode finissant pile sur une limite change
    d'intervalle.
    """
    if algorithm in ("PPO", "DQN"):
        log = pd.read_csv(out / "training_kpis.csv")
        position = log["timesteps"] - 1
    else:
        log = pd.read_csv(out / "training_diagnostics.csv")
        position = log["episode"]
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    hours = meta["env"]["episode_minutes"] / 60.0
    index = np.minimum(position * bins // budget, bins - 1)
    return (log["tonnage_t"] / hours).groupby(index).mean().round(0).astype(int).tolist()


def selection_metrics(
    policy: PolicyFn, scenario: Scenario, seeds: Sequence[int] = TUNING_SEEDS,
) -> dict[str, Any]:
    """Productivité, part d'ATTENDRE et part des cycles par pelle sur ``seeds``."""
    tph, waits, shares = [], [], Counter()
    for seed in seeds:
        env = make_env(scenario, log_cycles=True)
        kpis = run_single_episode(env, policy, seed)
        tph.append(kpis["productivity_tph"])
        waits.append(kpis["wait_share_pct"])
        shares.update(c.shovel_id for c in env.cycle_log if c.kind == "cycle")
    total = max(sum(shares.values()), 1)
    return {
        "productivity_mean": float(np.mean(tph)),
        "productivity_std": float(np.std(tph, ddof=1)) if len(tph) > 1 else 0.0,
        "wait_share_pct": float(np.mean(waits)),
        "cycle_share_pct": {sid: 100.0 * shares[sid] / total for sid in SHOVEL_IDS},
        "seeds": list(seeds),
    }


def run_trial(
    number: int,
    root: str | Path,
    budget: int | None = None,
    scenario: Scenario | None = None,
    seeds: Sequence[int] = TUNING_SEEDS,
) -> dict[str, Any]:
    """Entraîne (ou reprend) et évalue l'essai ``number`` ; écrit ``trial_result.json``.

    ``budget`` remplace le budget de l'essai (pas pour PPO et DQN, épisodes pour les agents
    tabulaires) et ``scenario`` le régime nominal : réservés aux tests, le résultat n'est
    alors plus celui de l'essai déclaré. Un modèle déjà entraîné n'est pas réentraîné.
    """
    trial = TRIALS[number]
    if trial.commit:
        raise ValueError(
            f"L'essai {number} a été fait avec le code du commit {trial.commit} (récompense "
            "de l'option B, abandonnée) : le refaire depuis ce commit, pas sur le code actuel."
        )
    scenario = scenario or SCENARIOS["nominal"]
    out = Path(root) / f"essai_{number}"
    deep = trial.algorithm in ("PPO", "DQN")
    budget = budget or (scenario.total_timesteps if deep else None)

    if not model_file(out, trial.algorithm).exists():
        kwargs = dict(
            seed=TRAIN_SEEDS[0], output_dir=str(out), truck_count=scenario.truck_count,
            shovel_count=scenario.shovel_count, dump_count=scenario.dump_count,
            episode_minutes=scenario.episode_minutes,
            breakdown_probability=scenario.breakdown_probability,
            reward_weights=scenario.reward_weights,
            resume=(checkpoint_dir(out) / STATE_FILE).exists(),
            **trial.overrides,
        )
        if deep:
            kwargs.update(total_timesteps=budget, progress_bar=False)
        elif budget:
            kwargs.update(n_episodes=budget)
        TRAINERS[trial.algorithm](**kwargs)

    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    training = meta["training"]
    budget = training["total_timesteps"] if deep else training["n_episodes"]
    policy = agent_policy_fn(load_model(out, trial.algorithm, make_env(scenario)),
                             trial.algorithm)
    start = time.perf_counter()
    metrics = selection_metrics(policy, scenario, seeds)
    result = {
        "trial": number,
        "algorithm": trial.algorithm,
        "description": trial.description,
        "reported_in_journal": trial.reported,
        "budget": budget,
        "budget_unit": "pas" if deep else "épisodes",
        "training_time_min": meta["training_time_s"] / 60.0,
        "training_curve_tph": training_curve(out, trial.algorithm, budget),
        "resumed": bool(training.get("resumed_from_timesteps")
                        or training.get("resumed_from_episodes")),
        **metrics,
        "evaluation_time_s": time.perf_counter() - start,
    }
    (out / RESULT_FILE).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    print_summary(result)
    return result


def print_summary(result: dict[str, Any]) -> None:
    """Les trois lignes des scripts d'origine (DUREE, COURBE, RESULT)."""
    algo, shares = result["algorithm"], result["cycle_share_pct"]
    print(f"DUREE {algo} : {result['training_time_min']:.1f} min d'entraînement", flush=True)
    print(f"COURBE {algo} (t/h par dixième du budget) : {result['training_curve_tph']}",
          flush=True)
    print(f"RESULT {algo} : {result['productivity_mean']:.1f} ± "
          f"{result['productivity_std']:.1f} t/h, ATTENDRE {result['wait_share_pct']:.1f} %, "
          "pelles " + " ".join(f"{sid} {shares[sid]:.0f}%" for sid in SHOVEL_IDS), flush=True)
    print(f"(noté dans le journal : {result['reported_in_journal']})", flush=True)


def list_trials() -> str:
    """Tous les essais déclarés, refaisables ou archivés."""
    lines = [f"{n} [archivé] {text}" for n, text in ARCHIVED_TRIALS.items()]
    lines += [f"{t.number} [{t.algorithm}{', commit ' + t.commit if t.commit else ''}] "
              f"{t.description} — journal : {t.reported}" for t in TRIALS.values()]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refaire un essai de la mise au point des agents.")
    parser.add_argument("trial", type=int, nargs="?", choices=sorted(TRIALS))
    parser.add_argument("out_dir", type=Path, nargs="?", help="dossier des essais")
    parser.add_argument("--list", action="store_true", help="lister les essais déclarés")
    args = parser.parse_args()
    if args.list or args.trial is None:
        print(list_trials())
    elif args.out_dir is None:
        parser.error("préciser le dossier de sortie")
    else:
        run_trial(args.trial, args.out_dir)
