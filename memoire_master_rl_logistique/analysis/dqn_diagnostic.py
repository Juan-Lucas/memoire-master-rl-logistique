"""Diagnostic de l'effondrement de DQN sur ATTENDRE (M6 du rapport du 17/11/2026).

Hypothèse de l'article : avec l'actualisation par minute, la cible d'ATTENDRE est
r_ATT + γ_min^Δt · max Q(s′, ·) avec Δt ≥ 0,2 min et s′ proche de s, donc Q(s, ATTENDRE)
est presque égale à la valeur de la meilleure affectation ; l'écart devient plus petit que
l'erreur d'approximation du réseau, et ATTENDRE peut passer en tête.

Ce module le **mesure** sur les modèles existants, sans entraînement et sans graine
d'évaluation : pour chaque modèle DQN, sur les états rencontrés par MTWT (situations
d'exploitation normales, les mêmes pour tous les modèles) et sur ceux que le modèle
rencontre lui-même (graines de sélection), il relève

- ``gap`` = Q(s, ATTENDRE) − max_{a ≠ ATTENDRE} Q(s, a) ;
- la part des états où ATTENDRE est l'action de plus grande valeur ;
- l'échelle des valeurs, max_a Q(s, a), et l'écart relatif ``gap`` / échelle ;
- l'écart attendu pour une attente de 0,2 min si Q(s′, ·) = Q(s, ·) :
  −(1 − γ_min^0,2) · max Q − coût d'ATTENDRE.

    python -m memoire_master_rl_logistique.analysis.dqn_diagnostic DOSSIER_SORTIE \\
        --models data/results_v2/campagne
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th

from memoire_master_rl_logistique.experiments.policies import heuristic_policies, make_env
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS
from memoire_master_rl_logistique.rl.model_loading import load_model
from memoire_master_rl_logistique.rl.model_paths import model_dir
from memoire_master_rl_logistique.rl.time_discount import per_minute_gamma
from memoire_master_rl_logistique.seeds import TRAIN_SEEDS, TUNING_SEEDS
from memoire_master_rl_logistique.simulation.reward import RewardWeights

STATES_FILE = "dqn_q_values.csv"
SUMMARY_FILE = "dqn_diagnostic_summary.csv"
SHORT_WAIT_MIN = 0.2
OPPORTUNITY_CYCLE_MIN = 23.2  # coût d'opportunité d'ATTENDRE : w1 × durée / 23,2 min


def q_values(model, obs: np.ndarray) -> np.ndarray:
    """Valeurs Q du réseau en ligne pour une observation."""
    with th.no_grad():
        tensor, _ = model.policy.obs_to_tensor(obs)
        return model.q_net(tensor).cpu().numpy().ravel()


def collect_states(env_factory: Callable, policy_fn: Callable, seeds: Sequence[int]) -> list[np.ndarray]:
    """Observations des décisions d'une politique sur ``seeds``."""
    states = []
    for seed in seeds:
        env = env_factory()
        obs, info = env.reset(seed=seed)
        while True:
            states.append(obs.copy())
            obs, _, terminated, truncated, info = env.step(policy_fn(obs, info, env))
            if terminated or truncated:
                break
    return states


def wait_cost(weights: RewardWeights, minutes: float = SHORT_WAIT_MIN) -> float:
    """Coût immédiat d'une ATTENDRE de ``minutes`` : pénalité d'attente et coût d'opportunité."""
    return weights.wait * min(minutes / 30.0, 1.0) + weights.production * minutes / OPPORTUNITY_CYCLE_MIN


def diagnose(model, states: Sequence[np.ndarray], wait_action: int, gamma: float,
             weights: RewardWeights) -> pd.DataFrame:
    """Une ligne par état : écart d'ATTENDRE à la meilleure affectation et écart attendu."""
    g02 = per_minute_gamma(gamma) ** SHORT_WAIT_MIN
    rows = []
    for obs in states:
        q = q_values(model, obs)
        best_other = float(np.max(np.delete(q, wait_action)))
        scale = float(np.max(q))
        rows.append({"q_wait": float(q[wait_action]), "q_best_assignment": best_other,
                     "gap": float(q[wait_action]) - best_other, "scale": scale,
                     "wait_first": bool(np.argmax(q) == wait_action),
                     "expected_gap": -(1 - g02) * scale - wait_cost(weights)})
    return pd.DataFrame(rows)


def summarise(states: pd.DataFrame) -> pd.DataFrame:
    keys = ["train_seed", "states_of"]
    g = states.groupby(keys)
    out = pd.DataFrame({
        "n_states": g.size(),
        "wait_first_share": g["wait_first"].mean(),
        "median_gap": g["gap"].median(),
        "median_abs_gap": g["gap"].apply(lambda x: x.abs().median()),
        "median_scale": g["scale"].median(),
        "median_relative_gap_pct": g.apply(lambda d: 100 * (d["gap"] / d["scale"].abs()).median(),
                                           include_groups=False),
        "median_expected_gap": g["expected_gap"].median(),
    })
    return out.reset_index()


def run(out_dir: Path, models_dir: Path, regime: str = "nominal",
        train_seeds: Sequence[int] = TRAIN_SEEDS, seeds: Sequence[int] = TUNING_SEEDS[:5]) -> pd.DataFrame:
    scenario = SCENARIOS[regime]
    env = make_env(scenario)
    wait_action = env.action_space.n - 1
    weights = RewardWeights.from_tuple(scenario.reward_weights)
    reference = collect_states(lambda: make_env(scenario), heuristic_policies(scenario)["MTWT"], seeds)
    frames = []
    for train_seed in train_seeds:
        model = load_model(model_dir(models_dir, "DQN", regime, train_seed), "DQN", env)
        own = collect_states(lambda: make_env(scenario),
                             lambda obs, info, e: int(model.predict(obs, deterministic=True)[0]), seeds)
        for label, states in (("MTWT", reference), ("own", own)):
            d = diagnose(model, states, wait_action, model.gamma, weights)
            d.insert(0, "states_of", label)
            d.insert(0, "train_seed", train_seed)
            frames.append(d)
    data = pd.concat(frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_dir / STATES_FILE, index=False)
    summary = summarise(data)
    summary.to_csv(out_dir / SUMMARY_FILE, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.round(4).to_string(index=False))
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.out_dir, args.models)


if __name__ == "__main__":
    main()
