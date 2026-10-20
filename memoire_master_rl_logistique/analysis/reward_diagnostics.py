"""Mesures de l'analyse de l'échec des agents à valeurs (19 et 20 octobre 2026).

Voir « Analyse de l'échec de DQN », « Résultat des agents tabulaires » et « Option B sur
le papier » dans ``docs/journal_corrections.md``. Trois mesures, en lecture seule, sans
réentraîner :

1. **La récompense classe-t-elle les politiques comme la productivité ?**
   (``reward_by_policy``) Pour des politiques fixées, on mesure :
   - la productivité ;
   - le nombre de décisions ;
   - la récompense par décision, au total et terme par terme (production, équilibre,
     distance, attente, opportunité).

   Résultat noté dans le journal : aller toujours à la pelle 3 plutôt qu'à la pelle 1
   coûte 9 % de productivité, mais ne change presque pas la récompense par décision.

2. **Le réseau de DQN distingue-t-il les actions ?** (``q_value_gaps``) Sur un poste
   joué par l'agent, on mesure :
   - la valeur Q moyenne ;
   - l'écart entre les deux meilleures actions ;
   - l'écart entre la meilleure et la pire ;
   - les actions choisies.

   Résultat noté : Q ≈ 174 pour toutes les actions, et un écart moyen de 0,05 entre les
   deux meilleures.

3. **Avec l'option B, le classement suit-il la productivité ?** (``option_b_rankings``)
   Douze politiques fixées (les heuristiques et les 6 routes constantes), trois régimes.
   Corrélation de Spearman entre la productivité et la récompense par décision, avant et
   après l'option B (``docs/option_b_recompense.md``).

Depuis l'option B, ``reward_per_decision`` est la récompense actuelle. La récompense d'avant
s'en déduit : on rajoute le coût d'opportunité des cycles, calculé à partir du journal des
cycles (``reward_per_decision_before_b``). Les mesures du 19 octobre se refont donc sur le
code actuel.

Lancement (quelques minutes) :
    python -m memoire_master_rl_logistique.analysis.reward_diagnostics DOSSIER
        [--dqn-model DOSSIER_DU_MODELE_DQN]
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from memoire_master_rl_logistique.experiments.policies import (
    PolicyFn,
    heuristic_policies,
    make_env,
)
from memoire_master_rl_logistique.experiments.run_benchmark import run_single_episode
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS, Scenario
from memoire_master_rl_logistique.seeds import TUNING_SEEDS
from memoire_master_rl_logistique.simulation.reward import (
    OPPORTUNITY_CYCLE_MIN,
    RewardWeights,
)

# Politiques de la mesure du journal : les deux meilleures heuristiques, l'Affectation
# fixe, la Pelle la plus proche, et des actions constantes (0 = pelle 1 → verse 1,
# 2 = pelle 2 → verse 1, 4 = pelle 3 → verse 1, le choix de DQN et de SARSA).
REFERENCE_HEURISTICS = ("MTWT", "LP", "Fixed", "Nearest")
CONSTANT_ACTIONS = (0, 2, 4)
DIAGNOSTIC_SEEDS = TUNING_SEEDS[:5]  # graines de sélection, jamais celles de l'évaluation


def constant_action_policy(action: int) -> PolicyFn:
    """Politique qui choisit toujours ``action``."""
    def policy(obs, info, env):
        return action
    return policy


def reference_policies(scenario: Scenario) -> dict[str, PolicyFn]:
    """Les politiques de la mesure du journal."""
    heuristics = heuristic_policies(scenario)
    policies = {name: heuristics[name] for name in REFERENCE_HEURISTICS}
    policies.update({f"Toujours action {a}": constant_action_policy(a) for a in CONSTANT_ACTIONS})
    return policies


def option_b_policies(scenario: Scenario) -> dict[str, PolicyFn]:
    """Les 12 politiques de la mesure de l'option B : heuristiques et routes constantes."""
    policies = dict(heuristic_policies(scenario))
    n_routes = scenario.shovel_count * scenario.dump_count
    policies.update({f"Toujours action {a}": constant_action_policy(a) for a in range(n_routes)})
    return policies


def reward_by_policy(
    policies: Mapping[str, PolicyFn],
    scenario: Scenario | None = None,
    seeds: Sequence[int] = DIAGNOSTIC_SEEDS,
) -> pd.DataFrame:
    """Une ligne par politique : productivité, décisions, récompense par décision.

    Les termes par décision (``per_decision_<terme>``) sont les contributions pondérées
    (``MineEnv.episode_reward_terms``), sommées sur tous les postes puis divisées par le
    nombre total de décisions ; leur somme est ``reward_per_decision``.
    ``reward_per_decision_before_b`` : la même récompense sans le coût d'opportunité des
    cycles (option B), c'est-à-dire celle d'avant le 20 octobre 2026.
    ``truck_min_per_shift`` : minutes-camion dans le poste, toutes décisions comprises
    (constant : nombre de camions × durée du poste).
    """
    scenario = scenario or SCENARIOS["nominal"]
    end = scenario.episode_minutes
    weights = RewardWeights.from_tuple(scenario.reward_weights)
    rows = []
    for name, policy in policies.items():
        tph, decisions, total, terms = [], 0, 0.0, {}
        cycle_min = all_min = 0.0
        for seed in seeds:
            env = make_env(scenario, log_cycles=True)
            kpis = run_single_episode(env, policy, seed)
            tph.append(kpis["productivity_tph"])
            decisions += kpis["decisions"]
            total += kpis["total_reward"]
            for term, value in env.episode_reward_terms.items():
                terms[term] = terms.get(term, 0.0) + value
            for record in env.cycle_log:
                minutes = max(0.0, min(record.end_min, end) - min(record.start_min, end))
                all_min += minutes
                cycle_min += minutes if record.kind == "cycle" else 0.0
        cycle_cost = weights.opportunity * cycle_min / OPPORTUNITY_CYCLE_MIN
        rows.append({
            "policy": name,
            "productivity_tph": float(np.mean(tph)),
            "decisions_per_shift": decisions / len(seeds),
            "reward_per_shift": total / len(seeds),
            "reward_per_decision": total / decisions,
            "reward_per_decision_before_b": (total + cycle_cost) / decisions,
            "truck_min_per_shift": all_min / len(seeds),
            **{f"per_decision_{term}": value / decisions for term, value in terms.items()},
        })
    return pd.DataFrame(rows)


def rank_agreement(table: pd.DataFrame, column: str) -> float:
    """Corrélation de Spearman entre la productivité et ``column`` (1 = même classement)."""
    return float(spearmanr(table["productivity_tph"], table[column]).statistic)


def option_b_rankings(
    scenarios: Sequence[str] = ("nominal", "high_load", "high_breakdown"),
    seeds: Sequence[int] = DIAGNOSTIC_SEEDS,
) -> pd.DataFrame:
    """Accord des classements, avant et après l'option B, par régime (tableau 2 de
    ``docs/option_b_recompense.md``)."""
    rows = []
    for name in scenarios:
        scenario = SCENARIOS[name]
        table = reward_by_policy(option_b_policies(scenario), scenario, seeds)
        rows.append({"scenario": name, "n_policies": len(table),
                     "spearman_before_b": rank_agreement(table, "reward_per_decision_before_b"),
                     "spearman_option_b": rank_agreement(table, "reward_per_decision")})
    return pd.DataFrame(rows)


def q_value_gaps(model: Any, scenario: Scenario | None = None,
                 seed: int = TUNING_SEEDS[0]) -> dict[str, Any]:
    """Valeurs Q d'un modèle DQN le long d'un poste qu'il joue (actions gloutonnes)."""
    import torch  # seulement pour cette mesure

    scenario = scenario or SCENARIOS["nominal"]
    env = make_env(scenario)
    obs, _ = env.reset(seed=seed)
    q_values, done = [], False
    while not done:
        with torch.no_grad():
            q = model.q_net(torch.as_tensor(obs[None], dtype=torch.float32)).cpu().numpy()[0]
        q_values.append(q)
        obs, _, terminated, truncated, _ = env.step(int(q.argmax()))
        done = terminated or truncated
    q = np.array(q_values)
    ordered = np.sort(q, axis=1)
    return {
        "seed": seed,
        "decisions": len(q),
        "mean_abs_q": float(np.abs(q).mean()),
        "mean_q_by_action": q.mean(axis=0).round(3).tolist(),
        "gap_best_second": float((ordered[:, -1] - ordered[:, -2]).mean()),
        "gap_best_worst": float((ordered[:, -1] - ordered[:, 0]).mean()),
        "chosen_actions": np.bincount(q.argmax(axis=1), minlength=q.shape[1]).tolist(),
    }


def main(out_dir: str | Path, dqn_model: str | Path | None = None) -> pd.DataFrame:
    """Écrit ``reward_by_policy.csv``, ``option_b_rankings.csv`` (et ``q_value_gaps.json``
    avec ``dqn_model``)."""
    out_dir = Path(out_dir)
    targets = [out_dir / "reward_by_policy.csv", out_dir / "option_b_rankings.csv"]
    targets += [out_dir / "q_value_gaps.json"] if dqn_model else []
    existing = [p for p in targets if p.exists()]
    if existing:
        raise FileExistsError(f"Déjà présent : {existing[0]} (choisir un autre dossier)")
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario = SCENARIOS["nominal"]
    table = reward_by_policy(reference_policies(scenario), scenario)
    table.to_csv(targets[0], index=False)
    print(table.round(3).to_string(index=False))
    rankings = option_b_rankings()
    rankings.to_csv(targets[1], index=False)
    print(rankings.round(3).to_string(index=False))
    if dqn_model:
        from memoire_master_rl_logistique.rl.model_loading import load_model

        gaps = q_value_gaps(load_model(dqn_model, "DQN", make_env(scenario)), scenario)
        targets[2].write_text(json.dumps(gaps, indent=2), encoding="utf-8")
        print(gaps)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesures de l'échec des agents à valeurs.")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--dqn-model", type=Path, default=None,
                        help="dossier d'un modèle DQN entraîné (ex. essai 7)")
    args = parser.parse_args()
    main(args.out_dir, args.dqn_model)
