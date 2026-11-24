"""Inférence entre modèles (M8 du rapport du 17/11/2026), sur le CSV de la campagne.

Rien n'est simulé : les 30 réplications et les 5 graines d'entraînement de la campagne sont
relues. Cinq sorties :

1. ``per_seed`` : ATTENDRE (%) et carburant par tonne de chaque modèle PPO et DQN, par
   graine d'entraînement (le Tableau 17 de l'article ne donne que la productivité) ;
2. ``hierarchical`` : IC à 95 % de l'écart agent − référence par un **bootstrap
   hiérarchique** qui rééchantillonne les graines d'entraînement, puis les réplications
   (appariées avec la référence), à côté de l'IC usuel qui ne rééchantillonne que les
   réplications ;
3. ``effect_sizes`` : d de Cohen **apparié** (d_z = moyenne des différences / écart type des
   différences, sur les 30 réplications) à côté du d calculé jusqu'ici avec l'écart type
   groupé des deux échantillons, et écart pratique en % et en camions équivalents ;
4. ``interaction`` : interaction politique × régime sur toutes les politiques et sur les
   seules politiques compétitives (MTWT, LP, PPO, FIFO, Affectation fixe).

    python -m memoire_master_rl_logistique.analysis.model_inference CSV DOSSIER_SORTIE
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from memoire_master_rl_logistique.analysis.statistical_tests import (
    PAIRS,
    average_over_training_seeds,
    cohens_d,
    interaction_test,
    load_results,
    paired_bootstrap_ci,
)
from memoire_master_rl_logistique.experiments.scenarios import MAIN_SCENARIOS

COMPETITIVE = ("MTWT", "LP", "PPO", "FIFO", "Fixed")
AGENTS = ("PPO", "DQN")
PRODUCTIVITY = "productivity_tph"


def cohens_dz(diff) -> float:
    """d de Cohen pour des différences appariées : moyenne / écart type des différences."""
    diff = np.asarray(diff, dtype=float)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def hierarchical_bootstrap_ci(agent: pd.DataFrame, reference: pd.Series, n_boot: int = 10_000,
                              seed: int = 0) -> tuple[float, float]:
    """IC à 95 % de la moyenne de (agent − référence), deux niveaux d'aléa.

    ``agent`` : une ligne par graine d'entraînement, une colonne par graine d'évaluation ;
    ``reference`` : une valeur par graine d'évaluation (heuristique). Chaque tirage
    rééchantillonne les graines d'entraînement, puis les graines d'évaluation (les mêmes pour
    l'agent et la référence, ce qui garde l'appariement)."""
    rng = np.random.default_rng(seed)
    values = agent[reference.index].to_numpy()
    ref = reference.to_numpy()
    n_train, n_eval = values.shape
    means = np.empty(n_boot)
    for b in range(n_boot):
        t = rng.integers(0, n_train, n_train)
        e = rng.integers(0, n_eval, n_eval)
        means[b] = (values[np.ix_(t, e)] - ref[e]).mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def per_seed(raw: pd.DataFrame) -> pd.DataFrame:
    agents = raw[raw["policy"].isin(AGENTS) & raw["scenario"].isin(MAIN_SCENARIOS)]
    return (agents.groupby(["scenario", "policy", "train_seed"])
            [["wait_share_pct", "specific_fuel_l_per_ton", PRODUCTIVITY]].mean().reset_index())


def hierarchical(raw: pd.DataFrame, n_boot: int = 10_000) -> pd.DataFrame:
    rows = []
    averaged = average_over_training_seeds(raw)
    for scenario in MAIN_SCENARIOS:
        d = raw[raw["scenario"] == scenario]
        for agent, ref in PAIRS:
            if agent not in AGENTS + ("Q-Learning", "SARSA"):
                continue
            ref_values = d[d["policy"] == ref].set_index("seed")[PRODUCTIVITY].sort_index()
            agent_values = d[d["policy"] == agent].pivot_table(index="train_seed", columns="seed",
                                                               values=PRODUCTIVITY)
            a = averaged[(averaged["scenario"] == scenario) & (averaged["policy"] == agent)] \
                .set_index("seed")[PRODUCTIVITY].sort_index()
            diff = a - ref_values
            low, high = paired_bootstrap_ci(diff.to_numpy())
            hlow, hhigh = hierarchical_bootstrap_ci(agent_values, ref_values, n_boot)
            rows.append({"scenario": scenario, "pair": f"{agent}-{ref}", "mean_diff": diff.mean(),
                         "ci_low": low, "ci_high": high, "h_ci_low": hlow, "h_ci_high": hhigh})
    return pd.DataFrame(rows)


def effect_sizes(raw: pd.DataFrame) -> pd.DataFrame:
    data = average_over_training_seeds(raw)
    rows = []
    for scenario in MAIN_SCENARIOS:
        d = data[data["scenario"] == scenario]
        table = d.pivot(index="seed", columns="policy", values=PRODUCTIVITY)
        per_truck = d.pivot(index="seed", columns="policy", values=PRODUCTIVITY).div(
            d.groupby("policy")["truck_count"].first(), axis=1)
        for a, b in PAIRS:
            diff = table[a] - table[b]
            rows.append({"scenario": scenario, "pair": f"{a}-{b}", "mean_diff": diff.mean(),
                         "d_pooled": cohens_d(table[a], table[b]), "d_z": cohens_dz(diff),
                         "diff_pct": 100 * diff.mean() / table[b].mean(),
                         "trucks_equivalent": diff.mean() / per_truck[b].mean()})
    return pd.DataFrame(rows)


def interaction(raw: pd.DataFrame) -> pd.DataFrame:
    data = average_over_training_seeds(raw)
    rows = []
    for label, subset in (("all", data), ("competitive", data[data["policy"].isin(COMPETITIVE)])):
        rows.append({"policies": label, **interaction_test(subset)})
    return pd.DataFrame(rows)


def run(csv_path: Path, out_dir: Path, n_boot: int = 10_000) -> dict[str, pd.DataFrame]:
    raw = load_results(csv_path)
    raw = raw[raw["scenario"].isin(MAIN_SCENARIOS)]
    outputs = {"per_seed": per_seed(raw), "hierarchical": hierarchical(raw, n_boot),
               "effect_sizes": effect_sizes(raw), "interaction": interaction(raw)}
    out_dir.mkdir(parents=True, exist_ok=True)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        for name, table in outputs.items():
            table.to_csv(out_dir / f"{name}.csv", index=False)
            print(f"\n===== {name} =====")
            print(table.round(3).to_string(index=False))
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)
    run(args.csv_path, args.out_dir)


if __name__ == "__main__":
    main()
